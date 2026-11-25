"""Unit tests for the execution seam internals (runner.run_experiment, training.train_head; T-06a).

Seam ① contract tests live in tests/test_execution_seam.py; these cover the T-06a decisions
(fixed tuning split, one LR selection reused by other seeds, early stopping on the tuning split)
and the input checks. All run on CPU with fake providers.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from layer_fusion.contracts import (  # noqa: E402
    BackboneCondition,
    CacheMetadata,
    DatasetName,
    FusionMethod,
    InputSetting,
    LengthBucket,
    RunKind,
    TrackKind,
)
from layer_fusion.providers import FakeLayerVectorProvider  # noqa: E402
from layer_fusion.runner import make_run_id, run_experiment  # noqa: E402
from layer_fusion.training import TrainingDivergedError  # noqa: E402


def _imdb_l512_metadata() -> CacheMetadata:
    return CacheMetadata(
        model_name="answerdotai/ModernBERT-base",
        transformers_version="5.23.0",
        attn_implementation="sdpa",
        torch_compile=False,
        input_setting=InputSetting.L512,
        max_length_cap=512,
        truncation_rule="head_128_tail_382",
        include_special_tokens_in_pool=True,
        dtype="float32",
    )


def _predictions(record):  # type: ignore[no-untyped-def]
    return [p.prediction for p in record.predictions]


def test_same_config_gives_identical_record(noisy_provider, make_run_config) -> None:  # type: ignore[no-untyped-def]
    provider = noisy_provider()
    config = make_run_config(provider)
    first = run_experiment(config, provider, device="cpu")
    second = run_experiment(config, provider, device="cpu")
    assert first.lr_tuning_scores == second.lr_tuning_scores
    assert _predictions(first) == _predictions(second)
    assert first.layer_weights == second.layer_weights


def test_run_does_not_touch_global_torch_rng(noisy_provider, make_run_config) -> None:  # type: ignore[no-untyped-def]
    provider = noisy_provider()
    torch.manual_seed(123)
    expected = torch.rand(3)
    torch.manual_seed(123)
    run_experiment(make_run_config(provider, epochs=1), provider, device="cpu")
    assert torch.equal(torch.rand(3), expected)


def test_fixed_lr_run_reproduces_selection_run_model(noisy_provider, make_run_config) -> None:  # type: ignore[no-untyped-def]
    """Other seeds reuse the selected LR; at the selection seed the model is the same."""
    provider = noisy_provider()
    selection = run_experiment(make_run_config(provider), provider, device="cpu")
    fixed = make_run_config(
        provider, fixed_lr=selection.selected_lr, lr_selection_seed=selection.config.seed
    )
    rerun = run_experiment(fixed, provider, device="cpu")
    assert _predictions(rerun) == _predictions(selection)
    assert rerun.train_seconds > 0
    # A reuse run trains one candidate only and points at the selection run.
    assert rerun.selected_lr == selection.selected_lr
    assert rerun.lr_tuning_scores == {}
    assert rerun.lr_search_seconds is None
    assert rerun.lr_selection_seed == selection.config.seed

    other_seed = run_experiment(
        replace(fixed, seed=7), provider, device="cpu"
    )
    assert other_seed.selected_lr == selection.selected_lr
    assert other_seed.lr_selection_seed == selection.config.seed


def test_selection_picks_lowest_tuning_error(noisy_provider, make_run_config) -> None:  # type: ignore[no-untyped-def]
    provider = noisy_provider()
    record = run_experiment(make_run_config(provider), provider, device="cpu")
    best_error = min(record.lr_tuning_scores.values())
    assert record.lr_tuning_scores[record.selected_lr] == best_error


def test_early_stopping_restores_best_tuning_epoch(noisy_provider) -> None:  # type: ignore[no-untyped-def]
    from layer_fusion.fusion import build_fusion_head
    from layer_fusion.training import RowView, evaluate, train_head

    provider = noisy_provider()
    train = provider.get_split("train")
    rows = np.arange(len(train.doc_ids))
    train_view = RowView(train.vectors, train.labels, rows[:240])
    tuning_view = RowView(train.vectors, train.labels, rows[240:])
    device = torch.device("cpu")
    head, outcome = train_head(
        lambda: build_fusion_head(FusionMethod.B0, 3),
        train_view,
        tuning_view,
        lr=0.1,
        epochs=12,
        patience=2,
        batch_size=32,
        seed=0,
        device=device,
    )
    errors = [row["tuning_error"] for row in outcome.learning_curve]
    scores = [(row["tuning_error"], row["tuning_loss"]) for row in outcome.learning_curve]
    # (error, loss) lexicographic, first minimum: ties on both never count as an improvement
    best_epoch = min(range(len(scores)), key=lambda i: (scores[i], i)) + 1
    assert outcome.best_epoch == best_epoch
    assert outcome.best_tuning_error == errors[best_epoch - 1]
    assert not outcome.diverged
    # Stopped exactly `patience` epochs after the best one (or ran out of epochs).
    assert len(errors) == min(12, best_epoch + 2)
    _, restored_error, _ = evaluate(head, tuning_view, 32, device)
    assert restored_error == pytest.approx(outcome.best_tuning_error)


def test_float16_cache_is_upcast_per_batch(noisy_provider, make_run_config) -> None:  # type: ignore[no-untyped-def]
    provider = noisy_provider(dtype=np.float16)
    assert provider.get_split("train").vectors.dtype == np.float16
    record = run_experiment(make_run_config(provider, epochs=2), provider, device="cpu")
    assert len(record.predictions) == len(provider.get_split("eval").doc_ids)


def test_imdb_records_carry_length_buckets_and_truncated_ratio(
    noisy_provider, make_run_config
) -> None:  # type: ignore[no-untyped-def]
    meta = _imdb_l512_metadata()
    base = noisy_provider(num_classes=2, eval_size=40, metadata=meta)
    lengths = [100, 600, 1500, 512] * 10
    provider = FakeLayerVectorProvider(
        {
            "train": base.get_split("train"),
            "eval": replace(base.get_split("eval"), token_lengths=lengths),
        },
        meta,
    )
    config = make_run_config(
        provider, dataset=DatasetName.IMDB, input_setting=InputSetting.L512, epochs=2
    )
    record = run_experiment(config, provider, device="cpu")
    assert [p.token_length for p in record.predictions] == lengths
    assert [p.length_bucket for p in record.predictions[:4]] == [
        LengthBucket.LE_512,
        LengthBucket.BETWEEN_512_1024,
        LengthBucket.GT_1024,
        LengthBucket.LE_512,
    ]
    assert record.truncated_doc_ratio == pytest.approx(0.5)  # 600 and 1500 exceed 512
    assert record.eval_error_rate_matched is None and record.eval_error_rate_mismatched is None


def test_early_stopping_counts_equal_error_with_lower_loss_as_improvement(
    noisy_provider, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    """Same tuning error but lower tuning loss resets patience (same order as the LR tie-break)."""
    from layer_fusion import training
    from layer_fusion.fusion import build_fusion_head

    provider = noisy_provider()
    train = provider.get_split("train")
    rows = np.arange(len(train.doc_ids))
    # Tuning error stays 0.5 every epoch while the loss keeps falling.
    losses = iter([1.0, 0.9, 0.8, 0.7])
    monkeypatch.setattr(
        training, "evaluate", lambda *a, **k: (next(losses), 0.5, np.zeros(0, dtype=np.int64))
    )
    _, outcome = training.train_head(
        lambda: build_fusion_head(FusionMethod.B0, 3),
        training.RowView(train.vectors, train.labels, rows[:240]),
        training.RowView(train.vectors, train.labels, rows[240:]),
        lr=1e-3,
        epochs=4,
        patience=1,
        batch_size=64,
        seed=0,
        device=torch.device("cpu"),
    )
    assert len(outcome.learning_curve) == 4  # never stopped early
    assert outcome.best_epoch == 4 and outcome.best_tuning_loss == pytest.approx(0.7)


def test_train_head_rejects_empty_budget(noisy_provider) -> None:  # type: ignore[no-untyped-def]
    from layer_fusion.fusion import build_fusion_head
    from layer_fusion.training import RowView, train_head

    train = noisy_provider().get_split("train")
    view = RowView(train.vectors, train.labels, np.arange(10))
    kwargs = dict(lr=1e-3, epochs=1, patience=1, batch_size=8, seed=0, device=torch.device("cpu"))
    for bad in ({"epochs": 0}, {"patience": 0}, {"batch_size": 0}, {"lr": float("inf")}):
        with pytest.raises(ValueError):
            train_head(
                lambda: build_fusion_head(FusionMethod.B0, 3), view, view, **{**kwargs, **bad}
            )


def test_optimizer_decays_weight_matrices_only() -> None:
    """Role-based decay: layer-weighting params (F2 logits, F4 gate) and biases get 0.0."""
    from layer_fusion.fusion import build_fusion_head
    from layer_fusion.training import WEIGHT_DECAY, make_optimizer

    def decay_map(module: torch.nn.Module) -> dict[str, float]:
        by_id = {
            id(p): group["weight_decay"]
            for group in make_optimizer(module, 1e-3).param_groups
            for p in group["params"]
        }
        named = dict(module.named_parameters())
        assert len(by_id) == len(named)
        return {name: by_id[id(param)] for name, param in named.items()}

    assert decay_map(build_fusion_head(FusionMethod.F2, 3)) == {
        "classifier.weight": WEIGHT_DECAY,
        "classifier.bias": 0.0,
        "layer_logits": 0.0,
    }
    assert decay_map(build_fusion_head(FusionMethod.F4, 3)) == {
        "classifier.weight": WEIGHT_DECAY,
        "classifier.bias": 0.0,
        "gate.weight": 0.0,
        "gate.bias": 0.0,
    }
    assert decay_map(build_fusion_head(FusionMethod.B0_PLUS, 3)) == {
        "mlp_in.weight": WEIGHT_DECAY,
        "mlp_in.bias": 0.0,
        "classifier.weight": WEIGHT_DECAY,
        "classifier.bias": 0.0,
    }

    # Wrapper modules (e.g. FinetuneModel(backbone, head)) inherit the same role-based rule.
    class _Wrapper(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = torch.nn.Linear(4, 4)
            self.head = build_fusion_head(FusionMethod.F4, 3, hidden_size=4)

    wrapper_decay = decay_map(_Wrapper())
    assert wrapper_decay["backbone.weight"] == WEIGHT_DECAY
    assert wrapper_decay["backbone.bias"] == 0.0
    assert wrapper_decay["head.classifier.weight"] == WEIGHT_DECAY
    assert wrapper_decay["head.gate.weight"] == 0.0
    assert wrapper_decay["head.gate.bias"] == 0.0


def test_diverged_candidate_is_excluded_from_selection(noisy_provider, make_run_config) -> None:  # type: ignore[no-untyped-def]
    provider = noisy_provider()
    config = make_run_config(provider, lr_candidates=(1e10, 1e-3, 1e-1))
    record = run_experiment(config, provider, device="cpu")
    assert record.lr_tuning_scores[1e10] == float("inf")
    assert record.selected_lr != 1e10
    assert all(np.isfinite(w) for w in record.layer_weights or [])


def test_all_candidates_diverging_raises(noisy_provider, make_run_config) -> None:  # type: ignore[no-untyped-def]
    provider = noisy_provider()
    config = make_run_config(provider, run_kind=RunKind.DEBUG, lr_candidates=(1e10,))
    with pytest.raises(TrainingDivergedError):
        run_experiment(config, provider, device="cpu")
    fixed = make_run_config(
        provider, lr_candidates=(1e-3, 1e-1, 1e10), fixed_lr=1e10, lr_selection_seed=0
    )
    with pytest.raises(TrainingDivergedError):
        run_experiment(fixed, provider, device="cpu")


def test_non_finite_input_vectors_are_rejected(noisy_provider, make_run_config) -> None:  # type: ignore[no-untyped-def]
    base = noisy_provider()
    train = base.get_split("train")
    poisoned = train.vectors.copy()
    poisoned[0, 0, 0] = np.nan
    provider = FakeLayerVectorProvider(
        {"train": replace(train, vectors=poisoned), "eval": base.get_split("eval")},
        base.metadata,
    )
    with pytest.raises(TrainingDivergedError):
        run_experiment(make_run_config(provider, epochs=1), provider, device="cpu")

    eval_batch = base.get_split("eval")
    poisoned_eval = eval_batch.vectors.copy()
    poisoned_eval[0, 0, 0] = np.inf
    provider = FakeLayerVectorProvider(
        {"train": train, "eval": replace(eval_batch, vectors=poisoned_eval)}, base.metadata
    )
    with pytest.raises(ValueError, match="eval loss"):
        run_experiment(make_run_config(provider, epochs=1), provider, device="cpu")


def test_time_fields_split_training_and_tuning_evaluation(noisy_provider, make_run_config) -> None:  # type: ignore[no-untyped-def]
    provider = noisy_provider()
    record = run_experiment(make_run_config(provider), provider, device="cpu")
    assert record.train_seconds > 0
    assert record.tuning_eval_seconds is not None and record.tuning_eval_seconds > 0
    assert record.lr_search_seconds is not None
    assert record.lr_search_seconds >= record.train_seconds + record.tuning_eval_seconds


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_run_does_not_touch_global_cuda_rng(noisy_provider, make_run_config) -> None:  # type: ignore[no-untyped-def]
    provider = noisy_provider()
    torch.cuda.manual_seed_all(123)
    expected = torch.rand(3, device="cuda")
    torch.cuda.manual_seed_all(123)
    run_experiment(make_run_config(provider, epochs=1), provider, device="cuda")
    assert torch.equal(torch.rand(3, device="cuda"), expected)


def test_run_id_is_deterministic(noisy_provider, make_run_config) -> None:  # type: ignore[no-untyped-def]
    config = make_run_config(noisy_provider(), seed=3, run_kind=RunKind.DEBUG)
    assert make_run_id(config) == "mnli-full-F2-all-frozen-none-s3-debug"
    repro = replace(config, track=TrackKind.REPRODUCTION)
    assert make_run_id(repro) == "repro-mnli-full-F2-all-frozen-none-s3-debug"


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"epochs": None}, ValueError),
        ({"batch_size": 0}, ValueError),
        ({"early_stopping_patience": None}, ValueError),
        ({"early_stopping_patience": 0}, ValueError),  # would behave exactly like 1
        ({"lr_candidates": (1e-3, float("inf"), 1e-1)}, ValueError),
        ({"lr_candidates": (1e-3, float("nan"), 1e-1)}, ValueError),
        ({"track": TrackKind.REPRODUCTION}, ValueError),  # run_reproduction's job (T-06e)
        ({"dataset": DatasetName.IMDB}, ValueError),  # IMDB without input_setting
        ({"dataset": DatasetName.SST2, "input_setting": InputSetting.L512}, ValueError),
        ({"lr_candidates": ()}, ValueError),
        ({"lr_candidates": (1e-3, 1e-2)}, ValueError),  # main runs need exactly 3 (ADR 0007)
        ({"lr_candidates": (1e-3, 1e-3, 1e-2)}, ValueError),
        ({"fixed_lr": 0.5, "lr_selection_seed": 0}, ValueError),  # not a candidate
        ({"fixed_lr": 1e-3}, ValueError),  # missing lr_selection_seed
        ({"lr_selection_seed": 0}, ValueError),  # seed without fixed_lr
        # A cache has no trainable backbone: finetune needs a BackboneProvider (T-06d).
        ({"backbone_condition": BackboneCondition.FINETUNE}, ValueError),
    ],
)
def test_incomplete_or_inconsistent_budget_is_rejected(
    noisy_provider, make_run_config, overrides, error
) -> None:  # type: ignore[no-untyped-def]
    provider = noisy_provider()
    with pytest.raises(error):
        run_experiment(make_run_config(provider, **overrides), provider, device="cpu")


def test_debug_run_may_use_fewer_lr_candidates(noisy_provider, make_run_config) -> None:  # type: ignore[no-untyped-def]
    provider = noisy_provider()
    config = make_run_config(
        provider, run_kind=RunKind.DEBUG, lr_candidates=(1e-3,), expected_cache_meta=None, epochs=1
    )
    record = run_experiment(config, provider, device="cpu")
    assert list(record.lr_tuning_scores) == [1e-3]


def test_labels_outside_dataset_classes_are_rejected(noisy_provider, make_run_config) -> None:  # type: ignore[no-untyped-def]
    provider = noisy_provider()  # 3 classes
    config = make_run_config(provider, dataset=DatasetName.SST2)  # SST-2 has 2
    with pytest.raises(ValueError, match="range\\(2\\)"):
        run_experiment(config, provider, device="cpu")


@pytest.mark.parametrize("bad_genre", [None, "telephone"])
def test_mnli_with_missing_or_unknown_genres_is_rejected(
    noisy_provider, make_run_config, bad_genre
) -> None:  # type: ignore[no-untyped-def]
    base = noisy_provider()
    eval_batch = base.get_split("eval")
    if bad_genre is None:
        genres = None
    else:  # one document outside matched/mismatched would silently drop out of both rates
        genres = [bad_genre, *eval_batch.genre_splits[1:]]
    provider = FakeLayerVectorProvider(
        {"train": base.get_split("train"), "eval": replace(eval_batch, genre_splits=genres)},
        base.metadata,
    )
    with pytest.raises(ValueError, match="matched"):
        run_experiment(make_run_config(provider, epochs=1), provider, device="cpu")


def test_imdb_eval_without_token_lengths_is_rejected(noisy_provider, make_run_config) -> None:  # type: ignore[no-untyped-def]
    base = noisy_provider(num_classes=2, metadata=_imdb_l512_metadata())
    provider = FakeLayerVectorProvider(
        {
            "train": base.get_split("train"),
            "eval": replace(base.get_split("eval"), token_lengths=None),
        },
        base.metadata,
    )
    config = make_run_config(
        provider, dataset=DatasetName.IMDB, input_setting=InputSetting.L512, epochs=1
    )
    with pytest.raises(ValueError, match="token_lengths"):
        run_experiment(config, provider, device="cpu")


def test_input_setting_on_non_imdb_dataset_is_rejected(noisy_provider, make_run_config) -> None:  # type: ignore[no-untyped-def]
    # Provider metadata agrees with the run, so only the dataset rule can reject it.
    provider = noisy_provider(num_classes=2, metadata=_imdb_l512_metadata())
    config = make_run_config(
        provider, dataset=DatasetName.SST2, input_setting=InputSetting.L512, epochs=1
    )
    with pytest.raises(ValueError, match="IMDB only"):
        run_experiment(config, provider, device="cpu")


# --- F4 per-document layer weights in the result record (SPEC module 6, US 54) ----------


def _random_gate_f4(num_classes: int = 3):  # type: ignore[no-untyped-def]
    """F4 head whose gate is not at its zero init, so weights differ between documents."""
    from layer_fusion.fusion import build_fusion_head

    head = build_fusion_head(FusionMethod.F4, num_classes)
    gen = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for name, param in head.named_parameters():
            if name.startswith("gate"):
                param.copy_(torch.randn(param.shape, generator=gen))
    return head


def test_eval_layer_weights_match_the_public_extractor_in_view_order(noisy_provider) -> None:  # type: ignore[no-untyped-def]
    from layer_fusion.fusion import extract_document_layer_weights
    from layer_fusion.training import RowView, evaluate, evaluate_with_layer_weights

    batch = noisy_provider().get_split("eval")
    rows = np.random.default_rng(0).permutation(len(batch.doc_ids))[:40].astype(np.int64)
    view = RowView(batch.vectors, batch.labels, rows)
    head = _random_gate_f4()
    device = torch.device("cpu")

    loss, error, predicted, weights = evaluate_with_layer_weights(head, view, 16, device)
    expected = extract_document_layer_weights(head, torch.from_numpy(batch.vectors[rows]))
    np.testing.assert_allclose(np.asarray(weights), np.asarray(expected), rtol=1e-5, atol=1e-6)
    assert (loss, error) == pytest.approx(evaluate(head, view, 16, device)[:2])
    np.testing.assert_array_equal(predicted, evaluate(head, view, 16, device)[2])


@pytest.mark.parametrize(
    "method",
    [FusionMethod.B0, FusionMethod.F1, FusionMethod.F2, FusionMethod.F3, FusionMethod.B0_PLUS],
)
def test_eval_layer_weights_are_none_without_f4(noisy_provider, method) -> None:  # type: ignore[no-untyped-def]
    from layer_fusion.fusion import build_fusion_head
    from layer_fusion.training import RowView, evaluate_with_layer_weights

    batch = noisy_provider().get_split("eval")
    view = RowView(batch.vectors, batch.labels, np.arange(len(batch.doc_ids), dtype=np.int64))
    head = build_fusion_head(method, 3)
    assert evaluate_with_layer_weights(head, view, 16, torch.device("cpu"))[3] is None


def test_f4_record_carries_per_document_layer_weights(  # type: ignore[no-untyped-def]
    noisy_provider, make_run_config, tmp_path
) -> None:
    from layer_fusion.layers import NUM_LAYER_VECTORS
    from layer_fusion.records import load_result_record, save_result_record

    provider = noisy_provider()
    record = run_experiment(
        make_run_config(provider, fusion_method=FusionMethod.F4),
        provider, device="cpu",
    )
    weights = np.asarray([p.layer_weights for p in record.predictions])
    assert weights.shape == (len(provider.get_split("eval").doc_ids), NUM_LAYER_VECTORS)
    assert (weights > 0).all()
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-5)
    assert np.ptp(weights, axis=0).max() > 0  # the trained gate differs between documents
    assert record.layer_weights is None  # dataset-level weights stay F2 only

    loaded = load_result_record(save_result_record(record, tmp_path))
    assert [p.layer_weights for p in loaded.predictions] == [
        p.layer_weights for p in record.predictions
    ]


@pytest.mark.parametrize(
    "method",
    [FusionMethod.B0, FusionMethod.F1, FusionMethod.F2, FusionMethod.F3, FusionMethod.B0_PLUS],
)
def test_other_records_have_no_per_document_layer_weights(  # type: ignore[no-untyped-def]
    noisy_provider, make_run_config, method
) -> None:
    provider = noisy_provider()
    record = run_experiment(
        make_run_config(provider, fusion_method=method, epochs=2),
        provider, device="cpu",
    )
    assert all(p.layer_weights is None for p in record.predictions)
    assert (record.layer_weights is not None) == (method is FusionMethod.F2)


def test_prediction_records_pair_each_document_with_its_own_weights(  # type: ignore[no-untyped-def]
    noisy_provider, make_run_config
) -> None:
    from layer_fusion.runner import prediction_records

    provider = noisy_provider(eval_size=4)
    batch = provider.get_split("eval")
    config = make_run_config(provider, fusion_method=FusionMethod.F4)
    weights = [[float(i)] * 23 for i in range(4)]
    records = prediction_records(batch, np.zeros(4, dtype=np.int64), config, weights)
    assert [(r.doc_id, r.layer_weights[0]) for r in records] == [
        (doc_id, float(i)) for i, doc_id in enumerate(batch.doc_ids)
    ]
    with pytest.raises(ValueError, match="3 per-document layer weights for 4 eval documents"):
        prediction_records(batch, np.zeros(4, dtype=np.int64), config, weights[:3])
