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
)
from layer_fusion.providers import FakeLayerVectorProvider  # noqa: E402
from layer_fusion.runner import make_run_id, run_experiment  # noqa: E402


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
    best_epoch = int(np.argmin(errors)) + 1  # first minimum: only strict decreases count
    assert outcome.best_epoch == best_epoch
    assert outcome.best_tuning_error == errors[best_epoch - 1]
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
    meta = CacheMetadata(
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


def test_run_id_is_deterministic(noisy_provider, make_run_config) -> None:  # type: ignore[no-untyped-def]
    config = make_run_config(noisy_provider(), seed=3, run_kind=RunKind.DEBUG)
    assert make_run_id(config) == "mnli-full-F2-all-frozen-none-s3-debug"


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"epochs": None}, ValueError),
        ({"batch_size": 0}, ValueError),
        ({"early_stopping_patience": None}, ValueError),
        ({"lr_candidates": ()}, ValueError),
        ({"lr_candidates": (1e-3, 1e-2)}, ValueError),  # main runs need exactly 3 (ADR 0007)
        ({"lr_candidates": (1e-3, 1e-3, 1e-2)}, ValueError),
        ({"fixed_lr": 0.5, "lr_selection_seed": 0}, ValueError),  # not a candidate
        ({"fixed_lr": 1e-3}, ValueError),  # missing lr_selection_seed
        ({"lr_selection_seed": 0}, ValueError),  # seed without fixed_lr
        ({"backbone_condition": BackboneCondition.FINETUNE}, NotImplementedError),  # T-06d
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


def test_mnli_without_genres_is_rejected(noisy_provider, make_run_config) -> None:  # type: ignore[no-untyped-def]
    base = noisy_provider()
    provider = FakeLayerVectorProvider(
        {
            "train": base.get_split("train"),
            "eval": replace(base.get_split("eval"), genre_splits=None),
        },
        base.metadata,
    )
    with pytest.raises(ValueError, match="matched"):
        run_experiment(make_run_config(provider, epochs=1), provider, device="cpu")
