"""Seam 1 tests: Execution seam (RunConfig + LayerVectorProvider -> ResultRecord).

All tests run on CPU using FakeLayerVectorProvider without GPU or real datasets.
Un-skip each test as T-05 / T-06 are implemented (SPEC.md §Seam ①).
"""

import pytest


@pytest.mark.skip(reason="TODO(T-05b): Enable after F4 and B0+ are implemented")
def test_trainable_parameter_counts_and_b0_plus_matched_to_f4() -> None:
    """Each fusion method matches definition and B0+ matches F4 parameter count."""
    raise NotImplementedError


def test_per_layer_norm_makes_f1_invariant_to_single_layer_scaling() -> None:
    """Scaling one layer vector sharply must not alter F1 output thanks to per-layer LN."""
    torch = pytest.importorskip("torch")
    from layer_fusion.contracts import FusionMethod
    from layer_fusion.fusion import build_fusion_head

    head = build_fusion_head(FusionMethod.F1, num_classes=3)
    vectors = torch.randn(4, 23, 768, generator=torch.Generator().manual_seed(0))
    scaled = vectors.clone()
    scaled[:, 11] *= 1000.0
    assert torch.allclose(head(vectors), head(scaled), atol=1e-4)


# The fusion-head tests below use only the public fusion API (build_fusion_head, forward,
# count_trainable_parameters, extract_layer_weights). Revisit routing them through
# run_experiment once T-06a implements the runner.


def test_f2_has_no_global_scale_gamma() -> None:
    """F2 has no global scalar scale parameter gamma."""
    torch = pytest.importorskip("torch")
    from layer_fusion.contracts import FusionMethod
    from layer_fusion.fusion import (
        build_fusion_head,
        count_trainable_parameters,
        extract_layer_weights,
    )

    f2 = build_fusion_head(FusionMethod.F2, num_classes=3)
    b0 = build_fusion_head(FusionMethod.B0, num_classes=3)
    # Same classifier as B0 plus exactly one weight per layer vector: no extra scalar.
    assert count_trainable_parameters(f2) == count_trainable_parameters(b0) + 23

    # Whatever values the parameters take, the layer weights stay a convex combination,
    # so they cannot act as an overall scale.
    torch.manual_seed(0)
    with torch.no_grad():
        for param in f2.parameters():
            param.normal_(std=3.0)
    weights = extract_layer_weights(f2)
    assert all(w > 0 for w in weights)
    assert sum(weights) == pytest.approx(1.0, abs=1e-6)


def test_f2_global_and_local_subsets_exclude_embedding() -> None:
    """global_only has 8 weights, local_only has 14 weights, embedding excluded in both."""
    torch = pytest.importorskip("torch")
    from layer_fusion.contracts import FusionMethod, LayerSubset
    from layer_fusion.fusion import build_fusion_head, extract_layer_weights

    vectors = torch.randn(4, 23, 768, generator=torch.Generator().manual_seed(0))
    changed_embedding = vectors.clone()
    changed_embedding[:, 0] += 100.0 * torch.randn(
        4, 768, generator=torch.Generator().manual_seed(1)
    )
    for subset, expected in ((LayerSubset.GLOBAL_ONLY, 8), (LayerSubset.LOCAL_ONLY, 14)):
        head = build_fusion_head(FusionMethod.F2, 3, subset)
        assert len(extract_layer_weights(head)) == expected
        # The embedding vector (slot 0) must not influence the output.
        assert torch.allclose(head(vectors), head(changed_embedding), atol=1e-5)


def test_small_data_sampling_shared_within_seed_and_varies_across_seeds(
    monkeypatch, noisy_provider, make_run_config
) -> None:  # type: ignore[no-untyped-def]
    """1,000 per class; same doc_ids per seed across fusion methods, distinct across seeds."""
    pytest.importorskip("torch")
    from collections import Counter

    from layer_fusion import runner
    from layer_fusion.contracts import DataAmount, DocumentRecord, FusionMethod
    from layer_fusion.data import make_tuning_split

    assert runner.SMALL_PER_CLASS == 1000  # the real setting; shrunk below to fit a CPU test
    per_class = 5
    monkeypatch.setattr(runner, "SMALL_PER_CLASS", per_class)
    provider = noisy_provider()

    def small_ids(fusion_method: FusionMethod, seed: int) -> list[str]:
        config = make_run_config(
            provider, data_amount=DataAmount.SMALL, fusion_method=fusion_method, seed=seed,
            epochs=1,
        )
        ids = runner.run_experiment(config, provider, device="cpu").small_sample_doc_ids
        assert ids is not None
        return ids

    seed0_b0 = small_ids(FusionMethod.B0, 0)
    assert small_ids(FusionMethod.F2, 0) == seed0_b0
    assert small_ids(FusionMethod.B0, 1) != seed0_b0

    train = provider.get_split("train")
    label_of = dict(zip(train.doc_ids, train.labels.tolist(), strict=True))
    assert Counter(label_of[i] for i in seed0_b0) == {0: per_class, 1: per_class, 2: per_class}
    # Drawn from the 90% outside the (fixed) tuning split, which stays shared with full runs.
    records = [
        DocumentRecord(doc_id=d, text_a="", label=label_of[d]) for d in train.doc_ids
    ]
    _, tuning = make_tuning_split(records, runner.TUNING_FRACTION, seed=runner.TUNING_SPLIT_SEED)
    assert not set(seed0_b0) & {r.doc_id for r in tuning}


def test_identical_tuning_budget_across_fusion_methods(
    noisy_provider, make_run_config
) -> None:  # type: ignore[no-untyped-def]
    """All fusion methods share number of LR candidates, epochs, and early stopping criteria."""
    pytest.importorskip("torch")
    from layer_fusion.contracts import FusionMethod
    from layer_fusion.runner import run_experiment

    provider = noisy_provider()
    base = make_run_config(provider)
    for fusion_method in (FusionMethod.B0, FusionMethod.F1, FusionMethod.F2, FusionMethod.F3):
        record = run_experiment(
            make_run_config(provider, fusion_method=fusion_method), provider, device="cpu"
        )
        # Every candidate was actually trained and scored on the tuning split.
        assert list(record.lr_tuning_scores) == list(base.lr_candidates)
        assert record.selected_lr in base.lr_candidates
        assert (record.config.epochs, record.config.early_stopping_patience) == (
            base.epochs,
            base.early_stopping_patience,
        )
        assert 1 <= len(record.learning_curve) <= base.epochs


def test_lr_selection_depends_only_on_tuning_split_not_eval_labels(
    noisy_provider, make_run_config
) -> None:  # type: ignore[no-untyped-def]
    """Changing eval split labels must not alter the selected learning rate."""
    pytest.importorskip("torch")
    from dataclasses import replace

    import numpy as np

    from layer_fusion.providers import FakeLayerVectorProvider
    from layer_fusion.runner import run_experiment

    provider = noisy_provider()
    eval_batch = provider.get_split("eval")
    flipped = replace(eval_batch, labels=(eval_batch.labels + 1) % 3)
    relabeled = FakeLayerVectorProvider(
        {"train": provider.get_split("train"), "eval": flipped}, provider.metadata
    )
    config = make_run_config(provider)
    original = run_experiment(config, provider, device="cpu")
    changed = run_experiment(config, relabeled, device="cpu")

    # The candidates really differ on the tuning split, so the choice is not a trivial tie.
    assert len(set(original.lr_tuning_scores.values())) > 1
    assert changed.selected_lr == original.selected_lr
    assert changed.lr_tuning_scores == original.lr_tuning_scores
    # Same trained head, so only the eval error (computed against new labels) moves.
    assert [p.prediction for p in changed.predictions] == [
        p.prediction for p in original.predictions
    ]
    assert changed.eval_error_rate != original.eval_error_rate
    assert np.isclose(
        changed.eval_error_rate,
        np.mean([p.prediction != p.label for p in changed.predictions]),
    )


def test_mnli_result_record_includes_matched_and_mismatched_error_rates(
    noisy_provider, make_run_config
) -> None:  # type: ignore[no-untyped-def]
    """MNLI ResultRecord exposes both eval_error_rate_matched and eval_error_rate_mismatched."""
    pytest.importorskip("torch")
    import numpy as np

    from layer_fusion.runner import run_experiment

    provider = noisy_provider()
    record = run_experiment(make_run_config(provider), provider, device="cpu")
    for genre, rate in (
        ("matched", record.eval_error_rate_matched),
        ("mismatched", record.eval_error_rate_mismatched),
    ):
        rows = [p for p in record.predictions if p.genre_split == genre]
        assert rows
        assert rate == pytest.approx(np.mean([p.prediction != p.label for p in rows]))


def test_mismatched_provider_metadata_raises_error() -> None:
    """Provider creation config differing from RunConfig must fail before training."""
    from dataclasses import replace

    from layer_fusion.contracts import (
        BackboneCondition,
        DataAmount,
        DatasetName,
        FusionMethod,
        RunConfig,
    )
    from layer_fusion.providers import CacheConfigMismatchError, FakeLayerVectorProvider
    from layer_fusion.runner import run_experiment

    provider = FakeLayerVectorProvider.make_synthetic()  # sdpa, compile off
    expected = replace(provider.metadata, attn_implementation="flash_attention_2")
    config = RunConfig(
        dataset=DatasetName.MNLI,
        data_amount=DataAmount.FULL,
        fusion_method=FusionMethod.F2,
        backbone_condition=BackboneCondition.FROZEN,
        seed=0,
        expected_cache_meta=expected,
    )
    with pytest.raises(CacheConfigMismatchError):
        run_experiment(config, provider)


def test_result_record_contains_required_telemetry_fields(
    noisy_provider, make_run_config
) -> None:  # type: ignore[no-untyped-def]
    """ResultRecord has trainable_params, train_seconds, learning_curve, predictions, run_kind."""
    pytest.importorskip("torch")
    from layer_fusion.contracts import FusionMethod, RunKind
    from layer_fusion.fusion import build_fusion_head, count_trainable_parameters
    from layer_fusion.runner import run_experiment

    provider = noisy_provider()
    for fusion_method, run_kind in (
        (FusionMethod.F2, RunKind.MAIN),
        (FusionMethod.B0, RunKind.DEBUG),
    ):
        config = make_run_config(provider, fusion_method=fusion_method, run_kind=run_kind)
        record = run_experiment(config, provider, device="cpu")

        assert record.config.run_kind == run_kind
        assert record.trainable_params == count_trainable_parameters(
            build_fusion_head(fusion_method, num_classes=3)
        )
        assert record.train_seconds > 0
        assert record.lr_search_seconds is not None
        assert record.lr_search_seconds >= record.train_seconds
        assert record.lr_selection_seed == config.seed
        assert record.learning_curve
        assert set(record.learning_curve[0]) == {
            "epoch", "train_loss", "tuning_loss", "tuning_error",
        }
        eval_batch = provider.get_split("eval")
        assert [p.doc_id for p in record.predictions] == eval_batch.doc_ids
        assert [p.label for p in record.predictions] == eval_batch.labels.tolist()
        assert (record.transformers_version, record.attn_implementation, record.torch_compile) == (
            provider.metadata.transformers_version,
            provider.metadata.attn_implementation,
            provider.metadata.torch_compile,
        )
        if fusion_method == FusionMethod.F2:
            assert record.layer_weights is not None and len(record.layer_weights) == 23
        else:
            assert record.layer_weights is None
