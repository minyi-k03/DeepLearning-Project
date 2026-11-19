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


@pytest.mark.skip(
    reason="TODO(T-06a): Enable when run_experiment wires DataAmount.SMALL into ResultRecord "
    "(data-level sampling is covered by tests/test_data.py)"
)
def test_small_data_sampling_shared_within_seed_and_varies_across_seeds() -> None:
    """1,000 per class; same doc_ids per seed across fusion methods, distinct across seeds."""
    raise NotImplementedError


@pytest.mark.skip(reason="TODO(T-06): Enable after runner is implemented")
def test_identical_tuning_budget_across_fusion_methods() -> None:
    """All fusion methods share number of LR candidates, epochs, and early stopping criteria."""
    raise NotImplementedError


@pytest.mark.skip(reason="TODO(T-06): Enable after runner is implemented")
def test_lr_selection_depends_only_on_tuning_split_not_eval_labels() -> None:
    """Changing eval split labels must not alter the selected learning rate."""
    raise NotImplementedError


@pytest.mark.skip(reason="TODO(T-06): Enable after MNLI eval reporting is implemented")
def test_mnli_result_record_includes_matched_and_mismatched_error_rates() -> None:
    """MNLI ResultRecord exposes both eval_error_rate_matched and eval_error_rate_mismatched."""
    raise NotImplementedError


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


@pytest.mark.skip(reason="TODO(T-06): Enable after runner is implemented")
def test_result_record_contains_required_telemetry_fields() -> None:
    """ResultRecord has trainable_params, train_seconds, learning_curve, predictions, run_kind."""
    raise NotImplementedError
