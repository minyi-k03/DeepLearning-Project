"""Seam 1 tests: Execution seam (RunConfig + LayerVectorProvider -> ResultRecord).

All tests run on CPU using FakeLayerVectorProvider without GPU or real datasets.
Un-skip each test as T-05 / T-06 are implemented (SPEC.md §Seam ①).
"""

import pytest


@pytest.mark.skip(reason="TODO(T-05): Enable after fusion heads are implemented")
def test_trainable_parameter_counts_and_b0_plus_matched_to_f4() -> None:
    """Each fusion method matches definition and B0+ matches F4 parameter count."""
    raise NotImplementedError


@pytest.mark.skip(reason="TODO(T-05): Enable after PerLayerNorm and F1 are implemented")
def test_per_layer_norm_makes_f1_invariant_to_single_layer_scaling() -> None:
    """Scaling one layer vector sharply must not alter F1 output thanks to per-layer LN."""
    raise NotImplementedError


@pytest.mark.skip(reason="TODO(ADR 0005 / T-05): Enable after F2 is implemented")
def test_f2_has_no_global_scale_gamma() -> None:
    """F2 has no global scalar scale parameter gamma."""
    raise NotImplementedError


@pytest.mark.skip(reason="TODO(T-05): Enable after F2 subset options are implemented")
def test_f2_global_and_local_subsets_exclude_embedding() -> None:
    """global_only has 8 weights, local_only has 14 weights, embedding excluded in both."""
    raise NotImplementedError


@pytest.mark.skip(reason="TODO(T-03 / T-06): Enable after small-data sampling is wired")
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
