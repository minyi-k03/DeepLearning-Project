"""Seam 2 tests: Judgment seam (ResultRecord[] + DecisionRules -> JudgmentReport).

All tests run on CPU using hand-crafted ResultRecords.
Un-skip each test as T-07 is implemented (SPEC.md §Seam ②).
"""

import pytest


@pytest.mark.skip(reason="TODO(T-07): Enable after judge_hypotheses is implemented")
def test_difference_smaller_than_seed_std_is_no_effect() -> None:
    """If error-rate difference < seed std, verdict must be NO_EFFECT."""
    raise NotImplementedError


@pytest.mark.skip(reason="TODO(T-07): Enable after judge_hypotheses is implemented")
def test_non_significant_mcnemar_is_no_effect_even_if_above_seed_std() -> None:
    """If difference >= seed std but McNemar is not significant, verdict must be NO_EFFECT."""
    raise NotImplementedError


@pytest.mark.skip(reason="TODO(T-07): Enable after judge_hypotheses is implemented")
def test_passes_both_std_and_mcnemar_is_supported() -> None:
    """Passing both seed-std and paired McNemar thresholds yields SUPPORTED."""
    raise NotImplementedError


@pytest.mark.skip(reason="TODO(T-07): Enable after compute_paired_mcnemar is implemented")
def test_mcnemar_pairs_predictions_by_doc_id() -> None:
    """Reordering predictions by doc_id gives the exact same McNemar p-value."""
    raise NotImplementedError


@pytest.mark.skip(reason="TODO(T-07): Enable after judge_hypotheses is implemented")
def test_debug_and_reproduction_records_are_excluded_from_judgment() -> None:
    """Records with run_kind=debug or track=reproduction must not affect verdicts."""
    raise NotImplementedError


@pytest.mark.skip(reason="TODO(T-07): Enable after judge_hypotheses is implemented")
def test_exploratory_hypotheses_receive_no_confirmatory_verdict() -> None:
    """H4 and H5c must have Verdict.EXPLORATORY rather than SUPPORTED/NO_EFFECT."""
    raise NotImplementedError


@pytest.mark.skip(reason="TODO(T-07): Enable after seed consistency check is implemented")
def test_low_seed_consistency_holds_layer_weight_interpretation() -> None:
    """When seed consistency fails thresholds, layer-weight interpretation is marked HELD."""
    raise NotImplementedError


@pytest.mark.skip(reason="TODO(T-07): Enable after CoG and global ratio are implemented")
def test_center_of_gravity_and_global_ratio_match_hand_calculation() -> None:
    """All weight on layer 22 => CoG == 22.0 and global_ratio == 1.0."""
    raise NotImplementedError


@pytest.mark.skip(reason="TODO(T-07): Enable after H5b decomposition is implemented")
def test_h5b_report_splits_mixing_effect_and_weight_learning_effect() -> None:
    """H5b details must expose F1 - B0 (mixing) and F2 - F1 (weight learning)."""
    raise NotImplementedError
