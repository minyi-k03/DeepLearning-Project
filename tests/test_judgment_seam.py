"""Seam 2 tests: Judgment seam (ResultRecord[] + DecisionRules -> JudgmentReport).

All tests run on CPU using hand-crafted ResultRecords.
Un-skip each test as T-07 is implemented (SPEC.md §Seam ②).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path

import pytest

from layer_fusion.contracts import (
    BackboneCondition,
    DataAmount,
    DatasetName,
    DecisionRules,
    FusionMethod,
    InputSetting,
    PredictionRecord,
    ResultRecord,
    RunConfig,
    RunKind,
    TrackKind,
    Verdict,
)
from layer_fusion.judgment import compute_paired_mcnemar, judge_hypotheses, load_decision_rules

RULES_PATH = Path(__file__).resolve().parent.parent / "configs" / "decision_rules.yaml"


def make_rules(**overrides: object) -> DecisionRules:
    """Real (draft) rules file with synthetic values for the still-TBD thresholds.

    The seam takes rule values as arguments (T-07b); none of these numbers are decisions.
    """
    base = replace(
        load_decision_rules(str(RULES_PATH)),
        mcnemar_alpha=0.05,
        multi_seed_mcnemar_aggregation="all_seeds_significant",
        h1_target_fusion_rule="F1",
    )
    return replace(base, **overrides)


def make_result(
    method: FusionMethod,
    seed: int,
    error_docs: Iterable[int],
    *,
    n_docs: int,
    run_kind: RunKind = RunKind.MAIN,
    track: TrackKind = TrackKind.MAIN,
) -> ResultRecord:
    """MNLI / frozen / full ResultRecord whose wrong predictions are exactly ``error_docs``."""
    wrong = set(error_docs)
    predictions = [
        PredictionRecord(doc_id=f"mnli-{i:05d}", label=1, prediction=0 if i in wrong else 1)
        for i in range(n_docs)
    ]
    config = RunConfig(
        dataset=DatasetName.MNLI,
        data_amount=DataAmount.FULL,
        fusion_method=method,
        backbone_condition=BackboneCondition.FROZEN,
        seed=seed,
        run_kind=run_kind,
        track=track,
    )
    return ResultRecord(
        run_id=f"{method.value}-{run_kind.value}-{track.value}-s{seed}",
        config=config,
        selected_lr=1e-3,
        lr_tuning_scores={1e-3: 0.0},
        eval_error_rate=len(wrong) / n_docs,
        predictions=predictions,
        trainable_params=0,
        train_seconds=0.0,
    )


def make_pair_records(
    baseline_errors: dict[int, set[int]],
    candidate_errors: dict[int, set[int]],
    *,
    n_docs: int,
) -> list[ResultRecord]:
    """B0 (baseline) and F1 (candidate) records, one per seed key."""
    records = [
        make_result(FusionMethod.B0, seed, errors, n_docs=n_docs)
        for seed, errors in baseline_errors.items()
    ]
    records += [
        make_result(FusionMethod.F1, seed, errors, n_docs=n_docs)
        for seed, errors in candidate_errors.items()
    ]
    return records


def clear_improvement_records() -> tuple[list[ResultRecord], int]:
    """3 seeds: B0 errs on 60+s of 200 docs, F1 only on the first 40+s of those (-10 points)."""
    n_docs = 200
    baseline = {s: set(range(60 + s)) for s in range(3)}
    candidate = {s: set(range(40 + s)) for s in range(3)}
    return make_pair_records(baseline, candidate, n_docs=n_docs), n_docs


def test_difference_smaller_than_seed_std_is_no_effect() -> None:
    """If error-rate difference < seed std, verdict must be NO_EFFECT."""
    # 2,000 docs. B0 error is 600/800/1000 docs across seeds (std 0.10); F1 fixes the same
    # 50 docs in every seed (-2.5 points, below the std). The 50 one-way flips are far past
    # McNemar significance, so only the seed-std rule can produce NO_EFFECT here.
    n_docs = 2000
    baseline = {s: set(range(count)) for s, count in enumerate((600, 800, 1000))}
    candidate = {s: set(range(50, count)) for s, count in enumerate((600, 800, 1000))}
    records = make_pair_records(baseline, candidate, n_docs=n_docs)

    h1 = judge_hypotheses(records, make_rules()).judgments["H1"]

    assert h1.mean_diff == pytest.approx(0.025)
    assert h1.seed_std == pytest.approx(0.10)
    assert h1.mcnemar_p_value is not None and h1.mcnemar_p_value < 0.05
    assert h1.verdict == Verdict.NO_EFFECT


def test_non_significant_mcnemar_is_no_effect_even_if_above_seed_std() -> None:
    """If difference >= seed std but McNemar is not significant, verdict must be NO_EFFECT."""
    # 200 docs per seed: 40 docs both models miss, 40 only B0 misses, 34 only F1 misses
    # (-3 points). The disagreements go both ways (40 vs 34), so McNemar cannot call it. Seed
    # jitter (s extra shared errors) keeps the seed std at 0.005, well below the 3 points.
    n_docs = 200
    baseline: dict[int, set[int]] = {}
    candidate: dict[int, set[int]] = {}
    for s in range(3):
        shared = set(range(40)) | set(range(150, 150 + s))
        baseline[s] = shared | set(range(40, 80))
        candidate[s] = shared | set(range(80, 114))
    records = make_pair_records(baseline, candidate, n_docs=n_docs)

    h1 = judge_hypotheses(records, make_rules()).judgments["H1"]

    assert h1.mean_diff == pytest.approx(0.03)
    assert h1.seed_std == pytest.approx(0.005)
    assert h1.mean_diff >= h1.seed_std
    assert h1.mcnemar_p_value is not None and h1.mcnemar_p_value >= 0.05
    assert h1.verdict == Verdict.NO_EFFECT


def test_passes_both_std_and_mcnemar_is_supported() -> None:
    """Passing both seed-std and paired McNemar thresholds yields SUPPORTED."""
    records, _ = clear_improvement_records()

    report = judge_hypotheses(records, make_rules())

    h1 = report.judgments["H1"]
    assert h1.kind == "confirmatory"
    assert h1.verdict == Verdict.SUPPORTED
    # mean B0 error 0.305, mean F1 error 0.205 -> improvement 0.10, seed std 0.005
    assert h1.mean_diff == pytest.approx(0.10)
    assert h1.seed_std == pytest.approx(0.005)
    assert h1.mcnemar_p_value is not None and h1.mcnemar_p_value < 0.05


def test_mcnemar_pairs_predictions_by_doc_id() -> None:
    """Reordering predictions by doc_id gives the exact same McNemar p-value."""
    # 12 documents: A and B disagree on d00..d08 (A right on 7, B right on 2), agree elsewhere.
    a_correct = [True] * 7 + [False] * 2 + [True, True, False]
    b_correct = [False] * 7 + [True] * 2 + [True, True, False]
    preds_a = [
        PredictionRecord(doc_id=f"d{i:02d}", label=1, prediction=1 if ok else 0)
        for i, ok in enumerate(a_correct)
    ]
    preds_b = [
        PredictionRecord(doc_id=f"d{i:02d}", label=1, prediction=1 if ok else 0)
        for i, ok in enumerate(b_correct)
    ]
    reference = compute_paired_mcnemar(preds_a, preds_b)
    assert 0.0 < reference < 1.0

    shuffled_b = [preds_b[i] for i in (5, 11, 0, 8, 3, 9, 1, 10, 7, 2, 6, 4)]
    assert shuffled_b != preds_b
    assert compute_paired_mcnemar(preds_a, shuffled_b) == reference
    assert compute_paired_mcnemar(list(reversed(preds_a)), shuffled_b) == reference


def _noise_records(
    *, run_kind: RunKind, track: TrackKind, seeds: tuple[int, ...]
) -> list[ResultRecord]:
    """B0 / F1 records where F1 is much WORSE than B0 - would flip H1 if they were counted."""
    n_docs = 200
    out = []
    for seed in seeds:
        out.append(
            make_result(
                FusionMethod.B0, seed, range(20), n_docs=n_docs, run_kind=run_kind, track=track
            )
        )
        out.append(
            make_result(
                FusionMethod.F1, seed, range(120), n_docs=n_docs, run_kind=run_kind, track=track
            )
        )
    return out


def test_debug_and_reproduction_records_are_excluded_from_judgment() -> None:
    """Records with run_kind=debug or track=reproduction must not affect verdicts."""
    records, _ = clear_improvement_records()
    reference = judge_hypotheses(records, make_rules()).judgments["H1"]

    noisy = (
        records
        + _noise_records(run_kind=RunKind.DEBUG, track=TrackKind.MAIN, seeds=(10, 11))
        + _noise_records(run_kind=RunKind.MAIN, track=TrackKind.REPRODUCTION, seeds=(20, 21))
    )
    with_noise = judge_hypotheses(noisy, make_rules()).judgments["H1"]

    assert with_noise.verdict == reference.verdict == Verdict.SUPPORTED
    assert with_noise.mean_diff == reference.mean_diff
    assert with_noise.seed_std == reference.seed_std
    assert with_noise.mcnemar_p_value == reference.mcnemar_p_value
    assert with_noise.details["n_seeds"] == 3

    # Debug / reproduction records alone give nothing to judge: no H1 row, not a NO_EFFECT.
    only_excluded = _noise_records(
        run_kind=RunKind.DEBUG, track=TrackKind.MAIN, seeds=(10, 11)
    ) + _noise_records(run_kind=RunKind.MAIN, track=TrackKind.REPRODUCTION, seeds=(20, 21))
    assert "H1" not in judge_hypotheses(only_excluded, make_rules()).judgments


def test_exploratory_hypotheses_receive_no_confirmatory_verdict() -> None:
    """H4 and H5c must have Verdict.EXPLORATORY rather than SUPPORTED/NO_EFFECT."""
    records, _ = clear_improvement_records()

    for data in (records, []):
        judgments = judge_hypotheses(data, make_rules()).judgments
        for hypothesis_id in ("H4", "H5c"):
            row = judgments[hypothesis_id]
            assert row.kind == "exploratory"
            assert row.verdict == Verdict.EXPLORATORY
            assert row.mean_diff is None and row.mcnemar_p_value is None
        exploratory_verdicts = {j.verdict for j in judgments.values() if j.kind == "exploratory"}
        assert exploratory_verdicts == {Verdict.EXPLORATORY}


# --- T-07b: rule handling around the seam (TBD values are refused, never guessed) ---------


@pytest.mark.parametrize("noisy_side", ["candidate", "baseline"])
def test_seed_std_is_the_larger_of_the_two_settings(noisy_side: str) -> None:
    """The std compared with the difference is max(std_B0, std_candidate), whichever is noisy."""
    n_docs = 200
    stable_count = 60 if noisy_side == "candidate" else 50
    stable = {s: set(range(stable_count)) for s in range(3)}
    noisy_counts = (30, 40, 50) if noisy_side == "candidate" else (60, 70, 80)
    noisy = {s: set(range(count)) for s, count in enumerate(noisy_counts)}
    baseline, candidate = (stable, noisy) if noisy_side == "candidate" else (noisy, stable)

    h1 = judge_hypotheses(
        make_pair_records(baseline, candidate, n_docs=n_docs), make_rules()
    ).judgments["H1"]

    # noisy setting: 15/20/25% (candidate) or 30/35/40% (baseline) -> sample std 0.05, the other
    # setting is constant (std 0). Mean errors: 30% vs 20%, or 35% vs 25% -> improvement 0.10.
    assert h1.seed_std == pytest.approx(0.05)
    assert h1.mean_diff == pytest.approx(0.10)


def test_a_seed_where_the_candidate_is_significantly_worse_does_not_count_as_significant() -> None:
    """McNemar is two-sided; for 'candidate beats B0' only seeds where it is better count."""
    n_docs = 200
    seeds = range(5)
    baseline = {s: set(range(60)) for s in seeds}
    # F1 fixes 40 docs in seeds 0, 1, 3, 4 (error 10%) but breaks 8 more in seed 2 (error 34%).
    candidate = {s: set(range(68)) if s == 2 else set(range(20)) for s in seeds}
    records = make_pair_records(baseline, candidate, n_docs=n_docs)

    h1 = judge_hypotheses(records, make_rules()).judgments["H1"]

    # mean improvement 0.152 clears the seed std (0.1075) and every raw per-seed McNemar p is
    # below 0.05 - but seed 2 is significant in the WRONG direction, so not every seed supports.
    assert h1.mean_diff >= h1.seed_std
    assert all(p < 0.05 for p in h1.details["per_seed_mcnemar_p"].values())
    assert h1.verdict == Verdict.NO_EFFECT
    assert h1.mcnemar_p_value == 1.0


def test_candidate_worse_than_baseline_is_never_supported() -> None:
    """A significant difference in the wrong direction must not read as support."""
    records, _ = clear_improvement_records()
    swapped = [
        replace(
            rec,
            config=replace(
                rec.config,
                fusion_method=FusionMethod.F1
                if rec.config.fusion_method is FusionMethod.B0
                else FusionMethod.B0,
            ),
        )
        for rec in records
    ]

    h1 = judge_hypotheses(swapped, make_rules()).judgments["H1"]

    assert h1.mean_diff == pytest.approx(-0.10)
    assert all(p < 0.05 for p in h1.details["per_seed_mcnemar_p"].values())  # raw, two-sided
    assert h1.mcnemar_p_value == 1.0  # no seed has the candidate ahead
    assert h1.verdict == Verdict.NO_EFFECT


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"mcnemar_alpha": None}, "mcnemar_alpha is TBD"),
        ({"multi_seed_mcnemar_aggregation": None}, "multi_seed_mcnemar_aggregation is TBD"),
        ({"multi_seed_mcnemar_aggregation": "fisher"}, "unknown multi_seed_mcnemar_aggregation"),
        ({"h1_target_fusion_rule": None}, "h1_target_fusion_rule is TBD"),
        ({"h1_target_fusion_rule": ["F1", "F2"]}, "unsupported h1_target_fusion_rule"),
    ],
)
def test_unset_or_unknown_rule_values_are_refused(
    override: dict[str, object], message: str
) -> None:
    records, _ = clear_improvement_records()
    with pytest.raises(ValueError, match=message):
        judge_hypotheses(records, make_rules(**override))


def test_majority_aggregation_can_support_where_all_seeds_cannot() -> None:
    """Seeds 0-1 clearly improve, seed 2 improves by 4 docs only (McNemar p = 0.125)."""
    n_docs = 200
    baseline = {s: set(range(60 + s)) for s in range(3)}
    candidate = {0: set(range(40)), 1: set(range(41)), 2: set(range(58))}
    records = make_pair_records(baseline, candidate, n_docs=n_docs)

    strict = judge_hypotheses(records, make_rules()).judgments["H1"]
    majority = judge_hypotheses(
        records, make_rules(multi_seed_mcnemar_aggregation="majority_of_seeds")
    ).judgments["H1"]

    assert strict.mean_diff >= strict.seed_std
    assert strict.mcnemar_p_value == pytest.approx(0.125)
    assert strict.verdict == Verdict.NO_EFFECT
    assert majority.mcnemar_p_value < 0.05
    assert majority.verdict == Verdict.SUPPORTED


def test_seed_sets_must_match_and_two_seeds_are_needed_for_a_std() -> None:
    records, _ = clear_improvement_records()
    without_seed_2 = [
        r for r in records if not (r.config.fusion_method is FusionMethod.F1 and r.config.seed == 2)
    ]
    with pytest.raises(ValueError, match="seeds"):
        judge_hypotheses(without_seed_2, make_rules())

    one_seed = [r for r in records if r.config.seed == 0]
    with pytest.raises(ValueError, match="at least 2 seeds"):
        judge_hypotheses(one_seed, make_rules())


def test_two_records_for_one_seed_are_refused() -> None:
    records, _ = clear_improvement_records()
    with pytest.raises(ValueError, match="more than one record for seed"):
        judge_hypotheses(records + [records[0]], make_rules())


@pytest.mark.parametrize(
    ("config_change", "field"),
    [
        ({"input_setting": InputSetting.L128}, "input_setting"),
        ({"epochs": 7}, "epochs"),
        ({"lr_candidates": (1e-4,)}, "lr_candidates"),
    ],
)
def test_records_from_different_experiment_cells_are_refused(
    config_change: dict[str, object], field: str
) -> None:
    """Only seed and fusion method may vary inside one comparison (SPEC module 7: same epochs
    and early stopping for every fusion method; input setting is IMDB-only). A stray record is
    an error, not something to average in."""
    records, _ = clear_improvement_records()
    stray = records[-1]
    records[-1] = replace(stray, config=replace(stray.config, **config_change))

    with pytest.raises(ValueError, match=f"one experiment cell.*{field}"):
        judge_hypotheses(records, make_rules())


def test_lr_reuse_fields_may_differ_inside_one_cell() -> None:
    """T-06a: the first seed of each setting selects the LR (fixed_lr=None) and the other seeds
    reuse it (fixed_lr + lr_selection_seed), each fusion method with its own choice. Those
    records belong to one cell and must still be judged."""
    records, _ = clear_improvement_records()
    chosen = {FusionMethod.B0: 1e-3, FusionMethod.F1: 3e-4}

    def as_run(rec: ResultRecord) -> ResultRecord:
        reuse = rec.config.seed != 0
        config = replace(
            rec.config,
            lr_candidates=(1e-3, 3e-4, 1e-4),
            fixed_lr=chosen[rec.config.fusion_method] if reuse else None,
            lr_selection_seed=0 if reuse else None,
        )
        return replace(rec, config=config)

    reused = [as_run(rec) for rec in records]

    assert judge_hypotheses(reused, make_rules()).judgments["H1"].verdict == Verdict.SUPPORTED


def test_h1_is_left_out_when_candidate_records_are_missing() -> None:
    """No F1 runs yet is 'not judged', not a NO_EFFECT verdict."""
    records, _ = clear_improvement_records()
    only_b0 = [r for r in records if r.config.fusion_method is FusionMethod.B0]

    assert "H1" not in judge_hypotheses(only_b0, make_rules()).judgments


def test_hypothesis_kinds_must_come_from_the_rules_file() -> None:
    records, _ = clear_improvement_records()
    with pytest.raises(ValueError, match="hypotheses"):
        judge_hypotheses(records, make_rules(extra={}))


@pytest.mark.skip(reason="TODO(T-07c): Enable after seed consistency check is implemented")
def test_low_seed_consistency_holds_layer_weight_interpretation() -> None:
    """When seed consistency fails thresholds, layer-weight interpretation is marked HELD."""
    raise NotImplementedError


@pytest.mark.skip(reason="TODO(T-07c): Enable after CoG and global ratio are implemented")
def test_center_of_gravity_and_global_ratio_match_hand_calculation() -> None:
    """All weight on layer 22 => CoG == 22.0 and global_ratio == 1.0."""
    raise NotImplementedError


@pytest.mark.skip(reason="TODO(T-07c): Enable after H5b decomposition is implemented")
def test_h5b_report_splits_mixing_effect_and_weight_learning_effect() -> None:
    """H5b details must expose F1 - B0 (mixing) and F2 - F1 (weight learning)."""
    raise NotImplementedError
