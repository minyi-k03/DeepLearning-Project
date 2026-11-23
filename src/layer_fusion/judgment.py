"""Judgment seam: pure hypothesis evaluation from ResultRecords + DecisionRules.

References: docs/SPEC.md, ADR 0007.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Sequence
from dataclasses import fields
from typing import Any

import yaml
from scipy.stats import binomtest

from layer_fusion.contracts import (
    BackboneCondition,
    DataAmount,
    DatasetName,
    DecisionRules,
    FusionMethod,
    HypothesisJudgment,
    JudgmentReport,
    LayerSubset,
    PredictionRecord,
    ResultRecord,
    RunConfig,
    RunKind,
    TrackKind,
    Verdict,
)

_CRITERIA_FIELDS = (
    "mcnemar_alpha",
    "multiple_comparison_correction",
    "multi_seed_mcnemar_aggregation",
    "min_bucket_samples",
    "below_min_sample_handling",
    "core_num_seeds",
    "secondary_num_seeds",
    "core_comparisons_for_5_seeds",
    "finetune_num_seeds",
    "best_fusion_selection_rule",
)
_LAYER_WEIGHT_FIELDS = (
    "include_embedding_in_cog",
    "include_embedding_in_global_denom",
    "seed_rank_corr_min",
    "cog_seed_std_max",
)
_REQUIRED_RULE_KEYS = ("freeze_date", "status", "core_num_seeds", "secondary_num_seeds")


def load_decision_rules(path: str, *, require_locked: bool = False) -> DecisionRules:
    """Load frozen decision rules from configs/decision_rules.yaml.

    Values are taken as written: a ``null`` (TBD until the 11/30 lock) stays ``None`` and is
    never filled with a guess (AGENTS.md §2). The per-hypothesis ``type``/``priority`` block
    is kept as ``rules.extra["hypotheses"]`` so judgment never hard-codes which hypotheses
    are confirmatory (ADR 0007). ``require_locked=True`` refuses a file whose status is not
    ``locked``; use it for judging main-experiment results.
    """
    with open(path, encoding="utf-8") as handle:
        doc = yaml.safe_load(handle)
    if not isinstance(doc, dict):
        raise ValueError(f"{path}: decision rules must be a YAML mapping")

    criteria = doc.get("criteria") or {}
    layer_weights = doc.get("layer_weight_interpretation") or {}
    hypotheses = doc.get("hypotheses") or {}

    values: dict[str, Any] = {key: doc.get(key) for key in ("freeze_date", "status")}
    values.update({key: criteria.get(key) for key in _CRITERIA_FIELDS})
    values.update({key: layer_weights.get(key) for key in _LAYER_WEIGHT_FIELDS})
    missing = [key for key in _REQUIRED_RULE_KEYS if values.get(key) is None]
    if missing:
        raise ValueError(f"{path}: required decision rule keys missing: {missing}")
    if require_locked and values["status"] != "locked":
        raise ValueError(
            f"{path}: status is {values['status']!r}; main-experiment judgment needs 'locked'"
        )

    extra: dict[str, Any] = {"hypotheses": hypotheses}
    known = set(_CRITERIA_FIELDS) | set(_LAYER_WEIGHT_FIELDS)
    for block in (criteria, layer_weights):
        extra.update({key: val for key, val in block.items() if key not in known})

    return DecisionRules(
        h1_target_fusion_rule=(hypotheses.get("H1") or {}).get("target_fusion_rule"),
        h5a_target_fusion_rule=(hypotheses.get("H5a") or {}).get("target_fusion_rule"),
        extra=extra,
        **values,
    )


def compute_paired_mcnemar(
    preds_a: Sequence[PredictionRecord],
    preds_b: Sequence[PredictionRecord],
) -> float:
    """Compute paired McNemar p-value aligned by doc_id.

    Two-sided exact binomial test on the discordant pairs (b: only A correct, c: only B
    correct). Documents are paired by ``doc_id``, never by position. With no discordant
    pairs there is no evidence of a difference, so the p-value is 1.0.
    """
    by_id_a = _index_by_doc_id(preds_a, "preds_a")
    by_id_b = _index_by_doc_id(preds_b, "preds_b")
    if not by_id_a or not by_id_b:
        # Without this, two empty runs would give p = 1.0 and read as NO_EFFECT.
        raise ValueError("McNemar needs at least one document on each side")
    if by_id_a.keys() != by_id_b.keys():
        only_a = sorted(by_id_a.keys() - by_id_b.keys())[:3]
        only_b = sorted(by_id_b.keys() - by_id_a.keys())[:3]
        raise ValueError(
            "McNemar needs the same documents on both sides; "
            f"only in A: {only_a}, only in B: {only_b} (first 3 shown)"
        )

    only_a_correct = 0
    only_b_correct = 0
    for doc_id, rec_a in by_id_a.items():
        rec_b = by_id_b[doc_id]
        if rec_a.label != rec_b.label:
            raise ValueError(
                f"doc_id {doc_id!r} has different labels ({rec_a.label} vs {rec_b.label})"
            )
        a_ok = rec_a.prediction == rec_a.label
        b_ok = rec_b.prediction == rec_b.label
        if a_ok and not b_ok:
            only_a_correct += 1
        elif b_ok and not a_ok:
            only_b_correct += 1

    discordant = only_a_correct + only_b_correct
    if discordant == 0:
        return 1.0
    result = binomtest(min(only_a_correct, only_b_correct), discordant, 0.5)
    return float(result.pvalue)


def _index_by_doc_id(preds: Sequence[PredictionRecord], name: str) -> dict[str, PredictionRecord]:
    indexed: dict[str, PredictionRecord] = {}
    for rec in preds:
        if rec.doc_id in indexed:
            raise ValueError(f"{name} has duplicate doc_id {rec.doc_id!r}")
        indexed[rec.doc_id] = rec
    return indexed


def compute_center_of_gravity(
    weights: Sequence[float],
    include_embedding: bool,
) -> float:
    """Compute center of gravity (무게중심) over layer weights."""
    raise NotImplementedError("TODO: Implement center of gravity calculation (T-07c)")


def compute_global_layer_weight_ratio(
    weights: Sequence[float],
    include_embedding_in_denom: bool,
) -> float:
    """Compute global layer weight share (전역 층 가중치 비율)."""
    raise NotImplementedError("TODO: Implement global layer weight ratio (T-07c)")


def judge_hypotheses(
    records: Sequence[ResultRecord],
    rules: DecisionRules,
) -> JudgmentReport:
    """Judgment seam: (ResultRecord[], DecisionRules) -> JudgmentReport.

    Contract:
    - Excludes run_kind == DEBUG and track == REPRODUCTION records.
    - Confirmatory hypotheses (H1, H3, H2, H4', H5b, H5a): SUPPORTED only if
      difference >= seed std AND paired McNemar is significant; else NO_EFFECT.
    - Exploratory hypotheses (H4, H5c): EXPLORATORY (no SUPPORTED/NO_EFFECT verdict).
    - Holds layer-weight interpretation if seed consistency fails thresholds.
    - Reports H5b decomposed into mixing effect (F1 - B0) and weight learning (F2 - F1).

    T-07b implements record filtering, the exploratory rows and H1. A confirmatory hypothesis
    with no registered evaluator (H3, H2, H4', H5a, H5b: T-07c) or without the records it
    needs is left out of ``judgments`` rather than reported as NO_EFFECT. Hypothesis kinds come
    from ``rules.extra["hypotheses"]`` (see ``load_decision_rules``), never from this module.
    """
    kinds = _hypothesis_kinds(rules)
    main_records = select_main_records(records)

    judgments: dict[str, HypothesisJudgment] = {}
    for hypothesis_id, kind in kinds.items():
        if kind == "exploratory":
            judgments[hypothesis_id] = HypothesisJudgment(
                hypothesis_id=hypothesis_id,
                kind=kind,
                verdict=Verdict.EXPLORATORY,
                details={"note": "exploratory: trend only, no confirmatory verdict (ADR 0007)"},
            )
            continue
        evaluator = _CONFIRMATORY_EVALUATORS.get(hypothesis_id)
        if evaluator is None:
            continue
        judgment = evaluator(main_records, rules, kind)
        if judgment is not None:
            judgments[hypothesis_id] = judgment
    return JudgmentReport(judgments=judgments)


def select_main_records(records: Sequence[ResultRecord]) -> list[ResultRecord]:
    """Drop debug and reproduction-track records; they never enter judgment (ADR 0004, 0009)."""
    return [
        rec
        for rec in records
        if rec.config.run_kind is not RunKind.DEBUG
        and rec.config.track is not TrackKind.REPRODUCTION
    ]


def seed_mean_std(values: Sequence[float]) -> tuple[float, float]:
    """Mean and sample standard deviation (ddof=1) across seeds; needs at least 2 seeds."""
    if len(values) < 2:
        raise ValueError(f"seed std needs at least 2 seeds, got {len(values)}")
    if not all(math.isfinite(v) for v in values):
        raise ValueError(f"seed values must be finite, got {list(values)}")
    return statistics.fmean(values), statistics.stdev(values)


def _kth_smallest_p(p_values: Sequence[float], k: int) -> float:
    return sorted(p_values)[k - 1]


# Each strategy returns the representative p-value: significant exactly when it is < alpha.
# SPEC leaves the combination rule TBD (fixed at the 11/30 lock, T-07a). These two strategies
# and their names are T-07b implementation choices so the seam can be exercised; T-07a picks
# one (or adds another here). A rules value of None or any other name is refused.
MCNEMAR_SEED_AGGREGATIONS: dict[str, Callable[[Sequence[float]], float]] = {
    # every seed must be significant -> the weakest (largest) p decides
    "all_seeds_significant": lambda ps: _kth_smallest_p(ps, len(ps)),
    # strictly more than half of the seeds must be significant -> the (n//2+1)-th smallest p
    "majority_of_seeds": lambda ps: _kth_smallest_p(ps, len(ps) // 2 + 1),
}


def _hypothesis_kinds(rules: DecisionRules) -> dict[str, str]:
    hypotheses = rules.extra.get("hypotheses")
    if not hypotheses:
        raise ValueError(
            "rules.extra['hypotheses'] is missing; load the rules with load_decision_rules"
        )
    return {hypothesis_id: body["type"] for hypothesis_id, body in hypotheses.items()}


# Inside one B0-vs-candidate comparison only these RunConfig fields may differ. Everything else
# (input setting, epochs, early stopping, lr candidates, cache, ...) must match, because SPEC
# module 7 trains every fusion method with the same epochs and early stopping, and averaging
# records from different cells would silently judge a different experiment. fixed_lr and
# lr_selection_seed differ by design: the first seed of each setting selects the LR and the
# other seeds reuse it (SPEC §결정 필요 사항, T-06a), each fusion method with its own choice.
_COMPARISON_FREE_FIELDS = frozenset({"seed", "fusion_method", "fixed_lr", "lr_selection_seed"})


def _require_one_cell(hypothesis_id: str, records: Sequence[ResultRecord]) -> None:
    reference = records[0]
    for rec in records[1:]:
        differing = [
            f.name
            for f in fields(RunConfig)
            if f.name not in _COMPARISON_FREE_FIELDS
            and getattr(rec.config, f.name) != getattr(reference.config, f.name)
        ]
        if differing:
            raise ValueError(
                f"{hypothesis_id}: records must come from one experiment cell (only "
                f"{sorted(_COMPARISON_FREE_FIELDS)} may vary); {reference.run_id!r} and "
                f"{rec.run_id!r} differ in {differing}"
            )


def _records_by_seed(records: Sequence[ResultRecord], label: str) -> dict[int, ResultRecord]:
    by_seed: dict[int, ResultRecord] = {}
    for rec in records:
        if rec.config.seed in by_seed:
            raise ValueError(f"{label}: more than one record for seed {rec.config.seed}")
        by_seed[rec.config.seed] = rec
    return by_seed


def _judge_comparison(
    hypothesis_id: str,
    kind: str,
    baseline: Sequence[ResultRecord],
    candidate: Sequence[ResultRecord],
    rules: DecisionRules,
) -> HypothesisJudgment:
    """SUPPORTED iff the candidate improves on the baseline by at least the seed std and the
    paired McNemar test is significant; otherwise NO_EFFECT.

    Improvement = mean baseline error rate - mean candidate error rate (positive: candidate
    better). Seed std = the larger sample std of the two settings (T-07b implementation choice;
    SPEC/ADR 0007 only say "seed std"). McNemar is paired per seed (two-sided exact), a seed
    counts only if the candidate has the lower error rate in it, and the per-seed p-values are
    combined by ``rules.multi_seed_mcnemar_aggregation``. ``details["per_seed_mcnemar_p"]``
    keeps the raw two-sided p-values.
    """
    if rules.mcnemar_alpha is None:
        raise ValueError("mcnemar_alpha is TBD (fixed at the 11/30 lock, T-07a)")
    aggregation_name = rules.multi_seed_mcnemar_aggregation
    if aggregation_name is None:
        raise ValueError("multi_seed_mcnemar_aggregation is TBD (fixed at the 11/30 lock, T-07a)")
    if aggregation_name not in MCNEMAR_SEED_AGGREGATIONS:
        raise ValueError(
            f"unknown multi_seed_mcnemar_aggregation {aggregation_name!r}; "
            f"supported: {sorted(MCNEMAR_SEED_AGGREGATIONS)}"
        )

    _require_one_cell(hypothesis_id, [*baseline, *candidate])
    base_by_seed = _records_by_seed(baseline, f"{hypothesis_id} baseline")
    cand_by_seed = _records_by_seed(candidate, f"{hypothesis_id} candidate")
    if base_by_seed.keys() != cand_by_seed.keys():
        raise ValueError(
            f"{hypothesis_id}: baseline seeds {sorted(base_by_seed)} != "
            f"candidate seeds {sorted(cand_by_seed)}"
        )
    seeds = sorted(base_by_seed)

    base_mean, base_std = seed_mean_std([base_by_seed[s].eval_error_rate for s in seeds])
    cand_mean, cand_std = seed_mean_std([cand_by_seed[s].eval_error_rate for s in seeds])
    improvement = base_mean - cand_mean
    seed_std = max(base_std, cand_std)

    per_seed_p = {
        seed: compute_paired_mcnemar(base_by_seed[seed].predictions, cand_by_seed[seed].predictions)
        for seed in seeds
    }
    # McNemar is two-sided, but "candidate beats baseline" is directional: a seed where the
    # candidate is not better cannot support it, however significant the difference is.
    counted_p = [
        per_seed_p[seed]
        if cand_by_seed[seed].eval_error_rate < base_by_seed[seed].eval_error_rate
        else 1.0
        for seed in seeds
    ]
    representative_p = MCNEMAR_SEED_AGGREGATIONS[aggregation_name](counted_p)

    # A negative improvement can never reach the (non-negative) seed std, so a significant
    # difference in the wrong direction is NO_EFFECT without a separate direction check.
    supported = improvement >= seed_std and representative_p < rules.mcnemar_alpha
    return HypothesisJudgment(
        hypothesis_id=hypothesis_id,
        kind=kind,
        verdict=Verdict.SUPPORTED if supported else Verdict.NO_EFFECT,
        mean_diff=improvement,
        seed_std=seed_std,
        mcnemar_p_value=representative_p,
        details={
            "per_seed_mcnemar_p": per_seed_p,
            "aggregation": aggregation_name,
            "alpha": rules.mcnemar_alpha,
            "n_seeds": len(seeds),
            "baseline_mean_error": base_mean,
            "candidate_mean_error": cand_mean,
        },
    )


def _mnli_frozen_full(records: Sequence[ResultRecord], method: FusionMethod) -> list[ResultRecord]:
    return [
        rec
        for rec in records
        if rec.config.dataset is DatasetName.MNLI
        and rec.config.backbone_condition is BackboneCondition.FROZEN
        and rec.config.data_amount is DataAmount.FULL
        and rec.config.layer_subset is LayerSubset.ALL
        and rec.config.fusion_method is method
    ]


def _evaluate_h1(
    records: Sequence[ResultRecord],
    rules: DecisionRules,
    kind: str,
) -> HypothesisJudgment | None:
    """H1: with a frozen backbone, the target fusion beats B0 on MNLI (full data)."""
    baseline = _mnli_frozen_full(records, FusionMethod.B0)
    if not baseline:
        return None
    target = rules.h1_target_fusion_rule
    if target is None:
        raise ValueError("h1_target_fusion_rule is TBD (fixed at the 11/30 lock, T-07a)")
    if not isinstance(target, str):
        raise ValueError(
            f"unsupported h1_target_fusion_rule shape: {target!r} (T-07b takes one name)"
        )
    candidate = _mnli_frozen_full(records, FusionMethod(target))
    if not candidate:
        return None
    return _judge_comparison("H1", kind, baseline, candidate, rules)


_CONFIRMATORY_EVALUATORS: dict[
    str, Callable[[Sequence[ResultRecord], DecisionRules, str], HypothesisJudgment | None]
] = {
    "H1": _evaluate_h1,
}
