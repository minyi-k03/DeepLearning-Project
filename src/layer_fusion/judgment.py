"""Judgment seam: pure hypothesis evaluation from ResultRecords + DecisionRules.

References: docs/SPEC.md, ADR 0007.
"""

from __future__ import annotations

from collections.abc import Sequence

from layer_fusion.contracts import (
    DecisionRules,
    JudgmentReport,
    PredictionRecord,
    ResultRecord,
)


def load_decision_rules(path: str) -> DecisionRules:
    """Load frozen decision rules from configs/decision_rules.yaml."""
    raise NotImplementedError("TODO: Implement decision rules loader (T-07)")


def compute_paired_mcnemar(
    preds_a: Sequence[PredictionRecord],
    preds_b: Sequence[PredictionRecord],
) -> float:
    """Compute paired McNemar p-value aligned by doc_id."""
    raise NotImplementedError("TODO: Implement paired McNemar test (T-07)")


def compute_center_of_gravity(
    weights: Sequence[float],
    include_embedding: bool,
) -> float:
    """Compute center of gravity (무게중심) over layer weights."""
    raise NotImplementedError("TODO: Implement center of gravity calculation (T-07)")


def compute_global_layer_weight_ratio(
    weights: Sequence[float],
    include_embedding_in_denom: bool,
) -> float:
    """Compute global layer weight share (전역 층 가중치 비율)."""
    raise NotImplementedError("TODO: Implement global layer weight ratio (T-07)")


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
    """
    raise NotImplementedError("TODO: Implement judgment seam (T-07)")
