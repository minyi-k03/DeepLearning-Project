"""Contract conformance (T-01c): contracts.py must carry every field SPEC.md names.

The expected field lists below are transcribed from docs/SPEC.md §데이터 형태(계약),
not derived from contracts.py, so they can disagree with the code.
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import pytest
import yaml

from layer_fusion.contracts import (
    CacheMetadata,
    DecisionRules,
    PredictionRecord,
    ResultRecord,
    RunConfig,
)

ROOT = Path(__file__).resolve().parent.parent
DECISION_RULES_YAML = ROOT / "configs" / "decision_rules.yaml"

# SPEC §데이터 형태 → contract field names.
SPEC_FIELDS: dict[str, tuple[type, set[str]]] = {
    # 실행 설정
    "실행 설정": (
        RunConfig,
        {
            "dataset",
            "data_amount",
            "fusion_method",
            "layer_subset",
            "backbone_condition",
            "input_setting",
            "seed",
            "lr_candidates",
            "epochs",
            "early_stopping_patience",
            "run_kind",
            "track",
            "fixed_lr",  # 고정 학습률과 그 학습률을 고른 시드 (T-06a)
            "lr_selection_seed",
        },
    ),
    # 결과 레코드 (실행 설정은 `config`로 포함)
    "결과 레코드": (
        ResultRecord,
        {
            "config",
            "selected_lr",
            "lr_tuning_scores",
            "eval_error_rate",
            "eval_error_rate_matched",
            "eval_error_rate_mismatched",
            "predictions",
            "trainable_params",
            "train_seconds",
            "learning_curve",
            "layer_weights",  # F2
            "small_sample_doc_ids",
            "transformers_version",
            "attn_implementation",
            "torch_compile",
            "lr_selection_seed",  # 학습률을 고른 시드 (T-06a)
            "lr_search_seconds",  # 학습률 탐색 시간 (T-06a)
        },
    ),
    # 문서 ID별 예측·정답, F4 층 가중치
    "문서별 예측": (
        PredictionRecord,
        {"doc_id", "label", "prediction", "layer_weights"},
    ),
    # 캐시 생성 설정: 모델 · 버전 · 길이(입력 설정, 상한) · 자르기 방식
    "캐시 생성 설정": (
        CacheMetadata,
        {
            "model_name",
            "transformers_version",
            "input_setting",
            "max_length_cap",
            "truncation_rule",
        },
    ),
    # 판정 규칙 문서: 판정 기준 · 최소 표본 기준 · 시드 일관성 기준값
    "판정 규칙 문서": (
        DecisionRules,
        {
            "mcnemar_alpha",
            "multiple_comparison_correction",
            "multi_seed_mcnemar_aggregation",
            "min_bucket_samples",
            "below_min_sample_handling",
            "seed_rank_corr_min",
            "cog_seed_std_max",
        },
    ),
    # SPEC §"11/30 판정 규칙 문서에 고정할 것": 임베딩 벡터 처리, 핵심 비교·파인튜닝 시드,
    # H1·H5a 판정 결합 방식, "가장 좋은 결합 방식" 선정 기준
    "11/30 고정 항목": (
        DecisionRules,
        {
            "include_embedding_in_cog",
            "include_embedding_in_global_denom",
            "core_comparisons_for_5_seeds",
            "finetune_num_seeds",
            "h1_target_fusion_rule",
            "h5a_target_fusion_rule",
            "best_fusion_selection_rule",
        },
    ),
    # 사용자 스토리 16(잘린 문서 비율), 17(문서별 토큰 길이·길이 구간), 53(문서 ID별 예측)
    "문서별 길이(US 16·17)": (PredictionRecord, {"token_length", "length_bucket"}),
    "잘린 문서 비율(US 16)": (ResultRecord, {"truncated_doc_ratio"}),
}


def _field_names(contract: type) -> set[str]:
    return {f.name for f in fields(contract)}


@pytest.mark.parametrize("name", SPEC_FIELDS)
def test_contract_carries_every_spec_field(name: str) -> None:
    contract, expected = SPEC_FIELDS[name]
    missing = expected - _field_names(contract)
    assert not missing, f"{contract.__name__} is missing SPEC fields: {sorted(missing)}"


def _decision_rules_yaml_keys() -> set[str]:
    """Rule keys in decision_rules.yaml, named as DecisionRules fields."""
    doc = yaml.safe_load(DECISION_RULES_YAML.read_text(encoding="utf-8"))
    keys = {"freeze_date", "status"}
    keys |= set(doc["criteria"])
    keys |= set(doc["layer_weight_interpretation"])
    for hypothesis_id, body in doc["hypotheses"].items():
        if "target_fusion_rule" in body:
            keys.add(f"{hypothesis_id.lower()}_target_fusion_rule")
    return keys


def test_every_decision_rules_yaml_key_has_a_typed_field() -> None:
    """Rule keys (criteria, layer_weight_interpretation, target_fusion_rule) need typed fields.

    Per-hypothesis `type`/`priority` are not covered here: DecisionRules has no field for them yet.
    """
    untyped = _decision_rules_yaml_keys() - _field_names(DecisionRules)
    assert not untyped, f"decision_rules.yaml keys with no DecisionRules field: {sorted(untyped)}"


def test_tbd_rule_fields_default_to_none() -> None:
    """TBD rule values stay None until the team fixes them (AGENTS.md §2)."""
    rules = DecisionRules(
        freeze_date="2026-11-30",
        status="draft",
        mcnemar_alpha=None,
        multiple_comparison_correction=None,
        multi_seed_mcnemar_aggregation=None,
        min_bucket_samples=None,
        below_min_sample_handling=None,
        core_num_seeds=5,
        secondary_num_seeds=3,
        include_embedding_in_cog=None,
        include_embedding_in_global_denom=None,
        seed_rank_corr_min=None,
        cog_seed_std_max=None,
    )
    assert rules.h1_target_fusion_rule is None
    assert rules.h5a_target_fusion_rule is None
    assert rules.core_comparisons_for_5_seeds is None
    assert rules.finetune_num_seeds is None
    assert rules.best_fusion_selection_rule is None
