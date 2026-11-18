"""Typed data contracts matching SPEC.md and GLOSSARY.md."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class DatasetName(str, Enum):
    """Supported datasets and AG News fallback (ADR 0003)."""

    MNLI = "mnli"
    SST2 = "sst2"
    IMDB = "imdb"
    AG_NEWS = "ag_news"


class DataAmount(str, Enum):
    """Dataset size condition (GLOSSARY: 데이터 양 조건)."""

    FULL = "full"
    SMALL = "small"  # 1,000 examples per class


class FusionMethod(str, Enum):
    """Layer fusion rules (GLOSSARY: 결합 방식)."""

    B0 = "B0"
    F1 = "F1"
    F2 = "F2"
    F3 = "F3"
    F4 = "F4"
    B0_PLUS = "B0+"


class LayerSubset(str, Enum):
    """Layer subset option for F2 ablation (ADR 0006)."""

    ALL = "all"
    GLOBAL_ONLY = "global_only"
    LOCAL_ONLY = "local_only"


class BackboneCondition(str, Enum):
    """Backbone training mode (ADR 0004)."""

    FROZEN = "frozen"
    FINETUNE = "finetune"


class InputSetting(str, Enum):
    """IMDB input truncation settings (ADR 0006)."""

    L128 = "L128"
    L512 = "L512"
    LFULL = "Lfull"


class RunKind(str, Enum):
    """Execution classification (ADR 0009)."""

    MAIN = "main"
    DEBUG = "debug"


class TrackKind(str, Enum):
    """Experimental track separation (ADR 0004)."""

    MAIN = "main"
    REPRODUCTION = "reproduction"


class LengthBucket(str, Enum):
    """Token length bucket for IMDB error reporting (ADR 0006)."""

    LE_512 = "<=512"
    BETWEEN_512_1024 = "512-1024"
    GT_1024 = ">1024"


class Verdict(str, Enum):
    """Confirmatory / exploratory hypothesis verdict."""

    SUPPORTED = "지지"
    NO_EFFECT = "효과 없음"
    EXPLORATORY = "탐색적(판정 없음)"
    HELD = "해석 보류"


@dataclass(frozen=True)
class DocumentRecord:
    """Single dataset example with stable doc_id."""

    doc_id: str
    text_a: str
    label: int
    text_b: str | None = None
    split: str = "train"
    genre_split: str | None = None  # e.g. "matched" | "mismatched" for MNLI


@dataclass(frozen=True)
class CacheMetadata:
    """Metadata stored alongside extracted layer-vector caches (SPEC.md §데이터 형태)."""

    model_name: str
    transformers_version: str
    attn_implementation: str
    torch_compile: bool
    input_setting: InputSetting | None
    max_length_cap: int | None
    truncation_rule: str
    include_special_tokens_in_pool: bool | None = None
    dtype: str = "float32"


@dataclass(frozen=True)
class RunConfig:
    """Execution configuration passed into the Execution seam (SPEC.md §데이터 형태)."""

    dataset: DatasetName
    data_amount: DataAmount
    fusion_method: FusionMethod
    backbone_condition: BackboneCondition
    seed: int
    layer_subset: LayerSubset = LayerSubset.ALL
    input_setting: InputSetting | None = None
    lr_candidates: tuple[float, ...] = ()
    epochs: int | None = None
    early_stopping_patience: int | None = None
    batch_size: int | None = None
    run_kind: RunKind = RunKind.MAIN
    track: TrackKind = TrackKind.MAIN
    expected_cache_meta: CacheMetadata | None = None


@dataclass(frozen=True)
class PredictionRecord:
    """Per-document prediction used for paired McNemar and length-bucket analysis."""

    doc_id: str
    label: int
    prediction: int
    token_length: int | None = None
    length_bucket: LengthBucket | None = None
    genre_split: str | None = None  # "matched" | "mismatched" for MNLI
    layer_weights: list[float] | None = None  # populated for F4


@dataclass
class ResultRecord:
    """Output produced by a single execution at the Execution seam."""

    run_id: str
    config: RunConfig
    selected_lr: float
    lr_tuning_scores: dict[float, float]
    eval_error_rate: float
    predictions: list[PredictionRecord]
    trainable_params: int
    train_seconds: float
    learning_curve: list[dict[str, float]] = field(default_factory=list)
    eval_error_rate_matched: float | None = None
    eval_error_rate_mismatched: float | None = None
    layer_weights: list[float] | None = None  # dataset-level weights for F2
    small_sample_doc_ids: list[str] | None = None
    transformers_version: str = ""
    attn_implementation: str = ""
    torch_compile: bool | None = None
    truncated_doc_ratio: float | None = None


@dataclass
class DecisionRules:
    """Judgment rules loaded from configs/decision_rules.yaml (locked 11/30)."""

    freeze_date: str
    status: str
    mcnemar_alpha: float | None
    multiple_comparison_correction: str | None
    multi_seed_mcnemar_aggregation: str | None
    min_bucket_samples: int | None
    below_min_sample_handling: str | None
    core_num_seeds: int
    secondary_num_seeds: int
    include_embedding_in_cog: bool | None
    include_embedding_in_global_denom: bool | None
    seed_rank_corr_min: float | None
    cog_seed_std_max: float | None
    # TBD until 11/30 lock (T-07a): values stay None (AGENTS.md §2). The types below are
    # provisional placeholders, not decisions; T-07a/T-07b narrow or replace them.
    h1_target_fusion_rule: str | None = None
    h5a_target_fusion_rule: str | None = None
    core_comparisons_for_5_seeds: list[str] | None = None
    finetune_num_seeds: int | None = None
    best_fusion_selection_rule: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class HypothesisJudgment:
    """Single row of the hypothesis judgment table."""

    hypothesis_id: str
    kind: str  # "confirmatory" | "exploratory"
    verdict: Verdict
    mean_diff: float | None = None
    seed_std: float | None = None
    mcnemar_p_value: float | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class JudgmentReport:
    """Output produced by the Judgment seam."""

    judgments: dict[str, HypothesisJudgment]
    summary_rows: list[dict[str, Any]] = field(default_factory=list)
    layer_weight_analysis: dict[str, Any] = field(default_factory=dict)
    length_bucket_analysis: dict[str, Any] = field(default_factory=dict)
