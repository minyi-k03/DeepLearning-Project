"""Execution seam, sequential queue runner, and reproduction track runner.

References: docs/SPEC.md, ADR 0004, ADR 0009.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from layer_fusion.contracts import (
    BackboneCondition,
    DataAmount,
    DatasetName,
    DocumentRecord,
    PredictionRecord,
    ResultRecord,
    RunConfig,
    RunKind,
    TrackKind,
)
from layer_fusion.data import (
    DATASET_SPECS,
    TRAIN_SPLIT,
    make_tuning_split,
    sample_small_dataset,
)
from layer_fusion.inputs import assign_length_bucket
from layer_fusion.providers import LayerVectorBatch, LayerVectorProvider, TokenBatch

# SPEC §결정 필요 사항 (T-06a): one tuning split for every seed, drawn with a constant seed.
TUNING_SPLIT_SEED = 0
TUNING_FRACTION = 0.10  # GLOSSARY 튜닝 분할: 학습셋의 10%
SMALL_PER_CLASS = 1000  # GLOSSARY 데이터 양 조건 "소량": 클래스당 1,000건
NUM_MAIN_LR_CANDIDATES = 3  # ADR 0007 튜닝 예산: 학습률 후보 3개
MNLI_GENRES = frozenset({"matched", "mismatched"})


def make_run_id(config: RunConfig) -> str:
    """Deterministic run id built from the setting; also the record file name (T-06b).

    No timestamp is added: one setting + seed + run kind is one record file, so a finished
    run is found by name and is never silently run twice (``records.save_result_record``).

    Reproduction-track ids start with ``repro-`` so they never collide with main-track ids.
    """
    input_setting = config.input_setting.value if config.input_setting else "none"
    prefix = ["repro"] if config.track == TrackKind.REPRODUCTION else []
    return "-".join(
        prefix
        + [
            config.dataset.value,
            config.data_amount.value,
            config.fusion_method.value,
            config.layer_subset.value,
            config.backbone_condition.value,
            input_setting,
            f"s{config.seed}",
            config.run_kind.value,
        ]
    )


def check_training_budget(config: RunConfig) -> None:
    """Reject a RunConfig whose tuning budget is incomplete or inconsistent (ADR 0007).

    Values stay TBD in configs/week1_env.yaml; this only checks that a run states them.
    """
    for name in ("epochs", "batch_size"):
        value = getattr(config, name)
        if value is None or value < 1:
            raise ValueError(f"RunConfig.{name} must be a positive integer, got {value}")
    if config.early_stopping_patience is None or config.early_stopping_patience < 1:
        raise ValueError(
            "RunConfig.early_stopping_patience must be >= 1, "
            f"got {config.early_stopping_patience}"
        )
    candidates = config.lr_candidates
    if not candidates:
        raise ValueError("RunConfig.lr_candidates is empty")
    if len(set(candidates)) != len(candidates) or not all(
        math.isfinite(lr) and lr > 0 for lr in candidates
    ):
        raise ValueError(
            f"lr_candidates must be distinct finite positive numbers, got {candidates}"
        )
    if config.run_kind == RunKind.MAIN and len(candidates) != NUM_MAIN_LR_CANDIDATES:
        raise ValueError(
            f"main runs need exactly {NUM_MAIN_LR_CANDIDATES} lr_candidates (ADR 0007), "
            f"got {len(candidates)}"
        )
    if config.fixed_lr is None:
        if config.lr_selection_seed is not None:
            raise ValueError("lr_selection_seed is only meaningful together with fixed_lr")
        return
    if config.fixed_lr not in candidates:
        raise ValueError(f"fixed_lr {config.fixed_lr} is not one of lr_candidates {candidates}")
    if config.lr_selection_seed is None:
        raise ValueError("fixed_lr needs lr_selection_seed (the seed of the selection run)")


def _check_labels(batch: LayerVectorBatch | TokenBatch, num_classes: int, split: str) -> None:
    labels = np.asarray(batch.labels)
    if labels.size and (labels.min() < 0 or labels.max() >= num_classes):
        raise ValueError(
            f"{split} labels must lie in range({num_classes}), "
            f"got {int(labels.min())}..{int(labels.max())}"
        )


def _check_run_inputs(config: RunConfig, eval_batch: LayerVectorBatch | TokenBatch) -> None:
    """Reject settings and eval splits that would silently drop report fields (PR #15 review)."""
    if config.track == TrackKind.REPRODUCTION:
        raise ValueError(
            "run_experiment is the main-track frozen seam; the reproduction track runs through "
            "run_reproduction_track (ADR 0004)"
        )
    if config.dataset == DatasetName.IMDB:
        if config.input_setting is None:
            raise ValueError("IMDB runs need RunConfig.input_setting (L128/L512/Lfull)")
        if eval_batch.token_lengths is None:
            raise ValueError("IMDB eval split needs token_lengths for the length buckets (US 6)")
    elif config.input_setting is not None:
        raise ValueError(
            f"input_setting applies to IMDB only, got {config.input_setting} for {config.dataset}"
        )
    if config.dataset == DatasetName.MNLI:
        if eval_batch.genre_splits is None:
            raise ValueError("MNLI eval split needs genre_splits (matched / mismatched, US 5)")
        unknown = set(eval_batch.genre_splits) - MNLI_GENRES
        if unknown:
            raise ValueError(
                f"MNLI genre_splits must be in {sorted(MNLI_GENRES)}, got {sorted(unknown)}"
            )


def _check_finetune(config: RunConfig, provider: Any) -> None:
    """The finetune condition is for the MNLI core comparison only (SPEC, ADR 0004)."""
    if not hasattr(provider, "fresh_model"):
        raise ValueError(
            "finetune runs need a BackboneProvider (a cache has no trainable backbone)"
        )
    if config.dataset != DatasetName.MNLI:
        raise ValueError(
            f"the finetune condition is for MNLI only (ADR 0004), got {config.dataset.value}"
        )
    if config.data_amount != DataAmount.FULL:
        raise ValueError("the finetune condition runs on the full MNLI data only (SPEC 실험 행렬)")


@dataclass
class _Candidate:
    """One trained LR candidate of a run."""

    lr: float
    head: Any  # torch.nn.Module (torch is imported lazily)
    outcome: Any  # training.TrainOutcome


def _select_candidate(candidates: list[_Candidate]) -> _Candidate:
    """Lowest (tuning error, tuning loss), then candidate order; diverged candidates excluded."""
    from layer_fusion.training import TrainingDivergedError

    usable = [i for i, c in enumerate(candidates) if not c.outcome.diverged]
    if not usable:
        lrs = [c.lr for c in candidates]
        raise TrainingDivergedError(f"every lr candidate diverged: {lrs}")
    best = min(
        usable,
        key=lambda i: (
            candidates[i].outcome.best_tuning_error,
            candidates[i].outcome.best_tuning_loss,
            i,
        ),
    )
    return candidates[best]


def _training_rows(
    train: LayerVectorBatch | TokenBatch, config: RunConfig, num_classes: int
) -> tuple[np.ndarray, np.ndarray, list[str] | None]:
    """Row indices of (training rows, tuning rows) and the small-sample doc_ids, if any.

    The provider has only ``train``/``eval``; the tuning split is chosen by doc_id inside train
    (PR #11 decision) by reusing data.make_tuning_split with a constant seed. A small sample is
    drawn from the remaining 90% with the run seed, so all fusion methods of a seed share it.
    """
    records = [
        DocumentRecord(doc_id=doc_id, text_a="", label=int(label), split=TRAIN_SPLIT)
        for doc_id, label in zip(train.doc_ids, np.asarray(train.labels), strict=True)
    ]
    train_part, tuning = make_tuning_split(records, TUNING_FRACTION, seed=TUNING_SPLIT_SEED)
    row_of = {doc_id: i for i, doc_id in enumerate(train.doc_ids)}
    small_ids: list[str] | None = None
    if config.data_amount == DataAmount.SMALL:
        train_part, small_ids = sample_small_dataset(
            train_part, SMALL_PER_CLASS, seed=config.seed, num_classes=num_classes
        )
    train_rows = np.sort(np.fromiter((row_of[r.doc_id] for r in train_part), dtype=np.int64))
    tuning_rows = np.sort(np.fromiter((row_of[r.doc_id] for r in tuning), dtype=np.int64))
    return train_rows, tuning_rows, small_ids


def _mnli_genre_error_rates(
    predictions: list[PredictionRecord],
) -> tuple[float, float]:
    """Error rates on the matched and mismatched eval sets (US 5, H4′)."""
    rates: dict[str, float] = {}
    for genre in ("matched", "mismatched"):
        rows = [p for p in predictions if p.genre_split == genre]
        if not rows:
            raise ValueError(f"MNLI eval split has no {genre} documents (genre_splits missing?)")
        rates[genre] = float(np.mean([p.prediction != p.label for p in rows]))
    return rates["matched"], rates["mismatched"]


def _resolve_device(device: Any) -> Any:
    import torch

    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def run_experiment(
    config: RunConfig,
    provider: LayerVectorProvider,
    *,
    device: str | None = None,
) -> ResultRecord:
    """Execution seam: (RunConfig, LayerVectorProvider) -> ResultRecord.

    Frozen backbone: the provider hands out cached layer vectors and only the head trains.
    Finetune (T-06d): a ``BackboneProvider`` hands out token ids and ModernBERT + head train
    together (``finetune.FinetuneModel``); everything below is shared, so both conditions
    produce the same record format. ``trainable_params`` then counts the backbone too.

    Contract (SPEC §결정 필요 사항, T-06a):
    - Validates provider metadata against the RunConfig before any training (US 26).
    - The tuning split is 10% of train, fixed across seeds; training uses the other 90% (or a
      small sample of it). The eval split is never used for LR choice or early stopping.
    - Without ``fixed_lr`` this is a selection run: every lr_candidate is trained with the same
      epochs / early stopping, the lowest tuning error wins (ties: lower tuning loss, then
      candidate order), and that model is reported. Early stopping uses the same order.
      With ``fixed_lr`` only that LR is trained.
    - ``train_seconds`` covers only the reported model's training steps; its per-epoch tuning
      evaluation is ``tuning_eval_seconds`` and the whole LR search is ``lr_search_seconds``.
    - A candidate whose loss turns non-finite is diverged and excluded (score ``inf``); if no
      usable model is left, ``TrainingDivergedError`` is raised.
    - MNLI records carry matched and mismatched error rates.
    """
    # Fail before any training when the provider was built under a different setting (US 26).
    provider.validate_for_run(config)
    check_training_budget(config)

    from layer_fusion.fusion import (
        build_fusion_head,
        count_trainable_parameters,
        extract_layer_weights,
    )
    from layer_fusion.training import RowView, TrainingDivergedError, evaluate, train_head

    finetune = config.backbone_condition == BackboneCondition.FINETUNE
    if finetune:
        _check_finetune(config, provider)
    torch_device = _resolve_device(device)
    num_classes = DATASET_SPECS[config.dataset].num_classes
    train = provider.get_split("train")
    eval_batch = provider.get_split("eval")
    _check_labels(train, num_classes, "train")
    _check_labels(eval_batch, num_classes, "eval")
    _check_run_inputs(config, eval_batch)

    train_rows, tuning_rows, small_ids = _training_rows(train, config, num_classes)
    def make_head() -> Any:
        return build_fusion_head(config.fusion_method, num_classes, config.layer_subset)

    if finetune:
        from layer_fusion.finetune import FinetuneModel, TokenRowView

        pool_special = provider.metadata.include_special_tokens_in_pool

        def view(batch: Any, rows: np.ndarray) -> Any:
            return TokenRowView(
                batch.input_ids, batch.labels, rows,
                provider.pad_id, provider.special_ids, pool_special,
            )

        def make_model() -> Any:
            return FinetuneModel(provider.fresh_model(), make_head())
    else:

        def view(batch: Any, rows: np.ndarray) -> Any:
            return RowView(batch.vectors, batch.labels, rows)

        make_model = make_head

    train_view = view(train, train_rows)
    tuning_view = view(train, tuning_rows)

    def fit(lr: float) -> _Candidate:
        head, outcome = train_head(
            make_model,
            train_view,
            tuning_view,
            lr=lr,
            epochs=config.epochs,
            patience=config.early_stopping_patience,
            batch_size=config.batch_size,
            seed=config.seed,
            device=torch_device,
        )
        # Park the trained model on the CPU: with the finetune condition each candidate is a
        # whole ModernBERT, and keeping all of them on the GPU would multiply its memory use.
        return _Candidate(lr, head.to("cpu"), outcome)

    lr_tuning_scores: dict[float, float] = {}
    lr_search_seconds: float | None = None
    if config.fixed_lr is None:
        candidates = [fit(lr) for lr in config.lr_candidates]
        chosen = _select_candidate(candidates)
        lr_tuning_scores = {
            c.lr: math.inf if c.outcome.diverged else c.outcome.best_tuning_error
            for c in candidates
        }
        lr_search_seconds = sum(c.outcome.seconds for c in candidates)
        lr_selection_seed = config.seed
    else:
        chosen = fit(config.fixed_lr)
        if chosen.outcome.diverged:
            raise TrainingDivergedError(f"training at fixed_lr={config.fixed_lr} diverged")
        lr_selection_seed = config.lr_selection_seed
    selected_lr, outcome = chosen.lr, chosen.outcome
    head = chosen.head.to(torch_device)  # only the reported model goes back to the device
    if config.fixed_lr is None:
        for candidate in candidates:
            candidate.head = None  # drop the other candidates' models

    eval_view = view(eval_batch, np.arange(len(eval_batch.doc_ids), dtype=np.int64))
    eval_loss, eval_error, predicted = evaluate(head, eval_view, config.batch_size, torch_device)
    if not math.isfinite(eval_loss):
        raise ValueError("eval loss is not finite (non-finite eval vectors?)")

    token_lengths = eval_batch.token_lengths
    genres = eval_batch.genre_splits
    labels = np.asarray(eval_batch.labels)
    predictions = [
        PredictionRecord(
            doc_id=doc_id,
            label=int(labels[i]),
            prediction=int(predicted[i]),
            token_length=None if token_lengths is None else int(token_lengths[i]),
            length_bucket=(
                assign_length_bucket(int(token_lengths[i]))
                if token_lengths is not None and config.input_setting is not None
                else None
            ),
            genre_split=None if genres is None else genres[i],
        )
        for i, doc_id in enumerate(eval_batch.doc_ids)
    ]

    matched = mismatched = None
    if config.dataset == DatasetName.MNLI:
        matched, mismatched = _mnli_genre_error_rates(predictions)

    metadata = provider.metadata
    truncated_doc_ratio = None
    if (
        config.input_setting is not None
        and metadata.max_length_cap is not None
        and token_lengths is not None
    ):
        truncated_doc_ratio = float(np.mean(np.asarray(token_lengths) > metadata.max_length_cap))

    return ResultRecord(
        run_id=make_run_id(config),
        config=config,
        selected_lr=selected_lr,
        lr_tuning_scores=lr_tuning_scores,
        eval_error_rate=eval_error,
        predictions=predictions,
        trainable_params=count_trainable_parameters(head),
        train_seconds=outcome.train_seconds,
        learning_curve=outcome.learning_curve,
        eval_error_rate_matched=matched,
        eval_error_rate_mismatched=mismatched,
        layer_weights=extract_layer_weights(head.head if finetune else head),
        small_sample_doc_ids=small_ids,
        transformers_version=metadata.transformers_version,
        attn_implementation=metadata.attn_implementation,
        torch_compile=metadata.torch_compile,
        truncated_doc_ratio=truncated_doc_ratio,
        lr_selection_seed=lr_selection_seed,
        lr_search_seconds=lr_search_seconds,
        tuning_eval_seconds=outcome.tuning_eval_seconds,
    )


def run_reproduction_track(
    dataset: str,
    seeds: Sequence[int],
    output_dir: str,
) -> list[ResultRecord]:
    """Run paper fine-tuning reproduction track (MNLI 89.1 / SST-2 96.0, x3 runs, ADR 0004)."""
    raise NotImplementedError("TODO: Implement reproduction track runner (T-06/T-08)")
