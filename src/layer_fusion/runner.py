"""Execution seam, sequential queue runner, and reproduction track runner.

References: docs/SPEC.md, ADR 0004, ADR 0009.
"""

from __future__ import annotations

from collections.abc import Sequence
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
)
from layer_fusion.data import (
    DATASET_SPECS,
    TRAIN_SPLIT,
    make_tuning_split,
    sample_small_dataset,
)
from layer_fusion.inputs import assign_length_bucket
from layer_fusion.providers import LayerVectorBatch, LayerVectorProvider

# SPEC §결정 필요 사항 (T-06a): one tuning split for every seed, drawn with a constant seed.
TUNING_SPLIT_SEED = 0
TUNING_FRACTION = 0.10  # GLOSSARY 튜닝 분할: 학습셋의 10%
SMALL_PER_CLASS = 1000  # GLOSSARY 데이터 양 조건 "소량": 클래스당 1,000건
NUM_MAIN_LR_CANDIDATES = 3  # ADR 0007 튜닝 예산: 학습률 후보 3개


def make_run_id(config: RunConfig) -> str:
    """Deterministic run id built from the setting (T-06b may add a timestamp to file names)."""
    input_setting = config.input_setting.value if config.input_setting else "none"
    return "-".join(
        [
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
    if config.backbone_condition == BackboneCondition.FINETUNE:
        raise NotImplementedError("finetune condition is implemented in T-06d (BackboneProvider)")
    for name in ("epochs", "batch_size"):
        value = getattr(config, name)
        if value is None or value < 1:
            raise ValueError(f"RunConfig.{name} must be a positive integer, got {value}")
    if config.early_stopping_patience is None or config.early_stopping_patience < 0:
        raise ValueError(
            "RunConfig.early_stopping_patience must be >= 0, "
            f"got {config.early_stopping_patience}"
        )
    candidates = config.lr_candidates
    if not candidates:
        raise ValueError("RunConfig.lr_candidates is empty")
    if len(set(candidates)) != len(candidates) or any(lr <= 0 for lr in candidates):
        raise ValueError(f"lr_candidates must be distinct and positive, got {candidates}")
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


def _check_labels(batch: LayerVectorBatch, num_classes: int, split: str) -> None:
    labels = np.asarray(batch.labels)
    if labels.size and (labels.min() < 0 or labels.max() >= num_classes):
        raise ValueError(
            f"{split} labels must lie in range({num_classes}), "
            f"got {int(labels.min())}..{int(labels.max())}"
        )


def _training_rows(
    train: LayerVectorBatch, config: RunConfig, num_classes: int
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
    """Execution seam: (RunConfig, LayerVectorProvider) -> ResultRecord (frozen backbone).

    Contract (SPEC §결정 필요 사항, T-06a):
    - Validates provider metadata against the RunConfig before any training (US 26).
    - The tuning split is 10% of train, fixed across seeds; training uses the other 90% (or a
      small sample of it). The eval split is never used for LR choice or early stopping.
    - Without ``fixed_lr`` this is a selection run: every lr_candidate is trained with the same
      epochs / early stopping, the lowest tuning error wins (ties: lower tuning loss, then
      candidate order), and that model is reported. With ``fixed_lr`` only that LR is trained.
    - ``train_seconds`` covers only the reported model's training loop; the whole LR search is
      ``lr_search_seconds``.
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
    from layer_fusion.training import RowView, evaluate, train_head

    torch_device = _resolve_device(device)
    num_classes = DATASET_SPECS[config.dataset].num_classes
    train = provider.get_split("train")
    eval_batch = provider.get_split("eval")
    _check_labels(train, num_classes, "train")
    _check_labels(eval_batch, num_classes, "eval")

    train_rows, tuning_rows, small_ids = _training_rows(train, config, num_classes)
    train_view = RowView(train.vectors, train.labels, train_rows)
    tuning_view = RowView(train.vectors, train.labels, tuning_rows)

    def make_head() -> Any:
        return build_fusion_head(config.fusion_method, num_classes, config.layer_subset)

    def fit(lr: float) -> tuple[Any, Any]:
        return train_head(
            make_head,
            train_view,
            tuning_view,
            lr=lr,
            epochs=config.epochs,
            patience=config.early_stopping_patience,
            batch_size=config.batch_size,
            seed=config.seed,
            device=torch_device,
        )

    lr_tuning_scores: dict[float, float] = {}
    lr_search_seconds: float | None = None
    if config.fixed_lr is None:
        fits = [(lr, *fit(lr)) for lr in config.lr_candidates]
        best = min(
            range(len(fits)),
            key=lambda i: (fits[i][2].best_tuning_error, fits[i][2].best_tuning_loss, i),
        )
        selected_lr, head, outcome = fits[best]
        lr_tuning_scores = {lr: o.best_tuning_error for lr, _, o in fits}
        lr_search_seconds = sum(o.seconds for _, _, o in fits)
        lr_selection_seed = config.seed
    else:
        selected_lr = config.fixed_lr
        head, outcome = fit(selected_lr)
        lr_selection_seed = config.lr_selection_seed

    eval_view = RowView(
        eval_batch.vectors, eval_batch.labels, np.arange(len(eval_batch.doc_ids), dtype=np.int64)
    )
    _, eval_error, predicted = evaluate(head, eval_view, config.batch_size, torch_device)

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
        train_seconds=outcome.seconds,
        learning_curve=outcome.learning_curve,
        eval_error_rate_matched=matched,
        eval_error_rate_mismatched=mismatched,
        layer_weights=extract_layer_weights(head),
        small_sample_doc_ids=small_ids,
        transformers_version=metadata.transformers_version,
        attn_implementation=metadata.attn_implementation,
        torch_compile=metadata.torch_compile,
        truncated_doc_ratio=truncated_doc_ratio,
        lr_selection_seed=lr_selection_seed,
        lr_search_seconds=lr_search_seconds,
    )


def run_sequential_queue(
    configs: Sequence[RunConfig],
    provider_factory: object,
    output_dir: str,
) -> list[ResultRecord]:
    """Run main timed experiments sequentially, one at a time on GPU (ADR 0009)."""
    raise NotImplementedError("TODO: Implement sequential queue runner (T-06)")


def run_reproduction_track(
    dataset: str,
    seeds: Sequence[int],
    output_dir: str,
) -> list[ResultRecord]:
    """Run paper fine-tuning reproduction track (MNLI 89.1 / SST-2 96.0, x3 runs, ADR 0004)."""
    raise NotImplementedError("TODO: Implement reproduction track runner (T-06/T-08)")
