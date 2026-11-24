"""Reproduction track: the paper's fine-tuning setting, kept apart from the main track (T-06e).

ADR 0004, SPEC US 45–46: MNLI and SST-2 are fine-tuned on the full training set with the
paper's classification head (``FusionMethod.PAPER_HEAD``) and the paper's learning rate,
weight decay and epochs (arXiv 2412.13663v2 Table 6, in ``configs/experiment_matrix.yaml``).
Records carry ``track=reproduction``, so judgment and the main results table never see them.

Decision proposals recorded in SPEC §결정 필요 사항 (pending review):
- no tuning split: the paper fine-tunes on the whole training set;
- the Table 6 epochs are run in full and the **last** model is evaluated. The paper's early
  stopping criterion is not stated, and choosing the epoch on the dev set would pick the model
  on the evaluation set. Per-epoch dev error is kept in the learning curve only;
- AdamW with the paper weight decay on weight matrices only (bias / LayerNorm excluded).

Values the paper does not give (batch size, warmup ratio, LR scheduler, seeds) stay null in
the config; main runs refuse them and only debug runs may pass them on the command line.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml

from layer_fusion.contracts import (
    BackboneCondition,
    DataAmount,
    DatasetName,
    FusionMethod,
    ReproductionSettings,
    ResultRecord,
    RunConfig,
    RunKind,
    TrackKind,
)

REPRODUCTION_DATASETS = (DatasetName.MNLI, DatasetName.SST2)  # ADR 0004
PAPER_VALUES = ("lr", "weight_decay", "epochs")
UNSTATED_VALUES = ("batch_size", "warmup_ratio", "lr_scheduler")


@dataclass(frozen=True)
class ReproductionOverrides:
    """Debug-only values for what the paper does not state; ``None`` = take the config."""

    batch_size: int | None = None
    warmup_ratio: float | None = None
    lr_scheduler: str | None = None

    def is_empty(self) -> bool:
        return all(getattr(self, f.name) is None for f in fields(self))


# ---------------------------------------------------------------- configs


def expand_reproduction(
    matrix_path: str | Path,
    env_path: str | Path,
    *,
    datasets: Sequence[DatasetName] | None = None,
    seeds: list[int] | None = None,
    run_kind: RunKind = RunKind.MAIN,
    overrides: ReproductionOverrides | None = None,
) -> list[RunConfig]:
    """One RunConfig per (reproduction entry, seed), MNLI before SST-2, seeds ascending.

    TBD values are never guessed (AGENTS.md §2): a missing paper value, or an unstated value
    left null, refuses the expansion with everything that is missing. Seeds and the unstated
    values may be given directly by debug runs only.
    """
    from layer_fusion.cache_build import load_env_config
    from layer_fusion.matrix import _check_seeds

    overrides = overrides or ReproductionOverrides()
    if run_kind is RunKind.MAIN and (seeds is not None or not overrides.is_empty()):
        raise ValueError(
            "seed and setting overrides are for debug runs only; main reproduction runs take "
            "the locked config values"
        )
    matrix = yaml.safe_load(Path(matrix_path).read_text(encoding="utf-8"))
    entries = matrix.get("reproduction_track") or []
    wanted = REPRODUCTION_DATASETS if datasets is None else tuple(datasets)
    unknown = [d for d in wanted if d not in REPRODUCTION_DATASETS]
    if unknown:
        raise ValueError(f"the reproduction track covers MNLI and SST-2 only, got {unknown}")
    selected = [e for e in entries if DatasetName(e["dataset"]) in wanted]
    selected.sort(key=lambda e: REPRODUCTION_DATASETS.index(DatasetName(e["dataset"])))
    if not selected:
        raise ValueError(f"no reproduction_track entry for {[d.value for d in wanted]}")

    problems: list[str] = []
    resolved: list[tuple[dict[str, Any], dict[str, Any], list[int] | None]] = []
    for entry in selected:
        values = {name: entry.get(name) for name in PAPER_VALUES + UNSTATED_VALUES}
        for name in UNSTATED_VALUES:
            if getattr(overrides, name) is not None:
                values[name] = getattr(overrides, name)
        problems += [f"{entry['id']}: {n} is TBD" for n in values if values[n] is None]
        entry_seeds = seeds if seeds is not None else entry.get("seeds")
        if entry_seeds is None:
            problems.append(f"{entry['id']}: seeds is TBD")
        resolved.append((entry, values, entry_seeds))
    if problems:
        raise ValueError(
            "cannot expand the reproduction track (TBD values):\n- " + "\n- ".join(problems)
        )

    env = load_env_config(env_path)
    meta = env.cache_metadata(None, "float32")  # layer vectors are not stored: float32 live
    configs: list[RunConfig] = []
    for entry, values, entry_seeds in resolved:
        for seed in _check_seeds(entry_seeds, entry["id"]):
            config = RunConfig(
                dataset=DatasetName(entry["dataset"]),
                data_amount=DataAmount.FULL,
                fusion_method=FusionMethod.PAPER_HEAD,
                backbone_condition=BackboneCondition.FINETUNE,
                seed=seed,
                lr_candidates=(float(values["lr"]),),
                epochs=int(values["epochs"]),
                batch_size=int(values["batch_size"]),
                run_kind=run_kind,
                track=TrackKind.REPRODUCTION,
                expected_cache_meta=meta,
                reproduction=ReproductionSettings(
                    weight_decay=float(values["weight_decay"]),
                    warmup_ratio=float(values["warmup_ratio"]),
                    lr_scheduler=str(values["lr_scheduler"]),
                ),
            )
            check_reproduction_config(config)
            configs.append(config)
    return configs


def check_reproduction_config(config: RunConfig) -> None:
    """Refuse anything that is not a complete reproduction-track setting."""
    expected = {
        "track": TrackKind.REPRODUCTION,
        "fusion_method": FusionMethod.PAPER_HEAD,
        "backbone_condition": BackboneCondition.FINETUNE,
        "data_amount": DataAmount.FULL,
    }
    wrong = {k: getattr(config, k) for k, v in expected.items() if getattr(config, k) is not v}
    if wrong:
        raise ValueError(f"not a reproduction-track config (ADR 0004): {wrong}")
    if config.dataset not in REPRODUCTION_DATASETS:
        raise ValueError(f"the reproduction track covers MNLI and SST-2 only, got {config.dataset}")
    settings = config.reproduction
    if settings is None:
        raise ValueError("reproduction runs need RunConfig.reproduction (paper settings)")
    if len(config.lr_candidates) != 1 or config.fixed_lr is not None:
        raise ValueError("a reproduction run uses exactly the paper learning rate")
    lr = config.lr_candidates[0]
    if not (math.isfinite(lr) and lr > 0):
        raise ValueError(f"lr must be a finite positive number, got {lr}")
    for name in ("epochs", "batch_size"):
        value = getattr(config, name)
        if value is None or value < 1:
            raise ValueError(f"RunConfig.{name} must be a positive integer, got {value}")
    if not (math.isfinite(settings.weight_decay) and settings.weight_decay >= 0):
        raise ValueError(f"weight_decay must be >= 0, got {settings.weight_decay}")
    if not 0 <= settings.warmup_ratio < 1:
        raise ValueError(f"warmup_ratio must be in [0, 1), got {settings.warmup_ratio}")
    if config.input_setting is not None:
        raise ValueError("reproduction runs use the untruncated MNLI / SST-2 input")


# ---------------------------------------------------------------- model and optimizer


def make_paper_optimizer(model: Any, lr: float, weight_decay: float) -> Any:
    """AdamW with the paper weight decay on weight matrices only; bias and LayerNorm weights
    (all 1-D) get none (Hugging Face convention; the paper does not say)."""
    import torch

    params = [p for p in model.parameters() if p.requires_grad]
    groups = [
        {"params": [p for p in params if p.ndim >= 2], "weight_decay": weight_decay},
        {"params": [p for p in params if p.ndim < 2], "weight_decay": 0.0},
    ]
    return torch.optim.AdamW([g for g in groups if g["params"]], lr=lr)


def make_scheduler(optimizer: Any, settings: ReproductionSettings, total_steps: int) -> Any:
    """``transformers.get_scheduler`` with ``warmup_ratio`` of all steps as warmup."""
    from transformers import get_scheduler

    return get_scheduler(
        settings.lr_scheduler,
        optimizer,
        num_warmup_steps=round(settings.warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )


def default_model_factory(provider: Any) -> Callable[[int], Any]:
    """``num_labels -> ModernBertForSequenceClassification`` under the locked setting."""

    def factory(num_labels: int) -> Any:
        from transformers import AutoModelForSequenceClassification

        meta = provider.metadata
        return AutoModelForSequenceClassification.from_pretrained(
            meta.model_name,
            num_labels=num_labels,
            attn_implementation=meta.attn_implementation,
        )

    return factory


# ---------------------------------------------------------------- one run


def run_reproduction(
    config: RunConfig,
    provider: Any,
    *,
    device: str | None = None,
    model_factory: Callable[[int], Any] | None = None,
) -> ResultRecord:
    """Fine-tune the paper model for the Table 6 epochs and evaluate the last model.

    ``provider`` is a ``BackboneProvider`` (tokenized splits; its own backbone is not used).
    ``train_seconds`` covers the training steps only, as in the main track; the per-epoch dev
    evaluation goes to the learning curve and is not counted.
    """
    import torch

    from layer_fusion.data import DATASET_SPECS
    from layer_fusion.finetune import PaperHeadModel, TokenRowView
    from layer_fusion.runner import (
        _check_labels,
        _resolve_device,
        genre_error_rates,
        make_run_id,
        prediction_records,
    )
    from layer_fusion.training import TrainingDivergedError, _rng_devices, _sync, evaluate

    check_reproduction_config(config)
    provider.validate_for_run(config)
    settings = cast(ReproductionSettings, config.reproduction)  # checked above
    torch_device = _resolve_device(device)
    num_classes = DATASET_SPECS[config.dataset].num_classes
    train = provider.get_split("train")
    eval_batch = provider.get_split("eval")
    _check_labels(train, num_classes, "train")
    _check_labels(eval_batch, num_classes, "eval")
    if config.dataset == DatasetName.MNLI and eval_batch.genre_splits is None:
        raise ValueError("MNLI eval split needs genre_splits (matched / mismatched, US 5)")

    def view(batch: Any) -> TokenRowView:
        rows = np.arange(len(batch.doc_ids), dtype=np.int64)  # whole split: no tuning split
        return TokenRowView(
            batch.input_ids, batch.labels, rows, provider.pad_id, provider.special_ids, True
        )

    train_view, eval_view = view(train), view(eval_batch)
    if len(train_view) == 0:
        raise ValueError("cannot train on an empty split")
    factory = model_factory or default_model_factory(provider)
    batch_size, epochs = cast(int, config.batch_size), cast(int, config.epochs)  # checked
    lr = config.lr_candidates[0]
    steps_per_epoch = math.ceil(len(train_view) / batch_size)

    curve: list[dict[str, float]] = []
    train_seconds = 0.0
    with torch.random.fork_rng(devices=_rng_devices()):
        torch.manual_seed(config.seed)
        model = PaperHeadModel(factory(num_classes)).to(torch_device)
        optimizer = make_paper_optimizer(model, lr, settings.weight_decay)
        scheduler = make_scheduler(optimizer, settings, steps_per_epoch * epochs)
        shuffle_rng = np.random.default_rng(config.seed)
        for epoch in range(1, epochs + 1):
            _sync(torch_device)
            started = time.perf_counter()
            model.train()
            order = shuffle_rng.permutation(len(train_view))
            train_loss = torch.zeros((), device=torch_device)
            for start in range(0, len(order), batch_size):
                rows = np.sort(train_view.rows[order[start : start + batch_size]])
                x, y = train_view.load(rows, torch_device)
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.cross_entropy(model(x), y)
                loss.backward()
                optimizer.step()
                scheduler.step()
                train_loss += loss.detach() * len(rows)
            _sync(torch_device)
            train_seconds += time.perf_counter() - started
            mean_train_loss = float(train_loss.item()) / len(train_view)
            eval_loss, eval_error, predicted = evaluate(model, eval_view, batch_size, torch_device)
            curve.append(
                {
                    "epoch": float(epoch),
                    "train_loss": mean_train_loss,
                    "eval_loss": eval_loss,
                    "eval_error": eval_error,
                }
            )
            if not (math.isfinite(mean_train_loss) and math.isfinite(eval_loss)):
                raise TrainingDivergedError(
                    f"reproduction run diverged at epoch {epoch} (lr={lr})"
                )

    # The last epoch's evaluation is the reported one: the last model, never the best epoch.
    predictions = prediction_records(eval_batch, predicted, config)
    matched, mismatched = genre_error_rates(predictions, config)
    meta = provider.metadata
    return ResultRecord(
        run_id=make_run_id(config),
        config=config,
        selected_lr=lr,
        lr_tuning_scores={},
        eval_error_rate=eval_error,
        predictions=predictions,
        trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
        train_seconds=train_seconds,
        learning_curve=curve,
        eval_error_rate_matched=matched,
        eval_error_rate_mismatched=mismatched,
        transformers_version=meta.transformers_version,
        attn_implementation=meta.attn_implementation,
        torch_compile=meta.torch_compile,
    )
