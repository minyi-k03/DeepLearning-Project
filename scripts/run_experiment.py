#!/usr/bin/env python3
"""Run one experiment on a layer-vector cache and save its result record (T-06b).

Example (debug run on a sampled cache):

    python scripts/run_experiment.py --dataset mnli --fusion F2 --seed 0 \\
        --cache-dir data/cache_debug/mnli_limit200 \\
        --lr-candidates 1e-3 --epochs 1 --early-stopping-patience 1 --batch-size 32

Debug runs only: main timed runs go through the sequential queue (``scripts/run_queue.py``,
T-06c), which holds the GPU lock (ADR 0009), so ``--run-kind main`` is refused here. The
tuning budget comes from ``configs/week1_env.yaml``; TBD values are passed on the command
line. A reuse run (``--fixed-lr`` + ``--lr-selection-seed``) is checked against its
selection-run record in ``--output-dir`` before training.

Finetune condition (T-06d): ``--backbone finetune`` loads MNLI and ModernBERT instead of a
cache and trains both; ``--limit N`` keeps N random documents per split, e.g.

    python scripts/run_experiment.py --dataset mnli --backbone finetune --fusion B0 --seed 0 \
        --limit 2000 --finetune-lr-candidates 2e-5 --epochs 1 --early-stopping-patience 1 \
        --batch-size 32
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from layer_fusion.cache_build import cache_name
from layer_fusion.contracts import (
    MAIN_FUSION_METHODS,
    BackboneCondition,
    DataAmount,
    DatasetName,
    FusionMethod,
    InputSetting,
    LayerSubset,
    RunConfig,
    RunKind,
)
from layer_fusion.matrix import TrainingBudget, resolve_budget
from layer_fusion.records import (
    find_selection_record_for,
    load_result_records,
    record_path,
    save_result_record,
)

DEFAULT_CACHE_ROOT = Path("data/cache")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single layer-fusion experiment.")
    parser.add_argument("--dataset", required=True, choices=[d.value for d in DatasetName])
    parser.add_argument(
        "--fusion", required=True, choices=[m.value for m in MAIN_FUSION_METHODS]
    )
    parser.add_argument(
        "--backbone", default="frozen", choices=[b.value for b in BackboneCondition]
    )
    parser.add_argument("--data-amount", default="full", choices=[a.value for a in DataAmount])
    parser.add_argument("--layer-subset", default="all", choices=[s.value for s in LayerSubset])
    parser.add_argument("--input-setting", default=None, choices=[s.value for s in InputSetting])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-kind", default="debug", choices=[k.value for k in RunKind])
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="layer-vector cache folder (default: data/cache/<dataset>[_<input setting>])",
    )
    parser.add_argument("--output-dir", default="results/runs")
    parser.add_argument("--env-config", default="configs/week1_env.yaml")
    budget = parser.add_argument_group("tuning budget overrides (debug runs only)")
    budget.add_argument("--lr-candidates", type=float, nargs="+", default=None)
    budget.add_argument("--epochs", type=int, default=None)
    budget.add_argument("--early-stopping-patience", type=int, default=None)
    budget.add_argument("--batch-size", type=int, default=None)
    budget.add_argument("--finetune-lr-candidates", type=float, nargs="+", default=None)
    parser.add_argument(
        "--limit", type=int, default=None,
        help="finetune only: N random documents per split (debug sample)",
    )
    reuse = parser.add_argument_group("LR reuse (the other seeds of a setting)")
    reuse.add_argument("--fixed-lr", type=float, default=None)
    reuse.add_argument("--lr-selection-seed", type=int, default=None)
    parser.add_argument("--device", default=None, help="cpu / cuda (default: cuda if available)")
    parser.add_argument(
        "--overwrite", action="store_true", help="replace an existing record (debug runs only)"
    )
    return parser.parse_args(argv)


def build_run_config(args: argparse.Namespace) -> RunConfig:
    """Debug RunConfig from parsed arguments; refuses main runs and TBD budget values."""
    run_kind = RunKind(args.run_kind)
    if run_kind is RunKind.MAIN:
        raise ValueError(
            "main runs go through the sequential queue (scripts/run_queue.py, T-06c), which "
            "holds the GPU lock (ADR 0009); this CLI runs debug only"
        )
    overrides = TrainingBudget(
        lr_candidates=None if args.lr_candidates is None else tuple(args.lr_candidates),
        epochs=args.epochs,
        early_stopping_patience=args.early_stopping_patience,
        batch_size=args.batch_size,
        finetune_lr_candidates=(
            None if args.finetune_lr_candidates is None else tuple(args.finetune_lr_candidates)
        ),
    )
    backbone = BackboneCondition(args.backbone)
    if args.limit is not None and backbone is not BackboneCondition.FINETUNE:
        raise ValueError("--limit samples the finetune data; frozen runs use a cache folder")
    budget = resolve_budget(args.env_config, run_kind, overrides)
    if budget.missing(backbone):
        raise ValueError(
            f"tuning budget is TBD in {args.env_config}: {budget.missing(backbone)} "
            "(pass them on the command line)"
        )
    return RunConfig(
        dataset=DatasetName(args.dataset),
        data_amount=DataAmount(args.data_amount),
        fusion_method=FusionMethod(args.fusion),
        backbone_condition=backbone,
        seed=args.seed,
        layer_subset=LayerSubset(args.layer_subset),
        input_setting=None if args.input_setting is None else InputSetting(args.input_setting),
        **budget.run_config_fields(backbone),
        run_kind=run_kind,
        fixed_lr=args.fixed_lr,
        lr_selection_seed=args.lr_selection_seed,
    )


def check_lr_reuse(config: RunConfig, output_dir: str | Path) -> None:
    """A reuse run must take the LR its selection run in ``output_dir`` chose (SPEC §결정 필요
    사항, T-06a); checked before training so a made-up LR never produces a record."""
    if config.fixed_lr is None and config.lr_selection_seed is None:
        return
    runs_dir = Path(output_dir)
    records = load_result_records(runs_dir) if runs_dir.is_dir() else []
    find_selection_record_for(config, records, label="--fixed-lr")


def make_backbone_provider(config: RunConfig, args: argparse.Namespace):  # type: ignore[no-untyped-def]
    """ModernBERT + the dataset, under the locked setting of ``--env-config`` (T-06d)."""
    from layer_fusion.cache_build import load_env_config
    from layer_fusion.providers import BackboneProvider

    return BackboneProvider.from_env(
        config.dataset, load_env_config(args.env_config), limit=args.limit
    )


def main(argv: Sequence[str] | None = None) -> Path:
    """Run one experiment and return the saved record path."""
    args = parse_args(argv)
    try:
        config = build_run_config(args)
        check_lr_reuse(config, args.output_dir)
    except (ValueError, LookupError) as exc:
        raise SystemExit(f"error: {exc}") from exc

    # Heavy imports (torch) only after the arguments are known to be valid.
    from layer_fusion.providers import CacheProvider
    from layer_fusion.runner import make_run_id, run_experiment

    # Refuse before training, not after hours of GPU time.
    out_path = record_path(args.output_dir, make_run_id(config))
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"error: {out_path} already exists; result records are never overwritten")
    if config.backbone_condition is BackboneCondition.FINETUNE:
        provider = make_backbone_provider(config, args)
    else:
        cache_dir = args.cache_dir or DEFAULT_CACHE_ROOT / cache_name(
            config.dataset, config.input_setting
        )
        try:
            provider = CacheProvider(str(cache_dir))
        except FileNotFoundError as exc:
            raise SystemExit(f"error: {exc}") from exc

    record = run_experiment(config, provider, device=args.device)
    path = save_result_record(record, args.output_dir, overwrite=args.overwrite)
    print(
        f"[run_experiment] {record.run_id}: eval error {record.eval_error_rate:.4f}, "
        f"lr {record.selected_lr:g}, train {record.train_seconds:.1f}s -> {path}"
    )
    return path


if __name__ == "__main__":
    main(sys.argv[1:])
