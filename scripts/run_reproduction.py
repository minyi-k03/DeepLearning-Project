#!/usr/bin/env python3
"""Run the reproduction track: paper fine-tuning of MNLI / SST-2, x3 seeds (T-06e, ADR 0004).

Main run (holds the shared-GPU lock of the main queue, ADR 0009; values from the config):

    python scripts/run_reproduction.py

Debug run (no lock, recorded as debug; what the paper does not state goes on the command line):

    python scripts/run_reproduction.py --debug --datasets mnli --seeds 0 --limit 2000 \\
        --batch-size 32 --warmup-ratio 0.06 --lr-scheduler linear

Records go to ``results/repro`` (separate from the main runs); a debug ``--limit N`` sample
writes under ``results/repro/limitN`` so samples of different sizes never pass for each other
on resume. An existing record is skipped only when it was made with the same config; the first
failing run stops the command. Each result is printed next to the paper's error rate.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path

from layer_fusion.contracts import DatasetName, RunConfig, RunKind
from layer_fusion.records import record_path, save_result_record
from layer_fusion.reproduction import (
    REPRODUCTION_DATASETS,
    ReproductionOverrides,
    expand_reproduction,
    paper_error_rates,
)
from layer_fusion.run_queue import DEFAULT_LOCK_PATH, MainQueueLockedError, main_queue_lock


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the paper reproduction track.")
    parser.add_argument("--matrix", default="configs/experiment_matrix.yaml")
    parser.add_argument("--env-config", default="configs/week1_env.yaml")
    parser.add_argument(
        "--datasets", nargs="+", default=None, choices=[d.value for d in REPRODUCTION_DATASETS]
    )
    parser.add_argument("--output-dir", default="results/repro")
    parser.add_argument("--lock-file", default=str(DEFAULT_LOCK_PATH))
    parser.add_argument("--device", default=None, help="cpu / cuda (default: cuda if available)")
    parser.add_argument("--dry-run", action="store_true", help="print the runs, run nothing")
    parser.add_argument("--debug", action="store_true", help="no GPU lock, recorded as debug")
    debug = parser.add_argument_group("debug-only values (the paper does not state them)")
    debug.add_argument("--seeds", type=int, nargs="+", default=None)
    debug.add_argument("--batch-size", type=int, default=None)
    debug.add_argument("--warmup-ratio", type=float, default=None)
    debug.add_argument("--lr-scheduler", default=None)
    debug.add_argument("--limit", type=int, default=None, help="N random documents per split")
    return parser.parse_args(argv)


def build_configs(args: argparse.Namespace) -> list[RunConfig]:
    run_kind = RunKind.DEBUG if args.debug else RunKind.MAIN
    if run_kind is RunKind.MAIN and args.limit is not None:
        raise ValueError("--limit is debug-only: a sample cannot make a main record")
    # Seeds and the unstated values are refused for main runs inside expand_reproduction.
    return expand_reproduction(
        args.matrix,
        args.env_config,
        datasets=None if args.datasets is None else [DatasetName(d) for d in args.datasets],
        seeds=args.seeds,
        run_kind=run_kind,
        overrides=ReproductionOverrides(
            batch_size=args.batch_size,
            warmup_ratio=args.warmup_ratio,
            lr_scheduler=args.lr_scheduler,
        ),
    )


def make_provider(dataset: DatasetName, args: argparse.Namespace):  # type: ignore[no-untyped-def]
    """Tokenized splits under the locked setting; the paper model is built per run."""
    from layer_fusion.cache_build import load_env_config
    from layer_fusion.providers import BackboneProvider

    return BackboneProvider.from_env(
        dataset, load_env_config(args.env_config), limit=args.limit, load_backbone=False
    )


def _versus_paper(record, paper: dict[DatasetName, float]) -> str:  # type: ignore[no-untyped-def]
    """Our error rate(s) next to the published one (US 45). MNLI shows matched and the
    combined matched+mismatched rate: which one the paper reports is not stated (TBD)."""
    text = f"eval error {record.eval_error_rate:.4f}"
    if record.eval_error_rate_matched is not None:
        text += f" (matched {record.eval_error_rate_matched:.4f})"
    published = paper.get(record.config.dataset)
    return text if published is None else f"{text} vs paper {published:.4f}"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        configs = build_configs(args)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc

    from layer_fusion.reproduction import run_reproduction
    from layer_fusion.run_queue import matching_record
    from layer_fusion.runner import make_run_id

    out = Path(args.output_dir)
    if args.limit is not None:
        out = out / f"limit{args.limit}"  # the sample size is not part of the RunConfig
    paper = paper_error_rates(args.matrix)
    kind = configs[0].run_kind.value
    if args.dry_run:
        print(f"[repro] {kind}, {len(configs)} runs (dry run, nothing started):")
        for config in configs:
            run_id = make_run_id(config)
            exists = record_path(out, run_id).exists()
            print(f"  {run_id}{' (record exists)' if exists else ''}")
        return 0

    lock = main_queue_lock(args.lock_file) if configs[0].run_kind is RunKind.MAIN else nullcontext()
    providers: dict[DatasetName, object] = {}
    try:
        with lock:
            for config in configs:
                run_id = make_run_id(config)
                path = record_path(out, run_id)
                try:
                    if path.exists():
                        matching_record(path, config)
                        print(f"[repro] skip {run_id} (record exists)")
                        continue
                    if config.dataset not in providers:
                        providers[config.dataset] = make_provider(config.dataset, args)
                    record = run_reproduction(config, providers[config.dataset], device=args.device)
                    save_result_record(record, out)
                except Exception as exc:
                    raise SystemExit(
                        f"error: {run_id} failed ({type(exc).__name__}: {exc}); fix it and rerun "
                        "(finished records are skipped)"
                    ) from exc
                print(f"[repro] {run_id}: {_versus_paper(record, paper)}, "
                      f"train {record.train_seconds:.1f}s -> {path}")
    except MainQueueLockedError as exc:
        raise SystemExit(f"error: {exc}") from exc
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
