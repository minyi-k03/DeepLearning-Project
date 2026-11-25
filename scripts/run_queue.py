#!/usr/bin/env python3
"""Run the experiment matrix through the sequential queue (T-06c, ADR 0009).

Main queue (holds the GPU lock; seeds and tuning budget come from the locked configs):

    python scripts/run_queue.py --experiments mnli_frozen_full

Debug queue (no lock, recorded as debug; seeds, budget and fusion methods on the command line):

    python scripts/run_queue.py --debug --experiments mnli_frozen_full \\
        --seeds 0 1 --fusion-methods B0 F2 --lr-candidates 1e-3 --epochs 1 \\
        --early-stopping-patience 1 --batch-size 32

Fine-tune queue (T-08f): ``BEST_FUSION`` is chosen from the MNLI frozen-full selection runs by
the decision rule (main queues need ``status: locked`` rules; debug queues read debug records):

    python scripts/run_queue.py --experiments mnli_finetune_full --best-fusion-from results/runs

``--dry-run`` prints the order (and which records already exist) without running anything.
Order: MNLI > SST-2 > IMDB. Existing records are skipped, so rerunning the same command
resumes an interrupted queue; the first failing run stops it.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from layer_fusion.contracts import MAIN_FUSION_METHODS, FusionMethod, RunConfig, RunKind
from layer_fusion.matrix import TrainingBudget, expand_experiment_matrix, select_best_fusion
from layer_fusion.records import record_path
from layer_fusion.run_queue import (
    DEFAULT_LOCK_PATH,
    MainQueueLockedError,
    QueueRunError,
    cache_provider_factory,
    check_queue,
    order_by_priority,
    run_queue,
)

DEBUG_ONLY = (
    "seeds",
    "fusion_methods",
    "lr_candidates",
    "epochs",
    "early_stopping_patience",
    "batch_size",
    "finetune_lr_candidates",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run main timed experiments sequentially, one at a time on the GPU."
    )
    parser.add_argument("--matrix", default="configs/experiment_matrix.yaml")
    parser.add_argument("--env-config", default="configs/week1_env.yaml")
    parser.add_argument(
        "--cache-dir", default="data/cache", help="root holding <dataset>[_<input setting>]/"
    )
    parser.add_argument("--output-dir", default="results/runs")
    parser.add_argument("--experiments", nargs="+", default=None, help="matrix ids (default: all)")
    parser.add_argument("--lock-file", default=str(DEFAULT_LOCK_PATH))
    parser.add_argument("--device", default=None, help="cpu / cuda (default: cuda if available)")
    parser.add_argument("--dry-run", action="store_true", help="print the order, run nothing")
    parser.add_argument(
        "--debug", action="store_true", help="debug queue: no GPU lock, recorded as debug"
    )
    parser.add_argument(
        "--best-fusion-from",
        default=None,
        help="runs dir of the MNLI frozen-full results; fills BEST_FUSION by the decision rule",
    )
    parser.add_argument("--rules", default="configs/decision_rules.yaml")
    debug = parser.add_argument_group("debug-only overrides")
    debug.add_argument("--seeds", type=int, nargs="+", default=None)
    debug.add_argument(
        "--fusion-methods",
        nargs="+",
        default=None,
        choices=[m.value for m in MAIN_FUSION_METHODS],
        help="keep only these fusion methods of the selected experiments",
    )
    debug.add_argument("--lr-candidates", type=float, nargs="+", default=None)
    debug.add_argument("--epochs", type=int, default=None)
    debug.add_argument("--early-stopping-patience", type=int, default=None)
    debug.add_argument("--batch-size", type=int, default=None)
    debug.add_argument("--finetune-lr-candidates", type=float, nargs="+", default=None)
    return parser.parse_args(argv)


def build_queue(args: argparse.Namespace) -> list[RunConfig]:
    """Expanded, filtered and priority-ordered configs; refuses TBD values and main-run
    overrides before anything runs."""
    run_kind = RunKind.DEBUG if args.debug else RunKind.MAIN
    if run_kind is RunKind.MAIN:
        given = ["--" + n.replace("_", "-") for n in DEBUG_ONLY if getattr(args, n) is not None]
        if given:
            raise ValueError(
                f"{given} are debug-only; main runs take the locked values (AGENTS.md §2)"
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
    configs = expand_experiment_matrix(
        args.matrix,
        args.env_config,
        experiment_ids=args.experiments,
        seeds=args.seeds,
        run_kind=run_kind,
        budget_overrides=None if overrides.is_empty() else overrides,
        best_fusion=_best_fusion(args, run_kind),
    )
    if args.fusion_methods is not None:
        keep = {FusionMethod(m) for m in args.fusion_methods}
        configs = [c for c in configs if c.fusion_method in keep]
    check_queue(configs)
    return order_by_priority(configs)


def _best_fusion(args: argparse.Namespace, run_kind: RunKind) -> FusionMethod | None:
    """The method BEST_FUSION stands for, chosen by the decision rule (or None)."""
    if args.best_fusion_from is None:
        return None
    from layer_fusion.judgment import load_decision_rules
    from layer_fusion.records import load_result_records

    try:
        records = load_result_records(args.best_fusion_from)
    except FileNotFoundError as exc:
        raise ValueError(str(exc)) from exc
    rules = load_decision_rules(args.rules, require_locked=run_kind is RunKind.MAIN)
    choice = select_best_fusion(records, rules.best_fusion_selection_rule, run_kind=run_kind)
    scores = ", ".join(f"{m.value} {s:.4f}" for m, s in choice.scores.items())
    print(
        f"[best-fusion] {choice.method.value} ({rules.best_fusion_selection_rule}; "
        f"tuning error at the selected LR: {scores})"
    )
    return choice.method


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        configs = build_queue(args)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc

    # Heavy imports (torch) only after the queue is known to be valid.
    from layer_fusion.runner import make_run_id, run_experiment

    kind = configs[0].run_kind.value
    if args.dry_run:
        print(f"[queue] {kind} queue, {len(configs)} runs (dry run, nothing started):")
        for i, config in enumerate(configs, start=1):
            run_id = make_run_id(config)
            exists = record_path(args.output_dir, run_id).exists()
            print(f"  {i:3d}. {run_id}{' (record exists, skip)' if exists else ''}")
        return 0

    def run(config, provider):  # type: ignore[no-untyped-def]
        return run_experiment(config, provider, device=args.device)

    print(f"[queue] {kind} queue, {len(configs)} runs -> {args.output_dir}")
    try:
        result = run_queue(
            configs,
            cache_provider_factory(args.cache_dir, args.env_config),
            args.output_dir,
            run=run,
            run_id_of=make_run_id,
            lock_path=args.lock_file,
        )
    except MainQueueLockedError as exc:
        raise SystemExit(f"error: {exc}") from exc
    except QueueRunError as exc:
        raise SystemExit(f"error: {exc}\ncause: {exc.__cause__!r}") from exc
    print(
        f"[queue] finished: {len(result.completed)} run, {len(result.skipped)} skipped "
        f"(records already existed), {sum(result.seconds.values()):.1f}s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
