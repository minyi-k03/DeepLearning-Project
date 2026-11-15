#!/usr/bin/env python3
"""CLI skeleton for a single experiment or debugging run (T-06)."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single layer-fusion experiment.")
    parser.add_argument("--dataset", required=True, choices=["mnli", "sst2", "imdb", "ag_news"])
    parser.add_argument("--fusion", required=True, choices=["B0", "F1", "F2", "F3", "F4", "B0+"])
    parser.add_argument("--backbone", default="frozen", choices=["frozen", "finetune"])
    parser.add_argument("--data-amount", default="full", choices=["full", "small"])
    parser.add_argument(
        "--layer-subset", default="all", choices=["all", "global_only", "local_only"]
    )
    parser.add_argument("--input-setting", default=None, choices=["L128", "L512", "Lfull"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-kind", default="debug", choices=["main", "debug"])
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--output-dir", default="results/runs")
    return parser.parse_args()


def main() -> None:
    _ = parse_args()
    raise NotImplementedError("TODO: Wire run_experiment CLI (T-06)")


if __name__ == "__main__":
    main()
