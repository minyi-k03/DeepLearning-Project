#!/usr/bin/env python3
"""CLI skeleton for sequential main experiment queue runner (ADR 0009, T-06)."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run main timed experiments sequentially one at a time on GPU."
    )
    parser.add_argument("--matrix", default="configs/experiment_matrix.yaml")
    parser.add_argument("--env-config", default="configs/week1_env.yaml")
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--output-dir", default="results/runs")
    return parser.parse_args()


def main() -> None:
    _ = parse_args()
    raise NotImplementedError("TODO: Wire sequential queue runner (T-06)")


if __name__ == "__main__":
    main()
