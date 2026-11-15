#!/usr/bin/env python3
"""CLI skeleton for ModernBERT-base 23-layer vector extraction and caching (T-04)."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract and cache ModernBERT-base layer vectors.")
    parser.add_argument("--dataset", required=True, choices=["mnli", "sst2", "imdb", "ag_news"])
    parser.add_argument("--input-setting", default=None, choices=["L128", "L512", "Lfull"])
    parser.add_argument("--env-config", default="configs/week1_env.yaml")
    parser.add_argument("--output-dir", default="data/cache")
    return parser.parse_args()


def main() -> None:
    _ = parse_args()
    raise NotImplementedError("TODO: Wire extraction pipeline (T-04)")


if __name__ == "__main__":
    main()
