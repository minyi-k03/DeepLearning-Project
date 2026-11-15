#!/usr/bin/env python3
"""T-02b: measure IMDB token lengths with the ModernBERT-base tokenizer (SPEC US15-17, ADR 0006).

Token length includes special tokens ([CLS] ... [SEP]), matching the L128 / L512 / Lfull
definitions. The Lfull cap rule (ADR 0006): if the observed maximum is <= 4,096, no document
is truncated; otherwise the cap stays at 4,096 and the truncated-document ratio is reported.

Usage:
  python scripts/measure_imdb_lengths.py
  python scripts/measure_imdb_lengths.py --output docs/measurements/imdb_length_check.json
"""

from __future__ import annotations

import argparse
import json
import platform
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np

from layer_fusion.inputs import assign_length_bucket

LFULL_CAP_LIMIT = 4096
PERCENTILES = (50, 90, 95, 99, 99.9)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-name", default="answerdotai/ModernBERT-base")
    parser.add_argument("--dataset-name", default="stanfordnlp/imdb")
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--output", default="docs/measurements/imdb_length_check.json")
    return parser.parse_args()


def count_tokens(texts: list[str], tokenizer: object, batch_size: int) -> np.ndarray:
    lengths: list[int] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        enc = tokenizer(batch, add_special_tokens=True, truncation=False)
        lengths.extend(len(ids) for ids in enc["input_ids"])
    return np.asarray(lengths, dtype=np.int64)


def summarize(lengths: np.ndarray) -> dict[str, object]:
    buckets = Counter(assign_length_bucket(int(n)).value for n in lengths)
    over_cap = int((lengths > LFULL_CAP_LIMIT).sum())
    return {
        "num_docs": int(lengths.size),
        "max_tokens": int(lengths.max()),
        "min_tokens": int(lengths.min()),
        "mean_tokens": round(float(lengths.mean()), 2),
        "percentiles": {f"p{p:g}": int(np.percentile(lengths, p)) for p in PERCENTILES},
        "length_buckets": {k: buckets.get(k, 0) for k in ("<=512", "512-1024", ">1024")},
        "docs_over_512": int((lengths > 512).sum()),
        "docs_over_lfull_cap_limit": over_cap,
        "truncated_ratio_at_cap_limit": round(over_cap / lengths.size, 6),
    }


def main() -> None:
    args = parse_args()

    import datasets
    import transformers
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    report: dict[str, object] = {
        "task": "T-02b",
        "measured_on": date.today().isoformat(),
        "model_name": args.model_name,
        "dataset_name": args.dataset_name,
        "length_definition": "input_ids length with special tokens, no truncation, raw text",
        "environment": {
            "python": platform.python_version(),
            "transformers": transformers.__version__,
            "datasets": datasets.__version__,
            "tokenizer_class": type(tokenizer).__name__,
            "tokenizer_model_max_length": int(tokenizer.model_max_length),
        },
        "splits": {},
    }

    all_lengths = []
    for split in args.splits:
        ds = load_dataset(args.dataset_name, split=split)
        lengths = count_tokens(ds["text"], tokenizer, args.batch_size)
        all_lengths.append(lengths)
        report["splits"][split] = summarize(lengths)
        print(f"[{split}] {json.dumps(report['splits'][split], ensure_ascii=False)}")

    combined = np.concatenate(all_lengths)
    report["all_splits"] = summarize(combined)
    observed_max = int(combined.max())
    report["decision"] = {
        "observed_max_tokens": observed_max,
        "lfull_max_length_cap": LFULL_CAP_LIMIT,
        "any_doc_truncated_in_lfull": observed_max > LFULL_CAP_LIMIT,
        "truncated_ratio_in_lfull": report["all_splits"]["truncated_ratio_at_cap_limit"],
        "rule": "ADR 0006: max <= 4096 -> no document truncated; max > 4096 -> keep cap 4096 "
        "and report the truncated ratio",
    }
    print(f"[decision] {json.dumps(report['decision'], ensure_ascii=False)}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
