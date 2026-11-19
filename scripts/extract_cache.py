#!/usr/bin/env python3
"""T-04b: extract ModernBERT-base 23-layer vectors and save them as a layer-vector cache.

The locked environment (model, transformers version, attention, compile, Lfull cap, pooling,
cache dtype) comes from configs/week1_env.yaml; a value still ``null`` there stops the run.
Each split is written to ``<output-dir>/<split>.partial`` and published as ``<split>`` only on
success; existing caches are never overwritten (AGENTS.md §7). GPU runs of the full datasets
belong to T-08b and run one at a time on the shared GPU (ADR 0009).

Usage:
  # full cache (T-08b)
  python scripts/extract_cache.py --dataset mnli --device cuda
  python scripts/extract_cache.py --dataset imdb --input-setting L512 --device cuda
  # small debug cache (only under data/cache_debug/, never mixed with main caches)
  python scripts/extract_cache.py --dataset sst2 --limit 50 --dtype float16
  # float16 range measurement: float32 extraction, report only, nothing saved
  python scripts/extract_cache.py --dataset mnli --limit 2000 --report-only \
      --report-output docs/measurements/float16_range_check.json
  # float16 worst case: the 300 shortest train inputs (largest means), report only
  python scripts/extract_cache.py --dataset sst2 --splits train --shortest 300 --report-only \
      --report-output docs/measurements/float16_range_check.json
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import date
from pathlib import Path

from layer_fusion.cache_build import (
    CACHE_DTYPES,
    build_cache,
    cache_name,
    check_input_setting,
    check_output_dir,
    default_output_dir,
    load_env_config,
    resolve_cache_dtype,
    sample_records,
    select_shortest,
)
from layer_fusion.contracts import DatasetName, InputSetting
from layer_fusion.data import load_dataset_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True, choices=[d.value for d in DatasetName])
    parser.add_argument("--input-setting", default=None, choices=[s.value for s in InputSetting])
    parser.add_argument("--env-config", default="configs/week1_env.yaml")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="default: data/cache/<dataset>[_<setting>] (data/cache_debug/... for a sample)",
    )
    parser.add_argument("--splits", nargs="+", default=["train", "eval"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    sample = parser.add_mutually_exclusive_group()
    sample.add_argument("--limit", type=int, default=None, help="random docs per split (debug)")
    sample.add_argument("--shortest", type=int, default=None,
                        help="the N shortest inputs per split (float16 worst case, debug)")
    parser.add_argument("--seed", type=int, default=0, help="seed for --limit sampling")
    parser.add_argument("--dtype", default=None, choices=list(CACHE_DTYPES),
                        help="override cache_dtype (debug samples only)")
    parser.add_argument("--data-dir", default=None, help="Hugging Face datasets cache dir")
    parser.add_argument("--report-only", action="store_true",
                        help="extract in float32 and only report value ranges (nothing saved)")
    parser.add_argument("--report-output", default=None,
                        help="JSON file to merge this run's report into")
    return parser.parse_args()


def merge_report(path: Path, key: str, entry: dict) -> None:
    """Add this run under ``runs[key]`` of a JSON file, keeping other runs."""
    data = json.loads(path.read_text("utf-8")) if path.is_file() else {"runs": {}}
    data["runs"][key] = entry
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    dataset = DatasetName(args.dataset)
    setting = InputSetting(args.input_setting) if args.input_setting else None
    check_input_setting(dataset, setting)
    env = load_env_config(args.env_config)
    sample_tag = None
    if args.limit is not None:
        sample_tag = f"limit{args.limit}"
    elif args.shortest is not None:
        sample_tag = f"shortest{args.shortest}"
    sampled = sample_tag is not None
    dtype = "float32" if args.report_only else resolve_cache_dtype(env, args.dtype, sampled)

    output_dir = None
    if not args.report_only:
        output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(
            dataset, setting, sample_tag
        )
        check_output_dir(output_dir, sampled)

    splits = load_dataset_splits(dataset, data_dir=args.data_dir)
    unknown = sorted(set(args.splits) - set(splits))
    if unknown:
        raise SystemExit(f"unknown split(s) {unknown}; available: {sorted(splits)}")

    tokenizer = None
    if args.shortest is not None:
        from transformers import AutoTokenizer

        from layer_fusion.inputs import build_tokenized_inputs

        tokenizer = AutoTokenizer.from_pretrained(env.model_name)
        records_by_split = {}
        for name in args.splits:
            lengths = build_tokenized_inputs(
                splits[name], tokenizer, setting, env.lfull_max_length_cap
            ).input_lengths
            records_by_split[name] = select_shortest(splits[name], lengths, args.shortest)
    else:
        records_by_split = {
            name: sample_records(splits[name], args.limit, args.seed) for name in args.splits
        }
    label = cache_name(dataset, setting, sep="/")
    for name, records in records_by_split.items():
        print(f"[{label}] {name}: {len(records)} docs")

    reports = build_cache(
        records_by_split,
        env,
        output_dir,
        input_setting=setting,
        dtype=dtype,
        device=args.device,
        batch_size=args.batch_size,
        tokenizer=tokenizer,
    )
    for name, report in reports.items():
        print(f"[{name}] {json.dumps(report, ensure_ascii=False)}")
    if output_dir is not None:
        print(f"[saved] {output_dir} ({dtype})")

    if args.report_output:
        key = label + (f"/{sample_tag}" if args.shortest is not None else "")
        merge_report(
            Path(args.report_output),
            key,
            {
                "date": date.today().isoformat(),
                "platform": platform.platform(),
                "device": args.device,
                "dtype_extracted": dtype,
                "limit_per_split": args.limit,
                "shortest_per_split": args.shortest,
                "seed": args.seed,
                "include_special_tokens_in_pool": env.include_special_tokens_in_pool,
                "splits": reports,
            },
        )
        print(f"[report] merged into {args.report_output} as {key!r}")


if __name__ == "__main__":
    main()
