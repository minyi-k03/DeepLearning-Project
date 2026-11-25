#!/usr/bin/env python3
"""T-08a: result records -> Judgment seam -> report artifacts (tables and figures).

Wires existing pieces only (records.load_result_records, judgment.load_decision_rules /
judge_hypotheses, reporting.generate_report_artifacts); nothing is judged or summarised here.

- Main-experiment judgment needs the frozen rules (``status: locked``, the 11/30 lock).
  ``--allow-draft-rules`` runs on a draft file for smoke/debug checks only; TBD values stay
  TBD (AGENTS.md §2).
- Debug and reproduction-track records are read but never judged or tabulated (ADR 0004,
  0009); the printed summary says how many were left out.
- An output directory that already holds report artifacts is refused, never overwritten
  (AGENTS.md §7).

Example (debug smoke):
  python scripts/judge_and_report.py --runs-dir results/debug_runs/t08a \\
      --allow-draft-rules --output-dir results/debug_runs/t08a_report
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate hypotheses from result records and frozen decision rules."
    )
    parser.add_argument("--runs-dir", default="results/runs")
    parser.add_argument("--rules", default="configs/decision_rules.yaml")
    parser.add_argument("--output-dir", default="results/reports")
    parser.add_argument(
        "--allow-draft-rules",
        action="store_true",
        help="accept rules whose status is not 'locked' (smoke/debug checks only)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> list[Path]:
    """Judge the records in ``--runs-dir`` and write the report; return the written files."""
    args = parse_args(argv)

    from layer_fusion.judgment import judge_hypotheses, load_decision_rules, select_main_records
    from layer_fusion.records import load_result_records
    from layer_fusion.reporting import generate_report_artifacts

    try:
        rules = load_decision_rules(args.rules, require_locked=not args.allow_draft_rules)
        records = load_result_records(args.runs_dir)
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"error: {exc}") from exc

    if rules.status != "locked":
        print(f"[judge] draft rules (not for the report): {args.rules} status={rules.status!r}")
    main_records = select_main_records(records)
    print(
        f"[judge] records: {len(records)} loaded, {len(main_records)} main, "
        f"{len(records) - len(main_records)} excluded (debug/reproduction)"
    )

    try:
        report = judge_hypotheses(records, rules)
        paths = generate_report_artifacts(records, report, args.output_dir)
    except (ValueError, FileExistsError) as exc:
        raise SystemExit(f"error: {exc}") from exc

    for hypothesis_id, judgment in report.judgments.items():
        print(f"[judge] {hypothesis_id}: {judgment.verdict.value} ({judgment.kind})")
    print(f"[report] {len(paths)} file(s) in {args.output_dir}")
    for path in paths:
        print(f"  {path.name}")
    return paths


if __name__ == "__main__":
    main(sys.argv[1:])
