"""Reporting module: summary tables, learning curves, layer-weight & length plots (SPEC.md)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from layer_fusion.contracts import JudgmentReport, ResultRecord


def generate_summary_table(records: Sequence[ResultRecord]) -> list[dict[str, object]]:
    """Build table with error rate (mean +- std), trainable params, and train time."""
    raise NotImplementedError("TODO: Implement summary table generation (T-09)")


def generate_report_artifacts(
    records: Sequence[ResultRecord],
    judgment: JudgmentReport,
    output_dir: str | Path,
) -> list[Path]:
    """Export result tables, learning curves, layer weight plots, bucket & view ratio tables."""
    raise NotImplementedError("TODO: Implement report artifact generation (T-09)")
