"""Reporting module: summary tables, learning curves, layer-weight & length plots (SPEC.md)."""

from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from collections.abc import Sequence
from functools import cache
from pathlib import Path
from typing import NamedTuple

from layer_fusion.contracts import (
    InputSetting,
    JudgmentReport,
    LayerSubset,
    LengthBucket,
    ResultRecord,
)
from layer_fusion.inputs import compute_view_ratio, max_length_cap
from layer_fusion.judgment import select_main_records
from layer_fusion.layers import EMBEDDING_SLOT, slots_for_subset


class _Cell(NamedTuple):
    """One experiment cell: records that differ only in seed are pooled (ADR 0007)."""

    dataset: str
    data_amount: str
    backbone_condition: str
    input_setting: str  # "" when the dataset has no input setting (MNLI, SST-2)
    fusion_method: str
    layer_subset: str

    @property
    def condition(self) -> tuple[str, ...]:
        """The cell without fusion method and layer subset: what methods are compared under."""
        return self[:4]


def _cell_key(record: ResultRecord) -> _Cell:
    config = record.config
    return _Cell(
        config.dataset.value,
        config.data_amount.value,
        config.backbone_condition.value,
        "" if config.input_setting is None else config.input_setting.value,
        config.fusion_method.value,
        config.layer_subset.value,
    )


def _mean_std(values: Sequence[float]) -> tuple[float, float | None]:
    """Mean and sample std (ddof=1); the std is ``None`` for one seed, never a fake 0."""
    return statistics.fmean(values), (statistics.stdev(values) if len(values) > 1 else None)


def _group_main_records(
    records: Sequence[ResultRecord],
) -> dict[_Cell, list[ResultRecord]]:
    """Group records into experiment cells; debug and reproduction records never enter."""
    cells: dict[_Cell, list[ResultRecord]] = defaultdict(list)
    for record in select_main_records(records):
        cells[_cell_key(record)].append(record)
    return cells


def generate_summary_table(records: Sequence[ResultRecord]) -> list[dict[str, object]]:
    """Build table with error rate (mean +- std), trainable params, and train time."""
    rows: list[dict[str, object]] = []
    for key, cell in _group_main_records(records).items():
        params = {r.trainable_params for r in cell}
        if len(params) > 1:
            raise ValueError(f"trainable_params differ inside one cell {key}: {sorted(params)}")
        error_mean, error_std = _mean_std([r.eval_error_rate for r in cell])
        seconds_mean, seconds_std = _mean_std([r.train_seconds for r in cell])
        rows.append(
            {
                **key._asdict(),
                "n_seeds": len(cell),
                "eval_error_rate_mean": error_mean,
                "eval_error_rate_std": error_std,
                "trainable_params": cell[0].trainable_params,
                "train_seconds_mean": seconds_mean,
                "train_seconds_std": seconds_std,
            }
        )
    return rows


def _length_bucket_rows(
    records: Sequence[ResultRecord], min_bucket_samples: int | None
) -> list[dict[str, object]]:
    """Per cell and length bucket: document count and error rate (mean +- std over seeds).

    ``below_min_samples`` stays ``None`` while the threshold is undecided (SPEC §결정 필요 사항).
    """
    rows: list[dict[str, object]] = []
    for key, cell in _group_main_records(records).items():
        for bucket in LengthBucket:
            in_bucket = [[p for p in r.predictions if p.length_bucket is bucket] for r in cell]
            counts = {len(preds) for preds in in_bucket}
            if counts == {0}:
                continue
            if len(counts) > 1:
                raise ValueError(
                    f"n_docs in bucket {bucket.value!r} differs across seeds {sorted(counts)} "
                    f"in cell {key}: seeds must be evaluated on the same documents"
                )
            error_rates = [
                sum(p.prediction != p.label for p in preds) / len(preds) for preds in in_bucket
            ]
            error_mean, error_std = _mean_std(error_rates)
            n_docs = counts.pop()
            rows.append(
                {
                    **key._asdict(),
                    "length_bucket": bucket.value,
                    "n_docs": n_docs,
                    "n_seeds": len(cell),
                    "error_rate_mean": error_mean,
                    "error_rate_std": error_std,
                    "below_min_samples": (
                        None if min_bucket_samples is None else n_docs < min_bucket_samples
                    ),
                }
            )
    return rows


_cached_view_ratio = cache(compute_view_ratio)  # O(length) each; many documents share a length


def _view_ratio_rows(records: Sequence[ResultRecord]) -> list[dict[str, object]]:
    """Mean local-layer view ratio per input setting and length bucket, one row set per setting.

    The view ratio is a property of the document and the model input, not of the fusion method,
    so each document is counted once per input setting however many runs evaluated it. The
    ratio uses the input length the model actually saw (``min(token_length, cap)``): an L128
    input has 128 tokens whatever the original review length (SPEC US 18, 19).
    """
    documents: dict[str, dict[str, tuple[int, LengthBucket]]] = defaultdict(dict)
    for record in select_main_records(records):
        setting = record.config.input_setting
        if setting is None:
            continue
        for pred in record.predictions:
            if pred.token_length is None or pred.length_bucket is None:
                continue
            seen = documents[setting.value].setdefault(
                pred.doc_id, (pred.token_length, pred.length_bucket)
            )
            if seen != (pred.token_length, pred.length_bucket):
                raise ValueError(
                    f"doc {pred.doc_id!r} has different token length/bucket across runs "
                    f"in input setting {setting.value}"
                )

    rows: list[dict[str, object]] = []
    for setting_name, docs in documents.items():
        cap = max_length_cap(InputSetting(setting_name), None)  # Lfull: never truncated
        ratios: dict[LengthBucket, list[float]] = defaultdict(list)
        for length, bucket in docs.values():
            ratios[bucket].append(_cached_view_ratio(length if cap is None else min(length, cap)))
        groups = [(b.value, ratios[b]) for b in LengthBucket if ratios[b]]
        groups.append(("all", [r for b in LengthBucket for r in ratios[b]]))
        for bucket_name, values in groups:
            rows.append(
                {
                    "input_setting": setting_name,
                    "length_bucket": bucket_name,
                    "n_docs": len(values),
                    "mean_view_ratio": statistics.fmean(values),
                }
            )
    return rows


def _cell_title(key: Sequence[str]) -> str:
    return " / ".join(part for part in key if part)


def _layer_weight_rows(records: Sequence[ResultRecord]) -> list[dict[str, object]]:
    """Seed mean +- std of dataset-level layer weights (F2) per cell and layer.

    The embedding vector has no layer number (GLOSSARY), so its row is labelled "embedding".
    """
    rows: list[dict[str, object]] = []
    for key, cell in _group_main_records(records).items():
        with_weights = [r for r in cell if r.layer_weights is not None]
        if not with_weights:
            continue
        slots = slots_for_subset(LayerSubset(key.layer_subset))
        for record in with_weights:
            n_weights = len(record.layer_weights or ())
            if n_weights != len(slots):
                raise ValueError(
                    f"layer_weights of run {record.run_id!r} has {n_weights} entries, "
                    f"but layer subset {key.layer_subset} has {len(slots)} slots"
                )
        for position, slot in enumerate(slots):
            mean, std = _mean_std([r.layer_weights[position] for r in with_weights])
            rows.append(
                {
                    **key._asdict(),
                    "layer": "embedding" if slot == EMBEDDING_SLOT else slot,
                    "n_seeds": len(with_weights),
                    "weight_mean": mean,
                    "weight_std": std,
                }
            )
    return rows


def _plot_layer_weights(rows: Sequence[dict[str, object]], path: Path) -> Path:
    """Bar chart of seed-mean layer weights (error bars = seed std), one panel per cell."""
    from matplotlib.figure import Figure  # lazy: tables work without matplotlib

    panels: dict[_Cell, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        panels[_Cell(*(str(row[name]) for name in _Cell._fields))].append(row)
    fig = Figure(figsize=(8, 2.8 * len(panels)), layout="constrained")
    axes = fig.subplots(len(panels), 1, squeeze=False)[:, 0]
    for ax, (key, panel) in zip(axes, panels.items(), strict=True):
        has_embedding = any(r["layer"] == "embedding" for r in panel)
        ax.bar(
            ["emb" if r["layer"] == "embedding" else str(r["layer"]) for r in panel],
            [float(r["weight_mean"]) for r in panel],
            yerr=[float(r["weight_std"] or 0.0) for r in panel],
            capsize=2,
        )
        ax.set_title(_cell_title(key), fontsize=9)
        ax.set_xlabel(
            "layer" + (" (emb = embedding vector, no layer number)" if has_embedding else "")
        )
        ax.set_ylabel("weight (mean +- std over seeds)")
    fig.savefig(path, dpi=150)
    return path


def _method_label(record: ResultRecord) -> str:
    """Legend label: the fusion method, plus the layer subset when it is not the default."""
    config = record.config
    suffix = "" if config.layer_subset is LayerSubset.ALL else f"({config.layer_subset.value})"
    return config.fusion_method.value + suffix


def _plot_learning_curves(records: Sequence[ResultRecord], path: Path) -> Path | None:
    """Tuning-split error rate per epoch, one panel per experiment condition, one line per seed.

    Seeds stop at different epochs under early stopping, so lines are drawn per seed instead of
    averaged. Returns ``None`` when no main record carries a learning curve.
    """
    curves = [r for r in select_main_records(records) if r.learning_curve]
    if not curves:
        return None
    from matplotlib.figure import Figure  # lazy: tables work without matplotlib
    from matplotlib.ticker import MaxNLocator

    panels: dict[tuple[str, ...], list[ResultRecord]] = defaultdict(list)
    for record in curves:
        panels[_cell_key(record).condition].append(record)
    fig = Figure(figsize=(7, 3.2 * len(panels)), layout="constrained")
    axes = fig.subplots(len(panels), 1, squeeze=False)[:, 0]
    for ax, (key, panel) in zip(axes, panels.items(), strict=True):
        colors = {
            label: f"C{i}" for i, label in enumerate(sorted({_method_label(r) for r in panel}))
        }
        drawn: set[str] = set()
        for record in panel:
            label = _method_label(record)
            ax.plot(
                [pt["epoch"] for pt in record.learning_curve],
                [pt["tuning_error"] for pt in record.learning_curve],
                color=colors[label],
                alpha=0.7,
                marker="o",
                markersize=3,
                label=None if label in drawn else label,
            )
            drawn.add(label)
        ax.set_title(_cell_title(key), fontsize=9)
        ax.set_xlabel("epoch")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylabel("tuning-split error rate")
        ax.legend(fontsize=8)
    fig.savefig(path, dpi=150)
    return path


def _hypothesis_rows(judgment: JudgmentReport) -> list[dict[str, object]]:
    """One row per judged hypothesis, copied from the judgment seam (nothing is re-judged)."""
    return [
        {
            "hypothesis_id": hypothesis_id,
            "kind": j.kind,
            "verdict": j.verdict.value,
            "mean_diff": j.mean_diff,
            "seed_std": j.seed_std,
            "mcnemar_p_value": j.mcnemar_p_value,
        }
        for hypothesis_id, j in judgment.judgments.items()
    ]


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> Path:
    """Write rows as UTF-8 CSV; ``None`` becomes an empty cell. Empty input -> header-less file."""
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: "" if v is None else v for k, v in row.items()})
    return path


def generate_report_artifacts(
    records: Sequence[ResultRecord],
    judgment: JudgmentReport,
    output_dir: str | Path,
    *,
    min_bucket_samples: int | None = None,
) -> list[Path]:
    """Export result tables, learning curves, layer weight plots, bucket & view ratio tables.

    Only main-track, non-debug records are used (ADR 0004, 0009). A table with no rows is not
    written. ``min_bucket_samples`` is the length-bucket minimum sample rule (TBD until locked).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tables = {
        "summary_table.csv": generate_summary_table(records),
        "length_bucket_table.csv": _length_bucket_rows(records, min_bucket_samples),
        "view_ratio_table.csv": _view_ratio_rows(records),
        "layer_weights_table.csv": _layer_weight_rows(records),
        "hypothesis_table.csv": _hypothesis_rows(judgment),
    }
    paths = [_write_csv(out / name, rows) for name, rows in tables.items() if rows]
    curves_path = _plot_learning_curves(records, out / "learning_curves.png")
    if curves_path is not None:
        paths.append(curves_path)
    if tables["layer_weights_table.csv"]:
        paths.append(
            _plot_layer_weights(tables["layer_weights_table.csv"], out / "layer_weights.png")
        )
    return paths
