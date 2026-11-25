"""Reporting module: summary tables, learning curves, layer-weight & length plots (SPEC.md)."""

from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import fields
from functools import cache
from pathlib import Path
from typing import NamedTuple

from layer_fusion.contracts import (
    FusionMethod,
    InputSetting,
    JudgmentReport,
    LayerSubset,
    LengthBucket,
    ResultRecord,
    RunConfig,
)
from layer_fusion.inputs import assign_length_bucket, compute_view_ratio, max_length_cap
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


# The only RunConfig fields the seeds of one cell may differ in: the seed, and the learning rate
# the first seed selects and the other seeds reuse (SPEC §결정 필요 사항, T-06a). The judgment
# seam's one-cell rule is the same, plus fusion_method, which is part of the cell here.
_SEED_FREE_FIELDS = frozenset({"seed", "fixed_lr", "lr_selection_seed"})

# Every artifact filename generate_report_artifacts can write; an output directory that already
# contains any of these is refused before writing (AGENTS.md §7: no overwrite or stale leftovers).
_REPORT_ARTIFACTS = (
    "summary_table.csv",
    "length_bucket_table.csv",
    "view_ratio_table.csv",
    "layer_weights_table.csv",
    "hypothesis_table.csv",
    "learning_curves.png",
    "layer_weights.png",
)

_Document = tuple[int | None, LengthBucket | None]  # (token_length, length_bucket)


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
    if not all(math.isfinite(v) for v in values):
        raise ValueError(f"values must be finite, got {list(values)}")
    return statistics.fmean(values), (statistics.stdev(values) if len(values) > 1 else None)


def _documents(record: ResultRecord) -> dict[str, _Document]:
    """The run's evaluated documents; a repeated doc_id or a length/bucket mismatch is refused.

    The runner records ``token_length`` for every dataset but a length bucket only under an
    input setting (IMDB), so MNLI/SST-2 predictions carry a length and no bucket.
    """
    has_setting = record.config.input_setting is not None
    documents: dict[str, _Document] = {}
    for pred in record.predictions:
        if pred.doc_id in documents:
            raise ValueError(f"run {record.run_id!r} evaluates doc {pred.doc_id!r} twice")
        expected = (
            assign_length_bucket(pred.token_length)
            if has_setting and pred.token_length is not None
            else None
        )
        if pred.length_bucket is not expected:
            raise ValueError(
                f"doc {pred.doc_id!r} of run {record.run_id!r}: token length "
                f"{pred.token_length} does not match length bucket {pred.length_bucket}"
            )
        documents[pred.doc_id] = (pred.token_length, pred.length_bucket)
    return documents


def _check_learning_rate(key: _Cell, cell: Sequence[ResultRecord]) -> None:
    """Enforce SPEC's one-selection-run-per-cell and identical reused-LR rule (T-06a).

    Inside one cell, exactly one run selects the learning rate (``config.fixed_lr is None``) and
    every other seed reuses that exact learning rate while pointing ``lr_selection_seed`` back to
    that selection run.
    """
    selection_runs: list[ResultRecord] = []
    reuse_runs: list[ResultRecord] = []
    for record in cell:
        if not (math.isfinite(record.selected_lr) and record.selected_lr > 0):
            raise ValueError(
                f"run {record.run_id!r} of cell {key}: selected_lr must be finite and positive, "
                f"got {record.selected_lr}"
            )
        config = record.config
        if config.fixed_lr is None:
            if config.lr_selection_seed is not None:
                raise ValueError(
                    f"run {record.run_id!r} of cell {key}: selection run (fixed_lr=None) "
                    f"cannot set config.lr_selection_seed={config.lr_selection_seed}"
                )
            if record.lr_selection_seed not in (None, config.seed):
                raise ValueError(
                    f"run {record.run_id!r} of cell {key}: selection run seed {config.seed} "
                    f"disagrees with record.lr_selection_seed={record.lr_selection_seed}"
                )
            selection_runs.append(record)
        else:
            if config.lr_selection_seed is None:
                raise ValueError(
                    f"run {record.run_id!r} of cell {key}: reuse run (fixed_lr={config.fixed_lr}) "
                    "must specify config.lr_selection_seed"
                )
            if config.lr_selection_seed == config.seed:
                raise ValueError(
                    f"run {record.run_id!r} of cell {key}: reuse run lr_selection_seed cannot "
                    f"reference its own seed {config.seed}"
                )
            if record.selected_lr != config.fixed_lr:
                raise ValueError(
                    f"run {record.run_id!r} of cell {key}: selected_lr {record.selected_lr} "
                    f"!= config.fixed_lr {config.fixed_lr}"
                )
            if record.lr_selection_seed not in (None, config.lr_selection_seed):
                raise ValueError(
                    f"run {record.run_id!r} of cell {key}: record.lr_selection_seed "
                    f"{record.lr_selection_seed} != config.lr_selection_seed "
                    f"{config.lr_selection_seed}"
                )
            reuse_runs.append(record)

    if len(selection_runs) != 1:
        raise ValueError(
            f"cell {key} must have exactly one LR selection run (fixed_lr=None), "
            f"got {len(selection_runs)}: {[r.run_id for r in selection_runs]}"
        )
    selection = selection_runs[0]
    for record in reuse_runs:
        if record.config.lr_selection_seed != selection.config.seed:
            raise ValueError(
                f"run {record.run_id!r} of cell {key} references lr_selection_seed "
                f"{record.config.lr_selection_seed}, but the cell's selection run is seed "
                f"{selection.config.seed} ({selection.run_id!r})"
            )
        if record.selected_lr != selection.selected_lr:
            raise ValueError(
                f"run {record.run_id!r} of cell {key} used selected_lr={record.selected_lr}, "
                f"but selection run {selection.run_id!r} chose {selection.selected_lr}"
            )


def _check_cell(key: _Cell, cell: Sequence[ResultRecord]) -> None:
    """Refuse what would make a seed mean +- std lie about the cell.

    That is a seed counted twice, a run trained under another setting (epochs, batch size, ...),
    mixed learning rates or broken LR-reuse provenance, or seeds evaluated on different documents.
    """
    reference = cell[0]
    reference_docs = _documents(reference)
    seeds = {reference.config.seed}
    for record in cell[1:]:
        if record.config.seed in seeds:
            raise ValueError(f"cell {key} has more than one record for seed {record.config.seed}")
        seeds.add(record.config.seed)
        differing = [
            f.name
            for f in fields(RunConfig)
            if f.name not in _SEED_FREE_FIELDS
            and getattr(record.config, f.name) != getattr(reference.config, f.name)
        ]
        if differing:
            raise ValueError(
                f"runs {reference.run_id!r} and {record.run_id!r} of cell {key} differ in "
                f"{differing}; seeds may differ only in {sorted(_SEED_FREE_FIELDS)}"
            )
        if _documents(record) != reference_docs:
            raise ValueError(
                f"runs {reference.run_id!r} and {record.run_id!r} of cell {key}: seeds must be "
                "evaluated on the same documents (doc ids, token lengths and buckets)"
            )
    _check_learning_rate(key, cell)


def _cells(main_records: Sequence[ResultRecord]) -> dict[_Cell, list[ResultRecord]]:
    """Group main records (debug and reproduction already dropped) into checked cells."""
    cells: dict[_Cell, list[ResultRecord]] = defaultdict(list)
    for record in main_records:
        cells[_cell_key(record)].append(record)
    for key, cell in cells.items():
        _check_cell(key, cell)
    return cells


def _summary_rows(cells: dict[_Cell, list[ResultRecord]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for key, cell in cells.items():
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


def generate_summary_table(records: Sequence[ResultRecord]) -> list[dict[str, object]]:
    """Build table with error rate (mean +- std), trainable params, and train time.

    Debug and reproduction records are left out (ADR 0004, 0009).
    """
    return _summary_rows(_cells(select_main_records(records)))


def _length_bucket_rows(
    cells: dict[_Cell, list[ResultRecord]], min_bucket_samples: int | None
) -> list[dict[str, object]]:
    """Per cell and length bucket: document count and error rate (mean +- std over seeds).

    The seeds of a cell share their documents (``_check_cell``), so every seed has the same
    count per bucket. ``below_min_samples`` stays ``None`` while the threshold is undecided
    (SPEC §결정 필요 사항).
    """
    rows: list[dict[str, object]] = []
    for key, cell in cells.items():
        for bucket in LengthBucket:
            in_bucket = [[p for p in r.predictions if p.length_bucket is bucket] for r in cell]
            n_docs = len(in_bucket[0])
            if n_docs == 0:
                continue
            error_mean, error_std = _mean_std(
                [sum(p.prediction != p.label for p in preds) / n_docs for preds in in_bucket]
            )
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


def _view_ratio_rows(main_records: Sequence[ResultRecord]) -> list[dict[str, object]]:
    """Mean local-layer view ratio per input setting and length bucket, one row set per setting.

    The view ratio is a property of the document and the model input, not of the fusion method,
    so each document is counted once per input setting however many runs evaluated it; those
    runs must share their documents. The ratio uses the input length the model actually saw:
    L128/L512 cut a longer review to 128/512 tokens (SPEC US 18, 19). Lfull does not cut; a
    review over the Lfull cap is refused at tokenization (inputs.py), so its input length is
    its token length.
    """
    documents: dict[str, dict[str, _Document]] = {}
    for record in main_records:
        setting = record.config.input_setting
        if setting is None or not record.predictions:
            continue
        docs = _documents(record)
        if docs != documents.setdefault(setting.value, docs):
            raise ValueError(
                f"runs of input setting {setting.value} must be evaluated on the same documents; "
                f"run {record.run_id!r} differs"
            )

    rows: list[dict[str, object]] = []
    for setting_name, docs in documents.items():
        cap = max_length_cap(InputSetting(setting_name), None)  # None for Lfull: not cut
        ratios: dict[LengthBucket, list[float]] = defaultdict(list)
        for length, bucket in docs.values():
            if length is None or bucket is None:
                continue
            ratios[bucket].append(_cached_view_ratio(length if cap is None else min(length, cap)))
        if not ratios:
            continue
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


def _layer_weight_rows(cells: dict[_Cell, list[ResultRecord]]) -> list[dict[str, object]]:
    """Seed mean +- std of dataset-level layer weights (F2) per cell and layer.

    The embedding vector has no layer number (GLOSSARY), so its row is labelled "embedding".
    """
    rows: list[dict[str, object]] = []
    for key, cell in cells.items():
        missing = [r.run_id for r in cell if r.layer_weights is None]
        if missing and (key.fusion_method == FusionMethod.F2.value or len(missing) < len(cell)):
            raise ValueError(
                f"layer_weights missing for runs {missing} of cell {key}: every F2 run and "
                "every seed of a cell with layer weights must carry them"
            )
        if missing:
            continue
        slots = slots_for_subset(LayerSubset(key.layer_subset))
        weights = [r.layer_weights or [] for r in cell]
        for record, record_weights in zip(cell, weights, strict=True):
            if len(record_weights) != len(slots):
                raise ValueError(
                    f"layer_weights of run {record.run_id!r} has {len(record_weights)} entries, "
                    f"but layer subset {key.layer_subset} has {len(slots)} slots"
                )
        for position, slot in enumerate(slots):
            mean, std = _mean_std([w[position] for w in weights])
            rows.append(
                {
                    **key._asdict(),
                    "layer": "embedding" if slot == EMBEDDING_SLOT else slot,
                    "n_seeds": len(cell),
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
            yerr=[float(r["weight_std"] or 0.0) for r in panel],  # one seed: no error bar
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


def _plot_learning_curves(main_records: Sequence[ResultRecord], path: Path) -> Path | None:
    """Tuning-split error rate per epoch, one panel per experiment condition, one line per seed.

    Seeds stop at different epochs under early stopping, so lines are drawn per seed instead of
    averaged. Returns ``None`` when no main record carries a learning curve.
    """
    curves = [r for r in main_records if r.learning_curve]
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


def _check_output_dir(out: Path) -> None:
    """Refuse an output directory that already holds report artifacts (AGENTS.md §7)."""
    if not out.exists():
        return
    if not out.is_dir():
        raise FileExistsError(f"output_dir {out} exists and is not a directory")
    existing = [name for name in _REPORT_ARTIFACTS if (out / name).exists()]
    if existing:
        raise FileExistsError(
            f"output directory {out} already contains report artifacts {existing}; "
            "refusing to overwrite or leave stale files (AGENTS.md §7)"
        )


def generate_report_artifacts(
    records: Sequence[ResultRecord],
    judgment: JudgmentReport,
    output_dir: str | Path,
    *,
    min_bucket_samples: int | None = None,
) -> list[Path]:
    """Export result tables, learning curves, layer weight plots, bucket & view ratio tables.

    Debug and reproduction records are dropped here, once, before any table or figure is built
    (ADR 0004, 0009). A table with no rows is not written. Existing report artifacts in
    ``output_dir`` are refused before writing so previous outputs are neither overwritten nor left
    behind alongside a newer export (AGENTS.md §7). ``min_bucket_samples`` is the length-bucket
    minimum sample rule (TBD until locked).
    """
    main = select_main_records(records)
    cells = _cells(main)
    tables = {
        "summary_table.csv": _summary_rows(cells),
        "length_bucket_table.csv": _length_bucket_rows(cells, min_bucket_samples),
        "view_ratio_table.csv": _view_ratio_rows(main),
        "layer_weights_table.csv": _layer_weight_rows(cells),
        "hypothesis_table.csv": _hypothesis_rows(judgment),
    }
    out = Path(output_dir)
    _check_output_dir(out)
    has_curves = any(r.learning_curve for r in main)
    if not any(tables.values()) and not has_curves:
        return []
    out.mkdir(parents=True, exist_ok=True)
    paths = [_write_csv(out / name, rows) for name, rows in tables.items() if rows]
    curves_path = _plot_learning_curves(main, out / "learning_curves.png")
    if curves_path is not None:
        paths.append(curves_path)
    if tables["layer_weights_table.csv"]:
        paths.append(
            _plot_layer_weights(tables["layer_weights_table.csv"], out / "layer_weights.png")
        )
    return paths
