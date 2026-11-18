"""Dataset loading, 10% tuning split, and small-data sampling (SPEC.md, ADR 0003).

Open decisions (SPEC.md §결정 필요 사항) are deliberately *not* fixed here:

- Whether the tuning split is one fixed split or re-drawn per seed is decided by the caller through
  the ``seed`` it passes to :func:`make_tuning_split`.
- Whether the small-data sample may include tuning-split documents is decided by which
  ``train_records`` the caller passes to :func:`sample_small_dataset`.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from layer_fusion.contracts import DatasetName, DocumentRecord

TRAIN_SPLIT = "train"
EVAL_SPLIT = "eval"
TUNING_SPLIT = "tuning"


@dataclass(frozen=True)
class EvalSource:
    """One public evaluation split and the genre tag its records receive."""

    hf_split: str
    genre_split: str | None = None


@dataclass(frozen=True)
class DatasetSpec:
    """Where a dataset lives on the Hugging Face Hub and how its rows map to DocumentRecord."""

    hf_path: str
    hf_config: str | None
    train_split: str
    eval_sources: tuple[EvalSource, ...]
    text_a_column: str
    text_b_column: str | None
    label_column: str
    num_classes: int


# MNLI / SST-2: GLUE test labels are hidden, so public validation is the eval set (ADR 0003).
# IMDB has no validation split: test is the eval set. AG News is the MNLI-rejection fallback.
DATASET_SPECS: dict[DatasetName, DatasetSpec] = {
    DatasetName.MNLI: DatasetSpec(
        hf_path="nyu-mll/glue",
        hf_config="mnli",
        train_split="train",
        eval_sources=(
            EvalSource("validation_matched", "matched"),
            EvalSource("validation_mismatched", "mismatched"),
        ),
        text_a_column="premise",
        text_b_column="hypothesis",
        label_column="label",
        num_classes=3,
    ),
    DatasetName.SST2: DatasetSpec(
        hf_path="nyu-mll/glue",
        hf_config="sst2",
        train_split="train",
        eval_sources=(EvalSource("validation"),),
        text_a_column="sentence",
        text_b_column=None,
        label_column="label",
        num_classes=2,
    ),
    DatasetName.IMDB: DatasetSpec(
        hf_path="stanfordnlp/imdb",
        hf_config=None,
        train_split="train",
        eval_sources=(EvalSource("test"),),
        text_a_column="text",
        text_b_column=None,
        label_column="label",
        num_classes=2,
    ),
    DatasetName.AG_NEWS: DatasetSpec(
        hf_path="fancyzhx/ag_news",
        hf_config=None,
        train_split="train",
        eval_sources=(EvalSource("test"),),
        text_a_column="text",
        text_b_column=None,
        label_column="label",
        num_classes=4,
    ),
}


def make_doc_id(dataset: DatasetName, hf_split: str, row_index: int) -> str:
    """Stable document ID: dataset + source split + row position (e.g. ``mnli-train-000012``).

    The source split name is part of the ID so MNLI matched/mismatched rows never collide.
    """
    return f"{DatasetName(dataset).value}-{hf_split}-{row_index:06d}"


def rows_to_records(
    rows: Iterable[Mapping[str, Any]],
    dataset: DatasetName,
    hf_split: str,
    split: str,
    genre_split: str | None = None,
) -> list[DocumentRecord]:
    """Convert raw dataset rows into DocumentRecords, keeping original order.

    Raises ``ValueError`` on a label outside ``range(num_classes)`` (e.g. GLUE's hidden ``-1``)
    instead of silently dropping the row.
    """
    dataset = DatasetName(dataset)
    spec = DATASET_SPECS[dataset]
    records: list[DocumentRecord] = []
    for index, row in enumerate(rows):
        label = int(row[spec.label_column])
        if not 0 <= label < spec.num_classes:
            raise ValueError(
                f"{make_doc_id(dataset, hf_split, index)}: label {label} is outside "
                f"0..{spec.num_classes - 1} for {dataset.value}"
            )
        text_b = row[spec.text_b_column] if spec.text_b_column else None
        records.append(
            DocumentRecord(
                doc_id=make_doc_id(dataset, hf_split, index),
                text_a=row[spec.text_a_column],
                label=label,
                text_b=text_b,
                split=split,
                genre_split=genre_split,
            )
        )
    return records


def load_dataset_splits(
    dataset: DatasetName,
    data_dir: str | None = None,
) -> dict[str, list[DocumentRecord]]:
    """Load dataset splits with stable doc_ids and MNLI matched/mismatched tags.

    MNLI / SST-2 use public validation as eval split; IMDB uses test as eval split.
    Returns ``{"train": [...], "eval": [...]}``. The MNLI eval list holds matched records first,
    then mismatched ones, each tagged through ``genre_split``. ``data_dir`` is the HF cache dir.
    Raw text is kept as-is (e.g. IMDB ``<br />`` tags are not stripped).
    """
    from datasets import load_dataset  # lazy: not needed by CPU seam tests

    dataset = DatasetName(dataset)
    spec = DATASET_SPECS[dataset]

    def fetch(hf_split: str) -> Any:
        return load_dataset(spec.hf_path, spec.hf_config, split=hf_split, cache_dir=data_dir)

    train = rows_to_records(fetch(spec.train_split), dataset, spec.train_split, TRAIN_SPLIT)
    eval_records: list[DocumentRecord] = []
    for source in spec.eval_sources:
        eval_records += rows_to_records(
            fetch(source.hf_split), dataset, source.hf_split, EVAL_SPLIT, source.genre_split
        )
    return {TRAIN_SPLIT: train, EVAL_SPLIT: eval_records}


def _rng_for(purpose: str, seed: int) -> random.Random:
    """Local RNG with a per-purpose stream.

    A string seed is hashed with SHA-512 (stable across processes, unaffected by PYTHONHASHSEED),
    so the tuning split and the small sample never share one random stream for the same seed.
    Sharing it would make the small sample a prefix of the tuning draw (identical documents).
    """
    return random.Random(f"layer_fusion.data:{purpose}:{seed}")


def _check_input(records: Sequence[DocumentRecord], allowed_splits: set[str]) -> None:
    """Reject empty input, duplicate doc_ids, and records from a split that must not be used."""
    if not records:
        raise ValueError("no records given")
    seen: set[str] = set()
    for record in records:
        if record.doc_id in seen:
            raise ValueError(f"duplicate doc_id {record.doc_id!r}")
        seen.add(record.doc_id)
        if record.split not in allowed_splits:
            raise ValueError(
                f"{record.doc_id}: split {record.split!r} is not allowed here "
                f"(expected one of {sorted(allowed_splits)})"
            )


def _group_by_label(records: Sequence[DocumentRecord]) -> dict[int, list[DocumentRecord]]:
    """Group by label; each group is sorted by doc_id so draws never depend on input order."""
    groups: dict[int, list[DocumentRecord]] = defaultdict(list)
    for record in records:
        groups[record.label].append(record)
    return {label: sorted(groups[label], key=lambda r: r.doc_id) for label in sorted(groups)}


def make_tuning_split(
    train_records: list[DocumentRecord],
    fraction: float = 0.10,
    *,
    seed: int,
) -> tuple[list[DocumentRecord], list[DocumentRecord]]:
    """Split train_records into (train_subset, tuning_split) with tuning used only for LR choice.

    The split is stratified by label (each class gives ``round(fraction * class_size)`` documents)
    and deterministic for a given ``seed``, independent of input order. Tuning records get
    ``split="tuning"``; both lists keep the original relative order. ``seed`` is required: pass a
    constant for one fixed split, or the run seed to re-draw it per seed (still open in SPEC).
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must be in (0, 1), got {fraction}")
    _check_input(train_records, {TRAIN_SPLIT})
    rng = _rng_for("tuning-split", seed)
    tuning_ids: set[str] = set()
    for label, group in _group_by_label(train_records).items():
        n_tuning = round(fraction * len(group))
        if not 0 < n_tuning < len(group):
            raise ValueError(
                f"class {label} has {len(group)} documents: a {fraction} tuning split would take "
                f"{n_tuning}, leaving an empty tuning or training part"
            )
        tuning_ids.update(r.doc_id for r in rng.sample(group, n_tuning))
    train_subset = [r for r in train_records if r.doc_id not in tuning_ids]
    tuning = [replace(r, split=TUNING_SPLIT) for r in train_records if r.doc_id in tuning_ids]
    return train_subset, tuning


def sample_small_dataset(
    train_records: list[DocumentRecord],
    per_class: int = 1000,
    *,
    seed: int,
    num_classes: int,
) -> tuple[list[DocumentRecord], list[str]]:
    """Sample `per_class` examples per class for a given seed.

    Within the same seed, all fusion methods must receive the exact same doc_ids.
    Returns ``(sampled_records, sampled_doc_ids)`` sorted by doc_id; the ID list is what the result
    record stores. ``seed`` is required so a forgotten seed cannot silently reuse one sample.
    Every label in ``range(num_classes)`` must be present with at least ``per_class`` documents.
    Eval records are rejected; whether tuning records may be included is decided by the caller.
    """
    if per_class <= 0:
        raise ValueError(f"per_class must be positive, got {per_class}")
    _check_input(train_records, {TRAIN_SPLIT, TUNING_SPLIT})
    groups = _group_by_label(train_records)
    if set(groups) != set(range(num_classes)):
        raise ValueError(f"labels {sorted(groups)} do not match range({num_classes})")
    rng = _rng_for("small-sample", seed)
    sampled: list[DocumentRecord] = []
    for label, group in groups.items():
        if len(group) < per_class:
            raise ValueError(
                f"class {label} has {len(group)} documents, fewer than per_class={per_class}"
            )
        sampled += rng.sample(group, per_class)
    sampled.sort(key=lambda r: r.doc_id)
    return sampled, [r.doc_id for r in sampled]
