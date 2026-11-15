"""Dataset loading, 10% tuning split, and small-data sampling (SPEC.md, ADR 0003)."""

from __future__ import annotations

from layer_fusion.contracts import DatasetName, DocumentRecord


def load_dataset_splits(
    dataset: DatasetName,
    data_dir: str | None = None,
) -> dict[str, list[DocumentRecord]]:
    """Load dataset splits with stable doc_ids and MNLI matched/mismatched tags.

    MNLI / SST-2 use public validation as eval split; IMDB uses test as eval split.
    """
    raise NotImplementedError("TODO: Implement dataset loading (T-03)")


def make_tuning_split(
    train_records: list[DocumentRecord],
    fraction: float = 0.10,
    seed: int = 0,
) -> tuple[list[DocumentRecord], list[DocumentRecord]]:
    """Split train_records into (train_subset, tuning_split) with tuning used only for LR choice."""
    raise NotImplementedError("TODO: Implement 10% tuning split (T-03)")


def sample_small_dataset(
    train_records: list[DocumentRecord],
    per_class: int = 1000,
    seed: int = 0,
) -> tuple[list[DocumentRecord], list[str]]:
    """Sample `per_class` examples per class for a given seed.

    Within the same seed, all fusion methods must receive the exact same doc_ids.
    """
    raise NotImplementedError("TODO: Implement small-data per-class sampling (T-03)")
