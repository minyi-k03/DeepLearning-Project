"""Layer vector provider interface and cache metadata validation (SPEC.md §모듈 구성)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from layer_fusion.contracts import CacheMetadata, DocumentRecord, RunConfig
from layer_fusion.layers import HIDDEN_SIZE, NUM_LAYER_VECTORS


class CacheConfigMismatchError(ValueError):
    """Raised when a provider's creation metadata does not match the RunConfig."""


@dataclass
class LayerVectorBatch:
    """Container returned by a LayerVectorProvider for a split."""

    doc_ids: list[str]
    labels: np.ndarray            # shape [N]
    vectors: np.ndarray           # shape [N, 23, 768]
    metadata: CacheMetadata
    genre_splits: list[str | None] | None = None
    token_lengths: list[int] | None = None


class LayerVectorProvider(Protocol):
    """Interface injecting layer vectors into the Execution seam."""

    @property
    def metadata(self) -> CacheMetadata:
        ...

    def validate_for_run(self, config: RunConfig) -> None:
        ...

    def get_split(self, split_name: str) -> LayerVectorBatch:
        ...


def validate_metadata_match(actual: CacheMetadata, expected: CacheMetadata) -> None:
    """Verify actual cache metadata matches expected run metadata; raise on mismatch."""
    if actual != expected:
        raise CacheConfigMismatchError(
            f"Cache metadata mismatch.\nExpected: {expected}\nActual:   {actual}"
        )


class CacheProvider:
    """Loads pre-extracted [N, 23, 768] layer vectors from disk (frozen backbone mode)."""

    def __init__(self, cache_dir: str) -> None:
        self.cache_dir = cache_dir
        raise NotImplementedError("TODO: Implement CacheProvider (T-04)")

    @property
    def metadata(self) -> CacheMetadata:
        raise NotImplementedError

    def validate_for_run(self, config: RunConfig) -> None:
        if config.expected_cache_meta is not None:
            validate_metadata_match(self.metadata, config.expected_cache_meta)

    def get_split(self, split_name: str) -> LayerVectorBatch:
        raise NotImplementedError


class BackboneProvider:
    """Computes layer vectors live through ModernBERT-base (finetune mode)."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        raise NotImplementedError("TODO: Implement BackboneProvider (T-04/T-06)")


class FakeLayerVectorProvider:
    """Deterministic in-memory provider for CPU-only Seam 1 tests without GPU/data."""

    def __init__(
        self,
        splits: dict[str, LayerVectorBatch],
        metadata: CacheMetadata,
    ) -> None:
        self._splits = splits
        self._metadata = metadata
        for batch in splits.values():
            if batch.vectors.ndim != 3 or batch.vectors.shape[1:] != (
                NUM_LAYER_VECTORS,
                HIDDEN_SIZE,
            ):
                raise ValueError(
                    f"Expected shape [N, {NUM_LAYER_VECTORS}, {HIDDEN_SIZE}], "
                    f"got {batch.vectors.shape}"
                )

    @property
    def metadata(self) -> CacheMetadata:
        return self._metadata

    def validate_for_run(self, config: RunConfig) -> None:
        if config.expected_cache_meta is not None:
            validate_metadata_match(self._metadata, config.expected_cache_meta)

    def get_split(self, split_name: str) -> LayerVectorBatch:
        if split_name not in self._splits:
            raise KeyError(f"Unknown split {split_name!r}")
        return self._splits[split_name]

    @classmethod
    def make_synthetic(
        cls,
        num_classes: int = 3,
        train_size: int = 60,
        eval_size: int = 30,
        seed: int = 0,
        metadata: CacheMetadata | None = None,
        doc_records: Sequence[DocumentRecord] | None = None,
    ) -> FakeLayerVectorProvider:
        """Create a small synthetic provider for local CPU seam testing."""
        raise NotImplementedError("TODO: Implement synthetic fixture builder if needed (T-06)")
