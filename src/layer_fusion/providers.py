"""Layer vector provider interface and cache metadata validation (SPEC.md §모듈 구성)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from layer_fusion.contracts import CacheMetadata, DocumentRecord, RunConfig, RunKind
from layer_fusion.inputs import truncation_rule_name
from layer_fusion.layers import HIDDEN_SIZE, NUM_LAYER_VECTORS

SYNTHETIC_SPLITS = ("train", "eval")


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


def check_provider_for_run(metadata: CacheMetadata, config: RunConfig) -> None:
    """Fail before training when the provider was built under a different setting (SPEC US 26).

    - ``config.input_setting`` must equal ``metadata.input_setting`` (``None`` included, so an
      MNLI/SST-2 run cannot take an IMDB L128/L512/Lfull cache).
    - Main runs must declare ``expected_cache_meta``; only debug runs may skip it, so a
      forgotten expectation cannot let a cache from another setting slip through.
    - With ``expected_cache_meta`` every CacheMetadata field must match (ADR 0005).
    """
    if config.input_setting != metadata.input_setting:
        raise CacheConfigMismatchError(
            f"input_setting mismatch: run={config.input_setting}, "
            f"provider={metadata.input_setting}"
        )
    if config.expected_cache_meta is None:
        if config.run_kind == RunKind.MAIN:
            raise CacheConfigMismatchError(
                "main runs must set RunConfig.expected_cache_meta (only debug runs may omit it)"
            )
        return
    validate_metadata_match(metadata, config.expected_cache_meta)


class CacheProvider:
    """Loads pre-extracted [N, 23, 768] layer vectors from disk (frozen backbone mode)."""

    def __init__(self, cache_dir: str) -> None:
        self.cache_dir = cache_dir
        raise NotImplementedError("TODO: Implement CacheProvider (T-04)")

    @property
    def metadata(self) -> CacheMetadata:
        raise NotImplementedError

    def validate_for_run(self, config: RunConfig) -> None:
        check_provider_for_run(self.metadata, config)

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
        check_provider_for_run(self._metadata, config)

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
        """Create a small deterministic provider for CPU seam tests (splits ``train``/``eval``).

        Vectors are [N, 23, 768] float32 noise plus a small class-dependent mean shift, so
        heads trained on them have a learnable signal. Labels are class-balanced. With
        ``num_classes == 3`` (MNLI-like) the eval split is half matched / half mismatched.
        ``doc_records`` overrides doc_ids, labels, splits and genres (e.g. from data.py).
        """
        rng = np.random.default_rng(seed)
        class_means = rng.normal(0.0, 1.0, size=(num_classes, NUM_LAYER_VECTORS, HIDDEN_SIZE))
        if metadata is None:
            metadata = CacheMetadata(
                model_name="answerdotai/ModernBERT-base",
                transformers_version="5.23.0",
                attn_implementation="sdpa",
                torch_compile=False,
                input_setting=None,
                max_length_cap=None,
                truncation_rule=truncation_rule_name(None),
                include_special_tokens_in_pool=None,
                dtype="float32",
            )

        plan: dict[str, tuple[list[str], list[int], list[str | None] | None]] = {}
        if doc_records is not None:
            for name in SYNTHETIC_SPLITS:
                rows = [r for r in doc_records if r.split == name]
                genres = [r.genre_split for r in rows]
                plan[name] = (
                    [r.doc_id for r in rows],
                    [r.label for r in rows],
                    genres if any(g is not None for g in genres) else None,
                )
        else:
            for name, size in zip(SYNTHETIC_SPLITS, (train_size, eval_size), strict=True):
                labels = [i % num_classes for i in range(size)]
                rng.shuffle(labels)
                genres: list[str | None] | None = None
                if name == "eval" and num_classes == 3:
                    half = size // 2
                    genres = ["matched"] * half + ["mismatched"] * (size - half)
                plan[name] = ([f"{name}-{i:05d}" for i in range(size)], labels, genres)

        splits: dict[str, LayerVectorBatch] = {}
        for name, (doc_ids, labels, genres) in plan.items():
            label_arr = np.asarray(labels, dtype=np.int64)
            noise = rng.normal(0.0, 1.0, size=(len(doc_ids), NUM_LAYER_VECTORS, HIDDEN_SIZE))
            vectors = (noise + 0.5 * class_means[label_arr]).astype(np.float32)
            splits[name] = LayerVectorBatch(
                doc_ids=doc_ids,
                labels=label_arr,
                vectors=vectors,
                metadata=metadata,
                genre_splits=genres,
            )
        return cls(splits, metadata)
