"""Shared pytest fixtures for CPU seam tests."""

from __future__ import annotations

import shutil
import sys
import weakref
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from layer_fusion.contracts import (  # noqa: E402
    BackboneCondition,
    CacheMetadata,
    DataAmount,
    DatasetName,
    FusionMethod,
    InputSetting,
    RunConfig,
)


@pytest.fixture
def noisy_provider():  # type: ignore[no-untyped-def]
    """Factory for a fake provider whose classes overlap, so tuning errors differ across LRs.

    ``FakeLayerVectorProvider.make_synthetic`` is separable at every LR (all tuning errors 0),
    which would make LR-selection tests vacuous. Doc ids, labels and genres are kept from it.
    """
    from dataclasses import replace

    from layer_fusion.providers import FakeLayerVectorProvider

    def make(
        num_classes: int = 3,
        train_size: int = 300,
        eval_size: int = 90,
        signal: float = 0.06,
        seed: int = 0,
        metadata: CacheMetadata | None = None,
        dtype: type = np.float32,
    ) -> FakeLayerVectorProvider:
        base = FakeLayerVectorProvider.make_synthetic(
            num_classes=num_classes,
            train_size=train_size,
            eval_size=eval_size,
            seed=seed,
            metadata=metadata,
        )
        rng = np.random.default_rng(seed + 1)
        class_direction = rng.normal(size=(num_classes, 23, 768))
        splits = {}
        for name in ("train", "eval"):
            batch = base.get_split(name)
            noise = rng.normal(size=batch.vectors.shape)
            vectors = (noise + signal * class_direction[batch.labels]).astype(dtype)
            splits[name] = replace(batch, vectors=vectors)
        return FakeLayerVectorProvider(splits, base.metadata)

    return make


@pytest.fixture
def make_run_config():  # type: ignore[no-untyped-def]
    """Factory for a main MNLI frozen RunConfig with a small, complete tuning budget."""

    def make(provider, **overrides) -> RunConfig:  # type: ignore[no-untyped-def]
        values = dict(
            dataset=DatasetName.MNLI,
            data_amount=DataAmount.FULL,
            fusion_method=FusionMethod.F2,
            backbone_condition=BackboneCondition.FROZEN,
            seed=0,
            lr_candidates=(1e-4, 1e-3, 1e-1),
            epochs=5,
            early_stopping_patience=2,
            batch_size=32,
            expected_cache_meta=provider.metadata,
        )
        values.update(overrides)
        return RunConfig(**values)

    return make


@pytest.fixture
def sample_cache_metadata() -> CacheMetadata:
    return CacheMetadata(
        model_name="answerdotai/ModernBERT-base",
        transformers_version="5.0.0-tbd",
        attn_implementation="sdpa",
        torch_compile=False,
        input_setting=InputSetting.L512,
        max_length_cap=512,
        truncation_rule="head_128_tail_382",
        include_special_tokens_in_pool=True,
        dtype="float32",
    )


@pytest.fixture
def windows_file_locks(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Emulate Windows on any OS: a folder holding an open np.memmap cannot be renamed or removed.

    POSIX lets a directory with open mapped files be renamed or deleted, so a leaked memmap
    handle passes on macOS/Linux CI but fails on Windows (PR #13 review: WinError 5 on rename,
    a silently surviving ``.partial`` after ``rmtree(ignore_errors=True)``). Every memmap is
    tracked by a weak reference to its mmap; one still alive and not closed counts as a lock.
    """
    opened: list[tuple[weakref.ref, Path]] = []
    original_new = np.memmap.__new__

    def tracking_new(subtype, filename, *args, **kwargs):  # type: ignore[no-untyped-def]
        array = original_new(subtype, filename, *args, **kwargs)
        mm = getattr(array, "_mmap", None)
        if mm is not None and array.filename is not None:
            opened.append((weakref.ref(mm), Path(array.filename).resolve()))
        return array

    def locked_files(folder: Path) -> list[Path]:
        root = Path(folder).resolve()
        return [
            path
            for ref, path in opened
            if (mm := ref()) is not None and not mm.closed and root in path.parents
        ]

    original_rename = Path.rename
    original_rmtree = shutil.rmtree

    def rename(self: Path, target):  # type: ignore[no-untyped-def]
        if locked := locked_files(self):
            raise PermissionError(f"[emulated WinError 5] {self} holds open memmaps: {locked}")
        return original_rename(self, target)

    def rmtree(path, ignore_errors=False, *args, **kwargs):  # type: ignore[no-untyped-def]
        if locked := locked_files(Path(path)):
            if ignore_errors:
                return None  # Windows: deletion silently fails, the folder stays
            raise PermissionError(f"[emulated WinError 32] {path} holds open memmaps: {locked}")
        return original_rmtree(path, ignore_errors, *args, **kwargs)

    monkeypatch.setattr(np.memmap, "__new__", staticmethod(tracking_new))
    monkeypatch.setattr(Path, "rename", rename)
    monkeypatch.setattr(shutil, "rmtree", rmtree)
    yield
