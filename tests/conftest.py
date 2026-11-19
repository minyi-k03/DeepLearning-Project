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

from layer_fusion.contracts import CacheMetadata, InputSetting  # noqa: E402


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
