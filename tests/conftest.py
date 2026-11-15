"""Shared pytest fixtures for CPU seam tests."""

from __future__ import annotations

import sys
from pathlib import Path

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
