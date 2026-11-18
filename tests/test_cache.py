"""Layer-vector cache files and CacheProvider (T-04b). CPU only: no model, fake vectors."""

from dataclasses import replace

import numpy as np
import pytest

from layer_fusion.contracts import CacheMetadata, DocumentRecord, InputSetting
from layer_fusion.extraction import check_float16_range, save_layer_cache
from layer_fusion.providers import (
    CacheConfigMismatchError,
    CacheProvider,
    FakeLayerVectorProvider,
)


def _meta(**overrides: object) -> CacheMetadata:
    base = CacheMetadata(
        model_name="answerdotai/ModernBERT-base",
        transformers_version="5.23.0",
        attn_implementation="sdpa",
        torch_compile=False,
        input_setting=InputSetting.L512,
        max_length_cap=512,
        truncation_rule="head_128_tail_382",
        include_special_tokens_in_pool=True,
        dtype="float16",
    )
    return replace(base, **overrides)


def _vectors(n: int, dtype: str = "float16") -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.normal(size=(n, 23, 768)).astype(dtype)


def test_round_trip_keeps_order_dtype_and_metadata(tmp_path) -> None:
    vectors = _vectors(3)
    save_layer_cache(
        tmp_path,
        "eval",
        vectors,
        ["b", "a", "c"],
        [1, 0, 2],
        _meta(),
        token_lengths=[10, 700, 2000],
        genre_splits=["matched", "mismatched", "matched"],
    )
    provider = CacheProvider(str(tmp_path))
    batch = provider.get_split("eval")
    assert provider.metadata == _meta()
    assert batch.doc_ids == ["b", "a", "c"] and batch.labels.tolist() == [1, 0, 2]
    assert batch.vectors.dtype == np.float16 and np.array_equal(batch.vectors, vectors)
    assert batch.token_lengths == [10, 700, 2000]
    assert batch.genre_splits == ["matched", "mismatched", "matched"]


def test_vectors_are_memory_mapped_not_loaded(tmp_path) -> None:
    """MNLI train is ~14 GB in float16; the container RAM cap is 24 GB, so read lazily."""
    save_layer_cache(tmp_path, "train", _vectors(4), list("abcd"), [0, 1, 0, 1], _meta())
    vectors = CacheProvider(str(tmp_path)).get_split("train").vectors
    assert isinstance(vectors, np.memmap)


def test_more_splits_with_same_metadata_are_allowed(tmp_path) -> None:
    save_layer_cache(tmp_path, "train", _vectors(2), ["a", "b"], [0, 1], _meta())
    save_layer_cache(tmp_path, "eval", _vectors(1), ["c"], [1], _meta())
    assert set(CacheProvider(str(tmp_path)).splits) == {"train", "eval"}


def test_existing_cache_is_never_overwritten(tmp_path) -> None:
    """AGENTS.md §7: do not overwrite experiment caches."""
    save_layer_cache(tmp_path, "train", _vectors(2), ["a", "b"], [0, 1], _meta())
    with pytest.raises(FileExistsError):
        save_layer_cache(tmp_path, "train", _vectors(2), ["a", "b"], [0, 1], _meta())


def test_other_metadata_in_same_directory_is_rejected(tmp_path) -> None:
    save_layer_cache(tmp_path, "train", _vectors(2), ["a", "b"], [0, 1], _meta())
    with pytest.raises(CacheConfigMismatchError):
        save_layer_cache(
            tmp_path, "eval", _vectors(1), ["c"], [1], _meta(input_setting=InputSetting.LFULL)
        )


def test_shape_and_length_mismatches_are_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="23, 768"):
        bad = np.zeros((2, 22, 768), "float16")
        save_layer_cache(tmp_path, "train", bad, ["a", "b"], [0, 1], _meta())
    with pytest.raises(ValueError, match="doc_ids"):
        save_layer_cache(tmp_path, "train", _vectors(2), ["a"], [0, 1], _meta())
    with pytest.raises(ValueError, match="dtype"):
        save_layer_cache(tmp_path, "train", _vectors(2, "float32"), ["a", "b"], [0, 1], _meta())


def test_unknown_split_and_missing_cache_raise(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        CacheProvider(str(tmp_path))
    save_layer_cache(tmp_path, "train", _vectors(1), ["a"], [0], _meta())
    with pytest.raises(KeyError):
        CacheProvider(str(tmp_path)).get_split("eval")


def test_float16_range_check() -> None:
    check_float16_range(3_791.3)
    with pytest.raises(ValueError, match="float16"):
        check_float16_range(70_000.0)


def test_make_synthetic_rejects_unknown_splits() -> None:
    """Tuning records must not be silently dropped (PR #11 review)."""
    records = [
        DocumentRecord(doc_id="t0", text_a="x", label=0, split="train"),
        DocumentRecord(doc_id="u0", text_a="y", label=1, split="tuning"),
    ]
    with pytest.raises(ValueError, match="tuning"):
        FakeLayerVectorProvider.make_synthetic(doc_records=records)
