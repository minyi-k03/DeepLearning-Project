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

# PR #13 review: leaked memmap handles break rename/rmtree on Windows; emulate that here.
pytestmark = pytest.mark.usefixtures("windows_file_locks")


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
    with pytest.raises(ValueError, match="view_ratios has 1 entries"):
        save_layer_cache(
            tmp_path, "train", _vectors(2), ["a", "b"], [0, 1], _meta(), view_ratios=[1.0]
        )
    assert not any(tmp_path.iterdir())  # length errors are caught before anything is written


def test_a_split_with_mismatched_index_files_is_rejected_on_load(tmp_path) -> None:
    """Writer and reader share one length rule (check_split_lengths)."""
    split_dir = save_layer_cache(
        tmp_path, "train", _vectors(2), ["a", "b"], [0, 1], _meta(), view_ratios=[1.0, 0.5]
    )
    np.save(split_dir / "view_ratios.npy", np.asarray([1.0]))
    with pytest.raises(ValueError, match="view_ratios has 1 entries, expected 2"):
        CacheProvider(str(tmp_path)).get_split("train")


def test_unknown_split_and_missing_cache_raise(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        CacheProvider(str(tmp_path))
    save_layer_cache(tmp_path, "train", _vectors(1), ["a"], [0], _meta())
    with pytest.raises(KeyError):
        CacheProvider(str(tmp_path)).get_split("eval")


def test_float16_range_check_uses_the_float16_maximum() -> None:
    """No extra safety margin beyond the float16 range (precision is reported separately)."""
    check_float16_range(60_000.0)
    with pytest.raises(ValueError, match="float16"):
        check_float16_range(70_000.0)


def test_any_existing_split_directory_blocks_saving(tmp_path) -> None:
    """A split left half-written (index files but no vectors) is not silently overwritten."""
    (tmp_path / "train").mkdir()
    (tmp_path / "train" / "doc_ids.json").write_text("[]", encoding="utf-8")
    with pytest.raises(FileExistsError):
        save_layer_cache(tmp_path, "train", _vectors(1), ["a"], [0], _meta())


def test_failed_save_leaves_no_split_behind(tmp_path, monkeypatch) -> None:
    """Writes go to a temporary folder that only becomes <split> when every file is written."""
    import layer_fusion.extraction as extraction

    real_save = np.save
    calls = {"n": 0}

    def flaky_save(path, array):  # vectors.npy (memmap) is already written, labels.npy fails
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("disk full")
        real_save(path, array)

    monkeypatch.setattr(extraction.np, "save", flaky_save)
    with pytest.raises(OSError):
        save_layer_cache(tmp_path, "train", _vectors(2), ["a", "b"], [0, 1], _meta())
    assert not (tmp_path / "train").exists()
    assert not any(p.name.startswith("train") for p in tmp_path.iterdir())


def test_make_synthetic_rejects_unknown_splits() -> None:
    """Tuning records must not be silently dropped (PR #11 review)."""
    records = [
        DocumentRecord(doc_id="t0", text_a="x", label=0, split="train"),
        DocumentRecord(doc_id="u0", text_a="y", label=1, split="tuning"),
    ]
    with pytest.raises(ValueError, match="tuning"):
        FakeLayerVectorProvider.make_synthetic(doc_records=records)


def test_view_ratios_and_report_round_trip_through_the_writer(tmp_path) -> None:
    """US 18: per-document view ratios are stored next to the vectors and loaded back."""
    import json

    from layer_fusion.extraction import (
        ExtractionReport,
        ExtractionResult,
        begin_split,
        finish_split,
    )
    from layer_fusion.inputs import TokenizedInputs

    out = begin_split(tmp_path, "eval", 2, _meta())
    out[:] = _vectors(2)
    tokenized = TokenizedInputs(
        doc_ids=["a", "b"], input_ids=[[1], [2]], token_lengths=[100, 900],
        input_lengths=[100, 512], length_buckets=None, truncated=[False, True],
        truncated_doc_ratio=0.5, view_ratios=[1.0, 0.25],
    )
    report = ExtractionReport(
        num_docs=2, max_abs_value=3.0, max_abs_slot=21, max_abs_dim=251,
        float16_round_trip_max_error=0.001, min_input_length=100, max_input_length=512,
        truncated_doc_ratio=0.5, elapsed_seconds=1.2, device="cpu",
    )
    result = ExtractionResult(out, ["a", "b"], [0, 1], None, _meta(), tokenized, report)
    split_dir = finish_split(tmp_path, "eval", result)

    batch = CacheProvider(str(tmp_path)).get_split("eval")
    assert batch.view_ratios == [1.0, 0.25] and batch.token_lengths == [100, 900]
    assert np.array_equal(batch.vectors, _vectors(2))
    assert json.loads((split_dir / "report.json").read_text("utf-8"))["max_abs_value"] == 3.0


def test_aborted_split_leaves_nothing_and_can_be_retried(tmp_path) -> None:
    from layer_fusion.extraction import abort_split, begin_split

    begin_split(tmp_path, "train", 2, _meta())
    abort_split(tmp_path, "train")
    assert not any(p.name.startswith("train") for p in tmp_path.iterdir())
    begin_split(tmp_path, "train", 2, _meta())  # a clean retry is allowed


def test_stale_partial_split_blocks_a_new_run(tmp_path) -> None:
    """A .partial left by a crashed run is not reused or overwritten silently (AGENTS.md §7)."""
    from layer_fusion.extraction import begin_split

    begin_split(tmp_path, "train", 2, _meta())
    with pytest.raises(FileExistsError, match="partial"):
        begin_split(tmp_path, "train", 2, _meta())


def test_windows_lock_emulation_catches_an_open_memmap(tmp_path) -> None:
    """Guard for the fixture itself: an open memmap must block rename, or the tests above
    would pass on POSIX without proving anything about Windows."""
    folder = tmp_path / "locked"
    folder.mkdir()
    held = np.lib.format.open_memmap(folder / "v.npy", mode="w+", dtype="float16", shape=(1,))
    with pytest.raises(PermissionError, match="open memmaps"):
        folder.rename(tmp_path / "moved")
    from layer_fusion.extraction import close_memmap

    close_memmap(held)
    folder.rename(tmp_path / "moved")  # released handle: allowed again


def test_abort_split_closes_the_memmap_and_warns_if_the_folder_survives(tmp_path) -> None:
    from layer_fusion.extraction import abort_split, begin_split

    out = begin_split(tmp_path, "train", 2, _meta())
    abort_split(tmp_path, "train", out)  # passing the memmap lets it be removed
    assert not (tmp_path / "train.partial").exists()

    out = begin_split(tmp_path, "train", 2, _meta())
    with pytest.warns(RuntimeWarning, match="could not remove"):
        abort_split(tmp_path, "train")  # memmap still open elsewhere: folder stays, loudly
    assert (tmp_path / "train.partial").exists()
    abort_split(tmp_path, "train", out)
    assert not (tmp_path / "train.partial").exists()
