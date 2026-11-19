"""Cache build helpers behind scripts/extract_cache.py (T-04b PR 2). CPU only, no download."""

from dataclasses import replace
from pathlib import Path

import pytest

from layer_fusion.cache_build import (
    ExtractionEnv,
    build_cache,
    cache_name,
    check_input_setting,
    check_output_dir,
    default_output_dir,
    load_env_config,
    resolve_cache_dtype,
    sample_records,
    select_shortest,
)
from layer_fusion.contracts import DatasetName, DocumentRecord, InputSetting

REPO_ROOT = Path(__file__).resolve().parents[1]


def _env(**overrides: object) -> ExtractionEnv:
    base = ExtractionEnv(
        model_name="answerdotai/ModernBERT-base",
        transformers_version="5.23.0",
        attn_implementation="sdpa",
        torch_compile=False,
        lfull_max_length_cap=4096,
        include_special_tokens_in_pool=True,
        cache_dtype=None,
    )
    return replace(base, **overrides)


def _records(n: int) -> list[DocumentRecord]:
    return [DocumentRecord(doc_id=f"d{i:03d}", text_a=f"text {i}", label=i % 2) for i in range(n)]


def test_env_config_reads_the_locked_runtime_values() -> None:
    env = load_env_config(REPO_ROOT / "configs" / "week1_env.yaml")
    assert env.model_name == "answerdotai/ModernBERT-base"
    assert env.transformers_version == "5.23.0"
    assert env.attn_implementation == "sdpa" and env.torch_compile is False
    assert env.lfull_max_length_cap == 4096


def test_undecided_cache_dtype_is_never_guessed() -> None:
    with pytest.raises(ValueError, match="cache_dtype is null"):
        resolve_cache_dtype(_env(cache_dtype=None), override=None)


def test_main_cache_uses_the_locked_dtype_only() -> None:
    """A main cache cannot drift from week1_env.yaml through --dtype (Spec review, PR #13)."""
    locked = _env(cache_dtype="float16")
    assert resolve_cache_dtype(locked, override=None) == "float16"
    assert resolve_cache_dtype(locked, override="float16") == "float16"
    with pytest.raises(ValueError, match="differs from the locked cache_dtype"):
        resolve_cache_dtype(locked, override="float32")
    with pytest.raises(ValueError, match="differs from the locked cache_dtype"):
        resolve_cache_dtype(_env(cache_dtype=None), override="float16")


def test_debug_sample_may_override_the_dtype() -> None:
    assert resolve_cache_dtype(_env(cache_dtype="float16"), "float32", debug=True) == "float32"
    assert resolve_cache_dtype(_env(cache_dtype=None), "float16", debug=True) == "float16"


def test_cache_folder_names() -> None:
    assert cache_name(DatasetName.MNLI, None) == "mnli"
    assert cache_name(DatasetName.IMDB, InputSetting.L512, sep="/") == "imdb/L512"
    assert default_output_dir(DatasetName.IMDB, InputSetting.L512, None) == Path(
        "data/cache/imdb_L512"
    )
    assert default_output_dir(DatasetName.SST2, None, "limit50") == Path(
        "data/cache_debug/sst2_limit50"
    )


def test_sampled_cache_cannot_be_written_into_the_main_cache_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AGENTS.md §4: a --limit/--shortest subset must never look like a main cache."""
    monkeypatch.chdir(tmp_path)
    check_output_dir(Path("data/cache_debug/sst2_limit50"), sampled=True)
    check_output_dir(Path("data/cache/sst2"), sampled=False)
    for bad in ("data/cache/sst2", "elsewhere/sst2", "data/cache_debug_x/sst2"):
        with pytest.raises(ValueError, match="cache_debug"):
            check_output_dir(Path(bad), sampled=True)


def test_shortest_selection_keeps_original_order_and_breaks_ties_by_position() -> None:
    records = _records(6)
    lengths = [9, 3, 7, 3, 1, 8]
    picked = select_shortest(records, lengths, 3)
    assert [r.doc_id for r in picked] == ["d001", "d003", "d004"]
    with pytest.raises(ValueError, match="lengths"):
        select_shortest(records, lengths[:-1], 3)


def test_unknown_cache_dtype_is_rejected() -> None:
    with pytest.raises(ValueError, match="cache dtype"):
        resolve_cache_dtype(_env(cache_dtype="bfloat16"), override=None)


@pytest.mark.parametrize(
    ("dataset", "setting"),
    [(DatasetName.IMDB, None), (DatasetName.MNLI, InputSetting.L512)],
    ids=["imdb-without-setting", "mnli-with-setting"],
)
def test_input_setting_is_required_for_imdb_only(
    dataset: DatasetName, setting: InputSetting | None
) -> None:
    with pytest.raises(ValueError, match="input-setting"):
        check_input_setting(dataset, setting)


def test_limited_sample_is_deterministic_and_keeps_order() -> None:
    records = _records(100)
    first = sample_records(records, 10, seed=0)
    assert [r.doc_id for r in first] == [r.doc_id for r in sample_records(records, 10, seed=0)]
    assert [r.doc_id for r in first] == sorted(r.doc_id for r in first)
    assert len(first) == 10


def test_no_limit_keeps_every_record() -> None:
    records = _records(5)
    assert sample_records(records, None) == records
    assert sample_records(records, 50) == records


def test_failed_extraction_leaves_no_split_behind(tmp_path: Path) -> None:
    """An error mid-build removes <split>.partial, so no half-written cache remains."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    env = _env(transformers_version="0.0.0-not-installed")  # extractor refuses this version
    with pytest.raises(ValueError, match="locked"):
        build_cache(
            {"train": _records(3)}, env, tmp_path, dtype="float16",
            model=object(), tokenizer=object(),
        )
    assert not any(p.name.startswith("train") for p in tmp_path.iterdir())


def test_saving_requires_the_pooling_decision(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="include_special_tokens_in_pool"):
        build_cache(
            {"train": _records(2)}, _env(include_special_tokens_in_pool=None), tmp_path,
            model=object(), tokenizer=object(),
        )
    assert not any(tmp_path.iterdir())
