"""Cache build helpers behind scripts/extract_cache.py (T-04b PR 2). CPU only, no download."""

from dataclasses import replace
from pathlib import Path

import pytest

from layer_fusion.cache_build import (
    ExtractionEnv,
    build_cache,
    check_input_setting,
    load_env_config,
    resolve_cache_dtype,
    sample_records,
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


def test_dtype_override_wins_over_the_config() -> None:
    assert resolve_cache_dtype(_env(cache_dtype="float16"), override="float32") == "float32"
    assert resolve_cache_dtype(_env(cache_dtype="float16"), override=None) == "float16"


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
