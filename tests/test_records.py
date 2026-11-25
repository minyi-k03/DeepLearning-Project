"""Result record JSON storage (T-06b): round trip, overwrite refusal, strict loading."""

from __future__ import annotations

import json
import math
import os
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("torch")

from layer_fusion import judgment  # noqa: E402
from layer_fusion.contracts import (  # noqa: E402
    FusionMethod,
    InputSetting,
    LengthBucket,
    PredictionRecord,
    RunKind,
)
from layer_fusion.records import (  # noqa: E402
    RECORD_FORMAT,
    find_lr_selection_record,
    find_selection_record_for,
    load_result_record,
    load_result_records,
    result_record_to_dict,
    save_result_record,
)
from layer_fusion.runner import run_experiment  # noqa: E402


@pytest.fixture
def diverged_record(noisy_provider, make_run_config):  # type: ignore[no-untyped-def]
    """A real F2 MNLI selection-run record whose first LR candidate diverged (score inf)."""
    provider = noisy_provider()
    config = make_run_config(provider, lr_candidates=(1e10, 1e-3, 1e-1), epochs=2)
    return run_experiment(config, provider, device="cpu")


def test_round_trip_of_a_real_record_is_exact(diverged_record, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    record = diverged_record
    assert record.lr_tuning_scores[1e10] == math.inf  # the case JSON cannot hold natively
    assert record.layer_weights is not None and record.eval_error_rate_matched is not None

    path = save_result_record(record, tmp_path)
    assert path == tmp_path / f"{record.run_id}.json"
    loaded = load_result_record(path)

    assert loaded == record
    assert list(loaded.lr_tuning_scores) == list(record.lr_tuning_scores)  # order kept
    assert loaded.config.fusion_method is FusionMethod.F2  # enums, not strings
    assert loaded.config.run_kind is RunKind.MAIN
    assert isinstance(loaded.config.lr_candidates, tuple)
    assert loaded.config.expected_cache_meta == record.config.expected_cache_meta


def test_saved_file_is_standard_json(diverged_record, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    path = save_result_record(diverged_record, tmp_path)
    text = path.read_text(encoding="utf-8")
    assert "Infinity" not in text and "NaN" not in text
    data = json.loads(text)
    assert data["record_format"] == RECORD_FORMAT
    assert [1e10, "inf"] in data["lr_tuning_scores"]


def test_length_buckets_and_input_setting_round_trip(diverged_record, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    pred = PredictionRecord(
        doc_id="imdb-0", label=1, prediction=0, token_length=700,
        length_bucket=LengthBucket.BETWEEN_512_1024, layer_weights=[0.5, 0.5],
    )
    record = replace(
        diverged_record,
        run_id="imdb-like",
        config=replace(diverged_record.config, input_setting=InputSetting.L512),
        predictions=[pred],
    )
    loaded = load_result_record(save_result_record(record, tmp_path))
    assert loaded == record
    assert loaded.predictions[0].length_bucket is LengthBucket.BETWEEN_512_1024


def test_reloaded_debug_records_are_kept_out_of_judgment(diverged_record, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    debug = replace(
        diverged_record,
        run_id="debug-run",
        config=replace(diverged_record.config, run_kind=RunKind.DEBUG),
    )
    save_result_record(diverged_record, tmp_path)
    save_result_record(debug, tmp_path)
    loaded = load_result_records(tmp_path)
    assert {r.run_id for r in loaded} == {diverged_record.run_id, "debug-run"}
    reloaded_debug = next(r for r in loaded if r.run_id == "debug-run")
    assert reloaded_debug.config.run_kind is RunKind.DEBUG  # judgment filters with `is`
    assert [r.run_id for r in judgment.select_main_records(loaded)] == [diverged_record.run_id]


def test_existing_record_is_never_overwritten(diverged_record, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    path = save_result_record(diverged_record, tmp_path)
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        save_result_record(diverged_record, tmp_path)
    with pytest.raises(ValueError, match="debug"):
        save_result_record(diverged_record, tmp_path, overwrite=True)
    assert path.read_bytes() == before


def test_debug_record_may_be_overwritten(diverged_record, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    debug = replace(diverged_record, config=replace(diverged_record.config, run_kind=RunKind.DEBUG))
    save_result_record(debug, tmp_path)
    changed = replace(debug, eval_error_rate=0.5)
    path = save_result_record(changed, tmp_path, overwrite=True)
    assert load_result_record(path).eval_error_rate == 0.5


@pytest.mark.skipif(os.name != "posix", reason="POSIX file modes")
def test_saved_record_follows_the_umask(diverged_record, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    # Records written as root inside the container must stay readable from the host account.
    old = os.umask(0o022)
    try:
        path = save_result_record(diverged_record, tmp_path)
    finally:
        os.umask(old)
    assert path.stat().st_mode & 0o777 == 0o644


def test_non_finite_value_fails_the_save(diverged_record, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    broken = replace(diverged_record, eval_error_rate=math.nan)
    with pytest.raises(ValueError):
        save_result_record(broken, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_run_id_must_be_a_plain_file_name(diverged_record, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError, match="file name"):
        save_result_record(replace(diverged_record, run_id="../escape"), tmp_path)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda d: d.update(record_format=99), "record_format"),
        (lambda d: d.update(record_format=True), "record_format"),
        (lambda d: d.update(config=None), "RunConfig"),
        (lambda d: d.update(eval_error_rate=math.nan), "non-standard JSON constant NaN"),
        (lambda d: d.update(eval_error_rate=math.inf), "non-standard JSON constant Infinity"),
        (lambda d: d["config"].update(run_kind=None), "RunKind"),
        (lambda d: d["config"].update(dataset=None), "DatasetName"),
        (lambda d: d.pop("record_format"), "record_format"),
        (lambda d: d.update(surprise=1), "unknown keys"),
        (lambda d: d.pop("eval_error_rate"), "missing keys"),
        (lambda d: d["config"].update(fusion_method="F9"), "F9"),
        (lambda d: d["predictions"][0].update(extra=1), "unknown keys"),
    ],
)
def test_malformed_files_are_refused(diverged_record, tmp_path: Path, mutate, message) -> None:  # type: ignore[no-untyped-def]
    data = result_record_to_dict(diverged_record)
    mutate(data)
    path = tmp_path / f"{diverged_record.run_id}.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_result_record(path)


def test_corrupt_json_and_renamed_files_are_refused(diverged_record, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_result_record(bad)
    path = save_result_record(diverged_record, tmp_path / "ok")
    renamed = path.rename(path.with_name("other-name.json"))
    with pytest.raises(ValueError, match="does not match run_id"):
        load_result_record(renamed)


def _reuse_of(selection, seed: int, **changes):  # type: ignore[no-untyped-def]
    config = replace(
        selection.config, seed=seed, fixed_lr=selection.selected_lr,
        lr_selection_seed=selection.config.seed, **changes,
    )
    return replace(selection, run_id=f"reuse-s{seed}", config=config)


def test_find_lr_selection_record(diverged_record) -> None:  # type: ignore[no-untyped-def]
    selection = diverged_record
    reuse = _reuse_of(selection, seed=1)
    other_method = replace(
        selection, run_id="other",
        config=replace(selection.config, fusion_method=FusionMethod.F1),
    )
    records = [other_method, selection, reuse]
    assert find_lr_selection_record(reuse, records) is selection
    assert find_lr_selection_record(selection, records) is selection  # its own selection


def test_lr_selection_record_must_exist_and_match(diverged_record) -> None:  # type: ignore[no-untyped-def]
    selection = diverged_record
    reuse = _reuse_of(selection, seed=1)
    other_setting = replace(
        selection, config=replace(selection.config, fusion_method=FusionMethod.F1)
    )
    with pytest.raises(LookupError):
        find_lr_selection_record(reuse, [other_setting, reuse])
    with pytest.raises(ValueError, match="2 selection-run records"):
        find_lr_selection_record(reuse, [selection, selection, reuse])
    wrong_lr = _reuse_of(selection, seed=1)
    wrong_lr = replace(wrong_lr, config=replace(wrong_lr.config, fixed_lr=1e-1 + 1e-9))
    with pytest.raises(ValueError, match="selected_lr"):
        find_lr_selection_record(wrong_lr, [selection])

def test_reuse_cannot_reference_its_own_seed(diverged_record) -> None:  # type: ignore[no-untyped-def]
    config = replace(
        diverged_record.config,
        fixed_lr=diverged_record.selected_lr,
        lr_selection_seed=diverged_record.config.seed,
    )
    with pytest.raises(ValueError, match="own seed"):
        find_selection_record_for(config, [diverged_record])
