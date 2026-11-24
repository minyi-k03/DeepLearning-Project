"""Sequential queue runner (T-06c, ADR 0009): lock, order, LR reuse, resume, fail-fast, CLI."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from layer_fusion.contracts import (
    DatasetName,
    FusionMethod,
    InputSetting,
    RunConfig,
    RunKind,
    TrackKind,
)
from layer_fusion.records import load_result_record, load_result_records, record_path
from layer_fusion.run_queue import (
    MainQueueLockedError,
    QueueRunError,
    main_queue_lock,
    order_by_priority,
    run_queue,
)

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------- helpers


def _run_id(config: RunConfig) -> str:
    """Same shape as runner.make_run_id, without importing torch."""
    setting = config.input_setting.value if config.input_setting else "none"
    return (
        f"{config.dataset.value}-{config.fusion_method.value}-{setting}-s{config.seed}"
        f"-{config.run_kind.value}"
    )


class FakeRun:
    """Stand-in execution seam: records the configs it saw, LR = first candidate (or fixed)."""

    def __init__(self, fail_on: set[int] | None = None) -> None:
        self.seen: list[RunConfig] = []
        self.fail_on = fail_on or set()

    def __call__(self, config: RunConfig, provider: object):  # type: ignore[no-untyped-def]
        from layer_fusion.contracts import PredictionRecord, ResultRecord

        self.seen.append(config)
        if len(self.seen) in self.fail_on:
            raise RuntimeError("all LR candidates diverged")
        lr = config.fixed_lr if config.fixed_lr is not None else config.lr_candidates[1]
        return ResultRecord(
            run_id=_run_id(config),
            config=config,
            selected_lr=lr,
            lr_tuning_scores={lr: 0.3},
            eval_error_rate=0.25,
            predictions=[PredictionRecord(doc_id="d0", label=0, prediction=0)],
            train_seconds=1.0,
            trainable_params=10,
        )


def _config(dataset: DatasetName = DatasetName.MNLI, seed: int = 0, **kw) -> RunConfig:  # type: ignore[no-untyped-def]
    values = dict(
        dataset=dataset,
        data_amount="full",
        fusion_method=FusionMethod.F2,
        backbone_condition="frozen",
        seed=seed,
        lr_candidates=(1e-4, 1e-3, 1e-2),
        epochs=2,
        early_stopping_patience=1,
        batch_size=8,
        run_kind=RunKind.MAIN,
    )
    values.update(kw)
    from layer_fusion.contracts import BackboneCondition, DataAmount

    values["data_amount"] = DataAmount(values["data_amount"])
    values["backbone_condition"] = BackboneCondition(values["backbone_condition"])
    return RunConfig(**values)


def _queue(configs, out: Path, lock: Path, run: FakeRun | None = None):  # type: ignore[no-untyped-def]
    return run_queue(
        configs,
        lambda config: None,
        out,
        run=run or FakeRun(),
        run_id_of=_run_id,
        lock_path=lock,
        log=lambda _msg: None,
    )


# ---------------------------------------------------------------- lock (US 47, ADR 0009)


def test_second_main_queue_is_refused_while_the_lock_is_held(tmp_path: Path) -> None:
    lock = tmp_path / "main_queue.lock"
    run = FakeRun()
    with main_queue_lock(lock):
        with pytest.raises(MainQueueLockedError, match="ADR 0009"):
            _queue([_config()], tmp_path / "runs", lock, run)
    assert run.seen == []  # refused before any run started
    assert not (tmp_path / "runs").exists()


def test_main_queue_started_in_another_process_is_refused(tmp_path: Path) -> None:
    """Completion criterion of T-06c: two main queues at once -> one is refused."""
    lock = tmp_path / "main_queue.lock"
    holder = textwrap.dedent(
        f"""
        import sys, time
        sys.path.insert(0, {str(ROOT / "src")!r})
        from layer_fusion.run_queue import main_queue_lock
        with main_queue_lock({str(lock)!r}):
            print("locked", flush=True)
            sys.stdin.readline()
        """
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", holder],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert proc.stdout is not None and proc.stdout.readline().strip() == "locked"
        with pytest.raises(MainQueueLockedError, match=r"pid=\d+"):
            _queue([_config()], tmp_path / "runs", lock)
    finally:
        proc.communicate("\n", timeout=30)
    # Holder gone -> the OS released the lock; the next main queue runs.
    assert _queue([_config()], tmp_path / "runs", lock).completed


def test_lock_is_released_when_a_run_fails(tmp_path: Path) -> None:
    lock = tmp_path / "main_queue.lock"
    with pytest.raises(QueueRunError):
        _queue([_config()], tmp_path / "runs", lock, FakeRun(fail_on={1}))
    with main_queue_lock(lock):  # would raise if the failed queue still held it
        pass


def test_debug_queue_runs_without_the_lock(tmp_path: Path) -> None:
    lock = tmp_path / "main_queue.lock"
    with main_queue_lock(lock):
        result = _queue([_config(run_kind=RunKind.DEBUG)], tmp_path / "runs", lock)
    assert result.completed == ["mnli-F2-none-s0-debug"]
    record = load_result_record(record_path(tmp_path / "runs", result.completed[0]))
    assert record.config.run_kind is RunKind.DEBUG


# ---------------------------------------------------------------- order (US 51)


def test_order_is_mnli_then_sst2_then_imdb_keeping_matrix_order() -> None:
    imdb = [
        _config(DatasetName.IMDB, input_setting=InputSetting.L128),
        _config(DatasetName.IMDB, input_setting=InputSetting.L512),
    ]
    sst2 = _config(DatasetName.SST2)
    mnli = [_config(fusion_method=FusionMethod.B0), _config(fusion_method=FusionMethod.F1)]
    ordered = order_by_priority([imdb[0], sst2, mnli[0], imdb[1], mnli[1]])
    assert ordered == [mnli[0], mnli[1], sst2, imdb[0], imdb[1]]


def test_queue_runs_in_priority_order_whatever_order_it_is_given(tmp_path: Path) -> None:
    run = FakeRun()
    given = [
        _config(DatasetName.IMDB, input_setting=InputSetting.L512),
        _config(DatasetName.SST2),
        _config(DatasetName.MNLI),
    ]
    _queue(given, tmp_path / "runs", tmp_path / "l", run)
    assert [c.dataset for c in run.seen] == [
        DatasetName.MNLI, DatasetName.SST2, DatasetName.IMDB
    ]


# ---------------------------------------------------------------- LR reuse and resume


def test_other_seeds_reuse_the_lr_of_the_first_seed(tmp_path: Path) -> None:
    run = FakeRun()
    _queue([_config(seed=s) for s in (0, 1, 2)], tmp_path / "runs", tmp_path / "l", run)
    assert [c.seed for c in run.seen] == [0, 1, 2]
    assert run.seen[0].fixed_lr is None
    assert {(c.fixed_lr, c.lr_selection_seed) for c in run.seen[1:]} == {(1e-3, 0)}


def test_real_execution_seam_reuses_the_selected_lr(noisy_provider, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("torch")
    from layer_fusion.runner import make_run_id, run_experiment

    provider = noisy_provider(train_size=120, eval_size=60)
    configs = [
        _config(seed=s, lr_candidates=(1e-3, 1e-2, 1e-1), expected_cache_meta=provider.metadata)
        for s in (0, 1)
    ]
    result = run_queue(
        configs,
        lambda config: provider,
        tmp_path / "runs",
        run=lambda c, p: run_experiment(c, p, device="cpu"),
        run_id_of=make_run_id,
        lock_path=tmp_path / "l",
        log=lambda _msg: None,
    )
    selection, reuse = (
        load_result_record(record_path(tmp_path / "runs", rid)) for rid in result.completed
    )
    assert reuse.config.fixed_lr == selection.selected_lr
    assert reuse.config.lr_selection_seed == 0
    assert reuse.selected_lr == selection.selected_lr


def test_existing_records_are_skipped_and_never_rewritten(tmp_path: Path) -> None:
    out = tmp_path / "runs"
    configs = [_config(seed=s) for s in (0, 1)]
    first = _queue(configs, out, tmp_path / "l")
    before = {p: p.read_bytes() for p in out.glob("*.json")}
    run = FakeRun()
    again = _queue(configs, out, tmp_path / "l", run)
    assert run.seen == []
    assert again.skipped == first.completed and again.completed == []
    assert {p: p.read_bytes() for p in out.glob("*.json")} == before


def test_resume_takes_the_lr_from_the_selection_record_on_disk(tmp_path: Path) -> None:
    out = tmp_path / "runs"
    _queue([_config(seed=0)], out, tmp_path / "l")  # only the selection run finished
    run = FakeRun()
    result = _queue([_config(seed=s) for s in (0, 1)], out, tmp_path / "l", run)
    assert result.skipped == ["mnli-F2-none-s0-main"]
    assert [(c.seed, c.fixed_lr, c.lr_selection_seed) for c in run.seen] == [(1, 1e-3, 0)]


def test_first_failure_stops_the_queue_and_keeps_finished_records(tmp_path: Path) -> None:
    out = tmp_path / "runs"
    configs = [_config(fusion_method=m) for m in (FusionMethod.B0, FusionMethod.F1,
                                                   FusionMethod.F2)]
    run = FakeRun(fail_on={2})
    with pytest.raises(QueueRunError) as info:
        _queue(configs, out, tmp_path / "l", run)
    assert len(run.seen) == 2  # F2 never started
    assert info.value.run_id == "mnli-F1-none-s0-main"
    assert info.value.result.completed == ["mnli-B0-none-s0-main"]
    assert [r.run_id for r in load_result_records(out)] == ["mnli-B0-none-s0-main"]


# ---------------------------------------------------------------- refused queues


@pytest.mark.parametrize(
    ("configs", "message"),
    [
        ([], "empty"),
        ([_config(), _config(DatasetName.SST2, run_kind=RunKind.DEBUG)], "main or all debug"),
        ([_config(track=TrackKind.REPRODUCTION)], "T-06e"),
        ([_config(seed=1, fixed_lr=1e-3, lr_selection_seed=0)], "selection-form"),
        ([_config(seed=1), _config(seed=0)], "must run first"),
        ([_config(seed=0), _config(seed=0)], "twice"),
    ],
)
def test_invalid_queues_are_refused_before_any_run(tmp_path: Path, configs, message) -> None:  # type: ignore[no-untyped-def]
    run = FakeRun()
    with pytest.raises(ValueError, match=message):
        _queue(configs, tmp_path / "runs", tmp_path / "l", run)
    assert run.seen == []


# ---------------------------------------------------------------- CLI


_spec = importlib.util.spec_from_file_location("run_queue_cli", ROOT / "scripts" / "run_queue.py")
assert _spec is not None and _spec.loader is not None
cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cli)


def test_cli_main_queue_with_tbd_matrix_is_refused_before_the_lock(tmp_path: Path) -> None:
    lock = tmp_path / "main_queue.lock"
    with pytest.raises(SystemExit, match="TBD"):
        cli.main(["--experiments", "mnli_frozen_full", "--lock-file", str(lock),
                  "--output-dir", str(tmp_path / "runs")])
    assert not lock.exists()


def test_cli_main_queue_refuses_debug_overrides(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="debug-only"):
        cli.main(["--seeds", "0", "--output-dir", str(tmp_path / "runs")])


@pytest.fixture
def cache_root(noisy_provider, tmp_path: Path) -> Path:  # type: ignore[no-untyped-def]
    """An MNLI cache whose metadata is the locked extraction setting of week1_env.yaml."""
    pytest.importorskip("torch")
    from layer_fusion.cache_build import load_env_config, resolve_cache_dtype
    from layer_fusion.extraction import save_layer_cache

    env = load_env_config(ROOT / "configs" / "week1_env.yaml")
    dtype = resolve_cache_dtype(env, None)
    meta = env.cache_metadata(None, dtype)
    provider = noisy_provider(train_size=120, eval_size=60, metadata=meta,
                              dtype=np.dtype(dtype).type)
    root = tmp_path / "cache"
    for split in ("train", "eval"):
        batch = provider.get_split(split)
        save_layer_cache(
            root / "mnli", split, batch.vectors, list(batch.doc_ids),
            [int(x) for x in batch.labels], meta, genre_splits=batch.genre_splits,
        )
    return root


def _debug_args(cache_root: Path, out: Path, *extra: str) -> list[str]:
    return [
        "--debug", "--experiments", "mnli_frozen_full", "--seeds", "0", "1",
        "--fusion-methods", "B0", "F2", "--lr-candidates", "1e-3", "1e-2", "--epochs", "1",
        "--early-stopping-patience", "1", "--batch-size", "32", "--device", "cpu",
        "--cache-dir", str(cache_root), "--output-dir", str(out),
        "--lock-file", str(out.parent / "main_queue.lock"), *extra,
    ]


def test_cli_dry_run_prints_the_order_and_runs_nothing(  # type: ignore[no-untyped-def]
    cache_root: Path, tmp_path: Path, capsys
) -> None:
    out = tmp_path / "runs"
    assert cli.main(_debug_args(cache_root, out, "--dry-run")) == 0
    printed = capsys.readouterr().out
    assert "4 runs" in printed and "mnli-full-B0-all-frozen-none-s0-debug" in printed
    assert not out.exists()


def test_cli_debug_queue_writes_records_that_load_back(cache_root: Path, tmp_path: Path) -> None:
    out = tmp_path / "runs"
    assert cli.main(_debug_args(cache_root, out)) == 0
    records = {r.run_id: r for r in load_result_records(out)}
    assert sorted(records) == sorted(
        f"mnli-full-{m}-all-frozen-none-s{s}-debug" for m in ("B0", "F2") for s in (0, 1)
    )
    for method in ("B0", "F2"):
        selection = records[f"mnli-full-{method}-all-frozen-none-s0-debug"]
        reuse = records[f"mnli-full-{method}-all-frozen-none-s1-debug"]
        assert reuse.config.fixed_lr == selection.selected_lr
    assert all(r.config.run_kind is RunKind.DEBUG for r in records.values())
    assert not (tmp_path / "main_queue.lock").exists()  # debug never touches the lock
    # Rerunning resumes: everything is skipped, nothing is rewritten.
    assert cli.main(_debug_args(cache_root, out)) == 0
    assert sorted(r.run_id for r in load_result_records(out)) == sorted(records)
