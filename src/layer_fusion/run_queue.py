"""Sequential queue runner for timed experiments (T-06c, SPEC 모듈 9, ADR 0009).

Main runs go one at a time while holding an OS file lock, so a second main queue started
on the same machine (same lock file) is refused instead of sharing the GPU and skewing the
training times the results table compares (US 47). Debug runs use the same ordering and
LR-reuse logic but never take the lock: they may run alongside anything and are recorded
as ``run_kind=debug``, which keeps them out of judgment and reports (US 48, 49).

Order: MNLI > SST-2 > IMDB, matrix order kept inside a dataset (US 51, ADR 0003). The first
seed of a setting selects the LR; the other seeds reuse it (T-06a/T-06b). A record already
on disk is skipped, never overwritten (AGENTS.md §7), so an interrupted queue resumes by
running the same command again. The first failing run stops the queue.

The lock only excludes processes that see the same lock file: every container on the GPU
server must mount the same repository path for it to work.
"""

from __future__ import annotations

import datetime as _dt
import os
import socket
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import IO, Any

from layer_fusion.contracts import DatasetName, ResultRecord, RunConfig, RunKind, TrackKind
from layer_fusion.matrix import with_selected_lr
from layer_fusion.records import (
    load_result_record,
    record_path,
    same_setting,
    save_result_record,
)

DATASET_PRIORITY = (DatasetName.MNLI, DatasetName.SST2, DatasetName.IMDB)
# Anchored at the repository root, not the working directory: two main queues started from
# different folders must still see the same lock file (T-06c completion criterion).
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOCK_PATH = REPO_ROOT / "results" / "main_queue.lock"
# Windows locks a byte range and refuses reads of it; the holder line lives at the start of
# the file, so the locked byte sits far past it and a refused process can still read who
# holds the lock. POSIX flock locks the whole file and ignores this offset.
_WINDOWS_LOCK_OFFSET = 1 << 20

RunFn = Callable[[RunConfig, Any], ResultRecord]
ProviderFactory = Callable[[RunConfig], Any]
RunIdFn = Callable[[RunConfig], str]


class MainQueueLockedError(RuntimeError):
    """Another main queue holds the GPU lock (ADR 0009)."""


class QueueRunError(RuntimeError):
    """A run failed; the queue stopped there. ``result`` lists what finished before it."""

    def __init__(self, run_id: str, result: QueueResult) -> None:
        super().__init__(
            f"queue stopped at {run_id} after {len(result.completed)} completed and "
            f"{len(result.skipped)} skipped runs; fix the cause and rerun the same command "
            "(finished records are skipped)"
        )
        self.run_id = run_id
        self.result = result


@dataclass
class QueueResult:
    completed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    seconds: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------- lock


def _try_lock(handle: IO[str]) -> bool:
    """Non-blocking exclusive lock on an open file; False when another holder has it."""
    if os.name == "nt":
        import msvcrt

        handle.seek(_WINDOWS_LOCK_OFFSET)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True
    import fcntl

    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return False
    return True


def _unlock(handle: IO[str]) -> None:
    if os.name == "nt":
        import msvcrt

        handle.seek(_WINDOWS_LOCK_OFFSET)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def main_queue_lock(path: str | Path = DEFAULT_LOCK_PATH) -> Iterator[None]:
    """Hold the main-queue lock for the ``with`` block, or raise ``MainQueueLockedError``.

    An OS lock (``flock`` / ``msvcrt.locking``) is released by the OS when the process dies,
    so a crashed queue never leaves a stale lock to be deleted by hand. The file keeps the
    holder's pid, host and start time for the error message; it is not deleted on exit.
    A relative ``path`` is resolved against the working directory once, here.
    """
    lock_path = Path(path).resolve()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    # "a+" never truncates, so a refused process cannot wipe the holder's details.
    handle = open(lock_path, "a+", encoding="utf-8")  # noqa: SIM115 - held across the yield
    try:
        if not _try_lock(handle):
            handle.seek(0)
            holder = handle.read().strip() or "unknown holder"
            raise MainQueueLockedError(
                f"another main queue holds {lock_path} ({holder}); main runs go one at a "
                "time on the shared GPU (ADR 0009)"
            )
        try:
            handle.seek(0)
            handle.truncate()
            started = _dt.datetime.now().isoformat(timespec="seconds")
            handle.write(f"pid={os.getpid()} host={socket.gethostname()} started={started}\n")
            handle.flush()
            yield
        finally:
            _unlock(handle)
    finally:
        handle.close()


# ---------------------------------------------------------------- order and checks


def order_by_priority(configs: Sequence[RunConfig]) -> list[RunConfig]:
    """MNLI > SST-2 > IMDB (US 51); a stable sort keeps the matrix order inside a dataset."""
    return sorted(configs, key=lambda c: DATASET_PRIORITY.index(c.dataset))


def check_queue(configs: Sequence[RunConfig]) -> RunKind:
    """Refuse a queue that mixes run kinds, carries reproduction runs (T-06e) or comes with
    LRs already filled in (the queue fills them from the selection runs); returns its kind."""
    if not configs:
        raise ValueError("the queue is empty")
    kinds = {c.run_kind for c in configs}
    if len(kinds) > 1:
        raise ValueError("a queue is either all main or all debug runs, not both")
    if any(c.track is TrackKind.REPRODUCTION for c in configs):
        raise ValueError("reproduction-track runs have their own runner (T-06e)")
    if any(c.fixed_lr is not None or c.lr_selection_seed is not None for c in configs):
        raise ValueError(
            "queue configs must be selection-form (fixed_lr=None); the queue reuses the LR "
            "of each setting's first seed itself"
        )
    return kinds.pop()


def _check_selection_first(ordered: Sequence[RunConfig]) -> None:
    """Each setting's selection run (lowest seed) must precede its reuse seeds, and a setting
    may not list the same seed twice; checked before any run starts."""
    seen: list[RunConfig] = []
    for config in ordered:
        same = [s for s in seen if same_setting(s, config)]
        if any(s.seed == config.seed for s in same):
            raise ValueError(f"seed {config.seed} appears twice for one setting")
        if same and config.seed < min(s.seed for s in same):
            raise ValueError(
                f"seed {config.seed} comes after seed {same[0].seed} of the same setting; "
                "the lowest seed selects the LR and must run first"
            )
        seen.append(config)


def _selection_seeds(configs: Sequence[RunConfig]) -> list[RunConfig]:
    """The selection run of every setting: its first config (the lowest seed, checked by
    :func:`_check_selection_first`)."""
    selections: list[RunConfig] = []
    for config in configs:
        if not any(same_setting(s, config) for s in selections):
            selections.append(config)
    return selections


# ---------------------------------------------------------------- queue


def run_queue(
    configs: Sequence[RunConfig],
    provider_factory: ProviderFactory,
    output_dir: str | Path,
    *,
    run: RunFn,
    run_id_of: RunIdFn,
    lock_path: str | Path = DEFAULT_LOCK_PATH,
    log: Callable[[str], None] = print,
) -> QueueResult:
    """Run ``configs`` one at a time in priority order and save each record.

    Main queues hold :func:`main_queue_lock` for the whole queue; debug queues do not.
    ``run`` is the execution seam (``runner.run_experiment``) and ``run_id_of`` its run-id
    function, injected so the queue stays importable and testable without torch.
    """
    kind = check_queue(configs)
    ordered = order_by_priority(configs)
    _check_selection_first(ordered)
    if kind is RunKind.MAIN:
        with main_queue_lock(lock_path):
            return _run_all(ordered, provider_factory, Path(output_dir), run, run_id_of, log)
    return _run_all(ordered, provider_factory, Path(output_dir), run, run_id_of, log)


def _run_all(
    ordered: list[RunConfig],
    provider_factory: ProviderFactory,
    output_dir: Path,
    run: RunFn,
    run_id_of: RunIdFn,
    log: Callable[[str], None],
) -> QueueResult:
    selections = _selection_seeds(ordered)
    selection_records: list[ResultRecord] = []  # seen so far, from disk or this queue
    result = QueueResult()
    total = len(ordered)
    for index, config in enumerate(ordered, start=1):
        selection = next(s for s in selections if same_setting(s, config))
        run_id = run_id_of(config)
        try:
            if config.seed != selection.seed:
                config = with_selected_lr(
                    config, _selection_record(selection, selection_records, output_dir, run_id_of)
                )
            run_id = run_id_of(config)
            path = record_path(output_dir, run_id)
            if path.exists():
                existing = _matching_record(path, config)
                log(f"[queue {index}/{total}] skip {run_id} (record exists)")
                result.skipped.append(run_id)
                if config.seed == selection.seed:
                    selection_records.append(existing)
                continue
            log(f"[queue {index}/{total}] run {run_id}")
            started = time.perf_counter()
            record = run(config, provider_factory(config))
            save_result_record(record, output_dir)
        except Exception as exc:
            log(f"[queue {index}/{total}] FAILED {run_id}: {type(exc).__name__}: {exc}")
            raise QueueRunError(run_id, result) from exc
        seconds = time.perf_counter() - started
        result.completed.append(run_id)
        result.seconds[run_id] = seconds
        if config.seed == selection.seed:
            selection_records.append(record)
        log(
            f"[queue {index}/{total}] done {run_id}: eval error {record.eval_error_rate:.4f}, "
            f"lr {record.selected_lr:g}, {seconds:.1f}s"
        )
    return result


def _matching_record(path: Path, config: RunConfig) -> ResultRecord:
    """The record at ``path``, which must have been made with exactly ``config``.

    The run id leaves out the tuning budget and the expected cache metadata, so a record from
    an earlier budget has the same file name; skipping it silently would mix budgets in one
    results table. Such a record is refused instead of skipped (it is never overwritten).
    """
    existing = load_result_record(path)
    if existing.config != config:
        differing = [
            f.name
            for f in fields(RunConfig)
            if getattr(existing.config, f.name) != getattr(config, f.name)
        ]
        raise ValueError(
            f"{path} exists but was made with another config (differs in {differing}); "
            "move it aside or use another --output-dir"
        )
    return existing


def _selection_record(
    selection: RunConfig,
    selection_records: list[ResultRecord],
    output_dir: Path,
    run_id_of: RunIdFn,
) -> ResultRecord:
    """The record of ``selection``: run earlier in this queue, or found on disk."""
    for record in selection_records:
        if record.config == selection:
            return record
    path = record_path(output_dir, run_id_of(selection))
    if not path.exists():
        raise LookupError(
            f"selection-run record {path} is missing: seed {selection.seed} of this setting "
            "must run before the seeds that reuse its LR"
        )
    record = _matching_record(path, selection)
    selection_records.append(record)
    return record


def cache_provider_factory(
    cache_root: str | Path, env_config: str | Path | None = None
) -> ProviderFactory:
    """Config -> its provider; each one is built once and shared by the runs that read it.

    Frozen runs read ``CacheProvider(<cache_root>/<dataset>[_<input setting>])``. Finetune runs
    (T-06d) get ``BackboneProvider.from_env`` under the locked setting of ``env_config``.
    """
    from layer_fusion.cache_build import cache_name, load_env_config
    from layer_fusion.contracts import BackboneCondition
    from layer_fusion.providers import BackboneProvider, CacheProvider

    opened: dict[tuple[str, str], Any] = {}

    def factory(config: RunConfig) -> Any:
        name = cache_name(config.dataset, config.input_setting)
        key = (config.backbone_condition.value, name)
        if key not in opened:
            if config.backbone_condition is BackboneCondition.FINETUNE:
                if env_config is None:
                    raise ValueError("finetune runs need env_config (configs/week1_env.yaml)")
                opened[key] = BackboneProvider.from_env(config.dataset, load_env_config(env_config))
            else:
                opened[key] = CacheProvider(str(Path(cache_root) / name))
        return opened[key]

    return factory
