"""Result record JSON storage: save / load ResultRecord files (T-06b, SPEC US 48, 52).

One run -> one ``<output_dir>/<run_id>.json`` file. Loading restores every enum, tuple and float
key exactly, so a reloaded record compares equal to the record the Execution seam returned and
can go straight into the Judgment seam.

Format rules:
- ``record_format`` is stored at the top level; a file in another format is refused.
- Only standard JSON is written (no NaN / Infinity). The one legitimate non-finite value, the
  ``inf`` tuning score of a diverged LR candidate (SPEC §결정 필요 사항, T-06a), is stored as the
  string ``"inf"``; any other non-finite value fails the save instead of writing a broken file.
- Parsing is strict: unknown or missing keys raise ``ValueError`` (same policy as
  ``cache_metadata_from_dict``).
- Existing result files are never overwritten (AGENTS.md §7); only debug records may opt in.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, fields, replace
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from layer_fusion.contracts import (
    BackboneCondition,
    DataAmount,
    DatasetName,
    FusionMethod,
    InputSetting,
    LayerSubset,
    LengthBucket,
    PredictionRecord,
    ReproductionSettings,
    ResultRecord,
    RunConfig,
    RunKind,
    TrackKind,
)
from layer_fusion.providers import cache_metadata_from_dict, cache_metadata_to_dict

RECORD_FORMAT = 1
_ADDED_RUN_CONFIG_FIELD = "reproduction"  # optional in records written before T-06e
RECORD_SUFFIX = ".json"
_INF = "inf"

_RUN_CONFIG_ENUMS: dict[str, type[Enum]] = {
    "dataset": DatasetName,
    "data_amount": DataAmount,
    "fusion_method": FusionMethod,
    "backbone_condition": BackboneCondition,
    "layer_subset": LayerSubset,
    "input_setting": InputSetting,
    "run_kind": RunKind,
    "track": TrackKind,
}


def _check_keys(data: dict[str, Any], cls: type, where: str) -> None:
    if not isinstance(data, dict):
        raise ValueError(f"{where}: expected a JSON object")
    expected = {f.name for f in fields(cls)}
    unknown = sorted(data.keys() - expected)
    missing = sorted(expected - data.keys())
    if unknown or missing:
        raise ValueError(f"{where}: unknown keys {unknown}, missing keys {missing}")


def run_config_to_dict(config: RunConfig) -> dict[str, Any]:
    """JSON-ready dict of a RunConfig (enum values as plain strings)."""
    data: dict[str, Any] = {}
    for f in fields(RunConfig):
        value = getattr(config, f.name)
        if isinstance(value, Enum):
            value = value.value
        elif f.name == "lr_candidates":
            value = list(value)
        elif f.name == "expected_cache_meta" and value is not None:
            value = cache_metadata_to_dict(value)
        elif f.name == "reproduction" and value is not None:
            value = asdict(value)
        data[f.name] = value
    return data


def run_config_from_dict(data: dict[str, Any]) -> RunConfig:
    """Inverse of :func:`run_config_to_dict`.

    ``reproduction`` was added after record format 1 was first written (T-06e); a record
    without it is a main-track record and reads as ``None``.
    """
    values = dict(data) if isinstance(data, dict) else data
    if isinstance(values, dict):
        values.setdefault(_ADDED_RUN_CONFIG_FIELD, None)
    _check_keys(values, RunConfig, "RunConfig")
    for name, enum_cls in _RUN_CONFIG_ENUMS.items():
        if name != "input_setting" or values[name] is not None:
            values[name] = enum_cls(values[name])
    values["lr_candidates"] = tuple(float(lr) for lr in values["lr_candidates"])
    if values["expected_cache_meta"] is not None:
        values["expected_cache_meta"] = cache_metadata_from_dict(values["expected_cache_meta"])
    if values["reproduction"] is not None:
        _check_keys(values["reproduction"], ReproductionSettings, "RunConfig.reproduction")
        values["reproduction"] = ReproductionSettings(**values["reproduction"])
    return RunConfig(**values)


def _prediction_to_dict(pred: PredictionRecord) -> dict[str, Any]:
    data = {f.name: getattr(pred, f.name) for f in fields(PredictionRecord)}
    if pred.length_bucket is not None:
        data["length_bucket"] = pred.length_bucket.value
    return data


def _prediction_from_dict(data: dict[str, Any]) -> PredictionRecord:
    _check_keys(data, PredictionRecord, "PredictionRecord")
    values = dict(data)
    if values["length_bucket"] is not None:
        values["length_bucket"] = LengthBucket(values["length_bucket"])
    return PredictionRecord(**values)


def _score_to_json(score: float) -> float | str:
    if math.isinf(score) and score > 0:
        return _INF
    return score


def _score_from_json(score: float | str) -> float:
    if score == _INF:
        return math.inf
    if isinstance(score, str):
        raise ValueError(f"lr_tuning_scores: unexpected string score {score!r}")
    return float(score)


def result_record_to_dict(record: ResultRecord) -> dict[str, Any]:
    """JSON-ready dict of a ResultRecord, with ``record_format`` at the top level.

    ``lr_tuning_scores`` (float keys) is stored as ``[[lr, score], ...]`` in insertion order,
    because JSON object keys are strings.
    """
    data: dict[str, Any] = {"record_format": RECORD_FORMAT}
    for f in fields(ResultRecord):
        value = getattr(record, f.name)
        if f.name == "config":
            value = run_config_to_dict(value)
        elif f.name == "predictions":
            value = [_prediction_to_dict(p) for p in value]
        elif f.name == "lr_tuning_scores":
            value = [[lr, _score_to_json(score)] for lr, score in value.items()]
        data[f.name] = value
    return data


def result_record_from_dict(data: dict[str, Any]) -> ResultRecord:
    """Inverse of :func:`result_record_to_dict`; refuses another ``record_format``."""
    values = dict(data)
    record_format = values.pop("record_format", None)
    if type(record_format) is not int or record_format != RECORD_FORMAT:
        raise ValueError(
            f"record_format {record_format!r} is not supported (expected {RECORD_FORMAT})"
        )
    _check_keys(values, ResultRecord, "ResultRecord")
    values["config"] = run_config_from_dict(values["config"])
    values["predictions"] = [_prediction_from_dict(p) for p in values["predictions"]]
    values["lr_tuning_scores"] = {
        float(lr): _score_from_json(score) for lr, score in values["lr_tuning_scores"]
    }
    return ResultRecord(**values)


def _json_default(value: Any) -> Any:
    # numpy scalars (e.g. np.float32 error rates) -> plain Python numbers.
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def record_path(output_dir: str | Path, run_id: str) -> Path:
    """``<output_dir>/<run_id>.json``; the run id must be a plain file name."""
    if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError(f"run_id {run_id!r} is not a plain file name")
    return Path(output_dir) / f"{run_id}{RECORD_SUFFIX}"


def save_result_record(
    record: ResultRecord,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write one record to ``<output_dir>/<run_id>.json`` and return the path.

    An existing file raises ``FileExistsError`` (AGENTS.md §7: results are never overwritten).
    ``overwrite=True`` is accepted for debug records only. The JSON is written to a temporary
    file in the same folder and moved into place with ``os.replace``, so an interrupted save
    never leaves half a record behind.
    """
    if overwrite and record.config.run_kind is not RunKind.DEBUG:
        raise ValueError("overwrite is allowed for debug records only (AGENTS.md §7)")
    path = record_path(output_dir, record.run_id)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; result records are never overwritten")
    # Serialize before touching the disk: a NaN or unsupported value fails here, not halfway.
    text = json.dumps(
        result_record_to_dict(record),
        ensure_ascii=False,
        indent=1,
        allow_nan=False,
        default=_json_default,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return path


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value}")


def load_result_record(path: str | Path) -> ResultRecord:
    """Read one record file; its name must be ``<run_id>.json``."""
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_json_constant)
    except ValueError as exc:
        raise ValueError(f"{path}: not valid JSON ({exc})") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path}: a result record must be a JSON object")
    try:
        record = result_record_from_dict(data)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{path}: {exc}") from exc
    if path.name != f"{record.run_id}{RECORD_SUFFIX}":
        raise ValueError(f"{path}: file name does not match run_id {record.run_id!r}")
    return record


def load_result_records(runs_dir: str | Path) -> list[ResultRecord]:
    """Read every ``*.json`` record in ``runs_dir`` (sorted by file name)."""
    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        raise FileNotFoundError(f"no runs directory at {runs_dir}")
    return [load_result_record(p) for p in sorted(runs_dir.glob(f"*{RECORD_SUFFIX}"))]


def same_setting(a: RunConfig, b: RunConfig) -> bool:
    """True when two configs differ only in their seed and LR-reuse fields.

    That is one setting combination (SPEC §결정 필요 사항, T-06a): the LR is selected once per
    setting by its first seed and the other seeds reuse it. Deliberately stricter than the
    SPEC key (dataset, data amount, fusion method, layer subset, backbone, input setting):
    run kind, track, tuning budget and expected cache metadata must match too, so a debug
    or differently budgeted selection run is never reused for another run.
    """
    neutral = {"seed": 0, "fixed_lr": None, "lr_selection_seed": None}
    return replace(a, **neutral) == replace(b, **neutral)


def find_selection_record_for(
    config: RunConfig,
    records: Sequence[ResultRecord],
    *,
    label: str = "run",
) -> ResultRecord:
    """The selection-run record a reuse config (``fixed_lr`` set) must take its LR from.

    The match is the record of the same setting whose seed is ``config.lr_selection_seed``
    and that searched the candidates itself; its selected LR must equal ``config.fixed_lr``
    (SPEC §결정 필요 사항, T-06a). Usable before training, so a reuse run with a made-up
    LR is refused instead of producing a record.
    """
    if config.fixed_lr is None or config.lr_selection_seed is None:
        raise ValueError(f"{label}: a reuse run needs both fixed_lr and lr_selection_seed")
    if config.seed == config.lr_selection_seed:
        raise ValueError(f"{label}: a reuse run cannot reference its own seed")
    matches = [
        r
        for r in records
        if r.config.fixed_lr is None
        and r.config.seed == config.lr_selection_seed
        and same_setting(r.config, config)
    ]
    if not matches:
        raise LookupError(
            f"{label}: no selection-run record for seed {config.lr_selection_seed}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"{label}: {len(matches)} selection-run records for seed "
            f"{config.lr_selection_seed}: {[r.run_id for r in matches]}"
        )
    selection = matches[0]
    if selection.selected_lr != config.fixed_lr:
        raise ValueError(
            f"{label}: fixed_lr {config.fixed_lr} != selected_lr "
            f"{selection.selected_lr} of {selection.run_id}"
        )
    return selection


def find_lr_selection_record(
    record: ResultRecord,
    records: Sequence[ResultRecord],
) -> ResultRecord:
    """The selection-run record that chose ``record``'s learning rate.

    A selection run (``fixed_lr is None``) is its own selection record; a reuse run is
    resolved by :func:`find_selection_record_for`.
    """
    if record.config.fixed_lr is None:
        return record
    return find_selection_record_for(record.config, records, label=record.run_id)
