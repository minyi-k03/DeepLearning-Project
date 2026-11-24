"""Expand configs/experiment_matrix.yaml into RunConfig lists (T-06b, SPEC §실험 행렬).

TBD values (``null``) are never guessed (AGENTS.md §2): a selected experiment whose seeds,
fusion methods or tuning budget are still open is refused with the list of what is missing.
Only debug runs may pass seeds or budget values directly; main-run values come from the
locked config files (11/30 decision rules, ``week1_env.yaml``).

Every expanded config is a selection run (``fixed_lr=None``). The sequential queue (T-06c)
runs the first seed of each setting, then fills the other seeds with :func:`with_selected_lr`.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

import yaml

from layer_fusion.cache_build import load_env_config, resolve_cache_dtype
from layer_fusion.contracts import (
    BackboneCondition,
    DataAmount,
    DatasetName,
    FusionMethod,
    InputSetting,
    LayerSubset,
    ResultRecord,
    RunConfig,
    RunKind,
)
from layer_fusion.records import same_setting

# Placeholder in experiment_matrix.yaml for the method chosen after T-08d (T-07a rule).
BEST_FUSION_PLACEHOLDER = "BEST_FUSION"


@dataclass(frozen=True)
class TrainingBudget:
    """Tuning budget shared by every fusion method (ADR 0007); ``None`` = TBD."""

    lr_candidates: tuple[float, ...] | None = None
    epochs: int | None = None
    early_stopping_patience: int | None = None
    batch_size: int | None = None

    def merged(self, overrides: TrainingBudget | None) -> TrainingBudget:
        """This budget with every non-``None`` field of ``overrides`` taking precedence."""
        if overrides is None:
            return self
        return replace(
            self,
            **{
                f.name: getattr(overrides, f.name)
                for f in fields(TrainingBudget)
                if getattr(overrides, f.name) is not None
            },
        )

    def is_empty(self) -> bool:
        return all(getattr(self, f.name) is None for f in fields(TrainingBudget))

    def missing(self) -> list[str]:
        return [f.name for f in fields(TrainingBudget) if getattr(self, f.name) is None]

    def run_config_fields(self) -> dict[str, Any]:
        """The RunConfig keyword arguments of this budget; refuses a TBD (``None``) field."""
        if self.missing():
            raise ValueError(f"tuning budget is TBD: {self.missing()}")
        return {
            "lr_candidates": self.lr_candidates,
            "epochs": self.epochs,
            "early_stopping_patience": self.early_stopping_patience,
            "batch_size": self.batch_size,
        }


def resolve_budget(
    env_path: str | Path,
    run_kind: RunKind,
    overrides: TrainingBudget | None = None,
) -> TrainingBudget:
    """The budget of ``week1_env.yaml`` with ``overrides`` applied; debug runs only may
    override (main runs take the locked values, AGENTS.md §2). TBD fields stay ``None``."""
    if run_kind is RunKind.MAIN and overrides is not None and not overrides.is_empty():
        raise ValueError(
            "budget overrides are for debug runs only; main runs take the locked values"
        )
    return load_training_budget(env_path).merged(overrides)


def load_training_budget(env_path: str | Path) -> TrainingBudget:
    """Read ``training_budget`` of week1_env.yaml; ``null`` stays ``None``.

    ``early_stopping`` is read as the patience in epochs (an integer); any other shape is
    refused until the team decides it.
    """
    data = yaml.safe_load(Path(env_path).read_text(encoding="utf-8"))
    block = data.get("training_budget") or {}
    lr_candidates = block.get("lr_candidates")
    early_stopping = block.get("early_stopping")
    if early_stopping is not None and (
        isinstance(early_stopping, bool) or not isinstance(early_stopping, int)
    ):
        raise ValueError(
            f"training_budget.early_stopping must be an integer patience, got {early_stopping!r}"
        )
    return TrainingBudget(
        lr_candidates=None if lr_candidates is None else tuple(float(x) for x in lr_candidates),
        epochs=block.get("epochs"),
        early_stopping_patience=early_stopping,
        batch_size=block.get("batch_size"),
    )


def _as_list(experiment: dict[str, Any], plural: str, singular: str) -> list[Any]:
    """``input_settings: [..]`` or ``input_setting: x`` (``null`` -> ``[None]``)."""
    if plural in experiment:
        values = experiment[plural]
        if not isinstance(values, list) or not values:
            raise ValueError(f"{experiment['id']}: {plural} must be a non-empty list")
        return values
    return [experiment.get(singular)]


def _experiment_problems(experiment: dict[str, Any], seeds: list[int] | None) -> list[str]:
    exp_id = experiment["id"]
    problems = []
    if seeds is None:
        problems.append(f"{exp_id}: seeds is TBD")
    methods = experiment.get("fusion_methods")
    if methods is None:
        problems.append(f"{exp_id}: fusion_methods is TBD")
    elif BEST_FUSION_PLACEHOLDER in methods:
        problems.append(
            f"{exp_id}: fusion_methods has {BEST_FUSION_PLACEHOLDER} "
            "(chosen from T-08d results by the 11/30 rule)"
        )
    return problems


def _check_seeds(seeds: Any, where: str) -> list[int]:
    if (
        not isinstance(seeds, list | tuple)
        or not seeds
        or not all(isinstance(s, int) and not isinstance(s, bool) for s in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError(f"{where}: seeds must be a non-empty list of distinct integers")
    return sorted(seeds)


def expand_experiment_matrix(
    matrix_path: str | Path,
    env_path: str | Path,
    *,
    experiment_ids: list[str] | None = None,
    seeds: list[int] | None = None,
    run_kind: RunKind = RunKind.MAIN,
    budget_overrides: TrainingBudget | None = None,
) -> list[RunConfig]:
    """Expand the selected experiments into one RunConfig per (input setting, layer subset,
    fusion method, seed), in that nesting order with seeds ascending.

    The first seed of each setting is its LR-selection run. ``expected_cache_meta`` is built
    from the locked extraction setting in ``week1_env.yaml`` (main runs must declare it).
    The ``reproduction_track`` block is not expanded here (T-06e).
    """
    if run_kind is RunKind.MAIN and seeds is not None:
        raise ValueError(
            "seed overrides are for debug runs only; main runs take the locked values"
        )
    budget = resolve_budget(env_path, run_kind, budget_overrides)
    matrix = yaml.safe_load(Path(matrix_path).read_text(encoding="utf-8"))
    experiments = matrix.get("experiments") or []
    by_id = {exp["id"]: exp for exp in experiments}
    if experiment_ids is not None:
        unknown = sorted(set(experiment_ids) - by_id.keys())
        if unknown:
            raise ValueError(f"unknown experiment ids {unknown}; known: {sorted(by_id)}")
        experiments = [by_id[exp_id] for exp_id in experiment_ids]

    problems =[f"training_budget.{name} is TBD" for name in budget.missing()]
    for exp in experiments:
        problems += _experiment_problems(exp, seeds if seeds is not None else exp.get("seeds"))
    if problems:
        raise ValueError("cannot expand the experiment matrix (TBD values):\n- " + "\n- ".join(
            problems
        ))

    env = load_env_config(env_path)
    dtype = resolve_cache_dtype(env, None)
    configs: list[RunConfig] = []
    for exp in experiments:
        exp_seeds = _check_seeds(seeds if seeds is not None else exp["seeds"], exp["id"])
        input_settings = [
            None if v is None else InputSetting(v)
            for v in _as_list(exp, "input_settings", "input_setting")
        ]
        layer_subsets = [LayerSubset(v) for v in _as_list(exp, "layer_subsets", "layer_subset")]
        methods = [FusionMethod(m) for m in exp["fusion_methods"]]
        for input_setting, subset, method, seed in itertools.product(
            input_settings, layer_subsets, methods, exp_seeds
        ):
            configs.append(
                RunConfig(
                    dataset=DatasetName(exp["dataset"]),
                    data_amount=DataAmount(exp["data_amount"]),
                    fusion_method=method,
                    backbone_condition=BackboneCondition(exp["backbone_condition"]),
                    seed=seed,
                    layer_subset=subset,
                    input_setting=input_setting,
                    **budget.run_config_fields(),
                    run_kind=run_kind,
                    expected_cache_meta=env.cache_metadata(input_setting, dtype),
                )
            )
    return configs


def with_selected_lr(config: RunConfig, selection_record: ResultRecord) -> RunConfig:
    """``config`` reusing the LR chosen by ``selection_record`` (SPEC §결정 필요 사항, T-06a).

    The selection record must be a selection run of the same setting with another seed.
    """
    selection = selection_record.config
    if selection.fixed_lr is not None:
        raise ValueError(f"{selection_record.run_id} is not a selection run (it reused an LR)")
    if not same_setting(selection, config):
        raise ValueError(f"{selection_record.run_id} is from another setting than this config")
    if selection.seed == config.seed:
        raise ValueError(f"seed {config.seed} is the selection run itself")
    return replace(
        config,
        fixed_lr=selection_record.selected_lr,
        lr_selection_seed=selection.seed,
    )
