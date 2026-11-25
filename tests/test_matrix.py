"""experiment_matrix.yaml -> RunConfig expansion (T-06b). CPU only, no data."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from layer_fusion.cache_build import load_env_config
from layer_fusion.contracts import (
    FusionMethod,
    InputSetting,
    LayerSubset,
    ResultRecord,
    RunKind,
)
from layer_fusion.matrix import (
    TrainingBudget,
    expand_experiment_matrix,
    load_training_budget,
    with_selected_lr,
)

ROOT = Path(__file__).resolve().parent.parent
REAL_MATRIX = ROOT / "configs" / "experiment_matrix.yaml"
REAL_ENV = ROOT / "configs" / "week1_env.yaml"

BUDGET = {"lr_candidates": [1e-4, 1e-3, 1e-2], "epochs": 3, "early_stopping": 2, "batch_size": 32}


def _env_file(tmp_path: Path, **budget: object) -> Path:
    """A copy of the real week1_env.yaml with the training budget replaced."""
    data = yaml.safe_load(REAL_ENV.read_text(encoding="utf-8"))
    data["training_budget"].update(budget)
    path = tmp_path / "env.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


def _matrix_file(tmp_path: Path, experiments: list[dict]) -> Path:
    path = tmp_path / "matrix.yaml"
    path.write_text(yaml.safe_dump({"experiments": experiments}), encoding="utf-8")
    return path


MNLI = {
    "id": "mnli_frozen_full", "dataset": "mnli", "backbone_condition": "frozen",
    "data_amount": "full", "input_setting": None, "fusion_methods": ["B0", "F2"],
    "layer_subset": "all", "seeds": [2, 0, 1],
}
IMDB_ABLATION = {
    "id": "imdb_ablation", "dataset": "imdb", "backbone_condition": "frozen",
    "data_amount": "full", "input_settings": ["L128", "Lfull"], "fusion_methods": ["F2"],
    "layer_subsets": ["global_only", "local_only"], "seeds": [0, 1],
}


def test_real_config_files_are_refused_while_values_are_tbd() -> None:
    """Today the real matrix and budget are TBD; expansion must say what is missing."""
    with pytest.raises(ValueError) as info:
        expand_experiment_matrix(REAL_MATRIX, REAL_ENV)
    message = str(info.value)
    assert "training_budget.epochs is TBD" in message
    assert "mnli_frozen_full: seeds is TBD" in message
    assert "mnli_frozen_small: fusion_methods is TBD" in message
    assert "mnli_finetune_full: fusion_methods has BEST_FUSION" in message


def test_real_budget_is_read_as_tbd() -> None:
    assert load_training_budget(REAL_ENV).missing() == [
        "lr_candidates", "epochs", "early_stopping_patience", "batch_size",
    ]


def test_expansion_order_and_fields(tmp_path: Path) -> None:
    env = _env_file(tmp_path, **BUDGET)
    configs = expand_experiment_matrix(_matrix_file(tmp_path, [MNLI]), env)
    assert [(c.fusion_method, c.seed) for c in configs] == [
        (FusionMethod.B0, 0), (FusionMethod.B0, 1), (FusionMethod.B0, 2),
        (FusionMethod.F2, 0), (FusionMethod.F2, 1), (FusionMethod.F2, 2),
    ]
    first = configs[0]
    assert first.run_kind is RunKind.MAIN
    assert first.lr_candidates == (1e-4, 1e-3, 1e-2)
    assert (first.epochs, first.early_stopping_patience, first.batch_size) == (3, 2, 32)
    assert first.fixed_lr is None and first.lr_selection_seed is None
    expected = load_env_config(env).cache_metadata(None, "float16")
    assert all(c.expected_cache_meta == expected for c in configs)


def test_input_settings_and_layer_subsets_multiply(tmp_path: Path) -> None:
    env = _env_file(tmp_path, **BUDGET)
    configs = expand_experiment_matrix(_matrix_file(tmp_path, [IMDB_ABLATION]), env)
    assert len(configs) == 2 * 2 * 1 * 2
    assert [(c.input_setting, c.layer_subset) for c in configs[::2]] == [
        (InputSetting.L128, LayerSubset.GLOBAL_ONLY),
        (InputSetting.L128, LayerSubset.LOCAL_ONLY),
        (InputSetting.LFULL, LayerSubset.GLOBAL_ONLY),
        (InputSetting.LFULL, LayerSubset.LOCAL_ONLY),
    ]
    lfull = configs[-1].expected_cache_meta
    assert lfull is not None and lfull.input_setting is InputSetting.LFULL
    assert lfull.max_length_cap == 4096


def test_experiment_selection(tmp_path: Path) -> None:
    env = _env_file(tmp_path, **BUDGET)
    tbd = {**MNLI, "id": "tbd_one", "seeds": None}
    matrix = _matrix_file(tmp_path, [MNLI, tbd])
    configs = expand_experiment_matrix(matrix, env, experiment_ids=["mnli_frozen_full"])
    assert len(configs) == 6  # the TBD experiment was not selected, so it does not block
    with pytest.raises(ValueError, match="unknown experiment ids"):
        expand_experiment_matrix(matrix, env, experiment_ids=["nope"])


def test_debug_runs_may_fill_tbd_seeds_and_budget() -> None:
    configs = expand_experiment_matrix(
        REAL_MATRIX, REAL_ENV,
        experiment_ids=["mnli_frozen_full"],
        seeds=[0],
        run_kind=RunKind.DEBUG,
        budget_overrides=TrainingBudget((1e-3,), 1, 1, 16),
    )
    assert len(configs) == 6  # B0, F1, F2, F3, F4, B0+
    assert all(c.run_kind is RunKind.DEBUG and c.seed == 0 for c in configs)


def test_main_runs_cannot_override_seeds_or_budget(tmp_path: Path) -> None:
    env = _env_file(tmp_path, **BUDGET)
    matrix = _matrix_file(tmp_path, [MNLI])
    with pytest.raises(ValueError, match="debug runs only"):
        expand_experiment_matrix(matrix, env, seeds=[0, 1])
    with pytest.raises(ValueError, match="debug runs only"):
        expand_experiment_matrix(matrix, env, budget_overrides=TrainingBudget(epochs=1))


@pytest.mark.parametrize("seeds", [[], [0, 0], [0.5], "0,1"])
def test_bad_seeds_are_refused(tmp_path: Path, seeds: object) -> None:
    env = _env_file(tmp_path, **BUDGET)
    with pytest.raises(ValueError, match="seeds"):
        expand_experiment_matrix(_matrix_file(tmp_path, [{**MNLI, "seeds": seeds}]), env)


def test_non_integer_early_stopping_is_refused(tmp_path: Path) -> None:
    env = _env_file(tmp_path, **{**BUDGET, "early_stopping": {"patience": 2}})
    with pytest.raises(ValueError, match="early_stopping"):
        load_training_budget(env)


def _selection_record(config) -> ResultRecord:  # type: ignore[no-untyped-def]
    return ResultRecord(
        run_id="sel", config=config, selected_lr=1e-3, lr_tuning_scores={1e-3: 0.1},
        eval_error_rate=0.2, predictions=[], trainable_params=1, train_seconds=1.0,
    )


def test_with_selected_lr_fills_the_reuse_fields(tmp_path: Path) -> None:
    env = _env_file(tmp_path, **BUDGET)
    configs = expand_experiment_matrix(_matrix_file(tmp_path, [MNLI]), env)
    selection = _selection_record(configs[0])
    reuse = with_selected_lr(configs[1], selection)
    assert (reuse.fixed_lr, reuse.lr_selection_seed, reuse.seed) == (1e-3, 0, 1)
    assert replace(reuse, fixed_lr=None, lr_selection_seed=None) == configs[1]


def test_with_selected_lr_refuses_mismatches(tmp_path: Path) -> None:
    env = _env_file(tmp_path, **BUDGET)
    configs = expand_experiment_matrix(_matrix_file(tmp_path, [MNLI]), env)
    selection = _selection_record(configs[0])
    with pytest.raises(ValueError, match="another setting"):
        with_selected_lr(configs[4], selection)  # F2 config, B0 selection
    with pytest.raises(ValueError, match="selection run itself"):
        with_selected_lr(configs[0], selection)
    reused = _selection_record(with_selected_lr(configs[1], selection))
    with pytest.raises(ValueError, match="not a selection run"):
        with_selected_lr(configs[2], reused)


def test_budget_helpers_shared_by_cli_and_expansion(tmp_path: Path) -> None:
    from layer_fusion.matrix import resolve_budget

    env = _env_file(tmp_path, **BUDGET)
    fields_ = resolve_budget(env, RunKind.MAIN).run_config_fields()
    assert fields_ == {
        "lr_candidates": (1e-4, 1e-3, 1e-2), "epochs": 3,
        "early_stopping_patience": 2, "batch_size": 32,
    }
    debug = resolve_budget(env, RunKind.DEBUG, TrainingBudget(epochs=1))
    assert debug.epochs == 1 and debug.batch_size == 32
    with pytest.raises(ValueError, match="debug runs only"):
        resolve_budget(env, RunKind.MAIN, TrainingBudget(epochs=1))
    with pytest.raises(ValueError, match="TBD"):
        resolve_budget(REAL_ENV, RunKind.MAIN).run_config_fields()


MNLI_FINETUNE = {
    "id": "mnli_finetune_full", "dataset": "mnli", "backbone_condition": "finetune",
    "data_amount": "full", "input_setting": None, "fusion_methods": ["B0", "F2"],
    "layer_subset": "all", "seeds": [0],
}


def test_finetune_settings_take_the_finetune_lr_candidates_and_float32_meta(
    tmp_path: Path,
) -> None:
    """T-06d: finetune LRs come from ``finetune_lr_candidates``; layer vectors are computed in
    float32 live, so the expected metadata does not carry the float16 cache dtype."""
    env = _env_file(tmp_path, **BUDGET, finetune_lr_candidates=[1e-5, 2e-5, 5e-5])
    matrix = _matrix_file(tmp_path, [MNLI, MNLI_FINETUNE])
    configs = expand_experiment_matrix(matrix, env)
    frozen = [c for c in configs if c.backbone_condition.value == "frozen"]
    finetune = [c for c in configs if c.backbone_condition.value == "finetune"]
    assert {c.lr_candidates for c in frozen} == {(1e-4, 1e-3, 1e-2)}
    assert {c.lr_candidates for c in finetune} == {(1e-5, 2e-5, 5e-5)}
    assert {c.expected_cache_meta.dtype for c in finetune} == {"float32"}
    assert {c.expected_cache_meta.dtype for c in frozen} == {"float16"}


def test_finetune_lr_candidates_tbd_blocks_only_finetune_experiments(tmp_path: Path) -> None:
    env = _env_file(tmp_path, **BUDGET)  # finetune_lr_candidates stays null in the real file
    assert expand_experiment_matrix(_matrix_file(tmp_path, [MNLI]), env)
    with pytest.raises(ValueError, match="training_budget.finetune_lr_candidates is TBD"):
        expand_experiment_matrix(_matrix_file(tmp_path, [MNLI, MNLI_FINETUNE]), env)
