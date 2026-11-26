"""experiment_matrix.yaml -> RunConfig expansion (T-06b). CPU only, no data."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from layer_fusion.cache_build import load_env_config
from layer_fusion.contracts import (
    DataAmount,
    DatasetName,
    FusionMethod,
    InputSetting,
    LayerSubset,
    ResultRecord,
    RunKind,
    TrackKind,
)
from layer_fusion.matrix import (
    BEST_FUSION_RULE,
    TrainingBudget,
    expand_experiment_matrix,
    load_training_budget,
    select_best_fusion,
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


def test_real_matrix_seed_lists_follow_the_decision_rules() -> None:
    """T-07a: seed counts live in two files; they must agree and the lists must be nested."""
    rules = yaml.safe_load((ROOT / "configs" / "decision_rules.yaml").read_text(encoding="utf-8"))
    criteria = rules["criteria"]
    experiments = {
        exp["id"]: exp
        for exp in yaml.safe_load(REAL_MATRIX.read_text(encoding="utf-8"))["experiments"]
    }
    core_ids = set(criteria["core_comparisons_for_5_seeds"])
    assert core_ids <= experiments.keys()

    for exp_id, exp in experiments.items():
        expected = criteria["core_num_seeds"] if exp_id in core_ids else (
            criteria["finetune_num_seeds"]
            if exp["backbone_condition"] == "finetune"
            else criteria["secondary_num_seeds"]
        )
        assert exp["seeds"] == list(range(expected)), exp_id  # nested: [0..n-1]


def test_real_config_files_are_refused_only_for_best_fusion() -> None:
    """The budget is filled (11/26); only BEST_FUSION (picked from T-08d) blocks expansion."""
    with pytest.raises(ValueError) as info:
        expand_experiment_matrix(REAL_MATRIX, REAL_ENV)
    message = str(info.value)
    assert "training_budget" not in message  # 11/26 T-08d budget: batch 32, epochs 20, patience 3
    assert "seeds is TBD" not in message  # fixed at the T-07a draft: core [0..4], others [0..2]
    assert "fusion_methods is TBD" not in message  # agenda B (11/25): small and SST-2 lists
    assert "lr_candidates is TBD" not in message  # agenda A (11/25): both candidate lists
    assert "mnli_finetune_full: fusion_methods has BEST_FUSION" in message


def test_real_budget_has_no_tbd_left() -> None:
    assert load_training_budget(REAL_ENV).missing() == []


def test_real_config_holds_the_t08d_budget_values() -> None:
    """11/26 T-08d budget (issue #32): a generous epoch cap ended by early stopping and batch 32,
    shared by both backbone conditions (no fine-tune-only epochs or batch)."""
    budget = load_training_budget(REAL_ENV)
    assert budget.epochs == 20
    assert budget.early_stopping_patience == 3
    assert budget.batch_size == 32
    env = yaml.safe_load(REAL_ENV.read_text(encoding="utf-8"))
    assert env["training_budget"]["extract_time_included_in_train_time"] is False


def test_real_config_holds_the_agenda_a_and_b_values() -> None:
    """11/25 decisions: LR candidates (agenda A) and the small / SST-2 fusion lists (agenda B)."""
    budget = load_training_budget(REAL_ENV)
    assert budget.lr_candidates == (1e-4, 1e-3, 1e-2)
    assert budget.finetune_lr_candidates == (2e-5, 5e-5, 8e-5)
    experiments = {
        exp["id"]: exp
        for exp in yaml.safe_load(REAL_MATRIX.read_text(encoding="utf-8"))["experiments"]
    }
    assert experiments["mnli_frozen_small"]["fusion_methods"] == ["B0", "F1", "F2"]
    assert experiments["sst2_frozen_full"]["fusion_methods"] == ["B0", "F1", "F2"]
    rules = yaml.safe_load((ROOT / "configs" / "decision_rules.yaml").read_text(encoding="utf-8"))
    h1_target = rules["hypotheses"]["H1"]["target_fusion_rule"]
    # H2 compares the small and full effects of the H1 method against B0
    assert {"B0", h1_target} <= set(experiments["mnli_frozen_small"]["fusion_methods"])
    assert "F2" in experiments["sst2_frozen_full"]["fusion_methods"]  # H4 compares F2 weights


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


def test_budget_helpers_shared_by_cli_and_expansion(tmp_path: Path, tbd_env: Path) -> None:
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
        resolve_budget(tbd_env, RunKind.MAIN).run_config_fields()


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
    env = _env_file(tmp_path, **BUDGET, finetune_lr_candidates=None)  # TBD on purpose
    assert expand_experiment_matrix(_matrix_file(tmp_path, [MNLI]), env)
    with pytest.raises(ValueError, match="training_budget.finetune_lr_candidates is TBD"):
        expand_experiment_matrix(_matrix_file(tmp_path, [MNLI, MNLI_FINETUNE]), env)


# --- BEST_FUSION: the fine-tune comparison method chosen by the T-07a rule (before T-08f) ----

ALL_METHODS = ["B0", "F1", "F2", "F3", "F4", "B0+"]


def _frozen_records(tmp_path: Path, scores: dict[str, float], kind: RunKind = RunKind.MAIN):  # type: ignore[no-untyped-def]
    """MNLI frozen-full records: a selection run (seed 0) whose selected LR scored ``scores``
    on the tuning split, plus a reuse run (seed 1) with a tempting low eval error rate."""
    matrix = _matrix_file(tmp_path, [{**MNLI, "fusion_methods": ALL_METHODS, "seeds": [0, 1]}])
    configs = expand_experiment_matrix(
        matrix, _env_file(tmp_path, **BUDGET), run_kind=kind,
        **({"seeds": [0, 1]} if kind is RunKind.DEBUG else {}),
    )
    records = []
    for config in configs:
        method = config.fusion_method.value
        if config.seed == 0:
            records.append(ResultRecord(
                run_id=f"{method}-s0", config=config, selected_lr=1e-3,
                lr_tuning_scores={1e-4: 0.9, 1e-3: scores.get(method, 0.5), 1e-2: 0.95},
                eval_error_rate=0.4, predictions=[], trainable_params=1, train_seconds=1.0,
            ))
    for sel in list(records):
        reuse = with_selected_lr(replace(sel.config, seed=1), sel)
        records.append(ResultRecord(
            run_id=f"{sel.config.fusion_method.value}-s1", config=reuse, selected_lr=1e-3,
            lr_tuning_scores={}, eval_error_rate=0.01, predictions=[], trainable_params=1,
            train_seconds=1.0,
        ))
    return records


def test_best_fusion_is_the_lowest_tuning_error_of_the_selection_runs(tmp_path: Path) -> None:
    records = _frozen_records(tmp_path, {"F1": 0.30, "F2": 0.25, "F3": 0.20, "F4": 0.22})
    # a lower eval error rate elsewhere never decides (the eval set is not used to choose)
    records = [replace(r, eval_error_rate=0.0) if r.run_id == "F1-s0" else r for r in records]
    choice = select_best_fusion(records, BEST_FUSION_RULE)
    assert choice.method is FusionMethod.F3
    assert choice.scores == {FusionMethod.F1: 0.30, FusionMethod.F2: 0.25,
                             FusionMethod.F3: 0.20, FusionMethod.F4: 0.22}
    assert choice.run_ids == {FusionMethod.F1: "F1-s0", FusionMethod.F2: "F2-s0",
                              FusionMethod.F3: "F3-s0", FusionMethod.F4: "F4-s0"}


def test_best_fusion_tie_goes_to_the_lower_number_and_divergence_never_wins(
    tmp_path: Path,
) -> None:
    tie = _frozen_records(tmp_path, {"F1": 0.3, "F2": 0.2, "F3": 0.2, "F4": 0.2})
    assert select_best_fusion(tie, BEST_FUSION_RULE).method is FusionMethod.F2
    inf = float("inf")
    diverged = _frozen_records(tmp_path, {"F1": inf, "F2": 0.4, "F3": inf, "F4": 0.3})
    assert select_best_fusion(diverged, BEST_FUSION_RULE).method is FusionMethod.F4
    all_diverged = _frozen_records(tmp_path, {m: inf for m in ("F1", "F2", "F3", "F4")})
    with pytest.raises(ValueError, match="diverged"):
        select_best_fusion(all_diverged, BEST_FUSION_RULE)


def test_best_fusion_ignores_b0_b0plus_and_records_outside_the_main_frozen_mnli_cell(
    tmp_path: Path,
) -> None:
    records = _frozen_records(tmp_path, {"B0": 0.0, "B0+": 0.0, "F1": 0.3, "F2": 0.2,
                                         "F3": 0.25, "F4": 0.4})
    f1 = next(r for r in records if r.run_id == "F1-s0")
    outsiders = [
        replace(f1, run_id="debug", config=replace(f1.config, run_kind=RunKind.DEBUG),
                lr_tuning_scores={1e-3: 0.0}),
        replace(f1, run_id="repro", config=replace(f1.config, track=TrackKind.REPRODUCTION),
                lr_tuning_scores={1e-3: 0.0}),
        replace(f1, run_id="small", config=replace(f1.config, data_amount=DataAmount.SMALL),
                lr_tuning_scores={1e-3: 0.0}),
        replace(f1, run_id="sst2", config=replace(f1.config, dataset=DatasetName.SST2),
                lr_tuning_scores={1e-3: 0.0}),
    ]
    assert select_best_fusion(records + outsiders, BEST_FUSION_RULE).method is FusionMethod.F2


def test_debug_selection_reads_debug_records_only(tmp_path: Path) -> None:
    main = _frozen_records(tmp_path, {"F1": 0.1, "F2": 0.3, "F3": 0.3, "F4": 0.3})
    debug = _frozen_records(tmp_path, {"F1": 0.3, "F2": 0.3, "F3": 0.1, "F4": 0.3}, RunKind.DEBUG)
    both = main + debug
    assert select_best_fusion(both, BEST_FUSION_RULE).method is FusionMethod.F1
    chosen = select_best_fusion(both, BEST_FUSION_RULE, run_kind=RunKind.DEBUG)
    assert chosen.method is FusionMethod.F3


def test_best_fusion_refuses_incomplete_or_doubled_selection_runs(tmp_path: Path) -> None:
    records = _frozen_records(tmp_path, {"F1": 0.3, "F2": 0.2, "F3": 0.25, "F4": 0.4})
    without_f3 = [r for r in records if r.config.fusion_method is not FusionMethod.F3]
    with pytest.raises(ValueError, match="F3"):
        select_best_fusion(without_f3, BEST_FUSION_RULE)
    f2 = next(r for r in records if r.run_id == "F2-s0")
    doubled = records + [replace(f2, run_id="F2-s0-again")]
    with pytest.raises(ValueError, match="F2"):
        select_best_fusion(doubled, BEST_FUSION_RULE)


@pytest.mark.parametrize(("rule", "match"), [(None, "TBD"), ("lowest_eval_error", "unknown")])
def test_best_fusion_rule_must_be_the_known_one(tmp_path: Path, rule, match) -> None:  # type: ignore[no-untyped-def]
    records = _frozen_records(tmp_path, {"F1": 0.3, "F2": 0.2, "F3": 0.25, "F4": 0.4})
    with pytest.raises(ValueError, match=match):
        select_best_fusion(records, rule)


def test_best_fusion_fills_the_placeholder_only_with_a_fusion_method(tmp_path: Path) -> None:
    env = _env_file(tmp_path, **BUDGET, finetune_lr_candidates=[1e-5, 2e-5, 5e-5])
    finetune = {**MNLI_FINETUNE, "fusion_methods": ["B0", "BEST_FUSION"], "seeds": [0, 1]}
    matrix = _matrix_file(tmp_path, [finetune])
    with pytest.raises(ValueError, match="BEST_FUSION"):
        expand_experiment_matrix(matrix, env)
    configs = expand_experiment_matrix(matrix, env, best_fusion=FusionMethod.F3)
    assert [(c.fusion_method, c.seed) for c in configs] == [
        (FusionMethod.B0, 0), (FusionMethod.B0, 1), (FusionMethod.F3, 0), (FusionMethod.F3, 1)
    ]
    for wrong in (FusionMethod.B0, FusionMethod.B0_PLUS):
        with pytest.raises(ValueError, match="F1 to F4"):
            expand_experiment_matrix(matrix, env, best_fusion=wrong)


def test_main_selection_runs_must_be_seed_0_as_the_rule_says(tmp_path: Path) -> None:
    """Rule text: 'the LR-selection run (seed 0)'. A main selection run on another seed is refused;
    debug queues stay flexible."""
    main = _frozen_records(tmp_path, {"F1": 0.3, "F2": 0.2, "F3": 0.25, "F4": 0.4})
    shifted = [
        replace(r, config=replace(r.config, seed=7)) if r.run_id == "F2-s0" else r for r in main
    ]
    with pytest.raises(ValueError, match="seed 0"):
        select_best_fusion(shifted, BEST_FUSION_RULE)
    debug = _frozen_records(tmp_path, {"F1": 0.3, "F2": 0.2, "F3": 0.25, "F4": 0.4}, RunKind.DEBUG)
    debug = [
        replace(r, config=replace(r.config, seed=7)) if r.run_id == "F2-s0" else r for r in debug
    ]
    assert select_best_fusion(debug, BEST_FUSION_RULE, run_kind=RunKind.DEBUG).method is (
        FusionMethod.F2
    )


def test_best_fusion_ignores_runs_with_an_input_setting(tmp_path: Path) -> None:
    records = _frozen_records(tmp_path, {"F1": 0.3, "F2": 0.2, "F3": 0.25, "F4": 0.4})
    f1 = next(r for r in records if r.run_id == "F1-s0")
    odd = replace(f1, run_id="odd", config=replace(f1.config, input_setting=InputSetting.L512),
                  lr_tuning_scores={1e-3: 0.0})
    assert select_best_fusion(records + [odd], BEST_FUSION_RULE).method is FusionMethod.F2


@pytest.mark.parametrize("bad", [float("nan"), float("-inf")])
def test_nan_or_negative_infinite_scores_are_refused_not_skipped(tmp_path: Path, bad) -> None:  # type: ignore[no-untyped-def]
    """Only +inf means 'diverged'; NaN / -inf is a corrupted record and must not be dropped."""
    records = _frozen_records(tmp_path, {"F1": bad, "F2": 0.2, "F3": 0.25, "F4": 0.4})
    with pytest.raises(ValueError, match="F1"):
        select_best_fusion(records, BEST_FUSION_RULE)
