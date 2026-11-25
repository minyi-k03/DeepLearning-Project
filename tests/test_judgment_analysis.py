"""T-07c tests: per-hypothesis evaluators and analyses at the judgment seam.

Everything runs on CPU with hand-built ResultRecords (SPEC §Seam ②). The rule values used here
(alpha, aggregation, consistency thresholds) are synthetic arguments, not 11/30 decisions.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import replace
from pathlib import Path

import pytest

from layer_fusion.contracts import (
    BackboneCondition,
    DataAmount,
    DatasetName,
    DecisionRules,
    FusionMethod,
    InputSetting,
    LayerSubset,
    PredictionRecord,
    ResultRecord,
    RunConfig,
    Verdict,
)
from layer_fusion.inputs import assign_length_bucket
from layer_fusion.judgment import (
    compute_center_of_gravity,
    compute_global_layer_weight_ratio,
    judge_hypotheses,
    load_decision_rules,
)

RULES_PATH = Path(__file__).resolve().parent.parent / "configs" / "decision_rules.yaml"


def make_rules(**overrides: object) -> DecisionRules:
    base = replace(
        load_decision_rules(str(RULES_PATH)),
        mcnemar_alpha=0.05,
        multi_seed_mcnemar_aggregation="all_seeds_significant",
        h1_target_fusion_rule="F1",
        h5a_target_fusion_rule="F2",
    )
    return replace(base, **overrides)


def make_record(
    method: FusionMethod,
    seed: int,
    error_docs: Iterable[int],
    *,
    n_docs: int,
    dataset: DatasetName = DatasetName.MNLI,
    amount: DataAmount = DataAmount.FULL,
    backbone: BackboneCondition = BackboneCondition.FROZEN,
    input_setting: InputSetting | None = None,
    layer_subset: LayerSubset = LayerSubset.ALL,
    genres: bool = False,
    token_lengths: Sequence[int] | None = None,
    layer_weights: list[float] | None = None,
    doc_layer_weights: Callable[[int], list[float]] | None = None,
) -> ResultRecord:
    """Record whose wrong predictions are exactly ``error_docs`` (labels alternate 0, 1).

    ``genres=True`` marks the first half of the documents "matched" and the second half
    "mismatched" and fills the per-genre error rates. ``token_lengths`` gives IMDB-style
    per-document lengths (the length bucket follows only under an input setting, as the runner
    records it).
    """
    wrong = set(error_docs)
    predictions = []
    for i in range(n_docs):
        label = i % 2
        predictions.append(
            PredictionRecord(
                doc_id=f"{dataset.value}-{i:05d}",
                label=label,
                prediction=1 - label if i in wrong else label,
                token_length=None if token_lengths is None else token_lengths[i],
                length_bucket=(
                    assign_length_bucket(token_lengths[i])
                    if token_lengths is not None and input_setting is not None
                    else None
                ),
                genre_split=(("matched" if i < n_docs // 2 else "mismatched") if genres else None),
                layer_weights=None if doc_layer_weights is None else doc_layer_weights(i),
            )
        )
    half = n_docs // 2
    matched_errors = sum(1 for i in wrong if i < half)
    mismatched_errors = len(wrong) - matched_errors
    config = RunConfig(
        dataset=dataset,
        data_amount=amount,
        fusion_method=method,
        backbone_condition=backbone,
        seed=seed,
        layer_subset=layer_subset,
        input_setting=input_setting,
    )
    return ResultRecord(
        run_id=f"{dataset.value}-{amount.value}-{backbone.value}-{method.value}-s{seed}",
        config=config,
        selected_lr=1e-3,
        lr_tuning_scores={1e-3: 0.0},
        eval_error_rate=len(wrong) / n_docs,
        predictions=predictions,
        trainable_params=0,
        train_seconds=0.0,
        eval_error_rate_matched=matched_errors / half if genres else None,
        eval_error_rate_mismatched=mismatched_errors / (n_docs - half) if genres else None,
        layer_weights=layer_weights,
    )


# --- H4': the same B0-vs-target comparison, on the mismatched genres only -------------------


def genre_records(
    b0_errors: Callable[[int], set[int]], f1_errors: Callable[[int], set[int]]
) -> list[ResultRecord]:
    """400 MNLI docs (0-199 matched, 200-399 mismatched), 3 seeds, B0 and F1."""
    records = []
    for seed in range(3):
        records.append(
            make_record(FusionMethod.B0, seed, b0_errors(seed), n_docs=400, genres=True)
        )
        records.append(
            make_record(FusionMethod.F1, seed, f1_errors(seed), n_docs=400, genres=True)
        )
    return records


def test_h4_prime_judges_the_mismatched_genres() -> None:
    # Mismatched docs 200+: B0 errs on 80+s of them, F1 only on the first 40+s of those.
    records = genre_records(
        lambda s: set(range(200, 280 + s)), lambda s: set(range(200, 240 + s))
    )

    h4p = judge_hypotheses(records, make_rules()).judgments["H4_prime"]

    assert h4p.kind == "confirmatory"
    assert h4p.verdict == Verdict.SUPPORTED
    # mismatched error: B0 mean 81/200 = .405, F1 mean 41/200 = .205 -> improvement 0.2
    assert h4p.mean_diff == pytest.approx(0.20)
    assert h4p.seed_std == pytest.approx(0.005)
    assert h4p.mcnemar_p_value is not None and h4p.mcnemar_p_value < 0.05


def test_h4_prime_ignores_a_gain_that_exists_only_on_matched_genres() -> None:
    # F1 fixes 60 matched docs per seed but nothing on the mismatched half.
    records = genre_records(
        lambda s: set(range(80 + s)) | set(range(200, 240)),
        lambda s: set(range(20 + s)) | set(range(200, 240)),
    )

    judgments = judge_hypotheses(records, make_rules()).judgments

    assert judgments["H1"].verdict == Verdict.SUPPORTED  # the whole set improves ...
    assert judgments["H4_prime"].verdict == Verdict.NO_EFFECT  # ... the unseen genres do not
    assert judgments["H4_prime"].mean_diff == pytest.approx(0.0)


def test_h4_prime_is_left_out_without_genre_information() -> None:
    """Records that carry no matched/mismatched split give nothing to judge: no row."""
    records = [
        replace(rec, eval_error_rate_matched=None, eval_error_rate_mismatched=None)
        for rec in genre_records(lambda s: set(range(60 + s)), lambda s: set(range(20 + s)))
    ]
    stripped = [
        replace(
            rec,
            predictions=[replace(p, genre_split=None) for p in rec.predictions],
        )
        for rec in records
    ]

    judgments = judge_hypotheses(stripped, make_rules()).judgments

    assert "H1" in judgments
    assert "H4_prime" not in judgments


def test_h4_prime_refuses_a_run_that_lacks_the_genre_split_its_peers_have() -> None:
    records = genre_records(lambda s: set(range(60 + s)), lambda s: set(range(20 + s)))
    records[0] = replace(records[0], eval_error_rate_mismatched=None)

    with pytest.raises(ValueError, match="mismatched"):
        judge_hypotheses(records, make_rules())


# --- IMDB fixtures and H5a: does the long input itself help (L512 vs Lfull)? ----------------

IMDB_DOCS = 200
# docs 0-99 fit in 512 tokens, 100-149 are 513-1024 long, 150-199 are over 1024.
IMDB_LENGTHS = [300] * 100 + [700] * 50 + [1500] * 50


def imdb_record(
    method: FusionMethod,
    seed: int,
    setting: InputSetting,
    errors: Iterable[int],
    **kwargs: object,
) -> ResultRecord:
    return make_record(
        method,
        seed,
        errors,
        n_docs=IMDB_DOCS,
        dataset=DatasetName.IMDB,
        input_setting=setting,
        token_lengths=IMDB_LENGTHS,
        **kwargs,  # type: ignore[arg-type]
    )


def h5a_records(
    l512_errors: Callable[[int], set[int]], lfull_errors: Callable[[int], set[int]]
) -> list[ResultRecord]:
    records = []
    for seed in range(3):
        records.append(imdb_record(FusionMethod.F2, seed, InputSetting.L512, l512_errors(seed)))
        records.append(imdb_record(FusionMethod.F2, seed, InputSetting.LFULL, lfull_errors(seed)))
    return records


def test_h5a_supported_when_the_full_input_beats_l512() -> None:
    # The gain sits entirely in the long documents (index 100+): 20 fewer errors per seed.
    records = h5a_records(
        lambda s: set(range(100, 160 + s)), lambda s: set(range(100, 140 + s))
    )

    h5a = judge_hypotheses(records, make_rules()).judgments["H5a"]

    assert h5a.kind == "confirmatory"
    assert h5a.verdict == Verdict.SUPPORTED
    # L512 mean error 61/200 = .305, Lfull 41/200 = .205 -> improvement 0.10, seed std 0.005
    assert h5a.mean_diff == pytest.approx(0.10)
    assert h5a.seed_std == pytest.approx(0.005)
    assert h5a.mcnemar_p_value is not None and h5a.mcnemar_p_value < 0.05


def test_h5a_is_no_effect_when_the_full_input_is_no_better() -> None:
    same = lambda s: set(range(100, 150 + s))  # noqa: E731
    h5a = judge_hypotheses(h5a_records(same, same), make_rules()).judgments["H5a"]

    assert h5a.mean_diff == pytest.approx(0.0)
    assert h5a.verdict == Verdict.NO_EFFECT


def test_h5a_is_left_out_without_both_input_settings() -> None:
    both = h5a_records(lambda s: {1}, lambda s: {1})
    only_l512 = [r for r in both if r.config.input_setting is InputSetting.L512]

    assert "H5a" not in judge_hypotheses(only_l512, make_rules()).judgments


def test_h5a_target_rule_must_be_set_once_imdb_records_exist() -> None:
    records = h5a_records(lambda s: {1}, lambda s: {1})

    with pytest.raises(ValueError, match="h5a_target_fusion_rule is TBD"):
        judge_hypotheses(records, make_rules(h5a_target_fusion_rule=None))


def test_h5a_still_refuses_records_that_differ_beyond_the_input_setting() -> None:
    records = h5a_records(lambda s: {1}, lambda s: {1})
    records[-1] = replace(records[-1], config=replace(records[-1].config, epochs=9))

    with pytest.raises(ValueError, match="one experiment cell.*epochs"):
        judge_hypotheses(records, make_rules())


# --- H2: is the fusion effect larger with small data than with full data? -------------------


def amount_records(
    errors: dict[tuple[DataAmount, FusionMethod], Callable[[int], set[int]]],
    *,
    backbone_of: Callable[[DataAmount], BackboneCondition] = lambda amount: (
        BackboneCondition.FROZEN
    ),
) -> list[ResultRecord]:
    """200 MNLI docs, 3 seeds, one record per (data amount, method) in ``errors``."""
    return [
        make_record(
            method, seed, errors_of(seed), n_docs=200, amount=amount, backbone=backbone_of(amount)
        )
        for (amount, method), errors_of in errors.items()
        for seed in range(3)
    ]


SMALL, FULL = DataAmount.SMALL, DataAmount.FULL
B0, F1 = FusionMethod.B0, FusionMethod.F1


def test_h2_supported_when_the_effect_is_larger_with_small_data() -> None:
    # Small data: B0 errs on 90+s docs, F1 on 30+2s (effect .30 - s/200 per seed).
    # Full data:  B0 errs on 60+s docs, F1 on 40+s  (effect .10).
    records = amount_records(
        {
            (SMALL, B0): lambda s: set(range(90 + s)),
            (SMALL, F1): lambda s: set(range(30 + 2 * s)),
            (FULL, B0): lambda s: set(range(60 + s)),
            (FULL, F1): lambda s: set(range(40 + s)),
        }
    )

    h2 = judge_hypotheses(records, make_rules()).judgments["H2"]

    assert h2.kind == "confirmatory"
    assert h2.verdict == Verdict.SUPPORTED
    # small effect mean (60-1)/200 = .295 ... per seed (60 - s)/200; full effect .10 every seed
    assert h2.mean_diff == pytest.approx(0.195)
    assert h2.seed_std == pytest.approx(0.005)
    assert h2.mcnemar_p_value is not None and h2.mcnemar_p_value < 0.05


def test_h2_is_no_effect_when_the_effect_is_not_larger_with_small_data() -> None:
    # The same four settings with the roles swapped: the full-data effect is the big one.
    records = amount_records(
        {
            (FULL, B0): lambda s: set(range(90 + s)),
            (FULL, F1): lambda s: set(range(30 + 2 * s)),
            (SMALL, B0): lambda s: set(range(60 + s)),
            (SMALL, F1): lambda s: set(range(40 + s)),
        }
    )

    h2 = judge_hypotheses(records, make_rules()).judgments["H2"]

    assert h2.mean_diff == pytest.approx(-0.195)
    assert h2.mcnemar_p_value == 1.0  # no seed has the small-data effect ahead
    assert h2.verdict == Verdict.NO_EFFECT


def test_h2_is_no_effect_when_the_two_effects_differ_by_less_than_the_seed_std() -> None:
    # Small-data effect per seed .30 / .45 / .35 (B0 errs 90 / 120 / 100 docs, F1 always 30),
    # full-data effect .30 every seed: gap .067 but the small-data effect itself swings with
    # std .076. Seeds 1 and 2 are significant, so with majority aggregation only the seed-std
    # rule can say NO_EFFECT here.
    small_b0_errors = (90, 120, 100)
    records = amount_records(
        {
            (SMALL, B0): lambda s: set(range(small_b0_errors[s])),
            (SMALL, F1): lambda s: set(range(30)),
            (FULL, B0): lambda s: set(range(90 + s)),
            (FULL, F1): lambda s: set(range(30 + s)),
        }
    )
    rules = make_rules(multi_seed_mcnemar_aggregation="majority_of_seeds")

    h2 = judge_hypotheses(records, rules).judgments["H2"]

    assert h2.mean_diff == pytest.approx(0.0667, abs=1e-3)
    assert h2.seed_std == pytest.approx(0.0764, abs=1e-3)
    assert h2.mcnemar_p_value is not None and h2.mcnemar_p_value < 0.05
    assert h2.verdict == Verdict.NO_EFFECT


def test_h2_is_left_out_without_small_data_records() -> None:
    records = amount_records(
        {(FULL, B0): lambda s: {1}, (FULL, F1): lambda s: {2}},
    )

    assert "H2" not in judge_hypotheses(records, make_rules()).judgments


def test_h2_uses_the_h1_target_method() -> None:
    """With a different target (F2), F1 records alone do not make an H2 row."""
    records = amount_records(
        {
            (SMALL, B0): lambda s: set(range(90 + s)),
            (SMALL, F1): lambda s: set(range(30 + s)),
            (FULL, B0): lambda s: set(range(60 + s)),
            (FULL, F1): lambda s: set(range(40 + s)),
        }
    )

    judgments = judge_hypotheses(records, make_rules(h1_target_fusion_rule="F2")).judgments

    assert "H2" not in judgments


def test_a_tbd_h5a_rule_is_not_demanded_before_both_input_settings_have_records() -> None:
    """One input setting is nothing to compare, so the (still TBD) target rule stays unasked."""
    lfull_only = [
        imdb_record(FusionMethod.F2, seed, InputSetting.LFULL, {1}) for seed in range(3)
    ]

    judgments = judge_hypotheses(lfull_only, make_rules(h5a_target_fusion_rule=None)).judgments

    assert "H5a" not in judgments


# --- H3: does fine-tuning the backbone shrink the fusion effect? ----------------------------

FROZEN, FINETUNE = BackboneCondition.FROZEN, BackboneCondition.FINETUNE


def backbone_records(
    errors: dict[tuple[BackboneCondition, FusionMethod], Callable[[int], set[int]]],
    *,
    seeds_of: Callable[[BackboneCondition], Iterable[int]] = lambda backbone: range(3),
) -> list[ResultRecord]:
    return [
        make_record(method, seed, errors_of(seed), n_docs=200, backbone=backbone)
        for (backbone, method), errors_of in errors.items()
        for seed in seeds_of(backbone)
    ]


def shrinking_effect_errors(
    finetune_methods: tuple[FusionMethod, ...] = (B0, F1),
) -> dict[tuple[BackboneCondition, FusionMethod], Callable[[int], set[int]]]:
    """Frozen: B0 90+s vs F1 30+2s errors (effect .295). Fine-tuned: 60+s vs 50+s (.05)."""
    errors: dict[tuple[BackboneCondition, FusionMethod], Callable[[int], set[int]]] = {
        (FROZEN, B0): lambda s: set(range(90 + s)),
        (FROZEN, F1): lambda s: set(range(30 + 2 * s)),
        (FINETUNE, B0): lambda s: set(range(60 + s)),
    }
    for method in finetune_methods:
        if method is not B0:
            errors[FINETUNE, method] = lambda s: set(range(50 + s))
    return errors


def test_h3_supported_when_fine_tuning_shrinks_the_effect() -> None:
    h3 = judge_hypotheses(backbone_records(shrinking_effect_errors()), make_rules()).judgments["H3"]

    assert h3.kind == "confirmatory"
    assert h3.verdict == Verdict.SUPPORTED
    # frozen effect (60-s)/200 -> mean .295; fine-tuned effect 10/200 = .05 -> gap .245
    assert h3.mean_diff == pytest.approx(0.245)
    assert h3.seed_std == pytest.approx(0.005)
    assert h3.mcnemar_p_value is not None and h3.mcnemar_p_value < 0.05
    assert h3.details["effect_mean"] == pytest.approx({"frozen": 0.295, "finetune": 0.05})


def test_h3_is_no_effect_when_fine_tuning_does_not_shrink_the_effect() -> None:
    errors = shrinking_effect_errors()
    # Swap the two conditions: now fine-tuning is where F1 helps most.
    swapped = {
        (FINETUNE if backbone is FROZEN else FROZEN, method): fn
        for (backbone, method), fn in errors.items()
    }

    h3 = judge_hypotheses(backbone_records(swapped), make_rules()).judgments["H3"]

    assert h3.mean_diff == pytest.approx(-0.245)
    assert h3.mcnemar_p_value == 1.0
    assert h3.verdict == Verdict.NO_EFFECT


def test_h3_takes_its_candidate_from_the_fine_tuned_records() -> None:
    """The fine-tune condition runs B0 and the best method; whichever that is gets judged."""
    errors = shrinking_effect_errors(finetune_methods=(B0, FusionMethod.F2))
    errors[FROZEN, FusionMethod.F2] = errors.pop((FROZEN, F1))

    h3 = judge_hypotheses(backbone_records(errors), make_rules()).judgments["H3"]

    assert h3.verdict == Verdict.SUPPORTED


def test_h3_is_left_out_without_fine_tuned_records_or_their_frozen_counterpart() -> None:
    frozen_only = {k: v for k, v in shrinking_effect_errors().items() if k[0] is FROZEN}
    assert "H3" not in judge_hypotheses(backbone_records(frozen_only), make_rules()).judgments

    no_frozen_f1 = {k: v for k, v in shrinking_effect_errors().items() if k != (FROZEN, F1)}
    assert "H3" not in judge_hypotheses(backbone_records(no_frozen_f1), make_rules()).judgments


def test_h3_refuses_more_than_one_fine_tuned_candidate() -> None:
    errors = shrinking_effect_errors(finetune_methods=(B0, F1, FusionMethod.F2))

    with pytest.raises(ValueError, match="one fusion method besides B0"):
        judge_hypotheses(backbone_records(errors), make_rules())


def test_h3_judges_on_the_common_seeds_when_the_conditions_ran_different_seed_counts() -> None:
    """T-07a: 5 frozen seeds (core) vs 3 fine-tuned seeds -> judged on the common seeds 0-2."""
    records = backbone_records(
        shrinking_effect_errors(),
        seeds_of=lambda backbone: range(5) if backbone is FROZEN else range(3),
    )

    h3 = judge_hypotheses(records, make_rules()).judgments["H3"]

    assert h3.verdict == Verdict.SUPPORTED
    # Seeds 0-2 only: frozen effect (60-s)/200 -> .295, fine-tuned .05, as with 3 frozen seeds.
    assert h3.mean_diff == pytest.approx(0.245)
    assert h3.details["n_seeds"] == 3
    assert h3.details["seeds"] == [0, 1, 2]
    assert h3.details["excluded_seeds"] == [3, 4]


def test_h3_with_matching_seed_sets_excludes_no_seed() -> None:
    h3 = judge_hypotheses(backbone_records(shrinking_effect_errors()), make_rules()).judgments["H3"]

    assert h3.details["seeds"] == [0, 1, 2]
    assert h3.details["excluded_seeds"] == []


def test_h3_refuses_fewer_than_two_common_seeds_between_the_backbone_conditions() -> None:
    # Frozen seeds 0-2 and fine-tuned seeds 2-4 share only seed 2: no seed std, no judgment.
    records = backbone_records(
        shrinking_effect_errors(),
        seeds_of=lambda backbone: range(3) if backbone is FROZEN else range(2, 5),
    )

    with pytest.raises(ValueError, match="at least 2 common seeds"):
        judge_hypotheses(records, make_rules())


def test_h2_judges_on_the_common_seeds_of_the_small_and_full_data_runs() -> None:
    """T-07a: 3 small-data seeds (secondary) vs 5 full-data seeds (core) -> common seeds 0-2."""
    errors = {
        (SMALL, B0): lambda s: set(range(90 + s)),
        (SMALL, F1): lambda s: set(range(30 + 2 * s)),
        (FULL, B0): lambda s: set(range(60 + s)),
        (FULL, F1): lambda s: set(range(40 + s)),
    }
    records = [
        make_record(method, seed, errors_of(seed), n_docs=200, amount=amount)
        for (amount, method), errors_of in errors.items()
        for seed in (range(3) if amount is SMALL else range(5))
    ]

    h2 = judge_hypotheses(records, make_rules()).judgments["H2"]

    assert h2.verdict == Verdict.SUPPORTED
    assert h2.mean_diff == pytest.approx(0.195)  # same hand-worked numbers as the 3-seed case
    assert h2.details["seeds"] == [0, 1, 2]
    assert h2.details["excluded_seeds"] == [3, 4]


# --- H5b: judged with F2 (F1 is auxiliary) -----------------------------------------------


def h5b_records(
    errors: dict[FusionMethod, tuple[Callable[[int], set[int]], Callable[[int], set[int]]]],
) -> list[ResultRecord]:
    """IMDB records per method: (L512 errors, Lfull errors) as functions of the seed."""
    return [
        imdb_record(method, seed, setting, fn(seed))
        for method, (l512, lfull) in errors.items()
        for seed in range(3)
        for setting, fn in ((InputSetting.L512, l512), (InputSetting.LFULL, lfull))
    ]


def test_h5b_decomposition_is_absent_when_f1_runs_are_missing() -> None:
    """F1 is the auxiliary arm: without it H5b is still judged, just not decomposed."""
    records = h5b_records(
        {
            B0: (lambda s: set(range(100, 180 + s)), lambda s: set(range(100, 170))),
            FusionMethod.F2: (lambda s: set(range(100, 160 + s)), lambda s: set(range(100, 130))),
        }
    )

    h5b = judge_hypotheses(records, make_rules()).judgments["H5b"]

    assert h5b.verdict == Verdict.SUPPORTED
    assert h5b.details["decomposition"] is None


def test_h5b_is_no_effect_when_the_longer_input_helps_b0_as_much_as_f2() -> None:
    same = (lambda s: set(range(100, 160 + s)), lambda s: set(range(100, 130)))
    records = h5b_records({B0: same, FusionMethod.F2: same})

    h5b = judge_hypotheses(records, make_rules()).judgments["H5b"]

    assert h5b.mean_diff == pytest.approx(0.0)
    assert h5b.mcnemar_p_value == 1.0
    assert h5b.verdict == Verdict.NO_EFFECT


def test_h5b_is_left_out_without_both_arms_at_both_input_settings() -> None:
    records = h5b_records(
        {FusionMethod.F2: (lambda s: set(range(100, 160)), lambda s: set(range(100, 130)))}
    )

    assert "H5b" not in judge_hypotheses(records, make_rules()).judgments


def test_h5b_refuses_f1_runs_from_a_different_experiment_cell() -> None:
    records = h5b_records(
        {
            B0: (lambda s: set(range(100, 180 + s)), lambda s: set(range(100, 170))),
            FusionMethod.F1: (lambda s: set(range(100, 170 + s)), lambda s: set(range(100, 150))),
            FusionMethod.F2: (lambda s: set(range(100, 160 + s)), lambda s: set(range(100, 130))),
        }
    )
    stray = next(r for r in records if r.config.fusion_method is FusionMethod.F1)
    records[records.index(stray)] = replace(stray, config=replace(stray.config, epochs=9))

    with pytest.raises(ValueError, match="one experiment cell.*epochs"):
        judge_hypotheses(records, make_rules())


# --- Length buckets and the <=512 sanity check (SPEC US 20, 66, 67) -------------------------

L512_CELL = "imdb/full/frozen/L512/F2/all"


def test_length_bucket_analysis_reports_counts_and_error_rates_per_bucket() -> None:
    records = h5a_records(
        lambda s: set(range(100, 160 + s)), lambda s: set(range(100, 140 + s))
    )

    buckets = judge_hypotheses(records, make_rules()).length_bucket_analysis["buckets"][L512_CELL]

    assert {name: row["n_docs"] for name, row in buckets.items()} == {
        "<=512": 100,
        "512-1024": 50,
        ">1024": 50,
    }
    assert buckets["<=512"]["error_rate_mean"] == 0.0  # every error sits in the long documents
    assert buckets["512-1024"]["error_rate_mean"] == pytest.approx(1.0)  # docs 100-149 all wrong
    # >1024 = docs 150-199; errors are docs 150..159+s -> (10 + s) / 50, mean over s = 0,1,2
    assert buckets[">1024"]["error_rate_mean"] == pytest.approx(11 / 50)
    assert buckets[">1024"]["error_rate_std"] == pytest.approx(0.02)  # 1/50 per seed step
    assert buckets[">1024"]["n_seeds"] == 3


def test_below_min_samples_follows_the_rule_and_stays_unset_while_it_is_tbd() -> None:
    records = h5a_records(lambda s: {150}, lambda s: {150})

    undecided = judge_hypotheses(
        records, make_rules(min_bucket_samples=None)
    ).length_bucket_analysis["buckets"]
    decided = judge_hypotheses(records, make_rules(min_bucket_samples=60)).length_bucket_analysis[
        "buckets"
    ]

    assert {row["below_min_samples"] for row in undecided[L512_CELL].values()} == {None}
    assert {name: row["below_min_samples"] for name, row in decided[L512_CELL].items()} == {
        "<=512": False,
        "512-1024": True,
        ">1024": True,
    }


def test_le512_check_reports_the_short_documents_of_l512_and_lfull() -> None:
    # Both settings err on the same 10 short documents in seed 0; Lfull gets 2 more wrong.
    records = h5a_records(
        lambda s: set(range(10)) | set(range(100, 150)),
        lambda s: set(range(12)) | set(range(100, 120)),
    )

    check = judge_hypotheses(records, make_rules()).length_bucket_analysis["le512_check"]["F2"]

    assert check["same_documents"] is True
    assert check["n_docs"] == 100
    assert check["error_rate"] == pytest.approx({"L512": 0.10, "Lfull": 0.12})
    assert check["error_rate_gap"] == pytest.approx(-0.02)  # L512 minus Lfull


def test_le512_check_refuses_runs_whose_short_documents_differ() -> None:
    """Doc 0 looks 600 tokens long under Lfull but 300 under L512: the inputs are not the same."""
    records = h5a_records(lambda s: {1}, lambda s: {1})
    lfull = next(r for r in records if r.config.input_setting is InputSetting.LFULL)
    longer = [
        replace(p, token_length=600, length_bucket=assign_length_bucket(600))
        if p.doc_id == "imdb-00000"
        else p
        for p in lfull.predictions
    ]
    records[records.index(lfull)] = replace(lfull, predictions=longer)

    with pytest.raises(ValueError, match="short documents"):
        judge_hypotheses(records, make_rules())


# --- F4: does the document-level global-layer share follow the document length? -------------

IMDB_F4_CELL = "imdb/full/frozen/Lfull/F4/all"
F4_RULES = {"include_embedding_in_global_denom": False}
# Global share of label-0 docs by length. Powers of two keep 1 - share exact, so the .25s of
# both labels tie exactly (0.2 and 1 - 0.8 would differ in the last bit and break the ties).
_SHARE_BY_LENGTH = {300: 0.25, 700: 0.5, 1500: 0.75}


def f4_document_weights(doc: int) -> list[float]:
    """Slot 22 (a global layer) gets the share r, slot 21 (a local layer) the rest.

    Label 0 (even docs) leans on the global layer more as documents get longer, label 1 (odd
    docs) the other way round.
    """
    share = _SHARE_BY_LENGTH[IMDB_LENGTHS[doc]]
    share = share if doc % 2 == 0 else 1.0 - share
    weights = [0.0] * 23
    weights[22] = share
    weights[21] = 1.0 - share
    return weights


def f4_records(**kwargs: object) -> list[ResultRecord]:
    return [
        imdb_record(
            FusionMethod.F4,
            seed,
            InputSetting.LFULL,
            {1},
            doc_layer_weights=f4_document_weights,
            **kwargs,
        )
        for seed in range(3)
    ]


def test_f4_length_correlation_is_reported_overall_and_per_label() -> None:
    analysis = judge_hypotheses(f4_records(), make_rules(**F4_RULES)).layer_weight_analysis

    corr = analysis[IMDB_F4_CELL]["length_global_ratio_corr"]

    assert corr["label_0"] == pytest.approx(1.0)  # longer -> more global weight
    assert corr["label_1"] == pytest.approx(-1.0)  # longer -> less global weight
    # Each length group holds the same mix of .25 and .75 shares (and 700 is all .5), so the two
    # labels cancel exactly in the pooled correlation - why the per-label split matters.
    assert corr["all"] == pytest.approx(0.0)
    assert analysis[IMDB_F4_CELL]["n_seeds"] == 3
    assert analysis[IMDB_F4_CELL]["n_docs"] == IMDB_DOCS


def test_f4_cells_without_document_weights_are_not_analysed() -> None:
    records = [replace(rec, predictions=[replace(p, layer_weights=None) for p in rec.predictions])
               for rec in f4_records()]

    assert IMDB_F4_CELL not in judge_hypotheses(records, make_rules()).layer_weight_analysis


def test_f4_correlation_refuses_a_document_missing_its_weights_or_length() -> None:
    records = f4_records()
    first = records[0]
    broken = [replace(first.predictions[0], layer_weights=None), *first.predictions[1:]]
    records[0] = replace(first, predictions=broken)

    with pytest.raises(ValueError, match="layer_weights"):
        judge_hypotheses(records, make_rules(**F4_RULES))


def test_a_constant_length_gives_no_correlation_rather_than_a_fake_zero() -> None:
    """Spearman is undefined when every document has the same length."""
    records = [
        replace(
            rec,
            predictions=[replace(p, token_length=300, length_bucket=None) for p in rec.predictions],
        )
        for rec in f4_records()
    ]

    corr = judge_hypotheses(records, make_rules(**F4_RULES)).layer_weight_analysis[IMDB_F4_CELL][
        "length_global_ratio_corr"
    ]

    assert corr == {"all": None, "label_0": None, "label_1": None}


# --- Exploratory H4 / H5c: evidence goes into details, the verdict stays EXPLORATORY --------

EXPLORATORY_RULES = {
    "include_embedding_in_cog": False,
    "include_embedding_in_global_denom": False,
    "seed_rank_corr_min": 0.5,
    "cog_seed_std_max": 2.0,
}


def one_hot(slot: int, size: int = 23) -> list[float]:
    weights = [0.0] * size
    weights[slot] = 1.0
    return weights


def f2_records(
    dataset: DatasetName,
    setting: InputSetting | None,
    slots_by_seed: Sequence[int],
    *,
    errors: int = 20,
    layer_subset: LayerSubset = LayerSubset.ALL,
    size: int = 23,
    n_docs: int = 200,
) -> list[ResultRecord]:
    """F2 records whose whole weight sits on one slot per seed (hand-checkable summaries)."""
    return [
        make_record(
            FusionMethod.F2,
            seed,
            range(errors),
            n_docs=n_docs,
            dataset=dataset,
            input_setting=setting,
            layer_subset=layer_subset,
            layer_weights=one_hot(slot, size),
        )
        for seed, slot in enumerate(slots_by_seed)
    ]


def sst2_pair(error_b0: int, error_f1: int) -> list[ResultRecord]:
    return [
        make_record(method, seed, range(count + seed), n_docs=200, dataset=DatasetName.SST2)
        for method, count in ((B0, error_b0), (F1, error_f1))
        for seed in range(3)
    ]


def test_h4_reports_gravity_and_sst2_comparison_without_a_verdict() -> None:
    records = (
        f2_records(DatasetName.MNLI, None, [20, 20, 20])
        + f2_records(DatasetName.SST2, None, [5, 5, 5])
        + sst2_pair(60, 40)
    )

    h4 = judge_hypotheses(records, make_rules(**EXPLORATORY_RULES)).judgments["H4"]

    assert h4.verdict == Verdict.EXPLORATORY
    assert h4.mean_diff is None and h4.mcnemar_p_value is None and h4.seed_std is None
    assert h4.details["center_of_gravity"] == pytest.approx({"mnli": 20.0, "sst2": 5.0})
    assert h4.details["layer_weight_interpretation"] is None
    sst2 = h4.details["sst2_error_comparison"]  # B0 vs the H1 target (F1) on SST-2
    assert sst2["mean_diff"] == pytest.approx(0.10)
    assert sst2["mcnemar_p_value"] < 0.05
    assert "verdict" not in sst2


def test_h4_holds_the_layer_weight_reading_when_a_dataset_fails_seed_consistency() -> None:
    records = f2_records(DatasetName.MNLI, None, [20, 20, 20]) + f2_records(
        DatasetName.SST2, None, [3, 21, 3]  # the SST-2 centre of gravity swings 3 -> 21 -> 3
    )

    h4 = judge_hypotheses(records, make_rules(**EXPLORATORY_RULES)).judgments["H4"]

    assert h4.verdict == Verdict.EXPLORATORY  # held reading, not a confirmatory verdict
    assert h4.details["layer_weight_interpretation"] == Verdict.HELD


def test_h4_without_sst2_or_f2_records_has_only_its_note() -> None:
    h4 = judge_hypotheses([], make_rules()).judgments["H4"]

    assert h4.verdict == Verdict.EXPLORATORY
    assert h4.details["note"]
    assert h4.details.get("sst2_error_comparison") is None
    assert h4.details.get("center_of_gravity", {}) == {}


def test_h5c_collects_per_setting_summaries_ablation_and_f4_correlation() -> None:
    records = (
        f2_records(DatasetName.IMDB, InputSetting.L512, [8, 8, 8], errors=30)
        + f2_records(DatasetName.IMDB, InputSetting.LFULL, [16, 16, 16], errors=30)
        + f2_records(
            DatasetName.IMDB, InputSetting.LFULL, [0, 0, 0], errors=40,
            layer_subset=LayerSubset.GLOBAL_ONLY, size=8,
        )
        + f2_records(
            DatasetName.IMDB, InputSetting.LFULL, [0, 0, 0], errors=50,
            layer_subset=LayerSubset.LOCAL_ONLY, size=14,
        )
        + f4_records()
    )

    h5c = judge_hypotheses(records, make_rules(**EXPLORATORY_RULES)).judgments["H5c"]

    assert h5c.verdict == Verdict.EXPLORATORY
    assert h5c.mean_diff is None and h5c.mcnemar_p_value is None
    by_setting = h5c.details["by_input_setting"]
    assert by_setting["L512"]["center_of_gravity_mean"] == pytest.approx(8.0)
    assert by_setting["L512"]["global_ratio_mean"] == pytest.approx(0.0)  # layer 8 is local
    assert by_setting["Lfull"]["center_of_gravity_mean"] == pytest.approx(16.0)
    assert by_setting["Lfull"]["global_ratio_mean"] == pytest.approx(1.0)  # layer 16 is global
    assert h5c.details["ablation_error_rate"]["Lfull"] == pytest.approx(
        {"all": 0.15, "global_only": 0.20, "local_only": 0.25}
    )
    assert h5c.details["f4_length_global_ratio_corr"]["label_0"] == pytest.approx(1.0)
    assert h5c.details["layer_weight_interpretation"] is None


# --- Review fixes: auxiliary analyses must not block, or crash, unrelated verdicts ----------


def test_tbd_layer_weight_rules_do_not_block_other_verdicts() -> None:
    """F2 weights with the layer-weight rules still TBD: H1 is judged, the cell says why not."""
    h1_records = [
        make_record(method, seed, range(count + seed), n_docs=200)
        for method, count in ((B0, 60), (F1, 40))
        for seed in range(3)
    ]
    records = h1_records + f2_records(DatasetName.MNLI, None, [20, 20, 20]) + f4_records()

    tbd_layer_weight_rules = make_rules(
        include_embedding_in_cog=None,
        include_embedding_in_global_denom=None,
        seed_rank_corr_min=None,
        cog_seed_std_max=None,
    )

    report = judge_hypotheses(records, tbd_layer_weight_rules)

    assert report.judgments["H1"].verdict == Verdict.SUPPORTED
    cell = report.layer_weight_analysis["mnli/full/frozen/F2/all"]
    assert "include_embedding_in_cog is TBD" in cell["unavailable"]
    assert "center_of_gravity_mean" not in cell
    assert "include_embedding_in_global_denom is TBD" in (
        report.layer_weight_analysis[IMDB_F4_CELL]["unavailable"]
    )
    # The exploratory rows still come out, without numbers they cannot compute.
    assert report.judgments["H4"].verdict == Verdict.EXPLORATORY
    assert report.judgments["H5c"].verdict == Verdict.EXPLORATORY


def test_a_single_seed_f2_cell_cannot_show_consistency_so_it_is_held() -> None:
    records = f2_records(DatasetName.IMDB, InputSetting.LFULL, [16])

    cell = judge_hypotheses(records, make_rules(**EXPLORATORY_RULES)).layer_weight_analysis[
        "imdb/full/frozen/Lfull/F2/all"
    ]

    assert cell["n_seeds"] == 1
    assert cell["seed_consistent"] is False
    assert cell["interpretation"] == Verdict.HELD
    assert cell["cog_seed_std"] is None and cell["rank_corr_mean"] is None
    assert cell["center_of_gravity_mean"] == pytest.approx(16.0)


def test_f4_cells_outside_imdb_are_not_length_analysed() -> None:
    """MNLI F4 runs also carry document weights; they have no token length and no US 72 role."""
    records = [
        make_record(
            FusionMethod.F4,
            seed,
            {1},
            n_docs=200,
            doc_layer_weights=lambda i: one_hot(22),
        )
        for seed in range(3)
    ]

    analysis = judge_hypotheses(records, make_rules(**F4_RULES)).layer_weight_analysis

    assert analysis == {}


def test_h5b_skips_the_decomposition_when_f1_covers_other_seeds() -> None:
    """F1 is the auxiliary arm: seeds that do not line up cost the split, not the F2 verdict."""
    records = h5b_records(
        {
            B0: (lambda s: set(range(100, 180 + s)), lambda s: set(range(100, 170))),
            FusionMethod.F1: (lambda s: set(range(100, 170 + s)), lambda s: set(range(100, 150))),
            FusionMethod.F2: (lambda s: set(range(100, 160 + s)), lambda s: set(range(100, 130))),
        }
    )
    records = [
        r for r in records if not (r.config.fusion_method is F1 and r.config.seed == 2)
    ]

    h5b = judge_hypotheses(records, make_rules()).judgments["H5b"]

    assert h5b.verdict == Verdict.SUPPORTED
    assert h5b.details["decomposition"] is None
    assert "seeds" in h5b.details["decomposition_skipped"]


# --- Cross-review fixes (T-07c, 2nd review) ------------------------------------------------


def test_effect_difference_does_not_count_a_seed_whose_documents_favour_the_other_effect() -> None:
    """Per-document gain differences run from -2 to +2, so the net gap and the document count
    can point in opposite directions.

    Small data: B0 errs on docs 0-14, F1 on none -> gain +1 on docs 0-14 (effect 15/200).
    Full data: B0 errs on docs 15-43, F1 on docs 0-14 -> gain -1 on 0-14, +1 on 15-43 (14/200).
    Net, small wins by 1/200, but 29 documents favour full data against 15 (+2 each) for small:
    the two-sided sign test is significant (p ~ .049) in the direction H2 does not claim.
    """
    records = amount_records(
        {
            (SMALL, B0): lambda s: set(range(15)),
            (SMALL, F1): lambda s: set(),
            (FULL, B0): lambda s: set(range(15, 44)),
            (FULL, F1): lambda s: set(range(15)),
        }
    )

    h2 = judge_hypotheses(records, make_rules(mcnemar_alpha=0.05)).judgments["H2"]

    assert h2.mean_diff == pytest.approx(0.005)  # net gap positive, seed std 0
    assert h2.details["per_seed_sign_test_p"][0] < 0.05  # significant, but the other way
    assert h2.details["per_seed_documents_ahead"][0] == {"small": 15, "full": 29}
    assert h2.mcnemar_p_value == 1.0
    assert h2.verdict == Verdict.NO_EFFECT


def test_h4_sst2_comparison_is_unavailable_while_the_h1_target_is_tbd() -> None:
    """Exploratory evidence never stops the report: the comparison says why it is missing."""
    report = judge_hypotheses(sst2_pair(60, 40), make_rules(h1_target_fusion_rule=None))

    comparison = report.judgments["H4"].details["sst2_error_comparison"]

    assert "h1_target_fusion_rule is TBD" in comparison["unavailable"]


def test_h4_sst2_comparison_needs_no_significance_level() -> None:
    """No verdict is drawn for H4, so alpha (TBD) is not needed for its numbers."""
    report = judge_hypotheses(sst2_pair(60, 40), make_rules(mcnemar_alpha=None))

    comparison = report.judgments["H4"].details["sst2_error_comparison"]

    assert comparison["mean_diff"] == pytest.approx(0.10)
    assert comparison["mcnemar_p_value"] < 0.05


@pytest.mark.parametrize(
    ("summary", "rule"),
    [
        (compute_center_of_gravity, "include_embedding_in_cog"),
        (compute_global_layer_weight_ratio, "include_embedding_in_global_denom"),
    ],
)
def test_layer_weight_summaries_refuse_a_tbd_embedding_rule(
    summary: Callable[..., float], rule: str
) -> None:
    with pytest.raises(ValueError, match=f"{rule} is TBD"):
        summary(one_hot(22), None)


def test_le512_check_of_an_arm_with_unpaired_seeds_is_unavailable_not_fatal() -> None:
    """F1 at L512 has 3 seeds but only 2 at Lfull: the H5b verdict and the F2 check survive."""
    records = h5b_records(
        {
            B0: (lambda s: set(range(100, 180 + s)), lambda s: set(range(100, 170))),
            FusionMethod.F1: (lambda s: set(range(100, 170 + s)), lambda s: set(range(100, 150))),
            FusionMethod.F2: (lambda s: set(range(100, 160 + s)), lambda s: set(range(100, 130))),
        }
    )
    records = [
        r
        for r in records
        if not (
            r.config.fusion_method is F1
            and r.config.input_setting is InputSetting.LFULL
            and r.config.seed == 2
        )
    ]

    report = judge_hypotheses(records, make_rules())

    assert report.judgments["H5b"].verdict == Verdict.SUPPORTED
    checks = report.length_bucket_analysis["le512_check"]
    assert "seeds" in checks["F1"]["unavailable"]
    assert checks["F2"]["same_documents"] is True


def test_f4_length_correlation_leaves_out_l512_runs() -> None:
    """SPEC US 72 asks for Lfull F4 only: under L512 the long documents are cut."""
    records = [
        replace(rec, config=replace(rec.config, input_setting=InputSetting.L512))
        for rec in f4_records()
    ]

    assert judge_hypotheses(records, make_rules(**F4_RULES)).layer_weight_analysis == {}


def test_h3_details_carry_the_mean_error_of_each_arm() -> None:
    """SPEC US 61: the gain against fine-tuning needs the raw error rates, not only the effects."""
    h3 = judge_hypotheses(backbone_records(shrinking_effect_errors()), make_rules()).judgments["H3"]

    errors = h3.details["mean_error"]
    # frozen: B0 (90+s)/200 -> .455, F1 (30+2s)/200 -> .16; fine-tuned: B0 .305, F1 .255
    assert errors["frozen"] == pytest.approx({"reference": 0.455, "improved": 0.16})
    assert errors["finetune"] == pytest.approx({"reference": 0.305, "improved": 0.255})


def test_le512_check_refuses_short_documents_without_a_token_length() -> None:
    """A missing length on both sides is not 'the same length': nothing was checked."""
    records = h5a_records(lambda s: {1}, lambda s: {1})
    records = [
        replace(
            rec,
            predictions=[
                replace(p, token_length=None) if p.doc_id == "imdb-00000" else p
                for p in rec.predictions
            ],
        )
        for rec in records
    ]

    with pytest.raises(ValueError, match="token length"):
        judge_hypotheses(records, make_rules())
