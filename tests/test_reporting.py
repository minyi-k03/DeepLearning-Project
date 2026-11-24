"""Reporting module tests (T-09a): hand-crafted ResultRecords, CPU only.

Expected values are worked by hand from the records built here, never recomputed with the
module's own formulas (tdd skill: no tautological assertions).
"""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import pytest

from layer_fusion.contracts import (
    BackboneCondition,
    DataAmount,
    DatasetName,
    FusionMethod,
    HypothesisJudgment,
    InputSetting,
    JudgmentReport,
    LayerSubset,
    LengthBucket,
    PredictionRecord,
    ResultRecord,
    RunConfig,
    RunKind,
    TrackKind,
    Verdict,
)
from layer_fusion.reporting import generate_report_artifacts, generate_summary_table


def make_record(
    method: FusionMethod = FusionMethod.B0,
    seed: int = 0,
    error_rate: float = 0.10,
    *,
    dataset: DatasetName = DatasetName.MNLI,
    input_setting: InputSetting | None = None,
    params: int = 2307,
    seconds: float = 10.0,
    run_kind: RunKind = RunKind.MAIN,
    track: TrackKind = TrackKind.MAIN,
) -> ResultRecord:
    config = RunConfig(
        dataset=dataset,
        data_amount=DataAmount.FULL,
        fusion_method=method,
        backbone_condition=BackboneCondition.FROZEN,
        seed=seed,
        input_setting=input_setting,
        run_kind=run_kind,
        track=track,
    )
    return ResultRecord(
        run_id=f"{method.value}-{run_kind.value}-{track.value}-s{seed}",
        config=config,
        selected_lr=1e-3,
        lr_tuning_scores={1e-3: 0.0},
        eval_error_rate=error_rate,
        predictions=[],
        trainable_params=params,
        train_seconds=seconds,
    )


def row_for(rows: list[dict[str, object]], method: str) -> dict[str, object]:
    matching = [r for r in rows if r["fusion_method"] == method]
    assert len(matching) == 1, f"expected one {method} row, got {len(matching)}"
    return matching[0]


def test_summary_table_reports_mean_std_params_and_time_per_fusion_method() -> None:
    records = [
        make_record(FusionMethod.B0, 0, 0.10, params=2307, seconds=10.0),
        make_record(FusionMethod.B0, 1, 0.12, params=2307, seconds=12.0),
        make_record(FusionMethod.B0, 2, 0.14, params=2307, seconds=14.0),
        make_record(FusionMethod.F1, 0, 0.08, params=2307, seconds=11.0),
        make_record(FusionMethod.F1, 1, 0.08, params=2307, seconds=13.0),
    ]

    rows = generate_summary_table(records)

    assert len(rows) == 2
    b0 = row_for(rows, "B0")
    assert b0["n_seeds"] == 3
    assert b0["eval_error_rate_mean"] == pytest.approx(0.12)
    assert b0["eval_error_rate_std"] == pytest.approx(0.02)  # sample std (ddof=1) of 0.10/.12/.14
    assert b0["trainable_params"] == 2307
    assert b0["train_seconds_mean"] == pytest.approx(12.0)
    assert b0["train_seconds_std"] == pytest.approx(2.0)
    f1 = row_for(rows, "F1")
    assert f1["n_seeds"] == 2
    assert f1["eval_error_rate_std"] == pytest.approx(0.0)


def test_summary_table_excludes_debug_and_reproduction_records() -> None:
    records = [
        make_record(FusionMethod.B0, 0, 0.10),
        make_record(FusionMethod.B0, 1, 0.20),
        make_record(FusionMethod.B0, 2, 0.90, run_kind=RunKind.DEBUG),
        make_record(FusionMethod.B0, 3, 0.95, track=TrackKind.REPRODUCTION),
        make_record(FusionMethod.F1, 0, 0.50, run_kind=RunKind.DEBUG),
    ]

    rows = generate_summary_table(records)

    assert [r["fusion_method"] for r in rows] == ["B0"]  # F1 had only a debug run
    assert rows[0]["n_seeds"] == 2
    assert rows[0]["eval_error_rate_mean"] == pytest.approx(0.15)


def test_single_seed_cell_has_no_std_instead_of_a_fake_zero() -> None:
    rows = generate_summary_table([make_record(FusionMethod.B0, 0, 0.10)])

    assert rows[0]["n_seeds"] == 1
    assert rows[0]["eval_error_rate_mean"] == pytest.approx(0.10)
    assert rows[0]["eval_error_rate_std"] is None
    assert rows[0]["train_seconds_std"] is None


def test_cells_differing_in_input_setting_or_dataset_are_not_pooled() -> None:
    records = [
        make_record(
            FusionMethod.B0, 0, 0.10, dataset=DatasetName.IMDB, input_setting=InputSetting.L512
        ),
        make_record(
            FusionMethod.B0, 1, 0.10, dataset=DatasetName.IMDB, input_setting=InputSetting.L512
        ),
        make_record(
            FusionMethod.B0, 0, 0.30, dataset=DatasetName.IMDB, input_setting=InputSetting.LFULL
        ),
        make_record(
            FusionMethod.B0, 1, 0.30, dataset=DatasetName.IMDB, input_setting=InputSetting.LFULL
        ),
        make_record(FusionMethod.B0, 0, 0.50, dataset=DatasetName.MNLI),
        make_record(FusionMethod.B0, 1, 0.50, dataset=DatasetName.MNLI),
    ]

    rows = generate_summary_table(records)

    by_cell = {(r["dataset"], r["input_setting"]): r["eval_error_rate_mean"] for r in rows}
    assert by_cell == pytest.approx(
        {("imdb", "L512"): 0.10, ("imdb", "Lfull"): 0.30, ("mnli", ""): 0.50}
    )


def test_trainable_params_that_differ_inside_a_cell_are_refused() -> None:
    records = [
        make_record(FusionMethod.F1, 0, 0.10, params=2307),
        make_record(FusionMethod.F1, 1, 0.10, params=2308),
    ]

    with pytest.raises(ValueError, match="trainable_params"):
        generate_summary_table(records)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def artifact(paths: list[Path], name: str) -> Path:
    matching = [p for p in paths if p.name == name]
    assert len(matching) == 1, f"expected one {name} in {[p.name for p in paths]}"
    return matching[0]


def test_artifacts_write_summary_csv_without_debug_or_reproduction_records(
    tmp_path: Path,
) -> None:
    records = [
        make_record(FusionMethod.B0, 0, 0.10),
        make_record(FusionMethod.B0, 1, 0.20),
        make_record(FusionMethod.B0, 2, 0.90, run_kind=RunKind.DEBUG),
        make_record(FusionMethod.B0, 3, 0.95, track=TrackKind.REPRODUCTION),
    ]

    paths = generate_report_artifacts(records, JudgmentReport(judgments={}), tmp_path)

    rows = read_csv(artifact(paths, "summary_table.csv"))
    assert len(rows) == 1
    assert rows[0]["fusion_method"] == "B0"
    assert rows[0]["n_seeds"] == "2"
    assert float(rows[0]["eval_error_rate_mean"]) == pytest.approx(0.15)
    assert rows[0]["trainable_params"] == "2307"
    assert all(p.is_file() and p.parent == tmp_path for p in paths)


# Hand-set IMDB eval documents: (token_length, bucket). Buckets are written out literally
# (<=512 / 512-1024 / >1024) so the test does not lean on assign_length_bucket.
IMDB_DOCS = [
    (100, LengthBucket.LE_512),
    (400, LengthBucket.LE_512),
    (600, LengthBucket.BETWEEN_512_1024),
    (900, LengthBucket.BETWEEN_512_1024),
    (1100, LengthBucket.GT_1024),
    (2000, LengthBucket.GT_1024),
]


def make_imdb_record(
    method: FusionMethod,
    seed: int,
    wrong_docs: set[int],
    *,
    input_setting: InputSetting = InputSetting.L512,
    docs: list[tuple[int, LengthBucket]] = IMDB_DOCS,
) -> ResultRecord:
    record = make_record(
        method,
        seed,
        len(wrong_docs) / len(docs),
        dataset=DatasetName.IMDB,
        input_setting=input_setting,
    )
    record.predictions = [
        PredictionRecord(
            doc_id=f"imdb-{i:03d}",
            label=1,
            prediction=0 if i in wrong_docs else 1,
            token_length=length,
            length_bucket=bucket,
        )
        for i, (length, bucket) in enumerate(docs)
    ]
    return record


def test_length_bucket_table_reports_doc_counts_and_error_rate_per_bucket(
    tmp_path: Path,
) -> None:
    records = [
        make_imdb_record(FusionMethod.F1, 0, {0, 4}),  # wrong: 1 of 2 in <=512, 1 of 2 in >1024
        make_imdb_record(FusionMethod.F1, 1, set()),
    ]

    paths = generate_report_artifacts(records, JudgmentReport(judgments={}), tmp_path)

    rows = read_csv(artifact(paths, "length_bucket_table.csv"))
    by_bucket = {r["length_bucket"]: r for r in rows}
    assert list(by_bucket) == ["<=512", "512-1024", ">1024"]  # report order, not alphabetical
    assert {b: int(r["n_docs"]) for b, r in by_bucket.items()} == {
        "<=512": 2,
        "512-1024": 2,
        ">1024": 2,
    }
    assert float(by_bucket["<=512"]["error_rate_mean"]) == pytest.approx(0.25)
    assert float(by_bucket["<=512"]["error_rate_std"]) == pytest.approx(0.35355339)
    assert float(by_bucket["512-1024"]["error_rate_mean"]) == pytest.approx(0.0)
    assert by_bucket["<=512"]["fusion_method"] == "F1"
    assert by_bucket["<=512"]["input_setting"] == "L512"
    assert by_bucket["<=512"]["data_amount"] == "full"  # same cell columns as the summary
    assert by_bucket["<=512"]["backbone_condition"] == "frozen"


def test_below_min_sample_flag_is_left_blank_while_the_threshold_is_tbd(tmp_path: Path) -> None:
    records = [make_imdb_record(FusionMethod.F1, s, set()) for s in range(2)]

    paths = generate_report_artifacts(records, JudgmentReport(judgments={}), tmp_path)

    rows = read_csv(artifact(paths, "length_bucket_table.csv"))
    assert {r["below_min_samples"] for r in rows} == {""}  # SPEC: threshold undecided, not guessed


@pytest.mark.parametrize(("threshold", "expected"), [(3, "True"), (2, "False")])
def test_below_min_sample_flag_marks_buckets_smaller_than_the_threshold(
    tmp_path: Path, threshold: int, expected: str
) -> None:
    records = [make_imdb_record(FusionMethod.F1, s, set()) for s in range(2)]  # 2 docs / bucket

    paths = generate_report_artifacts(
        records, JudgmentReport(judgments={}), tmp_path, min_bucket_samples=threshold
    )

    rows = read_csv(artifact(paths, "length_bucket_table.csv"))
    assert {r["below_min_samples"] for r in rows} == {expected}


def test_seeds_that_disagree_on_bucket_doc_counts_are_refused(tmp_path: Path) -> None:
    shorter = IMDB_DOCS[:-1]  # drops one >1024 doc
    records = [
        make_imdb_record(FusionMethod.F1, 0, set()),
        make_imdb_record(FusionMethod.F1, 1, set(), docs=shorter),
    ]

    with pytest.raises(ValueError, match="n_docs"):
        generate_report_artifacts(records, JudgmentReport(judgments={}), tmp_path)


def test_no_length_bucket_file_when_no_record_has_length_buckets(tmp_path: Path) -> None:
    records = [make_record(FusionMethod.B0, s, 0.1) for s in range(2)]  # MNLI: no buckets

    paths = generate_report_artifacts(records, JudgmentReport(judgments={}), tmp_path)

    assert "length_bucket_table.csv" not in {p.name for p in paths}
    assert not (tmp_path / "length_bucket_table.csv").exists()


def test_view_ratio_table_uses_the_model_input_length_per_input_setting(tmp_path: Path) -> None:
    """View ratio follows the truncated input (L128 -> 128 tokens), counted once per document.

    Hand-derived: a local token sees 129 tokens, fewer at the edges, so for input length
    L > 128 the visible-token total is 2 * (65 + ... + 128) + 129 * (L - 128)
    = 12352 + 129 * (L - 128), and the ratio is that total / L**2.
    For L = 100 the total is 2970 + 2800 + 2970 = 8740
    (tokens 0-35 see 65..100, tokens 36-63 see all 100, tokens 64-99 see 100..65).
    """
    l128_docs = [
        (300, LengthBucket.LE_512),
        (600, LengthBucket.BETWEEN_512_1024),
        (2000, LengthBucket.GT_1024),
    ]
    l512_docs = [(100, LengthBucket.LE_512), (1100, LengthBucket.GT_1024)]
    lfull_docs = [(1100, LengthBucket.GT_1024)]
    records = [
        make_imdb_record(
            FusionMethod.B0, 0, set(), input_setting=InputSetting.L128, docs=l128_docs
        ),
        make_imdb_record(
            FusionMethod.F1, 0, set(), input_setting=InputSetting.L128, docs=l128_docs
        ),
        make_imdb_record(
            FusionMethod.B0, 0, set(), input_setting=InputSetting.L512, docs=l512_docs
        ),
        make_imdb_record(
            FusionMethod.B0, 1, set(), input_setting=InputSetting.L512, docs=l512_docs
        ),
        make_imdb_record(
            FusionMethod.B0, 0, set(), input_setting=InputSetting.LFULL, docs=lfull_docs
        ),
    ]

    paths = generate_report_artifacts(records, JudgmentReport(judgments={}), tmp_path)

    rows = read_csv(artifact(paths, "view_ratio_table.csv"))
    table = {(r["input_setting"], r["length_bucket"]): r for r in rows}
    assert float(table["L128", "all"]["mean_view_ratio"]) == pytest.approx(0.754, abs=1e-3)
    assert float(table["L128", ">1024"]["mean_view_ratio"]) == pytest.approx(0.754, abs=1e-3)
    assert int(table["L128", "all"]["n_docs"]) == 3  # two methods, same docs: counted once
    assert float(table["L512", "<=512"]["mean_view_ratio"]) == pytest.approx(8740 / 100**2)
    assert float(table["L512", ">1024"]["mean_view_ratio"]) == pytest.approx(61888 / 512**2)
    assert float(table["L512", "all"]["mean_view_ratio"]) == pytest.approx(
        (8740 / 100**2 + 61888 / 512**2) / 2
    )
    assert int(table["L512", "all"]["n_docs"]) == 2  # two seeds: counted once
    assert float(table["Lfull", ">1024"]["mean_view_ratio"]) == pytest.approx(137740 / 1100**2)


PNG_SIGNATURE = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])


def make_f2_record(
    seed: int,
    weights: list[float] | None,
    *,
    subset: LayerSubset = LayerSubset.GLOBAL_ONLY,
    run_kind: RunKind = RunKind.MAIN,
) -> ResultRecord:
    record = make_record(FusionMethod.F2, seed, 0.1, run_kind=run_kind)
    record.config = replace(record.config, layer_subset=subset)
    record.layer_weights = weights
    return record


def test_layer_weight_table_gives_seed_mean_and_std_per_layer_and_a_figure(
    tmp_path: Path,
) -> None:
    pytest.importorskip("matplotlib")
    # Global-only F2 has 8 weights, one per global layer 1, 4, 7, 10, 13, 16, 19, 22.
    records = [
        make_f2_record(0, [0.125] * 8),
        make_f2_record(1, [0.5, 0.5, 0, 0, 0, 0, 0, 0]),
        make_f2_record(2, [9.0] * 8, run_kind=RunKind.DEBUG),  # must not enter the mean
        make_record(FusionMethod.F1, 0, 0.1),  # no layer weights: not part of this table
    ]

    paths = generate_report_artifacts(records, JudgmentReport(judgments={}), tmp_path)

    rows = read_csv(artifact(paths, "layer_weights_table.csv"))
    assert [r["layer"] for r in rows] == ["1", "4", "7", "10", "13", "16", "19", "22"]
    assert {r["fusion_method"] for r in rows} == {"F2"}
    assert float(rows[0]["weight_mean"]) == pytest.approx(0.3125)  # (0.125 + 0.5) / 2
    assert float(rows[0]["weight_std"]) == pytest.approx(0.26516504)  # |0.5 - 0.125| / sqrt(2)
    assert float(rows[7]["weight_mean"]) == pytest.approx(0.0625)  # (0.125 + 0) / 2
    assert artifact(paths, "layer_weights.png").read_bytes().startswith(PNG_SIGNATURE)


def test_all_layers_f2_labels_the_embedding_slot_instead_of_calling_it_layer_zero(
    tmp_path: Path,
) -> None:
    pytest.importorskip("matplotlib")
    weights = [1 / 23] * 23
    records = [make_f2_record(s, weights, subset=LayerSubset.ALL) for s in range(2)]

    paths = generate_report_artifacts(records, JudgmentReport(judgments={}), tmp_path)

    rows = read_csv(artifact(paths, "layer_weights_table.csv"))
    assert len(rows) == 23
    assert rows[0]["layer"] == "embedding"  # GLOSSARY: the embedding vector has no layer number
    assert [r["layer"] for r in rows[1:]] == [str(i) for i in range(1, 23)]


def test_layer_weight_count_must_match_the_layer_subset(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    records = [make_f2_record(0, [0.5, 0.5], subset=LayerSubset.GLOBAL_ONLY)]  # needs 8

    with pytest.raises(ValueError, match="layer_weights"):
        generate_report_artifacts(records, JudgmentReport(judgments={}), tmp_path)


def test_no_layer_weight_files_without_layer_weights(tmp_path: Path) -> None:
    records = [make_record(FusionMethod.B0, s, 0.1) for s in range(2)]

    paths = generate_report_artifacts(records, JudgmentReport(judgments={}), tmp_path)

    assert not {"layer_weights_table.csv", "layer_weights.png"} & {p.name for p in paths}


def curve(*tuning_errors: float) -> list[dict[str, float]]:
    return [
        {
            "epoch": float(i + 1),
            "train_loss": 1.0 / (i + 1),
            "tuning_loss": 1.1 / (i + 1),
            "tuning_error": err,
        }
        for i, err in enumerate(tuning_errors)
    ]


def test_learning_curve_figure_is_written_from_main_records(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    records = []
    for method, errors in ((FusionMethod.B0, (0.5, 0.4, 0.3)), (FusionMethod.F1, (0.4, 0.3))):
        for seed in range(2):
            record = make_record(method, seed, 0.3)
            record.learning_curve = curve(*errors)  # seeds may stop at different epochs
            records.append(record)

    paths = generate_report_artifacts(records, JudgmentReport(judgments={}), tmp_path)

    assert artifact(paths, "learning_curves.png").read_bytes().startswith(PNG_SIGNATURE)


def test_no_learning_curve_figure_when_only_debug_records_have_curves(tmp_path: Path) -> None:
    debug = make_record(FusionMethod.B0, 0, 0.3, run_kind=RunKind.DEBUG)
    debug.learning_curve = curve(0.5, 0.4)
    main = make_record(FusionMethod.B0, 1, 0.3)  # main record without a curve

    paths = generate_report_artifacts([debug, main], JudgmentReport(judgments={}), tmp_path)

    assert "learning_curves.png" not in {p.name for p in paths}


def test_hypothesis_table_copies_the_judgment_report_without_judging_again(
    tmp_path: Path,
) -> None:
    judgment = JudgmentReport(
        judgments={
            "H1": HypothesisJudgment("H1", "confirmatory", Verdict.SUPPORTED, 0.02, 0.01, 0.003),
            "H4": HypothesisJudgment("H4", "exploratory", Verdict.EXPLORATORY),
        }
    )

    paths = generate_report_artifacts([], judgment, tmp_path)

    rows = read_csv(artifact(paths, "hypothesis_table.csv"))
    assert [r["hypothesis_id"] for r in rows] == ["H1", "H4"]
    assert rows[0]["kind"] == "confirmatory"
    assert rows[0]["verdict"] == "지지"
    assert float(rows[0]["mean_diff"]) == pytest.approx(0.02)
    assert float(rows[0]["mcnemar_p_value"]) == pytest.approx(0.003)
    assert rows[1]["verdict"] == "탐색적(판정 없음)"
    assert rows[1]["mean_diff"] == ""  # exploratory: no number is invented
