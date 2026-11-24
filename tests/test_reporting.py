"""Reporting module tests (T-09a): hand-crafted ResultRecords, CPU only.

Expected values are worked by hand from the records built here, never recomputed with the
module's own formulas (tdd skill: no tautological assertions).
"""

from __future__ import annotations

import csv
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np
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


def with_config(record: ResultRecord, **changes: object) -> ResultRecord:
    record.config = replace(record.config, **changes)
    return record


def test_two_records_for_the_same_seed_in_one_cell_are_refused() -> None:
    # Pooled, they would give n_seeds=2 and a std of 0 from what is really one seed.
    records = [make_record(FusionMethod.B0, 0, 0.10), make_record(FusionMethod.B0, 0, 0.10)]

    with pytest.raises(ValueError, match="seed 0"):
        generate_summary_table(records)


def test_records_trained_under_different_settings_are_not_pooled_into_one_cell() -> None:
    records = [
        make_record(FusionMethod.B0, 0, 0.10),
        with_config(make_record(FusionMethod.B0, 1, 0.12), epochs=20),
    ]

    with pytest.raises(ValueError, match="epochs"):
        generate_summary_table(records)


def test_seeds_of_one_cell_may_differ_in_the_reused_learning_rate_fields() -> None:
    # SPEC §결정 필요 사항 (T-06a): the first seed selects the LR, the other seeds reuse it.
    records = [
        make_record(FusionMethod.B0, 0, 0.10),
        with_config(make_record(FusionMethod.B0, 1, 0.12), fixed_lr=1e-3, lr_selection_seed=0),
    ]

    rows = generate_summary_table(records)

    assert rows[0]["n_seeds"] == 2


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
@pytest.mark.parametrize("n_seeds", [1, 2])
def test_non_finite_error_rates_are_refused(bad: float, n_seeds: int) -> None:
    records = [make_record(FusionMethod.B0, 0, bad)]
    records += [make_record(FusionMethod.B0, s, 0.10) for s in range(1, n_seeds)]

    with pytest.raises(ValueError, match="finite"):
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


def renamed(record: ResultRecord, prefix: str) -> ResultRecord:
    """Same lengths and buckets, other documents."""
    record.predictions = [
        replace(p, doc_id=f"{prefix}-{i:03d}") for i, p in enumerate(record.predictions)
    ]
    return record


@pytest.mark.parametrize(
    "seed_1",
    [
        pytest.param(
            lambda: make_imdb_record(FusionMethod.F1, 1, set(), docs=IMDB_DOCS[:-1]),
            id="one-document-fewer",
        ),
        pytest.param(
            lambda: renamed(make_imdb_record(FusionMethod.F1, 1, set()), "other"),
            id="same-counts-other-documents",
        ),
    ],
)
def test_seeds_evaluated_on_different_documents_are_refused(
    tmp_path: Path, seed_1: Callable[[], ResultRecord]
) -> None:
    records = [make_imdb_record(FusionMethod.F1, 0, set()), seed_1()]

    with pytest.raises(ValueError, match="same documents"):
        generate_report_artifacts(records, JudgmentReport(judgments={}), tmp_path)


def test_mnli_seeds_evaluated_on_different_documents_are_refused() -> None:
    # MNLI has no input setting, so only the per-cell check guards the seed mean.
    records = [make_record(FusionMethod.B0, s, 0.5) for s in range(2)]
    records[0].predictions = [PredictionRecord("mnli-0", 1, 1), PredictionRecord("mnli-1", 1, 0)]
    records[1].predictions = [PredictionRecord("mnli-0", 1, 1), PredictionRecord("mnli-2", 1, 0)]

    with pytest.raises(ValueError, match="same documents"):
        generate_summary_table(records)


def test_runs_of_one_input_setting_evaluated_on_different_documents_are_refused(
    tmp_path: Path,
) -> None:
    # Different cells, so the per-cell check passes; the view ratio table pools per setting.
    records = [
        make_imdb_record(FusionMethod.B0, 0, set()),
        make_imdb_record(FusionMethod.F1, 0, set(), docs=IMDB_DOCS[:-1]),
    ]

    with pytest.raises(ValueError, match="same documents"):
        generate_report_artifacts(records, JudgmentReport(judgments={}), tmp_path)


def test_a_document_evaluated_twice_in_one_run_is_refused(tmp_path: Path) -> None:
    record = make_imdb_record(FusionMethod.F1, 0, set())
    record.predictions.append(record.predictions[0])

    with pytest.raises(ValueError, match="imdb-000"):
        generate_report_artifacts([record], JudgmentReport(judgments={}), tmp_path)


@pytest.mark.parametrize(
    ("token_length", "bucket"),
    [
        pytest.param(100, LengthBucket.GT_1024, id="100-tokens-tagged-over-1024"),
        pytest.param(None, LengthBucket.LE_512, id="bucket-without-length"),
        pytest.param(600, None, id="imdb-length-without-bucket"),
    ],
)
def test_predictions_whose_length_and_bucket_disagree_are_refused(
    tmp_path: Path, token_length: int | None, bucket: LengthBucket | None
) -> None:
    record = make_imdb_record(FusionMethod.F1, 0, set(), docs=[(100, LengthBucket.LE_512)])
    record.predictions = [
        replace(record.predictions[0], token_length=token_length, length_bucket=bucket)
    ]

    with pytest.raises(ValueError, match="length"):
        generate_report_artifacts([record], JudgmentReport(judgments={}), tmp_path)


def test_mnli_token_lengths_without_length_buckets_are_accepted(tmp_path: Path) -> None:
    """The runner records token_length for every dataset, a bucket only under an input setting."""
    records = [make_record(FusionMethod.B0, s, 0.5) for s in range(2)]
    for record in records:
        record.predictions = [
            PredictionRecord("mnli-0", label=1, prediction=0, token_length=40),
            PredictionRecord("mnli-1", label=1, prediction=1, token_length=900),
        ]

    paths = generate_report_artifacts(records, JudgmentReport(judgments={}), tmp_path)

    assert {p.name for p in paths} == {"summary_table.csv"}  # no bucket or view ratio table


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
    assert [r["length_bucket"] for r in rows if r["input_setting"] == "L128"] == [
        "<=512",
        "512-1024",
        ">1024",
        "all",
    ]
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


def test_length_bucket_and_view_ratio_tables_leave_out_debug_and_reproduction_runs(
    tmp_path: Path,
) -> None:
    main = [make_imdb_record(FusionMethod.F1, s, set()) for s in range(2)]
    debug = make_imdb_record(FusionMethod.F1, 2, {0, 1}, docs=IMDB_DOCS[:2])  # a debug subset
    debug.config = replace(debug.config, run_kind=RunKind.DEBUG)
    reproduction = make_imdb_record(FusionMethod.F1, 3, {0, 1, 2, 3, 4, 5})
    reproduction.config = replace(reproduction.config, track=TrackKind.REPRODUCTION)
    extra = make_imdb_record(  # another doc set and lengths in L128: only seen if not excluded
        FusionMethod.B0, 0, set(), input_setting=InputSetting.L128, docs=[(50, LengthBucket.LE_512)]
    )
    extra.config = replace(extra.config, run_kind=RunKind.DEBUG)

    main_only = generate_report_artifacts(main, JudgmentReport(judgments={}), tmp_path / "main")
    mixed = generate_report_artifacts(
        [*main, debug, reproduction, extra], JudgmentReport(judgments={}), tmp_path / "mixed"
    )

    for name in ("length_bucket_table.csv", "view_ratio_table.csv"):
        assert read_csv(artifact(mixed, name)) == read_csv(artifact(main_only, name)), name
    buckets = read_csv(artifact(mixed, "length_bucket_table.csv"))
    assert {(r["n_docs"], r["n_seeds"], float(r["error_rate_mean"])) for r in buckets} == {
        ("2", "2", 0.0)
    }
    assert {r["input_setting"] for r in read_csv(artifact(mixed, "view_ratio_table.csv"))} == {
        "L512"
    }


PNG_SIGNATURE = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])


def pixels(path: Path) -> np.ndarray:
    """Decoded image, so figures are compared by what they show, not by PNG metadata."""
    from matplotlib.image import imread

    return imread(path)


def make_f2_record(
    seed: int,
    weights: list[float] | None,
    *,
    subset: LayerSubset = LayerSubset.GLOBAL_ONLY,
    run_kind: RunKind = RunKind.MAIN,
    track: TrackKind = TrackKind.MAIN,
) -> ResultRecord:
    record = make_record(FusionMethod.F2, seed, 0.1, run_kind=run_kind, track=track)
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
        make_f2_record(3, [7.0] * 8, track=TrackKind.REPRODUCTION),  # nor this one
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


def test_single_seed_f2_has_no_weight_std_and_still_gets_a_figure(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")

    paths = generate_report_artifacts(
        [make_f2_record(0, [0.125] * 8)], JudgmentReport(judgments={}), tmp_path
    )

    rows = read_csv(artifact(paths, "layer_weights_table.csv"))
    assert {(r["n_seeds"], r["weight_std"]) for r in rows} == {("1", "")}  # no fake 0
    assert artifact(paths, "layer_weights.png").read_bytes().startswith(PNG_SIGNATURE)


def test_layer_weight_figure_shows_the_weights(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")

    def figure(weights: list[float], name: str) -> np.ndarray:
        paths = generate_report_artifacts(
            [make_f2_record(0, weights)], JudgmentReport(judgments={}), tmp_path / name
        )
        return pixels(artifact(paths, "layer_weights.png"))

    flat = figure([0.125] * 8, "flat")
    peaked = figure([0.5, 0.5, 0, 0, 0, 0, 0, 0], "peaked")

    assert flat.shape == peaked.shape  # same cell, same layout
    assert not np.array_equal(flat, peaked)


@pytest.mark.parametrize(
    "weights_by_seed",
    [
        pytest.param([[0.125] * 8, None], id="one-seed-missing"),
        pytest.param([None, None], id="all-seeds-missing"),
    ],
)
def test_every_f2_run_must_carry_its_layer_weights(
    tmp_path: Path, weights_by_seed: list[list[float] | None]
) -> None:
    # Otherwise the summary says n_seeds=2 while the weight table silently uses fewer seeds.
    records = [make_f2_record(s, w) for s, w in enumerate(weights_by_seed)]

    with pytest.raises(ValueError, match="layer_weights"):
        generate_report_artifacts(records, JudgmentReport(judgments={}), tmp_path)


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


def curve_record(
    method: FusionMethod, seed: int, *errors: float, **kwargs: RunKind | TrackKind
) -> ResultRecord:
    record = make_record(method, seed, 0.3, **kwargs)  # type: ignore[arg-type]
    record.learning_curve = curve(*errors)
    return record


def curve_figure(records: list[ResultRecord], out: Path) -> np.ndarray:
    paths = generate_report_artifacts(records, JudgmentReport(judgments={}), out)
    return pixels(artifact(paths, "learning_curves.png"))


def test_learning_curve_figure_leaves_out_debug_and_reproduction_curves(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    main = [curve_record(FusionMethod.B0, s, 0.5, 0.4, 0.3) for s in range(2)]
    excluded = [
        curve_record(FusionMethod.B0, 2, 0.9, 0.05, 0.9, run_kind=RunKind.DEBUG),
        curve_record(FusionMethod.B0, 3, 0.05, 0.95, 0.05, track=TrackKind.REPRODUCTION),
    ]

    assert np.array_equal(
        curve_figure([*main, *excluded], tmp_path / "mixed"), curve_figure(main, tmp_path / "main")
    )


def test_learning_curve_figure_shows_the_tuning_errors(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")

    falling = curve_figure([curve_record(FusionMethod.B0, 0, 0.5, 0.4, 0.3)], tmp_path / "a")
    rising = curve_figure([curve_record(FusionMethod.B0, 0, 0.3, 0.4, 0.5)], tmp_path / "b")

    assert falling.shape == rising.shape
    assert not np.array_equal(falling, rising)


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
