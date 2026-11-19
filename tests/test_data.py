"""Unit tests for the data module (T-03a, SPEC US 1, 3-11). CPU only, tiny fake rows."""

from __future__ import annotations

import random
import sys
import types
from collections import Counter
from dataclasses import FrozenInstanceError, replace

import pytest

from layer_fusion.contracts import DatasetName, DocumentRecord
from layer_fusion.data import (
    DATASET_SPECS,
    EVAL_SPLIT,
    TRAIN_SPLIT,
    TUNING_SPLIT,
    load_dataset_splits,
    make_doc_id,
    make_tuning_split,
    rows_to_records,
    sample_small_dataset,
)


def make_records(per_class: int, num_classes: int = 3, split: str = TRAIN_SPLIT):
    """Label-sorted fake records (like IMDB train) so order-dependence would show up."""
    return [
        DocumentRecord(doc_id=f"d-{label}-{i:05d}", text_a=f"t{label}{i}", label=label, split=split)
        for label in range(num_classes)
        for i in range(per_class)
    ]


def tuning_ids(records, seed: int) -> list[str]:
    return [r.doc_id for r in make_tuning_split(records, seed=seed)[1]]


def sample_ids(records, seed: int, per_class: int = 10, num_classes: int = 3) -> list[str]:
    return sample_small_dataset(records, per_class=per_class, seed=seed, num_classes=num_classes)[1]


# --- dataset specs / row conversion ---------------------------------------------------------


def test_every_dataset_name_has_a_spec() -> None:
    """Switching dataset (MNLI -> AG News) must be possible by name alone (US 11)."""
    assert set(DATASET_SPECS) == set(DatasetName)


def test_eval_sources_follow_adr_0003() -> None:
    mnli = DATASET_SPECS[DatasetName.MNLI]
    assert [(s.hf_split, s.genre_split) for s in mnli.eval_sources] == [
        ("validation_matched", "matched"),
        ("validation_mismatched", "mismatched"),
    ]
    assert DATASET_SPECS[DatasetName.SST2].eval_sources[0].hf_split == "validation"
    assert DATASET_SPECS[DatasetName.IMDB].eval_sources[0].hf_split == "test"
    assert DATASET_SPECS[DatasetName.AG_NEWS].eval_sources[0].hf_split == "test"


def test_mnli_rows_become_pair_records_with_genre_and_stable_ids() -> None:
    rows = [
        {"premise": "p0", "hypothesis": "h0", "label": 2},
        {"premise": "p1", "hypothesis": "h1", "label": 0},
    ]
    records = rows_to_records(
        rows, DatasetName.MNLI, "validation_matched", EVAL_SPLIT, genre_split="matched"
    )
    assert [r.doc_id for r in records] == [
        "mnli-validation_matched-000000",
        "mnli-validation_matched-000001",
    ]
    assert (records[0].text_a, records[0].text_b, records[0].label) == ("p0", "h0", 2)
    assert {r.genre_split for r in records} == {"matched"}
    assert {r.split for r in records} == {EVAL_SPLIT}


def test_matched_and_mismatched_doc_ids_never_collide() -> None:
    assert make_doc_id(DatasetName.MNLI, "validation_matched", 5) != make_doc_id(
        DatasetName.MNLI, "validation_mismatched", 5
    )


def test_single_sentence_dataset_has_no_text_b() -> None:
    records = rows_to_records(
        [{"sentence": "good", "label": 1}], DatasetName.SST2, "validation", EVAL_SPLIT
    )
    assert records[0].text_b is None
    assert records[0].genre_split is None


def test_ag_news_rows_accept_four_classes() -> None:
    """AG News fallback (ADR 0003): 4 classes, single text input."""
    rows = [{"text": f"news {i}", "label": i} for i in range(4)]
    records = rows_to_records(rows, DatasetName.AG_NEWS, "train", TRAIN_SPLIT)
    assert [r.label for r in records] == [0, 1, 2, 3]
    assert records[3].doc_id == "ag_news-train-000003"
    with pytest.raises(ValueError, match="label 4"):
        rows_to_records([{"text": "x", "label": 4}], DatasetName.AG_NEWS, "train", TRAIN_SPLIT)


def test_text_is_kept_raw() -> None:
    """IMDB `<br />` stays (team decision in PR #8)."""
    raw = "great.<br /><br />really"
    records = rows_to_records([{"text": raw, "label": 1}], DatasetName.IMDB, "test", EVAL_SPLIT)
    assert records[0].text_a == raw


def test_hidden_or_out_of_range_label_raises() -> None:
    with pytest.raises(ValueError, match="label -1"):
        rows_to_records([{"sentence": "x", "label": -1}], DatasetName.SST2, "test", EVAL_SPLIT)
    with pytest.raises(ValueError, match="label 2"):
        rows_to_records([{"sentence": "x", "label": 2}], DatasetName.SST2, "train", TRAIN_SPLIT)


def test_document_record_is_immutable() -> None:
    record = make_records(1)[0]
    with pytest.raises(FrozenInstanceError):
        record.label = 9  # type: ignore[misc]


# --- load_dataset_splits with a fake `datasets` module ---------------------------------------


@pytest.fixture
def fake_hf(monkeypatch: pytest.MonkeyPatch) -> list[tuple]:
    """Inject a fake `datasets.load_dataset` so the loader runs without network or the package."""
    calls: list[tuple] = []
    rows = {
        "train": [{"premise": "p", "hypothesis": "h", "label": i % 3} for i in range(4)],
        "validation_matched": [{"premise": "pm", "hypothesis": "hm", "label": 0}] * 2,
        "validation_mismatched": [{"premise": "px", "hypothesis": "hx", "label": 1}] * 3,
    }

    def load_dataset(path, name=None, *, split, cache_dir=None):
        calls.append((path, name, split, cache_dir))
        return rows[split]

    module = types.ModuleType("datasets")
    module.load_dataset = load_dataset  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", module)
    return calls


def test_load_dataset_splits_mnli_orders_matched_then_mismatched(fake_hf: list[tuple]) -> None:
    splits = load_dataset_splits(DatasetName.MNLI, data_dir="cache-dir")
    assert set(splits) == {TRAIN_SPLIT, EVAL_SPLIT}
    assert len(splits[TRAIN_SPLIT]) == 4
    assert {r.split for r in splits[TRAIN_SPLIT]} == {TRAIN_SPLIT}
    assert [r.genre_split for r in splits[EVAL_SPLIT]] == ["matched"] * 2 + ["mismatched"] * 3
    assert {r.split for r in splits[EVAL_SPLIT]} == {EVAL_SPLIT}
    assert len({r.doc_id for r in splits[EVAL_SPLIT]}) == 5
    assert fake_hf == [
        ("nyu-mll/glue", "mnli", "train", "cache-dir"),
        ("nyu-mll/glue", "mnli", "validation_matched", "cache-dir"),
        ("nyu-mll/glue", "mnli", "validation_mismatched", "cache-dir"),
    ]


# --- tuning split ---------------------------------------------------------------------------


def test_tuning_split_is_disjoint_complete_and_about_ten_percent() -> None:
    records = make_records(per_class=100)
    train, tuning = make_tuning_split(records, fraction=0.10, seed=0)
    assert len(tuning) == 30  # 10 per class
    assert Counter(r.label for r in tuning) == {0: 10, 1: 10, 2: 10}
    assert {r.doc_id for r in train}.isdisjoint({r.doc_id for r in tuning})
    assert {r.doc_id for r in train} | {r.doc_id for r in tuning} == {r.doc_id for r in records}


def test_tuning_records_are_tagged_and_originals_untouched() -> None:
    records = make_records(per_class=20)
    train, tuning = make_tuning_split(records, seed=1)
    assert {r.split for r in tuning} == {TUNING_SPLIT}
    assert {r.split for r in train} == {TRAIN_SPLIT}
    assert {r.split for r in records} == {TRAIN_SPLIT}


def test_tuning_split_keeps_original_order() -> None:
    records = make_records(per_class=50)
    index = {r.doc_id: i for i, r in enumerate(records)}
    train, tuning = make_tuning_split(records, seed=3)
    for part in (train, tuning):
        positions = [index[r.doc_id] for r in part]
        assert positions == sorted(positions)


def test_tuning_split_is_deterministic_per_seed_and_varies_across_seeds() -> None:
    records = make_records(per_class=100)
    assert tuning_ids(records, seed=0) == tuning_ids(records, seed=0)
    assert tuning_ids(records, seed=0) != tuning_ids(records, seed=1)


def test_tuning_split_does_not_depend_on_input_order() -> None:
    records = make_records(per_class=50)
    assert set(tuning_ids(records, seed=0)) == set(tuning_ids(list(reversed(records)), seed=0))


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.5])
def test_tuning_split_rejects_invalid_fraction(fraction: float) -> None:
    with pytest.raises(ValueError, match="fraction"):
        make_tuning_split(make_records(per_class=10), fraction=fraction, seed=0)


def test_tuning_split_rejects_class_too_small_for_a_nonempty_split() -> None:
    """round(0.1 * 5) == 0 would give an empty tuning split for that class."""
    with pytest.raises(ValueError, match="class 0"):
        make_tuning_split(make_records(per_class=5), seed=0)


@pytest.mark.parametrize(
    "bad_records",
    [
        [],
        make_records(per_class=20) + [replace(make_records(per_class=1)[0], label=1)],
        make_records(per_class=20, split=EVAL_SPLIT),
    ],
    ids=["empty", "duplicate-doc-id", "eval-records"],
)
def test_tuning_split_rejects_bad_input(bad_records: list[DocumentRecord]) -> None:
    with pytest.raises(ValueError):
        make_tuning_split(bad_records, seed=0)


def test_seed_is_required() -> None:
    """A forgotten seed must not silently reuse seed 0 (US 7)."""
    records = make_records(per_class=20)
    with pytest.raises(TypeError):
        make_tuning_split(records)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        sample_small_dataset(records, per_class=5, num_classes=3)  # type: ignore[call-arg]


# --- small-data sampling --------------------------------------------------------------------


def test_small_sample_has_exactly_per_class_per_label_and_matching_ids() -> None:
    records = make_records(per_class=50, num_classes=3)
    sampled, ids = sample_small_dataset(records, per_class=10, seed=0, num_classes=3)
    assert Counter(r.label for r in sampled) == {0: 10, 1: 10, 2: 10}
    assert ids == [r.doc_id for r in sampled]
    assert len(set(ids)) == len(ids)


def test_small_sample_is_same_within_seed_and_differs_across_seeds() -> None:
    records = make_records(per_class=50)
    assert sample_ids(records, seed=0) == sample_ids(records, seed=0)
    assert set(sample_ids(records, seed=0)) != set(sample_ids(records, seed=1))


def test_small_sample_does_not_depend_on_input_order() -> None:
    records = make_records(per_class=50)
    assert sample_ids(records, seed=2) == sample_ids(list(reversed(records)), seed=2)


def test_small_sample_only_contains_input_documents() -> None:
    records = make_records(per_class=50)
    sampled, _ = sample_small_dataset(records, per_class=10, seed=0, num_classes=3)
    assert set(sampled) <= set(records)


def test_small_sample_and_tuning_split_use_independent_random_streams() -> None:
    """Same seed must not make the small sample a prefix of the tuning draw (review of T-03a).

    With a shared stream, 100% of a class's sample landed inside the tuning split.
    Expected overlap for independent draws is about `fraction` (10%).
    """
    records = make_records(per_class=2000)
    tuning = set(tuning_ids(records, seed=0))
    sample = sample_ids(records, seed=0, per_class=200)
    overlap = sum(doc_id in tuning for doc_id in sample) / len(sample)
    assert overlap < 0.25


def test_small_sample_leaves_global_random_state_untouched() -> None:
    random.seed(123)
    before = random.getstate()
    sample_ids(make_records(per_class=50), seed=0)
    make_tuning_split(make_records(per_class=50), seed=0)
    assert random.getstate() == before


def test_small_sample_accepts_tuning_records_but_rejects_eval() -> None:
    """Whether tuning docs may be sampled is an open SPEC decision, so both train/tuning pass."""
    train, tuning = make_tuning_split(make_records(per_class=100), seed=0)
    sample_small_dataset(train + tuning, per_class=10, seed=0, num_classes=3)
    with pytest.raises(ValueError, match="split 'eval'"):
        sample_small_dataset(
            make_records(per_class=50, split=EVAL_SPLIT), per_class=10, seed=0, num_classes=3
        )


def test_small_sample_fails_loudly_when_a_class_is_too_small() -> None:
    """A silently smaller sample would break the 'per class 1,000' comparison."""
    with pytest.raises(ValueError, match="fewer than per_class"):
        sample_ids(make_records(per_class=5), seed=0, per_class=10)


@pytest.mark.parametrize(
    ("records", "num_classes"),
    [([], 3), (make_records(per_class=50, num_classes=2), 3)],
    ids=["empty", "missing-class"],
)
def test_small_sample_rejects_empty_input_or_missing_class(
    records: list[DocumentRecord], num_classes: int
) -> None:
    with pytest.raises(ValueError):
        sample_small_dataset(records, per_class=10, seed=0, num_classes=num_classes)


def test_small_sample_rejects_duplicate_doc_ids() -> None:
    records = make_records(per_class=20)
    with pytest.raises(ValueError, match="duplicate doc_id"):
        sample_ids(records + [records[0]], seed=0)


def test_small_sample_rejects_non_positive_per_class() -> None:
    with pytest.raises(ValueError, match="per_class"):
        sample_ids(make_records(per_class=20), seed=0, per_class=0)
