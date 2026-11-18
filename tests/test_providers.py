"""Layer vector provider tests (T-04a): synthetic fake provider and run/cache validation."""

from dataclasses import replace

import numpy as np
import pytest

from layer_fusion.contracts import (
    BackboneCondition,
    CacheMetadata,
    DataAmount,
    DatasetName,
    DocumentRecord,
    FusionMethod,
    InputSetting,
    RunConfig,
    RunKind,
)
from layer_fusion.providers import CacheConfigMismatchError, FakeLayerVectorProvider


def _config(**overrides: object) -> RunConfig:
    base = RunConfig(
        dataset=DatasetName.MNLI,
        data_amount=DataAmount.FULL,
        fusion_method=FusionMethod.B0,
        backbone_condition=BackboneCondition.FROZEN,
        seed=0,
    )
    return replace(base, **overrides)


def test_make_synthetic_shapes_and_splits() -> None:
    provider = FakeLayerVectorProvider.make_synthetic(train_size=12, eval_size=6)
    train, eval_ = provider.get_split("train"), provider.get_split("eval")
    assert train.vectors.shape == (12, 23, 768) and train.vectors.dtype == np.float32
    assert eval_.vectors.shape == (6, 23, 768)
    assert len(train.doc_ids) == 12 and train.labels.shape == (12,)
    with pytest.raises(KeyError):
        provider.get_split("test")


def test_make_synthetic_is_deterministic_per_seed() -> None:
    a = FakeLayerVectorProvider.make_synthetic(seed=1).get_split("train")
    b = FakeLayerVectorProvider.make_synthetic(seed=1).get_split("train")
    c = FakeLayerVectorProvider.make_synthetic(seed=2).get_split("train")
    assert np.array_equal(a.vectors, b.vectors) and np.array_equal(a.labels, b.labels)
    assert a.doc_ids == b.doc_ids
    assert not np.array_equal(a.vectors, c.vectors)


def test_make_synthetic_balanced_labels_and_unique_doc_ids() -> None:
    train = FakeLayerVectorProvider.make_synthetic(num_classes=3, train_size=60).get_split("train")
    assert sorted(np.bincount(train.labels).tolist()) == [20, 20, 20]
    assert len(set(train.doc_ids)) == 60
    assert train.doc_ids[0] == "train-00000"


def test_make_synthetic_classes_carry_learnable_signal() -> None:
    train = FakeLayerVectorProvider.make_synthetic(train_size=90).get_split("train")
    means = [train.vectors[train.labels == k].mean(axis=0) for k in range(3)]
    assert not np.allclose(means[0], means[1], atol=0.1)


def test_make_synthetic_mnli_style_genre_splits() -> None:
    eval_ = FakeLayerVectorProvider.make_synthetic(num_classes=3, eval_size=10).get_split("eval")
    assert eval_.genre_splits is not None
    assert eval_.genre_splits.count("matched") == 5
    assert eval_.genre_splits.count("mismatched") == 5
    binary = FakeLayerVectorProvider.make_synthetic(num_classes=2).get_split("eval")
    assert binary.genre_splits is None


def test_make_synthetic_uses_given_doc_records() -> None:
    records = [
        DocumentRecord(doc_id="mnli-a", text_a="x", label=2, split="train"),
        DocumentRecord(doc_id="mnli-b", text_a="y", label=0, split="train"),
        DocumentRecord(doc_id="mnli-c", text_a="z", label=1, split="eval", genre_split="matched"),
    ]
    provider = FakeLayerVectorProvider.make_synthetic(doc_records=records)
    train, eval_ = provider.get_split("train"), provider.get_split("eval")
    assert train.doc_ids == ["mnli-a", "mnli-b"] and train.labels.tolist() == [2, 0]
    assert eval_.doc_ids == ["mnli-c"] and eval_.genre_splits == ["matched"]


def test_default_metadata_matches_locked_runtime() -> None:
    meta = FakeLayerVectorProvider.make_synthetic().metadata
    assert (meta.transformers_version, meta.attn_implementation, meta.torch_compile) == (
        "5.23.0",
        "sdpa",
        False,
    )
    assert meta.truncation_rule == "none"


def test_matching_expected_metadata_passes(sample_cache_metadata: CacheMetadata) -> None:
    provider = FakeLayerVectorProvider.make_synthetic(metadata=sample_cache_metadata)
    provider.validate_for_run(
        _config(input_setting=InputSetting.L512, expected_cache_meta=sample_cache_metadata)
    )


def test_mismatched_expected_metadata_raises(sample_cache_metadata: CacheMetadata) -> None:
    provider = FakeLayerVectorProvider.make_synthetic(metadata=sample_cache_metadata)
    other = replace(sample_cache_metadata, attn_implementation="flash_attention_2")
    with pytest.raises(CacheConfigMismatchError):
        provider.validate_for_run(_config(expected_cache_meta=other))


def test_input_setting_mismatch_raises_even_without_expected_meta(
    sample_cache_metadata: CacheMetadata,
) -> None:
    provider = FakeLayerVectorProvider.make_synthetic(metadata=sample_cache_metadata)  # L512
    with pytest.raises(CacheConfigMismatchError, match="input_setting"):
        provider.validate_for_run(_config(input_setting=InputSetting.LFULL, run_kind=RunKind.DEBUG))


def test_main_run_requires_expected_metadata() -> None:
    provider = FakeLayerVectorProvider.make_synthetic()
    with pytest.raises(CacheConfigMismatchError, match="expected_cache_meta"):
        provider.validate_for_run(_config(run_kind=RunKind.MAIN))
    provider.validate_for_run(_config(run_kind=RunKind.DEBUG))  # debug runs may skip it
