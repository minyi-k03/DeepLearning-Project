"""Unit tests for canonical ModernBERT-base layer structure (layers.py)."""

import pytest

from layer_fusion.contracts import LayerSubset
from layer_fusion.layers import (
    EMBEDDING_SLOT,
    GLOBAL_LAYERS,
    LAYER_RANGES,
    LOCAL_LAYERS,
    NUM_LAYERS,
    NUM_LAYER_VECTORS,
    get_layer_range,
    is_global_layer,
    slots_for_subset,
)


def test_layer_counts_and_partition() -> None:
    assert NUM_LAYERS == 22
    assert NUM_LAYER_VECTORS == 23
    assert len(GLOBAL_LAYERS) == 8
    assert len(LOCAL_LAYERS) == 14
    assert set(GLOBAL_LAYERS).isdisjoint(LOCAL_LAYERS)
    assert set(GLOBAL_LAYERS) | set(LOCAL_LAYERS) == set(range(1, NUM_LAYERS + 1))


def test_subset_slots_exclude_embedding_for_global_and_local() -> None:
    all_slots = slots_for_subset(LayerSubset.ALL)
    global_slots = slots_for_subset(LayerSubset.GLOBAL_ONLY)
    local_slots = slots_for_subset(LayerSubset.LOCAL_ONLY)

    assert len(all_slots) == 23
    assert EMBEDDING_SLOT in all_slots
    assert len(global_slots) == 8
    assert EMBEDDING_SLOT not in global_slots
    assert len(local_slots) == 14
    assert EMBEDDING_SLOT not in local_slots


def test_layer_ranges_and_helpers() -> None:
    assert LAYER_RANGES["lower"] == tuple(range(1, 8))
    assert LAYER_RANGES["middle"] == tuple(range(8, 16))
    assert LAYER_RANGES["upper"] == tuple(range(16, 23))
    assert is_global_layer(1)
    assert is_global_layer(22)
    assert not is_global_layer(2)
    assert get_layer_range(1) == "lower"
    assert get_layer_range(8) == "middle"
    assert get_layer_range(22) == "upper"
    with pytest.raises(ValueError):
        is_global_layer(0)
