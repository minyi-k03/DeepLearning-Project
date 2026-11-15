"""Canonical ModernBERT-base layer structure definition (SPEC.md, GLOSSARY.md, ADR 0006).

Slot 0 is the embedding vector (임베딩 벡터); it has no layer number and belongs to
neither global nor local layers.
Slots 1..22 correspond to transformer layers 1..22.
"""

from __future__ import annotations

from layer_fusion.contracts import LayerSubset

NUM_LAYERS = 22
NUM_LAYER_VECTORS = 23  # 1 embedding slot + 22 layers
HIDDEN_SIZE = 768
EMBEDDING_SLOT = 0

# Global layers: 8 layers (every 3rd layer including first and last: 1, 4, 7, 10, 13, 16, 19, 22)
GLOBAL_LAYERS: tuple[int, ...] = (1, 4, 7, 10, 13, 16, 19, 22)

# Local layers: remaining 14 layers in 1..22
LOCAL_LAYERS: tuple[int, ...] = tuple(
    idx for idx in range(1, NUM_LAYERS + 1) if idx not in GLOBAL_LAYERS
)

# Layer ranges defined in GLOSSARY.md / SPEC.md
LAYER_RANGES: dict[str, tuple[int, ...]] = {
    "lower": tuple(range(1, 8)),     # 1~7층
    "middle": tuple(range(8, 16)),   # 8~15층
    "upper": tuple(range(16, 23)),   # 16~22층
}

# Local attention receptive field: +/- 64 tokens + self = 129 tokens
LOCAL_WINDOW_HALF = 64
LOCAL_WINDOW_TOTAL = 2 * LOCAL_WINDOW_HALF + 1


def is_global_layer(layer_index: int) -> bool:
    """Return True if 1-based layer_index (1..22) is a global attention layer."""
    if layer_index < 1 or layer_index > NUM_LAYERS:
        raise ValueError(f"Layer index must be in 1..{NUM_LAYERS}, got {layer_index}")
    return layer_index in GLOBAL_LAYERS


def get_layer_range(layer_index: int) -> str:
    """Return 'lower', 'middle', or 'upper' for a 1-based layer_index (1..22)."""
    for name, indices in LAYER_RANGES.items():
        if layer_index in indices:
            return name
    raise ValueError(f"Layer index must be in 1..{NUM_LAYERS}, got {layer_index}")


def slots_for_subset(subset: LayerSubset) -> tuple[int, ...]:
    """Return slot indices (0..22) for the given LayerSubset.

    For GLOBAL_ONLY and LOCAL_ONLY, embedding slot 0 is excluded per SPEC.md / ADR 0006.
    """
    if subset == LayerSubset.ALL:
        return tuple(range(NUM_LAYER_VECTORS))
    if subset == LayerSubset.GLOBAL_ONLY:
        return GLOBAL_LAYERS
    if subset == LayerSubset.LOCAL_ONLY:
        return LOCAL_LAYERS
    raise ValueError(f"Unsupported layer subset: {subset}")
