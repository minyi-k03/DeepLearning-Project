"""Per-layer normalization and layer fusion heads (SPEC.md, ADR 0005, ADR 0008).

Processing order:
1. Backbone final norm on last layer tokens -> token-mean pooling -> 23 layer vectors
2. Parameter-free per-layer LayerNorm applied to each of the 23 layer vectors
3. Fusion rule (B0 / F1 / F2 / F3 / F4 / B0+) -> classifier logits

Note:
- F2 has NO global scalar gamma (γ) per ADR 0005.
- F4 exact gating architecture and F2/F4 weight normalization are TBD before main runs.
"""

from __future__ import annotations

from layer_fusion.contracts import FusionMethod, LayerSubset


def build_fusion_head(
    fusion_method: FusionMethod,
    num_classes: int,
    layer_subset: LayerSubset = LayerSubset.ALL,
    hidden_size: int = 768,
) -> object:
    """Instantiate the requested fusion head with parameter-free per-layer LayerNorm."""
    raise NotImplementedError("TODO: Implement fusion heads (B0, F1, F2, F3, F4, B0+) (T-05)")


def count_trainable_parameters(module: object) -> int:
    """Return number of trainable parameters in a fusion head / model."""
    raise NotImplementedError("TODO: Implement trainable parameter counter (T-05)")


def extract_layer_weights(module: object) -> list[float] | None:
    """Expose dataset-level (F2) or input-level (F4) layer weights if applicable."""
    raise NotImplementedError("TODO: Implement layer weight extraction (T-05)")
