"""Per-layer normalization and layer fusion heads (SPEC.md, ADR 0005, ADR 0008).

Processing order:
1. Backbone final norm on last layer tokens -> token-mean pooling -> 23 layer vectors
2. Parameter-free per-layer LayerNorm applied to each of the 23 layer vectors
3. Fusion rule (B0 / F1 / F2 / F3 / F4 / B0+) -> classifier logits

Heads take ``layer_vectors`` of shape ``[batch, 23, hidden]`` (slot 0 = embedding vector,
slots 1..22 = layers; slot 22 is the final-normed last layer) and return logits
``[batch, num_classes]``.

Note:
- F2 has NO global scalar gamma (γ) per ADR 0005. Its layer weights are ``softmax`` of one
  learnable logit per layer (non-negative, sum to 1), so the fused vector is a convex
  combination and carries no extra scale freedom. Logits start at 0, i.e. F2 starts as F1.
  The softmax choice for F2 is recorded in SPEC.md §결정 필요 사항
  ("F2·F4 층 가중치 정규화 방식").
- F4 (T-05b): a linear gate ``Linear(hidden -> 23)`` reads the mean of the 23 per-layer-normed
  vectors and emits per-document layer weights via softmax (positive, sum to 1); gate params
  start at 0, so F4 starts as F1. Per-document weights are exposed by
  ``extract_document_layer_weights``; wiring them into result records is the runner's job.
- B0+ (T-05b): last layer vector -> ReLU MLP, hidden size 26 (3 classes) / 25 (2 classes) from
  ADR 0008, so its trainable parameter count matches F4's within ``PARAM_MATCH_TOLERANCE``.
  F4's gate form and that tolerance are SPEC decision proposals pending teammate approval.
"""

from __future__ import annotations

import torch
from torch import nn

from layer_fusion.contracts import FusionMethod, LayerSubset
from layer_fusion.layers import HIDDEN_SIZE, NUM_LAYER_VECTORS, NUM_LAYERS, slots_for_subset

LAST_LAYER_SLOT = NUM_LAYERS  # slot 22: final-normed output of the last layer
F3_NUM_LAST_LAYERS = 4  # layers 19..22
# B0+ MLP hidden size by class count (ADR 0008: MNLI 26 / 2 classes 25). Other class counts,
# e.g. the AG News 4-class fallback, are TBD in SPEC and deliberately not defined here.
B0_PLUS_HIDDEN_SIZE = {3: 26, 2: 25}
# B0+ and F4 count as "matched" when their trainable parameter counts differ by at most this
# fraction of F4's (SPEC decision proposal, T-05b). The MLP hidden size is an integer, so an
# exact match is impossible (each extra unit adds ~772 parameters).
PARAM_MATCH_TOLERANCE = 0.01


class PerLayerNorm(nn.Module):
    """LayerNorm over the hidden dimension of each layer vector, with no learnable parameters."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)

    def forward(self, layer_vectors: torch.Tensor) -> torch.Tensor:
        return self.norm(layer_vectors)


class _FusionHead(nn.Module):
    """Common base: select slots -> per-layer norm -> combine -> linear classifier.

    Per-layer norm acts on each layer vector independently, so selecting slots first and
    normalizing afterwards equals normalizing all 23 and selecting.
    """

    def __init__(
        self,
        slots: tuple[int, ...],
        num_classes: int,
        hidden_size: int,
        classifier_in_features: int,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}")
        if hidden_size < 1:
            raise ValueError(f"hidden_size must be >= 1, got {hidden_size}")
        self.slots = slots
        self.hidden_size = hidden_size
        self.register_buffer("_slot_index", torch.tensor(slots, dtype=torch.long), persistent=False)
        self.layer_norm = PerLayerNorm(hidden_size)
        self.classifier = nn.Linear(classifier_in_features, num_classes)

    def _select_and_normalize(self, layer_vectors: torch.Tensor) -> torch.Tensor:
        if layer_vectors.ndim != 3 or layer_vectors.shape[1:] != (
            NUM_LAYER_VECTORS,
            self.hidden_size,
        ):
            raise ValueError(
                f"layer_vectors must have shape [batch, {NUM_LAYER_VECTORS}, "
                f"{self.hidden_size}], got {tuple(layer_vectors.shape)}"
            )
        return self.layer_norm(layer_vectors.index_select(1, self._slot_index))

    def combine(self, selected: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, layer_vectors: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.combine(self._select_and_normalize(layer_vectors)))


class B0Head(_FusionHead):
    """B0: last layer vector only -> linear classifier."""

    def __init__(self, num_classes: int, hidden_size: int) -> None:
        super().__init__((LAST_LAYER_SLOT,), num_classes, hidden_size, hidden_size)

    def combine(self, selected: torch.Tensor) -> torch.Tensor:
        return selected[:, 0]


class F1Head(_FusionHead):
    """F1: simple mean of all 23 layer vectors (embedding vector included)."""

    def __init__(self, num_classes: int, hidden_size: int) -> None:
        super().__init__(slots_for_subset(LayerSubset.ALL), num_classes, hidden_size, hidden_size)

    def combine(self, selected: torch.Tensor) -> torch.Tensor:
        return selected.mean(dim=1)


class F2Head(_FusionHead):
    """F2: one dataset-level weight per layer (softmax of learnable logits), no global gamma."""

    def __init__(
        self,
        num_classes: int,
        hidden_size: int,
        layer_subset: LayerSubset = LayerSubset.ALL,
    ) -> None:
        slots = slots_for_subset(layer_subset)
        super().__init__(slots, num_classes, hidden_size, hidden_size)
        self.layer_subset = layer_subset
        self.layer_logits = nn.Parameter(torch.zeros(len(slots)))

    def layer_weights(self) -> torch.Tensor:
        """Normalized layer weights in ``self.slots`` order (non-negative, sum to 1)."""
        return torch.softmax(self.layer_logits, dim=0)

    def combine(self, selected: torch.Tensor) -> torch.Tensor:
        return torch.einsum("l,blh->bh", self.layer_weights(), selected)


class F3Head(_FusionHead):
    """F3: concatenate the last 4 layer vectors (layers 19~22) -> linear classifier."""

    def __init__(self, num_classes: int, hidden_size: int) -> None:
        slots = tuple(range(LAST_LAYER_SLOT - F3_NUM_LAST_LAYERS + 1, LAST_LAYER_SLOT + 1))
        super().__init__(slots, num_classes, hidden_size, hidden_size * len(slots))

    def combine(self, selected: torch.Tensor) -> torch.Tensor:
        return selected.flatten(start_dim=1)


class F4Head(_FusionHead):
    """F4: per-document layer weights from a linear gate, then weighted sum -> classifier.

    The gate reads the mean of the 23 per-layer-normed vectors (the F1 vector) and emits one
    logit per layer; ``softmax`` turns them into weights (positive, sum to 1) that differ per
    document. Gate weight and bias start at 0, so F4 starts as F1 (like F2).
    """

    def __init__(self, num_classes: int, hidden_size: int) -> None:
        slots = slots_for_subset(LayerSubset.ALL)
        super().__init__(slots, num_classes, hidden_size, hidden_size)
        self.gate = nn.Linear(hidden_size, len(slots))
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def document_layer_weights(self, selected: torch.Tensor) -> torch.Tensor:
        """Per-document weights ``[batch, 23]`` from per-layer-normed vectors."""
        return torch.softmax(self.gate(selected.mean(dim=1)), dim=-1)

    def combine(self, selected: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bl,blh->bh", self.document_layer_weights(selected), selected)

    def layer_weights_for(self, layer_vectors: torch.Tensor) -> torch.Tensor:
        """Public path: raw ``[batch, 23, hidden]`` layer vectors -> per-document weights."""
        return self.document_layer_weights(self._select_and_normalize(layer_vectors))


class B0PlusHead(_FusionHead):
    """B0+: last layer vector -> 2-layer MLP (ReLU), parameter count matched to F4 (ADR 0008).

    Same count, different shape: a nonlinear MLP instead of F4's gated linear fusion.
    """

    def __init__(self, num_classes: int, hidden_size: int) -> None:
        if num_classes not in B0_PLUS_HIDDEN_SIZE:
            raise ValueError(
                f"B0+ hidden size is only defined for num_classes in "
                f"{sorted(B0_PLUS_HIDDEN_SIZE)} (ADR 0008); got num_classes={num_classes}. "
                "The AG News (4-class) size is still TBD in SPEC."
            )
        mlp_hidden = B0_PLUS_HIDDEN_SIZE[num_classes]
        super().__init__((LAST_LAYER_SLOT,), num_classes, hidden_size, mlp_hidden)
        self.mlp_in = nn.Linear(hidden_size, mlp_hidden)

    def combine(self, selected: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.mlp_in(selected[:, 0]))


def build_fusion_head(
    fusion_method: FusionMethod,
    num_classes: int,
    layer_subset: LayerSubset = LayerSubset.ALL,
    hidden_size: int = HIDDEN_SIZE,
) -> nn.Module:
    """Instantiate the requested fusion head with parameter-free per-layer LayerNorm.

    ``layer_subset`` other than ``ALL`` is only valid for F2 (SPEC: 층 부분집합은 F2만).
    """
    if layer_subset != LayerSubset.ALL and fusion_method != FusionMethod.F2:
        raise ValueError(
            f"layer_subset={layer_subset.value} is only supported for F2, "
            f"got {fusion_method.value}"
        )
    if fusion_method == FusionMethod.B0:
        return B0Head(num_classes, hidden_size)
    if fusion_method == FusionMethod.F1:
        return F1Head(num_classes, hidden_size)
    if fusion_method == FusionMethod.F2:
        return F2Head(num_classes, hidden_size, layer_subset)
    if fusion_method == FusionMethod.F3:
        return F3Head(num_classes, hidden_size)
    if fusion_method == FusionMethod.F4:
        return F4Head(num_classes, hidden_size)
    if fusion_method == FusionMethod.B0_PLUS:
        return B0PlusHead(num_classes, hidden_size)
    raise ValueError(f"Unsupported fusion method: {fusion_method}")


def count_trainable_parameters(module: nn.Module) -> int:
    """Return number of trainable parameters in a fusion head / model."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def extract_layer_weights(module: nn.Module) -> list[float] | None:
    """Expose dataset-level (F2) layer weights if applicable.

    F2 returns its normalized weights in ``module.slots`` order (23 for all layers, 8 for
    global-only, 14 for local-only). Other heads have no dataset-level layer weights ->
    ``None``; F4's weights depend on the document, see ``extract_document_layer_weights``.
    """
    if isinstance(module, F2Head):
        return [float(w) for w in module.layer_weights().detach().cpu()]
    return None


def extract_document_layer_weights(
    module: nn.Module, layer_vectors: torch.Tensor
) -> list[list[float]] | None:
    """Expose F4's per-document layer weights: ``[batch][23]`` (slot order), else ``None``.

    ``layer_vectors`` is the raw ``[batch, 23, hidden]`` input the head's ``forward`` takes.
    """
    if isinstance(module, F4Head):
        with torch.no_grad():
            return module.layer_weights_for(layer_vectors).cpu().tolist()
    return None
