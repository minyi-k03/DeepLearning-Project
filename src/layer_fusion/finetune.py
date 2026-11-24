"""Finetune condition: ModernBERT and the fusion head trained together (T-06d, US 44, ADR 0004).

The frozen path feeds cached [N, 23, 768] layer vectors to a head. Here a batch of token ids
goes through a trainable backbone and the 23 layer vectors are pooled on the fly with the
same rule as the extractor (``extraction.masked_token_mean``: padding excluded, special tokens
pooled per ``include_special_tokens_in_pool``; the 23rd vector is the final-norm output,
ADR 0005). :class:`TokenRowView` and :class:`FinetuneModel` match the interface that
``training.train_head`` / ``training.evaluate`` already use (``view.load(rows, device) ->
(x, y)``, ``model(x)``), so the training loop, LR selection and early stopping are shared
with the frozen path unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from layer_fusion.extraction import masked_token_mean
from layer_fusion.layers import NUM_LAYER_VECTORS

TokenInputs = tuple[torch.Tensor, torch.Tensor, torch.Tensor]  # ids, attention, pool mask


@dataclass(frozen=True)
class TokenRowView:
    """Rows ``rows`` of a tokenized split, padded per batch (same interface as ``RowView``)."""

    input_ids: Sequence[Sequence[int]]
    labels: np.ndarray  # [N]
    rows: np.ndarray  # int64 row indices
    pad_id: int
    special_ids: tuple[int, ...]  # [CLS], [SEP]
    pool_special_tokens: bool

    def __len__(self) -> int:
        return len(self.rows)

    def load(self, rows: np.ndarray, device: torch.device) -> tuple[TokenInputs, torch.Tensor]:
        """``(ids, attention, pool_mask)`` padded to the longest row, and int64 labels."""
        seqs = [self.input_ids[int(i)] for i in rows]
        width = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), width), self.pad_id, dtype=torch.long)
        attention = torch.zeros((len(seqs), width), dtype=torch.long)
        for r, seq in enumerate(seqs):
            ids[r, : len(seq)] = torch.as_tensor(seq, dtype=torch.long)
            attention[r, : len(seq)] = 1
        pool_mask = attention
        if not self.pool_special_tokens:
            special = torch.as_tensor(self.special_ids, dtype=torch.long)
            pool_mask = attention * (~torch.isin(ids, special)).long()
        y = torch.from_numpy(np.asarray(self.labels[rows], dtype=np.int64))
        return (ids.to(device), attention.to(device), pool_mask.to(device)), y.to(device)


class FinetuneModel(nn.Module):
    """Trainable backbone -> 23 pooled layer vectors -> fusion head."""

    def __init__(self, backbone: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def layer_vectors(self, inputs: TokenInputs) -> torch.Tensor:
        """[B, 23, 768] float32 layer vectors, with gradients flowing into the backbone."""
        ids, attention, pool_mask = inputs
        hidden_states = self.backbone(
            input_ids=ids, attention_mask=attention, output_hidden_states=True
        ).hidden_states
        if len(hidden_states) != NUM_LAYER_VECTORS:
            raise ValueError(
                f"expected {NUM_LAYER_VECTORS} hidden states, got {len(hidden_states)}"
            )
        return torch.stack([masked_token_mean(h.float(), pool_mask) for h in hidden_states], dim=1)

    def forward(self, inputs: TokenInputs) -> torch.Tensor:
        return self.head(self.layer_vectors(inputs))


class PaperHeadModel(nn.Module):
    """Reproduction track (T-06e): the paper's sequence-classification model behind the same
    token-view interface, so ``training.evaluate`` works on it unchanged."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, inputs: TokenInputs) -> torch.Tensor:
        ids, attention, _pool_mask = inputs  # the paper head pools with the attention mask
        return self.model(input_ids=ids, attention_mask=attention).logits
