"""Fusion-head training loop for the frozen-backbone execution seam (SPEC.md 모듈 7, T-06a).

Rules (SPEC §결정 필요 사항, T-06a):
- Train on the training rows only; the tuning split decides early stopping (and, in the runner,
  the learning rate). The eval split is never seen here.
- Early stopping watches the tuning error rate; only a strict decrease counts as an improvement,
  and the head is restored to its best epoch.
- Optimizer ``torch.optim.AdamW`` with default settings, cross-entropy loss.
- Head initialisation and shuffling both derive from ``seed`` without touching the global RNG,
  so every LR candidate of a run starts from the same weights and sees the same batch order.

Cached vectors may be a float16 ``np.memmap``; each mini-batch is gathered with sorted row
indices (disk locality) and upcast to float32 (T-04b agreement), never the whole split.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class RowView:
    """Rows ``rows`` of a provider split, without copying the (possibly memory-mapped) vectors."""

    vectors: np.ndarray  # [N, 23, 768], any float dtype
    labels: np.ndarray  # [N]
    rows: np.ndarray  # int64 row indices into vectors / labels

    def __len__(self) -> int:
        return len(self.rows)

    def load(self, rows: np.ndarray, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Float32 vectors and int64 labels for ``rows`` (in the given order)."""
        x = np.asarray(self.vectors[rows], dtype=np.float32)
        y = np.asarray(self.labels[rows], dtype=np.int64)
        return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


@dataclass
class TrainOutcome:
    """Result of training one head at one learning rate."""

    # per epoch: epoch, train_loss, tuning_loss, tuning_error
    learning_curve: list[dict[str, float]]
    best_epoch: int
    best_tuning_error: float
    best_tuning_loss: float
    seconds: float  # wall time of the training loop (incl. per-epoch tuning evaluation)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def evaluate(
    head: nn.Module, view: RowView, batch_size: int, device: torch.device
) -> tuple[float, float, np.ndarray]:
    """Return ``(mean cross-entropy, error rate, predictions in view order)``."""
    if len(view) == 0:
        raise ValueError("cannot evaluate an empty split")
    head.eval()
    total_loss = torch.zeros((), device=device)
    predictions = np.empty(len(view), dtype=np.int64)
    for start in range(0, len(view), batch_size):
        rows = view.rows[start : start + batch_size]
        x, y = view.load(rows, device)
        logits = head(x)
        total_loss += nn.functional.cross_entropy(logits, y, reduction="sum")
        predictions[start : start + len(rows)] = logits.argmax(dim=1).cpu().numpy()
    labels = np.asarray(view.labels[view.rows], dtype=np.int64)
    error = float(np.mean(predictions != labels))
    return float(total_loss.item()) / len(view), error, predictions


def train_head(
    make_head: Callable[[], nn.Module],
    train_view: RowView,
    tuning_view: RowView,
    *,
    lr: float,
    epochs: int,
    patience: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> tuple[nn.Module, TrainOutcome]:
    """Train a fresh head and return it restored to the epoch with the lowest tuning error.

    Stops early after ``patience`` consecutive epochs without a strict tuning-error decrease.
    """
    if len(train_view) == 0:
        raise ValueError("cannot train on an empty split")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        head = make_head()
    head.to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr)
    shuffle_rng = np.random.default_rng(seed)

    curve: list[dict[str, float]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_error, best_loss, best_epoch = math.inf, math.inf, 0
    epochs_without_improvement = 0

    _sync(device)
    start_time = time.perf_counter()
    for epoch in range(1, epochs + 1):
        head.train()
        order = shuffle_rng.permutation(len(train_view))
        train_loss = torch.zeros((), device=device)
        for start in range(0, len(order), batch_size):
            rows = np.sort(train_view.rows[order[start : start + batch_size]])
            x, y = train_view.load(rows, device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(head(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.detach() * len(rows)
        tuning_loss, tuning_error, _ = evaluate(head, tuning_view, batch_size, device)
        curve.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss.item()) / len(train_view),
                "tuning_loss": tuning_loss,
                "tuning_error": tuning_error,
            }
        )
        if tuning_error < best_error:
            best_error, best_loss, best_epoch = tuning_error, tuning_loss, epoch
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break
    _sync(device)
    seconds = time.perf_counter() - start_time

    assert best_state is not None  # epochs >= 1 and the first epoch always improves on inf
    head.load_state_dict(best_state)
    return head, TrainOutcome(
        learning_curve=curve,
        best_epoch=best_epoch,
        best_tuning_error=best_error,
        best_tuning_loss=best_loss,
        seconds=seconds,
    )
