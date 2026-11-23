"""Fusion-head training loop for the frozen-backbone execution seam (SPEC.md 모듈 7, T-06a).

Rules (SPEC §결정 필요 사항, T-06a):
- Train on the training rows only; the tuning split decides early stopping (and, in the runner,
  the learning rate). The eval split is never seen here.
- Early stopping compares (tuning error rate, tuning loss) lexicographically, the same order as
  the LR tie-break; the head is restored to its best epoch.
- Optimizer ``torch.optim.AdamW``; weight decay ``WEIGHT_DECAY`` only on weight matrices
  (ndim >= 2), none on biases or F2 layer logits (which would pull F2 towards uniform = F1).
  Cross-entropy loss.
- An epoch whose training or tuning loss is not finite marks the run as diverged; it stops there
  and that epoch never becomes the best state.
- Head initialisation and shuffling both derive from ``seed`` without touching the global (CPU or
  CUDA) RNG, so every LR candidate of a run starts from the same weights and batch order.
- ``train_seconds`` times only the training steps; per-epoch tuning evaluation is timed apart.

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

WEIGHT_DECAY = 0.01  # AdamW default, applied to weight matrices only


class TrainingDivergedError(RuntimeError):
    """Training produced a non-finite loss and no usable model is left."""


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
    best_epoch: int  # 0 when the first epoch already diverged
    best_tuning_error: float  # inf when diverged before any finite epoch
    best_tuning_loss: float
    train_seconds: float  # wall time of the training steps only
    tuning_eval_seconds: float  # wall time of the per-epoch tuning evaluation
    diverged: bool = False

    @property
    def seconds(self) -> float:
        """Total loop wall time (training steps + tuning evaluation)."""
        return self.train_seconds + self.tuning_eval_seconds


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _rng_devices() -> list[int]:
    """CUDA devices whose RNG ``torch.manual_seed`` changes (all of them, when CUDA is up)."""
    return list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []


def make_optimizer(head: nn.Module, lr: float) -> torch.optim.Optimizer:
    """AdamW with weight decay on weight matrices only (same rule for every fusion method)."""
    params = [p for p in head.parameters() if p.requires_grad]
    groups = [
        {"params": [p for p in params if p.ndim >= 2], "weight_decay": WEIGHT_DECAY},
        {"params": [p for p in params if p.ndim < 2], "weight_decay": 0.0},
    ]
    return torch.optim.AdamW([g for g in groups if g["params"]], lr=lr)


@torch.no_grad()
def evaluate(
    head: nn.Module, view: RowView, batch_size: int, device: torch.device
) -> tuple[float, float, np.ndarray]:
    """Return ``(mean cross-entropy, error rate, predictions in view order)``.

    The loss is ``nan``/``inf`` when the head or the inputs are not finite; callers check it,
    because ``argmax`` over NaN logits still yields (meaningless) class 0 predictions.
    """
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
    """Train a fresh head and return it restored to its best tuning epoch.

    Stops early after ``patience`` consecutive epochs without improving
    ``(tuning_error, tuning_loss)``, or as soon as a loss is not finite (``diverged``). A
    diverged outcome with ``best_epoch == 0`` returns a head that must not be used.
    """
    for name, value in (("epochs", epochs), ("patience", patience), ("batch_size", batch_size)):
        if value < 1:
            raise ValueError(f"{name} must be >= 1, got {value}")
    if not (math.isfinite(lr) and lr > 0):
        raise ValueError(f"lr must be a finite positive number, got {lr}")
    if len(train_view) == 0:
        raise ValueError("cannot train on an empty split")
    with torch.random.fork_rng(devices=_rng_devices()):
        torch.manual_seed(seed)
        head = make_head()
    head.to(device)
    optimizer = make_optimizer(head, lr)
    shuffle_rng = np.random.default_rng(seed)

    curve: list[dict[str, float]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_error, best_loss, best_epoch = math.inf, math.inf, 0
    epochs_without_improvement = 0
    diverged = False
    train_seconds = tuning_eval_seconds = 0.0

    for epoch in range(1, epochs + 1):
        _sync(device)
        start_time = time.perf_counter()
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
        _sync(device)
        eval_start = time.perf_counter()
        train_seconds += eval_start - start_time

        tuning_loss, tuning_error, _ = evaluate(head, tuning_view, batch_size, device)
        tuning_eval_seconds += time.perf_counter() - eval_start  # evaluate() syncs via .item()
        mean_train_loss = float(train_loss.item()) / len(train_view)
        curve.append(
            {
                "epoch": float(epoch),
                "train_loss": mean_train_loss,
                "tuning_loss": tuning_loss,
                "tuning_error": tuning_error,
            }
        )
        if not (math.isfinite(mean_train_loss) and math.isfinite(tuning_loss)):
            diverged = True
            break
        if (tuning_error, tuning_loss) < (best_error, best_loss):
            best_error, best_loss, best_epoch = tuning_error, tuning_loss, epoch
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is not None:
        head.load_state_dict(best_state)
    return head, TrainOutcome(
        learning_curve=curve,
        best_epoch=best_epoch,
        best_tuning_error=best_error,
        best_tuning_loss=best_loss,
        train_seconds=train_seconds,
        tuning_eval_seconds=tuning_eval_seconds,
        diverged=diverged,
    )
