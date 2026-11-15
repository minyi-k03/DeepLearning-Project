"""Execution seam, sequential queue runner, and reproduction track runner.

References: docs/SPEC.md, ADR 0004, ADR 0009.
"""

from __future__ import annotations

from collections.abc import Sequence

from layer_fusion.contracts import ResultRecord, RunConfig
from layer_fusion.providers import LayerVectorProvider


def run_experiment(
    config: RunConfig,
    provider: LayerVectorProvider,
) -> ResultRecord:
    """Execution seam: (RunConfig, LayerVectorProvider) -> ResultRecord.

    Contract:
    - Validates provider metadata against config.expected_cache_meta.
    - Evaluates all lr_candidates strictly on the 10% tuning split (never eval split).
    - Trains with fixed epochs and early stopping across all fusion methods.
    - Evaluates on the eval split (MNLI reports both matched and mismatched error rates).
    """
    raise NotImplementedError("TODO: Implement execution seam run_experiment (T-06)")


def run_sequential_queue(
    configs: Sequence[RunConfig],
    provider_factory: object,
    output_dir: str,
) -> list[ResultRecord]:
    """Run main timed experiments sequentially, one at a time on GPU (ADR 0009)."""
    raise NotImplementedError("TODO: Implement sequential queue runner (T-06)")


def run_reproduction_track(
    dataset: str,
    seeds: Sequence[int],
    output_dir: str,
) -> list[ResultRecord]:
    """Run paper fine-tuning reproduction track (MNLI 89.1 / SST-2 96.0, x3 runs, ADR 0004)."""
    raise NotImplementedError("TODO: Implement reproduction track runner (T-06/T-08)")
