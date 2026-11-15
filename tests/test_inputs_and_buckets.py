"""Auxiliary unit tests from SPEC.md §보조 단위 테스트."""

import pytest

from layer_fusion.contracts import LengthBucket
from layer_fusion.inputs import assign_length_bucket, compute_view_ratio


def test_length_bucket_boundaries() -> None:
    assert assign_length_bucket(512) == LengthBucket.LE_512
    assert assign_length_bucket(513) == LengthBucket.BETWEEN_512_1024
    assert assign_length_bucket(1024) == LengthBucket.BETWEEN_512_1024
    assert assign_length_bucket(1025) == LengthBucket.GT_1024


def test_view_ratio_at_128_matches_adr_0006() -> None:
    ratio_128 = compute_view_ratio(128)
    assert abs(ratio_128 - 0.754) < 0.001


@pytest.mark.skip(reason="TODO(T-03): Enable after build_tokenized_inputs is implemented")
def test_head_tail_truncation_budgets() -> None:
    """L128 = 32 + 94 + special tokens (128), L512 = 128 + 382 + special tokens (512)."""
    raise NotImplementedError
