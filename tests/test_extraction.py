"""Seam 3 tests: Extraction verification with real ModernBERT-base (SPEC.md §Seam ③).

Run after Week 1 transformers version / attention / compile lock (T-02, T-04).
"""

import pytest

pytestmark = pytest.mark.extraction


@pytest.mark.skip(reason="TODO(T-02/T-04): Requires locked transformers 5.x + ModernBERT-base")
def test_23_layer_vectors_and_final_norm_contract() -> None:
    """23 vectors per doc and slot 22 matches final-normed token-mean output."""
    raise NotImplementedError


@pytest.mark.skip(reason="TODO(T-04): Requires ModernBERT-base extraction")
def test_cache_shape_and_doc_id_label_order() -> None:
    """Cache shape is [N, 23, 768] and doc_id/label order matches source dataset."""
    raise NotImplementedError


@pytest.mark.skip(reason="TODO(T-04): Requires ModernBERT-base extraction")
def test_mean_pooling_excludes_padding_tokens() -> None:
    """Same document with different batch padding yields identical layer vectors."""
    raise NotImplementedError


@pytest.mark.skip(reason="TODO(T-04): Requires ModernBERT-base tokenizer & extraction")
def test_imdb_head_tail_truncation_lengths() -> None:
    """L128 = 32 + 94 (+ special -> 128), L512 = 128 + 382 (+ special -> 512)."""
    raise NotImplementedError


@pytest.mark.skip(reason="TODO(T-04): Requires ModernBERT-base extraction")
def test_short_docs_under_512_identical_between_l512_and_lfull() -> None:
    """Documents <= 512 tokens produce identical vectors under L512 and Lfull."""
    raise NotImplementedError


@pytest.mark.skip(reason="TODO(T-04): Requires CacheProvider")
def test_cache_records_creation_metadata_and_rejects_mismatch() -> None:
    """Saved cache records model/version/length/truncation and rejects mismatched RunConfig."""
    raise NotImplementedError
