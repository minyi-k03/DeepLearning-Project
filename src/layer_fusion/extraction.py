"""ModernBERT-base 23-layer vector extraction and cache serialization (SPEC.md, ADR 0005)."""

from __future__ import annotations

from pathlib import Path

from layer_fusion.contracts import CacheMetadata, DocumentRecord, InputSetting


def verify_final_norm_contract(
    model_name: str,
    attn_implementation: str,
    device: str = "cpu",
) -> bool:
    """Week 1 check: confirm slot 22 (23rd hidden state) equals final-normed output."""
    raise NotImplementedError("TODO: Implement Week 1 final-norm check (T-02)")


def extract_layer_vectors(
    records: list[DocumentRecord],
    model_name: str,
    transformers_version: str,
    attn_implementation: str,
    torch_compile: bool,
    input_setting: InputSetting | None = None,
    lfull_max_length_cap: int | None = None,
    include_special_tokens_in_pool: bool | None = None,
    batch_size: int = 32,
    dtype: str = "float32",
) -> tuple[object, CacheMetadata]:
    """Extract [N, 23, 768] padding-excluded mean-pooled vectors for each document."""
    raise NotImplementedError("TODO: Implement ModernBERT extraction (T-04)")


def save_layer_cache(
    output_dir: str | Path,
    split_name: str,
    vectors: object,
    doc_ids: list[str],
    labels: list[int],
    metadata: CacheMetadata,
) -> Path:
    """Save extracted layer vectors and CacheMetadata to disk."""
    raise NotImplementedError("TODO: Implement cache writer (T-04)")
