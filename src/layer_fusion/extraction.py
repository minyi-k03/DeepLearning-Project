"""ModernBERT-base 23-layer vector extraction and cache serialization (SPEC.md, ADR 0005)."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from layer_fusion.contracts import CacheMetadata, DocumentRecord, InputSetting
from layer_fusion.inputs import (
    TRUNCATION_BUDGETS,
    TokenizedInputs,
    build_tokenized_inputs,
    truncation_rule_name,
)
from layer_fusion.layers import HIDDEN_SIZE, NUM_LAYER_VECTORS
from layer_fusion.providers import (
    CACHE_METADATA_FILE,
    DOC_IDS_FILE,
    GENRE_SPLITS_FILE,
    LABELS_FILE,
    PARTIAL_SUFFIX,
    TOKEN_LENGTHS_FILE,
    VECTORS_FILE,
    CacheConfigMismatchError,
    cache_metadata_from_dict,
    cache_metadata_to_dict,
)

FLOAT16_MAX = float(np.finfo(np.float16).max)  # 65,504

# Short inputs of different lengths so the check runs on a padded batch.
DEFAULT_CONTRACT_TEXTS: tuple[str, ...] = (
    "A short sentence.",
    "The movie was long, but the ending made the whole evening worth it.",
    "Premise and hypothesis are joined into one input for MNLI.",
)


def masked_token_mean(hidden: Any, attention_mask: Any) -> Any:
    """Mean over tokens where ``attention_mask`` is 1: [B, T, H] x [B, T] -> [B, H].

    Padding positions are excluded (SPEC US 23). Whether special tokens are pooled is decided
    by the caller through the mask (``include_special_tokens_in_pool``: agreed as true on PR #8,
    docs/03-plan.md §8; becomes final when configs/week1_env.yaml is filled in T-04b PR 2).
    """
    mask = attention_mask.to(hidden.dtype).unsqueeze(-1)
    summed = (hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


def check_final_norm_contract(
    model_name: str,
    attn_implementation: str,
    device: str = "cpu",
    torch_compile: bool = False,
    texts: list[str] | tuple[str, ...] | None = None,
    atol: float = 1e-5,
) -> dict[str, Any]:
    """Week 1 check (ADR 0005, SPEC US 24): report how the 23rd hidden state is defined.

    Captures the raw output of the last encoder layer with a forward hook and compares the
    last entry of ``hidden_states`` against ``final_norm(raw)``, ``last_hidden_state`` and
    the raw output itself, both per token and as padding-excluded token means.
    """
    import torch
    import transformers
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, attn_implementation=attn_implementation)
    model = model.to(device).eval()

    captured: dict[str, Any] = {}

    def _capture(_module: Any, _inputs: Any, output: Any) -> None:
        captured["raw"] = output[0] if isinstance(output, tuple) else output

    handle = model.layers[-1].register_forward_hook(_capture)
    forward = torch.compile(model) if torch_compile else model
    enc = tokenizer(
        list(texts or DEFAULT_CONTRACT_TEXTS), padding=True, return_tensors="pt"
    ).to(device)
    try:
        with torch.no_grad():
            out = forward(**enc, output_hidden_states=True)
            hidden_states = out.hidden_states
            last = hidden_states[-1]
            raw = captured["raw"]
            normed_raw = model.final_norm(raw)
    finally:
        handle.remove()

    mask = enc["attention_mask"]
    expected_slots = model.config.num_hidden_layers + 1
    last_eq_normed = bool(torch.allclose(last, normed_raw, atol=atol))
    last_eq_lhs = bool(torch.allclose(last, out.last_hidden_state, atol=atol))
    last_eq_raw = bool(torch.allclose(last, raw, atol=atol))
    token_mean_ok = bool(
        torch.allclose(
            masked_token_mean(last, mask), masked_token_mean(normed_raw, mask), atol=atol
        )
    )
    slot_norms = [
        round(float(masked_token_mean(h, mask).norm(dim=-1).mean()), 2) for h in hidden_states
    ]

    passed = (
        len(hidden_states) == expected_slots
        and last_eq_normed
        and last_eq_lhs
        and token_mean_ok
        and not last_eq_raw
    )
    return {
        "model_name": model_name,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "device": str(device),
        "attn_implementation": attn_implementation,
        "torch_compile": torch_compile,
        "atol": atol,
        "num_docs": int(mask.shape[0]),
        "num_hidden_states": len(hidden_states),
        "expected_hidden_states": expected_slots,
        "last_equals_final_norm_of_raw": last_eq_normed,
        "last_equals_last_hidden_state": last_eq_lhs,
        "last_equals_raw": last_eq_raw,
        "token_mean_matches": token_mean_ok,
        "slot_mean_norms": slot_norms,
        "passed": passed,
    }


def verify_final_norm_contract(
    model_name: str,
    attn_implementation: str,
    device: str = "cpu",
) -> bool:
    """Week 1 check: confirm slot 22 (23rd hidden state) equals final-normed output."""
    return bool(check_final_norm_contract(model_name, attn_implementation, device)["passed"])


def check_finite(values: np.ndarray) -> None:
    """Raise if layer vectors contain NaN or inf (checked per batch for float32 and float16).

    Per batch because a running ``max(max_abs, nan)`` keeps the old value, so NaN would
    otherwise vanish from the summary and a corrupt cache could be written.
    """
    if not np.isfinite(values).all():
        bad = int(values.size - np.count_nonzero(np.isfinite(values)))
        raise ValueError(f"layer vectors contain {bad} non-finite (NaN/inf) values")


def check_float16_range(max_abs: float) -> None:
    """Raise if layer-vector values do not fit float16 (precision is reported, not judged)."""
    if not np.isfinite(max_abs):
        raise ValueError(f"max |value| is {max_abs}; NaN/inf cannot be stored in a cache")
    if max_abs > FLOAT16_MAX:
        raise ValueError(
            f"max |value| {max_abs:.1f} exceeds the float16 maximum {FLOAT16_MAX:.0f}; "
            "store this cache as float32 instead"
        )


def _max_length_cap(
    input_setting: InputSetting | None, lfull_max_length_cap: int | None
) -> int | None:
    """Fixed per-setting length recorded in CacheMetadata (never data-dependent)."""
    if input_setting in TRUNCATION_BUDGETS:
        return TRUNCATION_BUDGETS[input_setting][2]
    if input_setting is InputSetting.LFULL:
        return lfull_max_length_cap
    return None


@dataclass
class ExtractionResult:
    """Layer vectors in input-record order plus what is needed to save and audit them."""

    vectors: np.ndarray                      # [N, 23, 768] in metadata.dtype
    doc_ids: list[str]
    labels: list[int]
    genre_splits: list[str | None] | None    # None unless some record has a genre (MNLI)
    metadata: CacheMetadata
    tokenized: TokenizedInputs
    report: dict[str, Any] = field(default_factory=dict)


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
    device: str = "cpu",
    out: np.ndarray | None = None,
    model: Any = None,
    tokenizer: Any = None,
) -> ExtractionResult:
    """Extract [N, 23, 768] padding-excluded token means for each document (SPEC US 21–23).

    Documents are batched longest-first to limit padding and written back in the original
    record order (US 27). Each of the 23 hidden states is pooled right away, so memory stays
    O(batch x length x 768) even for Lfull. ``out`` may be a pre-allocated array such as an
    ``np.lib.format.open_memmap`` so the MNLI train cache never has to fit in RAM. A float16
    range error stops extraction mid-way, so a caller writing ``out`` to disk should write to a
    temporary path and keep it only on success (the T-04b PR 2 CLI does this).
    ``include_special_tokens_in_pool`` must be given explicitly (configs/week1_env.yaml).
    """
    import torch
    import transformers
    from transformers import AutoModel, AutoTokenizer

    if transformers.__version__ != transformers_version:
        raise ValueError(
            f"transformers {transformers.__version__} is installed, but the cache must be built "
            f"with the locked {transformers_version} (ADR 0005)"
        )
    if include_special_tokens_in_pool is None:
        raise ValueError("include_special_tokens_in_pool must be set (configs/week1_env.yaml)")
    if dtype not in ("float32", "float16"):
        raise ValueError(f"dtype must be float32 or float16, got {dtype!r}")

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    if model is None:
        model = AutoModel.from_pretrained(model_name, attn_implementation=attn_implementation)
    else:
        # An injected model must match what CacheMetadata will record (ADR 0005).
        model_attn = getattr(getattr(model, "config", None), "_attn_implementation", None)
        if model_attn is not None and model_attn != attn_implementation:
            raise ValueError(
                f"model uses attn_implementation={model_attn!r}, but {attn_implementation!r} "
                "would be recorded in the cache metadata"
            )
    model = model.to(device).eval()
    forward = torch.compile(model) if torch_compile else model

    tokenized = build_tokenized_inputs(records, tokenizer, input_setting, lfull_max_length_cap)
    n_docs = len(records)
    if out is None:
        out = np.empty((n_docs, NUM_LAYER_VECTORS, HIDDEN_SIZE), dtype=dtype)
    if out.shape != (n_docs, NUM_LAYER_VECTORS, HIDDEN_SIZE) or out.dtype != np.dtype(dtype):
        raise ValueError(f"out must be {dtype} [{n_docs}, 23, 768], got {out.dtype} {out.shape}")

    pad_id = tokenizer.pad_token_id
    special_ids = torch.tensor([tokenizer.cls_token_id, tokenizer.sep_token_id], device=device)
    order = sorted(range(n_docs), key=lambda i: tokenized.input_lengths[i], reverse=True)
    max_abs = 0.0
    max_fp16_error = 0.0
    with torch.no_grad():
        for start in range(0, n_docs, batch_size):
            idx = order[start : start + batch_size]
            width = max(tokenized.input_lengths[i] for i in idx)
            ids = torch.full((len(idx), width), pad_id, dtype=torch.long)
            attention = torch.zeros((len(idx), width), dtype=torch.long)
            for row, i in enumerate(idx):
                seq = tokenized.input_ids[i]
                ids[row, : len(seq)] = torch.tensor(seq)
                attention[row, : len(seq)] = 1
            ids, attention = ids.to(device), attention.to(device)
            pool_mask = attention
            if not include_special_tokens_in_pool:
                pool_mask = attention * (~torch.isin(ids, special_ids)).long()

            hidden_states = forward(
                input_ids=ids, attention_mask=attention, output_hidden_states=True
            ).hidden_states
            if len(hidden_states) != NUM_LAYER_VECTORS:
                raise ValueError(
                    f"expected {NUM_LAYER_VECTORS} hidden states, got {len(hidden_states)}"
                )
            pooled = torch.stack(
                [masked_token_mean(h.float(), pool_mask) for h in hidden_states], dim=1
            ).cpu().numpy()  # [B, 23, 768] float32

            check_finite(pooled)
            batch_max = float(np.abs(pooled).max())
            max_abs = max(max_abs, batch_max)
            # Measured for float32 runs too: the float16 decision is made from such a run.
            with np.errstate(over="ignore"):
                as_fp16 = pooled.astype(np.float16)
            max_fp16_error = max(
                max_fp16_error, float(np.abs(as_fp16.astype(np.float32) - pooled).max())
            )
            if dtype == "float16":
                check_float16_range(batch_max)
                out[idx] = as_fp16
            else:
                out[idx] = pooled

    metadata = CacheMetadata(
        model_name=model_name,
        transformers_version=transformers_version,
        attn_implementation=attn_implementation,
        torch_compile=torch_compile,
        input_setting=input_setting,
        max_length_cap=_max_length_cap(input_setting, lfull_max_length_cap),
        truncation_rule=truncation_rule_name(input_setting),
        include_special_tokens_in_pool=include_special_tokens_in_pool,
        dtype=dtype,
    )
    report = {
        "num_docs": n_docs,
        "max_abs_value": round(max_abs, 4),
        "float16_round_trip_max_error": round(max_fp16_error, 6),
        "max_input_length": max(tokenized.input_lengths, default=0),
        "truncated_doc_ratio": tokenized.truncated_doc_ratio,
    }
    genres = [r.genre_split for r in records]
    return ExtractionResult(
        vectors=out,
        doc_ids=tokenized.doc_ids,
        labels=[r.label for r in records],
        # Same rule as FakeLayerVectorProvider: no genre_splits.json for SST-2/IMDB caches.
        genre_splits=genres if any(g is not None for g in genres) else None,
        metadata=metadata,
        tokenized=tokenized,
        report=report,
    )


def save_layer_cache(
    output_dir: str | Path,
    split_name: str,
    vectors: np.ndarray,
    doc_ids: list[str],
    labels: list[int],
    metadata: CacheMetadata,
    token_lengths: list[int] | None = None,
    genre_splits: list[str | None] | None = None,
) -> Path:
    """Save one split of a layer-vector cache: ``<output_dir>/<split>/vectors.npy`` + index files.

    ``<output_dir>/metadata.json`` holds the CacheMetadata shared by every split. A directory
    built under other metadata is rejected, an existing split (or a ``.partial`` left by a failed
    run) is never overwritten, and files appear under ``<split>`` only once all are written
    (AGENTS.md §7). Returns the split directory.
    """
    output_dir = Path(output_dir)
    if vectors.ndim != 3 or vectors.shape[1:] != (NUM_LAYER_VECTORS, HIDDEN_SIZE):
        raise ValueError(
            f"vectors must be [N, {NUM_LAYER_VECTORS}, {HIDDEN_SIZE}], got {vectors.shape}"
        )
    if vectors.dtype != np.dtype(metadata.dtype):
        raise ValueError(f"vectors dtype {vectors.dtype} != metadata dtype {metadata.dtype}")
    n = vectors.shape[0]
    if len(doc_ids) != n or len(labels) != n:
        raise ValueError(f"doc_ids ({len(doc_ids)}) and labels ({len(labels)}) must match {n} rows")
    for name, values in (("token_lengths", token_lengths), ("genre_splits", genre_splits)):
        if values is not None and len(values) != n:
            raise ValueError(f"{name} has {len(values)} entries, expected {n}")

    meta_path = output_dir / CACHE_METADATA_FILE
    if meta_path.is_file():
        existing = cache_metadata_from_dict(json.loads(meta_path.read_text("utf-8")))
        if existing != metadata:
            raise CacheConfigMismatchError(
                f"{output_dir} already holds a cache built with {existing}, not {metadata}"
            )
    split_dir = output_dir / split_name
    partial_dir = output_dir / f"{split_name}{PARTIAL_SUFFIX}"
    for existing_dir in (split_dir, partial_dir):
        if existing_dir.exists():
            raise FileExistsError(
                f"{existing_dir} already exists (complete or left by a failed run); "
                "refusing to overwrite — inspect and remove it by hand if it is stale"
            )

    # Write everything into <split>.partial and rename only when all files are written, so a
    # failed save never leaves something that looks like a usable cache (AGENTS.md §7).
    output_dir.mkdir(parents=True, exist_ok=True)
    partial_dir.mkdir()
    try:
        np.save(partial_dir / VECTORS_FILE, np.asarray(vectors))
        (partial_dir / DOC_IDS_FILE).write_text(json.dumps(list(doc_ids)), encoding="utf-8")
        np.save(partial_dir / LABELS_FILE, np.asarray(labels, dtype=np.int64))
        if token_lengths is not None:
            np.save(partial_dir / TOKEN_LENGTHS_FILE, np.asarray(token_lengths, dtype=np.int64))
        if genre_splits is not None:
            (partial_dir / GENRE_SPLITS_FILE).write_text(
                json.dumps(list(genre_splits)), encoding="utf-8"
            )
    except BaseException:
        shutil.rmtree(partial_dir, ignore_errors=True)  # only the folder this call created
        raise
    if not meta_path.is_file():
        meta_path.write_text(
            json.dumps(cache_metadata_to_dict(metadata), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    partial_dir.rename(split_dir)
    return split_dir
