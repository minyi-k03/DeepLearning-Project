"""ModernBERT-base 23-layer vector extraction and cache serialization (SPEC.md, ADR 0005)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from layer_fusion.contracts import CacheMetadata, DocumentRecord, InputSetting

# Short inputs of different lengths so the check runs on a padded batch.
DEFAULT_CONTRACT_TEXTS: tuple[str, ...] = (
    "A short sentence.",
    "The movie was long, but the ending made the whole evening worth it.",
    "Premise and hypothesis are joined into one input for MNLI.",
)


def masked_token_mean(hidden: Any, attention_mask: Any) -> Any:
    """Mean over tokens where ``attention_mask`` is 1: [B, T, H] x [B, T] -> [B, H].

    Padding positions are excluded (SPEC US 23). Whether special tokens are pooled is a
    separate, still-open decision (``include_special_tokens_in_pool``); this helper only
    follows the mask it is given.
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
