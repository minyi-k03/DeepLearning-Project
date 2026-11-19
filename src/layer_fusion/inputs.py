"""Input construction: MNLI [SEP] pairing, IMDB 앞+뒤 자르기 (ADR 0006), buckets, view ratio."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from layer_fusion.contracts import DocumentRecord, InputSetting, LengthBucket
from layer_fusion.layers import LOCAL_WINDOW_HALF

# Head+tail content token budgets (excluding special tokens [CLS], [SEP])
TRUNCATION_BUDGETS: dict[InputSetting, tuple[int, int, int]] = {
    # (head_tokens, tail_tokens, total_with_special_tokens)
    InputSetting.L128: (32, 94, 128),
    InputSetting.L512: (128, 382, 512),
}


def assign_length_bucket(token_length: int) -> LengthBucket:
    """Assign IMDB token length to <=512, 512-1024, or >1024 (SPEC.md §보조 단위 테스트)."""
    if token_length < 1:
        raise ValueError(f"token_length must be positive, got {token_length}")
    if token_length <= 512:
        return LengthBucket.LE_512
    if token_length <= 1024:
        return LengthBucket.BETWEEN_512_1024
    return LengthBucket.GT_1024


def compute_view_ratio(token_length: int, window_half: int = LOCAL_WINDOW_HALF) -> float:
    """Compute average local-layer view ratio across tokens in a document of `token_length`.

    Each token i in [0, L-1] attends to [max(0, i - window_half), min(L-1, i + window_half)].
    For L=128 and window_half=64 (129-token window), the position-averaged ratio is ~0.7539.
    """
    if token_length < 1:
        raise ValueError(f"token_length must be positive, got {token_length}")
    total_visible = 0
    for i in range(token_length):
        left = max(0, i - window_half)
        right = min(token_length - 1, i + window_half)
        total_visible += right - left + 1
    return total_visible / (token_length * token_length)


def max_length_cap(
    input_setting: InputSetting | None, lfull_max_length_cap: int | None
) -> int | None:
    """Fixed per-setting length recorded in CacheMetadata.max_length_cap (never data-dependent).

    L128/L512 -> 128/512, Lfull -> the configured Lfull cap, MNLI/SST-2 (no setting) -> None.
    """
    if input_setting in TRUNCATION_BUDGETS:
        return TRUNCATION_BUDGETS[input_setting][2]
    if input_setting is InputSetting.LFULL:
        return lfull_max_length_cap
    return None


def truncation_rule_name(input_setting: InputSetting | None) -> str:
    """Name recorded in CacheMetadata.truncation_rule for an input setting (T-04b uses this)."""
    if input_setting in TRUNCATION_BUDGETS:
        head, tail, _ = TRUNCATION_BUDGETS[input_setting]
        return f"head_{head}_tail_{tail}"
    return "none"


@dataclass(frozen=True)
class TokenizedInputs:
    """Per-document model inputs, all lists aligned with the input record order.

    ``input_ids`` are unpadded (padding happens per batch in the extractor). ``token_lengths`` are
    lengths *before* truncation with special tokens; ``length_buckets`` use them (IMDB only).
    ``view_ratios`` are local-layer view ratios of the actual model input (SPEC US 18).
    """

    doc_ids: list[str]
    input_ids: list[list[int]]
    token_lengths: list[int]
    input_lengths: list[int]
    length_buckets: list[LengthBucket] | None
    truncated: list[bool]
    truncated_doc_ratio: float
    view_ratios: list[float]


def _batch_content_ids(tokenizer: Any, texts: Sequence[str]) -> list[list[int]]:
    """Tokenize many texts in one call (MNLI has ~390k pairs) without special tokens."""
    if not texts:
        return []
    return [list(ids) for ids in tokenizer(list(texts), add_special_tokens=False)["input_ids"]]


def build_tokenized_inputs(
    records: list[DocumentRecord],
    tokenizer: Any,
    input_setting: InputSetting | None = None,
    lfull_max_length_cap: int | None = None,
) -> TokenizedInputs:
    """Tokenize documents: [SEP]-joined pairs for MNLI, 앞+뒤 자르기 for IMDB (GLOSSARY).

    - ``input_setting is None`` (MNLI / SST-2): ``[CLS] a [SEP]`` or ``[CLS] a [SEP] b [SEP]``,
      never truncated (SPEC US 2).
    - L128 / L512: keep the first ``head`` and last ``tail`` content tokens so the total with
      special tokens is exactly 128 / 512 (ADR 0006, SPEC US 13).
    - Lfull: no truncation up to ``lfull_max_length_cap``. ADR 0006 keeps the 4,096 cap and
      asks for the truncated-document ratio, but does not say which part of a longer document
      to keep, so such a document raises instead of guessing (team agreement on PR #8; the
      observed IMDB maximum is 3,270 tokens).
    """
    if input_setting is InputSetting.LFULL and lfull_max_length_cap is None:
        raise ValueError("Lfull requires lfull_max_length_cap (configs/week1_env.yaml)")
    if input_setting is not None:
        for record in records:
            if record.text_b is not None:
                raise ValueError(
                    f"{input_setting.value} expects a single text per document, "
                    f"but {record.doc_id!r} has text_b"
                )

    cls_id, sep_id = tokenizer.cls_token_id, tokenizer.sep_token_id
    ids_a_all = _batch_content_ids(tokenizer, [r.text_a for r in records])
    ids_b_iter = iter(
        _batch_content_ids(tokenizer, [r.text_b for r in records if r.text_b is not None])
    )

    input_ids: list[list[int]] = []
    token_lengths: list[int] = []
    truncated: list[bool] = []
    for record, ids_a in zip(records, ids_a_all, strict=True):
        if record.text_b is not None:
            full = [cls_id, *ids_a, sep_id, *next(ids_b_iter), sep_id]
        else:
            full = [cls_id, *ids_a, sep_id]
        # Untruncated length with special tokens, taken before any cut below.
        token_lengths.append(len(full))

        cut = False
        if input_setting in TRUNCATION_BUDGETS:
            head, tail, total = TRUNCATION_BUDGETS[input_setting]
            if len(full) > total:
                full = [cls_id, *ids_a[:head], *ids_a[len(ids_a) - tail :], sep_id]
                cut = True
        elif input_setting is InputSetting.LFULL and len(full) > lfull_max_length_cap:
            raise ValueError(
                f"{record.doc_id!r} has {len(full)} tokens > Lfull cap {lfull_max_length_cap}; "
                "ADR 0006 keeps the cap but does not decide which part to keep"
            )
        input_ids.append(full)
        truncated.append(cut)

    input_lengths = [len(ids) for ids in input_ids]
    return TokenizedInputs(
        doc_ids=[r.doc_id for r in records],
        input_ids=input_ids,
        token_lengths=token_lengths,
        input_lengths=input_lengths,
        length_buckets=(
            [assign_length_bucket(n) for n in token_lengths] if input_setting is not None else None
        ),
        truncated=truncated,
        truncated_doc_ratio=(sum(truncated) / len(truncated)) if truncated else 0.0,
        view_ratios=[compute_view_ratio(n) for n in input_lengths],
    )
