"""Input construction: MNLI [SEP] pairing, IMDB head+tail truncation, buckets, view ratio."""

from __future__ import annotations

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


def build_tokenized_inputs(
    records: list[DocumentRecord],
    tokenizer: object,
    input_setting: InputSetting | None = None,
    lfull_max_length_cap: int | None = None,
) -> dict[str, object]:
    """Tokenize documents: [SEP]-joined pairs for MNLI, head+tail truncation for IMDB."""
    raise NotImplementedError("TODO: Implement tokenization & head+tail truncation (T-03)")
