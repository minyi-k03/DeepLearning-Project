"""Auxiliary unit tests from SPEC.md §보조 단위 테스트 and input construction (T-03b)."""

import pytest

from layer_fusion.contracts import DocumentRecord, InputSetting, LengthBucket
from layer_fusion.inputs import (
    TokenizedInputs,
    assign_length_bucket,
    build_tokenized_inputs,
    compute_view_ratio,
    max_length_cap,
    truncation_rule_name,
)


class FakeTokenizer:
    """Whitespace tokenizer: word 'w<k>' -> id k + 10. CLS=1, SEP=2. No HF download needed.

    Like HF tokenizers it accepts one text or a list of texts, and counts calls so tests can
    check that documents are tokenized in batches.
    """

    cls_token_id = 1
    sep_token_id = 2

    def __init__(self) -> None:
        self.calls = 0

    def _encode(self, text: str, add_special_tokens: bool) -> list[int]:
        ids = [int(w[1:]) + 10 for w in text.split()]
        if add_special_tokens:
            ids = [self.cls_token_id, *ids, self.sep_token_id]
        return ids

    def __call__(
        self, text: str | list[str], add_special_tokens: bool = True
    ) -> dict[str, list[int] | list[list[int]]]:
        self.calls += 1
        if isinstance(text, list):
            return {"input_ids": [self._encode(t, add_special_tokens) for t in text]}
        return {"input_ids": self._encode(text, add_special_tokens)}


def _doc(n_words: int, doc_id: str = "d0", text_b: str | None = None) -> DocumentRecord:
    return DocumentRecord(
        doc_id=doc_id, text_a=" ".join(f"w{k}" for k in range(n_words)), label=0, text_b=text_b
    )


def test_length_bucket_boundaries() -> None:
    assert assign_length_bucket(512) == LengthBucket.LE_512
    assert assign_length_bucket(513) == LengthBucket.BETWEEN_512_1024
    assert assign_length_bucket(1024) == LengthBucket.BETWEEN_512_1024
    assert assign_length_bucket(1025) == LengthBucket.GT_1024


def test_view_ratio_at_128_matches_adr_0006() -> None:
    ratio_128 = compute_view_ratio(128)
    assert abs(ratio_128 - 0.754) < 0.001


@pytest.mark.parametrize(
    ("setting", "head", "tail", "total"),
    [(InputSetting.L128, 32, 94, 128), (InputSetting.L512, 128, 382, 512)],
)
def test_head_tail_truncation_budgets(
    setting: InputSetting, head: int, tail: int, total: int
) -> None:
    """L128 = 32 + 94 + special tokens (128), L512 = 128 + 382 + special tokens (512)."""
    n_words = 2000
    out = build_tokenized_inputs([_doc(n_words)], FakeTokenizer(), input_setting=setting)
    ids = out.input_ids[0]
    content = [k + 10 for k in range(n_words)]

    assert len(ids) == total
    assert ids[0] == FakeTokenizer.cls_token_id and ids[-1] == FakeTokenizer.sep_token_id
    assert ids[1 : 1 + head] == content[:head]
    assert ids[1 + head : -1] == content[-tail:]
    assert out.truncated == [True]
    assert out.token_lengths == [n_words + 2]
    assert out.input_lengths == [total]


def test_short_doc_is_identical_between_l512_and_lfull() -> None:
    docs = [_doc(300)]
    l512 = build_tokenized_inputs(docs, FakeTokenizer(), InputSetting.L512)
    lfull = build_tokenized_inputs(docs, FakeTokenizer(), InputSetting.LFULL, 4096)
    assert l512.input_ids == lfull.input_ids
    assert l512.truncated == lfull.truncated == [False]


def test_every_imdb_doc_gets_token_length_and_bucket() -> None:
    docs = [_doc(100, "a"), _doc(700, "b"), _doc(2000, "c")]
    out = build_tokenized_inputs(docs, FakeTokenizer(), InputSetting.LFULL, 4096)
    assert out.doc_ids == ["a", "b", "c"]
    assert out.token_lengths == [102, 702, 2002]
    assert out.length_buckets == [
        LengthBucket.LE_512,
        LengthBucket.BETWEEN_512_1024,
        LengthBucket.GT_1024,
    ]
    assert out.truncated_doc_ratio == 0.0


def test_truncated_doc_ratio_and_buckets_use_untruncated_length() -> None:
    docs = [_doc(50, "short"), _doc(2000, "long")]
    out = build_tokenized_inputs(docs, FakeTokenizer(), InputSetting.L128)
    assert out.truncated == [False, True]
    assert out.truncated_doc_ratio == 0.5
    assert out.length_buckets == [LengthBucket.LE_512, LengthBucket.GT_1024]


def test_mnli_pair_is_single_sep_joined_input() -> None:
    doc = DocumentRecord(doc_id="m", text_a="w1 w2", label=1, text_b="w3")
    out = build_tokenized_inputs([doc], FakeTokenizer())
    assert out.input_ids == [[1, 11, 12, 2, 13, 2]]
    assert out.length_buckets is None
    assert out.truncated == [False]


def test_lfull_requires_cap() -> None:
    with pytest.raises(ValueError, match="lfull_max_length_cap"):
        build_tokenized_inputs([_doc(10)], FakeTokenizer(), InputSetting.LFULL)


def test_lfull_over_cap_raises_because_kept_part_is_undecided() -> None:
    """ADR 0006 keeps the cap but not which part to keep; refuse instead of guessing."""
    with pytest.raises(ValueError, match="which part to keep"):
        build_tokenized_inputs([_doc(5000)], FakeTokenizer(), InputSetting.LFULL, 4096)


def test_imdb_settings_reject_paired_text() -> None:
    with pytest.raises(ValueError, match="single"):
        build_tokenized_inputs([_doc(10, text_b="w1")], FakeTokenizer(), InputSetting.L128)


def test_truncation_rule_names() -> None:
    assert truncation_rule_name(InputSetting.L128) == "head_32_tail_94"
    assert truncation_rule_name(InputSetting.L512) == "head_128_tail_382"
    assert truncation_rule_name(InputSetting.LFULL) == "none"
    assert truncation_rule_name(None) == "none"


def test_result_is_a_typed_object_with_view_ratios() -> None:
    """Each document also gets its local-layer view ratio (SPEC US 18, ADR 0006)."""
    docs = [_doc(126, "short"), _doc(2000, "long")]
    out = build_tokenized_inputs(docs, FakeTokenizer(), InputSetting.L128)
    assert isinstance(out, TokenizedInputs)
    assert out.view_ratios == [compute_view_ratio(128), compute_view_ratio(128)]
    assert abs(out.view_ratios[0] - 0.754) < 0.001
    mnli = build_tokenized_inputs(
        [DocumentRecord(doc_id="m", text_a="w1 w2", label=0, text_b="w3")], FakeTokenizer()
    )
    assert mnli.view_ratios == [1.0]  # 6 tokens <= 129-token window


def test_documents_are_tokenized_in_batches() -> None:
    """Large datasets (MNLI ~390k) must not call the tokenizer once per document."""
    tokenizer = FakeTokenizer()
    pairs = [
        DocumentRecord(doc_id=f"m{i}", text_a="w1 w2", label=0, text_b="w3") for i in range(50)
    ]
    build_tokenized_inputs(pairs, tokenizer)
    assert tokenizer.calls == 2  # one batch for text_a, one for text_b


def test_max_length_cap_is_fixed_per_setting_not_by_data() -> None:
    assert max_length_cap(InputSetting.L128, 4096) == 128
    assert max_length_cap(InputSetting.L512, 4096) == 512
    assert max_length_cap(InputSetting.LFULL, 4096) == 4096
    assert max_length_cap(None, 4096) is None
