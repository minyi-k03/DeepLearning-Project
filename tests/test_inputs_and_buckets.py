"""Auxiliary unit tests from SPEC.md §보조 단위 테스트 and input construction (T-03b)."""

import pytest

from layer_fusion.contracts import DocumentRecord, InputSetting, LengthBucket
from layer_fusion.inputs import (
    assign_length_bucket,
    build_tokenized_inputs,
    compute_view_ratio,
    truncation_rule_name,
)


class FakeTokenizer:
    """Whitespace tokenizer: word 'w<k>' -> id k + 10. CLS=1, SEP=2. No HF download needed."""

    cls_token_id = 1
    sep_token_id = 2

    def __call__(self, text: str, add_special_tokens: bool = True) -> dict[str, list[int]]:
        ids = [int(w[1:]) + 10 for w in text.split()]
        if add_special_tokens:
            ids = [self.cls_token_id, *ids, self.sep_token_id]
        return {"input_ids": ids}


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
    ids = out["input_ids"][0]
    content = [k + 10 for k in range(n_words)]

    assert len(ids) == total
    assert ids[0] == FakeTokenizer.cls_token_id and ids[-1] == FakeTokenizer.sep_token_id
    assert ids[1 : 1 + head] == content[:head]
    assert ids[1 + head : -1] == content[-tail:]
    assert out["truncated"] == [True]
    assert out["token_lengths"] == [n_words + 2]
    assert out["input_lengths"] == [total]


def test_short_doc_is_identical_between_l512_and_lfull() -> None:
    docs = [_doc(300)]
    l512 = build_tokenized_inputs(docs, FakeTokenizer(), InputSetting.L512)
    lfull = build_tokenized_inputs(docs, FakeTokenizer(), InputSetting.LFULL, 4096)
    assert l512["input_ids"] == lfull["input_ids"]
    assert l512["truncated"] == lfull["truncated"] == [False]


def test_every_imdb_doc_gets_token_length_and_bucket() -> None:
    docs = [_doc(100, "a"), _doc(700, "b"), _doc(2000, "c")]
    out = build_tokenized_inputs(docs, FakeTokenizer(), InputSetting.LFULL, 4096)
    assert out["doc_ids"] == ["a", "b", "c"]
    assert out["token_lengths"] == [102, 702, 2002]
    assert out["length_buckets"] == [
        LengthBucket.LE_512,
        LengthBucket.BETWEEN_512_1024,
        LengthBucket.GT_1024,
    ]
    assert out["truncated_doc_ratio"] == 0.0


def test_truncated_doc_ratio_and_buckets_use_untruncated_length() -> None:
    docs = [_doc(50, "short"), _doc(2000, "long")]
    out = build_tokenized_inputs(docs, FakeTokenizer(), InputSetting.L128)
    assert out["truncated"] == [False, True]
    assert out["truncated_doc_ratio"] == 0.5
    assert out["length_buckets"] == [LengthBucket.LE_512, LengthBucket.GT_1024]


def test_mnli_pair_is_single_sep_joined_input() -> None:
    doc = DocumentRecord(doc_id="m", text_a="w1 w2", label=1, text_b="w3")
    out = build_tokenized_inputs([doc], FakeTokenizer())
    assert out["input_ids"] == [[1, 11, 12, 2, 13, 2]]
    assert out["length_buckets"] is None
    assert out["truncated"] == [False]


def test_lfull_requires_cap_and_refuses_undecided_over_cap_truncation() -> None:
    with pytest.raises(ValueError, match="cap"):
        build_tokenized_inputs([_doc(10)], FakeTokenizer(), InputSetting.LFULL)
    with pytest.raises(ValueError, match="ADR 0006"):
        build_tokenized_inputs([_doc(5000)], FakeTokenizer(), InputSetting.LFULL, 4096)


def test_imdb_settings_reject_paired_text() -> None:
    with pytest.raises(ValueError, match="single"):
        build_tokenized_inputs([_doc(10, text_b="w1")], FakeTokenizer(), InputSetting.L128)


def test_truncation_rule_names() -> None:
    assert truncation_rule_name(InputSetting.L128) == "head_32_tail_94"
    assert truncation_rule_name(InputSetting.L512) == "head_128_tail_382"
    assert truncation_rule_name(InputSetting.LFULL) == "none"
    assert truncation_rule_name(None) == "none"
