"""Seam 3 tests: Extraction verification with real ModernBERT-base (SPEC.md §Seam ③).

Run after Week 1 transformers version / attention / compile lock (T-02, T-04).
"""

import pytest

pytestmark = pytest.mark.extraction

MODEL_NAME = "answerdotai/ModernBERT-base"


def _require_local_modernbert() -> None:
    """Skip unless torch/transformers are installed and weights are already in the HF cache.

    verify.py must not trigger a network download, so Seam 3 runs only where the model
    has been fetched once (e.g. via scripts/check_runtime_lock.py).
    """
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from huggingface_hub import try_to_load_from_cache

    if not isinstance(try_to_load_from_cache(MODEL_NAME, "model.safetensors"), str):
        pytest.skip(f"{MODEL_NAME} weights not in local HF cache")


def test_masked_token_mean_ignores_padding_positions() -> None:
    """Padding positions do not change the pooled vector, whatever their values are."""
    torch = pytest.importorskip("torch")
    from layer_fusion.extraction import masked_token_mean

    hidden = torch.zeros(2, 4, 3)
    hidden[0, :2] = torch.tensor([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
    hidden[1, :4] = 1.0
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])

    pooled = masked_token_mean(hidden, mask)
    hidden[0, 2:] = 1e6  # garbage in padding slots
    pooled_garbage = masked_token_mean(hidden, mask)

    assert torch.allclose(pooled[0], torch.tensor([2.0, 3.0, 4.0]))
    assert torch.allclose(pooled[1], torch.ones(3))
    assert torch.equal(pooled, pooled_garbage)


def test_23_layer_vectors_and_final_norm_contract() -> None:
    """23 vectors per doc and slot 22 matches final-normed token-mean output."""
    _require_local_modernbert()
    from layer_fusion.extraction import check_final_norm_contract, verify_final_norm_contract

    report = check_final_norm_contract(MODEL_NAME, attn_implementation="sdpa", device="cpu")

    assert report["num_hidden_states"] == 23
    assert report["last_equals_final_norm_of_raw"] is True
    assert report["last_equals_last_hidden_state"] is True
    assert report["token_mean_matches"] is True
    # Slot 22 must be the normed output, not the raw (pre-norm-scale) last layer output.
    assert report["last_equals_raw"] is False
    assert report["passed"] is True
    assert verify_final_norm_contract(MODEL_NAME, attn_implementation="sdpa") is True


@pytest.mark.skip(reason="TODO(T-04): Requires ModernBERT-base extraction")
def test_cache_shape_and_doc_id_label_order() -> None:
    """Cache shape is [N, 23, 768] and doc_id/label order matches source dataset."""
    raise NotImplementedError


@pytest.mark.skip(reason="TODO(T-04): Requires ModernBERT-base extraction")
def test_mean_pooling_excludes_padding_tokens() -> None:
    """Same document with different batch padding yields identical layer vectors."""
    raise NotImplementedError


def test_imdb_head_tail_truncation_lengths() -> None:
    """L128 = 32 + 94 (+ special -> 128), L512 = 128 + 382 (+ special -> 512)."""
    pytest.importorskip("transformers")
    from huggingface_hub import try_to_load_from_cache

    if not isinstance(try_to_load_from_cache(MODEL_NAME, "tokenizer.json"), str):
        pytest.skip(f"{MODEL_NAME} tokenizer not in local HF cache")
    from transformers import AutoTokenizer

    from layer_fusion.contracts import DocumentRecord, InputSetting
    from layer_fusion.inputs import build_tokenized_inputs

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    long_text = " ".join(f"word{i}" for i in range(3000))
    doc = DocumentRecord(doc_id="imdb-long", text_a=long_text, label=1)
    content = tokenizer(long_text, add_special_tokens=False)["input_ids"]

    for setting, head, tail, total in [
        (InputSetting.L128, 32, 94, 128),
        (InputSetting.L512, 128, 382, 512),
    ]:
        ids = build_tokenized_inputs([doc], tokenizer, input_setting=setting)["input_ids"][0]
        assert len(ids) == total
        assert ids[0] == tokenizer.cls_token_id and ids[-1] == tokenizer.sep_token_id
        assert ids[1 : 1 + head] == content[:head]
        assert ids[1 + head : -1] == content[-tail:]


def test_mnli_pair_matches_real_tokenizer_pair_encoding() -> None:
    """The manual [CLS] a [SEP] b [SEP] equals ModernBERT's own pair encoding (SPEC US 2)."""
    pytest.importorskip("transformers")
    from huggingface_hub import try_to_load_from_cache

    if not isinstance(try_to_load_from_cache(MODEL_NAME, "tokenizer.json"), str):
        pytest.skip(f"{MODEL_NAME} tokenizer not in local HF cache")
    from transformers import AutoTokenizer

    from layer_fusion.contracts import DocumentRecord
    from layer_fusion.inputs import build_tokenized_inputs

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    premise, hypothesis = "A man is playing a guitar on stage.", "Someone performs music."
    pair = DocumentRecord(doc_id="mnli-0", text_a=premise, label=0, text_b=hypothesis)
    built = build_tokenized_inputs([pair], tokenizer)["input_ids"][0]
    assert built == tokenizer(premise, hypothesis, add_special_tokens=True)["input_ids"]


@pytest.mark.skip(reason="TODO(T-04): Requires ModernBERT-base extraction")
def test_short_docs_under_512_identical_between_l512_and_lfull() -> None:
    """Documents <= 512 tokens produce identical vectors under L512 and Lfull."""
    raise NotImplementedError


@pytest.mark.skip(reason="TODO(T-04): Requires CacheProvider")
def test_cache_records_creation_metadata_and_rejects_mismatch() -> None:
    """Saved cache records model/version/length/truncation and rejects mismatched RunConfig."""
    raise NotImplementedError
