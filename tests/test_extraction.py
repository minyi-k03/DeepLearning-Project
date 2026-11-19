"""Seam 3 tests: Extraction verification with real ModernBERT-base (SPEC.md §Seam ③).

Run after Week 1 transformers version / attention / compile lock (T-02, T-04).
"""

import numpy as np
import pytest

pytestmark = [pytest.mark.extraction, pytest.mark.usefixtures("windows_file_locks")]

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


@pytest.fixture(scope="module")
def modernbert() -> tuple[object, object]:
    """Load ModernBERT-base once for the Seam 3 extraction tests (CPU, sdpa, eval mode)."""
    _require_local_modernbert()
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME, attn_implementation="sdpa").eval()
    return model, tokenizer


def _extract(modernbert: tuple[object, object], records: list, **overrides: object) -> object:
    import transformers

    from layer_fusion.extraction import extract_layer_vectors

    model, tokenizer = modernbert
    kwargs: dict[str, object] = {
        "model_name": MODEL_NAME,
        "transformers_version": transformers.__version__,
        "attn_implementation": "sdpa",
        "torch_compile": False,
        "include_special_tokens_in_pool": True,
        "batch_size": 2,
        "model": model,
        "tokenizer": tokenizer,
    }
    kwargs.update(overrides)
    return extract_layer_vectors(records, **kwargs)


def _docs(*texts: str) -> list:
    from layer_fusion.contracts import DocumentRecord

    return [DocumentRecord(doc_id=f"d{i}", text_a=t, label=i % 2) for i, t in enumerate(texts)]


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


def test_cache_shape_and_doc_id_label_order(modernbert, tmp_path) -> None:
    """Cache shape is [N, 23, 768] and doc_id/label order matches source dataset."""
    from layer_fusion.extraction import save_layer_cache
    from layer_fusion.providers import CacheProvider

    # Lengths deliberately unsorted: the extractor batches by length and must restore order.
    records = _docs("short one.", "a much longer sentence " * 20, "mid sized text here.", "x.")
    result = _extract(modernbert, records)
    assert result.vectors.shape == (4, 23, 768)
    save_layer_cache(
        tmp_path, "train", result.vectors, result.doc_ids, result.labels, result.metadata
    )
    batch = CacheProvider(str(tmp_path)).get_split("train")
    assert batch.vectors.shape == (4, 23, 768)
    assert batch.doc_ids == [r.doc_id for r in records]
    assert batch.labels.tolist() == [r.label for r in records]

    # Order restoration: each row equals that document extracted on its own.
    alone = _extract(modernbert, [records[2]])
    assert np.allclose(batch.vectors[2], alone.vectors[0], atol=1e-4)


def test_mean_pooling_excludes_padding_tokens(modernbert) -> None:
    """Same document with different batch padding yields identical layer vectors."""
    short, long = _docs("a short review.", "a long review that pads the batch " * 30)
    alone = _extract(modernbert, [short]).vectors[0]
    padded = _extract(modernbert, [short, long], batch_size=2).vectors[0]
    assert np.allclose(alone, padded, atol=1e-4)


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
        ids = build_tokenized_inputs([doc], tokenizer, input_setting=setting).input_ids[0]
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
    built = build_tokenized_inputs([pair], tokenizer).input_ids[0]
    assert built == tokenizer(premise, hypothesis, add_special_tokens=True)["input_ids"]


def test_short_docs_under_512_identical_between_l512_and_lfull(modernbert) -> None:
    """Documents <= 512 tokens produce identical vectors under L512 and Lfull."""
    from layer_fusion.contracts import InputSetting

    records = _docs("this movie was fine.", "an average review " * 40)
    l512 = _extract(modernbert, records, input_setting=InputSetting.L512)
    lfull = _extract(
        modernbert, records, input_setting=InputSetting.LFULL, lfull_max_length_cap=4096
    )
    assert max(l512.tokenized.token_lengths) <= 512
    assert np.allclose(l512.vectors, lfull.vectors, atol=1e-5)


def test_cache_records_creation_metadata_and_rejects_mismatch(modernbert, tmp_path) -> None:
    """Saved cache records model/version/length/truncation and rejects mismatched RunConfig."""
    from dataclasses import replace

    import transformers

    from layer_fusion.contracts import (
        BackboneCondition,
        DataAmount,
        DatasetName,
        FusionMethod,
        InputSetting,
        RunConfig,
    )
    from layer_fusion.extraction import save_layer_cache
    from layer_fusion.providers import CacheConfigMismatchError, CacheProvider

    result = _extract(modernbert, _docs("good film.", "bad film."), input_setting=InputSetting.L128)
    meta = result.metadata
    assert (meta.model_name, meta.transformers_version) == (MODEL_NAME, transformers.__version__)
    assert (meta.input_setting, meta.truncation_rule) == (InputSetting.L128, "head_32_tail_94")
    # The cap is the setting's fixed length, not the longest document seen in this batch,
    # otherwise two L128 caches of different data would never match.
    assert meta.max_length_cap == 128
    save_layer_cache(tmp_path, "train", result.vectors, result.doc_ids, result.labels, meta)

    provider = CacheProvider(str(tmp_path))
    assert provider.metadata == meta
    config = RunConfig(
        dataset=DatasetName.IMDB,
        data_amount=DataAmount.FULL,
        fusion_method=FusionMethod.B0,
        backbone_condition=BackboneCondition.FROZEN,
        seed=0,
        input_setting=InputSetting.L128,
        expected_cache_meta=meta,
    )
    provider.validate_for_run(config)
    with pytest.raises(CacheConfigMismatchError):
        provider.validate_for_run(
            replace(config, expected_cache_meta=replace(meta, torch_compile=True))
        )


def test_extraction_refuses_an_unlocked_transformers_version() -> None:
    """Caches must be built with the locked version (ADR 0005); checked before loading the model."""
    pytest.importorskip("transformers")
    from layer_fusion.extraction import extract_layer_vectors

    with pytest.raises(ValueError, match="locked"):
        extract_layer_vectors(
            _docs("x."),
            model_name=MODEL_NAME,
            transformers_version="0.0.0",
            attn_implementation="sdpa",
            torch_compile=False,
            include_special_tokens_in_pool=True,
        )


def test_special_tokens_can_be_left_out_of_the_mean(modernbert) -> None:
    """include_special_tokens_in_pool=False pools content tokens only (both modes supported)."""
    records = _docs("a short sentence for pooling.")
    with_special = _extract(modernbert, records, include_special_tokens_in_pool=True)
    without = _extract(modernbert, records, include_special_tokens_in_pool=False)
    assert without.vectors.shape == (1, 23, 768)
    assert not np.allclose(with_special.vectors, without.vectors, atol=1e-3)
    assert without.metadata.include_special_tokens_in_pool is False


def test_float32_extraction_still_reports_float16_round_trip_error(modernbert) -> None:
    """The float16 decision is made from a float32 measurement run, so report it there too."""
    result = _extract(modernbert, _docs("measure the value range."))
    assert result.metadata.dtype == "float32"
    assert result.report.max_abs_value > 0
    assert result.report.float16_round_trip_max_error >= 0


def test_report_locates_the_largest_value(modernbert) -> None:
    """The float16 evidence names where max |value| sits (slot, dim), so it can be re-checked."""
    result = _extract(modernbert, _docs("a", "a much longer second document for the batch."))
    report = result.report
    at_location = float(np.abs(result.vectors[:, report.max_abs_slot, report.max_abs_dim]).max())
    assert at_location == pytest.approx(report.max_abs_value, abs=1e-3)
    assert report.max_abs_value == pytest.approx(float(np.abs(result.vectors).max()), abs=1e-3)
    assert report.min_input_length == min(result.tokenized.input_lengths)


# --- PR #12 review: non-finite values, injected model, genre_splits -------------------------


@pytest.mark.parametrize(
    "max_abs", [float("nan"), float("inf"), 65505.0], ids=["nan", "inf", "over"]
)
def test_float16_range_check_rejects_nan_inf_and_overflow(max_abs: float) -> None:
    from layer_fusion.extraction import check_float16_range

    with pytest.raises(ValueError):
        check_float16_range(max_abs)


def test_float16_range_check_accepts_the_float16_maximum() -> None:
    from layer_fusion.extraction import check_float16_range

    check_float16_range(65504.0)


def test_non_finite_layer_vectors_are_rejected() -> None:
    from layer_fusion.extraction import check_finite

    vectors = np.zeros((2, 23, 768), dtype=np.float32)
    vectors[1, 5, 7] = np.nan
    with pytest.raises(ValueError, match="1 non-finite"):
        check_finite(vectors)


def test_injected_model_with_other_attention_is_rejected() -> None:
    """The cache would record attn_implementation='sdpa' for a model actually running eager."""
    from types import SimpleNamespace

    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from layer_fusion.extraction import extract_layer_vectors

    eager_model = SimpleNamespace(config=SimpleNamespace(_attn_implementation="eager"))
    with pytest.raises(ValueError, match="attn_implementation='eager'"):
        extract_layer_vectors(
            _docs("a."),
            model_name=MODEL_NAME,
            transformers_version=transformers.__version__,
            attn_implementation="sdpa",
            torch_compile=False,
            include_special_tokens_in_pool=True,
            model=eager_model,
            tokenizer=object(),
        )


def test_genre_splits_are_none_for_records_without_genres(modernbert) -> None:
    """SST-2/IMDB records carry no genre, so no all-null genre_splits.json gets written."""
    result = _extract(modernbert, _docs("first.", "second."))
    assert result.genre_splits is None


def test_cache_build_writes_a_loadable_cache_with_view_ratios(modernbert, tmp_path) -> None:
    """scripts/extract_cache.py path: begin_split -> extract into memmap -> finish_split."""
    import transformers

    from layer_fusion.cache_build import ExtractionEnv, build_cache
    from layer_fusion.providers import CacheProvider

    model, tokenizer = modernbert
    env = ExtractionEnv(
        model_name=MODEL_NAME,
        transformers_version=transformers.__version__,
        attn_implementation="sdpa",
        torch_compile=False,
        lfull_max_length_cap=4096,
        include_special_tokens_in_pool=True,
        cache_dtype="float16",
    )
    records = _docs("a much longer sentence " * 10, "short.", "mid sized text.")
    reports = build_cache(
        {"train": records}, env, tmp_path, dtype="float16", batch_size=2,
        model=model, tokenizer=tokenizer,
    )
    assert reports["train"]["num_docs"] == 3
    batch = CacheProvider(str(tmp_path)).get_split("train")
    assert batch.vectors.dtype.name == "float16" and batch.vectors.shape == (3, 23, 768)
    assert batch.doc_ids == ["d0", "d1", "d2"]
    assert batch.view_ratios is not None and len(batch.view_ratios) == 3
    assert batch.view_ratios[1] == 1.0  # short documents fit in one local window
