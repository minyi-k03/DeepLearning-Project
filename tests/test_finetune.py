"""Finetune condition (T-06d): BackboneProvider + run_experiment on CPU with a fake backbone.

The fake backbone and tokenizer mimic the Hugging Face interface the provider uses
(``tokenizer(texts, add_special_tokens=False)["input_ids"]``, ``model(input_ids=...,
attention_mask=..., output_hidden_states=True).hidden_states``), so the seam is tested
without ModernBERT weights. One test at the end uses the real model when it is cached.
"""

from __future__ import annotations

import copy
import importlib.util
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from layer_fusion.cache_build import load_env_config  # noqa: E402
from layer_fusion.contracts import (  # noqa: E402
    BackboneCondition,
    DataAmount,
    DatasetName,
    DocumentRecord,
    FusionMethod,
    InputSetting,
    ResultRecord,
    RunConfig,
    RunKind,
)
from layer_fusion.extraction import masked_token_mean  # noqa: E402
from layer_fusion.finetune import FinetuneModel, TokenRowView  # noqa: E402
from layer_fusion.fusion import build_fusion_head  # noqa: E402
from layer_fusion.layers import HIDDEN_SIZE, NUM_LAYER_VECTORS  # noqa: E402
from layer_fusion.providers import BackboneProvider, CacheConfigMismatchError  # noqa: E402
from layer_fusion.records import load_result_record, save_result_record  # noqa: E402
from layer_fusion.runner import run_experiment  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ENV = load_env_config(ROOT / "configs" / "week1_env.yaml")
META = ENV.cache_metadata(None, "float32")
PAD, CLS, SEP = 0, 1, 2
VOCAB = 40


class FakeTokenizer:
    pad_token_id, cls_token_id, sep_token_id = PAD, CLS, SEP

    def __call__(self, texts, add_special_tokens=False):  # type: ignore[no-untyped-def]
        assert add_special_tokens is False
        return {"input_ids": [[3 + int(w) % (VOCAB - 3) for w in t.split()] for t in texts]}


class FakeBackbone(nn.Module):
    """Embedding + 22 trainable per-layer affine maps -> 23 hidden states [B, T, 768]."""

    def __init__(self, num_hidden: int = NUM_LAYER_VECTORS) -> None:
        super().__init__()
        self.config = SimpleNamespace(_attn_implementation=META.attn_implementation)
        self.embed = nn.Embedding(VOCAB, HIDDEN_SIZE)
        self.scales = nn.Parameter(torch.ones(num_hidden - 1, HIDDEN_SIZE))
        self.shifts = nn.Parameter(torch.zeros(num_hidden - 1, HIDDEN_SIZE))

    def forward(self, input_ids, attention_mask, output_hidden_states):  # type: ignore[no-untyped-def]
        assert output_hidden_states
        h = self.embed(input_ids)
        states = [h]
        for scale, shift in zip(self.scales, self.shifts, strict=True):
            h = torch.tanh(h * scale + shift)
            states.append(h)
        return SimpleNamespace(hidden_states=tuple(states))


class DropoutBackbone(FakeBackbone):
    """Fake trainable backbone whose forward pass consumes the torch RNG."""

    def __init__(self) -> None:
        super().__init__()
        self.dropout = nn.Dropout(0.5)

    def forward(self, input_ids, attention_mask, output_hidden_states):  # type: ignore[no-untyped-def]
        result = super().forward(input_ids, attention_mask, output_hidden_states)
        return SimpleNamespace(
            hidden_states=tuple(self.dropout(state) for state in result.hidden_states)
        )


def _records(n: int, split: str, seed: int) -> list[DocumentRecord]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        label = int(rng.integers(3))
        # The label shows up as a word, so the task is learnable through the backbone.
        words = [str(label + 30)] + [str(x) for x in rng.integers(3, 30, rng.integers(2, 9))]
        genre = None if split == "train" else ("matched" if i % 2 == 0 else "mismatched")
        out.append(
            DocumentRecord(
                doc_id=f"mnli-{split}-{i}", text_a=" ".join(words),
                text_b=" ".join(str(x) for x in rng.integers(3, 30, 4)), label=label,
                split=split, genre_split=genre,
            )
        )
    return out


class SpyProvider(BackboneProvider):
    """Keeps every fresh backbone it hands out, with a copy of its starting weights."""

    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        self.handed_out: list[tuple[nn.Module, dict[str, torch.Tensor]]] = []

    def fresh_model(self):  # type: ignore[no-untyped-def]
        model = super().fresh_model()
        start = {k: v.detach().clone() for k, v in model.state_dict().items()}
        self.handed_out.append((model, start))
        return model


def _provider(*, limited: bool = False, backbone: nn.Module | None = None) -> SpyProvider:
    splits = {"train": _records(80, "train", 0), "eval": _records(30, "eval", 1)}
    return SpyProvider(splits, META, backbone or FakeBackbone(), FakeTokenizer(), limited=limited)


def _config(**kw) -> RunConfig:  # type: ignore[no-untyped-def]
    values = dict(
        dataset=DatasetName.MNLI,
        data_amount=DataAmount.FULL,
        fusion_method=FusionMethod.F2,
        backbone_condition=BackboneCondition.FINETUNE,
        seed=0,
        lr_candidates=(1e-3, 1e-2, 3e-2),
        epochs=2,
        early_stopping_patience=1,
        batch_size=16,
        expected_cache_meta=META,
    )
    values.update(kw)
    return RunConfig(**values)


# ---------------------------------------------------------------- same record format


def test_finetune_record_has_the_frozen_record_format(
    noisy_provider, make_run_config, tmp_path: Path
) -> None:  # type: ignore[no-untyped-def]
    """Completion criterion of T-06d: same fields filled as a frozen record, and it loads back
    and enters judgment like one."""
    from layer_fusion.judgment import select_main_records

    finetuned = run_experiment(_config(), _provider(), device="cpu")
    frozen_provider = noisy_provider()
    frozen = run_experiment(make_run_config(frozen_provider), frozen_provider, device="cpu")

    filled = {f.name for f in fields(ResultRecord) if getattr(finetuned, f.name) is not None}
    assert filled == {f.name for f in fields(ResultRecord) if getattr(frozen, f.name) is not None}
    assert finetuned.run_id == "mnli-full-F2-all-finetune-none-s0-main"
    assert len(finetuned.predictions) == 30
    assert finetuned.eval_error_rate_matched is not None
    assert finetuned.eval_error_rate_mismatched is not None
    assert {p.genre_split for p in finetuned.predictions} == {"matched", "mismatched"}
    assert len(finetuned.layer_weights) == NUM_LAYER_VECTORS  # F2 weights from the head
    assert set(finetuned.lr_tuning_scores) == set(_config().lr_candidates)

    path = save_result_record(finetuned, tmp_path)
    reloaded = load_result_record(path)
    assert reloaded == finetuned
    assert select_main_records([reloaded]) == [reloaded]


def test_trainable_params_include_the_backbone() -> None:
    record = run_experiment(_config(fusion_method=FusionMethod.B0), _provider(), device="cpu")
    head = build_fusion_head(FusionMethod.B0, 3)
    backbone = sum(p.numel() for p in FakeBackbone().parameters())
    assert record.trainable_params == backbone + sum(p.numel() for p in head.parameters())


# ---------------------------------------------------------------- the backbone really trains


def test_backbone_weights_change_and_every_candidate_starts_from_pretrained() -> None:
    provider = _provider()
    pristine = {k: v.clone() for k, v in provider.fresh_model().state_dict().items()}
    provider.handed_out.clear()
    run_experiment(_config(), provider, device="cpu")

    assert len(provider.handed_out) == 3  # one fresh backbone per LR candidate
    models = [model for model, _ in provider.handed_out]
    assert len({id(m) for m in models}) == 3  # never shared between candidates
    for model, start in provider.handed_out:
        for name, value in start.items():
            torch.testing.assert_close(value, pristine[name])  # starts pretrained
        trained = model.state_dict()
        assert any(not torch.equal(trained[k], start[k]) for k in start)  # gradients flowed


def test_fixed_lr_run_trains_one_fresh_backbone() -> None:
    provider = _provider()
    record = run_experiment(_config(seed=1, fixed_lr=1e-2, lr_selection_seed=0), provider,
                            device="cpu")
    assert len(provider.handed_out) == 1
    assert record.selected_lr == 1e-2 and record.lr_tuning_scores == {}


def test_same_seed_finetune_is_independent_of_global_torch_rng_state() -> None:
    """The run seed controls backbone dropout, not whatever ran before this experiment."""
    pristine = DropoutBackbone()
    config = _config(
        run_kind=RunKind.DEBUG,
        fixed_lr=1e-2,
        lr_selection_seed=0,
        lr_candidates=(1e-2,),
        epochs=2,
    )

    torch.manual_seed(111)
    first = run_experiment(config, _provider(backbone=copy.deepcopy(pristine)), device="cpu")
    torch.manual_seed(999)
    second = run_experiment(config, _provider(backbone=copy.deepcopy(pristine)), device="cpu")

    assert second.learning_curve == first.learning_curve
    assert second.predictions == first.predictions
    assert second.layer_weights == first.layer_weights


# ---------------------------------------------------------------- pooling = extractor rule


def _view(provider: BackboneProvider, rows: list[int], pool_special: bool = True) -> TokenRowView:
    batch = provider.get_split("eval")
    return TokenRowView(batch.input_ids, batch.labels, np.asarray(rows), provider.pad_id,
                        provider.special_ids, pool_special)


def test_layer_vectors_ignore_padding_and_match_the_extractor_mean() -> None:
    provider = _provider()
    model = FinetuneModel(provider.fresh_model(), build_fusion_head(FusionMethod.B0, 3)).eval()
    lengths = [len(x) for x in provider.get_split("eval").input_ids]
    short, long = int(np.argmin(lengths)), int(np.argmax(lengths))
    assert lengths[short] < lengths[long]
    view = _view(provider, [short, long])
    with torch.no_grad():
        alone = model.layer_vectors(view.load(np.asarray([short]), torch.device("cpu"))[0])
        padded = model.layer_vectors(view.load(np.asarray([short, long]), torch.device("cpu"))[0])
        torch.testing.assert_close(alone[0], padded[0])  # padding never enters the mean

        (ids, attention, pool_mask), _ = view.load(np.asarray([short]), torch.device("cpu"))
        hidden = model.backbone(input_ids=ids, attention_mask=attention,
                                output_hidden_states=True).hidden_states
        expected = torch.stack([masked_token_mean(h, attention) for h in hidden], dim=1)
        torch.testing.assert_close(alone, expected)
    assert alone.shape == (1, NUM_LAYER_VECTORS, HIDDEN_SIZE)


def test_special_tokens_leave_the_pool_when_the_setting_says_so() -> None:
    provider = _provider()
    (ids, attention, pool_mask), _ = _view(provider, [0], pool_special=False).load(
        np.asarray([0]), torch.device("cpu")
    )
    assert pool_mask.sum() == attention.sum() - int(((ids == CLS) | (ids == SEP)).sum())
    assert int(((ids == CLS) | (ids == SEP)).sum()) == 3  # [CLS] a [SEP] b [SEP]


# ---------------------------------------------------------------- refused runs


@pytest.mark.parametrize(
    ("config", "provider_kw", "error", "match"),
    [
        (_config(data_amount=DataAmount.SMALL), {}, ValueError, "full MNLI"),
        (_config(dataset=DatasetName.SST2), {}, ValueError, "MNLI only"),
        (_config(dataset=DatasetName.IMDB, input_setting=InputSetting.L512), {},
         CacheConfigMismatchError, "input_setting"),
        (_config(), {"limited": True}, CacheConfigMismatchError, "debug runs only"),
        (_config(backbone_condition=BackboneCondition.FROZEN), {}, CacheConfigMismatchError,
         "finetune condition"),
        (_config(expected_cache_meta=replace(META, dtype="float16")), {},
         CacheConfigMismatchError, "dtype"),
    ],
)
def test_out_of_scope_finetune_runs_are_refused(config, provider_kw, error, match) -> None:  # type: ignore[no-untyped-def]
    provider = _provider(**provider_kw)
    with pytest.raises(error, match=match):
        run_experiment(config, provider, device="cpu")
    assert provider.handed_out == []  # refused before any training


def test_limited_sample_is_fine_for_debug_runs() -> None:
    config = _config(run_kind=RunKind.DEBUG, lr_candidates=(1e-2,), epochs=1)
    assert run_experiment(config, _provider(limited=True), device="cpu").run_id.endswith("debug")


def test_backbone_with_wrong_hidden_state_count_is_refused() -> None:
    with pytest.raises(ValueError, match="23 hidden states"):
        run_experiment(_config(), _provider(backbone=FakeBackbone(num_hidden=13)), device="cpu")


def test_provider_refuses_a_model_with_another_attention_implementation() -> None:
    backbone = FakeBackbone()
    backbone.config._attn_implementation = "eager"
    with pytest.raises(ValueError, match="attn_implementation"):
        _provider(backbone=backbone)


# ---------------------------------------------------------------- CLI


_spec = importlib.util.spec_from_file_location(
    "run_experiment_cli_ft", ROOT / "scripts" / "run_experiment.py"
)
assert _spec is not None and _spec.loader is not None
cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cli)


def _cli_args(out: Path, *extra: str) -> list[str]:
    return [
        "--dataset", "mnli", "--backbone", "finetune", "--fusion", "B0", "--seed", "0",
        "--finetune-lr-candidates", "1e-2", "--epochs", "1", "--early-stopping-patience", "1",
        "--batch-size", "16", "--device", "cpu", "--output-dir", str(out), *extra,
    ]


def test_cli_finetune_debug_run_writes_a_record(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    seen = {}

    def fake_provider(config, args):  # type: ignore[no-untyped-def]
        seen["limit"] = args.limit
        return _provider(limited=args.limit is not None)

    monkeypatch.setattr(cli, "make_backbone_provider", fake_provider)
    path = cli.main(_cli_args(tmp_path / "runs", "--limit", "50"))
    record = load_result_record(path)
    assert seen["limit"] == 50
    assert record.config.backbone_condition is BackboneCondition.FINETUNE
    assert record.config.lr_candidates == (1e-2,)
    assert record.config.run_kind is RunKind.DEBUG


def test_cli_finetune_lr_candidates_are_required(tmp_path: Path) -> None:
    args = [a for a in _cli_args(tmp_path / "runs") if a not in {"--finetune-lr-candidates"}]
    args.remove("1e-2")
    with pytest.raises(SystemExit, match="finetune_lr_candidates"):
        cli.main(args)


def test_cli_limit_is_refused_for_frozen_runs(tmp_path: Path) -> None:
    args = _cli_args(tmp_path / "runs", "--limit", "5")
    args[args.index("finetune")] = "frozen"
    with pytest.raises(SystemExit, match="--limit"):
        cli.main(args)


# ---------------------------------------------------------------- real ModernBERT (if cached)


def test_real_modernbert_finetune_debug_run_on_cpu() -> None:
    pytest.importorskip("transformers")
    from transformers import AutoModel, AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(ENV.model_name, local_files_only=True)
        model = AutoModel.from_pretrained(
            ENV.model_name, attn_implementation=ENV.attn_implementation, local_files_only=True
        )
    except OSError:
        pytest.skip(f"{ENV.model_name} weights not in local HF cache")
    text = ["a man is playing a guitar", "the cat sleeps", "it rains", "people talk loudly"]
    splits = {
        "train": [
            DocumentRecord(f"t{i}", t, i % 3, text_b="someone does something", split="train")
            for i, t in enumerate(text * 8)  # 32 docs: >= 10 per class for the tuning split
        ],
        "eval": [
            DocumentRecord(f"e{i}", t, i % 3, text_b="a thing happens", split="eval",
                           genre_split="matched" if i % 2 == 0 else "mismatched")
            for i, t in enumerate(text)
        ],
    }
    provider = BackboneProvider(splits, META, model, tokenizer, limited=True)
    config = _config(run_kind=RunKind.DEBUG, fusion_method=FusionMethod.B0,
                     lr_candidates=(2e-5,), epochs=1, batch_size=4)
    record = run_experiment(config, provider, device="cpu")
    assert len(record.predictions) == 4
    assert record.trainable_params > 100_000_000  # ModernBERT-base itself trains


def test_finetune_optimizer_keeps_the_role_based_decay_rule() -> None:
    """PR #18 rule through the FinetuneModel wrapper: F4 gate and bias get no decay, while
    the backbone's and classifier's weight matrices get 0.01 (SPEC 학습 루프)."""
    from layer_fusion.training import make_optimizer

    model = FinetuneModel(FakeBackbone(), build_fusion_head(FusionMethod.F4, 3))
    decay_of = {
        id(p): group["weight_decay"]
        for group in make_optimizer(model, 1e-3).param_groups
        for p in group["params"]
    }
    gate = [m for m in model.head.modules() if isinstance(m, nn.Linear)
            and m.out_features == NUM_LAYER_VECTORS]
    assert gate, "F4 head should own a gate producing one logit per layer vector"
    assert decay_of[id(gate[0].weight)] == 0.0 and decay_of[id(gate[0].bias)] == 0.0
    assert decay_of[id(model.backbone.embed.weight)] == 0.01
    assert all(decay_of[id(p)] == 0.0 for p in model.parameters() if p.ndim < 2)
