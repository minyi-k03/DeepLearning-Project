"""Reproduction track (T-06e, ADR 0004): paper fine-tuning, kept out of judgment.

A fake Hugging Face sequence-classification model and tokenizer stand in for ModernBERT, so
the runner, config expansion and CLI are checked on the CPU. One test at the end uses the
real model when its weights are cached locally.
"""

from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from layer_fusion import reproduction  # noqa: E402
from layer_fusion.cache_build import load_env_config  # noqa: E402
from layer_fusion.contracts import (  # noqa: E402
    BackboneCondition,
    DataAmount,
    DatasetName,
    DocumentRecord,
    FusionMethod,
    ReproductionSettings,
    RunConfig,
    RunKind,
    TrackKind,
)
from layer_fusion.providers import BackboneProvider, CacheConfigMismatchError  # noqa: E402
from layer_fusion.records import load_result_record, save_result_record  # noqa: E402
from layer_fusion.reproduction import (  # noqa: E402
    ReproductionOverrides,
    expand_reproduction,
    make_paper_optimizer,
    make_scheduler,
    run_reproduction,
)

ROOT = Path(__file__).resolve().parent.parent
MATRIX = ROOT / "configs" / "experiment_matrix.yaml"
ENV_PATH = ROOT / "configs" / "week1_env.yaml"
META = load_env_config(ENV_PATH).cache_metadata(None, "float32")
VOCAB = 40
SETTINGS = ReproductionSettings(weight_decay=0.01, warmup_ratio=0.1, lr_scheduler="linear")


class FakeTokenizer:
    pad_token_id, cls_token_id, sep_token_id = 0, 1, 2

    def __call__(self, texts, add_special_tokens=False):  # type: ignore[no-untyped-def]
        return {"input_ids": [[3 + int(w) % (VOCAB - 3) for w in t.split()] for t in texts]}


class FakeSeqClassifier(nn.Module):
    """HF-like ``model(input_ids, attention_mask).logits`` with mean pooling; counts the
    training examples it sees so tests can tell which rows were trained on."""

    def __init__(self, num_labels: int) -> None:
        super().__init__()
        self.config = SimpleNamespace(_attn_implementation=META.attn_implementation)
        self.embed = nn.Embedding(VOCAB, 16)
        self.norm = nn.LayerNorm(16)
        self.classifier = nn.Linear(16, num_labels)
        self.trained_examples = 0

    def forward(self, input_ids, attention_mask):  # type: ignore[no-untyped-def]
        if self.training:
            self.trained_examples += int(input_ids.shape[0])
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (self.embed(input_ids) * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        return SimpleNamespace(logits=self.classifier(self.norm(pooled)))


def _records(n: int, split: str, seed: int, num_labels: int = 3) -> list[DocumentRecord]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        label = int(rng.integers(num_labels))
        words = [str(label + 30)] + [str(x) for x in rng.integers(3, 30, rng.integers(2, 8))]
        genre = None if split == "train" else ("matched" if i % 2 == 0 else "mismatched")
        out.append(DocumentRecord(f"mnli-{split}-{i}", " ".join(words), label,
                                  text_b="5 6 7", split=split, genre_split=genre))
    return out


def _provider(*, limited: bool = False, train: int = 64) -> BackboneProvider:
    splits = {"train": _records(train, "train", 0), "eval": _records(30, "eval", 1)}
    return BackboneProvider(splits, META, None, FakeTokenizer(), limited=limited)


def _config(**kw) -> RunConfig:  # type: ignore[no-untyped-def]
    values = dict(
        dataset=DatasetName.MNLI,
        data_amount=DataAmount.FULL,
        fusion_method=FusionMethod.PAPER_HEAD,
        backbone_condition=BackboneCondition.FINETUNE,
        seed=0,
        lr_candidates=(5e-2,),
        epochs=3,
        batch_size=16,
        track=TrackKind.REPRODUCTION,
        expected_cache_meta=META,
        reproduction=SETTINGS,
    )
    values.update(kw)
    return RunConfig(**values)


class Factory:
    """Model factory that keeps the models it built."""

    def __init__(self) -> None:
        self.models: list[FakeSeqClassifier] = []

    def __call__(self, num_labels: int) -> FakeSeqClassifier:
        self.models.append(FakeSeqClassifier(num_labels))
        return self.models[-1]


# ---------------------------------------------------------------- completion criterion


def test_reproduction_record_is_marked_and_left_out_of_judgment(tmp_path: Path) -> None:
    """T-06e: the record says reproduction track + paper head and judgment drops it."""
    from layer_fusion.judgment import select_main_records

    record = run_reproduction(_config(), _provider(), device="cpu", model_factory=Factory())
    assert record.run_id == "repro-mnli-full-paper_head-all-finetune-none-s0-main"
    assert record.config.track is TrackKind.REPRODUCTION
    assert record.config.fusion_method is FusionMethod.PAPER_HEAD
    assert select_main_records([record]) == []
    assert record.selected_lr == 5e-2 and record.lr_tuning_scores == {}
    assert record.eval_error_rate_matched is not None
    assert record.eval_error_rate_mismatched is not None
    assert len(record.predictions) == 30

    reloaded = load_result_record(save_result_record(record, tmp_path))
    assert reloaded == record
    assert reloaded.config.reproduction == SETTINGS


def test_the_last_epoch_model_is_reported_not_the_best() -> None:
    # Chosen so that an earlier epoch beats the last one on the dev set (fixed seed).
    record = run_reproduction(_config(lr_candidates=(0.2,), epochs=6), _provider(),
                              device="cpu", model_factory=Factory())
    errors = [point["eval_error"] for point in record.learning_curve]
    assert len(errors) == 6
    assert min(errors) < errors[-1]  # precondition: "best" and "last" disagree here
    assert record.eval_error_rate == errors[-1]
    reported = np.mean([p.prediction != p.label for p in record.predictions])
    assert reported == pytest.approx(errors[-1])  # predictions come from the last model too


def test_training_uses_the_whole_training_set_every_epoch() -> None:
    """No tuning split is held out (paper setting)."""
    factory = Factory()
    run_reproduction(_config(epochs=2), _provider(train=64), device="cpu", model_factory=factory)
    assert factory.models[0].trained_examples == 2 * 64


def test_same_seed_gives_the_same_record() -> None:
    a = run_reproduction(_config(), _provider(), device="cpu", model_factory=Factory())
    b = run_reproduction(_config(), _provider(), device="cpu", model_factory=Factory())
    assert [p.prediction for p in a.predictions] == [p.prediction for p in b.predictions]
    assert a.learning_curve == b.learning_curve


# ---------------------------------------------------------------- optimizer and schedule


def test_weight_decay_skips_bias_and_layer_norm() -> None:
    model = FakeSeqClassifier(3)
    decay_of = {
        id(p): g["weight_decay"]
        for g in make_paper_optimizer(model, 1e-3, 5e-6).param_groups
        for p in g["params"]
    }
    assert decay_of[id(model.embed.weight)] == 5e-6
    assert decay_of[id(model.classifier.weight)] == 5e-6
    for p in (model.classifier.bias, model.norm.weight, model.norm.bias):
        assert decay_of[id(p)] == 0.0


def test_scheduler_warms_up_then_decays() -> None:
    param = nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([param], lr=1.0)
    scheduler = make_scheduler(optimizer, replace(SETTINGS, warmup_ratio=0.25), 8)
    lrs = []
    for _ in range(8):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    peak = lrs.index(max(lrs))
    assert peak == 2  # 2 warmup steps of 8
    assert lrs[:peak + 1] == sorted(lrs[:peak + 1])
    assert lrs[peak:] == sorted(lrs[peak:], reverse=True)


# ---------------------------------------------------------------- config expansion


def test_real_config_refuses_main_runs_while_unstated_values_are_tbd() -> None:
    with pytest.raises(ValueError) as info:
        expand_reproduction(MATRIX, ENV_PATH)
    message = str(info.value)
    for missing in ("repro_mnli: batch_size is TBD", "repro_sst2: warmup_ratio is TBD",
                    "repro_mnli: lr_scheduler is TBD", "repro_sst2: seeds is TBD"):
        assert missing in message


def test_debug_expansion_takes_the_paper_values() -> None:
    configs = expand_reproduction(
        MATRIX, ENV_PATH, seeds=[1, 0], run_kind=RunKind.DEBUG,
        overrides=ReproductionOverrides(batch_size=32, warmup_ratio=0.06, lr_scheduler="linear"),
    )
    assert [(c.dataset, c.seed) for c in configs] == [
        (DatasetName.MNLI, 0), (DatasetName.MNLI, 1), (DatasetName.SST2, 0), (DatasetName.SST2, 1)
    ]
    mnli, sst2 = configs[0], configs[2]
    assert (mnli.lr_candidates, mnli.reproduction.weight_decay, mnli.epochs) == ((5e-5,), 5e-6, 1)
    assert (sst2.lr_candidates, sst2.reproduction.weight_decay, sst2.epochs) == ((8e-5,), 1e-5, 2)
    assert {c.expected_cache_meta.dtype for c in configs} == {"float32"}
    assert all(c.track is TrackKind.REPRODUCTION and c.run_kind is RunKind.DEBUG
               for c in configs)


def test_main_expansion_refuses_overrides() -> None:
    with pytest.raises(ValueError, match="debug runs only"):
        expand_reproduction(MATRIX, ENV_PATH, seeds=[0])
    with pytest.raises(ValueError, match="debug runs only"):
        expand_reproduction(MATRIX, ENV_PATH, overrides=ReproductionOverrides(batch_size=8))


# ---------------------------------------------------------------- refused runs


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"track": TrackKind.MAIN}, "reproduction-track"),
        ({"fusion_method": FusionMethod.B0}, "reproduction-track"),
        ({"dataset": DatasetName.IMDB}, "MNLI and SST-2"),
        ({"reproduction": None}, "paper settings"),
        ({"lr_candidates": (1e-3, 2e-3)}, "exactly the paper"),
        ({"batch_size": None}, "batch_size"),
        ({"reproduction": replace(SETTINGS, warmup_ratio=1.0)}, "warmup_ratio"),
    ],
)
def test_incomplete_or_foreign_configs_are_refused(change, match) -> None:  # type: ignore[no-untyped-def]
    factory = Factory()
    with pytest.raises(ValueError, match=match):
        run_reproduction(_config(**change), _provider(), device="cpu", model_factory=factory)
    assert factory.models == []  # refused before building a model


def test_a_limited_sample_cannot_make_a_main_record() -> None:
    with pytest.raises(CacheConfigMismatchError, match="debug runs only"):
        run_reproduction(_config(), _provider(limited=True), device="cpu",
                         model_factory=Factory())
    debug = _config(run_kind=RunKind.DEBUG, epochs=1)
    assert run_reproduction(debug, _provider(limited=True), device="cpu",
                            model_factory=Factory()).run_id.endswith("-debug")


def test_paper_head_never_enters_the_main_track(noisy_provider, make_run_config) -> None:  # type: ignore[no-untyped-def]
    from layer_fusion.fusion import build_fusion_head
    from layer_fusion.run_queue import check_queue
    from layer_fusion.runner import run_experiment

    with pytest.raises(ValueError, match="Unsupported fusion method"):
        build_fusion_head(FusionMethod.PAPER_HEAD, 3)
    provider = noisy_provider()
    with pytest.raises(ValueError, match="paper_head"):
        run_experiment(make_run_config(provider, fusion_method=FusionMethod.PAPER_HEAD),
                       provider, device="cpu")
    with pytest.raises(ValueError, match="T-06e"):
        check_queue([_config()])


def test_backbone_free_provider_refuses_to_hand_out_a_backbone() -> None:
    with pytest.raises(RuntimeError, match="without a backbone"):
        _provider().fresh_model()


def test_old_records_without_reproduction_settings_still_load(tmp_path: Path) -> None:
    from layer_fusion.records import load_result_records

    record = run_reproduction(_config(epochs=1), _provider(), device="cpu",
                              model_factory=Factory())
    main_like = replace(record, run_id="old", config=replace(record.config, reproduction=None))
    path = save_result_record(main_like, tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    del data["config"]["reproduction"]  # a record written before T-06e
    path.write_text(json.dumps(data), encoding="utf-8")
    assert load_result_records(tmp_path)[0].config.reproduction is None


# ---------------------------------------------------------------- CLI


_spec = importlib.util.spec_from_file_location(
    "run_repro_cli", ROOT / "scripts" / "run_reproduction.py"
)
assert _spec is not None and _spec.loader is not None
cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cli)

DEBUG_ARGS = ["--debug", "--datasets", "mnli", "--seeds", "0", "--batch-size", "16",
              "--warmup-ratio", "0.1", "--lr-scheduler", "linear", "--device", "cpu"]


@pytest.fixture
def fake_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "make_provider", lambda dataset, args: _provider(
        limited=args.limit is not None))
    monkeypatch.setattr(reproduction, "default_model_factory", lambda provider: Factory())


def test_cli_debug_run_writes_a_record_without_the_lock(tmp_path: Path, fake_backends) -> None:  # type: ignore[no-untyped-def]
    lock = tmp_path / "main_queue.lock"
    out = tmp_path / "repro"
    args = [*DEBUG_ARGS, "--limit", "64", "--output-dir", str(out), "--lock-file", str(lock)]
    assert cli.main(args) == 0
    [path] = out.glob("*.json")
    record = load_result_record(path)
    assert record.config.track is TrackKind.REPRODUCTION
    assert record.config.run_kind is RunKind.DEBUG
    assert record.config.lr_candidates == (5e-5,)  # the paper value, not an override
    assert not lock.exists()
    before = path.read_bytes()
    assert cli.main(args) == 0  # rerun: skipped, never rewritten
    assert path.read_bytes() == before


def test_cli_main_run_is_refused_while_values_are_tbd(tmp_path: Path) -> None:
    lock = tmp_path / "main_queue.lock"
    with pytest.raises(SystemExit, match="TBD"):
        cli.main(["--output-dir", str(tmp_path / "repro"), "--lock-file", str(lock)])
    assert not lock.exists()
    with pytest.raises(SystemExit, match="debug-only"):
        cli.main(["--seeds", "0", "--output-dir", str(tmp_path / "repro")])


def test_cli_dry_run_lists_runs(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    out = tmp_path / "repro"
    assert cli.main([*DEBUG_ARGS, "--seeds", "0", "1", "--dry-run", "--output-dir", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "repro-mnli-full-paper_head-all-finetune-none-s1-debug" in printed
    assert not out.exists()


# ---------------------------------------------------------------- real model (if cached)


def test_real_paper_model_reproduction_debug_run_on_cpu() -> None:
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(META.model_name, local_files_only=True)
        from transformers import AutoConfig

        AutoConfig.from_pretrained(META.model_name, local_files_only=True)
    except OSError:
        pytest.skip(f"{META.model_name} not in local HF cache")
    text = ["a man plays a guitar", "the cat sleeps", "it rains today", "people talk loudly"]
    splits = {
        "train": [DocumentRecord(f"t{i}", t, i % 3, text_b="someone acts", split="train")
                  for i, t in enumerate(text * 8)],
        "eval": [DocumentRecord(f"e{i}", t, i % 3, text_b="a thing", split="eval",
                                genre_split="matched" if i % 2 == 0 else "mismatched")
                 for i, t in enumerate(text)],
    }
    provider = BackboneProvider(splits, META, None, tokenizer, limited=True)
    config = _config(run_kind=RunKind.DEBUG, lr_candidates=(5e-5,), epochs=1, batch_size=8)
    record = run_reproduction(config, provider, device="cpu")
    assert len(record.predictions) == 4
    assert record.trainable_params > 100_000_000  # the whole ModernBERT + paper head trains
