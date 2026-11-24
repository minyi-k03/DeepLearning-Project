"""scripts/run_experiment.py (T-06b): one command line -> one record file that loads back."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("torch")

from layer_fusion.contracts import FusionMethod, RunKind  # noqa: E402
from layer_fusion.extraction import save_layer_cache  # noqa: E402
from layer_fusion.records import load_result_records  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "run_experiment_cli", ROOT / "scripts" / "run_experiment.py"
)
assert _spec is not None and _spec.loader is not None
cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cli)


@pytest.fixture
def cache_dir(noisy_provider, tmp_path: Path) -> Path:  # type: ignore[no-untyped-def]
    """A small MNLI-like cache written to disk the same way extract_cache.py writes one."""
    provider = noisy_provider(train_size=120, eval_size=60)
    out = tmp_path / "cache" / "mnli"
    for split in ("train", "eval"):
        batch = provider.get_split(split)
        save_layer_cache(
            out, split, batch.vectors, list(batch.doc_ids), [int(x) for x in batch.labels],
            provider.metadata, genre_splits=batch.genre_splits,
        )
    return out


def _debug_args(cache_dir: Path, out_dir: Path, *extra: str) -> list[str]:
    return [
        "--dataset", "mnli", "--fusion", "F2", "--seed", "0",
        "--cache-dir", str(cache_dir), "--output-dir", str(out_dir),
        "--lr-candidates", "1e-3", "1e-2", "--epochs", "2",
        "--early-stopping-patience", "1", "--batch-size", "32", "--device", "cpu",
        *extra,
    ]


def test_one_command_writes_a_loadable_record(cache_dir: Path, tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    out_dir = tmp_path / "runs"
    path = cli.main(_debug_args(cache_dir, out_dir))
    assert path.parent == out_dir and path.is_file()
    assert "eval error" in capsys.readouterr().out

    [record] = load_result_records(out_dir)
    assert record.config.run_kind is RunKind.DEBUG  # the CLI default
    assert record.config.fusion_method is FusionMethod.F2
    assert record.config.expected_cache_meta is None  # debug runs may skip it
    assert len(record.predictions) == 60
    assert set(record.lr_tuning_scores) == {1e-3, 1e-2}


def test_rerun_is_refused_unless_debug_overwrite(cache_dir: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "runs"
    cli.main(_debug_args(cache_dir, out_dir))
    with pytest.raises(SystemExit, match="already exists"):
        cli.main(_debug_args(cache_dir, out_dir))
    assert cli.main(_debug_args(cache_dir, out_dir, "--overwrite")).is_file()


def _reuse_args(cache_dir: Path, out_dir: Path, lr: float, selection_seed: int = 0) -> list[str]:
    args = _debug_args(
        cache_dir, out_dir, "--fixed-lr", repr(lr), "--lr-selection-seed", str(selection_seed)
    )
    args[args.index("--seed") + 1] = "1"
    return args


def test_lr_reuse_run_takes_the_lr_of_its_selection_record(cache_dir: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "runs"
    selection = load_result_records(cli.main(_debug_args(cache_dir, out_dir)).parent)[0]
    cli.main(_reuse_args(cache_dir, out_dir, selection.selected_lr))
    reuse = next(r for r in load_result_records(out_dir) if r.config.seed == 1)
    assert reuse.selected_lr == selection.selected_lr and reuse.lr_selection_seed == 0


def test_lr_reuse_with_another_lr_is_refused(cache_dir: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "runs"
    selection = load_result_records(cli.main(_debug_args(cache_dir, out_dir)).parent)[0]
    other = next(lr for lr in (1e-3, 1e-2) if lr != selection.selected_lr)
    with pytest.raises(SystemExit, match="selected_lr"):
        cli.main(_reuse_args(cache_dir, out_dir, other))
    assert len(load_result_records(out_dir)) == 1  # no reuse record was written


def test_lr_reuse_without_a_selection_record_is_refused(cache_dir: Path, tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="no selection-run record"):
        cli.main(_reuse_args(cache_dir, tmp_path / "runs", 1e-2))
    with pytest.raises(SystemExit, match="both fixed_lr and lr_selection_seed"):
        cli.main(_debug_args(cache_dir, tmp_path / "runs", "--fixed-lr", "1e-2"))


def test_main_runs_are_refused_here(cache_dir: Path, tmp_path: Path) -> None:
    """Main timed runs need the queue's GPU lock (ADR 0009, T-06c), so this CLI refuses them."""
    with pytest.raises(SystemExit, match="sequential queue"):
        cli.main(_debug_args(cache_dir, tmp_path / "runs", "--run-kind", "main"))
    assert not (tmp_path / "runs").exists()


def test_tbd_budget_must_be_passed_on_the_command_line(cache_dir: Path, tmp_path: Path) -> None:
    """week1_env.yaml has the budget as TBD today, so it must come from the command line."""
    args = [
        "--dataset", "mnli", "--fusion", "F2", "--seed", "0",
        "--cache-dir", str(cache_dir), "--output-dir", str(tmp_path / "runs"),
        "--env-config", str(ROOT / "configs" / "week1_env.yaml"), "--epochs", "1",
    ]
    with pytest.raises(SystemExit, match="TBD"):
        cli.main(args)
    assert not (tmp_path / "runs").exists()


def test_missing_cache_is_reported(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="no layer-vector cache"):
        cli.main(_debug_args(tmp_path / "nowhere", tmp_path / "runs"))


def test_default_cache_dir_follows_dataset_and_input_setting(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(tmp_path)
    args = _debug_args(Path("unused"), tmp_path / "runs")
    i = args.index("--cache-dir")
    del args[i : i + 2]
    args[args.index("mnli")] = "imdb"
    args += ["--input-setting", "L512", "--env-config", str(ROOT / "configs" / "week1_env.yaml")]
    with pytest.raises(SystemExit, match=r"data[\\/]cache[\\/]imdb_L512"):  # Windows too
        cli.main(args)
