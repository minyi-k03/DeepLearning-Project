"""scripts/judge_and_report.py (T-08a): result records -> judgment -> report artifacts."""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("torch")

from layer_fusion.contracts import FusionMethod, RunKind  # noqa: E402
from layer_fusion.records import save_result_record  # noqa: E402
from layer_fusion.runner import run_experiment  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RULES = ROOT / "configs" / "decision_rules.yaml"  # status: locked (T-07a); fixtures re-draft it
_spec = importlib.util.spec_from_file_location(
    "judge_and_report_cli", ROOT / "scripts" / "judge_and_report.py"
)
assert _spec is not None and _spec.loader is not None
cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cli)


@pytest.fixture
def runs_dir(noisy_provider, make_run_config, tmp_path: Path):  # type: ignore[no-untyped-def]
    """Write records to a runs directory: ``write(main=[methods], debug=[methods])``."""
    provider = noisy_provider(train_size=120, eval_size=40)
    out = tmp_path / "runs"

    def write(main=(), debug=()):  # type: ignore[no-untyped-def]
        for kind, methods in ((RunKind.MAIN, main), (RunKind.DEBUG, debug)):
            for method in methods:
                config = make_run_config(provider, fusion_method=method, epochs=2, run_kind=kind)
                save_result_record(run_experiment(config, provider, device="cpu"), out)
        return out

    return write


def _args(runs: Path, out: Path, *extra: str, rules: Path = RULES) -> list[str]:
    return ["--runs-dir", str(runs), "--rules", str(rules), "--output-dir", str(out), *extra]


@pytest.fixture
def exploratory_rules(tmp_path: Path) -> Path:
    """The real rules, re-drafted, with only the exploratory hypotheses kept.

    Dropping the confirmatory hypotheses lets main records of any fusion method reach the
    report without needing the full set of B0/F-method cells they judge. ``status`` is put
    back to ``draft`` so these tests keep exercising the ``--allow-draft-rules`` path.
    """
    import yaml

    doc = yaml.safe_load(RULES.read_text(encoding="utf-8"))
    doc["status"] = "draft"
    doc["hypotheses"] = {
        hid: spec for hid, spec in doc["hypotheses"].items() if spec["type"] == "exploratory"
    }
    assert doc["hypotheses"], "expected exploratory hypotheses in the draft rules"
    path = tmp_path / "rules.yaml"
    path.write_text(yaml.safe_dump(doc, allow_unicode=True), encoding="utf-8")
    return path


@pytest.fixture
def tbd_rules(tmp_path: Path) -> Path:
    """The rules re-drafted with the McNemar level put back to TBD (null), as before the lock."""
    import yaml

    doc = yaml.safe_load(RULES.read_text(encoding="utf-8"))
    doc["status"] = "draft"
    doc["criteria"]["mcnemar_alpha"] = None
    path = tmp_path / "tbd_rules.yaml"
    path.write_text(yaml.safe_dump(doc, allow_unicode=True), encoding="utf-8")
    return path


def test_confirmatory_hypotheses_with_tbd_rules_are_refused_not_guessed(  # type: ignore[no-untyped-def]
    runs_dir, tbd_rules, tmp_path: Path
) -> None:
    out = tmp_path / "report"
    with pytest.raises(SystemExit, match="is TBD"):
        cli.main(_args(runs_dir(main=[FusionMethod.B0, FusionMethod.F2]), out,
                       "--allow-draft-rules", rules=tbd_rules))
    assert not out.exists()


def test_draft_rules_are_refused_without_the_flag(  # type: ignore[no-untyped-def]
    runs_dir, exploratory_rules, tmp_path: Path
) -> None:
    out = tmp_path / "report"
    with pytest.raises(SystemExit, match="needs 'locked'"):
        cli.main(_args(runs_dir(debug=[FusionMethod.B0]), out, rules=exploratory_rules))
    assert not out.exists()


def test_debug_records_never_reach_the_result_table(  # type: ignore[no-untyped-def]
    runs_dir, exploratory_rules, tmp_path: Path, capsys
) -> None:
    out = tmp_path / "report"
    runs = runs_dir(debug=[FusionMethod.B0, FusionMethod.F4])
    paths = cli.main(_args(runs, out, "--allow-draft-rules", rules=exploratory_rules))

    printed = capsys.readouterr().out
    assert "draft rules (not for the report)" in printed
    assert "records: 2 loaded, 0 main, 2 excluded (debug/reproduction)" in printed
    assert not (out / "summary_table.csv").exists()
    assert all(p.name != "summary_table.csv" for p in paths)


def test_only_main_cells_appear_when_records_are_mixed(  # type: ignore[no-untyped-def]
    runs_dir, exploratory_rules, tmp_path: Path, capsys
) -> None:
    out = tmp_path / "report"
    runs = runs_dir(main=[FusionMethod.B0], debug=[FusionMethod.F2, FusionMethod.F4])
    paths = cli.main(_args(runs, out, "--allow-draft-rules", rules=exploratory_rules))

    assert "records: 3 loaded, 1 main, 2 excluded (debug/reproduction)" in capsys.readouterr().out
    assert out / "summary_table.csv" in paths
    with open(out / "summary_table.csv", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["fusion_method"] for row in rows] == ["B0"]
    for path in paths:
        if path.suffix == ".csv":
            assert "debug" not in path.read_text(encoding="utf-8")


def test_existing_report_is_never_overwritten(  # type: ignore[no-untyped-def]
    runs_dir, exploratory_rules, tmp_path: Path
) -> None:
    out = tmp_path / "report"
    runs = runs_dir(main=[FusionMethod.B0])
    cli.main(_args(runs, out, "--allow-draft-rules", rules=exploratory_rules))
    before = (out / "summary_table.csv").read_bytes()
    with pytest.raises(SystemExit, match="already contains report artifacts"):
        cli.main(_args(runs, out, "--allow-draft-rules", rules=exploratory_rules))
    assert (out / "summary_table.csv").read_bytes() == before


def test_missing_runs_directory_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="no runs directory"):
        cli.main(_args(tmp_path / "nope", tmp_path / "report", "--allow-draft-rules"))
