"""T-07b unit tests for the judgment core (loader, paired McNemar, seed aggregation).

Seam ② behavior lives in tests/test_judgment_seam.py; these tests pin the public helper
functions and their error paths on hand-crafted data (no GPU, no real dataset).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from layer_fusion.contracts import PredictionRecord
from layer_fusion.judgment import compute_paired_mcnemar, load_decision_rules

ROOT = Path(__file__).resolve().parent.parent


def _preds(correct: list[bool]) -> list[PredictionRecord]:
    """Binary task, label 1 everywhere; prediction is right or wrong per entry."""
    return [
        PredictionRecord(doc_id=f"d{i:03d}", label=1, prediction=1 if ok else 0)
        for i, ok in enumerate(correct)
    ]


def _pair_with_discordance(b: int, c: int, both_right: int = 0, both_wrong: int = 0):
    """A right/B wrong on `b` docs, A wrong/B right on `c` docs (plus concordant docs)."""
    a_ok = [True] * b + [False] * c + [True] * both_right + [False] * both_wrong
    b_ok = [False] * b + [True] * c + [True] * both_right + [False] * both_wrong
    return _preds(a_ok), _preds(b_ok)


def test_mcnemar_matches_hand_calculated_exact_p_value() -> None:
    # b=8, c=2 -> n=10, two-sided exact p = 2 * (1 + 10 + 45) / 1024 = 0.109375
    a, b = _pair_with_discordance(8, 2, both_right=30, both_wrong=5)
    assert compute_paired_mcnemar(a, b) == pytest.approx(0.109375)

    # b=10, c=0 -> p = 2 * 0.5**10
    a, b = _pair_with_discordance(10, 0)
    assert compute_paired_mcnemar(a, b) == pytest.approx(0.001953125)


def test_mcnemar_without_discordant_pairs_is_not_significant() -> None:
    a, b = _pair_with_discordance(0, 0, both_right=20, both_wrong=3)
    assert compute_paired_mcnemar(a, b) == 1.0


def test_mcnemar_rejects_different_document_sets() -> None:
    a, b = _pair_with_discordance(3, 1)
    with pytest.raises(ValueError, match="same documents"):
        compute_paired_mcnemar(a, b[:-1])


def test_mcnemar_rejects_duplicate_doc_ids() -> None:
    a, b = _pair_with_discordance(3, 1)
    with pytest.raises(ValueError, match="duplicate doc_id"):
        compute_paired_mcnemar(a + [a[0]], b + [b[0]])


def test_mcnemar_rejects_documents_with_different_labels() -> None:
    a, b = _pair_with_discordance(3, 1)
    b[0] = PredictionRecord(doc_id=b[0].doc_id, label=0, prediction=b[0].prediction)
    with pytest.raises(ValueError, match="different labels"):
        compute_paired_mcnemar(a, b)


# --- load_decision_rules -------------------------------------------------------------------

RULES_YAML = ROOT / "configs" / "decision_rules.yaml"


def test_loader_reads_the_real_rules_file_and_keeps_tbd_values_unset() -> None:
    rules = load_decision_rules(str(RULES_YAML))

    assert rules.status == "draft"
    assert rules.freeze_date == "2026-11-30"
    assert rules.core_num_seeds == 5
    assert rules.secondary_num_seeds == 3
    # TBD until the 11/30 lock: never guessed, left as None (AGENTS.md §2).
    assert rules.mcnemar_alpha is None
    assert rules.multi_seed_mcnemar_aggregation is None
    assert rules.h1_target_fusion_rule is None
    assert rules.seed_rank_corr_min is None


def test_loader_exposes_hypothesis_type_and_priority_from_the_file() -> None:
    hypotheses = load_decision_rules(str(RULES_YAML)).extra["hypotheses"]

    assert hypotheses["H1"]["type"] == "confirmatory"
    assert hypotheses["H1"]["priority"] == 1
    assert hypotheses["H4"]["type"] == "exploratory"
    assert hypotheses["H5c"]["type"] == "exploratory"
    assert {h for h, body in hypotheses.items() if body["type"] == "exploratory"} == {"H4", "H5c"}


def test_loader_can_refuse_rules_that_are_not_locked() -> None:
    with pytest.raises(ValueError, match="locked"):
        load_decision_rules(str(RULES_YAML), require_locked=True)


def test_loader_rejects_a_file_without_the_required_keys(tmp_path: Path) -> None:
    bad = tmp_path / "rules.yaml"
    bad.write_text("freeze_date: '2026-11-30'\nstatus: draft\ncriteria: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="core_num_seeds"):
        load_decision_rules(str(bad))
