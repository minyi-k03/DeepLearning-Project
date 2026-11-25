"""Judgment seam: pure hypothesis evaluation from ResultRecords + DecisionRules.

References: docs/SPEC.md, ADR 0007.
"""

from __future__ import annotations

import itertools
import math
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass, fields, replace
from typing import Any

import yaml
from scipy.stats import binomtest, spearmanr

from layer_fusion.contracts import (
    BackboneCondition,
    DataAmount,
    DatasetName,
    DecisionRules,
    FusionMethod,
    HypothesisJudgment,
    InputSetting,
    JudgmentReport,
    LayerSubset,
    LengthBucket,
    PredictionRecord,
    ResultRecord,
    RunConfig,
    RunKind,
    TrackKind,
    Verdict,
)
from layer_fusion.layers import EMBEDDING_SLOT, GLOBAL_LAYERS, slots_for_subset

_CRITERIA_FIELDS = (
    "mcnemar_alpha",
    "multiple_comparison_correction",
    "multi_seed_mcnemar_aggregation",
    "min_bucket_samples",
    "below_min_sample_handling",
    "core_num_seeds",
    "secondary_num_seeds",
    "core_comparisons_for_5_seeds",
    "finetune_num_seeds",
    "best_fusion_selection_rule",
)
_LAYER_WEIGHT_FIELDS = (
    "include_embedding_in_cog",
    "include_embedding_in_global_denom",
    "seed_rank_corr_min",
    "cog_seed_std_max",
)
_REQUIRED_RULE_KEYS = ("freeze_date", "status", "core_num_seeds", "secondary_num_seeds")


def load_decision_rules(path: str, *, require_locked: bool = False) -> DecisionRules:
    """Load frozen decision rules from configs/decision_rules.yaml.

    Values are taken as written: a ``null`` (TBD until the 11/30 lock) stays ``None`` and is
    never filled with a guess (AGENTS.md §2). The per-hypothesis ``type``/``priority`` block
    is kept as ``rules.extra["hypotheses"]`` so judgment never hard-codes which hypotheses
    are confirmatory (ADR 0007). ``require_locked=True`` refuses a file whose status is not
    ``locked``; use it for judging main-experiment results.
    """
    with open(path, encoding="utf-8") as handle:
        doc = yaml.safe_load(handle)
    if not isinstance(doc, dict):
        raise ValueError(f"{path}: decision rules must be a YAML mapping")

    criteria = doc.get("criteria") or {}
    layer_weights = doc.get("layer_weight_interpretation") or {}
    hypotheses = doc.get("hypotheses") or {}

    values: dict[str, Any] = {key: doc.get(key) for key in ("freeze_date", "status")}
    values.update({key: criteria.get(key) for key in _CRITERIA_FIELDS})
    values.update({key: layer_weights.get(key) for key in _LAYER_WEIGHT_FIELDS})
    missing = [key for key in _REQUIRED_RULE_KEYS if values.get(key) is None]
    if missing:
        raise ValueError(f"{path}: required decision rule keys missing: {missing}")
    if require_locked and values["status"] != "locked":
        raise ValueError(
            f"{path}: status is {values['status']!r}; main-experiment judgment needs 'locked'"
        )

    extra: dict[str, Any] = {"hypotheses": hypotheses}
    known = set(_CRITERIA_FIELDS) | set(_LAYER_WEIGHT_FIELDS)
    for block in (criteria, layer_weights):
        extra.update({key: val for key, val in block.items() if key not in known})

    return DecisionRules(
        h1_target_fusion_rule=(hypotheses.get("H1") or {}).get("target_fusion_rule"),
        h5a_target_fusion_rule=(hypotheses.get("H5a") or {}).get("target_fusion_rule"),
        extra=extra,
        **values,
    )


def compute_paired_mcnemar(
    preds_a: Sequence[PredictionRecord],
    preds_b: Sequence[PredictionRecord],
) -> float:
    """Compute paired McNemar p-value aligned by doc_id.

    Two-sided exact binomial test on the discordant pairs (b: only A correct, c: only B
    correct). Documents are paired by ``doc_id``, never by position. With no discordant
    pairs there is no evidence of a difference, so the p-value is 1.0.
    """
    by_id_a = _index_by_doc_id(preds_a, "preds_a")
    by_id_b = _index_by_doc_id(preds_b, "preds_b")
    if not by_id_a or not by_id_b:
        # Without this, two empty runs would give p = 1.0 and read as NO_EFFECT.
        raise ValueError("McNemar needs at least one document on each side")
    if by_id_a.keys() != by_id_b.keys():
        only_a = sorted(by_id_a.keys() - by_id_b.keys())[:3]
        only_b = sorted(by_id_b.keys() - by_id_a.keys())[:3]
        raise ValueError(
            "McNemar needs the same documents on both sides; "
            f"only in A: {only_a}, only in B: {only_b} (first 3 shown)"
        )

    only_a_correct = 0
    only_b_correct = 0
    for doc_id, rec_a in by_id_a.items():
        rec_b = by_id_b[doc_id]
        if rec_a.label != rec_b.label:
            raise ValueError(
                f"doc_id {doc_id!r} has different labels ({rec_a.label} vs {rec_b.label})"
            )
        a_ok = rec_a.prediction == rec_a.label
        b_ok = rec_b.prediction == rec_b.label
        if a_ok and not b_ok:
            only_a_correct += 1
        elif b_ok and not a_ok:
            only_b_correct += 1

    discordant = only_a_correct + only_b_correct
    if discordant == 0:
        return 1.0
    result = binomtest(min(only_a_correct, only_b_correct), discordant, 0.5)
    return float(result.pvalue)


def _index_by_doc_id(preds: Sequence[PredictionRecord], name: str) -> dict[str, PredictionRecord]:
    indexed: dict[str, PredictionRecord] = {}
    for rec in preds:
        if rec.doc_id in indexed:
            raise ValueError(f"{name} has duplicate doc_id {rec.doc_id!r}")
        indexed[rec.doc_id] = rec
    return indexed


def _slot_weights(weights: Sequence[float], layer_subset: LayerSubset) -> dict[int, float]:
    """Map a layer-weight vector to ``{slot: weight}`` (slot 0 = embedding, 1..22 = layer number).

    ``weights`` must follow ``slots_for_subset(layer_subset)`` order, as the F2 head exposes it.
    """
    slots = slots_for_subset(layer_subset)
    if len(weights) != len(slots):
        raise ValueError(
            f"layer subset {layer_subset.value} expects {len(slots)} weights, got {len(weights)}"
        )
    if not all(math.isfinite(w) for w in weights):
        raise ValueError(f"layer weights must be finite, got {list(weights)}")
    if any(w < 0 for w in weights):
        raise ValueError(f"layer weights must be non-negative, got {list(weights)}")
    return dict(zip(slots, weights, strict=True))


def _without_embedding_unless(slot_weights: dict[int, float], keep: bool) -> dict[int, float]:
    if keep:
        return slot_weights
    return {slot: w for slot, w in slot_weights.items() if slot != EMBEDDING_SLOT}


def _total_weight(slot_weights: dict[int, float]) -> float:
    total = sum(slot_weights.values())
    if total <= 0:
        raise ValueError("the counted layers carry no weight, so the summary is undefined")
    return total


def compute_center_of_gravity(
    weights: Sequence[float],
    include_embedding: bool,
    *,
    layer_subset: LayerSubset = LayerSubset.ALL,
) -> float:
    """Compute center of gravity (무게중심): the layer-number mean weighted by layer weights.

    ``weights`` follow ``slots_for_subset(layer_subset)`` order (23 for ALL, 8 global-only, 14
    local-only). The embedding vector has no layer number (GLOSSARY); ``include_embedding`` is
    the rule value (``include_embedding_in_cog``, TBD until the 11/30 lock): ``False`` drops it
    and renormalises over layers 1..22, ``True`` counts it at position 0. It has no effect on
    the ablation subsets, which never contain the embedding vector.
    """
    counted = _without_embedding_unless(_slot_weights(weights, layer_subset), include_embedding)
    total = _total_weight(counted)
    return sum(slot * w for slot, w in counted.items()) / total


def compute_global_layer_weight_ratio(
    weights: Sequence[float],
    include_embedding_in_denom: bool,
    *,
    layer_subset: LayerSubset = LayerSubset.ALL,
) -> float:
    """Compute global layer weight share (전역 층 가중치 비율).

    The share of the counted weight that sits on global layers. ``include_embedding_in_denom``
    is the rule value (``include_embedding_in_global_denom``, TBD until the 11/30 lock): whether
    the embedding vector's weight stays in the denominator (it is never in the numerator, being
    neither a global nor a local layer). Global-only weights give 1.0 and local-only give 0.0.
    """
    counted = _without_embedding_unless(
        _slot_weights(weights, layer_subset), include_embedding_in_denom
    )
    total = _total_weight(counted)
    return sum(w for slot, w in counted.items() if slot in GLOBAL_LAYERS) / total


def judge_hypotheses(
    records: Sequence[ResultRecord],
    rules: DecisionRules,
) -> JudgmentReport:
    """Judgment seam: (ResultRecord[], DecisionRules) -> JudgmentReport.

    Contract:
    - Excludes run_kind == DEBUG and track == REPRODUCTION records.
    - Confirmatory hypotheses (H1, H3, H2, H4', H5b, H5a): SUPPORTED only if
      difference >= seed std AND paired McNemar is significant; else NO_EFFECT.
    - Exploratory hypotheses (H4, H5c): EXPLORATORY (no SUPPORTED/NO_EFFECT verdict).
    - Holds layer-weight interpretation if seed consistency fails thresholds.
    - Reports H5b decomposed into mixing effect (F1 - B0) and weight learning (F2 - F1).

    T-07b implements record filtering, the exploratory rows and H1. A confirmatory hypothesis
    with no registered evaluator (H3, H2, H4', H5a, H5b: T-07c) or without the records it
    needs is left out of ``judgments`` rather than reported as NO_EFFECT. Hypothesis kinds come
    from ``rules.extra["hypotheses"]`` (see ``load_decision_rules``), never from this module.
    """
    kinds = _hypothesis_kinds(rules)
    main_records = select_main_records(records)
    layer_weight_analysis = _layer_weight_analysis(main_records, rules)
    exploratory_evidence: dict[str, Callable[[], dict[str, Any]]] = {
        "H4": lambda: _h4_details(main_records, rules, layer_weight_analysis),
        "H5c": lambda: _h5c_details(main_records, layer_weight_analysis),
    }

    judgments: dict[str, HypothesisJudgment] = {}
    for hypothesis_id, kind in kinds.items():
        if kind == "exploratory":
            evidence = exploratory_evidence.get(hypothesis_id)
            judgments[hypothesis_id] = HypothesisJudgment(
                hypothesis_id=hypothesis_id,
                kind=kind,
                verdict=Verdict.EXPLORATORY,
                details={
                    "note": "exploratory: trend only, no confirmatory verdict (ADR 0007)",
                    **(evidence() if evidence is not None else {}),
                },
            )
            continue
        evaluator = _CONFIRMATORY_EVALUATORS.get(hypothesis_id)
        if evaluator is None:
            continue
        judgment = evaluator(main_records, rules, kind)
        if judgment is not None:
            judgments[hypothesis_id] = judgment
    return JudgmentReport(
        judgments=judgments,
        layer_weight_analysis=layer_weight_analysis,
        length_bucket_analysis=_length_bucket_analysis(main_records, rules),
    )


def _cell_key(
    dataset: DatasetName,
    method: FusionMethod,
    setting: InputSetting | None = None,
    subset: LayerSubset = LayerSubset.ALL,
) -> str:
    """Key of a frozen-backbone / full-data cell, e.g. ``imdb/full/frozen/Lfull/F2/all``."""
    return _cell_label_of(
        dataset, DataAmount.FULL, BackboneCondition.FROZEN, setting, method, subset
    )


def _cell_label_of(
    dataset: DatasetName,
    amount: DataAmount,
    backbone: BackboneCondition,
    setting: InputSetting | None,
    method: FusionMethod,
    subset: LayerSubset,
) -> str:
    parts = (
        dataset.value,
        amount.value,
        backbone.value,
        "" if setting is None else setting.value,
        method.value,
        subset.value,
    )
    return "/".join(part for part in parts if part)


def _cell_label(config: RunConfig) -> str:
    """Stable key of one experiment cell, e.g. ``imdb/full/frozen/Lfull/F2/all``."""
    return _cell_label_of(
        config.dataset,
        config.data_amount,
        config.backbone_condition,
        config.input_setting,
        config.fusion_method,
        config.layer_subset,
    )


def _tbd_rules(rules: DecisionRules, names: Sequence[str]) -> str | None:
    """Message naming the rule values among ``names`` that are still TBD, else ``None``.

    The layer-weight summaries are auxiliary to the verdicts, so a TBD value there is reported
    in the analysis cell (``"unavailable"``) instead of stopping every other hypothesis.
    """
    missing = [name for name in names if getattr(rules, name) is None]
    if not missing:
        return None
    return "; ".join(f"{name} is TBD (fixed at the 11/30 lock, T-07a)" for name in missing)


def _spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Spearman rank correlation, or ``None`` when it is undefined (fewer than 2 points, or one
    side constant): an undefined correlation is never reported as 0 or as a perfect match."""
    if len(x) < 2 or len(set(x)) < 2 or len(set(y)) < 2:
        return None
    return float(spearmanr(x, y).statistic)


def _mean_pairwise_rank_corr(weight_sets: Sequence[Sequence[float]]) -> float | None:
    """Mean Spearman correlation over all seed pairs; ``None`` if any pair is undefined.

    A correlation is undefined when one seed's weights are all equal (no ranking). That is
    treated as no consistency evidence by the caller, never as a perfect or zero correlation.
    """
    correlations = [_spearman(a, b) for a, b in itertools.combinations(weight_sets, 2)]
    if any(c is None for c in correlations):
        return None
    return statistics.fmean(c for c in correlations if c is not None)


def _analyse_layer_weight_cell(
    cell: Sequence[ResultRecord], rules: DecisionRules
) -> dict[str, Any]:
    """Center of gravity, global ratio and seed consistency of one F2 cell (시드 일관성).

    Layer weights are interpreted only if both consistency thresholds hold: the mean seed-pair
    rank correlation is at least ``seed_rank_corr_min`` and the seed std of the center of
    gravity is at most ``cog_seed_std_max``. Otherwise ``interpretation`` is ``Verdict.HELD``.
    """
    unavailable = _tbd_rules(
        rules,
        (
            "include_embedding_in_cog",
            "include_embedding_in_global_denom",
            "seed_rank_corr_min",
            "cog_seed_std_max",
        ),
    )
    if unavailable is not None:
        return {"unavailable": unavailable}
    include_cog = rules.include_embedding_in_cog
    include_denominator = rules.include_embedding_in_global_denom
    min_corr = rules.seed_rank_corr_min
    max_cog_std = rules.cog_seed_std_max

    label = _cell_label(cell[0].config)
    by_seed = _records_by_seed(cell, label)
    seeds = sorted(by_seed)
    subset = cell[0].config.layer_subset
    weight_sets = [by_seed[s].layer_weights or [] for s in seeds]

    cog = [
        compute_center_of_gravity(w, include_cog, layer_subset=subset) for w in weight_sets
    ]
    ratios = [
        compute_global_layer_weight_ratio(w, include_denominator, layer_subset=subset)
        for w in weight_sets
    ]
    cog_mean = statistics.fmean(cog)
    # One seed cannot show consistency (no std, no pair): the reading is held, never assumed.
    cog_std = seed_mean_std(cog)[1] if len(cog) > 1 else None
    rank_corr = _mean_pairwise_rank_corr(weight_sets) if len(weight_sets) > 1 else None
    consistent = (
        rank_corr is not None
        and cog_std is not None
        and rank_corr >= min_corr
        and cog_std <= max_cog_std
    )
    return {
        "n_seeds": len(seeds),
        "seeds": seeds,
        "center_of_gravity_per_seed": cog,
        "center_of_gravity_mean": cog_mean,
        "cog_seed_std": cog_std,
        "global_ratio_per_seed": ratios,
        "global_ratio_mean": statistics.fmean(ratios),
        "rank_corr_mean": rank_corr,
        "rank_corr_min": min_corr,
        "cog_seed_std_max": max_cog_std,
        "seed_consistent": consistent,
        "interpretation": None if consistent else Verdict.HELD,
    }


def _f4_length_correlation(
    label: str, cell: Sequence[ResultRecord], rules: DecisionRules
) -> dict[str, Any]:
    """Rank correlation of document length and document-level global layer share (F4, H5c aid).

    Per seed: the share of each document's layer weights on global layers against its token
    length, over all documents and within each label (SPEC US 72, ADR 0006: exploratory, only
    "the gate reacts to length"). The reported value is the mean over seeds; ``None`` if the
    correlation is undefined in any seed.
    """
    unavailable = _tbd_rules(rules, ("include_embedding_in_global_denom",))
    if unavailable is not None:
        return {"unavailable": unavailable}
    include_denominator = bool(rules.include_embedding_in_global_denom)
    by_seed = _records_by_seed(cell, label)
    per_seed: dict[int, dict[str, float | None]] = {}
    n_docs = 0
    for seed, rec in by_seed.items():
        lengths: list[float] = []
        shares: list[float] = []
        labels: list[int] = []
        for pred in rec.predictions:
            if pred.layer_weights is None or pred.token_length is None:
                raise ValueError(
                    f"{label}: doc {pred.doc_id!r} of run {rec.run_id!r} lacks layer_weights "
                    "or token_length"
                )
            lengths.append(pred.token_length)
            shares.append(
                compute_global_layer_weight_ratio(pred.layer_weights, include_denominator)
            )
            labels.append(pred.label)
        n_docs = len(lengths)
        groups = {"all": [True] * n_docs}
        for value in sorted(set(labels)):
            groups[f"label_{value}"] = [lab == value for lab in labels]
        per_seed[seed] = {
            name: _spearman(
                [x for x, keep in zip(lengths, mask, strict=True) if keep],
                [y for y, keep in zip(shares, mask, strict=True) if keep],
            )
            for name, mask in groups.items()
        }

    names = sorted({name for corr in per_seed.values() for name in corr})
    mean_corr: dict[str, float | None] = {}
    for name in names:
        values = [corr.get(name) for corr in per_seed.values()]
        mean_corr[name] = (
            None
            if any(v is None for v in values)
            else statistics.fmean(v for v in values if v is not None)
        )
    return {
        "length_global_ratio_corr": mean_corr,
        "per_seed": per_seed,
        "n_seeds": len(per_seed),
        "n_docs": n_docs,
    }


def _held_if_any(analysis: dict[str, dict[str, Any]], labels: Sequence[str]) -> Verdict | None:
    """``Verdict.HELD`` if any listed layer-weight cell failed its seed-consistency gate."""
    return (
        Verdict.HELD
        if any(analysis[label].get("interpretation") is Verdict.HELD for label in labels)
        else None
    )


def _h4_details(
    records: Sequence[ResultRecord],
    rules: DecisionRules,
    analysis: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """H4 evidence (exploratory): F2 center of gravity MNLI vs SST-2, SST-2 B0 comparison."""
    cells = {
        "mnli": _cell_key(DatasetName.MNLI, FusionMethod.F2),
        "sst2": _cell_key(DatasetName.SST2, FusionMethod.F2),
    }
    present = {
        name: label
        for name, label in cells.items()
        if "center_of_gravity_mean" in analysis.get(label, {})
    }
    sst2_b0 = _records_for(records, DatasetName.SST2, FusionMethod.B0)
    comparison = None
    if sst2_b0:
        target = _records_for(records, DatasetName.SST2, _h1_target_method(rules))
        if target:
            judged = _judge_comparison("H4", "exploratory", sst2_b0, target, rules)
            # Numbers only: an exploratory hypothesis never carries a verdict (ADR 0007).
            comparison = {
                "mean_diff": judged.mean_diff,
                "seed_std": judged.seed_std,
                "mcnemar_p_value": judged.mcnemar_p_value,
                "n_seeds": judged.details["n_seeds"],
            }
    return {
        "center_of_gravity": {n: analysis[c]["center_of_gravity_mean"] for n, c in present.items()},
        "global_ratio": {n: analysis[c]["global_ratio_mean"] for n, c in present.items()},
        "layer_weight_interpretation": _held_if_any(analysis, list(present.values())),
        "sst2_error_comparison": comparison,
    }


def _h5c_details(
    records: Sequence[ResultRecord], analysis: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """H5c evidence (exploratory): per-setting F2 summaries, global/local ablation, F4 length."""
    settings = (InputSetting.L128, InputSetting.L512, InputSetting.LFULL)
    by_setting: dict[str, dict[str, float]] = {}
    ablation: dict[str, dict[str, float]] = {}
    for setting in settings:
        cell = analysis.get(_cell_key(DatasetName.IMDB, FusionMethod.F2, setting))
        if cell is not None and "center_of_gravity_mean" in cell:
            by_setting[setting.value] = {
                "center_of_gravity_mean": cell["center_of_gravity_mean"],
                "global_ratio_mean": cell["global_ratio_mean"],
            }
        errors = {
            subset.value: _mean_error(recs)
            for subset in LayerSubset
            if (recs := _imdb_records(records, FusionMethod.F2, setting, subset))
        }
        if LayerSubset.GLOBAL_ONLY.value in errors or LayerSubset.LOCAL_ONLY.value in errors:
            ablation[setting.value] = errors
    f4_cell = analysis.get(_cell_key(DatasetName.IMDB, FusionMethod.F4, InputSetting.LFULL), {})
    imdb_labels = [label for label in analysis if label.startswith("imdb/")]
    f4_corr = f4_cell.get("length_global_ratio_corr")
    return {
        "by_input_setting": by_setting,
        "ablation_error_rate": ablation,
        "f4_length_global_ratio_corr": f4_corr,
        "layer_weight_interpretation": _held_if_any(analysis, imdb_labels),
    }


def _layer_weight_analysis(
    main_records: Sequence[ResultRecord], rules: DecisionRules
) -> dict[str, dict[str, Any]]:
    """Per F2 cell: layer-weight summaries and the seed-consistency gate (SPEC US 68, 69); per F4
    cell with document weights: the length / global-share rank correlation (SPEC US 72)."""
    cells: dict[str, list[ResultRecord]] = {}
    for rec in main_records:
        if rec.config.fusion_method is FusionMethod.F2:
            cells.setdefault(_cell_label(rec.config), []).append(rec)

    analysis: dict[str, dict[str, Any]] = {}
    for label, cell in cells.items():
        with_weights = [rec for rec in cell if rec.layer_weights is not None]
        if not with_weights:
            continue  # nothing to interpret; the report module insists on complete cells
        if len(with_weights) != len(cell):
            missing = [rec.run_id for rec in cell if rec.layer_weights is None]
            raise ValueError(f"{label}: layer_weights missing for runs {missing}")
        analysis[label] = _analyse_layer_weight_cell(cell, rules)

    f4_cells: dict[str, list[ResultRecord]] = {}
    for rec in main_records:
        if (
            rec.config.fusion_method is FusionMethod.F4
            and rec.config.dataset is DatasetName.IMDB  # SPEC US 72: Lfull F4 on IMDB only
            and any(p.layer_weights is not None for p in rec.predictions)
        ):
            f4_cells.setdefault(_cell_label(rec.config), []).append(rec)
    for label, cell in f4_cells.items():
        analysis[label] = _f4_length_correlation(label, cell, rules)
    return analysis


def _bucket_rows(
    label: str, cell: Sequence[ResultRecord], min_bucket_samples: int | None
) -> dict[str, dict[str, Any]]:
    """Per length bucket of one cell: document count and error rate (mean, std over seeds)."""
    by_seed = _records_by_seed(cell, label)
    rows: dict[str, dict[str, Any]] = {}
    for bucket in LengthBucket:
        in_bucket = [
            [p for p in by_seed[seed].predictions if p.length_bucket is bucket]
            for seed in sorted(by_seed)
        ]
        counts = {len(docs) for docs in in_bucket}
        if len(counts) != 1:
            raise ValueError(f"{label}: seeds disagree on the document count of bucket {bucket}")
        n_docs = counts.pop()
        if n_docs == 0:
            continue
        errors = [sum(p.prediction != p.label for p in docs) / n_docs for docs in in_bucket]
        rows[bucket.value] = {
            "n_docs": n_docs,
            "n_seeds": len(errors),
            "error_rate_mean": statistics.fmean(errors),
            "error_rate_std": statistics.stdev(errors) if len(errors) > 1 else None,
            # None while the minimum sample rule is TBD (T-07a); never guessed.
            "below_min_samples": (
                None if min_bucket_samples is None else n_docs < min_bucket_samples
            ),
        }
    return rows


def _short_documents(rec: ResultRecord) -> dict[str, tuple[int | None, int]]:
    """doc_id -> (token length, label) for the documents in the <=512 bucket."""
    return {
        p.doc_id: (p.token_length, p.label)
        for p in rec.predictions
        if p.length_bucket is LengthBucket.LE_512
    }


def _le512_check(
    label: str, l512: Sequence[ResultRecord], lfull: Sequence[ResultRecord]
) -> dict[str, Any]:
    """Pipeline sanity check: up to 512 tokens L512 and Lfull feed the model the same input.

    The two settings must evaluate the same short documents with the same token lengths (else
    ``ValueError``). The error-rate gap is reported, not judged: the classifiers were trained on
    different training inputs, so equal predictions are not required and no threshold is set.
    """
    by_l512 = _records_by_seed(l512, f"{label} L512")
    by_lfull = _records_by_seed(lfull, f"{label} Lfull")
    if by_l512.keys() != by_lfull.keys():
        raise ValueError(
            f"{label}: L512 seeds {sorted(by_l512)} != Lfull seeds {sorted(by_lfull)}"
        )
    rates: dict[str, list[float]] = {"L512": [], "Lfull": []}
    n_docs = 0
    for seed in sorted(by_l512):
        short_l512 = _short_documents(by_l512[seed])
        short_lfull = _short_documents(by_lfull[seed])
        if short_l512 != short_lfull:
            raise ValueError(
                f"{label} seed {seed}: L512 and Lfull disagree on the short documents "
                "(<=512 tokens) or their token lengths"
            )
        n_docs = len(short_l512)
        for name, rec in (("L512", by_l512[seed]), ("Lfull", by_lfull[seed])):
            wrong = sum(
                p.prediction != p.label for p in rec.predictions if p.doc_id in short_l512
            )
            rates[name].append(wrong / n_docs if n_docs else math.nan)
    error_rate = {name: statistics.fmean(values) for name, values in rates.items()}
    return {
        "same_documents": True,
        "n_docs": n_docs,
        "error_rate": error_rate,
        "error_rate_gap": error_rate["L512"] - error_rate["Lfull"],
    }


def _length_bucket_analysis(
    main_records: Sequence[ResultRecord], rules: DecisionRules
) -> dict[str, Any]:
    """Per cell error rates by length bucket, plus the L512 = Lfull check on short documents."""
    cells: dict[str, list[ResultRecord]] = {}
    for rec in main_records:
        if any(p.length_bucket is not None for p in rec.predictions):
            cells.setdefault(_cell_label(rec.config), []).append(rec)
    if not cells:
        return {}

    checks: dict[str, Any] = {}
    methods = {
        (rec.config.fusion_method, rec.config.layer_subset)
        for cell in cells.values()
        for rec in cell
        if rec.config.dataset is DatasetName.IMDB
    }
    for method, subset in sorted(methods, key=lambda m: (m[0].value, m[1].value)):
        l512 = _imdb_records(main_records, method, InputSetting.L512, subset)
        lfull = _imdb_records(main_records, method, InputSetting.LFULL, subset)
        if l512 and lfull:
            key = method.value if subset is LayerSubset.ALL else f"{method.value}/{subset.value}"
            checks[key] = _le512_check(f"imdb {key}", l512, lfull)
    return {
        "buckets": {
            label: _bucket_rows(label, cell, rules.min_bucket_samples)
            for label, cell in cells.items()
        },
        "le512_check": checks,
    }


def select_main_records(records: Sequence[ResultRecord]) -> list[ResultRecord]:
    """Drop debug and reproduction-track records; they never enter judgment (ADR 0004, 0009)."""
    return [
        rec
        for rec in records
        if rec.config.run_kind is not RunKind.DEBUG
        and rec.config.track is not TrackKind.REPRODUCTION
    ]


def seed_mean_std(values: Sequence[float]) -> tuple[float, float]:
    """Mean and sample standard deviation (ddof=1) across seeds; needs at least 2 seeds."""
    if len(values) < 2:
        raise ValueError(f"seed std needs at least 2 seeds, got {len(values)}")
    if not all(math.isfinite(v) for v in values):
        raise ValueError(f"seed values must be finite, got {list(values)}")
    return statistics.fmean(values), statistics.stdev(values)


def _kth_smallest_p(p_values: Sequence[float], k: int) -> float:
    return sorted(p_values)[k - 1]


# Each strategy returns the representative p-value: significant exactly when it is < alpha.
# SPEC leaves the combination rule TBD (fixed at the 11/30 lock, T-07a). These two strategies
# and their names are T-07b implementation choices so the seam can be exercised; T-07a picks
# one (or adds another here). A rules value of None or any other name is refused.
MCNEMAR_SEED_AGGREGATIONS: dict[str, Callable[[Sequence[float]], float]] = {
    # every seed must be significant -> the weakest (largest) p decides
    "all_seeds_significant": lambda ps: _kth_smallest_p(ps, len(ps)),
    # strictly more than half of the seeds must be significant -> the (n//2+1)-th smallest p
    "majority_of_seeds": lambda ps: _kth_smallest_p(ps, len(ps) // 2 + 1),
}


def _hypothesis_kinds(rules: DecisionRules) -> dict[str, str]:
    hypotheses = rules.extra.get("hypotheses")
    if not hypotheses:
        raise ValueError(
            "rules.extra['hypotheses'] is missing; load the rules with load_decision_rules"
        )
    return {hypothesis_id: body["type"] for hypothesis_id, body in hypotheses.items()}


# Inside one B0-vs-candidate comparison only these RunConfig fields may differ. Everything else
# (input setting, epochs, early stopping, lr candidates, cache, ...) must match, because SPEC
# module 7 trains every fusion method with the same epochs and early stopping, and averaging
# records from different cells would silently judge a different experiment. fixed_lr and
# lr_selection_seed differ by design: the first seed of each setting selects the LR and the
# other seeds reuse it (SPEC §결정 필요 사항, T-06a), each fusion method with its own choice.
_COMPARISON_FREE_FIELDS = frozenset({"seed", "fusion_method", "fixed_lr", "lr_selection_seed"})


# H5a/H5b compare input settings, so the input setting is one more field allowed to vary.
_LENGTH_COMPARISON_FREE_FIELDS = _COMPARISON_FREE_FIELDS | {"input_setting"}


def _require_one_cell(
    hypothesis_id: str,
    records: Sequence[ResultRecord],
    free_fields: frozenset[str] = _COMPARISON_FREE_FIELDS,
) -> None:
    """Refuse records that differ in a field outside ``free_fields``.

    A comparison that is *about* one more field (H5a: input setting) passes it in
    ``free_fields``; everything else must still match.
    """
    reference = records[0]
    for rec in records[1:]:
        differing = [
            f.name
            for f in fields(RunConfig)
            if f.name not in free_fields
            and getattr(rec.config, f.name) != getattr(reference.config, f.name)
        ]
        if differing:
            raise ValueError(
                f"{hypothesis_id}: records must come from one experiment cell (only "
                f"{sorted(free_fields)} may vary); {reference.run_id!r} and "
                f"{rec.run_id!r} differ in {differing}"
            )


def _records_by_seed(records: Sequence[ResultRecord], label: str) -> dict[int, ResultRecord]:
    by_seed: dict[int, ResultRecord] = {}
    for rec in records:
        if rec.config.seed in by_seed:
            raise ValueError(f"{label}: more than one record for seed {rec.config.seed}")
        by_seed[rec.config.seed] = rec
    return by_seed


def _mcnemar_rules(rules: DecisionRules) -> tuple[float, str]:
    """The significance level and seed aggregation name; TBD or unknown values are refused."""
    if rules.mcnemar_alpha is None:
        raise ValueError("mcnemar_alpha is TBD (fixed at the 11/30 lock, T-07a)")
    aggregation_name = rules.multi_seed_mcnemar_aggregation
    if aggregation_name is None:
        raise ValueError("multi_seed_mcnemar_aggregation is TBD (fixed at the 11/30 lock, T-07a)")
    if aggregation_name not in MCNEMAR_SEED_AGGREGATIONS:
        raise ValueError(
            f"unknown multi_seed_mcnemar_aggregation {aggregation_name!r}; "
            f"supported: {sorted(MCNEMAR_SEED_AGGREGATIONS)}"
        )
    return rules.mcnemar_alpha, aggregation_name


def _judge_comparison(
    hypothesis_id: str,
    kind: str,
    baseline: Sequence[ResultRecord],
    candidate: Sequence[ResultRecord],
    rules: DecisionRules,
    free_fields: frozenset[str] = _COMPARISON_FREE_FIELDS,
) -> HypothesisJudgment:
    """SUPPORTED iff the candidate improves on the baseline by at least the seed std and the
    paired McNemar test is significant; otherwise NO_EFFECT.

    Improvement = mean baseline error rate - mean candidate error rate (positive: candidate
    better). Seed std = the larger sample std of the two settings (T-07b implementation choice;
    SPEC/ADR 0007 only say "seed std"). McNemar is paired per seed (two-sided exact), a seed
    counts only if the candidate has the lower error rate in it, and the per-seed p-values are
    combined by ``rules.multi_seed_mcnemar_aggregation``. ``details["per_seed_mcnemar_p"]``
    keeps the raw two-sided p-values.
    """
    alpha, aggregation_name = _mcnemar_rules(rules)

    _require_one_cell(hypothesis_id, [*baseline, *candidate], free_fields)
    base_by_seed = _records_by_seed(baseline, f"{hypothesis_id} baseline")
    cand_by_seed = _records_by_seed(candidate, f"{hypothesis_id} candidate")
    if base_by_seed.keys() != cand_by_seed.keys():
        raise ValueError(
            f"{hypothesis_id}: baseline seeds {sorted(base_by_seed)} != "
            f"candidate seeds {sorted(cand_by_seed)}"
        )
    seeds = sorted(base_by_seed)

    base_mean, base_std = seed_mean_std([base_by_seed[s].eval_error_rate for s in seeds])
    cand_mean, cand_std = seed_mean_std([cand_by_seed[s].eval_error_rate for s in seeds])
    improvement = base_mean - cand_mean
    seed_std = max(base_std, cand_std)

    per_seed_p = {
        seed: compute_paired_mcnemar(base_by_seed[seed].predictions, cand_by_seed[seed].predictions)
        for seed in seeds
    }
    # McNemar is two-sided, but "candidate beats baseline" is directional: a seed where the
    # candidate is not better cannot support it, however significant the difference is.
    counted_p = [
        per_seed_p[seed]
        if cand_by_seed[seed].eval_error_rate < base_by_seed[seed].eval_error_rate
        else 1.0
        for seed in seeds
    ]
    representative_p = MCNEMAR_SEED_AGGREGATIONS[aggregation_name](counted_p)

    # A negative improvement can never reach the (non-negative) seed std, so a significant
    # difference in the wrong direction is NO_EFFECT without a separate direction check.
    supported = improvement >= seed_std and representative_p < alpha
    return HypothesisJudgment(
        hypothesis_id=hypothesis_id,
        kind=kind,
        verdict=Verdict.SUPPORTED if supported else Verdict.NO_EFFECT,
        mean_diff=improvement,
        seed_std=seed_std,
        mcnemar_p_value=representative_p,
        details={
            "per_seed_mcnemar_p": per_seed_p,
            "aggregation": aggregation_name,
            "alpha": alpha,
            "n_seeds": len(seeds),
            "baseline_mean_error": base_mean,
            "candidate_mean_error": cand_mean,
        },
    )


@dataclass(frozen=True)
class _Effect:
    """An error-rate improvement: ``reference`` records minus ``improved`` records, per seed.

    ``free_fields`` are the config fields allowed to differ between the two sides (the
    comparison is one experiment cell otherwise).
    """

    reference: Sequence[ResultRecord]
    improved: Sequence[ResultRecord]
    free_fields: frozenset[str] = _COMPARISON_FREE_FIELDS


def _effect_by_seed(
    hypothesis_id: str, name: str, effect: _Effect
) -> dict[int, tuple[ResultRecord, ResultRecord]]:
    _require_one_cell(hypothesis_id, [*effect.reference, *effect.improved], effect.free_fields)
    reference = _records_by_seed(effect.reference, f"{hypothesis_id} {name} reference")
    improved = _records_by_seed(effect.improved, f"{hypothesis_id} {name} improved")
    if reference.keys() != improved.keys():
        raise ValueError(
            f"{hypothesis_id}: {name} reference seeds {sorted(reference)} != "
            f"improved seeds {sorted(improved)}"
        )
    return {seed: (reference[seed], improved[seed]) for seed in sorted(reference)}


def _document_gains(reference: ResultRecord, improved: ResultRecord) -> dict[str, int]:
    """Per document: +1 if only the improved run is right, -1 if only the reference is, else 0."""
    by_id_ref = _index_by_doc_id(reference.predictions, "reference")
    by_id_imp = _index_by_doc_id(improved.predictions, "improved")
    if by_id_ref.keys() != by_id_imp.keys():
        raise ValueError(
            f"runs {reference.run_id!r} and {improved.run_id!r} must evaluate the same documents"
        )
    gains: dict[str, int] = {}
    for doc_id, rec_ref in by_id_ref.items():
        rec_imp = by_id_imp[doc_id]
        if rec_ref.label != rec_imp.label:
            raise ValueError(f"doc_id {doc_id!r} has different labels across the two runs")
        gains[doc_id] = int(rec_imp.prediction == rec_imp.label) - int(
            rec_ref.prediction == rec_ref.label
        )
    return gains


def _paired_sign_test(gains_a: dict[str, int], gains_b: dict[str, int]) -> float:
    """Two-sided exact sign test on the documents where effect A and effect B gain differently.

    The generalisation of McNemar to a difference of effects: with per-document gains in
    {-1, 0, +1}, count documents where A gains more than B against the reverse. No such
    document means no evidence of a difference (p = 1.0).
    """
    if gains_a.keys() != gains_b.keys():
        raise ValueError("the two effects must be measured on the same documents")
    a_ahead = sum(1 for doc in gains_a if gains_a[doc] > gains_b[doc])
    b_ahead = sum(1 for doc in gains_a if gains_a[doc] < gains_b[doc])
    if a_ahead + b_ahead == 0:
        return 1.0
    return float(binomtest(min(a_ahead, b_ahead), a_ahead + b_ahead, 0.5).pvalue)


def _judge_effect_difference(
    hypothesis_id: str,
    kind: str,
    larger: _Effect,
    smaller: _Effect,
    rules: DecisionRules,
    labels: tuple[str, str],
) -> HypothesisJudgment:
    """SUPPORTED iff the ``larger`` effect exceeds the ``smaller`` one by a clear margin.

    For H2/H3/H5b the claim is "effect X is bigger than effect Y" (small vs full data, frozen
    vs fine-tuned backbone, F2 vs B0 gain from the longer input). An effect is the seed-mean
    error-rate improvement (reference minus improved). SUPPORTED needs both: the effect gap is
    at least the larger of the two effects' seed stds, and the paired sign test on per-document
    gains (see ``_paired_sign_test``) is significant after the seed aggregation of
    ``rules.multi_seed_mcnemar_aggregation``. A seed counts only if its own larger effect
    exceeds its smaller one. This operationalisation is a T-07c implementation choice
    (SPEC §결정 필요 사항); T-07a confirms or replaces it.
    """
    alpha, aggregation_name = _mcnemar_rules(rules)
    pairs_larger = _effect_by_seed(hypothesis_id, labels[0], larger)
    pairs_smaller = _effect_by_seed(hypothesis_id, labels[1], smaller)
    if pairs_larger.keys() != pairs_smaller.keys():
        raise ValueError(
            f"{hypothesis_id}: seeds of the {labels[0]} effect {sorted(pairs_larger)} != seeds "
            f"of the {labels[1]} effect {sorted(pairs_smaller)}"
        )
    seeds = sorted(pairs_larger)

    def effect_values(pairs: dict[int, tuple[ResultRecord, ResultRecord]]) -> list[float]:
        return [pairs[s][0].eval_error_rate - pairs[s][1].eval_error_rate for s in seeds]

    values_larger = effect_values(pairs_larger)
    values_smaller = effect_values(pairs_smaller)
    mean_larger, std_larger = seed_mean_std(values_larger)
    mean_smaller, std_smaller = seed_mean_std(values_smaller)
    difference = mean_larger - mean_smaller
    seed_std = max(std_larger, std_smaller)

    per_seed_p = {
        seed: _paired_sign_test(
            _document_gains(*pairs_larger[seed]), _document_gains(*pairs_smaller[seed])
        )
        for seed in seeds
    }
    counted_p = [
        per_seed_p[seed] if values_larger[i] > values_smaller[i] else 1.0
        for i, seed in enumerate(seeds)
    ]
    representative_p = MCNEMAR_SEED_AGGREGATIONS[aggregation_name](counted_p)

    supported = difference >= seed_std and representative_p < alpha
    return HypothesisJudgment(
        hypothesis_id=hypothesis_id,
        kind=kind,
        verdict=Verdict.SUPPORTED if supported else Verdict.NO_EFFECT,
        mean_diff=difference,
        seed_std=seed_std,
        mcnemar_p_value=representative_p,
        details={
            "per_seed_sign_test_p": per_seed_p,
            "aggregation": aggregation_name,
            "alpha": alpha,
            "n_seeds": len(seeds),
            "effect_mean": {labels[0]: mean_larger, labels[1]: mean_smaller},
            "effect_seed_std": {labels[0]: std_larger, labels[1]: std_smaller},
        },
    )


def _records_for(
    records: Sequence[ResultRecord],
    dataset: DatasetName,
    method: FusionMethod,
    backbone: BackboneCondition = BackboneCondition.FROZEN,
    amount: DataAmount = DataAmount.FULL,
) -> list[ResultRecord]:
    """Records of one dataset / method / backbone / data amount (all layers, F2 subset ALL)."""
    return [
        rec
        for rec in records
        if rec.config.dataset is dataset
        and rec.config.backbone_condition is backbone
        and rec.config.data_amount is amount
        and rec.config.layer_subset is LayerSubset.ALL
        and rec.config.fusion_method is method
    ]


def _mnli_records(
    records: Sequence[ResultRecord],
    method: FusionMethod,
    backbone: BackboneCondition = BackboneCondition.FROZEN,
    amount: DataAmount = DataAmount.FULL,
) -> list[ResultRecord]:
    return _records_for(records, DatasetName.MNLI, method, backbone, amount)


def _target_method(rules: DecisionRules, field: str) -> FusionMethod:
    """The single fusion method named by a target rule (``h1_...`` / ``h5a_...``); TBD refused."""
    target = getattr(rules, field)
    if target is None:
        raise ValueError(f"{field} is TBD (fixed at the 11/30 lock, T-07a)")
    if not isinstance(target, str):
        raise ValueError(f"unsupported {field} shape: {target!r} (one fusion method name expected)")
    return FusionMethod(target)


def _h1_target_method(rules: DecisionRules) -> FusionMethod:
    """The representative fusion method of the H1 rule (reused by H2 and H4')."""
    return _target_method(rules, "h1_target_fusion_rule")


def _evaluate_h1(
    records: Sequence[ResultRecord],
    rules: DecisionRules,
    kind: str,
) -> HypothesisJudgment | None:
    """H1: with a frozen backbone, the target fusion beats B0 on MNLI (full data)."""
    baseline = _mnli_records(records, FusionMethod.B0)
    if not baseline:
        return None
    target = _h1_target_method(rules)
    candidate = _mnli_records(records, target)
    if not candidate:
        return None
    return _judge_comparison("H1", kind, baseline, candidate, rules)


def _evaluate_h2(
    records: Sequence[ResultRecord],
    rules: DecisionRules,
    kind: str,
) -> HypothesisJudgment | None:
    """H2: the target fusion's gain over B0 is larger with small data than with full data."""
    if not _mnli_records(records, FusionMethod.B0, amount=DataAmount.SMALL):
        return None
    target = _h1_target_method(rules)
    groups = {
        (amount, method): _mnli_records(records, method, amount=amount)
        for amount in (DataAmount.SMALL, DataAmount.FULL)
        for method in (FusionMethod.B0, target)
    }
    if not all(groups.values()):
        return None
    return _judge_effect_difference(
        "H2",
        kind,
        _Effect(groups[DataAmount.SMALL, FusionMethod.B0], groups[DataAmount.SMALL, target]),
        _Effect(groups[DataAmount.FULL, FusionMethod.B0], groups[DataAmount.FULL, target]),
        rules,
        ("small", "full"),
    )


def _evaluate_h3(
    records: Sequence[ResultRecord],
    rules: DecisionRules,
    kind: str,
) -> HypothesisJudgment | None:
    """H3: the fusion gain over B0 is smaller once the backbone is fine-tuned (MNLI, full data).

    The fine-tune condition runs B0 and the best fusion method only (SPEC), so the candidate is
    the one non-B0 method found in those records; its frozen counterpart is the same method on
    the frozen backbone. ``rules`` is only needed for the significance settings.
    """
    finetune_b0 = _mnli_records(records, FusionMethod.B0, BackboneCondition.FINETUNE)
    candidates = {
        rec.config.fusion_method
        for rec in records
        if rec.config.dataset is DatasetName.MNLI
        and rec.config.backbone_condition is BackboneCondition.FINETUNE
        and rec.config.data_amount is DataAmount.FULL
        and rec.config.layer_subset is LayerSubset.ALL
        and rec.config.fusion_method is not FusionMethod.B0
    }
    if not finetune_b0 or not candidates:
        return None
    if len(candidates) > 1:
        raise ValueError(
            "H3: the fine-tuned records must hold one fusion method besides B0, "
            f"got {sorted(m.value for m in candidates)}"
        )
    (target,) = candidates
    groups = {
        (backbone, method): _mnli_records(records, method, backbone)
        for backbone in (BackboneCondition.FROZEN, BackboneCondition.FINETUNE)
        for method in (FusionMethod.B0, target)
    }
    if not all(groups.values()):
        return None
    frozen, finetune = BackboneCondition.FROZEN, BackboneCondition.FINETUNE
    return _judge_effect_difference(
        "H3",
        kind,
        _Effect(groups[frozen, FusionMethod.B0], groups[frozen, target]),
        _Effect(groups[finetune, FusionMethod.B0], groups[finetune, target]),
        rules,
        ("frozen", "finetune"),
    )


def _has_genre_split(rec: ResultRecord) -> bool:
    return rec.eval_error_rate_mismatched is not None and any(
        p.genre_split == "mismatched" for p in rec.predictions
    )


def _mismatched_view(rec: ResultRecord) -> ResultRecord:
    """The record restricted to MNLI's mismatched genres (error rate and predictions)."""
    if not _has_genre_split(rec) or rec.eval_error_rate_mismatched is None:
        raise ValueError(
            f"H4_prime: run {rec.run_id!r} has no mismatched error rate / predictions, "
            "but other runs of the comparison do"
        )
    mismatched = [p for p in rec.predictions if p.genre_split == "mismatched"]
    return replace(rec, predictions=mismatched, eval_error_rate=rec.eval_error_rate_mismatched)


def _evaluate_h4_prime(
    records: Sequence[ResultRecord],
    rules: DecisionRules,
    kind: str,
) -> HypothesisJudgment | None:
    """H4': the H1 comparison still holds on the mismatched (unseen-genre) evaluation set.

    Runs without any genre split are not judged here (no row), but a comparison where only some
    runs carry the split is a pipeline error, not something to average over.
    """
    baseline = _mnli_records(records, FusionMethod.B0)
    if not baseline:
        return None
    candidate = _mnli_records(records, _h1_target_method(rules))
    if not candidate:
        return None
    if not any(_has_genre_split(rec) for rec in (*baseline, *candidate)):
        return None
    return _judge_comparison(
        "H4_prime",
        kind,
        [_mismatched_view(rec) for rec in baseline],
        [_mismatched_view(rec) for rec in candidate],
        rules,
    )


def _imdb_records(
    records: Sequence[ResultRecord],
    method: FusionMethod,
    setting: InputSetting,
    layer_subset: LayerSubset = LayerSubset.ALL,
) -> list[ResultRecord]:
    return [
        rec
        for rec in records
        if rec.config.dataset is DatasetName.IMDB
        and rec.config.backbone_condition is BackboneCondition.FROZEN
        and rec.config.data_amount is DataAmount.FULL
        and rec.config.layer_subset is layer_subset
        and rec.config.fusion_method is method
        and rec.config.input_setting is setting
    ]


def _has_both_input_settings(records: Sequence[ResultRecord]) -> bool:
    """IMDB frozen/full records exist for both L512 and Lfull (something to compare)."""
    settings = {
        rec.config.input_setting
        for rec in records
        if rec.config.dataset is DatasetName.IMDB
        and rec.config.backbone_condition is BackboneCondition.FROZEN
        and rec.config.data_amount is DataAmount.FULL
    }
    return {InputSetting.L512, InputSetting.LFULL} <= settings


def _evaluate_h5a(
    records: Sequence[ResultRecord],
    rules: DecisionRules,
    kind: str,
) -> HypothesisJudgment | None:
    """H5a: with the target fusion on IMDB, Lfull (no cut) beats L512 (head+tail cut).

    Judged on the whole evaluation set. The input setting is the one field allowed to differ
    between the two sides; the per-length-bucket breakdown is reported separately.
    """
    if not _has_both_input_settings(records):
        return None
    method = _target_method(rules, "h5a_target_fusion_rule")
    baseline = _imdb_records(records, method, InputSetting.L512)
    candidate = _imdb_records(records, method, InputSetting.LFULL)
    if not baseline or not candidate:
        return None
    return _judge_comparison(
        "H5a",
        kind,
        baseline,
        candidate,
        rules,
        _LENGTH_COMPARISON_FREE_FIELDS,
    )


_LONG_INPUT_SETTINGS = (InputSetting.L512, InputSetting.LFULL)


def _mean_error(records: Sequence[ResultRecord]) -> float:
    return statistics.fmean(rec.eval_error_rate for rec in records)


def _h5b_decomposition(
    hypothesis_id: str,
    arms: dict[FusionMethod, dict[InputSetting, list[ResultRecord]]],
) -> dict[str, Any]:
    """Split the F2-vs-B0 story into the mixing effect and the weight learning effect.

    Both are improvements (positive = lower error). ``by_input_setting`` compares the error
    rates inside one input setting: mixing = B0 - F1, weight learning = F1 - F2. The
    ``long_input_gain`` entry splits the H5b statistic itself: each method's gain is its L512
    error minus its Lfull error, ``mixing_gain_difference`` = gain(F1) - gain(B0) and
    ``weight_learning_gain_difference`` = gain(F2) - gain(F1), which add up to the H5b
    ``mean_diff`` = gain(F2) - gain(B0). They are differences of gains, not the GLOSSARY effects
    (those are the ``by_input_setting`` values).
    """
    b0, f1, f2 = arms[FusionMethod.B0], arms[FusionMethod.F1], arms[FusionMethod.F2]
    for setting in _LONG_INPUT_SETTINGS:
        _require_one_cell(hypothesis_id, [*b0[setting], *f1[setting], *f2[setting]])
    error = {
        (method, setting): _mean_error(arm[setting])
        for method, arm in arms.items()
        for setting in _LONG_INPUT_SETTINGS
    }
    gain = {
        method: error[method, InputSetting.L512] - error[method, InputSetting.LFULL]
        for method in arms
    }
    return {
        "by_input_setting": {
            setting.value: {
                "mixing_effect": error[FusionMethod.B0, setting] - error[FusionMethod.F1, setting],
                "weight_learning_effect": (
                    error[FusionMethod.F1, setting] - error[FusionMethod.F2, setting]
                ),
            }
            for setting in _LONG_INPUT_SETTINGS
        },
        "long_input_gain": {
            "mixing_gain_difference": gain[FusionMethod.F1] - gain[FusionMethod.B0],
            "weight_learning_gain_difference": gain[FusionMethod.F2] - gain[FusionMethod.F1],
        },
    }


def _decomposition_obstacle(
    arms: dict[FusionMethod, dict[InputSetting, list[ResultRecord]]],
) -> str | None:
    """Why the auxiliary F1 split cannot be made (F1 missing or on other seeds), else ``None``.

    F1 only feeds the decomposition, so a gap costs the split, not the F2 verdict.
    """
    if not all(arms[FusionMethod.F1].values()):
        return "no F1 runs at both input settings"
    seed_sets = {
        frozenset(rec.config.seed for rec in arm[setting])
        for arm in arms.values()
        for setting in _LONG_INPUT_SETTINGS
    }
    if len(seed_sets) != 1:
        return "B0, F1 and F2 runs must cover the same seeds"
    return None


def _evaluate_h5b(
    records: Sequence[ResultRecord],
    rules: DecisionRules,
    kind: str,
) -> HypothesisJudgment | None:
    """H5b: the longer input (Lfull vs L512) helps F2 more than it helps B0 (IMDB).

    F2 decides; F1 is auxiliary and only feeds ``details["decomposition"]`` (``None`` when F1
    runs are missing). Judged on the whole evaluation set.
    """
    methods = (FusionMethod.B0, FusionMethod.F1, FusionMethod.F2)
    arms = {
        method: {
            setting: _imdb_records(records, method, setting) for setting in _LONG_INPUT_SETTINGS
        }
        for method in methods
    }
    b0, f2 = arms[FusionMethod.B0], arms[FusionMethod.F2]
    if not all(b0.values()) or not all(f2.values()):
        return None
    for setting in _LONG_INPUT_SETTINGS:
        _require_one_cell("H5b", [*b0[setting], *f2[setting]])
    judgment = _judge_effect_difference(
        "H5b",
        kind,
        _Effect(f2[InputSetting.L512], f2[InputSetting.LFULL], _LENGTH_COMPARISON_FREE_FIELDS),
        _Effect(b0[InputSetting.L512], b0[InputSetting.LFULL], _LENGTH_COMPARISON_FREE_FIELDS),
        rules,
        ("F2", "B0"),
    )
    skipped = _decomposition_obstacle(arms)
    judgment.details["decomposition"] = (
        _h5b_decomposition("H5b", arms) if skipped is None else None
    )
    if skipped is not None:
        judgment.details["decomposition_skipped"] = skipped
    return judgment


_CONFIRMATORY_EVALUATORS: dict[
    str, Callable[[Sequence[ResultRecord], DecisionRules, str], HypothesisJudgment | None]
] = {
    "H1": _evaluate_h1,
    "H2": _evaluate_h2,
    "H3": _evaluate_h3,
    "H4_prime": _evaluate_h4_prime,
    "H5b": _evaluate_h5b,
    "H5a": _evaluate_h5a,
}
