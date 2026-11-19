"""Build layer-vector caches from configs/week1_env.yaml (T-04b, scripts/extract_cache.py).

The fixed environment (model, transformers version, attention, compile, Lfull cap, pooling,
cache dtype) is read from ``week1_env.yaml`` only, so every cache records the locked setting.
A value that is still ``null`` there stops the build instead of being guessed (AGENTS.md §2).
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from layer_fusion.contracts import CacheMetadata, DatasetName, DocumentRecord, InputSetting
from layer_fusion.extraction import (
    abort_split,
    begin_split,
    extract_layer_vectors,
    finish_split,
    make_cache_metadata,
)

CACHE_DTYPES = ("float32", "float16")
MAIN_CACHE_ROOT = Path("data/cache")         # full caches used by main runs (T-08b)
DEBUG_CACHE_ROOT = Path("data/cache_debug")  # --limit / --shortest samples, git-ignored


@dataclass(frozen=True)
class ExtractionEnv:
    """The locked extraction setting from configs/week1_env.yaml."""

    model_name: str
    transformers_version: str
    attn_implementation: str
    torch_compile: bool
    lfull_max_length_cap: int | None
    include_special_tokens_in_pool: bool | None
    cache_dtype: str | None

    def cache_metadata(self, input_setting: InputSetting | None, dtype: str) -> CacheMetadata:
        """CacheMetadata of a cache built under this setting (pooling must be decided)."""
        if self.include_special_tokens_in_pool is None:
            raise ValueError(
                "include_special_tokens_in_pool is null in week1_env.yaml (not decided yet)"
            )
        return make_cache_metadata(
            self.model_name,
            self.transformers_version,
            self.attn_implementation,
            self.torch_compile,
            input_setting,
            self.lfull_max_length_cap,
            self.include_special_tokens_in_pool,
            dtype,
        )


def load_env_config(path: str | Path) -> ExtractionEnv:
    """Read the extraction fields of week1_env.yaml (``null`` stays ``None``)."""
    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    lock = data["runtime_lock"]
    return ExtractionEnv(
        model_name=data["backbone"]["model_name"],
        transformers_version=lock["transformers_version"],
        attn_implementation=lock["attn_implementation"],
        torch_compile=lock["torch_compile"],
        lfull_max_length_cap=data["imdb_length_check"]["lfull_max_length_cap"],
        include_special_tokens_in_pool=data["training_budget"]["include_special_tokens_in_pool"],
        cache_dtype=lock["cache_dtype"],
    )


def resolve_cache_dtype(env: ExtractionEnv, override: str | None, debug: bool = False) -> str:
    """The dtype to store; never a guessed default.

    A main cache always uses the locked ``cache_dtype`` (``null`` stops the build, and a
    different ``--dtype`` is refused). Only a debug sample (``debug=True``) may override it.
    """
    if not debug and override is not None and override != env.cache_dtype:
        raise ValueError(
            f"--dtype {override} differs from the locked cache_dtype {env.cache_dtype!r} in "
            "week1_env.yaml; main caches use the locked value (override only with --limit)"
        )
    dtype = override or env.cache_dtype
    if dtype is None:
        raise ValueError(
            "cache_dtype is null in week1_env.yaml (not decided yet); "
            "a debug run (--limit) may pass --dtype explicitly"
        )
    if dtype not in CACHE_DTYPES:
        raise ValueError(f"cache dtype must be one of {CACHE_DTYPES}, got {dtype!r}")
    return dtype


def cache_name(dataset: DatasetName, input_setting: InputSetting | None, sep: str = "_") -> str:
    """``mnli`` / ``imdb_L512``: the cache folder name (``sep="/"`` gives the log label)."""
    return dataset.value + (f"{sep}{input_setting.value}" if input_setting else "")


def default_output_dir(
    dataset: DatasetName, input_setting: InputSetting | None, sample_tag: str | None
) -> Path:
    """``data/cache/<name>`` for a full cache, ``data/cache_debug/<name>_<tag>`` for a sample."""
    name = cache_name(dataset, input_setting)
    if sample_tag is not None:
        return DEBUG_CACHE_ROOT / f"{name}_{sample_tag}"
    return MAIN_CACHE_ROOT / name


def check_output_dir(output_dir: Path, sampled: bool) -> None:
    """A sampled (debug) cache may only be written under data/cache_debug/ (AGENTS.md §4).

    CacheMetadata does not record sampling, so a subset written into data/cache/ could later be
    read as a full main cache; the folder is the only thing that keeps them apart.
    """
    if not sampled:
        return
    debug_root = DEBUG_CACHE_ROOT.resolve()
    target = output_dir.resolve()
    if target != debug_root and debug_root not in target.parents:
        raise ValueError(
            f"a sampled cache (--limit/--shortest) must be written under {DEBUG_CACHE_ROOT}/, "
            f"not {output_dir}; main caches hold every document"
        )


def check_input_setting(dataset: DatasetName, input_setting: InputSetting | None) -> None:
    """IMDB needs L128/L512/Lfull (ADR 0006); MNLI/SST-2/AG News take no input setting."""
    if dataset is DatasetName.IMDB and input_setting is None:
        raise ValueError("IMDB caches need --input-setting L128, L512 or Lfull (ADR 0006)")
    if dataset is not DatasetName.IMDB and input_setting is not None:
        raise ValueError(f"--input-setting applies to IMDB only, not {dataset.value}")


def sample_records(
    records: Sequence[DocumentRecord], limit: int | None, seed: int = 0
) -> list[DocumentRecord]:
    """A fixed random subset of ``limit`` records in their original order (debug / measurement).

    ``None`` or a limit covering every record returns all records unchanged.
    """
    if limit is None or limit >= len(records):
        return list(records)
    if limit < 1:
        raise ValueError(f"limit must be positive, got {limit}")
    rng = random.Random(f"extract-limit:{seed}")
    keep = sorted(rng.sample(range(len(records)), limit))
    return [records[i] for i in keep]


def select_shortest(
    records: Sequence[DocumentRecord], input_lengths: Sequence[int], count: int
) -> list[DocumentRecord]:
    """The ``count`` records with the shortest model inputs, in their original order.

    Used for the float16 worst case: a short input averages few tokens, so one outlier
    dimension dominates its mean (docs/measurements/float16_range_check.json). Ties keep the
    earlier record.
    """
    if len(input_lengths) != len(records):
        raise ValueError(f"{len(input_lengths)} lengths for {len(records)} records")
    if count < 1:
        raise ValueError(f"count must be positive, got {count}")
    keep = sorted(sorted(range(len(records)), key=lambda i: input_lengths[i])[:count])
    return [records[i] for i in keep]


def build_cache(
    records_by_split: Mapping[str, Sequence[DocumentRecord]],
    env: ExtractionEnv,
    output_dir: str | Path | None,
    input_setting: InputSetting | None = None,
    dtype: str = "float32",
    device: str = "cpu",
    batch_size: int = 32,
    model: Any = None,
    tokenizer: Any = None,
) -> dict[str, dict[str, Any]]:
    """Extract every split and return each split's extraction report.

    With ``output_dir`` each split is written straight to disk through ``begin_split`` →
    ``extract_layer_vectors(out=memmap)`` → ``finish_split``; on any error the split's
    ``.partial`` folder is removed and the error re-raised. With ``output_dir=None`` nothing is
    saved (report-only measurement, e.g. the float16 range check).
    """
    if model is None or tokenizer is None:
        from transformers import AutoModel, AutoTokenizer

        tokenizer = tokenizer or AutoTokenizer.from_pretrained(env.model_name)
        model = model or AutoModel.from_pretrained(
            env.model_name, attn_implementation=env.attn_implementation
        )
    common = {
        "model_name": env.model_name,
        "transformers_version": env.transformers_version,
        "attn_implementation": env.attn_implementation,
        "torch_compile": env.torch_compile,
        "input_setting": input_setting,
        "lfull_max_length_cap": env.lfull_max_length_cap,
        "include_special_tokens_in_pool": env.include_special_tokens_in_pool,
    }
    reports: dict[str, dict[str, Any]] = {}
    for split_name, records in records_by_split.items():
        records = list(records)
        kwargs = dict(
            common, dtype=dtype, device=device, batch_size=batch_size,
            model=model, tokenizer=tokenizer,
        )
        if output_dir is None:
            result = extract_layer_vectors(records, **kwargs)
        else:
            metadata = env.cache_metadata(input_setting, dtype)
            out = begin_split(output_dir, split_name, len(records), metadata)
            try:
                result = extract_layer_vectors(records, out=out, **kwargs)
                finish_split(output_dir, split_name, result)
            except BaseException:
                abort_split(output_dir, split_name, out)  # closes the memmap first (Windows)
                raise
        reports[split_name] = result.report.to_dict()
    return reports
