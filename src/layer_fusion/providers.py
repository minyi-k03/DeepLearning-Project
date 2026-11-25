"""Layer vector provider interface and cache metadata validation (SPEC.md §모듈 구성)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from layer_fusion.contracts import CacheMetadata, DocumentRecord, InputSetting, RunConfig, RunKind
from layer_fusion.inputs import truncation_rule_name
from layer_fusion.layers import HIDDEN_SIZE, NUM_LAYER_VECTORS

SYNTHETIC_SPLITS = ("train", "eval")

# On-disk cache layout (T-04b): <cache_dir>/metadata.json + <cache_dir>/<split>/{files below}
CACHE_METADATA_FILE = "metadata.json"
VECTORS_FILE = "vectors.npy"
DOC_IDS_FILE = "doc_ids.json"
LABELS_FILE = "labels.npy"
TOKEN_LENGTHS_FILE = "token_lengths.npy"
GENRE_SPLITS_FILE = "genre_splits.json"
VIEW_RATIOS_FILE = "view_ratios.npy"   # local-layer view ratio per document (SPEC US 18)
REPORT_FILE = "report.json"            # extraction audit (max |value|, float16 error, time)
PARTIAL_SUFFIX = ".partial"  # a split being written; never read as a cache


def check_split_lengths(
    n_rows: int,
    doc_ids: Sequence[Any],
    labels: Sequence[Any],
    where: str = "",
    **optional: Sequence[Any] | None,
) -> None:
    """Raise unless doc_ids, labels and every given per-document list have ``n_rows`` entries.

    Shared by the cache writer and :class:`CacheProvider`, so a split is checked by one rule
    when it is written and when it is read. ``None`` optional lists are skipped.
    """
    prefix = f"{where}: " if where else ""
    if len(doc_ids) != n_rows or len(labels) != n_rows:
        raise ValueError(
            f"{prefix}doc_ids ({len(doc_ids)}) and labels ({len(labels)}) must match "
            f"{n_rows} rows"
        )
    for name, values in optional.items():
        if values is not None and len(values) != n_rows:
            raise ValueError(f"{prefix}{name} has {len(values)} entries, expected {n_rows}")


class CacheConfigMismatchError(ValueError):
    """Raised when a provider's creation metadata does not match the RunConfig."""


@dataclass
class LayerVectorBatch:
    """Container returned by a LayerVectorProvider for a split."""

    doc_ids: list[str]
    labels: np.ndarray            # shape [N]
    vectors: np.ndarray           # shape [N, 23, 768]
    metadata: CacheMetadata
    genre_splits: list[str | None] | None = None
    token_lengths: list[int] | None = None
    view_ratios: list[float] | None = None


class LayerVectorProvider(Protocol):
    """Interface injecting layer vectors into the Execution seam."""

    @property
    def metadata(self) -> CacheMetadata:
        ...

    def validate_for_run(self, config: RunConfig) -> None:
        ...

    def get_split(self, split_name: str) -> LayerVectorBatch:
        ...


def cache_metadata_to_dict(metadata: CacheMetadata) -> dict[str, Any]:
    """JSON-ready dict of CacheMetadata (enum values as plain strings)."""
    data = asdict(metadata)
    if metadata.input_setting is not None:
        data["input_setting"] = metadata.input_setting.value
    return data


def cache_metadata_from_dict(data: dict[str, Any]) -> CacheMetadata:
    """Inverse of :func:`cache_metadata_to_dict`."""
    data = dict(data)
    if data.get("input_setting") is not None:
        data["input_setting"] = InputSetting(data["input_setting"])
    return CacheMetadata(**data)


def validate_metadata_match(actual: CacheMetadata, expected: CacheMetadata) -> None:
    """Verify actual cache metadata matches expected run metadata; raise on mismatch."""
    if actual != expected:
        raise CacheConfigMismatchError(
            f"Cache metadata mismatch.\nExpected: {expected}\nActual:   {actual}"
        )


def check_provider_for_run(metadata: CacheMetadata, config: RunConfig) -> None:
    """Fail before training when the provider was built under a different setting (SPEC US 26).

    - ``config.input_setting`` must equal ``metadata.input_setting`` (``None`` included, so an
      MNLI/SST-2 run cannot take an IMDB L128/L512/Lfull cache).
    - Main runs must declare ``expected_cache_meta``; only debug runs may skip it, so a
      forgotten expectation cannot let a cache from another setting slip through.
    - With ``expected_cache_meta`` every CacheMetadata field must match (ADR 0005).
    """
    if config.input_setting != metadata.input_setting:
        raise CacheConfigMismatchError(
            f"input_setting mismatch: run={config.input_setting}, "
            f"provider={metadata.input_setting}"
        )
    if config.expected_cache_meta is None:
        if config.run_kind == RunKind.MAIN:
            raise CacheConfigMismatchError(
                "main runs must set RunConfig.expected_cache_meta (only debug runs may omit it)"
            )
        return
    validate_metadata_match(metadata, config.expected_cache_meta)


class CacheProvider:
    """Loads pre-extracted [N, 23, 768] layer vectors from disk (백본 고정 조건, ADR 0004).

    Vectors are memory-mapped (``np.load(..., mmap_mode="r")``) in their stored dtype, so a
    float16 MNLI cache (~14 GB) fits the 24 GB container limit. Consumers upcast each mini-batch
    to float32 before computing. (Cache dtype float16: SPEC §결정 필요 사항 and
    configs/week1_env.yaml ``cache_dtype``.)
    """

    def __init__(self, cache_dir: str) -> None:
        self.cache_dir = Path(cache_dir)
        meta_path = self.cache_dir / CACHE_METADATA_FILE
        if not meta_path.is_file():
            raise FileNotFoundError(f"no layer-vector cache at {self.cache_dir} ({meta_path.name})")
        self._metadata = cache_metadata_from_dict(json.loads(meta_path.read_text("utf-8")))

    @property
    def metadata(self) -> CacheMetadata:
        return self._metadata

    @property
    def splits(self) -> list[str]:
        return sorted(
            p.name
            for p in self.cache_dir.iterdir()
            if not p.name.endswith(PARTIAL_SUFFIX) and (p / VECTORS_FILE).is_file()
        )

    def validate_for_run(self, config: RunConfig) -> None:
        check_provider_for_run(self.metadata, config)

    def get_split(self, split_name: str) -> LayerVectorBatch:
        """Load one split, checking it against the cache metadata before handing it out.

        Only the .npy headers are read for the shape/dtype checks; vectors stay memory-mapped.
        """
        split_dir = self.cache_dir / split_name
        if split_name.endswith(PARTIAL_SUFFIX) or not (split_dir / VECTORS_FILE).is_file():
            raise KeyError(f"split {split_name!r} not in cache {self.cache_dir} ({self.splits})")
        vectors = np.load(split_dir / VECTORS_FILE, mmap_mode="r")
        if vectors.ndim != 3 or vectors.shape[1:] != (NUM_LAYER_VECTORS, HIDDEN_SIZE):
            raise ValueError(
                f"{split_dir}: vectors must be [N, {NUM_LAYER_VECTORS}, {HIDDEN_SIZE}], "
                f"got {vectors.shape}"
            )
        if vectors.dtype != np.dtype(self._metadata.dtype):
            raise ValueError(
                f"{split_dir}: vectors are {vectors.dtype}, metadata says {self._metadata.dtype}"
            )
        doc_ids = json.loads((split_dir / DOC_IDS_FILE).read_text("utf-8"))
        labels = np.load(split_dir / LABELS_FILE)
        token_lengths = None
        if (split_dir / TOKEN_LENGTHS_FILE).is_file():
            token_lengths = np.load(split_dir / TOKEN_LENGTHS_FILE).tolist()
        genre_splits = None
        if (split_dir / GENRE_SPLITS_FILE).is_file():
            genre_splits = json.loads((split_dir / GENRE_SPLITS_FILE).read_text("utf-8"))
        view_ratios = None
        if (split_dir / VIEW_RATIOS_FILE).is_file():
            view_ratios = np.load(split_dir / VIEW_RATIOS_FILE).tolist()
        check_split_lengths(
            vectors.shape[0],
            doc_ids,
            labels,
            where=str(split_dir),
            token_lengths=token_lengths,
            genre_splits=genre_splits,
            view_ratios=view_ratios,
        )
        return LayerVectorBatch(
            doc_ids=doc_ids,
            labels=labels,
            vectors=vectors,
            metadata=self._metadata,
            genre_splits=genre_splits,
            token_lengths=token_lengths,
            view_ratios=view_ratios,
        )


@dataclass
class TokenBatch:
    """A tokenized split for the finetune condition: token ids instead of layer vectors."""

    doc_ids: list[str]
    labels: np.ndarray            # shape [N]
    input_ids: list[list[int]]    # unpadded model inputs (padding happens per batch)
    metadata: CacheMetadata
    genre_splits: list[str | None] | None = None
    token_lengths: list[int] | None = None
    view_ratios: list[float] | None = None


class BackboneProvider:
    """Layer vectors computed live by a trainable ModernBERT (finetune condition, T-06d).

    Holds the tokenized splits and a pristine copy of the pretrained model; every training run
    starts from :meth:`fresh_model`, so LR candidates and seeds never share trained weights.
    ``metadata`` describes the extraction setting exactly like a cache's (``dtype`` is
    ``float32``: vectors are computed, not stored), so ``expected_cache_meta`` checks apply
    unchanged. A ``limited`` provider (debug ``--limit`` sample) refuses main runs, because
    unlike a cache folder nothing in its metadata marks it as a sample.
    """

    def __init__(
        self,
        splits: dict[str, Sequence[DocumentRecord]],
        metadata: CacheMetadata,
        model: Any,
        tokenizer: Any,
        *,
        limited: bool = False,
    ) -> None:
        from layer_fusion.inputs import build_tokenized_inputs

        if metadata.dtype != "float32":
            raise ValueError(f"finetune layer vectors are float32, metadata says {metadata.dtype}")
        model_attn = getattr(getattr(model, "config", None), "_attn_implementation", None)
        if model_attn is not None and model_attn != metadata.attn_implementation:
            raise ValueError(
                f"model uses attn_implementation={model_attn!r}, metadata records "
                f"{metadata.attn_implementation!r} (ADR 0005)"
            )
        self._metadata = metadata
        self._model = model.cpu()
        self.limited = limited
        self.pad_id = int(tokenizer.pad_token_id)
        self.special_ids = (int(tokenizer.cls_token_id), int(tokenizer.sep_token_id))
        input_setting = metadata.input_setting
        lfull_cap = metadata.max_length_cap if input_setting is InputSetting.LFULL else None
        self._batches: dict[str, TokenBatch] = {}
        for name, records in splits.items():
            records = list(records)
            tokenized = build_tokenized_inputs(records, tokenizer, input_setting, lfull_cap)
            labels = np.asarray([r.label for r in records], dtype=np.int64)
            genres = [r.genre_split for r in records]
            genre_splits = genres if any(g is not None for g in genres) else None
            check_split_lengths(
                len(records),
                tokenized.doc_ids,
                labels,
                where=f"backbone split {name!r}",
                token_lengths=tokenized.token_lengths,
                genre_splits=genre_splits,
                view_ratios=tokenized.view_ratios,
            )
            self._batches[name] = TokenBatch(
                doc_ids=tokenized.doc_ids,
                labels=labels,
                input_ids=tokenized.input_ids,
                metadata=metadata,
                genre_splits=genre_splits,
                token_lengths=tokenized.token_lengths,
                view_ratios=tokenized.view_ratios,
            )

    @classmethod
    def from_env(
        cls,
        dataset: Any,
        env: Any,
        *,
        limit: int | None = None,
        data_dir: str | None = None,
    ) -> BackboneProvider:
        """Load ``dataset`` and the locked ModernBERT setting of ``env`` (``ExtractionEnv``).

        ``limit`` keeps a fixed random sample of that many documents per split (debug only).
        """
        import transformers
        from transformers import AutoModel, AutoTokenizer

        from layer_fusion.cache_build import sample_records
        from layer_fusion.data import load_dataset_splits

        if transformers.__version__ != env.transformers_version:
            raise ValueError(
                f"transformers {transformers.__version__} is installed, but runs must use the "
                f"locked {env.transformers_version} (ADR 0005)"
            )
        if env.torch_compile:
            raise ValueError("torch_compile=true is not supported by the finetune path yet")
        splits = load_dataset_splits(dataset, data_dir)
        if limit is not None:
            splits = {name: sample_records(records, limit) for name, records in splits.items()}
        tokenizer = AutoTokenizer.from_pretrained(env.model_name)
        model = AutoModel.from_pretrained(
            env.model_name, attn_implementation=env.attn_implementation
        )
        return cls(
            splits,
            env.cache_metadata(None, "float32"),
            model,
            tokenizer,
            limited=limit is not None,
        )

    @property
    def metadata(self) -> CacheMetadata:
        return self._metadata

    def validate_for_run(self, config: RunConfig) -> None:
        from layer_fusion.contracts import BackboneCondition

        if config.backbone_condition is not BackboneCondition.FINETUNE:
            raise CacheConfigMismatchError(
                "BackboneProvider is for the finetune condition; frozen runs read a cache"
            )
        if self.limited and config.run_kind is RunKind.MAIN:
            raise CacheConfigMismatchError(
                "a --limit sample provider is for debug runs only (AGENTS.md §4)"
            )
        check_provider_for_run(self._metadata, config)

    def get_split(self, split_name: str) -> TokenBatch:
        if split_name not in self._batches:
            raise KeyError(f"Unknown split {split_name!r}")
        return self._batches[split_name]

    def fresh_model(self) -> Any:
        """A trainable copy of the pretrained backbone (on CPU)."""
        import copy

        model = copy.deepcopy(self._model)
        model.train()
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        return model


class FakeLayerVectorProvider:
    """Deterministic in-memory provider for CPU-only Seam 1 tests without GPU/data."""

    def __init__(
        self,
        splits: dict[str, LayerVectorBatch],
        metadata: CacheMetadata,
    ) -> None:
        self._splits = splits
        self._metadata = metadata
        for batch in splits.values():
            if batch.vectors.ndim != 3 or batch.vectors.shape[1:] != (
                NUM_LAYER_VECTORS,
                HIDDEN_SIZE,
            ):
                raise ValueError(
                    f"Expected shape [N, {NUM_LAYER_VECTORS}, {HIDDEN_SIZE}], "
                    f"got {batch.vectors.shape}"
                )

    @property
    def metadata(self) -> CacheMetadata:
        return self._metadata

    def validate_for_run(self, config: RunConfig) -> None:
        check_provider_for_run(self._metadata, config)

    def get_split(self, split_name: str) -> LayerVectorBatch:
        if split_name not in self._splits:
            raise KeyError(f"Unknown split {split_name!r}")
        return self._splits[split_name]

    @classmethod
    def make_synthetic(
        cls,
        num_classes: int = 3,
        train_size: int = 60,
        eval_size: int = 30,
        seed: int = 0,
        metadata: CacheMetadata | None = None,
        doc_records: Sequence[DocumentRecord] | None = None,
    ) -> FakeLayerVectorProvider:
        """Create a small deterministic provider for CPU seam tests (splits ``train``/``eval``).

        Vectors are [N, 23, 768] float32 noise plus a small class-dependent mean shift, so
        heads trained on them have a learnable signal. Labels are class-balanced. With
        ``num_classes == 3`` (MNLI-like) the eval split is half matched / half mismatched.
        ``doc_records`` overrides doc_ids, labels, splits and genres (e.g. from data.py).
        """
        rng = np.random.default_rng(seed)
        class_means = rng.normal(0.0, 1.0, size=(num_classes, NUM_LAYER_VECTORS, HIDDEN_SIZE))
        if metadata is None:
            metadata = CacheMetadata(
                model_name="answerdotai/ModernBERT-base",
                transformers_version="5.23.0",
                attn_implementation="sdpa",
                torch_compile=False,
                input_setting=None,
                max_length_cap=None,
                truncation_rule=truncation_rule_name(None),
                include_special_tokens_in_pool=None,
                dtype="float32",
            )

        plan: dict[str, tuple[list[str], list[int], list[str | None] | None]] = {}
        if doc_records is not None:
            unknown = sorted({r.split for r in doc_records} - set(SYNTHETIC_SPLITS))
            if unknown:
                # e.g. "tuning": select tuning docs by doc_id from the train split instead
                # (decided on PR #11) rather than silently dropping them here.
                raise ValueError(f"make_synthetic only builds {SYNTHETIC_SPLITS}, got {unknown}")
            for name in SYNTHETIC_SPLITS:
                rows = [r for r in doc_records if r.split == name]
                genres = [r.genre_split for r in rows]
                plan[name] = (
                    [r.doc_id for r in rows],
                    [r.label for r in rows],
                    genres if any(g is not None for g in genres) else None,
                )
        else:
            for name, size in zip(SYNTHETIC_SPLITS, (train_size, eval_size), strict=True):
                labels = [i % num_classes for i in range(size)]
                rng.shuffle(labels)
                genres: list[str | None] | None = None
                if name == "eval" and num_classes == 3:
                    half = size // 2
                    genres = ["matched"] * half + ["mismatched"] * (size - half)
                plan[name] = ([f"{name}-{i:05d}" for i in range(size)], labels, genres)

        splits: dict[str, LayerVectorBatch] = {}
        for name, (doc_ids, labels, genres) in plan.items():
            label_arr = np.asarray(labels, dtype=np.int64)
            noise = rng.normal(0.0, 1.0, size=(len(doc_ids), NUM_LAYER_VECTORS, HIDDEN_SIZE))
            vectors = (noise + 0.5 * class_means[label_arr]).astype(np.float32)
            splits[name] = LayerVectorBatch(
                doc_ids=doc_ids,
                labels=label_arr,
                vectors=vectors,
                metadata=metadata,
                genre_splits=genres,
            )
        return cls(splits, metadata)
