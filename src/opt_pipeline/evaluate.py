"""SST-2 validation evaluation: accuracy, F1 and label agreement for any runner.

The GLUE SST-2 *validation* split (872 labelled sentences) is used because the
test split is unlabelled. Predictions use dynamic padding (``longest`` per
batch), i.e. realistic serving inputs, so the accuracy numbers reflect what a
deployment would see rather than the fixed-shape benchmark batches.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from transformers import PreTrainedTokenizerBase

from opt_pipeline.baseline import encode
from opt_pipeline.config import PipelineConfig
from opt_pipeline.errors import ConfigError
from opt_pipeline.runners import Runner
from opt_pipeline.utils import get_logger

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class LabeledSplit:
    """Sentences with integer gold labels and the dataset's label names."""

    sentences: list[str]
    labels: npt.NDArray[np.int64]
    label_names: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.sentences)

    def head(self, n: int) -> LabeledSplit:
        """First ``n`` examples (for fast tests)."""
        return LabeledSplit(self.sentences[:n], self.labels[:n], self.label_names)


def load_sst2(cfg: PipelineConfig, *, max_samples: int | None = None) -> LabeledSplit:
    """Load the configured GLUE SST-2 split into the project-local data cache.

    Args:
        cfg: Pipeline configuration (dataset id/config/split, cache dir).
        max_samples: Optional cap; defaults to ``cfg.eval_max_samples``.

    Raises:
        ConfigError: If the split is unlabelled (GLUE test splits use ``-1``).
    """
    from datasets import load_dataset  # heavy import; only needed here

    # ``cache_dir`` holds the processed Arrow tables. With datasets 5.x the raw
    # parquet files (~3 MB) are always fetched through the shared Hugging Face
    # hub cache (~/.cache/huggingface/hub), regardless of DownloadConfig.
    ds = load_dataset(
        cfg.dataset_name,
        cfg.dataset_config,
        split=cfg.dataset_split,
        cache_dir=str(cfg.paths.data_cache_dir),
    )
    limit = max_samples or cfg.eval_max_samples
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    labels = np.asarray(ds["label"], dtype=np.int64)
    if (labels < 0).any():
        raise ConfigError(f"split {cfg.dataset_split!r} has no gold labels (found -1)")
    return LabeledSplit(
        sentences=[str(s) for s in ds["sentence"]],
        labels=labels,
        label_names=tuple(str(n) for n in ds.features["label"].names),
    )


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    """Binary classification metrics; the positive class is ``positive``."""

    n: int
    accuracy: float
    f1: float
    precision: float
    recall: float
    tp: int
    fp: int
    tn: int
    fn: int

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable view."""
        return asdict(self)


def classification_metrics(
    y_true: npt.ArrayLike, y_pred: npt.ArrayLike, *, positive: int = 1
) -> ClassificationMetrics:
    """Accuracy plus binary precision / recall / F1 for the ``positive`` label.

    F1 is defined as 0.0 when there are no true or predicted positives.
    """
    t = np.asarray(y_true)
    p = np.asarray(y_pred)
    if t.shape != p.shape or t.ndim != 1:
        raise ValueError(f"y_true {t.shape} and y_pred {p.shape} must be equal-length 1-D")
    if t.size == 0:
        raise ValueError("cannot compute metrics on zero examples")
    tp = int(np.sum((p == positive) & (t == positive)))
    fp = int(np.sum((p == positive) & (t != positive)))
    fn = int(np.sum((p != positive) & (t == positive)))
    tn = int(t.size - tp - fp - fn)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return ClassificationMetrics(
        n=int(t.size),
        accuracy=float(np.mean(t == p)),
        f1=float(f1),
        precision=float(precision),
        recall=float(recall),
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
    )


def label_agreement(a: npt.ArrayLike, b: npt.ArrayLike) -> float:
    """Fraction of positions where two prediction vectors agree."""
    x, y = np.asarray(a), np.asarray(b)
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch {x.shape} vs {y.shape}")
    return float(np.mean(x == y))


def model_to_dataset_labels(
    id2label: Mapping[int, str], dataset_label_names: Sequence[str]
) -> npt.NDArray[np.int64]:
    """Map model output indices to dataset label indices by (case-insensitive) name.

    Guards against silently scoring a model whose label order differs from the
    dataset's (here: model ``NEGATIVE/POSITIVE`` vs dataset ``negative/positive``).

    Raises:
        ConfigError: If a model label has no dataset counterpart.
    """
    lookup = {name.lower(): i for i, name in enumerate(dataset_label_names)}
    mapping = np.empty(len(id2label), dtype=np.int64)
    for idx, name in id2label.items():
        if name.lower() not in lookup:
            raise ConfigError(f"model label {name!r} not in dataset labels {dataset_label_names}")
        mapping[idx] = lookup[name.lower()]
    return mapping


def batched_logits(
    runner: Runner,
    tokenizer: PreTrainedTokenizerBase,
    sentences: Sequence[str],
    *,
    batch_size: int,
    max_length: int,
) -> npt.NDArray[np.float32]:
    """Logits for all ``sentences`` with per-batch dynamic padding (``longest``)."""
    if not sentences:
        raise ValueError("sentences must be non-empty")
    chunks: list[npt.NDArray[np.float32]] = []
    for start in range(0, len(sentences), batch_size):
        batch = encode(
            tokenizer,
            sentences[start : start + batch_size],
            max_length=max_length,
            padding="longest",
        )
        chunks.append(runner.predict_logits(batch))
    return np.concatenate(chunks).astype(np.float32, copy=False)


def predict_labels(
    runner: Runner,
    tokenizer: PreTrainedTokenizerBase,
    sentences: Sequence[str],
    *,
    batch_size: int,
    max_length: int,
) -> npt.NDArray[np.int64]:
    """Argmax predictions in the model's label space."""
    logits = batched_logits(
        runner, tokenizer, sentences, batch_size=batch_size, max_length=max_length
    )
    return logits.argmax(axis=-1).astype(np.int64)


def evaluate_variants(
    cfg: PipelineConfig, variant_names: list[str] | None = None
) -> dict[str, Any]:
    """Accuracy / F1 / agreement for several variants, in-process, sequentially.

    The PyTorch FP32 CPU reference is always evaluated (first) so every other
    variant can report label agreement and accuracy delta against it.
    """
    import gc
    from datetime import UTC, datetime

    import torch

    from opt_pipeline.baseline import load_id2label, load_tokenizer
    from opt_pipeline.runners import REFERENCE_VARIANT, build_runner, check_artifacts, get_variants

    names = variant_names or None
    specs = get_variants(names)
    if REFERENCE_VARIANT not in {s.name for s in specs}:
        specs = get_variants([REFERENCE_VARIANT]) + specs
    check_artifacts(specs, cfg)
    split = load_sst2(cfg)
    tokenizer = load_tokenizer(cfg)
    id2label = load_id2label(cfg)

    rows: list[dict[str, Any]] = []
    ref_preds: npt.NDArray[np.int64] | None = None
    ref_acc: float | None = None
    for spec in specs:
        if spec.device == "cuda" and not torch.cuda.is_available():
            rows.append({**spec.to_dict(), "status": "skipped", "error": "CUDA not available"})
            continue
        runner = build_runner(spec, cfg)
        metrics, preds = evaluate_runner(runner, tokenizer, id2label, split, cfg)
        if spec.name == REFERENCE_VARIANT:
            ref_preds, ref_acc = preds, metrics.accuracy
        row = {**spec.to_dict(), "status": "ok", **metrics.to_dict()}
        if ref_preds is not None and ref_acc is not None:
            row["label_agreement_vs_reference"] = label_agreement(preds, ref_preds)
            row["accuracy_drop_vs_reference"] = ref_acc - metrics.accuracy
        rows.append(row)
        log.info("[%s] accuracy=%.4f f1=%.4f", spec.name, metrics.accuracy, metrics.f1)
        del runner
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset": f"{cfg.dataset_name}/{cfg.dataset_config}:{cfg.dataset_split}",
        "n": len(split),
        "config": cfg.to_dict(),
        "variants": rows,
    }


def evaluate_runner(
    runner: Runner,
    tokenizer: PreTrainedTokenizerBase,
    id2label: Mapping[int, str],
    split: LabeledSplit,
    cfg: PipelineConfig,
) -> tuple[ClassificationMetrics, npt.NDArray[np.int64]]:
    """Score ``runner`` on ``split``.

    Returns:
        ``(metrics, predictions)`` where predictions are in *dataset* label
        space, ready for :func:`label_agreement` across variants.
    """
    mapping = model_to_dataset_labels(id2label, split.label_names)
    raw = predict_labels(
        runner,
        tokenizer,
        split.sentences,
        batch_size=cfg.eval_batch_size,
        max_length=cfg.max_seq_length,
    )
    preds = mapping[raw]
    positive = split.label_names.index("positive") if "positive" in split.label_names else 1
    return classification_metrics(split.labels, preds, positive=positive), preds
