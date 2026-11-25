"""PyTorch baseline inference runner.

This is the reference implementation every optimized variant is compared
against. It provides:

* :func:`seed_everything` -- deterministic seeding of Python, NumPy and PyTorch
  (CPU + CUDA), optionally forcing deterministic kernels.
* :func:`resolve_device` -- ``auto``/``cpu``/``cuda`` resolution with a clear
  error when a CUDA build of torch is missing.
* :func:`encode` -- tokenisation into ``int64`` NumPy arrays, shared verbatim
  with the ONNX Runtime path so both backends see identical inputs.
* :class:`TorchClassifier` -- model + tokenizer bound to a device/precision,
  with a raw :meth:`~TorchClassifier.forward` for timing loops and
  convenience :meth:`~TorchClassifier.predict` helpers.
"""

from __future__ import annotations

import os

# cuBLAS needs a fixed workspace config for deterministic GEMMs when
# ``torch.use_deterministic_algorithms(True)`` is on. It is read when the
# cuBLAS handle is created, so it must be set before any CUDA work -- i.e. at
# import time, not inside a function. ``setdefault`` respects user overrides.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
import numpy.typing as npt
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from opt_pipeline.config import (
    AttnImplementation,
    DeviceName,
    PipelineConfig,
    Precision,
)
from opt_pipeline.errors import ConfigError, DeviceUnavailableError
from opt_pipeline.utils import get_logger

log = get_logger(__name__)

PaddingStrategy = Literal["max_length", "longest"]
"""``max_length`` gives fixed shapes (benchmarks); ``longest`` is realistic serving."""

Encoded: TypeAlias = dict[str, npt.NDArray[np.int64]]
"""Tokenised batch: ``{"input_ids": (B, S), "attention_mask": (B, S)}`` as int64."""

MODEL_INPUT_NAMES: tuple[str, ...] = ("input_ids", "attention_mask")
"""DistilBERT takes no ``token_type_ids``; these names are also the ONNX input names."""

_DTYPES: dict[Precision, torch.dtype] = {"fp32": torch.float32, "fp16": torch.float16}


# --------------------------------------------------------------------------
# Reproducibility & device handling
# --------------------------------------------------------------------------
def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    """Seed every RNG the pipeline touches.

    Args:
        seed: Seed for ``random``, ``numpy`` and ``torch`` (all CUDA devices).
        deterministic: Also request deterministic kernels
            (``torch.use_deterministic_algorithms``, cuDNN deterministic, no
            cuDNN autotuning). Inference has no dropout, so this mainly pins
            reduction order; ``warn_only=True`` keeps ops without a
            deterministic variant usable instead of raising.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # seeds CPU and all CUDA devices
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(device: DeviceName | str) -> torch.device:
    """Map ``auto``/``cpu``/``cuda`` to a concrete :class:`torch.device`.

    Raises:
        DeviceUnavailableError: If ``cuda`` is requested but unavailable --
            usually a CPU-only torch wheel (``torch.version.cuda is None``) or
            a driver problem.
        ConfigError: If ``device`` is not one of the supported names.
    """
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise DeviceUnavailableError(
            "CUDA was requested but torch.cuda.is_available() is False "
            f"(torch {torch.__version__}, built for CUDA {torch.version.cuda}). "
            "If CUDA is None, a CPU wheel was installed -- run `uv sync` so torch "
            "is reinstalled from the CUDA 12.6 index pinned in pyproject.toml."
        )
    if device not in ("cpu", "cuda"):
        raise ConfigError(f"unsupported device {device!r}; expected auto, cpu or cuda")
    return torch.device(device)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_tokenizer(cfg: PipelineConfig) -> PreTrainedTokenizerBase:
    """Load the (fast) tokenizer into the project-local model cache."""
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        cfg.model_id,
        revision=cfg.revision,
        cache_dir=str(cfg.paths.model_cache_dir),
    )
    return tokenizer


def load_model(
    cfg: PipelineConfig,
    *,
    device: torch.device,
    precision: Precision = "fp32",
    attn_implementation: AttnImplementation | None = None,
) -> PreTrainedModel:
    """Load the sequence-classification model in eval mode on ``device``.

    Args:
        cfg: Pipeline configuration (model id, revision, cache dir).
        device: Target device.
        precision: ``fp32`` or ``fp16``. FP16 is only allowed on CUDA: CPU
            half-precision kernels are slow or missing for several ops.
        attn_implementation: Override for ``cfg.attn_implementation``.

    Raises:
        ConfigError: If FP16 is requested on CPU, or ``cfg.max_seq_length``
            exceeds the model's position embedding table.
    """
    if precision == "fp16" and device.type != "cuda":
        raise ConfigError("fp16 inference is only supported on CUDA; use fp32 on CPU")

    model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_id,
        revision=cfg.revision,
        cache_dir=str(cfg.paths.model_cache_dir),
        attn_implementation=attn_implementation or cfg.attn_implementation,
    )
    max_positions = int(getattr(model.config, "max_position_embeddings", cfg.max_seq_length))
    if cfg.max_seq_length > max_positions:
        raise ConfigError(
            f"max_seq_length={cfg.max_seq_length} exceeds the model's "
            f"max_position_embeddings={max_positions}"
        )
    model = model.to(device=device, dtype=_DTYPES[precision])
    model.eval()
    return model


def encode(
    tokenizer: PreTrainedTokenizerBase,
    texts: Sequence[str],
    *,
    max_length: int,
    padding: PaddingStrategy = "max_length",
) -> Encoded:
    """Tokenise ``texts`` into contiguous ``int64`` arrays keyed by input name.

    Used by both the PyTorch and ONNX Runtime paths so the two backends are
    always fed byte-identical inputs.
    """
    if not texts:
        raise ValueError("texts must be non-empty")
    batch = tokenizer(
        list(texts),
        padding=padding,
        truncation=True,
        max_length=max_length,
        return_tensors="np",
    )
    return {name: np.ascontiguousarray(batch[name], dtype=np.int64) for name in MODEL_INPUT_NAMES}


def softmax(logits: npt.NDArray[np.floating]) -> npt.NDArray[np.float64]:
    """Numerically stable row-wise softmax in float64."""
    x = logits.astype(np.float64)
    x -= x.max(axis=-1, keepdims=True)
    np.exp(x, out=x)
    x /= x.sum(axis=-1, keepdims=True)
    return x


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Prediction:
    """A single classification result."""

    text: str
    label: str
    score: float
    logits: tuple[float, ...]


class TorchClassifier:
    """PyTorch sequence classifier bound to one device and precision.

    Example:
        >>> clf = TorchClassifier(PipelineConfig(), device="cpu")  # doctest: +SKIP
        >>> clf.predict(["What a wonderful film!"])[0].label  # doctest: +SKIP
        'POSITIVE'
    """

    def __init__(
        self,
        cfg: PipelineConfig,
        *,
        device: DeviceName | str | None = None,
        precision: Precision = "fp32",
        attn_implementation: AttnImplementation | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
    ) -> None:
        """Load tokenizer and model.

        Args:
            cfg: Pipeline configuration.
            device: ``auto``/``cpu``/``cuda``; defaults to ``cfg.device``.
            precision: ``fp32`` or ``fp16`` (CUDA only).
            attn_implementation: Override for ``cfg.attn_implementation``.
            tokenizer: Optional pre-loaded tokenizer to share between runners.
        """
        self.cfg = cfg
        self.device = resolve_device(device or cfg.device)
        self.precision: Precision = precision
        self.tokenizer = tokenizer or load_tokenizer(cfg)
        self.model = load_model(
            cfg,
            device=self.device,
            precision=precision,
            attn_implementation=attn_implementation,
        )
        self.id2label: dict[int, str] = {
            int(k): str(v) for k, v in (self.model.config.id2label or {}).items()
        }
        log.info(
            "Loaded %s on %s (%s, attn=%s)",
            cfg.model_id,
            self.device,
            precision,
            attn_implementation or cfg.attn_implementation,
        )

    # -- input staging -----------------------------------------------------
    def encode(self, texts: Sequence[str], *, padding: PaddingStrategy = "max_length") -> Encoded:
        """Tokenise with the configured ``max_seq_length``."""
        return encode(self.tokenizer, texts, max_length=self.cfg.max_seq_length, padding=padding)

    def to_device(self, encoded: Mapping[str, npt.NDArray[np.int64]]) -> dict[str, torch.Tensor]:
        """Copy an encoded batch to the model's device (do this *outside* timing loops)."""
        return {
            name: torch.from_numpy(encoded[name]).to(self.device, non_blocking=False)
            for name in MODEL_INPUT_NAMES
        }

    # -- execution ---------------------------------------------------------
    @torch.inference_mode()
    def forward(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Run the model on pre-staged device tensors and return raw logits.

        Asynchronous on CUDA: callers that time this must bracket it with
        :meth:`synchronize`.
        """
        logits: torch.Tensor = self.model(**inputs).logits
        return logits

    def synchronize(self) -> None:
        """Block until all queued CUDA work has finished (no-op on CPU)."""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def predict_logits(
        self, encoded: Mapping[str, npt.NDArray[np.int64]]
    ) -> npt.NDArray[np.float32]:
        """Run an encoded batch and return logits as host ``float32``."""
        logits = self.forward(self.to_device(encoded))
        return logits.float().cpu().numpy()

    def predict(
        self, texts: Sequence[str], *, padding: PaddingStrategy = "longest"
    ) -> list[Prediction]:
        """Classify raw texts (tokenise -> forward -> softmax)."""
        logits = self.predict_logits(self.encode(texts, padding=padding))
        probs = softmax(logits)
        labels = probs.argmax(axis=-1)
        return [
            Prediction(
                text=text,
                label=self.id2label[int(idx)],
                score=float(prob[idx]),
                logits=tuple(float(v) for v in row),
            )
            for text, idx, prob, row in zip(texts, labels, probs, logits, strict=True)
        ]
