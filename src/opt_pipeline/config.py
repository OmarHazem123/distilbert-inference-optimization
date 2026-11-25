"""Central, immutable configuration for the inference-optimization pipeline.

Every stage (baseline, export, quantize, benchmark, profile) reads its settings
from a single :class:`PipelineConfig` instance so that runs are reproducible and
the exact configuration can be serialised next to the results it produced.

Example:
    >>> cfg = PipelineConfig()
    >>> cfg.batch_sizes
    (1, 8, 32)
    >>> fast = cfg.with_overrides(benchmark_iters=20, batch_sizes=[1, 8])
    >>> fast.paths.onnx_fp32.name
    'model_fp32.onnx'
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from opt_pipeline.errors import ConfigError

__all__ = [
    "DEFAULT_MODEL_ID",
    "DEFAULT_MODEL_REVISION",
    "AttnImplementation",
    "ConfigError",
    "DeviceName",
    "PathsConfig",
    "PipelineConfig",
    "Precision",
    "Tolerances",
]

DeviceName = Literal["auto", "cpu", "cuda"]
"""Execution target: ``auto`` picks CUDA when available, otherwise CPU."""

Precision = Literal["fp32", "fp16"]
"""Floating-point precision for the PyTorch baseline."""

AttnImplementation = Literal["eager", "sdpa"]
"""Attention kernel used by both the PyTorch baseline and the ONNX export.

``sdpa`` (``torch.nn.functional.scaled_dot_product_attention``) is the
transformers default; keeping baseline and export on the same implementation
makes the PyTorch-vs-ONNX comparison like-for-like."""

DEFAULT_MODEL_ID = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
"""Canonical Hub id (the legacy un-namespaced id redirects to this one)."""

DEFAULT_MODEL_REVISION = "714eb0fa89d2f80546fda750413ed43d93601a13"
"""Hub commit the benchmarks were produced with; pinned so weights cannot drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


@dataclass(frozen=True, slots=True)
class Tolerances:
    """Numerical acceptance thresholds used by parity checks and tests.

    Attributes:
        fp32_atol: Absolute tolerance for ``np.allclose`` between PyTorch FP32
            and ONNX Runtime FP32 logits. These should agree to ~1e-5; 1e-4
            leaves headroom for kernel-order differences across EPs.
        fp32_rtol: Relative tolerance for the same comparison.
        fp16_max_abs_logit_diff: Largest acceptable absolute logit deviation of
            an FP16 graph from the FP32 reference.
        min_label_agreement_fp16: Minimum fraction of argmax predictions an FP16
            model must share with the FP32 reference.
        min_label_agreement_int8: Same, for the dynamically quantized INT8 model.
        max_accuracy_drop_fp16: Largest acceptable absolute drop in SST-2
            validation accuracy (0.005 = 0.5 percentage points).
        max_accuracy_drop_int8: Same, for INT8 (quantization is lossier).
    """

    fp32_atol: float = 1e-4
    fp32_rtol: float = 1e-3
    fp16_max_abs_logit_diff: float = 5e-2
    min_label_agreement_fp16: float = 0.99
    min_label_agreement_int8: float = 0.95
    max_accuracy_drop_fp16: float = 0.005
    max_accuracy_drop_int8: float = 0.02

    def __post_init__(self) -> None:
        for name in ("fp32_atol", "fp32_rtol", "fp16_max_abs_logit_diff"):
            _require(getattr(self, name) >= 0.0, f"{name} must be >= 0")
        for name in (
            "min_label_agreement_fp16",
            "min_label_agreement_int8",
            "max_accuracy_drop_fp16",
            "max_accuracy_drop_int8",
        ):
            value = getattr(self, name)
            _require(0.0 <= value <= 1.0, f"{name} must be in [0, 1], got {value}")


@dataclass(frozen=True, slots=True)
class PathsConfig:
    """Filesystem layout for downloaded models, artifacts, and results.

    Paths are relative to the current working directory (the repository root
    when invoked through ``tasks.py`` / ``make``).
    """

    artifacts_dir: Path = Path("artifacts")
    results_dir: Path = Path("results")
    model_cache_dir: Path = Path("models")
    data_cache_dir: Path = Path("data")

    @property
    def onnx_dir(self) -> Path:
        """Directory holding all exported / converted ONNX graphs."""
        return self.artifacts_dir / "onnx"

    @property
    def onnx_fp32(self) -> Path:
        """FP32 ONNX graph exported from PyTorch."""
        return self.onnx_dir / "model_fp32.onnx"

    @property
    def onnx_fp16(self) -> Path:
        """FP16-converted ONNX graph."""
        return self.onnx_dir / "model_fp16.onnx"

    @property
    def onnx_int8(self) -> Path:
        """Dynamically quantized INT8 ONNX graph."""
        return self.onnx_dir / "model_int8.onnx"

    @property
    def profiles_dir(self) -> Path:
        """Directory for ``torch.profiler`` traces."""
        return self.artifacts_dir / "profiles"

    @property
    def benchmarks_json(self) -> Path:
        """Structured benchmark output consumed by the README table."""
        return self.results_dir / "benchmarks.json"


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Top-level configuration for every pipeline stage.

    Attributes:
        model_id: Hugging Face Hub model id.
        revision: Hub revision (branch, tag, or commit sha) to pin the weights.
        max_seq_length: Tokenizer truncation length and benchmark sequence
            length. SST-2 sentences are short; 128 covers >99% of them.
        batch_sizes: Batch sizes to benchmark, stored sorted and de-duplicated.
        warmup_runs: Untimed iterations per configuration (CUDA context, cuDNN
            autotuning, ORT arena growth, CPU caches).
        benchmark_iters: Timed iterations per configuration; p99 needs >= 100
            samples to be meaningful.
        seed: Global RNG seed for Python, NumPy, and PyTorch.
        device: Default execution target for the PyTorch baseline.
        attn_implementation: Attention kernel for baseline and export.
        opset_version: ONNX opset used by the exporter.
        dataset_name: Hub dataset id for evaluation.
        dataset_config: Dataset configuration name.
        dataset_split: Split used for accuracy / F1 evaluation.
        eval_max_samples: Optional cap on evaluation samples (``None`` = all
            872 SST-2 validation sentences).
        tolerances: Numerical acceptance thresholds.
        paths: Filesystem layout.
    """

    model_id: str = DEFAULT_MODEL_ID
    revision: str = DEFAULT_MODEL_REVISION
    max_seq_length: int = 128
    batch_sizes: tuple[int, ...] = (1, 8, 32)
    warmup_runs: int = 10
    benchmark_iters: int = 100
    seed: int = 42
    device: DeviceName = "auto"
    attn_implementation: AttnImplementation = "sdpa"
    opset_version: int = 17
    dataset_name: str = "nyu-mll/glue"
    dataset_config: str = "sst2"
    dataset_split: str = "validation"
    eval_max_samples: int | None = None
    tolerances: Tolerances = field(default_factory=Tolerances)
    paths: PathsConfig = field(default_factory=PathsConfig)

    def __post_init__(self) -> None:
        # Normalise batch sizes (accept any iterable, e.g. a list from the CLI)
        # into a sorted, de-duplicated tuple. ``object.__setattr__`` is the
        # sanctioned way to assign inside a frozen dataclass's __post_init__.
        sizes = tuple(sorted(set(self.batch_sizes)))
        object.__setattr__(self, "batch_sizes", sizes)

        _require(bool(self.model_id), "model_id must be non-empty")
        _require(self.max_seq_length > 0, "max_seq_length must be > 0")
        _require(len(sizes) > 0, "batch_sizes must contain at least one value")
        _require(all(b > 0 for b in sizes), f"batch sizes must be > 0, got {sizes}")
        _require(self.warmup_runs >= 0, "warmup_runs must be >= 0")
        _require(self.benchmark_iters > 0, "benchmark_iters must be > 0")
        _require(self.seed >= 0, "seed must be >= 0")
        _require(self.device in ("auto", "cpu", "cuda"), f"invalid device {self.device!r}")
        _require(
            self.attn_implementation in ("eager", "sdpa"),
            f"invalid attn_implementation {self.attn_implementation!r}",
        )
        _require(self.opset_version >= 14, "opset_version must be >= 14")
        _require(
            self.eval_max_samples is None or self.eval_max_samples > 0,
            "eval_max_samples must be None or > 0",
        )

    def with_overrides(self, **overrides: Any) -> PipelineConfig:
        """Return a copy with the given fields replaced.

        ``None`` values are ignored so CLI flags that were not supplied do not
        clobber defaults. Iterables for ``batch_sizes`` are accepted.

        Raises:
            ConfigError: If an unknown field is given or a value is invalid.
        """
        known = set(self.__dataclass_fields__)
        unknown = set(overrides) - known
        _require(not unknown, f"unknown config fields: {sorted(unknown)}")
        changes = {k: v for k, v in overrides.items() if v is not None}
        if "batch_sizes" in changes and isinstance(changes["batch_sizes"], Iterable):
            changes["batch_sizes"] = tuple(changes["batch_sizes"])
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to JSON-compatible primitives (paths become strings)."""

        def _convert(value: Any) -> Any:
            if isinstance(value, Path):
                return value.as_posix()
            if isinstance(value, dict):
                return {k: _convert(v) for k, v in value.items()}
            if isinstance(value, list | tuple):
                return [_convert(v) for v in value]
            return value

        result: dict[str, Any] = _convert(asdict(self))
        return result
