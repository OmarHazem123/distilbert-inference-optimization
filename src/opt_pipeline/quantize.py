"""ONNX Runtime dynamic INT8 quantization and FP16 graph conversion.

* :func:`quantize_int8` -- ``quant_pre_process`` (shape inference + basic graph
  cleanup, as ORT recommends), then ``quantize_dynamic``: weights are INT8
  ahead of time, activation scales are computed per batch at run time. No
  calibration data is needed. Targets the CPU execution provider.
* :func:`convert_fp16` -- casts float initializers and ops to FP16 with
  ``keep_io_types=True`` (int64 ids in, float32 logits out), so the FP16 graph
  is a drop-in replacement. Targets the CUDA execution provider.
* :func:`compare_with_reference` -- logit deviation and label agreement of a
  converted graph against the PyTorch FP32 reference, gated by the configured
  tolerances.

Every artifact is written atomically (temp sibling file + ``os.replace``).
"""

from __future__ import annotations

import tempfile
import time
import warnings
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import onnx

from opt_pipeline.baseline import TorchClassifier, seed_everything
from opt_pipeline.config import PipelineConfig
from opt_pipeline.errors import MissingArtifactError, ParityError
from opt_pipeline.evaluate import batched_logits
from opt_pipeline.export_onnx import OrtDevice
from opt_pipeline.runners import OnnxClassifier
from opt_pipeline.utils import atomic_output, file_size_mb, get_logger

log = get_logger(__name__)

ConvertedPrecision = Literal["int8", "fp16"]


@dataclass(frozen=True, slots=True)
class ConversionResult:
    """Outcome of one quantization / conversion step."""

    precision: ConvertedPrecision
    source: str
    output: str
    source_size_mb: float
    size_mb: float
    seconds: float
    settings: dict[str, Any]

    @property
    def compression(self) -> float:
        """Source size divided by output size."""
        return self.source_size_mb / self.size_mb

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable view including the compression ratio."""
        return {**asdict(self), "compression": round(self.compression, 3)}

    def summary(self) -> str:
        """One-line human-readable summary."""
        return (
            f"{self.precision.upper()}: {self.source_size_mb:.1f} MB -> {self.size_mb:.1f} MB "
            f"({self.compression:.2f}x smaller) in {self.seconds:.1f}s -> {self.output}"
        )


def _require_source(path: Path) -> None:
    if not path.exists():
        raise MissingArtifactError(f"{path} not found -- run `opt-pipeline export` first")


# --------------------------------------------------------------------------
# INT8 dynamic quantization
# --------------------------------------------------------------------------
def quantize_int8(
    cfg: PipelineConfig, source: Path | None = None, output: Path | None = None
) -> ConversionResult:
    """Dynamically quantize the FP32 ONNX graph to INT8 weights.

    Args:
        cfg: Pipeline configuration; ``cfg.quantization`` selects per-channel,
            reduce-range and the quantized operator types.
        source: FP32 ONNX graph; defaults to ``cfg.paths.onnx_fp32``.
        output: Destination; defaults to ``cfg.paths.onnx_int8``.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from onnxruntime.quantization.shape_inference import quant_pre_process

    source = source or cfg.paths.onnx_fp32
    output = output or cfg.paths.onnx_int8
    _require_source(source)
    q = cfg.quantization
    settings: dict[str, Any] = {
        "method": "onnxruntime.quantization.quantize_dynamic",
        "weight_type": "QInt8",
        "activations": "dynamic QUInt8 (per-batch scale)",
        "per_channel": q.int8_per_channel,
        "reduce_range": q.int8_reduce_range,
        "op_types_to_quantize": list(q.int8_op_types),
        # ORT's *symbolic* shape inference cannot resolve the attention-mask
        # subgraph transformers 5 emits ("Incomplete symbolic shape
        # inference"), so it is skipped; ONNX shape inference + ORT basic
        # graph optimizations still run.
        "pre_process": "quant_pre_process(skip_symbolic_shape=True)",
    }
    log.info("Quantizing %s -> %s (INT8 dynamic, %s)", source, output, settings)

    start = time.perf_counter()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Same directory as the destination so the final os.replace is a rename.
    with tempfile.TemporaryDirectory(dir=output.parent, ignore_cleanup_errors=True) as tmpdir:
        preprocessed = Path(tmpdir) / "preprocessed.onnx"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # quant tooling is chatty about shape inference
            quant_pre_process(str(source), str(preprocessed), skip_symbolic_shape=True)
        with atomic_output(output) as tmp:
            quantize_dynamic(
                model_input=preprocessed,
                model_output=tmp,
                weight_type=QuantType.QInt8,
                per_channel=q.int8_per_channel,
                reduce_range=q.int8_reduce_range,
                op_types_to_quantize=list(q.int8_op_types),
            )
            onnx.checker.check_model(str(tmp))
    result = ConversionResult(
        precision="int8",
        source=source.as_posix(),
        output=output.as_posix(),
        source_size_mb=file_size_mb(source),
        size_mb=file_size_mb(output),
        seconds=time.perf_counter() - start,
        settings=settings,
    )
    log.info(result.summary())
    return result


# --------------------------------------------------------------------------
# FP16 conversion
# --------------------------------------------------------------------------
def convert_fp16(
    cfg: PipelineConfig, source: Path | None = None, output: Path | None = None
) -> ConversionResult:
    """Convert the FP32 ONNX graph to FP16 (inputs/outputs keep their types).

    Constants outside the FP16 range (e.g. the ``finfo(float32).min`` used for
    attention masking) are clamped to +/-65504 by the converter, which keeps
    masked softmax finite.
    """
    from onnxruntime.transformers.float16 import convert_float_to_float16

    source = source or cfg.paths.onnx_fp32
    output = output or cfg.paths.onnx_fp16
    _require_source(source)
    settings: dict[str, Any] = {
        "method": "onnxruntime.transformers.float16.convert_float_to_float16",
        "keep_io_types": cfg.quantization.fp16_keep_io_types,
    }
    log.info("Converting %s -> %s (FP16, %s)", source, output, settings)

    start = time.perf_counter()
    model = onnx.load(str(source))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # value-clamping notices for out-of-range constants
        model_fp16 = convert_float_to_float16(
            model, keep_io_types=cfg.quantization.fp16_keep_io_types
        )
    with atomic_output(output) as tmp:
        onnx.save(model_fp16, str(tmp))
        onnx.checker.check_model(str(tmp))
    result = ConversionResult(
        precision="fp16",
        source=source.as_posix(),
        output=output.as_posix(),
        source_size_mb=file_size_mb(source),
        size_mb=file_size_mb(output),
        seconds=time.perf_counter() - start,
        settings=settings,
    )
    log.info(result.summary())
    return result


# --------------------------------------------------------------------------
# Verification against the FP32 reference
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class AgreementReport:
    """How closely a converted graph tracks the PyTorch FP32 reference."""

    precision: ConvertedPrecision
    provider: str
    n: int
    max_abs_logit_diff: float
    mean_abs_logit_diff: float
    label_agreement: float
    min_label_agreement: float
    max_abs_logit_diff_limit: float | None

    @property
    def passed(self) -> bool:
        """Label agreement (and, for FP16, logit deviation) within tolerance."""
        ok = self.label_agreement >= self.min_label_agreement
        if self.max_abs_logit_diff_limit is not None:
            ok = ok and self.max_abs_logit_diff <= self.max_abs_logit_diff_limit
        return ok

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable view including the verdict."""
        return {**asdict(self), "passed": self.passed}

    def summary(self) -> str:
        """One-line human-readable summary."""
        limit = (
            f" (limit {self.max_abs_logit_diff_limit:g})"
            if self.max_abs_logit_diff_limit is not None
            else ""
        )
        return (
            f"{'PASS' if self.passed else 'FAIL'}  {self.precision.upper()} on {self.provider}: "
            f"label agreement {self.label_agreement:.2%} (min {self.min_label_agreement:.0%}) "
            f"over {self.n} sentences, max|logit diff| {self.max_abs_logit_diff:.3e}{limit}"
        )


def compare_with_reference(
    cfg: PipelineConfig,
    onnx_path: Path,
    *,
    precision: ConvertedPrecision,
    device: OrtDevice,
    sentences: Sequence[str],
    reference: TorchClassifier | None = None,
    raise_on_failure: bool = False,
) -> AgreementReport:
    """Compare a converted ONNX graph against PyTorch FP32 (CPU) on ``sentences``.

    INT8 is gated on label agreement only: its logits legitimately move by
    ~1e-2..1e-1, while the argmax is what users consume. FP16 is additionally
    gated on max absolute logit deviation.
    """
    seed_everything(cfg.seed)
    ref = reference or TorchClassifier(cfg, device="cpu", precision="fp32")
    candidate = OnnxClassifier(onnx_path, device=device)
    kwargs = {"batch_size": cfg.eval_batch_size, "max_length": cfg.max_seq_length}
    expected = batched_logits(ref, ref.tokenizer, sentences, **kwargs)
    actual = batched_logits(candidate, ref.tokenizer, sentences, **kwargs)
    diff = np.abs(actual - expected)

    tol = cfg.tolerances
    report = AgreementReport(
        precision=precision,
        provider=candidate.provider,
        n=len(sentences),
        max_abs_logit_diff=float(diff.max()),
        mean_abs_logit_diff=float(diff.mean()),
        label_agreement=float(np.mean(actual.argmax(-1) == expected.argmax(-1))),
        min_label_agreement=(
            tol.min_label_agreement_int8 if precision == "int8" else tol.min_label_agreement_fp16
        ),
        max_abs_logit_diff_limit=tol.fp16_max_abs_logit_diff if precision == "fp16" else None,
    )
    log.info(report.summary())
    if raise_on_failure and not report.passed:
        raise ParityError(report.summary())
    return report
