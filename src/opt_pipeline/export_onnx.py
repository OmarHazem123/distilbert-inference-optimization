"""Export the Hugging Face model to ONNX and verify numerical parity.

Pipeline:

1. :func:`export_onnx` traces the FP32 model on CPU with dynamic
   ``batch_size`` / ``sequence_length`` axes at ``cfg.opset_version`` (17),
   validates it with ``onnx.checker``, and writes it atomically.
2. :func:`verify_parity` runs PyTorch FP32 (reference) and ONNX Runtime side by
   side over several shapes -- variable batch sizes and sequence lengths -- and
   checks ``np.allclose`` within the configured tolerances.

:func:`create_ort_session` is the single place ONNX Runtime sessions are built.
It refuses to silently fall back from CUDA to CPU, so a mis-installed CUDA/cuDNN
can never be reported as a "GPU" result.
"""

from __future__ import annotations

import sys
import warnings
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import onnx
import onnxruntime as ort
import torch

from opt_pipeline.baseline import (
    MODEL_INPUT_NAMES,
    PaddingStrategy,
    TorchClassifier,
    encode,
    seed_everything,
)
from opt_pipeline.config import ExporterName, PipelineConfig
from opt_pipeline.errors import (
    ExportError,
    MissingArtifactError,
    ParityError,
    ProviderUnavailableError,
)
from opt_pipeline.utils import atomic_output, file_size_mb, get_logger, utf8_stdio

# NOTE on Windows CUDA DLLs: the ORT CUDA provider DLL is loaded lazily, at the
# first CUDA session creation. ``import torch`` (above) has already loaded the
# CUDA 12 / cuDNN 9 DLLs bundled in torch/lib into the process, so ORT resolves
# them from there -- no system CUDA Toolkit or cuDNN install required.

log = get_logger(__name__)

OrtDevice = Literal["cpu", "cuda"]

OUTPUT_NAMES: tuple[str, ...] = ("logits",)
DYNAMIC_AXES: dict[str, dict[int, str]] = {
    "input_ids": {0: "batch_size", 1: "sequence_length"},
    "attention_mask": {0: "batch_size", 1: "sequence_length"},
    "logits": {0: "batch_size"},
}

PARITY_TEXTS: tuple[str, ...] = (
    "ok",
    "Terrible.",
    "A gorgeous, witty, seductive movie.",
    "The plot is thin and the acting is wooden, but the soundtrack almost saves it.",
    "I expected nothing and was still disappointed.",
    "an utterly charming and hilarious film that reminded me why I love cinema",
    "It's not the worst thing I've seen this year, though it comes close at times.",
    # Longer than max_seq_length tokens -> exercises truncation.
    " ".join(["This film is a masterpiece of quiet, patient storytelling."] * 30),
)
"""Fixed, varied-length sentences for parity checks (incl. one that truncates)."""


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------
class _LogitsOnly(torch.nn.Module):
    """Wrap a HF model so the traced graph has exactly one output: ``logits``."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        logits: torch.Tensor = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        return logits


def export_onnx(
    cfg: PipelineConfig,
    output_path: Path | None = None,
    *,
    exporter: ExporterName | None = None,
) -> Path:
    """Export the FP32 model to ONNX with dynamic batch/sequence axes.

    The export runs on CPU in FP32 so the resulting graph is device-agnostic;
    precision/device specialisation happens later (quantize / EP choice).

    Args:
        cfg: Pipeline configuration (model id, opset, max length, paths).
        output_path: Destination; defaults to ``cfg.paths.onnx_fp32``.
        exporter: Override for ``cfg.exporter`` (``torchscript`` or ``dynamo``).

    Returns:
        Path to the validated ONNX file.

    Raises:
        onnx.checker.ValidationError: If the exported graph is malformed.
        ExportError: If the written opset differs from ``cfg.opset_version``
            (e.g. the dynamo exporter could not down-convert from opset 18).
    """
    output_path = output_path or cfg.paths.onnx_fp32
    exporter = exporter or cfg.exporter
    seed_everything(cfg.seed)

    clf = TorchClassifier(cfg, device="cpu", precision="fp32")
    wrapper = _LogitsOnly(clf.model).eval()

    # Batch of 2 with unequal lengths so the tracer cannot specialise on a
    # batch size of 1 or on a fully-unmasked sequence.
    sample = clf.encode(["a short one", "a noticeably longer example sentence"], padding="longest")
    args = tuple(torch.from_numpy(sample[name]) for name in MODEL_INPUT_NAMES)

    log.info(
        "Exporting %s -> %s (opset %d, %s exporter)",
        cfg.model_id,
        output_path,
        cfg.opset_version,
        exporter,
    )
    with atomic_output(output_path) as tmp:
        # no_grad rather than inference_mode: the TorchScript tracer cannot
        # record ops on inference-mode tensors. Exporter warnings are captured
        # and summarised (see _log_export_warnings) instead of spamming stderr.
        with torch.no_grad(), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if exporter == "torchscript":
                _export_torchscript(wrapper, args, tmp, cfg.opset_version)
            elif exporter == "dynamo":
                _export_dynamo(wrapper, args, tmp, cfg.opset_version)
            else:  # pragma: no cover - guarded by PipelineConfig validation
                raise ExportError(f"unknown exporter {exporter!r}")
        onnx.checker.check_model(str(tmp), full_check=True)
        _assert_opset(tmp, cfg.opset_version, exporter)

    _log_export_warnings(caught)
    log.info("Exported %.1f MB ONNX graph to %s", file_size_mb(output_path), output_path)
    return output_path


def _export_torchscript(
    model: torch.nn.Module, args: tuple[torch.Tensor, ...], path: Path, opset: int
) -> None:
    """Legacy TorchScript exporter: emits ``opset`` natively, honours ``dynamic_axes``."""
    torch.onnx.export(
        model,
        args,
        str(path),
        input_names=list(MODEL_INPUT_NAMES),
        output_names=list(OUTPUT_NAMES),
        dynamic_axes=DYNAMIC_AXES,
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )


def _export_dynamo(
    model: torch.nn.Module, args: tuple[torch.Tensor, ...], path: Path, opset: int
) -> None:
    """``torch.export``-based exporter.

    It builds at opset 18 and runs the ONNX version converter to reach lower
    opsets; :func:`_assert_opset` verifies the conversion actually happened.
    ``dynamic_axes`` is translated to ``dynamic_shapes`` internally.
    """
    with utf8_stdio():
        torch.onnx.export(
            model,
            args,
            str(path),
            input_names=list(MODEL_INPUT_NAMES),
            output_names=list(OUTPUT_NAMES),
            dynamic_axes=DYNAMIC_AXES,
            opset_version=opset,
            dynamo=True,
            external_data=False,  # single self-contained file (< 2 GB)
        )


def _log_export_warnings(caught: list[warnings.WarningMessage]) -> None:
    """Summarise exporter warnings in one line; full text at DEBUG.

    With transformers 5 + TorchScript tracing these are known-benign:
    ``TracerWarning`` on shape-only Python branches (mask padding length,
    SDPA ``is_causal`` -- always False for an encoder) and an ``aten::index``
    note about negative indices (none are used). They *would* matter if a
    branch depended on batch size or sequence length, which is exactly what
    :func:`verify_parity` checks by running many unseen shapes.
    """
    if not caught:
        return
    kinds = Counter(w.category.__name__ for w in caught)
    log.info(
        "Exporter emitted %d warning(s) %s; generalisation is verified by the parity check",
        len(caught),
        dict(kinds),
    )
    for w in caught:
        log.debug("%s: %s (%s:%s)", w.category.__name__, w.message, w.filename, w.lineno)


def _assert_opset(path: Path, expected: int, exporter: str) -> None:
    model = onnx.load(str(path), load_external_data=False)
    opsets = {imp.domain or "ai.onnx": imp.version for imp in model.opset_import}
    if opsets.get("ai.onnx") != expected:
        raise ExportError(
            f"{exporter} exporter produced default-domain opset {opsets.get('ai.onnx')}, "
            f"expected {expected} (all imports: {opsets}). The dynamo exporter only "
            "down-converts from opset 18 when the ONNX version converter succeeds; "
            "use --exporter torchscript or --opset 18."
        )


def onnx_opset(path: Path) -> int:
    """Return the default-domain (``ai.onnx``) opset of an ONNX file."""
    model = onnx.load(str(path), load_external_data=False)
    for imp in model.opset_import:
        if imp.domain in ("", "ai.onnx"):
            return int(imp.version)
    raise ValueError(f"{path} has no default-domain opset import")


# --------------------------------------------------------------------------
# ONNX Runtime
# --------------------------------------------------------------------------
def create_ort_session(
    path: Path,
    *,
    device: OrtDevice = "cpu",
    intra_op_num_threads: int | None = None,
    optimization_level: ort.GraphOptimizationLevel = ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    log_severity_level: int = 3,
) -> ort.InferenceSession:
    """Create an ONNX Runtime session on the requested device, never silently.

    ORT quietly falls back to ``CPUExecutionProvider`` when the CUDA provider
    cannot load (missing ``cudnn64_9.dll``, CUDA major mismatch, ...). That
    would make "GPU" benchmark numbers meaningless, so the active provider is
    checked after construction.

    Args:
        path: ONNX model file.
        device: ``cpu`` or ``cuda``.
        intra_op_num_threads: CPU thread-pool size (``None`` = ORT default,
            i.e. physical cores).
        optimization_level: ORT graph optimization level.
        log_severity_level: ORT log level (0 verbose ... 3 error, 4 fatal).

    Raises:
        ProviderUnavailableError: If ``cuda`` was requested but the CUDA EP
            is not the session's primary provider.
    """
    options = ort.SessionOptions()
    options.graph_optimization_level = optimization_level
    # 3 = ERROR. Silences the CUDA EP "Memcpy nodes are added" warning (a few
    # shape subgraphs stay on CPU -- expected for BERT-style graphs). Provider
    # fallback, the warning that matters, is checked explicitly below.
    options.log_severity_level = log_severity_level
    if intra_op_num_threads is not None:
        options.intra_op_num_threads = intra_op_num_threads

    providers: list[str | tuple[str, dict[str, Any]]]
    if device == "cuda":
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise ProviderUnavailableError(
                "CUDAExecutionProvider is not compiled into this onnxruntime build "
                f"(available: {ort.get_available_providers()}). Is the CPU-only "
                "`onnxruntime` package shadowing `onnxruntime-gpu`?"
            )
        providers = [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    session = ort.InferenceSession(str(path), sess_options=options, providers=providers)

    active = session.get_providers()
    if device == "cuda" and active[0] != "CUDAExecutionProvider":
        raise ProviderUnavailableError(
            f"Requested CUDAExecutionProvider but the session is running on {active}. "
            "The CUDA 12 / cuDNN 9 DLLs could not be loaded; make sure the CUDA build "
            "of torch is installed and imported before onnxruntime."
        )
    return session


def run_onnx(
    session: ort.InferenceSession, encoded: Mapping[str, npt.NDArray[np.int64]]
) -> npt.NDArray[np.float32]:
    """Run an encoded batch through an ORT session and return float32 logits."""
    feed = {inp.name: encoded[inp.name] for inp in session.get_inputs()}
    (logits,) = session.run(list(OUTPUT_NAMES), feed)
    return np.asarray(logits, dtype=np.float32)


# --------------------------------------------------------------------------
# Parity verification
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ParityCase:
    """Comparison result for one input shape."""

    name: str
    batch_size: int
    sequence_length: int
    max_abs_diff: float
    allclose: bool
    label_agreement: float


@dataclass(frozen=True, slots=True)
class ParityReport:
    """Aggregate PyTorch-vs-ONNX parity result."""

    onnx_path: str
    provider: str
    opset: int
    atol: float
    rtol: float
    cases: list[ParityCase] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True if every case is within tolerance and labels agree exactly."""
        return all(c.allclose and c.label_agreement == 1.0 for c in self.cases)

    @property
    def max_abs_diff(self) -> float:
        """Worst absolute logit deviation across all cases."""
        return max(c.max_abs_diff for c in self.cases)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable view including derived fields."""
        return {**asdict(self), "passed": self.passed, "max_abs_diff": self.max_abs_diff}

    def summary(self) -> str:
        """Human-readable multi-line summary."""
        header = (
            f"  {'case':<22} {'batch':>5} {'seq':>4} "
            f"{'max|diff|':>11} {'allclose':>8} {'labels':>7}"
        )
        lines = [
            f"Parity PyTorch FP32 vs ONNX ({self.provider}, opset {self.opset}) "
            f"atol={self.atol:g} rtol={self.rtol:g}: {'PASS' if self.passed else 'FAIL'}",
            header,
        ]
        for c in self.cases:
            lines.append(
                f"  {c.name:<22} {c.batch_size:>5} {c.sequence_length:>4} "
                f"{c.max_abs_diff:>11.3e} {c.allclose!s:>8} {c.label_agreement:>7.1%}"
            )
        lines.append(f"  overall max|diff| = {self.max_abs_diff:.3e}")
        return "\n".join(lines)


def _parity_batches(
    texts: Sequence[str],
) -> list[tuple[str, Sequence[str], PaddingStrategy]]:
    """Shapes to test, exercising both dynamic axes.

    * ``batch/max_length`` -- every text, fixed ``max_seq_length`` (benchmark shape).
    * ``half/longest`` -- first half of the texts padded to *their* longest, so
      batch size and sequence length both differ from the export trace. (With
      :data:`PARITY_TEXTS` the truncating sentence is last, so this stays short.)
    * ``single[i]`` -- each text alone: batch 1, a different length each time,
      and an all-ones attention mask (no padding at all).
    """
    half = texts[: max(1, len(texts) // 2)]
    cases: list[tuple[str, Sequence[str], PaddingStrategy]] = [
        ("batch/max_length", texts, "max_length"),
        ("half/longest", half, "longest"),
    ]
    cases += [(f"single[{i}]", [t], "longest") for i, t in enumerate(texts)]
    return cases


def verify_parity(
    cfg: PipelineConfig,
    onnx_path: Path | None = None,
    *,
    device: OrtDevice = "cpu",
    texts: Sequence[str] = PARITY_TEXTS,
    reference: TorchClassifier | None = None,
    raise_on_failure: bool = True,
) -> ParityReport:
    """Compare ONNX Runtime logits with the PyTorch FP32 CPU reference.

    Exercises the dynamic axes: a full batch padded to ``max_seq_length``, a
    smaller batch padded to its own longest sentence, and every sentence alone
    (batch 1, each with a different sequence length).

    Args:
        cfg: Pipeline configuration (tolerances, max length).
        onnx_path: ONNX file; defaults to ``cfg.paths.onnx_fp32``.
        device: ORT execution provider to test (``cpu`` or ``cuda``).
        texts: Input sentences.
        reference: Optional pre-loaded PyTorch FP32 CPU classifier to reuse.
        raise_on_failure: Raise :class:`ParityError` instead of returning a
            failing report.

    Returns:
        A :class:`ParityReport` with per-shape results.
    """
    onnx_path = onnx_path or cfg.paths.onnx_fp32
    if not onnx_path.exists():
        raise MissingArtifactError(f"{onnx_path} not found -- run `opt-pipeline export` first")
    seed_everything(cfg.seed)

    ref = reference or TorchClassifier(cfg, device="cpu", precision="fp32")
    if ref.device.type != "cpu" or ref.precision != "fp32":
        raise ValueError("the parity reference must be PyTorch FP32 on CPU")
    session = create_ort_session(onnx_path, device=device)
    tol = cfg.tolerances

    cases: list[ParityCase] = []
    for name, batch_texts, padding in _parity_batches(texts):
        enc = encode(ref.tokenizer, batch_texts, max_length=cfg.max_seq_length, padding=padding)
        expected = ref.predict_logits(enc)
        actual = run_onnx(session, enc)
        if actual.shape != expected.shape:
            raise ParityError(f"{name}: shape {actual.shape} != reference {expected.shape}")
        cases.append(
            ParityCase(
                name=name,
                batch_size=int(enc["input_ids"].shape[0]),
                sequence_length=int(enc["input_ids"].shape[1]),
                max_abs_diff=float(np.max(np.abs(actual - expected))),
                allclose=bool(
                    np.allclose(actual, expected, atol=tol.fp32_atol, rtol=tol.fp32_rtol)
                ),
                label_agreement=float(np.mean(actual.argmax(-1) == expected.argmax(-1))),
            )
        )

    report = ParityReport(
        onnx_path=onnx_path.as_posix(),
        provider=session.get_providers()[0],
        opset=onnx_opset(onnx_path),
        atol=tol.fp32_atol,
        rtol=tol.fp32_rtol,
        cases=cases,
    )
    log.info("\n%s", report.summary())
    if raise_on_failure and not report.passed:
        raise ParityError(report.summary())
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m opt_pipeline.export_onnx``: export with defaults, then verify on CPU."""
    del argv
    cfg = PipelineConfig()
    path = export_onnx(cfg)
    verify_parity(cfg, path, device="cpu")
    return 0


if __name__ == "__main__":
    sys.exit(main())
