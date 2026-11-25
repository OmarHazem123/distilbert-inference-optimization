"""Benchmark variants and a uniform runner interface over PyTorch and ONNX Runtime.

A *variant* is one (backend, device, precision) combination. Every runner
exposes the same host-to-host contract -- int64 NumPy inputs in, float32 NumPy
logits out -- so timing and evaluation code is backend-agnostic and the
measured boundary is identical for all variants (see ``benchmark.py``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import numpy.typing as npt

from opt_pipeline.baseline import TorchClassifier
from opt_pipeline.config import PipelineConfig
from opt_pipeline.errors import ConfigError, MissingArtifactError
from opt_pipeline.export_onnx import OrtDevice, create_ort_session, run_onnx

Backend = Literal["pytorch", "onnxruntime"]
VariantPrecision = Literal["fp32", "fp16", "int8"]


class Runner(Protocol):
    """Anything that maps an encoded batch to host float32 logits, synchronously."""

    @property
    def device_type(self) -> str:
        """``"cpu"`` or ``"cuda"``."""
        ...

    def predict_logits(
        self, encoded: Mapping[str, npt.NDArray[np.int64]]
    ) -> npt.NDArray[np.float32]:
        """Run one batch: host int64 arrays in, host float32 logits out."""
        ...

    def synchronize(self) -> None:
        """Block until all device work issued so far has completed."""
        ...


@dataclass(frozen=True, slots=True)
class VariantSpec:
    """One benchmarked configuration."""

    name: str
    backend: Backend
    device: OrtDevice
    precision: VariantPrecision
    description: str

    def artifact_path(self, cfg: PipelineConfig) -> Path:
        """File this variant loads (ONNX graph, or the HF safetensors checkpoint)."""
        if self.backend == "pytorch":
            return pytorch_checkpoint_path(cfg)
        paths = {
            "fp32": cfg.paths.onnx_fp32,
            "fp16": cfg.paths.onnx_fp16,
            "int8": cfg.paths.onnx_int8,
        }
        return paths[self.precision]

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable view."""
        return asdict(self)


VARIANTS: dict[str, VariantSpec] = {
    spec.name: spec
    for spec in (
        # --- CPU ---------------------------------------------------------
        VariantSpec(
            "pytorch-cpu-fp32", "pytorch", "cpu", "fp32", "PyTorch eager, FP32 (CPU baseline)"
        ),
        VariantSpec("ort-cpu-fp32", "onnxruntime", "cpu", "fp32", "ONNX Runtime CPU EP, FP32"),
        VariantSpec(
            "ort-cpu-int8", "onnxruntime", "cpu", "int8", "ONNX Runtime CPU EP, dynamic INT8"
        ),
        # --- GPU ---------------------------------------------------------
        VariantSpec(
            "pytorch-cuda-fp32", "pytorch", "cuda", "fp32", "PyTorch eager, FP32 (GPU baseline)"
        ),
        VariantSpec(
            "pytorch-cuda-fp16",
            "pytorch",
            "cuda",
            "fp16",
            "PyTorch eager, FP16 weights+activations",
        ),
        VariantSpec("ort-cuda-fp32", "onnxruntime", "cuda", "fp32", "ONNX Runtime CUDA EP, FP32"),
        VariantSpec(
            "ort-cuda-fp16", "onnxruntime", "cuda", "fp16", "ONNX Runtime CUDA EP, FP16 graph"
        ),
    )
}
"""All variants, in execution order (CPU first, then GPU).

Not benchmarked by design: dynamic INT8 on CUDA (dynamic quantization is a CPU
optimization in ONNX Runtime) and the FP16 graph on CPU (the CPU EP is an FP32
engine; FP16 there only adds casts)."""

REFERENCE_VARIANT = "pytorch-cpu-fp32"
"""Accuracy / label-agreement reference for every other variant."""

DEVICE_BASELINES: dict[str, str] = {"cpu": "pytorch-cpu-fp32", "cuda": "pytorch-cuda-fp32"}
"""Speed-up of each variant is reported relative to PyTorch FP32 on the same device."""


def get_variants(names: list[str] | None = None) -> list[VariantSpec]:
    """Resolve variant names (``None`` = all) preserving canonical order.

    Raises:
        ConfigError: On an unknown variant name.
    """
    if not names:
        return list(VARIANTS.values())
    unknown = [n for n in names if n not in VARIANTS]
    if unknown:
        raise ConfigError(f"unknown variant(s) {unknown}; choose from {list(VARIANTS)}")
    return [spec for name, spec in VARIANTS.items() if name in names]


def check_artifacts(specs: list[VariantSpec], cfg: PipelineConfig) -> None:
    """Fail fast, listing *every* missing ONNX artifact at once.

    Raises:
        MissingArtifactError: If any ONNX Runtime variant's graph is absent.
    """
    missing = [
        f"  {spec.name}: {spec.artifact_path(cfg)}"
        for spec in specs
        if spec.backend == "onnxruntime" and not spec.artifact_path(cfg).exists()
    ]
    if missing:
        raise MissingArtifactError(
            "missing ONNX artifacts -- run `opt-pipeline export` and `opt-pipeline quantize`:\n"
            + "\n".join(missing)
        )


def pytorch_checkpoint_path(cfg: PipelineConfig) -> Path:
    """Locate the cached ``model.safetensors`` for ``cfg.model_id@revision``.

    Raises:
        MissingArtifactError: If the checkpoint has not been downloaded yet.
    """
    from huggingface_hub import try_to_load_from_cache

    found = try_to_load_from_cache(
        cfg.model_id,
        "model.safetensors",
        cache_dir=str(cfg.paths.model_cache_dir),
        revision=cfg.revision,
    )
    if not isinstance(found, str):
        raise MissingArtifactError(
            f"model.safetensors for {cfg.model_id}@{cfg.revision[:8]} is not in "
            f"{cfg.paths.model_cache_dir}; run `opt-pipeline predict` or `export` to download it"
        )
    return Path(found)


class OnnxClassifier:
    """ONNX Runtime runner with the same contract as :class:`TorchClassifier`."""

    def __init__(
        self, path: Path, *, device: OrtDevice, intra_op_num_threads: int | None = None
    ) -> None:
        """Create a session; raises if the requested provider cannot be used."""
        if not path.exists():
            raise MissingArtifactError(
                f"{path} not found -- run `opt-pipeline export` / `opt-pipeline quantize` first"
            )
        self.path = path
        self._device: OrtDevice = device
        self.session = create_ort_session(
            path, device=device, intra_op_num_threads=intra_op_num_threads
        )

    @property
    def device_type(self) -> str:
        """``"cpu"`` or ``"cuda"``."""
        return self._device

    @property
    def provider(self) -> str:
        """Primary execution provider of the session."""
        return str(self.session.get_providers()[0])

    def predict_logits(
        self, encoded: Mapping[str, npt.NDArray[np.int64]]
    ) -> npt.NDArray[np.float32]:
        """Run one batch. ``session.run`` returns host arrays, so it is synchronous."""
        return run_onnx(self.session, encoded)

    def synchronize(self) -> None:
        """No-op: ``InferenceSession.run`` blocks until outputs are on the host."""


def build_runner(spec: VariantSpec, cfg: PipelineConfig) -> TorchClassifier | OnnxClassifier:
    """Instantiate the runner for ``spec``.

    Raises:
        MissingArtifactError: If an ONNX artifact has not been generated.
        DeviceUnavailableError / ProviderUnavailableError: If CUDA is unusable.
    """
    if spec.backend == "pytorch":
        if spec.precision == "int8":
            raise ConfigError("the PyTorch backend has no int8 variant")
        return TorchClassifier(cfg, device=spec.device, precision=spec.precision)
    return OnnxClassifier(spec.artifact_path(cfg), device=spec.device)
