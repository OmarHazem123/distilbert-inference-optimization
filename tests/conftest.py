"""Shared pytest fixtures.

Expensive fixtures (model load, ONNX export, quantization) are session-scoped
and write only into a pytest temp directory -- tests never touch the real
``artifacts/`` or ``results/``. The Hugging Face model and dataset caches are
shared with the project (``models/``, ``data/``) so nothing is re-downloaded.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from opt_pipeline.baseline import TorchClassifier
from opt_pipeline.config import PathsConfig, PipelineConfig
from opt_pipeline.evaluate import LabeledSplit

REPO_ROOT = Path(__file__).resolve().parents[1]


def _cuda_available() -> bool:
    import torch

    return bool(torch.cuda.is_available())


requires_cuda = pytest.mark.skipif(not _cuda_available(), reason="CUDA GPU not available")
"""Combine with ``@pytest.mark.gpu`` on tests that need the CUDA EP / torch CUDA."""


@pytest.fixture(scope="session")
def tmp_cfg(tmp_path_factory: pytest.TempPathFactory) -> PipelineConfig:
    """Default config whose artifacts/results live in a session temp dir."""
    base = tmp_path_factory.mktemp("pipeline")
    return PipelineConfig(
        paths=PathsConfig(
            artifacts_dir=base / "artifacts",
            results_dir=base / "results",
            model_cache_dir=REPO_ROOT / "models",
            data_cache_dir=REPO_ROOT / "data",
        )
    )


@pytest.fixture(scope="session")
def reference(tmp_cfg: PipelineConfig) -> TorchClassifier:
    """PyTorch FP32 CPU classifier -- the numerical reference."""
    return TorchClassifier(tmp_cfg, device="cpu", precision="fp32")


@pytest.fixture(scope="session")
def onnx_fp32(tmp_cfg: PipelineConfig) -> Path:
    """FP32 graph exported with the default (TorchScript) exporter."""
    from opt_pipeline.export_onnx import export_onnx

    return export_onnx(tmp_cfg)


@pytest.fixture(scope="session")
def onnx_int8(tmp_cfg: PipelineConfig, onnx_fp32: Path) -> Path:
    """Dynamically quantized INT8 graph built from :func:`onnx_fp32`."""
    from opt_pipeline.quantize import quantize_int8

    return Path(quantize_int8(tmp_cfg, source=onnx_fp32).output)


@pytest.fixture(scope="session")
def onnx_fp16(tmp_cfg: PipelineConfig, onnx_fp32: Path) -> Path:
    """FP16 graph (keep_io_types) built from :func:`onnx_fp32`."""
    from opt_pipeline.quantize import convert_fp16

    return Path(convert_fp16(tmp_cfg, source=onnx_fp32).output)


@pytest.fixture(scope="session")
def sst2_slice(tmp_cfg: PipelineConfig) -> LabeledSplit:
    """First 200 SST-2 validation sentences: enough that a 95% agreement
    threshold tolerates a handful of flips rather than zero."""
    from opt_pipeline.evaluate import load_sst2

    return load_sst2(tmp_cfg, max_samples=200)
