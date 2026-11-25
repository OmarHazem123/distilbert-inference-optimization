"""Numerical parity of the exported / converted graphs against PyTorch.

All tests here load the real model (and most export it), so the module is
marked ``slow``; GPU-only tests are additionally marked ``gpu`` and skipped
without CUDA.
"""

from __future__ import annotations

from itertools import cycle, islice
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest

from opt_pipeline.baseline import MODEL_INPUT_NAMES, Encoded, TorchClassifier, encode
from opt_pipeline.config import PipelineConfig
from opt_pipeline.evaluate import LabeledSplit
from opt_pipeline.export_onnx import (
    PARITY_TEXTS,
    create_ort_session,
    export_onnx,
    onnx_opset,
    run_onnx,
    verify_parity,
)
from opt_pipeline.quantize import compare_with_reference
from opt_pipeline.utils import file_size_mb

from .conftest import requires_cuda

pytestmark = pytest.mark.slow

# (batch_size, sequence_length) pairs, none equal to the export trace shape (2, 9).
SHAPES = [(1, 8), (3, 16), (4, 37), (8, 128)]


def fixed_batch(clf: TorchClassifier, batch_size: int, seq_len: int) -> Encoded:
    """Exactly ``(batch_size, seq_len)`` tokens: pad/truncate to ``seq_len``."""
    texts = list(islice(cycle(PARITY_TEXTS), batch_size))
    return encode(clf.tokenizer, texts, max_length=seq_len, padding="max_length")


@pytest.fixture(scope="module")
def cpu_session(onnx_fp32: Path) -> ort.InferenceSession:
    return create_ort_session(onnx_fp32, device="cpu")


# --------------------------------------------------------------------------
# Graph structure
# --------------------------------------------------------------------------
class TestExportedGraph:
    def test_opset_is_17(self, onnx_fp32: Path) -> None:
        assert onnx_opset(onnx_fp32) == 17

    def test_io_names_and_symbolic_dynamic_axes(self, onnx_fp32: Path) -> None:
        graph = onnx.load(str(onnx_fp32), load_external_data=False).graph
        assert [i.name for i in graph.input] == list(MODEL_INPUT_NAMES)
        assert [o.name for o in graph.output] == ["logits"]
        for tensor in graph.input:
            dims = tensor.type.tensor_type.shape.dim
            assert len(dims) == 2
            # Symbolic, not fixed: dim_param set and no concrete dim_value.
            for dim in dims:
                assert dim.dim_param, f"{tensor.name} has a fixed dimension {dim}"
                assert not dim.HasField("dim_value")
        logits_dims = graph.output[0].type.tensor_type.shape.dim
        assert logits_dims[0].dim_param  # dynamic batch
        assert logits_dims[1].dim_value == 2  # NEGATIVE / POSITIVE


# --------------------------------------------------------------------------
# Output shapes and FP32 numerical equivalence
# --------------------------------------------------------------------------
@pytest.mark.parametrize(("batch_size", "seq_len"), SHAPES)
def test_output_shapes(
    reference: TorchClassifier, cpu_session: ort.InferenceSession, batch_size: int, seq_len: int
) -> None:
    batch = fixed_batch(reference, batch_size, seq_len)
    assert batch["input_ids"].shape == (batch_size, seq_len)
    assert reference.predict_logits(batch).shape == (batch_size, 2)
    assert run_onnx(cpu_session, batch).shape == (batch_size, 2)


@pytest.mark.parametrize(("batch_size", "seq_len"), SHAPES)
def test_fp32_numerically_equivalent(
    tmp_cfg: PipelineConfig,
    reference: TorchClassifier,
    cpu_session: ort.InferenceSession,
    batch_size: int,
    seq_len: int,
) -> None:
    batch = fixed_batch(reference, batch_size, seq_len)
    expected = reference.predict_logits(batch)
    actual = run_onnx(cpu_session, batch)
    tol = tmp_cfg.tolerances
    np.testing.assert_allclose(actual, expected, atol=tol.fp32_atol, rtol=tol.fp32_rtol)
    np.testing.assert_array_equal(actual.argmax(-1), expected.argmax(-1))


def test_verify_parity_report(
    tmp_cfg: PipelineConfig, onnx_fp32: Path, reference: TorchClassifier
) -> None:
    report = verify_parity(tmp_cfg, onnx_fp32, device="cpu", reference=reference)
    assert report.passed
    assert report.opset == 17
    assert report.provider == "CPUExecutionProvider"
    assert len(report.cases) == 2 + len(PARITY_TEXTS)
    assert report.max_abs_diff < tmp_cfg.tolerances.fp32_atol


# --------------------------------------------------------------------------
# Dynamic sequence length handling
# --------------------------------------------------------------------------
@pytest.mark.parametrize("seq_len", [2, 5, 64, 128, 256, 512])
def test_dynamic_sequence_lengths(
    tmp_cfg: PipelineConfig,
    reference: TorchClassifier,
    cpu_session: ort.InferenceSession,
    seq_len: int,
) -> None:
    """Sequence lengths far from the trace shape -- up to the 512-position
    limit, beyond the configured ``max_seq_length`` -- still match PyTorch."""
    batch = fixed_batch(reference, 2, seq_len)
    tol = tmp_cfg.tolerances
    np.testing.assert_allclose(
        run_onnx(cpu_session, batch),
        reference.predict_logits(batch),
        atol=tol.fp32_atol,
        rtol=tol.fp32_rtol,
    )


def test_padding_does_not_change_logits(
    reference: TorchClassifier, cpu_session: ort.InferenceSession
) -> None:
    """A sentence scored alone (no padding) equals the same sentence padded
    inside a longer batch -- i.e. the attention mask survived export."""
    text = "A gorgeous, witty, seductive movie."
    alone = encode(reference.tokenizer, [text], max_length=128, padding="longest")
    padded = encode(
        reference.tokenizer, [text, PARITY_TEXTS[-1]], max_length=128, padding="longest"
    )
    assert padded["input_ids"].shape[1] > alone["input_ids"].shape[1]
    np.testing.assert_allclose(
        run_onnx(cpu_session, padded)[0],
        run_onnx(cpu_session, alone)[0],
        atol=1e-4,
    )


# --------------------------------------------------------------------------
# INT8 (dynamic quantization, CPU)
# --------------------------------------------------------------------------
class TestInt8:
    def test_is_much_smaller(self, onnx_fp32: Path, onnx_int8: Path) -> None:
        assert file_size_mb(onnx_fp32) / file_size_mb(onnx_int8) >= 3.0

    def test_handles_dynamic_shapes(self, reference: TorchClassifier, onnx_int8: Path) -> None:
        session = create_ort_session(onnx_int8, device="cpu")
        for batch_size, seq_len in SHAPES:
            logits = run_onnx(session, fixed_batch(reference, batch_size, seq_len))
            assert logits.shape == (batch_size, 2)
            assert np.isfinite(logits).all()

    def test_prediction_agreement_within_tolerance(
        self,
        tmp_cfg: PipelineConfig,
        onnx_int8: Path,
        reference: TorchClassifier,
        sst2_slice: LabeledSplit,
    ) -> None:
        report = compare_with_reference(
            tmp_cfg,
            onnx_int8,
            precision="int8",
            device="cpu",
            sentences=sst2_slice.sentences,
            reference=reference,
        )
        assert report.label_agreement >= tmp_cfg.tolerances.min_label_agreement_int8, (
            report.summary()
        )


# --------------------------------------------------------------------------
# FP16 (CUDA)
# --------------------------------------------------------------------------
class TestFp16:
    def test_keeps_io_types(self, onnx_fp16: Path) -> None:
        graph = onnx.load(str(onnx_fp16), load_external_data=False).graph
        int64, float32 = onnx.TensorProto.INT64, onnx.TensorProto.FLOAT
        assert all(i.type.tensor_type.elem_type == int64 for i in graph.input)
        assert graph.output[0].type.tensor_type.elem_type == float32

    @pytest.mark.gpu
    @requires_cuda
    def test_ort_fp16_within_tolerance(
        self,
        tmp_cfg: PipelineConfig,
        onnx_fp16: Path,
        reference: TorchClassifier,
        sst2_slice: LabeledSplit,
    ) -> None:
        report = compare_with_reference(
            tmp_cfg,
            onnx_fp16,
            precision="fp16",
            device="cuda",
            sentences=sst2_slice.sentences,
            reference=reference,
        )
        assert report.provider == "CUDAExecutionProvider"
        assert report.passed, report.summary()

    @pytest.mark.gpu
    @requires_cuda
    def test_pytorch_fp16_within_tolerance(
        self, tmp_cfg: PipelineConfig, reference: TorchClassifier, sst2_slice: LabeledSplit
    ) -> None:
        from opt_pipeline.evaluate import batched_logits

        fp16 = TorchClassifier(tmp_cfg, device="cuda", precision="fp16")
        kwargs = {"batch_size": 32, "max_length": tmp_cfg.max_seq_length}
        expected = batched_logits(reference, reference.tokenizer, sst2_slice.sentences, **kwargs)
        actual = batched_logits(fp16, reference.tokenizer, sst2_slice.sentences, **kwargs)
        tol = tmp_cfg.tolerances
        assert np.abs(actual - expected).max() <= tol.fp16_max_abs_logit_diff
        agreement = np.mean(actual.argmax(-1) == expected.argmax(-1))
        assert agreement >= tol.min_label_agreement_fp16


# --------------------------------------------------------------------------
# CUDA execution provider
# --------------------------------------------------------------------------
@pytest.mark.gpu
@requires_cuda
class TestCudaProvider:
    def test_no_silent_cpu_fallback(self, onnx_fp32: Path) -> None:
        session = create_ort_session(onnx_fp32, device="cuda")
        assert session.get_providers()[0] == "CUDAExecutionProvider"

    def test_fp32_cuda_parity(
        self, tmp_cfg: PipelineConfig, onnx_fp32: Path, reference: TorchClassifier
    ) -> None:
        report = verify_parity(tmp_cfg, onnx_fp32, device="cuda", reference=reference)
        assert report.passed, report.summary()


# --------------------------------------------------------------------------
# Alternative exporter
# --------------------------------------------------------------------------
def test_dynamo_exporter_reaches_opset_17_with_parity(
    tmp_cfg: PipelineConfig, reference: TorchClassifier, tmp_path: Path
) -> None:
    path = export_onnx(tmp_cfg, tmp_path / "dynamo.onnx", exporter="dynamo")
    assert onnx_opset(path) == 17
    report = verify_parity(tmp_cfg, path, device="cpu", reference=reference)
    assert report.passed, report.summary()
