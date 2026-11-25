"""Reproduce the INT8 settings sweep that justifies ``QuantizationConfig``'s defaults.

Quantizes the exported FP32 graph with six dynamic-INT8 settings, scores each on
the full SST-2 validation split against PyTorch FP32, takes a rough batch-8 p50
(20 iterations; *not* the benchmark methodology), and writes
``results/int8_sweep.json`` plus a Markdown table on stdout.

Usage (after ``python tasks.py export``)::

    uv run python scripts/int8_sweep.py
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

import numpy as np

from opt_pipeline.baseline import TorchClassifier, load_id2label
from opt_pipeline.benchmark import _cpu_name, make_batch
from opt_pipeline.config import PipelineConfig, QuantizationConfig
from opt_pipeline.evaluate import evaluate_runner, label_agreement, load_sst2
from opt_pipeline.quantize import quantize_int8
from opt_pipeline.runners import OnnxClassifier
from opt_pipeline.utils import atomic_write_json, utf8_stdio

SETTINGS: dict[str, QuantizationConfig] = {
    "matmul / per-channel": QuantizationConfig(True, False, ("MatMul",)),
    "matmul+gather / per-channel": QuantizationConfig(True, False, ("MatMul", "Gather")),
    "matmul / per-tensor": QuantizationConfig(False, False, ("MatMul",)),
    "matmul+gather / per-tensor": QuantizationConfig(False, False, ("MatMul", "Gather")),
    "matmul / per-channel / reduce_range": QuantizationConfig(True, True, ("MatMul",)),
    "matmul+gather / per-channel / reduce_range": QuantizationConfig(
        True, True, ("MatMul", "Gather")
    ),
}


def rough_p50_ms(runner: OnnxClassifier, batch: Any, iters: int = 20) -> float:
    """Median of ``iters`` timed calls after 5 warmups (a quick indicator only)."""
    for _ in range(5):
        runner.predict_logits(batch)
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        runner.predict_logits(batch)
        samples.append((time.perf_counter() - start) * 1e3)
    return float(np.median(samples))


def main() -> int:
    logging.getLogger("opt_pipeline").setLevel(logging.WARNING)
    cfg = PipelineConfig()
    split = load_sst2(cfg)
    reference = TorchClassifier(cfg, device="cpu")
    id2label = load_id2label(cfg)
    ref_metrics, ref_preds = evaluate_runner(reference, reference.tokenizer, id2label, split, cfg)
    batch = make_batch(reference.tokenizer, 8, cfg.max_seq_length)
    fp32_p50 = rough_p50_ms(OnnxClassifier(cfg.paths.onnx_fp32, device="cpu"), batch)

    rows: list[dict[str, Any]] = []
    out_dir = cfg.paths.artifacts_dir / "int8_sweep"
    for i, (label, quant) in enumerate(SETTINGS.items()):
        variant_cfg = cfg.with_overrides(quantization=quant)
        result = quantize_int8(variant_cfg, output=out_dir / f"int8_{i}.onnx")
        runner = OnnxClassifier(out_dir / f"int8_{i}.onnx", device="cpu")
        metrics, preds = evaluate_runner(runner, reference.tokenizer, id2label, split, cfg)
        rows.append(
            {
                "label": label,
                "settings": asdict(quant),
                "size_mb": result.size_mb,
                "accuracy": metrics.accuracy,
                "f1": metrics.f1,
                "accuracy_drop": ref_metrics.accuracy - metrics.accuracy,
                "label_agreement": label_agreement(preds, ref_preds),
                "rough_bs8_p50_ms": rough_p50_ms(runner, batch),
            }
        )
        print(f"done: {label}", file=sys.stderr, flush=True)
        del runner

    payload = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "cpu": _cpu_name(),
        "dataset": f"{cfg.dataset_name}/{cfg.dataset_config}:{cfg.dataset_split}",
        "n": len(split),
        "reference": {"accuracy": ref_metrics.accuracy, "f1": ref_metrics.f1},
        "ort_fp32_rough_bs8_p50_ms": fp32_p50,
        "rows": rows,
    }
    output = cfg.paths.results_dir / "int8_sweep.json"
    atomic_write_json(output, payload)

    with utf8_stdio():
        print(
            f"Reference PyTorch FP32: accuracy {ref_metrics.accuracy:.2%}, F1 {ref_metrics.f1:.4f}"
        )
        print(f"ORT FP32 rough bs8 p50: {fp32_p50:.0f} ms\n")
        print("| Setting | Size (MB) | Accuracy | Drop | Agreement with FP32 | bs8 p50 (ms) |")
        print("|---|---:|---:|---:|---:|---:|")
        for r in rows:
            print(
                f"| {r['label']} | {r['size_mb']:.1f} | {r['accuracy']:.2%} "
                f"| {r['accuracy_drop'] * 100:.2f} pp | {r['label_agreement']:.1%} "
                f"| {r['rough_bs8_p50_ms']:.0f} |"
            )
        print(f"\nWritten to {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
