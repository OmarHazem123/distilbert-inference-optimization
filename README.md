# Production Inference: DistilBERT Optimization & Benchmarking

An end-to-end, reproducible pipeline that takes
[`distilbert-base-uncased-finetuned-sst-2-english`](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english)
from a PyTorch baseline to optimized ONNX Runtime variants (FP32, FP16, INT8). It
measures the speed-vs-accuracy trade-off of each variant **on real hardware**.

| | |
|---|---|
| **Model** | DistilBERT (66M params), binary sentiment (SST-2) |
| **Backends** | PyTorch (CPU/CUDA, FP32/FP16) and ONNX Runtime (CPU EP, CUDA EP) |
| **Optimizations** | ONNX export (opset 17), dynamic INT8 quantization, FP16 graph conversion |
| **Metrics** | p50/p95/p99 latency, throughput, peak RAM/VRAM, model size, SST-2 accuracy & F1 |

> **Status:** Phase 1 is complete (scaffold, config, PyTorch baseline, ONNX export + parity).
> Phase 2 adds quantization, benchmarking, profiling and the test suite.

---

## Reference hardware

| Component | Spec |
|---|---|
| GPU | NVIDIA GeForce GTX 1650 Ti, 4 GB VRAM, Turing `sm_75`, **no Tensor Cores** |
| OS | Windows 11 + Git Bash |
| Stack | Python 3.12, torch 2.14.1+cu126, onnxruntime-gpu 1.26.0 (CUDA 12 / cuDNN 9) |

GTX 16-series cards lack Tensor Cores, so FP16 GEMMs get none of the speed-up
that Tensor Cores give on RTX cards. Expect FP16 to be at best modestly faster
than FP32 on this GPU, and possibly slower. The benchmark measures this rather
than assuming it.

---

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and an NVIDIA driver supporting CUDA 12.x.
`make` is **not** required. Every task runs through `tasks.py` (stdlib only).

```bash
python tasks.py setup     # uv sync -> .venv with CUDA torch + onnxruntime-gpu
python tasks.py doctor    # verify torch CUDA build + a real ORT CUDA session
python tasks.py predict   # sanity-check the PyTorch baseline
python tasks.py export    # export ONNX (opset 17) + parity check on CPU and CUDA EPs
```

Extra CLI flags go after `--`, e.g. `python tasks.py export -- --parity-device cpu`.
If you do have `make`, `make export` and similar targets call the same tasks.

The CLI is also available directly:

```bash
uv run opt-pipeline --help
uv run opt-pipeline predict "What a wonderful film" --device cuda --precision fp16
```

### Why the CUDA wheels are pinned

On Windows, PyPI's `torch` is a **CPU-only** build, and any `pip install`/`uv add`
that re-resolves torch can silently replace the CUDA build. `pyproject.toml`
prevents this:

- `torch` is bound to the `https://download.pytorch.org/whl/cu126` index via
  `[tool.uv.sources]`, with `explicit = true` so no other package resolves from it.
- `onnxruntime-gpu` is capped at `<1.27`. **1.26.0 is the last PyPI release built for
  CUDA 12**; 1.27+ targets CUDA 13. It shares the CUDA 12 / cuDNN 9 DLLs bundled in
  the torch wheel, so no system-wide CUDA Toolkit or cuDNN install is needed.
- The CPU-only `onnxruntime` package is deliberately **not** a dependency. It
  installs the same `onnxruntime/` module and would shadow the GPU build.
- `create_ort_session` raises if the CUDA EP fails to load, instead of letting ORT
  silently fall back to CPU and mislabel CPU numbers as "GPU".

---

## Project layout

```
src/opt_pipeline/
  config.py          Frozen dataclass config: model (+ pinned revision), seq len, batch sizes,
                     warmup/iterations, tolerances, paths
  errors.py          PipelineError hierarchy (CLI prints these cleanly; bugs keep tracebacks)
  utils.py           Logging, atomic file writes (tmp + os.replace), file sizes
  baseline.py        PyTorch runner: deterministic seeding, CPU/CUDA, FP32/FP16, shared tokenisation
  export_onnx.py     ONNX export (opset 17, dynamic batch/sequence axes) + PyTorch parity check
  cli.py             `opt-pipeline` CLI (env, predict, export)
tasks.py             Cross-platform task runner (make replacement)
Makefile             Thin wrapper around tasks.py
```

Generated, git-ignored: `.venv/`, `models/` (HF model cache), `data/` (datasets),
`artifacts/` (ONNX graphs, profiler traces). `results/benchmarks.json` is tracked.

---

## Results

### ONNX export parity (PyTorch FP32 vs ONNX Runtime FP32)

Measured on the reference hardware with `python tasks.py export`. The reference is PyTorch
FP32 on CPU. Each EP runs 10 input shapes: batch sizes 1, 4 and 8, sequence lengths 3 to 128,
covering padded, unpadded and truncated inputs.

| ONNX Runtime EP | Opset | Shapes | Max abs logit diff | `np.allclose` (atol 1e-4, rtol 1e-3) | Label agreement |
|---|---|---|---|---|---|
| `CPUExecutionProvider` | 17 | 10 | 9.98e-06 | ✅ all | 100% |
| `CUDAExecutionProvider` | 17 | 10 | 4.35e-06 | ✅ all | 100% |

The exported FP32 graph is 255.6 MB. The full per-shape report is written to
`artifacts/onnx/model_fp32.parity.json`.

**Exporter note:** the export uses the TorchScript exporter (`dynamo=False`). It honours
`opset_version=17` and `dynamic_axes` natively. torch 2.14 marks it deprecated, and `uv.lock`
pins torch, so this is reproducible today. The dynamo exporter was also tested: it exports at
opset 18, auto-converts down to 17, and passes the same parity check (max diff 1.5e-05). On
Windows it needs `PYTHONUTF8=1`, because its progress output contains emoji that the cp1252
console cannot encode.

### Benchmarks

_Phase 2. The table will be generated from measured `results/benchmarks.json`._

---

## License

[MIT](LICENSE) © 2026 Omar Hazem
