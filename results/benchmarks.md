_Generated 2026-11-25T08:09:32+00:00 from `results/benchmarks.json` (schema v1). CPU: Intel(R) Core(TM) i5-10300H CPU @ 2.50GHz (4C/8T, 19.8 GB RAM, on AC power). GPU: NVIDIA GeForce GTX 1650 Ti (4096 MB, driver 591.59). torch 2.14.1+cu126, onnxruntime 1.26.0. 100 timed iterations per cell, sequence length 128._

#### CPU latency & throughput

| Variant | Batch | p50 (ms) | p95 (ms) | p99 (ms) | Throughput (samples/s) | Speed-up vs PyTorch FP32 (p50) | Peak RAM Δ (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `pytorch-cpu-fp32` | 1 | 71.87 | 78.09 | 128.99 | 13.6 | 1.00× | 304 |
| `pytorch-cpu-fp32` | 8 | 540.36 | 772.51 | 821.42 | 14.0 | 1.00× | 353 |
| `pytorch-cpu-fp32` | 32 | 1910.81 | 2065.92 | 2147.36 | 16.7 | 1.00× | 500 |
| `ort-cpu-fp32` | 1 | 57.40 | 63.11 | 63.73 | 17.6 | 1.25× | 292 |
| `ort-cpu-fp32` | 8 | 486.89 | 515.77 | 533.69 | 16.4 | 1.11× | 334 |
| `ort-cpu-fp32` | 32 | 2065.43 | 2131.90 | 2203.03 | 15.5 | 0.93× | 489 |
| `ort-cpu-int8` | 1 | 38.68 | 40.76 | 42.54 | 26.1 | 1.86× | 103 |
| `ort-cpu-int8` | 8 | 329.19 | 344.24 | 352.45 | 24.3 | 1.64× | 141 |
| `ort-cpu-int8` | 32 | 1396.51 | 1475.96 | 1520.36 | 22.9 | 1.37× | 261 |

#### GPU latency & throughput

| Variant | Batch | p50 (ms) | p95 (ms) | p99 (ms) | Throughput (samples/s) | Speed-up vs PyTorch FP32 (p50) | VRAM device Δ (MB) | VRAM torch alloc / reserved (MB) |
|---|---:|---:|---:|---:|---:|---:|---: | ---:|
| `pytorch-cuda-fp32` | 1 | 9.11 | 10.35 | 10.89 | 107.3 | 1.00× | 368 | 294 / 326 |
| `pytorch-cuda-fp32` | 8 | 51.19 | 51.76 | 52.16 | 156.4 | 1.00× | 412 | 322 / 370 |
| `pytorch-cuda-fp32` | 32 | 224.53 | 225.46 | 226.05 | 142.6 | 1.00× | 532 | 421 / 490 |
| `pytorch-cuda-fp16` | 1 | 34.82 | 35.44 | 35.73 | 28.6 | 0.26× | 224 | 165 / 182 |
| `pytorch-cuda-fp16` | 8 | 217.38 | 217.89 | 218.26 | 36.8 | 0.24× | 244 | 181 / 202 |
| `pytorch-cuda-fp16` | 32 | 862.55 | 863.04 | 863.22 | 37.1 | 0.26× | 312 | 230 / 270 |
| `ort-cuda-fp32` | 1 | 7.02 | 7.82 | 7.98 | 140.0 | 1.30× | 590 | — |
| `ort-cuda-fp32` | 8 | 43.04 | 43.51 | 44.49 | 185.8 | 1.19× | 590 | — |
| `ort-cuda-fp32` | 32 | 198.19 | 199.39 | 199.56 | 161.5 | 1.13× | 590 | — |
| `ort-cuda-fp16` | 1 | 28.98 | 29.26 | 29.72 | 34.5 | 0.31× | 230 | — |
| `ort-cuda-fp16` | 8 | 220.11 | 220.40 | 221.04 | 36.3 | 0.23× | 358 | — |
| `ort-cuda-fp16` | 32 | 877.16 | 877.34 | 877.40 | 36.5 | 0.26× | 358 | — |

#### Accuracy & size (SST-2 validation)

| Variant | Provider | Model size (MB) | Accuracy | F1 | Δ accuracy vs PyTorch FP32 | Label agreement | Within tolerance |
|---|---|---:|---:|---:|---:|---:|:---:|
| `pytorch-cpu-fp32` | torch-cpu | 255.4 | 91.06% | 0.9137 | +0.00 pp | 100.00% | — |
| `ort-cpu-fp32` | CPUExecutionProvider | 255.6 | 91.06% | 0.9137 | +0.00 pp | 100.00% | — |
| `ort-cpu-int8` | CPUExecutionProvider | 64.5 | 91.06% | 0.9135 | +0.00 pp | 98.39% | ✅ |
| `pytorch-cuda-fp32` | torch-cuda | 255.4 | 91.06% | 0.9137 | +0.00 pp | 100.00% | — |
| `pytorch-cuda-fp16` | torch-cuda | 255.4 | 91.06% | 0.9137 | +0.00 pp | 100.00% | ✅ |
| `ort-cuda-fp32` | CUDAExecutionProvider | 255.6 | 91.06% | 0.9137 | +0.00 pp | 100.00% | — |
| `ort-cuda-fp16` | CUDAExecutionProvider | 127.9 | 91.06% | 0.9137 | +0.00 pp | 100.00% | ✅ |
