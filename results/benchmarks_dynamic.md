_Generated 2026-11-25T09:25:58+00:00 from `results/benchmarks_dynamic.json` (schema v2). CPU: Intel(R) Core(TM) i5-10300H CPU @ 2.50GHz (4C/8T, 19.8 GB RAM, on AC power). GPU: NVIDIA GeForce GTX 1650 Ti (4096 MB, driver 591.59). torch 2.14.1+cu126, onnxruntime 1.26.0. 100 timed batches per cell, drawn (seed 42) from the validation sentences; each batch is padded to its longest sentence._

#### Workload (identical for every variant)

| Batch | Padded length: mean / p50 / max | Real tokens per sentence | Real share of padded tokens | Real share at fixed 128 |
|---:|---:|---:|---:|---:|
| 1 | 25.1 / 24 / 52 | 25.1 | 100.0% | 19.6% |
| 8 | 41.2 / 41 / 55 | 25.3 | 61.3% | 19.7% |
| 32 | 47.5 / 48 / 55 | 25.2 | 53.0% | 19.7% |

#### CPU latency & throughput (dynamic padding)

| Variant | Batch | p50 (ms) | p95 (ms) | p99 (ms) | Throughput (samples/s) | Speed-up vs PyTorch FP32 (p50) | Speed-up vs 128-token padding (p50) | Peak RAM Δ (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `pytorch-cpu-fp32` | 1 | 28.19 | 33.93 | 37.94 | 35.1 | 1.00× | 2.55× | 293 |
| `pytorch-cpu-fp32` | 8 | 160.98 | 205.07 | 225.40 | 49.2 | 1.00× | 3.36× | 318 |
| `pytorch-cpu-fp32` | 32 | 655.72 | 817.22 | 857.44 | 48.2 | 1.00× | 2.91× | 387 |
| `ort-cpu-fp32` | 1 | 15.82 | 22.30 | 24.56 | 62.6 | 1.78× | 3.63× | 281 |
| `ort-cpu-fp32` | 8 | 132.93 | 179.27 | 187.02 | 59.5 | 1.21× | 3.66× | 307 |
| `ort-cpu-fp32` | 32 | 672.35 | 813.69 | 845.21 | 47.2 | 0.98× | 3.07× | 391 |
| `ort-cpu-int8` | 1 | 8.34 | 13.96 | 22.68 | 107.1 | 3.38× | 4.64× | 92 |
| `ort-cpu-int8` | 8 | 91.54 | 120.20 | 130.43 | 86.5 | 1.76× | 3.60× | 109 |
| `ort-cpu-int8` | 32 | 445.99 | 525.14 | 534.73 | 71.5 | 1.47× | 3.13× | 168 |

#### GPU latency & throughput (dynamic padding)

| Variant | Batch | p50 (ms) | p95 (ms) | p99 (ms) | Throughput (samples/s) | Speed-up vs PyTorch FP32 (p50) | Speed-up vs 128-token padding (p50) | VRAM device Δ (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `pytorch-cuda-fp32` | 1 | 5.02 | 6.43 | 6.92 | 188.5 | 1.00× | 1.81× | 368 |
| `pytorch-cuda-fp32` | 8 | 20.42 | 22.22 | 22.41 | 399.7 | 1.00× | 2.51× | 372 |
| `pytorch-cuda-fp32` | 32 | 79.71 | 97.55 | 98.37 | 395.0 | 1.00× | 2.82× | 458 |
| `pytorch-cuda-fp16` | 1 | 8.66 | 13.69 | 14.11 | 107.4 | 0.58× | 4.02× | 222 |
| `pytorch-cuda-fp16` | 8 | 92.25 | 107.51 | 107.63 | 89.5 | 0.22× | 2.36× | 228 |
| `pytorch-cuda-fp16` | 32 | 332.84 | 397.49 | 397.65 | 93.0 | 0.24× | 2.59× | 264 |
| `ort-cuda-fp32` | 1 | 3.47 | 6.47 | 8.19 | 261.5 | 1.45× | 2.02× | 582 |
| `ort-cuda-fp32` | 8 | 16.84 | 23.44 | 23.72 | 458.9 | 1.21× | 2.56× | 582 |
| `ort-cuda-fp32` | 32 | 71.56 | 84.67 | 85.00 | 455.6 | 1.11× | 2.77× | 582 |
| `ort-cuda-fp16` | 1 | 13.09 | 14.97 | 15.33 | 75.9 | 0.38× | 2.21× | 324 |
| `ort-cuda-fp16` | 8 | 90.15 | 111.75 | 111.79 | 93.3 | 0.23× | 2.44× | 324 |
| `ort-cuda-fp16` | 32 | 346.64 | 381.51 | 386.97 | 91.8 | 0.23× | 2.53× | 326 |

The fixed-padding p50 comes from `results/benchmarks.json` (generated 2026-11-25T08:09:32+00:00), a separate run. The ratios therefore include drift between runs, which has reached about 8% on this laptop's CPU, so they are reliable for the multi-fold effects of padding but not for small differences.
Accuracy does not depend on the benchmark padding mode, because evaluation always pads dynamically. This run re-measured 91.06% for every variant.
