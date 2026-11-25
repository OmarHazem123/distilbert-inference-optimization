"""Latency, throughput, memory, size and accuracy benchmark across all variants.

Methodology (also recorded in every ``benchmarks.json``):

* **Isolation** -- each variant runs in a fresh subprocess, so peak RAM, the
  CUDA context, allocator caches and thread pools never leak between variants,
  and one failing variant (e.g. CUDA OOM) cannot abort the whole run.
* **Timing boundary** -- host int64 NumPy arrays in, host float32 logits out,
  identically for every backend. PyTorch therefore pays its host->device copy
  and device->host readback exactly like ONNX Runtime does. Tokenisation is
  excluded (identical for all variants). CUDA timings bracket each iteration
  with ``torch.cuda.synchronize()``.
* **Shapes** -- ``cfg.padding`` selects the workload. ``max_length``: one
  fixed ``(batch, max_seq_length)`` batch, repeated (worst-case, identical
  work). ``dynamic``: ``cfg.benchmark_iters`` distinct batches of real SST-2
  validation sentences, each padded to its own longest sentence. Either way
  the workload is rebuilt deterministically in every worker and fingerprinted
  (sha256 of all token ids and masks); results whose variants were timed on
  different workloads fail validation.
* **Warmup** -- an untimed shape pass (one iteration per distinct padded
  length, longest first, at least ``MEMORY_PASS_ITERATIONS``) so per-shape
  kernel selection and allocator-arena growth happen outside the timed loop,
  then ``cfg.warmup_runs`` untimed iterations, then ``cfg.benchmark_iters``
  timed iterations with the garbage collector disabled.
* **Memory** -- measured in the untimed pass, never alongside the timed loop:
  sampled peak RSS (psutil); for CUDA, ``torch.cuda.max_memory_allocated`` /
  ``max_memory_reserved`` (PyTorch only -- ONNX Runtime allocations are
  invisible to it) and a device-level ``cudaMemGetInfo`` delta that captures
  *any* runtime's allocations, including ORT's power-of-two arena.
* **Accuracy** -- full SST-2 validation split with dynamic padding; label
  agreement and accuracy delta versus PyTorch FP32 on CPU.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from itertools import cycle, islice
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import psutil

from opt_pipeline import __version__
from opt_pipeline.baseline import (
    MODEL_INPUT_NAMES,
    Encoded,
    TorchClassifier,
    encode,
    load_id2label,
    load_tokenizer,
    seed_everything,
)
from opt_pipeline.config import PaddingMode, PipelineConfig
from opt_pipeline.errors import ConfigError, MissingArtifactError, PipelineError
from opt_pipeline.evaluate import LabeledSplit, evaluate_runner, label_agreement, load_sst2
from opt_pipeline.export_onnx import PARITY_TEXTS
from opt_pipeline.runners import (
    DEVICE_BASELINES,
    REFERENCE_VARIANT,
    Runner,
    VariantSpec,
    build_runner,
    check_artifacts,
    get_variants,
)
from opt_pipeline.utils import atomic_write_json, atomic_write_text, file_size_mb, get_logger

log = get_logger(__name__)

SCHEMA_VERSION = 2
SUPPORTED_SCHEMA_VERSIONS = (1, 2)
"""Version 2 adds a per-result ``workload`` block (padding, sequence-length
statistics, fingerprint). Version 1 files -- written before dynamic padding
existed -- still validate and render."""
_MB = 1024 * 1024
MEMORY_PASS_ITERATIONS = 3
"""Minimum untimed shape-pass iterations, run under the RSS sampler."""

README_MARKERS: dict[str, tuple[str, str]] = {
    "max_length": ("<!-- BENCHMARKS:START -->", "<!-- BENCHMARKS:END -->"),
    "dynamic": ("<!-- BENCHMARKS-DYNAMIC:START -->", "<!-- BENCHMARKS-DYNAMIC:END -->"),
}
"""README marker pair each padding mode's generated tables live between."""
README_START, README_END = README_MARKERS["max_length"]


# ==========================================================================
# Latency
# ==========================================================================
@dataclass(frozen=True, slots=True)
class LatencyStats:
    """Summary statistics of per-iteration latencies, in milliseconds."""

    mean: float
    std: float
    min: float
    p50: float
    p95: float
    p99: float
    max: float

    @classmethod
    def from_samples(cls, samples_ms: npt.ArrayLike) -> LatencyStats:
        """Compute stats; percentiles use NumPy's default linear interpolation."""
        s = np.asarray(samples_ms, dtype=np.float64)
        if s.ndim != 1 or s.size == 0:
            raise ValueError("need a non-empty 1-D array of latency samples")
        p50, p95, p99 = np.percentile(s, [50, 95, 99])
        return cls(
            mean=float(s.mean()),
            std=float(s.std(ddof=1)) if s.size > 1 else 0.0,
            min=float(s.min()),
            p50=float(p50),
            p95=float(p95),
            p99=float(p99),
            max=float(s.max()),
        )

    def to_dict(self) -> dict[str, float]:
        """JSON-serialisable view."""
        return asdict(self)


def time_callable(
    fn: Callable[[], object],
    *,
    iterations: int,
    warmup: int = 0,
    sync: Callable[[], None] | None = None,
) -> npt.NDArray[np.float64]:
    """Time ``fn`` ``iterations`` times after ``warmup`` untimed calls.

    Each sample spans ``fn()`` plus ``sync()`` (e.g. ``torch.cuda.synchronize``)
    so asynchronous device work is fully accounted for. The garbage collector
    is disabled during the timed loop to keep collection pauses out of p99.

    Returns:
        Per-iteration latencies in milliseconds.
    """
    if iterations <= 0:
        raise ValueError("iterations must be > 0")
    for _ in range(warmup):
        fn()
        if sync is not None:
            sync()
    samples = np.empty(iterations, dtype=np.float64)
    gc.collect()
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for i in range(iterations):
            start = time.perf_counter()
            fn()
            if sync is not None:
                sync()
            samples[i] = (time.perf_counter() - start) * 1e3
    finally:
        if gc_was_enabled:
            gc.enable()
    return samples


# ==========================================================================
# Memory
# ==========================================================================
def rss_mb() -> float:
    """Current resident set size of this process in MiB."""
    return psutil.Process().memory_info().rss / _MB


def process_peak_rss_mb() -> float | None:
    """OS-reported lifetime peak RSS of this process (``None`` if unavailable)."""
    if sys.platform == "win32":
        peak = getattr(psutil.Process().memory_info(), "peak_wset", None)  # bytes
        return None if peak is None else float(peak) / _MB
    import resource

    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB, macOS bytes.
    return float(maxrss) / (_MB if sys.platform == "darwin" else 1024)


class RssSampler:
    """Background thread recording the peak RSS while a block runs.

    Only used in untimed passes: the sampler thread would otherwise compete
    with the measured work for cores and the GIL.
    """

    def __init__(self, interval_s: float = 0.002) -> None:
        self.interval_s = interval_s
        self.peak_mb = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="rss-sampler", daemon=True)
        self._process = psutil.Process()

    def _sample(self) -> None:
        self.peak_mb = max(self.peak_mb, self._process.memory_info().rss / _MB)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval_s)

    def __enter__(self) -> RssSampler:
        self._sample()
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()
        self._sample()


class CudaMemoryProbe:
    """Device-level and PyTorch-allocator VRAM measurements.

    The constructor creates this process's CUDA context *before* reading the
    baseline, so the context itself is never charged to the model. ONNX
    Runtime's CUDA EP uses the same primary context through the runtime API.
    """

    def __init__(self) -> None:
        import torch

        self._torch = torch
        torch.cuda.init()  # type: ignore[no-untyped-call]
        self.baseline_used_mb = self.used_mb()

    def used_mb(self) -> float:
        """Device memory in use by *all* processes (``total - free``), in MiB."""
        free, total = self._torch.cuda.mem_get_info()
        return float(total - free) / _MB

    def delta_mb(self) -> float:
        """Device memory used now, relative to the post-context baseline."""
        return self.used_mb() - self.baseline_used_mb

    def reset_torch_peaks(self) -> None:
        """Reset PyTorch's caching-allocator peak counters."""
        self._torch.cuda.reset_peak_memory_stats()

    def torch_peaks_mb(self) -> tuple[float, float]:
        """``(max_memory_allocated, max_memory_reserved)`` in MiB."""
        return (
            self._torch.cuda.max_memory_allocated() / _MB,
            self._torch.cuda.max_memory_reserved() / _MB,
        )


# ==========================================================================
# Per-batch measurement
# ==========================================================================
@dataclass(frozen=True, slots=True)
class BatchResult:
    """All measurements for one (variant, batch size).

    ``sequence_length`` is the *longest* padded length that was timed (equal to
    ``max_seq_length`` for fixed padding); ``workload`` holds the full
    sequence-length statistics and the workload fingerprint.
    """

    batch_size: int
    sequence_length: int
    warmup: int
    iterations: int
    latency_ms: LatencyStats
    throughput_samples_per_s: float
    ram_peak_mb: float
    ram_delta_mb: float
    vram_torch_allocated_peak_mb: float | None
    vram_torch_reserved_peak_mb: float | None
    vram_device_delta_mb: float | None
    workload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable view."""
        out = asdict(self)
        out["latency_ms"] = self.latency_ms.to_dict()
        return out


# ==========================================================================
# Workloads (what is timed)
# ==========================================================================
@dataclass(frozen=True, slots=True)
class Workload:
    """The exact pre-tokenised batches one (variant, batch size) cell is timed on.

    Timed iteration ``i`` runs ``batches[i % len(batches)]``: fixed padding has
    one batch repeated; dynamic padding has one distinct batch per iteration.
    """

    padding: PaddingMode
    batches: tuple[Encoded, ...]

    def __post_init__(self) -> None:
        if not self.batches:
            raise ValueError("a workload needs at least one batch")
        sizes = {int(b["input_ids"].shape[0]) for b in self.batches}
        if len(sizes) != 1:
            raise ValueError(f"all batches must share one batch size, got {sorted(sizes)}")

    @property
    def batch_size(self) -> int:
        """Rows per batch."""
        return int(self.batches[0]["input_ids"].shape[0])

    def shape_pass(self) -> list[Encoded]:
        """One batch per distinct padded length, longest first.

        Running these untimed before warmup moves per-shape costs (oneDNN
        primitive creation, cuBLAS heuristics, allocator-arena growth) out of
        the timed loop, and the longest batch sets the memory peak.
        """
        by_length: dict[int, Encoded] = {}
        for batch in self.batches:
            by_length.setdefault(int(batch["input_ids"].shape[1]), batch)
        return [by_length[n] for n in sorted(by_length, reverse=True)]

    def sha256(self) -> str:
        """Fingerprint of every batch's shapes, token ids and attention mask."""
        digest = hashlib.sha256()
        for batch in self.batches:
            for name in MODEL_INPUT_NAMES:
                array = np.ascontiguousarray(batch[name], dtype=np.int64)
                digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
                digest.update(array.tobytes())
        return digest.hexdigest()

    def summary(self, iterations: int) -> dict[str, Any]:
        """Statistics of the ``iterations`` batches the timed loop runs."""
        timed = [self.batches[i % len(self.batches)] for i in range(iterations)]
        lengths = np.array([b["input_ids"].shape[1] for b in timed], dtype=np.float64)
        real = sum(int(b["attention_mask"].sum()) for b in timed)
        padded = sum(int(b["attention_mask"].size) for b in timed)
        return {
            "padding": self.padding,
            "distinct_batches": len(self.batches),
            "distinct_shapes": len(set(lengths.tolist())),
            "seq_len_mean": float(lengths.mean()),
            "seq_len_p50": float(np.percentile(lengths, 50)),
            "seq_len_max": int(lengths.max()),
            "mean_real_tokens": real / (len(timed) * self.batch_size),
            "real_token_fraction": real / padded,
            "sha256": self.sha256(),
        }


def make_batch(
    tokenizer: Any, batch_size: int, max_length: int, texts: Sequence[str] = PARITY_TEXTS
) -> Encoded:
    """Fixed-shape ``(batch_size, max_length)`` batch cycling over ``texts``."""
    chosen = list(islice(cycle(texts), batch_size))
    return encode(tokenizer, chosen, max_length=max_length, padding="max_length")


def dynamic_workload(
    tokenizer: Any,
    sentences: Sequence[str],
    *,
    batch_size: int,
    n_batches: int,
    max_length: int,
    seed: int,
) -> Workload:
    """``n_batches`` batches of real sentences, each padded to its longest member.

    Sentences are taken in order from successive seeded permutations of
    ``sentences``; when fewer than ``batch_size`` remain, a fresh permutation
    starts, so no sentence repeats within a batch. The draw depends only on
    ``(seed, batch_size)``: every variant -- each in its own worker process --
    rebuilds the identical sequence.

    Raises:
        ConfigError: If ``batch_size`` exceeds the number of sentences.
    """
    if batch_size > len(sentences):
        raise ConfigError(f"batch size {batch_size} exceeds the {len(sentences)} sentences")
    if n_batches <= 0:
        raise ValueError("n_batches must be > 0")
    rng = np.random.default_rng([seed, batch_size])
    order, pos = rng.permutation(len(sentences)), 0
    batches: list[Encoded] = []
    while len(batches) < n_batches:
        if pos + batch_size > len(order):
            order, pos = rng.permutation(len(sentences)), 0
        chosen = [sentences[int(i)] for i in order[pos : pos + batch_size]]
        pos += batch_size
        batches.append(encode(tokenizer, chosen, max_length=max_length, padding="longest"))
    return Workload("dynamic", tuple(batches))


def build_workload(
    cfg: PipelineConfig, tokenizer: Any, batch_size: int, sentences: Sequence[str] | None
) -> Workload:
    """The workload ``cfg.padding`` prescribes for one batch size.

    Raises:
        ConfigError: If dynamic padding is requested without ``sentences``.
    """
    if cfg.padding == "max_length":
        return Workload("max_length", (make_batch(tokenizer, batch_size, cfg.max_seq_length),))
    if not sentences:
        raise ConfigError("dynamic padding needs the evaluation sentences")
    return dynamic_workload(
        tokenizer,
        sentences,
        batch_size=batch_size,
        n_batches=cfg.benchmark_iters,
        max_length=cfg.max_seq_length,
        seed=cfg.seed,
    )


# ==========================================================================
# Per-cell measurement
# ==========================================================================
def measure_workload(
    runner: Runner,
    workload: Workload,
    *,
    warmup: int,
    iterations: int,
    ram_baseline_mb: float,
    cuda_probe: CudaMemoryProbe | None = None,
    torch_allocator: bool = False,
) -> BatchResult:
    """Shape/memory pass (untimed, sampled), warmup, then the timed latency loop.

    Args:
        runner: Any :class:`~opt_pipeline.runners.Runner`.
        workload: Batches to run; timed iteration ``i`` uses batch
            ``i % len(workload.batches)``.
        warmup: Untimed iterations after the shape pass.
        iterations: Timed iterations.
        ram_baseline_mb: RSS before the model was loaded.
        cuda_probe: VRAM probe for CUDA variants.
        torch_allocator: Whether the runner allocates through PyTorch (so the
            allocator peak counters are meaningful).
    """
    shapes = workload.shape_pass()
    shape_iterations = max(MEMORY_PASS_ITERATIONS, len(shapes))
    if cuda_probe is not None and torch_allocator:
        cuda_probe.reset_torch_peaks()
    with RssSampler() as sampler:
        for i in range(shape_iterations):
            runner.predict_logits(shapes[i % len(shapes)])
            runner.synchronize()

    torch_alloc = torch_reserved = device_delta = None
    if cuda_probe is not None:
        device_delta = cuda_probe.delta_mb()
        if torch_allocator:
            torch_alloc, torch_reserved = cuda_probe.torch_peaks_mb()

    warm = cycle(workload.batches)
    for _ in range(warmup):
        runner.predict_logits(next(warm))
        runner.synchronize()

    timed = cycle(workload.batches)  # fresh iterator: timing always starts at batch 0

    def step() -> None:
        runner.predict_logits(next(timed))

    samples = time_callable(step, iterations=iterations, sync=runner.synchronize)
    stats = LatencyStats.from_samples(samples)
    summary = workload.summary(iterations)
    return BatchResult(
        batch_size=workload.batch_size,
        sequence_length=summary["seq_len_max"],
        warmup=warmup + shape_iterations,
        iterations=iterations,
        latency_ms=stats,
        throughput_samples_per_s=workload.batch_size * 1e3 / stats.mean,
        ram_peak_mb=sampler.peak_mb,
        ram_delta_mb=sampler.peak_mb - ram_baseline_mb,
        vram_torch_allocated_peak_mb=torch_alloc,
        vram_torch_reserved_peak_mb=torch_reserved,
        vram_device_delta_mb=device_delta,
        workload=summary,
    )


# ==========================================================================
# One variant (runs inside an isolated worker process)
# ==========================================================================
def run_variant(
    spec: VariantSpec,
    cfg: PipelineConfig,
    *,
    evaluate: bool = True,
    split: LabeledSplit | None = None,
) -> dict[str, Any]:
    """Benchmark one variant across ``cfg.batch_sizes`` and optionally evaluate it.

    Args:
        spec: Variant to run.
        cfg: Pipeline configuration (``cfg.padding`` selects the workload).
        evaluate: Also compute SST-2 accuracy / F1.
        split: Pre-loaded evaluation split (in-process mode); loaded on demand
            otherwise.

    Returns a JSON-serialisable dict; ``predictions`` (dataset label space) is
    included for the parent to compute cross-variant agreement and is
    stripped before results are written.
    """
    import torch

    # Benchmarks favour the fastest kernels; determinism only pins reduction
    # order, which is irrelevant to timing and may select slower algorithms.
    seed_everything(cfg.seed, deterministic=False)
    cuda_probe = CudaMemoryProbe() if spec.device == "cuda" else None

    # Dynamic workloads draw from the *whole* split, independent of any
    # evaluation cap. It is dataset memory, so it is loaded before the RAM
    # baseline and not charged to the model.
    workload_split: LabeledSplit | None = None
    if cfg.padding == "dynamic":
        workload_split = load_sst2(replace(cfg, eval_max_samples=None))
    gc.collect()
    ram_baseline = rss_mb()

    load_start = time.perf_counter()
    runner = build_runner(spec, cfg)
    load_seconds = time.perf_counter() - load_start
    if isinstance(runner, TorchClassifier):
        is_torch, tokenizer, id2label = True, runner.tokenizer, runner.id2label
        provider = f"torch-{spec.device}"
    else:
        is_torch, tokenizer, id2label = False, load_tokenizer(cfg), load_id2label(cfg)
        provider = runner.provider
    vram_after_load = cuda_probe.delta_mb() if cuda_probe is not None else None
    log.info("[%s] loaded in %.1fs (provider=%s)", spec.name, load_seconds, provider)

    sentences = workload_split.sentences if workload_split is not None else None
    results: list[dict[str, Any]] = []
    for batch_size in cfg.batch_sizes:
        workload = build_workload(cfg, tokenizer, batch_size, sentences)
        res = measure_workload(
            runner,
            workload,
            warmup=cfg.warmup_runs,
            iterations=cfg.benchmark_iters,
            ram_baseline_mb=ram_baseline,
            cuda_probe=cuda_probe,
            torch_allocator=is_torch,
        )
        log.info(
            "[%s] bs=%-3d len=%5.1f  p50=%8.2f ms  p95=%8.2f ms  p99=%8.2f ms  %8.1f samples/s",
            spec.name,
            batch_size,
            res.workload["seq_len_mean"],
            res.latency_ms.p50,
            res.latency_ms.p95,
            res.latency_ms.p99,
            res.throughput_samples_per_s,
        )
        results.append(res.to_dict())
        del workload

    process_peak = process_peak_rss_mb()  # before evaluation adds dataset memory

    eval_block: dict[str, Any] | None = None
    predictions: list[int] | None = None
    if evaluate:
        if split is None and workload_split is not None:
            cap = cfg.eval_max_samples
            split = workload_split if cap is None else workload_split.head(cap)
        split = split or load_sst2(cfg)
        eval_start = time.perf_counter()
        metrics, preds = evaluate_runner(runner, tokenizer, id2label, split, cfg)
        eval_block = {**metrics.to_dict(), "seconds": time.perf_counter() - eval_start}
        predictions = [int(p) for p in preds]
        log.info("[%s] SST-2 accuracy=%.4f f1=%.4f", spec.name, metrics.accuracy, metrics.f1)

    artifact = spec.artifact_path(cfg)
    return {
        **spec.to_dict(),
        "status": "ok",
        "error": None,
        "provider": provider,
        "artifact": artifact.as_posix(),
        "model_size_mb": file_size_mb(artifact),
        "load_seconds": load_seconds,
        "threads": {
            "torch_intra_op": torch.get_num_threads(),
            "ort_intra_op": "ORT default (one per physical core)" if not is_torch else None,
        },
        "ram_baseline_mb": ram_baseline,
        "process_peak_ram_mb": process_peak,
        "vram_after_load_delta_mb": vram_after_load,
        "results": results,
        "eval": eval_block,
        "predictions": predictions,
    }


def _failed_entry(spec: VariantSpec, status: str, message: str) -> dict[str, Any]:
    return {**spec.to_dict(), "status": status, "error": message, "results": [], "eval": None}


# ==========================================================================
# Isolation (subprocess per variant)
# ==========================================================================
def _child_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"  # dynamo exporter / HF progress output is not cp1252-safe
    env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    return env


def _tail(path: Path, lines: int = 25) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
    except OSError:
        return "<no log>"


def run_variant_isolated(
    spec: VariantSpec, cfg: PipelineConfig, *, evaluate: bool, log_path: Path
) -> dict[str, Any]:
    """Run :func:`run_variant` in a fresh interpreter; never raises for worker failures."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = Path(tmp) / "config.json"
        out_path = Path(tmp) / "result.json"
        atomic_write_json(cfg_path, cfg.to_dict())
        cmd = [
            sys.executable,
            "-m",
            "opt_pipeline.benchmark",
            "--worker",
            spec.name,
            "--config",
            str(cfg_path),
            "--output",
            str(out_path),
        ]
        if not evaluate:
            cmd.append("--no-eval")
        try:
            with log_path.open("w", encoding="utf-8") as log_file:
                proc = subprocess.run(
                    cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    env=_child_env(),
                    timeout=cfg.worker_timeout_s,
                    check=False,
                )
        except subprocess.TimeoutExpired:
            return _failed_entry(
                spec,
                "error",
                f"worker timed out after {cfg.worker_timeout_s}s\n{_tail(log_path)}",
            )
        if proc.returncode != 0 or not out_path.exists():
            return _failed_entry(
                spec, "error", f"worker exited with code {proc.returncode}\n{_tail(log_path)}"
            )
        result: dict[str, Any] = json.loads(out_path.read_text(encoding="utf-8"))
        return result


def _worker_main(argv: Sequence[str] | None = None) -> int:
    """Entry point of ``python -m opt_pipeline.benchmark --worker NAME ...``."""
    parser = argparse.ArgumentParser(description="isolated benchmark worker (internal)")
    parser.add_argument("--worker", required=True, metavar="VARIANT")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--no-eval", action="store_true")
    args = parser.parse_args(argv)

    cfg = PipelineConfig.from_dict(json.loads(args.config.read_text(encoding="utf-8")))
    (spec,) = get_variants([args.worker])
    try:
        result = run_variant(spec, cfg, evaluate=not args.no_eval)
    except PipelineError as exc:  # expected, actionable: report cleanly
        result = _failed_entry(spec, "error", f"{type(exc).__name__}: {exc}")
    except Exception:  # unexpected: keep the traceback for the log tail
        traceback.print_exc()
        return 1
    atomic_write_json(args.output, result)
    return 0


# ==========================================================================
# Orchestration
# ==========================================================================
def run_benchmarks(
    cfg: PipelineConfig,
    variant_names: list[str] | None = None,
    *,
    isolate: bool = True,
    evaluate: bool = True,
    output: Path | None = None,
) -> dict[str, Any]:
    """Benchmark the selected variants and write ``benchmarks.json`` atomically.

    Args:
        cfg: Pipeline configuration.
        variant_names: Subset of :data:`~opt_pipeline.runners.VARIANTS`
            (``None`` = all).
        isolate: Run each variant in its own subprocess (recommended). The
            in-process mode exists for fast tests.
        evaluate: Also compute SST-2 accuracy / F1.
        output: Destination; defaults to the canonical file for
            ``cfg.padding`` (``benchmarks.json`` / ``benchmarks_dynamic.json``).

    Returns:
        The validated results payload.
    """
    import torch

    specs = get_variants(variant_names)
    check_artifacts(specs, cfg)
    output = output or cfg.paths.benchmark_results_json(cfg.padding)
    environment = collect_environment()
    # Warm the dataset cache once, before the workers (which also need it for
    # dynamic workloads even when evaluation is off).
    split = load_sst2(cfg) if evaluate or cfg.padding == "dynamic" else None

    entries: list[dict[str, Any]] = []
    started = time.perf_counter()
    for order, spec in enumerate(specs):
        if spec.device == "cuda" and not torch.cuda.is_available():
            entry = _failed_entry(spec, "skipped", "CUDA is not available on this machine")
        elif isolate:
            log.info("Running %s in an isolated worker ...", spec.name)
            log_path = cfg.paths.benchmark_logs_dir / f"{spec.name}.log"
            entry = run_variant_isolated(spec, cfg, evaluate=evaluate, log_path=log_path)
        else:
            try:
                entry = run_variant(spec, cfg, evaluate=evaluate, split=split)
            except PipelineError as exc:
                entry = _failed_entry(spec, "error", f"{type(exc).__name__}: {exc}")
        entry["order"] = order
        if entry["status"] != "ok":
            log.warning("[%s] %s: %s", spec.name, entry["status"], entry["error"])
        entries.append(entry)

    attach_comparisons(entries, cfg)
    padding_reference: dict[str, Any] | None = None
    if cfg.padding == "dynamic":
        reference = load_padding_reference(cfg)
        if reference is not None:
            attach_padding_comparison(entries, reference)
            padding_reference = {
                "path": cfg.paths.benchmarks_json.as_posix(),
                "generated_at": reference["generated_at"],
                "note": "fixed max_length padding, measured in a separate run",
            }
    payload = build_payload(
        entries,
        cfg,
        environment=environment,
        isolate=isolate,
        evaluate=evaluate,
        duration_s=time.perf_counter() - started,
        padding_reference=padding_reference,
    )
    atomic_write_json(output, payload)
    log.info("Wrote %s", output)
    return payload


def build_payload(
    entries: list[dict[str, Any]],
    cfg: PipelineConfig,
    *,
    environment: dict[str, Any],
    isolate: bool,
    evaluate: bool,
    duration_s: float,
    padding_reference: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble and validate the top-level results document.

    Raises:
        PipelineError: If the assembled payload violates the schema.
    """
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "duration_s": round(duration_s, 1),
        "opt_pipeline_version": __version__,
        "environment": environment,
        "methodology": methodology(cfg, isolate=isolate, evaluate=evaluate),
        "config": cfg.to_dict(),
        "variants": entries,
    }
    if padding_reference is not None:
        payload["padding_reference"] = padding_reference
    errors = validate_results(payload)
    if errors:
        raise PipelineError("benchmark payload failed schema validation:\n" + "\n".join(errors))
    return payload


def methodology(cfg: PipelineConfig, *, isolate: bool, evaluate: bool) -> dict[str, Any]:
    """Machine-readable description of how the numbers were produced."""
    q = cfg.quantization
    if cfg.padding == "max_length":
        padding = f"max_length: one ({cfg.max_seq_length}-token) batch per batch size, repeated"
        percentiles = "spread reflects system jitter only (every iteration has the same shape)"
    else:
        padding = (
            f"dynamic: {cfg.benchmark_iters} distinct batches per batch size of real "
            f"{cfg.dataset_name}/{cfg.dataset_config} {cfg.dataset_split} sentences "
            f"(seeded draw, seed={cfg.seed}), each padded to its longest sentence "
            f"(truncation at {cfg.max_seq_length})"
        )
        percentiles = (
            "p95/p99 include real variation in padded length across batches, not only jitter"
        )
    return {
        "timing_boundary": "host int64 numpy inputs -> host float32 logits (tokenisation excluded)",
        "cuda_sync": "torch.cuda.synchronize() after every PyTorch CUDA iteration",
        "padding": padding,
        "percentiles": percentiles,
        "workload_identity": (
            "sha256 of every timed batch's token ids and mask; must match across variants"
        ),
        "warmup": (
            f"untimed shape pass (one iteration per distinct padded length, longest first, "
            f"min {MEMORY_PASS_ITERATIONS}) + {cfg.warmup_runs} untimed iterations"
        ),
        "iterations": cfg.benchmark_iters,
        "gc_disabled_during_timing": True,
        "isolation": "one subprocess per variant" if isolate else "in-process (no isolation)",
        "seeding": f"seed={cfg.seed}, deterministic algorithms off for benchmarking",
        "ram": (
            "peak RSS sampled every 2 ms during the untimed shape pass; delta vs pre-load RSS "
            "(includes the cell's pre-tokenised inputs)"
        ),
        "vram_torch": "torch.cuda.max_memory_allocated/reserved (PyTorch variants only)",
        "vram_device": "cudaMemGetInfo used-memory delta vs post-context baseline (all CUDA)",
        "evaluation": (
            f"{cfg.dataset_name}/{cfg.dataset_config} {cfg.dataset_split}, dynamic padding, "
            f"batch {cfg.eval_batch_size}"
            if evaluate
            else "skipped"
        ),
        "int8_settings": {
            "per_channel": q.int8_per_channel,
            "reduce_range": q.int8_reduce_range,
            "op_types": list(q.int8_op_types),
        },
    }


def attach_comparisons(entries: list[dict[str, Any]], cfg: PipelineConfig) -> None:
    """Add agreement / accuracy deltas / speed-ups in place and drop raw predictions."""
    by_name = {e["name"]: e for e in entries if e["status"] == "ok"}
    ref = by_name.get(REFERENCE_VARIANT)
    ref_preds = ref.get("predictions") if ref else None
    ref_acc = ref["eval"]["accuracy"] if ref and ref.get("eval") else None
    tol = cfg.tolerances

    for entry in entries:
        preds = entry.pop("predictions", None)
        if entry["status"] != "ok":
            continue
        if entry.get("eval") is not None and ref_preds is not None and preds is not None:
            agreement = label_agreement(preds, ref_preds)
            drop = float(ref_acc - entry["eval"]["accuracy"]) if ref_acc is not None else None
            entry["eval"]["label_agreement_vs_reference"] = agreement
            entry["eval"]["accuracy_drop_vs_reference"] = drop
            limits = {
                "int8": (tol.min_label_agreement_int8, tol.max_accuracy_drop_int8),
                "fp16": (tol.min_label_agreement_fp16, tol.max_accuracy_drop_fp16),
            }.get(entry["precision"])
            if limits is not None and drop is not None:
                entry["eval"]["tolerance"] = {
                    "min_label_agreement": limits[0],
                    "max_accuracy_drop": limits[1],
                    "passed": bool(agreement >= limits[0] and drop <= limits[1]),
                }

        base = by_name.get(DEVICE_BASELINES[entry["device"]])
        base_p50 = (
            {r["batch_size"]: r["latency_ms"]["p50"] for r in base["results"]} if base else {}
        )
        for res in entry["results"]:
            ref_p50 = base_p50.get(res["batch_size"])
            res["speedup_p50_vs_device_baseline"] = (
                ref_p50 / res["latency_ms"]["p50"] if ref_p50 else None
            )


_REFERENCE_MUST_MATCH = ("model_id", "revision", "max_seq_length", "attn_implementation")


def load_padding_reference(cfg: PipelineConfig) -> dict[str, Any] | None:
    """The published fixed-padding results, if comparable with ``cfg``.

    Returns ``None`` (with a warning) when the file is missing, invalid, not a
    ``max_length`` run, or was produced for a different model / revision /
    sequence length / attention kernel.
    """
    path = cfg.paths.benchmarks_json
    if not path.exists():
        log.warning("No %s: dynamic results will lack the vs-128-token comparison", path)
        return None
    try:
        reference = load_results(path)
    except PipelineError as exc:
        log.warning("Ignoring padding reference %s: %s", path, exc)
        return None
    ref_cfg = reference["config"]
    mismatched = [k for k in _REFERENCE_MUST_MATCH if ref_cfg.get(k) != getattr(cfg, k)]
    if ref_cfg.get("padding", "max_length") != "max_length" or mismatched:
        log.warning("Ignoring padding reference %s: differs in %s", path, mismatched or "padding")
        return None
    return reference


def attach_padding_comparison(entries: list[dict[str, Any]], reference: dict[str, Any]) -> None:
    """Add ``speedup_p50_vs_max_length`` per result, matched by variant and batch size.

    The reference comes from a *separate* run, so this ratio carries run-to-run
    drift; it is meaningful for the large effects padding produces, not for a
    few percent.
    """
    ref_p50 = {
        (variant["name"], res["batch_size"]): res["latency_ms"]["p50"]
        for variant in reference["variants"]
        if variant["status"] == "ok"
        for res in variant["results"]
    }
    for entry in entries:
        if entry["status"] != "ok":
            continue
        for res in entry["results"]:
            base = ref_p50.get((entry["name"], res["batch_size"]))
            res["speedup_p50_vs_max_length"] = base / res["latency_ms"]["p50"] if base else None


# ==========================================================================
# Environment
# ==========================================================================
def _cpu_name() -> str:
    if sys.platform == "win32":
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def _nvidia_driver() -> str | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return out.stdout.strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        return None


def collect_environment() -> dict[str, Any]:
    """Hardware / software fingerprint stored alongside the results."""
    import onnx
    import onnxruntime as ort
    import torch
    import transformers

    battery = psutil.sensors_battery() if hasattr(psutil, "sensors_battery") else None
    env: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": _cpu_name(),
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "ram_total_gb": round(psutil.virtual_memory().total / 1024**3, 1),
        "power_plugged": None if battery is None else bool(battery.power_plugged),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,  # type: ignore[no-untyped-call]
        "torch_num_threads": torch.get_num_threads(),
        "onnxruntime": ort.__version__,
        "onnxruntime_providers": ort.get_available_providers(),
        "onnx": onnx.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "gpu": None,
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        env["gpu"] = {
            "name": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
            "total_memory_mb": round(props.total_memory / _MB),
            "multiprocessors": props.multi_processor_count,
            "driver": _nvidia_driver(),
        }
    return env


# ==========================================================================
# Schema validation
# ==========================================================================
_TOP_LEVEL: dict[str, type | tuple[type, ...]] = {
    "schema_version": int,
    "generated_at": str,
    "duration_s": (int, float),
    "opt_pipeline_version": str,
    "environment": dict,
    "methodology": dict,
    "config": dict,
    "variants": list,
}
_VARIANT_OK: dict[str, type | tuple[type, ...]] = {
    "name": str,
    "backend": str,
    "device": str,
    "precision": str,
    "status": str,
    "provider": str,
    "model_size_mb": (int, float),
    "ram_baseline_mb": (int, float),
    "results": list,
}
_RESULT: dict[str, type | tuple[type, ...]] = {
    "batch_size": int,
    "sequence_length": int,
    "iterations": int,
    "latency_ms": dict,
    "throughput_samples_per_s": (int, float),
    "ram_peak_mb": (int, float),
    "ram_delta_mb": (int, float),
}
_LATENCY_KEYS = ("mean", "std", "min", "p50", "p95", "p99", "max")
_OPTIONAL_NUMBERS = (
    "vram_torch_allocated_peak_mb",
    "vram_torch_reserved_peak_mb",
    "vram_device_delta_mb",
)
_WORKLOAD: dict[str, type | tuple[type, ...]] = {
    "padding": str,
    "distinct_batches": int,
    "distinct_shapes": int,
    "seq_len_mean": (int, float),
    "seq_len_p50": (int, float),
    "seq_len_max": int,
    "mean_real_tokens": (int, float),
    "real_token_fraction": (int, float),
    "sha256": str,
}


def _check(obj: dict[str, Any], spec: dict[str, Any], where: str, errors: list[str]) -> None:
    for key, typ in spec.items():
        if key not in obj:
            errors.append(f"{where}: missing key {key!r}")
        elif not isinstance(obj[key], typ) or isinstance(obj[key], bool):
            errors.append(f"{where}.{key}: expected {typ}, got {type(obj[key]).__name__}")


def validate_results(payload: dict[str, Any]) -> list[str]:
    """Return a list of schema violations (empty list = valid).

    Version 2 additionally requires a ``workload`` block per result and that
    all variants were timed on the identical workload for each batch size.
    """
    errors: list[str] = []
    _check(payload, _TOP_LEVEL, "root", errors)
    if errors:
        return errors
    version = payload["schema_version"]
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        errors.append(f"root.schema_version: expected one of {SUPPORTED_SCHEMA_VERSIONS}")
        return errors
    padding = payload["config"].get("padding", "max_length")
    fingerprints: dict[int, set[str]] = {}
    for i, variant in enumerate(payload["variants"]):
        where = f"variants[{i}]"
        if not isinstance(variant, dict):
            errors.append(f"{where}: not an object")
            continue
        status = variant.get("status")
        if status not in ("ok", "skipped", "error"):
            errors.append(f"{where}.status: invalid {status!r}")
            continue
        if status != "ok":
            if not variant.get("error"):
                errors.append(f"{where}: non-ok variant must explain itself in 'error'")
            continue
        _check(variant, _VARIANT_OK, where, errors)
        for j, res in enumerate(variant.get("results", [])):
            rwhere = f"{where}.results[{j}]"
            _check(res, _RESULT, rwhere, errors)
            latency = res.get("latency_ms", {})
            for key in _LATENCY_KEYS:
                if not isinstance(latency.get(key), int | float):
                    errors.append(f"{rwhere}.latency_ms.{key}: missing or not a number")
            if all(isinstance(latency.get(k), int | float) for k in ("p50", "p95", "p99")) and not (
                latency["p50"] <= latency["p95"] <= latency["p99"]
            ):
                errors.append(f"{rwhere}.latency_ms: percentiles not monotonic")
            for key in _OPTIONAL_NUMBERS:
                if key not in res:
                    errors.append(f"{rwhere}: missing key {key!r}")
                elif res[key] is not None and not isinstance(res[key], int | float):
                    errors.append(f"{rwhere}.{key}: expected number or null")
            if version >= 2:
                _check_workload(res, rwhere, padding, fingerprints, errors)
        if variant.get("eval") is not None:
            for key in ("accuracy", "f1", "n"):
                if key not in variant["eval"]:
                    errors.append(f"{where}.eval: missing key {key!r}")
    for batch_size, hashes in sorted(fingerprints.items()):
        if len(hashes) > 1:
            errors.append(f"batch {batch_size}: variants were timed on different workloads")
    return errors


def _check_workload(
    res: dict[str, Any],
    where: str,
    padding: str,
    fingerprints: dict[int, set[str]],
    errors: list[str],
) -> None:
    workload = res.get("workload")
    if not isinstance(workload, dict):
        errors.append(f"{where}: missing key 'workload'")
        return
    before = len(errors)
    _check(workload, _WORKLOAD, f"{where}.workload", errors)
    if len(errors) > before:
        return
    if workload["padding"] != padding:
        errors.append(f"{where}.workload.padding: {workload['padding']!r} != config {padding!r}")
    if isinstance(res.get("batch_size"), int):
        fingerprints.setdefault(res["batch_size"], set()).add(workload["sha256"])


# ==========================================================================
# Markdown report
# ==========================================================================
def _fmt(value: float | None, spec: str = ".2f", suffix: str = "") -> str:
    return "—" if value is None else f"{value:{spec}}{suffix}"


def _payload_padding(payload: dict[str, Any]) -> str:
    padding: str = payload["config"].get("padding", "max_length")
    return padding


def _header(payload: dict[str, Any], source: str, workload: str) -> list[str]:
    env = payload["environment"]
    gpu = env.get("gpu") or {}
    power = {True: "on AC power", False: "ON BATTERY", None: "power state n/a"}[
        env.get("power_plugged")
    ]
    return [
        f"_Generated {payload['generated_at']} from `{source}` "
        f"(schema v{payload['schema_version']}). "
        f"CPU: {env['cpu']} ({env['cpu_physical_cores']}C/{env['cpu_logical_cores']}T, "
        f"{env['ram_total_gb']} GB RAM, {power}). "
        f"GPU: {gpu.get('name', 'n/a')} ({gpu.get('total_memory_mb', '?')} MB, "
        f"driver {gpu.get('driver') or 'n/a'}). "
        f"torch {env['torch']}, onnxruntime {env['onnxruntime']}. {workload}_",
        "",
    ]


def _not_measured(payload: dict[str, Any]) -> list[str]:
    skipped = [v for v in payload["variants"] if v["status"] != "ok"]
    if not skipped:
        return []
    return [
        "",
        "Not measured:",
        *(f"- `{v['name']}` ({v['status']}): {str(v['error']).splitlines()[0]}" for v in skipped),
    ]


def render_markdown(payload: dict[str, Any]) -> str:
    """Render the README tables for a results payload (either padding mode)."""
    if _payload_padding(payload) == "dynamic":
        return _render_dynamic(payload)
    ok = [v for v in payload["variants"] if v["status"] == "ok"]
    lines = _header(
        payload,
        "results/benchmarks.json",
        f"{payload['config']['benchmark_iters']} timed iterations per cell, "
        f"sequence length {payload['config']['max_seq_length']}.",
    )

    for device, title in (("cpu", "CPU"), ("cuda", "GPU")):
        rows = [v for v in ok if v["device"] == device]
        if not rows:
            continue
        mem_head = (
            "Peak RAM Δ (MB)"
            if device == "cpu"
            else "VRAM device Δ (MB) | VRAM torch alloc / reserved (MB)"
        )
        mem_sep = "---:" if device == "cpu" else "---: | ---:"
        lines += [
            f"#### {title} latency & throughput",
            "",
            f"| Variant | Batch | p50 (ms) | p95 (ms) | p99 (ms) | Throughput (samples/s) "
            f"| Speed-up vs PyTorch FP32 (p50) | {mem_head} |",
            f"|---|---:|---:|---:|---:|---:|---:|{mem_sep}|",
        ]
        for v in rows:
            for r in v["results"]:
                lat = r["latency_ms"]
                if device == "cpu":
                    mem = _fmt(r["ram_delta_mb"], ".0f")
                else:
                    torch_mem = (
                        "—"
                        if r["vram_torch_allocated_peak_mb"] is None
                        else f"{r['vram_torch_allocated_peak_mb']:.0f} / "
                        f"{r['vram_torch_reserved_peak_mb']:.0f}"
                    )
                    mem = f"{_fmt(r['vram_device_delta_mb'], '.0f')} | {torch_mem}"
                lines.append(
                    f"| `{v['name']}` | {r['batch_size']} | {lat['p50']:.2f} | {lat['p95']:.2f} "
                    f"| {lat['p99']:.2f} | {r['throughput_samples_per_s']:.1f} "
                    f"| {_fmt(r.get('speedup_p50_vs_device_baseline'), '.2f', '×')} | {mem} |"  # noqa: RUF001
                )
        lines.append("")

    lines += [
        "#### Accuracy & size (SST-2 validation)",
        "",
        "| Variant | Provider | Model size (MB) | Accuracy | F1 | Δ accuracy vs PyTorch FP32 "
        "| Label agreement | Within tolerance |",
        "|---|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for v in ok:
        ev = v.get("eval") or {}
        tol = ev.get("tolerance")
        verdict = "—" if tol is None else ("✅" if tol["passed"] else "❌")
        drop = ev.get("accuracy_drop_vs_reference")
        lines.append(
            f"| `{v['name']}` | {v['provider']} | {v['model_size_mb']:.1f} "
            f"| {_fmt(ev.get('accuracy'), '.2%')} | {_fmt(ev.get('f1'), '.4f')} "
            # "+ 0.0" turns IEEE -0.0 into 0.0 so a zero delta prints "+0.00".
            f"| {_fmt(None if drop is None else -drop * 100 + 0.0, '+.2f', ' pp')} "
            f"| {_fmt(ev.get('label_agreement_vs_reference'), '.2%')} | {verdict} |"
        )
    lines += _not_measured(payload)
    return "\n".join(lines) + "\n"


def _render_dynamic(payload: dict[str, Any]) -> str:
    """Tables for a dynamic-padding run: workload, then latency per device."""
    cfg = payload["config"]
    ok = [v for v in payload["variants"] if v["status"] == "ok"]
    lines = _header(
        payload,
        "results/benchmarks_dynamic.json",
        f"{cfg['benchmark_iters']} timed batches per cell, drawn (seed {cfg['seed']}) from "
        f"the {cfg['dataset_split']} sentences; each batch is padded to its longest sentence.",
    )

    if ok:
        lines += [
            "#### Workload (identical for every variant)",
            "",
            "| Batch | Padded length: mean / p50 / max | Real tokens per sentence "
            f"| Real share of padded tokens | Real share at fixed {cfg['max_seq_length']} |",
            "|---:|---:|---:|---:|---:|",
        ]
        for r in ok[0]["results"]:
            w = r["workload"]
            lines.append(
                f"| {r['batch_size']} | {w['seq_len_mean']:.1f} / {w['seq_len_p50']:.0f} / "
                f"{w['seq_len_max']} | {w['mean_real_tokens']:.1f} "
                f"| {w['real_token_fraction']:.1%} "
                f"| {w['mean_real_tokens'] / cfg['max_seq_length']:.1%} |"
            )
        lines.append("")

    for device, title in (("cpu", "CPU"), ("cuda", "GPU")):
        rows = [v for v in ok if v["device"] == device]
        if not rows:
            continue
        mem_head = "Peak RAM Δ (MB)" if device == "cpu" else "VRAM device Δ (MB)"
        lines += [
            f"#### {title} latency & throughput (dynamic padding)",
            "",
            "| Variant | Batch | p50 (ms) | p95 (ms) | p99 (ms) | Throughput (samples/s) "
            f"| Speed-up vs PyTorch FP32 (p50) | Speed-up vs {cfg['max_seq_length']}-token "
            f"padding (p50) | {mem_head} |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for v in rows:
            for r in v["results"]:
                lat = r["latency_ms"]
                mem = r["ram_delta_mb"] if device == "cpu" else r["vram_device_delta_mb"]
                lines.append(
                    f"| `{v['name']}` | {r['batch_size']} | {lat['p50']:.2f} | {lat['p95']:.2f} "
                    f"| {lat['p99']:.2f} | {r['throughput_samples_per_s']:.1f} "
                    f"| {_fmt(r.get('speedup_p50_vs_device_baseline'), '.2f', '×')} "  # noqa: RUF001
                    f"| {_fmt(r.get('speedup_p50_vs_max_length'), '.2f', '×')} "  # noqa: RUF001
                    f"| {_fmt(mem, '.0f')} |"
                )
        lines.append("")

    reference = payload.get("padding_reference")
    if reference:
        lines.append(
            f"The fixed-padding p50 comes from `{reference['path']}` (generated "
            f"{reference['generated_at']}), a separate run. The ratios therefore include "
            "drift between runs, which has reached about 8% on this laptop's CPU, so they are "
            "reliable for the multi-fold effects of padding but not for small differences."
        )
    else:
        lines.append(f"No {cfg['max_seq_length']}-token reference run was available.")
    # Dedupe exact values: pre-rounding would shift 0.910550... (794/872) across
    # the .2% rounding boundary and print a different figure than the main table.
    accuracies = sorted({v["eval"]["accuracy"] for v in ok if v.get("eval")})
    if accuracies:
        measured = (
            f"{accuracies[0]:.2%} for every variant"
            if len(accuracies) == 1
            else f"{accuracies[0]:.2%} to {accuracies[-1]:.2%}"
        )
        lines.append(
            "Accuracy does not depend on the benchmark padding mode, because evaluation always "
            f"pads dynamically. This run re-measured {measured}."
        )
    lines += _not_measured(payload)
    return "\n".join(lines) + "\n"


def write_report(payload: dict[str, Any], md_path: Path, readme: Path | None = None) -> str:
    """Write the Markdown tables to ``md_path``; optionally splice them into ``readme``.

    The README section is chosen by the payload's padding mode
    (:data:`README_MARKERS`).
    """
    markdown = render_markdown(payload)
    atomic_write_text(md_path, markdown)
    if readme is not None:
        update_readme(readme, markdown, markers=README_MARKERS[_payload_padding(payload)])
    return markdown


def update_readme(
    readme: Path, markdown: str, *, markers: tuple[str, str] = (README_START, README_END)
) -> None:
    """Replace the text between a pair of benchmark markers in ``readme``.

    Raises:
        PipelineError: If the markers are missing.
    """
    start_marker, end_marker = markers
    text = readme.read_text(encoding="utf-8")
    start, end = text.find(start_marker), text.find(end_marker)
    if start < 0 or end < start:
        raise PipelineError(f"{readme} lacks {start_marker} / {end_marker} markers")
    new = text[: start + len(start_marker)] + "\n" + markdown + text[end:]
    atomic_write_text(readme, new)


def load_results(path: Path) -> dict[str, Any]:
    """Load and validate a results file."""
    if not path.exists():
        raise MissingArtifactError(f"{path} not found -- run `opt-pipeline benchmark` first")
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    errors = validate_results(payload)
    if errors:
        raise PipelineError(f"{path} failed schema validation:\n" + "\n".join(errors))
    return payload


if __name__ == "__main__":
    sys.exit(_worker_main())
