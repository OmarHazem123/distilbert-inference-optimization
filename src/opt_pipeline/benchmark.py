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
* **Shapes** -- fixed ``(batch, max_seq_length)`` batches padded to
  ``max_length`` so every variant computes exactly the same work.
* **Warmup** -- a short untimed memory-measurement pass, then
  ``cfg.warmup_runs`` untimed iterations, then ``cfg.benchmark_iters`` timed
  iterations with the garbage collector disabled.
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
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from itertools import cycle, islice
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import psutil

from opt_pipeline import __version__
from opt_pipeline.baseline import (
    Encoded,
    TorchClassifier,
    encode,
    load_id2label,
    load_tokenizer,
    seed_everything,
)
from opt_pipeline.config import PipelineConfig
from opt_pipeline.errors import MissingArtifactError, PipelineError
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

SCHEMA_VERSION = 1
_MB = 1024 * 1024
MEMORY_PASS_ITERATIONS = 3
"""Untimed iterations run under the RSS sampler before the timed loop."""

README_START = "<!-- BENCHMARKS:START -->"
README_END = "<!-- BENCHMARKS:END -->"


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
    """All measurements for one (variant, batch size)."""

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

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable view."""
        out = asdict(self)
        out["latency_ms"] = self.latency_ms.to_dict()
        return out


def make_batch(
    tokenizer: Any, batch_size: int, max_length: int, texts: Sequence[str] = PARITY_TEXTS
) -> Encoded:
    """Fixed-shape ``(batch_size, max_length)`` batch cycling over ``texts``."""
    chosen = list(islice(cycle(texts), batch_size))
    return encode(tokenizer, chosen, max_length=max_length, padding="max_length")


def measure_batch(
    runner: Runner,
    batch: Encoded,
    *,
    warmup: int,
    iterations: int,
    ram_baseline_mb: float,
    cuda_probe: CudaMemoryProbe | None = None,
    torch_allocator: bool = False,
) -> BatchResult:
    """Memory pass (untimed, sampled) followed by the timed latency loop.

    Args:
        runner: Any :class:`~opt_pipeline.runners.Runner`.
        batch: Encoded fixed-shape batch.
        warmup: Untimed iterations before timing.
        iterations: Timed iterations.
        ram_baseline_mb: RSS before the model was loaded.
        cuda_probe: VRAM probe for CUDA variants.
        torch_allocator: Whether the runner allocates through PyTorch (so the
            allocator peak counters are meaningful).
    """

    def step() -> None:
        runner.predict_logits(batch)

    if cuda_probe is not None and torch_allocator:
        cuda_probe.reset_torch_peaks()
    with RssSampler() as sampler:
        for _ in range(MEMORY_PASS_ITERATIONS):
            step()
            runner.synchronize()

    torch_alloc = torch_reserved = device_delta = None
    if cuda_probe is not None:
        device_delta = cuda_probe.delta_mb()
        if torch_allocator:
            torch_alloc, torch_reserved = cuda_probe.torch_peaks_mb()

    samples = time_callable(step, iterations=iterations, warmup=warmup, sync=runner.synchronize)
    stats = LatencyStats.from_samples(samples)
    batch_size, seq_len = (int(d) for d in batch["input_ids"].shape)
    return BatchResult(
        batch_size=batch_size,
        sequence_length=seq_len,
        warmup=warmup + MEMORY_PASS_ITERATIONS,
        iterations=iterations,
        latency_ms=stats,
        throughput_samples_per_s=batch_size * 1e3 / stats.mean,
        ram_peak_mb=sampler.peak_mb,
        ram_delta_mb=sampler.peak_mb - ram_baseline_mb,
        vram_torch_allocated_peak_mb=torch_alloc,
        vram_torch_reserved_peak_mb=torch_reserved,
        vram_device_delta_mb=device_delta,
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

    Returns a JSON-serialisable dict; ``predictions`` (dataset label space) is
    included for the parent to compute cross-variant agreement and is
    stripped before results are written.
    """
    import torch

    # Benchmarks favour the fastest kernels; determinism only pins reduction
    # order, which is irrelevant to timing and may select slower algorithms.
    seed_everything(cfg.seed, deterministic=False)
    cuda_probe = CudaMemoryProbe() if spec.device == "cuda" else None
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

    results: list[dict[str, Any]] = []
    for batch_size in cfg.batch_sizes:
        batch = make_batch(tokenizer, batch_size, cfg.max_seq_length)
        res = measure_batch(
            runner,
            batch,
            warmup=cfg.warmup_runs,
            iterations=cfg.benchmark_iters,
            ram_baseline_mb=ram_baseline,
            cuda_probe=cuda_probe,
            torch_allocator=is_torch,
        )
        log.info(
            "[%s] bs=%-3d p50=%8.2f ms  p95=%8.2f ms  p99=%8.2f ms  %8.1f samples/s",
            spec.name,
            batch_size,
            res.latency_ms.p50,
            res.latency_ms.p95,
            res.latency_ms.p99,
            res.throughput_samples_per_s,
        )
        results.append(res.to_dict())

    process_peak = process_peak_rss_mb()  # before evaluation adds dataset memory

    eval_block: dict[str, Any] | None = None
    predictions: list[int] | None = None
    if evaluate:
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
        output: Destination; defaults to ``cfg.paths.benchmarks_json``.

    Returns:
        The validated results payload.
    """
    import torch

    specs = get_variants(variant_names)
    check_artifacts(specs, cfg)
    output = output or cfg.paths.benchmarks_json
    environment = collect_environment()
    split = load_sst2(cfg) if evaluate else None  # warm the cache once, before workers

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
    payload = build_payload(
        entries,
        cfg,
        environment=environment,
        isolate=isolate,
        evaluate=evaluate,
        duration_s=time.perf_counter() - started,
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
) -> dict[str, Any]:
    """Assemble and validate the top-level ``benchmarks.json`` document.

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
    errors = validate_results(payload)
    if errors:
        raise PipelineError("benchmark payload failed schema validation:\n" + "\n".join(errors))
    return payload


def methodology(cfg: PipelineConfig, *, isolate: bool, evaluate: bool) -> dict[str, Any]:
    """Machine-readable description of how the numbers were produced."""
    q = cfg.quantization
    return {
        "timing_boundary": "host int64 numpy inputs -> host float32 logits (tokenisation excluded)",
        "cuda_sync": "torch.cuda.synchronize() after every PyTorch CUDA iteration",
        "padding": f"max_length ({cfg.max_seq_length} tokens) for timed batches",
        "warmup": f"{MEMORY_PASS_ITERATIONS} memory-pass + {cfg.warmup_runs} untimed iterations",
        "iterations": cfg.benchmark_iters,
        "gc_disabled_during_timing": True,
        "isolation": "one subprocess per variant" if isolate else "in-process (no isolation)",
        "seeding": f"seed={cfg.seed}, deterministic algorithms off for benchmarking",
        "ram": "peak RSS sampled every 2 ms during an untimed pass; delta vs pre-load RSS",
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


def _check(obj: dict[str, Any], spec: dict[str, Any], where: str, errors: list[str]) -> None:
    for key, typ in spec.items():
        if key not in obj:
            errors.append(f"{where}: missing key {key!r}")
        elif not isinstance(obj[key], typ) or isinstance(obj[key], bool):
            errors.append(f"{where}.{key}: expected {typ}, got {type(obj[key]).__name__}")


def validate_results(payload: dict[str, Any]) -> list[str]:
    """Return a list of schema violations (empty list = valid)."""
    errors: list[str] = []
    _check(payload, _TOP_LEVEL, "root", errors)
    if errors:
        return errors
    if payload["schema_version"] != SCHEMA_VERSION:
        errors.append(f"root.schema_version: expected {SCHEMA_VERSION}")
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
        if variant.get("eval") is not None:
            for key in ("accuracy", "f1", "n"):
                if key not in variant["eval"]:
                    errors.append(f"{where}.eval: missing key {key!r}")
    return errors


# ==========================================================================
# Markdown report
# ==========================================================================
def _fmt(value: float | None, spec: str = ".2f", suffix: str = "") -> str:
    return "—" if value is None else f"{value:{spec}}{suffix}"


def render_markdown(payload: dict[str, Any]) -> str:
    """Render the README comparison tables from a results payload."""
    env = payload["environment"]
    gpu = env.get("gpu") or {}
    ok = [v for v in payload["variants"] if v["status"] == "ok"]
    lines: list[str] = []
    power = {True: "on AC power", False: "ON BATTERY", None: "power state n/a"}[
        env.get("power_plugged")
    ]
    lines += [
        f"_Generated {payload['generated_at']} from `results/benchmarks.json` "
        f"(schema v{payload['schema_version']}). "
        f"CPU: {env['cpu']} ({env['cpu_physical_cores']}C/{env['cpu_logical_cores']}T, "
        f"{env['ram_total_gb']} GB RAM, {power}). "
        f"GPU: {gpu.get('name', 'n/a')} ({gpu.get('total_memory_mb', '?')} MB, "
        f"driver {gpu.get('driver') or 'n/a'}). "
        f"torch {env['torch']}, onnxruntime {env['onnxruntime']}. "
        f"{payload['config']['benchmark_iters']} timed iterations per cell, "
        f"sequence length {payload['config']['max_seq_length']}._",
        "",
    ]

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
    skipped = [v for v in payload["variants"] if v["status"] != "ok"]
    if skipped:
        lines += ["", "Not measured:"]
        lines += [
            f"- `{v['name']}` ({v['status']}): {str(v['error']).splitlines()[0]}" for v in skipped
        ]
    return "\n".join(lines) + "\n"


def write_report(payload: dict[str, Any], md_path: Path, readme: Path | None = None) -> str:
    """Write the Markdown tables to ``md_path``; optionally splice them into ``readme``."""
    markdown = render_markdown(payload)
    atomic_write_text(md_path, markdown)
    if readme is not None:
        update_readme(readme, markdown)
    return markdown


def update_readme(readme: Path, markdown: str) -> None:
    """Replace the text between the benchmark markers in ``readme``.

    Raises:
        PipelineError: If the markers are missing.
    """
    text = readme.read_text(encoding="utf-8")
    start, end = text.find(README_START), text.find(README_END)
    if start < 0 or end < start:
        raise PipelineError(f"{readme} lacks {README_START} / {README_END} markers")
    new = text[: start + len(README_START)] + "\n" + markdown + text[end:]
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
