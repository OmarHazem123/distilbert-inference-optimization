"""Operator-level profiling of the PyTorch baseline with ``torch.profiler``.

Captures a few steady-state inference steps (after warmup), exports a Chrome
trace (open in ``chrome://tracing`` or https://ui.perfetto.dev), and reports
the top operators by *self* time -- time spent in the op itself, excluding
children -- which is where optimization effort pays off. On CUDA the ranking
uses self *device* time, so it lists the GPU kernels that dominate.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.profiler import ProfilerActivity, profile, schedule

from opt_pipeline.baseline import TorchClassifier, seed_everything
from opt_pipeline.benchmark import make_batch
from opt_pipeline.config import PipelineConfig, Precision
from opt_pipeline.export_onnx import OrtDevice
from opt_pipeline.utils import atomic_output, atomic_write_json, get_logger

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class OpStat:
    """Aggregated timing of one operator / kernel name across active steps."""

    name: str
    calls: int
    self_cpu_ms: float
    cpu_total_ms: float
    self_device_ms: float
    device_total_ms: float
    share_pct: float
    """Share of the ranking metric summed over the same row kind (ops or kernels)."""


@dataclass(frozen=True, slots=True)
class ProfileReport:
    """Result of one profiling run.

    ``top_ops`` ranks framework operators (``aten::*``) -- on CUDA by the GPU
    time of the kernels they launched, on CPU by self CPU time. ``top_kernels``
    (CUDA only) ranks the GPU kernels themselves. The two are kept apart, and
    the per-step ``ProfilerStep*`` wrapper is excluded, so no time is counted
    twice within a ranking.
    """

    device: str
    precision: str
    batch_size: int
    sequence_length: int
    active_steps: int
    sort_by: str
    trace_path: str
    total_ms: float
    """Summed ranking metric over all operators (GPU kernel time on CUDA)."""
    top_ops: list[OpStat] = field(default_factory=list)
    top_kernels: list[OpStat] = field(default_factory=list)
    table: str = ""

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable view (without the pre-rendered table)."""
        out = asdict(self)
        out.pop("table")
        return out


def profile_model(
    cfg: PipelineConfig,
    *,
    device: OrtDevice = "cpu",
    precision: Precision = "fp32",
    batch_size: int = 8,
    warmup_steps: int = 3,
    active_steps: int = 5,
    row_limit: int = 10,
    output_dir: Path | None = None,
) -> ProfileReport:
    """Profile ``active_steps`` forward passes and rank operators by self time.

    Args:
        cfg: Pipeline configuration (model, ``max_seq_length``).
        device: ``cpu`` or ``cuda``.
        precision: ``fp32`` or ``fp16`` (CUDA only).
        batch_size: Batch size of the profiled input (padded to max length).
        warmup_steps: Profiler warmup steps (recorded but discarded).
        active_steps: Steps whose events are aggregated.
        row_limit: Number of operators to report.
        output_dir: Where to write the trace / JSON; defaults to
            ``cfg.paths.profiles_dir``.
    """
    seed_everything(cfg.seed, deterministic=False)
    clf = TorchClassifier(cfg, device=device, precision=precision)
    batch = make_batch(clf.tokenizer, batch_size, cfg.max_seq_length)
    inputs = clf.to_device(batch)

    on_cuda = clf.device.type == "cuda"
    activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if on_cuda else [])
    sort_by = "self_cuda_time_total" if on_cuda else "self_cpu_time_total"
    stem = f"torch_{clf.device.type}_{precision}_bs{batch_size}"
    out_dir = output_dir or cfg.paths.profiles_dir
    trace_path = out_dir / f"{stem}.trace.json"

    def export_trace(prof: profile) -> None:
        with atomic_output(trace_path) as tmp:
            prof.export_chrome_trace(str(tmp))

    # One untimed call outside the profiler absorbs lazy init (cuBLAS handles,
    # CUDA context, allocator growth) so it cannot pollute the profile.
    clf.forward(inputs)
    clf.synchronize()

    log.info(
        "Profiling %s on %s (%s), batch %d x %d tokens, %d active steps",
        cfg.model_id,
        clf.device,
        precision,
        batch_size,
        cfg.max_seq_length,
        active_steps,
    )
    with profile(
        activities=activities,
        schedule=schedule(wait=1, warmup=warmup_steps, active=active_steps, repeat=1),
        on_trace_ready=export_trace,
        record_shapes=True,
    ) as prof:
        for _ in range(1 + warmup_steps + active_steps):
            clf.forward(inputs)
            clf.synchronize()
            prof.step()

    averages = prof.key_averages()
    table = averages.table(sort_by=sort_by, row_limit=row_limit)
    top_ops, total_us = _rank(averages, on_cuda=on_cuda, kernels=False, row_limit=row_limit)
    top_kernels = (
        _rank(averages, on_cuda=True, kernels=True, row_limit=row_limit)[0] if on_cuda else []
    )
    report = ProfileReport(
        device=clf.device.type,
        precision=precision,
        batch_size=batch_size,
        sequence_length=cfg.max_seq_length,
        active_steps=active_steps,
        sort_by=sort_by,
        trace_path=trace_path.as_posix(),
        total_ms=total_us / 1e3,
        top_ops=top_ops,
        top_kernels=top_kernels,
        table=table,
    )
    atomic_write_json(out_dir / f"{stem}.top_ops.json", report.to_dict())
    return report


def _is_kernel(evt: Any) -> bool:
    """True for GPU-side events (kernels, memcpys) as opposed to framework ops."""
    from torch._C._autograd import DeviceType  # re-exported (untyped) as torch.autograd.DeviceType

    return bool(getattr(evt, "device_type", DeviceType.CPU) != DeviceType.CPU)


def _rank(
    averages: Any, *, on_cuda: bool, kernels: bool, row_limit: int
) -> tuple[list[OpStat], float]:
    """Rank one row kind by self time; returns ``(top rows, total in us)``.

    ``ProfilerStep*`` is the per-step wrapper whose time is the sum of its
    children, so it is excluded rather than double-counted.
    """

    def self_device_us(evt: Any) -> float:
        return float(getattr(evt, "self_device_time_total", 0.0))

    def key(evt: Any) -> float:
        return self_device_us(evt) if on_cuda else float(evt.self_cpu_time_total)

    events = [
        e
        for e in averages
        if _is_kernel(e) == kernels and not str(e.key).startswith("ProfilerStep") and key(e) > 0
    ]
    total = sum(key(e) for e in events)
    ranked = sorted(events, key=key, reverse=True)[:row_limit]
    rows = [
        OpStat(
            name=str(e.key),
            calls=int(e.count),
            self_cpu_ms=float(e.self_cpu_time_total) / 1e3,
            cpu_total_ms=float(e.cpu_time_total) / 1e3,
            self_device_ms=self_device_us(e) / 1e3,
            device_total_ms=float(getattr(e, "device_time_total", 0.0)) / 1e3,
            share_pct=100.0 * key(e) / (total or 1.0),
        )
        for e in ranked
    ]
    return rows, total


def _format_rows(title: str, rows: list[OpStat], metric: str, on_cuda: bool) -> list[str]:
    lines = [title, f"{'#':>2}  {'name':<60} {'calls':>6} {metric:>15} {'share':>7}"]
    for i, op in enumerate(rows, 1):
        value = op.self_device_ms if on_cuda else op.self_cpu_ms
        name = op.name if len(op.name) <= 60 else op.name[:57] + "..."
        lines.append(f"{i:>2}  {name:<60} {op.calls:>6} {value:>15.3f} {op.share_pct:>6.1f}%")
    return lines


def format_top_ops(report: ProfileReport) -> str:
    """Compact tables of the top operators (and, on CUDA, kernels)."""
    on_cuda = report.device == "cuda"
    metric = "GPU ms" if on_cuda else "self CPU ms"
    header = (
        f"{report.device}/{report.precision}, batch {report.batch_size} x "
        f"{report.sequence_length}, {report.active_steps} steps, "
        f"total {'GPU kernel' if on_cuda else 'self CPU'} time {report.total_ms:.1f} ms"
    )
    lines = _format_rows(
        f"Top {len(report.top_ops)} operators by {metric} -- {header}",
        report.top_ops,
        metric,
        on_cuda,
    )
    if report.top_kernels:
        kernel_title = f"Top {len(report.top_kernels)} CUDA kernels by GPU ms"
        lines.append("")
        lines += _format_rows(kernel_title, report.top_kernels, metric, on_cuda)
    lines.append(f"Chrome trace: {report.trace_path}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m opt_pipeline.profile_bottlenecks``: profile with defaults."""
    del argv
    cfg = PipelineConfig()
    report = profile_model(cfg, device="cuda" if torch.cuda.is_available() else "cpu")
    print(report.table)
    print(format_top_ops(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
