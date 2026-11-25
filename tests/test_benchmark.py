"""Fast smoke tests for metric collection, the results schema and reporting.

Everything except the final ``slow`` test runs in well under a second per
test using a fake runner -- no model, no GPU, no downloads.
"""

from __future__ import annotations

import gc
import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from opt_pipeline.benchmark import (
    README_END,
    README_START,
    LatencyStats,
    RssSampler,
    attach_comparisons,
    build_payload,
    measure_batch,
    render_markdown,
    rss_mb,
    time_callable,
    update_readme,
    validate_results,
)
from opt_pipeline.cli import _config_from_args, build_parser
from opt_pipeline.config import PathsConfig, PipelineConfig, QuantizationConfig, Tolerances
from opt_pipeline.errors import ConfigError, PipelineError
from opt_pipeline.evaluate import classification_metrics, model_to_dataset_labels
from opt_pipeline.runners import VARIANTS
from opt_pipeline.utils import atomic_write_json


class FakeRunner:
    """Deterministic stand-in satisfying the ``Runner`` protocol."""

    device_type = "cpu"

    def __init__(self, work: int = 20_000) -> None:
        self.calls = 0
        self.syncs = 0
        self._work = np.ones(work, dtype=np.float32)

    def predict_logits(
        self, encoded: Mapping[str, npt.NDArray[np.int64]]
    ) -> npt.NDArray[np.float32]:
        self.calls += 1
        float(self._work.sum())  # a little real work so latencies are > 0
        return np.zeros((encoded["input_ids"].shape[0], 2), dtype=np.float32)

    def synchronize(self) -> None:
        self.syncs += 1


def fake_batch(batch_size: int = 4, seq_len: int = 16) -> dict[str, npt.NDArray[np.int64]]:
    ones = np.ones((batch_size, seq_len), dtype=np.int64)
    return {"input_ids": ones, "attention_mask": ones}


def fake_entry(name: str, p50: float, preds: list[int] | None, accuracy: float) -> dict[str, Any]:
    """A minimal, schema-valid ``ok`` variant entry."""
    spec = VARIANTS[name]
    result = measure_batch(
        FakeRunner(), fake_batch(), warmup=1, iterations=5, ram_baseline_mb=rss_mb()
    ).to_dict()
    result["latency_ms"].update(p50=p50, p95=p50 * 1.1, p99=p50 * 1.2)
    return {
        **spec.to_dict(),
        "status": "ok",
        "error": None,
        "provider": "fake",
        "artifact": "fake.onnx",
        "model_size_mb": 10.0,
        "ram_baseline_mb": 100.0,
        "results": [result],
        "eval": {"n": 4, "accuracy": accuracy, "f1": accuracy},
        "predictions": preds,
    }


# --------------------------------------------------------------------------
# Latency
# --------------------------------------------------------------------------
def test_latency_stats_match_numpy() -> None:
    samples = np.random.default_rng(0).gamma(2.0, 3.0, size=1000)
    stats = LatencyStats.from_samples(samples)
    p50, p95, p99 = np.percentile(samples, [50, 95, 99])
    assert stats.p50 == pytest.approx(p50)
    assert stats.p95 == pytest.approx(p95)
    assert stats.p99 == pytest.approx(p99)
    assert stats.mean == pytest.approx(samples.mean())
    assert stats.min <= stats.p50 <= stats.p95 <= stats.p99 <= stats.max


def test_latency_stats_reject_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        LatencyStats.from_samples([])


def test_time_callable_counts_warmup_and_sync() -> None:
    runner = FakeRunner()
    batch = fake_batch()
    samples = time_callable(
        lambda: runner.predict_logits(batch), iterations=7, warmup=3, sync=runner.synchronize
    )
    assert samples.shape == (7,)
    assert (samples > 0).all()
    assert runner.calls == 10  # warmup runs are executed but not recorded
    assert runner.syncs == 10  # every call, timed or not, is synchronised


def test_time_callable_restores_gc_even_on_error() -> None:
    assert gc.isenabled()

    def boom() -> None:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        time_callable(boom, iterations=3)
    assert gc.isenabled()


# --------------------------------------------------------------------------
# Memory
# --------------------------------------------------------------------------
def test_rss_sampler_sees_allocation() -> None:
    baseline = rss_mb()
    with RssSampler(interval_s=0.001) as sampler:
        block = np.ones(80 * 1024 * 1024 // 8)  # 80 MiB, touched by ones()
        time.sleep(0.05)
        del block
    assert sampler.peak_mb >= baseline + 60


def test_measure_batch_collects_all_metrics() -> None:
    runner = FakeRunner()
    result = measure_batch(
        runner, fake_batch(4, 16), warmup=2, iterations=9, ram_baseline_mb=rss_mb()
    )
    assert (result.batch_size, result.sequence_length, result.iterations) == (4, 16, 9)
    assert result.throughput_samples_per_s == pytest.approx(4e3 / result.latency_ms.mean)
    assert result.ram_peak_mb > 0
    # CPU runner: VRAM fields are explicitly null, never a misleading 0.
    assert result.vram_device_delta_mb is None
    assert result.vram_torch_allocated_peak_mb is None
    assert result.vram_torch_reserved_peak_mb is None


# --------------------------------------------------------------------------
# Results schema
# --------------------------------------------------------------------------
def _payload(entries: list[dict[str, Any]]) -> dict[str, Any]:
    return build_payload(
        entries,
        PipelineConfig(),
        environment={"cpu": "test", "cpu_physical_cores": 1, "cpu_logical_cores": 1},
        isolate=False,
        evaluate=True,
        duration_s=1.0,
    )


def test_payload_schema_roundtrip(tmp_path: Path) -> None:
    entries = [
        fake_entry("pytorch-cpu-fp32", 10.0, [1, 0, 1, 1], 0.75),
        fake_entry("ort-cpu-int8", 5.0, [1, 0, 1, 0], 0.50),
        {**VARIANTS["ort-cuda-fp16"].to_dict(), "status": "skipped", "error": "no GPU"},
    ]
    attach_comparisons(entries, PipelineConfig())
    payload = _payload(entries)
    assert validate_results(payload) == []

    path = atomic_write_json(tmp_path / "results" / "benchmarks.json", payload)
    reloaded = json.loads(path.read_text(encoding="utf-8"))
    assert validate_results(reloaded) == []
    assert reloaded["variants"][0]["results"][0]["latency_ms"]["p50"] == pytest.approx(10.0)
    assert not list(path.parent.glob("*.tmp*"))  # atomic write left no temp files


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p.pop("environment"), "missing key 'environment'"),
        (lambda p: p["variants"][0].pop("provider"), "missing key 'provider'"),
        (lambda p: p["variants"][0]["results"][0]["latency_ms"].pop("p99"), "p99"),
        (
            lambda p: p["variants"][0]["results"][0]["latency_ms"].update(p50=99.0, p95=1.0),
            "not monotonic",
        ),
        (lambda p: p["variants"][0].update(status="weird"), "invalid 'weird'"),
        (lambda p: p["variants"].append({"status": "skipped"}), "must explain itself"),
        (
            lambda p: p["variants"][0]["results"][0].update(vram_device_delta_mb="big"),
            "number or null",
        ),
    ],
)
def test_schema_validator_catches_violations(mutate: Any, message: str) -> None:
    payload = _payload([fake_entry("pytorch-cpu-fp32", 10.0, None, 0.9)])
    mutate(payload)
    errors = validate_results(payload)
    assert any(message in e for e in errors), errors


def test_build_payload_rejects_invalid_entries() -> None:
    with pytest.raises(PipelineError, match="schema validation"):
        _payload([{"status": "ok", "name": "x"}])


def test_attach_comparisons_against_reference() -> None:
    entries = [
        fake_entry("pytorch-cpu-fp32", 10.0, [1, 0, 1, 1], 0.75),
        fake_entry("ort-cpu-int8", 5.0, [1, 0, 1, 0], 0.50),
    ]
    attach_comparisons(entries, PipelineConfig())
    ref, int8 = entries
    assert "predictions" not in ref and "predictions" not in int8  # stripped
    assert int8["eval"]["label_agreement_vs_reference"] == pytest.approx(0.75)
    assert int8["eval"]["accuracy_drop_vs_reference"] == pytest.approx(0.25)
    assert int8["eval"]["tolerance"]["passed"] is False  # 25 pp drop > 2 pp allowed
    assert "tolerance" not in ref["eval"]  # fp32 variants are not gated
    assert int8["results"][0]["speedup_p50_vs_device_baseline"] == pytest.approx(2.0)
    assert ref["results"][0]["speedup_p50_vs_device_baseline"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def test_render_markdown_and_update_readme(tmp_path: Path) -> None:
    entries = [
        fake_entry("pytorch-cpu-fp32", 10.0, [1, 0], 1.0),
        fake_entry("ort-cpu-fp32", 8.0, [1, 0], 1.0),
        {**VARIANTS["ort-cuda-fp32"].to_dict(), "status": "skipped", "error": "no GPU"},
    ]
    attach_comparisons(entries, PipelineConfig())
    payload = _payload(entries)
    payload["environment"].update(
        ram_total_gb=8, power_plugged=True, torch="t", onnxruntime="o", gpu=None
    )
    markdown = render_markdown(payload)
    for needle in ("`pytorch-cpu-fp32`", "`ort-cpu-fp32`", "p50 (ms)", "1.25×", "no GPU"):  # noqa: RUF001
        assert needle in markdown

    readme = tmp_path / "README.md"
    readme.write_text(f"intro\n{README_START}\nold\n{README_END}\noutro\n", encoding="utf-8")
    update_readme(readme, markdown)
    text = readme.read_text(encoding="utf-8")
    assert "old" not in text
    assert text.startswith("intro\n") and text.endswith("outro\n")
    assert "`ort-cpu-fp32`" in text


def test_update_readme_requires_markers(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("no markers here\n", encoding="utf-8")
    with pytest.raises(PipelineError, match="markers"):
        update_readme(readme, "table")


# --------------------------------------------------------------------------
# Config round-trip (the isolated-worker protocol) and CLI mapping
# --------------------------------------------------------------------------
def test_config_dict_roundtrip() -> None:
    cfg = PipelineConfig(
        batch_sizes=(32, 1),
        exporter="dynamo",
        tolerances=Tolerances(fp32_atol=1e-5),
        quantization=QuantizationConfig(int8_op_types=("MatMul",)),
        paths=PathsConfig(artifacts_dir=Path("somewhere/else")),
    )
    clone = PipelineConfig.from_dict(json.loads(json.dumps(cfg.to_dict())))
    assert clone == cfg
    assert isinstance(clone.paths.artifacts_dir, Path)
    assert clone.batch_sizes == (1, 32)


def test_config_from_dict_rejects_unknown_keys() -> None:
    data = PipelineConfig().to_dict()
    data["tolerances"]["bogus"] = 1
    with pytest.raises(ConfigError, match="bogus"):
        PipelineConfig.from_dict(data)


def test_cli_flags_map_to_config() -> None:
    args = build_parser().parse_args(
        ["benchmark", "--variants", "ort-cpu-int8", "--batch-sizes", "8", "1", "--iters", "7"]
    )
    cfg = _config_from_args(args)
    assert args.variants == ["ort-cpu-int8"]
    assert cfg.batch_sizes == (1, 8)
    assert cfg.benchmark_iters == 7
    assert cfg.warmup_runs == PipelineConfig().warmup_runs  # untouched default


# --------------------------------------------------------------------------
# Evaluation helpers
# --------------------------------------------------------------------------
def test_classification_metrics() -> None:
    m = classification_metrics([1, 1, 0, 0, 1], [1, 0, 0, 1, 1])
    assert (m.tp, m.fp, m.tn, m.fn) == (2, 1, 1, 1)
    assert m.accuracy == pytest.approx(0.6)
    assert m.precision == pytest.approx(2 / 3)
    assert m.recall == pytest.approx(2 / 3)
    assert m.f1 == pytest.approx(2 / 3)


def test_model_labels_map_to_dataset_labels_by_name() -> None:
    mapping = model_to_dataset_labels({0: "NEGATIVE", 1: "POSITIVE"}, ("negative", "positive"))
    assert mapping.tolist() == [0, 1]
    with pytest.raises(ConfigError):
        model_to_dataset_labels({0: "NEUTRAL"}, ("negative", "positive"))


# --------------------------------------------------------------------------
# Real isolated worker (slow)
# --------------------------------------------------------------------------
@pytest.mark.slow
def test_isolated_worker_end_to_end(tmp_cfg: PipelineConfig, onnx_fp32: Path) -> None:
    """The default benchmark path: a real subprocess worker per variant."""
    from opt_pipeline.benchmark import run_benchmarks

    cfg = tmp_cfg.with_overrides(batch_sizes=[1], benchmark_iters=3, warmup_runs=1)
    output = cfg.paths.results_dir / "smoke.json"
    payload = run_benchmarks(cfg, ["ort-cpu-fp32"], isolate=True, evaluate=False, output=output)
    assert validate_results(json.loads(output.read_text(encoding="utf-8"))) == []
    (variant,) = payload["variants"]
    assert variant["status"] == "ok", variant.get("error")
    assert variant["provider"] == "CPUExecutionProvider"
    (result,) = variant["results"]
    assert result["batch_size"] == 1 and result["iterations"] == 3
    assert result["latency_ms"]["p50"] > 0
    assert (cfg.paths.benchmark_logs_dir / "ort-cpu-fp32.log").exists()


# --------------------------------------------------------------------------
# Profiler (slow: loads the model)
# --------------------------------------------------------------------------
@pytest.mark.slow
def test_profile_ranks_operators_without_double_counting(
    tmp_cfg: PipelineConfig, tmp_path: Path
) -> None:
    from opt_pipeline.profile_bottlenecks import format_top_ops, profile_model

    report = profile_model(
        tmp_cfg,
        device="cpu",
        batch_size=1,
        warmup_steps=1,
        active_steps=2,
        row_limit=10,
        output_dir=tmp_path,
    )
    # Assert the ranking *mechanism*, not timings: which op wins a 2-step CPU
    # profile can wobble under background load (hardware numbers live in the
    # README, produced by the real profile runs).
    names = [op.name for op in report.top_ops]
    assert "aten::addmm" in names  # linear layers are always among the top ops
    assert not any(n.startswith("ProfilerStep") for n in names)  # wrapper excluded
    times = [op.self_cpu_ms for op in report.top_ops]
    assert times == sorted(times, reverse=True)
    assert sum(op.share_pct for op in report.top_ops) <= 100.0 + 1e-6
    assert report.total_ms >= sum(times) - 1e-6
    assert report.top_kernels == []  # CPU run: no CUDA kernel ranking
    assert Path(report.trace_path).stat().st_size > 0
    assert (tmp_path / "torch_cpu_fp32_bs1.top_ops.json").exists()
    assert "aten::addmm" in format_top_ops(report)


@pytest.mark.parametrize(
    ("argv", "canonical", "filename"),
    [
        (["benchmark"], True, "benchmarks.json"),
        (["benchmark", "--update-readme"], True, "benchmarks.json"),
        (["benchmark", "--variants", "ort-cpu-int8"], False, "benchmarks.partial.json"),
        (["benchmark", "--iters", "5"], False, "benchmarks.partial.json"),
        (["benchmark", "--batch-sizes", "1"], False, "benchmarks.partial.json"),
        (["benchmark", "--skip-eval"], False, "benchmarks.partial.json"),
        (["benchmark", "--no-isolate"], False, "benchmarks.partial.json"),
        (["benchmark", "--variants", "ort-cpu-int8", "--output", "x.json"], False, "x.json"),
    ],
)
def test_partial_benchmarks_never_overwrite_published_results(
    argv: list[str], canonical: bool, filename: str
) -> None:
    """A subset / quick run must not clobber results/benchmarks.json or the README."""
    from opt_pipeline.cli import benchmark_output_path, is_canonical_benchmark

    args = build_parser().parse_args(argv)
    cfg = _config_from_args(args)
    assert is_canonical_benchmark(args, cfg) is canonical
    assert benchmark_output_path(args, cfg).name == filename
