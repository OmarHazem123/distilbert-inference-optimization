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
    README_MARKERS,
    README_START,
    LatencyStats,
    RssSampler,
    Workload,
    attach_comparisons,
    attach_padding_comparison,
    build_payload,
    dynamic_workload,
    load_padding_reference,
    measure_workload,
    render_markdown,
    rss_mb,
    time_callable,
    update_readme,
    validate_results,
    write_report,
)
from opt_pipeline.cli import _config_from_args, build_parser
from opt_pipeline.config import (
    PaddingMode,
    PathsConfig,
    PipelineConfig,
    QuantizationConfig,
    Tolerances,
)
from opt_pipeline.errors import ConfigError, PipelineError
from opt_pipeline.evaluate import classification_metrics, model_to_dataset_labels
from opt_pipeline.runners import VARIANTS
from opt_pipeline.utils import atomic_write_json, atomic_write_text


class FakeRunner:
    """Deterministic stand-in satisfying the ``Runner`` protocol.

    Records the padded sequence length of every batch it is given.
    """

    device_type = "cpu"

    def __init__(self, work: int = 20_000) -> None:
        self.calls = 0
        self.syncs = 0
        self.lengths: list[int] = []
        self._work = np.ones(work, dtype=np.float32)

    def predict_logits(
        self, encoded: Mapping[str, npt.NDArray[np.int64]]
    ) -> npt.NDArray[np.float32]:
        self.calls += 1
        self.lengths.append(int(encoded["input_ids"].shape[1]))
        float(self._work.sum())  # a little real work so latencies are > 0
        return np.zeros((encoded["input_ids"].shape[0], 2), dtype=np.float32)

    def synchronize(self) -> None:
        self.syncs += 1


class FakeTokenizer:
    """Whitespace 'tokenizer' with the call signature ``baseline.encode`` uses.

    A sentence of n words becomes ``n + 2`` tokens ([CLS] ... [SEP]).
    """

    def __call__(
        self,
        texts: list[str],
        *,
        padding: str,
        truncation: bool,
        max_length: int,
        return_tensors: str,
    ) -> dict[str, npt.NDArray[np.int64]]:
        words = [[1000 + len(w) for w in t.split()] for t in texts]
        rows = [[101, *ids[: max_length - 2], 102] for ids in words]
        width = max_length if padding == "max_length" else max(len(r) for r in rows)
        input_ids = np.zeros((len(rows), width), dtype=np.int64)
        mask = np.zeros_like(input_ids)
        for i, row in enumerate(rows):
            input_ids[i, : len(row)] = row
            mask[i, : len(row)] = 1
        return {"input_ids": input_ids, "attention_mask": mask}


SENTENCES = [" ".join(["word"] * n) for n in range(1, 41)]
"""40 sentences of distinct lengths (3..42 tokens), so a row's mask sum identifies it."""


def fake_batch(batch_size: int = 4, seq_len: int = 16) -> dict[str, npt.NDArray[np.int64]]:
    ones = np.ones((batch_size, seq_len), dtype=np.int64)
    return {"input_ids": ones, "attention_mask": ones}


def single(batch: dict[str, npt.NDArray[np.int64]]) -> Workload:
    """Fixed-padding workload of one repeated batch."""
    return Workload("max_length", (batch,))


def fake_entry(
    name: str,
    p50: float,
    preds: list[int] | None,
    accuracy: float,
    padding: PaddingMode = "max_length",
) -> dict[str, Any]:
    """A minimal, schema-valid ``ok`` variant entry."""
    spec = VARIANTS[name]
    result = measure_workload(
        FakeRunner(),
        Workload(padding, (fake_batch(),)),
        warmup=1,
        iterations=5,
        ram_baseline_mb=rss_mb(),
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


def test_measure_workload_collects_all_metrics() -> None:
    runner = FakeRunner()
    result = measure_workload(
        runner, single(fake_batch(4, 16)), warmup=2, iterations=9, ram_baseline_mb=rss_mb()
    )
    assert (result.batch_size, result.sequence_length, result.iterations) == (4, 16, 9)
    assert result.throughput_samples_per_s == pytest.approx(4e3 / result.latency_ms.mean)
    assert result.ram_peak_mb > 0
    # CPU runner: VRAM fields are explicitly null, never a misleading 0.
    assert result.vram_device_delta_mb is None
    assert result.vram_torch_allocated_peak_mb is None
    assert result.vram_torch_reserved_peak_mb is None
    # Fixed padding: one shape, so the shape pass is the 3-iteration minimum.
    assert runner.calls == 3 + 2 + 9
    assert result.warmup == 3 + 2
    assert result.workload["distinct_shapes"] == 1
    assert result.workload["seq_len_mean"] == 16


def test_measure_workload_runs_every_shape_untimed_then_each_batch_once() -> None:
    """Shape pass (longest first) -> warmup -> one timed iteration per batch, in order."""
    batches = tuple(fake_batch(2, n) for n in (5, 9, 5, 7))
    runner = FakeRunner()
    result = measure_workload(
        runner, Workload("dynamic", batches), warmup=2, iterations=4, ram_baseline_mb=rss_mb()
    )
    shape_pass, warmup, timed = runner.lengths[:3], runner.lengths[3:5], runner.lengths[5:]
    assert shape_pass == [9, 7, 5]  # every distinct shape, longest (memory peak) first
    assert warmup == [5, 9]
    assert timed == [5, 9, 5, 7]  # the timed loop restarts at batch 0
    assert runner.syncs == runner.calls
    assert result.sequence_length == 9  # longest padded length
    assert result.warmup == 3 + 2
    w = result.workload
    assert (w["padding"], w["distinct_batches"], w["distinct_shapes"]) == ("dynamic", 4, 3)
    assert w["seq_len_mean"] == pytest.approx(6.5)
    assert w["seq_len_max"] == 9


def test_workload_rejects_mixed_batch_sizes() -> None:
    with pytest.raises(ValueError, match="one batch size"):
        Workload("dynamic", (fake_batch(2, 5), fake_batch(3, 5)))
    with pytest.raises(ValueError, match="at least one"):
        Workload("dynamic", ())


# --------------------------------------------------------------------------
# Dynamic workloads
# --------------------------------------------------------------------------
def test_dynamic_workload_is_seeded_and_padded_per_batch() -> None:
    tok = FakeTokenizer()
    kwargs: dict[str, Any] = {"batch_size": 8, "n_batches": 20, "max_length": 128}
    a = dynamic_workload(tok, SENTENCES, seed=42, **kwargs)
    b = dynamic_workload(tok, SENTENCES, seed=42, **kwargs)
    c = dynamic_workload(tok, SENTENCES, seed=7, **kwargs)
    assert a.sha256() == b.sha256()  # every worker process rebuilds the same workload
    assert a.sha256() != c.sha256()
    assert len(a.batches) == 20  # 160 draws from 40 sentences: cycles permutations
    for batch in a.batches:
        mask = batch["attention_mask"]
        assert mask.shape[0] == 8
        assert mask[:, -1].any()  # padded to the longest sentence, not beyond
        assert len(set(mask.sum(axis=1).tolist())) == 8  # no sentence twice in a batch


def test_dynamic_workload_batch_of_one_has_no_padding() -> None:
    w = dynamic_workload(
        FakeTokenizer(), SENTENCES, batch_size=1, n_batches=10, max_length=128, seed=0
    )
    summary = w.summary(10)
    assert summary["real_token_fraction"] == 1.0
    assert summary["seq_len_mean"] == pytest.approx(summary["mean_real_tokens"])


def test_dynamic_workload_truncates_at_max_length() -> None:
    w = dynamic_workload(
        FakeTokenizer(), SENTENCES, batch_size=4, n_batches=5, max_length=16, seed=0
    )
    assert max(int(b["input_ids"].shape[1]) for b in w.batches) <= 16


def test_dynamic_workload_rejects_batch_larger_than_dataset() -> None:
    with pytest.raises(ConfigError, match="exceeds"):
        dynamic_workload(
            FakeTokenizer(), SENTENCES[:3], batch_size=4, n_batches=1, max_length=8, seed=0
        )


# --------------------------------------------------------------------------
# Atomic writes
# --------------------------------------------------------------------------
def test_atomic_write_text_writes_lf_on_every_platform(tmp_path: Path) -> None:
    """Text mode would write CRLF on Windows; generated results must be byte-stable."""
    path = atomic_write_text(tmp_path / "out.md", "a\nb\n")
    assert path.read_bytes() == b"a\nb\n"
    assert b"\r" not in atomic_write_json(tmp_path / "out.json", {"k": [1, 2]}).read_bytes()


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
        (lambda p: p["variants"][0]["results"][0].pop("workload"), "missing key 'workload'"),
        (lambda p: p["variants"][0]["results"][0]["workload"].pop("sha256"), "'sha256'"),
        (
            lambda p: p["variants"][0]["results"][0]["workload"].update(padding="dynamic"),
            "!= config 'max_length'",
        ),
        (lambda p: p.update(schema_version=99), "expected one of"),
    ],
)
def test_schema_validator_catches_violations(mutate: Any, message: str) -> None:
    payload = _payload([fake_entry("pytorch-cpu-fp32", 10.0, None, 0.9)])
    mutate(payload)
    errors = validate_results(payload)
    assert any(message in e for e in errors), errors


def test_schema_rejects_variants_timed_on_different_workloads() -> None:
    """The cross-variant identity check that makes "same work" verifiable."""
    entries = [
        fake_entry("pytorch-cpu-fp32", 10.0, None, 0.9),
        fake_entry("ort-cpu-fp32", 8.0, None, 0.9),
    ]
    payload = _payload(entries)
    assert validate_results(payload) == []
    payload["variants"][1]["results"][0]["workload"]["sha256"] = "0" * 64
    assert any("different workloads" in e for e in validate_results(payload))


def test_schema_v1_results_still_validate_and_render() -> None:
    """Results published before the workload block existed stay readable."""
    payload = _payload([fake_entry("pytorch-cpu-fp32", 10.0, None, 0.9)])
    payload["schema_version"] = 1
    payload["config"].pop("padding")
    for result in payload["variants"][0]["results"]:
        result.pop("workload")
    payload["environment"].update(
        ram_total_gb=8, power_plugged=True, torch="t", onnxruntime="o", gpu=None
    )
    assert validate_results(payload) == []
    assert "`pytorch-cpu-fp32`" in render_markdown(payload)


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


def _env(payload: dict[str, Any]) -> dict[str, Any]:
    payload["environment"].update(
        ram_total_gb=8, power_plugged=True, torch="t", onnxruntime="o", gpu=None
    )
    return payload


def _dynamic_payload(reference: dict[str, Any] | None) -> dict[str, Any]:
    entries = [
        fake_entry("pytorch-cpu-fp32", 10.0, [1, 0], 1.0, padding="dynamic"),
        fake_entry("ort-cpu-int8", 5.0, [1, 0], 1.0, padding="dynamic"),
    ]
    cfg = PipelineConfig(padding="dynamic")
    attach_comparisons(entries, cfg)
    if reference is not None:
        attach_padding_comparison(entries, reference)
    return _env(
        build_payload(
            entries,
            cfg,
            environment={"cpu": "test", "cpu_physical_cores": 1, "cpu_logical_cores": 1},
            isolate=False,
            evaluate=True,
            duration_s=1.0,
            padding_reference=None
            if reference is None
            else {"path": "results/benchmarks.json", "generated_at": "T0", "note": "n"},
        )
    )


def _fixed_payload(p50s: dict[str, float]) -> dict[str, Any]:
    entries = [fake_entry(name, p50, [1, 0], 1.0) for name, p50 in p50s.items()]
    attach_comparisons(entries, PipelineConfig())
    return _env(_payload(entries))


def test_padding_comparison_matches_variant_and_batch() -> None:
    reference = _fixed_payload({"pytorch-cpu-fp32": 40.0})  # no ort-cpu-int8 row
    payload = _dynamic_payload(reference)
    assert validate_results(payload) == []
    torch_row, int8_row = (v["results"][0] for v in payload["variants"])
    assert torch_row["speedup_p50_vs_max_length"] == pytest.approx(4.0)
    assert int8_row["speedup_p50_vs_max_length"] is None  # missing in the reference
    assert int8_row["speedup_p50_vs_device_baseline"] == pytest.approx(2.0)


def test_dynamic_report_goes_to_its_own_readme_section(tmp_path: Path) -> None:
    payload = _dynamic_payload(_fixed_payload({"pytorch-cpu-fp32": 40.0}))
    markdown = render_markdown(payload)
    for needle in (
        "Workload (identical for every variant)",
        "CPU latency & throughput (dynamic padding)",
        "Speed-up vs 128-token padding (p50)",
        "4.00×",  # noqa: RUF001
        "100.00% for every variant",
        "`results/benchmarks_dynamic.json`",
    ):
        assert needle in markdown, needle
    assert "Accuracy & size" not in markdown  # accuracy is not re-tabulated

    fixed_start, fixed_end = README_MARKERS["max_length"]
    dyn_start, dyn_end = README_MARKERS["dynamic"]
    readme = tmp_path / "README.md"
    readme.write_text(
        f"{fixed_start}\nFIXED-TABLES\n{fixed_end}\n{dyn_start}\nSTALE-DYNAMIC\n{dyn_end}\n",
        encoding="utf-8",
    )
    write_report(payload, tmp_path / "benchmarks_dynamic.md", readme=readme)
    text = readme.read_text(encoding="utf-8")
    assert "FIXED-TABLES" in text and "STALE-DYNAMIC" not in text
    assert text.index(dyn_start) < text.index("Workload (identical") < text.index(dyn_end)


def test_dynamic_report_without_reference_says_so() -> None:
    markdown = render_markdown(_dynamic_payload(None))
    assert "No 128-token reference run was available." in markdown
    assert "| — |" in markdown  # the vs-128 column is explicitly empty


def test_load_padding_reference_checks_compatibility(tmp_path: Path) -> None:
    cfg = PipelineConfig(padding="dynamic", paths=PathsConfig(results_dir=tmp_path))
    assert load_padding_reference(cfg) is None  # missing file
    reference = _fixed_payload({"pytorch-cpu-fp32": 40.0})
    atomic_write_json(cfg.paths.benchmarks_json, reference)
    loaded = load_padding_reference(cfg)
    assert loaded is not None and loaded["generated_at"] == reference["generated_at"]
    assert load_padding_reference(cfg.with_overrides(revision="other")) is None
    assert load_padding_reference(cfg.with_overrides(max_seq_length=64)) is None


def test_report_command_renders_every_canonical_results_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from opt_pipeline.cli import main

    monkeypatch.chdir(tmp_path)
    paths = PathsConfig()
    atomic_write_json(paths.benchmarks_json, _fixed_payload({"pytorch-cpu-fp32": 40.0}))
    atomic_write_json(paths.benchmarks_dynamic_json, _dynamic_payload(None))
    readme = tmp_path / "README.md"
    (fixed_start, fixed_end), (dyn_start, dyn_end) = README_MARKERS.values()
    readme.write_text(f"{fixed_start}\nA\n{fixed_end}\n{dyn_start}\nB\n{dyn_end}\n", "utf-8")
    assert main(["report", "--update-readme", "--readme", str(readme)]) == 0
    text = readme.read_text(encoding="utf-8")
    assert "\nA\n" not in text and "\nB\n" not in text
    assert paths.benchmarks_md.exists()
    assert paths.benchmarks_dynamic_json.with_suffix(".md").exists()


# --------------------------------------------------------------------------
# Config round-trip (the isolated-worker protocol) and CLI mapping
# --------------------------------------------------------------------------
def test_config_dict_roundtrip() -> None:
    cfg = PipelineConfig(
        batch_sizes=(32, 1),
        exporter="dynamo",
        padding="dynamic",
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


def test_config_rejects_unknown_padding() -> None:
    with pytest.raises(ConfigError, match="padding"):
        PipelineConfig(padding="bucketed")  # type: ignore[arg-type]


def test_cli_flags_map_to_config() -> None:
    args = build_parser().parse_args(
        ["benchmark", "--variants", "ort-cpu-int8", "--batch-sizes", "8", "1", "--iters", "7"]
    )
    cfg = _config_from_args(args)
    assert args.variants == ["ort-cpu-int8"]
    assert cfg.batch_sizes == (1, 8)
    assert cfg.benchmark_iters == 7
    assert cfg.warmup_runs == PipelineConfig().warmup_runs  # untouched default
    assert cfg.padding == "max_length"
    padded = _config_from_args(build_parser().parse_args(["benchmark", "--padding", "dynamic"]))
    assert padded.padding == "dynamic"


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


@pytest.mark.slow
def test_isolated_worker_dynamic_padding(tmp_cfg: PipelineConfig, onnx_fp32: Path) -> None:
    """Real SST-2 batches in a subprocess; the workload is reproducible across processes."""
    from opt_pipeline.baseline import load_tokenizer
    from opt_pipeline.benchmark import build_workload, run_benchmarks
    from opt_pipeline.evaluate import load_sst2

    cfg = tmp_cfg.with_overrides(
        padding="dynamic", batch_sizes=[1, 8], benchmark_iters=4, warmup_runs=1
    )
    output = cfg.paths.results_dir / "smoke_dynamic.json"
    payload = run_benchmarks(cfg, ["ort-cpu-fp32"], isolate=True, evaluate=False, output=output)
    assert validate_results(json.loads(output.read_text(encoding="utf-8"))) == []
    assert "padding_reference" not in payload  # tmp results dir has no benchmarks.json
    (variant,) = payload["variants"]
    assert variant["status"] == "ok", variant.get("error")

    tokenizer, sentences = load_tokenizer(cfg), load_sst2(cfg).sentences
    for res in variant["results"]:
        w = res["workload"]
        assert (w["padding"], w["distinct_batches"]) == ("dynamic", 4)
        assert w["seq_len_max"] < cfg.max_seq_length  # SST-2 never needs 128 tokens
        assert res["sequence_length"] == w["seq_len_max"]
        if res["batch_size"] == 1:
            assert w["real_token_fraction"] == 1.0
        rebuilt = build_workload(cfg, tokenizer, res["batch_size"], sentences)
        assert w["sha256"] == rebuilt.sha256()  # same batches in parent and worker


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
        (["benchmark", "--padding", "max_length"], True, "benchmarks.json"),
        (["benchmark", "--padding", "dynamic"], True, "benchmarks_dynamic.json"),
        (
            ["benchmark", "--padding", "dynamic", "--iters", "5"],
            False,
            "benchmarks_dynamic.partial.json",
        ),
        (
            ["benchmark", "--padding", "dynamic", "--variants", "ort-cpu-int8"],
            False,
            "benchmarks_dynamic.partial.json",
        ),
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
