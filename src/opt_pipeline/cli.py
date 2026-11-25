"""Command-line interface: ``opt-pipeline <command> [options]``.

Commands:
    env       Verify the torch CUDA build and the ONNX Runtime CUDA provider.
    predict   Classify sentences with the PyTorch baseline (sanity check).
    export    Export FP32 ONNX (opset 17, dynamic axes) and verify parity.
    quantize  Build INT8 (dynamic) and FP16 graphs and verify them on SST-2.
    benchmark Latency / throughput / memory / size / accuracy for all variants
              (``--padding dynamic``: real SST-2 sentence lengths).
    profile   torch.profiler trace + top-10 operators of the PyTorch baseline.
    evaluate  SST-2 validation accuracy / F1 / label agreement per variant.
    report    Re-render the Markdown tables (and README) from the results JSON files.

Heavy imports (torch, onnxruntime, transformers) happen inside each command so
``--help`` stays fast.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from opt_pipeline import __version__
from opt_pipeline.config import PipelineConfig
from opt_pipeline.errors import PipelineError
from opt_pipeline.utils import atomic_write_json, get_logger, utf8_stdio

log = get_logger("opt_pipeline.cli")

DEFAULT_PREDICT_TEXTS = (
    "A gorgeous, witty, seductive movie.",
    "The plot is thin and the acting is wooden.",
)
VARIANTS_HELP = (
    "variant names (default: all): pytorch-cpu-fp32 ort-cpu-fp32 ort-cpu-int8 "
    "pytorch-cuda-fp32 pytorch-cuda-fp16 ort-cuda-fp32 ort-cuda-fp16"
)
DEFAULT_README = Path("README.md")


# --------------------------------------------------------------------------
# Argument helpers
# --------------------------------------------------------------------------
def _common_parser() -> argparse.ArgumentParser:
    """Options shared by every model-touching command (all default to config)."""
    p = argparse.ArgumentParser(add_help=False)
    g = p.add_argument_group("model / config overrides")
    g.add_argument("--model-id", help="Hugging Face model id")
    g.add_argument("--revision", help="model revision (branch, tag or commit)")
    g.add_argument("--max-seq-length", type=int, help="tokenizer max length")
    g.add_argument("--seed", type=int, help="global RNG seed")
    g.add_argument("--attn-implementation", choices=["eager", "sdpa"], help="attention kernel")
    return p


def _config_from_args(args: argparse.Namespace) -> PipelineConfig:
    """Apply only the CLI flags the user actually passed on top of the defaults."""
    return PipelineConfig().with_overrides(
        model_id=getattr(args, "model_id", None),
        revision=getattr(args, "revision", None),
        max_seq_length=getattr(args, "max_seq_length", None),
        seed=getattr(args, "seed", None),
        attn_implementation=getattr(args, "attn_implementation", None),
        opset_version=getattr(args, "opset", None),
        exporter=getattr(args, "exporter", None),
        batch_sizes=getattr(args, "batch_sizes", None),
        padding=getattr(args, "padding", None),
        benchmark_iters=getattr(args, "iters", None),
        warmup_runs=getattr(args, "warmup", None),
        eval_max_samples=getattr(args, "max_samples", None),
    )


def _print_table(rows: list[dict[str, Any]], columns: list[tuple[str, str, str]]) -> None:
    """Print ``rows`` as an aligned table; ``columns`` = (key, header, format)."""
    table = [
        ["—" if row.get(key) is None else format(row[key], fmt) for key, _, fmt in columns]
        for row in rows
    ]
    headers = [header for _, header, _ in columns]
    widths = [max(len(c) for c in col) for col in zip(headers, *table, strict=True)]
    for line in (headers, *table):
        print("  ".join(f"{cell:<{w}}" for cell, w in zip(line, widths, strict=True)))


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------
def _cmd_env(args: argparse.Namespace) -> int:
    """Report the runtime stack and prove the ORT CUDA EP really loads."""
    import onnx
    import onnxruntime as ort
    import torch  # loads torch's bundled CUDA/cuDNN DLLs before any ORT CUDA session

    ok = True

    def check(label: str, passed: bool, detail: str) -> None:
        nonlocal ok
        ok &= passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {label:<28} {detail}")

    print(f"opt-pipeline {__version__} | python {sys.version.split()[0]}")
    print(f"torch {torch.__version__} | onnxruntime {ort.__version__} | onnx {onnx.__version__}")

    cuda_build = torch.version.cuda
    check("torch CUDA build", cuda_build is not None, f"CUDA {cuda_build}")
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        arch = f"sm_{major}{minor}"
        cudnn = torch.backends.cudnn.version()  # type: ignore[no-untyped-call]
        detail = f"{torch.cuda.get_device_name(0)} ({arch}), cuDNN {cudnn}"
        check("torch.cuda.is_available()", True, detail)
        check("GPU arch in torch build", arch in torch.cuda.get_arch_list(), arch)
    else:
        check("torch.cuda.is_available()", False, "no CUDA device visible to torch")

    available = ort.get_available_providers()
    check("ORT CUDA provider compiled", "CUDAExecutionProvider" in available, ", ".join(available))

    # A provider being "available" only means it is compiled in; it can still
    # fail to load its DLLs at session creation. Build a real session.
    with tempfile.TemporaryDirectory() as tmp:
        tiny = Path(tmp) / "relu.onnx"
        x = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [None])
        y = onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [None])
        graph = onnx.helper.make_graph([onnx.helper.make_node("Relu", ["x"], ["y"])], "g", [x], [y])
        model = onnx.helper.make_model(
            graph, opset_imports=[onnx.helper.make_opsetid("", 17)], ir_version=8
        )
        onnx.save(model, str(tiny))
        try:
            from opt_pipeline.export_onnx import create_ort_session

            sess = create_ort_session(tiny, device="cuda")
            check("ORT CUDA session", True, f"providers={sess.get_providers()}")
            del sess
        except Exception as exc:
            check("ORT CUDA session", False, str(exc).splitlines()[0])

    print("Environment OK" if ok else "Environment has problems (see FAIL lines above)")
    return 0 if ok else 1


def _cmd_predict(args: argparse.Namespace) -> int:
    from opt_pipeline.baseline import TorchClassifier, seed_everything

    cfg = _config_from_args(args)
    seed_everything(cfg.seed)
    clf = TorchClassifier(cfg, device=args.device, precision=args.precision)
    for pred in clf.predict(args.texts or list(DEFAULT_PREDICT_TEXTS)):
        print(f"{pred.label:<9} {pred.score:6.2%}  {pred.text}")
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    import torch

    from opt_pipeline.baseline import TorchClassifier
    from opt_pipeline.export_onnx import OrtDevice, export_onnx, verify_parity

    cfg = _config_from_args(args)
    path = export_onnx(cfg, args.output)
    if args.skip_parity:
        return 0

    devices: list[OrtDevice] = ["cpu"]
    if args.parity_device == "cuda" or (args.parity_device == "auto" and torch.cuda.is_available()):
        devices.append("cuda")

    reference = TorchClassifier(cfg, device="cpu", precision="fp32")
    reports = [
        verify_parity(cfg, path, device=d, reference=reference, raise_on_failure=False)
        for d in devices
    ]
    report_path = path.with_suffix(".parity.json")
    atomic_write_json(
        report_path, {"config": cfg.to_dict(), "reports": [r.to_dict() for r in reports]}
    )

    # Per-shape tables were already logged by verify_parity; print the verdicts.
    for report in reports:
        verdict = "PASS" if report.passed else "FAIL"
        print(f"{verdict}  {report.provider:<22} max|diff| = {report.max_abs_diff:.3e}")
    print(f"Parity report written to {report_path}")
    return 0 if all(r.passed for r in reports) else 1


def _cmd_quantize(args: argparse.Namespace) -> int:
    import torch

    from opt_pipeline.baseline import TorchClassifier
    from opt_pipeline.evaluate import load_sst2
    from opt_pipeline.quantize import (
        AgreementReport,
        ConversionResult,
        compare_with_reference,
        convert_fp16,
        quantize_int8,
    )

    cfg = _config_from_args(args)
    conversions: list[ConversionResult] = []
    if not args.skip_int8:
        conversions.append(quantize_int8(cfg))
    if not args.skip_fp16:
        conversions.append(convert_fp16(cfg))

    reports: list[AgreementReport] = []
    if not args.no_verify and conversions:
        split = load_sst2(cfg, max_samples=args.verify_samples)
        reference = TorchClassifier(cfg, device="cpu", precision="fp32")
        for conv in conversions:
            if conv.precision == "fp16" and not torch.cuda.is_available():
                print("FP16 graph targets the CUDA EP; no GPU available, verification skipped")
                continue
            reports.append(
                compare_with_reference(
                    cfg,
                    Path(conv.output),
                    precision=conv.precision,
                    device="cuda" if conv.precision == "fp16" else "cpu",
                    sentences=split.sentences,
                    reference=reference,
                )
            )

    summary_path = cfg.paths.onnx_dir / "quantization.json"
    atomic_write_json(
        summary_path,
        {
            "config": cfg.to_dict(),
            "conversions": [c.to_dict() for c in conversions],
            "verification": [r.to_dict() for r in reports],
        },
    )
    for conv in conversions:
        print(conv.summary())
    for rep in reports:
        print(rep.summary())
    print(f"Summary written to {summary_path}")
    return 0 if all(r.passed for r in reports) else 1


def is_canonical_benchmark(args: argparse.Namespace, cfg: PipelineConfig) -> bool:
    """True only for a complete, default-configuration benchmark (either padding mode).

    Only such a run may overwrite its tracked results file
    (``results/benchmarks.json`` or ``results/benchmarks_dynamic.json``) or its
    README tables -- a ``--variants`` subset or a quick ``--iters 5`` run would
    otherwise silently replace the published 7-variant results.
    """
    return (
        not args.variants
        and not args.skip_eval
        and not args.no_isolate
        and cfg.with_overrides(padding="max_length") == PipelineConfig()
    )


def benchmark_output_path(args: argparse.Namespace, cfg: PipelineConfig) -> Path:
    """``--output`` if given; else the canonical file, or a ``.partial`` sibling."""
    if args.output is not None:
        output: Path = args.output
        return output
    canonical = cfg.paths.benchmark_results_json(cfg.padding)
    if is_canonical_benchmark(args, cfg):
        return canonical
    return canonical.with_name(f"{canonical.stem}.partial.json")


def _cmd_benchmark(args: argparse.Namespace) -> int:
    from opt_pipeline.benchmark import run_benchmarks, write_report

    cfg = _config_from_args(args)
    canonical = is_canonical_benchmark(args, cfg)
    output = benchmark_output_path(args, cfg)
    readme = args.readme if args.update_readme and canonical else None
    payload = run_benchmarks(
        cfg,
        args.variants,
        isolate=not args.no_isolate,
        evaluate=not args.skip_eval,
        output=output,
    )
    md_path = output.with_suffix(".md")
    markdown = write_report(payload, md_path, readme=readme)
    print(markdown)
    print(f"Results: {output} | tables: {md_path}")
    if args.update_readme and not canonical:
        print(
            "note: README left unchanged -- only a complete default run (all variants, "
            "default config, evaluation on, isolated) updates the published tables"
        )
    failed = [v["name"] for v in payload["variants"] if v["status"] == "error"]
    if failed:
        print(f"error: variant(s) failed: {', '.join(failed)} (see artifacts/benchmark_logs/)")
        return 1
    return 0


def _cmd_profile(args: argparse.Namespace) -> int:
    import torch

    from opt_pipeline.profile_bottlenecks import format_top_ops, profile_model

    cfg = _config_from_args(args)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    report = profile_model(
        cfg,
        device=device,
        precision=args.precision,
        batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        active_steps=args.steps,
        row_limit=args.row_limit,
    )
    if args.full_table:
        print(report.table)
    print(format_top_ops(report))
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from opt_pipeline.evaluate import evaluate_variants

    cfg = _config_from_args(args)
    payload = evaluate_variants(cfg, args.variants)
    output = args.output or cfg.paths.evaluation_json
    atomic_write_json(output, payload)
    print(f"SST-2 {cfg.dataset_split}: {payload['n']} sentences")
    _print_table(
        payload["variants"],
        [
            ("name", "variant", ""),
            ("accuracy", "accuracy", ".4f"),
            ("f1", "f1", ".4f"),
            ("label_agreement_vs_reference", "agreement", ".2%"),
            ("accuracy_drop_vs_reference", "acc. drop", "+.4f"),
            ("status", "status", ""),
        ],
    )
    print(f"Written to {output}")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    from opt_pipeline.benchmark import load_results, write_report

    cfg = _config_from_args(args)
    if args.input is not None:
        sources: list[Path] = [args.input]
    else:  # every canonical results file that exists (fixed first, then dynamic)
        canonical = [cfg.paths.benchmark_results_json(p) for p in ("max_length", "dynamic")]
        sources = [p for p in canonical if p.exists()] or canonical[:1]
    readme = args.readme if args.update_readme else None
    for source in sources:
        print(write_report(load_results(source), source.with_suffix(".md"), readme=readme))
    return 0


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level parser (exposed for tests)."""
    parser = argparse.ArgumentParser(
        prog="opt-pipeline",
        description="Inference optimization & benchmarking pipeline for DistilBERT SST-2.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    common = _common_parser()

    p = sub.add_parser("env", help="verify CUDA torch + ONNX Runtime CUDA provider")
    p.set_defaults(func=_cmd_env)

    p = sub.add_parser("predict", parents=[common], help="classify sentences (PyTorch)")
    p.add_argument("texts", nargs="*", help="sentences to classify")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    p.set_defaults(func=_cmd_predict)

    p = sub.add_parser("export", parents=[common], help="export ONNX + verify parity")
    p.add_argument("--output", type=Path, help="destination .onnx (default from config)")
    p.add_argument("--opset", type=int, help="ONNX opset version (default 17)")
    p.add_argument(
        "--exporter",
        choices=["torchscript", "dynamo"],
        help="ONNX exporter (default torchscript; dynamo builds at 18 and down-converts)",
    )
    p.add_argument(
        "--parity-device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="ORT provider(s) to verify: cpu always; cuda too if 'cuda' or 'auto' with a GPU",
    )
    p.add_argument("--skip-parity", action="store_true", help="export only")
    p.set_defaults(func=_cmd_export)

    p = sub.add_parser("quantize", parents=[common], help="INT8 dynamic + FP16 conversion")
    p.add_argument("--skip-int8", action="store_true", help="do not build the INT8 graph")
    p.add_argument("--skip-fp16", action="store_true", help="do not build the FP16 graph")
    p.add_argument("--no-verify", action="store_true", help="skip SST-2 agreement check")
    p.add_argument(
        "--verify-samples", type=int, help="SST-2 sentences for verification (default: all)"
    )
    p.set_defaults(func=_cmd_quantize)

    p = sub.add_parser("benchmark", parents=[common], help="benchmark all variants")
    p.add_argument("--variants", nargs="+", metavar="NAME", help=VARIANTS_HELP)
    p.add_argument("--batch-sizes", type=int, nargs="+", help="batch sizes (default 1 8 32)")
    p.add_argument(
        "--padding",
        choices=["max_length", "dynamic"],
        help="timed batches: fixed 128-token padding (default) or real SST-2 sentences padded "
        "per batch (writes results/benchmarks_dynamic.json)",
    )
    p.add_argument("--iters", type=int, help="timed iterations per cell (default 100)")
    p.add_argument("--warmup", type=int, help="untimed warmup iterations (default 10)")
    p.add_argument("--max-samples", type=int, help="cap SST-2 evaluation sentences")
    p.add_argument("--skip-eval", action="store_true", help="skip SST-2 accuracy / F1")
    p.add_argument("--no-isolate", action="store_true", help="run variants in-process")
    p.add_argument(
        "--output",
        type=Path,
        help="results JSON (default results/benchmarks.json or benchmarks_dynamic.json)",
    )
    p.add_argument("--update-readme", action="store_true", help="splice tables into README")
    p.add_argument("--readme", type=Path, default=DEFAULT_README, help=argparse.SUPPRESS)
    p.set_defaults(func=_cmd_benchmark)

    p = sub.add_parser("profile", parents=[common], help="torch.profiler top-10 operators")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--steps", type=int, default=5, help="active (recorded) steps")
    p.add_argument("--warmup-steps", type=int, default=3)
    p.add_argument("--row-limit", type=int, default=10)
    p.add_argument("--full-table", action="store_true", help="also print torch's full table")
    p.set_defaults(func=_cmd_profile)

    p = sub.add_parser("evaluate", parents=[common], help="SST-2 accuracy / F1 per variant")
    p.add_argument("--variants", nargs="+", metavar="NAME", help=VARIANTS_HELP)
    p.add_argument("--max-samples", type=int, help="cap SST-2 evaluation sentences")
    p.add_argument("--output", type=Path, help="JSON output (default results/evaluation.json)")
    p.set_defaults(func=_cmd_evaluate)

    p = sub.add_parser("report", help="re-render Markdown tables from the results JSON")
    p.add_argument(
        "--input",
        type=Path,
        help="results JSON (default: benchmarks.json and, if present, benchmarks_dynamic.json)",
    )
    p.add_argument("--update-readme", action="store_true", help="splice tables into README")
    p.add_argument("--readme", type=Path, default=DEFAULT_README, help=argparse.SUPPRESS)
    p.set_defaults(func=_cmd_report)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Console-script entry point."""
    # Windows without Developer Mode cannot create symlinks; the HF cache still
    # works (it copies instead), so silence the repeated warning.
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    args = build_parser().parse_args(argv)
    try:
        with utf8_stdio():  # reports contain non-cp1252 symbols (delta, check marks)
            code: int = args.func(args)
    except (PipelineError, FileNotFoundError) as exc:
        # Expected, actionable failures: one clean line. Anything else is a
        # bug and keeps its full traceback.
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return code


if __name__ == "__main__":
    sys.exit(main())
