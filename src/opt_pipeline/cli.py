"""Command-line interface: ``opt-pipeline <command> [options]``.

Phase 1 commands:
    env      Verify the torch CUDA build and the ONNX Runtime CUDA provider.
    predict  Classify sentences with the PyTorch baseline (sanity check).
    export   Export FP32 ONNX (opset 17, dynamic axes) and verify parity.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from opt_pipeline import __version__
from opt_pipeline.config import PipelineConfig
from opt_pipeline.errors import PipelineError
from opt_pipeline.utils import atomic_write_json, get_logger

log = get_logger("opt_pipeline.cli")

DEFAULT_PREDICT_TEXTS = (
    "A gorgeous, witty, seductive movie.",
    "The plot is thin and the acting is wooden.",
)


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
    return PipelineConfig().with_overrides(
        model_id=getattr(args, "model_id", None),
        revision=getattr(args, "revision", None),
        max_seq_length=getattr(args, "max_seq_length", None),
        seed=getattr(args, "seed", None),
        attn_implementation=getattr(args, "attn_implementation", None),
        opset_version=getattr(args, "opset", None),
    )


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
    cuda_ok = torch.cuda.is_available()
    if cuda_ok:
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

    p_env = sub.add_parser("env", help="verify CUDA torch + ONNX Runtime CUDA provider")
    p_env.set_defaults(func=_cmd_env)

    p_pred = sub.add_parser("predict", parents=[common], help="classify sentences (PyTorch)")
    p_pred.add_argument("texts", nargs="*", help="sentences to classify")
    p_pred.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p_pred.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    p_pred.set_defaults(func=_cmd_predict)

    p_exp = sub.add_parser("export", parents=[common], help="export ONNX + verify parity")
    p_exp.add_argument("--output", type=Path, help="destination .onnx (default from config)")
    p_exp.add_argument("--opset", type=int, help="ONNX opset version (default 17)")
    p_exp.add_argument(
        "--parity-device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="ORT provider(s) to verify: cpu always; cuda too if 'cuda' or 'auto' with a GPU",
    )
    p_exp.add_argument("--skip-parity", action="store_true", help="export only")
    p_exp.set_defaults(func=_cmd_export)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Console-script entry point."""
    # Windows without Developer Mode cannot create symlinks; the HF cache still
    # works (it copies instead), so silence the repeated warning.
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    args = build_parser().parse_args(argv)
    try:
        code: int = args.func(args)
    except (PipelineError, FileNotFoundError) as exc:
        # Expected, actionable failures: one clean line. Anything else is a
        # bug and keeps its full traceback.
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return code


if __name__ == "__main__":
    sys.exit(main())
