#!/usr/bin/env python
"""Cross-platform task runner -- a ``make`` replacement that needs only Python.

Uses only the standard library, so it runs with any Python >= 3.9 (including
the system interpreter) and delegates real work to ``uv`` / the project CLI.

Usage:
    python tasks.py --list                  # show available tasks
    python tasks.py setup doctor            # run several tasks in order
    python tasks.py export -- --device cuda # forward extra args to the last task
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parent

TaskFn = Callable[[list[str]], None]
TASKS: dict[str, tuple[TaskFn, str]] = {}


def task(description: str) -> Callable[[TaskFn], TaskFn]:
    """Register a function as a task. Underscores in the name become dashes."""

    def decorator(fn: TaskFn) -> TaskFn:
        TASKS[fn.__name__.replace("_", "-")] = (fn, description)
        return fn

    return decorator


def _uv() -> str:
    exe = shutil.which("uv")
    if exe is None:
        sys.exit("error: `uv` not found on PATH -- install from https://docs.astral.sh/uv/")
    return exe


def run(*cmd: str) -> None:
    """Echo and run a command from the repo root; abort on non-zero exit."""
    print(f"\n$ {' '.join(cmd)}", flush=True)
    code = subprocess.call(list(cmd), cwd=ROOT)
    if code != 0:
        sys.exit(code)


def uv_run(*args: str) -> None:
    """Run a command inside the project's locked ``.venv``."""
    run(_uv(), "run", *args)


# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------
@task("Create/refresh .venv from uv.lock (torch from the CUDA 12.6 index)")
def setup(extra: list[str]) -> None:
    run(_uv(), "sync", *extra)


@task("Verify torch CUDA build, GPU visibility and ONNX Runtime CUDA provider")
def doctor(extra: list[str]) -> None:
    uv_run("opt-pipeline", "env", *extra)


# --------------------------------------------------------------------------
# Pipeline stages
# --------------------------------------------------------------------------
@task("Sanity-check the PyTorch baseline on sample sentences")
def predict(extra: list[str]) -> None:
    uv_run("opt-pipeline", "predict", *extra)


@task("Export FP32 ONNX (opset 17, dynamic axes) and verify parity vs PyTorch")
def export(extra: list[str]) -> None:
    uv_run("opt-pipeline", "export", *extra)


# --------------------------------------------------------------------------
# Quality
# --------------------------------------------------------------------------
_LINT_PATHS = ("src", "tests", "tasks.py")


@task("Lint and check formatting with ruff")
def lint(extra: list[str]) -> None:
    uv_run("ruff", "check", *_LINT_PATHS, *extra)
    uv_run("ruff", "format", "--check", *_LINT_PATHS)


@task("Auto-format and apply safe lint fixes")
def fmt(extra: list[str]) -> None:
    uv_run("ruff", "format", *_LINT_PATHS)
    uv_run("ruff", "check", "--fix", *_LINT_PATHS, *extra)


@task("Static type-check with mypy (strict)")
def typecheck(extra: list[str]) -> None:
    uv_run("mypy", *extra)


@task("Run the pytest suite")
def test(extra: list[str]) -> None:
    uv_run("pytest", *extra)


@task("Run lint + typecheck + test")
def check(extra: list[str]) -> None:
    lint([])
    typecheck([])
    test(extra)


# --------------------------------------------------------------------------
# Housekeeping
# --------------------------------------------------------------------------
@task("Delete generated artifacts and tool caches (keeps models/, data/, .venv)")
def clean(extra: list[str]) -> None:
    targets = [ROOT / name for name in ("artifacts", ".pytest_cache", ".ruff_cache", ".mypy_cache")]
    targets.append(ROOT / "__pycache__")
    for base in (ROOT / "src", ROOT / "tests"):  # never crawl .venv / models
        targets += list(base.rglob("__pycache__"))
    for path in targets:
        if path.exists():
            print(f"removing {path.relative_to(ROOT)}")
            shutil.rmtree(path)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def _print_tasks() -> None:
    width = max(map(len, TASKS))
    print("Available tasks:")
    for name, (_, description) in TASKS.items():
        print(f"  {name:<{width}}  {description}")


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    extra: list[str] = []
    if "--" in argv:
        split = argv.index("--")
        argv, extra = argv[:split], argv[split + 1 :]

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("tasks", nargs="*", metavar="TASK", help="task(s) to run in order")
    parser.add_argument("-l", "--list", action="store_true", help="list available tasks")
    args = parser.parse_args(argv)

    if args.list or not args.tasks:
        _print_tasks()
        return

    unknown = [t for t in args.tasks if t not in TASKS]
    if unknown:
        _print_tasks()
        sys.exit(f"\nerror: unknown task(s): {', '.join(unknown)}")

    for i, name in enumerate(args.tasks):
        fn, _ = TASKS[name]
        # Extra args after `--` go to the last task only.
        fn(extra if i == len(args.tasks) - 1 else [])


if __name__ == "__main__":
    main()
