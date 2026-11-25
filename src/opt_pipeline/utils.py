"""Small, dependency-free helpers shared across pipeline stages."""

from __future__ import annotations

import io
import json
import logging
import os
import sys
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    """Return a module logger, configuring the root handler once.

    Third-party libraries stay at WARNING (``httpx`` would otherwise log every
    Hub request at INFO); only the ``opt_pipeline`` namespace logs at INFO.

    Args:
        name: Logger name, typically ``__name__``.
    """
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=logging.WARNING, format=_LOG_FORMAT, datefmt="%H:%M:%S")
        logging.getLogger("opt_pipeline").setLevel(logging.INFO)
    return logging.getLogger(name)


@contextmanager
def atomic_output(path: str | os.PathLike[str]) -> Iterator[Path]:
    """Yield a temporary sibling path that atomically replaces ``path`` on success.

    The temp file lives in the *same directory* as the destination so that
    :func:`os.replace` is a same-volume rename (atomic on both POSIX and
    Windows/NTFS). If the body raises, the temp file is removed and the
    existing destination -- if any -- is left untouched, so readers never
    observe a half-written artifact.

    The original suffix is kept (``model.<hex>.tmp.onnx``) for tools that
    infer the format from the file extension.

    Args:
        path: Final destination path. Parent directories are created.

    Yields:
        Temporary path for the caller to write to.

    Example:
        >>> with atomic_output("artifacts/onnx/model.onnx") as tmp:  # doctest: +SKIP
        ...     torch.onnx.export(model, args, tmp)
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f"{dest.stem}.{uuid.uuid4().hex[:8]}.tmp{dest.suffix}")
    try:
        yield tmp
        if not tmp.exists():
            raise FileNotFoundError(f"atomic_output body did not create {tmp}")
        tmp.replace(dest)  # os.replace semantics: atomic, overwrites dest
    finally:
        tmp.unlink(missing_ok=True)


def atomic_write_text(path: str | os.PathLike[str], text: str) -> Path:
    """Atomically write UTF-8 text with LF line endings to ``path``.

    ``newline="\\n"`` disables text-mode newline translation, which would
    otherwise write CRLF on Windows: generated results must be byte-identical
    on every platform (and match the LF form ``.gitattributes`` stores).
    Returns the destination path.
    """
    dest = Path(path)
    with atomic_output(dest) as tmp:
        tmp.write_text(text, encoding="utf-8", newline="\n")
    return dest


def atomic_write_json(path: str | os.PathLike[str], payload: Any, *, indent: int = 2) -> Path:
    """Atomically serialise ``payload`` as pretty-printed JSON."""
    return atomic_write_text(path, json.dumps(payload, indent=indent, sort_keys=False) + "\n")


@contextmanager
def utf8_stdio() -> Iterator[None]:
    """Temporarily switch stdout/stderr to UTF-8 (an in-process ``PYTHONUTF8=1``).

    On Windows, Python encodes console/pipe output with the ANSI code page
    (cp1252), so printing e.g. the dynamo exporter's emoji progress markers or
    the report's delta / multiplication / check-mark symbols raises
    ``UnicodeEncodeError``. Streams are
    reconfigured to UTF-8 with ``backslashreplace`` and restored afterwards.
    Streams that are already UTF-8, or are not plain ``TextIOWrapper``s (e.g.
    pytest capture objects), are left alone.
    """
    restore: list[tuple[io.TextIOWrapper, str, str | None]] = []
    for stream in (sys.stdout, sys.stderr):
        encoding = (getattr(stream, "encoding", None) or "").lower().replace("-", "")
        if encoding != "utf8" and isinstance(stream, io.TextIOWrapper):
            restore.append((stream, stream.encoding, stream.errors))
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    try:
        yield
    finally:
        for stream, encoding, errors in restore:
            stream.reconfigure(encoding=encoding, errors=errors)


def file_size_mb(path: str | os.PathLike[str]) -> float:
    """Return on-disk size in mebibytes (MiB), including ONNX external data.

    If an ``<name>.data`` / ``<name>.onnx.data`` sidecar exists (ONNX external
    tensor storage for >2 GB models), it is counted too.
    """
    p = Path(path)
    total = p.stat().st_size
    for sidecar in (p.with_name(p.name + ".data"), p.with_suffix(".data")):
        if sidecar.exists() and sidecar != p:
            total += sidecar.stat().st_size
    return total / (1024 * 1024)
