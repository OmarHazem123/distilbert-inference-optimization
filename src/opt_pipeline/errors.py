"""Exception hierarchy for user-facing pipeline failures.

Everything derives from :class:`PipelineError`, so the CLI can print these as
one-line errors while unexpected exceptions (real bugs) keep full tracebacks.
Each class also subclasses the closest builtin, so ``except ValueError`` and
similar handlers keep working.
"""

from __future__ import annotations


class PipelineError(Exception):
    """Base class for expected, user-actionable pipeline errors."""


class ConfigError(PipelineError, ValueError):
    """A configuration value, or combination of values, is invalid."""


class DeviceUnavailableError(PipelineError, RuntimeError):
    """A requested torch device (e.g. CUDA) is not usable."""


class ProviderUnavailableError(PipelineError, RuntimeError):
    """A requested ONNX Runtime execution provider failed to load."""


class ParityError(PipelineError, AssertionError):
    """Optimized-model outputs diverge from the reference beyond tolerance."""


class ExportError(PipelineError, RuntimeError):
    """ONNX export or conversion produced an unusable graph."""


class MissingArtifactError(PipelineError, FileNotFoundError):
    """A required artifact (e.g. an ONNX graph) has not been generated yet."""
