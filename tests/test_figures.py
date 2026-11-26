"""README figures: every chart renders from the tracked results, deterministically.

The inputs are the committed ``results/*.json`` files (read-only); all output
goes to pytest temp directories.
"""

from __future__ import annotations

import copy
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from opt_pipeline.cli import build_parser
from opt_pipeline.config import PathsConfig, PipelineConfig
from opt_pipeline.errors import MissingArtifactError, PipelineError
from opt_pipeline.figures import (
    DEFAULT_FIGURES_DIR,
    FIGURE_NAMES,
    LIGHT,
    THEMES,
    FigureInputs,
    _fig_latency_padding,
    _is_shipped,
    load_figure_inputs,
    render_figures,
    render_png,
)

pytest.importorskip("matplotlib")

RESULTS = Path(__file__).resolve().parents[1] / "results"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _cfg(results_dir: Path) -> PipelineConfig:
    return replace(PipelineConfig(), paths=PathsConfig(results_dir=results_dir))


@pytest.fixture(scope="module")
def inputs() -> FigureInputs:
    return load_figure_inputs(_cfg(RESULTS))


def test_render_figures_writes_every_figure_in_both_themes(
    inputs: FigureInputs, tmp_path: Path
) -> None:
    written = render_figures(inputs, tmp_path)
    expected = {f"{name}-{theme.name}.png" for name in FIGURE_NAMES for theme in THEMES}
    assert {p.name for p in written} == expected
    assert {p.name for p in tmp_path.iterdir()} == expected  # no stray temp files
    for path in written:
        assert path.read_bytes().startswith(PNG_SIGNATURE)


def test_figures_are_byte_identical_across_renders(inputs: FigureInputs) -> None:
    first = render_png("latency_padding", inputs, LIGHT)
    assert render_png("latency_padding", inputs, LIGHT) == first
    # No software/version or timestamp metadata that would churn the committed PNGs.
    assert b"matplotlib" not in first.lower()


def test_unknown_figure_name_is_rejected(inputs: FigureInputs) -> None:
    with pytest.raises(PipelineError, match="unknown figure"):
        render_png("nope", inputs, LIGHT)


def test_failed_variants_are_left_out(inputs: FigureInputs) -> None:
    import matplotlib as mpl

    dynamic = copy.deepcopy(inputs.dynamic)
    failed = dynamic["variants"][0]
    failed["status"] = "error"
    with mpl.rc_context({"font.family": "DejaVu Sans"}):
        fig = _fig_latency_padding(replace(inputs, dynamic=dynamic), LIGHT)
    labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    assert failed["name"] not in labels
    assert "ort-cpu-int8" in labels


def test_exactly_one_sweep_row_is_the_shipped_int8_setting(inputs: FigureInputs) -> None:
    # Guards the highlight: if the QuantizationConfig defaults change, the sweep
    # (and its figure) must be re-run so the shipped setting is still in it.
    shipped = [row for row in inputs.sweep["rows"] if _is_shipped(row["settings"])]
    assert len(shipped) == 1


def test_load_figure_inputs_requires_the_int8_sweep(tmp_path: Path) -> None:
    for name in ("benchmarks.json", "benchmarks_dynamic.json"):
        shutil.copy(RESULTS / name, tmp_path / name)
    with pytest.raises(MissingArtifactError, match="int8_sweep"):
        load_figure_inputs(_cfg(tmp_path))


def test_cli_figures_defaults_to_docs_figures() -> None:
    args = build_parser().parse_args(["figures"])
    assert args.output_dir == DEFAULT_FIGURES_DIR == Path("docs/figures")
