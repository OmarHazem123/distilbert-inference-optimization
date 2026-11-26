"""README figures rendered from the tracked results files.

Every image in ``docs/figures/`` is drawn from ``results/benchmarks.json``
(fixed 128-token padding), ``results/benchmarks_dynamic.json`` (padding per
batch) and ``results/int8_sweep.json``, so the charts cannot disagree with the
tables. Each figure is written twice, for GitHub's light and dark themes; the
README picks one with ``<picture>``.

Output is deterministic for a given matplotlib / FreeType build: the bundled
DejaVu fonts, fixed sizes and DPI, and no timestamp or software metadata. A
re-run on unchanged results rewrites byte-identical PNGs.

matplotlib is a development dependency (``uv sync`` installs it). It is
imported lazily, so the benchmark workers and the other CLI commands never load
it, and only the object-oriented API is used (no pyplot global state, no GUI
backend).
"""

from __future__ import annotations

import io
import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from opt_pipeline.config import PipelineConfig, QuantizationConfig
from opt_pipeline.errors import MissingArtifactError, PipelineError
from opt_pipeline.utils import atomic_output, get_logger

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = [
    "DEFAULT_FIGURES_DIR",
    "FIGURE_NAMES",
    "THEMES",
    "FigureInputs",
    "Theme",
    "figure_path",
    "load_figure_inputs",
    "render_figures",
    "render_png",
]

log = get_logger(__name__)

DEFAULT_FIGURES_DIR = Path("docs/figures")
DPI = 200
WIDTH_IN = 8.0
HERO_BATCH_SIZE = 1  # the latency figure shows online (batch-1) serving


# --------------------------------------------------------------------------
# Themes: one palette per GitHub colour scheme. Variant hues are slots 1-3 of
# the validated categorical palette (all-pairs CVD-safe in both modes); the
# accent / neutral pair is for figures whose series are not variants.
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Theme:
    """Colours for one rendering of every figure."""

    name: str
    surface: str
    ink: str
    ink_secondary: str
    ink_muted: str
    grid: str
    baseline: str
    pytorch: str  # PyTorch variants
    ort: str  # ONNX Runtime FP32 / FP16 variants
    int8: str  # ONNX Runtime INT8
    accent: str  # padding per batch, real tokens, the shipped INT8 setting
    neutral: str  # 128-token padding, the other INT8 settings
    padding: str  # padding share of a stacked bar


LIGHT = Theme(
    name="light",
    surface="#fcfcfb",
    ink="#0b0b0b",
    ink_secondary="#52514e",
    ink_muted="#898781",
    grid="#e1e0d9",
    baseline="#c3c2b7",
    pytorch="#2a78d6",
    ort="#eb6834",
    int8="#1baf7a",
    accent="#6250d6",
    neutral="#898781",
    padding="#d6d4cc",
)
DARK = Theme(
    name="dark",
    surface="#1a1a19",
    ink="#f0efec",
    ink_secondary="#c3c2b7",
    ink_muted="#898781",
    grid="#2c2c2a",
    baseline="#4a4945",
    pytorch="#3987e5",
    ort="#d95926",
    int8="#199e70",
    accent="#9085e9",
    neutral="#898781",
    padding="#4a4945",
)
THEMES: tuple[Theme, ...] = (LIGHT, DARK)

# Both fonts ship inside matplotlib, so the output never depends on system fonts.
_SANS = "DejaVu Sans"
_MONO = "DejaVu Sans Mono"
_ENGINE = {"pytorch": "PyTorch", "onnxruntime": "ORT"}
_DEVICES = (("cpu", "CPU"), ("cuda", "GPU"))


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class FigureInputs:
    """The three results payloads the figures are drawn from."""

    fixed: dict[str, Any]  # results/benchmarks.json (128-token padding)
    dynamic: dict[str, Any]  # results/benchmarks_dynamic.json (padding per batch)
    sweep: dict[str, Any]  # results/int8_sweep.json


def load_figure_inputs(cfg: PipelineConfig | None = None) -> FigureInputs:
    """Load and validate the canonical results files named by ``cfg.paths``."""
    from opt_pipeline.benchmark import load_results  # heavy: imports torch

    paths = (cfg or PipelineConfig()).paths
    fixed = load_results(paths.benchmarks_json)
    dynamic = load_results(paths.benchmarks_dynamic_json)
    for payload, path in ((fixed, paths.benchmarks_json), (dynamic, paths.benchmarks_dynamic_json)):
        if not _ok_variants(payload):
            raise PipelineError(f"{path} has no successfully measured variant to plot")
    sweep_path = paths.int8_sweep_json
    if not sweep_path.exists():
        raise MissingArtifactError(
            f"{sweep_path} not found -- run `uv run python scripts/int8_sweep.py` first"
        )
    sweep: dict[str, Any] = json.loads(sweep_path.read_text(encoding="utf-8"))
    rows = sweep.get("rows")
    reference = sweep.get("reference")
    if (
        not isinstance(rows, list)
        or not rows
        or not isinstance(reference, dict)
        or not isinstance(reference.get("accuracy"), float)
        or any(not isinstance(row.get("accuracy"), float) for row in rows)
    ):
        raise PipelineError(f"{sweep_path} is not an INT8 sweep results file")
    return FigureInputs(fixed=fixed, dynamic=dynamic, sweep=sweep)


def _ok_variants(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [v for v in payload.get("variants", []) if v.get("status") == "ok"]


def _cells(variant: dict[str, Any] | None) -> dict[int, dict[str, Any]]:
    """Per-batch-size results of one variant (empty for a missing variant)."""
    if variant is None:
        return {}
    return {int(r["batch_size"]): r for r in variant.get("results") or []}


def _hardware(payload: dict[str, Any]) -> str:
    """'Intel Core i5-10300H · NVIDIA GeForce GTX 1650 Ti' from a payload's environment."""
    env = payload.get("environment") or {}
    cpu = re.sub(r"\((?:R|TM)\)", "", str(env.get("cpu") or "unknown CPU"))
    cpu = " ".join(re.sub(r"\s+CPU\s*@.*$", "", cpu).split())
    gpu = (env.get("gpu") or {}).get("name")
    return f"{cpu} · {gpu}" if gpu else cpu


# --------------------------------------------------------------------------
# Drawing helpers
# --------------------------------------------------------------------------
def _new_figure(height_in: float, theme: Theme) -> Figure:
    from matplotlib.figure import Figure

    return Figure(figsize=(WIDTH_IN, height_in), dpi=DPI, facecolor=theme.surface)


def _frac(fig: Figure, inches_from_top: float) -> float:
    """Figure-fraction y coordinate ``inches_from_top`` below the top edge."""
    return 1.0 - inches_from_top / float(fig.get_figheight())


def _header(fig: Figure, theme: Theme, title: str, subtitle: str) -> None:
    fig.text(
        0.03,
        _frac(fig, 0.16),
        title,
        ha="left",
        va="top",
        fontsize=13,
        fontweight="bold",
        color=theme.ink,
    )
    fig.text(
        0.03, _frac(fig, 0.46), subtitle, ha="left", va="top", fontsize=9, color=theme.ink_secondary
    )


def _footer(fig: Figure, theme: Theme, text: str) -> None:
    fig.text(
        0.03,
        0.15 / fig.get_figheight(),
        text,
        ha="left",
        va="bottom",
        fontsize=7.5,
        color=theme.ink_muted,
    )


def _style_axes(ax: Axes, theme: Theme, *, grid_axis: Literal["x", "y"], baseline: str) -> None:
    """Recessive chrome: hairline grid, one baseline spine, no tick marks."""
    ax.set_facecolor(theme.surface)
    for side, spine in ax.spines.items():
        spine.set_visible(side == baseline)
        spine.set_color(theme.baseline)
        spine.set_linewidth(0.8)
    ax.tick_params(length=0, colors=theme.ink_muted, labelcolor=theme.ink_secondary)
    ax.grid(axis=grid_axis, color=theme.grid, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(theme.ink_secondary)
    ax.yaxis.label.set_color(theme.ink_secondary)


def _row_labels(
    ax: Axes, theme: Theme, rows: Sequence[tuple[float, str, bool]], *, column_pt: float
) -> None:
    """Left-aligned y labels in a ``column_pt``-wide column; headings in bold sans."""
    ax.set_yticks([y for y, _, _ in rows], [label for _, label, _ in rows])
    ax.tick_params(axis="y", pad=column_pt)
    for tick, (_, _, heading) in zip(ax.get_yticklabels(), rows, strict=True):
        tick.set_horizontalalignment("left")
        if heading:
            tick.set_fontweight("bold")
            tick.set_color(theme.ink)
        else:
            tick.set_fontfamily(_MONO)


def _series_style(variant: dict[str, Any], theme: Theme) -> dict[str, Any]:
    """Colour = engine (INT8 has its own hue); dashes + square markers = FP16."""
    backend, precision = variant["backend"], variant["precision"]
    if precision == "int8":
        color = theme.int8
    else:
        color = theme.pytorch if backend == "pytorch" else theme.ort
    fp16 = precision == "fp16"
    return {
        "color": color,
        "linestyle": (0, (4, 2)) if fp16 else "solid",
        "marker": {"fp16": "s", "int8": "D"}.get(precision, "o"),
        "label": f"{_ENGINE.get(backend, backend)} {precision.upper()}",
    }


# --------------------------------------------------------------------------
# Figure 1: batch-1 latency, 128-token padding vs padding per batch
# --------------------------------------------------------------------------
def _fig_latency_padding(data: FigureInputs, theme: Theme) -> Figure:
    from matplotlib.lines import Line2D
    from matplotlib.ticker import NullLocator

    fixed_by_name = {v["name"]: v for v in _ok_variants(data.fixed)}
    rows: list[tuple[float, str, bool]] = []
    points: list[tuple[float, float, float, float]] = []  # y, fixed p50, dynamic p50, speed-up
    y = 0.0
    for device, heading in _DEVICES:
        group = []
        for variant in _ok_variants(data.dynamic):
            if variant["device"] != device:
                continue
            dyn = _cells(variant).get(HERO_BATCH_SIZE)
            ref = _cells(fixed_by_name.get(variant["name"])).get(HERO_BATCH_SIZE)
            if dyn is not None and ref is not None:
                group.append((variant["name"], ref, dyn))
        if not group:
            continue
        rows.append((y, heading, True))
        y += 1.0
        for name, ref, dyn in group:
            fixed_p50 = float(ref["latency_ms"]["p50"])
            dyn_p50 = float(dyn["latency_ms"]["p50"])
            speedup = dyn.get("speedup_p50_vs_max_length") or fixed_p50 / dyn_p50
            rows.append((y, name, False))
            points.append((y, fixed_p50, dyn_p50, float(speedup)))
            y += 1.0
        y += 0.35
    if not points:
        raise PipelineError("no variant has batch-1 results in both padding modes")

    fig = _new_figure(0.95 + 0.34 * len(rows) + 0.9, theme)
    fig.subplots_adjust(
        left=0.215, right=0.86, top=_frac(fig, 1.05), bottom=0.75 / fig.get_figheight()
    )
    ax = fig.add_subplot()
    _style_axes(ax, theme, grid_axis="x", baseline="bottom")
    ax.set_xscale("log")
    lo = min(p[2] for p in points) / 1.9
    hi = max(p[1] for p in points) * 1.35
    ax.set_xlim(lo, hi)
    ticks = [m * 10**e for e in range(-1, 5) for m in (1, 2, 5) if lo <= m * 10**e <= hi]
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel(f"p50 latency at batch {HERO_BATCH_SIZE}, ms (log scale)")
    ax.set_ylim(y - 0.35 - 0.5, -0.6)
    _row_labels(ax, theme, rows, column_pt=112)

    for py, fixed_p50, dyn_p50, speedup in points:
        ax.plot(
            [dyn_p50, fixed_p50],
            [py, py],
            color=theme.neutral,
            alpha=0.45,
            linewidth=2,
            solid_capstyle="round",
            zorder=1,
        )
        ax.annotate(
            f"{dyn_p50:.1f} ms",
            (dyn_p50, py),
            xytext=(-9, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=8.5,
            color=theme.ink_secondary,
        )
        ratio = f"{speedup:.2f}×"  # noqa: RUF001
        ax.annotate(
            ratio,
            (1.0, py),
            xycoords=("axes fraction", "data"),
            xytext=(14, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            color=theme.ink,
        )
    ys = [p[0] for p in points]
    for xs, color in (
        ([p[1] for p in points], theme.neutral),
        ([p[2] for p in points], theme.accent),
    ):
        ax.scatter(xs, ys, s=70, color=color, edgecolors=theme.surface, linewidths=1.6, zorder=3)
    ax.annotate(
        "speed-up",
        (1.0, rows[0][0]),
        xycoords=("axes fraction", "data"),
        xytext=(14, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=8.5,
        color=theme.ink_muted,
    )

    _header(
        fig,
        theme,
        "Padding each batch to its longest sentence cuts batch-1 latency",
        "DistilBERT SST-2, p50 latency; lower is better. The two padding modes were measured "
        "in separate runs.",
    )
    handles = [
        Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markersize=8,
            color=c,
            markeredgecolor=theme.surface,
            label=label,
        )
        for c, label in (
            (theme.neutral, "Padded to 128 tokens"),
            (theme.accent, "Padded per batch (real SST-2 lengths)"),
        )
    ]
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.025, _frac(fig, 0.7)),
        ncols=2,
        frameon=False,
        labelcolor=theme.ink_secondary,
        handletextpad=0.3,
        columnspacing=1.6,
    )
    _footer(
        fig, theme, f"results/benchmarks.json + benchmarks_dynamic.json · {_hardware(data.dynamic)}"
    )
    return fig


# --------------------------------------------------------------------------
# Figure 2: throughput by batch size, both padding modes
# --------------------------------------------------------------------------
def _fig_throughput(data: FigureInputs, theme: Theme) -> Figure:
    from matplotlib.lines import Line2D

    columns = (
        (data.fixed, "Padded to 128 tokens"),
        (data.dynamic, "Padded per batch (real lengths)"),
    )
    batch_sizes = sorted(
        {bs for payload, _ in columns for v in _ok_variants(payload) for bs in _cells(v)}
    )
    xpos = {bs: i for i, bs in enumerate(batch_sizes)}
    devices = [
        (d, label)
        for d, label in _DEVICES
        if any(v["device"] == d for p, _ in columns for v in _ok_variants(p))
    ]

    fig = _new_figure(1.05 + 2.45 * len(devices) + 0.55, theme)
    fig.subplots_adjust(
        left=0.1,
        right=0.97,
        top=_frac(fig, 1.45),
        bottom=0.75 / fig.get_figheight(),
        hspace=0.42,
        wspace=0.08,
    )
    axes = fig.subplots(len(devices), len(columns), sharey="row", squeeze=False)
    legend: dict[str, Line2D] = {}
    for r, (device, device_label) in enumerate(devices):
        row_max = 0.0
        for c, (payload, column_title) in enumerate(columns):
            ax = axes[r][c]
            _style_axes(ax, theme, grid_axis="y", baseline="bottom")
            for variant in _ok_variants(payload):
                if variant["device"] != device:
                    continue
                cells = _cells(variant)
                xs = [xpos[bs] for bs in batch_sizes if bs in cells]
                ys = [
                    float(cells[bs]["throughput_samples_per_s"])
                    for bs in batch_sizes
                    if bs in cells
                ]
                style = _series_style(variant, theme)
                (line,) = ax.plot(
                    xs,
                    ys,
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=2,
                    marker=style["marker"],
                    markersize=6.5,
                    markeredgecolor=theme.surface,
                    markeredgewidth=1.4,
                    solid_joinstyle="round",
                    dash_capstyle="round",
                    label=style["label"],
                    zorder=3,
                )
                legend.setdefault(style["label"], line)
                row_max = max(row_max, *ys)
            ax.set_xticks(range(len(batch_sizes)), [str(bs) for bs in batch_sizes])
            ax.set_xlim(-0.25, len(batch_sizes) - 0.75)
            if r == 0:
                ax.set_title(column_title, loc="left", color=theme.ink, fontweight="bold", pad=10)
            if r == len(devices) - 1:
                ax.set_xlabel("Batch size")
            if c == 0:
                ax.set_ylabel(f"{device_label}: samples / s")
        # Both panels of a row share this axis: zero baseline, headroom over the row's peak.
        axes[r][0].set_ylim(0, row_max * 1.08 if row_max > 0 else 1.0)

    _header(
        fig,
        theme,
        "Throughput by batch size and padding mode",
        "Higher is better; dashed lines are FP16. Rows share a y-axis. The two padding modes "
        "were measured in separate runs.",
    )
    fig.legend(
        handles=list(legend.values()),
        loc="upper left",
        bbox_to_anchor=(0.025, _frac(fig, 0.72)),
        ncols=len(legend),
        frameon=False,
        labelcolor=theme.ink_secondary,
        handlelength=2.4,
        columnspacing=1.4,
    )
    _footer(
        fig, theme, f"results/benchmarks.json + benchmarks_dynamic.json · {_hardware(data.dynamic)}"
    )
    return fig


# --------------------------------------------------------------------------
# Figure 3: real tokens vs padding per sentence
# --------------------------------------------------------------------------
def _fig_padding_tokens(data: FigureInputs, theme: Theme) -> Figure:
    from matplotlib.patches import Patch

    max_len = int(data.fixed["config"]["max_seq_length"])
    reference = _ok_variants(data.dynamic)[0]  # every variant timed the same batches
    bars: list[tuple[float, float, float]] = []  # y, real tokens, padded length
    rows: list[tuple[float, str, bool]] = []
    y = 0.0
    for bs, cell in sorted(_cells(reference).items()):
        workload = cell.get("workload") or {}
        real = workload.get("mean_real_tokens")
        padded = workload.get("seq_len_mean")
        if real is None or padded is None:
            continue
        rows.append((y, f"Batch {bs}", True))
        for label, length in (
            (f"padded to {max_len}", float(max_len)),
            ("padded per batch", float(padded)),
        ):
            y += 1.0
            rows.append((y, label, False))
            bars.append((y, float(real), length))
        y += 1.35
    if not bars:
        raise PipelineError("benchmarks_dynamic.json has no workload summaries to plot")

    fig = _new_figure(1.0 + 0.3 * len(rows) + 0.85, theme)
    fig.subplots_adjust(
        left=0.215, right=0.88, top=_frac(fig, 1.05), bottom=0.8 / fig.get_figheight()
    )
    ax = fig.add_subplot()
    _style_axes(ax, theme, grid_axis="x", baseline="left")
    ax.spines["left"].set_visible(False)
    height = 0.62
    for by, real, length in bars:
        ax.barh(
            by,
            real,
            height=height,
            color=theme.accent,
            edgecolor=theme.surface,
            linewidth=1.2,
            zorder=2,
        )
        if length > real:
            ax.barh(
                by,
                length - real,
                left=real,
                height=height,
                color=theme.padding,
                edgecolor=theme.surface,
                linewidth=1.2,
                zorder=2,
            )
        ax.annotate(
            f"{real / length:.1%} real",
            (length, by),
            xytext=(6, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=8.5,
            color=theme.ink_secondary,
        )
    ax.set_xlim(0, max_len)
    step = 32 if max_len >= 64 else 8
    ax.set_xticks(range(0, max_len + 1, step))
    ax.set_xlabel("Tokens computed per sentence (mean over the timed batches)")
    ax.set_ylim(y - 1.35 + 0.6, -0.6)
    _row_labels(ax, theme, rows, column_pt=112)
    for tick, (_, _, heading) in zip(ax.get_yticklabels(), rows, strict=True):
        if not heading:
            tick.set_fontfamily(_SANS)

    _header(
        fig,
        theme,
        f"Most of a {max_len}-token batch is padding",
        f"SST-2 validation sentences average {sum(b[1] for b in bars) / len(bars):.0f} tokens. "
        "Bigger batches pad to a longer sentence.",
    )
    handles = [
        Patch(color=theme.accent, label="Real tokens"),
        Patch(color=theme.padding, label="Padding"),
    ]
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.025, _frac(fig, 0.7)),
        ncols=2,
        frameon=False,
        labelcolor=theme.ink_secondary,
    )
    n_batches = {
        (c.get("workload") or {}).get("distinct_batches") for c in _cells(reference).values()
    }
    batches = f"{n_batches.pop()} seeded batches" if len(n_batches) == 1 else "seeded batches"
    _footer(fig, theme, f"results/benchmarks_dynamic.json · {batches} per batch size")
    return fig


# --------------------------------------------------------------------------
# Figure 4: INT8 settings sweep
# --------------------------------------------------------------------------
def _sweep_label(settings: dict[str, Any]) -> str:
    ops = "+".join(settings.get("int8_op_types") or [])
    scales = "per-channel" if settings.get("int8_per_channel") else "per-tensor"
    reduce_range = "on" if settings.get("int8_reduce_range") else "off"
    return f"{ops}, {scales}, reduce_range {reduce_range}"


def _is_shipped(settings: dict[str, Any]) -> bool:
    q = QuantizationConfig()
    return (
        bool(settings.get("int8_per_channel")) == q.int8_per_channel
        and bool(settings.get("int8_reduce_range")) == q.int8_reduce_range
        and tuple(settings.get("int8_op_types") or ()) == q.int8_op_types
    )


def _fig_int8_sweep(data: FigureInputs, theme: Theme) -> Figure:
    sweep_rows = data.sweep["rows"]
    reference = float(data.sweep["reference"]["accuracy"])
    fig = _new_figure(1.15 + 0.42 * len(sweep_rows) + 0.75, theme)
    fig.subplots_adjust(
        left=0.45, right=0.92, top=_frac(fig, 1.15), bottom=0.8 / fig.get_figheight()
    )
    ax = fig.add_subplot()
    _style_axes(ax, theme, grid_axis="x", baseline="left")
    labels = []
    for i, row in enumerate(sweep_rows):
        shipped = _is_shipped(row.get("settings") or {})
        accuracy = float(row["accuracy"])
        ax.barh(
            i,
            accuracy * 100,
            height=0.58,
            color=theme.accent if shipped else theme.neutral,
            zorder=2,
        )
        ax.annotate(
            f"{accuracy:.2%}",
            (accuracy * 100, i),
            xytext=(6, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=8.5,
            color=theme.ink if shipped else theme.ink_secondary,
            fontweight="bold" if shipped else "normal",
        )
        labels.append(_sweep_label(row.get("settings") or {}))
    ax.set_yticks(range(len(sweep_rows)), labels)
    for tick, row in zip(ax.get_yticklabels(), sweep_rows, strict=True):
        if _is_shipped(row.get("settings") or {}):
            tick.set_fontweight("bold")
            tick.set_color(theme.ink)
    ax.set_ylim(len(sweep_rows) - 0.4, -0.75)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel(f"Accuracy on SST-2 validation ({data.sweep.get('n', '?')} sentences)")
    ax.axvline(reference * 100, color=theme.ink_secondary, linewidth=1, zorder=3)
    ax.annotate(
        f"PyTorch FP32 {reference:.2%}",
        (reference * 100, -0.75),
        xytext=(0, 3),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color=theme.ink_secondary,
        annotation_clip=False,
    )

    _header(
        fig,
        theme,
        "Per-channel INT8 collapses without reduce_range on this CPU",
        "ORT dynamic INT8. Without VNNI, full-range per-channel weights overflow. "
        "The bold row is the shipped default.",
    )
    _footer(fig, theme, f"results/int8_sweep.json · {_hardware(data.dynamic)}")
    return fig


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
_BUILDERS: dict[str, Callable[[FigureInputs, Theme], Figure]] = {
    "latency_padding": _fig_latency_padding,
    "throughput": _fig_throughput,
    "padding_tokens": _fig_padding_tokens,
    "int8_sweep": _fig_int8_sweep,
}
FIGURE_NAMES: tuple[str, ...] = tuple(_BUILDERS)


def _require_matplotlib() -> None:
    try:
        import matplotlib  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise PipelineError(
            "figures need matplotlib, a dev dependency: run `uv sync` (or `python tasks.py setup`)"
        ) from exc


def figure_path(out_dir: Path, name: str, theme: Theme) -> Path:
    """Where figure ``name`` is written for ``theme``: ``<out_dir>/<name>-<theme>.png``."""
    return out_dir / f"{name}-{theme.name}.png"


def render_png(name: str, data: FigureInputs, theme: Theme) -> bytes:
    """Render one figure to PNG bytes (deterministic for a given matplotlib build)."""
    _require_matplotlib()
    import matplotlib as mpl

    if name not in _BUILDERS:
        raise PipelineError(f"unknown figure {name!r}; choose from {', '.join(FIGURE_NAMES)}")
    with mpl.rc_context(
        {
            "font.family": _SANS,
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.unicode_minus": False,
        }
    ):
        fig = _BUILDERS[name](data, theme)
        buf = io.BytesIO()
        fig.savefig(
            buf, format="png", dpi=DPI, facecolor=theme.surface, metadata={"Software": None}
        )
    return buf.getvalue()


def render_figures(data: FigureInputs, out_dir: Path = DEFAULT_FIGURES_DIR) -> list[Path]:
    """Write every figure in every theme to ``out_dir``; return the paths written."""
    written: list[Path] = []
    for theme in THEMES:
        for name in FIGURE_NAMES:
            path = figure_path(out_dir, name, theme)
            png = render_png(name, data, theme)
            with atomic_output(path) as tmp:
                tmp.write_bytes(png)
            log.info("wrote %s (%.0f KB)", path, len(png) / 1024)
            written.append(path)
    return written
