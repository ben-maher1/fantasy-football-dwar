"""Calibration plots for the validation report (static PNGs, light theme).

Small multiples, one panel per group, one series each, so no categorical
color is needed: the series is the reference palette's slot 1 and the
"perfect calibration" reference is a muted dashed line.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
SERIES = "#2a78d6"
REFERENCE = "#9a9893"


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=8, length=0)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def calibration_plot(coverage: dict[str, dict[str, float]], path: Path, title: str) -> None:
    """Stated vs actual coverage of DWAR's central intervals, per group."""
    groups = list(coverage)
    fig, axes = plt.subplots(1, len(groups), figsize=(2.4 * len(groups), 2.8), sharey=True, facecolor=SURFACE)
    for ax, g in zip(np.atleast_1d(axes), groups):
        stated = np.array([float(k) for k in coverage[g]])
        actual = np.array(list(coverage[g].values()))
        _style(ax)
        ax.plot([0, 1], [0, 1], color=REFERENCE, linewidth=1, linestyle=(0, (3, 3)))
        ax.plot(stated, actual, color=SERIES, linewidth=2, marker="o", markersize=4,
                markeredgecolor=SURFACE, markeredgewidth=1)
        for lv in (0.5, 0.9):
            k = int(np.argmin(np.abs(stated - lv)))
            ax.annotate(f"{actual[k]:.0%}", (stated[k], actual[k]), textcoords="offset points",
                        xytext=(-4, 6), ha="right", fontsize=8, color=TEXT)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(g, fontsize=9, color=TEXT, loc="left")
        ax.set_xticks([0, 0.5, 1], ["0%", "50%", "100%"])
        ax.set_yticks([0, 0.5, 1], ["0%", "50%", "100%"])
    np.atleast_1d(axes)[0].set_ylabel("actual coverage", fontsize=8, color=TEXT_2)
    fig.supxlabel("stated interval", fontsize=8, color=TEXT_2)
    fig.suptitle(title, fontsize=10, color=TEXT, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def pit_histograms(rows: pl.DataFrame, groups: dict[str, pl.DataFrame], path: Path, title: str) -> None:
    """Where each outcome fell in its forecast distribution. Flat is honest;
    a U means intervals too narrow, a hump too wide."""
    fig, axes = plt.subplots(1, len(groups), figsize=(2.4 * len(groups), 2.6), sharey=True, facecolor=SURFACE)
    bins = np.linspace(0, 1, 11)
    for ax, (g, r) in zip(np.atleast_1d(axes), groups.items()):
        _style(ax)
        counts, _ = np.histogram(r["pit"].to_numpy(), bins=bins)
        share = counts / max(counts.sum(), 1)
        ax.bar(bins[:-1], share, width=0.1 - 0.012, align="edge", color=SERIES)
        ax.axhline(0.1, color=REFERENCE, linewidth=1, linestyle=(0, (3, 3)))
        ax.set_title(f"{g} (n={r.height:,})", fontsize=9, color=TEXT, loc="left")
        ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    np.atleast_1d(axes)[0].set_ylabel("share of outcomes", fontsize=8, color=TEXT_2)
    fig.supxlabel("position of the actual result within the forecast (PIT)", fontsize=8, color=TEXT_2)
    fig.suptitle(title, fontsize=10, color=TEXT, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
