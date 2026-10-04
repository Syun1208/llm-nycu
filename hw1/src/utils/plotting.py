from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cycler import cycler

ACADEMIC_PALETTE = ("#4C72B0", "#55A868", "#DD8452", "#C44E52", "#8172B3", "#937860")
ACADEMIC_MARKERS = ("o", "s", "^", "D", "v", "P")
FIGURE_DPI = 300
DISPLAY_DPI = 150
FIGURE_SUFFIXES = (".pdf", ".png")
PANEL_SIZE = (4.6, 3.8)
PANEL_LABELS = "abcdefgh"
CELL_FONT_SIZE = 9
RADAR_SIZE = (5.2, 5.2)
RADAR_FILL_ALPHA = 0.08
RADAR_FLOOR_STEP = 0.1
RADAR_LABEL_ANGLE = 36
RADAR_LABEL_ZORDER = 10
BOX_ALPHA = 0.6
REFERENCE_COLOUR = "#555555"
EPSILON = 1e-9


def apply_academic_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "dejavuserif",
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#E5E5E5",
            "grid.linewidth": 0.6,
            "axes.prop_cycle": cycler(
                color=ACADEMIC_PALETTE,
                marker=ACADEMIC_MARKERS,
            ),
            "lines.linewidth": 1.8,
            "lines.markersize": 6,
            "legend.frameon": False,
            "figure.dpi": DISPLAY_DPI,
            "savefig.dpi": FIGURE_DPI,
            "savefig.bbox": "tight",
        }
    )


def build_styles(*, names: Sequence[str]) -> dict[str, dict[str, str]]:
    return {
        name: {"color": color, "marker": marker}
        for name, color, marker in zip(names, ACADEMIC_PALETTE, ACADEMIC_MARKERS)
    }


def save_figure(
    *,
    figure: plt.Figure,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    for suffix in FIGURE_SUFFIXES:
        figure.savefig(path.with_suffix(suffix))


def panel_title(
    *,
    index: int,
    title: str,
) -> str:
    return f"({PANEL_LABELS[index]}) {title}"


def plot_confusion_matrices(
    *,
    matrices: Mapping[str, np.ndarray],
    tick_labels: Sequence[str],
) -> plt.Figure:
    width, height = PANEL_SIZE
    figure, axes = plt.subplots(
        nrows=1,
        ncols=len(matrices),
        figsize=(width * len(matrices), height),
        squeeze=False,
        constrained_layout=True,
    )
    for index, (ax, (title, matrix)) in enumerate(zip(axes[0], matrices.items())):
        normalized = matrix / np.maximum(
            matrix.sum(
                axis=1,
                keepdims=True,
            ),
            1,
        )
        image = ax.imshow(
            X=normalized,
            cmap="Blues",
            vmin=0.0,
            vmax=1.0,
        )
        ax.set_xticks(
            ticks=range(len(tick_labels)),
            labels=tick_labels,
        )
        ax.set_yticks(
            ticks=range(len(tick_labels)),
            labels=tick_labels,
        )
        ax.set(
            xlabel="Predicted language",
            ylabel="True language",
            title=panel_title(
                index=index,
                title=title,
            ),
        )
        ax.grid(visible=False)
        for (row, column), value in np.ndenumerate(normalized):
            colour = "white" if value > 0.5 else "black"
            ax.text(
                x=column,
                y=row,
                s=f"{value:.2f}",
                ha="center",
                va="center",
                color=colour,
                fontsize=CELL_FONT_SIZE,
            )
    figure.colorbar(
        mappable=image,
        ax=axes[0].tolist(),
        shrink=0.85,
        label="Fraction of true language",
    )
    return figure


def plot_learning_curves(
    *,
    curves: Mapping[str, Mapping[str, Sequence[float]]],
    panels: Sequence[tuple[str, str, str]],
    styles: Mapping[str, Mapping[str, str]],
    labels: Mapping[str, str],
) -> plt.Figure:
    width, height = PANEL_SIZE
    figure, axes = plt.subplots(
        nrows=1,
        ncols=len(panels),
        figsize=(width * len(panels), height),
        squeeze=False,
        constrained_layout=True,
    )
    for index, (ax, (field, title, ylabel)) in enumerate(zip(axes[0], panels)):
        for series, values in curves.items():
            ax.plot(
                range(1, len(values[field]) + 1),
                values[field],
                label=labels[series],
                **styles[series],
            )
        ax.set(
            title=panel_title(
                index=index,
                title=title,
            ),
            xlabel="Epoch",
            ylabel=ylabel,
        )
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    handles, legend_labels = axes[0][0].get_legend_handles_labels()
    figure.legend(
        handles=handles,
        labels=legend_labels,
        loc="outside upper center",
        ncols=len(curves),
    )
    return figure


def plot_perplexity_vs_accuracy(
    *,
    points: Mapping[str, tuple[float, float]],
    margins: Mapping[str, np.ndarray],
    styles: Mapping[str, Mapping[str, str]],
) -> plt.Figure:
    width, height = PANEL_SIZE
    figure, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(2 * width, height),
        constrained_layout=True,
    )
    for name, (perplexity, accuracy) in points.items():
        axes[0].plot(
            perplexity,
            accuracy,
            linestyle="none",
            markersize=7,
            label=name,
            **styles[name],
        )
    axes[0].set(
        title=panel_title(
            index=0,
            title="Perplexity vs. accuracy",
        ),
        xlabel="Test corpus perplexity",
        ylabel="Test accuracy",
    )
    axes[0].legend(loc="best")
    boxes = axes[1].boxplot(
        x=list(margins.values()),
        tick_labels=list(margins),
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black"},
    )
    for box, name in zip(boxes["boxes"], margins):
        box.set(
            facecolor=styles[name]["color"],
            alpha=BOX_ALPHA,
        )
    axes[1].axhline(
        y=0,
        color=REFERENCE_COLOUR,
        linestyle="--",
        linewidth=1,
    )
    axes[1].set(
        title=panel_title(
            index=1,
            title="Decision margin per text",
        ),
        ylabel=r"$\log \mathrm{PPL}_{\mathrm{best\ wrong}} - \log \mathrm{PPL}_{\mathrm{true}}$",
    )
    return figure


def plot_accuracy_by_length(
    *,
    length_table: pd.DataFrame,
    styles: Mapping[str, Mapping[str, str]],
) -> plt.Figure:
    figure, ax = plt.subplots(
        figsize=PANEL_SIZE,
        constrained_layout=True,
    )
    for name in length_table.columns:
        ax.plot(
            length_table.index.astype(str),
            length_table[name],
            label=name,
            **styles[name],
        )
    ax.set(
        xlabel="Text length (characters)",
        ylabel="Test accuracy",
    )
    ax.legend(loc="lower right")
    return figure


def plot_radar(
    *,
    values: Mapping[str, np.ndarray],
    axis_labels: Sequence[str],
    legend_labels: Mapping[str, str],
    styles: Mapping[str, Mapping[str, str]],
) -> plt.Figure:
    angles = np.linspace(
        start=0.0,
        stop=2 * np.pi,
        num=len(axis_labels),
        endpoint=False,
    )
    closed_angles = np.append(angles, angles[0])
    floor = math.floor(min(series.min() for series in values.values()) / RADAR_FLOOR_STEP) * RADAR_FLOOR_STEP

    figure, ax = plt.subplots(
        figsize=RADAR_SIZE,
        subplot_kw={"projection": "polar"},
        constrained_layout=True,
    )
    for name, series in values.items():
        closed = np.append(series, series[0])
        ax.plot(
            closed_angles,
            closed,
            label=legend_labels[name],
            **styles[name],
        )
        ax.fill(
            closed_angles,
            closed,
            color=styles[name]["color"],
            alpha=RADAR_FILL_ALPHA,
        )
    ax.set_xticks(
        ticks=angles,
        labels=axis_labels,
    )
    ax.set_theta_offset(offset=np.pi / 2)
    ax.set_theta_direction(direction=-1)
    ax.set_ylim(
        bottom=floor,
        top=1.0,
    )
    ax.set_yticks(
        ticks=np.arange(
            start=floor,
            stop=1.0 + EPSILON,
            step=RADAR_FLOOR_STEP,
        )
    )
    ax.set_rlabel_position(value=RADAR_LABEL_ANGLE)
    ax.tick_params(
        axis="y",
        labelsize=8,
        labelcolor=REFERENCE_COLOUR,
    )
    ax.yaxis.set_zorder(level=RADAR_LABEL_ZORDER)
    for tick_label in ax.get_yticklabels():
        tick_label.set_bbox({"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1})
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.06),
        ncols=2,
    )
    return figure
