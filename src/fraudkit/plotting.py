"""Shared matplotlib style so every chart in the portfolio reads as one system.

Colours come from a colourblind-validated categorical palette, used in fixed
slot order. Legit transactions always take slot 1 (blue) and fraud slot 2
(orange), in every notebook.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
CLASS_COLORS = {0: SERIES[0], 1: SERIES[1]}
CLASS_NAMES = {0: "Legit", 1: "Fraud"}

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
NEUTRAL = "#f0efec"

# One hue, light -> dark, for magnitude (heatmaps, confusion matrices).
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "fk_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)
# Two poles with a grey midpoint, for signed values (correlations).
DIVERGING = LinearSegmentedColormap.from_list(
    "fk_div", ["#184f95", "#3987e5", NEUTRAL, "#e66767", "#a12e2e"]
)


def set_style():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "figure.dpi": 110,
        "figure.figsize": (8, 4.5),
        "font.family": "sans-serif",
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlecolor": INK,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.prop_cycle": mpl.cycler(color=SERIES),
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "legend.frameon": False,
        "legend.labelcolor": INK_2,
        "patch.linewidth": 0,
    })


def class_legend(ax, **kwargs):
    handles = [mpl.patches.Patch(color=CLASS_COLORS[c], label=CLASS_NAMES[c]) for c in (0, 1)]
    ax.legend(handles=handles, **kwargs)


def label_bars(ax, bars, fmt="{:.3f}", horizontal=False, pad=3):
    """Write each bar's value just past its end."""
    for b in bars:
        if horizontal:
            ax.annotate(fmt.format(b.get_width()), (b.get_width(), b.get_y() + b.get_height() / 2),
                        xytext=(pad, 0), textcoords="offset points", va="center", color=INK_2)
        else:
            ax.annotate(fmt.format(b.get_height()), (b.get_x() + b.get_width() / 2, b.get_height()),
                        xytext=(0, pad), textcoords="offset points", ha="center", color=INK_2)


def finish(fig=None):
    (fig or plt.gcf()).tight_layout()
    plt.show()


def class_hist(ax, values, labels, bins, density=True):
    """Overlaid per-class histograms: light fill plus a crisp outline, so overlaps stay readable."""
    for c in (0, 1):
        v = values[labels == c]
        ax.hist(v, bins=bins, density=density, histtype="stepfilled", alpha=0.25, color=CLASS_COLORS[c])
        ax.hist(v, bins=bins, density=density, histtype="step", linewidth=1.5, color=CLASS_COLORS[c])
