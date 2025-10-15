"""
style.py — The Sociopath-it World of Data Visualization 🎨
-----------------------------------------------------------
Core style engine for Sociopath-it plots.

Implements multiple thematic styles reflecting interpretive mood:
- 'fiery'     → intense reds, oranges, purples (high emotional contrast)
- 'viridis'   → balanced, perceptually uniform viridis-based default
- 'sentiment' → green-positive, red-negative moral economy tone
- 'plainjane' → light red/blue academic contrast
- 'reviewer3' → grayscale, journal-safe, publishable
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

# ══════════════════════════════════════════════════════════════════════════════
# I. STYLE CONFIGURATION SYSTEM
# ══════════════════════════════════════════════════════════════════════════════

AVAILABLE_STYLES = ["fiery", "viridis", "sentiment", "plainjane", "reviewer3"]
ACTIVE_STYLE = "viridis"

def set_style(mode: str = "viridis"):
    """
    Apply a Sociopath-it visual style theme.

    Parameters
    ----------
    mode : str
        One of {'fiery','viridis','sentiment','plainjane','reviewer3'}.
    """
    mode = mode.lower()
    if mode not in AVAILABLE_STYLES:
        raise ValueError(f"Invalid mode '{mode}'. Choose from {AVAILABLE_STYLES}.")

    plt.style.use("default")
    plt.rcParams.update({
        # Canvas and spines
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "black",
        # Grid
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": "grey",
        "grid.linestyle": ":",
        "grid.linewidth": 0.7,
        "axes.axisbelow": True,
        # Typography
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.titlesize": 20,
        "axes.titleweight": "bold",
        "axes.titlecolor": "#333333",
        "axes.labelsize": 12,
        "axes.labelweight": "bold",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "ytick.major.size": 0,
        "legend.frameon": False,
        "figure.dpi": 120,
    })

    global ACTIVE_STYLE
    ACTIVE_STYLE = mode


# ══════════════════════════════════════════════════════════════════════════════
# II. SEMANTIC COLOR PALETTES
# ══════════════════════════════════════════════════════════════════════════════

def generate_semantic_palette(groups: dict, mode: str = None):
    """
    Generate a semantic color palette depending on the active or specified style.

    Parameters
    ----------
    groups : dict
        e.g., {'positive': [...], 'neutral': [...], 'negative': [...]}
    mode : str, optional
        Override style manually.

    Returns
    -------
    palette : dict  {variable: RGBA tuple}
    """
    style = (mode or globals().get("ACTIVE_STYLE", "viridis")).lower()
    palette = {}

    if style == "fiery":
        cmaps = {"positive": cm.plasma, "neutral": cm.magma, "negative": cm.inferno}
        ranges = {"positive": (0.4, 0.9), "neutral": (0.3, 0.6), "negative": (0.2, 0.7)}

    elif style == "sentiment":
        cmaps = {"positive": cm.Greens, "neutral": cm.Greys, "negative": cm.Reds_r}
        ranges = {"positive": (0.5, 0.9), "neutral": (0.4, 0.7), "negative": (0.3, 0.8)}

    elif style == "plainjane":
        cmaps = {"positive": cm.Blues, "neutral": cm.Greys, "negative": cm.Reds}
        ranges = {"positive": (0.4, 0.85), "neutral": (0.4, 0.6), "negative": (0.4, 0.85)}

    elif style == "reviewer3":
        # Pure grayscale — perfect for publication
        cmaps = {"positive": cm.Greys, "neutral": cm.Greys, "negative": cm.Greys}
        ranges = {"positive": (0.25, 0.85), "neutral": (0.25, 0.85), "negative": (0.25, 0.85)}

    else:  # viridis
        cmaps = {"positive": cm.viridis, "neutral": cm.Greys, "negative": cm.autumn_r}
        ranges = {"positive": (0.4, 0.9), "neutral": (0.4, 0.7), "negative": (0.2, 0.7)}

    # Build palette
    for group, items in groups.items():
        if not items:
            continue
        g = group.lower()
        key = "positive" if g.startswith("pos") else \
              "neutral" if g.startswith("neu") else \
              "negative" if g.startswith("neg") else "positive"
        cmap, (low, high) = cmaps[key], ranges[key]
        vals = np.linspace(low, high, len(items))
        for item, v in zip(items, vals):
            palette[item] = cmap(v)
    return palette


# ══════════════════════════════════════════════════════════════════════════════
# II.B. CONTINUOUS COLORMAPS FOR HEATMAPS
# ══════════════════════════════════════════════════════════════════════════════

def get_continuous_cmap(mode: str = None):
    """
    Get continuous colormap for heatmaps, correlation matrices, etc.

    Parameters
    ----------
    mode : str, optional
        Style mode. If None, uses global ACTIVE_STYLE.

    Returns
    -------
    cmap_name : str
        Matplotlib colormap name suitable for continuous data.

    Examples
    --------
    Fiery: Dark red → white (heat aesthetic)
    Viridis: Standard perceptually uniform viridis
    Sentiment: Red-white-green diverging (correlation)
    Plainjane: Blue-white-red diverging
    Reviewer3: Grayscale white-to-black
    """
    style = (mode or globals().get("ACTIVE_STYLE", "viridis")).lower()

    cmap_mapping = {
        "fiery": "Reds",           # Dark red → white for "heat" aesthetic
        "viridis": "viridis",      # Perceptually uniform
        "sentiment": "RdYlGn",     # Red-yellow-green diverging
        "plainjane": "RdBu_r",     # Blue-white-red diverging
        "reviewer3": "Greys",      # Grayscale for publication
    }

    return cmap_mapping.get(style, "viridis")


# ══════════════════════════════════════════════════════════════════════════════
# III. TITLING SYSTEM
# ══════════════════════════════════════════════════════════════════════════════

def apply_titles(fig, title=None, subtitle=None, n=None):
    """
    Sociopath-it title logic:
    - If subtitle exists: title and subtitle at center-left
    - If no subtitle: centered title
    - Optional n count in bottom right
    """
    if title is None:
        return

    if subtitle:
        # Center-left placement for title + subtitle
        fig.text(
            0.02,
            0.97,
            f"{title}",
            fontsize=15,
            weight="bold",
            color="black",
            ha="left",
            va="center",
        )
        fig.text(
            0.02,
            0.94,
            f"{subtitle}",
            fontsize=11,
            color="grey",
            ha="left",
            va="center",
        )
    else:
        # Centered title when no subtitle
        fig.suptitle(
            f"{title}",
            fontsize=15,
            weight="bold",
            color="black",
            y=0.96,
            ha="center",
        )

    if n is not None:
        fig.text(
            0.99,
            0.01,
            f"(n = {n:,})",
            fontsize=9,
            color="grey",
            ha="right",
            va="bottom",
        )


# ══════════════════════════════════════════════════════════════════════════════
# IV. DATA ELEMENT DEFAULTS
# ══════════════════════════════════════════════════════════════════════════════

def get_data_element_kwargs():
    """Return default kwargs for Sociopath-it data elements."""
    return {"edgecolor": "white", "linewidth": 0.5}


# ══════════════════════════════════════════════════════════════════════════════
# V. LEGEND UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def draw_legend_group(ax, fig, title, var_list, palette,
                      start_x, start_y, line_height=0.04, spacing=0.04):
    """Draw structured legend group (bars/lines)."""
    y = start_y
    fig.text(start_x, y, title, transform=ax.transAxes,
             fontsize=11, weight="bold", color="#333333", ha="left", va="top")
    y -= line_height
    for var in var_list:
        if var not in palette:
            continue
        rect = Rectangle((start_x, y - 0.015), 0.015, 0.025,
                         facecolor=palette[var], transform=ax.transAxes, clip_on=False)
        ax.add_patch(rect)
        fig.text(start_x + 0.02, y, var,
                 transform=ax.transAxes, fontsize=10, color="#333333",
                 ha="left", va="center")
        y -= line_height
    return y - spacing


def draw_scatter_legend(ax, fig, title, var_list, palette,
                        start_x, start_y, line_height=0.04, spacing=0.04):
    """Draw structured legend group (scatter)."""
    y = start_y
    fig.text(start_x, y, title, transform=ax.transAxes,
             fontsize=11, weight="bold", color="#333333", ha="left", va="top")
    y -= line_height
    for var in var_list:
        if var not in palette:
            continue
        ax.add_line(Line2D([start_x], [y - 0.01],
                           transform=ax.transAxes,
                           marker="o", color="w",
                           markerfacecolor=palette[var],
                           markersize=8, markeredgecolor="grey"))
        fig.text(start_x + 0.02, y, var,
                 transform=ax.transAxes, fontsize=10, color="#333333",
                 ha="left", va="center")
        y -= line_height
    return y - spacing
