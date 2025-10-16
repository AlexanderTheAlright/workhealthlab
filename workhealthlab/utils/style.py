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
        # High contrast black and white first, then grayscale — perfect for publication
        cmaps = {"positive": cm.Greys, "neutral": cm.Greys, "negative": cm.Greys}
        ranges = {"positive": (0.25, 0.85), "neutral": (0.25, 0.85), "negative": (0.25, 0.85)}

    else:  # viridis
        cmaps = {"positive": cm.viridis, "neutral": cm.viridis, "negative": cm.viridis}
        ranges = {"positive": (0.2, 0.9), "neutral": (0.3, 0.7), "negative": (0.1, 0.6)}

    # Build palette
    for group, items in groups.items():
        if not items:
            continue
        g = group.lower()
        key = "positive" if g.startswith("pos") else \
              "neutral" if g.startswith("neu") else \
              "negative" if g.startswith("neg") else "positive"
        cmap, (low, high) = cmaps[key], ranges[key]

        # Special handling for reviewer3: use high contrast black/white first, then grayscale
        if style == "reviewer3":
            if len(items) == 1:
                # Single item gets black
                palette[items[0]] = (0, 0, 0, 1)
            elif len(items) == 2:
                # Two items get black and white with black border
                palette[items[0]] = (0, 0, 0, 1)  # Black
                palette[items[1]] = (1, 1, 1, 1)  # White
            else:
                # Three or more: black, white, then grayscale
                palette[items[0]] = (0, 0, 0, 1)  # Black
                palette[items[1]] = (1, 1, 1, 1)  # White
                # Remaining items use grayscale
                vals = np.linspace(low, high, len(items) - 2)
                for item, v in zip(items[2:], vals):
                    palette[item] = cmap(v)
        else:
            # Standard behavior for other styles
            vals = np.linspace(low, high, len(items))
            for item, v in zip(items, vals):
                palette[item] = cmap(v)
    return palette


# ══════════════════════════════════════════════════════════════════════════════
# II.B. CONTINUOUS COLORMAPS FOR HEATMAPS
# ══════════════════════════════════════════════════════════════════════════════

# Dictionary mapping style modes to their primary colormaps
COLORS_DICT = {
    "fiery": cm.inferno,
    "viridis": cm.viridis,
    "sentiment": cm.RdYlGn,
    "plainjane": cm.RdBu_r,
    "reviewer3": cm.Greys,
}


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


# ══════════════════════════════════════════════════════════════════════════════
# VI. SEMANTIC COLOR RETRIEVAL (THEME-AWARE)
# ══════════════════════════════════════════════════════════════════════════════

def get_color(semantic: str, mode: str = None) -> str:
    """
    Get theme-aware color for semantic use cases.

    For reviewer3, all colors map to grayscale (black/white/gray).
    For other themes, returns appropriate colors.

    Parameters
    ----------
    semantic : str
        One of: 'negative', 'positive', 'neutral', 'warning', 'highlight',
                'line', 'threshold', 'reference', 'primary', 'secondary'
    mode : str, optional
        Override style. If None, uses ACTIVE_STYLE.

    Returns
    -------
    color : str
        Color string (hex, rgb, or named color).

    Examples
    --------
    >>> get_color('warning')  # Returns 'red' in most themes, '#333333' in reviewer3
    >>> get_color('positive', mode='reviewer3')  # Returns '#666666'
    """
    style = (mode or globals().get("ACTIVE_STYLE", "viridis")).lower()

    if style == "reviewer3":
        # Pure grayscale mapping for publication
        color_map = {
            'negative': '#000000',      # Black for negative/warnings
            'positive': '#666666',      # Medium gray for positive
            'neutral': '#999999',       # Light gray for neutral
            'warning': '#000000',       # Black for warnings/thresholds
            'highlight': '#333333',     # Dark gray for highlights
            'line': '#666666',          # Medium gray for lines
            'threshold': '#000000',     # Black for thresholds
            'reference': '#000000',     # Black for reference lines
            'primary': '#333333',       # Dark gray primary
            'secondary': '#999999',     # Light gray secondary
            'increasing': '#666666',    # Medium gray
            'decreasing': '#333333',    # Dark gray
        }
    elif style == "sentiment":
        color_map = {
            'negative': '#d62728',      # Red
            'positive': '#2ca02c',      # Green
            'neutral': '#7f7f7f',       # Gray
            'warning': '#d62728',       # Red
            'highlight': '#ff7f0e',     # Orange
            'line': '#1f77b4',          # Blue
            'threshold': '#d62728',     # Red
            'reference': '#d62728',     # Red
            'primary': '#1f77b4',       # Blue
            'secondary': '#ff7f0e',     # Orange
            'increasing': '#2ca02c',    # Green
            'decreasing': '#d62728',    # Red
        }
    elif style == "fiery":
        color_map = {
            'negative': '#d62728',      # Red
            'positive': '#ff7f0e',      # Orange
            'neutral': '#8c564b',       # Brown
            'warning': '#d62728',       # Red
            'highlight': '#ff7f0e',     # Orange
            'line': '#e377c2',          # Pink
            'threshold': '#d62728',     # Red
            'reference': '#d62728',     # Red
            'primary': '#d62728',       # Red
            'secondary': '#ff7f0e',     # Orange
            'increasing': '#ff7f0e',    # Orange
            'decreasing': '#8c564b',    # Brown
        }
    elif style == "plainjane":
        color_map = {
            'negative': '#d62728',      # Red
            'positive': '#1f77b4',      # Blue
            'neutral': '#7f7f7f',       # Gray
            'warning': '#d62728',       # Red
            'highlight': '#ff7f0e',     # Orange
            'line': '#1f77b4',          # Blue
            'threshold': '#d62728',     # Red
            'reference': '#d62728',     # Red
            'primary': '#1f77b4',       # Blue
            'secondary': '#d62728',     # Red
            'increasing': '#1f77b4',    # Blue
            'decreasing': '#d62728',    # Red
        }
    else:  # viridis
        color_map = {
            'negative': '#d62728',      # Red
            'positive': '#2ca02c',      # Green
            'neutral': '#7f7f7f',       # Gray
            'warning': '#d62728',       # Red
            'highlight': '#ff7f0e',     # Orange
            'line': '#1f77b4',          # Blue
            'threshold': '#d62728',     # Red
            'reference': '#d62728',     # Red
            'primary': '#440154',       # Viridis dark purple
            'secondary': '#fde724',     # Viridis yellow
            'increasing': '#2ca02c',    # Green
            'decreasing': '#d62728',    # Red
        }

    return color_map.get(semantic, '#333333')
