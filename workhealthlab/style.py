"""
style.py — The Work Health World of Data Visualization 🎨
----------------------------------------------------------
Core style engine for WorkHealthLab plots.
Implements all conventions from the generative style guide:
- White canvas, minimal spines
- Semantic color grouping
- Structured typography and legend logic
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ══════════════════════════════════════════════════════════════════════════════
# I. BASE STYLE SETUP
# ══════════════════════════════════════════════════════════════════════════════

def set_workhealth_style():
    """Apply global WorkHealthLab matplotlib rcParams."""
    plt.style.use("default")
    plt.rcParams.update({
        # Canvas
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        # Spines
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "grey",
        # Grid
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": "grey",
        "grid.linestyle": ":",
        "grid.linewidth": 0.7,
        "axes.axisbelow": True,
        # Font and typography
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.titlesize": 20,
        "axes.titleweight": "bold",
        "axes.titlecolor": "#333333",
        "axes.labelsize": 12,
        "axes.labelweight": "bold",
        "axes.labelcolor": "grey",
        "xtick.color": "grey",
        "ytick.color": "grey",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "ytick.major.size": 0,
        "legend.frameon": False,
        "figure.dpi": 120,
    })

# ══════════════════════════════════════════════════════════════════════════════
# II. GENERATIVE COLOR PALETTES
# ══════════════════════════════════════════════════════════════════════════════

def generate_semantic_palette(groups: dict):
    """
    Generate a semantic color palette based on groupings.

    Parameters
    ----------
    groups : dict
        Dictionary like:
        {
          'positive': ['var1','var2'],
          'neutral': ['var3'],
          'negative': ['var4','var5']
        }

    Returns
    -------
    palette : dict
        Mapping {var_name: rgba_color_array}
    """
    palette = {}
    for group, items in groups.items():
        if len(items) == 0:
            continue

        if group.lower().startswith("pos"):
            cmap = cm.viridis
            vals = np.linspace(0.4, 0.9, len(items))
        elif group.lower().startswith("neu"):
            cmap = cm.Greys
            vals = np.linspace(0.4, 0.7, len(items))
        elif group.lower().startswith("neg"):
            cmap = cm.autumn_r
            vals = np.linspace(0.2, 0.7, len(items))
        else:
            cmap = cm.viridis
            vals = np.linspace(0.4, 0.9, len(items))

        for item, v in zip(items, vals):
            palette[item] = cmap(v)
    return palette


# ══════════════════════════════════════════════════════════════════════════════
# III. STRUCTURED TITLING
# ══════════════════════════════════════════════════════════════════════════════

def apply_titles(fig, main_title, subtitle=None, n=None):
    """
    Add standardized WorkHealthLab title block.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    main_title : str
    subtitle : str, optional
    n : int, optional
        Sample size to append in main title (e.g., n=300)
    """
    title = main_title
    if n is not None:
        title = f"{title} (n={n})"
    fig.text(0.01, 0.98, title, fontsize=20, fontweight="bold",
             ha="left", va="top", color="#333333")
    if subtitle:
        fig.text(0.01, 0.93, subtitle, fontsize=14, ha="left",
                 va="top", color="grey")


# ══════════════════════════════════════════════════════════════════════════════
# IV. DATA ELEMENT DEFAULTS
# ══════════════════════════════════════════════════════════════════════════════

def get_data_element_kwargs():
    """
    Return default kwargs for WorkHealthLab data elements.
    """
    return {
        "edgecolor": "white",
        "linewidth": 0.5,
    }


# ══════════════════════════════════════════════════════════════════════════════
# V. LEGEND UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

def draw_legend_group(ax, fig, title, var_list, palette,
                      start_x, start_y, line_height=0.04, spacing=0.04):
    """
    Draw structured legend group (bar/scatter compatible).
    """
    y = start_y
    fig.text(start_x, y, title, transform=ax.transAxes,
             fontsize=11, weight="bold", color="#333333", ha="left", va="top")
    y -= line_height
    for var in var_list:
        if var not in palette:
            continue
        rect_height = 0.025
        rect_y = y - (rect_height / 2)
        rect = Rectangle((start_x, rect_y), 0.015, rect_height,
                         facecolor=palette[var], transform=ax.transAxes, clip_on=False)
        ax.add_patch(rect)
        fig.text(start_x + 0.02, y, var, transform=ax.transAxes,
                 fontsize=10, color="#333333", ha="left", va="center")
        y -= line_height
    return y - spacing


def draw_scatter_legend(ax, fig, title, var_list, palette,
                        start_x, start_y, line_height=0.04, spacing=0.04):
    """
    Draw structured legend group for scatter plots.
    """
    y = start_y
    fig.text(start_x, y, title, transform=ax.transAxes,
             fontsize=11, weight="bold", color="#333333", ha="left", va="top")
    y -= line_height
    for var in var_list:
        if var not in palette:
            continue
        ax.add_line(Line2D([start_x], [y - 0.01],
                           transform=ax.transAxes, marker="o", color="w",
                           markerfacecolor=palette[var], markersize=8,
                           markeredgecolor="grey"))
        fig.text(start_x + 0.02, y, var, transform=ax.transAxes,
                 fontsize=10, color="#333333", ha="left", va="center")
        y -= line_height
    return y - spacing
