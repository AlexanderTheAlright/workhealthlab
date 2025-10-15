"""
factormap.py — Sociopath-it Visualization Module
------------------------------------------------
Universal factor map visualizer for MCA, PCA, CA, or any 2D reduction.

Features:
- Active vs supplementary variable mapping
- Smart text labeling with overlap avoidance
- Supports Sociopath-it styles: fiery, viridis, sentiment, plainjane, reviewer3
- Interactive Plotly version
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from adjustText import adjust_text
from ..utils.style import set_style, apply_titles

def factormap(
    active_coords,
    sup_coords=None,
    title="Factor Map",
    subtitle=None,
    style_mode="viridis",
    label_filter=30,
    figsize=(12, 10),
    label_suffix_map=None,
    active_label="Active Variables",
    sup_label="Supplementary Variables",
    annotate=True,
    dim_labels=("Dim 1", "Dim 2"),
    perc_var=None,
):
    """
    Plot a 2D factor map (e.g., from MCA, PCA, or correspondence analysis).

    Parameters
    ----------
    active_coords : pd.DataFrame
        DataFrame with columns [0, 1] representing first two dimensions.
    sup_coords : pd.DataFrame, optional
        DataFrame with same structure for supplementary variables.
    title, subtitle : str
        Plot titles.
    style_mode : str
        One of {'fiery', 'viridis', 'sentiment', 'plainjane', 'reviewer3'}.
    label_filter : int
        Number of top-magnitude points to label.
    figsize : tuple
        Figure size in inches.
    label_suffix_map : dict, optional
        Mapping for cleaning suffixes from variable names.
    active_label, sup_label : str
        Legend labels for groups.
    annotate : bool
        Whether to show text labels.
    dim_labels : tuple
        Names for the two plotted dimensions.
    perc_var : tuple, optional
        Percentage of variance explained for Dim1, Dim2.
    """
    set_style(style_mode)
    fig, ax = plt.subplots(figsize=figsize, dpi=130)
    fig.set_facecolor("white")
    ax.set_facecolor("white")

    # Prepare data
    active = active_coords.copy()
    active["r"] = np.sqrt(active[0] ** 2 + active[1] ** 2)
    sup = sup_coords.copy() if sup_coords is not None else pd.DataFrame()

    # Determine which points to label
    n_label = min(label_filter, len(active))
    label_index = set(active.nlargest(n_label, "r").index)
    if not sup.empty:
        label_index = label_index.union(set(sup.index))

    # Define color scheme by style
    style_colors = {
        "fiery": {"active": "#C53E3E", "sup": "#FFB703"},
        "viridis": {"active": "#3B528BFF", "sup": "#5DC863FF"},
        "sentiment": {"active": "#2E8B57", "sup": "#D62828"},
        "plainjane": {"active": "#4682B4", "sup": "#CD5C5C"},
        "reviewer3": {"active": "#4D4D4D", "sup": "#111111"},
    }
    colors = style_colors.get(style_mode, style_colors["viridis"])

    # Plot active
    ax.scatter(
        active[0], active[1],
        color=colors["active"], alpha=0.4, s=50, label=active_label,
        edgecolor="white", linewidth=0.5, zorder=2
    )

    # Plot supplementary
    if not sup.empty:
        ax.scatter(
            sup[0], sup[1],
            color=colors["sup"], alpha=0.8, s=120,
            edgecolor="black" if style_mode != "reviewer3" else "white",
            linewidth=0.6, label=sup_label, zorder=4
        )

    # Text labels with backgrounds
    if annotate:
        texts = []
        all_coords = pd.concat([active, sup]) if not sup.empty else active
        for idx, row in all_coords.iterrows():
            if idx in label_index:
                display_text = idx
                display_text = re.sub(r"^kw_|^jq_|^score_", "", display_text)
                display_text = re.sub(r"_uniq_high$", " (high)", display_text)
                display_text = re.sub(r"_\d+$", "", display_text)
                display_text = display_text.replace("_", " ")
                if label_suffix_map:
                    for k, v in label_suffix_map.items():
                        display_text = display_text.replace(k, v)
                texts.append(ax.text(
                    row[0], row[1],
                    display_text.lower(),
                    fontsize=11 if "score_" not in idx else 13,
                    fontweight="bold",
                    color="#111111",
                    ha="center", va="center",
                    bbox=dict(facecolor="white", edgecolor="#333333", linewidth=1.5,
                             boxstyle="round,pad=0.4", alpha=0.95),
                    zorder=6,
                ))

        adjust_text(
            texts, ax=ax,
            expand_points=(1.3, 1.3), expand_text=(1.2, 1.2),
            force_points=(1.0, 1.0), force_text=(1.5, 1.5),
            arrowprops=dict(arrowstyle="-", color="grey", lw=0.5, alpha=0.7),
        )

    # Axis + decorations
    ax.axhline(0, color="grey", lw=0.6, linestyle="--", alpha=0.7)
    ax.axvline(0, color="grey", lw=0.6, linestyle="--", alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("grey")
    ax.spines["bottom"].set_color("grey")
    ax.tick_params(axis="both", colors="grey", length=0)

    # Axis labels with explained variance
    if perc_var:
        ax.set_xlabel(f"{dim_labels[0]} ({perc_var[0]*100:.1f}%)", weight="bold", fontsize=12, color="black")
        ax.set_ylabel(f"{dim_labels[1]} ({perc_var[1]*100:.1f}%)", weight="bold", fontsize=12, color="black")
    else:
        ax.set_xlabel(dim_labels[0], weight="bold", fontsize=12, color="black")
        ax.set_ylabel(dim_labels[1], weight="bold", fontsize=12, color="black")

    # Legend outside to the right with larger text
    legend = ax.legend(
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor="grey",
        fontsize=10,
        title="Variable Type",
        title_fontsize=11,
    )
    legend.get_title().set_fontweight("bold")
    legend.get_frame().set_linewidth(1.5)
    legend.get_frame().set_alpha(0.95)

    # Title and layout
    apply_titles(fig, title, subtitle, n=len(active))
    plt.subplots_adjust(right=0.82)
    fig.tight_layout(rect=(0, 0, 0.85, 0.9 if subtitle else 0.94))
    plt.show()
    return fig, ax


def factormap_interactive(
    active_coords,
    sup_coords=None,
    title="Factor Map",
    subtitle=None,
    style_mode="viridis",
    label_filter=30,
    active_label="Active Variables",
    sup_label="Supplementary Variables",
    dim_labels=("Dim 1", "Dim 2"),
    perc_var=None,
):
    """Interactive Plotly factor map."""
    set_style(style_mode)

    # Prepare data
    active = active_coords.copy()
    active["r"] = np.sqrt(active[0] ** 2 + active[1] ** 2)
    sup = sup_coords.copy() if sup_coords is not None else pd.DataFrame()

    # Determine which points to label
    n_label = min(label_filter, len(active))
    label_index = set(active.nlargest(n_label, "r").index)
    if not sup.empty:
        label_index = label_index.union(set(sup.index))

    # Define color scheme
    style_colors = {
        "fiery": {"active": "#C53E3E", "sup": "#FFB703"},
        "viridis": {"active": "#3B528BFF", "sup": "#5DC863FF"},
        "sentiment": {"active": "#2E8B57", "sup": "#D62828"},
        "plainjane": {"active": "#4682B4", "sup": "#CD5C5C"},
        "reviewer3": {"active": "#4D4D4D", "sup": "#111111"},
    }
    colors = style_colors.get(style_mode, style_colors["viridis"])

    fig = go.Figure()

    # Plot active
    fig.add_trace(go.Scatter(
        x=active[0], y=active[1],
        mode="markers",
        marker=dict(size=8, color=colors["active"], opacity=0.5,
                   line=dict(color="white", width=0.5)),
        name=active_label,
        text=[f"<b>{idx}</b>" for idx in active.index],
        hovertemplate='%{text}<br>Dim1: %{x:.3f}<br>Dim2: %{y:.3f}<extra></extra>',
    ))

    # Plot supplementary
    if not sup.empty:
        fig.add_trace(go.Scatter(
            x=sup[0], y=sup[1],
            mode="markers",
            marker=dict(size=14, color=colors["sup"], opacity=0.9,
                       line=dict(color="black", width=1)),
            name=sup_label,
            text=[f"<b>{idx}</b>" for idx in sup.index],
            hovertemplate='%{text}<br>Dim1: %{x:.3f}<br>Dim2: %{y:.3f}<extra></extra>',
        ))

    # Add zero lines
    fig.add_hline(y=0, line=dict(color="grey", width=1, dash="dash"), opacity=0.5)
    fig.add_vline(x=0, line=dict(color="grey", width=1, dash="dash"), opacity=0.5)

    # Axis labels
    xaxis_title = f"{dim_labels[0]} ({perc_var[0]*100:.1f}%)" if perc_var else dim_labels[0]
    yaxis_title = f"{dim_labels[1]} ({perc_var[1]*100:.1f}%)" if perc_var else dim_labels[1]

    fig.update_layout(
        title=f"<b>{title}</b><br><span style='color:grey'>{subtitle or ''}</span>",
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        template="plotly_white",
        height=700,
        width=900,
        hovermode="closest",
        legend=dict(x=1.02, y=1, xanchor="left", yanchor="top"),
    )

    return fig
