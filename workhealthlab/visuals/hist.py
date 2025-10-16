"""
hist.py — Sociopath-it Visualization Module
-------------------------------------------
Distribution histograms styled according to the Generative Guide 🎨
with full support for multi-style themes (fiery, viridis, sentiment,
plainjane, reviewer3).

Features:
- Trace outline around distribution
- Threshold lines
- Two-variable coloring
- Interactive Plotly version
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib import cm
from scipy.interpolate import make_interp_spline

# Correct relative import (visuals → utils)
from ..utils.style import set_style, apply_titles, get_data_element_kwargs, get_color


# ══════════════════════════════════════════════════════════════════════════════
# HISTOGRAM FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def histogram(
    df,
    x,
    bins=20,
    title=None,
    subtitle=None,
    color=None,
    color_by=None,
    n=None,
    show_legend=True,
    style_mode="viridis",
    trace_outline=False,
    thresholds=None,
):
    """
    Sociopath-it styled histogram with dynamic theming.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing variable `x`.
    x : str
        Column name to plot.
    bins : int, default 20
        Number of histogram bins.
    title, subtitle : str, optional
        Titles displayed above plot.
    color : str or RGBA, optional
        Bar color; defaults to theme midtone.
    color_by : str, optional
        Column name to color bars by (creates two-variable visualization).
    n : int, optional
        Observation count for title suffix.
    show_legend : bool, default True
        Whether to display a legend box with variable info.
    style_mode : str, default "viridis"
        Visualization style preset. One of
        {'fiery','viridis','sentiment','plainjane','reviewer3'}.
    trace_outline : bool, default False
        Add smooth curve tracing the distribution top.
    thresholds : list, optional
        List of x-values to mark with vertical red dashed lines.
    """
    set_style(style_mode)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=130)

    vals = df[x].dropna()
    n = n or len(vals)

    # ─── Color logic ────────────────────────────────────────────────────────────
    if color_by and color_by in df.columns:
        # Two-variable coloring: split by color_by values
        color_var_vals = df[color_by].dropna().unique()
        cmap = cm.get_cmap("viridis")
        colors_map = {v: cmap(i / len(color_var_vals)) for i, v in enumerate(color_var_vals)}

        # Plot stacked histogram
        for i, cv in enumerate(color_var_vals):
            subset = df[df[color_by] == cv][x].dropna()
            ax.hist(
                subset,
                bins=bins,
                color=colors_map[cv],
                alpha=0.7,
                label=f"{color_by}={cv}",
                **get_data_element_kwargs()
            )
    else:
        # Single color
        color = color or (
            cm.plasma(0.7) if style_mode == "fiery"
            else cm.get_cmap("viridis")(0.6) if style_mode == "viridis"
            else cm.Greens(0.6) if style_mode == "sentiment"
            else cm.Blues(0.6) if style_mode == "plainjane"
            else cm.Greys(0.5)   # reviewer3
        )

        # ─── Plot ───────────────────────────────────────────────────────────────
        counts, bin_edges, patches = ax.hist(
            vals,
            bins=bins,
            color=color,
            **get_data_element_kwargs()
        )

        # ─── Trace outline ──────────────────────────────────────────────────────
        if trace_outline:
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            # Smooth curve with safety check for minimum points
            if len(bin_centers) >= 4:
                # Use spline smoothing for 4+ points
                k = min(3, len(bin_centers) - 1)
                spl = make_interp_spline(bin_centers, counts, k=k)
                x_smooth = np.linspace(bin_centers.min(), bin_centers.max(), 300)
                y_smooth = spl(x_smooth)
                ax.plot(x_smooth, y_smooth, color="grey", linewidth=2, alpha=0.8, zorder=3)
            elif len(bin_centers) >= 2:
                # Simple line for 2-3 points
                ax.plot(bin_centers, counts, color="grey", linewidth=2, alpha=0.8, zorder=3, marker='o')

    # ─── Threshold lines (without annotated number blocks) ────────────────────────────────────────────────────────
    if thresholds:
        for thresh in thresholds:
            ax.axvline(thresh, color=get_color('threshold', style_mode), linestyle="--", linewidth=1.5, alpha=0.8, zorder=4)

    # ─── Axis styling ───────────────────────────────────────────────────────────
    ax.set_xlabel(x.replace("_", " ").title(), fontsize=12, weight="bold", color="black")
    ax.set_ylabel("Frequency", fontsize=12, weight="bold", color="black")
    ax.grid(axis="y", color="grey", linestyle=":", linewidth=0.7)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)

    # ─── Titles ────────────────────────────────────────────────────────────────
    apply_titles(fig, title or f"Distribution of {x.replace('_',' ').title()}", subtitle, n=n)

    # ─── Legend ────────────────────────────────────────────────────────────────
    if show_legend:
        if color_by:
            legend = ax.legend(
                bbox_to_anchor=(1.02, 1.0),
                loc="upper left",
                frameon=True,
                facecolor="white",
                edgecolor="grey",
                fontsize=10,
                title=color_by.replace("_", " ").title(),
                title_fontsize=11,
            )
            legend.get_title().set_fontweight("bold")
            legend.get_frame().set_linewidth(1.5)
            legend.get_frame().set_alpha(0.95)
        else:
            legend_text = f"{x.replace('_', ' ').title()}  |  Bins: {bins}  |  N: {n:,}"
            leg = ax.legend(
                [legend_text],
                bbox_to_anchor=(1.02, 1.0),
                loc="upper left",
                frameon=True,
                facecolor="white",
                edgecolor="grey",
                fontsize=10,
                title="Distribution Info",
                title_fontsize=11,
            )
            leg.get_title().set_fontweight("bold")
            leg.get_frame().set_linewidth(1.5)
            leg.get_frame().set_alpha(0.95)

    # ─── Layout ────────────────────────────────────────────────────────────────
    has_subtitle = bool(subtitle and str(subtitle).strip())
    if show_legend:
        fig.tight_layout(rect=(0, 0, 0.85, 0.9 if has_subtitle else 0.94))
    else:
        fig.tight_layout(rect=(0, 0, 1, 0.9 if has_subtitle else 0.94))
    plt.show()
    return fig, ax


def histogram_interactive(
    df,
    x,
    bins=20,
    title=None,
    subtitle=None,
    color_by=None,
    style_mode="viridis",
    thresholds=None,
):
    """Interactive Plotly histogram."""
    set_style(style_mode)

    vals = df[x].dropna()
    n = len(vals)

    fig = go.Figure()

    if color_by and color_by in df.columns:
        # Multi-variable histogram
        color_var_vals = df[color_by].dropna().unique()
        cmap = cm.get_cmap("viridis")

        for i, cv in enumerate(color_var_vals):
            subset = df[df[color_by] == cv][x].dropna()
            rgba = cmap(i / len(color_var_vals))
            color_str = f"rgba({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)},{rgba[3]})"

            fig.add_trace(go.Histogram(
                x=subset,
                nbinsx=bins,
                name=f"{color_by}={cv}",
                marker_color=color_str,
                opacity=0.7,
            ))
    else:
        # Single histogram
        cmap = cm.get_cmap("viridis")
        rgba = cmap(0.6)
        color_str = f"rgba({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)},{rgba[3]})"

        fig.add_trace(go.Histogram(
            x=vals,
            nbinsx=bins,
            marker_color=color_str,
            marker_line=dict(color="white", width=0.5),
        ))

    # Threshold lines
    if thresholds:
        for thresh in thresholds:
            thresh_color = get_color('threshold', style_mode)
            fig.add_vline(x=thresh, line=dict(color=thresh_color, width=2, dash="dash"),
                         annotation_text=f"{thresh}", annotation_position="top")

    # Build title
    default_title = f"Distribution of {x.replace('_', ' ').title()}"
    full_title = f"<b>{title or default_title}</b><br><span style='color:grey'>{subtitle or ''}</span>"

    fig.update_layout(
        title=full_title,
        xaxis_title=x.replace("_", " ").title(),
        yaxis_title="Frequency",
        template="plotly_white",
        height=600,
        width=900,
        barmode="overlay" if color_by else "group",
        showlegend=bool(color_by),
    )

    return fig
