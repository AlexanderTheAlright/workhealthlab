"""
scatter.py — Sociopath-it Visualization Module
----------------------------------------------
Semantic scatterplots with optional grouping and regression lines.
Supports dynamic theme switching from Sociopath-it style engine.

Features:
- Static and interactive versions
- Group-based coloring
- Regression lines with confidence intervals
- LOWESS smoothing
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.stats import linregress
from matplotlib import cm

# Correct relative import (visuals → utils)
from ..utils.style import (
    set_style,
    apply_titles,
    generate_semantic_palette,
)


# ══════════════════════════════════════════════════════════════════════════════
# SCATTER FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def scatterplot(
    df,
    x,
    y,
    group=None,
    title="Scatter Plot",
    subtitle="",
    palette=None,
    ci=True,
    line=True,
    smooth=False,
    legend_title=None,
    alpha=0.8,
    s=50,
    figsize=(8, 6),
    style_mode="viridis",
):
    """
    Sociopath-it scatter plot with semantic colour logic and regression overlays.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    x, y : str
        Variables to plot.
    group : str, optional
        Column used to colour points by group.
    title, subtitle : str, optional
        Plot titles.
    palette : dict, optional
        Custom colour mapping; auto-generated if None.
    ci : bool, default True
        Show shaded confidence intervals around regression lines.
    line : bool, default True
        Draw regression line(s).
    smooth : bool, default False
        Apply LOWESS smoothing.
    legend_title : str, optional
        Custom legend title.
    alpha : float, default 0.8
        Marker transparency.
    s : int, default 50
        Marker size.
    figsize : tuple, default (8, 6)
        Figure size.
    style_mode : str, default 'viridis'
        One of {'fiery','viridis','sentiment','plainjane','reviewer3'}.
    """
    from statsmodels.nonparametric.smoothers_lowess import lowess

    # Activate style
    set_style(style_mode)
    fig, ax = plt.subplots(figsize=figsize, dpi=130)

    # ─── Palette logic ─────────────────────────────────────────────────────────
    if group and palette is None:
        groups = df[group].dropna().unique().tolist()
        thirds = max(1, len(groups) // 3)
        g_dict = {
            "positive": groups[:thirds],
            "neutral": groups[thirds : 2 * thirds],
            "negative": groups[2 * thirds :],
        }
        palette = generate_semantic_palette(g_dict, mode=style_mode)
    elif palette is None:
        # Default single tone by style
        import matplotlib.cm as cm
        palette = {"default": (
            cm.plasma(0.7) if style_mode == "fiery" else
            cm.get_cmap("viridis")(0.7) if style_mode == "viridis" else
            cm.Greens(0.7) if style_mode == "sentiment" else
            cm.Blues(0.7) if style_mode == "plainjane" else
            cm.Greys(0.5)
        )}

    # ─── Plot points and lines ────────────────────────────────────────────────
    if group:
        for g, dfg in df.groupby(group):
            color = palette.get(g, "grey")
            ax.scatter(
                dfg[x], dfg[y],
                s=s, alpha=alpha,
                color=color,
                edgecolor="white",
                linewidth=0.6,
                label=str(g),
                zorder=3,
            )

            if line and len(dfg) >= 3:
                xs = np.linspace(dfg[x].min(), dfg[x].max(), 200)
                if smooth:
                    fitted = lowess(dfg[y], dfg[x], frac=0.4, return_sorted=True)
                    ax.plot(fitted[:, 0], fitted[:, 1], color=color, lw=2, zorder=4)
                else:
                    slope, intercept, *_ = linregress(dfg[x], dfg[y])
                    y_pred = intercept + slope * xs
                    ax.plot(xs, y_pred, color=color, lw=2, zorder=4)
                    if ci:
                        resid = dfg[y] - (intercept + slope * dfg[x])
                        y_std = np.std(resid)
                        ci_band = 1.96 * y_std
                        ax.fill_between(
                            xs,
                            y_pred - ci_band,
                            y_pred + ci_band,
                            color=color,
                            alpha=0.15,
                            zorder=2,
                        )

    else:
        ax.scatter(df[x], df[y], s=s, alpha=alpha,
                   color=palette["default"],
                   edgecolor="white", linewidth=0.5)
        if line and len(df) >= 3:
            slope, intercept, *_ = linregress(df[x], df[y])
            xs = np.linspace(df[x].min(), df[x].max(), 200)
            y_pred = intercept + slope * xs
            ax.plot(xs, y_pred, color=palette["default"], lw=2)
            if ci:
                resid = df[y] - (intercept + slope * df[x])
                y_std = np.std(resid)
                ci_band = 1.96 * y_std
                ax.fill_between(xs, y_pred - ci_band, y_pred + ci_band,
                                color=palette["default"], alpha=0.15)

    # ─── Axis aesthetics ──────────────────────────────────────────────────────
    ax.set_xlabel(x.replace("_", " ").title(), fontsize=12, weight="bold", color="black")
    ax.set_ylabel(y.replace("_", " ").title(), fontsize=12, weight="bold", color="black")
    ax.grid(axis="y", color="grey", linestyle=":", linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)

    # ─── Titles & Legend ──────────────────────────────────────────────────────
    apply_titles(fig, title, subtitle)
    if group:
        leg_title = legend_title or group.replace("_", " ").title()
        legend = ax.legend(
            title=leg_title,
            bbox_to_anchor=(1.02, 1.0),
            loc="upper left",
            fontsize=10,
            title_fontsize=11,
            frameon=True,
            facecolor="white",
            edgecolor="grey",
        )
        legend.get_title().set_fontweight("bold")
        legend.get_frame().set_alpha(0.95)
        legend.get_frame().set_linewidth(1.5)

    has_subtitle = bool(subtitle and str(subtitle).strip())
    # Adjust layout based on whether legend is present
    if group:
        fig.tight_layout(rect=(0, 0, 0.85, 0.9 if has_subtitle else 0.94))
    else:
        fig.tight_layout(rect=(0, 0, 1, 0.9 if has_subtitle else 0.94))
    plt.show()
    return fig, ax


def scatterplot_interactive(
    df,
    x,
    y,
    group=None,
    title="Scatter Plot",
    subtitle="",
    line=True,
    style_mode="viridis",
):
    """Interactive Plotly scatterplot with regression lines."""
    set_style(style_mode)

    # Palette logic
    if group:
        groups = df[group].dropna().unique().tolist()
        thirds = max(1, len(groups) // 3)
        g_dict = {
            "positive": groups[:thirds],
            "neutral": groups[thirds : 2 * thirds],
            "negative": groups[2 * thirds :],
        }
        palette = generate_semantic_palette(g_dict, mode=style_mode)
        # Convert to hex
        palette_hex = {}
        for k, v in palette.items():
            if isinstance(v, tuple):
                palette_hex[k] = f"rgba({int(v[0]*255)},{int(v[1]*255)},{int(v[2]*255)},{v[3] if len(v)>3 else 1})"
            else:
                palette_hex[k] = v
        palette = palette_hex
    else:
        cmap = cm.get_cmap("viridis")
        rgba = cmap(0.7)
        palette = {"default": f"rgba({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)},{rgba[3]})"}

    fig = go.Figure()

    if group:
        for g, dfg in df.groupby(group):
            color = palette.get(g, "grey")

            # Scatter points
            fig.add_trace(go.Scatter(
                x=dfg[x], y=dfg[y],
                mode="markers",
                name=str(g),
                marker=dict(size=8, color=color, opacity=0.7, line=dict(color="white", width=0.5)),
            ))

            # Regression line
            if line and len(dfg) >= 3:
                slope, intercept, *_ = linregress(dfg[x], dfg[y])
                xs = np.linspace(dfg[x].min(), dfg[x].max(), 100)
                y_pred = intercept + slope * xs
                fig.add_trace(go.Scatter(
                    x=xs, y=y_pred,
                    mode="lines",
                    name=f"{g} (trend)",
                    line=dict(color=color, width=2),
                    showlegend=False,
                ))
    else:
        fig.add_trace(go.Scatter(
            x=df[x], y=df[y],
            mode="markers",
            marker=dict(size=8, color=palette["default"], opacity=0.7,
                       line=dict(color="white", width=0.5)),
            showlegend=False,
        ))

        # Regression line
        if line and len(df) >= 3:
            slope, intercept, *_ = linregress(df[x], df[y])
            xs = np.linspace(df[x].min(), df[x].max(), 100)
            y_pred = intercept + slope * xs
            fig.add_trace(go.Scatter(
                x=xs, y=y_pred,
                mode="lines",
                line=dict(color=palette["default"], width=2),
                showlegend=False,
            ))

    fig.update_layout(
        title=f"<b>{title}</b><br><span style='color:grey'>{subtitle}</span>",
        xaxis_title=x.replace("_", " ").title(),
        yaxis_title=y.replace("_", " ").title(),
        template="plotly_white",
        height=600,
        width=900,
        hovermode="closest",
    )

    return fig
