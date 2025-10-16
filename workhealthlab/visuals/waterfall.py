"""
waterfall.py — Sociopath-it Visualization Module
-----------------------------------------------
Impact waterfall chart showing cumulative deltas with clear color cues.

Features:
- Connector lines through bar centers
- Color-coded +/- annotations
- Interactive Plotly version
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib import cm
from ..utils.style import set_style, apply_titles


def waterfall(
    df,
    x,
    y,
    title=None,
    subtitle=None,
    style_mode="viridis",
    connector_lines=True,
    annotate=True,
    pos_color="#2E8B57",
    neg_color="#4682B4",
):
    """
    Sociopath-it impact waterfall plot.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    x, y : str
        Column names.
    title, subtitle : str
        Plot titles.
    style_mode : str
        Style theme.
    connector_lines : bool
        Show lines connecting bar centers.
    annotate : bool
        Show +/- value annotations.
    pos_color : str
        Base color for positive changes (green default).
    neg_color : str
        Base color for negative changes (blue default).
    """
    set_style(style_mode)
    df = df.sort_values(x).reset_index(drop=True)
    deltas = df[y].diff().dropna().values
    start_val = df[y].iloc[0]
    labels = [f"{p}→{n}" for p, n in zip(df[x].iloc[:-1], df[x].iloc[1:])]

    # Calculate color intensity
    vmax = np.max(deltas[deltas > 0], initial=1) if np.any(deltas > 0) else 1
    vmin = np.min(deltas[deltas < 0], initial=-1) if np.any(deltas < 0) else -1

    fig, ax = plt.subplots(figsize=(10, 6), dpi=130)
    fig.set_facecolor("white")
    ax.set_facecolor("white")

    # Use actual colormaps
    pos_cmap = cm.get_cmap("Greens")
    neg_cmap = cm.get_cmap("Blues_r")

    cumulative = start_val
    cumulative_tops = [start_val]

    for i, d in enumerate(deltas):
        is_pos = d >= 0
        # Calculate intensity (0.3 to 0.9 range for visibility)
        if is_pos and vmax != 0:
            norm_val = 0.3 + 0.6 * (d / vmax)
        elif not is_pos and vmin != 0:
            norm_val = 0.3 + 0.6 * (abs(d) / abs(vmin))
        else:
            norm_val = 0.6

        bar_col = pos_cmap(norm_val) if is_pos else neg_cmap(norm_val)

        # Draw bar
        ax.bar(i, d, bottom=cumulative, color=bar_col,
               edgecolor="white", linewidth=0.8, zorder=2)

        # Annotation with improved visibility
        if annotate:
            sign = "+" if is_pos else "−"
            text_y = cumulative + d/2
            # Convert RGBA to hex if needed for edge color
            edge_col = bar_col if isinstance(bar_col, str) else f"#{int(bar_col[0]*255):02x}{int(bar_col[1]*255):02x}{int(bar_col[2]*255):02x}"
            ax.text(i, text_y, f"{sign}{abs(d):.1f}",
                   ha="center", va="center",
                   fontsize=10, weight="bold",
                   color="#111111",  # Always dark text for readability
                   bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                            edgecolor=edge_col, linewidth=1.8, alpha=0.95),
                   zorder=3)

        cumulative += d
        cumulative_tops.append(cumulative)

    # Connector lines
    if connector_lines:
        for i in range(len(deltas)):
            x1, y1 = i, cumulative_tops[i]
            x2, y2 = i+1, cumulative_tops[i+1]
            ax.plot([x1, x2], [y1, y2], color="grey", linestyle="--",
                   linewidth=1.2, alpha=0.6, zorder=1)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10, color="grey")
    ax.set_ylabel(y.replace("_", " ").title(), fontsize=12, weight="bold", color="black")
    ax.grid(axis="y", linestyle=":", linewidth=0.7, color="grey", alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    apply_titles(fig, title or f"Waterfall of {y.replace('_', ' ').title()}", subtitle)
    fig.tight_layout(rect=(0, 0, 1, 0.9 if subtitle else 0.94))
    plt.show()
    return fig, ax


def waterfall_interactive(
    df,
    x,
    y,
    title=None,
    subtitle=None,
    connector_lines=True,
    pos_color="#2E8B57",
    neg_color="#4682B4",
):
    """Interactive Plotly waterfall chart."""
    df = df.sort_values(x).reset_index(drop=True)
    values = df[y].values
    deltas = np.diff(values)

    # Prepare measure types: relative for deltas, total for endpoints
    measures = ["absolute"] + ["relative"] * len(deltas)
    display_values = [values[0]] + list(deltas)

    # Calculate intensities for colors
    vmax = np.max(deltas[deltas > 0], initial=1) if np.any(deltas > 0) else 1
    vmin = np.min(deltas[deltas < 0], initial=-1) if np.any(deltas < 0) else -1

    # Color each bar
    colors = []
    pos_cmap = cm.get_cmap("Greens")
    neg_cmap = cm.get_cmap("Blues_r")

    # First bar
    first_rgba = pos_cmap(0.6)
    colors.append(f"rgba({int(first_rgba[0]*255)},{int(first_rgba[1]*255)},{int(first_rgba[2]*255)},{first_rgba[3]})")

    for d in deltas:
        if d >= 0:
            norm_val = 0.3 + 0.6 * (d / vmax) if vmax != 0 else 0.6
            rgba = pos_cmap(norm_val)
        else:
            norm_val = 0.3 + 0.6 * (abs(d) / abs(vmin)) if vmin != 0 else 0.6
            rgba = neg_cmap(norm_val)
        colors.append(f"rgba({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)},{rgba[3]})")

    labels = [df[x].iloc[0]] + [f"{p}→{n}" for p, n in zip(df[x].iloc[:-1], df[x].iloc[1:])]

    # Determine increasing/decreasing/totals colors from the provided colors list
    # For Plotly waterfall, we use increasing/decreasing/totals parameters instead of marker.color
    fig = go.Figure(go.Waterfall(
        x=labels,
        y=display_values,
        measure=measures,
        textposition="inside",
        text=[f"+{v:.1f}" if v > 0 else f"−{abs(v):.1f}" for v in display_values],
        connector={"line": {"color": "grey", "dash": "dash", "width": 1.5}} if connector_lines else {"visible": False},
        increasing={"marker": {"color": pos_color}},
        decreasing={"marker": {"color": neg_color}},
        totals={"marker": {"color": "steelblue"}},
    ))

    fig.update_layout(
        title=f"<b>{title or 'Waterfall Chart'}</b><br><span style='color:grey'>{subtitle or ''}</span>",
        yaxis_title=y.replace("_", " ").title(),
        template="plotly_white",
        height=600,
        width=1000,
        showlegend=False,
    )

    return fig
