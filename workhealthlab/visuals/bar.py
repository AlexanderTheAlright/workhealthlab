"""
bar.py — Sociopath-it Visualization Module 🧱
--------------------------------------------
Flexible categorical comparisons:
- vertical, horizontal, or stacked bars
- optional highlight bar
- consistent Sociopath-it styling
- Plotly interactive counterpart
"""

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import make_interp_spline
from ..utils.style import (
    set_style,
    generate_semantic_palette,
    apply_titles,
    get_data_element_kwargs,
)


# ══════════════════════════════════════════════════════════════════════════════
# STATIC VERSION
# ══════════════════════════════════════════════════════════════════════════════

def bar(
    df,
    x,
    y,
    title=None,
    subtitle=None,
    palette=None,
    n=None,
    style_mode="viridis",
    orientation="vertical",        # 'vertical', 'horizontal', 'stacked'
    highlight=None,                # highlight label
    highlight_color="#D62828",
    trace_line=False,
    trace_arrow=True,
    sort="none",                   # 'none', 'asc', or 'desc'
    group_spacing=None,            # e.g. [(0,2), (3,5)] or int for split index
):
    """
    Sociopath-it bar plot with optional sorting, grouping gaps, and curved trace line with arrowhead.
    """

    # ─────────────────────────────────────────────
    # Sort and group spacing
    # ─────────────────────────────────────────────
    df = df.copy()
    if sort == "asc":
        df = df.sort_values(y, ascending=True)
    elif sort == "desc":
        df = df.sort_values(y, ascending=False)

    # Add gaps between groups if requested
    if isinstance(group_spacing, int):
        split_points = [group_spacing]
    elif isinstance(group_spacing, (list, tuple)):
        split_points = [g[1] for g in group_spacing]
    else:
        split_points = []

    # Apply pseudo-gap by inserting NaN rows
    if split_points:
        dfs = []
        last = 0
        for sp in split_points:
            dfs.append(df.iloc[last:sp])
            dfs.append(
                {x: f"", y: np.nan}
            )  # add blank separator
            last = sp
        dfs.append(df.iloc[last:])
        df = (
            pd.concat([pd.DataFrame(d) if not isinstance(d, dict) else pd.DataFrame([d]) for d in dfs])
            .reset_index(drop=True)
        )

    # ─────────────────────────────────────────────
    # Styling setup
    # ─────────────────────────────────────────────
    set_style(style_mode)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=130)
    fig.set_facecolor("white")
    ax.set_facecolor("white")

    if palette is None:
        groups = {"positive": [v for v in df[x].dropna().unique().tolist() if v != ""]}
        palette = generate_semantic_palette(groups, mode=style_mode)

    colors = [
        "white" if v == "" else (
            highlight_color if (highlight and v == highlight) else palette.get(v, cm.get_cmap("viridis")(0.6))
        )
        for v in df[x]
    ]
    kwargs = get_data_element_kwargs()

    # ─────────────────────────────────────────────
    # Main plotting logic
    # ─────────────────────────────────────────────
    if orientation == "horizontal":
        ax.barh(df[x], df[y], color=colors, **kwargs)
        ax.set_xlabel(y.title(), fontsize=12, weight="bold", color="black")
        ax.set_ylabel("")
        for i, val in enumerate(df[y]):
            if not np.isnan(val):
                ax.text(val + (df[y].max() * 0.015), i, f"{val:,}", va="center", fontsize=9, color="grey")

    elif orientation == "stacked":
        cols = [c for c in df.columns if c not in [x, y]]
        bottom = np.zeros(len(df))
        for c in cols:
            vals = df[c].values
            ax.bar(df[x], vals, bottom=bottom, label=c, color=palette.get(c, cm.get_cmap("viridis")(0.6)), **kwargs)
            bottom += vals
        legend = ax.legend(
            bbox_to_anchor=(1.02, 1.0),
            loc="upper left",
            frameon=True,
            facecolor="white",
            edgecolor="grey",
            fontsize=10,
            title="Categories",
            title_fontsize=11,
        )
        legend.get_title().set_fontweight("bold")
        legend.get_frame().set_linewidth(1.5)
        legend.get_frame().set_alpha(0.95)
        ax.set_ylabel("Total", fontsize=12, weight="bold", color="black")
        ax.set_xlabel(x.title(), fontsize=12, weight="bold", color="black")

    else:  # vertical
        ax.bar(df[x], df[y], color=colors, **kwargs)
        ax.set_xlabel(x.title(), fontsize=12, weight="bold", color="black")
        ax.set_ylabel(y.title(), fontsize=12, weight="bold", color="black")

        for i, val in enumerate(df[y]):
            if not np.isnan(val):
                ax.text(i, val + (df[y].max() * 0.03), f"{val:,}", ha="center", fontsize=9, color="grey")

        # ─────────────────────────────────────────────
        # Optional trace line and arrowhead
        # ─────────────────────────────────────────────
        if trace_line:
            # Smooth curve between bar tops
            valid_mask = ~df[y].isna()
            x_idx = np.arange(len(df))[valid_mask]
            y_vals = df[y][valid_mask].values

            # cubic spline smoothing
            spl = make_interp_spline(x_idx, y_vals, k=2)
            xs = np.linspace(x_idx.min(), x_idx.max(), 300)
            ys = spl(xs)

            # draw the curve + dots
            ax.plot(xs, ys, color="grey", lw=1.3, alpha=0.85, zorder=3)
            ax.scatter(x_idx[:-1], y_vals[:-1], color="grey", s=15, zorder=4)

            if trace_arrow:
                # ---- Arrow at end of curve with better positioning ----
                x_end, y_end = xs[-1], ys[-1]
                # Use larger step back for clearer direction
                step_back = min(10, len(xs) // 10)
                x_prev, y_prev = xs[-step_back], ys[-step_back]

                ax.annotate(
                    "",
                    xy=(x_end, y_end),
                    xytext=(x_prev, y_prev),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color="grey",
                        lw=2.0,
                        alpha=0.85,
                        shrinkA=0,
                        shrinkB=0,
                    ),
                    zorder=6,
                )


    # ─────────────────────────────────────────────
    # Styling and finishing touches
    # ─────────────────────────────────────────────
    ax.grid(axis="y" if orientation != "horizontal" else "x", linestyle=":", color="grey", linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    apply_titles(fig, title or f"{y.title()} by {x.title()}", subtitle, n=n)
    # Adjust layout based on orientation (stacked has legend on right)
    if orientation == "stacked":
        fig.tight_layout(rect=(0, 0, 0.85, 0.9 if subtitle else 0.94))
    else:
        fig.tight_layout(rect=(0, 0, 1, 0.9 if subtitle else 0.94))
    plt.show()
    return fig, ax

import plotly.graph_objects as go
import numpy as np
from ..utils.style import set_style, generate_semantic_palette


def bar_interactive(
    df,
    x,
    y,
    title=None,
    subtitle=None,
    style_mode="viridis",
    orientation="vertical",
    highlight=None,
    highlight_color="#D62828",
    trace_line=False,
    color_mode="categorical",  # "categorical" or "continuous"
    group_col=None,
    show_values=True,
):
    """
    Sociopath-it interactive bar chart.

    Parameters
    ----------
    df : pd.DataFrame
        Data for plotting.
    x, y : str
        Variable names for categories and values.
    title, subtitle : str, optional
        Title and subtitle text.
    style_mode : str
        Sociopath-it visual mode ('viridis', 'reviewer3', etc.).
    orientation : str
        'vertical' or 'horizontal'.
    highlight : str, optional
        Category name to highlight.
    trace_line : bool, optional
        Draw connecting line across bar tops.
    color_mode : str
        "categorical" (distinct palette) or "continuous" (value gradient).
    group_col : str, optional
        Optional grouping column for coloring.
    show_values : bool, optional
        Annotate bars with numeric labels.
    """
    set_style(style_mode)

    # --- Color logic ---
    if color_mode == "continuous":
        from matplotlib import cm
        cmap = cm.get_cmap("viridis")
        norm_vals = (df[y] - df[y].min()) / (df[y].max() - df[y].min() + 1e-9)
        colors = [f"rgba({int(r*255)},{int(g*255)},{int(b*255)},{a:.2f})"
                  for r, g, b, a in cmap(norm_vals)]
    elif group_col and group_col in df.columns:
        groups = df[group_col].unique().tolist()
        thirds = max(1, len(groups) // 3)
        group_dict = {
            "positive": groups[:thirds],
            "neutral": groups[thirds:2*thirds],
            "negative": groups[2*thirds:]
        }
        palette = generate_semantic_palette(group_dict, mode=style_mode)
        colors = [palette.get(v, "#888888") for v in df[group_col]]
    else:
        colors = ["#D3D3D3" if highlight and v != highlight else highlight_color for v in df[x]]

    # --- Build figure ---
    fig = go.Figure()

    # Bar layer
    if orientation == "horizontal":
        fig.add_trace(
            go.Bar(
                y=df[x],
                x=df[y],
                orientation="h",
                marker_color=colors,
                hovertemplate="<b>%{y}</b><br>Value: %{x}<extra></extra>",
            )
        )
    else:
        fig.add_trace(
            go.Bar(
                x=df[x],
                y=df[y],
                marker_color=colors,
                hovertemplate="<b>%{x}</b><br>Value: %{y}<extra></extra>",
            )
        )

    # Trace line (optional)
    if trace_line:
        if orientation == "horizontal":
            fig.add_trace(
                go.Scatter(
                    x=df[y],
                    y=df[x],
                    mode="lines+markers",
                    line=dict(color="grey", width=1.2),
                    marker=dict(color="grey", size=6),
                    name="trend",
                    hoverinfo="skip",
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=df[x],
                    y=df[y],
                    mode="lines+markers",
                    line=dict(color="grey", width=1.2),
                    marker=dict(color="grey", size=6),
                    name="trend",
                    hoverinfo="skip",
                )
            )

    # Value annotations
    if show_values:
        if orientation == "horizontal":
            for i, val in enumerate(df[y]):
                fig.add_annotation(
                    x=val,
                    y=df[x][i],
                    text=f"<b>{val:.0f}</b>",
                    showarrow=False,
                    xanchor="left",
                    yanchor="middle",
                    font=dict(size=12, color="#333333"),
                    xshift=10,
                )
        else:
            for i, val in enumerate(df[y]):
                fig.add_annotation(
                    x=df[x][i],
                    y=val,
                    text=f"<b>{val:.0f}</b>",
                    showarrow=False,
                    yanchor="bottom",
                    font=dict(size=12, color="#333333"),
                    yshift=8,
                )

    # Layout styling
    title_dict = {}
    if subtitle:
        # Top-left corner when subtitle present
        title_dict = dict(
            text=f"<b>{title or f'{y.title()} by {x.title()}'}</b>"
                 + f"<br><span style='color:grey;font-size:14px;'>{subtitle}</span>",
            x=0.02,
            xanchor="left",
            yanchor="top",
            y=0.96,
        )
    else:
        # Centered when no subtitle
        title_dict = dict(
            text=f"<b>{title or f'{y.title()} by {x.title()}'}</b>",
            x=0.5,
            xanchor="center",
            yanchor="top",
            y=0.96,
        )

    fig.update_layout(
        template="plotly_white",
        height=600,
        margin=dict(t=90, b=50, l=60, r=30),
        title=title_dict,
        xaxis_title=dict(
            text=x.title() if orientation != "horizontal" else "",
            font=dict(size=12, color="black", family="Arial, sans-serif"),
        ),
        yaxis_title=dict(
            text=y.title() if orientation != "horizontal" else "",
            font=dict(size=12, color="black", family="Arial, sans-serif"),
        ),
        plot_bgcolor="white",
        showlegend=False,
    )

    # Fine-tuning axis fonts
    fig.update_xaxes(showgrid=False, tickfont=dict(size=12, color="#333"))
    fig.update_yaxes(showgrid=True, gridcolor="rgba(180,180,180,0.3)", tickfont=dict(size=12, color="#333"))

    return fig

