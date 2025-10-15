"""
heatmap.py — Sociopath-it Visualization Module
----------------------------------------------
Matrix or correlation heatmap with full theme styling.

Features:
- Static matplotlib/seaborn version
- Interactive Plotly version
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from ..utils.style import set_style, apply_titles, get_continuous_cmap


def heatmap(df, title=None, subtitle=None, cmap=None, annot=False, style_mode="viridis"):
    """
    Sociopath-it correlation heatmap.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe for correlation.
    title, subtitle : str, optional
        Plot titles.
    cmap : str, optional
        Custom colormap. If None, uses style_mode's continuous colormap.
    annot : bool, default False
        Show correlation values in cells.
    style_mode : str, default "viridis"
        Style theme: fiery (dark red heat), viridis, sentiment (RdYlGn),
        plainjane (RdBu), reviewer3 (grayscale).
    """
    set_style(style_mode)

    # Use style-specific continuous colormap if not provided
    if cmap is None:
        cmap = get_continuous_cmap(style_mode)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=130)
    sns.heatmap(df.corr(), cmap=cmap, annot=annot, fmt=".2f",
                cbar_kws={"shrink": 0.8}, center=0 if "Rd" in cmap else None,
                vmin=-1 if "Rd" in cmap else None, vmax=1 if "Rd" in cmap else None)
    apply_titles(fig, title or "Correlation Heatmap", subtitle)
    fig.tight_layout(rect=(0, 0, 1, 0.9 if subtitle else 0.94))
    plt.show()
    return fig, ax


def heatmap_interactive(df, title=None, subtitle=None, cmap=None, style_mode="viridis"):
    """
    Interactive Plotly correlation heatmap.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    title, subtitle : str, optional
        Plot titles.
    cmap : str, optional
        Colormap name. If None, uses style_mode's continuous colormap.
    style_mode : str, default "viridis"
        Style theme.
    """
    set_style(style_mode)

    # Use style-specific continuous colormap if not provided
    if cmap is None:
        cmap = get_continuous_cmap(style_mode)

    corr = df.corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale=cmap,
        text=corr.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation"),
        hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>',
    ))

    fig.update_layout(
        title=f"<b>{title or 'Correlation Heatmap'}</b><br><span style='color:grey'>{subtitle or ''}</span>",
        xaxis_title="",
        yaxis_title="",
        template="plotly_white",
        height=600,
        width=700,
        xaxis=dict(side="bottom"),
        yaxis=dict(autorange="reversed"),
    )

    return fig
