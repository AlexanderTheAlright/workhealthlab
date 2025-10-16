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
    fig.set_facecolor("white")
    ax.set_facecolor("white")

    # Create heatmap with borders - disable default annotations if we want custom white-bordered ones
    corr_data = df.corr()
    sns.heatmap(corr_data, cmap=cmap, annot=False, fmt=".2f",
                cbar_kws={"shrink": 0.8}, center=0 if "Rd" in cmap else None,
                vmin=-1 if "Rd" in cmap else None, vmax=1 if "Rd" in cmap else None,
                linewidths=1.5, linecolor='white',  # Add cell borders
                ax=ax)

    # Add custom white-bordered annotations if requested
    if annot:
        for i in range(len(corr_data.index)):
            for j in range(len(corr_data.columns)):
                value = corr_data.iloc[i, j]
                ax.text(j + 0.5, i + 0.5, f'{value:.2f}',
                       ha='center', va='center',
                       fontsize=10, weight='bold', color='black',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                edgecolor='#333333', linewidth=1.5, alpha=0.95))

    # Bold axis labels
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11, fontweight='bold', color='black')
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=11, fontweight='bold', color='black', rotation=0)

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

    # Create heatmap with annotations that have white backgrounds
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale=cmap,
        text=corr.values,
        texttemplate='<b>%{text:.2f}</b>',  # Bold text values
        textfont={"size": 11, "color": "black", "family": "Arial Black"},  # Bold font
        colorbar=dict(title="Correlation"),
        hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>',
        xgap=2,  # Add gap between cells for border effect
        ygap=2,
    ))

    # Add white-bordered annotations using scatter traces with text
    for i, row_label in enumerate(corr.index):
        for j, col_label in enumerate(corr.columns):
            value = corr.iloc[i, j]
            fig.add_annotation(
                x=j, y=i,
                text=f"<b>{value:.2f}</b>",
                showarrow=False,
                font=dict(size=11, color="black", family="Arial Black"),
                bgcolor="rgba(255, 255, 255, 0.95)",
                bordercolor="#333333",
                borderwidth=1.5,
                borderpad=4,
            )

    fig.update_layout(
        title=f"<b>{title or 'Correlation Heatmap'}</b><br><span style='color:grey'>{subtitle or ''}</span>",
        xaxis_title="",
        yaxis_title="",
        template="plotly_white",
        height=600,
        width=700,
        xaxis=dict(side="bottom", tickfont=dict(size=11, color="black", family="Arial Black")),  # Bold x-axis
        yaxis=dict(autorange="reversed", tickfont=dict(size=11, color="black", family="Arial Black")),  # Bold y-axis
    )

    return fig
