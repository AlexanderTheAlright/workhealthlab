"""
plotting.py — WorkHealthLab Visualization API
----------------------------------------------
Unified plotting interface for WorkHealthLab.
All functions automatically apply style, typography, and legend conventions.

Supported functions:
- scatterplot()
- barchart()
- histogram()
- heatmap()
- clusterplot()
- waterfallchart()
- factorplot()

Each function accepts:
    df : pandas.DataFrame
    x, y : str
    color / style kwargs
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram

from .style import (
    set_workhealth_style,
    generate_semantic_palette,
    apply_titles,
    get_data_element_kwargs,
    draw_legend_group,
)

# ══════════════════════════════════════════════════════════════════════════════
# SCATTER PLOT
# ══════════════════════════════════════════════════════════════════════════════

def scatterplot(df, x, y, group=None, title=None, subtitle=None, palette=None, n=None):
    """WorkHealthLab scatter plot."""
    set_workhealth_style()

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.set_facecolor("white")
    ax.set_facecolor("white")

    # Generate color palette if grouping variable exists
    if group and palette is None:
        groups = df[group].unique().tolist()
        group_dict = {"positive": groups[: len(groups)//3],
                      "neutral": groups[len(groups)//3: 2*len(groups)//3],
                      "negative": groups[2*len(groups)//3:]}
        palette = generate_semantic_palette(group_dict)

    # Plot data
    if group:
        for g, dfg in df.groupby(group):
            ax.scatter(
                dfg[x], dfg[y],
                s=50, alpha=0.8,
                color=palette.get(g, cm.Greys(0.5)),
                edgecolor="white", linewidth=0.5,
                label=g
            )
    else:
        ax.scatter(df[x], df[y], s=50, alpha=0.7, color=cm.viridis(0.7),
                   edgecolor="white", linewidth=0.5)

    ax.set_xlabel(x.replace("_", " ").title())
    ax.set_ylabel(y.replace("_", " ").title())
    apply_titles(fig, title or f"Scatter of {x} vs {y}", subtitle, n)

    if group:
        ax.legend(title=group.title(), bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.subplots_adjust(right=0.8)

    plt.tight_layout()
    return fig, ax


# ══════════════════════════════════════════════════════════════════════════════
# BAR CHART
# ══════════════════════════════════════════════════════════════════════════════

def barchart(df, x, y, title=None, subtitle=None, palette=None, n=None):
    """WorkHealthLab bar chart."""
    set_workhealth_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.set_facecolor("white")
    ax.set_facecolor("white")

    if palette is None:
        groups = {"positive": df[x].unique().tolist()}
        palette = generate_semantic_palette(groups)

    ax.bar(df[x], df[y], color=[palette[v] for v in df[x]], **get_data_element_kwargs())
    ax.set_xlabel(x.replace("_", " ").title())
    ax.set_ylabel(y.replace("_", " ").title())
    apply_titles(fig, title or f"{y.title()} by {x.title()}", subtitle, n)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig, ax


# ══════════════════════════════════════════════════════════════════════════════
# HISTOGRAM
# ══════════════════════════════════════════════════════════════════════════════

def histogram(df, x, bins=20, title=None, subtitle=None, color=None, n=None):
    """WorkHealthLab histogram."""
    set_workhealth_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    color = color or cm.viridis(0.6)
    ax.hist(df[x].dropna(), bins=bins, color=color, edgecolor="white", linewidth=0.5)
    ax.set_xlabel(x.replace("_", " ").title())
    ax.set_ylabel("Frequency")
    apply_titles(fig, title or f"Distribution of {x}", subtitle, n)
    plt.tight_layout()
    return fig, ax


# ══════════════════════════════════════════════════════════════════════════════
# HEATMAP
# ══════════════════════════════════════════════════════════════════════════════

def heatmap(df, title=None, subtitle=None, cmap="viridis", annot=False):
    """WorkHealthLab correlation or matrix heatmap."""
    set_workhealth_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df.corr(), cmap=cmap, annot=annot, fmt=".2f", cbar_kws={"shrink": 0.8})
    apply_titles(fig, title or "Heatmap", subtitle)
    plt.tight_layout()
    return fig, ax


# ══════════════════════════════════════════════════════════════════════════════
# CLUSTERING DIAGRAM
# ══════════════════════════════════════════════════════════════════════════════

def clusterplot(df, method="ward", metric="euclidean", title=None, subtitle=None):
    """WorkHealthLab hierarchical cluster dendrogram."""
    set_workhealth_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    Z = linkage(df.select_dtypes(include=[np.number]), method=method, metric=metric)
    dendrogram(Z, ax=ax, leaf_rotation=90, leaf_font_size=10)
    apply_titles(fig, title or "Hierarchical Clustering Dendrogram", subtitle)
    plt.tight_layout()
    return fig, ax


# ══════════════════════════════════════════════════════════════════════════════
# FACTOR ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def factorplot(df, n_factors=2, title=None, subtitle=None):
    """WorkHealthLab factor analysis scatter."""
    set_workhealth_style()
    scaler = StandardScaler()
    X = scaler.fit_transform(df.select_dtypes(include=[np.number]))
    fa = FactorAnalysis(n_components=n_factors, random_state=42)
    factors = fa.fit_transform(X)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(factors[:, 0], factors[:, 1], s=40, alpha=0.7, color=cm.viridis(0.7),
               edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Factor 1")
    ax.set_ylabel("Factor 2")
    apply_titles(fig, title or f"Factor Analysis ({n_factors} Factors)", subtitle)
    plt.tight_layout()
    return fig, ax


# ══════════════════════════════════════════════════════════════════════════════
# WATERFALL CHART (as per your provided version)
# ══════════════════════════════════════════════════════════════════════════════

def waterfallchart(df, x, y, title=None, subtitle=None, color_pos="Greens", color_neg="Blues_r"):
    """
    WorkHealthLab impact waterfall chart.
    Requires ordered categorical x and numeric y.
    """
    set_workhealth_style()

    # Prepare data
    df = df.sort_values(x)
    deltas = df[y].diff().dropna().values
    start_val = df[y].iloc[0]
    labels = [f"{p}→{n}" for p, n in zip(df[x].iloc[:-1], df[x].iloc[1:])]
    v_max, v_min = np.max(deltas[deltas > 0], initial=0), np.min(deltas[deltas < 0], initial=0)
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.set_facecolor("white")

    cumulative = start_val
    y_tops = [start_val]
    pos_cmap, neg_cmap = cm.get_cmap(color_pos), cm.get_cmap(color_neg)

    for i, d in enumerate(deltas):
        is_pos = d >= 0
        norm_val = abs(d / (v_max if is_pos else v_min)) if (v_max or v_min) else 1
        bar_col = (pos_cmap if is_pos else neg_cmap)(0.3 + 0.6 * norm_val)
        ax.bar(i, d, bottom=cumulative, color=bar_col, edgecolor="white", linewidth=0.5, width=0.7, zorder=10)

        lbl_txt = f"(+{100*d/cumulative:.0f}%)" if cumulative != 0 else ""
        lbl_y = cumulative + d + (0.05 * np.sign(d))
        ax.text(i, lbl_y, lbl_txt, ha="center",
                va="bottom" if d > 0 else "top", fontsize=9,
                color="#006400" if d > 0 else "#00008B", weight="bold")

        cumulative += d
        y_tops.append(cumulative)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10, color="grey")
    ax.set_ylabel(y.title())
    ax.grid(axis="y", color="grey", linestyle=":", linewidth=0.7)
    apply_titles(fig, title or f"Waterfall Chart of {y.title()} by {x.title()}", subtitle, n=len(df))
    plt.tight_layout()
    return fig, ax
