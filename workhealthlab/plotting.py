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

def scatterplot(
    df,
    x,
    y,
    group=None,
    title="Scatter Plot",
    subtitle="",
    palette=None,
    n=None,
    ci=True,
    line=True,
    smooth=False,
    legend_title=None,
    alpha=0.8,
    s=50,
    figsize=(8, 6),
):
    """
    WorkHealthLab scatter plot with optional semantic coloring and group-wise trend lines.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing variables.
    x, y : str
        Columns to plot.
    group : str, optional
        Column used to color-code points (e.g., Positive / Neutral / Negative).
    title, subtitle : str, optional
        Main and sub titles.
    palette : dict, optional
        Mapping of group names to colors.
    n : int, optional
        Sample size to display in title.
    ci : bool, default True
        Show 95% confidence interval area around trend lines.
    line : bool, default True
        Draw linear regression line for each group.
    smooth : bool, default False
        Apply light smoothing (lowess).
    legend_title : str, optional
        Custom title for legend.
    alpha : float, default 0.8
        Point transparency.
    s : float, default 50
        Marker size.
    figsize : tuple, default (8, 6)
        Figure size.
    """

    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from scipy.stats import linregress
    from .style import set_workhealth_style, apply_titles, generate_semantic_palette

    set_workhealth_style()
    fig, ax = plt.subplots(figsize=figsize, dpi=130)
    fig.set_facecolor("white")
    ax.set_facecolor("white")

    # ─── Generate color palette if needed ────────────────────────────────────────
    if group and palette is None:
        groups = df[group].unique().tolist()
        thirds = max(1, len(groups) // 3)
        group_dict = {
            "positive": groups[:thirds],
            "neutral": groups[thirds : 2 * thirds],
            "negative": groups[2 * thirds :],
        }
        palette = generate_semantic_palette(group_dict)
    elif palette is None:
        palette = {"default": plt.cm.viridis(0.7)}

    # ─── Plot scatter points ─────────────────────────────────────────────────────
    if group:
        for g, dfg in df.groupby(group):
            color = palette.get(g, plt.cm.Greys(0.5))
            ax.scatter(
                dfg[x],
                dfg[y],
                s=s,
                alpha=alpha,
                color=color,
                edgecolor="white",
                linewidth=0.6,
                label=str(g),
                zorder=3,
            )

            # ─── Add trend line per group ────────────────────────────────────────
            if line and len(dfg) >= 3:
                x_vals = np.linspace(dfg[x].min(), dfg[x].max(), 200)
                if smooth:
                    from statsmodels.nonparametric.smoothers_lowess import lowess

                    fitted = lowess(dfg[y], dfg[x], frac=0.4, return_sorted=True)
                    ax.plot(
                        fitted[:, 0],
                        fitted[:, 1],
                        color=color,
                        lw=2,
                        zorder=4,
                    )
                else:
                    slope, intercept, *_ = linregress(dfg[x], dfg[y])
                    y_pred = intercept + slope * x_vals
                    ax.plot(x_vals, y_pred, color=color, lw=2, zorder=4)

                    # Confidence interval
                    if ci:
                        y_std = np.std(dfg[y] - (intercept + slope * dfg[x]))
                        ci_band = 1.96 * y_std
                        ax.fill_between(
                            x_vals,
                            y_pred - ci_band,
                            y_pred + ci_band,
                            color=color,
                            alpha=0.15,
                            zorder=2,
                        )

    else:
        ax.scatter(df[x], df[y], s=s, alpha=alpha, color=palette["default"],
                   edgecolor="white", linewidth=0.5)

    # ─── Axis labeling ───────────────────────────────────────────────────────────
    ax.set_xlabel(x.replace("_", " ").title(), fontsize=12, weight="bold", color="grey")
    ax.set_ylabel(y.replace("_", " ").title(), fontsize=12, weight="bold", color="grey")

    # ─── Grid & spines ───────────────────────────────────────────────────────────
    ax.grid(axis="y", color="grey", linestyle=":", linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color("grey")

    # ─── Titles outside plot ─────────────────────────────────────────────────────
    apply_titles(fig, title, subtitle, n=n)

    # ─── Enhanced Legend ─────────────────────────────────────────────────────────
    if group:
        leg_title = legend_title or group.replace("_", " ").title()
        legend = ax.legend(
            title=leg_title,
            bbox_to_anchor=(1.02, 1.0),
            loc="upper left",
            fontsize=10.5,
            title_fontsize=12,
            frameon=True,
            facecolor="white",
            edgecolor="lightgrey",
            fancybox=True,
        )
        legend.get_frame().set_linewidth(1)
        legend.get_frame().set_alpha(0.9)
        plt.subplots_adjust(right=0.78)

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

# ══════════════════════════════════════════════════════════════════════════════
# STACKED RESPONSE BAR CHART (generic Likert-style visualization)
# ══════════════════════════════════════════════════════════════════════════════

def stacked_responses(
    df,
    response_col,
    group_col,
    group_map=None,
    order=None,
    response_order=None,
    title=None,
    subtitle=None,
    colors=None,
    figsize=(7, 4),
    cmap="viridis",
    annotate=True,
):
    """
    Create a stacked bar chart showing proportions of categorical responses
    (e.g., Likert items) across survey groups or waves.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the response and group columns.
    response_col : str
        Column name containing categorical responses.
    group_col : str
        Column name indicating grouping or survey wave.
    group_map : dict, optional
        Optional mapping of raw group IDs to human-readable labels.
        Example: {'wave1': 'April 2025', 'wave2': 'September 2025'}.
    order : list, optional
        Ordered list of group labels for the x-axis.
    response_order : list, optional
        Ordered list of response categories from lowest → highest agreement.
        Example: ["Strongly disagree", "Disagree", "Agree", "Strongly agree"].
    title : str, optional
        Main title text. Defaults to 'Response distribution by group'.
    subtitle : str, optional
        Subtitle below the main title.
    colors : list, optional
        List of RGBA or hex colors matching response_order length.
    figsize : tuple, default (7, 4)
        Figure size in inches.
    cmap : str, default 'viridis'
        Matplotlib colormap name used if colors not provided.
    annotate : bool, default True
        Whether to show percentage labels within stacked bars.

    Returns
    -------
    pd.DataFrame
        DataFrame of proportions used for plotting.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib import cm
    from matplotlib.ticker import FuncFormatter

    set_workhealth_style()

    # --- Validation ---
    if response_col not in df.columns or group_col not in df.columns:
        raise ValueError(f"Columns {response_col} or {group_col} not found in dataframe.")

    # --- Clean subset ---
    sub = df.dropna(subset=[response_col, group_col]).copy()

    # Map group IDs → readable labels (if provided)
    if group_map is not None:
        sub[group_col] = sub[group_col].map(group_map).fillna(sub[group_col])

    # Determine unique order of groups
    group_labels = order or list(sub[group_col].unique())

    # Determine response order (Likert levels)
    if response_order is None:
        # Use observed order sorted alphabetically (safer default)
        response_order = sorted(sub[response_col].dropna().unique().tolist())

    # --- Compute proportions ---
    grp = sub.groupby(group_col)[response_col].value_counts(normalize=True).unstack(fill_value=0)
    for r in response_order:
        if r not in grp.columns:
            grp[r] = 0
    grp = grp[response_order]
    grp = grp.reindex(group_labels)
    out = grp.reset_index().rename(columns={group_col: "group"})
    out["group"] = pd.Categorical(out["group"], categories=group_labels, ordered=True)

    # --- Setup plot ---
    fig, ax = plt.subplots(figsize=figsize, dpi=130)
    fig.set_facecolor("white")
    ax.set_facecolor("white")

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.7)
    ax.grid(axis="x", visible=False)

    # --- Colors ---
    if colors is None:
        cmap_obj = cm.get_cmap(cmap)
        step = np.linspace(0.4, 0.9, len(response_order))
        colors = [cmap_obj(s) for s in step]

    # --- Draw bars ---
    x = np.arange(len(out))
    width = 0.6
    bottoms = np.zeros(len(out))
    for col, color in zip(response_order, colors):
        vals = out[col].values
        bars = ax.bar(
            x, vals, width, bottom=bottoms,
            label=col, color=color, edgecolor="white", linewidth=0.6
        )

        if annotate:
            for i, (bar, v) in enumerate(zip(bars, vals)):
                if np.isnan(v) or v <= 0:
                    continue
                x_c = bar.get_x() + bar.get_width() / 2
                y_c = bottoms[i] + v / 2
                ax.text(
                    x_c, y_c, f"{v:.0%}",
                    ha="center", va="center",
                    fontsize=8, weight="bold", family="Arial", color="#333333"
                )
        bottoms += vals

    # --- Axes ---
    ax.set_xticks(x)
    ax.set_xticklabels(out["group"], fontsize=9, family="Arial")
    ax.set_ylabel("Share of respondents", fontsize=10, family="Arial")
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0%}"))

    # --- Titles ---
    default_title = "Response distribution by group"
    apply_titles(fig, title or default_title, subtitle)

    # --- Legend (reverse order so top = positive) ---
    handles, labels = ax.get_legend_handles_labels()
    leg = ax.legend(
        handles[::-1], labels[::-1],
        loc="upper left", bbox_to_anchor=(1.02, 1.0),
        frameon=True, fontsize=8, title="Response"
    )
    leg.get_frame().set_edgecolor("lightgray")
    leg.get_frame().set_linewidth(0.8)
    leg.get_frame().set_facecolor("#f9f9f9")
    if leg.get_title():
        leg.get_title().set_fontfamily("Arial")

    plt.tight_layout()
    plt.show()
    return out


# ══════════════════════════════════════════════════════════════════════════════
# THEME DISTRIBUTION PIE CHART
# ══════════════════════════════════════════════════════════════════════════════

def pie_chart(
    df,
    category_col,
    value_col=None,
    title="Distribution of Categories",
    subtitle=None,
    cmap="viridis",
    startangle=140,
    autopct="%1.1f%%",
    figsize=(7, 7),
    top_n=None,
    min_pct=None,
):
    """
    Plot a pie chart showing the distribution of a categorical variable.
    Designed for displaying coded open-text data themes, categories, or labels.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing category and (optional) weights.
    category_col : str
        Column name representing the categorical variable.
    value_col : str, optional
        Optional column for weights. If None, simple counts are used.
    title : str, default "Distribution of Categories"
        Main title of the plot.
    subtitle : str, optional
        Subtitle placed below the title.
    cmap : str, default "viridis"
        Matplotlib colormap for slice colors.
    startangle : int, default 140
        Starting rotation angle for the pie chart.
    autopct : str, default "%1.1f%%"
        Format for automatic percentage labels.
    figsize : tuple, default (7, 7)
        Figure size in inches.
    top_n : int, optional
        Show only the top N categories (aggregating remainder as "Other").
    min_pct : float, optional
        Collapse any category below this threshold (0–1) into "Other".

    Returns
    -------
    pd.Series
        Series of category proportions used in the plot.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib import cm

    set_workhealth_style()

    # --- Validate ---
    if category_col not in df.columns:
        raise ValueError(f"'{category_col}' not found in dataframe.")
    if value_col and value_col not in df.columns:
        raise ValueError(f"'{value_col}' not found in dataframe.")

    # --- Aggregate data ---
    if value_col:
        counts = df.groupby(category_col)[value_col].sum().sort_values(ascending=False)
    else:
        counts = df[category_col].value_counts(dropna=False)

    total = counts.sum()
    proportions = counts / total

    # --- Apply top_n or min_pct cutoffs ---
    if top_n is not None:
        if len(proportions) > top_n:
            top_items = proportions.iloc[:top_n]
            other_sum = proportions.iloc[top_n:].sum()
            proportions = pd.concat([top_items, pd.Series({"Other": other_sum})])
    elif min_pct is not None:
        small = proportions[proportions < min_pct].sum()
        proportions = proportions[proportions >= min_pct]
        if small > 0:
            proportions.loc["Other"] = small

    proportions = proportions.sort_values(ascending=False)

    # --- Color palette ---
    cmap_obj = cm.get_cmap(cmap)
    colors = [cmap_obj(x) for x in np.linspace(0.3, 0.85, len(proportions))]

    # --- Plot ---
    fig, ax = plt.subplots(figsize=figsize, dpi=130)
    wedges, texts, autotexts = ax.pie(
        proportions,
        labels=proportions.index,
        autopct=autopct,
        startangle=startangle,
        counterclock=False,
        wedgeprops={"linewidth": 1, "edgecolor": "white"},
        textprops={"fontsize": 10, "family": "Arial"},
        colors=colors,
    )

    for t in texts:
        t.set_fontweight("bold")
        t.set_color("#333333")

    ax.set_title(title, fontsize=14, fontweight="bold", family="Arial", pad=20)
    apply_titles(fig, title, subtitle)

    plt.tight_layout()
    plt.show()
    return proportions

# ══════════════════════════════════════════════════════════════════════════════
# DISTRIBUTION HISTOGRAM (flexible, grouped + colored by variable)
# ══════════════════════════════════════════════════════════════════════════════

def distribution_chart(
    df,
    value_col,
    group_col,
    color_by=None,
    bins=10,
    exclude=None,
    figsize=(12, 8),
    title="Distribution of Values",
    subtitle="Distributions stacked by group; colors reflect group attributes",
    cmap_strategy="semantic",  # "semantic" or "continuous"
    color_thresholds=None,     # tuple of (low, high) if semantic
    cmap_positive="viridis",
    cmap_neutral="Greys",
    cmap_negative="autumn_r",
):
    """
    Create a stacked histogram of a continuous variable grouped by a categorical variable,
    optionally colored by another column or by mean group value.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    value_col : str
        Column name containing numeric or continuous values.
    group_col : str
        Column representing categories or groups to stack.
    color_by : str, optional
        Column used to define coloring. If None, uses the mean of `value_col` per group.
    bins : int, default 10
        Number of bins for the histogram.
    exclude : list, optional
        Groups to exclude.
    figsize : tuple, default (12, 8)
        Figure size in inches.
    title : str
        Main title.
    subtitle : str
        Subtitle below main title.
    cmap_strategy : {"semantic", "continuous"}, default "semantic"
        - "semantic" assigns distinct colormaps to high/neutral/low groups.
        - "continuous" uses one gradient colormap scaled to the `color_by` variable.
    color_thresholds : tuple of (low, high), optional
        Thresholds to define semantic groups if `cmap_strategy="semantic"`.
        Defaults to 33rd and 67th percentiles.
    cmap_positive, cmap_neutral, cmap_negative : str
        Colormap names for semantic color grouping.

    Returns
    -------
    dict
        Dictionary containing group means and color palette.
    """
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.patches import Rectangle

    set_workhealth_style()
    exclude = exclude or []

    # --- Data cleaning ---
    sub = df[~df[group_col].isin(exclude)].copy()
    sub[value_col] = pd.to_numeric(sub[value_col], errors="coerce")
    sub = sub.dropna(subset=[value_col])

    # --- Compute group-level color metric ---
    if color_by and color_by in sub.columns:
        metric = sub.groupby(group_col)[color_by].mean()
    else:
        metric = sub.groupby(group_col)[value_col].mean()

    # --- Color generation strategy ---
    from .style import generate_semantic_palette
    if cmap_strategy == "semantic":
        # Define thresholds if not provided
        if color_thresholds is None:
            q_low, q_high = metric.quantile([0.33, 0.67])
        else:
            q_low, q_high = color_thresholds

        pos_vars = sorted(metric[metric > q_high].index.tolist())
        neg_vars = sorted(metric[metric < q_low].index.tolist())
        neu_vars = sorted(metric[(metric >= q_low) & (metric <= q_high)].index.tolist())

        groups = {"positive": pos_vars, "neutral": neu_vars, "negative": neg_vars}
        palette = generate_semantic_palette(groups)

    else:  # continuous colormap strategy
        cmap = cm.get_cmap(cmap_positive)
        normed = (metric - metric.min()) / (metric.max() - metric.min() + 1e-9)
        palette = {g: cmap(v) for g, v in zip(metric.index, normed)}

    # --- Plot ---
    fig, ax = plt.subplots(figsize=figsize, dpi=130)
    fig.set_facecolor("white")
    ax.set_facecolor("white")

    hue_order = list(metric.index)
    sns.histplot(
        data=sub,
        x=value_col,
        hue=group_col,
        hue_order=hue_order,
        multiple="stack",
        bins=bins,
        palette=palette,
        edgecolor="white",
        linewidth=0.5,
        legend=False,
        ax=ax,
    )

    # --- Aesthetics ---
    ax.grid(axis="y", color="grey", linestyle=":", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("grey")
    ax.spines["bottom"].set_color("grey")
    ax.tick_params(axis="x", colors="grey")
    ax.tick_params(axis="y", colors="grey", length=0)
    ax.set_xlabel(value_col.replace("_", " ").title(), fontsize=12, weight="bold", color="grey")
    ax.set_ylabel("Count", fontsize=12, weight="bold", color="grey")

    n = len(sub)
    apply_titles(fig, f"{title} (n={n})", subtitle)

    # --- Legend drawing ---
    line_height = 0.035
    group_spacing = 0.04
    col1_x, col2_x = 1.02, 1.26
    y_start = 1.0 - (line_height * 1.5)
    fig.text(col1_x, 1.0, group_col.replace("_", " ").title(),
             transform=ax.transAxes, fontsize=12, weight="bold", color="grey", ha="left", va="top")

    def draw_group(title, vars_list, x, y):
        if not vars_list:
            return y
        fig.text(x, y, title, transform=ax.transAxes,
                 fontsize=11, weight="bold", color="#333333", ha="left", va="top")
        y -= line_height
        for var in vars_list:
            if var not in palette:
                continue
            rect = Rectangle((x, y - (line_height * 0.7)), 0.015, 0.025,
                             facecolor=palette[var], transform=ax.transAxes, clip_on=False)
            ax.add_patch(rect)
            fig.text(x + 0.02, y, str(var), transform=ax.transAxes,
                     fontsize=10, color="#333333", ha="left", va="top")
            y -= line_height
        return y - group_spacing

    if cmap_strategy == "semantic":
        y_next = draw_group("High Values", groups["positive"], col1_x, y_start)
        draw_group("Medium Values", groups["neutral"], col1_x, y_next)
        draw_group("Low Values", groups["negative"], col2_x, y_start)
    else:
        # For continuous color mapping, single legend
        draw_group("Groups", list(palette.keys()), col1_x, y_start)

    plt.subplots_adjust(right=0.72)
    plt.show()

    return {"metric": metric, "palette": palette}

# ══════════════════════════════════════════════════════════════════════════════
# TREND CHART (flexible line/bar visualizer)
# ══════════════════════════════════════════════════════════════════════════════

def trend_chart(
    df,
    x,
    y,
    group=None,
    kind="line",  # 'line', 'bar', or 'both'
    color_by=None,
    title="Trend Over Time",
    subtitle=None,
    figsize=(10, 6),
    marker="o",
    linestyle="-",
    linewidth=2.0,
    bar_alpha=0.25,
    palette_strategy="semantic",  # 'semantic' or 'continuous'
    cmap_positive="viridis",
    cmap_neutral="Greys",
    cmap_negative="autumn_r",
    color_thresholds=None,
    legend_title=None,
    smooth=False,
    annotate_points=True,
    fmt="{:.1f}",
):
    """
    Create a flexible, WorkHealthLab-styled trend visualizer.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    x : str
        Column representing the time or categorical sequence.
    y : str
        Column representing the numeric value to plot.
    group : str, optional
        Column name representing subgroups (lines or bars per category).
    kind : {"line", "bar", "both"}, default "line"
        Whether to plot lines, bars, or both.
    color_by : str, optional
        Column used to color-code groups. Defaults to group mean of `y`.
    title : str, optional
        Main title.
    subtitle : str, optional
        Subtitle below the main title.
    figsize : tuple, default (10, 6)
        Figure size.
    marker : str, default "o"
        Matplotlib marker for line plots.
    linestyle : str, default "-"
        Line style for trend lines.
    linewidth : float, default 2.0
        Line width for trend lines.
    bar_alpha : float, default 0.25
        Opacity of background bars when `kind='both'`.
    palette_strategy : {"semantic", "continuous"}, default "semantic"
        Determines how colors are assigned to groups.
    cmap_positive, cmap_neutral, cmap_negative : str
        Colormaps for semantic grouping.
    color_thresholds : tuple, optional
        Low/high thresholds for semantic color assignment (defaults to quantiles).
    legend_title : str, optional
        Title for legend.
    smooth : bool, default False
        Apply light smoothing to line trends (LOESS-style).
    annotate_points : bool, default True
        Annotate final values for each line.
    fmt : str, default "{:.1f}"
        Format string for annotations.

    Returns
    -------
    dict
        Dictionary containing assigned colors and figure reference.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.ndimage import gaussian_filter1d
    from .style import generate_semantic_palette, set_workhealth_style, apply_titles

    set_workhealth_style()

    df = df.copy()
    exclude = df[y].isna()
    if exclude.any():
        df = df[~exclude]

    # --- Color assignment ---
    if group:
        if color_by and color_by in df.columns:
            metric = df.groupby(group)[color_by].mean()
        else:
            metric = df.groupby(group)[y].mean()
    else:
        metric = pd.Series({y: df[y].mean()})

    if palette_strategy == "semantic":
        if color_thresholds is None:
            q_low, q_high = metric.quantile([0.33, 0.67])
        else:
            q_low, q_high = color_thresholds

        pos_vars = metric[metric > q_high].index.tolist()
        neg_vars = metric[metric < q_low].index.tolist()
        neu_vars = metric[(metric >= q_low) & (metric <= q_high)].index.tolist()

        palette = generate_semantic_palette({
            "positive": pos_vars,
            "neutral": neu_vars,
            "negative": neg_vars
        })
    else:
        cmap = plt.cm.get_cmap(cmap_positive)
        normed = (metric - metric.min()) / (metric.max() - metric.min() + 1e-9)
        palette = {g: cmap(v) for g, v in zip(metric.index, normed)}

    # --- Plot setup ---
    fig, ax = plt.subplots(figsize=figsize, dpi=130)
    fig.set_facecolor("white")
    ax.set_facecolor("white")

    # --- Plotting logic ---
    if kind in ["bar", "both"]:
        if group:
            wide_df = df.pivot(index=x, columns=group, values=y)
            wide_df.plot(kind="bar", stacked=False, alpha=bar_alpha,
                         color=[palette.get(g, "lightgrey") for g in wide_df.columns],
                         edgecolor="white", ax=ax)
        else:
            ax.bar(df[x], df[y], color="lightgrey", alpha=bar_alpha, edgecolor="white")

    if kind in ["line", "both"]:
        if group:
            for g, g_df in df.groupby(group):
                color = palette.get(g, "grey")
                g_df = g_df.sort_values(x)
                xs, ys = g_df[x], g_df[y]
                if smooth and len(ys) > 3:
                    ys = gaussian_filter1d(ys, sigma=1)
                ax.plot(xs, ys, color=color, label=str(g),
                        marker=marker, linestyle=linestyle,
                        linewidth=linewidth)
                if annotate_points:
                    ax.text(xs.iloc[-1], ys.iloc[-1],
                            fmt.format(ys.iloc[-1]),
                            ha="left", va="center",
                            fontsize=9, color=color, weight="bold")
        else:
            xs, ys = df[x], df[y]
            ax.plot(xs, ys, color="grey", marker=marker, linewidth=linewidth)
            if annotate_points:
                ax.text(xs.iloc[-1], ys.iloc[-1],
                        fmt.format(ys.iloc[-1]),
                        ha="left", va="center", fontsize=9, color="grey")

    # --- Aesthetics ---
    ax.grid(axis="y", color="grey", linestyle=":", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("grey")
    ax.spines["bottom"].set_color("grey")
    ax.tick_params(axis="x", colors="grey", rotation=30)
    ax.tick_params(axis="y", colors="grey", length=0)
    ax.set_xlabel(x.replace("_", " ").title(), fontsize=12, weight="bold", color="grey")
    ax.set_ylabel(y.replace("_", " ").title(), fontsize=12, weight="bold", color="grey")

    # --- Titles ---
    n = len(df)
    apply_titles(fig, f"{title} (n={n})", subtitle)

    # --- Legend ---
    if group:
        leg_title = legend_title or group.replace("_", " ").title()
        ax.legend(title=leg_title, frameon=False, fontsize=9, loc="best")

    plt.tight_layout()
    plt.show()

    return {"palette": palette, "figure": fig}
