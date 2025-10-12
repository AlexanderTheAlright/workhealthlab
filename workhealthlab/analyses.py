# workhealthlab/analyses.py
"""
WorkHealthLab Analyses
----------------------
Tools for weighted survey analysis:
 - Weighted summaries
 - Weighted crosstabs
 - Weighted regressions (OLS & logit)
 - Weighted barplots (WorkHealthLab aesthetic)

Usage:
    from workhealthlab import analyses as wha
    df = wha.load_dta_like(PATH)
    wha.weighted_summary(df, "jobsat")
    wha.weighted_regression(df, "jobsat ~ mastery + meaning", weight="weight")
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from typing import Optional
import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console

console = Console()


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _detect_weight_col(df: pd.DataFrame) -> Optional[str]:
    """Detect a survey weight variable by common naming conventions."""
    candidates = ["weight", "wgt", "pweight", "pw", "wt"]
    for c in df.columns:
        if c.lower() in candidates:
            return c
    return None


def _weighted_mean(x: pd.Series, w: pd.Series) -> float:
    mask = ~x.isna() & ~w.isna()
    if not mask.any():
        return np.nan
    return np.average(x[mask], weights=w[mask])


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def weighted_summary(df: pd.DataFrame, var: str, weight: Optional[str] = None) -> pd.Series:
    """Weighted descriptive summary for a numeric variable."""
    weight = weight or _detect_weight_col(df)
    if weight is None:
        raise ValueError("No weight column found or provided.")

    x, w = df[var], df[weight]
    mean = _weighted_mean(x, w)
    var_w = np.average((x - mean) ** 2, weights=w)
    se = np.sqrt(var_w / len(x.dropna()))

    res = pd.Series({
        "N": len(x.dropna()),
        "Weighted Mean": mean,
        "Std. Error": se,
        "Min": x.min(),
        "Max": x.max()
    })
    console.print(f"📊 Weighted Summary for [bold]{var}[/bold] (weight={weight})")
    console.print(res)
    return res


def weighted_crosstab(
    df: pd.DataFrame,
    row: str,
    col: Optional[str] = None,
    weight: Optional[str] = None,
    normalize: bool = True,
) -> pd.DataFrame:
    """Produce weighted cross-tabulation (% if normalize=True)."""
    weight = weight or _detect_weight_col(df)
    if weight is None:
        raise ValueError("No weight column found or provided.")

    if col is None:
        grouped = df.groupby(row, observed=True)[weight].sum()
        total = grouped.sum()
        out = (grouped / total * 100) if normalize else grouped
        out = out.rename("Weighted %")
    else:
        pivot = pd.pivot_table(
            df,
            values=weight,
            index=row,
            columns=col,
            aggfunc=np.sum,
            observed=True
        )
        if normalize:
            pivot = pivot.div(pivot.sum(axis=0), axis=1) * 100
        out = pivot

    console.print(f"📈 Weighted Crosstab: [bold]{row}[/bold] × [bold]{col or 'Total'}[/bold]")
    display(out.style.format("{:.1f}").background_gradient(cmap="Blues"))
    return out


def weighted_regression(
    df: pd.DataFrame,
    formula: str,
    weight: Optional[str] = None,
    model: str = "ols",
    robust: bool = True,
):
    """Run weighted OLS or logistic regression with clean output."""
    weight = weight or _detect_weight_col(df)
    if weight is None:
        console.print("⚠️ No weights found — running unweighted model.")
        weight = None

    if model.lower() == "ols":
        mod = smf.wls(formula, data=df, weights=df[weight]) if weight else smf.ols(formula, data=df)
        res = mod.fit(cov_type="HC3" if robust else "nonrobust")
    elif model.lower() in ["logit", "logistic"]:
        mod = smf.glm(formula, data=df, family=sm.families.Binomial(), freq_weights=df[weight] if weight else None)
        res = mod.fit(cov_type="HC3" if robust else "nonrobust")
    else:
        raise ValueError("model must be 'ols' or 'logit'")

    console.rule(f"[bold blue]Weighted {model.upper()} Regression[/bold blue]")
    console.print(res.summary())
    return res


def weighted_barplot(df, var, weight=None, title=None, order=None):
    """Weighted bar chart (WorkHealthLab minimalist style)."""
    weight = weight or _detect_weight_col(df)
    if weight is None:
        raise ValueError("No weight column found or provided.")

    weighted_counts = df.groupby(var, observed=True)[weight].sum()
    prop = weighted_counts / weighted_counts.sum()
    summary = prop.reset_index()
    summary.columns = [var, "Weighted %"]
    summary["Weighted %"] *= 100

    sns.set_theme(style="whitegrid", rc={"axes.facecolor": "white"})
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        summary[var],
        summary["Weighted %"],
        color=plt.cm.viridis(0.6),
        edgecolor="white"
    )

    ax.set_ylabel("Weighted %")
    ax.set_xlabel(var.replace("_", " ").title())
    ax.grid(axis="y", linestyle=":", color="grey", alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", rotation=45)

    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", loc="left")

    for i, v in enumerate(summary["Weighted %"]):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=10)

    plt.tight_layout()
    plt.show()

    return summary
