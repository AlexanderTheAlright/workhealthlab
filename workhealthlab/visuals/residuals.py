"""
residuals.py  Sociopath-it Residuals Diagnostic Plots
-------------------------------------------------------
Diagnostic plots for regression model assumptions.

Features:
- Residuals vs fitted values
- Q-Q plots for normality
- Scale-location plots
- Residuals vs leverage
- Combined diagnostic panel
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Optional
import warnings
from scipy import stats

warnings.filterwarnings('ignore')

try:
    from ..utils.style import apply_titles, COLORS_DICT
except ImportError:
    def apply_titles(*args, **kwargs):
        pass
    COLORS_DICT = {'viridis': plt.cm.viridis}


def residuals(
    model,
    plot_type: str = "all",
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    style_mode: str = "viridis",
    figsize: tuple = (12, 10),
    output_path: Optional[str] = None,
):
    """
    Create diagnostic residual plots for regression model.

    Parameters
    ----------
    model : RegressionModel
        Fitted regression model.
    plot_type : str, default "all"
        Type of plot: "all", "fitted", "qq", "scale", "leverage".
        - "all": 2x2 panel with all diagnostic plots
        - "fitted": Residuals vs fitted values
        - "qq": Q-Q plot for normality
        - "scale": Scale-location plot (sqrt standardized residuals vs fitted)
        - "leverage": Residuals vs leverage
    title : str, optional
        Plot title.
    subtitle : str, optional
        Plot subtitle.
    style_mode : str, default "viridis"
        Color scheme.
    figsize : tuple, default (12, 10)
        Figure size.
    output_path : str, optional
        Save path.

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    All diagnostics:
    >>> from sociopathit.analyses.regress import ols
    >>> model = ols(df, 'income', ['education', 'age'])
    >>> residuals(model, title='Model Diagnostics')

    Just residuals vs fitted:
    >>> residuals(model, plot_type='fitted')

    Just Q-Q plot:
    >>> residuals(model, plot_type='qq')
    """
    # Extract model information
    if not hasattr(model, 'results'):
        raise TypeError("model must have results attribute (fitted regression model)")

    results = model.results
    fitted = results.fittedvalues
    resid = results.resid
    standardized_resid = resid / np.sqrt(results.scale)

    # Get color
    color_map = COLORS_DICT.get(style_mode, plt.cm.viridis)
    if callable(color_map):
        main_color = color_map(0.6)
    else:
        main_color = 'steelblue'

    # Create plots based on type
    if plot_type == "all":
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()

        # 1. Residuals vs Fitted
        _plot_residuals_fitted(axes[0], fitted, resid, main_color)

        # 2. Q-Q Plot
        _plot_qq(axes[1], standardized_resid, main_color)

        # 3. Scale-Location
        _plot_scale_location(axes[2], fitted, standardized_resid, main_color)

        # 4. Residuals vs Leverage
        _plot_leverage(axes[3], results, standardized_resid, main_color)

    else:
        fig, ax = plt.subplots(figsize=(10, 6))

        if plot_type == "fitted":
            _plot_residuals_fitted(ax, fitted, resid, main_color)
        elif plot_type == "qq":
            _plot_qq(ax, standardized_resid, main_color)
        elif plot_type == "scale":
            _plot_scale_location(ax, fitted, standardized_resid, main_color)
        elif plot_type == "leverage":
            _plot_leverage(ax, results, standardized_resid, main_color)
        else:
            raise ValueError(f"Invalid plot_type: {plot_type}")

    # Title
    if title or subtitle:
        apply_titles(fig, title=title, subtitle=subtitle)

    # Layout
    fig.tight_layout(rect=(0, 0, 1, 0.94 if subtitle or title else 0.98))

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')

    return fig


def _plot_residuals_fitted(ax, fitted, resid, color):
    """Plot residuals vs fitted values."""
    ax.scatter(fitted, resid, alpha=0.5, s=30, color=color, edgecolors='grey', linewidth=0.5)

    # Add horizontal line at 0
    ax.axhline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)

    # Add lowess smoothing line
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        smoothed = lowess(resid, fitted, frac=0.3)
        ax.plot(smoothed[:, 0], smoothed[:, 1], color='blue', linewidth=2, alpha=0.8, label='Lowess')
    except:
        pass

    ax.set_xlabel('Fitted Values', fontsize=11, weight='bold', color='black')
    ax.set_ylabel('Residuals', fontsize=11, weight='bold', color='black')
    ax.set_title('Residuals vs Fitted', fontsize=12, weight='bold', pad=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)


def _plot_qq(ax, standardized_resid, color):
    """Q-Q plot for normality assessment."""
    stats.probplot(standardized_resid, dist="norm", plot=None)
    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(standardized_resid)))
    sample_quantiles = np.sort(standardized_resid)

    # Scatter plot
    ax.scatter(theoretical_quantiles, sample_quantiles, alpha=0.5, s=30,
              color=color, edgecolors='grey', linewidth=0.5)

    # Reference line
    x_range = [theoretical_quantiles.min(), theoretical_quantiles.max()]
    ax.plot(x_range, x_range, 'r--', linewidth=1.5, alpha=0.7)

    ax.set_xlabel('Theoretical Quantiles', fontsize=11, weight='bold', color='black')
    ax.set_ylabel('Standardized Residuals', fontsize=11, weight='bold', color='black')
    ax.set_title('Normal Q-Q Plot', fontsize=12, weight='bold', pad=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)


def _plot_scale_location(ax, fitted, standardized_resid, color):
    """Scale-location plot (spread-location plot)."""
    sqrt_abs_resid = np.sqrt(np.abs(standardized_resid))

    ax.scatter(fitted, sqrt_abs_resid, alpha=0.5, s=30, color=color,
              edgecolors='grey', linewidth=0.5)

    # Add lowess smoothing line
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        smoothed = lowess(sqrt_abs_resid, fitted, frac=0.3)
        ax.plot(smoothed[:, 0], smoothed[:, 1], color='blue', linewidth=2, alpha=0.8)
    except:
        pass

    ax.set_xlabel('Fitted Values', fontsize=11, weight='bold', color='black')
    ax.set_ylabel('√|Standardized Residuals|', fontsize=11, weight='bold', color='black')
    ax.set_title('Scale-Location Plot', fontsize=12, weight='bold', pad=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)


def _plot_leverage(ax, results, standardized_resid, color):
    """Residuals vs leverage plot with Cook's distance."""
    try:
        from statsmodels.stats.outliers_influence import OLSInfluence

        influence = OLSInfluence(results)
        leverage = influence.hat_matrix_diag
        cooks_d = influence.cooks_distance[0]

        ax.scatter(leverage, standardized_resid, alpha=0.5, s=30, color=color,
                  edgecolors='grey', linewidth=0.5)

        # Add Cook's distance contours
        x_range = np.linspace(0, max(leverage), 100)
        for d in [0.5, 1.0]:
            y_pos = np.sqrt(d * len(standardized_resid) * (1 - x_range) / x_range)
            y_neg = -y_pos
            ax.plot(x_range, y_pos, '--', color='red', alpha=0.5, linewidth=1)
            ax.plot(x_range, y_neg, '--', color='red', alpha=0.5, linewidth=1)

        # Highlight high leverage points
        high_leverage = leverage > (2 * len(results.params) / len(leverage))
        if high_leverage.any():
            ax.scatter(leverage[high_leverage], standardized_resid[high_leverage],
                      s=80, facecolors='none', edgecolors='red', linewidth=2)

    except Exception as e:
        warnings.warn(f"Could not compute leverage statistics: {e}")
        ax.text(0.5, 0.5, 'Leverage plot unavailable', ha='center', va='center',
               transform=ax.transAxes, fontsize=12, color='grey')

    ax.set_xlabel('Leverage', fontsize=11, weight='bold', color='black')
    ax.set_ylabel('Standardized Residuals', fontsize=11, weight='bold', color='black')
    ax.set_title('Residuals vs Leverage', fontsize=12, weight='bold', pad=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)


def residuals_histogram(
    model,
    bins: int = 30,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    style_mode: str = "viridis",
    figsize: tuple = (10, 6),
    output_path: Optional[str] = None,
):
    """
    Plot histogram of residuals with normal distribution overlay.

    Parameters
    ----------
    model : RegressionModel
        Fitted regression model.
    bins : int, default 30
        Number of histogram bins.
    title : str, optional
        Plot title.
    subtitle : str, optional
        Plot subtitle.
    style_mode : str, default "viridis"
        Color scheme.
    figsize : tuple, default (10, 6)
        Figure size.
    output_path : str, optional
        Save path.

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> residuals_histogram(model, title='Distribution of Residuals')
    """
    results = model.results
    resid = results.resid
    standardized_resid = resid / np.sqrt(results.scale)

    # Get color
    color_map = COLORS_DICT.get(style_mode, plt.cm.viridis)
    if callable(color_map):
        main_color = color_map(0.6)
    else:
        main_color = 'steelblue'

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Histogram
    n, bins_edges, patches = ax.hist(standardized_resid, bins=bins, density=True,
                                     alpha=0.7, color=main_color, edgecolor='grey')

    # Overlay normal distribution
    x_range = np.linspace(standardized_resid.min(), standardized_resid.max(), 100)
    ax.plot(x_range, stats.norm.pdf(x_range), 'r-', linewidth=2, alpha=0.8,
           label='Normal Distribution')

    # Labels
    ax.set_xlabel('Standardized Residuals', fontsize=12, weight='bold', color='black')
    ax.set_ylabel('Density', fontsize=12, weight='bold', color='black')

    # Legend
    legend = ax.legend(
        bbox_to_anchor=(1.02, 1.0),
        loc='upper left',
        frameon=True,
        facecolor='white',
        edgecolor='grey',
        fontsize=10,
    )
    legend.get_frame().set_linewidth(1.5)
    legend.get_frame().set_alpha(0.95)

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Title
    if title or subtitle:
        apply_titles(fig, title=title, subtitle=subtitle)

    # Layout
    fig.tight_layout(rect=(0, 0, 0.85, 0.9 if subtitle else 0.94))

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')

    return fig
