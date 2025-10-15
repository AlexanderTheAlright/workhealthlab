"""
margins.py  Sociopath-it Marginal Effects Plots
-------------------------------------------------
Visualize predicted values and marginal effects from regression models.

Features:
- Margin plots showing predicted values across predictor ranges
- Confidence intervals
- Multiple predictor comparison
- Interaction effects visualization
- Support for both continuous and categorical predictors
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Optional, List, Union
import warnings

warnings.filterwarnings('ignore')

try:
    from ..utils.style import apply_titles, COLORS_DICT
except ImportError:
    def apply_titles(*args, **kwargs):
        pass
    COLORS_DICT = {'viridis': plt.cm.viridis}


def margins(
    model,
    variable: str,
    at_values: Optional[Union[List, np.ndarray]] = None,
    other_vars: Optional[dict] = None,
    ci: bool = True,
    ci_level: float = 0.95,
    n_points: int = 50,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    style_mode: str = "viridis",
    figsize: tuple = (10, 6),
    output_path: Optional[str] = None,
):
    """
    Plot marginal effects (predicted values) from a regression model.

    Shows how the predicted outcome changes across values of a focal variable,
    holding other variables at specified values (typically means).

    Parameters
    ----------
    model : RegressionModel or MultilevelModel
        Fitted regression model from sociopathit.analyses.regress.
    variable : str
        Focal variable to vary (x-axis).
    at_values : list or array, optional
        Specific values to evaluate. If None, uses range of observed values.
    other_vars : dict, optional
        Values to hold other variables at. If None, uses means.
        Example: {'age': 30, 'gender': 'F'}
    ci : bool, default True
        Show confidence intervals.
    ci_level : float, default 0.95
        Confidence level for intervals.
    n_points : int, default 50
        Number of points to evaluate along variable range.
    title : str, optional
        Plot title.
    subtitle : str, optional
        Plot subtitle.
    xlabel : str, optional
        X-axis label. Defaults to variable name.
    ylabel : str, optional
        Y-axis label. Defaults to outcome name.
    style_mode : str, default "viridis"
        Color scheme.
    figsize : tuple, default (10, 6)
        Figure size.
    output_path : str, optional
        Save plot to file.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object.

    Examples
    --------
    Simple margin plot:
    >>> from sociopathit.analyses.regress import ols
    >>> model = ols(df, 'income', ['education', 'age', 'gender'])
    >>> margins(model, 'education', title='Income by Education')

    Margin plot at specific values:
    >>> margins(model, 'education', other_vars={'age': 30, 'gender': 'F'})

    Margin plot with specific evaluation points:
    >>> margins(model, 'age', at_values=[20, 30, 40, 50, 60])
    """
    # Extract model information
    try:
        from ..analyses.regress import RegressionModel, MultilevelModel
        is_valid_model = isinstance(model, (RegressionModel, MultilevelModel))
    except:
        is_valid_model = hasattr(model, 'predict') and hasattr(model, 'df')

    if not is_valid_model:
        raise TypeError("model must be a fitted RegressionModel or MultilevelModel")

    # Get data and variable info
    df = model.df
    inputs = getattr(model, 'inputs', []) or getattr(model, 'fixed', [])
    outcome = model.outcome

    if variable not in inputs:
        raise ValueError(f"{variable} not found in model inputs")

    # Determine values to evaluate
    if at_values is None:
        var_min = df[variable].min()
        var_max = df[variable].max()
        at_values = np.linspace(var_min, var_max, n_points)

    # Create prediction data
    pred_data = pd.DataFrame({variable: at_values})

    # Set other variables
    if other_vars is None:
        other_vars = {}

    for var in inputs:
        if var != variable and var not in other_vars:
            # Use mean for continuous, mode for categorical
            if df[var].dtype in [np.float64, np.int64]:
                other_vars[var] = df[var].mean()
            else:
                other_vars[var] = df[var].mode()[0]

    for var, val in other_vars.items():
        pred_data[var] = val

    # Generate predictions
    try:
        predictions = model.predict(pred_data)

        # Get confidence intervals if available
        if ci:
            try:
                # Get prediction standard errors (approximate)
                from scipy import stats
                residual_std = np.sqrt(model.results.scale) if hasattr(model.results, 'scale') else np.std(model.results.resid)
                z_crit = stats.norm.ppf(1 - (1 - ci_level) / 2)
                margin_error = z_crit * residual_std

                ci_lower = predictions - margin_error
                ci_upper = predictions + margin_error
            except:
                ci = False
                warnings.warn("Could not calculate confidence intervals")
    except Exception as e:
        raise RuntimeError(f"Error generating predictions: {e}")

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Get colors
    color_map = COLORS_DICT.get(style_mode, plt.cm.viridis)
    if callable(color_map):
        main_color = color_map(0.6)
    else:
        main_color = color_map[0] if hasattr(color_map, '__getitem__') else 'steelblue'

    # Plot predictions
    ax.plot(at_values, predictions, color=main_color, linewidth=2.5, label='Predicted')

    # Plot confidence interval
    if ci:
        ax.fill_between(at_values, ci_lower, ci_upper, color=main_color, alpha=0.2, label=f'{int(ci_level*100)}% CI')

    # Add actual data points (optional, with transparency)
    if len(df) < 1000:
        # Sample if too many points
        df_plot = df.sample(min(200, len(df)))
        ax.scatter(df_plot[variable], df_plot[outcome], alpha=0.1, color='grey', s=10, zorder=0)

    # Labels
    xlabel = xlabel or variable.replace('_', ' ').title()
    ylabel = ylabel or f"Predicted {outcome.replace('_', ' ').title()}"

    ax.set_xlabel(xlabel, fontsize=12, weight='bold', color='black')
    ax.set_ylabel(ylabel, fontsize=12, weight='bold', color='black')

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
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    # Title
    if title or subtitle:
        apply_titles(fig, title=title, subtitle=subtitle)

    # Layout
    fig.tight_layout(rect=(0, 0, 0.85, 0.9 if subtitle else 0.94))

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')

    return fig


def margins_comparison(
    model,
    variable: str,
    by: str,
    by_values: Optional[List] = None,
    at_values: Optional[Union[List, np.ndarray]] = None,
    other_vars: Optional[dict] = None,
    ci: bool = True,
    ci_level: float = 0.95,
    n_points: int = 50,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    style_mode: str = "viridis",
    figsize: tuple = (10, 6),
    output_path: Optional[str] = None,
):
    """
    Plot marginal effects comparing across groups or levels of another variable.

    Useful for visualizing interaction effects.

    Parameters
    ----------
    model : RegressionModel
        Fitted regression model.
    variable : str
        Focal variable to vary (x-axis).
    by : str
        Variable to compare across (different lines).
    by_values : list, optional
        Specific values of 'by' variable to plot. If None, uses quartiles or unique values.
    at_values : list or array, optional
        Values of focal variable to evaluate.
    other_vars : dict, optional
        Values for other variables.
    ci : bool, default True
        Show confidence intervals.
    ci_level : float, default 0.95
        Confidence level.
    n_points : int, default 50
        Number of evaluation points.
    title : str, optional
        Plot title.
    subtitle : str, optional
        Plot subtitle.
    xlabel : str, optional
        X-axis label.
    ylabel : str, optional
        Y-axis label.
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
    Interaction between education and gender:
    >>> margins_comparison(model, 'education', by='gender')

    Age effect at different income levels:
    >>> margins_comparison(model, 'age', by='income',
    ...                    by_values=[25000, 50000, 75000])
    """
    # Get model info
    df = model.df
    inputs = getattr(model, 'inputs', []) or getattr(model, 'fixed', [])
    outcome = model.outcome

    if variable not in inputs:
        raise ValueError(f"{variable} not found in model inputs")
    if by not in inputs:
        raise ValueError(f"{by} not found in model inputs")

    # Determine values to evaluate
    if at_values is None:
        var_min = df[variable].min()
        var_max = df[variable].max()
        at_values = np.linspace(var_min, var_max, n_points)

    # Determine by values
    if by_values is None:
        if df[by].dtype in [np.float64, np.int64]:
            # Use quartiles for continuous
            by_values = df[by].quantile([0.25, 0.50, 0.75]).values
        else:
            # Use unique values for categorical (max 5)
            by_values = df[by].value_counts().head(5).index.tolist()

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Get colors
    color_map = COLORS_DICT.get(style_mode, plt.cm.viridis)
    if callable(color_map):
        colors = [color_map(i / len(by_values)) for i in range(len(by_values))]
    else:
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(by_values)))

    # Plot each group
    for i, by_val in enumerate(by_values):
        # Create prediction data
        pred_data = pd.DataFrame({variable: at_values})
        pred_data[by] = by_val

        # Set other variables
        if other_vars is None:
            other_vars = {}

        for var in inputs:
            if var not in [variable, by] and var not in other_vars:
                if df[var].dtype in [np.float64, np.int64]:
                    other_vars[var] = df[var].mean()
                else:
                    other_vars[var] = df[var].mode()[0]

        for var, val in other_vars.items():
            if var not in [variable, by]:
                pred_data[var] = val

        # Generate predictions
        predictions = model.predict(pred_data)

        # Plot line
        label = f"{by}={by_val:.2f}" if isinstance(by_val, (int, float)) else f"{by}={by_val}"
        ax.plot(at_values, predictions, color=colors[i], linewidth=2.5, label=label)

        # Confidence interval (optional, makes plot busy)
        if ci and len(by_values) <= 3:
            try:
                from scipy import stats
                residual_std = np.sqrt(model.results.scale)
                z_crit = stats.norm.ppf(1 - (1 - ci_level) / 2)
                margin_error = z_crit * residual_std
                ci_lower = predictions - margin_error
                ci_upper = predictions + margin_error
                ax.fill_between(at_values, ci_lower, ci_upper, color=colors[i], alpha=0.15)
            except:
                pass

    # Labels
    xlabel = xlabel or variable.replace('_', ' ').title()
    ylabel = ylabel or f"Predicted {outcome.replace('_', ' ').title()}"

    ax.set_xlabel(xlabel, fontsize=12, weight='bold', color='black')
    ax.set_ylabel(ylabel, fontsize=12, weight='bold', color='black')

    # Legend
    legend = ax.legend(
        bbox_to_anchor=(1.02, 1.0),
        loc='upper left',
        frameon=True,
        facecolor='white',
        edgecolor='grey',
        fontsize=10,
        title=by.replace('_', ' ').title(),
        title_fontsize=11,
    )
    legend.get_title().set_fontweight('bold')
    legend.get_frame().set_linewidth(1.5)
    legend.get_frame().set_alpha(0.95)

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    # Title
    if title or subtitle:
        apply_titles(fig, title=title, subtitle=subtitle)

    # Layout
    fig.tight_layout(rect=(0, 0, 0.85, 0.9 if subtitle else 0.94))

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')

    return fig
