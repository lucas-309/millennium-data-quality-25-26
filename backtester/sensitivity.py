"""Parameter sensitivity analysis for hyperparameter robustness checks.

Grid sweeps over parameter combinations, returning a metric surface. Useful
for spotting cliffs in the parameter space (overfit regions).
"""
from __future__ import annotations

import itertools
from typing import Callable, Dict, List

import numpy as np
import pandas as pd


def parameter_sweep(
    param_grid: Dict[str, List],
    metric_fn: Callable[[Dict], float],
) -> pd.DataFrame:
    """Evaluate metric_fn over the Cartesian product of param_grid.

    Returns a long DataFrame with one row per param combo and columns for
    each parameter plus the resulting metric.
    """
    keys = list(param_grid.keys())
    combos = list(itertools.product(*[param_grid[k] for k in keys]))

    rows = []
    for combo in combos:
        params = dict(zip(keys, combo))
        metric = metric_fn(params)
        row = {**params, "metric": metric}
        rows.append(row)

    return pd.DataFrame(rows)


def sensitivity_heatmap_data(
    df: pd.DataFrame,
    row_param: str,
    col_param: str,
    metric_col: str = "metric",
) -> pd.DataFrame:
    """Pivot sweep results into a (row_param x col_param) metric grid."""
    return df.pivot_table(
        index=row_param,
        columns=col_param,
        values=metric_col,
        aggfunc="mean",
    )
