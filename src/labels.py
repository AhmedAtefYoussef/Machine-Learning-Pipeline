"""High-demand labels for Phase 5: an hour is positive when `cnt` exceeds a quantile of comparable hours.

The quantile is computed per cell of `group_by` (for example year x day type x hour) on the TRAINING frame only,
then looked up for any other frame, so validation and test rows never influence a threshold.

    label_i = 1  if  cnt_i > q_cell(i)       (q_cell = quantile of train cnt in the cell of row i)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def fit_thresholds(train_df: pd.DataFrame, group_by: list[str], quantile: float) -> pd.DataFrame:
    """Table [group_by..., threshold, n]: the `quantile` of train cnt (linear interpolation) and the size of each cell.

    An empty `group_by` gives one global row (the single-threshold rule of the Expectation).
    """
    if not group_by:
        return pd.DataFrame({"threshold": [float(train_df["cnt"].quantile(quantile))], "n": [len(train_df)]})
    cells = train_df.groupby(list(group_by))["cnt"]
    table = cells.quantile(quantile).rename("threshold").to_frame()
    table["n"] = cells.size()
    return table.reset_index()


def apply_thresholds(df: pd.DataFrame, table: pd.DataFrame, group_by: list[str]) -> np.ndarray:
    """0/1 labels of `df` (row order kept): 1 when cnt is above the threshold of the row's cell.

    A row whose cell is not in the table is an error: a silent default would hide a leak or a bug.
    """
    if not group_by:
        threshold = np.full(len(df), float(table["threshold"].iloc[0]))
    else:
        keys = df[list(group_by)].reset_index(drop=True)
        joined = keys.merge(table[list(group_by) + ["threshold"]], on=list(group_by), how="left")
        threshold = joined["threshold"].to_numpy(dtype=np.float64)
        if np.isnan(threshold).any():
            raise ValueError(f"{int(np.isnan(threshold).sum())} rows belong to cells missing from the threshold table")
    return (df["cnt"].to_numpy(dtype=np.float64) > threshold).astype(int)
