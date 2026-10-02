"""Capacity filtering shared by the notebook and modeling feature extraction."""

import numpy as np


def estimate_nominal_capacity(summary):
    """Median of per-cell median QD over cycles 1–5 (the notebook rule)."""
    keys = ["batch_id", "cell_id"] if "batch_id" in summary else ["cell_id"]
    return float(summary.loc[summary["cycle"] <= 5].groupby(keys)["QD"].median().median())


def filter_capacity(summary, *, lower=0.33, upper=1.33, nominal=None):
    """Return a copy with QD inside inclusive nominal-relative bounds.

    No target labels or future knee locations are used. ``nominal`` can be
    supplied from training/reference data rather than estimated on a new set.
    An empty or entirely missing nominal estimate returns an empty frame.
    """
    if not (np.isfinite(lower) and np.isfinite(upper) and 0 <= lower <= upper):
        raise ValueError("Require finite bounds 0 <= lower <= upper")
    if nominal is None:
        nominal = estimate_nominal_capacity(summary)
    if not np.isfinite(nominal):
        return summary.iloc[:0].copy()
    return summary.loc[summary["QD"].between(nominal * lower, nominal * upper)].copy()
