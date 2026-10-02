"""Full-lifetime diagnostic analysis; never use these results as predictors."""

import numpy as np


def find_knee_point(group, min_segment=50):
    """Notebook two-line minimum-SSE knee, or NaN if no valid split exists.

    This uses the entire supplied lifetime and leaks future information if
    included in an early-life prediction feature matrix.
    """
    if not isinstance(min_segment, (int, np.integer)) or min_segment < 2:
        raise ValueError("min_segment must be an integer >= 2")
    group = group.sort_values("cycle").dropna(subset=["cycle", "QD"])
    x = group["cycle"].to_numpy(dtype=float)
    y = group["QD"].to_numpy(dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    best_knee = np.nan
    best_error = np.inf
    # Keep the notebook's exclusive upper bound (no split at len - min_segment).
    for i in range(min_segment, len(x) - min_segment):
        x1, x2 = x[:i], x[i:]
        if np.ptp(x1) == 0 or np.ptp(x2) == 0:
            continue
        coef1 = np.polyfit(x1, y[:i], 1)
        coef2 = np.polyfit(x2, y[i:], 1)
        error = np.sum((y[:i] - np.polyval(coef1, x1)) ** 2)
        error += np.sum((y[i:] - np.polyval(coef2, x2)) ** 2)
        if error < best_error:
            best_error, best_knee = error, x[i]
    return best_knee
