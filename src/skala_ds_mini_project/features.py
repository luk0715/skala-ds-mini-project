"""Reusable early-cycle features from main.ipynb; no model fitting or plotting."""

import re
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

from .data import (
    BATCH_FILES,
    MULTI_SUMMARY_COLUMNS,
    cell_metadata,
    extract_summary,
    load_feature_batch,
    normalize_batch,
)
from .preprocessing import filter_capacity

EARLY_FEATURE_COLUMNS = (
    "QD_init",
    "QD_max_minus_init",
    "QD_diff_100_init",
    "QD_std",
    "IR_mean",
    "IR_diff_100_init",
    "Tavg_med",
    "Tmax_med",
    "Tmin_med",
    "chargetime_med",
)
DQ_STAT_COLUMNS = ("dQ_min", "dQ_mean", "dQ_var", "dQ_skew", "dQ_kurt", "dQ_2V")
DQ_LOG_COLUMNS = ("log_dQ_min", "log_dQ_mean", "log_dQ_var")
DQ_PREDICTORS = (*DQ_LOG_COLUMNS, "dQ_skew", "dQ_kurt")
CHARGE_FEATURE_COLUMNS = ("early_fade", "first_c", "switch_soc", "second_c", "avg_c")
# Deliberately explicit: adding diagnostic/target columns must not change X.
FEATURE_COLUMNS = (*EARLY_FEATURE_COLUMNS, *DQ_PREDICTORS, *CHARGE_FEATURE_COLUMNS)
FEATURE_TABLE_COLUMNS = (
    "batch_id",
    "cell_id",
    *EARLY_FEATURE_COLUMNS,
    "cycle_life",
    "log_cycle_life",
    *DQ_PREDICTORS,
    *CHARGE_FEATURE_COLUMNS,
)
DQ_COLUMNS = (
    "batch_id",
    "local_cell_id",
    "cell_id",
    "cycle_life",
    *DQ_STAT_COLUMNS,
    *DQ_LOG_COLUMNS,
    "log_cycle_life",
)
_POLICY = re.compile(r"(\d+(?:\.\d+)?)C\((\d+(?:\.\d+)?)%\)-(\d+(?:\.\d+)?)C(?:-newstructure)?")


def parse_policy(policy):
    """Return (first C-rate, switch SOC %, second C-rate), or three NaNs.

    Accept the dataset's optional ``-newstructure`` suffix; reject other text.
    """
    match = _POLICY.fullmatch(str(policy).strip())
    return tuple(map(float, match.groups())) if match else (np.nan, np.nan, np.nan)


def avg_c_rate(first_c, switch_soc, second_c):
    """Time-weighted average C-rate over 0–80% SOC; invalid inputs yield NaN."""
    first, soc, second = np.broadcast_arrays(
        np.asarray(first_c, dtype=float),
        np.asarray(switch_soc, dtype=float),
        np.asarray(second_c, dtype=float),
    )
    valid = (
        np.isfinite(first)
        & np.isfinite(soc)
        & np.isfinite(second)
        & (first > 0)
        & (second > 0)
        & (soc >= 0)
        & (soc <= 80)
    )
    result = np.full(first.shape, np.nan)
    result[valid] = 0.8 / (
        soc[valid] / 100 / first[valid] + (80 - soc[valid]) / 100 / second[valid]
    )
    return float(result) if result.ndim == 0 else result


def safe_log10(values, *, absolute=False):
    """log10 for finite positive values; zero/invalid values become NaN, not inf."""
    values = np.asarray(values, dtype=float)
    if absolute:
        values = np.abs(values)
    result = np.full(values.shape, np.nan)
    valid = np.isfinite(values) & (values > 0)
    result[valid] = np.log10(values[valid])
    return float(result) if result.ndim == 0 else result


def get_qdlin(cell, cycle_no):
    """Qdlin for a 1-based recorded cycle; None if absent/empty.

    Supports columnar mat73 cycles and scipy list-of-records cycles. It does
    not silently clip indices or interpolate unequal voltage grids.
    """
    if not isinstance(cycle_no, (int, np.integer)) or cycle_no < 1:
        raise ValueError("cycle_no must be a positive 1-based integer")
    cycles = cell.get("cycles")
    if cycles is None:
        return None
    if isinstance(cycles, Mapping):
        curves = cycles.get("Qdlin")
        if curves is None:
            return None
        # scipy may simplify a singleton cycle into a numeric vector.
        if isinstance(curves, np.ndarray) and curves.dtype.kind != "O" and curves.ndim == 1:
            curve = curves if cycle_no == 1 else None
        else:
            curve = curves[cycle_no - 1] if len(curves) >= cycle_no else None
    else:
        records = normalize_batch(cycles)
        curve = records[cycle_no - 1].get("Qdlin") if len(records) >= cycle_no else None
    if curve is None:
        return None
    values = np.asarray(curve, dtype=float).reshape(-1)
    return values if values.size else None


def dq_statistics(dq):
    """Notebook ΔQ moments (population variance, scipy default skew/kurtosis)."""
    dq = np.asarray(dq, dtype=float).reshape(-1)
    if not dq.size:
        return dict.fromkeys((*DQ_STAT_COLUMNS, *DQ_LOG_COLUMNS), np.nan)
    variance = float(dq.var())
    moments_valid = np.isfinite(dq).all() and variance > 0
    stats = {
        "dQ_min": float(dq.min()),
        "dQ_mean": float(dq.mean()),
        "dQ_var": variance,
        "dQ_skew": float(skew(dq)) if moments_valid else np.nan,
        "dQ_kurt": float(kurtosis(dq)) if moments_valid else np.nan,
        "dQ_2V": float(dq[-1]),  # Last point on the dataset's descending voltage axis.
    }
    for name in ("dQ_min", "dQ_mean", "dQ_var"):
        stats[f"log_{name}"] = safe_log10(stats[name], absolute=True)
    return stats


def extract_dq_features(
    batches,
    *,
    cycle_early=10,
    cycle_late=100,
    labeled_only=False,
    batch_id_start=1,
    cell_id_offset=0,
):
    """Return (statistics frame, {(batch_id, local_cell_id): ΔQ curve}).

    Missing curves are skipped; different curve lengths raise ValueError.
    Missing labels are retained by default. Supply batches in the same order
    as ``extract_summaries`` so global identities agree.
    """
    if not all(isinstance(n, (int, np.integer)) for n in (cycle_early, cycle_late)):
        raise ValueError("Cycle numbers must be integers")
    if not 1 <= cycle_early < cycle_late:
        raise ValueError("Require 1 <= cycle_early < cycle_late")
    records, curves = [], {}
    offset = cell_id_offset
    for batch_id, raw in enumerate(batches, start=batch_id_start):
        batch = normalize_batch(raw)
        for local_id, cell in enumerate(batch):
            life, _ = cell_metadata(cell)
            if labeled_only and not np.isfinite(life):
                continue
            early = get_qdlin(cell, cycle_early)
            late = get_qdlin(cell, cycle_late)
            if early is None or late is None:
                continue
            if early.shape != late.shape:
                raise ValueError(
                    f"Batch {batch_id} cell {local_id}: Qdlin length mismatch "
                    f"({early.size} vs {late.size})"
                )
            dq = late - early
            curves[batch_id, local_id] = dq
            records.append(
                {
                    "batch_id": batch_id,
                    "local_cell_id": local_id,
                    "cell_id": offset + local_id,
                    "cycle_life": life,
                    **dq_statistics(dq),
                    "log_cycle_life": safe_log10(life),
                }
            )
        offset += len(batch)
    return pd.DataFrame(records, columns=DQ_COLUMNS), curves


def early_cell_features(group):
    """Notebook summary features over cycles 2–100 (sample std, ddof=1)."""
    group = group.sort_values("cycle")
    early = group.loc[group["cycle"].between(2, 100)]
    qd_init = early.loc[early["cycle"] <= 10, "QD"].median()
    qd_end = early.loc[early["cycle"] >= 91, "QD"].median()
    ir_init = early.loc[early["cycle"] <= 10, "IR"].median()
    ir_end = early.loc[early["cycle"] >= 91, "IR"].median()
    return {
        "QD_init": qd_init,
        "QD_max_minus_init": early["QD"].max() - qd_init,
        "QD_diff_100_init": qd_end - qd_init,
        "QD_std": early["QD"].std(),
        "IR_mean": early["IR"].mean(),
        "IR_diff_100_init": ir_end - ir_init,
        "Tavg_med": early["Tavg"].median(),
        "Tmax_med": early["Tmax"].median(),
        "Tmin_med": early["Tmin"].median(),
        "chargetime_med": early["chargetime"].median(),
    }


def fade_slope(group, start=10, end=100):
    """QD linear slope on inclusive cycles 10–100, requiring 10 finite rows."""
    if start > end:
        raise ValueError("start must not exceed end")
    sub = group.loc[group["cycle"].between(start, end), ["cycle", "QD"]]
    sub = sub.loc[np.isfinite(sub).all(axis=1)].sort_values("cycle")
    if len(sub) < 10 or sub["cycle"].nunique() < 2:
        return np.nan
    return float(np.polyfit(sub["cycle"], sub["QD"], 1)[0])


def build_feature_table(summary, dq_features, *, labeled_only=True, nominal=None):
    """Build the notebook's 24-column table (20 predictors, two IDs, two targets).

    Capacity filtering uses [0.33, 1.33] × nominal, estimated from cycles 1–5
    unless supplied. Labeled mode requires a known positive target and a ΔQ
    record, matching the notebook joins. Inference mode retains unknown labels
    and missing ΔQ as NaNs. Cells with no retained summary rows are omitted.
    All inputs remain unmodified. No knee/full-lifetime metric is a predictor.
    """
    clean = filter_capacity(summary, nominal=nominal)
    records = []
    for (batch_id, cell_id), group in clean.groupby(["batch_id", "cell_id"], sort=True):
        metadata = group[["cycle_life", "charging_policy"]].drop_duplicates()
        if len(metadata) != 1:
            raise ValueError(f"Conflicting metadata for batch {batch_id}, cell {cell_id}")
        life, policy = metadata.iloc[0]
        first, soc, second = parse_policy(policy)
        records.append(
            {
                "batch_id": batch_id,
                "cell_id": cell_id,
                **early_cell_features(group),
                "cycle_life": life,
                "log_cycle_life": safe_log10(life),
                "early_fade": fade_slope(group),
                "first_c": first,
                "switch_soc": soc,
                "second_c": second,
                "avg_c": avg_c_rate(first, soc, second),
            }
        )
    if not records:
        return pd.DataFrame(columns=FEATURE_TABLE_COLUMNS, dtype=float)
    table = pd.DataFrame(records)
    table = table.merge(
        dq_features[["batch_id", "cell_id", *DQ_PREDICTORS]],
        on=["batch_id", "cell_id"],
        how="inner" if labeled_only else "left",
        validate="one_to_one",
    )
    if labeled_only:
        table = table.loc[table["log_cycle_life"].notna()]
    return table.reindex(columns=FEATURE_TABLE_COLUMNS).reset_index(drop=True)


def load_feature_inputs(data_dir, batch_files=BATCH_FILES, *, cycle_early=10, cycle_late=100):
    """Load one batch at a time; return (summary, ΔQ table, curves, voltage axis).

    Only summaries and the two requested Qdlin cycles are read for HDF5 files;
    raw waveforms are never decoded. Returned objects contain summaries and ΔQ
    only. Files/order are explicit; legacy MAT files load one full batch at a time.
    """
    summaries, features, curves = [], [], {}
    voltage = np.array([], dtype=float)
    offset = 0
    for batch_id, filename in enumerate(batch_files, start=1):
        batch = load_feature_batch(
            Path(data_dir) / filename, cycle_numbers=(cycle_early, cycle_late)
        )
        summaries.append(extract_summary(batch, batch_id=batch_id, cell_id_offset=offset))
        frame, batch_curves = extract_dq_features(
            [batch],
            cycle_early=cycle_early,
            cycle_late=cycle_late,
            batch_id_start=batch_id,
            cell_id_offset=offset,
        )
        features.append(frame)
        curves.update(batch_curves)
        if not voltage.size and batch:
            voltage = np.asarray(batch[0].get("Vdlin", []), dtype=float).reshape(-1).copy()
        offset += len(batch)
        del batch
    summary = (
        pd.concat(summaries, ignore_index=True)
        if summaries
        else pd.DataFrame(columns=MULTI_SUMMARY_COLUMNS)
    )
    dq = pd.concat(features, ignore_index=True) if features else pd.DataFrame(columns=DQ_COLUMNS)
    return summary, dq, curves, voltage
