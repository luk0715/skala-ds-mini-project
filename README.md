# SKALA battery feature extraction

Reusable computations from `notebooks/main.ipynb`, ready for the modeling step.
This refactor does **not** implement model training. EDA narrative and plots stay
in the notebook; loading, preprocessing and feature formulas live in Python.

## Setup

From the project root:

```sh
uv sync
```

Keep the three original MAT files in `data/`, or pass their directory explicitly.
The ordered filenames are defined in `skala_ds_mini_project.data.BATCH_FILES`;
loading never guesses the batch order from a directory listing. Use this same
Python environment as the Jupyter kernel when running the notebook.

## Build modeling inputs directly

Run with `uv run python`, or put this code in your modeling script:

```python
from pathlib import Path

from skala_ds_mini_project.features import (
    FEATURE_COLUMNS,
    build_feature_table,
    load_feature_inputs,
)

summary, dq_features, dq_curves, voltage = load_feature_inputs(Path("data"))
features = build_feature_table(summary, dq_features, labeled_only=True)

# Explicit predictor allowlist: never automatically include every numeric column.
X = features.loc[:, list(FEATURE_COLUMNS)]
y = features["log_cycle_life"]
identity = features[["batch_id", "cell_id"]]

# Optional export; creating the file is the caller's decision.
# features.to_csv("early_features.csv", index=False)
```

For the supplied data, the direct smoke run produced **129 rows × 24 columns**:
20 predictors, `batch_id`, `cell_id`, `cycle_life`, and `log_cycle_life`.
`summary` and `dq_features` also retain `local_cell_id` so rows can be traced back
to the raw batch. Batch IDs are 1-based; local/global cell IDs are 0-based.
Global cell IDs follow file order and then cell order, including reserved IDs
for cells without summary rows.

For unknown targets / inference:

```python
inference_features = build_feature_table(summary, dq_features, labeled_only=False)
X_inference = inference_features.loc[:, list(FEATURE_COLUMNS)]
```

Labeled mode requires a positive known target and an available ΔQ record,
matching the notebook's inner join. Inference mode retains missing targets and
missing ΔQ predictors as NaNs; it does not fill or invent observations. Cells
without any retained summary rows are omitted. For the supplied data, inference
mode produced 139 rows, including 10 unknown targets. Imputation, feature
selection and train/validation splits are intentionally left to modeling code.
Fit those decisions on training data only. A nominal capacity estimated on
training/reference data can be reused via `build_feature_table(..., nominal=...)`
instead of re-estimating it on each input set.

## Lower-level functions

- `data.py`: `load_mat`, `load_feature_batch`, `normalize_batch`,
  `to_list_of_dicts`, `extract_summary`, `extract_summaries`, and `BATCH_FILES`.
- `preprocessing.py`: `estimate_nominal_capacity` and `filter_capacity`.
- `features.py`: `parse_policy`, `avg_c_rate`, `get_qdlin`, `dq_statistics`,
  `extract_dq_features`, `early_cell_features`, `fade_slope`,
  `build_feature_table`, and `load_feature_inputs`.
- `analysis.py`: `find_knee_point`, a **full-lifetime diagnostic only**. A knee
  inferred from later cycles would leak future information into early-life models.

For already loaded batches, use `extract_summaries(batches)` and
`extract_dq_features(batches)` with the **same ordered batches**, then pass the
resulting frames to `build_feature_table`. `extract_dq_features` returns a frame
and a dictionary of ΔQ curves keyed by `(batch_id, local_cell_id)`.

## Preserved formulas and explicit edge cases

- Nominal QD: median of per-cell medians over cycles 1–5. Modeling filtering:
  inclusive `[0.33 × nominal, 1.33 × nominal]`.
- Summary features: cycles **2–100**, excluding cycle 1; initial medians use
  cycles 2–10, ending medians use cycles 91–100. `QD_std` uses pandas sample
  standard deviation (`ddof=1`). Temperature/charge-time features use medians.
- Early fade: linear QD slope over cycles **10–100**, with at least 10 finite
  observations. Too-short/constant-cycle inputs return NaN.
- ΔQ: **Q100 − Q10**, with 1-based cycle indexing, population variance
  (`ddof=0`), and scipy's default skewness/Fisher kurtosis. `dQ_2V` is the last
  value on the dataset's descending voltage grid. Unequal curve lengths raise
  an error rather than broadcasting; absent curves are skipped.
- Policy parsing accepts `5.6C(26%)-4.5C` and the `-newstructure` suffix.
  Average C-rate is `0.8 / (SOC/100/C1 + (80-SOC)/100/C2)`.
- `log_dQ_*` is `log10(abs(value))`; target log is `log10(cycle_life)`.
  Zero, nonfinite or invalid-log inputs become NaN rather than infinity.
  Unparseable policies and constant-curve skew/kurtosis also become NaN.
- Empty inputs retain column schemas; unequal summary/struct field lengths and
  duplicate feature join keys raise errors. Functions do not mutate inputs;
  normalization returns fresh dictionaries but shares nested arrays read-only.

## Memory and notebook behavior

`load_feature_inputs` processes files one at a time. For these HDF5 MAT files,
`load_feature_batch` directly reads summary/metadata references and only Qdlin
cycles 10 and 100: it never loads raw I/V/t waveforms. Other cycle positions are
`None`, so these compact cells are **not** substitutes for a full raw batch.
The selected-cycle loader is specific to this dataset's MATLAB reference layout.
Legacy MAT files use scipy and must load one full batch at a time.

`load_mat` remains a general full-file loader, with optional mat73 field
selection. Avoid using nested `only_include` paths as a memory guarantee:
mat73 follows referenced structs eagerly and resolving HDF5 object names in
these files can be very slow. Missing files and parse errors remain visible.

The notebook imports shared functions and calls the same feature pipeline.
All stored outputs/execution counts were cleared because the computation code
changed; rerun from the top to regenerate plots. The notebook's first structure
inspection now explicitly shows the selected Qdlin input rather than raw waveforms.
Modules do no import-time data loading, plotting, or warning suppression.

## Lightweight verification

No test suite or test dependencies were added. For an import/compile check:

```sh
uv run python -m compileall -q src/skala_ds_mini_project
uv run python -c "from skala_ds_mini_project.features import FEATURE_COLUMNS; print(len(FEATURE_COLUMNS))"
```

Run the direct input-building example above as a smoke check with local data.
The all-batch smoke run returned 116,722 summary rows, a 129×24 labeled table,
and a 139×24 inference table. Notebook code cells were syntax-checked; the full
EDA plotting sequence and expensive full-lifetime knee analysis were not rerun.
