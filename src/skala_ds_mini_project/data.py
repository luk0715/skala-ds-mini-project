"""MAT loading and cycle-summary normalization, without import-time I/O."""

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

BATCH_FILES = (
    "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
    "2018-02-20_batchdata_updated_struct_errorcorrect.mat",
    "2018-04-12_batchdata_updated_struct_errorcorrect.mat",
)
SUMMARY_FIELDS = {
    "QD": "QDischarge",
    "QC": "QCharge",
    "IR": "IR",
    "Tmax": "Tmax",
    "Tavg": "Tavg",
    "Tmin": "Tmin",
    "chargetime": "chargetime",
}
SUMMARY_COLUMNS = ("cell_id", "cycle", "cycle_life", "charging_policy", *SUMMARY_FIELDS)
MULTI_SUMMARY_COLUMNS = (*SUMMARY_COLUMNS, "batch_id", "local_cell_id")


def load_mat(path, *, only_include=None):
    """Load v7.3/HDF5 or older MAT files, preserving useful file errors.

    ``only_include`` is a sequence of mat73 HDF5 field paths. For legacy MAT
    files scipy can select top-level variables only, not nested fields.
    """
    import h5py
    import mat73
    from scipy.io import loadmat

    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(path)
    if not path.is_file():
        raise IsADirectoryError(path)
    if isinstance(only_include, str):
        only_include = [only_include]
    try:
        if h5py.is_hdf5(path):
            return mat73.loadmat(str(path), only_include=only_include, verbose=False)
        variables = (
            None
            if only_include is None
            else list(dict.fromkeys(field.strip("/").split("/")[0] for field in only_include))
        )
        return loadmat(path, simplify_cells=True, variable_names=variables)
    except (PermissionError, FileNotFoundError):
        raise
    except Exception as exc:
        # Do not reinterpret a corrupt HDF5 file as a legacy file.
        raise ValueError(f"Could not load MAT file {path}: {exc}") from exc


def load_feature_batch(path, *, cycle_numbers=(10, 100)):
    """Load summaries plus selected Qdlin cycles from this dataset's MAT layout.

    Read HDF5 references directly: mat73's nested selection still traverses
    referenced structs and resolves expensive HDF5 object names. Unselected
    cycle positions are None, preserving 1-based indexing. This is a feature
    input, not a full raw-data replacement. Legacy MAT files use load_mat.
    """
    import h5py

    cycle_numbers = tuple(cycle_numbers)
    if not cycle_numbers or any(
        not isinstance(n, (int, np.integer)) or n < 1 for n in cycle_numbers
    ):
        raise ValueError("cycle_numbers must contain positive 1-based integers")
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(path)
    if not path.is_file():
        raise IsADirectoryError(path)
    if not h5py.is_hdf5(path):
        mat = load_mat(path)
        if "batch" not in mat:
            raise ValueError(f"MAT file {path} has no 'batch' variable")
        return normalize_batch(mat["batch"])

    def numeric(dataset):
        if dataset.attrs.get("MATLAB_empty", False):
            return None
        return np.asarray(dataset[()], dtype=float).reshape(-1)

    with h5py.File(path, "r") as handle:
        if "batch" not in handle or not isinstance(handle["batch"], h5py.Group):
            raise ValueError(f"{path}: expected a MATLAB 'batch' struct group")
        raw = handle["batch"]
        if "summary" not in raw:
            raise ValueError(f"{path}: batch is missing summary")
        refs = {}
        for key in ("summary", "cycle_life", "policy_readable", "policy", "Vdlin", "cycles"):
            if key in raw:
                values = raw[key][()].reshape(-1)
                if h5py.check_dtype(ref=values.dtype) is None:
                    raise ValueError(f"{path}: expected MATLAB references in batch/{key}")
                refs[key] = values
        count = len(refs["summary"])
        if any(len(values) != count for values in refs.values()):
            raise ValueError(f"{path}: batch field length mismatch")
        batch = []
        for i in range(count):
            if not refs["summary"][i]:
                raise ValueError(f"{path}: cell {i} has no summary reference")
            summary = handle[refs["summary"][i]]
            cell = {
                "summary": {field: numeric(summary[field]) for field in SUMMARY_FIELDS.values()}
            }
            for key in ("cycle_life", "Vdlin"):
                if key in refs and refs[key][i]:
                    cell[key] = numeric(handle[refs[key][i]])
            for key in ("policy_readable", "policy"):
                if key in refs and refs[key][i]:
                    chars = handle[refs[key][i]][()].reshape(-1)
                    cell[key] = "".join(chr(int(char)) for char in chars)
            selected = [None] * max(cycle_numbers)
            if "cycles" in refs and refs["cycles"][i]:
                cycles = handle[refs["cycles"][i]]
                if "Qdlin" in cycles:
                    qrefs = cycles["Qdlin"][()].reshape(-1)
                    if h5py.check_dtype(ref=qrefs.dtype) is None:
                        raise ValueError(f"{path}: expected MATLAB references in cycles/Qdlin")
                    for number in cycle_numbers:
                        if number <= len(qrefs) and qrefs[number - 1]:
                            selected[number - 1] = numeric(handle[qrefs[number - 1]])
            cell["cycles"] = {"Qdlin": selected}
            batch.append(cell)
    return batch


def _record(value):
    while isinstance(value, np.ndarray) and value.size == 1:
        value = value.reshape(-1)[0]
    if isinstance(value, np.void) and value.dtype.names:
        return {name: value[name] for name in value.dtype.names}
    if not isinstance(value, Mapping):
        raise TypeError(f"Expected a MATLAB struct/dict, got {type(value).__name__}")
    return dict(value)


def normalize_batch(raw_batch):
    """Return fresh record dictionaries from columnar, list or NumPy structs.

    Nested arrays are shared (not copied); callers must treat them as read-only.
    Singleton scipy cells are accepted. Unequal column lengths raise rather
    than silently dropping cells or assigning incorrect identities.
    """
    if isinstance(raw_batch, Mapping):
        if not raw_batch:
            return []
        # scipy simplify_cells represents a single battery as one mapping.
        if isinstance(raw_batch.get("summary"), Mapping):
            return [dict(raw_batch)]
        columns = {}
        for key, values in raw_batch.items():
            if isinstance(values, (str, bytes, Mapping)) or not hasattr(values, "__len__"):
                raise ValueError(f"Batch field {key!r} must be a sequence of equal length")
            if isinstance(values, np.ndarray) and values.ndim == 0:
                raise ValueError(f"Batch field {key!r} must be a sequence of equal length")
            if isinstance(values, np.ndarray) and values.dtype.kind in "OV":
                values = values.reshape(-1)
            columns[key] = values
        lengths = {key: len(values) for key, values in columns.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"Batch field length mismatch: {lengths}")
        return [
            {key: values[i] for key, values in columns.items()}
            for i in range(next(iter(lengths.values())))
        ]
    if isinstance(raw_batch, np.void):
        return [_record(raw_batch)]
    if isinstance(raw_batch, np.ndarray):
        raw_batch = raw_batch.reshape(-1)
    return [_record(cell) for cell in raw_batch]


def to_list_of_dicts(raw):
    """Normalize a struct collection, including columnar cycle records."""
    return normalize_batch(raw)


def cell_metadata(cell):
    """Return a scalar target (NaN if unknown) and readable policy."""
    life = np.asarray(cell.get("cycle_life", np.nan), dtype=float).reshape(-1)
    if life.size != 1:
        raise ValueError("cycle_life must be a scalar")
    life = float(life[0])
    life = int(life) if np.isfinite(life) else np.nan
    policy = "unknown"
    for field in ("policy_readable", "policy"):
        value = cell.get(field)
        if value is None:
            continue
        value = np.asarray(value).squeeze()
        if value.ndim != 0:
            raise ValueError(f"{field} must be a scalar string")
        text = str(value.item())
        if text and text not in ("None", "nan"):
            policy = text
            break
    return life, policy


def extract_summary(batch, *, batch_id=None, cell_id_offset=0):
    """One row per recorded cycle; cycle numbering is 1-based.

    ``cell_id_offset`` reserves an identity even for a cell with no summaries.
    Supply ``batch_id`` to also retain the zero-based ``local_cell_id``.
    """
    frames = []
    for local_id, cell in enumerate(normalize_batch(batch)):
        summary = _record(cell["summary"])
        arrays = {
            name: np.asarray(summary[field], dtype=float).reshape(-1)
            for name, field in SUMMARY_FIELDS.items()
        }
        lengths = {name: len(values) for name, values in arrays.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"Cell {local_id} summary field length mismatch: {lengths}")
        life, policy = cell_metadata(cell)
        frame = pd.DataFrame(arrays)
        frame["cell_id"] = cell_id_offset + local_id
        frame["cycle"] = np.arange(1, len(frame) + 1)
        frame["cycle_life"] = life
        frame["charging_policy"] = policy
        if batch_id is not None:
            frame["batch_id"] = batch_id
            frame["local_cell_id"] = local_id
        frames.append(frame)
    columns = SUMMARY_COLUMNS if batch_id is None else MULTI_SUMMARY_COLUMNS
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True).reindex(columns=columns)


def extract_summaries(batches):
    """Combine batches in supplied order with stable, globally unique cell IDs.

    Batch IDs start at 1; local and global cell IDs start at 0. Pass batches in
    ``BATCH_FILES`` order to reproduce the notebook's mapping.
    """
    frames = []
    offset = 0
    for batch_id, raw in enumerate(batches, start=1):
        batch = normalize_batch(raw)
        frames.append(extract_summary(batch, batch_id=batch_id, cell_id_offset=offset))
        offset += len(batch)
    if not frames:
        return pd.DataFrame(columns=MULTI_SUMMARY_COLUMNS)
    return pd.concat(frames, ignore_index=True).reindex(columns=MULTI_SUMMARY_COLUMNS)
