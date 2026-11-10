"""Population Stability Index (PSI) helpers.

Shared by the reference-building pipeline stage and the API, so training-time
and serving-time distributions are binned in exactly the same way.
"""
import numpy as np
import pandas as pd

EPS = 1e-4  # floor for proportions, avoids log(0) when a bin is empty


def _bin_proportions(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Share of values falling in each bin defined by the cut points."""
    idx = np.searchsorted(edges, values, side="right")
    counts = np.bincount(idx, minlength=len(edges) + 1)
    return counts / max(counts.sum(), 1)


def numeric_reference(values, n_bins: int) -> dict:
    """Quantile bins from the training data and the share in each bin."""
    values = np.asarray(values, dtype=float)
    cuts = np.quantile(values, np.linspace(0, 1, n_bins + 1)[1:-1])
    edges = np.unique(cuts)
    return {"type": "numeric", "edges": edges.tolist(),
            "proportions": _bin_proportions(values, edges).tolist()}


def categorical_reference(values) -> dict:
    """Share of each category in the training data."""
    shares = pd.Series(values).astype(str).value_counts(normalize=True)
    return {"type": "categorical", "proportions": shares.to_dict()}


def psi_from_proportions(ref, cur) -> float:
    ref = np.clip(np.asarray(ref, dtype=float), EPS, None)
    cur = np.clip(np.asarray(cur, dtype=float), EPS, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def feature_psi(spec: dict, values) -> float:
    """PSI of a window of live values against one feature's reference."""
    if len(values) == 0:
        return 0.0
    if spec["type"] == "numeric":
        cur = _bin_proportions(np.asarray(values, dtype=float),
                               np.asarray(spec["edges"]))
        return psi_from_proportions(spec["proportions"], cur)
    cats = list(spec["proportions"])
    series = pd.Series(values).astype(str)
    counts = np.array([(series == c).sum() for c in cats], dtype=float)
    other = len(series) - counts.sum()  # values never seen in training
    cur = np.append(counts, other) / len(series)
    ref = np.append(list(spec["proportions"].values()), 0.0)
    return psi_from_proportions(ref, cur)
