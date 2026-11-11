"""Concept shifts: ways churn behaviour can change while the inputs look normal.

Each shift changes only the labels. A shift takes the raw feature frame, the
current 0/1 labels and a seed, and returns the new labels.
"""
import numpy as np


def _flip(labels, mask, from_value, prob, rng):
    """Flip labels equal to from_value, for rows in mask, with probability prob."""
    hit = mask & (labels == from_value) & (rng.random(len(labels)) < prob)
    out = labels.copy()
    out[hit] = 1 - from_value
    return out


def _dsl_vs_fiber(df, y, seed, up, down):
    """DSL customers start churning (up); fiber customers are retained (down)."""
    rng = np.random.default_rng(seed)
    y = _flip(y, (df["InternetService"] == "DSL").to_numpy(), 0, up, rng)
    return _flip(y, (df["InternetService"] == "Fiber optic").to_numpy(), 1, down, rng)


def _contract_flip(df, y, seed):
    """Month-to-month churners are retained; contract customers start churning."""
    rng = np.random.default_rng(seed)
    month = (df["Contract"] == "Month-to-month").to_numpy()
    y = _flip(y, month, 1, 0.5, rng)
    return _flip(y, ~month, 0, 0.5, rng)


SHIFTS = {
    "mild_dsl_fiber": lambda df, y, seed: _dsl_vs_fiber(df, y, seed, 0.25, 0.25),
    "dsl_fiber": lambda df, y, seed: _dsl_vs_fiber(df, y, seed, 0.4, 0.5),
    "contract_flip": _contract_flip,
}


def apply_shift(name: str, df, labels, seed: int):
    """Return the shifted 0/1 labels for the rows of df."""
    return SHIFTS[name](df, np.asarray(labels).astype(int), seed)
