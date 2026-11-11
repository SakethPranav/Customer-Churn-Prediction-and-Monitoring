"""Prediction log (SQLite): every served prediction, so true outcomes can be attached later.

Churn is only known weeks after the prediction, so each prediction gets an ID
and the outcome arrives later through the feedback endpoint.
"""
import json
import os
import sqlite3
import threading
import time
import uuid
from contextlib import closing

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    model_version TEXT,
    probability REAL NOT NULL,
    features TEXT NOT NULL,
    label INTEGER,
    labelled_at REAL
)"""


class PredictionLog:
    """Stores predictions and the outcomes reported for them later."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._lock = threading.Lock()
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with closing(self._connect()) as conn, conn:
            conn.execute(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def add(self, rows: list[dict], probs, model_version) -> list[str]:
        """Store served predictions and return one new ID per row."""
        now = time.time()
        ids = [uuid.uuid4().hex for _ in rows]
        data = [(i, now, str(model_version), float(p), json.dumps(r))
                for i, r, p in zip(ids, rows, probs)]
        with self._lock, closing(self._connect()) as conn, conn:
            conn.executemany(
                "INSERT INTO predictions "
                "(id, created_at, model_version, probability, features) "
                "VALUES (?, ?, ?, ?, ?)", data)
        return ids

    def label(self, items: list[tuple[str, int]]) -> dict:
        """Attach true outcomes (1 = churned). The latest report for an ID wins."""
        now = time.time()
        updated, unknown = 0, []
        with self._lock, closing(self._connect()) as conn, conn:
            for pid, outcome in items:
                cur = conn.execute(
                    "UPDATE predictions SET label = ?, labelled_at = ? WHERE id = ?",
                    (int(outcome), now, pid))
                if cur.rowcount:
                    updated += 1
                else:
                    unknown.append(pid)
        return {"updated": updated, "unknown": unknown}

    def counts(self) -> dict:
        with closing(self._connect()) as conn:
            total, labelled = conn.execute(
                "SELECT COUNT(*), COUNT(label) FROM predictions").fetchone()
        return {"predictions": total, "labelled": labelled}

    def performance(self, window: int, threshold: float, min_labelled: int,
                    min_per_class: int = 10, model_version: str | None = None) -> dict:
        """Recall, precision and PR-AUC over the most recently labelled rows.

        With model_version, only predictions made by that model version count,
        so the metrics always describe the model currently being served.
        """
        where, args = "label IS NOT NULL", []
        if model_version is not None:
            where += " AND model_version = ?"
            args.append(str(model_version))
        with closing(self._connect()) as conn:
            rows = conn.execute(
                f"SELECT probability, label FROM predictions WHERE {where} "
                "ORDER BY labelled_at DESC, rowid DESC LIMIT ?",
                (*args, window)).fetchall()
        p = np.array([r[0] for r in rows], dtype=float)
        y = np.array([r[1] for r in rows], dtype=int)
        n, pos = len(rows), int(y.sum())
        base = {"n_labelled": n, "positives": pos, "min_labelled": min_labelled,
                "threshold": threshold}
        if n < min_labelled or pos < min_per_class or n - pos < min_per_class:
            return {**base, "ready": False, "recall": None, "precision": None,
                    "pr_auc": None}
        pred = (p >= threshold).astype(int)
        return {**base, "ready": True,
                "recall": round(float(recall_score(y, pred)), 4),
                "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
                "pr_auc": round(float(average_precision_score(y, p)), 4)}

    def labelled(self) -> tuple[pd.DataFrame, pd.Series]:
        """All labelled rows as features and outcomes, for retraining."""
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT features, label FROM predictions "
                "WHERE label IS NOT NULL ORDER BY rowid").fetchall()
        X = pd.DataFrame([json.loads(f) for f, _ in rows])
        y = pd.Series([r[1] for r in rows], name="label", dtype=int)
        return X, y
