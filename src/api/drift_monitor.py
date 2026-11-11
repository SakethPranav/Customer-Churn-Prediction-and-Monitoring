"""Rolling-window input drift monitor (PSI against the training reference).

Window size, minimum rows and thresholds come from the settings block of the
reference file, which DVC builds from params.yaml, so the pipeline and the API
always use the same values.
"""
import json
import logging
import os
import threading
from collections import deque

import pandas as pd

from src.monitoring.drift import feature_psi

log = logging.getLogger("churn-api")

# Used only if the reference file has no settings block (older files).
DEFAULT_SETTINGS = {"window_size": 300, "min_window": 300,
                    "psi_warn": 0.1, "psi_alert": 0.25,
                    "perf_window": 500, "perf_min_labelled": 500,
                    "perf_min_per_class": 10, "perf_pr_auc_floor": 0.54}


class DriftMonitor:
    """Keeps the most recent requests and scores them against the reference."""

    def __init__(self) -> None:
        self.reference_path = os.getenv("REFERENCE_PATH", "monitoring/reference_stats.json")
        self._lock = threading.Lock()
        self._settings = dict(DEFAULT_SETTINGS)
        self._rows: deque = deque(maxlen=self._settings["window_size"])
        self._features: dict = {}
        self.reload_reference()

    def reload_reference(self) -> None:
        """(Re)load the reference file and start a fresh window.

        If the file cannot be read or its settings are invalid, the previous
        reference and settings are kept.
        """
        try:
            with open(self.reference_path) as f:
                data = json.load(f)
            features = data["features"]
            raw = {**DEFAULT_SETTINGS, **data.get("settings", {})}
            settings = {"window_size": int(raw["window_size"]),
                        "min_window": int(raw["min_window"]),
                        "psi_warn": float(raw["psi_warn"]),
                        "psi_alert": float(raw["psi_alert"]),
                        "perf_window": int(raw["perf_window"]),
                        "perf_min_labelled": int(raw["perf_min_labelled"]),
                        "perf_min_per_class": int(raw["perf_min_per_class"]),
                        "perf_pr_auc_floor": float(raw["perf_pr_auc_floor"])}
            if settings["min_window"] > settings["window_size"]:
                raise ValueError("min_window cannot exceed window_size")
            if settings["perf_min_labelled"] > settings["perf_window"]:
                raise ValueError("perf_min_labelled cannot exceed perf_window")
        except Exception:
            log.exception("Could not load drift reference from %s", self.reference_path)
            return
        with self._lock:
            self._features = features
            self._settings = settings
            self._rows = deque(maxlen=settings["window_size"])
        log.info("Drift reference loaded: %d features, window %d, warn %.2f, alert %.2f",
                 len(features), settings["window_size"],
                 settings["psi_warn"], settings["psi_alert"])

    @property
    def settings(self) -> dict:
        """Settings from the reference file, including the live-performance ones."""
        with self._lock:
            return dict(self._settings)

    def add(self, rows: list[dict]) -> None:
        with self._lock:
            self._rows.extend(rows)

    def report(self) -> dict:
        with self._lock:
            rows = list(self._rows)
            features = self._features
            s = dict(self._settings)
        n = len(rows)
        base = {"window_size": n, "min_window": s["min_window"]}
        if not features or n < s["min_window"]:
            return {**base, "ready": False, "max_psi": 0.0,
                    "status": "insufficient_data",
                    "feature_psi": {name: 0.0 for name in features}}
        df = pd.DataFrame(rows)
        psi = {name: round(feature_psi(spec, df[name]), 4)
               for name, spec in features.items()}
        worst = max(psi.values())
        status = ("alert" if worst >= s["psi_alert"]
                  else "warn" if worst >= s["psi_warn"] else "ok")
        return {**base, "ready": True, "max_psi": worst, "status": status,
                "feature_psi": psi}
