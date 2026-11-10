"""DVC stage: build reference feature distributions and monitor settings."""
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.monitoring.drift import categorical_reference, numeric_reference  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build-reference")

SETTING_KEYS = ("n_bins", "window_size", "min_window", "psi_warn", "psi_alert")


def main() -> None:
    with open("params.yaml") as f:
        p = yaml.safe_load(f)["monitoring"]
    try:
        if p["min_window"] > p["window_size"]:
            raise ValueError("monitoring.min_window cannot exceed monitoring.window_size")
        df = pd.read_csv(p["train_path"])
        wanted = p["numeric_features"] + p["categorical_features"]
        missing = [c for c in wanted if c not in df.columns]
        if missing:
            raise ValueError(f"Columns missing from {p['train_path']}: {missing}")
        features = {}
        for col in p["numeric_features"]:
            features[col] = numeric_reference(df[col], p["n_bins"])
        for col in p["categorical_features"]:
            features[col] = categorical_reference(df[col])
        settings = {k: p[k] for k in SETTING_KEYS}
        out = Path(p["reference_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"n_rows": len(df), "source": p["train_path"],
                                   "settings": settings, "features": features},
                                  indent=2))
        log.info("Wrote reference for %d features from %d rows to %s",
                 len(features), len(df), out)
    except Exception:
        log.exception("Building reference statistics failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
