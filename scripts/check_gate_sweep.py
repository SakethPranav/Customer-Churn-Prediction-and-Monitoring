"""Sweep live_weight and labelled-row count: promotion rate with and without a shift."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

from scripts.concept_shift import apply_shift  # noqa: E402
from src.training.retrain import paired_bootstrap  # noqa: E402
from src.training.train import build_pipeline, load_xy, make_estimator  # noqa: E402

SCENARIOS = 30
WEIGHTS = (5.0, 10.0)
ROWS = (600, 1000)
CONDITIONS = ("no_change", "dsl_fiber", "contract_flip")


def main() -> None:
    with open("params.yaml") as f:
        cfg = yaml.safe_load(f)
    p, r, split = cfg["train"], cfg["retrain"], cfg["split"]
    target, seed, n_hold = cfg["data"]["target"], p["random_state"], r["holdout_rows"]
    X_base, y_base = load_xy(split["train_path"], target)
    X_test, y_test = load_xy(split["test_path"], target)
    y_base_np, y_test_np = y_base.to_numpy(), y_test.to_numpy()
    champion = joblib.load("models/best_model.joblib")
    hp = p["models"]["random_forest"]

    print(f"{'rows':>5} {'weight':>6}  {'condition':14s} promoted  mean gain", flush=True)
    for n_rows in ROWS:
        for weight in WEIGHTS:
            for cond in CONDITIONS:
                promoted, gains = [], []
                for k in range(SCENARIOS):
                    idx = np.random.default_rng(5000 + k).permutation(len(X_test))[:n_rows]
                    X_live = X_test.iloc[idx]
                    y_live = (y_test_np[idx] if cond == "no_change" else
                              apply_shift(cond, X_live, y_test_np[idx], seed=4000 + k))
                    X_old, y_old = X_live.iloc[:-n_hold], y_live[:-n_hold]
                    X_hold, y_h = X_live.iloc[-n_hold:], y_live[-n_hold:]
                    X_tr = pd.concat([X_base, X_old], ignore_index=True)
                    y_tr = np.concatenate([y_base_np, y_old])
                    w = np.concatenate([np.ones(len(X_base)),
                                        np.full(len(X_old), weight)])
                    pos_weight = float(w[y_tr == 0].sum() / w[y_tr == 1].sum())
                    challenger = build_pipeline(
                        make_estimator("random_forest", hp, seed, pos_weight), X_tr)
                    challenger.fit(X_tr, y_tr, model__sample_weight=w)
                    p_new = challenger.predict_proba(X_hold)[:, 1]
                    p_old = champion.predict_proba(X_hold)[:, 1]
                    gain = (average_precision_score(y_h, p_new)
                            - average_precision_score(y_h, p_old))
                    diffs = paired_bootstrap(y_h, p_new, p_old,
                                             r["bootstrap_samples"], seed)
                    gains.append(gain)
                    promoted.append(bool(gain >= r["promote_margin"]
                                         and np.percentile(diffs, 5) > 0))
                print(f"{n_rows:5d} {weight:6.1f}  {cond:14s} "
                      f"{sum(promoted):2d}/{SCENARIOS}     {np.mean(gains):+.4f}", flush=True)


if __name__ == "__main__":
    main()
