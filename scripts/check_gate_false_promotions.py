"""Measure how often the retrain promotion gate promotes when nothing has changed.

Each scenario draws 600 rows from the held-out test set (the same population as
training), uses the first 400 as older labelled live rows and the last 200 as the
holdout, and applies the same rule as src/training/retrain.py. The champion is the
model trained on train.csv only (models/best_model.joblib). Only the random forest
is simulated, as it won the model choice in the real run.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

from src.training.retrain import paired_bootstrap  # noqa: E402
from src.training.train import build_pipeline, load_xy, make_estimator  # noqa: E402

SCENARIOS = 100
N_ROWS = 600


def main() -> None:
    with open("params.yaml") as f:
        cfg = yaml.safe_load(f)
    p, r, split = cfg["train"], cfg["retrain"], cfg["split"]
    target, seed, n_hold = cfg["data"]["target"], p["random_state"], r["holdout_rows"]
    X_base, y_base = load_xy(split["train_path"], target)
    X_test, y_test = load_xy(split["test_path"], target)
    champion = joblib.load("models/best_model.joblib")
    hp = p["models"]["random_forest"]

    gains, p5s, promoted = [], [], []
    for k in range(SCENARIOS):
        idx = np.random.default_rng(1000 + k).permutation(len(X_test))[:N_ROWS]
        X_live, y_live = X_test.iloc[idx], y_test.iloc[idx]
        X_old, y_old = X_live.iloc[:-n_hold], y_live.iloc[:-n_hold]
        X_hold, y_hold = X_live.iloc[-n_hold:], y_live.iloc[-n_hold:]
        X_tr = pd.concat([X_base, X_old], ignore_index=True)
        y_tr = pd.concat([y_base, y_old], ignore_index=True).to_numpy()
        w = np.concatenate([np.ones(len(X_base)),
                            np.full(len(X_old), float(r["live_weight"]))])
        pos_weight = float(w[y_tr == 0].sum() / w[y_tr == 1].sum())
        challenger = build_pipeline(
            make_estimator("random_forest", hp, seed, pos_weight), X_tr)
        challenger.fit(X_tr, y_tr, model__sample_weight=w)

        y_h = y_hold.to_numpy()
        p_new = challenger.predict_proba(X_hold)[:, 1]
        p_old = champion.predict_proba(X_hold)[:, 1]
        gain = average_precision_score(y_h, p_new) - average_precision_score(y_h, p_old)
        diffs = paired_bootstrap(y_h, p_new, p_old, r["bootstrap_samples"], seed)
        p5 = float(np.percentile(diffs, 5))
        gains.append(gain)
        p5s.append(p5)
        promoted.append(bool(gain >= r["promote_margin"] and p5 > 0))
        if (k + 1) % 10 == 0:
            print(f"{k + 1}/{SCENARIOS} done, promotions so far: {sum(promoted)}",
                  flush=True)

    g = np.array(gains)
    print(f"\nScenarios: {SCENARIOS} (no change in the data)")
    print(f"Promotion rate: {np.mean(promoted):.2f}")
    print(f"Gain: mean {g.mean():+.4f}, sd {g.std():.4f}, "
          f"5th/95th percentile {np.percentile(g, 5):+.4f} / {np.percentile(g, 95):+.4f}")
    print(f"Share with gain >= {r['promote_margin']}: {np.mean(g >= r['promote_margin']):.2f}")
    print(f"Share with bootstrap 5th percentile > 0: {np.mean(np.array(p5s) > 0):.2f}")


if __name__ == "__main__":
    main()
