"""For each concept shift: how badly does the champion degrade, and how often
does the retrain gate promote a better challenger afterwards?

Same setup as check_gate_false_promotions.py, but the labels follow the shifted
behaviour. Only the random forest is simulated.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

from scripts.concept_shift import SHIFTS, apply_shift  # noqa: E402
from src.training.retrain import paired_bootstrap  # noqa: E402
from src.training.train import build_pipeline, load_xy, make_estimator  # noqa: E402

SCENARIOS = 40
N_ROWS = 600


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
    p_champ_all = champion.predict_proba(X_test)[:, 1]
    print("Champion PR-AUC on the unshifted test set: "
          f"{average_precision_score(y_test_np, p_champ_all):.4f}")

    for name in SHIFTS:
        y_all = apply_shift(name, X_test, y_test_np, seed=1)
        print(f"\n[{name}] churn rate {y_all.mean():.3f} "
              f"(unshifted {y_test_np.mean():.3f}); champion PR-AUC on the "
              f"shifted test set {average_precision_score(y_all, p_champ_all):.4f}",
              flush=True)
        gains, promoted = [], []
        for k in range(SCENARIOS):
            idx = np.random.default_rng(3000 + k).permutation(len(X_test))[:N_ROWS]
            X_live = X_test.iloc[idx]
            y_live = apply_shift(name, X_live, y_test_np[idx], seed=2000 + k)
            X_old, y_old = X_live.iloc[:-n_hold], y_live[:-n_hold]
            X_hold, y_h = X_live.iloc[-n_hold:], y_live[-n_hold:]
            X_tr = pd.concat([X_base, X_old], ignore_index=True)
            y_tr = np.concatenate([y_base_np, y_old])
            w = np.concatenate([np.ones(len(X_base)),
                                np.full(len(X_old), float(r["live_weight"]))])
            pos_weight = float(w[y_tr == 0].sum() / w[y_tr == 1].sum())
            challenger = build_pipeline(
                make_estimator("random_forest", hp, seed, pos_weight), X_tr)
            challenger.fit(X_tr, y_tr, model__sample_weight=w)
            p_new = challenger.predict_proba(X_hold)[:, 1]
            p_old = champion.predict_proba(X_hold)[:, 1]
            gain = (average_precision_score(y_h, p_new)
                    - average_precision_score(y_h, p_old))
            diffs = paired_bootstrap(y_h, p_new, p_old, r["bootstrap_samples"], seed)
            p5 = float(np.percentile(diffs, 5))
            gains.append(gain)
            promoted.append(bool(gain >= r["promote_margin"] and p5 > 0))
        g = np.array(gains)
        print(f"  gate promotion rate {np.mean(promoted):.2f} over {SCENARIOS} "
              f"scenarios; gain mean {g.mean():+.4f}, 5th/95th percentile "
              f"{np.percentile(g, 5):+.4f} / {np.percentile(g, 95):+.4f}", flush=True)


if __name__ == "__main__":
    main()
