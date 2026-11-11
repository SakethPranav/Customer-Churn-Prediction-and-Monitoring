"""Retrain on base data plus labelled live rows; promote only if clearly better.

The challenger is compared with the current champion on the newest labelled
live rows, which neither model was trained on. Rejected challengers are logged
to MLflow but not registered.
"""
import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402
import yaml  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

from src.common.features import engineer_features  # noqa: E402
from src.training.train import (build_pipeline, file_md5, git_info,  # noqa: E402
                                load_xy, make_estimator)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("retrain")

EXIT_OK, EXIT_ERROR, EXIT_NOT_ENOUGH_DATA, EXIT_RELOAD_FAILED = 0, 1, 2, 3


def fetch_labelled(api_url: str, limit: int) -> list:
    """Fetch the newest labelled live rows (oldest outcome first)."""
    r = requests.get(f"{api_url}/labelled", params={"limit": limit}, timeout=60)
    r.raise_for_status()
    return r.json()["rows"]


def to_frame(rows: list, columns: list, text_cols: list):
    """Re-engineer features for logged raw rows, in the training column order."""
    X = engineer_features(pd.DataFrame([r["features"] for r in rows]))[columns]
    X = X.astype({c: "object" for c in text_cols})
    y = pd.Series([int(r["label"]) for r in rows])
    return X, y


def weighted_cv_pr_auc(name, hp, X, y, w, seed, folds):
    """Cross-validated PR-AUC with sample weights in training and scoring."""
    y_np = y.to_numpy()
    pos_weight = float(w[y_np == 0].sum() / w[y_np == 1].sum())
    cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
    scores = []
    for tr, va in cv.split(X, y_np):
        pipe = build_pipeline(make_estimator(name, hp, seed, pos_weight), X.iloc[tr])
        pipe.fit(X.iloc[tr], y_np[tr], model__sample_weight=w[tr])
        proba = pipe.predict_proba(X.iloc[va])[:, 1]
        scores.append(average_precision_score(y_np[va], proba, sample_weight=w[va]))
    return float(np.mean(scores)), float(np.std(scores))


def paired_bootstrap(y, p_new, p_old, n, seed):
    """Bootstrap the PR-AUC gain by resampling the same rows for both models."""
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        diffs.append(average_precision_score(y[idx], p_new[idx])
                     - average_precision_score(y[idx], p_old[idx]))
    return np.array(diffs)


def finish(result: dict, code: int) -> int:
    """Print a machine-readable summary line and return the exit code."""
    print("RESULT " + json.dumps(result))
    return code


def main() -> int:
    with open("params.yaml") as f:
        cfg = yaml.safe_load(f)
    p, r, split = cfg["train"], cfg["retrain"], cfg["split"]
    target, min_class = cfg["data"]["target"], cfg["monitoring"]["perf_min_per_class"]
    api_url = os.getenv("API_URL", r["api_url"])
    tracking = os.getenv("MLFLOW_TRACKING_URI", p["tracking_uri"])
    reg_name, seed = p["registered_model_name"], p["random_state"]

    try:
        rows = fetch_labelled(api_url, r["max_live_rows"])
        log.info("Labelled live rows available: %d", len(rows))
        n_hold = r["holdout_rows"]
        hold_labels = [int(x["label"]) for x in rows[-n_hold:]]
        if (len(rows) < r["min_labelled"] or sum(hold_labels) < min_class
                or n_hold - sum(hold_labels) < min_class):
            log.warning("Not enough labelled data to retrain (need %d rows, %d per "
                        "class in the holdout)", r["min_labelled"], min_class)
            return finish({"decision": "not_enough_data", "labelled": len(rows)},
                          EXIT_NOT_ENOUGH_DATA)

        X_base, y_base = load_xy(split["train_path"], target)
        columns = X_base.columns.tolist()
        text_cols = [c for c in columns if not pd.api.types.is_numeric_dtype(X_base[c])]
        X_live, y_live = to_frame(rows[:-n_hold], columns, text_cols)
        X_hold, y_hold = to_frame(rows[-n_hold:], columns, text_cols)
        X_tr = pd.concat([X_base, X_live], ignore_index=True)
        y_tr = pd.concat([y_base, y_live], ignore_index=True)
        w = np.concatenate([np.ones(len(X_base)),
                            np.full(len(X_live), float(r["live_weight"]))])
        log.info("Training rows: %d base + %d live (weight %.1f); holdout: %d",
                 len(X_base), len(X_live), r["live_weight"], len(X_hold))

        mlflow.set_tracking_uri(tracking)
        mlflow.set_experiment(p["experiment_name"])
        client = MlflowClient()
        champ_version = client.get_model_version_by_alias(reg_name, "champion")
        champion = mlflow.sklearn.load_model(f"models:/{reg_name}@champion")
        p_champ = champion.predict_proba(X_hold)[:, 1]

        cv_results = {}
        for name, hp in p["models"].items():
            cv_results[name] = weighted_cv_pr_auc(name, hp, X_tr, y_tr, w, seed,
                                                  p["cv_folds"])
            log.info("%s: weighted cv_pr_auc=%.4f (sd %.4f)", name, *cv_results[name])
        best = max(cv_results, key=lambda k: cv_results[k][0])

        y_np = y_tr.to_numpy()
        pos_weight = float(w[y_np == 0].sum() / w[y_np == 1].sum())
        challenger = build_pipeline(
            make_estimator(best, p["models"][best], seed, pos_weight), X_tr)
        challenger.fit(X_tr, y_np, model__sample_weight=w)
        p_new = challenger.predict_proba(X_hold)[:, 1]

        y_h = y_hold.to_numpy()
        ap_new = float(average_precision_score(y_h, p_new))
        ap_old = float(average_precision_score(y_h, p_champ))
        gain = ap_new - ap_old
        diffs = paired_bootstrap(y_h, p_new, p_champ, r["bootstrap_samples"], seed)
        gain_p5 = float(np.percentile(diffs, 5))
        promote = bool(gain >= r["promote_margin"] and gain_p5 > 0)
        log.info("Holdout PR-AUC: champion=%.4f challenger(%s)=%.4f gain=%+.4f "
                 "(5th percentile %+.4f)", ap_old, best, ap_new, gain, gain_p5)

        with mlflow.start_run(run_name=f"retrain_{best}"):
            mlflow.set_tags({**git_info(), "kind": "retrain", "model_type": best,
                             "train_data_md5": file_md5(split["train_path"]),
                             "promotion": "accepted" if promote else "rejected",
                             "champion_version_at_start": str(champ_version.version)})
            mlflow.log_params({"model_type": best, "live_weight": r["live_weight"],
                               "holdout_rows": len(X_hold), "n_live_train": len(X_live),
                               "n_base_train": len(X_base),
                               "promote_margin": r["promote_margin"],
                               "bootstrap_samples": r["bootstrap_samples"]})
            mlflow.log_metrics({"holdout_pr_auc_challenger": ap_new,
                                "holdout_pr_auc_champion": ap_old,
                                "holdout_gain": gain, "holdout_gain_p5": gain_p5,
                                **{f"weighted_cv_pr_auc_{k}": v[0]
                                   for k, v in cv_results.items()}})
            info = mlflow.sklearn.log_model(
                challenger, artifact_path="model", serialization_format="cloudpickle",
                signature=infer_signature(X_tr, challenger.predict_proba(X_tr)),
                input_example=X_tr.head(3))

        result = {"decision": "promoted" if promote else "rejected",
                  "model_type": best, "champion_version_before": int(champ_version.version),
                  "holdout_pr_auc_champion": round(ap_old, 4),
                  "holdout_pr_auc_challenger": round(ap_new, 4),
                  "gain": round(gain, 4), "gain_p5": round(gain_p5, 4)}
        if promote:
            version = mlflow.register_model(info.model_uri, reg_name)
            client.set_registered_model_alias(reg_name, "champion", version.version)
            result["champion_version_after"] = int(version.version)
            log.info("Promoted %s as %s v%s [champion]", best, reg_name, version.version)
            try:
                requests.post(f"{api_url}/reload", timeout=120).raise_for_status()
                result["reloaded"] = True
                log.info("API reloaded the new champion")
            except Exception:
                result["reloaded"] = False
                log.exception("Promoted, but /reload failed; call /reload manually")
                return finish(result, EXIT_RELOAD_FAILED)
        else:
            log.info("Challenger rejected; champion v%s unchanged", champ_version.version)
        return finish(result, EXIT_OK)
    except Exception:
        log.exception("Retraining failed")
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
