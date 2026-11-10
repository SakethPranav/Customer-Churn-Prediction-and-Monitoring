"""Model training stage with MLflow experiment tracking."""
import hashlib
import json
import logging
import subprocess
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, average_precision_score,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def git_info() -> dict:
    """Return the current Git commit hash and whether the tree has changes."""
    def run(*args: str) -> str:
        return subprocess.check_output(["git", *args], text=True).strip()
    try:
        return {"git_commit": run("rev-parse", "HEAD"),
                "git_dirty": str(bool(run("status", "--porcelain")))}
    except Exception:
        log.warning("Could not read Git information")
        return {"git_commit": "unknown", "git_dirty": "unknown"}


def file_md5(path: str) -> str:
    """MD5 hash of a file, used to identify the exact training data."""
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def load_xy(path: str, target: str):
    """Load a CSV and split it into features and target."""
    df = pd.read_csv(path)
    y = df[target]
    X = df.drop(columns=[target])
    text_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    return X.astype({c: "object" for c in text_cols}), y


def make_estimator(name: str, hp: dict, seed: int, pos_weight: float):
    """Create a classifier; class imbalance is handled via weights."""
    if name == "logistic_regression":
        return LogisticRegression(class_weight="balanced", random_state=seed, **hp)
    if name == "random_forest":
        return RandomForestClassifier(class_weight="balanced", random_state=seed,
                                      n_jobs=-1, **hp)
    if name == "xgboost":
        return XGBClassifier(scale_pos_weight=pos_weight, random_state=seed,
                             eval_metric="logloss", n_jobs=-1, **hp)
    raise ValueError(f"Unknown model type: {name}")


def build_pipeline(estimator, X: pd.DataFrame) -> Pipeline:
    """Scaling + one-hot encoding + classifier, fit on training data only."""
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    pre = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])
    return Pipeline([("prep", pre), ("model", estimator)])


def evaluate(pipe: Pipeline, X, y, threshold: float = 0.5) -> dict:
    """Compute test metrics for a fitted pipeline."""
    proba = pipe.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    return {
        "test_roc_auc": roc_auc_score(y, proba),
        "test_pr_auc": average_precision_score(y, proba),
        "test_recall": recall_score(y, pred),
        "test_precision": precision_score(y, pred),
        "test_f1": f1_score(y, pred),
    }


def main() -> None:
    with open("params.yaml") as f:
        cfg = yaml.safe_load(f)
    p, target = cfg["train"], cfg["data"]["target"]
    split = cfg["split"]

    try:
        X_train, y_train = load_xy(split["train_path"], target)
        X_test, y_test = load_xy(split["test_path"], target)
        pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
        tags = {**git_info(), "train_data_md5": file_md5(split["train_path"])}
        log.info("Train %s, test %s", X_train.shape, X_test.shape)

        mlflow.set_tracking_uri(p["tracking_uri"])
        mlflow.set_experiment(p["experiment_name"])
        cv = StratifiedKFold(p["cv_folds"], shuffle=True, random_state=p["random_state"])

        results = []
        for name, hp in p["models"].items():
            with mlflow.start_run(run_name=name) as run:
                pipe = build_pipeline(
                    make_estimator(name, hp, p["random_state"], pos_weight), X_train)
                mlflow.set_tags({**tags, "model_type": name})
                mlflow.log_params({"model_type": name, **hp,
                                   "cv_folds": p["cv_folds"],
                                   "random_state": p["random_state"],
                                   "n_train_rows": len(X_train),
                                   "n_input_features": X_train.shape[1]})

                cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv,
                                            scoring="average_precision")
                pipe.fit(X_train, y_train)
                metrics = {"cv_pr_auc_mean": float(cv_scores.mean()),
                           "cv_pr_auc_std": float(cv_scores.std()),
                           **evaluate(pipe, X_test, y_test)}
                mlflow.log_metrics(metrics)

                disp = ConfusionMatrixDisplay.from_predictions(
                    y_test, pipe.predict(X_test))
                mlflow.log_figure(disp.figure_, "confusion_matrix.png")
                mlflow.log_artifact("params.yaml")

                info = mlflow.sklearn.log_model(
                    pipe, artifact_path="model", serialization_format="cloudpickle",
                    signature=infer_signature(X_train, pipe.predict_proba(X_train)),
                    input_example=X_train.head(3))
                results.append({"name": name, "run_id": run.info.run_id,
                                "model_uri": info.model_uri, "pipe": pipe,
                                "metrics": metrics})
                log.info("%s: cv_pr_auc=%.4f test_pr_auc=%.4f test_recall=%.4f",
                         name, metrics["cv_pr_auc_mean"],
                         metrics["test_pr_auc"], metrics["test_recall"])

        # Select the winner by cross-validated score, never by test score.
        best = max(results, key=lambda r: r["metrics"]["cv_pr_auc_mean"])
        version = mlflow.register_model(best["model_uri"], p["registered_model_name"])
        MlflowClient().set_registered_model_alias(
            p["registered_model_name"], "champion", version.version)
        log.info("Best model: %s (run %s) registered as %s v%s [champion]",
                 best["name"], best["run_id"], p["registered_model_name"],
                 version.version)
    except Exception:
        log.exception("Training failed (tracking server: %s)",
                      p["tracking_uri"])
        sys.exit(1)

    Path("models").mkdir(exist_ok=True)
    joblib.dump(best["pipe"], "models/best_model.joblib")
    Path("metrics").mkdir(exist_ok=True)
    Path("metrics/train_metrics.json").write_text(json.dumps({
        "best_model": best["name"], "mlflow_run_id": best["run_id"],
        "registered_version": int(version.version),
        **{k: round(v, 4) for k, v in best["metrics"].items()},
    }, indent=2))


if __name__ == "__main__":
    main()
