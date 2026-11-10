"""Stratified train/test split stage."""
import logging
import sys

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    with open("params.yaml") as f:
        cfg = yaml.safe_load(f)
    params, target = cfg["split"], cfg["data"]["target"]

    try:
        df = pd.read_csv(params["input_path"])
        train, test = train_test_split(
            df, test_size=params["test_size"],
            random_state=params["random_state"], stratify=df[target],
        )
    except Exception:
        log.exception("Train/test split failed")
        sys.exit(1)

    train.to_csv(params["train_path"], index=False)
    test.to_csv(params["test_path"], index=False)
    log.info("Train %s (churn rate %.4f), test %s (churn rate %.4f)",
             train.shape, train[target].mean(), test.shape, test[target].mean())


if __name__ == "__main__":
    main()
