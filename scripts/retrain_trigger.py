"""Decide whether to retrain, and run src/training/retrain.py if so.

Triggers: live quality is degraded, or input drift is at alert level and enough
labelled rows exist to validate a challenger. After an attempt, a cooldown
requires new labelled rows before the next one.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
STATE_FILE = ROOT / "retrain_state.json"


def decide(perf: dict, drift: dict, cfg: dict, state: dict):
    """Return (action, signal, reason); action is 'retrain' or 'skip'."""
    if perf["status"] == "degraded":
        signal = "performance_degraded"
    elif drift["status"] == "alert":
        signal = "drift_alert"
    else:
        return ("skip", None, f"no trigger (performance {perf['status']}, "
                              f"drift {drift['status']})")
    labelled = perf["labelled_total"]
    if labelled < cfg["min_labelled"]:
        return ("skip", signal, f"{signal}, but only {labelled} labelled rows "
                                f"(need {cfg['min_labelled']})")
    if "labelled_total_at_last_attempt" in state:
        new = labelled - state["labelled_total_at_last_attempt"]
        if new < cfg["cooldown_new_labels"]:
            return ("skip", signal, f"{signal}, but cooling down: {new} new labelled "
                                    f"rows since the last attempt "
                                    f"(need {cfg['cooldown_new_labels']})")
    return "retrain", signal, f"{signal} with {labelled} labelled rows"


def get(api_url: str, path: str) -> dict:
    r = requests.get(f"{api_url}{path}", timeout=30)
    r.raise_for_status()
    return r.json()


def load_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text())
    except (OSError, ValueError):
        return {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="decide, but do not retrain")
    args = ap.parse_args()
    with open(ROOT / "params.yaml") as f:
        cfg = yaml.safe_load(f)["retrain"]
    api_url = os.getenv("API_URL", cfg["api_url"]).rstrip("/")
    try:
        perf, drift = get(api_url, "/performance"), get(api_url, "/drift")
    except requests.RequestException as err:
        print(f"Cannot read the API at {api_url}: {err}")
        return 1

    action, signal, reason = decide(perf, drift, cfg, load_state())
    summary = {"action": action, "signal": signal, "reason": reason,
               "performance_status": perf["status"], "drift_status": drift["status"],
               "labelled_total": perf["labelled_total"]}
    if action == "skip" or args.dry_run:
        if action == "retrain":
            summary["note"] = "dry run: retraining not started"
        print("RESULT " + json.dumps(summary))
        return 0

    print(f"Retraining: {reason}", flush=True)
    proc = subprocess.run([sys.executable, str(ROOT / "src/training/retrain.py")],
                          cwd=ROOT, stdout=subprocess.PIPE, text=True)
    outcome = None
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            outcome = json.loads(line[len("RESULT "):])
    summary["retrain_exit_code"] = proc.returncode
    summary["retrain"] = outcome
    if proc.returncode in (0, 3) and outcome:  # a decision was made: start the cooldown
        STATE_FILE.write_text(json.dumps({
            "labelled_total_at_last_attempt": perf["labelled_total"],
            "last_decision": outcome.get("decision"), "last_signal": signal}))
    print("RESULT " + json.dumps(summary))
    return 0 if proc.returncode in (0, 2) else proc.returncode


if __name__ == "__main__":
    sys.exit(main())
