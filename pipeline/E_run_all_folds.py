"""
Run B (band selection) → C (train + eval) for every fold and aggregate.

For each fold in 0..N_FOLDS-1:
  1. subprocess: python3 B_band_selection.py   (env: FOLD_INDEX, SELECTION_MODE)
  2. subprocess: python3 C_train_eval.py       (env: FOLD_INDEX, SELECTION_MODE)
  3. Load that fold's test_metrics.json into the aggregate.

Stops on the first failure (subprocess.run(check=True) raises). Outputs:
  pipeline/results/all_folds_{mode}_summary.csv   one row per fold + mean/std

Run AFTER pipeline/A_make_splits.py.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR  = Path("/Users/leoqian/mdanderson/EEGOutcomePrediction")
PIPELINE_DIR = PROJECT_DIR / "pipeline"
RESULTS_DIR  = PIPELINE_DIR / "results"

N_FOLDS        = 5
SELECTION_MODE = "top2_per_sheet"

# Metrics pulled from each fold's test_metrics.json into the summary table.
METRIC_COLS = [
    "model", "train_accuracy", "accuracy", "balanced_accuracy",
    "auc", "average_precision", "f1", "threshold",
    "n_features", "n_eeg_features", "n_clinical_features",
]

# Numeric columns that get a mean/std row at the bottom.
NUMERIC_COLS = [
    "train_accuracy", "accuracy", "balanced_accuracy",
    "auc", "average_precision", "f1", "threshold",
]


def run_fold(fold_index: int) -> dict:
    """Run B then C for one fold. Raises on first non-zero exit."""
    env = {**os.environ,
           "FOLD_INDEX":     str(fold_index),
           "SELECTION_MODE": SELECTION_MODE}

    print(f"\n{'#' * 72}\n# Fold {fold_index} — B (band selection)\n{'#' * 72}")
    subprocess.run(["python3", "B_band_selection.py"],
                   cwd=PIPELINE_DIR, env=env, check=True)

    print(f"\n{'#' * 72}\n# Fold {fold_index} — C (train + eval)\n{'#' * 72}")
    subprocess.run(["python3", "C_train_eval.py"],
                   cwd=PIPELINE_DIR, env=env, check=True)

    metrics_path = RESULTS_DIR / f"fold_{fold_index}" / SELECTION_MODE / "test_metrics.json"
    if not metrics_path.exists():
        raise RuntimeError(f"Missing metrics after C: {metrics_path}")
    with open(metrics_path) as f:
        return json.load(f)


def main():
    rows = []
    for fold in range(N_FOLDS):
        m = run_fold(fold)
        row = {"fold": fold}
        for c in METRIC_COLS:
            row[c] = m.get(c)
        rows.append(row)

    df = pd.DataFrame(rows)

    # Append mean/std summary rows for the numeric columns.
    summary = []
    for stat_name, fn in [("mean", np.mean), ("std", np.std)]:
        agg = {"fold": stat_name}
        for c in METRIC_COLS:
            if c in NUMERIC_COLS:
                agg[c] = round(float(fn(df[c].astype(float))), 4)
            else:
                agg[c] = ""
        summary.append(agg)
    out_df = pd.concat([df, pd.DataFrame(summary)], ignore_index=True)

    out_csv = RESULTS_DIR / f"all_folds_{SELECTION_MODE}_summary.csv"
    out_df.to_csv(out_csv, index=False)

    print(f"\n{'=' * 72}")
    print(f"All folds complete: mode={SELECTION_MODE}")
    print(f"{'=' * 72}")
    print(out_df.to_string(index=False))
    print(f"\nSummary CSV: {out_csv}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"\n[FAIL] Subprocess exited {e.returncode} on cmd: {e.cmd}", file=sys.stderr)
        print("Stopping (no aggregate written).", file=sys.stderr)
        sys.exit(e.returncode)
