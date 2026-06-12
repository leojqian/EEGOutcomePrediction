"""
Driver for the v2 pipeline: re-uses Module B's existing top2_per_sheet
band selections (from pipeline/bands/fold_{k}/top2_per_sheet/) and
re-runs only C_train_eval_v2.py for each fold.
"""
import json, os, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd

PROJECT_DIR  = Path("/Users/leoqian/mdanderson/EEGOutcomePrediction")
PIPELINE_DIR = PROJECT_DIR / "pipeline"
RESULTS_DIR  = PIPELINE_DIR / "results"

N_FOLDS    = 5
OUTPUT_TAG = "top2_per_sheet_v2"

METRIC_COLS = [
    "model", "train_accuracy", "accuracy", "balanced_accuracy",
    "auc", "average_precision", "f1", "threshold",
    "n_features", "n_eeg_features", "n_clinical_features",
]
NUMERIC_COLS = [
    "train_accuracy", "accuracy", "balanced_accuracy",
    "auc", "average_precision", "f1", "threshold",
]

def run_fold(k):
    env = {**os.environ, "FOLD_INDEX": str(k)}
    print(f"\n{'#'*72}\n# Fold {k} — C v2 (train+eval w/ OOF threshold + RF+SVM heads)\n{'#'*72}")
    subprocess.run(["python3", "C_train_eval_v2.py"],
                   cwd=PIPELINE_DIR, env=env, check=True)
    mp = RESULTS_DIR / f"fold_{k}" / OUTPUT_TAG / "test_metrics.json"
    if not mp.exists():
        raise RuntimeError(f"Missing metrics: {mp}")
    return json.load(open(mp))

def main():
    rows = []
    for k in range(N_FOLDS):
        m = run_fold(k)
        row = {"fold": k}
        for c in METRIC_COLS:
            row[c] = m.get(c)
        rows.append(row)

    df = pd.DataFrame(rows)
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

    out_csv = RESULTS_DIR / f"all_folds_{OUTPUT_TAG}_summary.csv"
    out_df.to_csv(out_csv, index=False)

    print(f"\n{'='*72}\nAll folds complete (v2)\n{'='*72}")
    print(out_df.to_string(index=False))
    print(f"\nSummary CSV: {out_csv}")

if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"\n[FAIL] exit {e.returncode}: {e.cmd}", file=sys.stderr)
        sys.exit(e.returncode)
