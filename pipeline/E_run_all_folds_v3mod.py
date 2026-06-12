"""
v3+modality driver: same splits as v1/v2 (label-stratified, the v2 baseline),
same band selection (top_2_per_sheet from pipeline/bands/), but Module C v3
which adds the 3 treatment-modality one-hot features (NFB, DL, NFB+DL) to
the clinical scalars.

Writes to: pipeline/results/fold_{k}/top2_per_sheet_v3_mod/
"""
import json, os, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd

PROJECT_DIR  = Path("/Users/leoqian/mdanderson/EEGOutcomePrediction")
PIPELINE_DIR = PROJECT_DIR / "pipeline"
RESULTS_DIR  = PIPELINE_DIR / "results"
N_FOLDS      = 5
OUTPUT_TAG   = "top2_per_sheet_v3_mod"

METRIC_COLS = ["model", "train_accuracy", "accuracy", "balanced_accuracy",
               "auc", "average_precision", "f1", "threshold",
               "n_features", "n_eeg_features", "n_clinical_features"]
NUMERIC_COLS = ["train_accuracy", "accuracy", "balanced_accuracy",
                "auc", "average_precision", "f1", "threshold"]

def run_fold(k):
    env = {**os.environ, "FOLD_INDEX": str(k)}
    print(f"\n{'#'*72}\n# v3-mod Fold {k} — Module C v3 (with modality features)\n{'#'*72}")
    subprocess.run(["python3", "C_train_eval_v3.py"],
                   cwd=PIPELINE_DIR, env=env, check=True)
    mp = RESULTS_DIR / f"fold_{k}" / OUTPUT_TAG / "test_metrics.json"
    if not mp.exists():
        raise RuntimeError(f"Missing metrics: {mp}")
    return json.load(open(mp))

def main():
    rows = []
    for k in range(N_FOLDS):
        m = run_fold(k)
        rows.append({"fold": k, **{c: m.get(c) for c in METRIC_COLS}})
    df = pd.DataFrame(rows)
    summary = []
    for stat_name, fn in [("mean", np.mean), ("std", np.std)]:
        agg = {"fold": stat_name}
        for c in METRIC_COLS:
            agg[c] = (round(float(fn(df[c].astype(float))), 4)
                      if c in NUMERIC_COLS else "")
        summary.append(agg)
    out_df = pd.concat([df, pd.DataFrame(summary)], ignore_index=True)
    out_csv = RESULTS_DIR / f"all_folds_{OUTPUT_TAG}_summary.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"\n{'='*72}\nv3-mod pipeline complete\n{'='*72}")
    print(out_df.to_string(index=False))
    print(f"\nSummary CSV: {out_csv}")

if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"\n[FAIL] {e.returncode}: {e.cmd}", file=sys.stderr)
        sys.exit(e.returncode)
