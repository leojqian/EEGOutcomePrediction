"""
Task A driver — baseline-pain prediction across one or more pain thresholds.

For each threshold thr:
  1. A_make_splits_taskA.py  (PAIN_THRESHOLD=thr) → splits_taskA{thr}/
  2. per fold k: B_band_selection.py then C_train_eval_taskA.py
                 (SPLITS_TAG=taskA{thr}, FOLD_INDEX=k)
  3. aggregate per-fold metrics + pooled out-of-fold (OOF) AUC/accuracy.

Usage:
  python3 E_run_all_folds_taskA.py            # threshold 6 (near-balanced)
  python3 E_run_all_folds_taskA.py 4 5 6      # sweep all three

Writes:
  pipeline/results_taskA{thr}/all_folds_taskA_top2_summary.csv   (per threshold)
  pipeline/results_taskA_threshold_comparison.csv                (across thresholds)
"""
import json, os, sys, subprocess
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score

PROJECT_DIR  = Path("/Users/leoqian/mdanderson/EEGOutcomePrediction")
PIPELINE_DIR = PROJECT_DIR / "pipeline"
N_FOLDS      = 5
OUTPUT_TAG   = "taskA_top2"
SELECTION_MODE = "top2_per_sheet"

METRIC_COLS = ["model", "train_accuracy", "accuracy", "balanced_accuracy",
               "auc", "average_precision", "f1", "threshold",
               "n_features", "n_eeg_features", "n_clinical_features"]
NUMERIC_COLS = ["train_accuracy", "accuracy", "balanced_accuracy",
                "auc", "average_precision", "f1", "threshold"]


def run(script_args, env):
    # Use the same interpreter that launched this driver (the venv python),
    # so child scripts inherit openpyxl/xgboost/etc.
    subprocess.run([sys.executable, *script_args],
                   cwd=PIPELINE_DIR, env={**os.environ, **env}, check=True)


def run_threshold(thr):
    tag = f"taskA{thr}"
    results_dir = PIPELINE_DIR / f"results_{tag}"
    print(f"\n{'#'*72}\n# Task A — threshold pain_t1 >= {thr}  (SPLITS_TAG={tag})\n{'#'*72}")

    # 1. splits
    run(["A_make_splits_taskA.py"], {"PAIN_THRESHOLD": str(thr)})

    # 2. per fold: band selection + train/eval
    rows, oof = [], []
    for k in range(N_FOLDS):
        env = {"SPLITS_TAG": tag, "FOLD_INDEX": str(k),
               "SELECTION_MODE": SELECTION_MODE}
        print(f"\n--- {tag} fold {k}: Module B (band selection) ---")
        run(["B_band_selection.py"], env)
        print(f"--- {tag} fold {k}: Module C (Task A train/eval) ---")
        run(["C_train_eval_taskA.py"], env)

        mp = results_dir / f"fold_{k}" / OUTPUT_TAG / "test_metrics.json"
        m = json.load(open(mp))
        rows.append({"fold": k, **{c: m.get(c) for c in METRIC_COLS}})

        pp = results_dir / f"fold_{k}" / OUTPUT_TAG / "test_predictions.csv"
        oof.append(pd.read_csv(pp))

    # per-fold summary
    df = pd.DataFrame(rows)
    summary = []
    for stat_name, fn in [("mean", np.mean), ("std", np.std)]:
        agg = {"fold": stat_name}
        for c in METRIC_COLS:
            agg[c] = (round(float(fn(df[c].astype(float))), 4)
                      if c in NUMERIC_COLS else "")
        summary.append(agg)
    out_df = pd.concat([df, pd.DataFrame(summary)], ignore_index=True)
    out_csv = results_dir / f"all_folds_{OUTPUT_TAG}_summary.csv"
    out_df.to_csv(out_csv, index=False)

    # pooled OOF
    oof_df = pd.concat(oof, ignore_index=True)
    pooled = {
        "threshold": thr,
        "n": int(len(oof_df)),
        "pos_rate": float(oof_df["y_true"].mean()),
        "pooled_auc": round(float(roc_auc_score(oof_df["y_true"], oof_df["proba"])), 4),
        "pooled_accuracy": round(float(accuracy_score(oof_df["y_true"], oof_df["y_pred"])), 4),
        "pooled_bal_acc": round(float(balanced_accuracy_score(oof_df["y_true"], oof_df["y_pred"])), 4),
        "mean_fold_auc": round(float(df["auc"].astype(float).mean()), 4),
        "std_fold_auc": round(float(df["auc"].astype(float).std()), 4),
        "mean_fold_bal_acc": round(float(df["balanced_accuracy"].astype(float).mean()), 4),
    }
    print(f"\n{'='*72}\nTask A threshold {thr} complete\n{'='*72}")
    print(out_df.to_string(index=False))
    print(f"\nPooled OOF (n={pooled['n']}, pos_rate={pooled['pos_rate']:.3f}): "
          f"AUC={pooled['pooled_auc']}  acc={pooled['pooled_accuracy']}  "
          f"bal_acc={pooled['pooled_bal_acc']}")
    print(f"Summary CSV: {out_csv}")
    return pooled


def main():
    thresholds = [int(a) for a in sys.argv[1:]] or [6]
    pooled_rows = [run_threshold(t) for t in thresholds]
    comp = pd.DataFrame(pooled_rows)
    comp_csv = PIPELINE_DIR / "results_taskA_threshold_comparison.csv"
    comp.to_csv(comp_csv, index=False)
    print(f"\n{'#'*72}\n# Task A threshold comparison\n{'#'*72}")
    print(comp.to_string(index=False))
    print(f"\nComparison CSV: {comp_csv}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"\n[FAIL] {e.returncode}: {e.cmd}", file=sys.stderr)
        sys.exit(e.returncode)
