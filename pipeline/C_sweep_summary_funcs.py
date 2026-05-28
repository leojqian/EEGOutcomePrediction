"""
Sweep SUMMARY_FUNCS combinations on the same fold + selection mode.

Reuses C_train_eval.run() — shares the (expensive) module-level setup
across all combos: fold IDs, selected bands from B, files_by_pid map.

Each combo's full artifacts go to:
  pipeline/results/fold_{K}/{mode}/sweep_summary_funcs/{label}/
A side-by-side comparison is written to:
  pipeline/results/fold_{K}/{mode}/sweep_summary_funcs/comparison.csv

Run AFTER A and B (same prerequisites as C_train_eval.py). The fold and
SELECTION_MODE follow whatever C_train_eval.py is currently set to.
"""

import pandas as pd

from C_train_eval import run, OUT_DIR, FOLD_INDEX, SELECTION_MODE

COMBOS = [
    ["mean",   "std"],      # current default
    ["mean",   "median"],
    ["median", "p90"],
    ["mean",   "p95"],
    ["p25",    "p75"],
]

SWEEP_DIR = OUT_DIR / "sweep_summary_funcs"
SWEEP_DIR.mkdir(parents=True, exist_ok=True)

rows = []
for combo in COMBOS:
    label = "_".join(combo)
    sub = SWEEP_DIR / label
    print(f"\n{'#' * 70}\n# summary_funcs = {combo}\n{'#' * 70}")
    metrics = run(summary_funcs=combo, out_dir=sub)
    rows.append({
        "summary_funcs": "+".join(combo),
        "n_features":    metrics["n_features"],
        "model":         metrics["model"],
        "threshold":     round(metrics["threshold"], 3),
        "train_acc":     round(metrics["train_accuracy"], 3),
        "val_acc":       round(metrics["accuracy"], 3),
        "val_bal_acc":   round(metrics["balanced_accuracy"], 3),
        "val_auc":       round(metrics["auc"], 3),
        "val_ap":        round(metrics["average_precision"], 3),
        "val_f1":        round(metrics["f1"], 3),
        "out_dir":       str(sub.relative_to(OUT_DIR.parent.parent)),
    })

df = pd.DataFrame(rows).sort_values("val_bal_acc", ascending=False).reset_index(drop=True)
df.to_csv(SWEEP_DIR / "comparison.csv", index=False)

print(f"\n{'=' * 70}")
print(f"Sweep complete: fold {FOLD_INDEX}, mode={SELECTION_MODE}")
print(f"{'=' * 70}")
print(df.to_string(index=False))
print(f"\nComparison: {SWEEP_DIR / 'comparison.csv'}")
print(f"Per-combo artifacts: {SWEEP_DIR}/<label>/")
