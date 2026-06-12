"""
Module A (Task A) — splits for BASELINE-PAIN prediction.

Task A (Figure 1): predict whether a patient's baseline (T1) pain
unpleasantness is at or above a threshold, from baseline EEG + age +
neuropathy duration. pain_t1 is the TARGET here, so (unlike Task B) it is
NOT stratified on and NOT used as a feature downstream.

Label:  y = 1[ pain_t1 >= PAIN_THRESHOLD ]   (high baseline pain).
Default PAIN_THRESHOLD = 6 (the cohort median → near-balanced split,
pos_rate ~0.59). Thresholds 4 and 5 are more imbalanced (0.78 / 0.71);
compare them on AUC rather than accuracy.

Writes per-threshold split dirs so several thresholds coexist:
  pipeline/splits_taskA{thr}/fold_{k}_train.npy / _val.npy
  pipeline/splits_taskA{thr}/labels.json
  pipeline/splits_taskA{thr}/manifest.json

Downstream Modules B and C read these via SPLITS_TAG=taskA{thr}.
"""
import os, re, glob, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

PROJECT_DIR   = Path("/Users/leoqian/mdanderson/EEGOutcomePrediction")
DATA_DIR      = PROJECT_DIR / "processeddata"
OUTCOMES_FILE = DATA_DIR / "Randomization factors and Primary outcome.xlsx"

PAIN_THRESHOLD = float(os.environ.get("PAIN_THRESHOLD", 6))
N_FOLDS        = 5
RANDOM_SEED    = 42

SPLITS_TAG = f"taskA{int(PAIN_THRESHOLD)}"
SPLITS_DIR = PROJECT_DIR / "pipeline" / f"splits_{SPLITS_TAG}"
SPLITS_DIR.mkdir(parents=True, exist_ok=True)


# %% [Load patients + baseline pain]
df_outcomes = pd.read_excel(OUTCOMES_FILE)
t1_rows = df_outcomes[df_outcomes["Event Name"] == "T1"].copy()
t1_rows["pain_t1"] = pd.to_numeric(t1_rows["Pain Unpleasantness"], errors="coerce")
pid_pain_t1 = t1_rows.dropna(subset=["pain_t1"]).set_index("Patient number")["pain_t1"].to_dict()

files_by_pid = {}
for f in sorted(glob.glob(str(DATA_DIR / "CIPN3*.xlsx"))):
    m = re.search(r"CIPN3(\d{3})", os.path.basename(f))
    if m:
        files_by_pid[int(m.group(1))] = f

pids = sorted(set(files_by_pid) & set(pid_pain_t1))
pids_arr = np.array(pids)
pain_arr = np.array([pid_pain_t1[int(p)] for p in pids])
labels = {int(p): int(pid_pain_t1[int(p)] >= PAIN_THRESHOLD) for p in pids}
y_all = np.array([labels[int(p)] for p in pids])

print(f"Task A | threshold pain_t1 >= {PAIN_THRESHOLD:g}")
print(f"Total patients (EEG & pain_t1): {len(pids)} | "
      f"pos(high pain)={(y_all==1).sum()} neg={(y_all==0).sum()} "
      f"pos_rate={y_all.mean():.3f}")


# %% [Stratified 5-fold split on the binary baseline-pain label]
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
manifest = []
print("\nFold composition:")
for k, (tr_idx, va_idx) in enumerate(skf.split(pids_arr, y_all)):
    train_ids = np.array(sorted(pids_arr[tr_idx]))
    val_ids   = np.array(sorted(pids_arr[va_idx]))
    assert len(set(train_ids) & set(val_ids)) == 0, "train/val overlap!"

    np.save(SPLITS_DIR / f"fold_{k}_train.npy", train_ids)
    np.save(SPLITS_DIR / f"fold_{k}_val.npy",   val_ids)

    y_tr = np.array([labels[int(p)] for p in train_ids])
    y_va = np.array([labels[int(p)] for p in val_ids])
    print(f"  Fold {k}: n_tr={len(train_ids)} n_va={len(val_ids)} | "
          f"pos_tr={y_tr.mean():.3f} pos_va={y_va.mean():.3f}")
    manifest.append({
        "fold": int(k),
        "n_train": int(len(train_ids)),
        "n_val":   int(len(val_ids)),
        "train_pos_rate": float(y_tr.mean()),
        "val_pos_rate":   float(y_va.mean()),
    })

# %% [Save labels + manifest]
with open(SPLITS_DIR / "labels.json", "w") as f:
    json.dump({str(p): int(labels[p]) for p in pids}, f, indent=2)
with open(SPLITS_DIR / "manifest.json", "w") as f:
    json.dump({
        "task": "A_baseline_pain",
        "pain_threshold": PAIN_THRESHOLD,
        "label_rule": "pain_t1 >= threshold",
        "n_folds": N_FOLDS,
        "stratification": "binary baseline-pain label",
        "random_seed": RANDOM_SEED,
        "total_patients": len(pids),
        "pos_rate": float(y_all.mean()),
        "folds": manifest,
    }, f, indent=2)

print(f"\nSaved Task A splits to {SPLITS_DIR}  (SPLITS_TAG={SPLITS_TAG})")
