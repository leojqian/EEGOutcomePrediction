"""
Notebook A — Create 5-fold stratified train/val splits over the 107 patients.
Saves train/val patient IDs per fold to disk so downstream notebooks (B, C)
can load the same patient lists.

Run ONCE (or whenever you want to re-create splits with a new seed). Outputs:
  pipeline/splits/fold_{k}_train.npy   (k = 0..N_FOLDS-1)
  pipeline/splits/fold_{k}_val.npy
  pipeline/splits/labels.json          (pid -> binary label)
  pipeline/splits/manifest.json        (split summary)
"""

# %% [Imports + config]
import os, re, glob, json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

PROJECT_DIR     = Path("/Users/leoqian/mdanderson/EEGOutcomePrediction")
DATA_DIR        = PROJECT_DIR / "processeddata"
OUTCOMES_FILE   = DATA_DIR / "Randomization factors and Primary outcome.xlsx"
SPLITS_DIR      = PROJECT_DIR / "pipeline" / "splits"

LABEL_THRESHOLD = -2.00      # Diff >= threshold => class 1 ("non-improver")
N_FOLDS         = 5
RANDOM_SEED     = 42

SPLITS_DIR.mkdir(parents=True, exist_ok=True)


# %% [Load patients + labels]
df_outcomes = pd.read_excel(OUTCOMES_FILE)
diff = (
    df_outcomes[df_outcomes["Event Name"].isin(["T1", "T2"])]
    .pivot_table(index="Patient number", columns="Event Name",
                 values="Pain Unpleasantness")
    .dropna()
)
diff["Diff"] = diff["T2"] - diff["T1"]

# Map patient_id -> file. Patient 124 has 2 files; we keep _NFB_DUL (last
# alphabetically) and silently drop _NFB. Print a notice so it stays visible.
files_by_pid = {}
for f in sorted(glob.glob(str(DATA_DIR / "CIPN3*.xlsx"))):
    m = re.search(r"CIPN3(\d{3})", os.path.basename(f))
    if not m:
        continue
    pid = int(m.group(1))
    if pid in files_by_pid:
        print(f"[NOTE] patient {pid} has multiple files; "
              f"keeping {os.path.basename(f)}, "
              f"dropping {os.path.basename(files_by_pid[pid])}")
    files_by_pid[pid] = f

pids = sorted(set(files_by_pid) & set(diff.index))
labels = (diff.loc[pids, "Diff"] >= LABEL_THRESHOLD).astype(int).to_dict()
y_all = np.array([labels[p] for p in pids])
print(f"\nTotal patients: {len(pids)} | "
      f"pos={(y_all==1).sum()} neg={(y_all==0).sum()}")


# %% [5-fold stratified split, save IDs per fold]
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
pids_arr = np.array(pids)

manifest = []
for k, (tr_idx, va_idx) in enumerate(skf.split(pids_arr, y_all)):
    train_ids = np.array(sorted(pids_arr[tr_idx]))
    val_ids   = np.array(sorted(pids_arr[va_idx]))
    assert len(set(train_ids) & set(val_ids)) == 0, "train/val overlap!"

    np.save(SPLITS_DIR / f"fold_{k}_train.npy", train_ids)
    np.save(SPLITS_DIR / f"fold_{k}_val.npy",   val_ids)

    y_tr = np.array([labels[int(p)] for p in train_ids])
    y_va = np.array([labels[int(p)] for p in val_ids])
    print(f"Fold {k}: train n={len(train_ids)} (pos={y_tr.sum()}, "
          f"neg={(y_tr==0).sum()}) | val n={len(val_ids)} "
          f"(pos={y_va.sum()}, neg={(y_va==0).sum()})")
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
        "label_threshold": LABEL_THRESHOLD,
        "n_folds": N_FOLDS,
        "random_seed": RANDOM_SEED,
        "total_patients": len(pids),
        "folds": manifest,
    }, f, indent=2)

print(f"\nSaved splits + labels + manifest to {SPLITS_DIR}")
