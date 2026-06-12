"""
Notebook A v3 — joint-stratified 5-fold splits.

v1 stratifies on outcome label only, which produced two folds (2 and 3)
whose val patients had pain_T1 distributions shifted up by ~0.7 NRS
relative to train. Because pain_T1 is the dominant predictor (SHAP),
that single-covariate drift is what cratered val accuracy on those folds.

v3 stratifies on the joint (outcome label × pain_T1 tercile), so every
fold's val set has approximately the same label balance AND
approximately the same low/mid/high baseline-pain composition.

Writes to:
  pipeline/splits_v3/fold_{k}_train.npy / _val.npy
  pipeline/splits_v3/labels.json
  pipeline/splits_v3/manifest.json
"""
import os, re, glob, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

PROJECT_DIR   = Path("/Users/leoqian/mdanderson/EEGOutcomePrediction")
DATA_DIR      = PROJECT_DIR / "processeddata"
OUTCOMES_FILE = DATA_DIR / "Randomization factors and Primary outcome.xlsx"
SPLITS_DIR    = PROJECT_DIR / "pipeline" / "splits_v3"

LABEL_THRESHOLD = -2.00
N_FOLDS         = 5
N_PAIN_BINS     = 3       # terciles
RANDOM_SEED     = 42

SPLITS_DIR.mkdir(parents=True, exist_ok=True)


# %% [Load patients + labels + pain_t1]
df_outcomes = pd.read_excel(OUTCOMES_FILE)
diff = (
    df_outcomes[df_outcomes["Event Name"].isin(["T1", "T2"])]
    .pivot_table(index="Patient number", columns="Event Name",
                 values="Pain Unpleasantness")
    .dropna()
)
diff["Diff"] = diff["T2"] - diff["T1"]

t1_rows = df_outcomes[df_outcomes["Event Name"] == "T1"].copy()
t1_rows["pain_t1"] = pd.to_numeric(t1_rows["Pain Unpleasantness"], errors="coerce")
pid_pain_t1 = t1_rows.set_index("Patient number")["pain_t1"].to_dict()

files_by_pid = {}
for f in sorted(glob.glob(str(DATA_DIR / "CIPN3*.xlsx"))):
    m = re.search(r"CIPN3(\d{3})", os.path.basename(f))
    if not m:
        continue
    pid = int(m.group(1))
    files_by_pid[pid] = f

pids = sorted(set(files_by_pid) & set(diff.index))
labels = (diff.loc[pids, "Diff"] >= LABEL_THRESHOLD).astype(int).to_dict()
pids_arr = np.array(pids)
y_all = np.array([labels[p] for p in pids])
pain_arr = np.array([pid_pain_t1.get(int(p), np.nan) for p in pids])

# tercile bins on pain_t1 (computed on the FULL cohort — splits-only use,
# does not leak into model training since pain_t1 is itself a feature)
qs = np.nanpercentile(pain_arr, [100/3, 200/3])
pain_bin = np.where(pain_arr <= qs[0], 0,
            np.where(pain_arr <= qs[1], 1, 2))
print(f"pain_t1 tercile cutoffs: ≤{qs[0]:.2f}, ≤{qs[1]:.2f}, else")
print(f"  bin counts: 0={(pain_bin==0).sum()} 1={(pain_bin==1).sum()} 2={(pain_bin==2).sum()}")

# Composite stratum: label × pain_bin → 6 strata
strata = y_all * 10 + pain_bin
print(f"Composite strata distribution:")
for s in sorted(set(strata)):
    lab = s // 10; pb = s % 10
    print(f"  label={lab} pain_bin={pb}: n={int((strata==s).sum())}")
print(f"\nTotal patients: {len(pids)} | "
      f"pos={(y_all==1).sum()} neg={(y_all==0).sum()}")


# %% [Joint-stratified 5-fold split]
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
manifest = []
print("\nFold composition under joint stratification:")
for k, (tr_idx, va_idx) in enumerate(skf.split(pids_arr, strata)):
    train_ids = np.array(sorted(pids_arr[tr_idx]))
    val_ids   = np.array(sorted(pids_arr[va_idx]))
    assert len(set(train_ids) & set(val_ids)) == 0, "train/val overlap!"

    np.save(SPLITS_DIR / f"fold_{k}_train.npy", train_ids)
    np.save(SPLITS_DIR / f"fold_{k}_val.npy",   val_ids)

    y_tr = np.array([labels[int(p)] for p in train_ids])
    y_va = np.array([labels[int(p)] for p in val_ids])
    p_tr = np.array([pid_pain_t1.get(int(p), np.nan) for p in train_ids])
    p_va = np.array([pid_pain_t1.get(int(p), np.nan) for p in val_ids])
    print(f"  Fold {k}: n_tr={len(train_ids)} n_va={len(val_ids)} | "
          f"pos_tr={y_tr.mean():.3f} pos_va={y_va.mean():.3f} | "
          f"painT1_tr={np.nanmean(p_tr):.2f} painT1_va={np.nanmean(p_va):.2f}")
    manifest.append({
        "fold": int(k),
        "n_train": int(len(train_ids)),
        "n_val":   int(len(val_ids)),
        "train_pos_rate": float(y_tr.mean()),
        "val_pos_rate":   float(y_va.mean()),
        "train_pain_t1_mean": float(np.nanmean(p_tr)),
        "val_pain_t1_mean":   float(np.nanmean(p_va)),
    })

# %% [Save labels + manifest]
with open(SPLITS_DIR / "labels.json", "w") as f:
    json.dump({str(p): int(labels[p]) for p in pids}, f, indent=2)
with open(SPLITS_DIR / "manifest.json", "w") as f:
    json.dump({
        "label_threshold": LABEL_THRESHOLD,
        "n_folds": N_FOLDS,
        "n_pain_bins": N_PAIN_BINS,
        "stratification": "label x pain_t1_tercile",
        "random_seed": RANDOM_SEED,
        "total_patients": len(pids),
        "folds": manifest,
    }, f, indent=2)

print(f"\nSaved v3 splits to {SPLITS_DIR}")
