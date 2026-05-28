"""
Notebook B — Band selection on fold K's 85 train patients.
Mirrors the 3_*.ipynb methodology verbatim:
  - For each sheet (coherence, phase lag, abs bandpower):
      For each of INNER_CV_FOLDS outer folds (patient-grouped):
          For each non-empty band subset:
              GridSearchCV(LogReg, full hyperparam grid, inner CV) → score
          Subset with max score wins → its bands get +1 in tally

The 22 val patients are NEVER loaded by this notebook. A leakage guard
asserts no val IDs make it into the train slice.

The selection rule is selectable via SELECTION_MODE:
  - "abc_existing":     7-fold tally; bands with count >= BAND_SELECTION_THRESHOLD survive.
  - "top2_per_sheet":   7-fold tally; the TOP_K_PER_SHEET highest-tallied bands survive.
  - "3ref_per_fold":    NO voting. Single exhaustive subset search on the full 85;
                        the subset with the highest inner-CV bal_acc is chosen
                        directly (matches 3_REF on a single outer fold).

Run AFTER pipeline/A_make_splits.py. Outputs (per sheet, in fold_{K}/{mode}/):
  {sheet_label}_band_counts.csv
  {sheet_label}_selected.json
  {sheet_label}_fold_winners.json
"""

# In B_band_selection.py: SELECTION_MODE = "abc_existing"
# In C_train_eval.py:     SELECTION_MODE = "abc_existing"

# In B_band_selection.py: SELECTION_MODE = "top2_per_sheet"
# In C_train_eval.py:     SELECTION_MODE = "top2_per_sheet"



# %% [Imports + config]
import os, re, glob, json, itertools, warnings
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV

warnings.filterwarnings("ignore")  # mute saga/L1 ConvergenceWarning floods

PROJECT_DIR     = Path("/Users/leoqian/mdanderson/EEGOutcomePrediction")
DATA_DIR        = PROJECT_DIR / "processeddata"
SPLITS_DIR      = PROJECT_DIR / "pipeline" / "splits"

# --- Methodology knobs (easy to change) -----------------------------------
# FOLD_INDEX / SELECTION_MODE accept env-var overrides so the all-folds
# driver (E_run_all_folds.py) can sweep them without editing this file.
FOLD_INDEX               = int(os.environ.get("FOLD_INDEX", 0))
INNER_CV_FOLDS           = 7         # outer band-tally folds
BAND_GS_INNER_FOLDS      = 5         # GridSearchCV's nested inner CV
BAND_SELECTION_THRESHOLD = 5         # used when SELECTION_MODE == "abc_existing"
TOP_K_PER_SHEET          = 2         # used when SELECTION_MODE == "top2_per_sheet"
SELECTION_MODE           = os.environ.get("SELECTION_MODE", "top2_per_sheet")
RANDOM_SEED              = 42

OUT_DIR = (PROJECT_DIR / "pipeline" / "bands"
           / f"fold_{FOLD_INDEX}" / SELECTION_MODE)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Sheets to process (label -> sheet name in xlsx). Edit if you want to run
# fewer sheets at a time, or swap to non-Z_ variants.
SHEETS = {
    "coherence": "Z_FFT_Coherence",
    "phaselag":  "Z_FFT_PhaseLag_PLI",
    "power_uv1": "Z_FFT_abs_bandpower_uV2",
}

CANDIDATE_BANDS = ["Delta", "Theta", "Alpha", "Beta", "HighBeta",
                   "Alpha1", "Alpha2", "Beta1", "Beta2", "Beta3"]

# LR hyperparam grid (verbatim from 3_*.ipynb)
LR_SOLVERS   = ["liblinear", "saga"]
LR_PENALTIES = ["l1", "l2"]
LR_C_VALUES  = [1e-3, 1e-2, 1e-1, 1, 10, 100]

def _make_lr_param_grid():
    grid = []
    for solver in LR_SOLVERS:
        for penalty in LR_PENALTIES:
            entry = {
                "clf__solver":  [solver],
                "clf__penalty": [penalty],
                "clf__C":       LR_C_VALUES,
            }
            if penalty == "l1":
                entry["clf__warm_start"] = [True]
            grid.append(entry)
    return grid


# %% [Load fold's 85 train IDs + leakage guard]
train_ids = np.load(SPLITS_DIR / f"fold_{FOLD_INDEX}_train.npy")
val_ids   = np.load(SPLITS_DIR / f"fold_{FOLD_INDEX}_val.npy")
with open(SPLITS_DIR / "labels.json") as f:
    labels = {int(p): int(y) for p, y in json.load(f).items()}

VAL_ID_SET = frozenset(int(p) for p in val_ids)
def assert_no_val_leak(pids, where):
    leaked = VAL_ID_SET & set(int(p) for p in pids)
    if leaked:
        raise RuntimeError(f"LEAKAGE in {where}: val pids {sorted(leaked)} present.")

assert_no_val_leak(train_ids, "Step B init")
y_train = np.array([labels[int(p)] for p in train_ids])
groups_train = np.array([int(p) for p in train_ids])  # 1 patient = 1 sample
print(f"Fold {FOLD_INDEX}: {len(train_ids)} train (pos={(y_train==1).sum()} "
      f"neg={(y_train==0).sum()}) | {len(val_ids)} val (NOT loaded here)")


# %% [Patient -> file map (train only)]
files_by_pid = {}
for f in sorted(glob.glob(str(DATA_DIR / "CIPN3*.xlsx"))):
    m = re.search(r"CIPN3(\d{3})", os.path.basename(f))
    if m:
        files_by_pid[int(m.group(1))] = f

train_files = [files_by_pid[int(p)] for p in train_ids]


# %% [Per-band feature compression (verbatim 3_*.ipynb 5-vector)]
def feats_from_single_band_column(df_band):
    """[col_mean, col_std, global_mean, global_std, global_median]
    — col and global entries are duplicates for a 1-col slice, kept for
    fidelity with the 3_*.ipynb reference."""
    A = df_band.values.astype(float)
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    col_mean = A.mean(axis=0)[0]
    col_std  = A.std(axis=0)[0]
    return np.array([col_mean, col_std, A.mean(), A.std(), float(np.median(A))],
                     dtype=float)

def load_band_cache(xlsx_path, sheet_name, bands):
    df_sheet = pd.read_excel(xlsx_path, sheet_name=sheet_name, index_col=0)
    out = {}
    for b in bands:
        if b in df_sheet.columns:
            out[b] = feats_from_single_band_column(df_sheet[[b]])
    return out

def build_X(cache_list, subset):
    rows = []
    for d in cache_list:
        if any(b not in d for b in subset):
            return None
        rows.append(np.concatenate([d[b] for b in subset]))
    return np.vstack(rows)


# %% [Band-selection loop, per sheet]
sel_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(class_weight="balanced", max_iter=50000,
                                tol=1e-3, random_state=RANDOM_SEED)),
])
sel_param_grid = _make_lr_param_grid()

# Outer band-tally CV (one winner per fold).
outer_band_cv = StratifiedGroupKFold(n_splits=INNER_CV_FOLDS, shuffle=True,
                                       random_state=RANDOM_SEED)
# GridSearchCV's nested inner CV (matches 3_'s inner_n_splits=5).
gs_inner_cv = StratifiedGroupKFold(n_splits=BAND_GS_INNER_FOLDS, shuffle=True,
                                     random_state=1)

for sheet_label, sheet_name in SHEETS.items():
    print(f"\n========== Sheet: {sheet_label} ({sheet_name}) ==========")
    train_cache = [load_band_cache(f, sheet_name, CANDIDATE_BANDS)
                   for f in train_files]
    available = [b for b in CANDIDATE_BANDS
                 if all(b in d for d in train_cache)]
    if not available:
        print("  [WARN] no bands available across all train patients; skipping.")
        continue
    print(f"  Available bands: {available}")

    subsets = [c for r in range(1, len(available) + 1)
               for c in itertools.combinations(available, r)]
    print(f"  Subsets to try: {len(subsets)} × {INNER_CV_FOLDS} outer folds × "
          f"{BAND_GS_INNER_FOLDS}-fold inner CV")

    inner_cv_bal_acc = None
    best_lr_params   = None
    fold_winners: list = []
    cnt = Counter()

    if SELECTION_MODE == "3ref_per_fold":
        # 3_REF on a single outer fold: ONE subset search on the full 85,
        # NO inner band-tally folds, NO voting. Pick the (subset, LR-params)
        # pair with the highest inner-CV bal_acc directly.
        best_subset, best_score, best_params = None, -np.inf, None
        progress_every = max(1, len(subsets) // 10)
        for si, subset in enumerate(subsets, 1):
            X_tr = build_X(train_cache, subset)
            if X_tr is None:
                continue
            gs = GridSearchCV(sel_pipe, param_grid=sel_param_grid,
                                scoring="balanced_accuracy", cv=gs_inner_cv,
                                n_jobs=-1, refit=False)
            gs.fit(X_tr, y_train, groups=groups_train)
            if gs.best_score_ > best_score:
                best_score, best_subset = gs.best_score_, subset
                best_params = gs.best_params_
            if si % progress_every == 0:
                print(f"    progress {si}/{len(subsets)} | current best "
                      f"{best_subset} bal_acc={best_score:.3f}")
        chosen = list(best_subset) if best_subset else []
        inner_cv_bal_acc = float(best_score) if best_subset else None
        best_lr_params   = best_params
        rule_str = "3_REF best subset on full 85 (no voting)"
        print(f"\n  Best subset: {chosen} | inner-CV bal_acc={best_score:.3f}")
        print(f"  Best LR (from subset search): {best_lr_params}")

    else:
        # Existing 7-fold tally machinery (abc_existing + top2_per_sheet).
        for fold_i, (tr_idx, _) in enumerate(
            outer_band_cv.split(train_ids, y_train, groups=groups_train), 1
        ):
            cache_tr = [train_cache[i] for i in tr_idx]
            y_tr = y_train[tr_idx]
            groups_tr = groups_train[tr_idx]

            best_subset, best_score = None, -np.inf
            for subset in subsets:
                X_tr = build_X(cache_tr, subset)
                if X_tr is None:
                    continue
                gs = GridSearchCV(sel_pipe, param_grid=sel_param_grid,
                                    scoring="balanced_accuracy", cv=gs_inner_cv,
                                    n_jobs=-1, refit=False)
                gs.fit(X_tr, y_tr, groups=groups_tr)
                if gs.best_score_ > best_score:
                    best_score, best_subset = gs.best_score_, subset
            fold_winners.append(list(best_subset) if best_subset else [])
            print(f"  Fold {fold_i}/{INNER_CV_FOLDS}: winner={best_subset} "
                  f"score={best_score:.3f}")

        for w in fold_winners:
            cnt.update(w)

        if SELECTION_MODE == "abc_existing":
            chosen = [b for b in available if cnt[b] >= BAND_SELECTION_THRESHOLD]
            rule_str = f">= {BAND_SELECTION_THRESHOLD}/{INNER_CV_FOLDS}"
        elif SELECTION_MODE == "top2_per_sheet":
            # Stable ranking: most-tallied first; ties broken by canonical order.
            ranked = sorted(available, key=lambda b: (-cnt.get(b, 0), available.index(b)))
            chosen = ranked[:TOP_K_PER_SHEET]
            rule_str = f"top {TOP_K_PER_SHEET} per sheet"
        else:
            raise ValueError(f"Unknown SELECTION_MODE: {SELECTION_MODE}")
        print(f"\n  Frequencies: {dict(cnt)}")
        print(f"  Selected ({rule_str}): {chosen}")

    # Save outputs (band_counts and fold_winners only meaningful in voting modes).
    if SELECTION_MODE != "3ref_per_fold":
        rows = [{
            "sheet": sheet_label, "band": b,
            "count": int(cnt[b]), "n_folds": INNER_CV_FOLDS,
            "selected": b in chosen,
        } for b in available]
        pd.DataFrame(rows).to_csv(OUT_DIR / f"{sheet_label}_band_counts.csv",
                                    index=False)
        with open(OUT_DIR / f"{sheet_label}_fold_winners.json", "w") as f:
            json.dump(fold_winners, f, indent=2)

    with open(OUT_DIR / f"{sheet_label}_selected.json", "w") as f:
        json.dump({
            "sheet_label":      sheet_label,
            "sheet_name":       sheet_name,
            "bands":            chosen,
            "selection_mode":   SELECTION_MODE,
            "threshold":        BAND_SELECTION_THRESHOLD,
            "top_k_per_sheet":  TOP_K_PER_SHEET,
            "n_folds":          INNER_CV_FOLDS,
            "inner_cv_bal_acc": inner_cv_bal_acc,
            "best_lr_params":   best_lr_params,
        }, f, indent=2)

print(f"\nDone. Outputs in {OUT_DIR}")
