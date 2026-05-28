"""
Single-fold EEG classification pipeline (85 train / 22 held-out).

Run as a script, or open in Jupyter/VS Code — `# %%` markers act as cells.

Methodology (per meeting notes):
  1. Fixed stratified 85/22 split, saved to disk.
  2. Band selection on the 85 train patients only, via inner CV with
     exhaustive band-subset search (mirrors 3_*.ipynb). A band is kept
     if it appears in the winning subset of >= BAND_SELECTION_THRESHOLD
     of the inner folds.
  3. Per-selected-band summary features (mean/std by default; median /
     p90 / p95 available manually). Cap total features at MAX_FEATURES.
  4. LogReg + XGBoost (if installed), tuned via inner CV on the 85.
  5. One-shot evaluation on the 22 held-out patients.

Leakage guards: TEST_ID_SET is checked at every step that touches data.
"""

# %% [Imports + config]
import os, re, glob, json, pickle, itertools
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    roc_auc_score, average_precision_score, f1_score,
)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# ---- Paths (edit if your layout differs) ---------------------------------
PROJECT_DIR   = Path("/Users/leoqian/mdanderson/EEGOutcomePrediction")
DATA_DIR      = PROJECT_DIR / "processeddata"
OUTCOMES_FILE = DATA_DIR / "Randomization factors and Primary outcome.xlsx"
RESULTS_DIR   = PROJECT_DIR / "results" / "single_fold_v1"
SPLIT_DIR     = RESULTS_DIR / "split"
FOLD_DIR      = RESULTS_DIR / "fold_0"

# ---- Methodology knobs (easy to change) ----------------------------------
LABEL_THRESHOLD          = -2.00     # Diff >= threshold → class 1
TRAIN_SIZE               = 85
TEST_SIZE                = 22
RANDOM_SEED              = 42
INNER_CV_FOLDS           = 7         # outer band-tally folds (=> 7 winners to vote)
BAND_GS_INNER_FOLDS      = 5         # GridSearchCV's nested inner CV (matches 3_*.ipynb)
BAND_SELECTION_THRESHOLD = 5         # band must win in >= this many of INNER_CV_FOLDS
MAX_FEATURES             = 14
SUMMARY_FUNCS            = ["mean", "std"]   # add "median", "p90", "p95" manually

# LR hyperparameter grid (verbatim from 3_*.ipynb)
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

# Sheets (label → sheet name in the .xlsx). "power_uv1" = absolute bandpower in µV².
# Switch to "Z_FFT_*" variants if you prefer z-scored versions.
SHEETS = {
    "coherence": "FFT_Coherence",
    "phase_lag": "FFT_PhaseLag_PLI",
    "power_uv1": "FFT_abs_bandpower_uV2",
}

# Bands eligible for selection. Band-subset search runs over all non-empty
# combinations of those that exist in the sheet.
CANDIDATE_BANDS = ["Delta", "Theta", "Alpha", "Beta", "HighBeta",
                   "Alpha1", "Alpha2", "Beta1", "Beta2", "Beta3"]

for d in (SPLIT_DIR, FOLD_DIR):
    d.mkdir(parents=True, exist_ok=True)


# %% [Step 0] Load patient IDs and labels
# PLACEHOLDER: replace this block if your label/file convention differs.
df_outcomes = pd.read_excel(OUTCOMES_FILE)
diff = (
    df_outcomes[df_outcomes["Event Name"].isin(["T1", "T2"])]
    .pivot_table(index="Patient number", columns="Event Name",
                 values="Pain Unpleasantness")
    .dropna()
)
diff["Diff"] = diff["T2"] - diff["T1"]

# Map patient_id → xlsx path
files_by_pid = {}
for f in sorted(glob.glob(str(DATA_DIR / "CIPN3*.xlsx"))):
    m = re.search(r"CIPN3(\d{3})", os.path.basename(f))
    if m:
        files_by_pid[int(m.group(1))] = f

# Patients with both an EEG file and a Diff outcome
pids = sorted(set(files_by_pid) & set(diff.index))
labels = (diff.loc[pids, "Diff"] >= LABEL_THRESHOLD).astype(int).to_dict()
y_all = np.array([labels[p] for p in pids])
print(f"Patients with EEG + label: {len(pids)} | "
      f"pos={(y_all==1).sum()} neg={(y_all==0).sum()}")


# %% [Step 1] Fixed 85/22 stratified split
expected_n = TRAIN_SIZE + TEST_SIZE
if len(pids) != expected_n:
    print(f"[NOTE] Found {len(pids)} patients, expected {expected_n}. "
          f"Adjust TRAIN_SIZE/TEST_SIZE or filter your dataset before splitting.")

train_ids, test_ids = train_test_split(
    pids, test_size=TEST_SIZE, stratify=y_all, random_state=RANDOM_SEED,
)
train_ids = np.array(sorted(train_ids))
test_ids  = np.array(sorted(test_ids))

assert len(set(train_ids) & set(test_ids)) == 0, "Train/test overlap!"
assert len(train_ids) + len(test_ids) == len(pids), "Lost patients in split."

np.save(SPLIT_DIR / "train_ids.npy", train_ids)
np.save(SPLIT_DIR / "test_ids.npy",  test_ids)
print(f"Saved split: train={len(train_ids)} test={len(test_ids)}")

# Leakage guard used by every later step
TEST_ID_SET = frozenset(int(p) for p in test_ids)
def assert_no_test_leak(pid_iter, where):
    leaked = TEST_ID_SET & set(int(p) for p in pid_iter)
    if leaked:
        raise RuntimeError(f"LEAKAGE in {where}: test pids {sorted(leaked)} present.")


# %% [Step 2] Band selection (training-only inner CV)
# For each sheet, run INNER_CV_FOLDS-fold stratified CV on the 85 train patients.
# In each inner fold: brute-force search over all non-empty band subsets
# (matching 3_*.ipynb), pick the subset with the best validation balanced
# accuracy. Tally band frequencies → keep bands that win in
# >= BAND_SELECTION_THRESHOLD folds.

def _band_summary_vec(col_values):
    """5-vector per band, verbatim from 3_*.ipynb:
       [col_mean, col_std, global_mean, global_std, global_median].
    For a 1-col slice, col_mean == global_mean and col_std == global_std (2 entries
    are duplicates), but we keep all 5 for fidelity with the reference notebooks."""
    A = np.asarray(col_values, dtype=float).reshape(-1, 1)
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    col_mean = A.mean(axis=0)[0]
    col_std  = A.std(axis=0)[0]
    return np.array(
        [col_mean, col_std, A.mean(), A.std(), float(np.median(A))],
        dtype=float,
    )

def _load_band_cache(pid, sheet):
    """Return {band_name: feature_vec} for one patient/sheet."""
    df_sheet = pd.read_excel(files_by_pid[pid], sheet_name=sheet, index_col=0)
    return {b: _band_summary_vec(df_sheet[b])
            for b in CANDIDATE_BANDS if b in df_sheet.columns}

def _build_X(cache_list, subset):
    """Stack per-patient band features for a given band subset; None if any band is missing."""
    rows = []
    for d in cache_list:
        if any(b not in d for b in subset):
            return None
        rows.append(np.concatenate([d[b] for b in subset]))
    return np.vstack(rows)

assert_no_test_leak(train_ids, "Step 2 band selection")
y_train = np.array([labels[p] for p in train_ids])

selected_bands = {}
band_count_records = []

# Selection model: LogReg with full hyperparam grid (matches 3_*.ipynb).
# GridSearchCV scores each subset by best mean inner-CV balanced accuracy.
sel_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(class_weight="balanced", max_iter=50000,
                                tol=1e-3, random_state=RANDOM_SEED)),
])
sel_param_grid = _make_lr_param_grid()

# Band-tally outer loop: 7 different train-patient subsets vote on bands.
outer_band_cv = StratifiedKFold(n_splits=INNER_CV_FOLDS, shuffle=True,
                                  random_state=RANDOM_SEED)
# Inner CV used by GridSearchCV when scoring each subset.
gs_inner_cv = StratifiedKFold(n_splits=BAND_GS_INNER_FOLDS, shuffle=True,
                               random_state=1)

for sheet_label, sheet_name in SHEETS.items():
    print(f"\n--- Band selection: {sheet_label} ({sheet_name}) ---")
    train_cache = [_load_band_cache(p, sheet_name) for p in train_ids]
    available = [b for b in CANDIDATE_BANDS
                 if all(b in d for d in train_cache)]
    if not available:
        print("  [WARN] no bands present across all train patients; skipping.")
        selected_bands[sheet_label] = []
        continue
    print(f"  Available bands: {available}")

    subsets = [c for r in range(1, len(available) + 1)
               for c in itertools.combinations(available, r)]
    print(f"  Subsets to try: {len(subsets)} (× {INNER_CV_FOLDS} outer folds × "
          f"{BAND_GS_INNER_FOLDS}-fold GridSearchCV inner CV)")

    fold_winners = []
    # tr_idx is the outer-train slice for one band-tally fold; va_idx is unused —
    # selection score comes from GridSearchCV's own nested inner CV (mirrors 3_).
    for fold_i, (tr_idx, _) in enumerate(outer_band_cv.split(train_ids, y_train), 1):
        cache_tr = [train_cache[i] for i in tr_idx]
        y_tr = y_train[tr_idx]

        best_subset, best_score = None, -np.inf
        for subset in subsets:
            X_tr = _build_X(cache_tr, subset)
            if X_tr is None:
                continue
            gs = GridSearchCV(sel_pipe, param_grid=sel_param_grid,
                                scoring="balanced_accuracy", cv=gs_inner_cv,
                                n_jobs=-1, refit=False)
            gs.fit(X_tr, y_tr)
            if gs.best_score_ > best_score:
                best_score, best_subset = gs.best_score_, subset
        fold_winners.append(best_subset)
        print(f"  Fold {fold_i}: winner={best_subset} gs.best_score_={best_score:.3f}")

    cnt = Counter()
    for w in fold_winners:
        cnt.update(w)
    chosen = [b for b in available if cnt[b] >= BAND_SELECTION_THRESHOLD]
    selected_bands[sheet_label] = chosen
    print(f"  Frequencies: {dict(cnt)}")
    print(f"  Selected (>= {BAND_SELECTION_THRESHOLD}/{INNER_CV_FOLDS}): {chosen}")

    for b in available:
        band_count_records.append({
            "sheet": sheet_label, "band": b,
            "count": cnt[b], "n_folds": INNER_CV_FOLDS,
            "selected": b in chosen,
        })

pd.DataFrame(band_count_records).to_csv(FOLD_DIR / "band_counts.csv", index=False)
with open(FOLD_DIR / "selected_bands.json", "w") as f:
    json.dump(selected_bands, f, indent=2)
print(f"\nSelected bands saved → {FOLD_DIR / 'selected_bands.json'}")


# %% [Step 3] Feature engineering — per-band summary features
# Total features = sum_over_sheets(len(selected_bands) * len(SUMMARY_FUNCS)).
# Capped at MAX_FEATURES. Tighten BAND_SELECTION_THRESHOLD or shorten
# SUMMARY_FUNCS if the assertion fires.

SUMMARY_FUNC_MAP = {
    "mean":   np.nanmean,
    "std":    np.nanstd,
    "median": np.nanmedian,
    "p90":    lambda a: np.nanpercentile(a, 90),
    "p95":    lambda a: np.nanpercentile(a, 95),
}

def extract_features(pid, selected_bands, summary_funcs):
    """Return (vec, names) of summary features for one patient, given selected bands."""
    vals, names = [], []
    for sheet_label, sheet_name in SHEETS.items():
        bands = selected_bands.get(sheet_label, [])
        if not bands:
            continue
        df_sheet = pd.read_excel(files_by_pid[pid], sheet_name=sheet_name,
                                  index_col=0)
        for b in bands:
            if b not in df_sheet.columns:
                continue
            col = df_sheet[b].values.astype(float)
            for fn_name in summary_funcs:
                vals.append(SUMMARY_FUNC_MAP[fn_name](col))
                names.append(f"{sheet_label}__{b}__{fn_name}")
    return np.array(vals, dtype=float), names

assert_no_test_leak(train_ids, "Step 3 train feature extraction")

X_train_rows, feature_names = [], None
for p in train_ids:
    vec, names = extract_features(p, selected_bands, SUMMARY_FUNCS)
    X_train_rows.append(vec)
    feature_names = names
X_train = np.nan_to_num(np.vstack(X_train_rows))

print(f"Train feature matrix: {X_train.shape} | features: {len(feature_names)}")
assert len(feature_names) <= MAX_FEATURES, (
    f"Too many features ({len(feature_names)} > {MAX_FEATURES}). "
    f"Tighten BAND_SELECTION_THRESHOLD or shorten SUMMARY_FUNCS."
)
pd.DataFrame({"feature": feature_names}).to_csv(FOLD_DIR / "feature_list.csv",
                                                  index=False)


# %% [Step 4] Model training on the 85 train patients
# Inner CV grid search; pick the best by balanced accuracy. Models stay simple.

assert_no_test_leak(train_ids, "Step 4 model training")
inner_cv_models = StratifiedKFold(n_splits=INNER_CV_FOLDS, shuffle=True,
                                    random_state=RANDOM_SEED)

models = {
    "logreg": (
        Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced",
                                        max_iter=50000, tol=1e-3,
                                        random_state=RANDOM_SEED)),
        ]),
        _make_lr_param_grid(),  # full L1/L2 × {liblinear,saga} × C grid (3_*.ipynb)
    ),
}
if HAS_XGB:
    models["xgboost"] = (
        XGBClassifier(eval_metric="logloss", random_state=RANDOM_SEED,
                       n_jobs=1, tree_method="hist"),
        {
            "n_estimators":  [100, 300],
            "max_depth":     [2, 3, 5],
            "learning_rate": [0.05, 0.1],
        },
    )
else:
    print("[NOTE] xgboost not installed — skipping. `pip install xgboost` to enable.")

train_results = []
fitted = {}
for name, (est, grid) in models.items():
    gs = GridSearchCV(est, grid, scoring="balanced_accuracy",
                       cv=inner_cv_models, n_jobs=-1, refit=True)
    gs.fit(X_train, y_train)
    fitted[name] = gs.best_estimator_
    print(f"{name}: CV bal_acc={gs.best_score_:.3f} | params={gs.best_params_}")
    train_results.append({
        "model": name,
        "cv_bal_acc": gs.best_score_,
        "params": json.dumps(gs.best_params_),
    })

pd.DataFrame(train_results).to_csv(FOLD_DIR / "train_results.csv", index=False)

best_name = max(train_results, key=lambda r: r["cv_bal_acc"])["model"]
with open(FOLD_DIR / "model.pkl", "wb") as f:
    pickle.dump({
        "name": best_name,
        "model": fitted[best_name],
        "feature_names": feature_names,
        "selected_bands": selected_bands,
        "summary_funcs": SUMMARY_FUNCS,
    }, f)
print(f"Best model: {best_name} → saved to {FOLD_DIR / 'model.pkl'}")


# %% [Step 5] One-shot evaluation on 22 held-out patients
# This is the FIRST time we touch the test patients. Bands, features, and
# model are already locked in.

y_test = np.array([labels[p] for p in test_ids])

X_test_rows = []
for p in test_ids:
    vec, names = extract_features(p, selected_bands, SUMMARY_FUNCS)
    assert names == feature_names, "Feature schema mismatch between train and test."
    X_test_rows.append(vec)
X_test = np.nan_to_num(np.vstack(X_test_rows))

best = fitted[best_name]

# Tune decision threshold on TRAIN probabilities (matches 3_*.ipynb).
p_train = best.predict_proba(X_train)[:, 1]
ths = np.linspace(0.05, 0.95, 181)
best_th, best_train_ba = 0.5, -np.inf
for t in ths:
    yhat = (p_train >= t).astype(int)
    ba = balanced_accuracy_score(y_train, yhat)
    if ba > best_train_ba:
        best_train_ba, best_th = ba, t
print(f"Threshold tuned on train: {best_th:.3f} "
      f"(train_bal_acc_at_threshold={best_train_ba:.3f})")

proba = best.predict_proba(X_test)[:, 1]
pred = (proba >= best_th).astype(int)

metrics = {
    "model": best_name,
    "threshold": float(best_th),
    "train_bal_acc_at_threshold": float(best_train_ba),
    "n_test": int(len(y_test)),
    "pos_rate": float(y_test.mean()),
    "accuracy": float(accuracy_score(y_test, pred)),
    "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
    "auc": float(roc_auc_score(y_test, proba)),
    "average_precision": float(average_precision_score(y_test, proba)),
    "f1": float(f1_score(y_test, pred, zero_division=0)),
}

pd.DataFrame({
    "pid": test_ids, "y_true": y_test, "y_pred": pred, "proba": proba,
}).to_csv(FOLD_DIR / "test_predictions.csv", index=False)

with open(FOLD_DIR / "test_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\n=== Held-out test results ===")
for k, v in metrics.items():
    print(f"  {k}: {v}")

if metrics["accuracy"] >= 0.67:
    print("\n[GO] accuracy >= 67% → can extend to multi-fold outer CV.")
else:
    print("\n[STAY] accuracy < 67% → keep refining this single 85/22 split "
          "(adjust BAND_SELECTION_THRESHOLD, SUMMARY_FUNCS, or model grid).")
