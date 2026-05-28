"""
Notebook D — Single-fold 3_REF replication.

Replicates the 3_REF_Z_FFT_*.ipynb methodology on FOLD_INDEX=0 only:
  * Per-sheet exhaustive band-subset search (NO voting, NO consensus tally).
  * Per-band 5-vector compression: [col_mean, col_std, g_mean, g_std, g_med]
    (verbatim from 3_REF.feats_from_single_band_column).
  * GridSearchCV(LR, full hyperparam grid, inner CV) per subset; pick the
    subset with the highest inner-CV balanced_accuracy directly.
  * Concatenate per-sheet selections → final feature matrix.
  * Re-tune LR and XGB hyperparameters on the concatenated features.
  * Threshold tuned on TRAIN probabilities (181 thresholds, max train bal_acc).
  * Single-shot eval on the 22 held-out patients.

Differences vs A/B/C:
  - No voting / no >= K threshold rule. Per-fold best subset is used directly.
  - No MAX_FEATURES cap. A 6-band selection → 30 features.
  - 5-vec features (not [mean, std]).

Single-fold scope: this script does NOT loop the 5 outer folds, and does NOT
average across the 5×2=10 outer folds the original 3_REF uses.

Run AFTER pipeline/A_make_splits.py.
Output: pipeline/results/fold_{FOLD_INDEX}/3ref/
"""

# %% [Imports + config]
from __future__ import annotations

import glob, itertools, json, os, pickle, re, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

PROJECT_DIR = Path("/Users/leoqian/mdanderson/EEGOutcomePrediction")
DATA_DIR    = PROJECT_DIR / "processeddata"
SPLITS_DIR  = PROJECT_DIR / "pipeline" / "splits"

# --- Methodology knobs ---------------------------------------------------
FOLD_INDEX      = 0
INNER_N_SPLITS  = 5             # 3_REF inner_n_splits=5
RANDOM_SEED     = 42

# Subset-search runtime cap. None = full 1023 search per sheet (matches 3_REF).
# Set to e.g. 5 to limit to subsets of size <= 5 if runtime is too high.
MAX_SUBSET_SIZE = None

SHEETS = {
    "coherence": "Z_FFT_Coherence",
    "phaselag":  "Z_FFT_PhaseLag_PLI",
    "power_uv1": "Z_FFT_abs_bandpower_uV2",
}
CANDIDATE_BANDS = [
    "Delta", "Theta", "Alpha", "Beta", "HighBeta",
    "Alpha1", "Alpha2", "Beta1", "Beta2", "Beta3",
]

# LR hyperparam grid (verbatim 3_REF).
LR_SOLVERS   = ["liblinear", "saga"]
LR_PENALTIES = ["l1", "l2"]
LR_C_VALUES  = [1e-3, 1e-2, 1e-1, 1, 10, 100]

def _make_lr_grid():
    grid = []
    for solver in LR_SOLVERS:
        for penalty in LR_PENALTIES:
            entry = {"clf__solver":  [solver],
                     "clf__penalty": [penalty],
                     "clf__C":       LR_C_VALUES}
            if penalty == "l1":
                entry["clf__warm_start"] = [True]
            grid.append(entry)
    return grid

# XGBoost grid — wider search (matches C_train_eval.py's expanded grid).
XGB_PARAM_GRID = {
    "n_estimators":     [50, 100, 200, 300],
    "max_depth":        [2, 3, 4, 5, 6, 7],
    "learning_rate":    [0.005, 0.01, 0.03, 0.05, 0.1, 0.2],
    "subsample":        [0.7, 0.85, 1.0],
    "colsample_bytree": [0.7, 0.85, 1.0],
    "reg_lambda":       [0.1, 1, 5, 10],
}

OUT_DIR = PROJECT_DIR / "pipeline" / "results" / f"fold_{FOLD_INDEX}" / "3ref"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# %% [Helpers]
def patient_file_map() -> dict[int, str]:
    files: dict[int, str] = {}
    for f in sorted(glob.glob(str(DATA_DIR / "CIPN3*.xlsx"))):
        m = re.search(r"CIPN3(\d{3})", os.path.basename(f))
        if m:
            files[int(m.group(1))] = f
    return files


def load_split(fold_index: int):
    train_ids = np.load(SPLITS_DIR / f"fold_{fold_index}_train.npy")
    val_ids   = np.load(SPLITS_DIR / f"fold_{fold_index}_val.npy")
    with open(SPLITS_DIR / "labels.json") as f:
        labels = {int(p): int(y) for p, y in json.load(f).items()}
    return train_ids, val_ids, labels


# Verbatim 3_REF feats_from_single_band_column.
def feats_from_single_band_column(df_band: pd.DataFrame) -> np.ndarray:
    """[col_mean, col_std, g_mean, g_std, g_med] — matches 3_REF exactly.
    The col_* and g_* entries duplicate for a 1-col slice; kept for fidelity."""
    A = df_band.values.astype(float)
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    col_mean = A.mean(axis=0)[0]
    col_std  = A.std(axis=0)[0]
    return np.array([col_mean, col_std, A.mean(), A.std(),
                     float(np.median(A))], dtype=float)


def load_band_cache(xlsx_path: str, sheet_name: str,
                    bands: list[str]) -> dict[str, np.ndarray]:
    """Per-patient (band -> 5-vec) cache. Matches 3_REF.load_bandwise_cache_for_file."""
    df_sheet = pd.read_excel(xlsx_path, sheet_name=sheet_name, index_col=0)
    return {b: feats_from_single_band_column(df_sheet[[b]])
            for b in bands if b in df_sheet.columns}


def build_X(cache_list: list[dict],
            subset: tuple[str, ...]) -> np.ndarray | None:
    """Stack per-patient 5-vecs across the bands in `subset`. Matches 3_REF.build_X_from_subset."""
    rows = []
    for d in cache_list:
        if any(b not in d for b in subset):
            return None
        rows.append(np.concatenate([d[b] for b in subset]))
    return np.vstack(rows)


def all_subsets(bands: list[str], max_size: int | None) -> list[tuple[str, ...]]:
    """All non-empty subsets, optionally capped at `max_size`.
    With max_size=None this matches 3_REF.all_nonempty_subsets exactly (1023 for 10 bands)."""
    cap = len(bands) if max_size is None else min(len(bands), max_size)
    out = []
    for r in range(1, cap + 1):
        out.extend(itertools.combinations(bands, r))
    return out


# %% [Load split + leakage guard]
files_by_pid = patient_file_map()
train_ids, val_ids, labels = load_split(FOLD_INDEX)
y_train = np.array([labels[int(p)] for p in train_ids])
y_val   = np.array([labels[int(p)] for p in val_ids])
groups_train = np.asarray(train_ids, dtype=int)

VAL_ID_SET = frozenset(int(p) for p in val_ids)
def assert_no_val_leak(pids, where):
    leaked = VAL_ID_SET & set(int(p) for p in pids)
    if leaked:
        raise RuntimeError(f"LEAKAGE in {where}: val pids {sorted(leaked)} present.")

assert_no_val_leak(train_ids, "Step D init")
print(f"Fold {FOLD_INDEX}: {len(train_ids)} train | "
      f"{len(val_ids)} held-out (NOT loaded yet)")
print(f"MAX_SUBSET_SIZE={MAX_SUBSET_SIZE} "
      f"({'full 3_REF search' if MAX_SUBSET_SIZE is None else 'capped'})")

train_files = [files_by_pid[int(p)] for p in train_ids]


# %% [Phase 1 — per-sheet subset search (3_REF's exhaustive search, ONE outer fold)]
sel_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    LogisticRegression(class_weight="balanced", max_iter=50000,
                                  tol=1e-3, random_state=RANDOM_SEED)),
])
sel_grid = _make_lr_grid()

# Inner CV — matches 3_REF's inner_n_splits=5 GroupKFold(random_state=1).
inner_cv = StratifiedGroupKFold(n_splits=INNER_N_SPLITS, shuffle=True,
                                 random_state=1)

per_sheet: dict = {}
total_subsets_evaluated = 0

for sheet_label, sheet_name in SHEETS.items():
    print(f"\n========== Sheet: {sheet_label} ({sheet_name}) ==========")
    train_cache = [load_band_cache(f, sheet_name, CANDIDATE_BANDS)
                   for f in train_files]
    available = [b for b in CANDIDATE_BANDS
                 if all(b in d for d in train_cache)]
    if not available:
        print("  [WARN] no bands available across all train patients; skipping.")
        per_sheet[sheet_label] = {"sheet_name": sheet_name, "best_subset": [],
                                   "best_score": None, "best_lr_params": None,
                                   "n_subsets_evaluated": 0}
        continue
    print(f"  Available bands: {available}")

    subsets = all_subsets(available, MAX_SUBSET_SIZE)
    print(f"  Subsets to try: {len(subsets)}")
    total_subsets_evaluated += len(subsets)

    # 3_REF: pick the (subset, LR-params) pair with the highest inner-CV
    # balanced_accuracy. No voting; no consensus.
    best_subset, best_score, best_params = None, -np.inf, None
    progress_every = max(1, len(subsets) // 10)
    for si, subset in enumerate(subsets, 1):
        X_tr = build_X(train_cache, subset)
        if X_tr is None:
            continue
        gs = GridSearchCV(sel_pipe, param_grid=sel_grid,
                          scoring="balanced_accuracy", cv=inner_cv,
                          n_jobs=-1, refit=False)
        gs.fit(X_tr, y_train, groups=groups_train)
        if gs.best_score_ > best_score:
            best_score, best_subset, best_params = (gs.best_score_, subset,
                                                     gs.best_params_)
        if si % progress_every == 0:
            print(f"    progress {si}/{len(subsets)} | current best "
                  f"{best_subset} bal_acc={best_score:.3f}")

    print(f"  Best subset: {best_subset} | inner-CV bal_acc={best_score:.3f}")
    print(f"  Best LR (from subset search): {best_params}")
    per_sheet[sheet_label] = {
        "sheet_name":          sheet_name,
        "best_subset":         list(best_subset) if best_subset else [],
        "best_score":          float(best_score) if best_subset else None,
        "best_lr_params":      best_params,
        "n_subsets_evaluated": len(subsets),
    }


# %% [Concatenate selected bands across sheets → final feature matrix]
def build_concat_X(pids, files_by_pid, per_sheet):
    """5-vec features over every (sheet, band) selection, concatenated."""
    feat_names = []
    for label, info in per_sheet.items():
        for band in info["best_subset"]:
            for stat in ["col_mean", "col_std", "g_mean", "g_std", "g_med"]:
                feat_names.append(f"{label}__{band}__{stat}")
    X_rows = []
    for pid in pids:
        row = []
        for label, info in per_sheet.items():
            if not info["best_subset"]:
                continue
            cache = load_band_cache(files_by_pid[int(pid)],
                                     info["sheet_name"],
                                     info["best_subset"])
            for band in info["best_subset"]:
                row.append(cache[band])
        X_rows.append(np.concatenate(row))
    return np.nan_to_num(np.vstack(X_rows)), feat_names


print("\n[Phase 2] Building concatenated feature matrix on the 85 train patients…")
X_train, feature_names = build_concat_X(train_ids, files_by_pid, per_sheet)
print(f"  X_train: {X_train.shape} | features: {len(feature_names)}")
if not feature_names:
    raise RuntimeError("No bands selected in any sheet — cannot build features.")


# %% [Phase 2 — final LR + XGB hyperparameter tuning on concat features]
final_inner_cv = StratifiedGroupKFold(n_splits=INNER_N_SPLITS, shuffle=True,
                                       random_state=RANDOM_SEED)

def fit_lr(X, y, groups, cv):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(class_weight="balanced", max_iter=50000,
                                      tol=1e-3, random_state=RANDOM_SEED)),
    ])
    gs = GridSearchCV(pipe, _make_lr_grid(), scoring="balanced_accuracy",
                       cv=cv, n_jobs=-1, refit=True)
    gs.fit(X, y, groups=groups)
    return gs.best_estimator_, gs.best_params_, float(gs.best_score_)

def fit_xgb(X, y, groups, cv):
    base = XGBClassifier(eval_metric="logloss", random_state=RANDOM_SEED,
                          n_jobs=1, tree_method="hist")
    gs = GridSearchCV(base, XGB_PARAM_GRID, scoring="balanced_accuracy",
                       cv=cv, n_jobs=-1, refit=True)
    gs.fit(X, y, groups=groups)
    return gs.best_estimator_, gs.best_params_, float(gs.best_score_)


print("\n[Phase 2] Tuning LR on concat features…")
lr_model, lr_params, lr_cv = fit_lr(X_train, y_train, groups_train,
                                     final_inner_cv)
print(f"  LR  CV bal_acc={lr_cv:.3f} | params={lr_params}")

if HAS_XGB:
    print("[Phase 2] Tuning XGB on concat features…")
    xgb_model, xgb_params, xgb_cv = fit_xgb(X_train, y_train, groups_train,
                                              final_inner_cv)
    print(f"  XGB CV bal_acc={xgb_cv:.3f} | params={xgb_params}")
else:
    xgb_model, xgb_params, xgb_cv = None, None, None
    print("[NOTE] xgboost not installed; reporting LR only.")

# Pick the better-performing model on inner CV.
candidates = [("logreg", lr_model, lr_params, lr_cv)]
if HAS_XGB:
    candidates.append(("xgboost", xgb_model, xgb_params, xgb_cv))
best_name, best_model, best_params_final, best_cv = max(candidates,
                                                          key=lambda r: r[3])
print(f"\nBest model by inner CV: {best_name} (bal_acc={best_cv:.3f})")


# %% [Threshold tuning on TRAIN — verbatim 3_REF loop]
p_train = best_model.predict_proba(X_train)[:, 1]
ths = np.linspace(0.05, 0.95, 181)
best_th, best_train_ba = 0.5, -np.inf
for t in ths:
    ba = balanced_accuracy_score(y_train, (p_train >= t).astype(int))
    if ba > best_train_ba:
        best_train_ba, best_th = ba, t
print(f"\nThreshold tuned on train: {best_th:.3f} "
      f"(train bal_acc={best_train_ba:.3f})")


# %% [Phase 3 — single-shot held-out evaluation; first + only val load]
print("\n[Phase 3] Loading the 22 held-out patients for the first time…")
X_val, val_feature_names = build_concat_X(val_ids, files_by_pid, per_sheet)
assert val_feature_names == feature_names, "Feature schema drift train→val."

p_val      = best_model.predict_proba(X_val)[:, 1]
y_pred_val = (p_val >= best_th).astype(int)

train_pred = (p_train >= best_th).astype(int)
train_acc  = float(accuracy_score(y_train, train_pred))
train_ba   = float(balanced_accuracy_score(y_train, train_pred))

cm_val   = confusion_matrix(y_val,   y_pred_val, labels=[0, 1]).tolist()
cm_train = confusion_matrix(y_train, train_pred, labels=[0, 1]).tolist()

metrics = {
    "fold":               int(FOLD_INDEX),
    "n_train":            int(len(train_ids)),
    "n_val":              int(len(val_ids)),
    "max_subset_size":    MAX_SUBSET_SIZE,
    "total_subsets_evaluated": total_subsets_evaluated,
    "selected_per_sheet": {k: v["best_subset"] for k, v in per_sheet.items()},
    "subset_search":      {
        k: {"n_subsets_evaluated": v["n_subsets_evaluated"],
            "inner_cv_bal_acc":    v["best_score"],
            "best_lr_params":      v["best_lr_params"]}
        for k, v in per_sheet.items()
    },
    "feature_names":      feature_names,
    "n_features":         int(len(feature_names)),
    "lr_best_params":     lr_params,
    "lr_cv_bal_acc":      lr_cv,
    "xgb_best_params":    xgb_params,
    "xgb_cv_bal_acc":     xgb_cv,
    "best_model":         best_name,
    "best_cv_bal_acc":    float(best_cv),
    "best_threshold":     float(best_th),
    "train_accuracy":     train_acc,
    "train_balanced_accuracy": train_ba,
    "val_accuracy":       float(accuracy_score(y_val, y_pred_val)),
    "val_balanced_accuracy":   float(balanced_accuracy_score(y_val, y_pred_val)),
    "val_auc":            float(roc_auc_score(y_val, p_val)),
    "val_average_precision":   float(average_precision_score(y_val, p_val)),
    "val_f1":             float(f1_score(y_val, y_pred_val, zero_division=0)),
    "confusion_matrix_val":   cm_val,
    "confusion_matrix_train": cm_train,
}


# %% [Save outputs]
pd.DataFrame({"feature": feature_names}).to_csv(
    OUT_DIR / "feature_list.csv", index=False,
)
pd.DataFrame({
    "pid": val_ids, "y_true": y_val, "y_pred": y_pred_val, "proba": p_val,
}).to_csv(OUT_DIR / "val_predictions.csv", index=False)
with open(OUT_DIR / "test_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
with open(OUT_DIR / "model.pkl", "wb") as f:
    pickle.dump({
        "best_model_name":    best_name,
        "best_model":         best_model,
        "lr_model":           lr_model,
        "lr_params":          lr_params,
        "xgb_model":          xgb_model,
        "xgb_params":         xgb_params,
        "feature_names":      feature_names,
        "selected_per_sheet": per_sheet,
        "threshold":          best_th,
        "fold_index":         FOLD_INDEX,
    }, f)


# %% [Report]
print(f"\n=== Run summary (fold {FOLD_INDEX}, 3_REF-style single fold) ===")
print(f"  n_train={metrics['n_train']}, n_held_out={metrics['n_val']}")
print(f"  candidate subsets evaluated (across all sheets): "
      f"{metrics['total_subsets_evaluated']}")
print(f"  selected per sheet: {metrics['selected_per_sheet']}")
print(f"  feature names ({metrics['n_features']}): {feature_names}")
print(f"  LR best params:  {lr_params} (CV bal_acc={lr_cv:.3f})")
if HAS_XGB:
    print(f"  XGB best params: {xgb_params} (CV bal_acc={xgb_cv:.3f})")
print(f"  Picked model:    {best_name} (CV bal_acc={best_cv:.3f})")
print(f"  Threshold tuned on train: {best_th:.3f}")
print(f"  TRAIN: acc={train_acc:.3f} | bal_acc={train_ba:.3f}")
print(f"  VAL:   acc={metrics['val_accuracy']:.3f} | "
      f"bal_acc={metrics['val_balanced_accuracy']:.3f} | "
      f"AUC={metrics['val_auc']:.3f}")
print(f"  Val confusion matrix [[TN, FP], [FN, TP]]: {cm_val}")
print(f"\nSaved outputs to {OUT_DIR}")
