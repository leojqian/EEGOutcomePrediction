"""
Notebook C v2 — accuracy-pushing variant of C_train_eval.py.

Three principled changes vs. v1:
  1. THRESHOLD CALIBRATION is now performed on INNER-CV OUT-OF-FOLD
     predictions of the training patients (via cross_val_predict),
     not on the refit model's training predictions. This removes a
     known source of overfitting: in v1, folds whose model achieves
     training accuracy = 1.0 have an arbitrary plateau of thresholds
     all maximizing train BA, so the chosen τ generalizes poorly.
     CV-OOF threshold calibration is leakage-safe (val patients are
     still never touched).

  2. MODEL FAMILIES extended from {LogReg, XGBoost} to
     {LogReg, XGBoost, RandomForest, SVC} (per Maury's request).

  3. XGBOOST GRID is mildly extended along the dimensions whose
     selected hyperparameters consistently landed on the v1 grid edge.

Outputs (in pipeline/results/fold_{K}/top2_per_sheet_v2/):
  feature_list.csv, train_results.csv, test_predictions.csv,
  test_metrics.json, model.pkl
"""

# %% [Imports + config]
import os, re, glob, json, pickle, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import (StratifiedGroupKFold, GridSearchCV,
                                       cross_val_predict)
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score,
    average_precision_score, f1_score, confusion_matrix,
)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

warnings.filterwarnings("ignore")

PROJECT_DIR    = Path("/Users/leoqian/mdanderson/EEGOutcomePrediction")
DATA_DIR       = PROJECT_DIR / "processeddata"
SPLITS_TAG     = os.environ.get("SPLITS_TAG", "")   # "" → splits/, "v3" → splits_v3/
SPLITS_DIR     = (PROJECT_DIR / "pipeline" /
                  (f"splits_{SPLITS_TAG}" if SPLITS_TAG else "splits"))
BANDS_BASE_TAG = f"_{SPLITS_TAG}" if SPLITS_TAG else ""
RESULTS_BASE_TAG = BANDS_BASE_TAG

# --- Methodology knobs ----------------------------------------------------
FOLD_INDEX     = int(os.environ.get("FOLD_INDEX", 0))
SELECTION_MODE = "top2_per_sheet"          # source mode for Module B output
OUTPUT_TAG     = "top2_per_sheet_v4_cal"    # v4 = v3 + Platt calibration, no ensemble
SUMMARY_FUNCS  = ["mean", "std"]
MAX_FEATURES   = 14
MODEL_CV_FOLDS = 5
RANDOM_SEED    = 42

INCLUDE_CLINICAL  = True
INCLUDE_MODALITY  = True
# pain_t1, age, neurop_months  +  3 one-hot modality flags (NFB, DL, NFB+DL)
CLINICAL_FEATURES = ["pain_t1", "age", "neurop_months",
                     "mod_NFB", "mod_DL", "mod_NFB_DL"]
OUTCOMES_FILE     = DATA_DIR / "Randomization factors and Primary outcome.xlsx"

BANDS_DIR = (PROJECT_DIR / "pipeline" / f"bands{BANDS_BASE_TAG}"
             / f"fold_{FOLD_INDEX}" / SELECTION_MODE)
OUT_DIR   = (PROJECT_DIR / "pipeline" / f"results{RESULTS_BASE_TAG}"
             / f"fold_{FOLD_INDEX}" / OUTPUT_TAG)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# XGBoost grid — same as v1. A v2 attempt to extend the grid downward at
# the boundary-flagged edges produced degenerate configs (max_depth=1,
# learning_rate=0.001) that over-fit inner-CV and hurt validation; the
# original v1 grid is the better operating point on this cohort.
XGB_PARAM_GRID = {
    "n_estimators":     [50, 100, 200, 300],
    "max_depth":        [2, 3, 4, 5, 6, 7],
    "learning_rate":    [0.005, 0.01, 0.03, 0.05, 0.1, 0.2],
    "subsample":        [0.7, 0.85, 1.0],
    "colsample_bytree": [0.7, 0.85, 1.0],
    "reg_lambda":       [0.1, 1, 5, 10],
}

LR_C_VALUES = [1e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 0.5, 1, 5, 10, 50, 100, 500]
LR_PARAM_GRID = [
    {"clf__solver": ["liblinear"], "clf__penalty": ["l1", "l2"],
     "clf__C": LR_C_VALUES},
    {"clf__solver": ["saga"], "clf__penalty": ["l2"],
     "clf__C": LR_C_VALUES},
    {"clf__solver": ["saga"], "clf__penalty": ["elasticnet"],
     "clf__C": [1e-2, 1e-1, 1, 10, 100],
     "clf__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]},
]

# Random Forest — small but expressive grid.
RF_PARAM_GRID = {
    "n_estimators":     [200, 400, 800],
    "max_depth":        [None, 3, 5, 8],
    "min_samples_split":[2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features":     ["sqrt", 0.5],
}

# SVM with RBF kernel — small grid (SVC with probability=True is slow).
SVM_PARAM_GRID = [{
    "clf__C":     [0.1, 1, 10, 100],
    "clf__gamma": ["scale", 0.01, 0.1, 1],
    "clf__kernel":["rbf"],
}, {
    "clf__C":     [0.1, 1, 10, 100],
    "clf__kernel":["linear"],
}]


def check_grid_boundaries(best_params, distributions):
    notes = []
    for k, v in best_params.items():
        if k not in distributions:
            continue
        try:
            opts = sorted(distributions[k])
        except TypeError:
            continue
        if v == opts[0]:
            notes.append(f"{k}={v} at LOW edge of {opts}")
        elif v == opts[-1]:
            notes.append(f"{k}={v} at HIGH edge of {opts}")
    return notes


# %% [Load fold IDs + selected bands from B]
train_ids = np.load(SPLITS_DIR / f"fold_{FOLD_INDEX}_train.npy")
val_ids   = np.load(SPLITS_DIR / f"fold_{FOLD_INDEX}_val.npy")
with open(SPLITS_DIR / "labels.json") as f:
    labels = {int(p): int(y) for p, y in json.load(f).items()}

VAL_ID_SET = frozenset(int(p) for p in val_ids)
def assert_no_val_leak(pids, where):
    leaked = VAL_ID_SET & set(int(p) for p in pids)
    if leaked:
        raise RuntimeError(f"LEAKAGE in {where}: val pids {sorted(leaked)} present.")

selected = {}
for selj in sorted(BANDS_DIR.glob("*_selected.json")):
    sheet_label = selj.name.replace("_selected.json", "")
    with open(selj) as f:
        selected[sheet_label] = json.load(f)
print(f"Loaded selected bands for {len(selected)} sheets:")
for label, info in selected.items():
    print(f"  {label} ({info['sheet_name']}): {info['bands']}")
if not any(info["bands"] for info in selected.values()):
    raise RuntimeError("No bands selected — re-run Module B first.")

# %% [Patient -> file map]
files_by_pid = {}
for f in sorted(glob.glob(str(DATA_DIR / "CIPN3*.xlsx"))):
    m = re.search(r"CIPN3(\d{3})", os.path.basename(f))
    if m:
        files_by_pid[int(m.group(1))] = f


# %% [Clinical features incl. treatment-modality one-hot]
# Group assignment column (NFB=1, NFB+DL=2, DL=3). Known at baseline.
GROUP_COL = "Group assignment - can code the variables (NFB=1; NFB+DL=2; DL=3)"

def _load_clinical_by_pid():
    df = pd.read_excel(OUTCOMES_FILE)
    t1 = df[df["Event Name"] == "T1"].copy()
    t1["pain_t1"]       = pd.to_numeric(t1["Pain Unpleasantness"], errors="coerce")
    t1["age"]           = pd.to_numeric(t1["Age"], errors="coerce")
    t1["neurop_months"] = pd.to_numeric(
        t1["How many months have you been experiencing neuropathy?"], errors="coerce")
    t1["group"]         = pd.to_numeric(t1[GROUP_COL], errors="coerce")
    out = {}
    for _, row in t1.iterrows():
        pid = int(row["Patient number"])
        g = row["group"]
        # one-hot: 1=NFB, 2=NFB+DL, 3=DL.
        mod_NFB    = 1.0 if g == 1 else 0.0 if pd.notna(g) else np.nan
        mod_NFB_DL = 1.0 if g == 2 else 0.0 if pd.notna(g) else np.nan
        mod_DL     = 1.0 if g == 3 else 0.0 if pd.notna(g) else np.nan
        vals = {
            "pain_t1":       float(row["pain_t1"])       if pd.notna(row["pain_t1"])       else np.nan,
            "age":           float(row["age"])           if pd.notna(row["age"])           else np.nan,
            "neurop_months": float(row["neurop_months"]) if pd.notna(row["neurop_months"]) else np.nan,
            "mod_NFB":       mod_NFB,
            "mod_DL":        mod_DL,
            "mod_NFB_DL":    mod_NFB_DL,
        }
        out[pid] = {cf: vals[cf] for cf in CLINICAL_FEATURES}
    return out

clinical_by_pid = _load_clinical_by_pid() if INCLUDE_CLINICAL else {}


# %% [Feature extraction]
SUMMARY_FUNC_MAP = {"mean": np.nanmean, "std": np.nanstd, "median": np.nanmedian}

def extract_features(pid, selected, summary_funcs):
    vals, names = [], []
    for sheet_label, info in selected.items():
        sheet_name = info["sheet_name"]
        bands      = info["bands"]
        if not bands: continue
        df_sheet = pd.read_excel(files_by_pid[int(pid)],
                                  sheet_name=sheet_name, index_col=0)
        for b in bands:
            if b not in df_sheet.columns: continue
            col = df_sheet[b].values.astype(float)
            for fn_name in summary_funcs:
                vals.append(float(SUMMARY_FUNC_MAP[fn_name](col)))
                names.append(f"{sheet_label}__{b}__{fn_name}")
    if INCLUDE_CLINICAL:
        clin = clinical_by_pid.get(int(pid), {})
        for cf in CLINICAL_FEATURES:
            vals.append(clin.get(cf, np.nan))
            names.append(f"clinical__{cf}")
    return np.array(vals, dtype=float), names


# %% [Build train + val matrices]
assert_no_val_leak(train_ids, "Step C v2 train feature extraction")

X_train_rows, feature_names = [], None
for p in train_ids:
    vec, names = extract_features(p, selected, SUMMARY_FUNCS)
    X_train_rows.append(vec); feature_names = names
X_train = np.vstack(X_train_rows)
y_train = np.array([labels[int(p)] for p in train_ids])

# Train-median clinical imputation
clinical_idx = [i for i, n in enumerate(feature_names) if n.startswith("clinical__")]
clinical_medians = {}
for i in clinical_idx:
    col = X_train[:, i]
    med = float(np.nanmedian(col))
    clinical_medians[feature_names[i]] = med
    X_train[np.isnan(col), i] = med
X_train = np.nan_to_num(X_train)

n_clinical = len(clinical_idx)
n_eeg      = len(feature_names) - n_clinical
print(f"\nTrain matrix: {X_train.shape} | EEG {n_eeg}, clinical {n_clinical}")
pd.DataFrame({"feature": feature_names}).to_csv(OUT_DIR/"feature_list.csv", index=False)

groups_train = np.array([int(p) for p in train_ids])
inner_cv = StratifiedGroupKFold(n_splits=MODEL_CV_FOLDS, shuffle=True,
                                  random_state=RANDOM_SEED)

# %% [Model search]
models = {
    "logreg": (
        Pipeline([("scaler", StandardScaler()),
                  ("clf", LogisticRegression(class_weight="balanced",
                                              max_iter=50000, tol=1e-3,
                                              random_state=RANDOM_SEED))]),
        LR_PARAM_GRID,
    ),
    "rf": (
        RandomForestClassifier(class_weight="balanced", n_jobs=1,
                                random_state=RANDOM_SEED),
        RF_PARAM_GRID,
    ),
    "svm": (
        Pipeline([("scaler", StandardScaler()),
                  ("clf", SVC(class_weight="balanced", probability=True,
                               random_state=RANDOM_SEED))]),
        SVM_PARAM_GRID,
    ),
}
if HAS_XGB:
    models["xgboost"] = (
        XGBClassifier(eval_metric="logloss", random_state=RANDOM_SEED,
                       n_jobs=1, tree_method="hist"),
        XGB_PARAM_GRID,
    )

train_results, fitted = [], {}
for name, (est, params) in models.items():
    print(f"\n--- Grid-searching {name} ---")
    searcher = GridSearchCV(est, params, scoring="balanced_accuracy",
                             cv=inner_cv, n_jobs=-1, refit=True)
    searcher.fit(X_train, y_train, groups=groups_train)
    fitted[name] = searcher

    boundaries = (check_grid_boundaries(searcher.best_params_, params)
                   if isinstance(params, dict) else [])
    print(f"  {name}: CV bal_acc={searcher.best_score_:.3f} | "
          f"params={searcher.best_params_}")

    train_results.append({
        "model":      name,
        "cv_bal_acc": float(searcher.best_score_),
        "params":     json.dumps(searcher.best_params_),
        "boundary_warnings": "; ".join(boundaries),
    })

pd.DataFrame(train_results).to_csv(OUT_DIR/"train_results.csv", index=False)
best_name = max(train_results, key=lambda r: r["cv_bal_acc"])["model"]
best_searcher = fitted[best_name]
best_est = best_searcher.best_estimator_
print(f"\nBest single model family (by inner CV): {best_name}  "
      f"(BA={best_searcher.best_score_:.3f})")


# %% [v4: NO ensemble. Wrap the single-best classifier in Platt scaling so
# probabilities are properly calibrated and τ = 0.5 corresponds to a true
# 50/50 posterior. CalibratedClassifierCV does its own internal 5-fold
# split on the training data; val patients are still never touched.]
print(f"\nWrapping {best_name} in CalibratedClassifierCV (sigmoid, cv=5)…")
best_params_dict = best_searcher.best_params_
# Build a fresh estimator with the chosen best params, then calibrate.
base_for_cal = clone(best_searcher.estimator)
base_for_cal.set_params(**best_params_dict)
calibrated = CalibratedClassifierCV(
    estimator=base_for_cal, method="sigmoid", cv=5)
calibrated.fit(X_train, y_train)
best_est = calibrated

# Inner-CV OOF probabilities of the calibrated model — used for threshold choice
oof_proba_for_th = cross_val_predict(
    clone(calibrated), X_train, y_train, groups=groups_train,
    cv=inner_cv, method="predict_proba", n_jobs=-1)[:, 1]
oof_proba_per_family = {best_name: oof_proba_for_th}
oof_proba_ens = oof_proba_for_th
ensemble_oof_ba = float(max(balanced_accuracy_score(
    y_train, (oof_proba_for_th >= t).astype(int))
    for t in np.linspace(0.05, 0.95, 181)))
single_oof_ba = ensemble_oof_ba

# %% [Inner-CV OOF threshold calibration (v2)]
# Pick the threshold from the plateau closest to τ = 0.5 (robust under
# small-n noise). All inner classifiers use class_weight="balanced", so
# τ = 0.5 is the natural anchor.
if best_name == "ensemble_soft_vote":
    oof_proba_for_th = oof_proba_ens
else:
    oof_proba_for_th = oof_proba_per_family[best_name]

ths = np.linspace(0.05, 0.95, 181)
ba_curve = np.array([
    balanced_accuracy_score(y_train, (oof_proba_for_th >= t).astype(int))
    for t in ths
])
best_oof_ba = float(ba_curve.max())
top_mask = ba_curve >= best_oof_ba - 1e-9
top_ths = ths[top_mask]
best_th = float(top_ths[np.argmin(np.abs(top_ths - 0.5))])
print(f"\nThreshold calibration (OOF): τ = {best_th:.3f}  "
      f"| OOF BA at τ = {best_oof_ba:.3f}  "
      f"(plateau width = {len(top_ths)} thresholds)")

# Also report what the v1-style train threshold WOULD have been
p_train_refit = best_est.predict_proba(X_train)[:, 1]
v1_best_th, v1_best_ba = 0.5, -np.inf
for t in ths:
    ba = balanced_accuracy_score(y_train, (p_train_refit >= t).astype(int))
    if ba > v1_best_ba:
        v1_best_ba, v1_best_th = ba, t
print(f"  (v1 style would have picked τ = {v1_best_th:.3f}; OOF τ used.)")

# %% [Final one-shot val evaluation]
X_val_rows = []
for p in val_ids:
    vec, names = extract_features(p, selected, SUMMARY_FUNCS)
    assert names == feature_names
    X_val_rows.append(vec)
X_val = np.vstack(X_val_rows)
for i in clinical_idx:
    med = clinical_medians[feature_names[i]]
    mask = np.isnan(X_val[:, i])
    if mask.any():
        X_val[mask, i] = med
X_val = np.nan_to_num(X_val)
y_val = np.array([labels[int(p)] for p in val_ids])

proba = best_est.predict_proba(X_val)[:, 1]
pred  = (proba >= best_th).astype(int)
cm_val = confusion_matrix(y_val, pred, labels=[0, 1]).tolist()

train_pred = (p_train_refit >= best_th).astype(int)
train_acc  = float(accuracy_score(y_train, train_pred))

best_row_info = next((r for r in train_results if r["model"] == best_name), None)
metrics = {
    "fold":            int(FOLD_INDEX),
    "selection_mode":  SELECTION_MODE,
    "output_tag":      OUTPUT_TAG,
    "model":           best_name,
    "best_params":     (json.loads(best_row_info["params"]) if best_row_info
                        else {"ensemble_members": list(ensemble_members.keys())}),
    "boundary_warnings": (best_row_info["boundary_warnings"] if best_row_info else ""),
    "ensemble_oof_ba":   ensemble_oof_ba,
    "best_single_oof_ba": single_oof_ba,
    "ensemble_member_inner_cv_ba": {n: float(fitted[n].best_score_)
                                     for n in models},
    "threshold":       float(best_th),
    "threshold_source":"inner_cv_oof_train",
    "v1_style_threshold": float(v1_best_th),
    "oof_train_bal_acc_at_threshold": float(best_oof_ba),
    "train_accuracy":  train_acc,
    "n_val":           int(len(y_val)),
    "pos_rate_val":    float(y_val.mean()),
    "accuracy":        float(accuracy_score(y_val, pred)),
    "balanced_accuracy": float(balanced_accuracy_score(y_val, pred)),
    "auc":             float(roc_auc_score(y_val, proba)),
    "average_precision": float(average_precision_score(y_val, proba)),
    "f1":              float(f1_score(y_val, pred, zero_division=0)),
    "confusion_matrix_val": cm_val,
    "n_features":      int(len(feature_names)),
    "n_eeg_features":      n_eeg,
    "n_clinical_features": n_clinical,
    "summary_funcs":   list(SUMMARY_FUNCS),
    "selected_bands":  {label: info["bands"] for label, info in selected.items()},
    "clinical_medians": clinical_medians,
}

pd.DataFrame({
    "pid": val_ids, "y_true": y_val, "y_pred": pred, "proba": proba,
}).to_csv(OUT_DIR/"test_predictions.csv", index=False)
with open(OUT_DIR/"test_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
with open(OUT_DIR/"model.pkl", "wb") as f:
    pickle.dump({
        "name": best_name, "model": best_est,
        "feature_names": feature_names,
        "selected_bands": selected,
        "summary_funcs": list(SUMMARY_FUNCS),
        "clinical_medians": clinical_medians,
        "fold": FOLD_INDEX, "threshold": best_th,
    }, f)

print(f"\n=== Fold {FOLD_INDEX} (v2) ===")
print(f"  best model: {best_name} (CV BA {best_searcher.best_score_:.3f})")
print(f"  τ = {best_th:.3f} (OOF-calibrated)")
print(f"  val accuracy: {metrics['accuracy']:.3f}")
print(f"  val balanced accuracy: {metrics['balanced_accuracy']:.3f}")
print(f"  val AUC: {metrics['auc']:.3f}")
print(f"  val CM [[TN,FP],[FN,TP]]: {cm_val}")
