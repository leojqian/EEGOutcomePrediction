"""
Notebook C — Feature engineering + final model + single-shot val evaluation.
Operates on FOLD_INDEX's 85 train patients; touches the 22 val patients
exactly once at the end.

SELECTION_MODE picks which B output to read (must match B's run):
  - "abc_existing"     → B's vote-threshold band list (Version 1)
  - "top2_per_sheet"   → B's top-2-per-sheet band list (Version 2)
  - "3ref_per_fold"    → B's per-fold best-subset list (3_REF style)
                         Triggers 5-vec features and disables the MAX_FEATURES cap.

Models: XGBoost (wide grid, ~5184 configs) + Logistic Regression (wide grid,
~80 configs incl. l1/l2/elasticnet). Both run full GridSearchCV; the better
inner-CV bal_acc wins. Boundary warnings flag any XGB best-param landing on
a grid edge so it can be re-extended in that direction.

Run AFTER pipeline/A_make_splits.py and pipeline/B_band_selection.py.

Outputs (in pipeline/results/fold_{K}/{mode}/):
  feature_list.csv
  train_results.csv
  test_predictions.csv
  test_metrics.json
  model.pkl

Manual experimentation knob: SUMMARY_FUNCS — start with ["mean", "std"],
swap in "median" / "p90" / "p95" to try different feature sets. The
MAX_FEATURES assertion guards the 12–14 feature cap.
"""

# %% [Imports + config]
import os, re, glob, json, pickle, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
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
SPLITS_DIR     = PROJECT_DIR / "pipeline" / "splits"

# --- Methodology knobs ----------------------------------------------------
# FOLD_INDEX / SELECTION_MODE accept env-var overrides so the all-folds
# driver (E_run_all_folds.py) can sweep them without editing this file.
FOLD_INDEX     = int(os.environ.get("FOLD_INDEX", 0))
SELECTION_MODE = os.environ.get("SELECTION_MODE", "top2_per_sheet")    # must match B
SUMMARY_FUNCS  = ["mean", "std"]    # try ["mean", "median"] / ["mean", "p90"] / etc.
MAX_FEATURES   = 14                 # hard cap (EEG only); clinical add-ons extend it below
MODEL_CV_FOLDS = 5                  # inner CV for grid search (per mentor)
RANDOM_SEED    = 42

# Clinical-feature add-on. When INCLUDE_CLINICAL=True, these scalars are
# appended to each patient's feature vector and imputed with TRAIN median
# (leakage-safe). Effective cap = MAX_FEATURES + len(CLINICAL_FEATURES).
INCLUDE_CLINICAL  = True
CLINICAL_FEATURES = ["pain_t1", "age", "neurop_months"]
OUTCOMES_FILE     = DATA_DIR / "Randomization factors and Primary outcome.xlsx"

BANDS_DIR = (PROJECT_DIR / "pipeline" / "bands"
             / f"fold_{FOLD_INDEX}" / SELECTION_MODE)
OUT_DIR   = (PROJECT_DIR / "pipeline" / "results"
             / f"fold_{FOLD_INDEX}" / SELECTION_MODE)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Full XGBoost grid — GridSearchCV evaluates every combination.
# 4 × 6 × 6 × 3 × 3 × 4 = 5184 configs × MODEL_CV_FOLDS folds = ~26k fits.
XGB_PARAM_GRID = {
    "n_estimators":     [50, 100, 200, 300],
    "max_depth":        [2, 3, 4, 5, 6, 7],
    "learning_rate":    [0.005, 0.01, 0.03, 0.05, 0.1, 0.2],
    "subsample":        [0.7, 0.85, 1.0],
    "colsample_bytree": [0.7, 0.85, 1.0],
    "reg_lambda":       [0.1, 1, 5, 10],
}

# Wide Logistic Regression grid — l1, l2, and elasticnet penalties with a
# 13-value C sweep across 6 orders of magnitude. Total ~77 configurations.
LR_C_VALUES = [1e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 0.5, 1, 5, 10, 50, 100, 500]
LR_PARAM_GRID = [
    {"clf__solver": ["liblinear"], "clf__penalty": ["l1", "l2"],
     "clf__C": LR_C_VALUES},
    {"clf__solver": ["saga"], "clf__penalty": ["l1"],
     "clf__C": LR_C_VALUES, "clf__warm_start": [True]},
    {"clf__solver": ["saga"], "clf__penalty": ["l2"],
     "clf__C": LR_C_VALUES},
    {"clf__solver": ["saga"], "clf__penalty": ["elasticnet"],
     "clf__C": [1e-2, 1e-1, 1, 10, 100],
     "clf__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]},
]


def check_grid_boundaries(best_params, distributions):
    """Flag any best-param value sitting on the low/high edge of its grid."""
    notes = []
    for k, v in best_params.items():
        if k not in distributions:
            continue
        opts = sorted(distributions[k])
        if v == opts[0]:
            notes.append(f"{k}={v} at LOW edge of {opts} — consider extending downward")
        elif v == opts[-1]:
            notes.append(f"{k}={v} at HIGH edge of {opts} — consider extending upward")
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

# Selected bands per sheet (from Notebook B).
selected = {}
for selj in sorted(BANDS_DIR.glob("*_selected.json")):
    sheet_label = selj.name.replace("_selected.json", "")
    with open(selj) as f:
        selected[sheet_label] = json.load(f)
print(f"Loaded selected bands for {len(selected)} sheets:")
for label, info in selected.items():
    print(f"  {label} ({info['sheet_name']}): {info['bands']}")
if not BANDS_DIR.exists() or not any(BANDS_DIR.glob("*_selected.json")):
    raise RuntimeError(
        f"No B output at {BANDS_DIR}. Run B with SELECTION_MODE='{SELECTION_MODE}' first."
    )
if not any(info["bands"] for info in selected.values()):
    raise RuntimeError("No bands selected in any sheet — re-run Notebook B "
                       "with a lower BAND_SELECTION_THRESHOLD or different mode.")


# %% [Patient -> file map]
files_by_pid = {}
for f in sorted(glob.glob(str(DATA_DIR / "CIPN3*.xlsx"))):
    m = re.search(r"CIPN3(\d{3})", os.path.basename(f))
    if m:
        files_by_pid[int(m.group(1))] = f


# %% [Clinical features — load once, NaN preserved for train-median imputation]
def _load_clinical_by_pid():
    """Read T1 baseline rows and return {pid: {pain_t1, age, neurop_months}}.
    NaN preserved so run() can impute with TRAIN median per fold.
    NeuropMonths uses pd.to_numeric(errors='coerce') for free-text entries.
    """
    df = pd.read_excel(OUTCOMES_FILE)
    t1 = df[df["Event Name"] == "T1"].copy()
    t1["pain_t1"]       = pd.to_numeric(t1["Pain Unpleasantness"], errors="coerce")
    t1["age"]           = pd.to_numeric(t1["Age"], errors="coerce")
    t1["neurop_months"] = pd.to_numeric(
        t1["How many months have you been experiencing neuropathy?"],
        errors="coerce",
    )
    out = {}
    for _, row in t1.iterrows():
        pid = int(row["Patient number"])
        out[pid] = {cf: float(row[cf]) if pd.notna(row[cf]) else np.nan
                    for cf in CLINICAL_FEATURES}
    return out

clinical_by_pid = _load_clinical_by_pid() if INCLUDE_CLINICAL else {}
if INCLUDE_CLINICAL:
    print(f"Clinical features loaded for {len(clinical_by_pid)} patients "
          f"({CLINICAL_FEATURES})")
else:
    print("Clinical features DISABLED (INCLUDE_CLINICAL=False).")


# %% [Feature engineering — per-patient summary stats over selected bands]
SUMMARY_FUNC_MAP = {
    "mean":   np.nanmean,
    "std":    np.nanstd,
    "median": np.nanmedian,
    "p25":    lambda a: np.nanpercentile(a, 25),
    "p75":    lambda a: np.nanpercentile(a, 75),
    "p90":    lambda a: np.nanpercentile(a, 90),
    "p95":    lambda a: np.nanpercentile(a, 95),
}

FIVE_VEC_STATS = ["col_mean", "col_std", "g_mean", "g_std", "g_med"]

def _five_vec(col_array: np.ndarray) -> list[float]:
    """3_REF-style [col_mean, col_std, g_mean, g_std, g_med].
    For a 1-col slice the col_* and g_* entries duplicate; kept verbatim."""
    A = col_array.reshape(-1, 1) if col_array.ndim == 1 else col_array
    A = np.nan_to_num(A.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    return [float(A.mean(axis=0)[0]), float(A.std(axis=0)[0]),
            float(A.mean()),          float(A.std()),
            float(np.median(A))]


def extract_features(pid, selected, summary_funcs):
    """Compute summary features for ONE patient. All summaries are computed
    per-patient — no cross-patient normalization, so this is leakage-safe.

    When SELECTION_MODE == "3ref_per_fold", emits the 5-vec per band; otherwise
    emits len(summary_funcs) scalars per band.
    """
    vals, names = [], []
    use_5vec = SELECTION_MODE == "3ref_per_fold"
    for sheet_label, info in selected.items():
        sheet_name = info["sheet_name"]
        bands      = info["bands"]
        if not bands:
            continue
        df_sheet = pd.read_excel(files_by_pid[int(pid)],
                                  sheet_name=sheet_name, index_col=0)
        for b in bands:
            if b not in df_sheet.columns:
                continue
            col = df_sheet[b].values.astype(float)
            if use_5vec:
                for stat, v in zip(FIVE_VEC_STATS, _five_vec(col)):
                    vals.append(v)
                    names.append(f"{sheet_label}__{b}__{stat}")
            else:
                for fn_name in summary_funcs:
                    vals.append(float(SUMMARY_FUNC_MAP[fn_name](col)))
                    names.append(f"{sheet_label}__{b}__{fn_name}")

    if INCLUDE_CLINICAL:
        clin = clinical_by_pid.get(int(pid), {})
        for cf in CLINICAL_FEATURES:
            vals.append(clin.get(cf, np.nan))    # NaN kept; imputed in run()
            names.append(f"clinical__{cf}")

    return np.array(vals, dtype=float), names

def run(summary_funcs=None, out_dir=None):
    """Feature extraction → grid search → one-shot val eval for ONE summary
    combo. Returns the metrics dict (also saved to test_metrics.json).

    Top-level setup (fold IDs, selected bands, files_by_pid) is shared
    across calls — only this function's body varies with summary_funcs.
    """
    if summary_funcs is None:
        summary_funcs = SUMMARY_FUNCS
    if out_dir is None:
        out_dir = OUT_DIR
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build train matrix (NaN preserved for clinical imputation below).
    assert_no_val_leak(train_ids, "Step C train feature extraction")
    X_train_rows, feature_names = [], None
    for p in train_ids:
        vec, names = extract_features(p, selected, summary_funcs)
        X_train_rows.append(vec)
        feature_names = names
    X_train = np.vstack(X_train_rows)
    y_train = np.array([labels[int(p)] for p in train_ids])

    # Clinical imputation: TRAIN median, applied later to val too. Compute
    # before nan_to_num so we don't impute NaN→0→median (would skew the median).
    clinical_idx = [i for i, n in enumerate(feature_names)
                    if n.startswith("clinical__")]
    clinical_medians = {}
    for i in clinical_idx:
        col = X_train[:, i]
        med = float(np.nanmedian(col))
        clinical_medians[feature_names[i]] = med
        n_imp = int(np.isnan(col).sum())
        X_train[np.isnan(col), i] = med
        if n_imp:
            print(f"  [impute] {feature_names[i]}: train median={med:.3f} "
                  f"applied to {n_imp} train patients")
    # EEG NaNs (from a missing band column) → 0, matching legacy behavior.
    X_train = np.nan_to_num(X_train)

    n_clinical = len(clinical_idx)
    n_eeg      = len(feature_names) - n_clinical
    effective_cap = MAX_FEATURES + n_clinical    # clinical extends the cap
    print(f"\nTrain matrix: {X_train.shape} | features: {len(feature_names)} "
          f"(EEG={n_eeg}, clinical={n_clinical})")
    if SELECTION_MODE == "3ref_per_fold":
        print(f"[NOTE] 3ref_per_fold: MAX_FEATURES cap disabled (3_REF style).")
    else:
        assert len(feature_names) <= effective_cap, (
            f"Too many features ({len(feature_names)} > {effective_cap}). "
            f"Cap = MAX_FEATURES({MAX_FEATURES}) + clinical({n_clinical}). "
            f"Drop a SUMMARY_FUNC, raise BAND_SELECTION_THRESHOLD in B, or "
            f"shorten the selected band list."
        )
    pd.DataFrame({"feature": feature_names}).to_csv(out_dir / "feature_list.csv",
                                                      index=False)

    # Model training — grid search on the 85, patient-grouped inner CV.
    groups_train = np.array([int(p) for p in train_ids])
    inner_cv = StratifiedGroupKFold(n_splits=MODEL_CV_FOLDS, shuffle=True,
                                      random_state=RANDOM_SEED)

    models: dict = {
        "logreg": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(class_weight="balanced",
                                            max_iter=50000, tol=1e-3,
                                            random_state=RANDOM_SEED)),
            ]),
            LR_PARAM_GRID,
        ),
    }
    if HAS_XGB:
        models["xgboost"] = (
            XGBClassifier(eval_metric="logloss", random_state=RANDOM_SEED,
                           n_jobs=1, tree_method="hist"),
            XGB_PARAM_GRID,
        )
    else:
        print("[NOTE] xgboost not installed (`pip install xgboost`).")

    train_results, fitted = [], {}
    for name, (est, params) in models.items():
        searcher = GridSearchCV(est, params, scoring="balanced_accuracy",
                                 cv=inner_cv, n_jobs=-1, refit=True)
        searcher.fit(X_train, y_train, groups=groups_train)
        fitted[name] = searcher.best_estimator_

        boundaries = (check_grid_boundaries(searcher.best_params_, params)
                       if isinstance(params, dict) else [])

        print(f"{name}: CV bal_acc={searcher.best_score_:.3f} | "
              f"params={searcher.best_params_}")
        for note in boundaries:
            print(f"  [BOUNDARY] {note}")

        train_results.append({
            "model":      name,
            "cv_bal_acc": float(searcher.best_score_),
            "params":     json.dumps(searcher.best_params_),
            "boundary_warnings": "; ".join(boundaries),
        })

    pd.DataFrame(train_results).to_csv(out_dir / "train_results.csv", index=False)
    best_name = max(train_results, key=lambda r: r["cv_bal_acc"])["model"]
    print(f"\nBest model (by inner CV): {best_name}")

    # One-shot validation evaluation — first time we touch the 22.
    y_val = np.array([labels[int(p)] for p in val_ids])
    X_val_rows = []
    for p in val_ids:
        vec, names = extract_features(p, selected, summary_funcs)
        assert names == feature_names, "Feature schema mismatch between train and val."
        X_val_rows.append(vec)
    X_val = np.vstack(X_val_rows)
    # Apply TRAIN-fit medians to val clinical NaNs (leakage-safe).
    for i in clinical_idx:
        med = clinical_medians[feature_names[i]]
        nan_mask = np.isnan(X_val[:, i])
        if nan_mask.any():
            print(f"  [impute] {feature_names[i]}: train median={med:.3f} "
                  f"applied to {int(nan_mask.sum())} val patients")
            X_val[nan_mask, i] = med
    X_val = np.nan_to_num(X_val)

    best = fitted[best_name]

    # Threshold tuned on TRAIN probabilities (matches 3_*.ipynb).
    p_train = best.predict_proba(X_train)[:, 1]
    ths = np.linspace(0.05, 0.95, 181)
    best_th, best_train_ba = 0.5, -np.inf
    for t in ths:
        ba = balanced_accuracy_score(y_train, (p_train >= t).astype(int))
        if ba > best_train_ba:
            best_train_ba, best_th = ba, t
    print(f"Threshold tuned on train: {best_th:.3f} "
          f"(train_bal_acc={best_train_ba:.3f})")

    proba = best.predict_proba(X_val)[:, 1]
    pred  = (proba >= best_th).astype(int)

    train_pred = (p_train >= best_th).astype(int)
    train_acc  = float(accuracy_score(y_train, train_pred))

    cm_val   = confusion_matrix(y_val,   pred,       labels=[0, 1]).tolist()
    cm_train = confusion_matrix(y_train, train_pred, labels=[0, 1]).tolist()

    selected_bands_flat = {label: info["bands"] for label, info in selected.items()}
    n_total_bands = sum(len(b) for b in selected_bands_flat.values())

    best_row = next(r for r in train_results if r["model"] == best_name)

    metrics = {
        "fold":            int(FOLD_INDEX),
        "selection_mode":  SELECTION_MODE,
        "model":           best_name,
        "best_params":     json.loads(best_row["params"]),
        "boundary_warnings": best_row["boundary_warnings"],
        "threshold":       float(best_th),
        "train_bal_acc_at_threshold": float(best_train_ba),
        "train_accuracy":  train_acc,
        "n_val":           int(len(y_val)),
        "pos_rate_val":    float(y_val.mean()),
        "accuracy":        float(accuracy_score(y_val, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_val, pred)),
        "auc":             float(roc_auc_score(y_val, proba)),
        "average_precision": float(average_precision_score(y_val, proba)),
        "f1":              float(f1_score(y_val, pred, zero_division=0)),
        "confusion_matrix_val":   cm_val,
        "confusion_matrix_train": cm_train,
        "n_features":      int(len(feature_names)),
        "n_eeg_features":      n_eeg,
        "n_clinical_features": n_clinical,
        "include_clinical":    INCLUDE_CLINICAL,
        "clinical_features":   CLINICAL_FEATURES if INCLUDE_CLINICAL else [],
        "clinical_medians":    clinical_medians,
        "summary_funcs":   list(summary_funcs),
        "selected_bands":  selected_bands_flat,
        "n_total_bands":   n_total_bands,
    }

    pd.DataFrame({
        "pid": val_ids, "y_true": y_val, "y_pred": pred, "proba": proba,
    }).to_csv(out_dir / "test_predictions.csv", index=False)
    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(out_dir / "model.pkl", "wb") as f:
        pickle.dump({
            "name": best_name,
            "model": best,
            "feature_names": feature_names,
            "selected_bands": selected,
            "summary_funcs": list(summary_funcs),
            "include_clinical": INCLUDE_CLINICAL,
            "clinical_features": CLINICAL_FEATURES if INCLUDE_CLINICAL else [],
            "clinical_medians": clinical_medians,
            "fold": FOLD_INDEX,
            "threshold": best_th,
        }, f)

    # Report + GO/NO-GO.
    print(f"\n=== Run summary (fold {FOLD_INDEX}, mode={SELECTION_MODE}, "
          f"summary_funcs={summary_funcs}) ===")
    print(f"  selected bands per sheet: {selected_bands_flat}")
    print(f"  n_total_bands: {n_total_bands}")
    print(f"  feature names ({len(feature_names)}): {feature_names}")
    print(f"  best model: {best_name}")
    print(f"  best params: {metrics['best_params']}")
    if metrics["boundary_warnings"]:
        print(f"  boundary warnings: {metrics['boundary_warnings']}")
    print(f"  train accuracy: {train_acc:.3f}")
    print(f"  val   accuracy: {metrics['accuracy']:.3f}")
    print(f"  val   balanced_accuracy: {metrics['balanced_accuracy']:.3f}")
    print(f"  val   AUC: {metrics['auc']:.3f}")
    print(f"  val confusion matrix [[TN, FP], [FN, TP]]: {cm_val}")

    if metrics["accuracy"] >= 0.67:
        print(f"\n[GO]   accuracy >= 67% → can extend to other folds.")
    else:
        print(f"\n[STAY] accuracy < 67% → iterate on this fold:\n"
              f"        - try different SUMMARY_FUNCS combinations\n"
              f"        - re-run B with a different BAND_SELECTION_THRESHOLD/mode\n"
              f"        - widen / narrow the model grids")

    return metrics


if __name__ == "__main__":
    run(SUMMARY_FUNCS, OUT_DIR)
