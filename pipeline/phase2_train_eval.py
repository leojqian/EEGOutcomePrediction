"""
Phase 2 — Feature construction + XGBoost training + single-shot val eval.

Refactor of pipeline/C_train_eval.py with the simpler design:
  - Build per-patient [mean, std] features over Phase 1's selected bands.
  - Cap total features at MAX_FEATURES (= 14).
  - GridSearchCV XGBoost (5-fold patient-grouped CV) on the 85 train patients.
  - Refit on all 85; evaluate ONCE on the 22 held-out patients.

Run AFTER pipeline/A_make_splits.py and pipeline/phase1_band_selection.py.

Outputs (in pipeline/results/phase2/fold_{K}/):
  feature_list.csv      — ordered feature names used in train/val matrices
  best_params.json      — best XGB hyperparameters + inner-CV bal_acc
  val_metrics.json      — accuracy, AUC, balanced_accuracy on the 22
  val_predictions.csv   — per-patient probabilities + predictions
  model.pkl             — refit XGB + the artifacts needed to replay it
"""

# %% [Imports + config]
from __future__ import annotations

import glob, json, os, pickle, re, warnings
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
except ImportError as e:
    raise ImportError("xgboost is required for Phase 2. `pip install xgboost`") from e

PROJECT_DIR = Path("/Users/leoqian/mdanderson/EEGOutcomePrediction")
DATA_DIR    = PROJECT_DIR / "processeddata"
SPLITS_DIR  = PROJECT_DIR / "pipeline" / "splits"

# --- Methodology knobs ---------------------------------------------------
FOLD_INDEX     = 0                  # must match Phase 1
SUMMARY_FUNCS  = ["mean", "std"]    # per-patient features per (group, band)
MAX_FEATURES   = 14                 # hard cap (low-n regularization)
GS_FOLDS       = 5                  # GridSearchCV inner folds
RANDOM_SEED    = 42

XGB_PARAM_GRID = {
    "n_estimators":  [50, 100, 200],
    "max_depth":     [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2],
}

PHASE1_DIR = PROJECT_DIR / "pipeline" / "results" / "phase1" / f"fold_{FOLD_INDEX}"
OUT_DIR    = PROJECT_DIR / "pipeline" / "results" / "phase2" / f"fold_{FOLD_INDEX}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_FUNC_MAP = {
    "mean":   np.nanmean,
    "std":    np.nanstd,
    "median": np.nanmedian,
    "p90":    lambda a: np.nanpercentile(a, 90),
    "p95":    lambda a: np.nanpercentile(a, 95),
}


# %% [Helpers]
def patient_file_map() -> dict[int, str]:
    files: dict[int, str] = {}
    for f in sorted(glob.glob(str(DATA_DIR / "CIPN3*.xlsx"))):
        m = re.search(r"CIPN3(\d{3})", os.path.basename(f))
        if m:
            files[int(m.group(1))] = f
    return files


def load_split(fold_index: int) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    train_ids = np.load(SPLITS_DIR / f"fold_{fold_index}_train.npy")
    val_ids   = np.load(SPLITS_DIR / f"fold_{fold_index}_val.npy")
    with open(SPLITS_DIR / "labels.json") as f:
        labels = {int(p): int(y) for p, y in json.load(f).items()}
    return train_ids, val_ids, labels


def load_phase1_selection() -> tuple[dict[str, list[str]], dict[str, str]]:
    """Returns ({sheet_label: [bands]}, {sheet_label: sheet_name})."""
    path = PHASE1_DIR / "selected_bands.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Phase 1 output missing: {path}. Run phase1_band_selection.py first."
        )
    with open(path) as f:
        info = json.load(f)
    return info["selected"], info["sheets"]


def build_feature_matrix(
    pids: Sequence[int],
    files: dict[int, str],
    selected: dict[str, list[str]],
    sheets: dict[str, str],
    summary_funcs: Sequence[str],
) -> tuple[np.ndarray, list[str]]:
    """Per-patient features for every (group, band, summary_func).

    Summaries are computed per-patient over channels/pairs → no cross-patient
    leakage by construction.
    """
    feature_names = [
        f"{group}__{band}__{fn}"
        for group, bands in selected.items() if bands
        for band in bands
        for fn in summary_funcs
    ]

    X_rows: list[list[float]] = []
    for pid in pids:
        row: list[float] = []
        for group, bands in selected.items():
            if not bands:
                continue
            df = pd.read_excel(files[int(pid)], sheet_name=sheets[group],
                               index_col=0)
            for band in bands:
                if band not in df.columns:
                    raise RuntimeError(
                        f"Patient {pid}: band {band} missing in {group}."
                    )
                col = df[band].values.astype(float)
                for fn in summary_funcs:
                    row.append(float(SUMMARY_FUNC_MAP[fn](col)))
        X_rows.append(row)
    return np.nan_to_num(np.vstack(X_rows)), feature_names


def train_xgboost(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray,
) -> tuple[XGBClassifier, dict, float]:
    """5-fold patient-grouped GridSearchCV on the 85; refit on all 85."""
    cv = StratifiedGroupKFold(n_splits=GS_FOLDS, shuffle=True,
                              random_state=RANDOM_SEED)
    base = XGBClassifier(eval_metric="logloss", random_state=RANDOM_SEED,
                         n_jobs=1, tree_method="hist")
    gs = GridSearchCV(base, XGB_PARAM_GRID, scoring="balanced_accuracy",
                      cv=cv, n_jobs=-1, refit=True)
    gs.fit(X, y, groups=groups)
    return gs.best_estimator_, gs.best_params_, float(gs.best_score_)


# %% [Main]
def main() -> None:
    files = patient_file_map()
    train_ids, val_ids, labels = load_split(FOLD_INDEX)
    selected, sheets = load_phase1_selection()
    print(f"Phase 1 selected: {selected}")

    y_train = np.array([labels[int(p)] for p in train_ids])
    y_val   = np.array([labels[int(p)] for p in val_ids])

    # --- Build TRAIN matrix (val patients are NOT touched yet) ---
    X_train, feature_names = build_feature_matrix(
        train_ids, files, selected, sheets, SUMMARY_FUNCS,
    )
    print(f"Train: X={X_train.shape}, n_features={len(feature_names)}")
    if len(feature_names) > MAX_FEATURES:
        raise RuntimeError(
            f"Too many features ({len(feature_names)} > {MAX_FEATURES}). "
            f"Tighten Phase 1 (lower SELECTION_K / raise THRESHOLD) or drop a summary."
        )
    pd.DataFrame({"feature": feature_names}).to_csv(
        OUT_DIR / "feature_list.csv", index=False,
    )

    # --- Hyperparameter tuning + final fit on the 85 ---
    groups_train = np.asarray(train_ids, dtype=int)
    model, best_params, cv_score = train_xgboost(X_train, y_train, groups_train)
    train_pred = model.predict(X_train)
    train_metrics = {
        "accuracy":          float(accuracy_score(y_train, train_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_train, train_pred)),
    }
    print(f"\n[Phase 2] Best params: {best_params}")
    print(f"          CV bal_acc:  {cv_score:.3f}")
    print(f"          Train acc:   {train_metrics['accuracy']:.3f} | "
          f"bal_acc: {train_metrics['balanced_accuracy']:.3f}")

    # --- Single-shot validation (FIRST + ONLY time we touch the 22) ---
    X_val, val_feature_names = build_feature_matrix(
        val_ids, files, selected, sheets, SUMMARY_FUNCS,
    )
    assert val_feature_names == feature_names, "Feature schema drift train→val."

    proba = model.predict_proba(X_val)[:, 1]
    pred  = (proba >= 0.5).astype(int)
    val_metrics = {
        "fold":              int(FOLD_INDEX),
        "n_val":             int(len(y_val)),
        "pos_rate_val":      float(y_val.mean()),
        "accuracy":          float(accuracy_score(y_val, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_val, pred)),
        "auc":               float(roc_auc_score(y_val, proba)),
        "n_features":        int(len(feature_names)),
        "summary_funcs":     SUMMARY_FUNCS,
        "selected_bands":    selected,
        "best_params":       best_params,
        "cv_bal_acc":        cv_score,
        "train_metrics":     train_metrics,
    }

    # --- Persist ---
    with open(OUT_DIR / "best_params.json", "w") as f:
        json.dump({"best_params": best_params, "cv_bal_acc": cv_score}, f, indent=2)
    with open(OUT_DIR / "val_metrics.json", "w") as f:
        json.dump(val_metrics, f, indent=2)
    pd.DataFrame({
        "pid": val_ids, "y_true": y_val, "y_pred": pred, "proba": proba,
    }).to_csv(OUT_DIR / "val_predictions.csv", index=False)
    with open(OUT_DIR / "model.pkl", "wb") as f:
        pickle.dump({
            "model":          model,
            "feature_names":  feature_names,
            "selected_bands": selected,
            "sheets":         sheets,
            "summary_funcs":  SUMMARY_FUNCS,
            "best_params":    best_params,
            "fold_index":     FOLD_INDEX,
        }, f)

    # --- Report ---
    print(f"\n=== Validation results (fold {FOLD_INDEX}, n={val_metrics['n_val']}) ===")
    print(f"  accuracy:           {val_metrics['accuracy']:.3f}")
    print(f"  balanced_accuracy:  {val_metrics['balanced_accuracy']:.3f}")
    print(f"  AUC:                {val_metrics['auc']:.3f}")
    print(f"  selected bands:     {selected}")
    print(f"  best params:        {best_params}")
    print(f"  train→val gap:      acc {train_metrics['accuracy']:.3f} → "
          f"{val_metrics['accuracy']:.3f}")
    print(f"\nSaved Phase 2 outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
