"""
Phase 1 — Per-group band selection on the fold's 85 train patients.

Refactor of pipeline/B_band_selection.py with the simpler design:
  for each feature group (sheet):
    for each of N_FOLDS folds (patient-grouped CV on the 85):
      for each candidate band individually:
        fit LR on per-patient [mean, std]; score balanced_accuracy on the fold's val slice
      record the top-scoring band as that fold's winner
    apply selection rule (top-K wins OR >= THRESHOLD wins)

The 22 validation patients are NEVER loaded by this script.

Run AFTER pipeline/A_make_splits.py. Outputs in
pipeline/results/phase1/fold_{K}/:
  selected_bands.json   — final per-group bands + the rule that picked them
  fold_winners.json     — per-fold winning band for each group
  win_counts.csv        — per-band win tally for each group
"""

# %% [Imports + config]
from __future__ import annotations

import glob, json, os, re, warnings
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")  # mute saga/L1 ConvergenceWarning floods

PROJECT_DIR = Path("/Users/leoqian/mdanderson/EEGOutcomePrediction")
DATA_DIR    = PROJECT_DIR / "processeddata"
SPLITS_DIR  = PROJECT_DIR / "pipeline" / "splits"

# --- Methodology knobs ---------------------------------------------------
FOLD_INDEX       = 0       # which 85/22 split from A_make_splits.py to use
SELECTION_FOLDS  = 7       # K in K-fold CV for band voting
RANDOM_SEED      = 42

# Selection rule. "top_k" picks the SELECTION_K most-frequent winners;
# "threshold" picks bands with count >= SELECTION_THRESHOLD.
SELECTION_RULE       = "top_k"
SELECTION_K          = 2
SELECTION_THRESHOLD  = 4

# Feature groups (label -> sheet name in xlsx).
SHEETS = {
    "coherence": "Z_FFT_Coherence",
    "phaselag":  "Z_FFT_PhaseLag_PLI",
    "power_uv1": "Z_FFT_abs_bandpower_uV2",
}
CANDIDATE_BANDS = [
    "Delta", "Theta", "Alpha", "Beta", "HighBeta",
    "Alpha1", "Alpha2", "Beta1", "Beta2", "Beta3",
]

# Per-patient summary stats fed to the LR scorer. Keep aligned with Phase 2
# so selection conditions match the downstream feature shape.
SELECTION_SUMMARY_FUNCS = ["mean", "std"]
SUMMARY_FUNC_MAP = {
    "mean":   np.nanmean,
    "std":    np.nanstd,
    "median": np.nanmedian,
    "p90":    lambda a: np.nanpercentile(a, 90),
    "p95":    lambda a: np.nanpercentile(a, 95),
}

OUT_DIR = PROJECT_DIR / "pipeline" / "results" / "phase1" / f"fold_{FOLD_INDEX}"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# %% [Helpers]
def patient_file_map() -> dict[int, str]:
    """Map patient ID -> xlsx path."""
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


def per_patient_band_features(
    pids: Sequence[int],
    files: dict[int, str],
    sheet_name: str,
    band: str,
    summary_funcs: Sequence[str],
) -> np.ndarray | None:
    """Build (n_patients, len(summary_funcs)) feature matrix for ONE band.

    Per-patient summaries → no cross-patient stats → leakage-safe.
    Returns None if any patient is missing the band column.
    """
    rows = []
    for p in pids:
        df = pd.read_excel(files[int(p)], sheet_name=sheet_name, index_col=0)
        if band not in df.columns:
            return None
        col = df[band].values.astype(float)
        rows.append([SUMMARY_FUNC_MAP[fn](col) for fn in summary_funcs])
    return np.nan_to_num(np.array(rows, dtype=float))


def score_band_one_fold(
    X: np.ndarray, y: np.ndarray, tr_idx: np.ndarray, va_idx: np.ndarray,
) -> float:
    """Fit LR on the fold's train slice; score balanced_accuracy on the val slice."""
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=10000,
                                   random_state=RANDOM_SEED)),
    ])
    pipe.fit(X[tr_idx], y[tr_idx])
    return balanced_accuracy_score(y[va_idx], pipe.predict(X[va_idx]))


def apply_selection_rule(counts: Counter) -> list[str]:
    if SELECTION_RULE == "top_k":
        return [b for b, _ in counts.most_common(SELECTION_K)]
    if SELECTION_RULE == "threshold":
        return [b for b, c in counts.items() if c >= SELECTION_THRESHOLD]
    raise ValueError(f"Unknown SELECTION_RULE: {SELECTION_RULE}")


def select_bands_for_group(
    sheet_label: str,
    sheet_name: str,
    train_ids: np.ndarray,
    y_train: np.ndarray,
    files: dict[int, str],
) -> tuple[list[str], list[str], dict[str, int]]:
    """Run the K-fold band vote for one feature group.

    Returns (selected_bands, fold_winners, win_counts).
    """
    print(f"\n[Phase 1] Group: {sheet_label} ({sheet_name})")

    # Pre-compute per-band feature matrices once; reused across folds.
    band_X: dict[str, np.ndarray] = {}
    for band in CANDIDATE_BANDS:
        X = per_patient_band_features(
            train_ids, files, sheet_name, band, SELECTION_SUMMARY_FUNCS,
        )
        if X is not None:
            band_X[band] = X
    if not band_X:
        print("  [WARN] no bands available across all train patients; skipping.")
        return [], [], {}
    print(f"  Available bands: {list(band_X)}")

    cv = StratifiedGroupKFold(n_splits=SELECTION_FOLDS, shuffle=True,
                              random_state=RANDOM_SEED)
    groups = np.asarray(train_ids, dtype=int)

    fold_winners: list[str] = []
    for fold_i, (tr_idx, va_idx) in enumerate(
        cv.split(train_ids, y_train, groups=groups), 1
    ):
        scores = {b: score_band_one_fold(X, y_train, tr_idx, va_idx)
                  for b, X in band_X.items()}
        winner = max(scores, key=scores.get)
        fold_winners.append(winner)
        print(f"  Fold {fold_i}/{SELECTION_FOLDS}: winner={winner} "
              f"(bal_acc={scores[winner]:.3f})")

    counts = Counter(fold_winners)
    selected = apply_selection_rule(counts)
    print(f"  Win counts: {dict(counts)}")
    print(f"  Selected ({SELECTION_RULE}): {selected}")
    return selected, fold_winners, dict(counts)


# %% [Main]
def main() -> None:
    files = patient_file_map()
    train_ids, val_ids, labels = load_split(FOLD_INDEX)

    # Leakage guard — Phase 1 must never see a val patient.
    val_set = frozenset(int(p) for p in val_ids)
    if val_set & set(int(p) for p in train_ids):
        raise RuntimeError("LEAKAGE: val IDs present in Phase 1 train slice.")

    y_train = np.array([labels[int(p)] for p in train_ids])
    print(f"Fold {FOLD_INDEX}: {len(train_ids)} train "
          f"(pos={(y_train==1).sum()}, neg={(y_train==0).sum()}) | "
          f"{len(val_ids)} val (NOT loaded here)")

    selected_per_group: dict[str, list[str]] = {}
    winners_per_group:  dict[str, list[str]] = {}
    counts_rows: list[dict] = []

    for sheet_label, sheet_name in SHEETS.items():
        bands, winners, counts = select_bands_for_group(
            sheet_label, sheet_name, train_ids, y_train, files,
        )
        selected_per_group[sheet_label] = bands
        winners_per_group[sheet_label] = winners
        for band in CANDIDATE_BANDS:
            counts_rows.append({
                "group":    sheet_label,
                "band":     band,
                "wins":     int(counts.get(band, 0)),
                "n_folds":  SELECTION_FOLDS,
                "selected": band in bands,
            })

    if not any(selected_per_group.values()):
        raise RuntimeError("No bands selected in any group — loosen the rule.")

    # --- Persist outputs ---
    with open(OUT_DIR / "selected_bands.json", "w") as f:
        json.dump({
            "fold_index": FOLD_INDEX,
            "rule":       SELECTION_RULE,
            "k":          SELECTION_K,
            "threshold":  SELECTION_THRESHOLD,
            "n_folds":    SELECTION_FOLDS,
            "summary_funcs": SELECTION_SUMMARY_FUNCS,
            "sheets":     SHEETS,
            "selected":   selected_per_group,
        }, f, indent=2)
    with open(OUT_DIR / "fold_winners.json", "w") as f:
        json.dump(winners_per_group, f, indent=2)
    pd.DataFrame(counts_rows).to_csv(OUT_DIR / "win_counts.csv", index=False)

    print(f"\nSaved Phase 1 outputs to {OUT_DIR}")
    print(f"Selected bands per group: {selected_per_group}")


if __name__ == "__main__":
    main()
