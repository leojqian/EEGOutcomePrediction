# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

MD Anderson — EEG patient stratification. The goal is to predict treatment response in chemotherapy-induced peripheral neuropathy (CIPN) from baseline resting-state EEG, for a single-center randomized trial of neurofeedback (NFB), duloxetine (DL), and NFB+DL (n=107 patients with paired T1/T2 pain ratings). The primary task is a binary classifier of clinically meaningful pain improvement (ΔPain = Pain(T2) − Pain(T1), threshold −2 NRS).

## Environment & commands

Python 3.11 in a local venv (`venv/`). There is no `requirements.txt`; key deps installed in the venv: scikit-learn 1.8, xgboost 3.2, pandas 3.0, numpy 2.3, mne 1.11, mne-connectivity 0.7, shap 0.51, scipy, openpyxl, matplotlib. Always run with the venv interpreter:

```bash
venv/bin/python3 pipeline/<script>.py          # or: source venv/bin/activate
```

Common workflows (run from inside `pipeline/`):

```bash
# Full cross-validated experiment (chains band-selection + train/eval across 5 folds,
# writes a cross-fold summary CSV). Pick the driver matching the experiment variant:
cd pipeline && python3 E_run_all_folds_v3mod.py   # current/canonical: modality features

# Run ONE fold by hand (scripts are parameterized by env vars, not CLI args):
FOLD_INDEX=2 SELECTION_MODE=top2_per_sheet python3 B_band_selection.py
FOLD_INDEX=2 python3 C_train_eval_v3.py

# Regenerate paper figures (-> reports/abc_paper_figs/*.png):
python3 figures/make_fig1_study_design.py

# Re-extract per-patient EEG features from raw txt -> processeddata/*.xlsx:
python3 scripts/eeg_feature_extraction.py
```

There is no test suite, linter, or build step — this is a research analysis repo, not an application.

## Architecture

### Data flow
`rawdata/` (raw EEG txt) → `scripts/eeg_feature_extraction.py` (Welch PSD power, coherence, PLI via mne/scipy) → `processeddata/CIPN3{NNN}_*.xlsx` (one workbook per patient, ~17 feature sheets) → the `pipeline/` modeling stages. Patient outcomes/demographics live in `processeddata/Randomization factors and Primary outcome.xlsx`.

**Data directories (`rawdata/`, `processeddata/`, `oldprocesseddata/`) are gitignored** and are not present on a fresh clone — the pipeline cannot run without them.

### The ABC pipeline (`pipeline/`)
The core is a deliberately leakage-controlled 5-fold cross-validation, organized as three sequential stages, each a standalone script writing on-disk artifacts the next stage consumes:

- **A — split generation** (`A_make_splits.py`, `A_make_splits_v3.py`): generates 5-fold stratified train/val splits once, writes patient-ID arrays + `labels.json` + `manifest.json` to `splits/` (label-only stratification) or `splits_v3/` (joint label × baseline-pain-tercile stratification).
- **B — per-fold band selection** (`B_band_selection.py`): for each fold, selects the most predictive EEG frequency bands using that fold's training patients **only**. Writes selected bands to `bands[_v3]/fold_{k}/{mode}/`.
- **C — per-fold train/eval** (`C_train_eval*.py`): builds the feature matrix, grid-searches model families, calibrates a decision threshold, and does a single one-shot evaluation on held-out patients. Writes metrics/predictions/`model.pkl` to `results[_v3]/fold_{k}/{tag}/`.
- **E — drivers** (`E_run_all_folds*.py`): chain B and/or C across folds 0–4 via subprocess and aggregate the per-fold `test_metrics.json` into `results/all_folds_{tag}_summary.csv`.

### Env-var contract (critical)
B and C are not configured by CLI flags but by environment variables; the E drivers set them per fold. When running a stage manually you must set these to match:
- `FOLD_INDEX` — which fold (0–4).
- `SPLITS_TAG` — `""` reads `splits/` + `bands/`; `"v3"` reads `splits_v3/` + `bands_v3/`.
- `SELECTION_MODE` — `top2_per_sheet` (default), `abc_existing`, or `3ref_per_fold`.

### Leakage-control invariant (do not break)
The central design principle: every fold-local decision (band selection, model selection, threshold calibration) is made on training patients only, and a runtime guard `assert_no_val_leak(pids, where)` is called at every data-loading site in B and C — it raises if a validation patient ID ever enters a training slice. Modules B and C are re-executed independently inside every fold. Preserve this when editing; any change that lets validation data influence training should fail loudly, not silently.

### Versioned C / E lineage
Many near-identical `C_train_eval*` and `E_run_all_folds*` files coexist — they are successive experiment variants, not dead copies. Each C version has a distinct `OUTPUT_TAG`; each E driver pairs with a specific C:
- `C_train_eval.py` (v1): LogReg + XGBoost; threshold calibrated on refit-train predictions. Driver: `E_run_all_folds.py` (runs B + C).
- `C_train_eval_v2.py`: adds RandomForest + SVC and a soft-vote ensemble; threshold calibrated on inner-CV out-of-fold predictions (leakage-safe). Tag `top2_per_sheet_v2`. Drivers: `E_run_all_folds_v2.py`, and `E_run_all_folds_v3.py` (same C_v2 but with `SPLITS_TAG=v3` joint-stratified splits).
- `C_train_eval_v3.py`: v2 + three treatment-modality one-hot features (NFB/DL/NFB+DL). Tag `top2_per_sheet_v3_mod`. Driver: `E_run_all_folds_v3mod.py` — this is the variant the paper reports.
- `C_train_eval_v4.py`: v3 + Platt (sigmoid) `CalibratedClassifierCV`, no ensemble. Tag `top2_per_sheet_v4_cal`.

`phase1_band_selection.py` / `phase2_train_eval.py` are older, simpler refactors of B/C (per-band scoring, XGBoost-only); `D_3ref_singlefold.py` is a single-fold reference variant. These are superseded by the A/B/C lineage.

### Reproducibility details
- All scripts hard-code `PROJECT_DIR = Path("/Users/leoqian/mdanderson/EEGOutcomePrediction")` — update this when moving the repo.
- All random seeds are fixed (`random_state = 42`).
- Scripts are written with `# %% [...]` cell markers (notebook-style) but run as plain Python files.

## Other contents
- `paper_draft.md` — full manuscript draft. `paper_methodology.md` / `_detailed.md` / `_overview.md` — the Methods section at three levels of detail.
- `reports/abc_paper_figs/` — paper figures + SHAP CSVs. `notebooks/` — exploratory and legacy analysis.
- `report.md` — earlier data-quality / exploratory write-up.
