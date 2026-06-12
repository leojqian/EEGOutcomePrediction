# Methods (detailed altitude)

## 2.1 Study cohort and design

The analysis cohort comprised **107 patients** enrolled in a single-center, randomized clinical trial of three treatments for chemotherapy-induced peripheral neuropathy (CIPN)–associated pain: neurofeedback (NFB), duloxetine (DL), and combined NFB+DL (Figure 1). Each patient was randomized to one of the three treatment arms at the baseline visit. The inclusion criterion for the present analysis was the availability of (i) a baseline resting-state EEG recording and (ii) paired pain-unpleasantness ratings at baseline (T1) and post-treatment (T2). Pain unpleasantness was measured on a 0–10 numeric rating scale (NRS).

Two baseline-predictable endpoints were defined (Figure 1, panels 4–5):

- **Task A — baseline pain.** Predict the patient's T1 pain-unpleasantness rating from baseline EEG features plus age and self-reported neuropathy duration. Because no treatment has yet been delivered at T1, T1 pain is a legitimate prediction target for a pre-treatment EEG biomarker.
- **Task B — pain change (primary).** Predict the clinically meaningful change in pain from baseline EEG plus baseline clinical scalars, where the target is the post-minus-pre difference `ΔPain = Pain(T2) − Pain(T1)`. Task B is the focus of the modeling pipeline described below; the same feature-construction and cross-validation machinery applies to Task A.

For Task B, `ΔPain` was dichotomized at the approximate minimum clinically important difference of −2 NRS points (`LABEL_THRESHOLD = −2.00`, [A_make_splits_v3.py](pipeline/A_make_splits_v3.py#L29)): patients with `ΔPain ≥ −2` were labeled **non-improvers (class 1)** and the remainder **improvers (class 0)**. The resulting cohort is approximately class-balanced, so balanced accuracy and majority-class accuracy nearly coincide at 0.50 as the chance reference.

## 2.2 Resting-state EEG acquisition

Baseline EEG was acquired from **19 scalp electrodes** in an eyes-closed resting state of approximately five minutes' duration (Figure 1, panel 2). Detailed acquisition parameters (reference montage, sampling rate, filtering, and artifact-rejection criteria) are summarized briefly here and will be specified in full by the EEG-acquisition collaborators in a companion methods note. Per-patient spectral and connectivity features were precomputed (Welch periodogram for power spectra; magnitude-squared coherence and phase-lag index for connectivity) and stored as one feature workbook per patient (`processeddata/CIPN3{NNN}_*.xlsx`), which the modeling pipeline consumes directly. The feature-extraction code is at [scripts/eeg_feature_extraction.py](scripts/eeg_feature_extraction.py).

## 2.3 EEG and clinical features

The EEG feature space (Figure 1, panel 3; Figure 2) spans three families: (a) **spectral power** in canonical frequency bands (absolute and relative band power, δ/θ/α/β/γ), (b) **functional connectivity** via inter-channel coherence, and (c) other summaries including total power (1–45 Hz), ultra-slow/very-low-frequency power (0.1–1 Hz), peak frequency, spectral entropy, and the phase-lag index (PLI).

The modeling pipeline draws on three pre-aggregated, z-scored feature sheets from each patient's workbook ([B_band_selection.py](pipeline/B_band_selection.py#L72)):

| Sheet | Content | Dimensions |
|---|---|---|
| `Z_FFT_abs_bandpower_uV2` | z-scored absolute band power | 19 channels × bands |
| `Z_FFT_Coherence` | z-scored inter-channel coherence | 171 channel pairs × bands |
| `Z_FFT_PhaseLag_PLI` | z-scored debiased phase-lag index | 171 channel pairs × bands |

Band selection draws from a 10-band candidate set `{Delta, Theta, Alpha, Beta, HighBeta, Alpha1, Alpha2, Beta1, Beta2, Beta3}` (`CANDIDATE_BANDS`, [B_band_selection.py:78](pipeline/B_band_selection.py#L78)).

In addition, six baseline clinical/treatment scalars — all known at the baseline visit and therefore admissible pre-treatment predictors — are appended to the feature vector in Module C ([C_train_eval_v3.py:72](pipeline/C_train_eval_v3.py#L72)): three continuous variables (T1 pain unpleasantness `pain_t1`, `age`, self-reported neuropathy duration `neurop_months`) and a one-hot encoding of the randomized treatment assignment (`mod_NFB`, `mod_DL`, `mod_NFB_DL`). Free-text entries in the neuropathy-duration field are coerced numerically (`pd.to_numeric(..., errors="coerce")`), and residual missing values are imputed with the per-fold training median (§2.6). Treatment assignment was made at baseline prior to any treatment delivery, so it is a baseline-known predictor rather than a leaked outcome.

## 2.4 The ABC pipeline — overview and leakage control

The analysis is organized as three sequential modules, each a single script that writes on-disk artifacts the next module consumes:

- **A — split generation** ([A_make_splits_v3.py](pipeline/A_make_splits_v3.py)): writes the 5-fold train/validation patient-ID arrays once.
- **B — per-fold band selection** ([B_band_selection.py](pipeline/B_band_selection.py)): for each outer fold, selects bands using only that fold's training patients.
- **C — per-fold feature engineering, model fitting, threshold calibration, and held-out evaluation** ([C_train_eval_v3.py](pipeline/C_train_eval_v3.py)).

Modules B and C are **re-executed independently inside every outer fold**, on that fold's training patients only, so that no fold-local decision (band selection, model selection, threshold calibration) is informed by its own held-out patients. A driver script chains the per-fold C runs and aggregates the five `test_metrics.json` files into a cross-fold summary ([E_run_all_folds_v3mod.py](pipeline/E_run_all_folds_v3mod.py)).

**Leakage control.** A runtime guard is instantiated at the top of Modules B and C from the validation-ID list (`VAL_ID_SET = frozenset(...)`) and invoked via `assert_no_val_leak(pids, where)` at every data-loading site; it raises a `RuntimeError` if a validation ID ever enters a training slice ([B_band_selection.py:108](pipeline/B_band_selection.py#L108), [C_train_eval_v3.py:149](pipeline/C_train_eval_v3.py#L149)). This makes accidental contamination fail loudly rather than silently inflate reported performance. All random seeds are fixed (`random_state = 42`) at the split-generation, band-tally, model-search, calibration, and classifier-initialization stages.

## 2.5 Module A — Cross-validation splits

A 5-fold split of patient identifiers was generated once with `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`. To prevent the dominant predictor (baseline pain) from drifting between train and validation, the split is stratified on a **joint stratum of outcome label × `pain_t1` tercile** ([A_make_splits_v3.py:67-74](pipeline/A_make_splits_v3.py#L67-L74)): the cohort's `pain_t1` values are binned into terciles, and the composite stratum `label × pain_bin` (six strata) is passed to the stratifier. This ensures that every fold's validation set has approximately the same outcome-label balance **and** the same low/mid/high baseline-pain composition. (The tercile cutoffs are computed once on the full cohort for split construction only; because `pain_t1` is itself a model feature, this does not leak outcome information into training.) Per-fold train/validation ID arrays, a patient→label dictionary (`labels.json`), and a split manifest (`manifest.json`) are written to `pipeline/splits_v3/`. Each fold has ~85–86 training and ~21–22 validation patients.

## 2.6 Module B — Per-fold band selection

For each outer fold and each of the three EEG sheets, Module B selects exactly **2 bands** (`SELECTION_MODE = "top2_per_sheet"`, `TOP_K_PER_SHEET = 2`) using only that fold's training patients. Selection proceeds in three stages:

1. **Inner band-tally cross-validation.** The fold's training patients are partitioned by a 7-fold `StratifiedGroupKFold` (`INNER_CV_FOLDS = 7`), stratified on the outcome label with the patient identifier as the group so each patient appears in exactly one tally fold ([B_band_selection.py:168](pipeline/B_band_selection.py#L168)).

2. **Exhaustive subset search per tally fold.** For each tally fold and each of the 1,023 non-empty subsets of the 10 candidate bands, a per-patient feature vector is built by concatenating, for every band in the subset, a 5-element per-band summary `[col_mean, col_std, global_mean, global_std, global_median]` of that band's sheet column (`feats_from_single_band_column`, [B_band_selection.py:131](pipeline/B_band_selection.py#L131)). A penalized logistic regression (`StandardScaler → LogisticRegression`, `class_weight="balanced"`, `max_iter=50000`, `tol=1e-3`) is fit with a nested inner 5-fold `StratifiedGroupKFold` `GridSearchCV` (`BAND_GS_INNER_FOLDS = 5`) over the grid `solver ∈ {liblinear, saga}`, `penalty ∈ {ℓ1, ℓ2}`, `C ∈ {1e-3, 1e-2, 1e-1, 1, 10, 100}`, scored by balanced accuracy.

3. **Tally and selection.** The winning subset of each tally fold is the one with the highest inner-CV balanced accuracy. After all seven tally folds, each band's count equals the number of winning subsets that contained it; the two top-tallied bands per sheet are retained (ties broken by canonical band order). This yields exactly 2 bands × 3 sheets = **6 bands per outer fold**.

Per-fold artifacts (`{sheet}_band_counts.csv`, `{sheet}_fold_winners.json`, `{sheet}_selected.json`) are written under `pipeline/bands_v3/fold_{k}/top2_per_sheet/`.

## 2.7 Module C — Feature engineering, model fitting, threshold calibration, evaluation

Module C consumes the fold's split (A) and selected bands (B) and proceeds in five stages.

**Feature construction.** For each patient and each selected band, two summary statistics — the **mean** and **standard deviation** across the sheet column (channels for power, channel pairs for connectivity; `SUMMARY_FUNCS = ["mean", "std"]`) — are computed. Six bands × two summaries = **12 EEG features**, plus the **6 baseline clinical/treatment scalars**, for a fixed dimension of **18 features per patient**. Missing values in the three continuous clinical scalars are imputed with the **fold-`k` training median only**, applied identically to that fold's validation patients ([C_train_eval_v3.py:243-250](pipeline/C_train_eval_v3.py#L243-L250)); EEG-side missingness (an absent band column) is zero-filled.

**Model search.** Four model families are searched inside an inner 5-fold `StratifiedGroupKFold` (`MODEL_CV_FOLDS = 5`, patient ID as group) over the fold's training patients, each scored by balanced accuracy with `GridSearchCV(refit=True)`:

- **Penalized logistic regression** — `StandardScaler → LogisticRegression` (`class_weight="balanced"`); grid over `solver ∈ {liblinear, saga}`, `penalty ∈ {ℓ1, ℓ2, elastic net}`, `C ∈ {1e-4 … 500}` (13 values), elastic-net `l1_ratio ∈ {0.1, 0.3, 0.5, 0.7, 0.9}`.
- **Random forest** — `RandomForestClassifier(class_weight="balanced")`; grid over `n_estimators ∈ {200, 400, 800}`, `max_depth ∈ {None, 3, 5, 8}`, `min_samples_split ∈ {2, 5, 10}`, `min_samples_leaf ∈ {1, 2, 4}`, `max_features ∈ {sqrt, 0.5}`.
- **Support vector machine** — `StandardScaler → SVC(class_weight="balanced", probability=True)`; RBF (`C ∈ {0.1, 1, 10, 100}`, `gamma ∈ {scale, 0.01, 0.1, 1}`) and linear (`C ∈ {0.1, 1, 10, 100}`) sub-grids; probabilities via Platt scaling.
- **Gradient-boosted trees** — `XGBClassifier(eval_metric="logloss", tree_method="hist")`; grid over `n_estimators ∈ {50, 100, 200, 300}`, `max_depth ∈ {2…7}`, `learning_rate ∈ {0.005 … 0.2}`, `subsample ∈ {0.7, 0.85, 1.0}`, `colsample_bytree ∈ {0.7, 0.85, 1.0}`, `reg_lambda ∈ {0.1, 1, 5, 10}`.

The best configuration of each family is refit on the full fold-`k` training set, and a boundary check flags any selected hyperparameter that lands on a grid edge so the grid can be extended in follow-up work.

**Ensemble adjudication.** A soft-vote ensemble averaging the `predict_proba` outputs of the four refit family champions is evaluated head-to-head against the single best family. To choose between them without touching validation data, inner-CV out-of-fold (OOF) probabilities are computed for each family and the ensemble via `cross_val_predict` on the same inner 5-fold splits; the candidate with the higher OOF balanced accuracy (maximized over thresholds) becomes the fold's final classifier, with ties broken toward the single-best family ([C_train_eval_v3.py:317-360](pipeline/C_train_eval_v3.py#L317-L360)).

**Threshold calibration.** The decision threshold τ is calibrated on the chosen classifier's inner-CV OOF training probabilities (not on refit training predictions, which removes a known source of overfitting when training accuracy saturates at 1.0). The balanced-accuracy-maximizing plateau `{τ : OOF-BA(τ) = max}` is enumerated over `τ ∈ {0.05, …, 0.95}`, and the threshold closest to 0.5 is retained — a robust anchor given that all inner classifiers use `class_weight="balanced"` ([C_train_eval_v3.py:362-382](pipeline/C_train_eval_v3.py#L362-L382)).

**Held-out evaluation.** Each fold ends with a single forward pass over its ~21–22 held-out validation patients, producing the predicted probability `p̂` and label `ŷ = 1[p̂ ≥ τ]`. This is the only time Module C touches the validation patients. Per-fold artifacts (`feature_list.csv`, `train_results.csv`, `test_predictions.csv`, `test_metrics.json`, `model.pkl`) are written under `pipeline/results/fold_{k}/top2_per_sheet_v3_mod/`.

## 2.8 Metrics and baselines

For each fold we report accuracy, balanced accuracy, ROC-AUC, average precision, F1 for the positive (non-improver) class, the calibrated threshold τ, and the validation confusion matrix `[[TN, FP], [FN, TP]]`. We additionally report (i) the mean ± standard deviation of each metric across the five folds and (ii) pooled out-of-fold metrics obtained by concatenating the five per-fold validation predictions into a single 107-patient vector. **ROC-AUC is the primary endpoint**: it is threshold-independent (robust to per-fold τ noise at ~21 validation patients per fold), it captures the model's ability to rank patients by risk, and it stabilizes faster than thresholded accuracy under pooling.

Two pre-specified baselines are evaluated under the identical 5-fold harness: a **majority-class baseline** (balanced accuracy 0.500 by construction, the chance reference) and a **clinical-only baseline** (Module C with the EEG features disabled, so only the baseline clinical/treatment scalars are used), which quantifies the incremental value of EEG features beyond baseline pain, age, neuropathy duration, and treatment arm.

## 2.9 Software and reproducibility

The pipeline was implemented in Python 3 using scikit-learn (cross-validation, preprocessing, logistic regression, random forest, SVM), XGBoost (gradient-boosted trees), and pandas/NumPy. All random seeds are fixed (`random_state = 42`) throughout, and every per-fold artifact (selected bands, fitted model, calibrated threshold, validation predictions, metrics) is written to a deterministic path, so any reported number can be audited end-to-end without re-running the models.
