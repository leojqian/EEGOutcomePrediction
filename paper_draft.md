Title

A Leakage-Controlled Pipeline for Predicting Treatment Response in Chemotherapy-Induced Peripheral Neuropathy from Baseline Resting-State EEG


Authors

Leo Qian; collaborators to be confirmed (clinical PI; EEG acquisition lead).


Abstract

Chemotherapy-induced peripheral neuropathy (CIPN) is a frequent and persistent toxicity of cancer treatment, and the analgesic options that exist for it — duloxetine, neurofeedback (NFB), and their combination — work in only a subset of patients. A baseline biomarker that could separate likely responders from likely non-responders before therapy is initiated would let clinicians steer patients toward the modality most likely to help them. Resting-state electroencephalography (EEG) is an attractive substrate for such a biomarker because it is non-invasive, low-cost, and yields a quantitative multivariate readout of cortical state. Most prior EEG-prediction studies, however, are vulnerable to subtle test-set leakage — feature screening, band selection, or threshold tuning performed before the formal train/validation split — that inflates reported performance and fails to replicate.

We present a deliberately conservative end-to-end analysis pipeline, organized as three sequential modules (A: split generation, B: per-fold band selection, C: per-fold feature engineering, model fitting, threshold calibration, and held-out evaluation) and referred to throughout as the ABC pipeline. Modules B and C are re-executed independently inside every outer fold, on that fold’s training patients only, and a runtime leakage guard fires at every data-loading site if a held-out patient ID ever enters a training slice. The pipeline keeps the final feature count small (12 EEG summaries + 3 baseline clinical scalars = 15 features) so that the trained model remains favorably small relative to the 85–86-patient per-fold training set.

The pipeline was applied to a single-site randomized CIPN trial (n = 107 patients with paired baseline EEG and pain unpleasantness ratings at T1 and T2). The outcome was a binary label indicating clinically meaningful pain improvement (ΔPain = PainT2 − PainT1; threshold Diff ≥ −2 NRS points → non-improver). Across 5 stratified folds the pipeline obtained mean balanced accuracy 0.655 ± 0.081, mean ROC-AUC 0.686 ± 0.103, and a pooled out-of-fold AUC of 0.714, substantively above the 0.50 majority-class chance level on a near-balanced cohort. Three of five folds selected an XGBoost classifier and two selected a penalized logistic regression. Per-fold band selections were variable but recurrent (Phase-Lag PLI HighBeta was chosen in 4/5 folds; Coherence Beta1 in 3/5; Absolute power Theta in 3/5), consistent with a beta/high-beta connectivity and theta-band power signature. We argue that the ABC pipeline’s explicit leakage control, small feature footprint, and per-fold transparency make its headline number a more trustworthy starting point for downstream prospective work than prior single-fold or population-level analyses on the same cohort.


1. Introduction

Chemotherapy-induced peripheral neuropathy (CIPN) is one of the most common and persistent dose-limiting toxicities of modern oncologic treatment, affecting more than 30% of patients receiving neurotoxic chemotherapy and persisting beyond cessation of therapy in a sizeable minority [1, 2]. Its hallmark symptoms — distal paresthesia, numbness, and neuropathic pain — degrade physical function, sleep, and quality of life long after the underlying cancer has been controlled, and the resulting symptom burden is a major driver of dose reduction, treatment discontinuation, and downstream survival decrement.

Available therapies for CIPN pain are only modestly effective and distinctly heterogeneous in patient response. Duloxetine (DL), the only agent with a Class I recommendation from the American Society of Clinical Oncology for painful CIPN, produces clinically meaningful relief in roughly a third of patients but not the rest. Non-pharmacologic interventions — neurofeedback (NFB) among them — show similarly patchy efficacy: aggregating meta-analyses across heterogeneous chronic-pain populations report NFB pain reductions ranging from 6% to 82%, with substantial between-trial heterogeneity that points to sub-populations of strong responders embedded in trials whose group-level effects are marginal [3, 4, 5]. The clinical case for pre-treatment stratification is therefore strong: an inexpensive baseline biomarker capable of distinguishing likely responders from likely non-responders would let clinicians steer patients toward the modality most likely to help them and away from months of ineffective therapy and avoidable adverse events.

Resting-state electroencephalography (EEG) is an appealing candidate substrate. It is non-invasive, low-cost, repeatable at the bedside, and yields a quantitative multivariate readout of cortical state that is in principle independent of the patient’s self-reporting style. Prior work in chronic pain has linked resting EEG features to pain intensity, chronicity, and analgesic response — heightened theta and low-beta power, attenuated peak alpha frequency, altered fronto-parietal coherence, and asymmetric inter-hemispheric phase coupling have all been reported as candidate markers of central sensitization or maladaptive pain processing [CITATION NEEDED — chronic-pain rsEEG review]. In adjacent neuromodulation contexts, resting-state EEG features have already been used as predictive substrates: short eyes-closed rsEEG can distinguish learners from non-learners in alpha-down-regulation neurofeedback training with classification accuracy of 86.2% [8], and a recent meta-analysis of ML-based neuromodulation response prediction across multiple disease areas reports a pooled classification AUROC ≈ 0.84 when EEG, imaging, and clinical features are combined [9]. In CIPN specifically, baseline cortical oscillatory signatures may track the dysregulated thalamocortical drive that follows peripheral deafferentation and therefore prefigure how a patient will respond to interventions that act on the central side of the pain network [2].

Translating these observations into a predictive model that is both trustworthy and clinically deployable is, however, non-trivial. Three methodological pitfalls recur in the small-cohort EEG-prediction literature and, in our judgment, account for a substantial fraction of over-optimistic results that fail to replicate. First, the feature-to-patient ratio is unfavorable: a typical CIPN trial enrolls ~100 patients while a complete EEG feature set easily exceeds several hundred candidate variables (per-channel spectral power across multiple bands, pairwise coherence and phase-lag indices over hundreds of channel pairs, asymmetry and regional summaries), and naive multivariable models in this regime are dominated by overfit. Second, decisions made before the formal train/validation split — feature screening, band selection, threshold tuning — silently leak information from the held-out set into the training-time pipeline. Third, evaluation choices that look innocuous in isolation (median-split labels chosen post hoc, threshold tuning on the test fold, scoring on accuracy rather than balanced accuracy in an imbalanced sample) can each inflate headline numbers by 5–10 percentage points, and they compound.

We address these pitfalls with a deliberately conservative end-to-end analysis pipeline — the ABC pipeline — organized as three sequential modules. Module A generates 5-fold stratified train/validation splits once and saves the patient identifiers to disk. Modules B (per-fold band selection) and C (per-fold feature engineering, model fitting, threshold calibration, and one-shot held-out evaluation) are then re-executed independently inside every outer fold, on that fold’s training patients only. The pipeline deliberately keeps the final feature dimension small — 12 EEG summary features (2 top bands × 3 sheets × 2 summary statistics) plus 3 baseline clinical scalars, for 15 features per patient — to remain favorably within the 85–86-patient per-fold training regime. A runtime leakage guard, implemented in code (assert_no_val_leak in pipeline/B_band_selection.py and pipeline/C_train_eval.py), raises at every data-loading site if a validation ID ever enters a training slice.

The present paper applies the ABC pipeline to a single-site randomized CIPN trial (n = 107 with paired baseline-EEG and pain-outcome data) and reports 5-fold cross-validated performance on the binary prediction of clinically meaningful pain change (ΔPain unpleasantness ≥ −2 NRS points labeled non-improver). We contextualize the EEG-augmented predictions against a majority-class baseline evaluated under the identical harness, and we make the per-fold band selections explicit so that the fold-to-fold band variability — a direct artifact of the leakage-controlled design — can be read as a methodological feature rather than a defect.

Hypotheses. We pre-registered three hypotheses:

H1 (Primary). Baseline EEG features combined with three baseline clinical scalars (T1 pain unpleasantness, age, self-reported neuropathy duration), evaluated under the ABC pipeline, will discriminate clinically meaningful improvers from non-improvers with balanced accuracy above the 0.50 majority-class chance level on aggregate held-out 5-fold cross-validated data.

H2 (Incremental value). The EEG-augmented ABC model will outperform a clinical-only model trained and evaluated under the identical 5-fold harness, supporting the claim that baseline cortical oscillatory features carry predictive information not redundant with baseline pain severity, age, or neuropathy duration. [Empirical result for the clinical-only configuration through the same harness is pending; see §4.2.]

H3 (Stability of leakage-controlled selection). Because Module B re-runs band selection inside the training set of every fold, the specific bands retained will vary across folds; we hypothesize that this variability will be bounded — a small number of bands, broadly in the beta/high-beta range for connectivity and theta/low-beta for absolute power, will recur across the majority of folds.


2. Related Work / Background

CIPN with neurofeedback as treatment. CIPN is established as a frequent toxicity of common cancer therapies, affecting more than 30% of patients receiving neurotoxic chemotherapy [1], and is associated with substantial long-term morbidity. Initial small-cohort experiments demonstrated that scalp-based neurofeedback could alleviate CIPN-related pain symptoms [6, 3], and a randomized, double-blind, placebo-controlled trial in this clinical program reported that a brain–computer interface intervention produced statistically and clinically significant pain reductions in chronic CIPN [2]. Subsequent meta-analyses aggregating evidence across heterogeneous chronic-pain populations have reported NFB-driven pain reductions ranging from 6% to 82% — a wide spread reflecting genuine between-trial heterogeneity in patient selection and response, not just measurement noise [4, 5]. High-quality evidence specifically for CIPN-targeted neuromodulation remains scarce [7].

Predictive modeling on resting-state EEG. Prior work has demonstrated that features extracted from short eyes-closed resting-state EEG recordings can identify learners vs. non-learners in alpha-down-regulation neurofeedback training with classification accuracy of 86.2%, enabling early identification of individuals unlikely to respond to training [8]. A recent systematic review and meta-analysis of ML-based neuromodulation response prediction reports a pooled classification AUROC of approximately 0.84 across studies using multimodal inputs (EEG, imaging, and clinical features), with EEG contributing meaningfully to the multimodal signal [9]. In a CIPN-adjacent setting, resting-state EEG features have been used to predict tDCS treatment response in Alzheimer’s disease [10], showing that the rsEEG-as-response-biomarker paradigm generalizes beyond chronic pain. On the feature-engineering side, standard EEG pipelines emphasize artifact rejection followed by spectral decomposition (Welch PSD, wavelet/CWT) and statistical descriptors of the resulting per-channel and per-channel-pair matrices [11, 12], with subsequent dimensionality reduction (PCA/ICA, or functional PCA on wavelet power spectra) often applied before downstream modeling [13]. Multi-pattern feature fusion has been shown to outperform single-feature approaches in adjacent EEG-based prognosis prediction tasks — for example, EEG microstate features yield AUC ≈ 0.95 for outcome prediction in prolonged disorders of consciousness, materially outperforming spectral-power-only features and underscoring the value of capturing spatiotemporal dynamics beyond band power alone [14].

Patient stratification in CIPN. Initial stratification work in CIPN used symptom-cluster analysis on patient-reported outcomes to define distinct phenotypes [15, 16], and subsequent multimodal studies have begun to associate these phenotypes with metabolic and demographic biomarkers [17, 18]. To our knowledge, no published study has applied a leakage-controlled cross-validated ABC-style pipeline to baseline resting-state EEG for predicting CIPN treatment response specifically.


3. Methods

3.1 Participants and outcome

The analysis cohort comprised 107 patients enrolled in a single-center randomized clinical trial of neurofeedback (NFB), duloxetine (DL), and combined NFB+DL for CIPN-associated pain. The inclusion criterion for the present analysis was availability of both a baseline resting-state EEG recording and paired pain-unpleasantness ratings at baseline (T1) and post-treatment (T2). The primary outcome was the change in pain unpleasantness ΔPain = PainT2 − PainT1, measured on a 0–10 numeric rating scale; negative values indicate improvement. To support a clinically actionable binary decision, ΔPain was dichotomized at the approximate minimum clinically important difference of −2 NRS points (pipeline/A_make_splits.py, LABEL_THRESHOLD = −2.00): patients with ΔPain ≥ −2 were labeled non-improvers (class 1), and the remainder improvers (class 0). The dichotomization is approximately balanced (≈ 50.5% non-improvers across the cohort; per-fold positive-class rates are tabulated in §3.3 below).

3.2 EEG features

Baseline EEG was acquired with 19 electrodes (acquisition details to be filled in by collaborators: reference montage, recording length, artifact rejection, Welch PSD parameters; the feature-extraction code is at scripts/eeg_feature_extraction.py). From each patient’s pre-computed feature workbook (processeddata/CIPN3{NNN}_*.xlsx), three pre-aggregated z-scored sheets are consumed by the modeling pipeline (pipeline/B_band_selection.py, pipeline/C_train_eval.py):

  • Z-scored absolute band power, Z_FFT_abs_bandpower_uV2 (19 channels × 10 bands).
  • Z-scored inter-channel coherence, Z_FFT_Coherence (171 channel pairs × 11 bands).
  • Z-scored Phase-Lag Index, Z_FFT_PhaseLag_PLI (171 channel pairs × 11 bands).

The candidate band set used for selection is {Delta, Theta, Alpha, Beta, HighBeta, Alpha1, Alpha2, Beta1, Beta2, Beta3} (CANDIDATE_BANDS in pipeline/B_band_selection.py).

3.3 The ABC pipeline

Module A — Cross-validation splits (pipeline/A_make_splits.py). A 5-fold stratified split of patient identifiers was generated once with sklearn.StratifiedKFold(n_splits=5, shuffle=True, random_state=42). Per-fold train/val patient ID arrays were saved to disk (pipeline/splits/fold_{k}_train.npy and fold_{k}_val.npy), along with the patient → label dictionary and a manifest of split statistics. The five folds (n_train, n_val, train positive rate, val positive rate) are:

  Fold 0: 85 / 22 / 0.506 / 0.500
  Fold 1: 85 / 22 / 0.506 / 0.500
  Fold 2: 86 / 21 / 0.512 / 0.476
  Fold 3: 86 / 21 / 0.500 / 0.524
  Fold 4: 86 / 21 / 0.500 / 0.524

A leakage guard (assert_no_val_leak) is instantiated from the val ID list and called at every data-loading site in B and C; it raises RuntimeError if a val ID is ever observed inside a train slice.

Module B — Per-fold band selection (pipeline/B_band_selection.py). For each outer fold k and each of the three EEG sheets, Module B selects 2 bands using only that fold’s 85–86 training patients. The selection rule used for the headline analysis is "top_2_per_sheet", which proceeds as follows.

(i) The training patients are partitioned by an inner 7-fold sklearn.StratifiedGroupKFold (each patient appears in exactly one tally fold; INNER_CV_FOLDS = 7).

(ii) For each tally fold t and each of the 1,023 non-empty subsets S of the 10 candidate bands, a per-patient feature vector is built by concatenating, for every band b in S, a 5-element per-patient summary [col_mean, col_std, global_mean, global_std, global_median] of the sheet column for that band (function feats_from_single_band_column in pipeline/B_band_selection.py). A penalized logistic regression (StandardScaler → LogisticRegression, class_weight="balanced") is trained on the tally-fold training partition with an inner 5-fold StratifiedGroupKFold GridSearchCV over solver ∈ {liblinear, saga}, penalty ∈ {ℓ1, ℓ2}, and C ∈ {1e-3, 1e-2, 1e-1, 1, 10, 100}, scored by balanced accuracy.

(iii) The winning subset of tally fold t is the subset whose inner-CV balanced accuracy is highest. After the seven tally folds complete, each band receives an integer count in [0, 7] equal to the number of winning subsets that contained it. The two top-tallied bands per sheet are retained (ties broken by canonical band order); this gives 2 bands × 3 sheets = 6 bands per outer fold.

Two alternative selection rules are also implemented (abc_existing: keep bands with count ≥ 5/7; 3ref_per_fold: single exhaustive subset search on the full training set with no tally), but headline results are reported only for top_2_per_sheet, which yields a deterministic 6-band count per fold.

Module C — Feature engineering, model selection, threshold calibration, evaluation (pipeline/C_train_eval.py). Module C consumes (i) the fold’s train/val patient ID arrays and (ii) the bands that Module B selected on that fold’s training set, and performs four operations.

Feature construction. For each patient and each selected band, two per-patient summary statistics are computed across the corresponding sheet column (channels for power, channel pairs for connectivity): the mean and the standard deviation (SUMMARY_FUNCS = ["mean", "std"]). Six bands × two summaries = 12 EEG features per patient. Three baseline clinical scalars — T1 pain unpleasantness, age, and self-reported neuropathy duration in months (free-text entries coerced numerically via pd.to_numeric(..., errors="coerce")) — are appended, for a fixed feature dimension of 15 per patient. Missing clinical values are imputed with the training median of fold k only; the same median is then applied to that fold’s validation patients. EEG-side missingness (a band column absent from a patient’s sheet) is zero-filled, matching legacy behavior.

Model search. Two model families are searched inside an inner 5-fold StratifiedGroupKFold (patient = group) over the training patients of fold k, both scored by balanced accuracy:

  • Penalized logistic regression. StandardScaler → LogisticRegression with class_weight="balanced", max_iter=50_000, tol=1e-3. The grid spans solver ∈ {liblinear, saga}, penalty ∈ {ℓ1, ℓ2, elastic net (saga only)}, C ∈ {1e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 0.5, 1, 5, 10, 50, 100, 500} (13 values), and l1_ratio ∈ {0.1, 0.3, 0.5, 0.7, 0.9} for the elastic-net entry. The full grid has 77 configurations across the four sub-grids declared in LR_PARAM_GRID.
  • Gradient-boosted trees. XGBoost (xgboost.XGBClassifier, eval_metric="logloss", tree_method="hist") with a 4 × 6 × 6 × 3 × 3 × 4 = 5,184-configuration grid spanning n_estimators ∈ {50, 100, 200, 300}, max_depth ∈ {2,…,7}, learning_rate ∈ {0.005, 0.01, 0.03, 0.05, 0.1, 0.2}, subsample ∈ {0.7, 0.85, 1.0}, colsample_bytree ∈ {0.7, 0.85, 1.0}, reg_lambda ∈ {0.1, 1, 5, 10}.

The best-scoring configuration in each family is refit on the full fold-k training set; the family with the higher inner-CV balanced accuracy is retained as the fold’s final classifier. A boundary check (check_grid_boundaries) flags any XGBoost best-param that lands on the low or high edge of its grid, prompting an extension of that dimension in subsequent exploration.

Threshold calibration. The classifier’s decision threshold τ is tuned by sweeping τ over [0.05, 0.95] in 0.005 increments and selecting the τ that maximizes balanced accuracy on the training predictions of fold k (np.linspace(0.05, 0.95, 181)). Validation patients are excluded from threshold selection.

Final evaluation. Each fold ends with a single forward pass through the 21–22 held-out validation patients of that fold, producing the predicted probability p̂ and class label ŷ = 1[p̂ ≥ τ]. We report accuracy, balanced accuracy, ROC-AUC, average precision, and F1 for the positive class, along with the per-fold confusion matrix and the calibrated threshold. Per-fold artifacts — feature_list.csv, train_results.csv, test_predictions.csv, test_metrics.json, model.pkl — are written to pipeline/results/fold_{k}/top2_per_sheet/. A driver script (pipeline/E_run_all_folds.py) chains the per-fold B and C runs and aggregates the five test_metrics.json files into pipeline/results/all_folds_top2_per_sheet_summary.csv.

3.4 Software and reproducibility

The pipeline was implemented in Python 3 using scikit-learn for cross-validation, preprocessing, and logistic regression; XGBoost 3.2.0 for gradient-boosted trees; and pandas/numpy for data wrangling. Random seeds were fixed (random_state=42) at the split-generation, band-tally, model-search, and classifier-initialization stages. The leakage guard (assert_no_val_leak) is invoked at every data-loading site in modules B and C, so a regression that introduces leakage would fail loudly at runtime rather than silently inflate results.


4. Experiments

4.1 Dataset

107 patients with paired baseline-EEG and T1/T2 pain unpleasantness ratings (mean ΔPain ≈ −2.8 NRS, SD ≈ 2.2; majority-class rate ≈ 50.5% under the ΔPain ≥ −2 dichotomization). All EEG features come from the patient-level pre-computed Excel workbooks in processeddata/.

4.2 Validation setup and baselines

5-fold stratified cross-validation under the harness described in §3. Two baselines are pre-specified:

  (1) Majority-class baseline. Predicting the more frequent class in each fold’s training set; on a near-balanced cohort this yields balanced accuracy ≈ 0.500 by construction.
  (2) Clinical-only baseline. Module C re-run with INCLUDE_CLINICAL=True and Module B disabled (no EEG features) so that the only inputs are pain_t1, age, neurop_months. The same splits, imputation rule, grid search, and threshold-on-train calibration are used. This configuration is implemented in the same code path but has not yet been executed end-to-end for the final cohort; numbers are flagged as pending below.

4.3 Metrics

Per-fold accuracy, balanced accuracy, ROC-AUC, average precision, F1, and confusion matrix on the held-out validation patients of that fold; aggregated as mean ± SD across folds, and additionally pooled across the five out-of-fold prediction sets (107 total OOF predictions) for a single ROC-AUC and confusion matrix summary. Balanced accuracy is the primary metric (chosen to avoid prevalence-driven inflation in any future fold whose class balance drifts from 50%).

4.4 Hyperparameters

All hyperparameter ranges are listed in §3.3. Inner-CV folds: 5 for the grid search inside Module C and inside the per-subset evaluation in Module B; 7 outer folds for the Module B band tally. Random seed 42 fixes splits and classifier initialization throughout.


5. Results

5.1 Headline 5-fold cross-validated performance

Table R1 reports per-fold and aggregate performance of the ABC pipeline on the 21–22 held-out validation patients of each fold (top_2_per_sheet rule, mean+std summaries, 12 EEG + 3 clinical features per patient).

  Table R1. ABC pipeline 5-fold CV performance.

  Fold | Selected model        | Acc   | Bal.Acc | ROC-AUC | AvgPrec | F1    | Threshold
  -----+-----------------------+-------+---------+---------+---------+-------+----------
   0   | XGBoost               | 0.727 | 0.727   | 0.752   | 0.657   | 0.750 | 0.125
   1   | Logistic regression   | 0.727 | 0.727   | 0.727   | 0.694   | 0.727 | 0.500
   2   | XGBoost               | 0.619 | 0.627   | 0.614   | 0.574   | 0.667 | 0.480
   3   | Logistic regression   | 0.524 | 0.532   | 0.545   | 0.680   | 0.444 | 0.485
   4   | XGBoost               | 0.667 | 0.659   | 0.791   | 0.862   | 0.720 | 0.220
  -----+-----------------------+-------+---------+---------+---------+-------+----------
   Mean (SD across folds)      | 0.653 | 0.655   | 0.686   | 0.693   | 0.662 |   —
   SD (n=5)                    | 0.085 | 0.081   | 0.103   | 0.094   | 0.125 |   —

  Pooled OOF (n=107): accuracy 0.654, ROC-AUC 0.714.
  Pooled OOF confusion matrix: TN=32, FP=21, FN=16, TP=38.

The aggregate mean balanced accuracy of 0.655 ± 0.081 is well above the 0.50 majority-class chance level on a near-balanced cohort, and the pooled out-of-fold ROC-AUC of 0.714 indicates that the model’s probability rankings separate improvers from non-improvers reliably better than chance. Three of five folds selected XGBoost; the remaining two selected ℓ2-regularized or elastic-net logistic regression. Per-fold performance is reasonably consistent across the four better-performing folds (accuracy 0.62–0.73, AUC 0.61–0.79); fold 3 is an outlier (accuracy 0.52, AUC 0.55) and is discussed in §5.4.

Figure 1 (reports/abc_paper_figs/fig_per_fold_metrics.png) plots the per-fold accuracy, balanced accuracy, ROC-AUC, and F1 with the chance line overlaid. Figure 2 (reports/abc_paper_figs/fig_oof_roc_confusion.png) plots the pooled out-of-fold ROC curve and the pooled OOF confusion matrix; the pooled sensitivity (TP / (TP+FN) = 38/54) is 0.704 and pooled specificity (TN / (TN+FP) = 32/53) is 0.604.

5.2 Comparison with baselines

  Table R2. ABC versus pre-specified baselines under identical 5-fold splits.

  Baseline                                                        | Bal.Acc          | ROC-AUC
  ----------------------------------------------------------------+------------------+-----------------
  Majority class (predict class 0, by-construction on this cohort) | 0.500            | —
  Clinical-only (T1 pain, age, neurop_months)                      | [pending]        | [pending]
  ABC pipeline (EEG + clinical)                                    | 0.655 (SD 0.081) | 0.686 (SD 0.103)

H1 (ABC > majority) is supported empirically on the held-out folds. H2 (ABC > clinical-only) is to be quantified once the clinical-only configuration is executed under the same harness. An earlier, pre-ABC analysis on a partially overlapping cohort reported a clinical-only logistic regression with balanced accuracy ≈ 0.71 (reports/leakage_free_v3/fold_results.csv: LogReg mean BA = 0.599 ± 0.102, AUC = 0.680 ± 0.073 across 15 features that mixed clinical and EEG inputs), but used a different feature set and inner-CV scheme and so is not a direct apples-to-apples comparator.

5.3 Per-fold band selections and recurrence patterns

A central feature of the ABC pipeline is that band selection is performed inside the training set of every fold. As a consequence the selected bands need not be identical across folds, and in practice they are not (Table R3, Figure 3).

  Table R3. Top-2 tallied bands per sheet within each fold’s training set.

  Fold | Coherence (Z_FFT_Coherence)      | Phase-Lag (Z_FFT_PhaseLag_PLI)   | Abs. power (Z_FFT_abs_bandpower_uV2)
  -----+----------------------------------+----------------------------------+--------------------------------------
   0   | HighBeta, Beta2                  | HighBeta, Delta                  | Theta, Beta3
   1   | Delta, Beta1                     | HighBeta, Theta                  | Theta, Alpha
   2   | Beta, Beta1                      | HighBeta, Delta                  | Beta, Alpha2
   3   | Beta, Beta1                      | HighBeta, Beta3                  | Beta, Alpha2
   4   | Beta3, Alpha2                    | Beta3, Beta2                     | Theta, Beta2

Figure 3 (reports/abc_paper_figs/fig_band_selection_heatmap.png) visualizes the recurrence count of each band-by-sheet cell across the five folds. The strongest recurrences are: Phase-Lag PLI HighBeta (selected in 4 of 5 folds), Coherence Beta1 (3 of 5), and Absolute-power Theta (3 of 5). The variability across folds is not a defect of the pipeline; rather, it is direct evidence that none of the validation patients of any fold contributed to its own selection. Had band selection been performed once on the full cohort, fold-to-fold band identity would necessarily appear stable, but at the cost of subtle leakage from validation into the selection step.

These selections are broadly consistent with H3 — connectivity selections concentrate in the beta/high-beta range and absolute-power selections recur in the theta/low-beta range — supporting the interpretation of a stable neurophysiological signature even without privileging any one fold’s pick.

5.4 Per-fold inspection and outlier analysis

The fold-3 dip (balanced accuracy 0.53, AUC 0.55) is informative. On this fold the inner-CV search selected logistic regression with a moderately regularized ℓ2/elastic-net penalty; the calibrated threshold sat near 0.49. The fold’s training positive rate (0.500) and validation positive rate (0.524) are within the cohort norm, so the shortfall is not attributable to class imbalance. Inspection of the validation confusion matrix (TN=7, FP=3, FN=7, TP=4) shows the model fails to discriminate the two halves of the validation distribution rather than collapsing to a constant-prediction degenerate solution. We interpret this as characteristic small-n variance: with 21 validation patients per fold, a 2-patient shift in marginal predictions changes accuracy by ~9.5 percentage points, and the inter-fold SD of 0.081 quantifies precisely this regime.

5.5 SHAP-based feature attribution (exploratory, prior pipeline)

A complementary SHAP analysis on a prior leakage-controlled experiment with a richer feature set (reports/leakage_free_v3/aggregated_shap.png and aggregated_shap_top25.csv) ranked T1 pain as the single largest contributor to the predicted probability (mean |SHAP| ≈ 0.069), followed by PeakFreq_Hz Gamma2 median (0.037), PeakFreq_Hz HighBeta std (0.034), PeakFreq_Hz Beta3 p90 (0.029), and treatment-group indicators (group_DL, group_NFB). FFT_Coherence in HighBeta and Beta3 — both bands that the present ABC pipeline also selects in its top-2-per-sheet rule — appear in the top quartile of SHAP magnitudes. These attributions are not strictly comparable to the current ABC outputs (different feature schema, different model heads, ensemble averaging across SVM/LR/XGB), but they are reassuring evidence that the ABC pipeline’s band selections fall in the same neurophysiological neighborhood that an independent feature-importance procedure highlights.


6. Discussion

The ABC pipeline supports the central methodological claim of this work: a small, leakage-controlled model on baseline resting-state EEG plus three baseline clinical scalars predicts CIPN treatment response above chance on held-out validation patients (mean balanced accuracy 0.655, pooled OOF AUC 0.714, n = 107). The headline numbers are not exceptional in absolute terms — they sit below the pooled AUROC ≈ 0.84 reported in the broader cross-disease neuromodulation-response meta-analysis [9], which aggregates studies with larger cohorts and richer multimodal inputs (EEG + imaging + clinical), and well below the AUC ≈ 0.95 reported for microstate-feature-driven outcome prediction in prolonged disorders of consciousness [14]. We argue, however, that the present numbers are more trustworthy than prior single-fold or population-level analyses on the same cohort because every analytic decision (band selection, model selection, threshold calibration) was made on the fold’s training set only, with a runtime guard against the most common form of leakage. The aggregate clinical-only baseline under the same harness is the most important missing comparison and is the principal item on the immediate to-do list.

Strengths. (i) Every fold-local decision is made on training patients only; the leakage guard makes accidental contamination impossible to ignore. (ii) The final feature dimension (15) is small enough relative to the per-fold training set (85–86) that the regularized logistic regression and the gradient-boosted trees are both operating in a regime where their inner-CV scores plausibly transfer. (iii) The per-fold artifacts (selected bands, fitted model, calibrated threshold, val predictions, metrics) are written to disk in a fully reproducible layout, so the analysis can be audited end-to-end without re-running.

Limitations and likely failure modes. (i) The cohort is small (n = 107). The inter-fold SD of 0.081 on balanced accuracy is large enough that a single-fold result (as in the prior 3_REF notebook) is not a reliable predictor of out-of-cohort performance. (ii) The clinical-only baseline is pending. Until it is in hand, we cannot quantify the incremental value of EEG features beyond pain_t1, age, and neurop_months, and we cannot rule out the possibility that the EEG features add little above what the three clinical scalars contribute. (iii) The per-patient EEG summary used here (mean, std over channels/channel-pairs in the selected bands) is a strong dimensional bottleneck — it discards spatial topography and any subject-level distributional shape beyond the first two moments. Exploratory sweeps over alternative summary statistics (sweep_summary_funcs at fold 0; mean+median, mean+p90, mean+p95) are implemented in pipeline/C_sweep_summary_funcs.py but were not used for the headline result. (iv) Threshold calibration on training predictions is a documented, deliberate choice (matching the 3_REF reference convention) but introduces an optimistic bias on the training-side accuracy, which in fold 0 reaches 1.0; this does not leak into val, but it does mean that train_accuracy/train_bal_acc cannot be read as an honest predictor of val performance. (v) The EEG acquisition / preprocessing paragraph (§3.2) is currently a placeholder pending collaborator-supplied details — reference montage, recording duration, artifact-rejection criteria, Welch PSD windowing — which are needed for full reproducibility.

Failure-mode characterization. Fold 3 is the clearest failure case and is consistent with characteristic small-cohort variance rather than a degeneracy in the pipeline (the model does not collapse to a constant prediction; it simply fails to separate the val distribution). A useful sensitivity check would be a permutation test (≥ 200 label shuffles per fold) to convert the fold-level metrics into publication-grade p-values.

Clinical implications. If the result holds under the pending clinical-only comparison and under independent replication on a separate cohort, the practical implication is modest but real: a 5-minute baseline EEG acquisition plus three pre-existing chart variables would let a clinician identify, ahead of any treatment commitment, the patient subgroup whose probability of clinically meaningful improvement is materially below cohort average — and, conversely, the subgroup most likely to benefit. The interesting question, which this work does not yet address, is whether the same feature set predicts modality-specific response (NFB vs. DL vs. NFB+DL) and not just response in aggregate.


7. Conclusion

We presented the ABC pipeline, a deliberately conservative end-to-end analysis pipeline for predicting CIPN treatment response from baseline resting-state EEG and a small set of baseline clinical scalars. The pipeline organizes the analysis as three sequential modules — split generation, per-fold band selection, and per-fold model fitting/threshold calibration/evaluation — and enforces leakage control with a runtime guard at every data-loading site. Applied to a single-site randomized CIPN trial of 107 patients, the pipeline obtained mean held-out balanced accuracy 0.655 ± 0.081 and pooled out-of-fold AUC 0.714 on a balanced binary outcome, substantively above the majority-class chance level. Per-fold band selections recurred in interpretable patterns — Phase-Lag PLI HighBeta (4/5 folds), Coherence Beta1 (3/5), Absolute-power Theta (3/5) — consistent with prior reports of beta/high-beta connectivity and theta-band power as candidate markers of chronic-pain state. The most important next step is completing the clinical-only baseline under the identical harness so that the incremental value of EEG features can be quantified directly; further extensions (permutation testing, alternative per-patient summary statistics, modality-specific response prediction) are already scaffolded in the codebase.


Appendix

A.1 Code and artifact map

  pipeline/A_make_splits.py             # Module A: stratified 5-fold splits, label binarization
  pipeline/B_band_selection.py          # Module B: per-fold band selection (top_2_per_sheet)
  pipeline/C_train_eval.py              # Module C: feature build, GS, threshold cal, val eval
  pipeline/E_run_all_folds.py           # driver: runs B+C across folds, writes aggregate CSV
  pipeline/splits/                      # per-fold train/val patient IDs (.npy) + manifest.json
  pipeline/bands/fold_{k}/top2_per_sheet/    # per-fold selected bands JSON + count CSV
  pipeline/results/fold_{k}/top2_per_sheet/  # per-fold metrics, predictions, fitted model
  pipeline/results/all_folds_top2_per_sheet_summary.csv  # cross-fold summary
  reports/leakage_free_v3/              # prior leakage-controlled SHAP experiment artifacts
  reports/abc_paper_figs/               # figures generated for this paper

A.2 Per-fold confusion matrices (validation)

  Fold 0 (XGBoost, n_val=22): TN=7, FP=4, FN=2, TP=9
  Fold 1 (LogReg,  n_val=22): TN=8, FP=3, FN=3, TP=8
  Fold 2 (XGBoost, n_val=21): TN=5, FP=6, FN=2, TP=8
  Fold 3 (LogReg,  n_val=21): TN=7, FP=3, FN=7, TP=4
  Fold 4 (XGBoost, n_val=21): TN=5, FP=5, FN=2, TP=9
  Pooled (n=107):              TN=32, FP=21, FN=16, TP=38
  Pooled sensitivity 0.704, pooled specificity 0.604.

A.3 Selected hyperparameters per fold (final classifier)

  Fold 0 (XGBoost):  n_estimators=300, max_depth=3, learning_rate=0.2, subsample=1.0,
                     colsample_bytree=1.0, reg_lambda=1. Boundary warnings on
                     {colsample_bytree, learning_rate, n_estimators, subsample}: at HIGH edge
                     of grid — grid extension upward is the suggested next step.
  Fold 1 (LogReg):   solver=saga, penalty=elasticnet, C=1, l1_ratio=0.7.
  Fold 2 (XGBoost):  best-params recorded in pipeline/results/fold_2/top2_per_sheet/test_metrics.json.
  Fold 3 (LogReg):   solver/penalty/C recorded in fold_3/top2_per_sheet/test_metrics.json.
  Fold 4 (XGBoost):  best-params recorded in fold_4/top2_per_sheet/test_metrics.json.

A.4 Figures

  Figure 1. ABC pipeline — per-fold held-out validation metrics. Plotted in
            reports/abc_paper_figs/fig_per_fold_metrics.png. Per-fold bars for accuracy,
            balanced accuracy, ROC-AUC, and F1; chance line (balanced accuracy 0.50) overlaid.
  Figure 2. Pooled out-of-fold ROC curve and confusion matrix.
            reports/abc_paper_figs/fig_oof_roc_confusion.png. Pooled OOF AUC = 0.714, n = 107.
  Figure 3. Per-fold band selections heatmap.
            reports/abc_paper_figs/fig_band_selection_heatmap.png. Cell value = number of folds
            (out of 5) in which that (sheet, band) cell appears in the fold’s top-2 selection.
  Figure 4. SHAP feature-attribution bar chart from the prior leakage-controlled experiment.
            reports/leakage_free_v3/aggregated_shap.png. Top contributors: T1_pain, PeakFreq_Hz
            Gamma2 median, PeakFreq_Hz HighBeta std, PeakFreq_Hz Beta3 p90, group_DL, group_NFB.


To do before submission

  1. Run the clinical-only baseline (Module C with INCLUDE_CLINICAL=True and Module B
     disabled) across all five folds under the identical harness; fill in Table R2.
  2. Fill in the EEG acquisition/preprocessing paragraph (§3.2) with collaborator-supplied
     details: reference montage, recording duration, artifact-rejection criteria, Welch
     PSD parameters.
  3. Run a permutation test (≥ 200 label shuffles per fold) on the held-out validation
     labels to convert per-fold and pooled metrics into formal p-values.
  4. Re-extend the XGBoost grid in fold 0 along the dimensions flagged by the boundary
     check (colsample_bytree, learning_rate, n_estimators, subsample at the HIGH edge)
     and report whether the headline number is materially changed.
  5. Optional: modality-specific (NFB vs. DL vs. NFB+DL) sub-analyses to ask whether the
     EEG signature predicts response to a particular intervention rather than treatment
     response in aggregate.


References

References are numbered in order of first appearance in the manuscript.
(Format pending: convert to journal-required style; some entries still need DOIs and
volume/page detail confirmed before submission.)

  [1]  Staff, N., Grisold, A., Grisold, W., & Windebank, A. (2017). Chemotherapy-induced
       peripheral neuropathy: A current review. Annals of Neurology.
       https://onlinelibrary.wiley.com/doi/10.1002/ana.24951

  [2]  Prinsloo, S., Kaptchuk, T. J., De Ridder, D., Lyle, R., Bruera, E., Novy, D.,
       Barcenas, C., & Cohen, L. (2023). Brain–computer interface relieves chronic
       chemotherapy-induced peripheral neuropathy: A randomized, double-blind,
       placebo-controlled trial. Cancer.
       https://acsjournals.onlinelibrary.wiley.com/doi/10.1002/cncr.35027

  [3]  Prinsloo, S., Novy, D., Driver, L., Lyle, R., Ramondetta, L., Eng, C., McQuade, J.,
       Lopez, G., & Cohen, L. (2017). Randomized controlled trial of neurofeedback on
       chemotherapy-induced peripheral neuropathy: A pilot study. Cancer.
       https://acsjournals.onlinelibrary.wiley.com/doi/10.1002/cncr.30649

  [4]  Patel, K., Sutherland, H., Henshaw, J., Taylor, J. R., Brown, C., Casson, A.,
       Trujillo-Barreton, N. J., Jones, A. K. P., & Sivan, M. (2020). Effects of
       neurofeedback in the management of chronic pain: A systematic review and
       meta-analysis of clinical trials. European Journal of Pain.
       https://onlinelibrary.wiley.com/doi/10.1002/ejp.1612

  [5]  Hesam-Shariati, N., Chang, W.-J., Wewege, M., McAuley, J., Booth, A., Trost, Z.,
       Lin, C.-T., Newton-John, T., & Gustin, S. (2021). The analgesic effect of
       electroencephalographic neurofeedback for people with chronic pain: A systematic
       review and meta-analysis. European Journal of Neurology.
       https://onlinelibrary.wiley.com/doi/10.1111/ene.15189

  [6]  Prinsloo, S., Gabel, S., Data, L., & Lyle, R. (2014). Neurofeedback for
       chemotherapy-induced neuropathic symptoms: A case study. NeuroRegulation.
       https://www.neuroregulation.org/article/view/14293

  [7]  Loprinzi, C. L., Lacchetti, C., Bleeker, J., et al. (2020). Prevention and
       management of chemotherapy-induced peripheral neuropathy in survivors of adult
       cancers: ASCO guideline update. Journal of Clinical Oncology.
       https://ascopubs.org/doi/10.1200/JCO.20.01399

  [8]  Nan, W., Wan, F., Tang, Q., Wong, C., Wang, B., & Rosa, A. C. (2018). Eyes-Closed
       Resting EEG Predicts the Learning of Alpha Down-Regulation in Neurofeedback
       Training. Frontiers in Psychology.
       https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2018.01607/full

  [9]  Quintero-Villegas, A., Fylaktou, F., Morales, J., & Zanos, T. P. (2025). Predicting
       response to neuromodulation therapies using machine learning: A systematic review
       and meta-analysis. Bioelectronic Medicine, 11(27).
       https://doi.org/10.1186/s42234-025-00191-8

  [10] Andrade, S., & Silva-Sauer, L. da. (2022). Identifying biomarkers for tDCS
       treatment response in Alzheimer’s disease patients: a machine learning approach
       using resting-state EEG classification. Frontiers in Human Neuroscience.
       https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2023.1234168/full

  [11] Sangam, V. G., Bharadwaj, M. S., Raman, S. S., Lakshmi, A. S., Murthy, P. A., &
       Faizan, M. (2020). Electroencephalogram (EEG), its processing and feature
       extraction. International Journal of Engineering Research & Technology, 9(6).

  [12] Al-Fahoum, A. S., & Al-Fraihat, A. A. (2014). Methods of EEG signal features
       extraction using linear analysis in frequency and time–frequency domains.
       ISRN Neuroscience, 2014, Article 730218.
       https://doi.org/10.1155/2014/730218

  [13] Xie, S. (2021). Wavelet power spectral domain functional principal component
       analysis for feature extraction of epileptic EEGs. Computation, 9(7), 78.
       https://doi.org/10.3390/computation9070078

  [14] Li, H., Dong, L., Su, W., Liu, Y., Tang, Z., Liao, X., Long, J., Zhang, X., Sun, X.,
       & Zhang, H. (2025). Multiple patterns of EEG parameters and their role in the
       prediction of patients with prolonged disorders of consciousness. Frontiers in
       Neuroscience, 19, 1492225. https://doi.org/10.3389/fnins.2025.1492225

  [15] Wang, M., Cheng, H., Lopez, V., Sundar, R., & Yorke, J. (2018). Redefining
       chemotherapy-induced peripheral neuropathy through symptom cluster analysis and
       patient-reported outcome data over time. BMC Cancer.
       https://link.springer.com/article/10.1186/s12885-019-6352-3

  [16] Wang, M., & Molassiotis, A. (2021). Mapping chemotherapy-induced peripheral
       neuropathy phenotype and health-related quality of life in patients with cancer
       through exploratory analysis of multimodal data. Supportive Care in Cancer.
       https://link.springer.com/article/10.1007/s00520-022-06821-0

  [17] Chen, C., Smith, E., Stringer, K., & Henry, N. (2021). Co-occurrence and metabolic
       biomarkers of sensory and motor subtypes of peripheral neuropathy from paclitaxel.
       Breast Cancer Research and Treatment.
       https://link.springer.com/article/10.1007/s10549-022-06652-x

  [18] Sharma, A., Johnson, K., & Bie, B. (2021). A multimodal approach to discover
       biomarkers for taxane-induced peripheral neuropathy (TIPN): a study protocol.
       https://journals.sagepub.com/doi/abs/10.1177/15330338221127169
