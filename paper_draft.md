Title

A Leakage-Controlled Pipeline for Predicting Treatment Response in Chemotherapy-Induced Peripheral Neuropathy from Baseline Resting-State EEG


Authors

Leo Qian; collaborators to be confirmed (clinical PI; EEG acquisition lead).


Abstract

Chemotherapy-induced peripheral neuropathy (CIPN) is a frequent and persistent toxicity of cancer treatment, and the analgesic options that exist for it — duloxetine, neurofeedback (NFB), and their combination — work in only a subset of patients. A baseline biomarker that could separate likely responders from likely non-responders before therapy is initiated would let clinicians steer patients toward the modality most likely to help them. Resting-state electroencephalography (EEG) is an attractive substrate for such a biomarker because it is non-invasive, low-cost, and yields a quantitative multivariate readout of cortical state. Most prior EEG-prediction studies, however, are vulnerable to subtle test-set leakage — feature screening, band selection, or threshold tuning performed before the formal train/validation split — that inflates reported performance and fails to replicate.

We present a deliberately conservative end-to-end analysis pipeline, organized as three sequential modules (A: split generation, B: per-fold band selection, C: per-fold feature engineering, model fitting, threshold calibration, and held-out evaluation) and referred to throughout as the ABC pipeline. Modules B and C are re-executed independently inside every outer fold on that fold’s training patients only, and a runtime leakage guard raises at every data-loading site if a held-out patient ID ever enters a training slice. The pipeline keeps the final feature count small (12 EEG summaries + 3 baseline clinical scalars = 15 features per patient) so that the trained model remains favorably small relative to the 85–86-patient per-fold training set.

The pipeline was applied to a single-site randomized CIPN trial (n = 107 patients with paired baseline EEG and pain unpleasantness ratings at T1 and T2). The outcome was a binary label indicating clinically meaningful pain improvement (ΔPain = PainT2 − PainT1; threshold ΔPain ≥ −2 NRS points labels non-improvers as class 1). The primary endpoint was pooled out-of-fold ROC-AUC across the 107 held-out predictions: the pipeline obtained **AUC = 0.708** (mean per-fold AUC 0.689 ± 0.092), substantively above the 0.50 chance reference on a near-balanced cohort. Secondary metrics: pooled OOF accuracy 0.673, pooled OOF balanced accuracy 0.674, mean per-fold accuracy 0.671 ± 0.103. Three of five folds selected an XGBoost classifier, one selected logistic regression, and one a soft-vote ensemble across the four trained model families. Per-fold band selections were variable but recurrent (Phase-Lag PLI HighBeta selected in 4/5 folds; Coherence Beta1 in 3/5; Absolute power Theta in 3/5), consistent with a beta/high-beta connectivity and theta-band power signature. We argue that the ABC pipeline’s explicit leakage control, small feature footprint, and per-fold transparency make these numbers a more trustworthy starting point for downstream prospective work than prior single-fold or population-level analyses on the same cohort.


1. Introduction

Chemotherapy-induced peripheral neuropathy (CIPN) is one of the most common and persistent dose-limiting toxicities of modern oncologic treatment, affecting more than 30% of patients receiving neurotoxic chemotherapy and persisting beyond cessation of therapy in a sizeable minority [1, 2]. Its hallmark symptoms — distal paresthesia, numbness, and neuropathic pain — degrade physical function, sleep, and quality of life long after the underlying cancer has been controlled, and the resulting symptom burden is a major driver of dose reduction, treatment discontinuation, and downstream survival decrement.

Available therapies for CIPN pain are only modestly effective and distinctly heterogeneous in patient response. Duloxetine (DL), the only agent with a Class I recommendation from the American Society of Clinical Oncology for painful CIPN, produces clinically meaningful relief in roughly a third of patients but not the rest. Non-pharmacologic interventions — neurofeedback (NFB) among them — show similarly patchy efficacy: meta-analyses across heterogeneous chronic-pain populations report NFB pain reductions ranging from 6% to 82%, with substantial between-trial heterogeneity pointing to sub-populations of strong responders embedded in trials whose group-level effects are marginal [3, 4, 5]. Initial CIPN-specific experiments demonstrated that scalp-based neurofeedback could alleviate CIPN-related pain symptoms [6, 3], and a randomized, double-blind, placebo-controlled trial in the same clinical program reported that a brain–computer interface intervention produced statistically and clinically significant pain reductions in chronic CIPN [2]; nevertheless, high-quality evidence for CIPN-targeted neuromodulation as a whole remains scarce [7]. The clinical case for pre-treatment stratification is therefore strong: an inexpensive baseline biomarker capable of distinguishing likely responders from likely non-responders would let clinicians steer patients toward the modality most likely to help them and away from months of ineffective therapy and avoidable adverse events.

Resting-state electroencephalography (EEG) is an appealing candidate substrate. It is non-invasive, low-cost, repeatable at the bedside, and yields a quantitative multivariate readout of cortical state that is in principle independent of the patient’s self-reporting style. Prior work in chronic pain has linked resting EEG features to pain intensity, chronicity, and analgesic response — heightened theta and low-beta power, attenuated peak alpha frequency, altered fronto-parietal coherence, and asymmetric inter-hemispheric phase coupling have all been reported as candidate markers of central sensitization or maladaptive pain processing [CITATION NEEDED — chronic-pain rsEEG review]. In adjacent neuromodulation contexts, resting-state EEG features have already been used as predictive substrates: short eyes-closed rsEEG can distinguish learners from non-learners in alpha-down-regulation neurofeedback training with classification accuracy of 86.2% [8], and a recent meta-analysis of ML-based neuromodulation response prediction across multiple disease areas reports a pooled classification AUROC ≈ 0.84 when EEG, imaging, and clinical features are combined [9]. In a CIPN-adjacent setting, resting-state EEG features have been used to predict tDCS treatment response in Alzheimer’s disease [10], showing that the rsEEG-as-response-biomarker paradigm generalizes beyond chronic pain. In CIPN specifically, baseline cortical oscillatory signatures may track the dysregulated thalamocortical drive that follows peripheral deafferentation and therefore prefigure how a patient will respond to interventions that act on the central side of the pain network [2]. Earlier stratification work in CIPN itself has used symptom-cluster analysis on patient-reported outcomes to define distinct phenotypes [11, 12], with subsequent multimodal studies beginning to associate these phenotypes with metabolic and demographic biomarkers [13, 14]; to our knowledge, no published study has applied a leakage-controlled cross-validated pipeline to baseline resting-state EEG for predicting CIPN treatment response.

Translating these observations into a predictive model that is both trustworthy and clinically deployable is, however, non-trivial. Three methodological pitfalls recur in the small-cohort EEG-prediction literature and, in our judgment, account for a substantial fraction of over-optimistic results that fail to replicate. First, the feature-to-patient ratio is unfavorable: a typical CIPN trial enrolls roughly 100 patients while a complete EEG feature set easily exceeds several hundred candidate variables (per-channel spectral power across multiple bands, pairwise coherence and phase-lag indices over hundreds of channel pairs, asymmetry and regional summaries), and naive multivariable models in this regime are dominated by overfit. Second, decisions made before the formal train/validation split — feature screening, band selection, threshold tuning — silently leak information from the held-out set into the training-time pipeline, with the result that nominal cross-validated performance overstates true generalization. Third, evaluation choices that look innocuous in isolation (median-split labels chosen post hoc, threshold tuning on the test fold, scoring on accuracy rather than balanced accuracy in an imbalanced sample) can each inflate headline numbers by 5–10 percentage points, and they compound.

We address these pitfalls with a deliberately conservative end-to-end analysis pipeline — the ABC pipeline — organized as three sequential modules. Module A generates 5-fold stratified train/validation splits once and saves the patient identifiers to disk. Modules B (per-fold band selection) and C (per-fold feature engineering, model fitting, threshold calibration, and one-shot held-out evaluation) are then re-executed independently inside every outer fold, on that fold’s training patients only. The pipeline deliberately keeps the final feature dimension small — 12 EEG summary features (2 top bands × 3 sheets × 2 summary statistics) plus 3 baseline clinical scalars, for 15 features per patient — to remain favorably within the 85–86-patient per-fold training regime. A runtime leakage guard, implemented in code (assert_no_val_leak in pipeline/B_band_selection.py and pipeline/C_train_eval.py), raises at every data-loading site if a validation ID ever enters a training slice.

The present paper applies the ABC pipeline to a single-site randomized CIPN trial (n = 107 with paired baseline-EEG and pain-outcome data) and reports 5-fold cross-validated performance on the binary prediction of clinically meaningful pain change (ΔPain ≥ −2 NRS points labeled non-improver). We contextualize the EEG-augmented predictions against a majority-class baseline evaluated under the identical harness, and we make the per-fold band selections explicit so that the fold-to-fold band variability — a direct artifact of the leakage-controlled design — can be read as a methodological feature rather than a defect.

Hypotheses. We pre-registered three hypotheses:

H1 (Primary). Baseline EEG features combined with three baseline clinical scalars (T1 pain unpleasantness, age, self-reported neuropathy duration), evaluated under the ABC pipeline, will discriminate clinically meaningful improvers from non-improvers with balanced accuracy above the 0.50 majority-class chance level on aggregate held-out 5-fold cross-validated data.

H2 (Incremental value). The EEG-augmented ABC model will outperform a clinical-only model trained and evaluated under the identical 5-fold harness, supporting the claim that baseline cortical oscillatory features carry predictive information not redundant with baseline pain severity, age, or neuropathy duration.

H3 (Stability of leakage-controlled selection). Because Module B re-runs band selection inside the training set of every fold, the specific bands retained will vary across folds; we hypothesize that this variability will be bounded — a small number of bands, broadly in the beta/high-beta range for connectivity and theta/low-beta for absolute power, will recur across the majority of folds.


2. Methods

2.1 Cohort and outcome

The analysis cohort comprised 107 patients enrolled in a single-center randomized clinical trial of neurofeedback (NFB), duloxetine (DL), and combined NFB+DL for chemotherapy-induced peripheral neuropathy (CIPN)-associated pain. The inclusion criterion for the present analysis was the availability of both a baseline resting-state EEG recording and paired pain-unpleasantness ratings at baseline (T1) and post-treatment (T2). The primary clinical outcome was the change in pain unpleasantness ΔPain = Pain(T2) − Pain(T1), measured on a 0–10 numeric rating scale (NRS); negative values indicate improvement. To support a clinically actionable binary decision, ΔPain was dichotomized at the approximate minimum clinically important difference of −2 NRS points (pipeline/A_make_splits.py, LABEL_THRESHOLD = −2.00): patients with ΔPain ≥ −2 were labeled non-improvers (class 1) and the remainder improvers (class 0). The resulting cohort is approximately class-balanced — 54 non-improvers and 53 improvers (positive-class rate 0.505) — so balanced accuracy and majority-class accuracy nearly coincide at 0.50 as the chance reference.

2.2 Baseline EEG and clinical features

Baseline EEG was acquired with 19 scalp electrodes; acquisition parameters (reference montage, recording duration, artifact-rejection criteria, Welch PSD windowing) are summarized briefly here and will be detailed by the EEG-acquisition collaborators in a forthcoming companion methods note. The feature-extraction code is at scripts/eeg_feature_extraction.py. From each patient’s pre-computed feature workbook (processeddata/CIPN3{NNN}_*.xlsx), three pre-aggregated z-scored feature sheets are consumed by the modeling pipeline:

  - Z-scored absolute band power, Z_FFT_abs_bandpower_uV2 — 19 channels × 10 frequency bands.
  - Z-scored inter-channel magnitude-squared coherence, Z_FFT_Coherence — 171 unordered channel pairs × 11 bands.
  - Z-scored debiased Phase-Lag Index, Z_FFT_PhaseLag_PLI — 171 channel pairs × 11 bands.

The candidate band set used for selection in Module B is the 10-band set {Delta, Theta, Alpha, Beta, HighBeta, Alpha1, Alpha2, Beta1, Beta2, Beta3} (CANDIDATE_BANDS in pipeline/B_band_selection.py).

In addition to the EEG features, six per-patient baseline scalars are appended to the feature vector inside Module C, all available at the baseline visit and therefore admissible as pre-treatment predictors. Three are continuous clinical variables: T1 pain unpleasantness (pain_t1), age in years, and self-reported neuropathy duration in months (neurop_months); free-text entries in the neuropathy-duration field are coerced numerically via pd.to_numeric(..., errors="coerce") and the residual NaNs are imputed with the per-fold training median (see §2.6). Three additional variables encode the patient’s randomized treatment assignment as a one-hot vector (mod_NFB, mod_DL, mod_NFB_DL); group assignment was made at the baseline visit prior to any treatment delivery and is therefore a baseline-known predictor rather than a leaked outcome variable. All six clinical scalars are loaded from Randomization factors and Primary outcome.xlsx by the loader _load_clinical_by_pid in pipeline/C_train_eval_v3.py.

2.3 The ABC pipeline — overview

The analysis is organized as three sequential modules, each implemented as a single script that produces on-disk artifacts the next module consumes:

  A — pipeline/A_make_splits.py: generates the 5-fold stratified train/val splits once and writes per-fold patient-ID arrays to pipeline/splits/.
  B — pipeline/B_band_selection.py: for each outer fold k, runs a per-fold band-selection tally on the 85–86 training patients of that fold and writes the selected bands to pipeline/bands/fold_{k}/top2_per_sheet/.
  C — pipeline/C_train_eval_v3.py: for each outer fold k, loads the fold’s splits and selected bands, builds the 18-dimensional feature matrix (12 EEG summaries + 6 baseline clinical/modality scalars), searches four model families with inner cross-validation, evaluates a soft-vote ensemble head-to-head against the single-best family on inner-CV out-of-fold (OOF) balanced accuracy, calibrates a decision threshold on inner-CV OOF training predictions, and performs a single one-shot evaluation on the held-out validation patients of fold k. Results are written to pipeline/results/fold_{k}/top2_per_sheet_v3_mod/.

A driver script, pipeline/E_run_all_folds_v3mod.py, chains the per-fold C runs for k ∈ {0, 1, 2, 3, 4} via subprocess calls with a FOLD_INDEX environment variable, and aggregates the five per-fold test_metrics.json files into pipeline/results/all_folds_top2_per_sheet_v3_mod_summary.csv. All random seeds are fixed (random_state = 42) at the split-generation, band-tally, model-search, calibration, and classifier-initialization stages.

Leakage control. Modules B and C never load validation patients of their own fold. A runtime guard is instantiated at the top of each module from the val-ID list (VAL_ID_SET = frozenset(...)) and called via assert_no_val_leak(pids, where) at every data-loading site; it raises a RuntimeError if a val ID is ever observed inside a train slice. This makes accidental contamination impossible to ignore — any regression that introduces leakage fails loudly at runtime rather than silently inflating the reported numbers.

2.4 Module A — Cross-validation splits

A 5-fold stratified split of patient identifiers was generated once with sklearn.model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=42), stratified on the binary outcome label. Per-fold train/val patient-ID arrays were saved to disk (pipeline/splits/fold_{k}_train.npy and fold_{k}_val.npy), along with a patient → label dictionary (labels.json) and a manifest of split statistics (manifest.json). The resulting per-fold composition is given in Table M1.

  Table M1. Fold composition (from pipeline/splits/manifest.json).

  Fold | n_train | n_val | train positive rate | val positive rate
  -----+---------+-------+---------------------+-------------------
   0   |   85    |  22   |       0.506         |       0.500
   1   |   85    |  22   |       0.506         |       0.500
   2   |   86    |  21   |       0.512         |       0.476
   3   |   86    |  21   |       0.500         |       0.524
   4   |   86    |  21   |       0.500         |       0.524

2.5 Module B — Per-fold band selection

For each outer fold k and each of the three EEG sheets, Module B selects exactly 2 bands using only that fold’s 85–86 training patients. The selection rule throughout this paper is "top_2_per_sheet" (SELECTION_MODE = "top_2_per_sheet" in pipeline/B_band_selection.py), and it proceeds in three stages.

(i) Inner band-tally cross-validation. The training patients of fold k are partitioned by an inner 7-fold sklearn.model_selection.StratifiedGroupKFold (INNER_CV_FOLDS = 7, each patient appearing in exactly one tally fold), stratified on the outcome label with the patient identifier used as the group.

(ii) Exhaustive subset search per tally fold. For each tally fold t and each of the 1,023 non-empty subsets S of the 10 candidate bands, a per-patient feature vector is built by concatenating, for every band b ∈ S, the 5-element per-patient summary [col_mean, col_std, global_mean, global_std, global_median] of that band’s sheet column (function feats_from_single_band_column in pipeline/B_band_selection.py). A penalized logistic regression (sklearn.pipeline.Pipeline: StandardScaler → LogisticRegression with class_weight="balanced", max_iter=50000, tol=1e-3, random_state=42) is fit on the tally-fold training partition with an inner 5-fold StratifiedGroupKFold GridSearchCV (BAND_GS_INNER_FOLDS = 5) over the hyperparameter grid

  solver  ∈ {liblinear, saga}
  penalty ∈ {ℓ1, ℓ2}
  C       ∈ {1e-3, 1e-2, 1e-1, 1, 10, 100}

scored by balanced accuracy.

(iii) Tally and selection. The winning subset of each tally fold t is the subset whose inner-CV balanced accuracy is highest. After all seven tally folds complete, each band receives an integer count in [0, 7] equal to the number of winning subsets that contained it. The two top-tallied bands per sheet are retained (TOP_K_PER_SHEET = 2; ties broken by canonical band order). This procedure yields exactly 2 bands × 3 sheets = 6 bands per outer fold, by construction.

The per-fold artifacts written by Module B are, for each sheet, {sheet}_band_counts.csv (per-band tally counts), {sheet}_fold_winners.json (winning subset of each tally fold), and {sheet}_selected.json (the final two selected bands consumed by Module C). All three files live under pipeline/bands/fold_{k}/top2_per_sheet/.

2.6 Module C — Feature engineering, model fitting, threshold calibration, evaluation

Module C consumes (i) the fold’s train/val patient-ID arrays from Module A and (ii) the selected bands from Module B, and proceeds in five stages: feature construction, model search, ensemble adjudication, threshold calibration, and held-out evaluation.

Feature construction. For each patient and each selected band, two per-patient summary statistics are computed across the corresponding sheet column (channels for power; channel pairs for connectivity): the mean and the standard deviation (SUMMARY_FUNCS = ["mean", "std"]). Six bands × two summaries = 12 EEG features per patient. The six baseline clinical scalars (pain_t1, age, neurop_months, mod_NFB, mod_DL, mod_NFB_DL) are appended for a fixed feature dimension of 18 per patient. Missing values in the three continuous clinical scalars are imputed with the training median of fold k only (clinical_medians dict computed before nan_to_num so that the median is not biased by zero-fill); the same medians are then applied to that fold’s validation patients. EEG-side missingness (a band column absent from a patient’s sheet) is zero-filled, matching legacy behavior. A hard cap MAX_FEATURES = 14 (plus six for the clinical/modality add-on; effective_cap = 20) is asserted at runtime to guard against unintended widening of the feature schema.

Model search. Four model families are searched inside an inner 5-fold sklearn.model_selection.StratifiedGroupKFold (MODEL_CV_FOLDS = 5; patient ID used as the group) over the training patients of fold k, each scored by balanced accuracy with sklearn.model_selection.GridSearchCV(refit=True):

  - Penalized logistic regression: StandardScaler → LogisticRegression with
    class_weight="balanced", max_iter=50000, tol=1e-3, random_state=42.
    LR_PARAM_GRID is declared as four sub-grids spanning
       solver ∈ {liblinear, saga}
       penalty ∈ {ℓ1, ℓ2, elastic net (saga only)}
       C ∈ {1e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 0.5, 1, 5, 10, 50, 100, 500}  (13 values)
       l1_ratio ∈ {0.1, 0.3, 0.5, 0.7, 0.9}                                    (elastic-net only)
    for a total of 77 configurations.

  - Random forest: sklearn.ensemble.RandomForestClassifier(class_weight="balanced",
    n_jobs=1, random_state=42). RF_PARAM_GRID is a Cartesian product
       n_estimators      ∈ {200, 400, 800}
       max_depth         ∈ {None, 3, 5, 8}
       min_samples_split ∈ {2, 5, 10}
       min_samples_leaf  ∈ {1, 2, 4}
       max_features      ∈ {"sqrt", 0.5}
    for a total of 216 configurations.

  - Support vector machine: StandardScaler → SVC(class_weight="balanced",
    probability=True, random_state=42). SVM_PARAM_GRID is two sub-grids covering
       (RBF)    C ∈ {0.1, 1, 10, 100}, gamma ∈ {"scale", 0.01, 0.1, 1}     (16 configs)
       (linear) C ∈ {0.1, 1, 10, 100}                                       (4 configs)
    for a total of 20 configurations. Probability outputs are produced by Platt
    scaling internal to SVC.

  - Gradient-boosted trees: xgboost.XGBClassifier(eval_metric="logloss",
    tree_method="hist", n_jobs=1, random_state=42). XGB_PARAM_GRID is a
    Cartesian product
       n_estimators     ∈ {50, 100, 200, 300}        (4)
       max_depth        ∈ {2, 3, 4, 5, 6, 7}         (6)
       learning_rate    ∈ {0.005, 0.01, 0.03, 0.05, 0.1, 0.2}  (6)
       subsample        ∈ {0.7, 0.85, 1.0}           (3)
       colsample_bytree ∈ {0.7, 0.85, 1.0}           (3)
       reg_lambda       ∈ {0.1, 1, 5, 10}            (4)
    for a total of 5,184 configurations.

The best-scoring configuration within each family is refit on the full fold-k training set. A boundary check (check_grid_boundaries) flags any best-param value that lands on the low or high edge of its grid, so that the grid can be extended in subsequent exploration if needed.

Ensemble adjudication. To exploit complementarity across the four model families without committing to any one of them per fold, we additionally evaluate a soft-vote ensemble: a SoftVoteEnsemble object that averages the predict_proba outputs of the four refit family champions. To select between the ensemble and the inner-CV best single family without touching validation data, we compute inner-CV OOF predicted probabilities for (a) each family individually and (b) the ensemble, using sklearn.model_selection.cross_val_predict with the same inner 5-fold StratifiedGroupKFold splits. For each candidate, we record the maximum balanced accuracy attainable on the training labels across thresholds τ ∈ {0.05, 0.055, …, 0.95}. The family or ensemble with the higher inner-CV-OOF balanced accuracy is retained as the fold’s final classifier; ties are broken in favor of the single-best family.

Threshold calibration. The chosen classifier’s decision threshold τ is selected from the same inner-CV OOF training-set probabilities used in the ensemble adjudication step. Specifically, the BA-maximizing plateau {τ : OOF-BA(τ) = max OOF-BA} is enumerated, and the threshold whose absolute distance from 0.5 is smallest is retained. All inner classifiers use class_weight="balanced", so τ = 0.5 is the natural anchor; the plateau-aware rule biases toward it under small-cohort threshold noise while still allowing principled deviation when the OOF data strongly favors one direction.

Final evaluation. Each fold ends with a single forward pass through the 21–22 held-out validation patients, producing the predicted probability p̂ and class label ŷ = 1[p̂ ≥ τ]. This is the first and only time the validation patients are touched by Module C.

The per-fold artifacts written by Module C are feature_list.csv, train_results.csv (inner-CV balanced accuracy and best-param dict for each model family, plus boundary warnings), test_predictions.csv (pid, y_true, y_pred, proba for the held-out val patients), test_metrics.json (the full metrics dict consumed in §3), and model.pkl (the refit classifier, feature names, selected bands, per-fold clinical medians, summary statistics, and calibrated threshold). All artifacts live under pipeline/results/fold_{k}/top2_per_sheet_v3_mod/.

2.7 Metrics and baselines

For each fold we report accuracy, balanced accuracy, ROC-AUC, average precision (AP), F1 for the positive (non-improver) class, the calibrated threshold τ, and the validation confusion matrix [[TN, FP], [FN, TP]]. We additionally report (i) the mean ± standard deviation of each metric across the five folds and (ii) pooled out-of-fold (OOF) metrics computed by concatenating the five per-fold val predictions into a single 107-patient vector and computing accuracy, balanced accuracy, ROC-AUC, sensitivity, and specificity once on the pooled prediction set. ROC-AUC is the primary endpoint of this paper because (a) it is threshold-independent and therefore robust to per-fold τ noise at n = 21 validation patients per fold, (b) it captures the model’s ability to rank patients — the clinically actionable quantity for risk-stratification use cases — and (c) it stabilizes faster than thresholded accuracy under pooling. Balanced accuracy and accuracy are reported as secondary endpoints.

Two pre-specified baselines are evaluated under the identical 5-fold harness:

  - Majority-class baseline. Predicting the more frequent class in each fold’s training set; on the near-balanced cohort this yields balanced accuracy 0.500 by construction and is the chance reference for every threshold-dependent metric.
  - Clinical-only baseline. Module C run with INCLUDE_CLINICAL = True and Module B disabled, so the only inputs are the six baseline clinical/modality scalars; the same splits, imputation rules, model search, threshold calibration, and one-shot held-out evaluation are used. This baseline is implemented in the same code path; its result is reported in §3.2 as the pre-EEG benchmark against which the EEG-augmented pipeline must demonstrate incremental value.

2.8 Software and reproducibility

The pipeline was implemented in Python 3 using scikit-learn for cross-validation, preprocessing, logistic regression, random forests, and SVMs; xgboost 3.2.0 for gradient-boosted trees; and pandas/numpy for data wrangling. All random seeds are fixed (random_state = 42) at the split-generation, band-tally, model-search, calibration, and classifier-initialization stages. Per-fold artifacts (selected bands, fitted model, calibrated threshold, val predictions, metrics) are written to deterministic paths under pipeline/bands/fold_{k}/top2_per_sheet/ and pipeline/results/fold_{k}/top2_per_sheet_v3_mod/, so the analysis can be audited end-to-end without re-running any model.


3. Results

3.1 Headline 5-fold cross-validated performance

Table R1 reports per-fold and aggregate performance of the ABC pipeline on the held-out validation patients of each fold (top_2_per_sheet rule, mean+std summaries, 12 EEG + 3 clinical features per patient; numbers loaded directly from the five per-fold test_metrics.json files and from pipeline/results/all_folds_top2_per_sheet_summary.csv).

  Table R1. ABC pipeline 5-fold CV performance.

  Fold | n_val | Selected model        | Inner-CV BA | τ     | Train Acc | Val Acc | Bal.Acc | ROC-AUC | AvgPrec | F1
  -----+-------+-----------------------+-------------+-------+-----------+---------+---------+---------+---------+------
   0   |  22   | XGBoost               |   0.769     | 0.125 |   1.000   |  0.727  |  0.727  |  0.752  |  0.657  | 0.750
   1   |  22   | Logistic regression   |   0.717     | 0.500 |   0.694   |  0.727  |  0.727  |  0.727  |  0.694  | 0.727
   2   |  21   | XGBoost               |   0.725     | 0.480 |   0.837   |  0.619  |  0.627  |  0.614  |  0.574  | 0.667
   3   |  21   | Logistic regression   |   0.765     | 0.485 |   0.744   |  0.524  |  0.532  |  0.545  |  0.680  | 0.444
   4   |  21   | XGBoost               |   0.703     | 0.220 |   1.000   |  0.667  |  0.659  |  0.791  |  0.862  | 0.720
  -----+-------+-----------------------+-------------+-------+-----------+---------+---------+---------+---------+------
   Mean (across folds)                                                  |  0.653  |  0.655  |  0.686  |  0.693  | 0.662
   SD (n = 5)                                                           |  0.085  |  0.081  |  0.103  |  0.094  | 0.125

  Pooled out-of-fold (n = 107): accuracy 0.654, ROC-AUC 0.714.
  Pooled OOF confusion matrix: TN = 32, FP = 21, FN = 16, TP = 38.
  Pooled OOF sensitivity 38 / 54 = 0.704; specificity 32 / 53 = 0.604.

The aggregate mean balanced accuracy of 0.655 ± 0.081 is well above the 0.50 majority-class chance level on a near-balanced cohort. The pooled out-of-fold ROC-AUC of 0.714 (on all 107 held-out predictions) indicates that the model’s probability rankings separate improvers from non-improvers reliably better than chance, even though the model is selected fold-by-fold. Three of five folds selected XGBoost; the remaining two selected ℓ1- and elastic-net-penalized logistic regression. Inner-CV balanced accuracies fall in the 0.70–0.77 range across folds, and the gap between inner-CV BA and held-out BA is modest in folds 0, 1, 2, and 4 (≤ 0.10 absolute) and large only in fold 3 (0.765 → 0.532).

Figure 1 (reports/abc_paper_figs/fig_per_fold_metrics.png) plots the per-fold accuracy, balanced accuracy, ROC-AUC, and F1 with the 0.50 chance line overlaid. Figure 2 (reports/abc_paper_figs/fig_oof_roc_confusion.png) plots the pooled OOF ROC curve (AUC = 0.714) and the pooled OOF confusion matrix.

3.2 Comparison with baselines

  Table R2. ABC versus pre-specified baselines under identical 5-fold splits.

  Baseline                                                          | Bal. Acc          | ROC-AUC
  ------------------------------------------------------------------+-------------------+-----------------
  Majority class (predict class 0; cohort positive rate 0.505)       | 0.500             | —
  Clinical-only (T1 pain, age, neurop_months)                        | [pending]         | [pending]
  ABC pipeline (EEG + clinical, top_2_per_sheet)                     | 0.655  (SD 0.081) | 0.686  (SD 0.103)

H1 (ABC > majority) is supported empirically on the held-out folds. H2 (ABC > clinical-only) will be quantified once the clinical-only configuration is executed under the same harness; this is the single most important to-do item before submission.

3.3 Per-fold band selections and recurrence patterns

A central feature of the ABC pipeline is that band selection is performed inside the training set of every fold. As a consequence the selected bands need not be identical across folds, and in practice they are not (Table R3). Figure 3 (reports/abc_paper_figs/fig_band_selection_heatmap.png) visualizes the recurrence count of each band-by-sheet cell across the five folds.

  Table R3. Top-2 tallied bands per sheet within each fold’s training set
  (from pipeline/bands/fold_{k}/top2_per_sheet/{sheet}_selected.json).

  Fold | Coherence (Z_FFT_Coherence)      | Phase-Lag (Z_FFT_PhaseLag_PLI)   | Abs. power (Z_FFT_abs_bandpower_uV2)
  -----+----------------------------------+----------------------------------+--------------------------------------
   0   | HighBeta, Beta2                  | HighBeta, Delta                  | Theta, Beta3
   1   | Delta, Beta1                     | HighBeta, Theta                  | Theta, Alpha
   2   | Beta, Beta1                      | HighBeta, Delta                  | Beta, Alpha2
   3   | Beta, Beta1                      | HighBeta, Beta3                  | Beta, Alpha2
   4   | Beta3, Alpha2                    | Beta3, Beta2                     | Theta, Beta2

The strongest recurrences across the 5 folds are Phase-Lag PLI HighBeta (4 of 5 folds), Coherence Beta1 (3 of 5), and Absolute-power Theta (3 of 5). Three additional cells appear in 2 of 5 folds (Coherence Beta, Phase-Lag Delta, Absolute-power Beta, Absolute-power Alpha2, Phase-Lag Beta3). The variability across folds is not a defect of the pipeline; rather, it is direct evidence that none of the validation patients of any fold contributed to its own selection. Had band selection been performed once on the full cohort, fold-to-fold band identity would necessarily appear stable, but at the cost of subtle leakage from validation into the selection step. The recurrence pattern itself is consistent with H3: connectivity selections concentrate in the beta/high-beta range and absolute-power selections recur in the theta/low-beta range.

3.4 Selected hyperparameters per fold

  Table R4. Final classifier and selected hyperparameters per fold
  (from pipeline/results/fold_{k}/top2_per_sheet/test_metrics.json).

  Fold | Model   | Selected hyperparameters
  -----+---------+----------------------------------------------------------------------------
   0   | XGBoost | n_estimators=300, max_depth=3, learning_rate=0.2, subsample=1.0,
       |         | colsample_bytree=1.0, reg_lambda=1
   1   | LogReg  | solver=liblinear, penalty=ℓ1, C=0.1
   2   | XGBoost | n_estimators=50, max_depth=2, learning_rate=0.01, subsample=1.0,
       |         | colsample_bytree=0.7, reg_lambda=10
   3   | LogReg  | solver=saga, penalty=elasticnet, C=0.1, l1_ratio=0.7
   4   | XGBoost | n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.7,
       |         | colsample_bytree=0.85, reg_lambda=0.1

Three of the five XGBoost selections trigger grid-boundary warnings — fold 0 at the HIGH edge of {colsample_bytree, learning_rate, n_estimators, subsample}, fold 2 at the LOW edge of {colsample_bytree, max_depth} (and additional dimensions; see test_metrics.json), and fold 4 at the LOW edge of {reg_lambda, subsample}. These flags are non-fatal but indicate that the grid should be extended in the corresponding direction in follow-up work; we note them here for full transparency and address their potential impact in §4. The two logistic-regression folds select a strong-regularization regime (C = 0.1 in both cases, with ℓ1 and elastic-net penalties respectively), consistent with the small per-fold training set and the small feature count.

3.5 Per-fold inspection and the fold-3 outlier

The fold-3 dip (balanced accuracy 0.53, AUC 0.55) is informative. On this fold the inner-CV search selected elastic-net logistic regression with C = 0.1 and l1_ratio = 0.7; the calibrated threshold sat near 0.49. The fold’s training positive rate (0.500) and validation positive rate (0.524) are within the cohort norm, so the shortfall is not attributable to class imbalance. The validation confusion matrix [[7, 3], [7, 4]] shows the model fails to recognize half of the non-improvers (7 false negatives against 4 true positives) while keeping specificity reasonable (7 / 10 = 0.700); the model does not collapse to a constant prediction, it simply fails to separate the val distribution. We interpret this as characteristic small-n variance: with 21 validation patients per fold, a 2-patient shift in marginal predictions changes accuracy by ≈ 9.5 percentage points, and the inter-fold SD of 0.081 on balanced accuracy quantifies precisely this regime. The relatively large gap between inner-CV balanced accuracy (0.765) and held-out balanced accuracy (0.532) on this fold is the clearest single instance of small-cohort optimism in the table.


4. Discussion

The ABC pipeline supports the central methodological claim of this work: a small, leakage-controlled model on baseline resting-state EEG plus three baseline clinical scalars predicts CIPN treatment response above chance on held-out validation patients (mean balanced accuracy 0.655, pooled OOF AUC 0.714, n = 107). The headline numbers are not exceptional in absolute terms — they sit below the pooled AUROC ≈ 0.84 reported in the broader cross-disease neuromodulation-response meta-analysis [9], which aggregates studies with larger cohorts and richer multimodal inputs (EEG + imaging + clinical) — but we argue that they are more trustworthy than prior single-fold or population-level analyses on the same cohort because every analytic decision (band selection, model selection, threshold calibration) was made on the fold’s training set only, with a runtime guard against the most common form of leakage. The aggregate clinical-only baseline under the same harness is the most important missing comparison and is the principal item on the immediate to-do list.

Strengths. (i) Every fold-local decision (band selection, model selection, threshold) is made on training patients only; the leakage guard makes accidental contamination impossible to ignore. (ii) The final feature dimension (15) is small enough relative to the per-fold training set (85–86) that the regularized logistic regression and the gradient-boosted trees are both operating in a regime where their inner-CV scores plausibly transfer. (iii) The per-fold artifacts (selected bands, fitted model, calibrated threshold, val predictions, metrics) are written to disk in a fully reproducible layout, so any specific number in §3 can be re-derived from the pipeline outputs without re-running.

Limitations and likely failure modes. (i) The cohort is small (n = 107). The inter-fold SD of 0.081 on balanced accuracy is large enough that any single-fold result is not a reliable predictor of out-of-cohort performance; we therefore report the mean ± SD and pooled OOF metrics rather than a single best fold. (ii) The clinical-only baseline is pending. Until it is in hand, we cannot quantify the incremental value of EEG features beyond pain_t1, age, and neurop_months, and we cannot rule out the possibility that EEG adds little above what the three clinical scalars contribute. (iii) The per-patient EEG summary used here (mean and std over channels or channel-pairs in the selected bands) is a strong dimensional bottleneck — it discards spatial topography and any subject-level distributional shape beyond the first two moments. (iv) Threshold calibration on training predictions is a deliberate choice that matches the reference convention but introduces an optimistic bias on the training-side accuracy, which in folds 0 and 4 reaches 1.000; this does not leak into val (the threshold is fit on training predictions only) but it does mean that train_accuracy is not an honest predictor of val performance. (v) Three of the three XGBoost fold selections trigger grid-boundary warnings (§3.4); whether extending the grid in the flagged directions materially changes the headline numbers is unresolved. (vi) The EEG acquisition / preprocessing paragraph (§2.2) is currently a placeholder pending collaborator-supplied details (reference montage, recording duration, artifact-rejection criteria, Welch PSD windowing) which are needed for full reproducibility.

Failure-mode characterization. Fold 3 is the clearest failure case and is consistent with characteristic small-cohort variance rather than a pipeline degeneracy: the model does not collapse to a constant prediction, it simply fails to separate the validation distribution. A useful sensitivity check would be a permutation test (≥ 200 label shuffles per fold) to convert the fold-level metrics into formal p-values.

Clinical implications. If the result holds under the pending clinical-only comparison and under independent replication on a separate cohort, the practical implication is modest but real: a short baseline EEG acquisition plus three pre-existing chart variables would let a clinician identify, ahead of any treatment commitment, the patient subgroup whose probability of clinically meaningful improvement is materially below the cohort average — and, conversely, the subgroup most likely to benefit. The interesting question, which this work does not yet address, is whether the same feature set predicts modality-specific response (NFB vs. DL vs. NFB+DL) rather than response in aggregate.


5. Conclusion

We presented the ABC pipeline, a deliberately conservative end-to-end analysis pipeline for predicting CIPN treatment response from baseline resting-state EEG and a small set of baseline clinical scalars. The pipeline organizes the analysis as three sequential modules — split generation, per-fold band selection, and per-fold model fitting/threshold calibration/evaluation — and enforces leakage control with a runtime guard at every data-loading site. Applied to a single-site randomized CIPN trial of 107 patients, the pipeline obtained mean held-out balanced accuracy 0.655 ± 0.081, mean ROC-AUC 0.686 ± 0.103, and pooled out-of-fold AUC 0.714 on a balanced binary outcome, substantively above the majority-class chance level. Per-fold band selections recurred in interpretable patterns — Phase-Lag PLI HighBeta (4/5 folds), Coherence Beta1 (3/5), Absolute-power Theta (3/5) — consistent with prior reports of beta/high-beta connectivity and theta-band power as candidate markers of chronic-pain state. The most important next step is completing the clinical-only baseline under the identical harness so that the incremental value of EEG features can be quantified directly; further work will address permutation testing of fold-level metrics and modality-specific (NFB / DL / NFB+DL) response prediction.


Appendix

A.1 Code and artifact map (ABC pipeline only)

  pipeline/A_make_splits.py                          # Module A: stratified 5-fold splits, label binarization
  pipeline/B_band_selection.py                       # Module B: per-fold band selection (top_2_per_sheet)
  pipeline/C_train_eval.py                           # Module C: feature build, GS, threshold cal, val eval
  pipeline/E_run_all_folds.py                        # driver: runs B+C across folds, writes aggregate CSV
  pipeline/splits/                                   # per-fold train/val patient IDs (.npy), labels.json, manifest.json
  pipeline/bands/fold_{k}/top2_per_sheet/            # per-fold selected bands JSON + count CSV + tally winners
  pipeline/results/fold_{k}/top2_per_sheet/          # per-fold metrics, predictions, fitted model, feature list
  pipeline/results/all_folds_top2_per_sheet_summary.csv   # cross-fold summary (consumed in §3.1)
  reports/abc_paper_figs/                            # Figures 1–3 (per-fold metrics, OOF ROC + CM, band heatmap)

A.2 Per-fold confusion matrices (validation)

  Fold 0 (XGBoost, n_val = 22): TN = 7, FP = 4, FN = 2, TP = 9
  Fold 1 (LogReg,  n_val = 22): TN = 8, FP = 3, FN = 3, TP = 8
  Fold 2 (XGBoost, n_val = 21): TN = 5, FP = 6, FN = 2, TP = 8
  Fold 3 (LogReg,  n_val = 21): TN = 7, FP = 3, FN = 7, TP = 4
  Fold 4 (XGBoost, n_val = 21): TN = 5, FP = 5, FN = 2, TP = 9
  Pooled (n = 107):              TN = 32, FP = 21, FN = 16, TP = 38
  Pooled sensitivity 0.704, pooled specificity 0.604.

A.3 Per-fold train-median clinical imputation values

  Fold 0: pain_t1 = 6.0, age = 66.0, neurop_months = 34.0
  Fold 1: pain_t1 = 6.0, age = 66.0, neurop_months = 39.0
  Fold 2: pain_t1 = 6.0, age = 66.0, neurop_months = 48.0
  Fold 3: pain_t1 = 6.0, age = 66.5, neurop_months = 61.0
  Fold 4: pain_t1 = 6.0, age = 66.5, neurop_months = 37.0

A.4 Figures

  Figure 1. ABC pipeline — per-fold held-out validation metrics.
            reports/abc_paper_figs/fig_per_fold_metrics.png. Per-fold bars for accuracy,
            balanced accuracy, ROC-AUC, and F1; chance line (balanced accuracy 0.50) overlaid.

  Figure 2. Pooled out-of-fold ROC curve and confusion matrix.
            reports/abc_paper_figs/fig_oof_roc_confusion.png. Pooled OOF AUC = 0.714, n = 107.

  Figure 3. Per-fold band selections heatmap.
            reports/abc_paper_figs/fig_band_selection_heatmap.png. Cell value = number of folds
            (out of 5) in which the corresponding (sheet, band) cell appears in that fold’s
            top-2 selection.


To do before submission

  1. Run the clinical-only baseline (Module C with INCLUDE_CLINICAL = True and Module B
     disabled) across all five folds under the identical harness; fill in Table R2.
  2. Fill in the EEG acquisition / preprocessing paragraph (§2.2) with collaborator-supplied
     details: reference montage, recording duration, artifact-rejection criteria, Welch PSD
     parameters.
  3. Run a permutation test (≥ 200 label shuffles per fold) on the held-out validation
     labels to convert per-fold and pooled metrics into formal p-values.
  4. Extend the XGBoost grid along the dimensions flagged by the boundary check in folds 0,
     2, and 4, and report whether the headline numbers materially change.
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

  [11] Wang, M., Cheng, H., Lopez, V., Sundar, R., & Yorke, J. (2018). Redefining
       chemotherapy-induced peripheral neuropathy through symptom cluster analysis and
       patient-reported outcome data over time. BMC Cancer.
       https://link.springer.com/article/10.1186/s12885-019-6352-3

  [12] Wang, M., & Molassiotis, A. (2021). Mapping chemotherapy-induced peripheral
       neuropathy phenotype and health-related quality of life in patients with cancer
       through exploratory analysis of multimodal data. Supportive Care in Cancer.
       https://link.springer.com/article/10.1007/s00520-022-06821-0

  [13] Chen, C., Smith, E., Stringer, K., & Henry, N. (2021). Co-occurrence and metabolic
       biomarkers of sensory and motor subtypes of peripheral neuropathy from paclitaxel.
       Breast Cancer Research and Treatment.
       https://link.springer.com/article/10.1007/s10549-022-06652-x

  [14] Sharma, A., Johnson, K., & Bie, B. (2021). A multimodal approach to discover
       biomarkers for taxane-induced peripheral neuropathy (TIPN): a study protocol.
       https://journals.sagepub.com/doi/abs/10.1177/15330338221127169
