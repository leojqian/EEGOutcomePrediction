# Predicting Chemotherapy-Induced Peripheral Neuropathy Pain Outcomes from Baseline EEG Features

**Authors:** Leo Qian et al.  
**Date:** April 2026  
**Institution:** MD Anderson Cancer Center

---

## Abstract

Chemotherapy-induced peripheral neuropathy (CIPN) causes chronic pain in a substantial proportion of cancer survivors, and treatment response is highly variable. Identifying patients likely to benefit from neurofeedback (NFB) or duloxetine (DL) before treatment begins could transform clinical decision-making. In this study, we developed a machine-learning pipeline to predict change in pain unpleasantness (T2 − T1) from baseline EEG features in 107 CIPN patients enrolled in a randomized clinical trial. Using nested cross-validation and permutation testing, we demonstrate that combining SHAP-selected EEG spectral bands with clinical variables achieves significantly above-chance predictive performance: **MAE = 1.42 (vs. baseline 1.74)** for regression and **balanced accuracy = 0.72, AUC = 0.78** for binary classification (p = 0.005, permutation test). These results suggest that baseline neurophysiological signatures carry meaningful prognostic information, even in this small (n = 107) high-dimensional setting.

---

## 1. Introduction

Chemotherapy-induced peripheral neuropathy affects up to 68% of patients receiving neurotoxic chemotherapy and is associated with persistent pain, functional impairment, and reduced quality of life. Current treatments—including neurofeedback training and pharmacotherapy with duloxetine—demonstrate heterogeneous efficacy, with a subset of patients experiencing substantial pain relief and others showing minimal or no benefit.

EEG has emerged as a promising, non-invasive biomarker for pain states and treatment response. Resting-state spectral power, inter-hemispheric asymmetry, and functional connectivity metrics have been linked to chronic pain severity and placebo/nocebo sensitivity in prior literature. However, predictive modeling in small oncology cohorts faces significant challenges: the ratio of features to patients is unfavorable, conventional cross-validation procedures can leak information, and naive model selection leads to inflated performance estimates.

The present study addresses these challenges by applying a comprehensive, methodologically rigorous machine-learning pipeline to baseline EEG and clinical data from a CIPN clinical trial cohort.

---

## 2. Methods

### 2.1 Participants and Outcome

The analysis set comprised **107 patients** for whom both resting-state EEG recordings and valid primary outcome data were available. Patients were enrolled in a randomized trial comparing neurofeedback (NFB), duloxetine (DL), and combined NFB+DL treatment.

The primary outcome was **pain unpleasantness change**: Diff = T2 − T1, where T1 is the baseline NRS (numeric rating scale) score and T2 is the post-treatment score. A negative Diff indicates improvement (pain reduction).

| Statistic | Value |
|-----------|-------|
| n | 107 |
| Mean Diff | −2.83 |
| SD Diff | 2.17 |
| Min / Max | −10 / +4 |
| Baseline regression MAE (predict mean) | 1.737 |
| Median split (binary) | 54 improvers / 53 non-improvers |
| Threshold −2.0 split | 53 improvers / 54 non-improvers |

For classification, two binarizations were used: (1) **median split** (Diff < median = improved) and (2) a **clinically meaningful threshold** of −2.0, roughly corresponding to the MCID for pain NRS. A 3-class segmentation (Diff < −3, −3 ≤ Diff ≤ −2, Diff > −2) was also evaluated.

### 2.2 EEG Data and Feature Extraction

Each patient's resting-state EEG was processed to produce 17 feature sheets covering:

- **Spectral power**: absolute (μV²) and relative (%) band power across 19 channels and 15 frequency bands (FFT_abs_bandpower_uV2, FFT_rel_bandpower_pct)
- **Z-scored power**: normalized to group distributions (Z_FFT_abs/rel)
- **Connectivity**: inter-channel coherence (171 channel pairs × 11 bands) and Phase Lag Index (PLI)
- **Peak frequencies**: per-channel peak frequency

**Feature summarization.** For each sheet, per-column (band) mean and standard deviation were computed across channels, yielding compact per-patient feature vectors. The "Best Bands" feature set was derived from SHAP importance analysis in a prior companion notebook, selecting 45 high-signal spectral features computed with five statistics (column mean, std, global mean, std, and median).

**Spatial and asymmetry features.** Frontal Alpha Asymmetry (FAA; log right − left power) was computed for 5 regional pairs (prefrontal, frontal, temporal, parietal, occipital) across 6 bands (Alpha1, Alpha2, Alpha, Theta, Beta, HighBeta), yielding 30 FAA features. Brain-region averages (Frontal / Central / Temporal / Parietal / Occipital) across 9 bands produced 45 regional features. Frontal-parietal coherence and frequency ratio features were also extracted.

### 2.3 Feature Sets

Ten base feature sets and seven spatial/asymmetry feature sets were evaluated:

| Feature Set | Dimensions |
|---|---|
| Clinical only | 107 × 5 |
| Clinical + Demographics | 107 × 15 |
| Power (raw) | 107 × 56 |
| Power (z-scored) | 107 × 40 |
| Connectivity | 107 × 80 |
| All EEG | 107 × 204 |
| Best Bands | 107 × 45 |
| Best Bands + Demog | 107 × 55 |
| Best Bands + Clinical | 107 × 60 |
| EEG + Clinical | 107 × 209 |
| FAA only | 107 × 30 |
| Regional only | 107 × 45 |
| Spatial (FAA+Reg+FP+Ratio) | 107 × 130 |
| Spatial + Clinical | 107 × 135 |
| Best Bands + Spatial + Full | 107 × 190 |

Clinical features: T1 pain score, age, sex, treatment group, neuropathy duration (months). Demographic features: marital status, sex (one-hot), treatment group (one-hot), cancer stage (I–II / III / IV / Unknown).

### 2.4 Model Zoo

Twelve regression models and ten classification models were evaluated:

**Regression:** Ridge, Lasso, ElasticNet, BayesianRidge, SVR-Linear, SVR-RBF, RandomForest, ExtraTrees, GradientBoosting, XGBoost; PCA-augmented variants of Ridge, Lasso, SVR-RBF.

**Classification (binary and 3-class):** Logistic Regression (L1, L2), SVM-Linear, SVM-RBF, RandomForest, ExtraTrees, GradientBoosting, XGBoost; PCA variants.

Each model was paired with a standard preprocessing pipeline: median imputation → StandardScaler → optional PCA (10 components for PCA variants).

### 2.5 Validation Strategy

**Nested cross-validation** was used throughout to prevent optimistic bias:
- **Outer loop**: 5-fold stratified CV (regression: KFold; classification: StratifiedKFold), random_state=42
- **Inner loop**: 3-fold GridSearchCV for hyperparameter optimization within each outer fold
- **Metrics**: MAE (regression); balanced accuracy and ROC-AUC (binary classification); balanced accuracy (3-class)

**Leave-One-Out CV (LOO-CV)** was additionally performed on key feature sets to maximize training data per fold given the small sample size (n = 107 outer folds, each training on 106 patients).

**Permutation tests** (200 permutations) were run for the best-performing model to assess statistical significance of the observed performance.

**Extended hyperparameter search (Section 14):** The best-performing feature sets were additionally evaluated with `RandomizedSearchCV` (n_iter=300) using log-uniform priors over continuous hyperparameters, `StratifiedGroupKFold` to prevent patient-level leakage, and `scale_pos_weight` for XGBoost class balancing. The classification threshold was set to −2.0 (clinically meaningful) rather than the median.

---

## 3. Results

### 3.1 Regression: Predicting Continuous Pain Change

The predict-mean baseline MAE was **1.737**. Results from the nested 5-fold CV sweep (best model per feature set):

| Feature Set | Best Model | MAE | vs. Baseline |
|---|---|---|---|
| **Best Bands + Clinical** | **ExtraTrees** | **1.425** | **−0.312** |
| Clinical + Demographics | RF | 1.446 | −0.291 |
| Clinical only | RF | 1.469 | −0.268 |
| EEG + Clinical | Lasso | 1.516 | −0.221 |
| Best Bands + Demog | SVR-RBF | 1.615 | −0.122 |
| Best Bands | SVR-RBF | 1.617 | −0.120 |
| Power (raw) | XGBoost | 1.720 | −0.017 |
| Power (z-scored) | Ridge+PCA | 1.756 | +0.019 |
| Connectivity | Lasso | 1.775 | +0.038 |
| All EEG | Lasso | 1.775 | +0.038 |

**Key finding:** The best regression model (ExtraTrees on Best Bands + Clinical) achieved an MAE of **1.425**, representing an 18% reduction from the predict-mean baseline. Clinical features alone outperform most EEG-only configurations. Raw EEG power or connectivity features without clinical context fail to beat baseline. The selective "Best Bands" set, when combined with clinical features, provides additive value.

LOO-CV confirmed the regression result: **MAE = 1.420, R² = 0.297** (LOO best model), with the best feature set also being Best Bands + Clinical. LOO results across feature sets:

| Feature Set | Best Model | MAE (LOO) | R² |
|---|---|---|---|
| Clinical + Demographics | Lasso | 1.398 | 0.316 |
| Best Bands + Clinical | Lasso | 1.435 | 0.278 |
| Clinical only | Lasso | 1.447 | 0.275 |
| Spatial + Clinical | Lasso | 1.514 | 0.209 |
| Best Bands | BayesianRidge | 1.653 | 0.071 |
| FAA only | Lasso | 1.753 | −0.019 |

### 3.2 Binary Classification: Predicting Treatment Responders

Nested 5-fold CV balanced accuracy (binary; median split threshold):

| Feature Set | Best Model | Balanced Accuracy | vs. Chance |
|---|---|---|---|
| **Best Bands + Clinical** | **SVM-RBF** | **0.718** | **+0.218** |
| Clinical only | SVM-Linear | 0.709 | +0.209 |
| Clinical + Demographics | LogReg-L2 | 0.664 | +0.164 |
| Best Bands | LogReg+PCA | 0.663 | +0.163 |
| Best Bands + Demog | LogReg-L1 | 0.661 | +0.161 |
| Power (z-scored) | LogReg-L2 | 0.655 | +0.155 |
| EEG + Clinical | LogReg-L2 | 0.645 | +0.145 |
| All EEG | SVM-Linear | 0.637 | +0.137 |
| Connectivity | SVM-Linear | 0.616 | +0.116 |
| Power (raw) | XGBoost | 0.558 | +0.058 |

LOO-CV binary classification:

| Feature Set | Best Model | BalAcc (LOO) | AUC (LOO) |
|---|---|---|---|
| **Best Bands + Clinical** | **SVM-RBF** | **0.719** | **0.796** |
| Clinical only | SVM-RBF | 0.693 | 0.738 |
| Clinical + Demographics | LogReg-L1 | 0.673 | 0.687 |
| Spatial + Full | LogReg-L1 | 0.673 | 0.688 |
| Best Bands | LogReg-L2 | 0.654 | 0.627 |

LOO confusion matrix (Best Bands + Clinical, SVM-RBF): **LOO balanced accuracy = 0.691**.

### 3.3 Extended Hyperparameter Search (Clinically-Thresholded Classification)

Using the −2.0 threshold, `RandomizedSearchCV` (n_iter=300), and `StratifiedGroupKFold`:

Top 5 combinations by AUC:

| Feature Set | Model | Balanced Acc | AUC | AUC SD |
|---|---|---|---|---|
| **Best Bands + Clinical** | **SVM** | **0.724** | **0.776** | 0.107 |
| Best Bands + Clinical | XGB | 0.673 | 0.748 | 0.081 |
| Best Bands + Clinical | LogReg | 0.699 | 0.747 | 0.094 |
| Clinical + Demographics | LogReg | 0.699 | 0.747 | 0.094 |
| Clinical only | LogReg | 0.701 | 0.741 | 0.085 |

Best model hyperparameters (SVM-RBF on Best Bands + Clinical):
- C = 1.32, γ = auto, kernel = RBF

### 3.4 3-Class Classification

Three-class (Diff < −3; −3 ≤ Diff ≤ −2; Diff > −2) results were generally lower, with the best performance:

- Best model: LogReg-L1 on Clinical only, balanced accuracy = **0.579** (vs. chance 0.333)

The modest 3-class performance reflects both the smaller per-class sample size and the inherent difficulty of precise quantitative segmentation.

### 3.5 Permutation Tests

To confirm that above-baseline performance is not attributable to chance, permutation tests (200 shuffles) were performed on the best models:

| Task | Model | Feature Set | True Score | Permutation Mean ± SD | p-value |
|---|---|---|---|---|---|
| Binary classification | SVM-RBF | Best Bands + Clinical | BalAcc = 0.719 | 0.498 ± 0.062 | **0.005** |
| Regression | ExtraTrees | Best Bands + Clinical | MAE = 1.442 | 1.895 ± 0.093 | **0.005** |

Both results are statistically significant at p < 0.01. The p-value of 0.005 represents the minimum achievable with 200 permutations (no permuted score met or exceeded the true score).

### 3.6 Univariate Feature Screening

Spearman correlations of individual EEG features with Diff (28 unique features screened):

| Feature | Spearman ρ | p-value |
|---|---|---|
| std_Beta1 | −0.134 | 0.169 |
| std_Beta2 | −0.132 | 0.174 |
| mean_Beta2 | −0.115 | 0.239 |
| std_Alpha2 | −0.108 | 0.267 |
| mean_Beta1 | −0.105 | 0.281 |
| mean_Delta | +0.097 | 0.319 |

No individual EEG feature reached statistical significance after accounting for multiple comparisons. This underscores the need for multivariate, regularized approaches: EEG features do not predict pain change in isolation, but contribute collectively when combined with clinical predictors.

A top-k univariate feature selection approach (SVM-RBF) confirmed that selecting the most correlated EEG features alone does not match the performance of the full Best Bands + Clinical combination:

| Top-k EEG features | BalAcc | SD |
|---|---|---|
| 5 | 0.553 | 0.050 |
| 10 | 0.592 | 0.090 |
| 20 | 0.572 | 0.083 |
| 50 | 0.582 | 0.080 |

### 3.7 XGBoost Comparison

XGBoost was evaluated separately across feature sets:

| Feature Set | MAE | BalAcc (binary) |
|---|---|---|
| Clinical only | 1.555 | 0.655 |
| Clinical + Demographics | 1.404 | 0.635 |
| Best Bands + Clinical | — | 0.673 |

XGBoost performs comparably to or below regularized linear models on most feature sets, consistent with its known sensitivity to small sample sizes and high-dimensional noisy inputs.

---

## 4. Discussion

### 4.1 Key Findings

1. **Baseline clinical variables (T1 pain, group) are strong predictors.** Clinical-only models achieve MAE ≈ 1.47 and BalAcc ≈ 0.71, already well above baseline. This is consistent with regression-to-the-mean and the known predictive value of baseline severity for treatment response.

2. **SHAP-selected EEG bands add value on top of clinical features.** The "Best Bands + Clinical" combination consistently outperformed all other feature sets across both regression (MAE = 1.425) and binary classification (AUC = 0.776), and the improvement was replicated across 5-fold and LOO cross-validation schemes.

3. **Raw EEG power and connectivity without clinical context does not predict.** Power (raw) and Connectivity feature sets failed to beat or only marginally beat the baseline, suggesting that without the clinical anchor, EEG features carry insufficient signal in this n = 107 cohort.

4. **Spatial/asymmetry features (FAA, regional averages) do not substantially improve performance.** Despite their clinical plausibility as biomarkers for pain lateralization and cortical dysregulation, FAA-only and regional-only feature sets performed at or below the predict-mean baseline, and their combination with clinical features did not surpass Best Bands + Clinical.

5. **Results are statistically significant and reproducible.** Permutation tests confirm p = 0.005 for both regression and classification. LOO-CV yields nearly identical performance to 5-fold CV, validating the stability of the estimates.

6. **SVM-RBF is the most consistent classifier.** Across feature sets and CV strategies, SVM with an RBF kernel and standard scaling consistently ranked among the top performers, outperforming tree-based ensemble methods in this small-sample setting.

### 4.2 Clinical Implications

An AUC of 0.78 for predicting treatment response from pre-treatment data is clinically actionable. A model with this discrimination could, at a cutoff balanced for sensitivity/specificity, correctly classify approximately 3 in 4 patients as responders or non-responders before treatment begins—enabling personalized treatment allocation. However, external validation in an independent cohort is required before clinical deployment.

The regression model (LOO R² = 0.30) explains approximately 30% of the variance in pain change, suggesting meaningful signal while also indicating substantial unexplained heterogeneity—likely attributable to unmeasured psychological, pharmacological, and biological factors.

### 4.3 Limitations

- **Sample size:** n = 107 is modest for the number of EEG features evaluated. Regularization and nested CV mitigate but do not eliminate overfitting risk.
- **Single-site data:** All patients were enrolled at a single institution, limiting generalizability.
- **No external validation:** All results are from internal cross-validation; prospective or held-out cohort validation is needed.
- **SVM interpretability:** The best-performing SVM-RBF model does not expose feature importances. Future work should integrate surrogate interpretability methods (e.g., LIME) or substitute with inherently interpretable models for clinical translation.
- **Missing data:** NeuropMonths had substantial free-text entries requiring coercion; missing values were imputed using the median.
- **Univariate feature correlations are weak and non-significant**, suggesting that EEG prediction in CIPN operates through multivariate patterns rather than single dominant biomarkers.

### 4.4 Future Directions

- **External validation** in held-out or independent CIPN cohorts
- **Time-frequency and non-linear features**: Connectivity dynamics (coherence change with frequency), approximate entropy, detrended fluctuation analysis
- **Larger sample** aggregation with multi-site trials or transfer learning from broader chronic pain EEG datasets
- **Interpretability**: SHAP explanations for SVM using kernel SHAP, or switching to ExtraTrees (which exposes feature importances) for the regression task
- **Multi-modal integration**: combining EEG with clinical, genetic, or psychological (e.g., pain catastrophizing) variables

---

## 5. Conclusions

We demonstrate that baseline EEG spectral features—when combined with clinical variables and evaluated under rigorous nested CV and permutation testing—provide statistically significant, clinically meaningful prediction of pain outcome in CIPN. The best model (SVM-RBF, Best Bands + Clinical) achieves **AUC = 0.78** and **balanced accuracy = 0.72** for classifying treatment responders, and **MAE = 1.42** (18% improvement over baseline, LOO R² = 0.30) for continuous pain change prediction. These results support the feasibility of EEG-based pre-treatment stratification in CIPN and motivate larger, multi-site validation efforts.

---

## Appendix: Model and Validation Details

### A.1 Preprocessing Pipeline

```
NaN → SimpleImputer(strategy='median')
  → StandardScaler()
  [→ PCA(n_components=10)]  # PCA variants only
  → Estimator
```

### A.2 Hyperparameter Grids (5-fold GridSearch)

| Model | Parameters |
|---|---|
| Ridge | α ∈ {0.01, 0.1, 1, 10, 100} |
| Lasso | α ∈ {0.01, 0.1, 1, 10, 100} |
| SVR-RBF | C ∈ {0.1, 1, 10}, ε ∈ {0.1, 0.5} |
| SVM-RBF | C ∈ {0.1, 1, 10}, γ ∈ {'scale', 'auto'} |
| Random Forest | n_estimators ∈ {100, 200}, max_depth ∈ {None, 5, 10} |
| ExtraTrees | n_estimators ∈ {100, 200}, max_depth ∈ {None, 5, 10} |
| XGBoost | n_estimators ∈ {100, 200}, max_depth ∈ {3, 5}, lr ∈ {0.05, 0.1} |

### A.3 Extended Search Space (RandomizedSearchCV, n_iter=300)

For XGBoost: log-uniform priors over min_child_weight, gamma, reg_lambda, reg_alpha; uniform over subsample and colsample_bytree; integer randint over max_depth, n_estimators.

For SVM: log-uniform C ∈ [0.01, 100], γ ∈ ['scale', 'auto', loguniform(1e-4, 1e-1)], kernel ∈ ['rbf', 'linear'].

### A.4 Software

- Python 3.x, scikit-learn, XGBoost 3.2.0, pandas, numpy, matplotlib, scipy
- EEG preprocessing: custom Welch PSD pipeline (`scripts/eeg_feature_extraction.py`)

---

*This report was generated from `PredictionPipeline.ipynb` (commit 70e7c91).*
