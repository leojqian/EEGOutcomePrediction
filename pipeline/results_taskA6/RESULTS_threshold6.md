# Task A results — threshold 6

**Task:** predict whether baseline pain `pain_t1 >= 6` (high vs. low baseline pain)
from resting EEG + age + neuropathy duration. Binary classification, leakage-controlled
5-fold CV (Module B band selection + Module C train/eval re-run per fold on training
patients only). `pain_t1` is the target (excluded as a feature); treatment modality excluded.

**Cohort:** n = 107, 63 high-pain / 44 low-pain (positive rate 0.589 — near-balanced,
so chance ≈ 0.50 for AUC and balanced accuracy).

## Per-fold (held-out validation)

| fold | model | n_val | train acc | val acc | bal acc | AUC | AP | F1 | τ |
|---|---|---|---|---|---|---|---|---|---|
| 0 | logreg | 22 | 0.659 | 0.545 | 0.513 | 0.590 | 0.719 | 0.643 | 0.500 |
| 1 | ensemble | 22 | 0.976 | 0.727 | 0.701 | 0.761 | 0.807 | 0.786 | 0.575 |
| 2 | ensemble | 21 | 1.000 | 0.333 | 0.341 | 0.346 | 0.561 | 0.364 | 0.580 |
| 3 | logreg | 21 | 0.651 | 0.476 | 0.458 | 0.435 | 0.524 | 0.560 | 0.500 |
| 4 | ensemble | 21 | 0.930 | 0.476 | 0.486 | 0.463 | 0.548 | 0.476 | 0.635 |
| **mean ± SD** | | | 0.843 ± 0.174 | 0.512 ± 0.143 | 0.500 ± 0.130 | **0.519 ± 0.161** | 0.632 ± 0.124 | 0.566 ± 0.161 | |

## Pooled out-of-fold (all 107 held-out predictions)

| metric | value |
|---|---|
| **AUC** | **0.499** |
| Accuracy | 0.514 |
| Balanced accuracy | 0.502 |
| Sensitivity (high-pain recall) | 36/63 = 0.571 |
| Specificity (low-pain recall) | 19/44 = 0.432 |
| Confusion `[[TN,FP],[FN,TP]]` | `[[19,25],[27,36]]` |

## Verdict

**Chance.** Pooled OOF AUC 0.499 and balanced accuracy 0.502 are indistinguishable from
random. Mean training accuracy 0.843 (two folds at/near 1.0) against held-out at chance is
textbook overfitting — the models memorize training patients and transfer nothing. Per-fold
AUC scatters 0.35–0.76; fold 1's 0.76 is a lucky draw that pooling washes out.

Baseline pain level is **not** recoverable from resting EEG + demographics in this cohort.
Contrast with Task B (treatment-response prediction), pooled OOF AUC ≈ 0.71 — EEG carries
signal about how pain *changes* with treatment, not about the current pain *level*.

## Per-fold selected bands (Module B, top-2 per sheet)

| fold | coherence | phase-lag | abs power |
|---|---|---|---|
| 0 | Beta2, Delta | Alpha2, Alpha | Alpha2, HighBeta |
| 1 | Beta3, Delta | HighBeta, Beta2 | Delta, Alpha2 |
| 2 | Alpha1, Delta | Alpha, Beta | Alpha2, Beta |
| 3 | Delta, Alpha | HighBeta, Theta | Alpha, Beta3 |
| 4 | Beta, Alpha1 | HighBeta, Beta | Alpha, Beta |

Recurrence (≥2/5 folds): coherence Delta 4/5; power Alpha2 3/5; phase-lag HighBeta 3/5.
Note: band recurrence is not meaningful when the downstream model is at chance — it reflects
which bands the inner-CV tally happened to favor, not a validated signal.

_Source: pipeline/results_taskA6/fold_*/taskA_top2/{test_metrics.json, test_predictions.csv}_
