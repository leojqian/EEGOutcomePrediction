#!/usr/bin/env python3
"""
Figure 3 (main) + per-fold supplementary figure for the ABC pipeline (Task B).

Main figure  -> reports/abc_paper_figs/fig3_performance.png
    Two panels, POOLED out-of-fold only:
      A. Pooled OOF ROC curve (all 107 held-out predictions)
      B. Pooled OOF confusion matrix

Supplementary figure -> reports/abc_paper_figs/figS_per_fold_performance.png
      A. Per-fold ROC curves (+ pooled overlay)
      B. Per-fold held-out validation metrics (acc / bal.acc / ROC-AUC / F1)

All numbers are read straight from the per-fold artifacts under
pipeline/results/fold_{k}/top2_per_sheet_v3_mod/.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "pipeline" / "results"
OUTDIR = ROOT / "reports" / "abc_paper_figs"
TAG = "top2_per_sheet_v3_mod"
FOLDS = [0, 1, 2, 3, 4]

# colour-blind-friendly palette
C_ROC = "#1f4e79"      # pooled ROC / primary
C_FOLDS = ["#4477AA", "#66CCEE", "#228833", "#CCBB44", "#EE6677"]
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "figure.dpi": 150,
})


def load_fold(k):
    base = RESULTS / f"fold_{k}" / TAG
    preds = pd.read_csv(base / "test_predictions.csv")
    metrics = json.loads((base / "test_metrics.json").read_text())
    return preds, metrics


def gather():
    rows, metrics = [], {}
    for k in FOLDS:
        preds, m = load_fold(k)
        preds = preds.assign(fold=k)
        rows.append(preds)
        metrics[k] = m
    pooled = pd.concat(rows, ignore_index=True)
    return pooled, metrics


def panel_pooled_roc(ax, pooled):
    y, p = pooled["y_true"].to_numpy(), pooled["proba"].to_numpy()
    fpr, tpr, _ = roc_curve(y, p)
    auc = roc_auc_score(y, p)
    ax.plot(fpr, tpr, color=C_ROC, lw=2.6, label=f"Pooled OOF  AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], ls="--", color="0.6", lw=1)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"A. Pooled out-of-fold ROC  (n = {len(y)})")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.set_aspect("equal")
    return auc


def panel_pooled_cm(ax, pooled):
    y, yhat = pooled["y_true"].to_numpy(), pooled["y_pred"].to_numpy()
    cm = confusion_matrix(y, yhat, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    acc = (tn + tp) / cm.sum()
    sens = tp / (tp + fn)   # non-improver recall (class 1)
    spec = tn / (tn + fp)   # improver recall (class 0)
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
    labels = ["Improver", "Non-improver"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    thr = cm.max() * 0.6
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    fontsize=20, fontweight="bold",
                    color="white" if cm[i, j] > thr else "#1f4e79")
    ax.set_title(f"B. Pooled OOF confusion matrix\nacc {acc:.3f} · sens {sens:.3f} · spec {spec:.3f}")
    return acc, sens, spec


def make_main(pooled):
    fig = plt.figure(figsize=(9.2, 4.3))
    gs = GridSpec(1, 2, width_ratios=[1, 1], wspace=0.32,
                  left=0.08, right=0.97, top=0.84, bottom=0.14)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    auc = panel_pooled_roc(ax0, pooled)
    panel_pooled_cm(ax1, pooled)
    fig.suptitle("Figure 3 — ABC pipeline pooled performance "
                 "(Task B: ΔPain ≥ −2, + treatment modality)",
                 fontsize=12, fontweight="bold")
    out = OUTDIR / "fig3_performance.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}  (pooled AUC={auc:.3f})")


def panel_perfold_roc(ax, pooled, metrics):
    for k in FOLDS:
        sub = pooled[pooled["fold"] == k]
        y, p = sub["y_true"].to_numpy(), sub["proba"].to_numpy()
        fpr, tpr, _ = roc_curve(y, p)
        auc = roc_auc_score(y, p)
        model = metrics[k]["model"]
        ax.plot(fpr, tpr, color=C_FOLDS[k], lw=1.5, alpha=0.9,
                label=f"Fold {k} ({model})  AUC={auc:.3f}")
    y, p = pooled["y_true"].to_numpy(), pooled["proba"].to_numpy()
    fpr, tpr, _ = roc_curve(y, p)
    ax.plot(fpr, tpr, color="black", lw=2.6,
            label=f"Pooled (n={len(y)})  AUC={roc_auc_score(y, p):.3f}")
    ax.plot([0, 1], [0, 1], ls="--", color="0.6", lw=1)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("A. ROC — per fold + pooled out-of-fold")
    ax.legend(loc="lower right", frameon=False, fontsize=7.5)
    ax.set_aspect("equal")


def panel_perfold_bars(ax, metrics):
    metric_keys = ["accuracy", "balanced_accuracy", "auc", "f1"]
    metric_lab = ["Accuracy", "Balanced Acc.", "ROC-AUC", "F1"]
    bar_colors = ["#4477AA", "#228833", "#EE6677", "#CCBB44"]
    n = len(FOLDS)
    x = np.arange(n)
    w = 0.2
    for i, key in enumerate(metric_keys):
        vals = [metrics[k][key] for k in FOLDS]
        ax.bar(x + (i - 1.5) * w, vals, w, label=metric_lab[i], color=bar_colors[i])
    ax.axhline(0.5, ls="--", color="0.5", lw=1)
    ax.text(n - 0.5, 0.51, "Chance (BA=0.50)", ha="right", va="bottom",
            fontsize=8, color="0.4")
    ax.set_xticks(x, [f"Fold {k}\n({metrics[k]['model']})" for k in FOLDS], fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("B. Per-fold held-out validation metrics")
    ax.legend(ncol=2, fontsize=8, frameon=False, loc="upper left")
    # mean +/- sd annotation
    means = {m: np.mean([metrics[k][m] for k in FOLDS]) for m in metric_keys}
    sds = {m: np.std([metrics[k][m] for k in FOLDS], ddof=1) for m in metric_keys}
    txt = "Mean ± SD across folds:\n" + "  ".join(
        f"{lab.split()[0]} {means[k]:.3f}±{sds[k]:.3f}"
        for lab, k in zip(metric_lab, metric_keys))
    ax.text(0.5, -0.22, txt, transform=ax.transAxes, ha="center",
            va="top", fontsize=7.5, color="0.25")


def make_supp(pooled, metrics):
    fig = plt.figure(figsize=(12.5, 4.6))
    gs = GridSpec(1, 2, width_ratios=[1, 1.25], wspace=0.22,
                  left=0.06, right=0.98, top=0.88, bottom=0.2)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    panel_perfold_roc(ax0, pooled, metrics)
    panel_perfold_bars(ax1, metrics)
    fig.suptitle("Figure S — ABC pipeline per-fold performance "
                 "(Task B: ΔPain ≥ −2, + treatment modality)",
                 fontsize=12, fontweight="bold")
    out = OUTDIR / "figS_per_fold_performance.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    pooled, metrics = gather()
    make_main(pooled)
    make_supp(pooled, metrics)


if __name__ == "__main__":
    main()
