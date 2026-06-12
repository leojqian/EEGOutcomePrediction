#!/usr/bin/env python3
"""
EEG feature-extraction figure (pulled out of the study-design overview).

Three panels:
  a) Spectral power   — schematic resting PSD with band shading (δ θ α β γ)
  b) Coherence        — schematic 19-channel head with inter-/intra-hemispheric links
  c) Other features   — total power, very-low-freq power, peak frequency,
                        spectral entropy, phase-lag index (PLI)
Footer: baseline clinical variables appended to every feature vector.

Output -> reports/abc_paper_figs/fig_feature_extraction.png

This is schematic (illustrative band boundaries / electrode layout), not a plot of
any single patient's data.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, Circle

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "reports" / "abc_paper_figs"
TITLE_C = "#1f3b66"

# canonical band edges (Hz) and colours for the schematic
BANDS = [
    ("δ", 1, 4,  "#cfe3f3"),
    ("θ", 4, 8,  "#bfe0cf"),
    ("α", 8, 13, "#fbe7b2"),
    ("β", 13, 30, "#f7c9a8"),
    ("γ", 30, 45, "#e8c2d8"),
]


def panel_spectral(ax):
    f = np.linspace(1, 45, 800)
    # 1/f background + alpha bump -> recognizable resting PSD
    psd = 1.0 / (f ** 1.1) + 0.30 * np.exp(-((f - 10) ** 2) / (2 * 1.6 ** 2))
    psd += 0.06 * np.exp(-((f - 20) ** 2) / (2 * 4 ** 2))
    for name, lo, hi, c in BANDS:
        ax.axvspan(lo, hi, color=c, alpha=0.9, zorder=0)
        ax.text((lo + hi) / 2, psd.max() * 1.02, name, ha="center",
                va="bottom", fontsize=11, color="#333")
    ax.plot(f, psd, color="#1f3b66", lw=2, zorder=3)
    ax.set_xlim(1, 45)
    ax.set_ylim(0, psd.max() * 1.15)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (μV²/Hz)")
    ax.set_yticks([])
    ax.set_title("a) Spectral power (frequency bands)")
    ax.text(0.5, -0.26, "Absolute & relative band power, per channel",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.5,
            color="0.3")


# rough 19-channel 10–20 layout on a unit head (x right, y front)
ELECTRODES = {
    "Fp1": (-.27, .80), "Fp2": (.27, .80),
    "F7": (-.62, .45), "F3": (-.33, .48), "Fz": (0, .50), "F4": (.33, .48), "F8": (.62, .45),
    "T3": (-.78, 0), "C3": (-.40, 0), "Cz": (0, 0), "C4": (.40, 0), "T4": (.78, 0),
    "T5": (-.62, -.45), "P3": (-.33, -.48), "Pz": (0, -.50), "P4": (.33, -.48), "T6": (.62, -.45),
    "O1": (-.27, -.80), "O2": (.27, -.80),
}


def panel_coherence(ax):
    ax.set_aspect("equal")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.15, 1.15)
    ax.axis("off")
    # head outline + nose
    ax.add_patch(Circle((0, 0), 1.0, fill=False, lw=2, color="#444"))
    ax.plot([-.12, 0, .12], [1.0, 1.13, 1.0], color="#444", lw=2)
    pts = ELECTRODES
    # a few representative inter-/intra-hemispheric links
    links = [("Fp1", "Fp2"), ("F3", "F4"), ("C3", "C4"), ("P3", "P4"),
             ("F3", "P3"), ("F4", "P4"), ("F7", "T5"), ("F8", "T6"),
             ("Fz", "Pz"), ("T3", "C3"), ("C4", "T4"), ("Fp2", "O1")]
    for a, b in links:
        xa, ya = pts[a]
        xb, yb = pts[b]
        ax.plot([xa, xb], [ya, yb], color="#2e6da4", lw=1.0, alpha=0.55, zorder=1)
    for (x, y) in pts.values():
        ax.add_patch(Circle((x, y), 0.055, color="#c0504d", zorder=2))
    ax.set_title("b) Coherence (functional connectivity)")
    ax.text(0.5, -0.06, "Mean coherence over intra- & inter-hemispheric pairs",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.5,
            color="0.3")


def panel_other(ax):
    ax.axis("off")
    ax.set_title("c) Other features")
    items = [
        "Total power (1–45 Hz)",
        "Ultra-slow / very-low-freq (ULV) power (0.1–1 Hz)",
        "Peak frequency",
        "Spectral entropy",
        "Phase-lag index (PLI)",
    ]
    y = 0.86
    for it in items:
        ax.text(0.06, y, "•", fontsize=13, color="#2e6da4", va="center")
        ax.text(0.13, y, it, fontsize=9.5, color="#333", va="center")
        y -= 0.165


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(12, 4.6))
    gs = GridSpec(1, 3, width_ratios=[1.25, 1.0, 1.05], wspace=0.32,
                  left=0.06, right=0.98, top=0.84, bottom=0.2)
    panel_spectral(fig.add_subplot(gs[0, 0]))
    panel_coherence(fig.add_subplot(gs[0, 1]))
    panel_other(fig.add_subplot(gs[0, 2]))

    fig.suptitle("EEG feature extraction", fontsize=14, fontweight="bold",
                 color=TITLE_C, x=0.52)
    fig.text(0.06, 0.045,
             "+ Baseline clinical variables (age, neuropathy duration, T1 pain) "
             "appended to every patient feature vector.",
             ha="left", va="center", fontsize=9, style="italic", color="#555")
    out = OUTDIR / "fig_feature_extraction.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
