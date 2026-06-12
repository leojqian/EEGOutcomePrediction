#!/usr/bin/env python3
"""
Figure 1 — Study design (ABC pipeline).

Reproducible matplotlib re-draw of the supplied reference graphic: a left-to-right
pipeline (participants -> resting-state EEG -> EEG feature extraction) that branches
into the two prediction tasks (A: baseline pain, B: pain change) and the three
randomized treatment arms (NFB / Duloxetine / NFB+DL).

The feature-extraction box is drawn inline (spectral-power PSD, coherence topomap,
other-features list + baseline clinical variables), matching the reference.

Mini-plots (PSD, topomap, NRS scale, ΔPain) are schematic illustrations, not plots
of any single patient's recording.

Output -> reports/abc_paper_figs/fig1_study_design.png
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import (FancyBboxPatch, FancyArrowPatch, Circle,
                                Ellipse, Polygon, Arc)
import matplotlib as mpl

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "reports" / "abc_paper_figs"

W, H = 16.0, 8.0          # background coordinate system (matches figsize)
TITLE_C = "#1f3b66"

PAL = {
    "step":  ("#f2f6fb", "#9fb4cc"),
    "eeg":   ("#eef4fb", "#2e6da4"),
    "feat":  ("#fdf8ec", "#e0a93b"),
    "sub":   ("#ffffff", "#e6d6ad"),
    "taskA": ("#fdeeee", "#c0504d"),
    "taskB": ("#eef4fb", "#2e6da4"),
    "nfb":   ("#eaf4e6", "#6aa84f"),
    "dl":    ("#fdf3e2", "#e69138"),
    "both":  ("#f0eaf7", "#8e7cc3"),
}

BANDS = [
    ("δ", "1–4 Hz", 1, 4,  "#cfe3f3"),
    ("θ", "4–8 Hz", 4, 8,  "#c6e7d4"),
    ("α", "8–12 Hz", 8, 12, "#fbe7b2"),
    ("β", "12–30 Hz", 12, 30, "#f7c9a8"),
    ("γ", "30–45 Hz", 30, 45, "#e8c2d8"),
]

ELECTRODES = {
    "Fp1": (-.27, .80), "Fp2": (.27, .80),
    "F7": (-.62, .45), "F3": (-.33, .48), "Fz": (0, .50), "F4": (.33, .48), "F8": (.62, .45),
    "T3": (-.78, 0), "C3": (-.40, 0), "Cz": (0, 0), "C4": (.40, 0), "T4": (.78, 0),
    "T5": (-.62, -.45), "P3": (-.33, -.48), "Pz": (0, -.50), "P4": (.33, -.48), "T6": (.62, -.45),
    "O1": (-.27, -.80), "O2": (.27, -.80),
}


# --------------------------------------------------------------------------- #
# primitives
# --------------------------------------------------------------------------- #
def box(ax, x, y, w, h, key, radius=0.08, lw=1.6, dashed=False):
    fill, edge = PAL[key] if key in PAL else key
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0,rounding_size={radius}",
                       linewidth=lw, edgecolor=edge, facecolor=fill,
                       linestyle="--" if dashed else "-", zorder=2)
    ax.add_patch(p)
    return p


def arrow(ax, p0, p1, color, lw=2.2, dashed=False, scale=18):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=scale, linewidth=lw,
        color=color, zorder=3, shrinkA=1, shrinkB=1,
        linestyle="--" if dashed else "-"))


def line(ax, p0, p1, color, lw=2.0):
    ax.add_line(Line2D([p0[0], p1[0]], [p0[1], p1[1]], color=color, lw=lw,
                       zorder=3, solid_capstyle="round"))


def inset(fig, x, y, w, h):
    return fig.add_axes([x / W, y / H, w / W, h / H])


# --------------------------------------------------------------------------- #
# small icons
# --------------------------------------------------------------------------- #
def icon_people(ax, cx, cy, s=1.0, c="#5b7fa6"):
    offs = [-0.32, 0.0, 0.32]
    for i, dx in enumerate(offs):
        z = 4 if i == 1 else 3
        sc = 1.12 if i == 1 else 0.92
        ax.add_patch(Circle((cx + dx * s, cy + 0.26 * s * sc), 0.13 * s * sc,
                            color=c, zorder=z))
        ax.add_patch(FancyBboxPatch((cx + dx * s - 0.16 * s * sc, cy - 0.30 * s * sc),
                                    0.32 * s * sc, 0.40 * s * sc,
                                    boxstyle="round,pad=0,rounding_size=0.06",
                                    color=c, zorder=z))


def icon_eeg_head(ax, cx, cy, s=1.0):
    """Top-down head wearing an EEG electrode cap."""
    edge = "#3a5a80"
    ax.add_patch(Polygon([[cx - 0.10 * s, cy + 0.40 * s], [cx, cy + 0.55 * s],
                          [cx + 0.10 * s, cy + 0.40 * s]], closed=True,
                         facecolor="#eef4fb", edgecolor=edge, lw=1.5, zorder=2))
    ax.add_patch(Ellipse((cx - 0.44 * s, cy), 0.09 * s, 0.20 * s,
                        facecolor="#eef4fb", edgecolor=edge, lw=1.4, zorder=2))
    ax.add_patch(Ellipse((cx + 0.44 * s, cy), 0.09 * s, 0.20 * s,
                        facecolor="#eef4fb", edgecolor=edge, lw=1.4, zorder=2))
    ax.add_patch(Circle((cx, cy), 0.42 * s, facecolor="#f6f9fd",
                       edgecolor=edge, lw=1.9, zorder=3))
    # electrode cap layout (10–20 subset) with connecting wires
    pos = {"Fz": (0, .24), "F3": (-.20, .20), "F4": (.20, .20),
           "C3": (-.26, 0), "Cz": (0, 0), "C4": (.26, 0),
           "P3": (-.20, -.20), "Pz": (0, -.22), "P4": (.20, -.20)}
    wires = [("F3", "Fz"), ("Fz", "F4"), ("C3", "Cz"), ("Cz", "C4"),
             ("P3", "Pz"), ("Pz", "P4"), ("F3", "C3"), ("C3", "P3"),
             ("Fz", "Cz"), ("Cz", "Pz"), ("F4", "C4"), ("C4", "P4")]
    for a, b in wires:
        xa, ya = pos[a]; xb, yb = pos[b]
        ax.add_line(Line2D([cx + xa * s, cx + xb * s], [cy + ya * s, cy + yb * s],
                           color="#9fb6d2", lw=0.8, zorder=4))
    for (dx, dy) in pos.values():
        ax.add_patch(Circle((cx + dx * s, cy + dy * s), 0.045 * s,
                           facecolor="#2e6da4", edgecolor="white", lw=0.6, zorder=5))


def draw_eeg_traces(ax, n_ch=5, seed=5):
    """Schematic multi-channel resting EEG (alpha-dominant, waxing/waning)."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 4, 1200)
    labels = ["Fz", "Cz", "Pz", "O1", "O2"][:n_ch]
    for i in range(n_ch):
        af = 9 + rng.uniform(-1.2, 1.2)
        env = 0.55 + 0.45 * np.sin(2 * np.pi * 0.4 * t + rng.uniform(0, 6))
        alpha = np.sin(2 * np.pi * af * t + rng.uniform(0, 6)) * env
        theta = 0.40 * np.sin(2 * np.pi * 5 * t + rng.uniform(0, 6))
        beta = 0.18 * np.sin(2 * np.pi * 19 * t + rng.uniform(0, 6))
        noise = np.convolve(rng.standard_normal(t.size), np.ones(9) / 9, mode="same")
        sig = alpha + theta + beta + 0.5 * noise
        sig = 0.42 * sig / np.max(np.abs(sig))
        y = (n_ch - 1 - i) + sig
        ax.plot(t, y, color="#1f4e79", lw=0.8, zorder=2)
        ax.text(-0.12, n_ch - 1 - i, labels[i], ha="right", va="center",
                fontsize=6.5, color="#666")
    ax.set_xlim(-0.55, 4)
    ax.set_ylim(-0.7, n_ch - 0.3)
    ax.axis("off")


def icon_monitor(ax, cx, cy, s=1.0, c="#6aa84f"):
    ax.add_patch(FancyBboxPatch((cx - 0.30 * s, cy - 0.16 * s), 0.60 * s, 0.42 * s,
                                boxstyle="round,pad=0,rounding_size=0.04",
                                fill=False, lw=1.6, edgecolor=c, zorder=4))
    xs = np.linspace(cx - 0.24 * s, cx + 0.24 * s, 60)
    ax.add_line(Line2D(xs, cy + 0.05 * s + 0.07 * s * np.sin((xs - cx) * 22),
                       color=c, lw=1.3, zorder=5))
    ax.add_line(Line2D([cx, cx], [cy - 0.16 * s, cy - 0.27 * s], color=c, lw=1.6, zorder=4))
    ax.add_line(Line2D([cx - 0.12 * s, cx + 0.12 * s], [cy - 0.27 * s] * 2, color=c, lw=1.6, zorder=4))


def icon_pill(ax, cx, cy, s=1.0, c="#e69138"):
    ax.add_patch(FancyBboxPatch((cx - 0.30 * s, cy - 0.12 * s), 0.60 * s, 0.24 * s,
                                boxstyle="round,pad=0,rounding_size=0.12",
                                fill=False, lw=1.6, edgecolor=c, zorder=4,
                                transform=ax.transData))
    ax.add_line(Line2D([cx, cx], [cy - 0.12 * s, cy + 0.12 * s], color=c, lw=1.4, zorder=5))


def icon_brain(ax, cx, cy, s=1.0, c="#8e7cc3"):
    ax.add_patch(Ellipse((cx, cy), 0.62 * s, 0.50 * s, fill=False, lw=1.6,
                         edgecolor=c, zorder=4))
    for dx in (-0.16, 0.0, 0.16):
        ys = np.linspace(cy - 0.18 * s, cy + 0.18 * s, 30)
        ax.add_line(Line2D(cx + dx * s + 0.05 * s * np.sin(ys * 14), ys,
                           color=c, lw=1.0, zorder=5))


# --------------------------------------------------------------------------- #
# mini data panels (real axes)
# --------------------------------------------------------------------------- #
def draw_psd(ax):
    f = np.linspace(1, 45, 800)
    psd = 1.0 / (f ** 1.15) + 0.32 * np.exp(-((f - 10) ** 2) / (2 * 1.5 ** 2))
    psd += 0.05 * np.exp(-((f - 20) ** 2) / (2 * 4 ** 2))
    geo = lambda lo, hi: np.sqrt(lo * hi)   # visual centre on a log axis
    for name, hz, lo, hi, col in BANDS:
        ax.axvspan(lo, hi, color=col, zorder=0)
        ax.text(geo(lo, hi), psd.max() * 1.06, name, ha="center", va="bottom",
                fontsize=9.5, fontweight="bold", color="#333")
    ax.fill_between(f, psd, color="#cdd6e0", zorder=1)
    ax.plot(f, psd, color="#33425a", lw=1.3, zorder=2)
    ax.set_xlim(1, 45)
    ax.set_ylim(0, psd.max() * 1.38)
    ax.set_xscale("log")
    ax.set_xticks([1, 4, 8, 12, 30, 45])
    ax.set_xticklabels(["1", "4", "8", "12", "30", "45"], fontsize=6.5)
    ax.set_yticks([])
    ax.tick_params(length=2.5, pad=1.5)
    ax.set_xlabel("Frequency (Hz)", fontsize=7.0, labelpad=2)
    ax.set_ylabel("Power (μV²/Hz)", fontsize=7.0, labelpad=2)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def draw_topomap(ax):
    ax.set_aspect("equal")
    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(-1.18, 1.18)
    ax.axis("off")
    ax.add_patch(Circle((0, 0), 1.0, fill=False, lw=1.4, color="#444"))
    ax.plot([-.13, 0, .13], [0.99, 1.14, 0.99], color="#444", lw=1.4)
    ax.add_patch(Ellipse((-1.0, 0), 0.10, 0.26, fill=False, lw=1.2, color="#444"))
    ax.add_patch(Ellipse((1.0, 0), 0.10, 0.26, fill=False, lw=1.2, color="#444"))
    cmap = mpl.colormaps["coolwarm"]
    rng = np.random.default_rng(7)
    links = [("Fp1", "Fp2"), ("F3", "F4"), ("C3", "C4"), ("P3", "P4"),
             ("F3", "P3"), ("F4", "P4"), ("F7", "T5"), ("F8", "T6"),
             ("Fz", "Pz"), ("T3", "C3"), ("C4", "T4"), ("Fp2", "O1"),
             ("Fp1", "C3"), ("Cz", "Pz"), ("F4", "C4")]
    for a, b in links:
        xa, ya = ELECTRODES[a]
        xb, yb = ELECTRODES[b]
        v = rng.uniform(0, 1)
        ax.plot([xa, xb], [ya, yb], color=cmap(v), lw=0.7 + 1.2 * v,
                alpha=0.75, zorder=1)
    for (x, y) in ELECTRODES.values():
        ax.add_patch(Circle((x, y), 0.05, facecolor="white",
                            edgecolor="#c0504d", lw=0.9, zorder=3))


def draw_colorbar(ax):
    grad = np.linspace(1, 0, 256).reshape(-1, 1)
    ax.imshow(grad, aspect="auto", cmap="coolwarm",
              extent=[0, 1, 0, 1], origin="upper")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(2.2, 1.0, "1", transform=ax.transAxes, fontsize=7.0, va="center")
    ax.text(2.2, 0.0, "0", transform=ax.transAxes, fontsize=7.0, va="center")
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)


def draw_nrs(ax):
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-1, 1)
    ax.axhline(0, color="#444", lw=1.3)
    for t in range(0, 11):
        h = 0.32 if t % 5 == 0 else 0.16
        ax.plot([t, t], [-h, h], color="#444", lw=1.0)
    for t in (0, 5, 10):
        ax.text(t, -0.62, str(t), ha="center", va="top", fontsize=8)
    ax.errorbar(8.6, 0, yerr=0.42, fmt="D", color="#c0504d", ms=6,
                elinewidth=1.6, capsize=2.5, zorder=4)
    ax.set_ylim(-1.15, 1)
    ax.axis("off")


def draw_deltapain(ax):
    ax.plot([0, 1], [1.0, 0.35], ls="--", marker="o", color="#2e6da4",
            lw=2.0, ms=8, mfc="#2e6da4")
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(0, 1.25)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["T1", "T2"], fontsize=8)
    ax.set_yticks([])
    ax.set_ylabel("NRS", fontsize=8, labelpad=2)
    ax.tick_params(length=2.5, pad=2)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


# --------------------------------------------------------------------------- #
def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    # ---- vertical layout bands ------------------------------------------
    # title 7.5 | main content 2.7–6.7 | randomised note 2.45 | arms 1.0–1.9 | footer 0.3–0.9
    MID = 4.70                       # vertical centre of the pipeline row
    ax.text(8, 7.50, "Study design", ha="center", va="center",
            fontsize=23, fontweight="bold", color=TITLE_C)

    # ---- 1. Participants -------------------------------------------------
    box(ax, 0.25, 3.10, 2.05, 3.20, "step")
    cx1 = 1.275
    ax.text(cx1, 5.95, "1. Participants", ha="center", fontsize=14,
            fontweight="bold", color=TITLE_C)
    icon_people(ax, cx1, 5.05, s=0.95)
    ax.text(cx1, 4.10, "CIPN patients\n(n = 107)", ha="center", fontsize=12.5,
            color="#333", linespacing=1.3)
    ax.text(cx1, 3.45, "Baseline visit (T1)", ha="center", fontsize=10,
            color="#555")

    # ---- 2. Resting-state EEG -------------------------------------------
    box(ax, 2.60, 2.90, 2.45, 3.60, "eeg")
    cx2 = 3.825
    ax.text(cx2, 6.18, "2. Resting-state EEG", ha="center", fontsize=12.5,
            fontweight="bold", color=TITLE_C)
    ax.text(cx2, 5.84, "(19 channels)", ha="center", fontsize=10.5, color="#555")
    icon_eeg_head(ax, cx2, 5.18, s=0.82)
    ax.text(cx2, 3.12, "Eyes closed · ~5 min", ha="center", fontsize=10.5,
            color="#555")

    # ---- 3. EEG feature extraction (centred amber box, x-centre = 8.0) ---
    box(ax, 5.30, 2.70, 5.40, 4.00, "feat", radius=0.1)
    ax.text(8.00, 6.42, "3. EEG feature extraction", ha="center", fontsize=15,
            fontweight="bold", color=TITLE_C)

    # sub-panel frames (y 3.55–5.55) and their headers (above frames)
    pa = (5.46, 1.59)   # (x0, width)
    pb = (7.20, 1.59)
    pc = (8.94, 1.59)
    for (x0, w) in (pa, pb, pc):
        box(ax, x0, 3.55, w, 2.00, "sub", radius=0.05, lw=1.0)
    cxa, cxb, cxc = pa[0] + pa[1] / 2, pb[0] + pb[1] / 2, pc[0] + pc[1] / 2
    ax.text(cxa, 5.97, "a) Spectral power", ha="center", fontsize=9.8,
            fontweight="bold", color="#333")
    ax.text(cxa, 5.73, "(Frequency bands)", ha="center", fontsize=7.6, color="#555")
    ax.text(cxb, 5.97, "b) Coherence", ha="center", fontsize=9.8,
            fontweight="bold", color="#333")
    ax.text(cxb, 5.73, "(Functional connectivity)", ha="center", fontsize=7.2, color="#555")
    ax.text(cxc, 5.97, "c) Other features", ha="center", fontsize=9.8,
            fontweight="bold", color="#333")

    # other-features list (inside panel c)
    items = ["Total power (1–45 Hz)",
             "Ultra-slow / VLF\npower (0.1–1 Hz)",
             "Peak frequency",
             "Spectral entropy",
             "Phase-lag index (PLI)"]
    yy = 5.32
    for it in items:
        ax.text(pc[0] + 0.10, yy, "•", fontsize=9, color="#2e6da4", va="top")
        ax.text(pc[0] + 0.26, yy, it, fontsize=7.5, color="#333", va="top",
                linespacing=1.12)
        yy -= 0.31 + 0.31 * it.count("\n")

    # captions under panels a & b (inside their frames)
    ax.text(cxa, 3.71, "Absolute & relative\nband power (δ,θ,α,β,γ)",
            ha="center", fontsize=7.2, color="#555", va="center", linespacing=1.2)
    ax.text(cxb, 3.71, "Mean coherence over\nintra/inter-hemisphere pairs",
            ha="center", fontsize=6.8, color="#555", va="center", linespacing=1.2)

    # baseline clinical strip
    box(ax, 5.45, 2.92, 5.10, 0.52, "feat", radius=0.06, lw=1.0)
    ax.text(8.00, 3.18, "+ Baseline clinical variables  "
            "(age, neuropathy duration, etc.)", ha="center", fontsize=9.8,
            color="#7a5a16", fontweight="bold")

    # ---- pipeline arrows -------------------------------------------------
    arrow(ax, (2.30, MID), (2.57, MID), "#2e6da4")
    arrow(ax, (5.05, MID), (5.27, MID), "#2e6da4")

    # ---- 4. Task A -------------------------------------------------------
    box(ax, 11.05, 4.75, 4.70, 1.95, "taskA", radius=0.07)
    ax.text(11.25, 6.42, "4. Task A — Predict baseline pain", ha="left",
            fontsize=12, fontweight="bold", color="#a23b39")
    ax.text(13.45, 5.92, "Target:\nPain at T1 (NRS)", ha="left", fontsize=10.5,
            color="#333", va="center", linespacing=1.25)
    ax.text(13.45, 5.10, "Inputs:\nEEG features + age\n+ neuropathy duration",
            ha="left", fontsize=9.6, color="#444", va="center", linespacing=1.25)

    # ---- 5. Task B -------------------------------------------------------
    box(ax, 11.05, 2.70, 4.70, 1.95, "taskB", radius=0.07)
    ax.text(11.25, 4.37, "5. Task B — Predict pain change", ha="left",
            fontsize=12, fontweight="bold", color="#1f4e79")
    ax.text(13.05, 3.87, "Target:\nΔPain = Pain(T2) − Pain(T1)", ha="left",
            fontsize=9.6, color="#333", va="center", linespacing=1.25)
    ax.text(13.05, 3.05, "Inputs:\nEEG features + T1 pain\n+ age + neuropathy duration",
            ha="left", fontsize=9.2, color="#444", va="center", linespacing=1.25)

    # branch arrows from feature box to the two tasks (orthogonal bus)
    line(ax, (10.70, MID), (10.90, MID), "#8a97a8")
    line(ax, (10.90, 3.55), (10.90, 5.72), "#8a97a8")
    arrow(ax, (10.90, 5.72), (11.02, 5.72), "#c0504d")
    arrow(ax, (10.90, 3.55), (11.02, 3.55), "#2e6da4")

    # ---- treatment arms (aligned under the task column) -----------------
    ax.text(13.40, 2.42, "(patient randomised to one arm)", ha="center",
            fontsize=8.8, style="italic", color="#555")
    arms = [("nfb", 11.05, 1.45, "NFB", icon_monitor, "#3d6b29"),
            ("dl", 12.65, 1.50, "Duloxetine", icon_pill, "#9c5a14"),
            ("both", 14.30, 1.45, "NFB + DL", icon_brain, "#5e4b86")]
    for key, x0, w, label, icon, tc in arms:
        box(ax, x0, 1.00, w, 0.90, key, radius=0.08)
        icon(ax, x0 + 0.34, 1.45, s=0.72)
        ax.text(x0 + 0.62 + (w - 0.62) / 2, 1.45, label, ha="center", va="center",
                fontsize=9.5, fontweight="bold", color=tc)
    for tx in (11.775, 13.40, 15.025):
        arrow(ax, (13.40, 2.28), (tx, 1.94), "#9a9a9a", lw=1.3, dashed=True, scale=11)

    # ---- footer (centred on the figure midline) -------------------------
    box(ax, 4.00, 0.30, 8.00, 0.62, ("#fbfbfb", "#bcbcbc"), radius=0.06,
        lw=1.1, dashed=True)
    icon_people(ax, 4.85, 0.61, s=0.5, c="#8a8a8a")
    ax.text(5.35, 0.61, "Single-center randomized trial    •    "
            "107 patients with paired baseline EEG + T1/T2 pain ratings",
            ha="left", va="center", fontsize=9.8, style="italic", color="#555")

    # ---- real mini-axes (added on top) ----------------------------------
    draw_eeg_traces(inset(fig, 2.86, 3.45, 1.98, 1.32))
    draw_psd(inset(fig, 5.62, 4.18, 1.30, 1.12))
    draw_topomap(inset(fig, 7.24, 4.05, 1.40, 1.40))
    draw_colorbar(inset(fig, 8.66, 4.25, 0.07, 0.90))
    draw_nrs(inset(fig, 11.30, 5.18, 1.85, 0.80))
    draw_deltapain(inset(fig, 11.30, 2.96, 1.70, 1.08))

    out = OUTDIR / "fig1_study_design.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
