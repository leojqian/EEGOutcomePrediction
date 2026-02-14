# ============================================================
# EEG TXT -> FFT/PSD Features + Connectivity (Coherence / PLI)
# Single output: one Excel file with many sheets.
#
# Dependencies:
#   pip install numpy pandas scipy mne mne-connectivity openpyxl
# ============================================================

import os
import re
import numpy as np
import pandas as pd
from scipy.signal import welch

import mne
from mne_connectivity import spectral_connectivity_epochs  # type: ignore[import-not-found]


# ----------------------------
# Bands (exact as you listed)
# ----------------------------
BANDS_FULL = {
    "Delta": (1.0, 4.0),
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 12.0),
    "Beta": (12.0, 25.0),
    "HighBeta": (25.0, 30.0),
    "Gamma": (30.0, 40.0),
    "HighGamma": (40.0, 50.0),
    "Alpha1": (8.0, 10.0),
    "Alpha2": (10.0, 12.0),
    "Beta1": (12.0, 15.0),
    "Beta2": (15.0, 18.0),
    "Beta3": (18.0, 25.0),
    "Gamma1": (30.0, 35.0),
    "Gamma2": (35.0, 40.0),
}

BANDS_Z = {
    "Delta": (1.0, 4.0),
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 12.0),
    "Beta": (12.0, 25.0),
    "HighBeta": (25.0, 30.0),
    "Alpha1": (8.0, 10.0),
    "Alpha2": (10.0, 12.0),
    "Beta1": (12.0, 15.0),
    "Beta2": (15.0, 18.0),
    "Beta3": (18.0, 25.0),
}

# Peak frequency bands: only Theta, Alpha, Beta, HighBeta are meaningful
# (Delta and Alpha2 have zero variance and are not informative)
BANDS_PEAK_FREQ = ["Theta", "Alpha", "Beta", "HighBeta"]


# ----------------------------
# 1) Parse your TXT file
# ----------------------------
def read_custom_eeg_txt(path: str):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.rstrip("\n") for ln in f]

    meta = {}
    header_line_idx = None
    ch_names_raw = None

    for i, ln in enumerate(lines):
        # metadata lines: "Field:\tValue"
        if ":" in ln and "\t" in ln:
            k, v = ln.split("\t", 1)
            k = k.strip().rstrip(":")
            meta[k] = v.strip()

        # channel header heuristic: many tokens containing "-"
        if "\t" in ln and re.search(r"[A-Za-z0-9]+-[A-Za-z0-9]+", ln):
            tokens = ln.split("\t")
            if sum(("-" in t) for t in tokens) >= 10:
                header_line_idx = i
                ch_names_raw = tokens
                break

    if header_line_idx is None or ch_names_raw is None:
        raise ValueError("Could not find channel header line in the TXT file.")

    data_rows = []
    for ln in lines[header_line_idx + 1 :]:
        if not ln.strip():
            continue
        parts = ln.split("\t")
        if len(parts) != len(ch_names_raw):
            continue
        try:
            row = [float(x) for x in parts]
            data_rows.append(row)
        except ValueError:
            continue

    if len(data_rows) == 0:
        raise ValueError("No numeric EEG rows found after the channel header.")

    data_uV = np.asarray(data_rows, dtype=np.float64)  # shape: (n_samples, n_channels)

    if "Sampling Rate" not in meta:
        raise ValueError("Sampling Rate not found in metadata.")
    sfreq = float(meta["Sampling Rate"])

    return meta, ch_names_raw, data_uV, sfreq


def rename_to_LE(ch_names_raw):
    out = []
    for ch in ch_names_raw:
        base = ch.split("-", 1)[0].strip()
        out.append(f"{base}-LE")
    return out


# ----------------------------
# 2) PSD via Welch (uV^2/Hz)
# ----------------------------
def welch_psd_uV(data_uV, sfreq, win_sec=2.0, overlap=0.5, fmin=0.5, fmax=50.0):
    # input: data_uV shape (n_samples, n_channels)
    nperseg = int(win_sec * sfreq)
    nperseg = max(nperseg, 8)
    noverlap = int(nperseg * overlap)

    x = data_uV.T  # shape: (n_channels, n_samples)
    freqs, psd = welch(
        x,
        fs=sfreq,
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        scaling="density",
        axis=-1,
    )
    # welch returns: freqs shape (n_freqs,), psd shape (n_channels, n_freqs)

    band = (freqs >= fmin) & (freqs <= fmax)
    return freqs[band], psd[:, band]  # freqs: (n_freqs_filtered,), psd: (n_channels, n_freqs_filtered)


def bandpower_from_psd(psd, freqs, f_lo, f_hi):
    # input: psd shape (n_channels, n_freqs), freqs shape (n_freqs,)
    m = (freqs >= f_lo) & (freqs < f_hi)
    if not np.any(m):
        return np.zeros(psd.shape[0])  # shape: (n_channels,)
    return np.trapezoid(psd[:, m], freqs[m], axis=1)  # output shape: (n_channels,)


def per_hz_power(psd, freqs, hz_min=1, hz_max=50):
    # input: psd shape (n_channels, n_freqs), freqs shape (n_freqs,)
    cols, mats = [], []
    for hz in range(hz_min, hz_max + 1):
        f_lo, f_hi = hz - 0.5, hz + 0.5
        mats.append(bandpower_from_psd(psd, freqs, f_lo, f_hi))  # each: (n_channels,)
        cols.append(f"{hz}Hz")
    mat = np.stack(mats, axis=1)  # shape: (n_channels, n_hz_bins)
    return mat, cols


def peak_freq_per_band(psd, freqs, bands_dict):
    # input: psd shape (n_channels, n_freqs), freqs shape (n_freqs,)
    out, cols = [], []
    for name, (f_lo, f_hi) in bands_dict.items():
        m = (freqs >= f_lo) & (freqs < f_hi)
        if not np.any(m):
            peak = np.full(psd.shape[0], np.nan)  # shape: (n_channels,)
        else:
            band_psd = psd[:, m]  # shape: (n_channels, n_freqs_in_band)
            band_freqs = freqs[m]  # shape: (n_freqs_in_band,)
            peak = band_freqs[np.argmax(band_psd, axis=1)]  # shape: (n_channels,)
        out.append(peak)
        cols.append(name)
    return np.stack(out, axis=1), cols  # output: (n_channels, n_bands), list


def zscore_across_rows(df: pd.DataFrame):
    mu = df.mean(axis=0)
    sd = df.std(axis=0, ddof=0).replace(0, np.nan)
    return (df - mu) / sd


# ----------------------------
# 3) MNE epochs for connectivity
# ----------------------------
def make_epochs_mne(data_uV, sfreq, ch_names, epoch_sec=8.0, overlap=0.5):
    # input: data_uV shape (n_samples, n_channels)
    data_V = (data_uV * 1e-6).T  # shape: (n_channels, n_samples)
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data_V, info, verbose=False)

    step = epoch_sec * (1.0 - overlap)
    overlap_sec = epoch_sec - step

    epochs = mne.make_fixed_length_epochs(
        raw, duration=epoch_sec, overlap=overlap_sec, preload=True, verbose=False
    )
    # output: epochs object with shape (n_epochs, n_channels, n_samples_per_epoch)
    return epochs


def build_unique_pair_indices(n_ch: int):
    """Return indices (i_idx, j_idx) for unique undirected pairs i<j."""
    ii, jj = [], []
    for i in range(n_ch):
        for j in range(i + 1, n_ch):
            ii.append(i)
            jj.append(j)
    return np.array(ii, dtype=int), np.array(jj, dtype=int)


def connectivity_per_band_fixed_pairs(epochs, bands_dict, method="coh"):
    """
    Compute connectivity on explicitly-defined unique pairs (i<j).
    Returns DataFrame: index = "ch1__ch2", columns = band names.
    """
    if spectral_connectivity_epochs is None:
        raise ImportError(
            "mne-connectivity is required for connectivity features. "
            "Install it with: pip install mne-connectivity"
        )
    # input: epochs shape (n_epochs, n_channels, n_samples_per_epoch)
    ch_names = epochs.ch_names
    n_ch = len(ch_names)
    i_idx, j_idx = build_unique_pair_indices(n_ch)  # each: (n_pairs,) where n_pairs = n_ch*(n_ch-1)/2
    pair_index = [f"{ch_names[i]}__{ch_names[j]}" for i, j in zip(i_idx, j_idx)]  # length: n_pairs

    cols = []
    mats = []

    for bname, (fmin, fmax) in bands_dict.items():
        con = spectral_connectivity_epochs(
            epochs,
            method=method,      # "coh", "pli", or "wpli"
            mode="fourier",
            sfreq=epochs.info["sfreq"],
            fmin=fmin,
            fmax=fmax,
            faverage=True,
            indices=(i_idx, j_idx),   # <<< critical fix
            verbose=False,
        )

        vals = con.get_data()
        # Expect (n_pairs, 1) or (n_pairs,)
        if vals.ndim == 2:
            vals = vals[:, 0]  # shape: (n_pairs,)
        elif vals.ndim != 1:
            raise ValueError(f"Unexpected connectivity data shape: {vals.shape}")

        if len(vals) != len(pair_index):
            raise ValueError(
                f"Connectivity returned {len(vals)} values, expected {len(pair_index)}. "
                f"Check MNE-connectivity version / indices handling."
            )

        mats.append(vals)  # each: (n_pairs,)
        cols.append(bname)

    mat = np.stack(mats, axis=1)  # shape: (n_pairs, n_bands)
    df = pd.DataFrame(mat, index=pair_index, columns=cols)  # shape: (n_pairs, n_bands)
    return df


# ----------------------------
# 4) Save everything into one Excel
# ----------------------------
def save_all_to_one_excel(results: dict, out_xlsx: str):
    tables = {
        "meta": pd.DataFrame([results["meta"]]).T.rename(columns={0: "value"}),
        "FFT_abs_bandpower_uV2": results["df_abs"],
        "Z_FFT_abs_bandpower_uV2": results["df_abs_z"],
        "FFT_rel_bandpower_pct": results["df_rel"],
        "Z_FFT_rel_bandpower_pct": results["df_rel_z"],
        "FFT_abs_1to50Hz_uV2": results["df_abs_hz"],
        "Z_FFT_abs_1to30Hz_uV2": results["df_abs_hz_z_1_30"],
        "Z_FFT_abs_1to50Hz_uV2": results["df_abs_hz_z_1_50"],
        "FFT_rel_1to50Hz_pct": results["df_rel_hz"],
        "Z_FFT_rel_1to30Hz_pct": results["df_rel_hz_z_1_30"],
        "Z_FFT_rel_1to50Hz_pct": results["df_rel_hz_z_1_50"],
        "PeakFreq_Hz": results["df_peak"],
        "Z_PeakFreq_Hz": results["df_peak_z"],
        "FFT_Coherence": results["df_coh"],
        "Z_FFT_Coherence": results["df_coh_z"],
        "FFT_PhaseLag_PLI": results["df_pli"],
        "Z_FFT_PhaseLag_PLI": results["df_pli_z"],
    }

    def safe_sheet(name):
        name = re.sub(r"[\[\]\:\*\?\/\\]", "_", name)
        return name[:31]

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        for name, df in tables.items():
            df.to_excel(writer, sheet_name=safe_sheet(name))


# ----------------------------
# 5) Main extraction function
# ----------------------------
def extract_fft_feature_tables(
    txt_path: str,
    out_prefix: str,
    win_sec_psd: float = 2.0,
    conn_epoch_sec: float = 8.0,
    drop_A1A2: bool = True,
    overlap: float = 0.5,
    output_one_excel: bool = True,
):
    meta, ch_raw, data_uV, sfreq = read_custom_eeg_txt(txt_path)
    # data_uV shape: (n_samples, n_channels)
    ch = rename_to_LE(ch_raw)

    # Drop A1/A2 if present (to match your 19 electrode list)
    if drop_A1A2:
        keep = [i for i, name in enumerate(ch) if not name.startswith(("A1-", "A2-"))]
        data_uV = data_uV[:, keep]  # shape: (n_samples, n_channels_filtered)
        ch = [ch[i] for i in keep]

    # ---------- PSD / FFT features ----------
    freqs, psd = welch_psd_uV(
        data_uV, sfreq, win_sec=win_sec_psd, overlap=overlap, fmin=0.5, fmax=50.0
    )
    # freqs shape: (n_freqs,), psd shape: (n_channels, n_freqs)

    total_power = bandpower_from_psd(psd, freqs, 1.0, 50.0)  # shape: (n_channels,)

    abs_band = {name: bandpower_from_psd(psd, freqs, f_lo, f_hi)
                for name, (f_lo, f_hi) in BANDS_FULL.items()}
    # abs_band: dict[str, np.ndarray] where each value has shape (n_channels,)
    #           keys = band names, values = bandpower arrays per channel
    df_abs = pd.DataFrame(abs_band, index=ch)  # shape: (n_channels, n_bands_full)
    df_abs.index.name = "Channel"

    rel_band = {name: 100.0 * (df_abs[name].values / np.maximum(total_power, 1e-12))
                for name in BANDS_FULL.keys()}
    # rel_band: dict with values shape (n_channels,) each
    df_rel = pd.DataFrame(rel_band, index=ch)  # shape: (n_channels, n_bands_full)
    df_rel.index.name = "Channel"

    df_abs_z = zscore_across_rows(df_abs[list(BANDS_Z.keys())])  # shape: (n_channels, n_bands_z)
    df_rel_z = zscore_across_rows(df_rel[list(BANDS_Z.keys())])  # shape: (n_channels, n_bands_z)

    abs_hz_mat, hz_cols = per_hz_power(psd, freqs, 1, 50)
    # abs_hz_mat shape: (n_channels, 50)
    df_abs_hz = pd.DataFrame(abs_hz_mat, index=ch, columns=hz_cols)  # shape: (n_channels, 50)
    df_abs_hz.index.name = "Channel"

    rel_hz_mat = 100.0 * (abs_hz_mat / np.maximum(total_power[:, None], 1e-12))
    # rel_hz_mat shape: (n_channels, 50)
    df_rel_hz = pd.DataFrame(rel_hz_mat, index=ch, columns=hz_cols)  # shape: (n_channels, 50)
    df_rel_hz.index.name = "Channel"

    df_abs_hz_z_1_50 = zscore_across_rows(df_abs_hz)  # shape: (n_channels, 50)
    df_rel_hz_z_1_50 = zscore_across_rows(df_rel_hz)  # shape: (n_channels, 50)
    df_abs_hz_z_1_30 = zscore_across_rows(df_abs_hz[[f"{i}Hz" for i in range(1, 31)]])  # shape: (n_channels, 30)
    df_rel_hz_z_1_30 = zscore_across_rows(df_rel_hz[[f"{i}Hz" for i in range(1, 31)]])  # shape: (n_channels, 30)

    peak_mat, peak_cols = peak_freq_per_band(psd, freqs, BANDS_FULL)
    # peak_mat shape: (n_channels, n_bands_full)
    df_peak = pd.DataFrame(peak_mat, index=ch, columns=peak_cols)  # shape: (n_channels, n_bands_full)
    df_peak.index.name = "Channel"
    # Only z-score peak frequency for meaningful bands (exclude Delta and Alpha2)
    df_peak_z = zscore_across_rows(df_peak[BANDS_PEAK_FREQ])  # shape: (n_channels, 4)

    # ---------- Connectivity features ----------
    # Use longer epochs for connectivity to make Delta/Theta stable
    epochs = make_epochs_mne(data_uV, sfreq, ch, epoch_sec=conn_epoch_sec, overlap=overlap)
    # epochs shape: (n_epochs, n_channels, n_samples_per_epoch)

    df_coh = connectivity_per_band_fixed_pairs(epochs, BANDS_Z, method="coh")
    # df_coh shape: (n_pairs, n_bands_z) where n_pairs = n_channels*(n_channels-1)/2
    df_coh_z = zscore_across_rows(df_coh)  # shape: (n_pairs, n_bands_z)

    df_pli = connectivity_per_band_fixed_pairs(epochs, BANDS_Z, method="pli")
    # df_pli shape: (n_pairs, n_bands_z)
    df_pli_z = zscore_across_rows(df_pli)  # shape: (n_pairs, n_bands_z)

    results = {
        "meta": meta,
        "channels": ch,
        "sfreq": sfreq,
        "df_abs": df_abs,
        "df_abs_z": df_abs_z,
        "df_rel": df_rel,
        "df_rel_z": df_rel_z,
        "df_abs_hz": df_abs_hz,
        "df_abs_hz_z_1_30": df_abs_hz_z_1_30,
        "df_abs_hz_z_1_50": df_abs_hz_z_1_50,
        "df_rel_hz": df_rel_hz,
        "df_rel_hz_z_1_30": df_rel_hz_z_1_30,
        "df_rel_hz_z_1_50": df_rel_hz_z_1_50,
        "df_peak": df_peak,
        "df_peak_z": df_peak_z,
        "df_coh": df_coh,
        "df_coh_z": df_coh_z,
        "df_pli": df_pli,
        "df_pli_z": df_pli_z,
    }

    if output_one_excel:
        out_xlsx = f"{out_prefix}_ALL_FEATURES.xlsx"
        save_all_to_one_excel(results, out_xlsx)
        print(f"Saved: {out_xlsx}")

    return results

