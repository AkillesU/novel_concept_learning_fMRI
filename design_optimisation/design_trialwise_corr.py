#!/usr/bin/env python3
"""
Python port of design_trialwise_corr.m (SPM-style HRF + DCT high-pass) for trial-wise collinearity diagnostics.

Matches the MATLAB pipeline:
  - Requires columns: run_id, trial_id, img_onset, img_dur, dec_onset_est, isi2_dur, fb_dur
  - Uses SPM-like canonical HRF sampled at dt = TR/microtime
  - Builds trial-wise encoding regressors (1 per trial), HRF convolved, sampled at TR
  - Builds decision/feedback nuisance regressors (condition-level, 1 each)
  - High-pass drift removal via SPM-like DCT basis (includes constant)
  - Computes:
      * RAW encoding trial-wise corr (drift removed + zscored)
      * PARTIALLED encoding trial-wise corr (residualise wrt decision+feedback+drift)
      * ALL-EVENTS corr (enc+dec+fb trial-wise; drift removed + zscored)
  - HRF sampling diagnostics over 0..window_s (default 32s)

Outputs a dict 'out' with per-run results and a pandas summary table.

Usage
  python design_trialwise_corr.py --csv design.csv --TR 1.792 --hp_cutoff 128 --microtime 16 --pad_s 32 --dec_dur_s 2.0

Notes on matching SPM:
  - spm_hrf in SPM normalizes the HRF by its sum. We do the same for closer alignment.
  - spm_dctmtx is implemented exactly (orthonormal DCT-II-like basis with constant column).
  - interp1(...,'linear','extrap') is matched with np.interp over the microtime grid; np.interp extrapolates linearly
    only at edges by clamping. To match MATLAB 'extrap' more faithfully, we implement explicit linear extrapolation.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd


# -------------------------
# SPM-like HRF (spm_hrf)
# -------------------------
def _gamma_pdf(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Gamma pdf with shape=a, scale=b (b>0). Uses math.gamma (no SciPy)."""
    from math import gamma
    x = np.maximum(x, 0.0)
    return (x ** (a - 1.0)) * np.exp(-x / b) / (gamma(a) * (b ** a))


def spm_hrf(dt: float,
            p: Tuple[float, ...] = (6, 16, 1, 1, 6, 0, 32)) -> np.ndarray:
    """
    Close port of SPM canonical HRF generator.

    p = (delay1, delay2, dispersion1, dispersion2, ratio, onset, length)
    SPM normalizes HRF by sum(hrf).
    """
    delay1, delay2, disp1, disp2, ratio, onset, length = p
    t = np.arange(0, length + 1e-12, dt)

    t_shift = t - onset
    h1 = _gamma_pdf(t_shift, delay1 / disp1, disp1)
    h2 = _gamma_pdf(t_shift, delay2 / disp2, disp2)
    hrf = h1 - h2 / ratio

    s = float(np.sum(hrf))
    if s == 0.0:
        return hrf
    return hrf / s


# -------------------------
# SPM-like DCT basis (spm_dctmtx)
# -------------------------
def spm_dctmtx(N: int, K: int) -> np.ndarray:
    """
    Orthonormal DCT basis as in SPM's spm_dctmtx(N,K), including constant.

    Column 0: constant = 1/sqrt(N)
    Columns k>=1: sqrt(2/N) * cos(pi*(2n+1)*k/(2N))  for n=0..N-1
    """
    if K < 1:
        K = 1
    n = np.arange(N)[:, None]  # (N,1)
    C = np.zeros((N, K), dtype=float)
    C[:, 0] = 1.0 / math.sqrt(N)
    if K > 1:
        k = np.arange(1, K)[None, :]  # (1,K-1)
        C[:, 1:] = math.sqrt(2.0 / N) * np.cos(math.pi * (2.0 * n + 1.0) * k / (2.0 * N))
    return C


def dct_basis(n_scans: int, TR: float, cutoff: float) -> np.ndarray:
    """Match MATLAB: K = floor(2*(n_scans*TR)/cutoff + 1); includes constant."""
    K = int(math.floor(2.0 * (n_scans * TR) / cutoff + 1.0))
    K = max(K, 1)
    return spm_dctmtx(n_scans, K)


# -------------------------
# Interpolation matching MATLAB interp1(...,'linear','extrap')
# -------------------------
def interp1_linear_extrap(x: np.ndarray, y: np.ndarray, xq: np.ndarray) -> np.ndarray:
    """
    1D linear interpolation with linear extrapolation (MATLAB-like).
    Assumes x is increasing.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    xq = np.asarray(xq, float)

    # In-range interp via np.interp (which clamps outside); we'll patch extrapolation ourselves.
    yq = np.interp(xq, x, y)

    # Left extrapolation
    left = xq < x[0]
    if np.any(left):
        slope = (y[1] - y[0]) / (x[1] - x[0])
        yq[left] = y[0] + slope * (xq[left] - x[0])

    # Right extrapolation
    right = xq > x[-1]
    if np.any(right):
        slope = (y[-1] - y[-2]) / (x[-1] - x[-2])
        yq[right] = y[-1] + slope * (xq[right] - x[-1])

    return yq


# -------------------------
# Design matrix builders
# -------------------------
def build_trialwise(t_micro: np.ndarray, t_scan: np.ndarray,
                    onsets: np.ndarray, durs: np.ndarray, hrf: np.ndarray) -> np.ndarray:
    """
    MATLAB-equivalent build_trialwise():
      - microtime boxcar u at dt resolution
      - convolve with hrf, truncate
      - sample at TR grid with interp1 linear extrap
    """
    onsets = np.asarray(onsets, float)
    durs = np.asarray(durs, float)
    dt = float(t_micro[1] - t_micro[0])
    n_trials = onsets.size
    X = np.zeros((t_scan.size, n_trials), dtype=float)

    for j in range(n_trials):
        u = np.zeros(t_micro.size, dtype=float)
        onset_idx = int(round(onsets[j] / dt))  # 0-based
        dur_idx = max(1, int(round(durs[j] / dt)))
        last_idx = min(u.size - 1, onset_idx + dur_idx - 1)

        if onset_idx < u.size:
            u[onset_idx:last_idx + 1] = 1.0

        x_micro = np.convolve(u, hrf, mode="full")[:u.size]
        X[:, j] = interp1_linear_extrap(t_micro, x_micro, t_scan)

    return X


def build_condition_level(t_micro: np.ndarray, t_scan: np.ndarray,
                          onsets: np.ndarray, durs: np.ndarray, hrf: np.ndarray) -> np.ndarray:
    """One regressor that is the sum of all events of that type."""
    onsets = np.asarray(onsets, float)
    durs = np.asarray(durs, float)
    dt = float(t_micro[1] - t_micro[0])

    u = np.zeros(t_micro.size, dtype=float)
    for j in range(onsets.size):
        onset_idx = int(round(onsets[j] / dt))
        dur_idx = max(1, int(round(durs[j] / dt)))
        last_idx = min(u.size - 1, onset_idx + dur_idx - 1)
        if onset_idx < u.size:
            u[onset_idx:last_idx + 1] += 1.0

    x_micro = np.convolve(u, hrf, mode="full")[:u.size]
    return interp1_linear_extrap(t_micro, x_micro, t_scan)


# -------------------------
# Regression helpers
# -------------------------
def dct_resid(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Remove drift terms: X - C*(C'X)"""
    return X - C @ (C.T @ X)


def partial_out(X: np.ndarray, N: np.ndarray) -> np.ndarray:
    """Residualise X wrt N: X - N*(pinv(N)*X)"""
    pinvN = np.linalg.pinv(N)
    return X - N @ (pinvN @ X)


def zscore_cols(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    mu = np.mean(X, axis=0, keepdims=True)
    sd = np.std(X, axis=0, ddof=0, keepdims=True)
    sd = np.maximum(sd, eps)
    return (X - mu) / sd


def corr_mat(X: np.ndarray) -> np.ndarray:
    """Column-wise Pearson correlation like MATLAB corr()."""
    return np.corrcoef(X, rowvar=False)


def summarize_corr(R: np.ndarray) -> Dict[str, float]:
    off = R[~np.eye(R.shape[0], dtype=bool)]
    abs_off = np.abs(off)
    return {
        "max_abs_r": float(np.max(abs_off)) if abs_off.size else float("nan"),
        "p95_abs_r": float(np.percentile(abs_off, 95)) if abs_off.size else float("nan"),
        "mean_abs_r": float(np.mean(abs_off)) if abs_off.size else float("nan"),
    }


# -------------------------
# HRF sampling diagnostics (same logic)
# -------------------------
def hrf_sampling_diagnostics(TR: float, onsets: np.ndarray, window_s: float = 32.0) -> Dict[str, Any]:
    nmax = int(math.ceil(window_s / TR))
    lags = []
    for o in np.asarray(onsets, float):
        for k in range(nmax + 1):
            t = (math.ceil(o / TR) + k) * TR
            lag = t - o
            if 0.0 <= lag <= window_s:
                lags.append(lag)
    lags = np.asarray(lags, float)

    binw = 0.1
    edges = np.arange(0.0, window_s + 1e-12, binw)
    counts, _ = np.histogram(lags, bins=edges)

    cov = {
        "early_0_4": float(np.mean((lags >= 0) & (lags < 4))) if lags.size else float("nan"),
        "rise_4_8": float(np.mean((lags >= 4) & (lags < 8))) if lags.size else float("nan"),
        "peak_8_16": float(np.mean((lags >= 8) & (lags < 16))) if lags.size else float("nan"),
        "late_16_32": float(np.mean((lags >= 16) & (lags <= 32))) if lags.size else float("nan"),
    }

    return {
        "window_s": float(window_s),
        "binw": float(binw),
        "edges": edges,
        "counts": counts,
        "lags": lags,
        "coverage": cov,
        "unique_lags": np.unique(np.round(lags, 3)) if lags.size else np.array([]),
    }


# -------------------------
# Pipeline
# -------------------------
REQUIRED_COLS = ["run_id", "trial_id", "img_onset", "img_dur", "dec_onset_est", "isi2_dur", "fb_dur"]


def design_trialwise_corr_pipeline(csv_path: str | Path,
                                  TR: float,
                                  hp_cutoff: float,
                                  microtime: int,
                                  pad_s: float,
                                  dec_dur_s: float) -> Dict[str, Any]:
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    for c in REQUIRED_COLS:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    run_ids = sorted(df["run_id"].unique().tolist())

    dt = TR / float(microtime)
    hrf = spm_hrf(dt)

    out: Dict[str, Any] = {
        "csv_path": str(csv_path),
        "TR": float(TR),
        "hp_cutoff": float(hp_cutoff),
        "microtime": int(microtime),
        "pad_s": float(pad_s),
        "dec_dur_s": float(dec_dur_s),
        "runs": [],
    }

    summary_rows = []

    for rid in run_ids:
        G = df[df["run_id"] == rid].copy()

        enc_on = G["img_onset"].to_numpy(float)
        enc_dur = G["img_dur"].to_numpy(float)

        dec_on = G["dec_onset_est"].to_numpy(float)
        dec_dur = np.full_like(dec_on, float(dec_dur_s), dtype=float)

        fb_on = G["dec_onset_est"].to_numpy(float) + float(dec_dur_s) + G["isi2_dur"].to_numpy(float)
        fb_dur = G["fb_dur"].to_numpy(float)

        run_end = float(np.max(fb_on + fb_dur)) + float(pad_s)
        n_scans = int(math.ceil(run_end / TR))

        t_scan = np.arange(n_scans, dtype=float) * TR
        t_micro = np.arange(int(math.ceil(run_end / dt)), dtype=float) * dt

        X_enc = build_trialwise(t_micro, t_scan, enc_on, enc_dur, hrf)
        x_dec = build_condition_level(t_micro, t_scan, dec_on, dec_dur, hrf)
        x_fb = build_condition_level(t_micro, t_scan, fb_on, fb_dur, hrf)

        C_dct = dct_basis(n_scans, TR, hp_cutoff)

        # RAW encoding corr (drift removed + zscore)
        X_enc_z = zscore_cols(dct_resid(X_enc, C_dct))
        R_raw = corr_mat(X_enc_z)
        stats_raw = summarize_corr(R_raw)

        # PARTIALLED encoding corr (regress out decision+feedback+drift)
        N = np.column_stack([x_dec, x_fb, C_dct])
        X_part = partial_out(X_enc, N)
        X_part_z = zscore_cols(X_part)
        R_part = corr_mat(X_part_z)
        stats_part = summarize_corr(R_part)

        # ALL-EVENTS corr (enc+dec+fb trial-wise, drift removed)
        X_dec_tw = build_trialwise(t_micro, t_scan, dec_on, dec_dur, hrf)
        X_fb_tw = build_trialwise(t_micro, t_scan, fb_on, fb_dur, hrf)
        X_all = np.column_stack([X_enc, X_dec_tw, X_fb_tw])
        X_all_z = zscore_cols(dct_resid(X_all, C_dct))
        R_all = corr_mat(X_all_z)
        stats_all = summarize_corr(R_all)

        samp = hrf_sampling_diagnostics(TR, enc_on, 32.0)

        runout = {
            "run_id": rid,
            "n_scans": n_scans,
            "n_trials": int(len(G)),
            "R_raw_encoding": R_raw,
            "R_part_encoding": R_part,
            "R_all_events": R_all,
            "stats_raw_encoding": stats_raw,
            "stats_part_encoding": stats_part,
            "stats_all_events": stats_all,
            "nuisance": {"x_dec": x_dec, "x_fb": x_fb},
            "hrf_sampling": samp,
        }
        out["runs"].append(runout)

        summary_rows.append({
            "run_id": rid,
            "n_scans": n_scans,
            "n_trials": int(len(G)),
            "enc_raw_max": stats_raw["max_abs_r"],
            "enc_raw_p95": stats_raw["p95_abs_r"],
            "enc_raw_mean": stats_raw["mean_abs_r"],
            "enc_part_max": stats_part["max_abs_r"],
            "enc_part_p95": stats_part["p95_abs_r"],
            "enc_part_mean": stats_part["mean_abs_r"],
            "all_max": stats_all["max_abs_r"],
            "all_p95": stats_all["p95_abs_r"],
            "all_mean": stats_all["mean_abs_r"],
        })

    out["summary_table"] = pd.DataFrame(summary_rows)
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--TR", type=float, required=True)
    ap.add_argument("--hp_cutoff", type=float, default=128.0)
    ap.add_argument("--microtime", type=int, default=16)
    ap.add_argument("--pad_s", type=float, default=32.0)
    ap.add_argument("--dec_dur_s", type=float, default=2.0)
    ap.add_argument("--save_summary_csv", type=str, default=None, help="Optional path to save the summary table as CSV.")
    args = ap.parse_args()

    out = design_trialwise_corr_pipeline(args.csv, args.TR, args.hp_cutoff, args.microtime, args.pad_s, args.dec_dur_s)
    print(out["summary_table"].to_string(index=False))

    if args.save_summary_csv:
        Path(args.save_summary_csv).parent.mkdir(parents=True, exist_ok=True)
        out["summary_table"].to_csv(args.save_summary_csv, index=False)


if __name__ == "__main__":
    main()
