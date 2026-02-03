#!/usr/bin/env python3
"""
HRF sampling viewer (GUI) with:
  1) Run selector in interactive mode
  2) Design-file selector (dropdown) over all CSVs in the same directory as the provided --csv
  3) Phase dispersion averaging modes:
       - This run
       - Avg across runs (within current CSV)
       - Avg across participants (across CSV files in directory)
         * can average across runs (per participant) OR use the selected run_id across participants

CSV format (unchanged):
  - run_id
  - img_onset   (seconds from run start)

Notes on "participants":
  - By default, each CSV file is treated as one participant/design.
  - If your directory contains multiple files per participant, you can optionally group them by a
    filename prefix via --participant_prefix_sep (default: none; i.e., no grouping).

Usage:
  python hrf_sampling_viewer_gui.py --csv path/to/design.csv --TR 1.792 --window 32 --microtime 16
"""

from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

# Tkinter GUI + Matplotlib embedding
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# -------------------------
# Canonical HRF (SPM-like)
# -------------------------
def spm_canonical_hrf(dt: float = 0.1,
                     length: float = 32.0,
                     p: tuple[float, ...] = (6, 16, 1, 1, 6, 0, 32)) -> tuple[np.ndarray, np.ndarray]:
    delay1, delay2, disp1, disp2, ratio, onset, _ = p
    t = np.arange(0, length + dt, dt)

    def gamma_pdf(x, a, b):
        from math import gamma
        x = np.maximum(x, 0)
        return (x ** (a - 1)) * np.exp(-x / b) / (gamma(a) * (b ** a))

    t_shift = t - onset
    hrf = gamma_pdf(t_shift, delay1 / disp1, disp1) - (gamma_pdf(t_shift, delay2 / disp2, disp2) / ratio)
    peak = float(np.max(hrf)) if float(np.max(hrf)) != 0.0 else 1.0
    return t, (hrf / peak)


# -----------------------------------------
# Utilities
# -----------------------------------------
def first_x_tr_times_after_onset(onset_s: float, TR: float, first_x: int, window_s: float) -> np.ndarray:
    if first_x <= 0:
        return np.array([], dtype=float)
    k0 = int(math.ceil(onset_s / TR))
    tr_times = (k0 + np.arange(first_x)) * TR
    return tr_times[tr_times <= (onset_s + window_s + 1e-9)]


def phase_hist(onsets: np.ndarray, TR: float, bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (bin_centers, probability_per_bin) over [0, TR)."""
    phases = np.mod(onsets.astype(float), TR)
    phases = phases[np.isfinite(phases)]
    edges = np.linspace(0.0, TR, bins + 1)
    counts, _ = np.histogram(phases, bins=edges)
    denom = float(np.sum(counts)) if np.sum(counts) > 0 else 1.0
    probs = counts / denom
    centers = (edges[:-1] + edges[1:]) / 2.0
    return centers, probs


def list_csvs_in_dir(csv_path: Path) -> list[Path]:
    d = csv_path.parent
    # include .csv (case-insensitive)
    files = sorted([p for p in d.iterdir() if p.is_file() and p.suffix.lower() == ".csv"])
    return files


def load_design(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for col in ["run_id", "img_onset"]:
        if col not in df.columns:
            raise ValueError(f"{csv_path.name} missing required column: {col}")
    return df


# -----------------------------------------
# GUI
# -----------------------------------------
class HRFSamplingApp:
    def __init__(self, root: tk.Tk, initial_csv: Path, TR: float, window_s: float, microtime: int,
                 first_x_default: int, phase_bins: int):
        self.root = root
        self.TR = float(TR)
        self.window_s = float(window_s)
        self.microtime = int(microtime)
        self.first_x_default = int(first_x_default)
        self.phase_bins = int(phase_bins)

        self.initial_csv = initial_csv.resolve()
        self.csv_files = list_csvs_in_dir(self.initial_csv)
        if not self.csv_files:
            self.csv_files = [self.initial_csv]

        # State
        self.current_csv: Path = self.initial_csv
        self.df: pd.DataFrame = load_design(self.current_csv)
        self.run_ids: list[int] = sorted(self.df["run_id"].unique().tolist())
        self.current_run: int = int(self.run_ids[0]) if self.run_ids else 1

        self.mode_var = tk.StringVar(value="Trial")  # Trial / Phase
        self.phase_avg_var = tk.StringVar(value="this_run")  # this_run / avg_runs / avg_participants
        self.participants_use_avg_runs_var = tk.BooleanVar(value=True)

        self.trial_idx_var = tk.IntVar(value=1)
        self.first_x_var = tk.IntVar(value=self.first_x_default)
        self.phase_bins_var = tk.IntVar(value=self.phase_bins)

        # Build UI
        self._build_controls()
        self._build_plot()

        self._refresh_all()

    def _build_controls(self):
        self.root.title("HRF Sampling Viewer")

        frm = ttk.Frame(self.root, padding=8)
        frm.pack(side=tk.TOP, fill=tk.X)

        # Row 1: file dropdown + run dropdown
        ttk.Label(frm, text="Design file:").grid(row=0, column=0, sticky="w")
        self.file_combo = ttk.Combobox(frm, state="readonly",
                                       values=[p.name for p in self.csv_files],
                                       width=45)
        self.file_combo.grid(row=0, column=1, sticky="w", padx=6)
        self.file_combo.set(self.current_csv.name)
        self.file_combo.bind("<<ComboboxSelected>>", lambda e: self._on_file_change())

        ttk.Label(frm, text="Run:").grid(row=0, column=2, sticky="w", padx=(16, 0))
        self.run_combo = ttk.Combobox(frm, state="readonly",
                                      values=[str(r) for r in self.run_ids],
                                      width=8)
        self.run_combo.grid(row=0, column=3, sticky="w", padx=6)
        self.run_combo.set(str(self.current_run))
        self.run_combo.bind("<<ComboboxSelected>>", lambda e: self._on_run_change())

        # Row 2: view mode toggle
        ttk.Label(frm, text="View:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.view_combo = ttk.Combobox(frm, state="readonly", values=["Trial", "Phase"], width=10)
        self.view_combo.grid(row=1, column=1, sticky="w", padx=6, pady=(8, 0))
        self.view_combo.set("Trial")
        self.view_combo.bind("<<ComboboxSelected>>", lambda e: self._refresh_plot())

        # Row 3: Trial controls
        trial_frm = ttk.Frame(frm)
        trial_frm.grid(row=2, column=0, columnspan=4, sticky="we", pady=(10, 0))

        ttk.Label(trial_frm, text="Trial:").grid(row=0, column=0, sticky="w")
        self.trial_spin = ttk.Spinbox(trial_frm, from_=1, to=9999, width=6, textvariable=self.trial_idx_var,
                                      command=self._refresh_plot)
        self.trial_spin.grid(row=0, column=1, sticky="w", padx=6)

        self.prev_btn = ttk.Button(trial_frm, text="Prev", command=self._prev_trial)
        self.prev_btn.grid(row=0, column=2, padx=6)
        self.next_btn = ttk.Button(trial_frm, text="Next", command=self._next_trial)
        self.next_btn.grid(row=0, column=3, padx=6)

        ttk.Label(trial_frm, text="First X TRs:").grid(row=0, column=4, sticky="w", padx=(16, 0))
        self.firstx_spin = ttk.Spinbox(trial_frm, from_=0, to=200, width=6, textvariable=self.first_x_var,
                                       command=self._refresh_plot)
        self.firstx_spin.grid(row=0, column=5, sticky="w", padx=6)

        # Row 4: Phase averaging controls
        phase_frm = ttk.LabelFrame(frm, text="Phase dispersion options", padding=8)
        phase_frm.grid(row=3, column=0, columnspan=4, sticky="we", pady=(10, 0))

        ttk.Label(phase_frm, text="Average:").grid(row=0, column=0, sticky="w")
        self.phase_avg_combo = ttk.Combobox(
            phase_frm, state="readonly",
            values=[
                "This run",
                "Avg across runs (this file)",
                "Avg across participants (directory)"
            ],
            width=28
        )
        self.phase_avg_combo.grid(row=0, column=1, sticky="w", padx=6)
        self.phase_avg_combo.set("This run")
        self.phase_avg_combo.bind("<<ComboboxSelected>>", lambda e: self._refresh_plot())

        ttk.Label(phase_frm, text="Phase bins:").grid(row=0, column=2, sticky="w", padx=(16, 0))
        self.phase_bins_spin = ttk.Spinbox(phase_frm, from_=5, to=100, width=6, textvariable=self.phase_bins_var,
                                           command=self._refresh_plot)
        self.phase_bins_spin.grid(row=0, column=3, sticky="w", padx=6)

        self.participants_use_avg_runs_chk = ttk.Checkbutton(
            phase_frm,
            text="Participants: average across runs (otherwise use selected run_id)",
            variable=self.participants_use_avg_runs_var,
            command=self._refresh_plot
        )
        self.participants_use_avg_runs_chk.grid(row=1, column=0, columnspan=4, sticky="w", pady=(6, 0))

    def _build_plot(self):
        # Matplotlib figure embedded in Tkinter
        self.fig = plt.Figure(figsize=(9.2, 5.2), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def _refresh_all(self):
        self._update_run_list()
        self._refresh_plot()

    def _on_file_change(self):
        name = self.file_combo.get()
        chosen = next((p for p in self.csv_files if p.name == name), None)
        if chosen is None:
            return
        self.current_csv = chosen
        self.df = load_design(self.current_csv)
        self._update_run_list()
        self._refresh_plot()

    def _update_run_list(self):
        self.run_ids = sorted(self.df["run_id"].unique().tolist())
        if not self.run_ids:
            self.run_ids = [1]
        # keep current run if possible
        if self.current_run not in self.run_ids:
            self.current_run = int(self.run_ids[0])
        self.run_combo["values"] = [str(r) for r in self.run_ids]
        self.run_combo.set(str(self.current_run))

        # update trial spin limits based on current run
        n_trials = self._n_trials_current_run()
        self.trial_spin.config(to=max(1, n_trials))
        if self.trial_idx_var.get() > n_trials:
            self.trial_idx_var.set(max(1, n_trials))

    def _on_run_change(self):
        try:
            self.current_run = int(self.run_combo.get())
        except Exception:
            return
        n_trials = self._n_trials_current_run()
        self.trial_spin.config(to=max(1, n_trials))
        if self.trial_idx_var.get() > n_trials:
            self.trial_idx_var.set(max(1, n_trials))
        self._refresh_plot()

    def _n_trials_current_run(self) -> int:
        dfr = self.df[self.df["run_id"] == self.current_run]
        return int(len(dfr))

    def _prev_trial(self):
        v = int(self.trial_idx_var.get())
        self.trial_idx_var.set(max(1, v - 1))
        self._refresh_plot()

    def _next_trial(self):
        n = self._n_trials_current_run()
        v = int(self.trial_idx_var.get())
        self.trial_idx_var.set(min(n, v + 1))
        self._refresh_plot()

    def _current_view(self) -> str:
        return self.view_combo.get()

    def _phase_avg_mode(self) -> str:
        m = self.phase_avg_combo.get()
        if m.startswith("This run"):
            return "this_run"
        if m.startswith("Avg across runs"):
            return "avg_runs"
        return "avg_participants"

    def _refresh_plot(self):
        view = self._current_view()
        self.ax.clear()

        if view == "Trial":
            self._plot_trial_view()
        else:
            self._plot_phase_view()

        self.fig.tight_layout()
        self.canvas.draw()

    def _plot_trial_view(self):
        TR = self.TR
        window_s = self.window_s
        micro = self.microtime
        dt = TR / float(micro)
        t, hrf = spm_canonical_hrf(dt=dt, length=window_s)

        dfr = self.df[self.df["run_id"] == self.current_run].copy().sort_values("img_onset")
        onsets = dfr["img_onset"].to_numpy(dtype=float)

        n_trials = len(onsets)
        if n_trials == 0:
            self.ax.set_title("No trials found for this run.")
            return

        trial_idx = int(self.trial_idx_var.get())
        trial_idx = max(1, min(n_trials, trial_idx))
        self.trial_idx_var.set(trial_idx)

        first_x = int(self.first_x_var.get())

        onset = float(onsets[trial_idx - 1])
        tr_times = first_x_tr_times_after_onset(onset, TR, first_x, window_s)
        lags = tr_times - onset
        y = np.interp(lags, t, hrf, left=np.nan, right=np.nan)

        self.ax.plot(t, hrf, linewidth=2, label="Canonical HRF (SPM-like)")
        for x in np.arange(0.0, window_s + 1e-9, TR):
            self.ax.axvline(x, linestyle=":", linewidth=1)
        self.ax.scatter(lags, y, s=70, alpha=0.95, label="First-X TR samples")

        self.ax.set_xlabel("Lag after img onset (s)  [TR times - onset]")
        self.ax.set_ylabel("Canonical HRF (a.u.)")
        self.ax.set_xlim(0.0, window_s)
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(loc="upper right")
        self.ax.set_title(
            f"{self.current_csv.name} | run {self.current_run} | trial {trial_idx}/{n_trials} | onset={onset:.3f}s | TR={TR:.3f}s | first_x={first_x}"
        )

    def _plot_phase_view(self):
        TR = self.TR
        bins = int(self.phase_bins_var.get())
        mode = self._phase_avg_mode()

        # helper: get onsets for a run in current df
        def onsets_for(df: pd.DataFrame, run_id: int | None, avg_runs: bool):
            if avg_runs:
                # stack all runs
                return df["img_onset"].to_numpy(dtype=float)
            if run_id is None:
                return df["img_onset"].to_numpy(dtype=float)
            return df[df["run_id"] == run_id]["img_onset"].to_numpy(dtype=float)

        if mode == "this_run":
            dfr = self.df[self.df["run_id"] == self.current_run].copy()
            on = dfr["img_onset"].to_numpy(dtype=float)
            centers, probs = phase_hist(on, TR, bins=bins)
            self.ax.bar(centers, probs, width=TR/bins*0.95, edgecolor="black", alpha=0.85)
            self.ax.set_title(f"Onset phase within TR | {self.current_csv.name} | run {self.current_run} | TR={TR:.3f}s")
        elif mode == "avg_runs":
            on = onsets_for(self.df, None, avg_runs=True)
            centers, probs = phase_hist(on, TR, bins=bins)
            self.ax.bar(centers, probs, width=TR/bins*0.95, edgecolor="black", alpha=0.85)
            self.ax.set_title(f"Onset phase within TR | {self.current_csv.name} | avg across runs | TR={TR:.3f}s")
        else:
            # avg across participants = average probability distributions across CSV files
            use_avg_runs = bool(self.participants_use_avg_runs_var.get())
            per_participant = []
            used = 0
            for f in self.csv_files:
                try:
                    dfp = load_design(f)
                except Exception:
                    continue
                on = onsets_for(dfp, self.current_run, avg_runs=use_avg_runs)
                if len(on) == 0:
                    continue
                centers, probs = phase_hist(on, TR, bins=bins)
                per_participant.append(probs)
                used += 1

            if used == 0:
                self.ax.set_title("No usable CSVs for participant averaging.")
                return

            avg_probs = np.mean(np.vstack(per_participant), axis=0)
            self.ax.bar(centers, avg_probs, width=TR/bins*0.95, edgecolor="black", alpha=0.85)

            if use_avg_runs:
                scope = "avg across runs (per participant)"
            else:
                scope = f"run {self.current_run} (per participant)"
            self.ax.set_title(f"Onset phase within TR | avg across participants | {scope} | TR={TR:.3f}s | n_files={used}")

        self.ax.set_xlabel("Onset phase within TR (s)  [onset % TR]")
        self.ax.set_ylabel("Probability per bin")
        self.ax.set_xlim(0.0, TR)
        self.ax.grid(True, alpha=0.25)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to one design CSV (directory is scanned for other CSVs).")
    ap.add_argument("--TR", type=float, required=True, help="TR in seconds.")
    ap.add_argument("--window", type=float, default=32.0, help="Peri-event window for trial view (s).")
    ap.add_argument("--microtime", type=int, default=16, help="Microtime bins per TR (dt = TR/microtime).")
    ap.add_argument("--first_x", type=int, default=6, help="Default first X TR samples in trial view.")
    ap.add_argument("--phase_bins", type=int, default=20, help="Default bins for phase histogram.")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    root = tk.Tk()
    app = HRFSamplingApp(root, csv_path, TR=args.TR, window_s=args.window, microtime=args.microtime,
                         first_x_default=args.first_x, phase_bins=args.phase_bins)
    root.mainloop()


if __name__ == "__main__":
    main()
