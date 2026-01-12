#!/usr/bin/env python3
"""
create_runs.py

GUI tool to generate event-related timing CSVs.
Supports:
 1. Standard (2-Event) & Self-Paced (3-Event) Designs.
 2. Advanced Jitter (Uniform, Normal, Exponential).
 3. Study 3 Specifics:
    - Counterbalances Groups (Latin Square rotation based on Sub ID).
    - Counterbalances Conditions (Cong/Med/Inc) to Groups.
    - No 1-back Group repetition.
"""

import csv
import os
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Optional
import dataclasses

# ==================== JITTER UTILITIES ==================== #

def sample_jitter(rng, mode, jmin, jmax, norm_mu=None, norm_sd=None, exp_scale=None, exp_reverse=False):
    """
    Full Jitter logic restored from original script.
    """
    if jmin == jmax:
        return float(jmin)
    span = jmax - jmin
    if span <= 0:
        return float(jmin)

    if mode == "uniform":
        return float(rng.uniform(jmin, jmax))

    if mode == "normal":
        mu = float(norm_mu) if norm_mu is not None else 0.5 * (jmin + jmax)
        mu = float(np.clip(mu, jmin, jmax))
        sigma = float(norm_sd) if (norm_sd is not None and norm_sd > 0) else span / 4.0
        # Rejection sampling to stay within bounds
        for _ in range(1000):
            val = rng.normal(mu, sigma)
            if jmin <= val <= jmax:
                return float(val)
        return float(np.clip(mu, jmin, jmax))

    if mode == "exponential":
        scale = float(exp_scale) if (exp_scale is not None and exp_scale > 0) else span / 3.0
        if not exp_reverse:
            val = jmin + rng.exponential(scale=scale)
        else:
            val = jmax - rng.exponential(scale=scale)
        return float(np.clip(val, jmin, jmax))

    return float(jmin)

# ==================== DATA CONFIGS ==================== #

@dataclass
class JitterConfig:
    mode: str
    norm_mu: Optional[float]
    norm_sd: Optional[float]
    exp_scale: Optional[float]
    exp_reverse: bool

@dataclass
class DesignConfig:
    design_type: str  # "2event" or "3event"
    n_runs: int
    trials_per_run: int
    start_time: float
    n_categories_to_select: int
    label_csv_path: str
    stim_dir: str = "blender_output_renamed"
    # Test run configuration
    include_test_run: bool = False
    n_test_runs: int = 1
    test_mode: str = "all_corners"  # all_corners | furthest_corner | opposite_gaussian
    test_repeats_per_space: int = 3
    test_samples_per_space: int = 3
    
    # Timing (2-Event)
    dec_dur: float = 0.0
    dec_fb_dur: float = 0.0
    jit_dec_fb_range: Tuple[float, float] = (0.0, 0.0)
    
    # Timing (3-Event)
    img_dur: float = 0.0
    max_dec_dur: float = 0.0
    fb_dur: float = 0.0
    jit_isi1_range: Tuple[float, float] = (0.0, 0.0)
    jit_isi2_range: Tuple[float, float] = (0.0, 0.0)
    
    # Common
    jit_iti_range: Tuple[float, float] = (0.0, 0.0)
    jitter_config: JitterConfig = None

# ==================== STUDY 3 LOGIC ==================== #
def load_and_parse_groups(csv_path):
    # Detect separator and load directly using 'ObjectSpace'
    df = pd.read_csv(csv_path, sep=None, dtype={'ObjectSpace': str}, engine="python")
    df.columns = [c.strip() for c in df.columns]
    return df


def find_closest_image(stim_dir, space_id, target_coords):
    """
    Finds the image file in the object-space folder closest to sampled coordinates.
    Expects structure: stim_dir / {space_id} / {space_id}_{F0:.3f}_{F1:.3f}.ext
    """
    from pathlib import Path
    import numpy as np
    
    space_path = Path(stim_dir) / str(space_id)
    if not space_path.exists():
        return "MISSING"
    
    best_img, min_dist = None, float('inf')
    for img_file in space_path.glob("*"):
        if img_file.suffix.lower() not in ['.png', '.jpg', '.jpeg']:
            continue
        try:
            # Parse format: {gid}_{F0}_{F1}.ext
            parts = img_file.stem.split('_')
            # Extract F0 and F1 from parts index 1 and 2
            feat_coords = np.array([float(p) for p in parts[1:3]])
            dist = np.linalg.norm(feat_coords - target_coords)
            if dist < min_dist:
                min_dist, best_img = dist, img_file.name
        except:
            continue
    return best_img if best_img else "MISSING"


def generate_design_rows(rng, cfg: DesignConfig, sub_id: int, space_cond_map_override: Optional[dict] = None) -> List[dict]:
    """
    Ensures ObjectSpace trials are balanced across the entire session 
    AND as balanced as possible within each individual run.
    """
    df_labels = load_and_parse_groups(cfg.label_csv_path)
    all_spaces = df_labels['ObjectSpace'].tolist()
    space_cond_map = space_cond_map_override if space_cond_map_override is not None else get_counterbalanced_map(all_spaces, sub_id, cfg.n_categories_to_select)
    active_spaces = list(space_cond_map.keys())
    dist_params = df_labels.set_index('ObjectSpace').to_dict('index')

    # --- SESSION BALANCING LOGIC ---
    total_session_trials = cfg.n_runs * cfg.trials_per_run
    n_cats = len(active_spaces)
    
    # 1. Determine guaranteed session-wide totals
    session_extras = rng.permutation(active_spaces)[:total_session_trials % n_cats]
    
    # 2. Pre-allocate items to runs to ensure run-level balance
    run_pools = [[] for _ in range(cfg.n_runs)]
    base_per_run = cfg.trials_per_run // n_cats
    for r_idx in range(cfg.n_runs):
        run_pools[r_idx].extend(active_spaces * base_per_run)
    
    # 3. Distribute leftovers from the session pool into the run pools
    session_leftovers = {s: (total_session_trials // n_cats) - (base_per_run * cfg.n_runs) + (1 if s in session_extras else 0) for s in active_spaces}
    leftover_items = []
    for s, count in session_leftovers.items():
        leftover_items.extend([s] * count)
    rng.shuffle(leftover_items)
    
    rem_per_run = cfg.trials_per_run % n_cats
    for r_idx in range(cfg.n_runs):
        run_pools[r_idx].extend(leftover_items[r_idx * rem_per_run : (r_idx + 1) * rem_per_run])
        rng.shuffle(run_pools[r_idx])
    
    # 4. Flatten and fix 1-back repeats while preserving the balanced run counts
    session_seq = []
    for p in run_pools: session_seq.extend(p)
    
    for _ in range(1000):
        conflict = False
        for i in range(1, len(session_seq)):
            if session_seq[i] == session_seq[i-1]:
                # Prioritize a swap within the same run to maintain run-level counts
                run_id = i // cfg.trials_per_run
                r_start, r_end = run_id * cfg.trials_per_run, (run_id + 1) * cfg.trials_per_run
                swap_idx = [j for j in range(r_start, r_end) if session_seq[j] != session_seq[i-1] 
                            and (j==0 or session_seq[j-1]!=session_seq[i]) 
                            and (j<len(session_seq)-1 and session_seq[j+1]!=session_seq[i])]
                if not swap_idx: # Session-wide fallback
                    swap_idx = [j for j in range(len(session_seq)) if session_seq[j] != session_seq[i-1]]
                if swap_idx:
                    k = rng.choice(swap_idx)
                    session_seq[i], session_seq[k] = session_seq[k], session_seq[i]
                else: conflict = True
        if not conflict: break

    # --- TRIAL PROCESSING ---
    rows = []
    trial_id_global = 0
    def get_jit(r):
        return sample_jitter(rng, cfg.jitter_config.mode, r[0], r[1],
                             cfg.jitter_config.norm_mu, cfg.jitter_config.norm_sd,
                             cfg.jitter_config.exp_scale, cfg.jitter_config.exp_reverse)

    for run_id in range(1, cfg.n_runs + 1):
        current_time = float(cfg.start_time)
        run_start = (run_id - 1) * cfg.trials_per_run
        run_end = run_id * cfg.trials_per_run
        space_seq = session_seq[run_start:run_end]
        
        for space_id in space_seq:
            cond = space_cond_map[space_id]
            info = dist_params[space_id]
            
            f0 = np.clip(rng.normal(info['Mean F0'], info['SD F0']), 0, 1)
            f1 = np.clip(rng.normal(info['Mean F1'], info['SD F1']), 0, 1)
            img_file = find_closest_image(cfg.stim_dir, space_id, np.array([f0, f1]))
            iti = get_jit(cfg.jit_iti_range)
            
            common = {
                "trial_id": trial_id_global, "run_id": run_id, "ObjectSpace": space_id, 
                "condition": cond, "image_file": img_file, "iti": iti,
                "sampled_f0": round(f0, 3), "sampled_f1": round(f1, 3)
            }

            if cfg.design_type == "3event":
                isi1, isi2 = get_jit(cfg.jit_isi1_range), get_jit(cfg.jit_isi2_range)
                t_img, t_dec = current_time, current_time + cfg.img_dur + isi1
                t_fb, t_end  = t_dec + cfg.max_dec_dur + isi2, t_dec + cfg.max_dec_dur + isi2 + cfg.fb_dur
                rows.append({
                    **common, "trial_duration_max": t_end - t_img, "img_onset": t_img,
                    "img_dur": cfg.img_dur, "isi1_dur": isi1, "isi1_type": "hidden",
                    "dec_onset_est": t_dec, "max_dec_dur": cfg.max_dec_dur,
                    "isi2_dur": isi2, "isi2_type": "fixation", "fb_dur": cfg.fb_dur
                })
                current_time = t_end + iti
            else:
                jit = get_jit(cfg.jit_dec_fb_range)
                t_trial, t_fb = current_time, current_time + cfg.dec_dur + jit
                t_end = t_fb + cfg.dec_fb_dur
                rows.append({
                    **common, "trial_onset": t_trial, "dec_onset": t_trial, "dec_dur": cfg.dec_dur,
                    "dec_fb_onset": t_fb, "dec_fb_dur": cfg.dec_fb_dur, "trial_duration": t_end - t_trial,
                    "dec_fb_jit": jit
                })
                current_time = t_end + iti
            trial_id_global += 1
    return rows


# 2. Restore the Latin Square (Order of selection changes per participant)
def get_counterbalanced_map(all_spaces, participant_id, n_select):
    """Counterbalances ObjectSpaces using Latin Square rotation."""
    conditions = ['Congruent', 'Medium', 'Incongruent']
    n_total = len(all_spaces)
    n_select = min(n_select, n_total)
        
    start_idx = (participant_id - 1) % n_total
    rotated = all_spaces[start_idx:] + all_spaces[:start_idx]
    selected = rotated[:n_select]
    
    space_cond_map = {}
    for i, space in enumerate(selected):
        cond_idx = (participant_id - 1 + i) % len(conditions)
        space_cond_map[space] = conditions[cond_idx]
    return space_cond_map




# ==================== COUNTERBALANCED RANDOMISATION (OPTIMISED) ==================== #

def select_spaces_rotated(all_spaces: List[str], participant_id: int, n_select: int) -> List[str]:
    """Deterministic rotation-based selection of ObjectSpaces (kept from original logic)."""
    n_total = len(all_spaces)
    if n_total == 0:
        return []
    n_select = int(min(max(n_select, 0), n_total))
    start_idx = (participant_id - 1) % n_total
    rotated = all_spaces[start_idx:] + all_spaces[:start_idx]
    return rotated[:n_select]


def make_balanced_condition_list(rng, n_items: int, conditions: List[str]) -> List[str]:
    """Return a list of conditions with counts as equal as possible (e.g., 2,2,2 or 3,3,2)."""
    k = len(conditions)
    if k == 0 or n_items <= 0:
        return []
    q, r = divmod(n_items, k)
    # Randomly choose which conditions get the +1 when n_items not divisible by k
    conds = list(conditions)
    rng.shuffle(conds)
    counts = {c: q for c in conditions}
    for c in conds[:r]:
        counts[c] += 1
    out = []
    for c in conditions:
        out.extend([c] * counts[c])
    rng.shuffle(out)
    return out


def optimise_space_condition_map(
    rng,
    selected_spaces: List[str],
    conditions: List[str],
    global_sc_counts: dict,
    global_pair_counts: dict,
    n_candidates: int = 200,
    lambda_pair: float = 1.0,
) -> dict:
    """
    Constrained randomisation:
      - Hard constraint: within-participant condition counts are as equal as possible.
      - Objective: diversify (space,condition) usage and avoid deterministic coupling between spaces' assignments.

    Scoring (lower is better):
      sc_term   = sum over spaces of current count for assigning that space to that condition
      pair_term = sum over (space_i, space_j) pairs of current count for that ordered condition-pair
      score = sc_term + lambda_pair * pair_term
    """
    if not selected_spaces:
        return {}

    # Ensure globals contain needed keys
    for s in selected_spaces:
        if s not in global_sc_counts:
            global_sc_counts[s] = {c: 0 for c in conditions}
        else:
            for c in conditions:
                global_sc_counts[s].setdefault(c, 0)

    def pair_key_and_condpair(sa, sb, ca, cb):
        # store pair key with sorted spaces; condpair aligned to that order
        if sa <= sb:
            return (sa, sb), (ca, cb)
        else:
            return (sb, sa), (cb, ca)

    best_map = None
    best_score = None

    n_candidates = int(max(1, n_candidates))
    # Candidate generation
    for _ in range(n_candidates):
        cond_list = make_balanced_condition_list(rng, len(selected_spaces), conditions)
        cand_map = {s: c for s, c in zip(selected_spaces, cond_list)}

        sc_term = 0
        for s, c in cand_map.items():
            sc_term += global_sc_counts[s][c]

        pair_term = 0
        if lambda_pair and len(selected_spaces) > 1:
            # Penalise repeating the same joint assignments between co-occurring spaces
            for i in range(len(selected_spaces)):
                for j in range(i + 1, len(selected_spaces)):
                    sa, sb = selected_spaces[i], selected_spaces[j]
                    ca, cb = cand_map[sa], cand_map[sb]
                    pk, cp = pair_key_and_condpair(sa, sb, ca, cb)
                    pair_term += global_pair_counts.get(pk, {}).get(cp, 0)

        score = sc_term + float(lambda_pair) * pair_term
        if best_score is None or score < best_score:
            best_score = score
            best_map = cand_map

    if best_map is None:
        # Fallback: greedy least-used per space (still balanced-ish)
        cond_list = make_balanced_condition_list(rng, len(selected_spaces), conditions)
        best_map = {s: c for s, c in zip(selected_spaces, cond_list)}

    # Update global trackers
    for s, c in best_map.items():
        global_sc_counts[s][c] = global_sc_counts[s].get(c, 0) + 1

    if len(selected_spaces) > 1:
        for i in range(len(selected_spaces)):
            for j in range(i + 1, len(selected_spaces)):
                sa, sb = selected_spaces[i], selected_spaces[j]
                ca, cb = best_map[sa], best_map[sb]
                pk, cp = pair_key_and_condpair(sa, sb, ca, cb)
                if pk not in global_pair_counts:
                    global_pair_counts[pk] = {}
                global_pair_counts[pk][cp] = global_pair_counts[pk].get(cp, 0) + 1

    return best_map

# 3. Add safety checks in the 'run' method to prevent IndexError
def parse_range(val_str, field_name):
    """Safely split comma-separated strings to avoid IndexError."""
    try:
        parts = [float(x) for x in val_str.split(',')]
        if len(parts) < 2:
            raise ValueError(f"{field_name} needs a Min and Max separated by a comma.")
        return parts[0], parts[1]
    except Exception:
        raise ValueError(f"Invalid format for {field_name}. Use 'Min,Max' (e.g., 0.5,1.5)")


def generate_sequence_no_repeat(rng, items, length):
    """Generates sequence ensuring no 1-back repeats."""
    if not items: return []
    if len(items) == 1: return items * length
    
    # Initial balanced pool
    base = length // len(items)
    rem = length % len(items)
    pool = items * base + list(rng.choice(items, size=rem, replace=False))
    rng.shuffle(pool)
    
    # Swapping heuristic to fix repeats
    for _ in range(500):
        conflict = False
        for i in range(1, len(pool)):
            if pool[i] == pool[i-1]:
                # Find swap candidate
                swap_idx = [j for j in range(len(pool)) if pool[j] != pool[i-1] and (j==0 or pool[j-1]!=pool[i]) and (j<len(pool)-1 and pool[j+1]!=pool[i])]
                if not swap_idx: 
                    # fallback: any non-match
                    swap_idx = [j for j in range(len(pool)) if pool[j] != pool[i-1]]
                
                if swap_idx:
                    k = rng.choice(swap_idx)
                    pool[i], pool[k] = pool[k], pool[i]
                else:
                    conflict = True # Could not fix
        if not conflict: break
    
    return pool


# ==================== GUI ==================== #

class DesignGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Study 3 Design Generator")
        self.geometry("1100x800")
        self._build_ui()
        self.update_estimates()

    def _init_vars(self):
        """Initialize all variables with traces for live updates."""
        # Global
        self.v_runs = tk.StringVar(value="2")
        self.v_trials = tk.StringVar(value="30")
        self.v_runs_s2 = tk.StringVar(value="2")
        self.v_trials_s2 = tk.StringVar(value="20")
        
        # 2-Event Timing
        self.v_2_dec = tk.StringVar(value="2.0")
        self.v_2_fb = tk.StringVar(value="1.0")
        self.v_2_jit = tk.StringVar(value="0.5,1.5")
        self.v_2_iti = tk.StringVar(value="1.0,3.0")
        
        # 3-Event Timing
        self.v_3_img = tk.StringVar(value="1.0")
        self.v_3_max = tk.StringVar(value="3.0")
        self.v_3_fb = tk.StringVar(value="1.0")
        self.v_3_isi1 = tk.StringVar(value="0.5,1.5")
        self.v_3_isi2 = tk.StringVar(value="0.5,1.0")
        self.v_3_iti = tk.StringVar(value="1.0,3.0")

        # Add traces to all timing-relevant variables
        vars_to_trace = [
            self.v_runs, self.v_trials, self.v_runs_s2, self.v_trials_s2,
            self.v_2_dec, self.v_2_fb, self.v_2_jit, self.v_2_iti,
            self.v_3_img, self.v_3_max, self.v_3_fb, self.v_3_isi1, self.v_3_isi2, self.v_3_iti
        ]
        for v in vars_to_trace:
            v.trace_add("write", lambda *a: self.update_estimates())

    def get_float(self, var, default=0.0):
        try:
            val = var.get().replace(',', '.') # Handle comma decimal separators
            return float(val)
        except: return default

    def get_int(self, var, default):
        try: return int(var.get())
        except: return default

    def _build_ui(self):
        main = ttk.Frame(self, padding=10)
        main.pack(fill="both", expand=True)

        # 1. File Inputs
        fr_file = ttk.LabelFrame(main, text="Files & Output")
        fr_file.pack(fill="x", pady=5)
        ttk.Label(fr_file, text="Label CSV:").pack(side="left")
        self.v_csv = tk.StringVar()
        ttk.Entry(fr_file, textvariable=self.v_csv, width=30).pack(side="left", padx=5)
        ttk.Button(fr_file, text="Browse", command=self.browse_csv).pack(side="left")
        ttk.Label(fr_file, text="Output Prefix:").pack(side="left", padx=10)
        self.v_prefix = tk.StringVar(value="study3")
        ttk.Entry(fr_file, textvariable=self.v_prefix, width=10).pack(side="left")

        # 2. Global Settings
        fr_glob = ttk.LabelFrame(main, text="Global Settings")
        fr_glob.pack(fill="x", pady=5)

        # We add trace to these variables to update timing automatically
        self.v_runs = tk.StringVar(value="2"); self.v_runs.trace_add("write", lambda *a: self.update_estimates())
        self.v_trials = tk.StringVar(value="30"); self.v_trials.trace_add("write", lambda *a: self.update_estimates())

        ttk.Label(fr_glob, text="Seed:").pack(side="left")
        self.v_seed = tk.StringVar(value="999")
        ttk.Entry(fr_glob, textvariable=self.v_seed, width=6).pack(side="left", padx=2)
        
        ttk.Label(fr_glob, text="Start Sub ID:").pack(side="left")
        self.v_start_id = tk.StringVar(value="1")
        ttk.Entry(fr_glob, textvariable=self.v_start_id, width=4).pack(side="left", padx=2)
        
        ttk.Label(fr_glob, text="# Participants:").pack(side="left")
        self.v_n_subs = tk.StringVar(value="5")
        ttk.Entry(fr_glob, textvariable=self.v_n_subs, width=4).pack(side="left", padx=2)
      

        # Session 1 and Session 2 specific parameters
        sep = ttk.Separator(fr_glob, orient="vertical")
        sep.pack(side="left", fill="y", padx=5)
        
        ttk.Label(fr_glob, text="S1 Runs:").pack(side="left")
        self.v_runs = tk.StringVar(value="2")
        ttk.Entry(fr_glob, textvariable=self.v_runs, width=3).pack(side="left", padx=2)
        ttk.Label(fr_glob, text="S1 Trials/Run:").pack(side="left")
        self.v_trials = tk.StringVar(value="30")
        ttk.Entry(fr_glob, textvariable=self.v_trials, width=4).pack(side="left", padx=2)
        
        ttk.Label(fr_glob, text="S2 Runs:").pack(side="left")
        self.v_runs_s2 = tk.StringVar(value="2")
        ttk.Entry(fr_glob, textvariable=self.v_runs_s2, width=3).pack(side="left", padx=2)
        ttk.Label(fr_glob, text="S2 Trials/Run:").pack(side="left")
        self.v_trials_s2 = tk.StringVar(value="20")
        ttk.Entry(fr_glob, textvariable=self.v_trials_s2, width=4).pack(side="left", padx=2)
        ttk.Label(fr_glob, text="N Categories to Select:").pack(side="left")
        self.v_n_cats = tk.StringVar(value="6")
        ttk.Entry(fr_glob, textvariable=self.v_n_cats, width=5).pack(side="left", padx=5)

        # Test run settings
        ttk.Label(fr_glob, text=" | Test Run:").pack(side="left", padx=8)
        self.v_include_test = tk.BooleanVar(value=False)
        ttk.Checkbutton(fr_glob, text="Include", variable=self.v_include_test).pack(side="left")
        ttk.Label(fr_glob, text="# Test Runs:").pack(side="left")
        self.v_n_test_runs = tk.StringVar(value="1")
        ttk.Entry(fr_glob, textvariable=self.v_n_test_runs, width=4).pack(side="left", padx=5)

        ttk.Label(fr_glob, text="Mode:").pack(side="left")
        self.v_test_mode = tk.StringVar(value="all_corners")
        ttk.OptionMenu(fr_glob, self.v_test_mode, "all_corners", "all_corners", "furthest_corner", "opposite_gaussian").pack(side="left")
        ttk.Label(fr_glob, text="Repeats:").pack(side="left")
        self.v_test_repeats = tk.StringVar(value="3")
        ttk.Entry(fr_glob, textvariable=self.v_test_repeats, width=4).pack(side="left", padx=2)
        ttk.Label(fr_glob, text="Samples:").pack(side="left")
        self.v_test_samples = tk.StringVar(value="3")
        ttk.Entry(fr_glob, textvariable=self.v_test_samples, width=4).pack(side="left", padx=2)

        # 3. Timing Tabs
        self.nb = ttk.Notebook(main)
        self.nb.pack(fill="x", pady=10)
        self.nb.bind("<<NotebookTabChanged>>", lambda e: self.update_estimates())
        
        self.tab2 = ttk.Frame(self.nb)
        self.nb.add(self.tab2, text="Standard (2-Event)")
        self._build_2event_tab(self.tab2)
        
        self.tab3 = ttk.Frame(self.nb)
        self.nb.add(self.tab3, text="Self-Paced (3-Event)")
        self._build_3event_tab(self.tab3)

        # 3.5 ESTIMATES PANEL (New Section)
        fr_est = ttk.LabelFrame(main, text="Estimated Timings")
        fr_est.pack(fill="x", pady=10)
        
        # Trial Row
        ttk.Label(fr_est, text="Trial Length (Mean):", font=("TkDefaultFont", 10, "bold")).grid(row=1, column=0, sticky="e", padx=5)
        self.lbl_trial_mean = ttk.Label(fr_est, text="--")
        self.lbl_trial_mean.grid(row=1, column=1)
        self.lbl_trial_mean_nojit = ttk.Label(fr_est, text="--")
        self.lbl_trial_mean_nojit.grid(row=1, column=2)

        # Min/Max Row
        ttk.Label(fr_est, text="Trial Range (Min-Max):").grid(row=2, column=0, sticky="e", padx=5)
        self.lbl_trial_range = ttk.Label(fr_est, text="--")
        self.lbl_trial_range.grid(row=2, column=1)
        self.lbl_trial_range_nojit = ttk.Label(fr_est, text="--")
        self.lbl_trial_range_nojit.grid(row=2, column=2)

        # Total Session 1 Row
        ttk.Label(fr_est, text="Total Session 1:", font=("TkDefaultFont", 10, "bold")).grid(row=3, column=0, sticky="e", padx=5)
        self.lbl_s1_total = ttk.Label(fr_est, text="--", foreground="blue")
        self.lbl_s1_total.grid(row=3, column=1)
        self.lbl_s1_total_nojit = ttk.Label(fr_est, text="--", foreground="green")
        self.lbl_s1_total_nojit.grid(row=3, column=2)

        # Total Session 2 Row
        ttk.Label(fr_est, text="Total Session 2:", font=("TkDefaultFont", 10, "bold")).grid(row=4, column=0, sticky="e", padx=5)
        self.lbl_s2_total = ttk.Label(fr_est, text="--", foreground="blue")
        self.lbl_s2_total.grid(row=4, column=1)
        self.lbl_s2_total_nojit = ttk.Label(fr_est, text="--", foreground="green")
        self.lbl_s2_total_nojit.grid(row=4, column=2)
        # 4. Jitter Settings (Restored)
        fr_jit = ttk.LabelFrame(main, text="Jitter Distribution Config")
        fr_jit.pack(fill="x", pady=5)
        
        ttk.Label(fr_jit, text="Mode:").pack(side="left")
        self.v_jmode = tk.StringVar(value="uniform")
        ttk.OptionMenu(fr_jit, self.v_jmode, "uniform", "uniform", "normal", "exponential").pack(side="left")
        
        ttk.Label(fr_jit, text="Norm Mu/SD:").pack(side="left", padx=5)
        self.v_nmu = tk.StringVar()
        self.v_nsd = tk.StringVar()
        ttk.Entry(fr_jit, textvariable=self.v_nmu, width=5).pack(side="left")
        ttk.Entry(fr_jit, textvariable=self.v_nsd, width=5).pack(side="left")
        
        ttk.Label(fr_jit, text="Exp Scale/Rev:").pack(side="left", padx=5)
        self.v_esc = tk.StringVar()
        ttk.Entry(fr_jit, textvariable=self.v_esc, width=5).pack(side="left")
        self.v_erev = tk.BooleanVar()
        ttk.Checkbutton(fr_jit, text="Rev", variable=self.v_erev).pack(side="left")

        # Action
        ttk.Button(main, text="Generate Designs", command=self.run).pack(pady=10)
        self.log_box = scrolledtext.ScrolledText(main, height=10)
        self.log_box.pack(fill="both")

    def _build_2event_tab(self, p):
        f = ttk.Frame(p); f.pack(fill="both", padx=10, pady=10)
        ttk.Label(f, text="Dec Dur:").pack(side="left")
        self.v_2_dec = tk.StringVar(value="2.0")
        ttk.Entry(f, textvariable=self.v_2_dec, width=5).pack(side="left")
        ttk.Label(f, text="Fb Dur:").pack(side="left", padx=5)
        self.v_2_fb = tk.StringVar(value="1.0")
        ttk.Entry(f, textvariable=self.v_2_fb, width=5).pack(side="left")
        ttk.Label(f, text="Jitter (Dec->Fb) Min/Max:").pack(side="left", padx=5)
        self.v_2_jit = tk.StringVar(value="0.5,1.5")
        ttk.Entry(f, textvariable=self.v_2_jit, width=8).pack(side="left")
        ttk.Label(f, text="ITI Min/Max:").pack(side="left", padx=5)
        self.v_2_iti = tk.StringVar(value="1.0,3.0")
        ttk.Entry(f, textvariable=self.v_2_iti, width=8).pack(side="left")

    def _build_3event_tab(self, p):
        f = ttk.Frame(p); f.pack(fill="both", padx=10, pady=10)
        # Simplified row layout
        r1 = ttk.Frame(f); r1.pack(fill="x")
        ttk.Label(r1, text="Img Dur:").pack(side="left")
        self.v_3_img = tk.StringVar(value="1.0")
        ttk.Entry(r1, textvariable=self.v_3_img, width=5).pack(side="left")
        ttk.Label(r1, text="Max Dec Dur:").pack(side="left", padx=5)
        self.v_3_max = tk.StringVar(value="3.0")
        ttk.Entry(r1, textvariable=self.v_3_max, width=5).pack(side="left")
        ttk.Label(r1, text="Fb Dur:").pack(side="left", padx=5)
        self.v_3_fb = tk.StringVar(value="1.0")
        ttk.Entry(r1, textvariable=self.v_3_fb, width=5).pack(side="left")
        
        r2 = ttk.Frame(f); r2.pack(fill="x", pady=5)
        ttk.Label(r2, text="ISI 1 (Hidden) Min/Max:").pack(side="left")
        self.v_3_isi1 = tk.StringVar(value="0.5,1.5")
        ttk.Entry(r2, textvariable=self.v_3_isi1, width=8).pack(side="left")
        ttk.Label(r2, text="ISI 2 (Fix) Min/Max:").pack(side="left", padx=5)
        self.v_3_isi2 = tk.StringVar(value="0.5,1.0")
        ttk.Entry(r2, textvariable=self.v_3_isi2, width=8).pack(side="left")
        ttk.Label(r2, text="ITI Min/Max:").pack(side="left", padx=5)
        self.v_3_iti = tk.StringVar(value="1.0,3.0")
        ttk.Entry(r2, textvariable=self.v_3_iti, width=8).pack(side="left")

    def browse_csv(self):
        f = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if f: self.v_csv.set(f)

    def log(self, txt):
        self.log_box.insert("end", txt+"\n"); self.log_box.see("end")

    def _save_session_data(self, rows, sub_id, sess_id, prefix, mode):
            """Helper to save one consolidated CSV for the entire session."""
            sess_dir = os.path.join(prefix, f"session-{sess_id}")
            os.makedirs(sess_dir, exist_ok=True)
        
            # Save all runs together in one file
            fname = f"{prefix}_sub-{sub_id:03d}_sess-{sess_id}_{mode}.csv"
            fpath = os.path.join(sess_dir, fname)
        
            with open(fpath, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=rows[0].keys())
                w.writeheader()
                w.writerows(rows)
        
            self.log(f"Generated Session {sess_id} master file: {fname}")

    def update_estimates(self):
        """Live calculation for trial, total task lengths, and test run durations."""
        try:
            tab = self.nb.index(self.nb.select())
            n_runs1 = self.get_int(self.v_runs, 0)
            n_trials1 = self.get_int(self.v_trials, 0)
            n_runs2 = self.get_int(self.v_runs_s2, 0)
            n_trials2 = self.get_int(self.v_trials_s2, 0)
            
            # Test Run Parameters
            include_test = self.v_include_test.get()
            n_test_runs = self.get_int(self.v_n_test_runs, 0)
            test_repeats = self.get_int(self.v_test_repeats, 3)
            test_samples = self.get_int(self.v_test_samples, 3)
            n_cats = self.get_int(self.v_n_cats, 6)
            
            # Total test trials across all test runs
            total_test_trials = n_cats * test_repeats * test_samples * n_test_runs
            
            # 1. Calculate Base and Jitter components
            if tab == 0: # 2-Event
                dec = self.get_float(self.v_2_dec)
                fb = self.get_float(self.v_2_fb)
                j_min, j_max = parse_range(self.v_2_jit.get(), "Jit")
                i_min, i_max = parse_range(self.v_2_iti.get(), "ITI")
                
                # Full Trial (with jitter)
                t_min, t_max = dec + fb + j_min + i_min, dec + fb + j_max + i_max
                # Without Jitter (ISI 2 equivalent in 2-event)
                tn_min, tn_max = dec + fb + i_min, dec + fb + i_max
                # Test Trial (No Jitter/Feedback)
                test_t_min, test_t_max = dec + i_min, dec + i_max
                
            else: # 3-Event
                img = self.get_float(self.v_3_img)
                dec = self.get_float(self.v_3_max)
                fb = self.get_float(self.v_3_fb)
                is1_min, is1_max = parse_range(self.v_3_isi1.get(), "ISI1")
                is2_min, is2_max = parse_range(self.v_3_isi2.get(), "ISI2")
                iti_min, iti_max = parse_range(self.v_3_iti.get(), "ITI")
                
                # Full Trial (with ISI 2)
                t_min = img + dec + fb + is1_min + is2_min + iti_min
                t_max = img + dec + fb + is1_max + is2_max + iti_max
                # Without ISI 2
                tn_min = img + dec + fb + is1_min + iti_min
                tn_max = img + dec + fb + is1_max + iti_max
                # Test Trial (No ISI 2 or Feedback)
                test_t_min = img + dec + is1_min + iti_min
                test_t_max = img + dec + is1_max + iti_max

            t_mean = (t_min + t_max) / 2
            tn_mean = (tn_min + tn_max) / 2
            test_t_mean = (test_t_min + test_t_max) / 2

            # 2. Update Labels
            self.lbl_trial_mean.config(text=f"{t_mean:.2f}s")
            self.lbl_trial_mean_nojit.config(text=f"(No-ISI2: {tn_mean:.2f}s)")
            self.lbl_trial_range.config(text=f"{t_min:.1f}s - {t_max:.1f}s")
            self.lbl_trial_range_nojit.config(text=f"(No-ISI2: {tn_min:.1f}s - {tn_max:.1f}s)")

            # Total session time without test runs
            s1_full = (n_runs1 * n_trials1 * t_mean) / 60
            s1_no_isi2 = (n_runs1 * n_trials1 * tn_mean) / 60
            s2_full = (n_runs2 * n_trials2 * t_mean) / 60
            s2_no_isi2 = (n_runs2 * n_trials2 * tn_mean) / 60
            
            # Test run duration (total across all specified test runs)
            test_dur = (total_test_trials * test_t_mean) / 60 if include_test else 0.0

            # 3. Final String Formatting
            self.lbl_s1_total.config(text=f"~{s1_full:.2f} min")
            self.lbl_s1_total_nojit.config(text=f"No-ISI2: ~{s1_no_isi2:.2f} min | test length: ~{test_dur:.2f} min")
            
            self.lbl_s2_total.config(text=f"~{s2_full:.2f} min")
            self.lbl_s2_total_nojit.config(text=f"No-ISI2: ~{s2_no_isi2:.2f} min | test length: ~{test_dur:.2f} min")

        except Exception:
            pass
    def run(self):
        if not os.path.exists(self.v_csv.get()):
            messagebox.showerror("Error", "CSV not found")
            return
        try:
            selected_tab = self.nb.select()
            mode = "2event" if selected_tab == str(self.tab2) else "3event"
            
            # Initial config from GUI (Session 1 parameters)
            jconf = JitterConfig(
                mode=self.v_jmode.get(), 
                norm_mu=self.get_float(self.v_nmu, None), 
                norm_sd=self.get_float(self.v_nsd, None),
                exp_scale=self.get_float(self.v_esc, None), 
                exp_reverse=self.v_erev.get()
            )
            
            cfg_s1 = DesignConfig(
                design_type=mode, 
                n_runs=self.get_int(self.v_runs, 2), 
                trials_per_run=self.get_int(self.v_trials, 30),
                start_time=0.0, 
                n_categories_to_select=self.get_int(self.v_n_cats, 10), 
                label_csv_path=self.v_csv.get(), 
                jitter_config=jconf
            )
            
            if mode == "2event":
                cfg_s1.dec_dur, cfg_s1.dec_fb_dur = self.get_float(self.v_2_dec, 2.0), self.get_float(self.v_2_fb, 1.0)
                cfg_s1.jit_dec_fb_range, cfg_s1.jit_iti_range = parse_range(self.v_2_jit.get(), "Jitter"), parse_range(self.v_2_iti.get(), "ITI")
            else:
                cfg_s1.img_dur, cfg_s1.max_dec_dur, cfg_s1.fb_dur = self.get_float(self.v_3_img, 1.0), self.get_float(self.v_3_max, 3.0), self.get_float(self.v_3_fb, 1.0)
                cfg_s1.jit_isi1_range, cfg_s1.jit_isi2_range, cfg_s1.jit_iti_range = parse_range(self.v_3_isi1.get(), "ISI1"), parse_range(self.v_3_isi2.get(), "ISI2"), parse_range(self.v_3_iti.get(), "ITI")

            # Create Session 2 Config by overriding S1
            cfg_s2 = dataclasses.replace(
                cfg_s1, 
                n_runs=self.get_int(self.v_runs_s2, 2), 
                trials_per_run=self.get_int(self.v_trials_s2, 20)
            )

            base_rng = np.random.default_rng(self.get_int(self.v_seed, 999))
            prefix = self.v_prefix.get()
            os.makedirs(prefix, exist_ok=True)
            
            df_master = load_and_parse_groups(self.v_csv.get())
            all_spaces_master = [str(s) for s in df_master['ObjectSpace'].tolist()]
            cond_master = ['Congruent', 'Medium', 'Incongruent']
            global_sc = {s: {c: 0 for c in cond_master} for s in all_spaces_master}
            global_pair = {}

            for i in range(self.get_int(self.v_n_subs, 5)):
                sub_id = self.get_int(self.v_start_id, 1) + i
                sub_rng = np.random.default_rng(base_rng.integers(0, 2**32))
                
                # IMPORTANT: Mapping and selection happen once per participant for retention consistency
                selected = select_spaces_rotated(all_spaces_master, sub_id, cfg_s1.n_categories_to_select)
                sc_map = optimise_space_condition_map(sub_rng, selected, cond_master, global_sc, global_pair)
                
                # Generate Session 1
                rows_s1 = generate_design_rows(sub_rng, cfg_s1, sub_id, sc_map)
                self._save_session_data(rows_s1, sub_id, 1, prefix, mode)
                
                # Generate Session 2 (Retains exact same sc_map)
                rows_s2 = generate_design_rows(sub_rng, cfg_s2, sub_id, sc_map)
                self._save_session_data(rows_s2, sub_id, 2, prefix, mode)
                
                self.log(f"Participant {sub_id} complete (Balanced sessions).")

            self.log("All participants generated."); messagebox.showinfo("Success", "Design files generated for 2 sessions.")
        except Exception as e:
            self.log(f"Error: {e}"); messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    app = DesignGUI()
    app.mainloop()