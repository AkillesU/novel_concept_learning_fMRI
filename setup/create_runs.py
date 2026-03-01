#!/usr/bin/env python3
"""
create_runs.py

Design generator GUI for event-related fMRI experiments.

Primary function:
 - Provide a small GUI to configure and generate event-timing CSVs for two
   supported designs:
     * Standard (2-event): Decision -> Feedback
     * Self-Paced (3-event): Image -> Decision (self-paced) -> Feedback
 - Configure flexible jitter distributions (uniform, truncated normal, exponential)
   for inter-stimulus intervals and other timing jitter.
 - Produce session CSV files per participant with built-in counterbalancing:
     * Latin-square style rotation of ObjectSpaces by participant ID.
     * Assignment of conditions (Congruent/Medium/Incongruent) to ObjectSpaces
       in a balanced manner.
     * Heuristics to avoid immediate 1-back repeats within the session.
 - Save per-session master CSV files into an output prefix folder.

Usage:
    python setup/create_runs.py

Notes / implementation details:
 - Jitter sampling supports rejection sampling for truncated normal values
   and an exponential mode that can be reversed to bias toward the high end.
 - The core balancing routine first ensures session-level counts are satisfied,
   then pre-allocates items to runs to preserve per-run balance, and finally
   applies a swap-based heuristic to eliminate adjacent repeats while keeping
   run counts stable.
 - The GUI estimates mean/min/max trial durations live so the operator can
   check approximate session lengths before generation.
 - No functional behaviour of generation logic is changed by this file's comments;
   comments are for readability and maintenance.
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
# Utility for sampling jitter values using common distributions.
# The function intentionally clamps/rejects values so returned jitter always
# lies within [jmin, jmax].
def sample_jitter(rng, mode, jmin, jmax, norm_mu=None, norm_sd=None, exp_scale=None, exp_reverse=False):
    """
    Sample a jitter value in [jmin, jmax] according to the requested distribution.

    Parameters:
      - rng: numpy random Generator instance
      - mode: 'uniform' | 'normal' | 'exponential' (fallback returns jmin)
      - jmin, jmax: inclusive numeric bounds for the sampled value
      - norm_mu, norm_sd: optional parameters for truncated normal (mu, sigma)
      - exp_scale: scale parameter (lambda) for exponential sampling
      - exp_reverse: if True and mode == 'exponential' bias toward jmax instead of jmin

    Behaviour:
      - Uniform: draws uniformly on [jmin, jmax].
      - Normal: performs truncated normal via rejection sampling (bounded to [jmin,jmax]).
      - Exponential: draws an exponential offset from jmin (or from jmax when exp_reverse).
      - If bounds equal or invalid span, returns jmin as deterministic fallback.
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
        # Rejection sampling to enforce bounds (limited iterations for safety)
        for _ in range(1000):
            val = rng.normal(mu, sigma)
            if jmin <= val <= jmax:
                return float(val)
        # Fallback to the bounded mean when rejection fails
        return float(np.clip(mu, jmin, jmax))

    if mode == "exponential":
        scale = float(exp_scale) if (exp_scale is not None and exp_scale > 0) else span / 3.0
        # Exponential offset from jmin (or reversed from jmax)
        if not exp_reverse:
            val = jmin + rng.exponential(scale=scale)
        else:
            val = jmax - rng.exponential(scale=scale)
        return float(np.clip(val, jmin, jmax))

    # Conservative fallback if an unknown mode was supplied
    return float(jmin)

# ==================== DATA CONFIGS ==================== #
# Small dataclasses that hold configuration values passed around the generator.

@dataclass
class JitterConfig:
    mode: str
    norm_mu: Optional[float]
    norm_sd: Optional[float]
    exp_scale: Optional[float]
    exp_reverse: bool

@dataclass
class DesignConfig:
    """
    High-level container for a design configuration used by the generator.

    Attributes:
      - design_type: '2event' or '3event'
      - n_runs, trials_per_run: run structure for session
      - start_time: initial offset for the first trial
      - n_categories_to_select: how many ObjectSpaces to select for this participant
      - label_csv_path: path to stimulus label CSV (must contain 'ObjectSpace' + feature stats)
      - jitter_config: instance of JitterConfig controlling distribution parameters
      - additional fields store the timing constants for the selected design type
    """
    design_type: str  # "2event" or "3event"
    n_runs: int
    trials_per_run: int
    start_time: float
    n_categories_to_select: int
    label_csv_path: str
    stim_dir: str = "images/task_images/"
    # Test run configuration
    include_test_run: bool = False
    # How many test runs to append (if enabled for this session)
    n_test_runs: int = 1
    # all_corners | furthest_corner | opposite_gaussian
    test_mode: str = "all_corners"

    # Legacy corner sampling params (used for all_corners / furthest_corner)
    test_repeats_per_space: int = 3
    test_samples_per_space: int = 3

    # Opposite-corner Gaussian sampling params (used for opposite_gaussian)
    # These are PER RUN, PER CATEGORY/ObjectSpace. Test runs can mix OOD (corner) and
    # in-distribution (learning mean/SD) samples.
    test_opposite_sd: float = 0.15
    test_corner_samples_per_space: int = 6
    test_indist_samples_per_space: int = 6

    # Optional: restrict which ObjectSpaces are used for test runs (subset of active spaces)
    # If None/empty, uses all active spaces for the participant.
    test_objectspaces: Optional[List[str]] = None

    # Optional: override max decision duration used for test runs (3-event only).
    # If None, defaults to cfg.max_dec_dur.
    test_max_dec_dur: Optional[float] = None

    # Timing (2-Event)
    dec_dur: float = 0.0
    dec_fb_dur: float = 0.0
    jit_dec_fb_range: Tuple[float, float] = (0.0, 0.0)
    
    # Timing (3-Event)
    img_dur: float = 0.0
    max_dec_dur: float = 0.0
    max_dec_dur_per_run: Optional[List[float]] = None
    fb_dur: float = 0.0
    jit_isi1_range: Tuple[float, float] = (0.0, 0.0)
    jit_isi2_range: Tuple[float, float] = (0.0, 0.0)
    
    # Common
    jit_iti_range: Tuple[float, float] = (0.0, 0.0)
    jitter_config: JitterConfig = None

    # Optional run start/end fixation events (encoded as extra rows in the design CSV)
    add_run_fixations: bool = True
    run_fix_start_dur: float = 10.0
    run_fix_end_dur: float = 5.0

# ==================== STUDY 3 LOGIC ==================== #
def load_and_parse_groups(csv_path):
    """Load the label CSV robustly (auto-detect separator) and normalise column names.

    Returns a pandas DataFrame where the 'ObjectSpace' column is kept as string.
    """
    df = pd.read_csv(csv_path, sep=",", dtype={'ObjectSpace': str}, engine="python")
    # Trim whitespace from column names to reduce common CSV formatting issues
    df.columns = [c.strip() for c in df.columns]
    return df



# ---------- LABEL CONSTRAINT HELPERS ---------- #
def _is_empty_cell(x) -> bool:
    """Treat NaN/None/empty/whitespace-only strings as empty."""
    if x is None:
        return True
    if isinstance(x, float) and np.isnan(x):
        return True
    s = str(x).strip()
    return s == "" or s.lower() in {"nan", "none"}


def allowed_conditions_from_label_row(row: pd.Series, condition_cols: List[str]) -> List[str]:
    """
    Determine which conditions are allowed for a given ObjectSpace row based on label cell presence.

    A condition is 'allowed' if the corresponding cell in that condition column is non-empty.
    This supports the new behaviour:
      - If exactly one condition has a non-empty label, that ObjectSpace is locked to that condition.
      - If multiple conditions have labels, the ObjectSpace can be randomised among those conditions.
      - Empty cells are never used/assigned.
    """
    allowed = []
    for c in condition_cols:
        if c in row.index and not _is_empty_cell(row[c]):
            allowed.append(c)
    return allowed


def constrained_space_condition_map(
    rng,
    selected_spaces: List[str],
    conditions: List[str],
    allowed_by_space: dict,
    global_sc_counts: dict,
    global_pair_counts: dict,
) -> dict:
    """
    Assign conditions to selected_spaces while respecting per-space allowed conditions.

    This function is ONLY used when constraints exist (empty label cells and/or fixed spaces).
    When there are no constraints (all spaces allow all conditions), the original optimiser
    is used unchanged to preserve existing behaviour.

    Strategy:
      1) Lock any space that allows only one condition.
      2) Compute a balanced target distribution across all selected spaces.
      3) Greedily assign remaining spaces, preferring conditions with remaining quota.
      4) If quotas become impossible due to constraints, fall back to choosing among allowed
         conditions to minimise imbalance (soft constraint), ensuring a valid assignment.
      5) Update global trackers in the same manner as optimise_space_condition_map.
    """
    if not selected_spaces:
        return {}

    # Ensure global trackers contain required keys (mirrors optimise_space_condition_map)
    for s in selected_spaces:
        if s not in global_sc_counts:
            global_sc_counts[s] = {c: 0 for c in conditions}
        else:
            for c in conditions:
                global_sc_counts[s].setdefault(c, 0)

    # Fixed assignments: spaces with only one allowed condition
    fixed = {}
    flexible = []
    for s in selected_spaces:
        allowed = allowed_by_space.get(s, conditions)
        if not allowed:
            raise ValueError(f"ObjectSpace '{s}' has no allowed conditions (all label cells empty).")
        if len(allowed) == 1:
            fixed[s] = allowed[0]
        else:
            flexible.append(s)

    # Balanced target counts across ALL selected spaces
    k = len(conditions)
    n = len(selected_spaces)
    base = n // k
    r = n % k
    target = {c: base for c in conditions}
    # Distribute remainder deterministically over conditions order (same as make_balanced_condition_list counting)
    for c in conditions[:r]:
        target[c] += 1

    # Adjust targets to respect fixed counts (ensure target[c] >= fixed_count[c])
    fixed_counts = {c: 0 for c in conditions}
    for s, c in fixed.items():
        fixed_counts[c] += 1

    # If fixed exceeds initial target, raise target and then reduce others if needed to keep sum == n
    for c in conditions:
        if fixed_counts[c] > target[c]:
            target[c] = fixed_counts[c]
    # Reduce other targets if we overshoot total n
    while sum(target.values()) > n:
        # pick a condition we can reduce (above fixed) with the highest current target
        reducible = [c for c in conditions if target[c] > fixed_counts[c]]
        if not reducible:
            break
        # reduce the most over-allocated condition
        c_max = max(reducible, key=lambda cc: target[cc])
        target[c_max] -= 1

    # Remaining quotas after fixed assignments
    remaining = {c: max(0, target[c] - fixed_counts[c]) for c in conditions}

    # Assign flexible spaces, hardest first (fewest allowed)
    flexible_sorted = sorted(flexible, key=lambda s: (len(allowed_by_space.get(s, conditions)), str(s)))

    assign = dict(fixed)
    cur_counts = dict(fixed_counts)

    for s in flexible_sorted:
        allowed = allowed_by_space.get(s, conditions)
        # Prefer conditions with remaining quota
        cand = [c for c in allowed if remaining.get(c, 0) > 0]
        if cand:
            # Choose among candidates with highest remaining quota (tie random)
            max_rem = max(remaining[c] for c in cand)
            best = [c for c in cand if remaining[c] == max_rem]
            c_pick = rng.choice(best) if len(best) > 1 else best[0]
            assign[s] = c_pick
            remaining[c_pick] -= 1
            cur_counts[c_pick] += 1
            continue

        # Soft fallback: pick among allowed to minimise imbalance vs target
        def imbalance_if_pick(c):
            # L1 distance to target after hypothetical pick
            tmp = dict(cur_counts)
            tmp[c] += 1
            return sum(abs(tmp[cc] - target[cc]) for cc in conditions)

        scores = [(imbalance_if_pick(c), c) for c in allowed]
        min_score = min(sc for sc, _ in scores)
        best = [c for sc, c in scores if sc == min_score]
        c_pick = rng.choice(best) if len(best) > 1 else best[0]
        assign[s] = c_pick
        cur_counts[c_pick] += 1

    # Update global trackers (mirrors optimise_space_condition_map)
    for s, c in assign.items():
        global_sc_counts[s][c] = global_sc_counts[s].get(c, 0) + 1

    if len(selected_spaces) > 1:
        # replicate the same canonical pair key used in optimise_space_condition_map
        def pair_key_and_condpair(sa, sb, ca, cb):
            if sa <= sb:
                return (sa, sb), (ca, cb)
            else:
                return (sb, sa), (cb, ca)

        for i in range(len(selected_spaces)):
            for j in range(i + 1, len(selected_spaces)):
                sa, sb = selected_spaces[i], selected_spaces[j]
                ca, cb = assign[sa], assign[sb]
                pk, cp = pair_key_and_condpair(sa, sb, ca, cb)
                if pk not in global_pair_counts:
                    global_pair_counts[pk] = {}
                global_pair_counts[pk][cp] = global_pair_counts[pk].get(cp, 0) + 1

    return assign

def find_closest_image(stim_dir, space_id, target_coords):
    """
    Find the stimulus file in stim_dir/<space_id> whose embedded F0/F1 coordinates
    are closest to the requested target_coords.

    The function expects filenames with the convention:
        {gid}_{F0}_{F1}.ext

    Returns:
      - filename (basename) of the closest image, or the string "MISSING" if none found.
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
            # Parse expected naming format and compute Euclidean distance
            parts = img_file.stem.split('_')
            feat_coords = np.array([float(p) for p in parts[1:3]])
            dist = np.linalg.norm(feat_coords - target_coords)
            if dist < min_dist:
                min_dist, best_img = dist, img_file.name
        except:
            # Ignore files that do not conform to naming assumptions
            continue
    return best_img if best_img else "MISSING"

# ---------- TEST RUN SAMPLING HELPERS ---------- #
def _corner_set():
    """Return the 4 corners of the [0,1]x[0,1] feature space."""
    return [np.array([0.0, 0.0]), np.array([0.0, 1.0]), np.array([1.0, 0.0]), np.array([1.0, 1.0])]


def _closest_corner(mean_xy: np.ndarray) -> np.ndarray:
    """Corner with minimum Euclidean distance to mean_xy."""
    corners = _corner_set()
    dists = [float(np.linalg.norm(c - mean_xy)) for c in corners]
    return corners[int(np.argmin(dists))]


def _furthest_corner(mean_xy: np.ndarray) -> np.ndarray:
    """Corner with maximum Euclidean distance to mean_xy."""
    corners = _corner_set()
    dists = [float(np.linalg.norm(c - mean_xy)) for c in corners]
    return corners[int(np.argmax(dists))]


def get_test_targets_for_space(mean_f0: float, mean_f1: float, mode: str) -> List[np.ndarray]:
    """Return a list of *target means* (coordinates) for test sampling for one ObjectSpace.

    Modes:
      - all_corners: return the three corners furthest from the *closest* corner to the learning mean.
      - furthest_corner: return only the single furthest corner from the learning mean.
      - opposite_gaussian: return only the single furthest corner from the learning mean, but downstream
        sampling uses a Gaussian cloud centred on that corner (mean=corner, SD=cfg.test_opposite_sd).

    Note:
      - The returned list is interpreted as a set of target coordinates to draw from. For
        opposite_gaussian, this list is used only to define the corner mean.
    """
    mean_xy = np.array([float(mean_f0), float(mean_f1)])
    mode = str(mode).strip().lower()

    if mode in {"furthest_corner", "opposite_gaussian"}:
        return [_furthest_corner(mean_xy)]

    # default: all_corners
    closest = _closest_corner(mean_xy)
    corners = _corner_set()
    return [c for c in corners if not np.allclose(c, closest)]


def generate_design_rows(rng, cfg: DesignConfig, sub_id: int, space_cond_map_override: Optional[dict] = None) -> List[dict]:
    """
    Generate a flat list of trial dictionaries for one session.

    Balancing approach (high-level):
     - Determine which ObjectSpaces are active for this subject (space_cond_map).
     - Compute session-level totals and distribute items to per-run pools so each
       run is as balanced as possible across the selected ObjectSpaces.
     - Flatten run pools into a session sequence and iteratively swap items to
       remove 1-back repeats while preserving run-level counts whenever possible.
     - For each trial, sample F0/F1 coordinates from the per-space feature distributions
       and select the best matching image, then compute timing onsets/durations
       according to design_type and jitter configuration.

    Returns:
      A list of dicts, each representing a trial row suitable for CSV writing.
    """
    df_labels = load_and_parse_groups(cfg.label_csv_path)
    all_spaces = df_labels['ObjectSpace'].tolist()
    # Allow caller to override the map; otherwise build a default counterbalanced map
    space_cond_map = space_cond_map_override if space_cond_map_override is not None else get_counterbalanced_map(all_spaces, sub_id, cfg.n_categories_to_select)
    active_spaces = list(space_cond_map.keys())
    # dist_params holds per-space statistics like Mean F0, SD F0, Mean F1, SD F1
    dist_params = df_labels.set_index('ObjectSpace').to_dict('index')

    # --- SESSION BALANCING LOGIC ---
    total_session_trials = cfg.n_runs * cfg.trials_per_run
    n_cats = len(active_spaces)
    
    # 1. Extra items to evenly distribute when totals do not divide evenly
    session_extras = rng.permutation(active_spaces)[:total_session_trials % n_cats]
    
    # 2. Pre-allocate each run's base allocations: repeat the active_spaces base_per_run times
    run_pools = [[] for _ in range(cfg.n_runs)]
    base_per_run = cfg.trials_per_run // n_cats
    for r_idx in range(cfg.n_runs):
        run_pools[r_idx].extend(active_spaces * base_per_run)
    
    # 3. Compute session leftovers (items needed to reach the total) and distribute
    session_leftovers = {s: (total_session_trials // n_cats) - (base_per_run * cfg.n_runs) + (1 if s in session_extras else 0) for s in active_spaces}
    leftover_items = []
    for s, count in session_leftovers.items():
        leftover_items.extend([s] * count)
    rng.shuffle(leftover_items)
    
    rem_per_run = cfg.trials_per_run % n_cats
    # Spread leftovers across run pools (slice-based) then shuffle each run to randomise order
    for r_idx in range(cfg.n_runs):
        run_pools[r_idx].extend(leftover_items[r_idx * rem_per_run : (r_idx + 1) * rem_per_run])
        rng.shuffle(run_pools[r_idx])
    
    # 4. Flatten runs into a session sequence and attempt to remove 1-back repeats
    session_seq = []
    for p in run_pools:
        session_seq.extend(p)
    
    # Iterative swapping heuristic: within-run swaps preferred to preserve per-run balance
    for _ in range(1000):
        conflict = False
        for i in range(1, len(session_seq)):
            if session_seq[i] == session_seq[i-1]:
                run_id = i // cfg.trials_per_run
                r_start, r_end = run_id * cfg.trials_per_run, (run_id + 1) * cfg.trials_per_run
                # Choose candidates within the same run that avoid creating new conflicts
                swap_idx = [j for j in range(r_start, r_end) if session_seq[j] != session_seq[i-1] 
                            and (j==0 or session_seq[j-1]!=session_seq[i]) 
                            and (j<len(session_seq)-1 and session_seq[j+1]!=session_seq[i])]
                if not swap_idx:  # Session-wide fallback if no within-run swap found
                    swap_idx = [j for j in range(len(session_seq)) if session_seq[j] != session_seq[i-1]]
                if swap_idx:
                    k = rng.choice(swap_idx)
                    session_seq[i], session_seq[k] = session_seq[k], session_seq[i]
                else:
                    conflict = True
        if not conflict:
            break

    # --- TRIAL PROCESSING: generate timing rows using the computed session_seq ---
    rows = []
    trial_id_global = 0
    def get_jit(r):
        return sample_jitter(rng, cfg.jitter_config.mode, r[0], r[1],
                             cfg.jitter_config.norm_mu, cfg.jitter_config.norm_sd,
                             cfg.jitter_config.exp_scale, cfg.jitter_config.exp_reverse)

    def _append_fixation_row(run_id: int, onset: float, dur: float, kind: str):
        """Append a fixation-only event row to `rows`.

        These rows are handled specially by the experimental task and simply
        display a fixation cross for `dur` seconds.
        """
        nonlocal trial_id_global

        base = {
            "trial_id": trial_id_global,
            "run_id": run_id,
            "ObjectSpace": "",
            "condition": "Fixation",
            "image_file": "",
            "iti": 0.0,
            "sampled_f0": "",
            "sampled_f1": "",
            "event_type": "fixation",
            "fix_dur": float(dur),
            "fix_kind": str(kind),
        }

        if cfg.design_type == "3event":
            rows.append({
                **base,
                "trial_duration_max": float(dur),
                "img_onset": float(onset),
                "img_dur": 0.0,
                "isi1_dur": 0.0,
                "isi1_type": "hidden",
                "dec_onset_est": float(onset),
                "max_dec_dur": 0.0,
                "isi2_dur": 0.0,
                "isi2_type": "fixation",
                "fb_dur": 0.0,
            })
        else:
            rows.append({
                **base,
                "trial_onset": float(onset),
                "dec_onset": float(onset),
                "dec_dur": 0.0,
                "dec_fb_onset": float(onset),
                "dec_fb_dur": 0.0,
                "trial_duration": float(dur),
                "dec_fb_jit": 0.0,
            })

        trial_id_global += 1

    for run_id in range(1, cfg.n_runs + 1):
        current_time = float(cfg.start_time)
        # If provided, use per-run max decision durations; otherwise fall back to cfg.max_dec_dur
        if getattr(cfg, 'max_dec_dur_per_run', None) is not None and len(cfg.max_dec_dur_per_run) >= run_id:
            _run_max_dec_dur = float(cfg.max_dec_dur_per_run[run_id - 1])
        else:
            _run_max_dec_dur = float(cfg.max_dec_dur)
        run_start = (run_id - 1) * cfg.trials_per_run
        run_end = run_id * cfg.trials_per_run
        space_seq = session_seq[run_start:run_end]

        # Optional run-start fixation (default: 10s)
        if bool(getattr(cfg, 'add_run_fixations', True)) and float(getattr(cfg, 'run_fix_start_dur', 0.0)) > 0:
            _append_fixation_row(run_id, current_time, float(cfg.run_fix_start_dur), kind='run_start')
            current_time += float(cfg.run_fix_start_dur)
        
        for space_id in space_seq:
            cond = space_cond_map[space_id]
            info = dist_params[space_id]
            
            # Sample feature coordinates from the per-space Gaussian and clamp to [0,1]
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
                # Compute ISI1/ISI2 and absolute onsets for self-paced design (max durations used for schedule)
                isi1, isi2 = get_jit(cfg.jit_isi1_range), get_jit(cfg.jit_isi2_range)
                t_img, t_dec = current_time, current_time + cfg.img_dur + isi1
                t_fb, t_end  = t_dec + _run_max_dec_dur + isi2, t_dec + _run_max_dec_dur + isi2 + cfg.fb_dur
                rows.append({
                    **common, "trial_duration_max": t_end - t_img, "img_onset": t_img,
                    "img_dur": cfg.img_dur, "isi1_dur": isi1, "isi1_type": "hidden",
                    "dec_onset_est": t_dec, "max_dec_dur": _run_max_dec_dur,
                    "isi2_dur": isi2, "isi2_type": "fixation", "fb_dur": cfg.fb_dur
                })
                current_time = t_end + iti
            else:
                # Standard 2-event timeline using configured dec and feedback durations and jitter
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

        # Optional run-end fixation (default: 5s)
        if bool(getattr(cfg, 'add_run_fixations', True)) and float(getattr(cfg, 'run_fix_end_dur', 0.0)) > 0:
            _append_fixation_row(run_id, current_time, float(cfg.run_fix_end_dur), kind='run_end')
            current_time += float(cfg.run_fix_end_dur)
    # --- OPTIONAL TEST RUNS (appended as additional runs) ---
    # Test runs are appended after the learning runs as additional run_id blocks.
    # Crucially, test trials are generated with *no feedback* and *no ISI2* in the exported schedule:
    #  - 3-event: fb_dur = 0.0 and isi2_dur = 0.0 (still includes ISI1 + ITI as configured)
    #  - 2-event: dec_fb_dur = 0.0 and dec_fb_jit = 0.0
    #
    # Additional features:
    #  - test_mode == 'opposite_gaussian' supports mixing OOD corner-cloud samples with in-distribution samples.
    #  - cfg.test_objectspaces can restrict the set of ObjectSpaces used in test runs (subset of active spaces).
    if bool(getattr(cfg, "include_test_run", False)) and int(getattr(cfg, "n_test_runs", 0) or 0) > 0:
        n_test_runs = int(getattr(cfg, "n_test_runs", 1) or 1)
        test_mode = str(getattr(cfg, "test_mode", "all_corners") or "all_corners").strip().lower()

        # Legacy params (all_corners / furthest_corner)
        repeats_per_space = int(getattr(cfg, "test_repeats_per_space", 1) or 1)
        samples_per_space = int(getattr(cfg, "test_samples_per_space", 1) or 1)

        # Opposite-corner Gaussian params (opposite_gaussian)
        opp_sd = float(getattr(cfg, "test_opposite_sd", 0.15) or 0.15)
        n_corner = int(getattr(cfg, "test_corner_samples_per_space", 0) or 0)
        n_indist = int(getattr(cfg, "test_indist_samples_per_space", 0) or 0)

        # Determine max decision window for 3-event test runs (override allowed)
        _test_max_dec_dur = None
        if cfg.design_type == "3event":
            if getattr(cfg, "test_max_dec_dur", None) is not None and str(getattr(cfg, "test_max_dec_dur")).strip() != "":
                _test_max_dec_dur = float(getattr(cfg, "test_max_dec_dur"))
            else:
                _test_max_dec_dur = float(cfg.max_dec_dur)

        # Restrict test spaces if requested (subset of active_spaces); otherwise use all active.
        req_spaces = getattr(cfg, "test_objectspaces", None)
        if req_spaces:
            req_spaces = [str(s).strip() for s in req_spaces if str(s).strip() != ""]
        test_spaces = [s for s in (req_spaces or active_spaces) if s in active_spaces]
        if not test_spaces:
            test_spaces = list(active_spaces)

        for t_run_idx in range(n_test_runs):
            test_run_id = cfg.n_runs + 1 + t_run_idx
            current_time = float(cfg.start_time)

            # Optional run-start fixation (default: 10s)
            if bool(getattr(cfg, "add_run_fixations", True)) and float(getattr(cfg, "run_fix_start_dur", 0.0)) > 0:
                _append_fixation_row(test_run_id, current_time, float(cfg.run_fix_start_dur), kind="run_start")
                current_time += float(cfg.run_fix_start_dur)

            test_trials = []

            if test_mode == "opposite_gaussian":
                # For each ObjectSpace: mix corner-cloud (OOD) and in-distribution samples.
                # Counts are PER RUN, PER SPACE.
                for space_id in test_spaces:
                    info = dist_params[space_id]
                    corner_mean = get_test_targets_for_space(info["Mean F0"], info["Mean F1"], "opposite_gaussian")[0]

                    # OOD samples: Gaussian cloud centred on opposite corner
                    for _ in range(max(0, n_corner)):
                        f0 = float(np.clip(rng.normal(float(corner_mean[0]), opp_sd), 0, 1))
                        f1 = float(np.clip(rng.normal(float(corner_mean[1]), opp_sd), 0, 1))
                        test_trials.append((space_id, f0, f1))

                    # In-distribution samples: Gaussian around learning mean/SD (as learning runs)
                    for _ in range(max(0, n_indist)):
                        f0 = float(np.clip(rng.normal(info['Mean F0'], info['SD F0']), 0, 1))
                        f1 = float(np.clip(rng.normal(info['Mean F1'], info['SD F1']), 0, 1))
                        test_trials.append((space_id, f0, f1))

                rng.shuffle(test_trials)

            else:
                # Legacy corner target list sampling
                per_space_targets = {}
                for space_id in test_spaces:
                    info = dist_params[space_id]
                    targets = get_test_targets_for_space(info["Mean F0"], info["Mean F1"], test_mode)

                    n_trials = max(0, repeats_per_space) * max(0, samples_per_space)
                    if n_trials <= 0 or len(targets) == 0:
                        per_space_targets[space_id] = []
                        continue

                    # Sample target corners; if more trials than targets, sample with replacement.
                    idxs = rng.choice(len(targets), size=n_trials, replace=(n_trials > len(targets)))
                    t_list = [targets[int(k)].copy() for k in idxs]
                    rng.shuffle(t_list)
                    per_space_targets[space_id] = t_list

                # Flatten trials across spaces and shuffle
                for space_id in test_spaces:
                    for tgt in per_space_targets.get(space_id, []):
                        f0, f1 = float(np.clip(tgt[0], 0, 1)), float(np.clip(tgt[1], 0, 1))
                        test_trials.append((space_id, f0, f1))
                rng.shuffle(test_trials)

            # Light no-1-back heuristic on ObjectSpace
            for _ in range(1000):
                conflict = False
                for i in range(1, len(test_trials)):
                    if test_trials[i][0] == test_trials[i-1][0]:
                        swap_idx = [j for j in range(len(test_trials)) if test_trials[j][0] != test_trials[i-1][0]]
                        if swap_idx:
                            k = int(rng.choice(swap_idx))
                            test_trials[i], test_trials[k] = test_trials[k], test_trials[i]
                        else:
                            conflict = True
                if not conflict:
                    break

            for space_id, f0, f1 in test_trials:
                cond = space_cond_map.get(space_id, "Test")
                img_file = find_closest_image(cfg.stim_dir, space_id, np.array([f0, f1]))
                iti = get_jit(cfg.jit_iti_range)

                common = {
                    "trial_id": trial_id_global, "run_id": test_run_id, "ObjectSpace": space_id,
                    "condition": cond, "image_file": img_file, "iti": iti,
                    "sampled_f0": round(float(f0), 3), "sampled_f1": round(float(f1), 3)
                }

                if cfg.design_type == "3event":
                    isi1 = get_jit(cfg.jit_isi1_range)
                    t_img = current_time
                    t_dec = current_time + cfg.img_dur + isi1
                    t_end = t_dec + float(_test_max_dec_dur)  # no ISI2, no feedback

                    rows.append({
                        **common,
                        "trial_duration_max": t_end - t_img,
                        "img_onset": t_img,
                        "img_dur": cfg.img_dur,
                        "isi1_dur": isi1,
                        "isi1_type": "hidden",
                        "dec_onset_est": t_dec,
                        "max_dec_dur": float(_test_max_dec_dur),
                        # Explicitly remove ISI2 + feedback in the exported design rows
                        "isi2_dur": 0.0,
                        "isi2_type": "fixation",
                        "fb_dur": 0.0,
                    })
                    current_time = t_end + iti

                else:
                    # 2-event test: decision only (no jitter, no feedback)
                    t_trial = current_time
                    t_end = t_trial + cfg.dec_dur

                    rows.append({
                        **common,
                        "trial_onset": t_trial,
                        "dec_onset": t_trial,
                        "dec_dur": cfg.dec_dur,
                        "dec_fb_onset": t_end,
                        "dec_fb_dur": 0.0,
                        "trial_duration": t_end - t_trial,
                        "dec_fb_jit": 0.0,
                    })
                    current_time = t_end + iti

                trial_id_global += 1

            # Optional run-end fixation (default: 5s)
            if bool(getattr(cfg, "add_run_fixations", True)) and float(getattr(cfg, "run_fix_end_dur", 0.0)) > 0:
                _append_fixation_row(test_run_id, current_time, float(cfg.run_fix_end_dur), kind="run_end")
                current_time += float(cfg.run_fix_end_dur)

    return rows



# 2. Restore the Latin Square (Order of selection changes per participant)
def get_counterbalanced_map(all_spaces, participant_id, n_select):
    """
    Map a rotated selection of ObjectSpaces to conditions using a simple
    Latin-square-like rotation based on participant_id. This provides a
    deterministic but participant-dependent assignment.
    """
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
    """
    Construct a shuffled list of conditions of length n_items where counts across
    conditions differ by at most one (e.g., for 8 items and 3 conditions -> 3,3,2).
    """
    k = len(conditions)
    if k == 0 or n_items <= 0:
        return []
    q, r = divmod(n_items, k)
    # Randomly choose which conditions receive the extra +1 when n_items not divisible by k
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
    Constrained randomisation to assign conditions to the selected spaces.

    Goals:
      - Within-participant: produce as balanced a distribution across conditions as possible.
      - Across participants: avoid repeating the same (space,condition) pairs too often
        and penalise frequently observed ordered pairs of assignments (pair_term).

    The routine generates multiple candidate maps, scores them, and selects
    the candidate with the lowest score. Global trackers are updated with the
    chosen assignment for use by subsequent participant iterations.
    """
    if not selected_spaces:
        return {}

    # Ensure global trackers contain required keys
    for s in selected_spaces:
        if s not in global_sc_counts:
            global_sc_counts[s] = {c: 0 for c in conditions}
        else:
            for c in conditions:
                global_sc_counts[s].setdefault(c, 0)

    def pair_key_and_condpair(sa, sb, ca, cb):
        # canonical ordering for pair keys and their corresponding condition-pair
        if sa <= sb:
            return (sa, sb), (ca, cb)
        else:
            return (sb, sa), (cb, ca)

    best_map = None
    best_score = None

    n_candidates = int(max(1, n_candidates))
    # Candidate-generation loop: sample many balanced assignments and score them
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
        # Fallback: simple balanced assignment if no candidate chosen
        cond_list = make_balanced_condition_list(rng, len(selected_spaces), conditions)
        best_map = {s: c for s, c in zip(selected_spaces, cond_list)}

    # Update global trackers to reflect the selected assignment
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
    """
    Parse a 'Min,Max' string into two floats.

    Raises a ValueError with a helpful message if parsing fails so callers can
    present the error to the user via the GUI.
    """
    try:
        parts = [float(x) for x in val_str.split(',')]
        if len(parts) < 2:
            raise ValueError(f"{field_name} needs a Min and Max separated by a comma.")
        return parts[0], parts[1]
    except Exception:
        raise ValueError(f"Invalid format for {field_name}. Use 'Min,Max' (e.g., 0.5,1.5)")


def generate_sequence_no_repeat(rng, items, length):
    """
    Create a sequence of the requested length using items such that there
    are no immediate (1-back) repeats if possible.

    The function builds an initial balanced pool and then applies a swapping
    heuristic to remove adjacent matches. If items has length 1, repeats are
    unavoidable and the sequence is the repeated item.
    """
    if not items: return []
    if len(items) == 1: return items * length
    
    # Initial balanced pool
    base = length // len(items)
    rem = length % len(items)
    pool = items * base + list(rng.choice(items, size=rem, replace=False))
    rng.shuffle(pool)
    
    # Swapping heuristic to fix 1-back repeats
    for _ in range(500):
        conflict = False
        for i in range(1, len(pool)):
            if pool[i] == pool[i-1]:
                # Find swap candidate that avoids creating new adjacent conflicts
                swap_idx = [j for j in range(len(pool)) if pool[j] != pool[i-1] and (j==0 or pool[j-1]!=pool[i]) and (j<len(pool)-1 and pool[j+1]!=pool[i])]
                if not swap_idx: 
                    # fallback: any non-match in the sequence
                    swap_idx = [j for j in range(len(pool)) if pool[j] != pool[i-1]]
                
                if swap_idx:
                    k = rng.choice(swap_idx)
                    pool[i], pool[k] = pool[k], pool[i]
                else:
                    conflict = True # Could not fix this conflict
        if not conflict: break
    
    return pool


# ==================== GUI ==================== #
# The remainder of the file implements a Tk-based interface for operator configuration
# and a 'run' method that saves per-subject session CSVs using the generation routines above.

class DesignGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Study 3 Design Generator")
        self.geometry("1100x800")
        self._build_ui()
        # Update the estimated timings display as soon as the window is created
        self.update_estimates()

    def _init_vars(self):
        """Initialize tracked Tk variables and attach traces for live estimate updates."""
        # Global defaults (these may be replaced by explicit widget initializers)
        self.v_runs = tk.StringVar(value="2")
        self.v_trials = tk.StringVar(value="30")
        self.v_runs_s2 = tk.StringVar(value="2")
        self.v_trials_s2 = tk.StringVar(value="20")
        
        # 2-Event Timing defaults
        self.v_2_dec = tk.StringVar(value="2.0")
        self.v_2_fb = tk.StringVar(value="1.0")
        self.v_2_jit = tk.StringVar(value="0.5,1.5")
        self.v_2_iti = tk.StringVar(value="1.0,3.0")
        
        # 3-Event Timing defaults
        self.v_3_img = tk.StringVar(value="1.0")
        self.v_3_max = tk.StringVar(value="3.0")
        self.v_3_fb = tk.StringVar(value="1.0")
        self.v_3_isi1 = tk.StringVar(value="0.5,1.5")
        self.v_3_isi2 = tk.StringVar(value="0.5,1.0")
        self.v_3_iti = tk.StringVar(value="1.0,3.0")

        # Add traces so updates recalc estimates on change
        vars_to_trace = [
            self.v_runs, self.v_trials, self.v_runs_s2, self.v_trials_s2,
            self.v_2_dec, self.v_2_fb, self.v_2_jit, self.v_2_iti,
            self.v_3_img, self.v_3_max, self.v_3_fb, self.v_3_isi1, self.v_3_isi2, self.v_3_iti
        ]
        for v in vars_to_trace:
            v.trace_add("write", lambda *a: self.update_estimates())

    def get_float(self, var, default=0.0):
        """Parse a Tk StringVar into float, allowing comma decimal separators."""
        try:
            val = var.get().replace(',', '.') # Handle comma decimal separators
            return float(val)
        except: return default

    def get_int(self, var, default):
        """Parse Tk StringVar into integer with safe fallback."""
        try: return int(var.get())
        except: return default

    def _build_ui(self):
        """Construct the GUI layout: file inputs, global settings, timing tabs, and action button."""
        # ---- Scrollable main container (prevents UI extending beyond screen) ----
        container = ttk.Frame(self)
        container.pack(fill="both", expand=True)

        canvas = tk.Canvas(container, highlightthickness=0)
        vscroll = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vscroll.set)

        vscroll.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        main = ttk.Frame(canvas, padding=10)
        main_window = canvas.create_window((0, 0), window=main, anchor="nw")

        def _on_main_configure(_event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_canvas_configure(event):
            # Keep the inner frame the same width as the canvas so widgets don't get clipped horizontally
            canvas.itemconfigure(main_window, width=event.width)

        main.bind("<Configure>", _on_main_configure)
        canvas.bind("<Configure>", _on_canvas_configure)

        # Mousewheel scrolling (Windows/macOS)
        def _on_mousewheel(event):
            # event.delta is typically 120 increments on Windows; on macOS it can be smaller.
            delta = event.delta
            if delta == 0:
                return
            canvas.yview_scroll(int(-1 * (delta / 120)), "units")

        # Linux (X11) uses Button-4/5
        def _on_mousewheel_linux(event):
            if event.num == 4:
                canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                canvas.yview_scroll(1, "units")

        # Bind to the canvas (not the whole app) to avoid hijacking scrolling in other windows
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        canvas.bind_all("<Button-4>", _on_mousewheel_linux)
        canvas.bind_all("<Button-5>", _on_mousewheel_linux)
        # -------------------------------------------------------------------

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
      

        # Session 1 and Session 2 specific parameters (compact layout)
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

        # Optional run start/end fixations (10s at start, 5s at end) encoded into the design file
        self.v_add_run_fix = tk.BooleanVar(value=True)
        ttk.Checkbutton(fr_glob, text="Add run fixations (10s start / 5s end)", variable=self.v_add_run_fix).pack(side="left", padx=8)

                
        # Test run settings (placed on separate rows so it doesn't exceed screen width)
        fr_test = ttk.LabelFrame(main, text="Test Runs")
        fr_test.pack(fill="x", pady=5)

        fr_test_row1 = ttk.Frame(fr_test)
        fr_test_row1.pack(fill="x", pady=(0, 2))
        fr_test_row2 = ttk.Frame(fr_test)
        fr_test_row2.pack(fill="x")

        self.v_include_test = tk.BooleanVar(value=False)
        ttk.Checkbutton(fr_test_row1, text="Include", variable=self.v_include_test, command=self.update_estimates).pack(side="left", padx=(0, 8))

        ttk.Label(fr_test_row1, text="Append to:").pack(side="left")
        # Where to append test runs (end of Session 1, Session 2, or both)
        self.v_test_append_to = tk.StringVar(value="both")
        ttk.OptionMenu(fr_test_row1, self.v_test_append_to, "both", "s1", "s2", "both",
                       command=lambda *_: self.update_estimates()).pack(side="left", padx=(2, 10))

        ttk.Label(fr_test_row1, text="# Test Runs:").pack(side="left")
        self.v_n_test_runs = tk.StringVar(value="1")
        ttk.Entry(fr_test_row1, textvariable=self.v_n_test_runs, width=4).pack(side="left", padx=(2, 10))

        ttk.Label(fr_test_row1, text="Mode:").pack(side="left")
        self.v_test_mode = tk.StringVar(value="all_corners")
        ttk.OptionMenu(fr_test_row1, self.v_test_mode, "all_corners", "all_corners", "furthest_corner", "opposite_gaussian",
                       command=lambda *_: self._on_test_mode_change()).pack(side="left", padx=(2, 10))

        # Legacy (corner list) controls
        ttk.Label(fr_test_row1, text="Repeats:").pack(side="left")
        self.v_test_repeats = tk.StringVar(value="3")
        self.ent_test_repeats = ttk.Entry(fr_test_row1, textvariable=self.v_test_repeats, width=4)
        self.ent_test_repeats.pack(side="left", padx=(2, 10))

        ttk.Label(fr_test_row1, text="Samples:").pack(side="left")
        self.v_test_samples = tk.StringVar(value="3")
        self.ent_test_samples = ttk.Entry(fr_test_row1, textvariable=self.v_test_samples, width=4)
        self.ent_test_samples.pack(side="left", padx=(2, 0))

        # Opposite-corner Gaussian mix controls (per run, per category)
        ttk.Label(fr_test_row2, text="OOD#:").pack(side="left")
        self.v_test_corner_n = tk.StringVar(value="6")
        self.ent_test_corner_n = ttk.Entry(fr_test_row2, textvariable=self.v_test_corner_n, width=4)
        self.ent_test_corner_n.pack(side="left", padx=(2, 10))

        ttk.Label(fr_test_row2, text="InDist#:").pack(side="left")
        self.v_test_indist_n = tk.StringVar(value="6")
        self.ent_test_indist_n = ttk.Entry(fr_test_row2, textvariable=self.v_test_indist_n, width=4)
        self.ent_test_indist_n.pack(side="left", padx=(2, 10))

        ttk.Label(fr_test_row2, text="Opp SD:").pack(side="left")
        self.v_test_opp_sd = tk.StringVar(value="0.15")
        self.ent_test_opp_sd = ttk.Entry(fr_test_row2, textvariable=self.v_test_opp_sd, width=6)
        self.ent_test_opp_sd.pack(side="left", padx=(2, 10))

        # Optional ObjectSpace restriction (comma-separated)
        ttk.Label(fr_test_row2, text="Test ObjectSpaces:").pack(side="left")
        self.v_test_objectspaces = tk.StringVar(value="")
        self.ent_test_objectspaces = ttk.Entry(fr_test_row2, textvariable=self.v_test_objectspaces, width=18)
        self.ent_test_objectspaces.pack(side="left", padx=(2, 10))

        # Optional test max decision duration override (3-event only; blank = use learning)
        ttk.Label(fr_test_row2, text="Test MaxDec:").pack(side="left")
        self.v_test_max_dec = tk.StringVar(value="")
        self.ent_test_max_dec = ttk.Entry(fr_test_row2, textvariable=self.v_test_max_dec, width=6)
        self.ent_test_max_dec.pack(side="left", padx=(2, 0))

# Keep estimates responsive
        for _v in [self.v_n_test_runs, self.v_test_repeats, self.v_test_samples,
                  self.v_test_corner_n, self.v_test_indist_n, self.v_test_opp_sd,
                  self.v_test_objectspaces, self.v_test_max_dec, self.v_test_append_to]:
            try:
                _v.trace_add("write", lambda *a: self.update_estimates())
            except Exception:
                pass

        # Apply initial enabling/disabling based on test mode
        self._on_test_mode_change()

        # 3. Timing Tabs (2-event vs 3-event configuration)
        self.nb = ttk.Notebook(main)
        self.nb.pack(fill="x", pady=10)
        self.nb.bind("<<NotebookTabChanged>>", lambda e: self.update_estimates())
        
        self.tab2 = ttk.Frame(self.nb)
        self.nb.add(self.tab2, text="Standard (2-Event)")
        self._build_2event_tab(self.tab2)
        
        self.tab3 = ttk.Frame(self.nb)
        self.nb.add(self.tab3, text="Self-Paced (3-Event)")
        self._build_3event_tab(self.tab3)

        # 3.5 ESTIMATES PANEL (New Section) - shows operator approximate timings
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
        """Construct controls for the 2-event timing parameters."""
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
        """Construct controls for the 3-event timing parameters."""
        f = ttk.Frame(p); f.pack(fill="both", padx=10, pady=10)
        # Simplified row layout
        r1 = ttk.Frame(f); r1.pack(fill="x")
        ttk.Label(r1, text="Img Dur:").pack(side="left")
        self.v_3_img = tk.StringVar(value="1.0")
        ttk.Entry(r1, textvariable=self.v_3_img, width=5).pack(side="left")
        ttk.Label(r1, text="Max Dec Dur:").pack(side="left", padx=5)
        self.v_3_max = tk.StringVar(value="3.0")
        ttk.Entry(r1, textvariable=self.v_3_max, width=5).pack(side="left")
        # Optional per-run max decision duration (vary max_dec_dur across runs)
        self.v_3_max_per_run = tk.BooleanVar(value=False)
        ttk.Checkbutton(r1, text="Per-run", variable=self.v_3_max_per_run,
                        command=self._refresh_per_run_dec_dur_ui).pack(side="left", padx=6)
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

        # Per-run decision durations panel (hidden unless enabled)
        self._per_run_dec_container = ttk.LabelFrame(f, text="Per-run Max Decision Duration (s)")
        self._per_run_dec_container.pack(fill="x", pady=(6, 0))
        self._per_run_dec_container.pack_forget()  # hidden by default

        # Hold per-run StringVars for each session
        self.v_3_max_per_run_s1 = []
        self.v_3_max_per_run_s2 = []

        # Refresh per-run UI whenever run counts change (only meaningful if enabled)
        try:
            self.v_runs.trace_add("write", lambda *a: self._refresh_per_run_dec_dur_ui())
            self.v_runs_s2.trace_add("write", lambda *a: self._refresh_per_run_dec_dur_ui())
        except Exception:
            pass

        self._refresh_per_run_dec_dur_ui()

    def _refresh_per_run_dec_dur_ui(self):
        """Rebuild per-run max decision duration widgets for session 1 and session 2.

        This does nothing unless the 'Per-run' checkbox is enabled.
        """
        if not hasattr(self, "v_3_max_per_run") or not hasattr(self, "_per_run_dec_container"):
            return

        enabled = bool(self.v_3_max_per_run.get())
        if not enabled:
            try:
                self._per_run_dec_container.pack_forget()
            except Exception:
                pass
            return

        # Ensure visible
        try:
            self._per_run_dec_container.pack(fill="x", pady=(6, 0))
        except Exception:
            pass

        # Clear previous widgets
        for child in list(self._per_run_dec_container.winfo_children()):
            try:
                child.destroy()
            except Exception:
                pass

        n_s1 = self.get_int(self.v_runs, 0)
        n_s2 = self.get_int(self.v_runs_s2, 0)

        default_txt = str(self.v_3_max.get()).strip()

        def _build_row(label: str, n: int):
            fr = ttk.Frame(self._per_run_dec_container)
            fr.pack(fill="x", pady=(2, 2))
            ttk.Label(fr, text=label, width=14).pack(side="left")

            vars_list = []
            for i in range(int(n)):
                v = tk.StringVar(value=default_txt)
                vars_list.append(v)
                ttk.Label(fr, text=f"R{i+1}:").pack(side="left", padx=(6, 1))
                ttk.Entry(fr, textvariable=v, width=6).pack(side="left")
            return vars_list

        self.v_3_max_per_run_s1 = _build_row("Session 1:", n_s1)
        self.v_3_max_per_run_s2 = _build_row("Session 2:", n_s2)



    def _on_test_mode_change(self):
        """Enable/disable test-run sampling controls based on selected test mode."""
        try:
            mode = str(self.v_test_mode.get()).strip().lower()
        except Exception:
            mode = "all_corners"

        # Legacy (corner list) controls
        legacy_state = "normal" if mode in {"all_corners", "furthest_corner"} else "disabled"
        # Opposite-corner Gaussian mix controls
        opp_state = "normal" if mode == "opposite_gaussian" else "disabled"

        for w in [getattr(self, "ent_test_repeats", None), getattr(self, "ent_test_samples", None)]:
            if w is not None:
                try:
                    w.configure(state=legacy_state)
                except Exception:
                    pass

        for w in [getattr(self, "ent_test_corner_n", None), getattr(self, "ent_test_indist_n", None), getattr(self, "ent_test_opp_sd", None)]:
            if w is not None:
                try:
                    w.configure(state=opp_state)
                except Exception:
                    pass

        # Always allow restriction / overrides
        for w in [getattr(self, "ent_test_objectspaces", None), getattr(self, "ent_test_max_dec", None)]:
            if w is not None:
                try:
                    w.configure(state="normal")
                except Exception:
                    pass

        # Refresh estimates
        self.update_estimates()
    def browse_csv(self):
        """Open a file dialog to select the label CSV and set the entry field."""
        f = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if f: self.v_csv.set(f)

    def log(self, txt):
        """Append a line to the GUI log box and scroll to the bottom."""
        self.log_box.insert("end", txt+"\n"); self.log_box.see("end")

    def _save_session_data(self, rows, sub_id, sess_id, prefix, mode):
        """Helper to save one consolidated CSV for a participant/session.

        Creates <prefix>/session-<sess_id>/ and writes a single master CSV containing
        all runs for that session. Fieldnames are inferred from the first row.

        Notes:
          - `prefix` is treated as an output directory (may contain path separators).
          - The filename uses the basename of `prefix` to avoid duplicating directories.
        """
        if not rows:
            self.log(f"[ERROR] No rows to save for Session {sess_id}. Skipping file write.")
            return

        sess_dir = os.path.join(prefix, f"session-{sess_id}")
        os.makedirs(sess_dir, exist_ok=True)

        # Save all runs together in one file
        prefix_name = os.path.basename(os.path.normpath(prefix))
        fname = f"{prefix_name}_sub-{sub_id:03d}_sess-{sess_id}_{mode}.csv"
        fpath = os.path.join(sess_dir, fname)

        with open(fpath, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)

        self.log(f"Generated Session {sess_id} master file: {fname}")

    def update_estimates(self):
        """Live calculation for trial, total task lengths, and test run durations.

        This uses the current GUI parameter values to compute approximate min/max/mean
        trial durations (with and without ISI2) so operators can preview session lengths.
        """
        try:
            tab = self.nb.index(self.nb.select())
            n_runs1 = self.get_int(self.v_runs, 0)
            n_trials1 = self.get_int(self.v_trials, 0)
            n_runs2 = self.get_int(self.v_runs_s2, 0)
            n_trials2 = self.get_int(self.v_trials_s2, 0)
            
            # Test Run Parameters
            include_test = bool(getattr(self, "v_include_test", tk.BooleanVar(value=False)).get())
            append_to = str(getattr(self, "v_test_append_to", tk.StringVar(value="both")).get()).strip().lower()
            n_test_runs = self.get_int(getattr(self, "v_n_test_runs", tk.StringVar(value="0")), 0)
            test_mode = str(getattr(self, "v_test_mode", tk.StringVar(value="all_corners")).get()).strip().lower()

            test_repeats = self.get_int(getattr(self, "v_test_repeats", tk.StringVar(value="3")), 3)
            test_samples = self.get_int(getattr(self, "v_test_samples", tk.StringVar(value="3")), 3)

            test_corner_n = self.get_int(getattr(self, "v_test_corner_n", tk.StringVar(value="6")), 6)
            test_indist_n = self.get_int(getattr(self, "v_test_indist_n", tk.StringVar(value="6")), 6)

            n_cats = self.get_int(self.v_n_cats, 6)

            # Total test trials across all test runs (estimation uses selected category count)
            if include_test and n_test_runs > 0:
                if test_mode == "opposite_gaussian":
                    total_test_trials = n_cats * max(0, (test_corner_n + test_indist_n)) * n_test_runs
                else:
                    total_test_trials = n_cats * max(0, test_repeats) * max(0, test_samples) * n_test_runs
            else:
                total_test_trials = 0
            # 1. Calculate Base and Jitter components based on selected tab
            if tab == 0: # 2-Event
                dec = self.get_float(self.v_2_dec)
                fb = self.get_float(self.v_2_fb)
                j_min, j_max = parse_range(self.v_2_jit.get(), "Jit")
                i_min, i_max = parse_range(self.v_2_iti.get(), "ITI")
                
                # Full Trial (with jitter)
                t_min, t_max = dec + fb + j_min + i_min, dec + fb + j_max + i_max
                # Without Jitter (e.g., no ISI2)
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

            # 2. Update GUI labels with formatted values
            self.lbl_trial_mean.config(text=f"{t_mean:.2f}s")
            self.lbl_trial_mean_nojit.config(text=f"(No-ISI2: {tn_mean:.2f}s)")
            self.lbl_trial_range.config(text=f"{t_min:.1f}s - {t_max:.1f}s")
            self.lbl_trial_range_nojit.config(text=f"(No-ISI2: {tn_min:.1f}s - {tn_max:.1f}s)")

            # Total session time without test runs (approximation in minutes)
            s1_full = (n_runs1 * n_trials1 * t_mean) / 60
            s1_no_isi2 = (n_runs1 * n_trials1 * tn_mean) / 60
            s2_full = (n_runs2 * n_trials2 * t_mean) / 60
            s2_no_isi2 = (n_runs2 * n_trials2 * tn_mean) / 60

            # Optional run-start/end fixations (10s start, 5s end per run)
            add_fix = bool(getattr(self, 'v_add_run_fix', tk.BooleanVar(value=True)).get())
            if add_fix:
                extra_s1 = (n_runs1 * (10.0 + 5.0)) / 60.0
                extra_s2 = (n_runs2 * (10.0 + 5.0)) / 60.0
                s1_full += extra_s1
                s1_no_isi2 += extra_s1
                s2_full += extra_s2
                s2_no_isi2 += extra_s2
            
                        # Test run duration (total across all specified test runs)
            test_dur_total = (total_test_trials * test_t_mean) / 60.0 if include_test else 0.0

            # Apply append target (Session 1, Session 2, or both)
            test_dur_s1 = test_dur_total if include_test and append_to in {"s1", "both"} else 0.0
            test_dur_s2 = test_dur_total if include_test and append_to in {"s2", "both"} else 0.0
            # 3. Final String Formatting for display
            self.lbl_s1_total.config(text=f"~{s1_full:.2f} min")
            self.lbl_s1_total_nojit.config(text=f"No-ISI2: ~{s1_no_isi2:.2f} min | test length: ~{test_dur_s1:.2f} min")
            
            self.lbl_s2_total.config(text=f"~{s2_full:.2f} min")
            self.lbl_s2_total_nojit.config(text=f"No-ISI2: ~{s2_no_isi2:.2f} min | test length: ~{test_dur_s2:.2f} min")

        except Exception:
            # Silent guard: GUI should remain responsive even if a transient parse error occurs
            pass
    def run(self):
        """Main entry invoked by the 'Generate Designs' button.

        Reads GUI values, assembles DesignConfig objects for session 1 and 2,
        and iterates over requested participants to write per-session CSV master files.
        """
        if not os.path.exists(self.v_csv.get()):
            messagebox.showerror("Error", "CSV not found")
            return
        try:
            selected_tab = self.nb.select()
            mode = "2event" if selected_tab == str(self.tab2) else "3event"
            
            # Construct jitter configuration from GUI controls
            jconf = JitterConfig(
                mode=self.v_jmode.get(), 
                norm_mu=self.get_float(self.v_nmu, None), 
                norm_sd=self.get_float(self.v_nsd, None),
                exp_scale=self.get_float(self.v_esc, None), 
                exp_reverse=self.v_erev.get()
            )
            
            # Build session 1 config from GUI values
            cfg_s1 = DesignConfig(
                design_type=mode, 
                n_runs=self.get_int(self.v_runs, 2), 
                trials_per_run=self.get_int(self.v_trials, 30),
                start_time=0.0, 
                n_categories_to_select=self.get_int(self.v_n_cats, 10), 
                label_csv_path=self.v_csv.get(), 
                jitter_config=jconf
            )

            #             # Test run configuration (optional)
            # Test runs are appended to the end of the sessions selected in the GUI.
            include_test = bool(getattr(self, "v_include_test", tk.BooleanVar(value=False)).get())
            append_to = str(getattr(self, "v_test_append_to", tk.StringVar(value="both")).get()).strip().lower()

            cfg_s1.include_test_run = bool(include_test and append_to in {"s1", "both"})
            cfg_s1.n_test_runs = self.get_int(getattr(self, "v_n_test_runs", tk.StringVar(value="1")), 1)
            cfg_s1.test_mode = str(getattr(self, "v_test_mode", tk.StringVar(value="all_corners")).get()).strip()
            cfg_s1.test_repeats_per_space = self.get_int(getattr(self, "v_test_repeats", tk.StringVar(value="3")), 3)
            cfg_s1.test_samples_per_space = self.get_int(getattr(self, "v_test_samples", tk.StringVar(value="3")), 3)

            # Opposite-corner Gaussian mix (per run, per category)
            cfg_s1.test_corner_samples_per_space = self.get_int(getattr(self, "v_test_corner_n", tk.StringVar(value="6")), 6)
            cfg_s1.test_indist_samples_per_space = self.get_int(getattr(self, "v_test_indist_n", tk.StringVar(value="6")), 6)
            cfg_s1.test_opposite_sd = self.get_float(getattr(self, "v_test_opp_sd", tk.StringVar(value="0.15")), 0.15)

            # Optional: restrict test ObjectSpaces (comma-separated). Empty -> all active spaces.
            raw_spaces = str(getattr(self, "v_test_objectspaces", tk.StringVar(value="")).get())
            spaces_list = [s.strip() for s in raw_spaces.split(",") if s.strip() != ""]
            cfg_s1.test_objectspaces = spaces_list if spaces_list else None

            # Optional: test max decision duration override (3-event only). Blank -> None.
            raw_test_max = str(getattr(self, "v_test_max_dec", tk.StringVar(value="")).get()).strip()
            cfg_s1.test_max_dec_dur = float(raw_test_max) if raw_test_max != "" else None

            # Optional run start/end fixations encoded into the design CSV
            if hasattr(self, 'v_add_run_fix'):
                cfg_s1.add_run_fixations = bool(self.v_add_run_fix.get())
            else:
                cfg_s1.add_run_fixations = True
            cfg_s1.run_fix_start_dur = 10.0
            cfg_s1.run_fix_end_dur = 5.0
            
            # Attach timing parameters depending on the selected mode
            if mode == "2event":
                cfg_s1.dec_dur, cfg_s1.dec_fb_dur = self.get_float(self.v_2_dec, 2.0), self.get_float(self.v_2_fb, 1.0)
                cfg_s1.jit_dec_fb_range, cfg_s1.jit_iti_range = parse_range(self.v_2_jit.get(), "Jitter"), parse_range(self.v_2_iti.get(), "ITI")
            else:
                cfg_s1.img_dur, cfg_s1.max_dec_dur, cfg_s1.fb_dur = self.get_float(self.v_3_img, 1.0), self.get_float(self.v_3_max, 3.0), self.get_float(self.v_3_fb, 1.0)
                cfg_s1.jit_isi1_range, cfg_s1.jit_isi2_range, cfg_s1.jit_iti_range = parse_range(self.v_3_isi1.get(), "ISI1"), parse_range(self.v_3_isi2.get(), "ISI2"), parse_range(self.v_3_iti.get(), "ITI")

                # Optional per-run max decision durations (session 1)
                if hasattr(self, "v_3_max_per_run") and bool(self.v_3_max_per_run.get()):
                    cfg_s1.max_dec_dur_per_run = [self.get_float(v, cfg_s1.max_dec_dur) for v in getattr(self, "v_3_max_per_run_s1", [])]
                else:
                    cfg_s1.max_dec_dur_per_run = None

            # Create Session 2 config via dataclasses.replace (S2 differs only by run counts)
            cfg_s2 = dataclasses.replace(
                cfg_s1, 
                n_runs=self.get_int(self.v_runs_s2, 2), 
                trials_per_run=self.get_int(self.v_trials_s2, 20)
            )

            # Apply test-run append target to Session 2
            cfg_s2.include_test_run = bool(include_test and append_to in {"s2", "both"})

            # Optional per-run max decision durations (session 2)
            if mode == "3event" and hasattr(self, "v_3_max_per_run") and bool(self.v_3_max_per_run.get()):
                cfg_s2.max_dec_dur_per_run = [self.get_float(v, cfg_s2.max_dec_dur) for v in getattr(self, "v_3_max_per_run_s2", [])]
            else:
                cfg_s2.max_dec_dur_per_run = None

            base_rng = np.random.default_rng(self.get_int(self.v_seed, 999))
            prefix = self.v_prefix.get()
            os.makedirs(prefix, exist_ok=True)
            
            # Load master label CSV and initialise global tracking structures for balancing
            df_master = load_and_parse_groups(self.v_csv.get())
            df_by_space = df_master.set_index('ObjectSpace', drop=False)
            all_spaces_master = [str(s) for s in df_master['ObjectSpace'].tolist()]
            cond_master = ['Congruent', 'Medium', 'Incongruent']
            global_sc = {s: {c: 0 for c in cond_master} for s in all_spaces_master}
            global_pair = {}

            # Iterate requested number of participants, creating a balanced map and generating rows
            for i in range(self.get_int(self.v_n_subs, 5)):
                sub_id = self.get_int(self.v_start_id, 1) + i
                sub_rng = np.random.default_rng(base_rng.integers(0, 2**32))
                
                # IMPORTANT: Mapping and selection happen once per participant for retention consistency
                selected = select_spaces_rotated(all_spaces_master, sub_id, cfg_s1.n_categories_to_select)

                # New behaviour: if any label cells are empty in the condition columns,
                # those empty options are excluded from assignment. If a space has exactly
                # one non-empty label across conditions, it is locked to that condition.
                allowed_by_space = {}
                for s in selected:
                    if s not in df_by_space.index:
                        raise ValueError(f"ObjectSpace '{s}' not found in label CSV.")
                    row = df_by_space.loc[s]
                    allowed = allowed_conditions_from_label_row(row, cond_master)
                    # If no empties (all three allowed), this will be equal to cond_master
                    allowed_by_space[s] = allowed if allowed else cond_master

                constraints_exist = any(len(allowed_by_space[s]) != len(cond_master) for s in selected)

                if constraints_exist:
                    sc_map = constrained_space_condition_map(
                        sub_rng,
                        selected,
                        cond_master,
                        allowed_by_space,
                        global_sc,
                        global_pair
                    )
                else:
                    # Original behaviour preserved: unconstrained optimiser
                    sc_map = optimise_space_condition_map(sub_rng, selected, cond_master, global_sc, global_pair)
                
                # Generate Session 1 rows and save
                rows_s1 = generate_design_rows(sub_rng, cfg_s1, sub_id, sc_map)
                self._save_session_data(rows_s1, sub_id, 1, prefix, mode)
                
                # Generate Session 2 rows (reusing the same sc_map so assignments remain consistent)
                rows_s2 = generate_design_rows(sub_rng, cfg_s2, sub_id, sc_map)
                self._save_session_data(rows_s2, sub_id, 2, prefix, mode)
                
                self.log(f"Participant {sub_id} complete (Balanced sessions).")

            self.log("All participants generated."); messagebox.showinfo("Success", "Design files generated for 2 sessions.")
        except Exception as e:
            # Surface the error both in the GUI log and a popup dialog
            self.log(f"Error: {e}"); messagebox.showerror("Error", str(e))


# ---- Hotfix: ensure GUI methods are bound on DesignGUI (in case of indentation issues) ----
def _designgui_browse_csv(self):
    """Open a file dialog to select the label CSV and set the entry field."""
    f = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
    if f:
        self.v_csv.set(f)

def _designgui_log(self, txt):
    """Append a line to the GUI log box and scroll to the bottom."""
    try:
        self.log_box.insert("end", txt + "\n")
        self.log_box.see("end")
    except Exception:
        pass

# Bind as methods if missing
if "DesignGUI" in globals():
    if not hasattr(DesignGUI, "browse_csv"):
        DesignGUI.browse_csv = _designgui_browse_csv
    if not hasattr(DesignGUI, "log"):
        DesignGUI.log = _designgui_log


if __name__ == "__main__":
    app = DesignGUI()
    app.mainloop()