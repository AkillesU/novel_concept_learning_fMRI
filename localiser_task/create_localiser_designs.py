#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate design CSVs for the category localiser (1-back), with counterbalancing and image banking.

What this script generates (per your updated requirements):
- A GUI that generates design files for:
    * P participants
    * S localiser-sessions per participant
    * R runs per localiser-session
- Output structure:
    OUTPUT_ROOT/
      sub-001/
        localiser_design_sub-001_ses-01.csv   (contains R runs)
        localiser_design_sub-001_ses-02.csv
        ...
      sub-002/
        ...

Design rules:
- Images are drawn from a parent directory with subfolders named after categories.
- Random sequencing of images within blocks across participants.
- For each participant and category, attempt to use unique images across ALL runs across ALL sessions.
  If reuse is needed, it will never repeat within the same run (so duplicates fall into different runs).
- Block/category order is counterbalanced:
    * within each participant across ALL runs (treat runs across sessions as one long sequence)
    * across participants (regular rotation across subjects)

Design CSV schema (event stream):
- event_type: "fixation" or "trial"
- fix_dur (for fixation rows)
- img_dur, isi_dur (for trial rows)
- category, block_index, trial_in_block, image_path, is_target (for trial rows)
- run, localiser_session, participant
"""

import os
import sys
import csv
import random
from typing import Dict, List, Tuple

# Tkinter GUI (no PsychoPy dependency)
import tkinter as tk
from tkinter import filedialog, messagebox

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# Defaults matched to run_localiser.py
DEFAULT_BASELINE_S = 16.0
DEFAULT_INTERBLOCK_S = 4.0
DEFAULT_IMG_S = 0.300
DEFAULT_ISI_S = 0.500
DEFAULT_TRIALS_PER_BLOCK = 20


# -----------------------------
# Filesystem helpers
# -----------------------------

def list_category_folders(parent_dir: str) -> List[str]:
    cats = []
    for name in sorted(os.listdir(parent_dir)):
        p = os.path.join(parent_dir, name)
        if os.path.isdir(p) and not name.startswith("."):
            cats.append(name)
    return cats


def list_images_in_folder(folder: str) -> List[str]:
    files = []
    for fn in sorted(os.listdir(folder)):
        ext = os.path.splitext(fn)[1].lower()
        if ext in IMG_EXTS:
            files.append(os.path.join(folder, fn))
    return files


# -----------------------------
# Counterbalancing + targets
# -----------------------------

def counterbalanced_order(categories: List[str], participant_num: int, global_run_idx: int) -> List[str]:
    """
    Regular counterbalancing:
      - Rotate base order by (participant_num + global_run_idx) mod n
      - Optional reversal to diversify sequences (participant parity + run parity)
    """
    n = len(categories)
    base = categories[:]  # sorted order is the base
    # Use 0-indexed offsets for clarity
    offset = ((participant_num - 1) + (global_run_idx - 1)) % n
    order = base[offset:] + base[:offset]

    # Simple diversity rule: reverse every other participant on every other run
    if (participant_num % 2 == 0) and (global_run_idx % 2 == 0):
        order = list(reversed(order))

    return order


def choose_target_positions_across_run(
    n_blocks: int,
    trials_per_block: int,
    n_targets_total: int,
    rng: random.Random,
) -> Dict[int, List[int]]:
    """
    Place exactly n_targets_total 1-back targets across the whole run.
    Targets are placed within blocks, on trial indices 1..(trials_per_block-1),
    meaning trial t repeats trial (t-1).
    """
    if n_targets_total < 0:
        raise ValueError("n_targets_per_run must be >= 0")

    candidates: List[Tuple[int, int]] = []
    for b in range(1, n_blocks + 1):
        for t in range(1, trials_per_block):
            candidates.append((b, t))
    rng.shuffle(candidates)

    chosen_by_block: Dict[int, List[int]] = {b: [] for b in range(1, n_blocks + 1)}
    chosen_total = 0

    # Strict non-adjacent targets in same block
    for b, t in candidates:
        if chosen_total >= n_targets_total:
            break
        if any(abs(t - c) <= 1 for c in chosen_by_block[b]):
            continue
        chosen_by_block[b].append(t)
        chosen_total += 1

    # Relax adjacency if needed
    if chosen_total < n_targets_total:
        for b, t in candidates:
            if chosen_total >= n_targets_total:
                break
            if t in chosen_by_block[b]:
                continue
            chosen_by_block[b].append(t)
            chosen_total += 1

    for b in chosen_by_block:
        chosen_by_block[b].sort()

    return chosen_by_block


# -----------------------------
# Image sampling + block building
# -----------------------------

def _sample_unique_images_for_run(
    cat_images: List[str],
    n_needed_unique: int,
    used_in_run: set,
    used_across_participant: set,
    rng: random.Random,
) -> List[str]:
    """
    Prefer images not used previously in this participant (across all runs, across all sessions).
    If not enough, allow reuse but never within the same run (used_in_run).
    """
    # 1) Fresh for participant and not used in run
    pool_fresh = [p for p in cat_images if (p not in used_across_participant) and (p not in used_in_run)]
    rng.shuffle(pool_fresh)

    chosen = []
    for p in pool_fresh:
        if len(chosen) >= n_needed_unique:
            break
        chosen.append(p)

    if len(chosen) >= n_needed_unique:
        return chosen

    # 2) Allow reuse across participant, but still avoid within-run duplicates
    pool_reuse = [p for p in cat_images if p not in used_in_run]
    rng.shuffle(pool_reuse)

    for p in pool_reuse:
        if len(chosen) >= n_needed_unique:
            break
        if p in chosen:
            continue
        chosen.append(p)

    # 3) Last resort: sample with replacement (should only happen if category very small)
    if len(chosen) < n_needed_unique:
        while len(chosen) < n_needed_unique:
            chosen.append(rng.choice(cat_images))

    return chosen


def build_block_sequence(
    cat_images: List[str],
    trials_per_block: int,
    target_positions: List[int],
    rng: random.Random,
    used_in_run: set,
    used_across_participant: set,
) -> Tuple[List[str], List[bool]]:
    """
    Build a trial sequence for a block:
    - No accidental immediate repeats (we add repeats only at target positions)
    - Use unique images within the run when possible
    """
    if len(cat_images) < 2:
        raise RuntimeError("Need at least 2 images per category to support 1-back without trivial repeats.")

    draw_pool = _sample_unique_images_for_run(
        cat_images,
        n_needed_unique=trials_per_block,
        used_in_run=used_in_run,
        used_across_participant=used_across_participant,
        rng=rng,
    )

    base: List[str] = []
    for _ in range(trials_per_block):
        if draw_pool:
            cand = draw_pool.pop(0)
        else:
            cand = rng.choice(cat_images)

        if base and cand == base[-1]:
            # Try to find a non-repeating alternative
            alt = None
            for j in range(len(draw_pool)):
                if draw_pool[j] != base[-1]:
                    alt = draw_pool.pop(j)
                    draw_pool.insert(0, cand)  # put cand back
                    cand = alt
                    break
            if alt is None:
                tries = 0
                while cand == base[-1] and tries < 50:
                    cand = rng.choice(cat_images)
                    tries += 1

        base.append(cand)

    is_target = [False] * trials_per_block
    for t in target_positions:
        if t <= 0 or t >= trials_per_block:
            continue
        base[t] = base[t - 1]
        is_target[t] = True

    # Update usage sets: mark all images that appear in this run (including repeats)
    for p in set(base):
        used_in_run.add(p)
        used_across_participant.add(p)

    return base, is_target


# -----------------------------
# Design generation
# -----------------------------

def generate_design_for_participant_session(
    participant_num: int,
    session_idx: int,
    n_runs: int,
    categories: List[str],
    cat_to_imgs: Dict[str, List[str]],
    n_targets_per_run: int,
    baseline_s: float,
    interblock_s: float,
    img_s: float,
    isi_s: float,
    trials_per_block: int,
    rng_seed_base: int,
    used_across_participant_by_cat: Dict[str, set],
    global_run_offset: int,
) -> List[dict]:
    """
    Generate ONE design file (rows) for a participant's localiser-session.
    The returned rows contain ALL runs (run column indicates run number).
    """
    rows_all: List[dict] = []

    for run_idx in range(1, n_runs + 1):
        global_run_idx = global_run_offset + run_idx  # 1..(S*R) over the full participant
        rng = random.Random(
            (rng_seed_base * 100_000)
            + (participant_num * 10_000)
            + (session_idx * 1_000)
            + run_idx
        )

        order = counterbalanced_order(categories, participant_num, global_run_idx)
        targets_by_block = choose_target_positions_across_run(
            n_blocks=len(order),
            trials_per_block=trials_per_block,
            n_targets_total=n_targets_per_run,
            rng=rng,
        )

        # Pre-run fixation
        rows_all.append({
            "participant": participant_num,
            "localiser_session": session_idx,
            "run": run_idx,
            "event_type": "fixation",
            "fix_dur": float(baseline_s),
            "block_index": "",
            "category": "",
            "trial_in_block": "",
            "image_path": "",
            "is_target": "",
            "img_dur": "",
            "isi_dur": "",
        })

        # Track per-run image usage to prevent duplicates within a run
        used_in_run_by_cat: Dict[str, set] = {c: set() for c in categories}

        for b_i, cat in enumerate(order, start=1):
            imgs = cat_to_imgs[cat]
            target_positions = targets_by_block.get(b_i, [])

            seq, is_target = build_block_sequence(
                imgs,
                trials_per_block=trials_per_block,
                target_positions=target_positions,
                rng=rng,
                used_in_run=used_in_run_by_cat[cat],
                used_across_participant=used_across_participant_by_cat[cat],
            )

            for t_i in range(trials_per_block):
                rows_all.append({
                    "participant": participant_num,
                    "localiser_session": session_idx,
                    "run": run_idx,
                    "event_type": "trial",
                    "fix_dur": "",
                    "block_index": b_i,
                    "category": cat,
                    "trial_in_block": t_i + 1,
                    "image_path": seq[t_i],
                    "is_target": int(is_target[t_i]),
                    "img_dur": float(img_s),
                    "isi_dur": float(isi_s),
                })

            # Inter-block blank
            if b_i < len(order):
                rows_all.append({
                    "participant": participant_num,
                    "localiser_session": session_idx,
                    "run": run_idx,
                    "event_type": "fixation",
                    "fix_dur": float(interblock_s),
                    "block_index": "",
                    "category": "",
                    "trial_in_block": "",
                    "image_path": "",
                    "is_target": "",
                    "img_dur": "",
                    "isi_dur": "",
                })

        # Post-run fixation
        rows_all.append({
            "participant": participant_num,
            "localiser_session": session_idx,
            "run": run_idx,
            "event_type": "fixation",
            "fix_dur": float(baseline_s),
            "block_index": "",
            "category": "",
            "trial_in_block": "",
            "image_path": "",
            "is_target": "",
            "img_dur": "",
            "isi_dur": "",
        })

    return rows_all


def write_csv(path: str, rows: List[dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "participant", "localiser_session", "run",
        "event_type",
        "fix_dur",
        "block_index", "category", "trial_in_block",
        "image_path", "is_target",
        "img_dur", "isi_dur",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# -----------------------------
# GUI
# -----------------------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Generate Localiser Design CSVs")
        self.geometry("860x600")

        self.parent_dir = tk.StringVar(value="images/localiser_images/")
        self.out_root = tk.StringVar(value=os.path.join(os.getcwd(), "designs_localiser"))

        self.n_participants = tk.IntVar(value=10)
        self.start_sub = tk.IntVar(value=1)
        self.sub_zero_pad = tk.IntVar(value=3)

        self.n_targets = tk.IntVar(value=5)
        self.n_runs = tk.IntVar(value=6)
        self.n_localiser_sessions = tk.IntVar(value=4)

        self.baseline_s = tk.DoubleVar(value=DEFAULT_BASELINE_S)
        self.interblock_s = tk.DoubleVar(value=DEFAULT_INTERBLOCK_S)
        self.img_s = tk.DoubleVar(value=DEFAULT_IMG_S)
        self.isi_s = tk.DoubleVar(value=DEFAULT_ISI_S)
        self.trials_per_block = tk.IntVar(value=DEFAULT_TRIALS_PER_BLOCK)
        self.seed = tk.IntVar(value=12345)

        self._build()

    def _browse_parent(self):
        p = filedialog.askdirectory(title="Select image parent directory")
        if p:
            self.parent_dir.set(p)

    def _browse_out(self):
        p = filedialog.askdirectory(title="Select output ROOT directory")
        if p:
            self.out_root.set(p)

    def _build(self):
        pad = {"padx": 8, "pady": 6}
        frm = tk.Frame(self)
        frm.pack(fill="both", expand=True, padx=12, pady=12)

        # Parent dir
        row = tk.Frame(frm); row.pack(fill="x", **pad)
        tk.Label(row, text="Image parent directory (subfolders = categories):", width=48, anchor="w").pack(side="left")
        tk.Entry(row, textvariable=self.parent_dir).pack(side="left", fill="x", expand=True, padx=6)
        tk.Button(row, text="Browse…", command=self._browse_parent).pack(side="left")

        # Output root
        row = tk.Frame(frm); row.pack(fill="x", **pad)
        tk.Label(row, text="Output ROOT directory (sub-XXX folders created here):", width=48, anchor="w").pack(side="left")
        tk.Entry(row, textvariable=self.out_root).pack(side="left", fill="x", expand=True, padx=6)
        tk.Button(row, text="Browse…", command=self._browse_out).pack(side="left")

        # Participants
        sep = tk.Label(frm, text="Participants:", font=("Arial", 12, "bold"), anchor="w")
        sep.pack(fill="x", padx=8, pady=(14, 4))

        row = tk.Frame(frm); row.pack(fill="x", **pad)
        tk.Label(row, text="Number of participants to create:", width=48, anchor="w").pack(side="left")
        tk.Entry(row, textvariable=self.n_participants, width=10).pack(side="left")

        row = tk.Frame(frm); row.pack(fill="x", **pad)
        tk.Label(row, text="Starting subject number (sub-XXX):", width=48, anchor="w").pack(side="left")
        tk.Entry(row, textvariable=self.start_sub, width=10).pack(side="left")
        tk.Label(row, text="Zero pad:", padx=12).pack(side="left")
        tk.Entry(row, textvariable=self.sub_zero_pad, width=5).pack(side="left")

        # Main knobs
        sep = tk.Label(frm, text="Design:", font=("Arial", 12, "bold"), anchor="w")
        sep.pack(fill="x", padx=8, pady=(14, 4))

        row = tk.Frame(frm); row.pack(fill="x", **pad)
        tk.Label(row, text="Localiser-sessions per participant (files per participant):", width=48, anchor="w").pack(side="left")
        tk.Entry(row, textvariable=self.n_localiser_sessions, width=10).pack(side="left")

        row = tk.Frame(frm); row.pack(fill="x", **pad)
        tk.Label(row, text="Runs per localiser-session (runs per file):", width=48, anchor="w").pack(side="left")
        tk.Entry(row, textvariable=self.n_runs, width=10).pack(side="left")

        row = tk.Frame(frm); row.pack(fill="x", **pad)
        tk.Label(row, text="1-back targets per run (total across run):", width=48, anchor="w").pack(side="left")
        tk.Entry(row, textvariable=self.n_targets, width=10).pack(side="left")

        row = tk.Frame(frm); row.pack(fill="x", **pad)
        tk.Label(row, text="Pre/post fixation duration (s):", width=48, anchor="w").pack(side="left")
        tk.Entry(row, textvariable=self.baseline_s, width=10).pack(side="left")

        # Timing extras
        sep = tk.Label(frm, text="Timing (advanced):", font=("Arial", 12, "bold"), anchor="w")
        sep.pack(fill="x", padx=8, pady=(14, 4))

        row = tk.Frame(frm); row.pack(fill="x", **pad)
        tk.Label(row, text="Inter-block blank duration (s):", width=48, anchor="w").pack(side="left")
        tk.Entry(row, textvariable=self.interblock_s, width=10).pack(side="left")

        row = tk.Frame(frm); row.pack(fill="x", **pad)
        tk.Label(row, text="Image duration (s):", width=48, anchor="w").pack(side="left")
        tk.Entry(row, textvariable=self.img_s, width=10).pack(side="left")

        row = tk.Frame(frm); row.pack(fill="x", **pad)
        tk.Label(row, text="ISI (blank) duration (s):", width=48, anchor="w").pack(side="left")
        tk.Entry(row, textvariable=self.isi_s, width=10).pack(side="left")

        row = tk.Frame(frm); row.pack(fill="x", **pad)
        tk.Label(row, text="Trials per block:", width=48, anchor="w").pack(side="left")
        tk.Entry(row, textvariable=self.trials_per_block, width=10).pack(side="left")

        row = tk.Frame(frm); row.pack(fill="x", **pad)
        tk.Label(row, text="Random seed (reproducibility):", width=48, anchor="w").pack(side="left")
        tk.Entry(row, textvariable=self.seed, width=10).pack(side="left")

        # Run button
        btn = tk.Button(frm, text="Generate design CSVs", height=2, command=self._run)
        btn.pack(pady=14)

        self.log = tk.Text(frm, height=10)
        self.log.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.log.insert("end", "Ready.\n")

    def _log(self, s: str):
        self.log.insert("end", s + "\n")
        self.log.see("end")
        self.update_idletasks()

    def _run(self):
        parent_dir = self.parent_dir.get().strip()
        out_root = self.out_root.get().strip()
        if not parent_dir or not os.path.isdir(parent_dir):
            messagebox.showerror("Error", f"Parent directory not found:\n{parent_dir}")
            return

        categories = list_category_folders(parent_dir)
        if not categories:
            messagebox.showerror("Error", "No category subfolders found in parent directory.")
            return

        # Build image bank
        cat_to_imgs: Dict[str, List[str]] = {}
        for c in categories:
            imgs = list_images_in_folder(os.path.join(parent_dir, c))
            if len(imgs) < 2:
                messagebox.showerror("Error", f"Category '{c}' has too few images ({len(imgs)}). Need at least 2.")
                return
            cat_to_imgs[c] = imgs

        n_participants = int(self.n_participants.get())
        start_sub = int(self.start_sub.get())
        zpad = int(self.sub_zero_pad.get())

        n_localiser_sessions = int(self.n_localiser_sessions.get())
        n_runs = int(self.n_runs.get())
        n_targets = int(self.n_targets.get())
        baseline_s = float(self.baseline_s.get())
        interblock_s = float(self.interblock_s.get())
        img_s = float(self.img_s.get())
        isi_s = float(self.isi_s.get())
        trials_per_block = int(self.trials_per_block.get())
        seed = int(self.seed.get())

        if n_participants <= 0:
            messagebox.showerror("Error", "Number of participants must be > 0.")
            return
        if n_localiser_sessions <= 0:
            messagebox.showerror("Error", "Localiser-sessions per participant must be > 0.")
            return
        if n_runs <= 0:
            messagebox.showerror("Error", "Runs per localiser-session must be > 0.")
            return

        self._log(f"Found {len(categories)} categories: {categories}")
        self._log(f"Output root: {out_root}")
        self._log(f"Participants: {n_participants} (sub-{start_sub:0{zpad}d} .. sub-{(start_sub+n_participants-1):0{zpad}d})")
        self._log(f"Localiser-sessions per participant: {n_localiser_sessions}; runs per file: {n_runs}")

        # Generate
        for p_i in range(n_participants):
            participant_num = start_sub + p_i
            sub_label = f"sub-{participant_num:0{zpad}d}"
            sub_dir = os.path.join(out_root, sub_label)

            # Track image usage across ALL sessions/runs within this participant
            used_across_participant_by_cat: Dict[str, set] = {c: set() for c in categories}

            for ses in range(1, n_localiser_sessions + 1):
                global_run_offset = (ses - 1) * n_runs  # so session 1 uses global runs 1..R, session 2 uses R+1..2R, etc.
                rows = generate_design_for_participant_session(
                    participant_num=participant_num,
                    session_idx=ses,
                    n_runs=n_runs,
                    categories=categories,
                    cat_to_imgs=cat_to_imgs,
                    n_targets_per_run=n_targets,
                    baseline_s=baseline_s,
                    interblock_s=interblock_s,
                    img_s=img_s,
                    isi_s=isi_s,
                    trials_per_block=trials_per_block,
                    rng_seed_base=seed,
                    used_across_participant_by_cat=used_across_participant_by_cat,
                    global_run_offset=global_run_offset,
                )
                fn = f"localiser_design_{sub_label}_ses-{ses:02d}.csv"
                out_path = os.path.join(sub_dir, fn)
                write_csv(out_path, rows)
                self._log(f"Wrote {out_path}")

        messagebox.showinfo("Done", f"Generated design CSVs under:\n{out_root}")


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
