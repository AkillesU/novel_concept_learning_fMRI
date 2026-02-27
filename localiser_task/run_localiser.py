#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run the localiser task FROM a design CSV, without importing run_localiser.py.

Requirements (met by this script):
- Independent: does not import run_localiser.
- Keeps the *same* on-screen instruction screens / trigger / between-run dialog behaviour as run_localiser.py.
- Primary design file lookup is automatic from GUI values:
    design_file_parent/{sub-xxx}/localiser_design_sub-xxx_ses_yy.csv
  using GUI fields:
    participant, localiser_session (GUI label), design_file_parent (default: localiser_task/design_files/)
- If the auto-located design file is missing/invalid, falls back to the previous behaviour:
    user browses for a design CSV via a file picker ("design_csv" field).

Design CSV expected columns (minimum):
  event_type

Recommended columns (as produced by generate_localiser_designs_gui.py or compatible):
  localiser_session, run, event_type, fix_dur,
  block_index, category, trial_in_block, image_path, is_target, img_dur, isi_dur

Rows:
  event_type == 'fixation' -> draw fixation for fix_dur seconds (responses ignored for scoring)
  event_type == 'trial'    -> show image for img_dur then blank for isi_dur; scoring matches run_localiser.py

Notes:
- image_path may be absolute or relative. If relative, it is resolved relative to the CSV's directory.
- Supports single-run CSVs or multi-run CSVs via a 'run' column.
- Outputs per-trial CSVs with the SAME columns as run_localiser.py.
"""

import os
import sys
import csv
import time
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from psychopy import visual, core, event, gui
from psychopy.hardware import keyboard

from PIL import Image, ImageOps


# -----------------------------
# Global look & feel / constants (copied from run_localiser.py)
# -----------------------------

# Language for on-screen text (manual toggle)
LANGUAGE = "japanese"  # "english" or "japanese"

DEFAULT_IMG_S = 0.300
DEFAULT_ISI_S = 0.500

# On-screen footprint (units='height'): 1.0 == full screen height.
DISPLAY_IMG_HEIGHT_FRAC = 0.60

# Match experimental_task.py look
BG_COLOR = "lightgrey"
FG_COLOR = "black"

SCANNER_KEYS = ["1", "2", "3", "4"]
PC_KEYS = ["1", "2", "9", "0"]
TRIGGER_KEY = "5"  # scanner trigger
QUIT_KEYS = ["escape"]

# PsychoPy ImageStim texture resolution (power-of-two). Larger helps retina displays.
DEFAULT_TEXRES = 2048

# Pixel resolution used for the image texture (NOT on-screen size).
DEFAULT_IMG_TEX_SIZE = 512

RUN_COL = "run"


# -----------------------------
# Small utilities (copied/adapted from run_localiser.py)
# -----------------------------

def ensure_data_dir(participant: str, session: str) -> str:
    out = os.path.join(os.getcwd(), "localiser_data", f"sub-{participant}", f"ses-{session}")
    os.makedirs(out, exist_ok=True)
    return out


def preprocess_image_to_cache(path: str, target_size: int, cache_dir: str) -> str:
    """
    Preprocess an image for presentation:
    - Apply EXIF orientation (prevents rotated/flipped JPEGs)
    - Convert to standard RGB (handles grayscale/CMYK/RGBA safely)
    - Center-crop to square
    - Resize to (target_size, target_size) pixels
    - Save as an 8-bit PNG in a cache directory

    Returns the cached PNG filepath.
    """
    os.makedirs(cache_dir, exist_ok=True)

    # Cache key: path + mtime + size + target_size
    try:
        st = os.stat(path)
        key_str = f"{path}|{st.st_mtime_ns}|{st.st_size}|{int(target_size)}"
    except OSError:
        key_str = f"{path}|{int(target_size)}"

    h = hashlib.sha1(key_str.encode("utf-8")).hexdigest()[:16]
    out_path = os.path.join(cache_dir, f"{h}_{int(target_size)}.png")
    if os.path.exists(out_path):
        return out_path

    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("RGB")

    w, h0 = img.size
    min_dim = min(w, h0)
    left = (w - min_dim) // 2
    top = (h0 - min_dim) // 2
    img = img.crop((left, top, left + min_dim, top + min_dim))

    resample = Image.LANCZOS if hasattr(Image, "LANCZOS") else Image.BICUBIC
    img = img.resize((int(target_size), int(target_size)), resample=resample)

    img.save(out_path, format="PNG")
    return out_path


def safe_wait_until(end_time_abs: float,
                    kb: keyboard.Keyboard,
                    allowed_keys: List[str],
                    start_time_abs: Optional[float] = None,
                    run_once: Optional[callable] = None) -> Tuple[Optional[str], Optional[float]]:
    """
    Wait until the absolute time `end_time_abs` (core.getTime() seconds) while collecting
    the FIRST keypress in allowed_keys.

    Returns (key, rt) where rt is relative to `start_time_abs` (defaults to time at function entry),
    or (None, None).

    Using absolute end times reduces cumulative drift/overshoot across a run.
    """
    if start_time_abs is None:
        start_time_abs = core.getTime()

    got_key = None
    got_rt = None
    did_run_once = False

    while True:
        now = core.getTime()
        if now >= float(end_time_abs):
            break

        if (run_once is not None) and (not did_run_once):
            try:
                run_once()
            finally:
                did_run_once = True

        if event.getKeys(QUIT_KEYS):
            core.quit()

        keys = kb.getKeys(keyList=allowed_keys + QUIT_KEYS, waitRelease=False, clear=False)
        if keys and got_key is None:
            k = keys[0]
            if k.name in QUIT_KEYS:
                core.quit()
            got_key = k.name
            try:
                got_rt = float(now - start_time_abs)
            except Exception:
                got_rt = None

        core.wait(0.0005)

    return got_key, got_rt


def safe_wait(duration: float, kb: keyboard.Keyboard, allowed_keys: List[str]) -> Tuple[Optional[str], Optional[float]]:
    """
    Backwards-compatible relative wait.
    Implemented via safe_wait_until for improved timing stability.
    """
    t0 = core.getTime()
    return safe_wait_until(t0 + float(duration), kb, allowed_keys=allowed_keys, start_time_abs=t0)


def between_run_dialog(next_run_idx: int, n_runs: int, trigger_key: str = TRIGGER_KEY):
    """Native GUI dialog shown between runs (outside fullscreen).

    Matches the *English* wording used in experimental_task.py, and provides a
    Japanese equivalent when LANGUAGE == "japanese".
    """
    if LANGUAGE == "japanese":
        title = f"施行開始 {next_run_idx}"
    else:
        title = f"Start run {next_run_idx}"

    dlg = gui.Dlg(title=title)

    if LANGUAGE == "japanese":
        dlg.addText("「OK」をクリックすると全画面で開きます。")
        dlg.addText(f"次にトリガー画面で '{trigger_key}'を押して、施行 {next_run_idx} を開始します。")
    else:
        dlg.addText("Click OK to open fullscreen.")
        dlg.addText(f"Then press '{trigger_key}' on the trigger screen to begin run {next_run_idx}.")

    dlg.show()
    if not dlg.OK:
        core.quit()


def show_text_screen(win: visual.Window, text: str, kb: keyboard.Keyboard, advance_keys: List[str]):
    """Instruction / pause screen styled like experimental_task.py."""
    stim = visual.TextStim(
        win,
        text=text,
        height=0.04,
        wrapWidth=0.90,
        color=FG_COLOR,
        pos=(0, 0.10),
    )
    if LANGUAGE == "japanese":
        cont_text = "続行するにはいずれかのボタンを押してください"
    else:
        cont_text = "Press any button to continue"

    cont = visual.TextStim(
        win,
        text=cont_text,
        height=0.03,
        color="grey",
        pos=(0, -0.40),
    )

    while True:
        if event.getKeys(QUIT_KEYS):
            core.quit()
        stim.draw()
        cont.draw()
        win.flip()

        keys = kb.getKeys(keyList=advance_keys + ["space", "return"] + QUIT_KEYS, waitRelease=False, clear=True)
        if keys:
            if keys[0].name in QUIT_KEYS:
                core.quit()
            return
        core.wait(0.01)


def wait_for_trigger(
    win: visual.Window,
    kb: keyboard.Keyboard,
    trigger_key: str = TRIGGER_KEY,
    allow_skip_keys: Optional[List[str]] = None,
    text: str = None,
) -> str:
    """Block until we receive the scanner trigger key (default '5')."""
    allow_skip_keys = allow_skip_keys or []

    if text is None:
        if LANGUAGE == "japanese":
            text = "MRI装置の起動を待っています"
        else:
            text = "Waiting for scanner to start"

    stim = visual.TextStim(
        win,
        text=text,
        height=0.04,
        wrapWidth=0.90,
        color=FG_COLOR,
        pos=(0, 0.05),
    )

    kb.clearEvents()
    while True:
        if event.getKeys(QUIT_KEYS):
            core.quit()
        stim.draw()
        win.flip()

        keys = kb.getKeys(
            keyList=[trigger_key] + QUIT_KEYS,
            waitRelease=False,
            clear=True,
        )
        if keys:
            k = keys[0].name
            if k in QUIT_KEYS:
                core.quit()
            return k


def create_window(fullscreen: bool, screen_index: int) -> visual.Window:
    """Create a PsychoPy window that matches run_localiser.py styling."""
    if fullscreen:
        win = visual.Window(
            fullscr=True,
            screen=screen_index,
            color=BG_COLOR,
            units="height",
            allowGUI=False,
            useRetina=True,
        )
    else:
        win = visual.Window(
            size=[1280, 720],
            fullscr=False,
            screen=screen_index,
            color=BG_COLOR,
            units="height",
            allowGUI=True,
            useRetina=True,
        )
    return win


# -----------------------------
# Design loading helpers
# -----------------------------

def _resolve_image_path(csv_dir: str, p: str) -> str:
    p = str(p)
    if not p:
        return p
    if os.path.isabs(p) and os.path.exists(p):
        return p
    cand = os.path.join(csv_dir, p)
    if os.path.exists(cand):
        return cand
    return p


def _read_design_rows(design_csv: str) -> List[Dict[str, str]]:
    with open(design_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError("Design CSV has no header row.")
        rows = [dict(r) for r in reader]
    if not rows:
        raise RuntimeError("Design CSV contains no rows.")
    if "event_type" not in rows[0]:
        raise RuntimeError("Design CSV missing required column: event_type")
    return rows


def _detect_unique_runs(rows: List[Dict[str, str]]) -> Optional[List[str]]:
    if not rows:
        return None
    if RUN_COL not in rows[0]:
        return None
    seen = []
    seen_set = set()
    for r in rows:
        v = r.get(RUN_COL, "")
        if v not in seen_set:
            seen_set.add(v)
            seen.append(v)
    return seen


def _filter_rows_for_run(rows: List[Dict[str, str]], run_value: Optional[str]) -> List[Dict[str, str]]:
    if run_value is None:
        return rows
    return [r for r in rows if r.get(RUN_COL, "") == run_value]


def _try_parse_float(x, default: float = 0.0) -> float:
    try:
        if x is None:
            return float(default)
        s = str(x).strip()
        if s == "":
            return float(default)
        return float(s)
    except Exception:
        return float(default)


def _try_parse_int(x, default: int = 0) -> int:
    try:
        if x is None:
            return int(default)
        s = str(x).strip()
        if s == "":
            return int(default)
        return int(float(s))
    except Exception:
        return int(default)


def _canonicalize_sub(participant: str) -> str:
    p = str(participant).strip()
    if p.lower().startswith("sub-"):
        p = p[4:]
    # pad digits to 3 if numeric and short (common BIDS convention)
    if p.isdigit() and len(p) < 3:
        p = p.zfill(3)
    return p


def _canonicalize_ses(session: str) -> str:
    s = str(session).strip()
    if s.lower().startswith("ses-"):
        s = s[4:]
    if s.isdigit() and len(s) < 2:
        s = s.zfill(2)
    return s


def _auto_design_csv_path(design_file_parent: str, participant: str, localiser_session: str) -> str:
    parent = design_file_parent or ""
    parent = parent.strip() if isinstance(parent, str) else str(parent)
    if not parent:
        parent = "localiser_task/design_files/"
    sub = _canonicalize_sub(participant)
    ses = _canonicalize_ses(localiser_session)

    # Try a small set of likely filename variants for robustness.
    candidates = []
    sub_dir = os.path.join(parent, f"sub-{sub}")
    candidates.append(os.path.join(sub_dir, f"localiser_design_sub-{sub}_ses_{ses}.csv"))
    candidates.append(os.path.join(sub_dir, f"localiser_design_sub-{sub}_ses-{ses}.csv"))
    candidates.append(os.path.join(sub_dir, f"localiser_design_sub-{sub}_ses_{localiser_session}.csv"))
    candidates.append(os.path.join(sub_dir, f"localiser_design_sub-{sub}_ses-{localiser_session}.csv"))

    for c in candidates:
        if os.path.isfile(c):
            return c

    # Return the primary expected path (even if it doesn't exist) for error messaging.
    return candidates[0]


# -----------------------------
# Params / GUI
# -----------------------------

@dataclass
class Params:
    language: str
    participant: str
    localiser_session: str
    design_file_parent: str
    design_csv: str  # optional manual override / fallback
    button_mode: str  # "scanner" or "pc"
    img_tex_size: int
    fullscreen: bool
    screen_index: int
    show_run_summary: bool = False


def get_params_from_gui() -> Params:
    # Keep the *front-end* aligned with run_localiser.py, but:
    # - session -> localiser_session
    # - parent_dir -> design_file_parent (for design file lookup)
    info = {
        "language": "japanese",
        "participant": "",
        "localiser_session": "01",
"design_file_parent": "localiser_task/design_files/",
        "design_csv": "",  # optional manual override / fallback
        "button_mode": "scanner",
        "img_tex_size": DEFAULT_IMG_TEX_SIZE,
        "fullscreen": True,
        "screen_index": 0,
        "show_run_summary": False,
    }

    dlg = gui.DlgFromDict(
        dictionary=info,
        title="fMRI Localiser Setup",
        order=[
            "language",
            "participant", "localiser_session",
"design_file_parent",
            "design_csv",
            "button_mode",
            "img_tex_size",
            "fullscreen",
            "screen_index",
            "show_run_summary",
        ],
        tip={
            "localiser_session": "Session label used to find the design file (and saved as ses-XX in output).",
            "design_file_parent": "Directory under which design files exist (default: localiser_task/design_files/).",
            "design_csv": "Optional: manually select a design CSV. Used if auto-lookup fails or if you want to override.",
            "button_mode": "scanner: keys 1,2,3,4 | pc: keys 1,2,9,0",
            "img_tex_size": "Pixel resolution of loaded images after crop/resize (display size is fixed at ~60% screen height).",
            "screen_index": "Which monitor to use (0 = primary).",
            "show_run_summary": "When enabled, show hit/false-alarm summary at the end of each run (default: off).",
        }
    )
    if not dlg.OK:
        sys.exit(0)

    # Set global language exactly like run_localiser.py
    global LANGUAGE
    lang = str(info.get("language", "english")).strip().lower()
    LANGUAGE = "japanese" if lang.startswith("jap") else "english"

    return Params(
        language=str(info.get("language", "english")),
        participant=str(info.get("participant", "001")),
        localiser_session=str(info.get("localiser_session", "01")),
        design_file_parent=str(info.get("design_file_parent", "localiser_task/design_files/")),
        design_csv=str(info.get("design_csv", "")).strip(),
        button_mode=str(info.get("button_mode", "scanner")).strip().lower(),
        img_tex_size=int(info.get("img_tex_size", DEFAULT_IMG_TEX_SIZE)),
        fullscreen=bool(info.get("fullscreen", True)),
        screen_index=int(info.get("screen_index", 0)),
        show_run_summary=bool(info.get("show_run_summary", False)),
    )


# -----------------------------
# Main task logic (design-driven)
# -----------------------------

def run_localiser_from_design(params: Params):
    resp_keys = SCANNER_KEYS if params.button_mode.lower() == "scanner" else PC_KEYS
    use_scanner_trigger = (params.button_mode.lower() == "scanner")
    start_keys = resp_keys + ([TRIGGER_KEY] if use_scanner_trigger else [])

    # Primary: auto-locate design CSV from participant/session + design_file_parent.
    auto_csv = _auto_design_csv_path(params.design_file_parent, params.participant, params.localiser_session)
    design_csv = auto_csv

    # If a manual design_csv is provided, prefer it (explicit override).
    if params.design_csv:
        design_csv = params.design_csv

    rows = None
    design_load_error = None

    # Try the chosen design_csv first (manual override if set, otherwise auto path).
    try:
        if not design_csv or not os.path.isfile(design_csv):
            raise FileNotFoundError(f"Design CSV not found: {design_csv}")
        rows = _read_design_rows(design_csv)
    except Exception as e:
        design_load_error = e
        rows = None

    # Backup: if auto (or override) failed, fall back to browse picker (previous behaviour).
    if rows is None:
        picked = gui.fileOpenDlg(
            prompt=(
                "Auto design CSV not found/invalid."
                f"\nTried: {design_csv}"
                f"\nError: {design_load_error}"
                "\n\nSelect localiser design CSV"
            ),
            allowed="CSV files (*.csv);;All files (*.*)"
        )
        if not picked:
            raise FileNotFoundError(f"No valid design CSV selected. Last error: {design_load_error}")
        design_csv = picked[0]
        rows = _read_design_rows(design_csv)

    run_values = _detect_unique_runs(rows)
    if run_values is None:
        runs = [(None, rows)]
    else:
        runs = [(rv, _filter_rows_for_run(rows, rv)) for rv in run_values]
    n_runs = len(runs)

    out_dir = ensure_data_dir(params.participant, params.localiser_session)
    cache_dir = os.path.join(out_dir, "_stim_cache_png")

    win = create_window(fullscreen=params.fullscreen, screen_index=params.screen_index)
    kb = keyboard.Keyboard()

    img_stim = visual.ImageStim(
        win,
        image=None,
        pos=(0, 0),
        size=(DISPLAY_IMG_HEIGHT_FRAC, DISPLAY_IMG_HEIGHT_FRAC),
        interpolate=True,
        texRes=DEFAULT_TEXRES,
    )
    fixation = visual.TextStim(win, text="+", height=0.10, color=FG_COLOR)

    img_cache: Dict[Tuple[str, int], str] = {}

    # Instructions (COPIED EXACTLY from run_localiser.py)
    if LANGUAGE == "japanese":
        instr1 = (
            "画像閲覧\n\n"
            "さまざまな画像のシーケンスが表示されます"
        )
        instr2 = (
            "同じ画像が2回続けて表示されることもあります。\n"
            "繰り返し画像（連続）が表示されたら、\n任意の応答ボタンを押します。"
        )
        instr3 = (
            "迅速かつ正確に応答するようにしてください。\n応答が遅いと思われても心配しないでください。\n"
            "シーケンス間の固定クロスに注目してください\n"
            "開始するには任意のボタンを押してください。"
        )
    else:
        instr1 = (
            "Image viewing\n\n"
            "You will see different sequences of images"
        )
        instr2 = (
            "Sometimes the SAME picture will appear twice in a row.\n"
            "When you see a repeat image (back-to-back), press ANY response button."
        )
        instr3 = (
            "Try to respond quickly and accurately\nbut don't worry if your response seems slow.\n\n"
            "Keep your eyes on the fixation cross between sequences.\n\n"
            "Press any button to start."
        )

    show_text_screen(win, instr1, kb, advance_keys=resp_keys)
    show_text_screen(win, instr2, kb, advance_keys=resp_keys)
    show_text_screen(win, instr3, kb, advance_keys=start_keys)

    csv_dir = os.path.dirname(os.path.abspath(design_csv))
    all_run_summaries = []

    try:
        for run_idx, (run_value, run_rows) in enumerate(runs, start=1):
            ts = time.strftime("%Y%m%d-%H%M%S")
            out_csv_path = os.path.join(
                out_dir,
                f"localiser_sub-{params.participant}_ses-{params.localiser_session}_run-{run_idx:02d}_{ts}.csv",
            )

            fieldnames = [
                "participant", "session", "run",
                "block_index", "category",
                "trial_in_block", "image_path",
                "is_target",
                "resp_key", "resp_rt_s", "correct",
                "trial_onset_s", "img_onset_s", "isi_onset_s",
                "img_tex_size", "img_display_height_frac",
            ]

            with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

                # Trigger screen (same layout as run_localiser.py)
                if LANGUAGE == "japanese":
                    trigger_text = f"施行: {run_idx}/{n_runs}\nMRI装置の起動を待っています"
                else:
                    trigger_text = f"Run: {run_idx}/{n_runs}\nWaiting for scanner to start…"

                wait_for_trigger(
                    win,
                    kb,
                    trigger_key=TRIGGER_KEY,
                    allow_skip_keys=resp_keys,
                    text=trigger_text,
                )

                kb.clearEvents()
                run_start = core.getTime()
                kb.clock.reset()

                hits = misses = fas = crs = 0


                def _next_trial_row(start_idx: int) -> Optional[Dict[str, str]]:
                    """Return the next row with event_type == 'trial' after start_idx, or None."""
                    for j in range(start_idx + 1, len(run_rows)):
                        rj = run_rows[j]
                        etj = str(rj.get("event_type", "")).strip().lower()
                        if etj == "trial":
                            return rj
                    return None

                for i, row in enumerate(run_rows):
                    et = str(row.get("event_type", "")).strip().lower()

                    
                    if et == "fixation":
                        dur = _try_parse_float(row.get("fix_dur", 0), default=0.0)

                        fixation.draw()
                        fix_flip_t = win.flip()

                        # Preload the next trial's image during long fixation windows so that the next image
                        # flip is not delayed by preprocessing/disk IO.
                        nxt = _next_trial_row(i)

                        def _preload_next():
                            if not nxt:
                                return
                            nxt_img_raw = nxt.get("image_path", "")
                            nxt_img = _resolve_image_path(csv_dir, nxt_img_raw)
                            if not nxt_img or (not os.path.exists(nxt_img)):
                                return
                            ck = (nxt_img, int(params.img_tex_size))
                            if ck not in img_cache:
                                img_cache[ck] = preprocess_image_to_cache(nxt_img, int(params.img_tex_size), cache_dir)

                        safe_wait_until(
                            fix_flip_t + float(dur),
                            kb,
                            allowed_keys=resp_keys,
                            start_time_abs=fix_flip_t,
                            run_once=_preload_next,
                        )
                        continue

                    if et != "trial":
                        # Unknown event -> ignore safely
                        continue

                    img_path_raw = row.get("image_path", "")
                    img_path = _resolve_image_path(csv_dir, img_path_raw)
                    if not img_path or not os.path.exists(img_path):
                        raise FileNotFoundError(f"Image not found: {img_path} (from {img_path_raw})")

                    img_dur = _try_parse_float(row.get("img_dur", DEFAULT_IMG_S), default=DEFAULT_IMG_S)
                    isi_dur = _try_parse_float(row.get("isi_dur", DEFAULT_ISI_S), default=DEFAULT_ISI_S)

                    block_index = row.get("block_index", "")
                    category = row.get("category", "")
                    trial_in_block = row.get("trial_in_block", "")

                    target = _try_parse_int(row.get("is_target", 0), default=0) == 1

                    trial_onset = core.getTime() - run_start

                    cache_key = (img_path, int(params.img_tex_size))
                    if cache_key not in img_cache:
                        img_cache[cache_key] = preprocess_image_to_cache(img_path, int(params.img_tex_size), cache_dir)

                    img_stim.image = img_cache[cache_key]

                    
                    # IMAGE: draw then flip; use flip timestamp as true onset.
                    img_stim.draw()
                    img_flip_t = win.flip()
                    img_onset = img_flip_t - run_start

                    kb.clearEvents()
                    img_end_t = img_flip_t + float(img_dur)
                    key_img, rt_img = safe_wait_until(img_end_t, kb, allowed_keys=resp_keys, start_time_abs=img_flip_t)

                    # ISI (blank): flip at image offset then wait isi_dur from that flip.
                    isi_flip_t = win.flip()
                    isi_onset = isi_flip_t - run_start

                    # Preload the NEXT trial's image during the ISI so the next image flip isn't delayed.
                    nxt = _next_trial_row(i)

                    def _preload_next():
                        if not nxt:
                            return
                        nxt_img_raw = nxt.get("image_path", "")
                        nxt_img = _resolve_image_path(csv_dir, nxt_img_raw)
                        if not nxt_img or (not os.path.exists(nxt_img)):
                            return
                        ck = (nxt_img, int(params.img_tex_size))
                        if ck not in img_cache:
                            img_cache[ck] = preprocess_image_to_cache(nxt_img, int(params.img_tex_size), cache_dir)

                    kb.clearEvents()
                    isi_end_t = isi_flip_t + float(isi_dur)
                    key_isi, rt_isi = safe_wait_until(isi_end_t, kb, allowed_keys=resp_keys, start_time_abs=isi_flip_t, run_once=_preload_next)

                    resp_key = resp_rt = None
                    if key_img is not None:
                        resp_key, resp_rt = key_img, rt_img
                    elif key_isi is not None:
                        resp_key, resp_rt = key_isi, rt_isi

                    if target and resp_key is not None:
                        correct = 1; hits += 1
                    elif target and resp_key is None:
                        correct = 0; misses += 1
                    elif (not target) and resp_key is not None:
                        correct = 0; fas += 1
                    else:
                        correct = 1; crs += 1

                    writer.writerow({
                        "participant": params.participant,
                        "session": params.localiser_session,
                        "run": run_idx,
                        "block_index": block_index,
                        "category": category,
                        "trial_in_block": trial_in_block,
                        "image_path": img_path,
                        "is_target": int(target),
                        "resp_key": resp_key if resp_key is not None else "",
                        "resp_rt_s": f"{resp_rt:.4f}" if resp_rt is not None else "",
                        "correct": int(correct),
                        "trial_onset_s": f"{trial_onset:.4f}",
                        "img_onset_s": f"{img_onset:.4f}",
                        "isi_onset_s": f"{isi_onset:.4f}",
                        "img_tex_size": int(params.img_tex_size),
                        "img_display_height_frac": float(DISPLAY_IMG_HEIGHT_FRAC),
                    })

                total_targets = hits + misses
                total_nontargets = fas + crs
                hit_rate = hits / total_targets if total_targets else 0.0
                fa_rate = fas / total_nontargets if total_nontargets else 0.0

                all_run_summaries.append({
                    "run": run_idx,
                    "hits": hits, "misses": misses, "false_alarms": fas, "correct_rejects": crs,
                    "hit_rate": hit_rate, "fa_rate": fa_rate,
                    "csv": os.path.basename(out_csv_path),
                    "design_csv": os.path.basename(design_csv),
                })

            # Between-run dialog: close fullscreen -> GUI -> reopen fullscreen (matches run_localiser.py flow)
            if run_idx < n_runs:
                try:
                    win.close()
                except Exception:
                    pass

                between_run_dialog(next_run_idx=run_idx + 1, n_runs=n_runs, trigger_key=TRIGGER_KEY)

                win = create_window(fullscreen=params.fullscreen, screen_index=params.screen_index)
                kb = keyboard.Keyboard()

                # Rebind window-specific stimuli
                img_stim = visual.ImageStim(
                    win,
                    image=None,
                    pos=(0, 0),
                    size=(DISPLAY_IMG_HEIGHT_FRAC, DISPLAY_IMG_HEIGHT_FRAC),
                    interpolate=True,
                    texRes=DEFAULT_TEXRES,
                )
                fixation = visual.TextStim(win, text="+", height=0.10, color=FG_COLOR)

        final_text = "終了した" if LANGUAGE == "japanese" else "All runs complete."
        show_text_screen(win, final_text, kb, advance_keys=resp_keys)

    finally:
        try:
            win.close()
        except Exception:
            pass
        core.quit()


if __name__ == "__main__":
    params = get_params_from_gui()
    run_localiser_from_design(params)
