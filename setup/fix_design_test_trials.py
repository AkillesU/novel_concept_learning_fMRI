from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from typing import List, Dict, Any, Optional

FIXATION_EVENT_VALUE = "fixation"


def is_fixation_row(row: pd.Series) -> bool:
    ev = str(row.get("event_type", "")).strip().lower()
    cond = str(row.get("condition", "")).strip().lower()
    return ev == FIXATION_EVENT_VALUE or cond == "fixation"


def object_value(row: pd.Series):
    val = row.get("ObjectSpace", None)
    if pd.isna(val):
        return None
    try:
        f = float(val)
        if f.is_integer():
            return int(f)
        return f
    except Exception:
        return val


def split_runs_by_fixation(df: pd.DataFrame) -> List[List[int]]:
    """
    Return trial-row index lists for each run, using fixation rows as separators.
    A run is the contiguous set of non-fixation rows between fixation rows.
    """
    runs = []
    current = []
    for idx, row in df.iterrows():
        if is_fixation_row(row):
            if current:
                runs.append(current)
                current = []
        else:
            current.append(idx)
    if current:
        runs.append(current)
    return runs


def adjacent_repeat_positions(df: pd.DataFrame, run_indices: List[int]) -> List[int]:
    """
    Return positions i (within run_indices) where trial i-1 and i have same object identity.
    """
    reps = []
    prev_obj = None
    for pos, idx in enumerate(run_indices):
        obj = object_value(df.loc[idx])
        if pos > 0 and obj is not None and obj == prev_obj:
            reps.append(pos)
        prev_obj = obj
    return reps


def get_swappable_columns(df: pd.DataFrame) -> List[str]:
    """
    Columns whose values move when swapping trials.
    Keep identifiers / run labels / row-order anchors fixed.
    """
    immovable = {
        "trial_id",
        "run_id",
    }
    return [c for c in df.columns if c not in immovable]


def would_be_clean_after_swap(objects: List[Any], i: int, j: int) -> bool:
    """
    Check whether swapping positions i and j removes adjacent duplicates at all affected adjacencies.
    """
    n = len(objects)
    test = list(objects)
    test[i], test[j] = test[j], test[i]

    affected = set()
    for k in (i, j):
        affected.update([k - 1, k, k + 1])
    affected = {p for p in affected if 1 <= p < n}

    for p in affected:
        if test[p] is not None and test[p] == test[p - 1]:
            return False
    return True


def choose_swap_target(objects: List[Any], repeat_pos: int) -> Optional[int]:
    """
    For a repeat at repeat_pos (same as repeat_pos-1), find the nearest later position
    whose swap resolves the local repeat without introducing new adjacent repeats.
    Fall back to earlier positions if needed.
    """
    n = len(objects)

    for j in range(repeat_pos + 1, n):
        if would_be_clean_after_swap(objects, repeat_pos, j):
            return j

    for j in range(0, repeat_pos - 1):
        if would_be_clean_after_swap(objects, repeat_pos, j):
            return j

    return None


def swap_trial_contents(df: pd.DataFrame, idx_a: int, idx_b: int, swappable_cols: List[str]) -> None:
    tmp = df.loc[idx_a, swappable_cols].copy()
    df.loc[idx_a, swappable_cols] = df.loc[idx_b, swappable_cols].values
    df.loc[idx_b, swappable_cols] = tmp.values


def reindex_trial_id_by_row_order(df: pd.DataFrame) -> None:
    """
    Reassign trial_id so that non-fixation rows are numbered in their current top-to-bottom order.
    Fixation rows are left untouched when they are blank/NaN; if a fixation row has a value,
    it is cleared to keep trial_id aligned with actual trials only.
    """
    if "trial_id" not in df.columns:
        return

    next_id = 1
    new_vals = []
    for _, row in df.iterrows():
        if is_fixation_row(row):
            new_vals.append(pd.NA)
        else:
            new_vals.append(next_id)
            next_id += 1

    df["trial_id"] = pd.array(new_vals, dtype="Int64")



def fix_run_repeats(
    df: pd.DataFrame,
    run_indices: List[int],
    swappable_cols: List[str],
    file_label: str = ""
) -> List[Dict[str, Any]]:
    """
    Iteratively resolve adjacent same-object repeats within one run by safe in-run swaps.
    """
    changes: List[Dict[str, Any]] = []

    while True:
        repeats = adjacent_repeat_positions(df, run_indices)
        if not repeats:
            break

        changed_this_pass = False
        for repeat_pos in repeats:
            objects = [object_value(df.loc[idx]) for idx in run_indices]
            if repeat_pos >= len(objects) or objects[repeat_pos] != objects[repeat_pos - 1]:
                continue

            j = choose_swap_target(objects, repeat_pos)
            if j is None:
                changes.append({
                    "type": "unresolved_repeat",
                    "file": file_label,
                    "run_id": df.loc[run_indices[repeat_pos], "run_id"] if "run_id" in df.columns else None,
                    "repeat_trial_id_prev": df.loc[run_indices[repeat_pos - 1], "trial_id"] if "trial_id" in df.columns else None,
                    "repeat_trial_id_curr": df.loc[run_indices[repeat_pos], "trial_id"] if "trial_id" in df.columns else None,
                    "object": objects[repeat_pos],
                })
                continue

            idx_a = run_indices[repeat_pos]
            idx_b = run_indices[j]

            before_a = df.loc[idx_a].to_dict()
            before_b = df.loc[idx_b].to_dict()

            swap_trial_contents(df, idx_a, idx_b, swappable_cols)

            changes.append({
                "type": "swap",
                "file": file_label,
                "run_id": df.loc[idx_a, "run_id"] if "run_id" in df.columns else None,
                "pos_a_in_run": repeat_pos,
                "pos_b_in_run": j,
                "trial_id_a_before_reindex": before_a.get("trial_id"),
                "trial_id_b_before_reindex": before_b.get("trial_id"),
                "object_a_before": before_a.get("ObjectSpace"),
                "object_b_before": before_b.get("ObjectSpace"),
                "image_a_before": before_a.get("image_file"),
                "image_b_before": before_b.get("image_file"),
                "object_a_after": df.loc[idx_a, "ObjectSpace"],
                "object_b_after": df.loc[idx_b, "ObjectSpace"],
                "image_a_after": df.loc[idx_a, "image_file"],
                "image_b_after": df.loc[idx_b, "image_file"],
            })
            changed_this_pass = True
            break

        if not changed_this_pass:
            break

    return changes



def process_file(
    in_path: Path,
    out_path: Path,
    only_test_runs: bool = False,
) -> Dict[str, Any]:
    df = pd.read_csv(in_path)
    original_df = df.copy(deep=True)

    runs = split_runs_by_fixation(df)
    swappable_cols = get_swappable_columns(df)

    all_changes: List[Dict[str, Any]] = []
    for run_indices in runs:
        if not run_indices:
            continue

        if only_test_runs and "fb_dur" in df.columns:
            fb = pd.to_numeric(df.loc[run_indices, "fb_dur"], errors="coerce").fillna(0)
            if not (fb == 0).all():
                continue

        all_changes.extend(fix_run_repeats(df, run_indices, swappable_cols, file_label=str(in_path)))

    if not df.equals(original_df):
        reindex_trial_id_by_row_order(df)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    changed = not df.equals(original_df)

    final_runs = split_runs_by_fixation(df)
    remaining = []
    for run_indices in final_runs:
        if only_test_runs and "fb_dur" in df.columns:
            fb = pd.to_numeric(df.loc[run_indices, "fb_dur"], errors="coerce").fillna(0)
            if not (fb == 0).all():
                continue

        reps = adjacent_repeat_positions(df, run_indices)
        if reps:
            remaining.append({
                "run_id": df.loc[run_indices[0], "run_id"] if "run_id" in df.columns and run_indices else None,
                "n_remaining_repeats": len(reps),
                "positions": reps,
            })

    return {
        "input_file": str(in_path),
        "output_file": str(out_path),
        "changed": changed,
        "n_changes": sum(1 for c in all_changes if c["type"] == "swap"),
        "n_unresolved": sum(1 for c in all_changes if c["type"] == "unresolved_repeat"),
        "changes": all_changes,
        "remaining_repeats": remaining,
        "trial_id_reindexed": changed and "trial_id" in df.columns,
    }



def iter_csv_files(input_root: Path):
    for p in input_root.rglob("*.csv"):
        if p.is_file():
            yield p



def main():
    parser = argparse.ArgumentParser(
        description="Resolve back-to-back same-object trials by safe in-run swaps."
    )
    parser.add_argument("input_dir", type=Path, help="Root directory containing CSV files.")
    parser.add_argument("output_dir", type=Path, help="Root directory for rewritten CSV files.")
    parser.add_argument(
        "--only-test-runs",
        action="store_true",
        help="Only modify runs where all fb_dur values are 0."
    )
    parser.add_argument(
        "--report-name",
        default="swap_report.txt",
        help="Name of the text report written into output_dir."
    )
    args = parser.parse_args()

    input_root = args.input_dir.resolve()
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    reports = []
    for in_file in iter_csv_files(input_root):
        rel = in_file.relative_to(input_root)
        out_file = output_root / rel
        report = process_file(in_file, out_file, only_test_runs=args.only_test_runs)
        reports.append(report)

    report_path = output_root / args.report_name
    with open(report_path, "w", encoding="utf-8") as f:
        for r in reports:
            f.write(f"FILE: {r['input_file']}\n")
            f.write(f"OUT : {r['output_file']}\n")
            f.write(f"CHANGED: {r['changed']}\n")
            f.write(f"SWAPS: {r['n_changes']}\n")
            f.write(f"UNRESOLVED: {r['n_unresolved']}\n")
            f.write(f"TRIAL_ID_REINDEXED: {r['trial_id_reindexed']}\n")
            if r["changes"]:
                f.write("DETAILS:\n")
                for c in r["changes"]:
                    if c["type"] == "swap":
                        f.write(
                            f"  swap run {c['run_id']} | "
                            f"row-pos {c['pos_a_in_run']} trial_id {c['trial_id_a_before_reindex']} "
                            f"(obj {c['object_a_before']}, img {c['image_a_before']}) "
                            f"<-> row-pos {c['pos_b_in_run']} trial_id {c['trial_id_b_before_reindex']} "
                            f"(obj {c['object_b_before']}, img {c['image_b_before']})\n"
                        )
                    else:
                        f.write(
                            f"  unresolved repeat run {c['run_id']} | "
                            f"trial_id {c['repeat_trial_id_prev']} and {c['repeat_trial_id_curr']} "
                            f"(obj {c['object']})\n"
                        )
            if r["remaining_repeats"]:
                f.write("REMAINING REPEATS AFTER PROCESSING:\n")
                for rem in r["remaining_repeats"]:
                    f.write(
                        f"  run {rem['run_id']} | remaining={rem['n_remaining_repeats']} | positions={rem['positions']}\n"
                    )
            f.write("\n" + "-" * 80 + "\n\n")

    print(f"Processed {len(reports)} file(s).")
    print(f"Report written to: {report_path}")


if __name__ == "__main__":
    main()
