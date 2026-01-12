import tkinter as tk
from tkinter import ttk, messagebox
from dataclasses import dataclass, field
from typing import List


@dataclass
class RunConfig:
    name: str
    num_trial_events: int = 3
    event_durations: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    isi_durations: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    num_trials: int = 10
    num_conditions: int = 2
    num_categories_per_condition: int = 2
    break_after_run: float = 30.0  # seconds; break *after* this run


class ExperimentDesignerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Experimental Design Visualiser")
        self.geometry("1200x700")

        self.runs: List[RunConfig] = []
        self.current_run_index: int = 0

        self._build_ui()
        # start with one default run
        self._add_default_run()
        self._select_run_in_listbox(0)
        self.update_all()

    # ---------- UI BUILDING ----------
    def _build_ui(self):
        # Main layout: left controls, right visualisation
        main_pane = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True)

        controls_frame = ttk.Frame(main_pane, padding=5)
        visuals_frame = ttk.Frame(main_pane, padding=5)
        main_pane.add(controls_frame, weight=1)
        main_pane.add(visuals_frame, weight=3)

        # --- Left: Run list + parameters ---
        self._build_run_list_frame(controls_frame)
        self._build_run_params_frame(controls_frame)

        # --- Right: summary + canvases ---
        self._build_visuals_frame(visuals_frame)

    def _build_run_list_frame(self, parent):
        frame = ttk.LabelFrame(parent, text="Runs", padding=5)
        frame.pack(fill=tk.X, expand=False, pady=(0, 5))

        list_frame = ttk.Frame(frame)
        list_frame.pack(fill=tk.X, expand=False)

        self.run_listbox = tk.Listbox(list_frame, height=5, exportselection=False)
        self.run_listbox.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.run_listbox.bind("<<ListboxSelect>>", self.on_run_selected)

        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.run_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.run_listbox.config(yscrollcommand=scrollbar.set)

        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, expand=False, pady=(5, 0))

        ttk.Button(btn_frame, text="Add run", command=self.add_run).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Clone run", command=self.clone_run).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Delete run", command=self.delete_run).pack(side=tk.LEFT, padx=2)

    def _build_run_params_frame(self, parent):
        frame = ttk.LabelFrame(parent, text="Run parameters", padding=5)
        frame.pack(fill=tk.BOTH, expand=True)

        # Tk variables
        self.var_run_name = tk.StringVar()
        self.var_num_events = tk.StringVar(value="3")
        self.var_event_durs = tk.StringVar(value="1, 1, 1")
        self.var_isi_durs = tk.StringVar(value="1, 1, 1")
        self.var_num_trials = tk.StringVar(value="10")
        self.var_num_conditions = tk.StringVar(value="2")
        self.var_num_categories = tk.StringVar(value="2")
        self.var_break_after = tk.StringVar(value="30")

        row = 0
        ttk.Label(frame, text="Run name:").grid(row=row, column=0, sticky="w")
        name_entry = ttk.Entry(frame, textvariable=self.var_run_name, width=18)
        name_entry.grid(row=row, column=1, sticky="ew", columnspan=2)
        row += 1

        # live-update run list name as user types
        self.var_run_name.trace_add("write", self.on_run_name_change)

        ttk.Label(frame, text="# trial events:").grid(row=row, column=0, sticky="w")
        self.spn_num_events = tk.Spinbox(
            frame,
            from_=1,
            to=10,
            textvariable=self.var_num_events,
            width=5,
            command=self.on_num_events_change,
        )
        self.spn_num_events.grid(row=row, column=1, sticky="w")
        row += 1

        ttk.Label(frame, text="Event lengths (s):").grid(row=row, column=0, sticky="nw")
        ttk.Entry(frame, textvariable=self.var_event_durs).grid(
            row=row, column=1, columnspan=2, sticky="ew"
        )
        row += 1
        ttk.Label(frame, text="comma-separated, one per event").grid(
            row=row, column=1, columnspan=2, sticky="w"
        )
        row += 1

        ttk.Label(frame, text="ISIs after each event (s):").grid(row=row, column=0, sticky="nw")
        ttk.Entry(frame, textvariable=self.var_isi_durs).grid(
            row=row, column=1, columnspan=2, sticky="ew"
        )
        row += 1
        ttk.Label(frame, text="same length as events (last = post-trial)").grid(
            row=row, column=1, columnspan=2, sticky="w"
        )
        row += 1

        ttk.Label(frame, text="# trials in this run:").grid(row=row, column=0, sticky="w")
        tk.Spinbox(frame, from_=1, to=1000, textvariable=self.var_num_trials, width=7).grid(
            row=row, column=1, sticky="w"
        )
        row += 1

        ttk.Label(frame, text="# conditions:").grid(row=row, column=0, sticky="w")
        tk.Spinbox(frame, from_=1, to=20, textvariable=self.var_num_conditions, width=7).grid(
            row=row, column=1, sticky="w"
        )
        row += 1

        ttk.Label(frame, text="# categories / condition:").grid(row=row, column=0, sticky="w")
        tk.Spinbox(frame, from_=1, to=20, textvariable=self.var_num_categories, width=7).grid(
            row=row, column=1, sticky="w"
        )
        row += 1

        ttk.Label(frame, text="Break after run (s):").grid(row=row, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.var_break_after, width=7).grid(
            row=row, column=1, sticky="w"
        )
        row += 1

        # Buttons
        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=row, column=0, columnspan=3, pady=(8, 0), sticky="ew")
        ttk.Button(btn_frame, text="Apply to run", command=self.save_current_run).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(btn_frame, text="Update visualisation", command=self.update_all).pack(
            side=tk.LEFT, padx=2
        )

        for col in range(3):
            frame.grid_columnconfigure(col, weight=1)

    def _build_visuals_frame(self, parent):
        # Summary label
        summary_frame = ttk.LabelFrame(parent, text="Summary", padding=5)
        summary_frame.pack(fill=tk.X, expand=False)

        self.lbl_summary = ttk.Label(summary_frame, text="", justify="left")
        self.lbl_summary.pack(anchor="w")

        # Trial structure canvas
        trial_frame = ttk.LabelFrame(parent, text="Trial-level structure (current run)", padding=5)
        trial_frame.pack(fill=tk.X, expand=False, pady=(5, 0))

        self.trial_canvas = tk.Canvas(trial_frame, height=120, bg="white")
        self.trial_canvas.pack(fill=tk.X, expand=True)

        # Experiment structure canvas
        exp_frame = ttk.LabelFrame(parent, text="Experiment structure (all runs)", padding=5)
        exp_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 0))

        self.exp_canvas = tk.Canvas(exp_frame, height=260, bg="white")
        self.exp_canvas.pack(fill=tk.BOTH, expand=True)

        # Redraw on resize
        self.trial_canvas.bind("<Configure>", lambda event: self.draw_trial_structure())
        self.exp_canvas.bind("<Configure>", lambda event: self.draw_experiment_structure())

    # ---------- Run list helpers ----------
    def _add_default_run(self):
        idx = len(self.runs)
        run = RunConfig(name=f"Run {idx+1}")
        self._ensure_lengths(run)
        self.runs.append(run)
        self.run_listbox.insert(tk.END, run.name)

    def add_run(self):
        # Add a new run with default trial-level parameters
        self._add_default_run()
        self._select_run_in_listbox(len(self.runs) - 1)
        self.update_all()

    def clone_run(self):
        if not self.runs:
            return
        idx = self.current_run_index if 0 <= self.current_run_index < len(self.runs) else 0
        src = self.runs[idx]
        clone = RunConfig(
            name=f"{src.name} (clone)",
            num_trial_events=src.num_trial_events,
            event_durations=list(src.event_durations),
            isi_durations=list(src.isi_durations),
            num_trials=src.num_trials,
            num_conditions=src.num_conditions,
            num_categories_per_condition=src.num_categories_per_condition,
            break_after_run=src.break_after_run,
        )
        self.runs.append(clone)
        self.run_listbox.insert(tk.END, clone.name)
        self._select_run_in_listbox(len(self.runs) - 1)
        self.update_all()

    def delete_run(self):
        if not self.runs:
            return
        idx = self.current_run_index
        if not (0 <= idx < len(self.runs)):
            return
        del self.runs[idx]
        self.run_listbox.delete(idx)
        if self.runs:
            new_idx = max(0, idx - 1)
            self._select_run_in_listbox(new_idx)
        else:
            self.current_run_index = 0
            self.clear_run_controls()
        self.update_all()

    def _select_run_in_listbox(self, index: int):
        if not self.runs:
            return
        self.run_listbox.selection_clear(0, tk.END)
        self.run_listbox.selection_set(index)
        self.run_listbox.activate(index)
        self.current_run_index = index
        self.load_run_into_controls(self.runs[index])

    def on_run_selected(self, event):
        if not self.runs:
            return
        selection = self.run_listbox.curselection()
        if not selection:
            return
        idx = selection[0]
        self.current_run_index = idx
        self.load_run_into_controls(self.runs[idx])
        self.update_all()

    # ---------- Run <-> controls ----------
    def clear_run_controls(self):
        self.var_run_name.set("")
        self.var_num_events.set("1")
        self.var_event_durs.set("1")
        self.var_isi_durs.set("1")
        self.var_num_trials.set("1")
        self.var_num_conditions.set("1")
        self.var_num_categories.set("1")
        self.var_break_after.set("0")

    def load_run_into_controls(self, run: RunConfig):
        self._ensure_lengths(run)
        self.var_run_name.set(run.name)
        self.var_num_events.set(str(run.num_trial_events))
        self.var_event_durs.set(", ".join(f"{d:g}" for d in run.event_durations))
        self.var_isi_durs.set(", ".join(f"{d:g}" for d in run.isi_durations))
        self.var_num_trials.set(str(run.num_trials))
        self.var_num_conditions.set(str(run.num_conditions))
        self.var_num_categories.set(str(run.num_categories_per_condition))
        self.var_break_after.set(f"{run.break_after_run:g}")

    def on_num_events_change(self):
        """When #events changes, adjust the length of the duration lists in the UI strings."""
        try:
            n = int(self.var_num_events.get())
        except ValueError:
            return
        n = max(1, n)

        # Helper to resize comma-separated list in a StringVar
        def resize_list(var: tk.StringVar, default_val: float = 1.0):
            text = var.get()
            parts = [p.strip() for p in text.split(",") if p.strip()]
            vals = []
            for p in parts:
                try:
                    vals.append(float(p))
                except ValueError:
                    pass
            if len(vals) < n:
                vals += [default_val] * (n - len(vals))
            elif len(vals) > n:
                vals = vals[:n]
            var.set(", ".join(f"{v:g}" for v in vals))

        resize_list(self.var_event_durs, 1.0)
        resize_list(self.var_isi_durs, 1.0)

    def _parse_float_list(self, text: str, expected_len: int, default_val: float = 1.0):
        if not text.strip():
            return [default_val] * expected_len
        parts = [p.strip() for p in text.split(",") if p.strip()]
        vals = []
        for p in parts:
            vals.append(float(p))  # may raise ValueError
        if len(vals) < expected_len:
            vals += [default_val] * (expected_len - len(vals))
        elif len(vals) > expected_len:
            vals = vals[:expected_len]
        return vals

    def save_current_run(self):
        if not self.runs:
            return
        idx = self.current_run_index
        if not (0 <= idx < len(self.runs)):
            return

        try:
            name = self.var_run_name.get().strip() or f"Run {idx+1}"
            num_events = int(self.var_num_events.get())
            num_events = max(1, num_events)

            event_durs = self._parse_float_list(self.var_event_durs.get(), num_events, 1.0)
            isi_durs = self._parse_float_list(self.var_isi_durs.get(), num_events, 1.0)

            num_trials = max(1, int(self.var_num_trials.get()))
            num_conditions = max(1, int(self.var_num_conditions.get()))
            num_categories = max(1, int(self.var_num_categories.get()))
            break_after = float(self.var_break_after.get())
            break_after = max(0.0, break_after)
        except ValueError:
            messagebox.showerror(
                "Invalid input",
                "Please ensure all numeric fields contain valid numbers (use comma-separated lists for durations).",
            )
            return

        run = self.runs[idx]
        run.name = name
        run.num_trial_events = num_events
        run.event_durations = event_durs
        run.isi_durations = isi_durs
        run.num_trials = num_trials
        run.num_conditions = num_conditions
        run.num_categories_per_condition = num_categories
        run.break_after_run = break_after
        self._ensure_lengths(run)

        # update listbox label
        self.run_listbox.delete(idx)
        self.run_listbox.insert(idx, run.name)
        self.run_listbox.selection_set(idx)
        self.run_listbox.activate(idx)

    def _ensure_lengths(self, run: RunConfig):
        """Ensure event_durations and isi_durations match num_trial_events."""
        n = max(1, run.num_trial_events)
        if len(run.event_durations) < n:
            run.event_durations += [1.0] * (n - len(run.event_durations))
        elif len(run.event_durations) > n:
            run.event_durations = run.event_durations[:n]
        if len(run.isi_durations) < n:
            run.isi_durations += [1.0] * (n - len(run.isi_durations))
        elif len(run.isi_durations) > n:
            run.isi_durations = run.isi_durations[:n]

    def on_run_name_change(self, *args):
        """Live-update the run name in the listbox as the user edits it."""
        if not self.runs:
            return
        idx = self.current_run_index
        if not (0 <= idx < len(self.runs)):
            return
        name = self.var_run_name.get().strip() or f"Run {idx+1}"
        # Update listbox display only (actual RunConfig updated on save_current_run)
        self.run_listbox.delete(idx)
        self.run_listbox.insert(idx, name)
        self.run_listbox.selection_set(idx)
        self.run_listbox.activate(idx)

    # ---------- Calculations ----------
    def compute_trial_length(self, run: RunConfig) -> float:
        return sum(run.event_durations) + sum(run.isi_durations)

    def format_time(self, seconds: float) -> str:
        total = int(round(seconds))
        minutes = total // 60
        sec = total % 60
        return f"{minutes} min {sec} s"

    def update_summary(self):
        if not self.runs:
            self.lbl_summary.config(text="No runs defined.")
            return

        # Ensure current run is saved before summarising
        self.save_current_run()

        total_time = 0.0
        total_trials = 0
        per_run_lines = []

        for i, run in enumerate(self.runs):
            trial_len = self.compute_trial_length(run)
            run_active = trial_len * run.num_trials
            # Breaks apply after run i, except after last one
            break_len = run.break_after_run if i < len(self.runs) - 1 else 0.0
            run_total = run_active + break_len
            total_time += run_total
            total_trials += run.num_trials

            cats = run.num_conditions * run.num_categories_per_condition
            trials_per_cat = run.num_trials / cats if cats > 0 else 0
            per_run_lines.append(
                f"{run.name}: trial={trial_len:.1f}s, run active={run_active:.1f}s, "
                f"break={break_len:.1f}s, total={run_total:.1f}s, "
                f"trials={run.num_trials}, trials/category≈{trials_per_cat:.2f}"
            )

        # Check if conditions/categories consistent across runs
        first = self.runs[0]
        base_cats = first.num_conditions * first.num_categories_per_condition
        consistent = True
        for run in self.runs[1:]:
            if (
                run.num_conditions != first.num_conditions
                or run.num_categories_per_condition != first.num_categories_per_condition
            ):
                consistent = False
                break

        total_trials_per_cat_str = "n/a (conditions/categories differ between runs)"
        if consistent and base_cats > 0:
            total_trials_per_cat = total_trials / base_cats
            total_trials_per_cat_str = (
                f"≈{total_trials_per_cat:.2f} (per category across experiment)"
            )

        summary_lines = [
            f"Total experiment duration: {self.format_time(total_time)} ({total_time:.1f} s)",
            f"Total number of runs: {len(self.runs)}",
            f"Total number of trials: {total_trials}",
            f"Trials per category overall: {total_trials_per_cat_str}",
            "",
            "Per-run details:",
            *per_run_lines,
        ]
        self.lbl_summary.config(text="\n".join(summary_lines))

    # ---------- Drawing ----------
    def draw_trial_structure(self):
        canvas = self.trial_canvas
        canvas.delete("all")
        if not self.runs:
            return

        run = self.runs[self.current_run_index]
        self._ensure_lengths(run)
        trial_len = self.compute_trial_length(run)
        if trial_len <= 0:
            return

        width = max(canvas.winfo_width(), 400)
        height = canvas.winfo_height()
        margin = 10
        usable_width = width - 2 * margin
        x = margin
        y1, y2 = 30, height - 20

        scale = usable_width / trial_len

        # Font size dynamically based on height
        font_size = max(6, min(10, int((y2 - y1) / 5)))
        trial_font = ("TkDefaultFont", font_size)

        # Draw axis baseline
        canvas.create_line(margin, y2 + 5, width - margin, y2 + 5)

        # Draw events and ISIs in sequence
        for i in range(run.num_trial_events):
            # Event block
            ev_dur = run.event_durations[i]
            ev_w = ev_dur * scale
            canvas.create_rectangle(x, y1, x + ev_w, y2, fill="#add8e6", outline="black")
            # adapt label based on width
            if ev_w >= 40:
                label_text = f"E{i+1}\n{ev_dur:g}s"
            else:
                label_text = f"E{i+1}"
            canvas.create_text(
                x + ev_w / 2,
                (y1 + y2) / 2,
                text=label_text,
                font=trial_font,
                width=max(ev_w - 4, 20),
                justify="center",
            )
            x += ev_w

            # ISI block after this event
            isi_dur = run.isi_durations[i]
            if isi_dur > 0:
                isi_w = isi_dur * scale
                canvas.create_rectangle(x, y1, x + isi_w, y2, fill="#dddddd", outline="black")
                base_label = "ITI" if i == run.num_trial_events - 1 else f"ISI{i+1}"
                if isi_w >= 40:
                    label = f"{base_label}\n{isi_dur:g}s"
                else:
                    label = base_label
                canvas.create_text(
                    x + isi_w / 2,
                    (y1 + y2) / 2,
                    text=label,
                    font=trial_font,
                    width=max(isi_w - 4, 20),
                    justify="center",
                )
                x += isi_w

        canvas.create_text(
            width / 2,
            10,
            text=f"One trial (length {trial_len:.1f} s)",
            font=("TkDefaultFont", 9, "bold"),
        )

    def draw_experiment_structure(self):
        canvas = self.exp_canvas
        canvas.delete("all")
        if not self.runs:
            return

        width = max(canvas.winfo_width(), 400)
        height = canvas.winfo_height()
        margin = 10
        usable_width = width - 2 * margin
        y1, y2 = 50, height - 60

        # Font size based on height
        font_size = max(7, min(11, int((y2 - y1) / 5)))
        run_font = ("TkDefaultFont", font_size)

        # Compute total duration for scaling (active + breaks, except after last)
        total_time = 0.0
        active_times = []
        break_times = []
        for i, run in enumerate(self.runs):
            trial_len = self.compute_trial_length(run)
            run_active = trial_len * run.num_trials
            active_times.append(run_active)
            if i < len(self.runs) - 1:
                break_len = max(0.0, run.break_after_run)
            else:
                break_len = 0.0
            break_times.append(break_len)
            total_time += run_active + break_len

        if total_time <= 0:
            return

        scale = usable_width / total_time
        x = margin
        for i, run in enumerate(self.runs):
            run_active = active_times[i]
            run_w = max(run_active * scale, 1.0)  # avoid zero-width blocks

            # Run block
            canvas.create_rectangle(x, y1, x + run_w, y2, fill="#c3f7c3", outline="black")
            trial_len = self.compute_trial_length(run)

            # Adapt label content based on width
            if run_w < 80:
                text = f"{run.name}\n{run.num_trials} tr"
            elif run_w < 140:
                text = f"{run.name}\n{run.num_trials} tr\ntrial={trial_len:.1f}s"
            else:
                text = (
                    f"{run.name}\n"
                    f"{run.num_trials} trials\n"
                    f"trial={trial_len:.1f}s\n"
                    f"run active={run_active:.1f}s"
                )

            canvas.create_text(
                x + run_w / 2,
                (y1 + y2) / 2,
                text=text,
                font=run_font,
                width=max(run_w - 6, 30),
                justify="center",
            )
            x += run_w

            # Break block (except after last run)
            break_len = break_times[i]
            if break_len > 0:
                break_w = max(break_len * scale, 1.0)
                canvas.create_rectangle(
                    x, y1, x + break_w, y2, fill="#f0e68c", outline="black"
                )
                if break_w < 40:
                    btext = "Break"
                else:
                    btext = f"Break\n{break_len:.1f}s"
                canvas.create_text(
                    x + break_w / 2,
                    (y1 + y2) / 2,
                    text=btext,
                    font=run_font,
                    width=max(break_w - 6, 30),
                    justify="center",
                )
                x += break_w

        canvas.create_text(
            width / 2,
            20,
            text="Experiment timeline (runs + breaks)",
            font=("TkDefaultFont", 10, "bold"),
        )

    # ---------- Top-level update ----------
    def update_all(self):
        if self.runs:
            self.save_current_run()
        self.update_summary()
        self.draw_trial_structure()
        self.draw_experiment_structure()


if __name__ == "__main__":
    app = ExperimentDesignerApp()
    app.mainloop()
