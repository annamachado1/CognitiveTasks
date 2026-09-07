#!/usr/bin/env python3

import traceback
import sys

try:
    from psychopy import visual, core, event, gui, monitors
    from psychopy.hardware import keyboard
    import numpy as np
    import random
    import csv
    import os
    import json
    import uuid
    import platform
    from datetime import datetime

    P = dict(
        dotsize=12,
        dotedgesize=1,
        linewidth=3,
        cuesize=25,
        cuewidth=5,
        displayhoriz=900,
        displayvert=900,
        buffer=5,

        numdotgroups=2,
        numtargets=1,
        # Minimum prey-chaser separation required before a chase may start,
        # in multiples of dotsize.
        chasedistthresh_mult=8,
        # False: separation must exceed the threshold on each axis independently.
        # True: Euclidean separation.
        use_euclidean_onset=False,

        # --- Block design ---
        block_defs=[
            dict(label="2 predators", predator_dots=2, numdotspergroup=1),
            dict(label="4 predators", predator_dots=4, numdotspergroup=2),
            dict(label="6 predators", predator_dots=6, numdotspergroup=3),
        ],
        block_duration_secs=120.0,
        showcue=False,

        frmrate_expected=60,
        cueduration=1.0,
        ITI=0.5,
        # Dead time after a trial starts or a press lands, during which further
        # presses are logged but not scored.
        blackout_secs=0.200,

        # --- Trial structure ---
        # A block is a sequence of trials. A trial ends as soon as a response is
        # recorded; every dot is then re-placed and the display restarts.
        end_trial_on_hit=True,
        end_trial_on_false_alarm=True,
        end_trial_on_catch=True,
        trial_blank_secs=0.5,

        # --- Feedback on correct detection ---
        flash_col=(0, 255, 0),
        flash_duration=1,      # display freezes while the caught chaser flashes

        stepsize=1,
        targetstepsize=0.5,
        dirchfreq=1.0,
        chasechxfreq1=0.5035,
        chasechxfreq2=0.5035,
        wiggle=0.2,
        targwiggle=0.03,
        # Prey is caught when the chaser is within this many dotsizes on both axes.
        catch_dist_mult=1.5,

        targ_col=(255, 0, 0),
        grp1_col=(0, 0, 0),
        grp2_col=(0, 0, 0),
        bkgd_col=(160, 160, 160),
        frame_col=(0, 0, 0),
        cue_col=(0, 225, 0),

        feedbacktype=0,
        # Keeps practice from reusing the real run's stimulus sequence.
        practice_seed_offset=100000,
    )

    DATA_DIR = "data"

    # Every event row carries all fields; blanks where not applicable.
    EVENT_FIELDS = [
        "datetime", "run_id", "subject", "practice", "seed",
        "block_order", "block_label", "predator_dots",
        "trial_num", "event_num", "event_type", "key_name", "rt_ms",
        "trial_start_time", "chase_onset_time", "button_press_time",
        "chase_finish_time",
        "chaser_idx", "chaser_group",
        "onset_dx", "onset_dy", "onset_dist", "onset_dist_from_trial_start",
        "prey_x", "prey_y", "chaser_x", "chaser_y",
    ]

    BLOCK_FIELDS = [
        "datetime", "run_id", "subject", "practice", "seed",
        "block_order", "block_label", "predator_dots",
        "block_duration_s", "n_frames", "n_trials", "n_chases",
        "n_hits", "n_misses", "n_false_alarms", "n_unresolved",
        "n_discarded", "n_other_keys", "time_at_risk_s",
        "frmrate_measured", "n_dropped_frames", "completed",
    ]


    class AbortExperiment(Exception):
        """Raised on escape so data buffers flush instead of hard-quitting."""
        pass


    def rgb255_to_psychopy(rgb):
        return [(c / 127.5) - 1 for c in rgb]


    def blank_event(event_type):
        ev = {k: "" for k in EVENT_FIELDS}
        ev["event_type"] = event_type
        return ev


    def get_subject_info():
        info = {
            "Subject Number": 1,
            "Practice": False,
            "Screen width (cm)": 52.0,
            "Viewing distance (cm)": 57.0,
        }
        dlg = gui.DlgFromDict(info, title="Chasing / Pursuit Task")
        if not dlg.OK:
            core.quit()
        return (int(info["Subject Number"]), bool(info["Practice"]),
                float(info["Screen width (cm)"]), float(info["Viewing distance (cm)"]))


    def make_writers(subj, practice):
        """Open the event and block CSVs in append mode, writing headers if new."""
        os.makedirs(DATA_DIR, exist_ok=True)
        tag = "practice" if practice else "main"

        ev_name = os.path.join(DATA_DIR, f"Chase1_s{subj}_{tag}_events.csv")
        ev_new = not os.path.exists(ev_name)
        ev_file = open(ev_name, "a", newline="")
        ev_writer = csv.writer(ev_file)
        if ev_new:
            ev_writer.writerow(EVENT_FIELDS)

        bk_name = os.path.join(DATA_DIR, f"Chase1_s{subj}_{tag}_blocks.csv")
        bk_new = not os.path.exists(bk_name)
        bk_file = open(bk_name, "a", newline="")
        bk_writer = csv.writer(bk_file)
        if bk_new:
            bk_writer.writerow(BLOCK_FIELDS)

        return ev_file, ev_writer, bk_file, bk_writer


    def write_sidecar(subj, practice, seed, run_id, mon_info, frmrate):
        """Dump parameters, seed, monitor geometry and versions next to the data."""
        os.makedirs(DATA_DIR, exist_ok=True)
        payload = dict(
            run_id=run_id,
            subject=subj,
            practice=practice,
            seed=int(seed),
            started=datetime.now().isoformat(timespec="milliseconds"),
            monitor=mon_info,
            frmrate_measured=frmrate,
            versions=dict(
                python=sys.version.split()[0],
                numpy=np.__version__,
                platform=platform.platform(),
            ),
            params={k: (list(v) if isinstance(v, tuple) else v) for k, v in P.items()},
        )
        try:
            from psychopy import __version__ as psychopy_version
            payload["versions"]["psychopy"] = psychopy_version
        except Exception:
            pass
        fname = os.path.join(DATA_DIR, f"Chase1_s{subj}_{run_id}_params.json")
        with open(fname, "w") as f:
            json.dump(payload, f, indent=2, default=str)


    class DotField:
        """Positions, headings and chase state for one block's dots.

        Index 0 is always the prey; indices 1..n are the predators, split into
        two selection pools (group 1 and group 2) that the chase lottery draws
        from. Coordinates are pixels, centred on the display.
        """

        def __init__(self, p, numdotspergroup, rng):
            self.p = p
            self.numdotspergroup = numdotspergroup
            self.numdots = p["numtargets"] + numdotspergroup * p["numdotgroups"]
            self.rng = rng

            self.half_w = p["displayhoriz"] / 2
            self.half_h = p["displayvert"] / 2
            self.chasedistthresh = p["dotsize"] * p["chasedistthresh_mult"]

            # Block-level counters, kept across trials.
            self.chase_onset_time = []
            self.n_caught = 0

            self.reset_positions()

        def reset_positions(self):
            """Scatter every dot onto a fresh 5x5 grid cell and clear chase state.

            Called at the start of each trial, so a response resets the whole
            display rather than only the dot that was responded to.
            """
            p, rng = self.p, self.rng
            cells = list(range(1, 26))
            rng.shuffle(cells)
            self.x = np.zeros(self.numdots)
            self.y = np.zeros(self.numdots)
            for i in range(self.numdots):
                cell = cells[i]
                jitter = rng.random() * 40 - 20
                self.x[i] = ((np.ceil(cell / 5) - 1) * (p["displayhoriz"] / 6)) \
                    - self.half_w + (p["displayhoriz"] / 6) + jitter
                self.y[i] = ((cell % 5) * (p["displayvert"] / 6)) \
                    - self.half_h + (p["displayvert"] / 6) + jitter

            self.direction = (rng.random(self.numdots) * 2 - 1) * np.pi
            self.chasing_now = np.zeros(self.numdots, dtype=bool)
            self.flash_until = {}
            self.active_chaser = None
            self.active_chase = None

            # Trial-start positions, used to measure how far a chaser had
            # travelled by the time it began chasing.
            self.start_x = self.x.copy()
            self.start_y = self.y.copy()

        def group1_idx(self):
            return list(range(1, self.numdotspergroup + 1))

        def group2_idx(self):
            return list(range(self.numdotspergroup + 1, self.numdots))

        def group_of(self, j):
            return 1 if j in self.group1_idx() else 2

        def geometry(self, j):
            """Signed offsets and distance of the prey relative to dot j."""
            dx = float(self.x[0] - self.x[j])
            dy = float(self.y[0] - self.y[j])
            return dict(
                dx=dx, dy=dy,
                dist=float(np.hypot(dx, dy)),
                prey_x=float(self.x[0]), prey_y=float(self.y[0]),
                chaser_x=float(self.x[j]), chaser_y=float(self.y[j]),
            )

        def dist_from_trial_start(self, j):
            """How far dot j has travelled from where this trial placed it."""
            return float(np.hypot(self.x[j] - self.start_x[j],
                                  self.y[j] - self.start_y[j]))

        def _wall_bounce(self, j):
            """Reflect dot j off a wall, with jitter so bounces are not mirror-exact."""
            p = self.p
            d = self.direction[j]
            wfx = self.half_w - p["buffer"] - abs(self.x[j])
            wfy = self.half_h - p["buffer"] - abs(self.y[j])
            near_vert = abs(wfx) < p["dotsize"]
            near_horiz = abs(wfy) < p["dotsize"]

            if near_vert:
                if self.x[j] < 0:
                    d = np.pi / 2 + (self.rng.random() - 0.5) * np.pi / 2
                else:
                    d = -np.pi / 2 + (self.rng.random() - 0.5) * np.pi / 2
                if near_horiz:
                    if self.x[j] < 0 and self.y[j] > 0:
                        d = 3 * np.pi / 4 + (self.rng.random() - 0.5) * np.pi / 4
                    elif self.x[j] > 0 and self.y[j] > 0:
                        d = 5 * np.pi / 4 + (self.rng.random() - 0.5) * np.pi / 4
                    elif self.x[j] < 0 and self.y[j] < 0:
                        d = np.pi / 4 + (self.rng.random() - 0.5) * np.pi / 4
                    elif self.x[j] > 0 and self.y[j] < 0:
                        d = 7 * np.pi / 4 + (self.rng.random() - 0.5) * np.pi / 4
            if near_horiz:
                if self.y[j] > 0:
                    d = np.pi + (self.rng.random() - 0.5) * np.pi / 2
                else:
                    d = 0 + (self.rng.random() - 0.5) * np.pi / 2
            return d

        def update_frame(self, p):
            """Advance every dot one frame.

            Order: wall bounces, then the chase lottery (only when nothing is
            already chasing), then pursuit or catch for the active chaser, then
            random heading wiggle, then the position step.

            Returns a list of catch records for this frame; a catch ends the
            trial, so the field is not re-placed here.
            """
            caught_this_frame = []

            for j in range(self.numdots):
                self.direction[j] = self._wall_bounce(j)

            if not self.chasing_now.any():
                chg1 = round(p["chasechxfreq1"] * self.rng.random())
                if chg1 == 1 and self.numdotspergroup > 0:
                    cand = self.rng.choice(self.group1_idx())
                    self._maybe_start_chase(cand)
                if not self.chasing_now.any():
                    chg2 = round(p["chasechxfreq2"] * self.rng.random())
                    if chg2 == 1 and self.numdotspergroup > 0:
                        cand = self.rng.choice(self.group2_idx())
                        self._maybe_start_chase(cand)

            for j in range(1, self.numdots):
                if not self.chasing_now[j]:
                    continue
                dx = self.x[0] - self.x[j]
                dy = self.y[0] - self.y[j]
                caught = (abs(dx) <= p["dotsize"] * p["catch_dist_mult"]
                          and abs(dy) <= p["dotsize"] * p["catch_dist_mult"])
                if caught:
                    self.n_caught += 1
                    caught_this_frame.append(dict(
                        chaser_idx=int(j),
                        chase=self.active_chase,
                        geom=self.geometry(j),
                    ))
                    self.chasing_now[:] = False
                    self.active_chaser = None
                    self.active_chase = None
                else:
                    self.direction[j] = np.arctan2(dx, dy)

            self.direction[0] += p["targwiggle"] * self._rand_dir_delta()
            for j in range(1, self.numdots):
                if not self.chasing_now[j]:
                    self.direction[j] += p["wiggle"] * self._rand_dir_delta()
            self.direction = np.mod(self.direction, 2 * np.pi)

            dx_step = p["stepsize"] * np.sin(self.direction)
            dy_step = p["stepsize"] * np.cos(self.direction)
            dx_step[0] = p["targetstepsize"] * np.sin(self.direction[0])
            dy_step[0] = p["targetstepsize"] * np.cos(self.direction[0])
            self.x = self.x + dx_step
            self.y = self.y + dy_step

            return caught_this_frame

        def _maybe_start_chase(self, j):
            """Start dot j chasing, if it is currently far enough from the prey.

            Freezes the onset geometry so later event rows can report how far
            apart the two dots were, and how far the chaser had already
            travelled, at the moment the chase began.
            """
            dx = abs(self.x[0] - self.x[j])
            dy = abs(self.y[0] - self.y[j])
            if self.p["use_euclidean_onset"]:
                far_enough = np.hypot(dx, dy) > self.chasedistthresh
            else:
                far_enough = dx > self.chasedistthresh and dy > self.chasedistthresh
            if far_enough:
                self.chasing_now[j] = True
                self.chase_onset_time.append(core.getTime())
                self.active_chaser = int(j)
                self.active_chase = dict(
                    chaser_idx=int(j),
                    chaser_group=self.group_of(j),
                    onset_geom=self.geometry(j),
                    onset_dist_from_trial_start=self.dist_from_trial_start(j),
                    onset_clock=core.getTime(),
                )

        def _rand_dir_delta(self):
            if round(self.p["dirchfreq"] * self.rng.random()) == 1:
                randval = self.rng.integers(1, 10)
                return -2 * np.pi + 4 * np.pi * (0.1 * randval)
            return 0.0

        def release_chasers(self):
            """Stop the active chase and send the chaser off on a random heading."""
            chasers = np.nonzero(self.chasing_now)[0]
            for j in chasers:
                self.direction[j] = (self.rng.random() * 2 - 1) * np.pi
            self.chasing_now[:] = False
            self.active_chaser = None
            self.active_chase = None
            return chasers

        def mark_flash(self, indices, duration):
            end_time = core.getTime() + duration
            for j in indices:
                self.flash_until[j] = end_time

        def clear_flash(self):
            """Drop all feedback highlighting; dots redraw in their own colour."""
            self.flash_until = {}


    def build_stimuli(win, p):
        frame = visual.Rect(
            win, width=p["displayhoriz"], height=p["displayvert"],
            lineColor=rgb255_to_psychopy(p["frame_col"]), lineWidth=4,
            fillColor=None,
        )
        prey_dot = visual.Circle(
            win, radius=p["dotsize"], fillColor=rgb255_to_psychopy(p["targ_col"]),
            lineColor=rgb255_to_psychopy((0, 0, 0)), lineWidth=p["dotedgesize"],
        )
        cue_ring = visual.Circle(
            win, radius=p["cuesize"], fillColor=None,
            lineColor=rgb255_to_psychopy(p["cue_col"]), lineWidth=p["cuewidth"],
        )
        text = visual.TextStim(win, text="", color="black", height=24, wrapWidth=800)
        return frame, prey_dot, cue_ring, text


    def set_fill(stim, color):
        """Assign a fill colour only when it actually changes."""
        if getattr(stim, "_cur_fill", None) != color:
            stim.fillColor = color
            stim._cur_fill = color


    def draw_frame(win, frame, prey_dot, group_dots_a, group_dots_b, field, p):
        """Draw the border, the prey and both predator groups for one frame.

        Each predator's colour is derived from field.flash_until on every frame
        rather than latched when the flash starts, so clearing that dict — at a
        trial reset, or via clear_flash() — puts every dot back to its normal
        colour on the very next frame.
        """
        frame.draw()
        prey_dot.pos = (field.x[0], field.y[0])
        prey_dot.draw()

        now = core.getTime()
        flash_color = rgb255_to_psychopy(p["flash_col"])

        for dots, idxs, base_color in (
                (group_dots_a, field.group1_idx(), rgb255_to_psychopy(p["grp1_col"])),
                (group_dots_b, field.group2_idx(), rgb255_to_psychopy(p["grp2_col"]))):
            for k, dot_idx in enumerate(idxs):
                dots[k].pos = (field.x[dot_idx], field.y[dot_idx])
                flashing = (dot_idx in field.flash_until
                            and now < field.flash_until[dot_idx])
                set_fill(dots[k], flash_color if flashing else base_color)
                dots[k].draw()

    def make_group_dots(win, p, n, color):
        return [visual.Circle(win, radius=p["dotsize"], fillColor=rgb255_to_psychopy(color),
                               lineColor=rgb255_to_psychopy((0, 0, 0)), lineWidth=p["dotedgesize"])
                for _ in range(n)]


    def key_time(key, fallback):
        """Prefer the keyboard's hardware timestamp; fall back to the frame clock."""
        t = getattr(key, "tDown", None)
        if t is None or not np.isfinite(t) or abs(t - fallback) > 1.0:
            return fallback, False
        return t, True


    def fill_onset(ev, chase):
        """Copy the frozen chase-onset identity and geometry onto an event row."""
        if not chase:
            return ev
        g = chase["onset_geom"]
        ev["chaser_idx"] = chase["chaser_idx"]
        ev["chaser_group"] = chase["chaser_group"]
        ev["onset_dx"] = f'{g["dx"]:.2f}'
        ev["onset_dy"] = f'{g["dy"]:.2f}'
        ev["onset_dist"] = f'{g["dist"]:.2f}'
        ev["onset_dist_from_trial_start"] = \
            f'{chase.get("onset_dist_from_trial_start", float("nan")):.2f}'
        return ev


    def fill_current(ev, geom):
        """Copy prey and chaser positions at the moment of the event."""
        ev["prey_x"] = f'{geom["prey_x"]:.2f}'
        ev["prey_y"] = f'{geom["prey_y"]:.2f}'
        ev["chaser_x"] = f'{geom["chaser_x"]:.2f}'
        ev["chaser_y"] = f'{geom["chaser_y"]:.2f}'
        return ev


    def hold_display(win, draw_fn, secs):
        """Hold a frozen display for `secs`, still honouring escape."""
        end = core.getTime() + secs
        while core.getTime() < end:
            draw_fn()
            win.flip()
            if event.getKeys(["escape"]):
                raise AbortExperiment()


    def run_block(win, kb, stim, p, rng, block_def, log_event, subj, run_id, out=None):
        """Run one block: repeated trials until block_duration_secs elapses.

        Each trial re-places every dot, then runs frame by frame until a
        response is recorded — a spacebar hit, a spacebar false alarm, or the
        chaser catching the prey. The trial then shows feedback, blanks, and the
        next trial begins.

        Events are written the moment they happen via log_event. `out` is filled
        in the finally block, so a summary survives an escape abort.
        """
        frame, prey_dot, text = stim["frame"], stim["prey_dot"], stim["text"]

        numdotspergroup = block_def["numdotspergroup"]
        field = DotField(p, numdotspergroup, rng)
        group_dots_a = make_group_dots(win, p, numdotspergroup, p["grp1_col"])
        group_dots_b = make_group_dots(win, p, numdotspergroup, p["grp2_col"])

        def draw_now():
            draw_frame(win, frame, prey_dot, group_dots_a, group_dots_b, field, p)

        def draw_blank():
            frame.draw()

        if p["showcue"]:
            cue_ring = stim["cue_ring"]
            draw_now()
            cue_ring.pos = (field.x[0], field.y[0])
            cue_ring.draw()
            win.flip()
            core.wait(p["cueduration"])

        draw_now()
        win.flip()
        core.wait(p["cueduration"])

        events_log = []
        n_correct = 0
        n_fa = 0
        n_miss = 0
        n_discarded = 0
        n_other_keys = 0
        n_frames = 0
        n_trials = 0
        time_at_risk = 0.0      # seconds with no chase running: the false-alarm denominator
        chase_onset_clock = None
        chase_onset_time = None

        tstate = dict(num=0, start=0.0)   # current trial number and start time
        event_counter = [0]

        def emit(ev):
            event_counter[0] += 1
            ev["event_num"] = event_counter[0]
            ev["trial_num"] = tstate["num"]
            ev["trial_start_time"] = f'{tstate["start"]:.4f}'
            ev["datetime"] = datetime.now().isoformat(timespec="milliseconds")
            log_event(ev)
            events_log.append(ev)

        kb.clearEvents()
        win.frameIntervals = []
        block_start_wall = datetime.now().isoformat(timespec="milliseconds")
        block_clock = core.Clock()
        prev_t = 0.0

        completed = False
        try:
            # Clock time, not frame count, so the block is a true 120 s.
            while block_clock.getTime() < p["block_duration_secs"]:

                # --- start a trial: every dot is re-placed ---
                tstate["num"] += 1
                n_trials += 1
                field.reset_positions()
                kb.clearEvents()
                tstate["start"] = block_clock.getTime()
                prev_t = block_clock.getTime()

                chase_onset_clock = None
                chase_onset_time = None
                blackout_active = True
                blackout_start = core.getTime()
                chase_initiated_this_frame = False
                blackout_lifted_this_frame = False

                trial_over = False
                trial_outcome = None
                hit_chasers = None

                while (not trial_over) and block_clock.getTime() < p["block_duration_secs"]:
                    n_frames += 1
                    t_now_block = block_clock.getTime()
                    dt = t_now_block - prev_t
                    prev_t = t_now_block
                    if not field.chasing_now.any():
                        time_at_risk += dt

                    n_active_before = field.chasing_now.any()
                    caught = field.update_frame(p)

                    for c in caught:
                        n_miss += 1
                        ev = blank_event("Miss")
                        ev["chase_onset_time"] = (f"{chase_onset_time:.4f}"
                                                  if chase_onset_time is not None else "")
                        ev["chase_finish_time"] = f"{block_clock.getTime():.4f}"
                        fill_onset(ev, c["chase"])
                        if not c["chase"]:
                            ev["chaser_idx"] = c["chaser_idx"]
                            ev["chaser_group"] = field.group_of(c["chaser_idx"])
                        fill_current(ev, c["geom"])
                        emit(ev)
                        chase_onset_clock = None
                        chase_onset_time = None
                        if p["end_trial_on_catch"]:
                            trial_over = True
                            trial_outcome = "Miss"

                    if not n_active_before and field.chasing_now.any():
                        chase_initiated_this_frame = True
                        chase_onset_clock = core.getTime()
                        chase_onset_time = block_clock.getTime()

                    now = core.getTime()
                    if blackout_active and (now - blackout_start) > p["blackout_secs"]:
                        blackout_active = False
                        blackout_lifted_this_frame = True

                    keys = kb.getKeys(waitRelease=False)
                    space_keys = [k for k in keys if k.name == "space"]
                    other_keys = [k for k in keys if k.name not in ("space", "escape")]
                    escape_keys = [k for k in keys if k.name == "escape"]

                    for k in other_keys:
                        n_other_keys += 1
                        ev = blank_event("Other Key")
                        ev["key_name"] = k.name
                        ev["button_press_time"] = f"{block_clock.getTime():.4f}"
                        emit(ev)

                    # A press is unscoreable if it lands during the blackout, or on
                    # the same frame the chase started or the blackout lifted —
                    # it cannot have been a response to what just appeared.
                    pressed = len(space_keys) > 0
                    discard_reason = None
                    possible_discard = (not blackout_active) and pressed
                    if blackout_active and pressed:
                        pressed = False
                        discard_reason = "blackout"
                    elif possible_discard and (chase_initiated_this_frame or blackout_lifted_this_frame):
                        pressed = False
                        discard_reason = ("onset frame" if chase_initiated_this_frame
                                          else "blackout lift")

                    if discard_reason:
                        for k in space_keys:
                            n_discarded += 1
                            ev = blank_event("Discarded Press")
                            ev["key_name"] = f"space ({discard_reason})"
                            t_press, _ = key_time(k, now)
                            ev["button_press_time"] = f"{block_clock.getTime() - (now - t_press):.4f}"
                            if field.active_chase:
                                fill_onset(ev, field.active_chase)
                                fill_current(ev, field.geometry(field.active_chase["chaser_idx"]))
                            emit(ev)

                    if pressed and not trial_over:
                        # RT comes off the key event, not the frame loop.
                        t_press, _ = key_time(space_keys[0], now)
                        press_block_time = block_clock.getTime() - (now - t_press)
                        if field.chasing_now.any():
                            chase = field.active_chase
                            j = chase["chaser_idx"] if chase else field.active_chaser
                            geom = field.geometry(j) if j is not None else None
                            rt_ms = ((t_press - chase_onset_clock) * 1000
                                     if chase_onset_clock else None)
                            ev = blank_event("Hit")
                            ev["rt_ms"] = f"{rt_ms:.1f}" if rt_ms is not None else ""
                            ev["chase_onset_time"] = (f"{chase_onset_time:.4f}"
                                                      if chase_onset_time is not None else "")
                            ev["button_press_time"] = f"{press_block_time:.4f}"
                            fill_onset(ev, chase)
                            if geom:
                                fill_current(ev, geom)
                            emit(ev)
                            hit_chasers = field.release_chasers()
                            n_correct += 1
                            chase_onset_clock = None
                            chase_onset_time = None
                            if p["end_trial_on_hit"]:
                                trial_over = True
                                trial_outcome = "Hit"
                        else:
                            ev = blank_event("False Alarm")
                            ev["button_press_time"] = f"{press_block_time:.4f}"
                            emit(ev)
                            n_fa += 1
                            if p["end_trial_on_false_alarm"]:
                                trial_over = True
                                trial_outcome = "False Alarm"
                        blackout_active = True
                        blackout_start = now
                        kb.clearEvents()

                    chase_initiated_this_frame = False
                    blackout_lifted_this_frame = False

                    draw_now()
                    win.flip()

                    if escape_keys or event.getKeys(["escape"]):
                        raise AbortExperiment()

                # --- end of trial: feedback, then blank the display ---
                if trial_over:
                    if (trial_outcome == "Hit" and hit_chasers is not None
                            and len(hit_chasers) and p["flash_duration"] > 0):
                        field.mark_flash(hit_chasers, p["flash_duration"])
                        hold_display(win, draw_now, p["flash_duration"])
                    field.clear_flash()
                    if p["trial_blank_secs"] > 0:
                        hold_display(win, draw_blank, p["trial_blank_secs"])

            completed = True

        finally:
            block_duration = block_clock.getTime()

            # A chase still running when the block clock expires is censored,
            # not counted as a miss.
            n_unresolved = 0
            if field.chasing_now.any():
                n_unresolved = 1
                ev = blank_event("Unresolved Chase")
                ev["chase_onset_time"] = (f"{chase_onset_time:.4f}"
                                          if chase_onset_time is not None else "")
                ev["chase_finish_time"] = f"{block_duration:.4f}"
                fill_onset(ev, field.active_chase)
                if field.active_chaser is not None:
                    fill_current(ev, field.geometry(field.active_chaser))
                emit(ev)

            # Terminator row; its absence in the CSV marks an aborted block.
            ev = blank_event("Block End" if completed else "Block Aborted")
            ev["chase_finish_time"] = f"{block_duration:.4f}"
            emit(ev)

            try:
                thresh = getattr(win, "refreshThreshold", None) or (1.0 / p["frmrate_expected"] * 1.5)
                n_dropped = int(sum(1 for i in win.frameIntervals if i > thresh))
            except Exception:
                n_dropped = ""

            summary = dict(
                label=block_def["label"],
                predator_dots=block_def["predator_dots"],
                n_correct=n_correct, n_miss=n_miss, n_fa=n_fa,
                n_unresolved=n_unresolved, n_discarded=n_discarded,
                n_other_keys=n_other_keys,
                n_trials=n_trials,
                n_chases=len(field.chase_onset_time),   # hit-rate denominator
                n_caught=field.n_caught,
                time_at_risk=time_at_risk,
                n_frames=n_frames, n_dropped=n_dropped,
                block_start_wall=block_start_wall,
                events=events_log, block_duration=block_duration,
                completed=completed,
            )
            if out is not None:
                out.update(summary)

        return summary


    def main():
        subj, practice, screen_w_cm, view_dist_cm = get_subject_info()
        seed = subj + (P["practice_seed_offset"] if practice else 0)
        rng = np.random.default_rng(seed)
        run_id = uuid.uuid4().hex[:8]   # distinguishes re-runs within an appended file

        # A calibrated monitor keeps pixel units convertible to degrees offline.
        mon = monitors.Monitor("expMonitor", width=screen_w_cm, distance=view_dist_cm)
        win = visual.Window(fullscr=True, color=rgb255_to_psychopy(P["bkgd_col"]),
                             units="pix", allowGUI=False, monitor=mon)
        try:
            mon.setSizePix(list(win.size))
        except Exception:
            pass

        kb = keyboard.Keyboard()
        event.globalKeys.clear()

        measured = win.getActualFrameRate(nIdentical=20, nMaxFrames=100, nWarmUpFrames=10, threshold=1)
        P["frmrate_measured"] = measured if measured else P["frmrate_expected"]

        win.recordFrameIntervals = True
        win.refreshThreshold = (1.0 / P["frmrate_measured"]) + 0.004

        mon_info = dict(width_cm=screen_w_cm, distance_cm=view_dist_cm,
                        size_pix=list(win.size))
        write_sidecar(subj, practice, seed, run_id, mon_info, P["frmrate_measured"])

        frame, prey_dot, cue_ring, text = build_stimuli(win, P)
        stim = dict(frame=frame, prey_dot=prey_dot, cue_ring=cue_ring, text=text)

        ev_file, ev_writer, bk_file, bk_writer = make_writers(subj, practice)

        text.text = (
            "Instructions: Chasing Detection Task\n\n"
            "1. One dot (the prey) will appear in red.\n"
            "2. All dots start moving randomly.\n"
            "3. At some point, one starts to chase the prey.\n"
            "4. Press the SPACEBAR as soon as you detect a chase.\n"
            "5. When you correctly detect a chase, that dot will briefly flash green.\n"
            "6. After each response the display resets and a new trial begins.\n"
            "7. Respond as quickly and accurately as possible.\n\n"
            "Press any key to begin."
        )
        text.draw()
        win.flip()
        instr_clock = core.Clock()
        event.waitKeys()
        instr_secs = instr_clock.getTime()

        exp_start = core.getTime()
        try:
            block_order = rng.permutation(len(P["block_defs"]))

            for pos, bidx in enumerate(block_order, start=1):
                block_def = P["block_defs"][bidx]

                core.wait(P["ITI"])

                def log_event(ev, _pos=pos, _bd=block_def):
                    """Stamp an event with run/block identity, write it, flush."""
                    ev["run_id"] = run_id
                    ev["subject"] = subj
                    ev["practice"] = int(practice)
                    ev["seed"] = seed
                    ev["block_order"] = _pos
                    ev["block_label"] = _bd["label"]
                    ev["predator_dots"] = _bd["predator_dots"]
                    ev_writer.writerow([ev.get(k, "") for k in EVENT_FIELDS])
                    ev_file.flush()

                result = {}
                try:
                    run_block(win, kb, stim, P, rng, block_def,
                              log_event, subj, run_id, out=result)
                finally:
                    if result:
                        bk_writer.writerow([
                            datetime.now().isoformat(timespec="milliseconds"),
                            run_id, subj, int(practice), seed, pos,
                            result["label"], result["predator_dots"],
                            f'{result["block_duration"]:.3f}',
                            result["n_frames"], result["n_trials"],
                            result["n_chases"],
                            result["n_correct"], result["n_miss"], result["n_fa"],
                            result["n_unresolved"], result["n_discarded"],
                            result["n_other_keys"],
                            f'{result["time_at_risk"]:.3f}',
                            f'{P["frmrate_measured"]:.3f}',
                            result["n_dropped"], int(result["completed"]),
                        ])
                        bk_file.flush()

            text.text = "End of Experiment\n\nThank you very much.\n\nPlease get the experimenter."
            text.draw()
            win.flip()
            core.wait(1.0)
            event.waitKeys()

        except AbortExperiment:
            print("\n--- ABORTED BY EXPERIMENTER (escape) ---\n")
        except Exception:
            print("\n--- RUNTIME ERROR ---")
            traceback.print_exc()
            print("---------------------\n")
        finally:
            try:
                fname = os.path.join(DATA_DIR, f"Chase1_s{subj}_{run_id}_params.json")
                with open(fname) as f:
                    payload = json.load(f)
                payload["instruction_screen_secs"] = round(instr_secs, 3)
                payload["experiment_secs"] = round(core.getTime() - exp_start, 3)
                payload["ended"] = datetime.now().isoformat(timespec="milliseconds")
                with open(fname, "w") as f:
                    json.dump(payload, f, indent=2, default=str)
            except Exception:
                pass
            ev_file.close()
            bk_file.close()
            win.close()
            core.quit()


    if __name__ == "__main__":
        main()

except Exception as e:
    print("\n--- CRITICAL STARTUP/IMPORT ERROR ---")
    traceback.print_exc()
    print("-------------------------------------\n")
    sys.exit(1)
