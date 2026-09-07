#!/usr/bin/env python3

import traceback
import sys

try:
    from psychopy import visual, core, event, gui, monitors
    from psychopy.hardware import keyboard
    import numpy as np
    import csv
    import os
    import json
    import uuid
    import platform
    from datetime import datetime

    P = dict(
        # --- Stream timing -------------------------------------------------
        # One symbol at a time, alternating between two positions.
        #
        # blank_dur is the experimental knob: lengthening it lengthens the
        # interval over which a symbol must be held in memory, since the
        # same-position (retention) interval is 2 x (symbol_dur + blank_dur).
        # symbol_dur is meant to stay fixed while blank_dur is varied.
        #
        # Defaults reproduce the original schematic: symbols every 1.0 s, each
        # position every 2.0 s, each location empty for 1.5 s between its own
        # symbols. Everything downstream is derived, so changing blank_dur
        # alone rescales the whole schedule consistently.
        symbol_dur=0.5,             # display duration — held fixed
        blank_dur=0.5,              # blank between successive symbols — varied
        stream_duration=300.0,      # 5 minutes of test stream
        practice_duration=45.0,
        run_practice=True,
        # Practice may be repeated if the participant clearly has not grasped
        # the rule. Set practice_max_runs=1 to disable the retry.
        practice_max_runs=2,
        practice_pass_hit_rate=0.5,

        # --- Response ------------------------------------------------------
        response_key="space",
        # A press counts as a hit if it lands within this many seconds of a
        # target onset. Kept below the 2.0 s gap between successive symbols at
        # one position, so a press can never be ambiguous between two items.
        response_window=1.5,

        # --- Sequence structure --------------------------------------------
        # Target: the symbol matches the previous symbol AT THE SAME POSITION
        #         (two presentations back). This is the event to respond to.
        # Lure:   the symbol matches the immediately preceding symbol at the
        #         OTHER position. Not a target; it exists so that "have I seen
        #         this symbol recently?" is not a sufficient strategy.
        targets_per_min=10.0,
        lures_per_min=10.0,
        # "per_minute" holds the rate per unit time, so a longer blank_dur means
        # fewer presentations but the same number of targets — and therefore a
        # higher chance of a target per symbol. "per_presentation" holds that
        # per-symbol chance fixed instead, so target count falls with blank_dur.
        # The two diverge as soon as blank_dur is varied; see README.
        target_rate_mode="per_minute",
        # Floor on the interval between any two repeat events, in seconds.
        # Converted to presentations at run time, so it holds whatever the
        # timing is set to.
        min_event_gap_s=2.0,
        first_event_min_s=6.0,
        # A same-position repeat at lag 2-3 is recent enough to be a plausible
        # memory intrusion, and is flagged so those false alarms can be told
        # apart from random ones. Beyond that the whole pool has been seen at
        # every location, so a longer lag carries no information.
        dist_repeat_max_lag=3,
        # Excludes visually confusable characters (B G I L O Q S Z 0 1 2 5 6 8)
        # so perceptual errors are not mistaken for memory errors.
        symbols="ACDEFHJKMNPRTUVWXY3479",

        # --- Auditory feedback ---------------------------------------------
        # Short tones fired the moment a response is classified. The whole
        # feature is off by default: set enabled=True to switch it on, and the
        # per-phase and per-event flags below decide where it applies. If no
        # sound backend is available the task prints a warning and runs
        # silently rather than failing.
        audio_feedback=dict(
            enabled=False,          # master switch — the only one you need to flip
            in_practice=True,       # where it applies, once enabled
            in_test=False,
            on_hit=True,
            on_false_alarm=True,
            on_miss=False,          # fires when a target's window closes unanswered
            hit_hz=1000.0, hit_secs=0.06,
            false_alarm_hz=350.0, false_alarm_secs=0.12,
            miss_hz=220.0, miss_secs=0.12,
            volume=0.5,
        ),

        # --- Display -------------------------------------------------------
        position_offset=220,        # pixels left and right of centre
        symbol_height=64,
        symbol_font="Arial",
        symbol_col=(0, 0, 0),
        bkgd_col=(160, 160, 160),
        # Small marker showing the currently empty location, as in the design
        # schematic. Both locations are always marked.
        show_placeholder=True,
        placeholder_radius=5,
        placeholder_col=(90, 90, 90),
        # Optional low-contrast mask filling the empty location through its
        # blank interval. Off by default; switch on if piloting shows
        # afterimages of the symbols.
        use_mask=False,
        mask_radius=42,
        mask_col=(140, 140, 140),

        frmrate_expected=60,
        practice_seed_offset=100000,
    )

    DATA_DIR = "data"

    EVENT_FIELDS = [
        "datetime", "run_id", "subject", "practice", "seed", "phase",
        "event_num", "event_type",
        "presentation_num", "onset_s", "position", "symbol",
        "prev_same_pos_symbol", "prev_other_pos_symbol",
        "key_name", "press_time_s", "rt_ms", "responded",
        "attributed_to_presentation",
        # same_pos_lag: how many symbols back AT THIS POSITION this symbol last
        # appeared. 1 = target. >=2 = a repeat at this location that is not a
        # target under the confirmed rule, and so a natural source of intrusions.
        "same_pos_lag",
        # How many further symbols had appeared between the target's onset and
        # the press. 0 = answered before the next symbol arrived.
        "presentations_after_target",
        "near_lure", "near_dist_repeat", "feedback_tone",
    ]

    BLOCK_FIELDS = [
        "datetime", "run_id", "subject", "practice", "seed", "phase",
        "stream_duration_s", "n_presentations", "n_frames",
        "n_targets", "n_hits", "n_misses",
        "n_lures", "n_lure_false_alarms",
        "n_false_alarms", "n_duplicate_presses", "n_other_keys",
        "hit_rate", "mean_rt_ms", "median_rt_ms",
        # Split-half, so a decline across the run is visible without any
        # separate analysis.
        "hit_rate_h1", "hit_rate_h2", "mean_rt_h1_ms", "mean_rt_h2_ms",
        "n_dist_repeats", "n_dist_repeat_false_alarms",
        "fa_per_min", "time_at_risk_s", "audio_feedback",
        "frmrate_measured", "n_dropped_frames", "completed",
    ]

    LEFT, RIGHT = 0, 1


    class AbortExperiment(Exception):
        """Raised on escape so data buffers flush instead of hard-quitting."""
        pass


    def rgb255_to_psychopy(rgb):
        return [(c / 127.5) - 1 for c in rgb]


    def blank_event(event_type):
        ev = {k: "" for k in EVENT_FIELDS}
        ev["event_type"] = event_type
        return ev


    def soa_of(p):
        """Onset-to-onset interval between successive symbols."""
        return p["symbol_dur"] + p["blank_dur"]


    def retention_interval(p):
        """Gap between successive symbols at one position — the memory interval."""
        return 2.0 * soa_of(p)


    def min_gap_presentations(p):
        """Minimum presentations between repeat events, given the current timing.

        Enforces three things at once: the requested floor in seconds, a gap
        wide enough that response windows never overlap, and a floor of 2 so no
        two events are adjacent (which is what guarantees a target can never
        also be a lure).
        """
        soa = soa_of(p)
        return max(2,
                   int(np.ceil(p["min_event_gap_s"] / soa)),
                   int(np.ceil(p["response_window"] / soa)))


    def check_timing(p):
        """Reject timing combinations that would make scoring ambiguous."""
        problems = []
        if p["response_window"] >= retention_interval(p):
            problems.append(
                f'response_window ({p["response_window"]}s) must be shorter than '
                f'the same-position interval ({retention_interval(p):.2f}s), or a '
                f'press cannot be attributed to one symbol')
        if p["blank_dur"] <= 0 or p["symbol_dur"] <= 0:
            problems.append("symbol_dur and blank_dur must both be positive")
        return problems


    def timing_summary(p):
        """Where the response window sits relative to the rest of the schedule.

        The window has to be short enough that a press cannot belong to two
        symbols (the hard limit, enforced by check_timing), but a window that
        merely runs past the NEXT symbol is a softer concern: a late response
        then overlaps a symbol the participant is trying to encode. Lengthening
        blank_dur is what buys room for longer responses.
        """
        soa = soa_of(p)
        w = p["response_window"]
        return dict(
            symbol_dur=p["symbol_dur"],
            blank_dur=p["blank_dur"],
            soa=soa,
            retention_interval=retention_interval(p),
            empty_at_location=retention_interval(p) - p["symbol_dur"],
            symbols_per_min=60.0 / soa,
            response_window=w,
            window_past_next_symbol=w - soa,
            max_window_before_ambiguity=retention_interval(p),
            n_presentations=int(p["stream_duration"] // soa),
        )


    def print_timing(p):
        s = timing_summary(p)
        print("\n--- timing ---")
        print(f"  display {s['symbol_dur']:.2f}s + blank {s['blank_dur']:.2f}s "
              f"= SOA {s['soa']:.2f}s   ({s['symbols_per_min']:.0f} symbols/min, "
              f"{s['n_presentations']} in the run)")
        print(f"  same position every {s['retention_interval']:.2f}s "
              f"(the memory interval); each location empty "
              f"{s['empty_at_location']:.2f}s")
        print(f"  response window {s['response_window']:.2f}s "
              f"(hard limit {s['max_window_before_ambiguity']:.2f}s)")
        if s["window_past_next_symbol"] > 0:
            print(f"  NOTE: the window runs {s['window_past_next_symbol']:.2f}s "
                  f"past the next symbol's onset, so a late response can overlap "
                  f"the following symbol. Lengthen blank_dur to buy room.")
        else:
            print("  the window closes before the next symbol appears")
        print("---------------\n")


    # ------------------------------------------------------------------ #
    #  Sequence construction                                             #
    # ------------------------------------------------------------------ #

    def build_sequence(rng, n_pres, p):
        """Build the symbol stream for one run.

        Positions strictly alternate, so presentation i sits at position i % 2
        and its same-position predecessor is presentation i - 2.

        Repeat events are placed first as bare slots, then labelled target or
        lure, then symbols are filled in around them. Every non-event symbol is
        chosen to differ from both the previous symbol and the previous symbol
        at its own position, so the only repeats in the stream are the intended
        ones — there are no accidental targets or lures.

        Returns a list of dicts, one per presentation.
        """
        soa = soa_of(p)
        min_gap = min_gap_presentations(p)
        first_ok = max(2, int(np.ceil(p["first_event_min_s"] / soa)))

        if p["target_rate_mode"] == "per_presentation":
            # Chance of a target per symbol is held at whatever the rate implies
            # under the reference 1 s timing, so stream statistics stay
            # comparable when blank_dur changes.
            n_targets = int(round(p["targets_per_min"] / 60.0 * n_pres))
            n_lures = int(round(p["lures_per_min"] / 60.0 * n_pres))
        else:
            stream_min = (n_pres * soa) / 60.0
            n_targets = int(round(p["targets_per_min"] * stream_min))
            n_lures = int(round(p["lures_per_min"] * stream_min))
        n_events = n_targets + n_lures

        span = n_pres - first_ok
        if n_events > 0 and span // min_gap < n_events:
            # Requested density does not fit under the minimum-gap floor.
            n_events = max(0, int(span // min_gap) - 1)
            n_targets = n_events // 2
            n_lures = n_events - n_targets

        # Slot placement. Gaps are min_gap plus a geometric draw, which gives a
        # flat hazard above the floor — the participant cannot learn to expect
        # an event from how long it has been since the last one.
        mean_gap = span / n_events if n_events else span
        extra = max(0.0, mean_gap - min_gap)
        p_geom = 1.0 / (1.0 + extra) if extra > 0 else 1.0

        slots = []
        i = first_ok + int(rng.integers(0, min_gap + 1))
        while i < n_pres and len(slots) < n_events:
            slots.append(int(i))
            i += min_gap + int(rng.geometric(p_geom)) - 1

        # Label the slots target or lure, keeping the two counts on track as we
        # go. A slot is barred from being a target only when that would make a
        # third consecutive same-position repeat (four identical symbols at one
        # location); back-to-back targets 2 s apart are allowed, since 2 s is
        # the intended minimum interval between repetitions.
        n_slots = len(slots)
        want_t = min(n_targets, n_slots)
        want_l = n_slots - want_t
        kind = {}
        for s in sorted(slots):
            barred = kind.get(s - 2) == "target" and kind.get(s - 4) == "target"
            if barred or want_t <= 0:
                lab = "lure" if want_l > 0 else "target"
            elif want_l <= 0:
                lab = "target"
            else:
                lab = "target" if rng.random() < want_t / (want_t + want_l) else "lure"
            kind[s] = lab
            if lab == "target":
                want_t -= 1
            else:
                want_l -= 1

        # Fill in symbols.
        syms = list(p["symbols"])
        seq_sym = [None] * n_pres
        for i in range(n_pres):
            k = kind.get(i)
            if k == "target":
                seq_sym[i] = seq_sym[i - 2]
            elif k == "lure":
                seq_sym[i] = seq_sym[i - 1]
            else:
                banned = set()
                if i >= 1:
                    banned.add(seq_sym[i - 1])
                if i >= 2:
                    banned.add(seq_sym[i - 2])
                choices = [s for s in syms if s not in banned]
                seq_sym[i] = choices[int(rng.integers(len(choices)))]

        # How many symbols back at its own position each symbol last appeared.
        # 1 is a target. 2 or more is a repeat at the same location that the
        # confirmed rule does NOT make a target — these are left uncontrolled on
        # purpose. Suppressing them would mean any symbol recognised at a
        # location must be a target, which would let a participant succeed on
        # location familiarity alone without tracking how recent it was.
        history = {LEFT: [], RIGHT: []}
        same_pos_lag = [0] * n_pres
        for i in range(n_pres):
            h = history[i % 2]
            for back, sym in enumerate(reversed(h), start=1):
                if sym == seq_sym[i]:
                    same_pos_lag[i] = back
                    break
            h.append(seq_sym[i])

        stream = []
        for i in range(n_pres):
            prev_same = seq_sym[i - 2] if i >= 2 else ""
            prev_other = seq_sym[i - 1] if i >= 1 else ""
            stream.append(dict(
                idx=i,
                onset=i * soa,
                position=i % 2,
                symbol=seq_sym[i],
                is_target=(seq_sym[i] == prev_same) if i >= 2 else False,
                is_lure=(seq_sym[i] == prev_other) if i >= 1 else False,
                same_pos_lag=same_pos_lag[i],
                is_dist_repeat=(2 <= same_pos_lag[i] <= p["dist_repeat_max_lag"]),
                prev_same=prev_same,
                prev_other=prev_other,
                responded=False,
                rt=None,
            ))
        return stream


    def validate_sequence(stream, p):
        """Check the stream obeys its own rules. Returns a list of problems."""
        problems = []
        soa = soa_of(p)
        min_gap = min_gap_presentations(p)

        for s in stream:
            if s["is_target"] and s["is_lure"]:
                problems.append(f"presentation {s['idx']} is both target and lure")

        events = [s["idx"] for s in stream if s["is_target"] or s["is_lure"]]
        for a, b in zip(events, events[1:]):
            if b - a < min_gap:
                problems.append(f"events {a} and {b} are only {b - a} apart")

        targets = [s["idx"] for s in stream if s["is_target"]]
        if targets and targets[0] * soa < p["first_event_min_s"]:
            problems.append(f"first target at {targets[0] * soa:.1f}s, "
                            f"before {p['first_event_min_s']}s")
        for a, b in zip(targets, targets[1:]):
            if (b - a) * soa < p["min_event_gap_s"]:
                problems.append(f"targets {a} and {b} less than "
                                f"{p['min_event_gap_s']} s apart")
        for s in stream:
            if s["is_target"] and s["position"] != stream[s["idx"] - 2]["position"]:
                problems.append(f"target {s['idx']} is not a same-position repeat")
        return problems


    # ------------------------------------------------------------------ #
    #  Setup                                                             #
    # ------------------------------------------------------------------ #

    def get_subject_info():
        info = {
            "Subject Number": 1,
            "Run practice first": True,
            "Screen width (cm)": 52.0,
            "Viewing distance (cm)": 57.0,
        }
        dlg = gui.DlgFromDict(info, title="VWM Repeat Detection Task")
        if not dlg.OK:
            core.quit()
        return (int(info["Subject Number"]), bool(info["Run practice first"]),
                float(info["Screen width (cm)"]), float(info["Viewing distance (cm)"]))


    def make_writers(subj):
        """Open the event and block CSVs in append mode, writing headers if new."""
        os.makedirs(DATA_DIR, exist_ok=True)

        ev_name = os.path.join(DATA_DIR, f"VWM_s{subj}_events.csv")
        ev_new = not os.path.exists(ev_name)
        ev_file = open(ev_name, "a", newline="")
        ev_writer = csv.writer(ev_file)
        if ev_new:
            ev_writer.writerow(EVENT_FIELDS)

        bk_name = os.path.join(DATA_DIR, f"VWM_s{subj}_blocks.csv")
        bk_new = not os.path.exists(bk_name)
        bk_file = open(bk_name, "a", newline="")
        bk_writer = csv.writer(bk_file)
        if bk_new:
            bk_writer.writerow(BLOCK_FIELDS)

        return ev_file, ev_writer, bk_file, bk_writer


    def write_stream_record(subj, run_id, phase, stream):
        """Save the whole stimulus stream, so the display can be reconstructed."""
        os.makedirs(DATA_DIR, exist_ok=True)
        fname = os.path.join(DATA_DIR, f"VWM_s{subj}_{run_id}_{phase}_sequence.csv")
        with open(fname, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["presentation_num", "onset_s", "position", "symbol",
                        "is_target", "is_lure", "same_pos_lag",
                        "prev_same_pos_symbol", "prev_other_pos_symbol"])
            for s in stream:
                w.writerow([s["idx"], f'{s["onset"]:.3f}',
                            "left" if s["position"] == LEFT else "right",
                            s["symbol"], int(s["is_target"]), int(s["is_lure"]),
                            s["same_pos_lag"],
                            s["prev_same"], s["prev_other"]])


    def write_sidecar(subj, seed, run_id, mon_info, frmrate):
        os.makedirs(DATA_DIR, exist_ok=True)
        payload = dict(
            run_id=run_id, subject=subj, seed=int(seed),
            started=datetime.now().isoformat(timespec="milliseconds"),
            monitor=mon_info, frmrate_measured=frmrate,
            versions=dict(python=sys.version.split()[0], numpy=np.__version__,
                          platform=platform.platform()),
            params={k: (list(v) if isinstance(v, tuple) else v) for k, v in P.items()},
            # Everything the timing implies, so a run's schedule is on record
            # without having to recompute it from symbol_dur and blank_dur.
            timing=timing_summary(P),
            derived_timing=dict(
                soa_s=soa_of(P),
                retention_interval_s=retention_interval(P),
                empty_interval_at_location_s=retention_interval(P) - P["symbol_dur"],
                symbols_per_min=60.0 / soa_of(P),
                n_presentations=int(P["stream_duration"] // soa_of(P)),
                min_event_gap_presentations=min_gap_presentations(P),
            ),
        )
        try:
            from psychopy import __version__ as psychopy_version
            payload["versions"]["psychopy"] = psychopy_version
        except Exception:
            pass
        fname = os.path.join(DATA_DIR, f"VWM_s{subj}_{run_id}_params.json")
        with open(fname, "w") as f:
            json.dump(payload, f, indent=2, default=str)


    class Feedback:
        """Short tones fired when a response is classified.

        Tones are built once and warmed at startup, because the first play on a
        cold audio backend can take long enough to drop a frame. Every audio
        call is guarded: if the backend is missing or fails, `ok` stays False
        and every play() is a no-op, so the task runs silently rather than
        crashing on a machine with no working sound.
        """

        def __init__(self, p):
            self.cfg = p["audio_feedback"]
            self.tones = {}
            self.ok = False
            if not self.cfg.get("enabled"):
                return
            try:
                from psychopy import sound
                specs = (("hit", self.cfg["hit_hz"], self.cfg["hit_secs"]),
                         ("false_alarm", self.cfg["false_alarm_hz"],
                          self.cfg["false_alarm_secs"]),
                         ("miss", self.cfg["miss_hz"], self.cfg["miss_secs"]))
                for name, hz, secs in specs:
                    try:
                        # hamming avoids the click a square-edged tone makes
                        s = sound.Sound(value=hz, secs=secs, hamming=True,
                                        volume=self.cfg["volume"], name=name)
                    except TypeError:
                        s = sound.Sound(value=hz, secs=secs,
                                        volume=self.cfg["volume"])
                    self.tones[name] = s
                self.ok = True
            except Exception as e:
                print(f"[auditory feedback unavailable: {e}] — running silently")
                return

            # Warm the backend so the first real tone is not the slow one.
            try:
                for s in self.tones.values():
                    s.volume = 0.0
                    s.play()
                    s.stop()
                    s.volume = self.cfg["volume"]
            except Exception:
                pass

        def wanted(self, kind, phase):
            c = self.cfg
            if not (self.ok and c.get("enabled")):
                return False
            if phase == "practice" and not c.get("in_practice"):
                return False
            if phase == "test" and not c.get("in_test"):
                return False
            return bool(c.get(f"on_{kind}"))

        def play(self, kind, phase):
            """Play the tone for `kind`; returns the name logged, or ""."""
            if not self.wanted(kind, phase):
                return ""
            s = self.tones.get(kind)
            if s is None:
                return ""
            try:
                s.stop()      # so rapid repeats retrigger rather than overlap
            except Exception:
                pass
            try:
                s.play()
                return kind
            except Exception:
                return ""


    def build_stimuli(win, p):
        off = p["position_offset"]
        sym = [visual.TextStim(win, text="", height=p["symbol_height"],
                               font=p["symbol_font"],
                               color=rgb255_to_psychopy(p["symbol_col"]),
                               pos=(-off, 0), alignText="center", anchorHoriz="center"),
               visual.TextStim(win, text="", height=p["symbol_height"],
                               font=p["symbol_font"],
                               color=rgb255_to_psychopy(p["symbol_col"]),
                               pos=(off, 0), alignText="center", anchorHoriz="center")]
        dot = [visual.Circle(win, radius=p["placeholder_radius"],
                             fillColor=rgb255_to_psychopy(p["placeholder_col"]),
                             lineColor=None, pos=(-off, 0)),
               visual.Circle(win, radius=p["placeholder_radius"],
                             fillColor=rgb255_to_psychopy(p["placeholder_col"]),
                             lineColor=None, pos=(off, 0))]
        mask = [visual.Circle(win, radius=p["mask_radius"],
                              fillColor=rgb255_to_psychopy(p["mask_col"]),
                              lineColor=None, pos=(-off, 0)),
                visual.Circle(win, radius=p["mask_radius"],
                              fillColor=rgb255_to_psychopy(p["mask_col"]),
                              lineColor=None, pos=(off, 0))]
        text = visual.TextStim(win, text="", color="black", height=24, wrapWidth=900)
        return sym, dot, mask, text


    # ------------------------------------------------------------------ #
    #  The stream                                                        #
    # ------------------------------------------------------------------ #

    def run_stream(win, kb, stim, p, stream, duration, phase, log_event,
                   feedback=None, out=None):
        """Present one continuous stream and score responses against it.

        The display is driven from elapsed time rather than a frame count, so
        the schedule cannot drift: on every frame the current presentation is
        floor(t / soa), and its symbol is visible for the first symbol_dur of
        that interval.

        A keypress is attributed to the most recent target whose response
        window is still open. Because repeat events are at least 2 s apart and
        the window is shorter than that, at most one target can be open at a
        time and the attribution is never ambiguous.

        `out` is filled in the finally block so a summary survives an abort.
        """
        sym_stim, dot_stim, mask_stim, text = stim
        soa, sdur = soa_of(p), p["symbol_dur"]
        window = p["response_window"]
        n_pres = len(stream)
        # How many presentations back a still-open response window can reach.
        lookback = int(np.ceil(window / soa)) + 1

        events_log = []
        event_counter = [0]

        def emit(ev):
            event_counter[0] += 1
            ev["event_num"] = event_counter[0]
            ev["phase"] = phase
            ev["datetime"] = datetime.now().isoformat(timespec="milliseconds")
            log_event(ev)
            events_log.append(ev)

        def presentation_row(s, etype):
            ev = blank_event(etype)
            ev["presentation_num"] = s["idx"]
            ev["onset_s"] = f'{s["onset"]:.4f}'
            ev["position"] = "left" if s["position"] == LEFT else "right"
            ev["symbol"] = s["symbol"]
            ev["prev_same_pos_symbol"] = s["prev_same"]
            ev["prev_other_pos_symbol"] = s["prev_other"]
            ev["same_pos_lag"] = s["same_pos_lag"]
            return ev

        n_hits = n_fa = n_dup = n_other = n_lure_fa = n_dist_fa = 0
        n_frames = 0
        rts = []
        rt_times = []

        # Onsets of lures and of same-position repeats that are not targets,
        # so a false alarm can be attributed to the thing that likely caused it.
        lure_onsets = [s["onset"] for s in stream if s["is_lure"]]
        dist_onsets = [s["onset"] for s in stream if s["is_dist_repeat"]]

        if feedback is None:
            feedback = Feedback(dict(audio_feedback=dict(enabled=False)))
        # Targets in onset order, walked by a pointer so a miss tone can fire
        # the moment a window closes unanswered rather than at stream end.
        target_idxs = [s["idx"] for s in stream if s["is_target"]]
        miss_ptr = 0
        miss_toned = set()

        kb.clearEvents()
        win.frameIntervals = []
        clock = core.Clock()
        completed = False
        last_drawn = None

        try:
            while clock.getTime() < duration:
                n_frames += 1
                t = clock.getTime()
                i = int(t // soa)
                if i >= n_pres:
                    break
                phase_t = t - i * soa
                showing = phase_t < sdur
                cur = stream[i]
                pos = cur["position"]

                # --- draw -------------------------------------------------
                for k in (LEFT, RIGHT):
                    if k == pos and showing:
                        sym_stim[k].text = cur["symbol"]
                        sym_stim[k].draw()
                    else:
                        if p["use_mask"]:
                            mask_stim[k].draw()
                        if p["show_placeholder"]:
                            dot_stim[k].draw()
                win.flip()

                # --- responses --------------------------------------------
                keys = kb.getKeys(waitRelease=False)
                for k in keys:
                    if k.name == "escape":
                        raise AbortExperiment()
                    if k.name != p["response_key"]:
                        n_other += 1
                        ev = blank_event("Other Key")
                        ev["key_name"] = k.name
                        ev["press_time_s"] = f"{clock.getTime():.4f}"
                        emit(ev)
                        continue

                    # Timestamp from the keyboard where it is trustworthy.
                    t_now = clock.getTime()
                    tk = getattr(k, "tDown", None)
                    if tk is not None and np.isfinite(tk):
                        # kb timestamps share the core clock; convert to stream time
                        t_press = t_now - (core.getTime() - tk)
                        if not (0.0 <= t_press <= t_now + 0.001):
                            t_press = t_now
                    else:
                        t_press = t_now

                    # Most recent target with an open window.
                    j = int(t_press // soa)
                    hit_on = None
                    for cand in range(j, max(-1, j - lookback - 1), -1):
                        if cand < 0 or cand >= n_pres:
                            continue
                        s = stream[cand]
                        if s["is_target"] and 0 <= t_press - s["onset"] <= window:
                            hit_on = s
                            break

                    if hit_on is None:
                        near = any(0 <= t_press - lo <= window for lo in lure_onsets)
                        near_d = any(0 <= t_press - lo <= window for lo in dist_onsets)
                        if near:
                            n_lure_fa += 1
                        if near_d:
                            n_dist_fa += 1
                        n_fa += 1
                        ev = blank_event("False Alarm")
                        ev["key_name"] = p["response_key"]
                        ev["press_time_s"] = f"{t_press:.4f}"
                        ev["near_lure"] = int(near)
                        ev["near_dist_repeat"] = int(near_d)
                        ev["feedback_tone"] = feedback.play("false_alarm", phase)
                        emit(ev)
                    elif hit_on["responded"]:
                        n_dup += 1
                        ev = blank_event("Duplicate Press")
                        ev["key_name"] = p["response_key"]
                        ev["press_time_s"] = f"{t_press:.4f}"
                        ev["attributed_to_presentation"] = hit_on["idx"]
                        emit(ev)
                    else:
                        hit_on["responded"] = True
                        rt = (t_press - hit_on["onset"]) * 1000.0
                        hit_on["rt"] = rt
                        rts.append(rt)
                        rt_times.append(hit_on["onset"])
                        n_hits += 1
                        ev = presentation_row(hit_on, "Hit")
                        ev["key_name"] = p["response_key"]
                        ev["press_time_s"] = f"{t_press:.4f}"
                        ev["rt_ms"] = f"{rt:.1f}"
                        ev["responded"] = 1
                        ev["attributed_to_presentation"] = hit_on["idx"]
                        ev["presentations_after_target"] = int(
                            (t_press - hit_on["onset"]) // soa)
                        ev["feedback_tone"] = feedback.play("hit", phase)
                        emit(ev)

                # A target whose window has closed with no press is a miss now,
                # not at stream end, so the tone lands while it still means
                # something to the participant.
                if feedback.wanted("miss", phase):
                    while miss_ptr < len(target_idxs):
                        s = stream[target_idxs[miss_ptr]]
                        if t < s["onset"] + window:
                            break
                        if not s["responded"] and s["idx"] not in miss_toned:
                            feedback.play("miss", phase)
                            miss_toned.add(s["idx"])
                        miss_ptr += 1

                last_drawn = i

            completed = True

        finally:
            elapsed = clock.getTime()
            shown = (last_drawn + 1) if last_drawn is not None else 0

            # Targets that were presented but never answered.
            n_targets = 0
            n_misses = 0
            for s in stream[:shown]:
                if not s["is_target"]:
                    continue
                n_targets += 1
                if not s["responded"]:
                    n_misses += 1
                    ev = presentation_row(s, "Miss")
                    ev["responded"] = 0
                    ev["feedback_tone"] = "miss" if s["idx"] in miss_toned else ""
                    emit(ev)

            # One row per lure, so lure-driven false alarms can be matched up.
            n_lures = 0
            for s in stream[:shown]:
                if s["is_lure"]:
                    n_lures += 1
                    ev = presentation_row(s, "Lure")
                    emit(ev)

            ev = blank_event("Stream End" if completed else "Stream Aborted")
            ev["press_time_s"] = f"{elapsed:.4f}"
            emit(ev)

            try:
                thresh = getattr(win, "refreshThreshold", None) or \
                    (1.0 / p["frmrate_expected"] * 1.5)
                n_dropped = int(sum(1 for x in win.frameIntervals if x > thresh))
            except Exception:
                n_dropped = ""

            # Split the run in half by target onset so a decline over time is
            # visible straight from the block row.
            half = elapsed / 2.0 if elapsed else 0.0
            t_h1 = [s for s in stream[:shown] if s["is_target"] and s["onset"] < half]
            t_h2 = [s for s in stream[:shown] if s["is_target"] and s["onset"] >= half]
            def _hr(ts):
                return (sum(1 for x in ts if x["responded"]) / len(ts)) if ts else ""
            rt_h1 = [x["rt"] for x in t_h1 if x["rt"] is not None]
            rt_h2 = [x["rt"] for x in t_h2 if x["rt"] is not None]
            n_dist = sum(1 for s in stream[:shown] if s["is_dist_repeat"])

            summary = dict(
                phase=phase, elapsed=elapsed, n_presentations=shown,
                hit_rate_h1=_hr(t_h1), hit_rate_h2=_hr(t_h2),
                mean_rt_h1=(float(np.mean(rt_h1)) if rt_h1 else ""),
                mean_rt_h2=(float(np.mean(rt_h2)) if rt_h2 else ""),
                n_dist_repeats=n_dist, n_dist_fa=n_dist_fa,
                n_frames=n_frames, n_targets=n_targets, n_hits=n_hits,
                n_misses=n_misses, n_lures=n_lures, n_lure_fa=n_lure_fa,
                n_fa=n_fa, n_dup=n_dup, n_other=n_other,
                hit_rate=(n_hits / n_targets) if n_targets else "",
                mean_rt=(float(np.mean(rts)) if rts else ""),
                median_rt=(float(np.median(rts)) if rts else ""),
                time_at_risk=max(0.0, elapsed - n_targets * p["response_window"]),
                n_dropped=n_dropped, completed=completed, events=events_log,
                audio_feedback=int(bool(
                    feedback.ok and (feedback.wanted("hit", phase)
                                     or feedback.wanted("false_alarm", phase)
                                     or feedback.wanted("miss", phase)))),
            )
            if out is not None:
                out.update(summary)

        return summary


    def show_text(win, text, body, wait_key=True, secs=None):
        text.text = body
        text.draw()
        win.flip()
        if wait_key:
            keys = event.waitKeys()
            if keys and "escape" in keys:
                raise AbortExperiment()
        elif secs:
            core.wait(secs)


    # ------------------------------------------------------------------ #

    def main():
        bad = check_timing(P)
        if bad:
            print("\n--- TIMING PARAMETERS REJECTED ---")
            for b in bad:
                print("   ", b)
            print("----------------------------------\n")
            sys.exit(1)

        print_timing(P)

        subj, do_practice, screen_w_cm, view_dist_cm = get_subject_info()
        seed = subj
        rng = np.random.default_rng(seed)
        run_id = uuid.uuid4().hex[:8]

        mon = monitors.Monitor("expMonitor", width=screen_w_cm, distance=view_dist_cm)
        win = visual.Window(fullscr=True, color=rgb255_to_psychopy(P["bkgd_col"]),
                            units="pix", allowGUI=False, monitor=mon)
        try:
            mon.setSizePix(list(win.size))
        except Exception:
            pass

        kb = keyboard.Keyboard()
        event.globalKeys.clear()

        measured = win.getActualFrameRate(nIdentical=20, nMaxFrames=100,
                                          nWarmUpFrames=10, threshold=1)
        P["frmrate_measured"] = measured if measured else P["frmrate_expected"]
        win.recordFrameIntervals = True
        win.refreshThreshold = (1.0 / P["frmrate_measured"]) + 0.004

        mon_info = dict(width_cm=screen_w_cm, distance_cm=view_dist_cm,
                        size_pix=list(win.size))
        write_sidecar(subj, seed, run_id, mon_info, P["frmrate_measured"])

        stim = build_stimuli(win, P)
        text = stim[3]
        feedback = Feedback(P)
        if P["audio_feedback"]["enabled"] and not feedback.ok:
            print("[audio feedback requested but unavailable — "
                  "the run will continue without it]")
        ev_file, ev_writer, bk_file, bk_writer = make_writers(subj)

        instr = (
            "Visual Working Memory: Repeat Detection\n\n"
            "Letters and numbers will appear one at a time, alternating between\n"
            "a LEFT and a RIGHT position.\n\n"
            "Press the SPACEBAR as soon as a symbol repeats "
            "AT THE SAME POSITION\nas the symbol shown there before it.\n\n"
            "A symbol that repeats at the OTHER position is not a target — "
            "ignore it.\n\n"
            "Respond as quickly as you can.\n\n"
            "Press any key to begin."
        )

        instr_clock = core.Clock()
        show_text(win, text, instr)
        instr_secs = instr_clock.getTime()

        exp_start = core.getTime()
        try:
            # Practice may run more than once if the participant has clearly
            # not grasped the rule; each repeat gets a fresh sequence.
            phases = []
            if do_practice:
                for k in range(max(1, int(P["practice_max_runs"]))):
                    phases.append(("practice", P["practice_duration"],
                                   np.random.default_rng(
                                       seed + P["practice_seed_offset"] + k)))
            phases.append(("test", P["stream_duration"], rng))

            practice_done = 0
            n_practice_total = sum(1 for x in phases if x[0] == "practice")
            for phase, duration, prng in phases:
                if phase == "practice":
                    if practice_done >= n_practice_total:
                        continue          # criterion already met
                    practice_done += 1
                n_pres = int(duration // soa_of(P))
                stream = build_sequence(prng, n_pres, P)
                problems = validate_sequence(stream, P)
                if problems:
                    print("--- SEQUENCE WARNINGS ---")
                    for x in problems:
                        print("   ", x)
                write_stream_record(subj, run_id, phase, stream)

                if phase == "practice":
                    show_text(win, text,
                              "Practice: about 45 seconds.\n\n"
                              "Press any key to start.")
                else:
                    show_text(win, text,
                              "Now the main task: 5 minutes without a break.\n\n"
                              "Press any key to start.")

                def log_event(ev, _phase=phase):
                    ev["run_id"] = run_id
                    ev["subject"] = subj
                    ev["practice"] = int(_phase == "practice")
                    ev["seed"] = seed
                    ev_writer.writerow([ev.get(k, "") for k in EVENT_FIELDS])
                    ev_file.flush()

                result = {}
                try:
                    run_stream(win, kb, stim, P, stream, duration, phase,
                               log_event, feedback=feedback, out=result)
                finally:
                    if result:
                        bk_writer.writerow([
                            datetime.now().isoformat(timespec="milliseconds"),
                            run_id, subj, int(phase == "practice"), seed, phase,
                            f'{result["elapsed"]:.3f}', result["n_presentations"],
                            result["n_frames"], result["n_targets"],
                            result["n_hits"], result["n_misses"],
                            result["n_lures"], result["n_lure_fa"],
                            result["n_fa"], result["n_dup"], result["n_other"],
                            (f'{result["hit_rate"]:.4f}'
                             if result["hit_rate"] != "" else ""),
                            (f'{result["mean_rt"]:.1f}'
                             if result["mean_rt"] != "" else ""),
                            (f'{result["median_rt"]:.1f}'
                             if result["median_rt"] != "" else ""),
                            (f'{result["hit_rate_h1"]:.4f}'
                             if result["hit_rate_h1"] != "" else ""),
                            (f'{result["hit_rate_h2"]:.4f}'
                             if result["hit_rate_h2"] != "" else ""),
                            (f'{result["mean_rt_h1"]:.1f}'
                             if result["mean_rt_h1"] != "" else ""),
                            (f'{result["mean_rt_h2"]:.1f}'
                             if result["mean_rt_h2"] != "" else ""),
                            result["n_dist_repeats"], result["n_dist_fa"],
                            (f'{result["n_fa"] / (result["elapsed"] / 60.0):.2f}'
                             if result["elapsed"] > 0 else ""),
                            f'{result["time_at_risk"]:.3f}',
                            result["audio_feedback"],
                            f'{P["frmrate_measured"]:.3f}',
                            result["n_dropped"], int(result["completed"]),
                        ])
                        bk_file.flush()

                # Summary after practice only — nothing interrupts the test stream.
                if phase == "practice" and result:
                    hr = result["hit_rate"]
                    n_practice = sum(1 for x in phases if x[0] == "practice")
                    retry = (hr != "" and hr < P["practice_pass_hit_rate"]
                             and practice_done < n_practice)
                    tail = ("It looks like the rule may not be clear yet, so we "
                            "will run the practice once more.\n\n"
                            "Remember: only a repeat at the SAME position counts."
                            if retry else "")
                    show_text(win, text,
                              "Practice complete.\n\n"
                              f"Repeats detected: {result['n_hits']} "
                              f"of {result['n_targets']}\n"
                              f"Missed: {result['n_misses']}\n"
                              f"Presses with no repeat: {result['n_fa']}\n\n"
                              + tail + "\n\nPress any key to continue.")
                    if not retry:
                        # Good enough — drop any remaining practice runs.
                        practice_done = n_practice

            show_text(win, text,
                      "End of Task\n\nThank you very much.\n\n"
                      "Please get the experimenter.")

        except AbortExperiment:
            print("\n--- ABORTED BY EXPERIMENTER (escape) ---\n")
        except Exception:
            print("\n--- RUNTIME ERROR ---")
            traceback.print_exc()
            print("---------------------\n")
        finally:
            try:
                fname = os.path.join(DATA_DIR, f"VWM_s{subj}_{run_id}_params.json")
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
