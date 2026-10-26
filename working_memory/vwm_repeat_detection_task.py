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
        # --- Timing ---
        # Memory interval = target_lag x (symbol_dur + blank_dur).
        # blank_dur is the manipulated variable; symbol_dur stays fixed.
        symbol_dur=0.5,
        blank_dur=0.5,
        stream_duration=300.0,
        practice_duration=45.0,
        run_practice=True,
        practice_max_runs=2,          # 1 disables the practice retry
        practice_pass_hit_rate=0.5,

        # --- Response ---
        response_key="space",
        response_window=1.5,          # seconds after target onset that count as a hit

        # --- Sequence ---
        target_lag=1,                 # 1 = same as previous symbol, 2 = same as two back
        targets_per_min=10.0,
        # "per_minute": fixed targets per minute; "per_presentation": fixed
        # chance per symbol. The two diverge when blank_dur changes.
        target_rate_mode="per_minute",
        min_repeat_gap=10,            # non-targets never repeat within this many symbols
        min_event_gap_s=2.0,
        first_event_min_s=6.0,
        # Visually confusable characters (B G I L O Q S Z 0 1 2 5 6 8) excluded.
        symbols="ACDEFHJKMNPRTUVWXY3479",

        # --- Display ---
        symbol_height=64,
        symbol_font="Arial",
        symbol_col=(0, 0, 0),
        bkgd_col=(160, 160, 160),
        show_placeholder=True,        # fixation dot during blanks
        placeholder_radius=5,
        placeholder_col=(90, 90, 90),
        use_mask=False,               # low-contrast mask during blanks
        mask_radius=42,
        mask_col=(140, 140, 140),

        frmrate_expected=60,
        practice_seed_offset=100000,
    )

    DATA_DIR = "data"
    FILE_PREFIX = "VWMcentre"

    EVENT_FIELDS = [
        "datetime", "run_id", "subject", "practice", "seed", "phase",
        "event_num", "event_type",
        "presentation_num", "onset_s", "symbol", "nback_symbol",
        "key_name", "press_time_s", "rt_ms", "responded",
        "attributed_to_presentation", "presentations_after_target",
        "feedback_tone",
    ]

    BLOCK_FIELDS = [
        "datetime", "run_id", "subject", "practice", "seed", "phase",
        "stream_duration_s", "n_presentations", "n_frames",
        "n_targets", "n_hits", "n_misses",
        "n_false_alarms", "n_duplicate_presses", "n_other_keys",
        "hit_rate", "mean_rt_ms", "median_rt_ms",
        "hit_rate_h1", "hit_rate_h2", "mean_rt_h1_ms", "mean_rt_h2_ms",
        "fa_per_min", "time_at_risk_s", "audio_feedback",
        "frmrate_measured", "n_dropped_frames", "completed",
    ]


    class AbortExperiment(Exception):
        """Raised on escape so data is saved before quitting."""
        pass


    def rgb255_to_psychopy(rgb):
        return [(c / 127.5) - 1 for c in rgb]


    def blank_event(event_type):
        ev = {k: "" for k in EVENT_FIELDS}
        ev["event_type"] = event_type
        return ev


    def soa_of(p):
        return p["symbol_dur"] + p["blank_dur"]


    def retention_interval(p):
        return int(p["target_lag"]) * soa_of(p)


    def no_repeat_window(p):
        # Never less than target_lag, so fillers cannot create accidental targets.
        return max(int(p["min_repeat_gap"]), int(p["target_lag"]))


    def min_gap_presentations(p):
        # Also guarantees that response windows of successive targets never overlap.
        soa = soa_of(p)
        return max(int(p["target_lag"]) + 1,
                   int(np.ceil(p["min_event_gap_s"] / soa)),
                   int(np.ceil(p["response_window"] / soa)))


    def check_params(p):
        problems = []
        if p["blank_dur"] <= 0 or p["symbol_dur"] <= 0:
            problems.append("symbol_dur and blank_dur must both be positive")
        if p["response_window"] <= 0:
            problems.append("response_window must be positive")
        if int(p["target_lag"]) < 1:
            problems.append("target_lag must be 1 or more")
        n_sym = len(set(p["symbols"]))
        if no_repeat_window(p) >= n_sym:
            problems.append(
                f"each symbol must differ from the previous {no_repeat_window(p)}, "
                f"but the pool only has {n_sym} distinct symbols — lower "
                f"min_repeat_gap or add symbols")
        return problems


    def timing_summary(p):
        soa = soa_of(p)
        w = p["response_window"]
        return dict(
            symbol_dur=p["symbol_dur"],
            blank_dur=p["blank_dur"],
            soa=soa,
            target_lag=int(p["target_lag"]),
            retention_interval=retention_interval(p),
            min_repeat_gap=int(p["min_repeat_gap"]),
            symbols_per_min=60.0 / soa,
            response_window=w,
            window_past_next_symbol=w - soa,
            min_target_spacing=min_gap_presentations(p) * soa,
            n_presentations=int(p["stream_duration"] // soa),
        )


    def print_timing(p):
        s = timing_summary(p)
        print("\n--- timing ---")
        print(f"  display {s['symbol_dur']:.2f}s + blank {s['blank_dur']:.2f}s "
              f"= SOA {s['soa']:.2f}s   ({s['symbols_per_min']:.0f} symbols/min, "
              f"{s['n_presentations']} in the run)")
        print(f"  target = repeat of the symbol {s['target_lag']} back; "
              f"memory interval {s['retention_interval']:.2f}s")
        print(f"  no other symbol repeats within {no_repeat_window(p)} symbols")
        print(f"  response window {s['response_window']:.2f}s; targets at "
              f"least {s['min_target_spacing']:.2f}s apart")
        if s["window_past_next_symbol"] > 0:
            print(f"  NOTE: the window runs {s['window_past_next_symbol']:.2f}s "
                  f"past the next symbol's onset, so a late response can overlap "
                  f"the following symbol. Lengthen blank_dur to buy room.")
        else:
            print("  the window closes before the next symbol appears")
        print("---------------\n")


    def rule_text(p):
        lag = int(p["target_lag"])
        if lag == 1:
            return "the same as the one just before it"
        if lag == 2:
            return "the same as the one TWO before it"
        return f"the same as the one {lag} before it"


    def data_path(subj, name):
        # Each participant's files go in data/<participant ID>/
        folder = os.path.join(DATA_DIR, str(subj))
        os.makedirs(folder, exist_ok=True)
        return os.path.join(folder, f"{FILE_PREFIX}_{name}")


    # ------------------------------------------------------------------ #
    #  Sequence construction                                             #
    # ------------------------------------------------------------------ #

    def build_sequence(rng, n_pres, p):
        soa = soa_of(p)
        lag = int(p["target_lag"])
        min_gap = min_gap_presentations(p)
        no_repeat = no_repeat_window(p)
        first_ok = max(lag, int(np.ceil(p["first_event_min_s"] / soa)))

        if p["target_rate_mode"] == "per_presentation":
            n_targets = int(round(p["targets_per_min"] / 60.0 * n_pres))
        else:
            stream_min = (n_pres * soa) / 60.0
            n_targets = int(round(p["targets_per_min"] * stream_min))

        span = n_pres - first_ok
        if n_targets > 0 and span // min_gap < n_targets:
            n_targets = max(0, int(span // min_gap) - 1)

        # Geometric gaps above the floor give a flat hazard, so target timing
        # cannot be anticipated.
        mean_gap = span / n_targets if n_targets else span
        extra = max(0.0, mean_gap - min_gap)
        p_geom = 1.0 / (1.0 + extra) if extra > 0 else 1.0

        slots = set()
        i = first_ok + int(rng.integers(0, min_gap + 1))
        while i < n_pres and len(slots) < n_targets:
            slots.add(int(i))
            i += min_gap + int(rng.geometric(p_geom)) - 1

        syms = list(dict.fromkeys(p["symbols"]))
        seq_sym = [None] * n_pres
        for i in range(n_pres):
            if i in slots:
                seq_sym[i] = seq_sym[i - lag]
            else:
                banned = set(seq_sym[max(0, i - no_repeat):i])
                choices = [s for s in syms if s not in banned]
                seq_sym[i] = choices[int(rng.integers(len(choices)))]

        last_seen = {}
        repeat_lag = [0] * n_pres
        for i, s in enumerate(seq_sym):
            if s in last_seen:
                repeat_lag[i] = i - last_seen[s]
            last_seen[s] = i

        stream = []
        for i in range(n_pres):
            nback = seq_sym[i - lag] if i >= lag else ""
            stream.append(dict(
                idx=i,
                onset=i * soa,
                symbol=seq_sym[i],
                is_target=(i >= lag and seq_sym[i] == nback),
                nback_symbol=nback,
                repeat_lag=repeat_lag[i],
                responded=False,
                rt=None,
            ))
        return stream


    def validate_sequence(stream, p):
        problems = []
        soa = soa_of(p)
        lag = int(p["target_lag"])
        min_gap = min_gap_presentations(p)
        no_repeat = no_repeat_window(p)

        targets = [s["idx"] for s in stream if s["is_target"]]
        for a, b in zip(targets, targets[1:]):
            if b - a < min_gap:
                problems.append(f"targets {a} and {b} are only {b - a} apart")
        if targets and targets[0] * soa < p["first_event_min_s"]:
            problems.append(f"first target at {targets[0] * soa:.1f}s, "
                            f"before {p['first_event_min_s']}s")

        for s in stream:
            if s["is_target"]:
                if s["repeat_lag"] != lag:
                    problems.append(f"target {s['idx']} repeats at lag "
                                    f"{s['repeat_lag']}, not {lag}")
            elif 0 < s["repeat_lag"] <= no_repeat:
                problems.append(f"presentation {s['idx']} ({s['symbol']}) "
                                f"repeats after only {s['repeat_lag'] - 1} "
                                f"other symbols")
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
        dlg = gui.DlgFromDict(info, title="VWM Repeat Detection Task (centre)")
        if not dlg.OK:
            core.quit()
        return (int(info["Subject Number"]), bool(info["Run practice first"]),
                float(info["Screen width (cm)"]), float(info["Viewing distance (cm)"]))


    def make_writers(subj):
        ev_name = data_path(subj, f"s{subj}_events.csv")
        ev_new = not os.path.exists(ev_name)
        ev_file = open(ev_name, "a", newline="")
        ev_writer = csv.writer(ev_file)
        if ev_new:
            ev_writer.writerow(EVENT_FIELDS)

        bk_name = data_path(subj, f"s{subj}_blocks.csv")
        bk_new = not os.path.exists(bk_name)
        bk_file = open(bk_name, "a", newline="")
        bk_writer = csv.writer(bk_file)
        if bk_new:
            bk_writer.writerow(BLOCK_FIELDS)

        return ev_file, ev_writer, bk_file, bk_writer


    def write_stream_record(subj, run_id, phase, stream):
        fname = data_path(subj, f"s{subj}_{run_id}_{phase}_sequence.csv")
        with open(fname, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["presentation_num", "onset_s", "symbol", "is_target",
                        "nback_symbol", "repeat_lag"])
            for s in stream:
                w.writerow([s["idx"], f'{s["onset"]:.3f}', s["symbol"],
                            int(s["is_target"]), s["nback_symbol"],
                            s["repeat_lag"]])


    def write_sidecar(subj, seed, run_id, mon_info, frmrate):
        payload = dict(
            run_id=run_id, subject=subj, seed=int(seed),
            started=datetime.now().isoformat(timespec="milliseconds"),
            monitor=mon_info, frmrate_measured=frmrate,
            versions=dict(python=sys.version.split()[0], numpy=np.__version__,
                          platform=platform.platform()),
            params={k: (list(v) if isinstance(v, tuple) else v) for k, v in P.items()},
            timing=timing_summary(P),
            derived_timing=dict(
                soa_s=soa_of(P),
                retention_interval_s=retention_interval(P),
                symbols_per_min=60.0 / soa_of(P),
                n_presentations=int(P["stream_duration"] // soa_of(P)),
                min_event_gap_presentations=min_gap_presentations(P),
                no_repeat_window=no_repeat_window(P),
            ),
        )
        try:
            from psychopy import __version__ as psychopy_version
            payload["versions"]["psychopy"] = psychopy_version
        except Exception:
            pass
        fname = data_path(subj, f"s{subj}_{run_id}_params.json")
        with open(fname, "w") as f:
            json.dump(payload, f, indent=2, default=str)


    class Feedback:
        """Feedback tones. Every call is a silent no-op if audio is unavailable."""

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

            # Warm up the backend so the first real tone does not drop a frame.
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
            if not self.wanted(kind, phase):
                return ""
            s = self.tones.get(kind)
            if s is None:
                return ""
            try:
                s.stop()
            except Exception:
                pass
            try:
                s.play()
                return kind
            except Exception:
                return ""


    def build_stimuli(win, p):
        sym = visual.TextStim(win, text="", height=p["symbol_height"],
                              font=p["symbol_font"],
                              color=rgb255_to_psychopy(p["symbol_col"]),
                              pos=(0, 0), alignText="center", anchorHoriz="center")
        dot = visual.Circle(win, radius=p["placeholder_radius"],
                            fillColor=rgb255_to_psychopy(p["placeholder_col"]),
                            lineColor=None, pos=(0, 0))
        mask = visual.Circle(win, radius=p["mask_radius"],
                             fillColor=rgb255_to_psychopy(p["mask_col"]),
                             lineColor=None, pos=(0, 0))
        text = visual.TextStim(win, text="", color="black", height=24, wrapWidth=900)
        return sym, dot, mask, text


    # ------------------------------------------------------------------ #
    #  The stream                                                        #
    # ------------------------------------------------------------------ #

    def run_stream(win, kb, stim, p, stream, duration, phase, log_event,
                   feedback=None, out=None):
        sym_stim, dot_stim, mask_stim, text = stim
        soa, sdur = soa_of(p), p["symbol_dur"]
        window = p["response_window"]
        n_pres = len(stream)
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
            ev["symbol"] = s["symbol"]
            ev["nback_symbol"] = s["nback_symbol"]
            return ev

        n_hits = n_fa = n_dup = n_other = 0
        n_frames = 0
        rts = []

        if feedback is None:
            feedback = Feedback(dict(audio_feedback=dict(enabled=False)))
        target_idxs = [s["idx"] for s in stream if s["is_target"]]
        miss_ptr = 0
        miss_toned = set()

        kb.clearEvents()
        win.frameIntervals = []
        clock = core.Clock()
        completed = False
        last_drawn = None
        loaded_symbol = None

        try:
            # Driven by elapsed time rather than frame count, so the schedule
            # cannot drift.
            while clock.getTime() < duration:
                n_frames += 1
                t = clock.getTime()
                i = int(t // soa)
                if i >= n_pres:
                    break
                showing = (t - i * soa) < sdur
                cur = stream[i]

                if showing:
                    # Re-rendering text every frame can drop frames; only
                    # update it when the symbol changes.
                    if loaded_symbol != cur["symbol"]:
                        sym_stim.text = cur["symbol"]
                        loaded_symbol = cur["symbol"]
                    sym_stim.draw()
                else:
                    if p["use_mask"]:
                        mask_stim.draw()
                    if p["show_placeholder"]:
                        dot_stim.draw()
                win.flip()

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

                    # Keyboard timestamps share the core clock; convert to
                    # stream time.
                    t_now = clock.getTime()
                    tk = getattr(k, "tDown", None)
                    if tk is not None and np.isfinite(tk):
                        t_press = t_now - (core.getTime() - tk)
                        if not (0.0 <= t_press <= t_now + 0.001):
                            t_press = t_now
                    else:
                        t_press = t_now

                    # Attribute the press to the most recent target whose
                    # response window is still open.
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
                        n_fa += 1
                        ev = blank_event("False Alarm")
                        ev["key_name"] = p["response_key"]
                        ev["press_time_s"] = f"{t_press:.4f}"
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

            ev = blank_event("Stream End" if completed else "Stream Aborted")
            ev["press_time_s"] = f"{elapsed:.4f}"
            emit(ev)

            try:
                thresh = getattr(win, "refreshThreshold", None) or \
                    (1.0 / p["frmrate_expected"] * 1.5)
                n_dropped = int(sum(1 for x in win.frameIntervals if x > thresh))
            except Exception:
                n_dropped = ""

            half = elapsed / 2.0 if elapsed else 0.0
            t_h1 = [s for s in stream[:shown] if s["is_target"] and s["onset"] < half]
            t_h2 = [s for s in stream[:shown] if s["is_target"] and s["onset"] >= half]
            def _hr(ts):
                return (sum(1 for x in ts if x["responded"]) / len(ts)) if ts else ""
            rt_h1 = [x["rt"] for x in t_h1 if x["rt"] is not None]
            rt_h2 = [x["rt"] for x in t_h2 if x["rt"] is not None]

            summary = dict(
                phase=phase, elapsed=elapsed, n_presentations=shown,
                hit_rate_h1=_hr(t_h1), hit_rate_h2=_hr(t_h2),
                mean_rt_h1=(float(np.mean(rt_h1)) if rt_h1 else ""),
                mean_rt_h2=(float(np.mean(rt_h2)) if rt_h2 else ""),
                n_frames=n_frames, n_targets=n_targets, n_hits=n_hits,
                n_misses=n_misses, n_fa=n_fa, n_dup=n_dup, n_other=n_other,
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
        bad = check_params(P)
        if bad:
            print("\n--- PARAMETERS REJECTED ---")
            for b in bad:
                print("   ", b)
            print("---------------------------\n")
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
            "Letters and numbers will appear one at a time\n"
            "in the centre of the screen.\n\n"
            "Press the SPACEBAR as soon as a symbol is\n"
            f"{rule_text(P)}.\n\n"
            "Respond as quickly as you can.\n\n"
            "Press any key to begin."
        )

        instr_clock = core.Clock()
        show_text(win, text, instr)
        instr_secs = instr_clock.getTime()

        exp_start = core.getTime()
        try:
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
                        continue
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
                            (f'{result["n_fa"] / (result["elapsed"] / 60.0):.2f}'
                             if result["elapsed"] > 0 else ""),
                            f'{result["time_at_risk"]:.3f}',
                            result["audio_feedback"],
                            f'{P["frmrate_measured"]:.3f}',
                            result["n_dropped"], int(result["completed"]),
                        ])
                        bk_file.flush()

                if phase == "practice" and result:
                    hr = result["hit_rate"]
                    n_practice = sum(1 for x in phases if x[0] == "practice")
                    retry = (hr != "" and hr < P["practice_pass_hit_rate"]
                             and practice_done < n_practice)
                    tail = ("It looks like the rule may not be clear yet, so we "
                            "will run the practice once more.\n\n"
                            f"Remember: press only when a symbol is "
                            f"{rule_text(P)}."
                            if retry else "")
                    show_text(win, text,
                              "Practice complete.\n\n"
                              f"Repeats detected: {result['n_hits']} "
                              f"of {result['n_targets']}\n"
                              f"Missed: {result['n_misses']}\n"
                              f"Presses with no repeat: {result['n_fa']}\n\n"
                              + tail + "\n\nPress any key to continue.")
                    if not retry:
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
                fname = data_path(subj, f"s{subj}_{run_id}_params.json")
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