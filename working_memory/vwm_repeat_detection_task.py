#!/usr/bin/env python3
# Visual working memory repeat-detection task (2-back).
# Letters and digits appear one at a time at the centre of the screen, each
# followed by a blank ISI drawn from a truncated normal distribution.
# Participants press SPACE when a symbol matches the one two before it.
# A short practice block (repeated once if performance is low) precedes a
# 15-minute test block. Data are saved to data/<subject>/.

import traceback
import sys
import bisect

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

    # ---- Parameters ----
    P = dict(
        # Timing (seconds). Each symbol is shown for symbol_dur, then a blank
        # ISI drawn from N(isi_mean, isi_sd); draws outside the ISI limits are
        # redrawn. isi_sd=0 gives a fixed ISI.
        symbol_dur=0.25,
        symbol_round_to_frames=True,  # snap symbol_dur to whole frames at the measured refresh rate
        isi_mean=2.0,
        isi_sd=0.3,
        isi_trunc_sd=1.0,       # ISI limits = isi_mean +/- isi_trunc_sd * isi_sd (follows the mean automatically)
        isi_min=None,           # only used if isi_trunc_sd is None: fixed limits in seconds
        isi_max=None,
        isi_round_to_frames=True,     # snap each ISI to whole frames
        stream_duration=900.0,
        practice_duration=45.0,
        run_practice=True,
        practice_max_runs=2,
        practice_pass_hit_rate=0.5,

        # Response
        response_key="space",
        response_window=1.5,
        late_response_window=1.0,     # presses this long after the hit window count as late, not FA

        # Sequence
        target_lag=2,                 # n in n-back
        targets_per_min=15.0,
        target_rate_mode="per_presentation",  # or "per_minute"
        min_repeat_gap=10,
        min_event_gap_s=2.0,
        first_event_min_s=6.0,
        symbols="ACDEFHJKMNPRTUVWXY3479",  # visually confusable characters excluded

        # Display (pixels, RGB 0-255)
        symbol_height=200,
        symbol_font="Arial",
        symbol_col=(0, 0, 0),
        bkgd_col=(160, 160, 160),
        show_placeholder=True,
        placeholder_radius=5,
        placeholder_col=(90, 90, 90),
        use_mask=False,
        mask_radius=84,
        mask_col=(140, 140, 140),

        frmrate_expected=60,
        practice_seed_offset=100000,
    )

    # ---- macOS pyglet fix ----
    # pyglet 1.5 on macOS can reuse a stale event wrapper and crash mid-run
    # ("ObjCInstance ... has no attribute b'type'"); this clears stale wrappers.
    def patch_pyglet_macos():
        if sys.platform != "darwin":
            return False
        try:
            from ctypes import c_void_p
            from pyglet.libs.darwin.cocoapy import runtime
            original_new = runtime.ObjCInstance.__new__

            def checked_new(cls, object_ptr):
                if not isinstance(object_ptr, c_void_p):
                    object_ptr = c_void_p(object_ptr)
                cached = cls._cached_objects.get(object_ptr.value)
                if (cached is not None and runtime.objc.object_getClass(object_ptr)
                        != cached.objc_class.ptr.value):
                    del cls._cached_objects[object_ptr.value]
                return original_new(cls, object_ptr)

            runtime.ObjCInstance.__new__ = staticmethod(checked_new)
            return True
        except Exception as e:
            print(f"[could not apply the macOS pyglet fix: {e}]")
            return False

    PYGLET_MAC_FIX = patch_pyglet_macos()


    def is_pyglet_mac_glitch(err):
        return isinstance(err, AttributeError) and "ObjCInstance" in str(err)

    # ---- Data files ----
    DATA_DIR = "data"
    FILE_PREFIX = "VWMcentre"

    EVENT_FIELDS = [
        "datetime", "run_id", "subject", "practice", "seed", "phase",
        "event_num", "event_type",
        "presentation_num", "onset_s", "symbol", "nback_symbol",
        "isi_before_s", "retention_s",
        "key_name", "press_time_s", "rt_ms", "responded", "late_response",
        "attributed_to_presentation", "presentations_after_target",
    ]

    BLOCK_FIELDS = [
        "datetime", "run_id", "subject", "practice", "seed", "phase",
        "stream_duration_s", "n_presentations", "n_frames",
        "n_targets", "n_hits", "n_misses", "n_late_responses",
        "n_false_alarms", "n_duplicate_presses", "n_other_keys",
        "hit_rate", "mean_rt_ms", "median_rt_ms", "mean_late_rt_ms",
        "hit_rate_h1", "hit_rate_h2", "mean_rt_h1_ms", "mean_rt_h2_ms",
        "fa_per_min", "time_at_risk_s",
        "frmrate_measured", "n_dropped_frames", "completed",
    ]


    # ---- Helpers ----
    class AbortExperiment(Exception):
        pass


    def rgb255_to_psychopy(rgb):
        return [(c / 127.5) - 1 for c in rgb]


    def blank_event(event_type):
        ev = {k: "" for k in EVENT_FIELDS}
        ev["event_type"] = event_type
        return ev


    def soa_of(p):
        return p["symbol_dur"] + p["isi_mean"]


    def min_soa(p):
        return p["symbol_dur"] + p["isi_min"]


    def max_soa(p):
        return p["symbol_dur"] + p["isi_max"]


    def retention_interval(p):
        return int(p["target_lag"]) * soa_of(p)


    def no_repeat_window(p):
        return max(int(p["min_repeat_gap"]), int(p["target_lag"]))


    def min_gap_presentations(p):
        soa = min_soa(p)
        return max(int(p["target_lag"]) + 1,
                   int(np.ceil(p["min_event_gap_s"] / soa)),
                   int(np.ceil(p["response_window"] / soa)))


    # ---- Timing and parameter checks ----
    def frame_dur(p):
        return 1.0 / float(p.get("frmrate_measured") or p["frmrate_expected"])


    def resolve_timing(p):
        """Work out the ISI limits and the frame-snapped symbol duration from
        the parameters. Called at startup and again once the refresh rate has
        been measured. The requested symbol duration is kept separately."""
        if p.get("isi_trunc_sd") is not None:
            k = float(p["isi_trunc_sd"])
            p["isi_min"] = float(p["isi_mean"]) - k * float(p["isi_sd"])
            p["isi_max"] = float(p["isi_mean"]) + k * float(p["isi_sd"])
        req = p.setdefault("symbol_dur_requested", p["symbol_dur"])
        if p.get("symbol_round_to_frames", False) and req > 0:
            fd = frame_dur(p)
            p["symbol_dur"] = max(1, int(round(req / fd))) * fd
        else:
            p["symbol_dur"] = req


    def check_params(p):
        problems = []
        if p["symbol_dur"] <= 0:
            problems.append("symbol_dur must be positive")
        if p["isi_sd"] < 0:
            problems.append("isi_sd cannot be negative")
        if p.get("isi_trunc_sd") is not None and p["isi_trunc_sd"] < 0:
            problems.append("isi_trunc_sd cannot be negative")
        if p["isi_min"] is None or p["isi_max"] is None:
            problems.append("set isi_trunc_sd, or set both isi_min and isi_max")
        elif not (0 < p["isi_min"] <= p["isi_mean"] <= p["isi_max"]):
            problems.append(
                f"ISI limits {p['isi_min']:.3f}-{p['isi_max']:.3f}s must satisfy "
                f"0 < isi_min <= isi_mean <= isi_max (with isi_trunc_sd, the "
                f"lower limit is isi_mean - isi_trunc_sd * isi_sd, so it must "
                f"stay above zero)")
        if p["response_window"] <= 0:
            problems.append("response_window must be positive")
        if p["late_response_window"] < 0:
            problems.append("late_response_window cannot be negative")
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
        lag = int(p["target_lag"])
        w = p["response_window"]
        return dict(
            symbol_dur=p["symbol_dur"],
            isi_mean=p["isi_mean"],
            isi_sd=p["isi_sd"],
            isi_min=p["isi_min"],
            isi_max=p["isi_max"],
            soa_mean=soa,
            soa_min=min_soa(p),
            soa_max=max_soa(p),
            target_lag=lag,
            retention_interval_mean=retention_interval(p),
            retention_interval_min=lag * min_soa(p),
            retention_interval_max=lag * max_soa(p),
            min_repeat_gap=int(p["min_repeat_gap"]),
            symbols_per_min=60.0 / soa,
            response_window=w,
            late_response_window=p["late_response_window"],
            window_past_next_symbol=w - min_soa(p),
            min_target_spacing=min_gap_presentations(p) * min_soa(p),
            n_presentations_approx=int(p["stream_duration"] // soa),
        )


    def print_timing(p):
        s = timing_summary(p)
        print("\n--- timing ---")
        fd = frame_dur(p)
        req = p.get("symbol_dur_requested", p["symbol_dur"])
        print(f"  refresh {1.0 / fd:.2f} Hz ({fd * 1000:.2f} ms/frame); symbol "
              f"{p['symbol_dur'] * 1000:.1f} ms "
              f"(~{p['symbol_dur'] / fd:.2f} frames)")
        if abs(req - p["symbol_dur"]) > 1e-6:
            print(f"  NOTE: symbol_dur {req * 1000:.1f} ms is not a whole number "
                  f"of frames, so it was snapped to {p['symbol_dur'] * 1000:.1f} ms")
        print(f"  display {s['symbol_dur']:.3f}s, then a blank ISI drawn from "
              f"N({s['isi_mean']:.2f}, {s['isi_sd']:.2f})s, kept within "
              f"{s['isi_min']:.2f}-{s['isi_max']:.2f}s")
        print(f"  SOA {s['soa_min']:.2f}-{s['soa_max']:.2f}s, mean "
              f"{s['soa_mean']:.2f}s   ({s['symbols_per_min']:.0f} symbols/min, "
              f"about {s['n_presentations_approx']} in the run)")
        print(f"  target = repeat of the symbol {s['target_lag']} back; memory "
              f"interval {s['retention_interval_min']:.2f}-"
              f"{s['retention_interval_max']:.2f}s, mean "
              f"{s['retention_interval_mean']:.2f}s")
        print(f"  no other symbol repeats within {no_repeat_window(p)} symbols")
        print(f"  response window {s['response_window']:.2f}s; targets at "
              f"least {s['min_target_spacing']:.2f}s apart")
        if s["late_response_window"] > 0:
            print(f"  presses {s['response_window']:.2f}-"
                  f"{s['response_window'] + s['late_response_window']:.2f}s "
                  f"after a target are logged as late responses")
        if s["window_past_next_symbol"] > 0:
            print(f"  NOTE: the window can run up to "
                  f"{s['window_past_next_symbol']:.2f}s past the next symbol's "
                  f"onset, so a late response can overlap the following symbol.")
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


    def duration_text(secs):
        if secs < 60:
            return f"{secs:g} seconds"
        mins = secs / 60
        return f"{mins:g} minute" + ("" if mins == 1 else "s")


    def data_path(subj, name):
        folder = os.path.join(DATA_DIR, str(subj))
        os.makedirs(folder, exist_ok=True)
        return os.path.join(folder, f"{FILE_PREFIX}_{name}")


    # ---- Sequence construction ----
    def draw_isis(rng, n, p):
        mean, sd = float(p["isi_mean"]), float(p["isi_sd"])
        lo, hi = float(p["isi_min"]), float(p["isi_max"])
        if sd <= 0 or hi <= lo:
            isis = np.full(n, mean)
        else:
            isis = rng.normal(mean, sd, n)
            bad = (isis < lo) | (isis > hi)
            while bad.any():
                isis[bad] = rng.normal(mean, sd, int(bad.sum()))
                bad = (isis < lo) | (isis > hi)
        if p.get("isi_round_to_frames", False):   # so logged ISIs match what was shown
            fd = 1.0 / float(p.get("frmrate_measured") or p["frmrate_expected"])
            frames = np.round(isis / fd)
            f_lo, f_hi = np.ceil(lo / fd - 1e-6), np.floor(hi / fd + 1e-6)
            if f_lo <= f_hi:
                frames = np.clip(frames, f_lo, f_hi)
            isis = frames * fd
        return isis


    def build_sequence(rng, duration, p):
        sdur = p["symbol_dur"]
        lag = int(p["target_lag"])
        min_gap = min_gap_presentations(p)
        no_repeat = no_repeat_window(p)

        n_max = int(np.ceil(duration / min_soa(p))) + 1
        isis = draw_isis(rng, n_max, p)
        onsets = np.concatenate([[0.0], np.cumsum(sdur + isis)[:-1]])
        ends = onsets + sdur + isis
        n_pres = int(np.sum(ends <= duration + 1e-9))
        onsets, ends, isis = onsets[:n_pres], ends[:n_pres], isis[:n_pres]
        stream_end = float(ends[-1]) if n_pres else 0.0

        first_ok = max(lag, int(np.searchsorted(
            onsets, p["first_event_min_s"] - 1e-9, side="left")))
        last_ok = int(np.searchsorted(
            onsets,
            stream_end - (p["response_window"] + p["late_response_window"]) + 1e-9,
            side="right")) - 1

        if p["target_rate_mode"] == "per_presentation":
            n_targets = int(round(p["targets_per_min"] / 60.0 * n_pres))
        else:
            n_targets = int(round(p["targets_per_min"] * stream_end / 60.0))

        span = last_ok + 1 - first_ok
        if span <= 0:
            n_targets = 0
        elif n_targets > 0 and span // min_gap < n_targets:
            n_targets = max(0, int(span // min_gap) - 1)

        mean_gap = span / n_targets if n_targets else span
        extra = max(0.0, mean_gap - min_gap)
        p_geom = 1.0 / (1.0 + extra) if extra > 0 else 1.0

        # Geometric gaps above min_gap give a flat hazard, so targets can't be anticipated.
        slots = set()
        i = first_ok + int(rng.integers(0, min_gap + 1))
        while i <= last_ok and len(slots) < n_targets:
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
                onset=float(onsets[i]),
                end=float(ends[i]),
                isi_after=float(isis[i]),
                isi_before=float(isis[i - 1]) if i > 0 else None,
                retention=float(onsets[i] - onsets[i - lag]) if i >= lag else None,
                symbol=seq_sym[i],
                is_target=(i >= lag and seq_sym[i] == nback),
                nback_symbol=nback,
                repeat_lag=repeat_lag[i],
                responded=False,
                rt=None,
                late_rt=None,
            ))
        return stream


    def validate_sequence(stream, p):
        problems = []
        lag = int(p["target_lag"])
        min_gap = min_gap_presentations(p)
        no_repeat = no_repeat_window(p)
        min_spacing_s = max(p["min_event_gap_s"], p["response_window"])

        targets = [s["idx"] for s in stream if s["is_target"]]
        for a, b in zip(targets, targets[1:]):
            if b - a < min_gap:
                problems.append(f"targets {a} and {b} are only {b - a} apart")
            gap_s = stream[b]["onset"] - stream[a]["onset"]
            if gap_s < min_spacing_s - 1e-9:
                problems.append(f"targets {a} and {b} are only {gap_s:.2f}s apart")
        if targets and stream[targets[0]]["onset"] < p["first_event_min_s"] - 1e-9:
            problems.append(f"first target at {stream[targets[0]]['onset']:.1f}s, "
                            f"before {p['first_event_min_s']}s")
        end = stream[-1]["end"] if stream else 0.0
        if targets and (stream[targets[-1]]["onset"] + p["response_window"]
                        + p["late_response_window"] > end + 1e-9):
            problems.append(f"last target at {stream[targets[-1]]['onset']:.1f}s "
                            f"cannot be fully answered before the stream ends "
                            f"at {end:.1f}s")

        for s in stream:
            if not (p["isi_min"] - 1e-6 <= s["isi_after"] <= p["isi_max"] + 1e-6):
                problems.append(f"presentation {s['idx']} has an ISI of "
                                f"{s['isi_after']:.3f}s, outside the limits")
            if s["is_target"]:
                if s["repeat_lag"] != lag:
                    problems.append(f"target {s['idx']} repeats at lag "
                                    f"{s['repeat_lag']}, not {lag}")
            elif 0 < s["repeat_lag"] <= no_repeat:
                problems.append(f"presentation {s['idx']} ({s['symbol']}) "
                                f"repeats after only {s['repeat_lag'] - 1} "
                                f"other symbols")
        return problems


    # ---- Setup ----
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


    def open_csv(path, header):
        if os.path.exists(path):
            with open(path, newline="") as f:
                existing = next(csv.reader(f), None)
            if existing != header:
                root, ext = os.path.splitext(path)
                old = f"{root}_old-{datetime.now():%Y%m%d-%H%M%S}{ext}"
                os.rename(path, old)
                print(f"[columns have changed: moved {path} to {old}]")
        new = not os.path.exists(path)
        fh = open(path, "a", newline="")
        writer = csv.writer(fh)
        if new:
            writer.writerow(header)
        return fh, writer


    def make_writers(subj):
        ev_file, ev_writer = open_csv(data_path(subj, f"s{subj}_events.csv"),
                                      EVENT_FIELDS)
        bk_file, bk_writer = open_csv(data_path(subj, f"s{subj}_blocks.csv"),
                                      BLOCK_FIELDS)
        return ev_file, ev_writer, bk_file, bk_writer


    def fmt_s(x):
        return "" if x is None else f"{x:.4f}"


    def write_stream_record(subj, run_id, phase, stream):
        fname = data_path(subj, f"s{subj}_{run_id}_{phase}_sequence.csv")
        with open(fname, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["presentation_num", "onset_s", "symbol", "is_target",
                        "nback_symbol", "repeat_lag", "isi_before_s",
                        "isi_after_s", "retention_s"])
            for s in stream:
                w.writerow([s["idx"], f'{s["onset"]:.4f}', s["symbol"],
                            int(s["is_target"]), s["nback_symbol"],
                            s["repeat_lag"], fmt_s(s["isi_before"]),
                            fmt_s(s["isi_after"]), fmt_s(s["retention"])])


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
                soa_mean_s=soa_of(P),
                soa_min_s=min_soa(P),
                soa_max_s=max_soa(P),
                retention_interval_mean_s=retention_interval(P),
                symbols_per_min=60.0 / soa_of(P),
                n_presentations_approx=int(P["stream_duration"] // soa_of(P)),
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


    # ---- Stream ----
    def run_stream(win, kb, stim, p, stream, duration, phase, log_event,
                   out=None):
        sym_stim, dot_stim, mask_stim, text = stim
        sdur = p["symbol_dur"]
        window = p["response_window"]
        late_win = p["late_response_window"]
        n_pres = len(stream)
        onsets = [s["onset"] for s in stream]
        stream_end = stream[-1]["end"] if stream else 0.0
        lookback = int(np.ceil((window + late_win) / min_soa(p))) + 1

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
            ev["isi_before_s"] = fmt_s(s["isi_before"])
            ev["retention_s"] = fmt_s(s["retention"])
            return ev

        n_hits = n_fa = n_dup = n_other = n_late = 0
        n_frames = 0
        rts = []
        late_rts = []

        kb.clearEvents()
        win.frameIntervals = []
        clock = kb.clock
        clock.reset()
        seen_presses = set()

        def guarded(fn):
            try:
                return fn()
            except AttributeError as e:
                if not is_pyglet_mac_glitch(e):
                    raise
                ev = blank_event("Input Error")
                ev["press_time_s"] = f"{clock.getTime():.4f}"
                emit(ev)
                return []
        completed = False
        last_drawn = None
        loaded_symbol = None

        try:
            while clock.getTime() < duration:
                n_frames += 1
                t = clock.getTime()
                if t >= stream_end:
                    break
                i = bisect.bisect_right(onsets, t) - 1
                cur = stream[i]
                showing = (t - cur["onset"]) < sdur

                if showing:
                    if loaded_symbol != cur["symbol"]:
                        sym_stim.text = cur["symbol"]
                        loaded_symbol = cur["symbol"]
                    sym_stim.draw()
                else:
                    if p["use_mask"]:
                        mask_stim.draw()
                    if p["show_placeholder"]:
                        dot_stim.draw()
                guarded(win.flip)

                keys = guarded(lambda: kb.getKeys(waitRelease=False))
                for k in keys:
                    press_id = (k.name, getattr(k, "tDown", None))
                    if press_id[1] is not None:
                        if press_id in seen_presses:
                            continue
                        seen_presses.add(press_id)
                    if k.name == "escape":
                        raise AbortExperiment()
                    if k.name != p["response_key"]:
                        n_other += 1
                        ev = blank_event("Other Key")
                        ev["key_name"] = k.name
                        ev["press_time_s"] = f"{clock.getTime():.4f}"
                        emit(ev)
                        continue

                    t_now = clock.getTime()
                    t_press = getattr(k, "rt", None)
                    if t_press is not None and np.isfinite(t_press) and t_press < 0:
                        continue
                    if (t_press is None or not np.isfinite(t_press)
                            or t_press > t_now + 0.001):
                        t_press = t_now

                    # Hit if inside a target's response window; otherwise late if just
                    # after one; otherwise a false alarm.
                    j = bisect.bisect_right(onsets, t_press) - 1
                    hit_on = late_on = None
                    for cand in range(j, max(-1, j - lookback - 1), -1):
                        if cand < 0 or cand >= n_pres:
                            continue
                        s = stream[cand]
                        if not s["is_target"]:
                            continue
                        dt = t_press - s["onset"]
                        if 0 <= dt <= window:
                            hit_on = s
                            break
                        if late_on is None and window < dt <= window + late_win:
                            late_on = s
                    owner = hit_on or late_on

                    if owner is None:
                        n_fa += 1
                        ev = blank_event("False Alarm")
                        ev["key_name"] = p["response_key"]
                        ev["press_time_s"] = f"{t_press:.4f}"
                        emit(ev)
                    elif owner["responded"] or owner["late_rt"] is not None:
                        n_dup += 1
                        ev = blank_event("Duplicate Press")
                        ev["key_name"] = p["response_key"]
                        ev["press_time_s"] = f"{t_press:.4f}"
                        ev["attributed_to_presentation"] = owner["idx"]
                        emit(ev)
                    elif hit_on is None:
                        rt = (t_press - late_on["onset"]) * 1000.0
                        late_on["late_rt"] = rt
                        late_rts.append(rt)
                        n_late += 1
                        ev = presentation_row(late_on, "Late Response")
                        ev["key_name"] = p["response_key"]
                        ev["press_time_s"] = f"{t_press:.4f}"
                        ev["rt_ms"] = f"{rt:.1f}"
                        ev["late_response"] = 1
                        ev["attributed_to_presentation"] = late_on["idx"]
                        ev["presentations_after_target"] = j - late_on["idx"]
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
                        ev["presentations_after_target"] = j - hit_on["idx"]
                        emit(ev)

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
                    ev["late_response"] = int(s["late_rt"] is not None)
                    emit(ev)

            covered, reach = 0.0, 0.0
            for s in stream[:shown]:
                if s["is_target"]:
                    a = max(s["onset"], reach)
                    b = min(s["onset"] + window + late_win, elapsed)
                    if b > a:
                        covered += b - a
                    reach = max(reach, b)

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
                n_misses=n_misses, n_late=n_late,
                n_fa=n_fa, n_dup=n_dup, n_other=n_other,
                hit_rate=(n_hits / n_targets) if n_targets else "",
                mean_rt=(float(np.mean(rts)) if rts else ""),
                median_rt=(float(np.median(rts)) if rts else ""),
                mean_late_rt=(float(np.mean(late_rts)) if late_rts else ""),
                time_at_risk=max(0.0, elapsed - covered),
                n_dropped=n_dropped, completed=completed, events=events_log,
            )
            if out is not None:
                out.update(summary)

        return summary


    def show_text(win, text, body, wait_key=True, secs=None):
        text.text = body
        for attempt in range(3):
            try:
                text.draw()
                win.flip()
                break
            except AttributeError as e:
                if not is_pyglet_mac_glitch(e):
                    raise
        if wait_key:
            while True:
                try:
                    keys = event.waitKeys()
                    break
                except AttributeError as e:
                    if not is_pyglet_mac_glitch(e):
                        raise
            if keys and "escape" in keys:
                raise AbortExperiment()
        elif secs:
            core.wait(secs)


    # ---- Main ----
    def main():
        resolve_timing(P)
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
        P["keyboard_backend"] = str(getattr(keyboard.Keyboard, "_backend", ""))
        P["macos_pyglet_fix"] = PYGLET_MAC_FIX
        if P["keyboard_backend"] == "event":
            print("[psychtoolbox is not installed, so key presses are only timed "
                  "to the nearest screen refresh. Install it with: "
                  "pip install psychtoolbox]")

        measured = win.getActualFrameRate(nIdentical=20, nMaxFrames=100,
                                          nWarmUpFrames=10, threshold=1)
        P["frmrate_measured"] = measured if measured else P["frmrate_expected"]
        resolve_timing(P)          # re-snap durations to the measured refresh rate
        print("[timing at the measured refresh rate]")
        print_timing(P)
        win.recordFrameIntervals = True
        win.refreshThreshold = (1.0 / P["frmrate_measured"]) + 0.004

        mon_info = dict(width_cm=screen_w_cm, distance_cm=view_dist_cm,
                        size_pix=list(win.size))
        write_sidecar(subj, seed, run_id, mon_info, P["frmrate_measured"])

        stim = build_stimuli(win, P)
        text = stim[3]
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
                label = f"practice{practice_done}" if phase == "practice" else phase
                stream = build_sequence(prng, duration, P)
                problems = validate_sequence(stream, P)
                if problems:
                    print("--- SEQUENCE WARNINGS ---")
                    for x in problems:
                        print("   ", x)
                write_stream_record(subj, run_id, label, stream)

                if phase == "practice":
                    show_text(win, text,
                              f"Practice: about {duration_text(P['practice_duration'])}.\n\n"
                              "Press any key to start.")
                else:
                    show_text(win, text,
                              f"Now the main task: {duration_text(P['stream_duration'])} "
                              "without a break.\n\n"
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
                    run_stream(win, kb, stim, P, stream, duration, label,
                               log_event, out=result)
                finally:
                    if result:
                        bk_writer.writerow([
                            datetime.now().isoformat(timespec="milliseconds"),
                            run_id, subj, int(phase == "practice"), seed, label,
                            f'{result["elapsed"]:.3f}', result["n_presentations"],
                            result["n_frames"], result["n_targets"],
                            result["n_hits"], result["n_misses"], result["n_late"],
                            result["n_fa"], result["n_dup"], result["n_other"],
                            (f'{result["hit_rate"]:.4f}'
                             if result["hit_rate"] != "" else ""),
                            (f'{result["mean_rt"]:.1f}'
                             if result["mean_rt"] != "" else ""),
                            (f'{result["median_rt"]:.1f}'
                             if result["median_rt"] != "" else ""),
                            (f'{result["mean_late_rt"]:.1f}'
                             if result["mean_late_rt"] != "" else ""),
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
                              f"Missed: {result['n_misses'] - result['n_late']}\n"
                              f"Pressed too late: {result['n_late']}\n"
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