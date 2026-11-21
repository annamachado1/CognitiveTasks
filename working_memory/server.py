#!/usr/bin/env python3
# Serves the browser version of the VWM repeat-detection task from this laptop
# and writes each session's data to data/<subject>/ in the same files and
# columns as the PsychoPy version (vwm_repeat_detection_task.py).
#
#   python3 server.py            # local only: http://localhost:8000/?key=...
#   python3 server.py --tunnel   # also opens a Cloudflare quick tunnel and
#                                # prints a public link to share
#
# Standard library only; no installs needed apart from cloudflared for --tunnel.

import argparse
import csv
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
TASK_HTML = os.path.join(HERE, "task.html")
FILE_PREFIX = "VWMcentre"
MAX_BODY = 8 * 1024 * 1024

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
    "planned_duration_s", "isi_mean_s", "target_rate", "target_rate_unit",
    "stream_duration_s", "n_presentations", "n_frames",
    "n_targets", "n_hits", "n_misses", "n_late_responses",
    "n_false_alarms", "n_duplicate_presses", "n_other_keys",
    "hit_rate", "mean_rt_ms", "median_rt_ms", "mean_late_rt_ms",
    "hit_rate_h1", "hit_rate_h2", "mean_rt_h1_ms", "mean_rt_h2_ms",
    "fa_per_min", "time_at_risk_s",
    "frmrate_measured", "n_dropped_frames", "completed",
]

SEQUENCE_FIELDS = [
    "presentation_num", "onset_s", "symbol", "is_target", "nback_symbol",
    "repeat_lag", "isi_before_s", "isi_after_s", "retention_s",
]

RUN_ID_RE = re.compile(r"^[0-9a-f]{8}$")
PHASE_RE = re.compile(r"^(practice\d{1,2}|test)$")

write_lock = threading.Lock()


def data_path(data_dir, subj, name):
    folder = os.path.join(data_dir, str(subj))
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f"{FILE_PREFIX}_{name}")


def append_rows(path, header, rows):
    """Same behaviour as open_csv in the PsychoPy script: if an existing file
    has different columns it is moved aside, then rows are appended."""
    if os.path.exists(path):
        with open(path, newline="") as f:
            existing = next(csv.reader(f), None)
        if existing != header:
            root, ext = os.path.splitext(path)
            old = f"{root}_old-{datetime.now():%Y%m%d-%H%M%S}{ext}"
            os.rename(path, old)
            print(f"[columns have changed: moved {path} to {old}]")
    new = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(header)
        w.writerows(rows)


def clean_rows(rows, n_cols):
    if not isinstance(rows, list):
        raise ValueError("rows must be a list")
    out = []
    for r in rows:
        if not isinstance(r, list) or len(r) != n_cols:
            raise ValueError("row has the wrong number of columns")
        cells = []
        for c in r:
            if c is None:
                cells.append("")
            elif isinstance(c, (str, int, float, bool)):
                s = str(int(c)) if isinstance(c, bool) else str(c)
                if len(s) > 200:
                    raise ValueError("cell too long")
                cells.append(s)
            else:
                raise ValueError("cells must be plain values")
        out.append(cells)
    return out


def handle_save(data_dir, msg):
    subj = msg.get("subject")
    if not isinstance(subj, int) or not (0 <= subj <= 999999):
        raise ValueError("subject must be a whole number")
    run_id = str(msg.get("run_id", ""))
    if not RUN_ID_RE.match(run_id):
        raise ValueError("bad run_id")
    kind = msg.get("kind")

    with write_lock:
        if kind == "events":
            rows = clean_rows(msg.get("rows"), len(EVENT_FIELDS))
            append_rows(data_path(data_dir, subj, f"s{subj}_events.csv"),
                        EVENT_FIELDS, rows)
            return f"{len(rows)} event rows"
        if kind == "blocks":
            rows = clean_rows(msg.get("rows"), len(BLOCK_FIELDS))
            append_rows(data_path(data_dir, subj, f"s{subj}_blocks.csv"),
                        BLOCK_FIELDS, rows)
            return f"{len(rows)} block rows"
        if kind == "sequence":
            phase = str(msg.get("phase", ""))
            if not PHASE_RE.match(phase):
                raise ValueError("bad phase")
            rows = clean_rows(msg.get("rows"), len(SEQUENCE_FIELDS))
            path = data_path(data_dir, subj,
                             f"s{subj}_{run_id}_{phase}_sequence.csv")
            with open(path, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(SEQUENCE_FIELDS)
                w.writerows(rows)
            return f"{phase} sequence"
        if kind == "params":
            payload = msg.get("json")
            if not isinstance(payload, dict):
                raise ValueError("params must be an object")
            path = data_path(data_dir, subj, f"s{subj}_{run_id}_params.json")
            with open(path, "w") as fh:
                json.dump(payload, fh, indent=2, default=str)
            return "params"
    raise ValueError("unknown kind")


def make_handler(data_dir, key):
    class Handler(BaseHTTPRequestHandler):
        server_version = "VWMTask/1.0"

        def log_message(self, fmt, *args):
            pass  # keep the console for data messages only

        def send_plain(self, code, text):
            body = text.encode()
            self.send_response(code)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
                return
            if path not in ("/", "/index.html", "/task.html"):
                self.send_plain(404, "Not found")
                return
            with open(TASK_HTML, "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.path.split("?", 1)[0] != "/api/save":
                self.send_plain(404, "Not found")
                return
            try:
                n = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                n = 0
            if n <= 0 or n > MAX_BODY:
                self.send_plain(413, "Body missing or too large")
                return
            try:
                msg = json.loads(self.rfile.read(n).decode("utf-8"))
            except Exception:
                self.send_plain(400, "Body is not valid JSON")
                return
            if not isinstance(msg, dict) or not secrets.compare_digest(
                    str(msg.get("key", "")), key):
                self.send_plain(403, "Wrong or missing key")
                return
            try:
                what = handle_save(data_dir, msg)
            except ValueError as e:
                self.send_plain(400, str(e))
                return
            print(f"[{datetime.now():%H:%M:%S}] subject {msg['subject']} "
                  f"run {msg['run_id']}: saved {what}")
            self.send_plain(200, "ok")

    return Handler


def start_tunnel(port, key):
    exe = shutil.which("cloudflared")
    if not exe:
        print("\ncloudflared is not installed, so no public link was made.\n"
              "  macOS:   brew install cloudflared\n"
              "  Windows: winget install --id Cloudflare.cloudflared\n"
              "Then run this script again with --tunnel.\n")
        return None
    proc = subprocess.Popen(
        [exe, "tunnel", "--no-autoupdate", "--url", f"http://localhost:{port}"],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    url_re = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")

    def watch():
        found = False
        for line in proc.stderr:
            m = url_re.search(line)
            if m and not found:
                found = True
                print("\n================================================")
                print("Share this link with collaborators:")
                print(f"  {m.group(0)}/?key={key}")
                print("It works for as long as this window stays open and")
                print("the laptop stays awake and online.")
                print("================================================\n")
        if not found:
            print("[cloudflared exited before giving a link]")

    threading.Thread(target=watch, daemon=True).start()
    return proc


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--data-dir", default="data",
                    help="where to write data (default: ./data, as in the "
                         "PsychoPy script)")
    ap.add_argument("--key", default=None,
                    help="access key required to save data (default: a new "
                         "random key each start)")
    ap.add_argument("--tunnel", action="store_true",
                    help="open a Cloudflare quick tunnel and print a public link")
    args = ap.parse_args()

    if not os.path.exists(TASK_HTML):
        sys.exit(f"task.html not found next to server.py ({TASK_HTML})")
    key = args.key or secrets.token_urlsafe(9)
    data_dir = os.path.abspath(args.data_dir)
    os.makedirs(data_dir, exist_ok=True)

    httpd = ThreadingHTTPServer(("127.0.0.1", args.port),
                                make_handler(data_dir, key))
    print(f"Serving the task at http://localhost:{args.port}/?key={key}")
    print(f"Data will be written to {data_dir}")

    proc = start_tunnel(args.port, key) if args.tunnel else None
    print("Press Ctrl+C to stop.\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        httpd.server_close()
        if proc:
            proc.terminate()


if __name__ == "__main__":
    main()