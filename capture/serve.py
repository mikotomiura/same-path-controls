"""Serve the page and keep what it sends (capture spec section 3b). Standard library only; 127.0.0.1 only.

Run as a child of the session:  python -m capture.serve --out page.ndjson   (prints 'PORT <n>')
The page long-polls GET /cmd for the session's commands (status, focus, arm, flush, quit), so that nothing is polled
while presses are typed. The session puts a command with POST /control {"cmd": ...} and reads GET /state.
Every message of the page is written to page.ndjson with the server's time_ns and perf_counter_ns at receipt; the log
of key downs is written one press per line.
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PAGE = Path(__file__).resolve().parent / "page" / "index.html"
POLL_S = 20.0
PAGE_MESSAGES = ("open", "status", "ready", "armed", "log")


class State:
    def __init__(self, out: Path):
        self.f = open(out, "x", encoding="utf-8", newline="\n")
        self.lock = threading.Lock()
        self.cmds: queue.Queue[str] = queue.Queue()
        self.counts = {m: 0 for m in PAGE_MESSAGES}
        self.last: dict[str, dict] = {}

    def keep(self, what: str, body: dict) -> None:
        stamp = {"server_time_ns": time.time_ns(), "server_perf_ns": time.perf_counter_ns()}
        with self.lock:
            self.counts[what] += 1
            if what == "log":
                entries = body.pop("entries")
                self.f.write(json.dumps({"type": "log", **body, **stamp, "n": len(entries)}) + "\n")
                for e in entries:
                    self.f.write(json.dumps({"type": "key", **e}) + "\n")
                body = {**body, "n": len(entries)}
            else:
                self.f.write(json.dumps({"type": what, **body, **stamp}) + "\n")
            self.f.flush()
            self.last[what] = {**body, **stamp}


def handler(state: State):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self.send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/cmd":
                try:
                    cmd = state.cmds.get(timeout=POLL_S)
                except queue.Empty:
                    cmd = "none"
                self.send(200, json.dumps({"cmd": cmd}).encode())
            elif self.path == "/state":
                with state.lock:
                    body = json.dumps({"counts": state.counts, "last": state.last}).encode()
                self.send(200, body)
            elif self.path == "/favicon.ico":
                self.send(204, b"")
            else:
                self.send(404, b"{}")

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            what = self.path.strip("/")
            if what == "control":
                state.cmds.put(body["cmd"])
            elif what in PAGE_MESSAGES:
                state.keep(what, body)
            else:
                self.send(404, b"{}")
                return
            self.send(200, b"{}")

    return H


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--port", type=int, default=0)
    a = ap.parse_args()
    state = State(a.out)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), handler(state))
    srv.daemon_threads = True
    print(f"PORT {srv.server_address[1]}", flush=True)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        sys.stdin.read()  # the session closes stdin to stop the server
    finally:
        srv.shutdown()
        state.f.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
