"""CPU and GPU utilisation and GPU temperature once per second (capture spec section 5b), in its own process.

    python -m capture.monitor --out monitor.ndjson      (the session closes stdin to stop it)

CPU: GetSystemTimes over each second (ctypes). GPU: nvidia-smi --query-gpu=utilization.gpu,temperature.gpu
--format=csv,noheader,nounits -l 1 as a child process, one line per second.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import subprocess
import sys
import threading
import time
from ctypes import wintypes
from pathlib import Path


def _system_times() -> tuple[int, int, int]:
    idle, kernel, user = wintypes.FILETIME(), wintypes.FILETIME(), wintypes.FILETIME()
    ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user))
    f = lambda x: (x.dwHighDateTime << 32) | x.dwLowDateTime  # noqa: E731
    return f(idle), f(kernel), f(user)


def cpu_share(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
    """Busy share between two GetSystemTimes readings (kernel time includes idle time)."""
    idle = b[0] - a[0]
    total = (b[1] - a[1]) + (b[2] - a[2])
    return 100.0 * (1.0 - idle / total) if total > 0 else float("nan")


def parse_gpu(line: str) -> dict | None:
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 2:
        return None
    try:
        return {"gpu_util": float(parts[0]), "gpu_temp": float(parts[1])}
    except ValueError:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    f = open(a.out, "x", encoding="utf-8", newline="\n")
    lock = threading.Lock()
    stop = threading.Event()

    def write(rec: dict) -> None:
        with lock:
            f.write(json.dumps(rec) + "\n")
            f.flush()

    def cpu_loop():
        prev = _system_times()
        while not stop.wait(1.0):
            now = _system_times()
            write({"src": "cpu", "time_ns": time.time_ns(), "perf_ns": time.perf_counter_ns(),
                   "cpu_util": round(cpu_share(prev, now), 2)})
            prev = now

    gpu = subprocess.Popen(["nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu",
                            "--format=csv,noheader,nounits", "-l", "1"], stdout=subprocess.PIPE, text=True)

    def gpu_loop():
        for line in gpu.stdout:
            g = parse_gpu(line)
            rec = {"src": "gpu", "time_ns": time.time_ns(), "perf_ns": time.perf_counter_ns()}
            write({**rec, **g} if g else {**rec, "unparsed": line.strip()})

    threads = [threading.Thread(target=cpu_loop, daemon=True), threading.Thread(target=gpu_loop, daemon=True)]
    for t in threads:
        t.start()
    print("READY", flush=True)
    try:
        sys.stdin.read()
    finally:
        stop.set()
        gpu.terminate()
        gpu.wait(5)
        threads[1].join(5)
        with lock:
            f.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
