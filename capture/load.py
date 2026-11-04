"""Background inference load (capture spec section 5a, conditions series.H1_load), in its own process.

    python -m capture.load --model <m> --prompt-file <p.txt> --out load.ndjson   (the session closes stdin to stop)

Generates from the prompt again and again; the output text is discarded (only its token count and times are kept).
A generation that ends before the capture is restarted with the same prompt, and the restarts are counted. Each
generation uses seed = its number (decided by the researcher, 2026-11-04; the spec fixes no seed for the load).
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from capture import ollama  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-file", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    prompt = ollama.essay_prompt(a.prompt_file.read_text(encoding="utf-8").strip())
    f = open(a.out, "x", encoding="utf-8", newline="\n")
    f.write(json.dumps({"type": "header", "model": ollama.model_record(a.model), "prompt": prompt,
                        "temperature": ollama.TEMPERATURE, "time_ns": time.time_ns()}) + "\n")
    f.flush()
    stop = threading.Event()
    first = threading.Event()
    failed: list[str] = []

    def loop():
        n = 0
        try:
            while not stop.is_set():
                t0 = time.perf_counter_ns()
                tokens, done = 0, False
                # the start is written at once, so that a generation still running is seen by the session's check
                f.write(json.dumps({"type": "start", "n": n, "start_perf_ns": t0}) + "\n")
                f.flush()
                with ollama.open_stream({"model": a.model, "prompt": prompt, "keep_alive": "10m",
                                     "options": {"temperature": ollama.TEMPERATURE, "seed": n}}) as r:
                    for line in r:
                        if stop.is_set():
                            break
                        d = json.loads(line)
                        if "error" in d:
                            raise RuntimeError(f"ollama: {d['error']}")
                        if d.get("response"):
                            tokens += 1
                            first.set()
                        if d.get("done"):
                            done = True
                            break
                f.write(json.dumps({"type": "generation", "n": n, "start_perf_ns": t0,
                                    "end_perf_ns": time.perf_counter_ns(), "tokens": tokens, "completed": done}) + "\n")
                f.flush()
                n += 1
        except BaseException as e:  # noqa: BLE001 - written and reported: a dead load fails the capture (review 2026-11-04)
            failed.append(f"{type(e).__name__}: {e}")
            f.write(json.dumps({"type": "error", "n": n, "perf_ns": time.perf_counter_ns(), "error": failed[-1]}) + "\n")
            f.flush()

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    end = time.monotonic() + 300
    while not first.is_set():
        if failed or not t.is_alive():
            print(f"FAIL load: {failed[-1] if failed else 'the generator stopped'}", flush=True)
            f.close()
            return 1
        if time.monotonic() > end:
            raise TimeoutError("the load model produced no token in 300 s")
        first.wait(0.1)
    print("READY", flush=True)
    sys.stdin.read()
    stop.set()
    t.join(30)
    if t.is_alive():  # a stream that sends nothing never sees stop: reported, the file is left to the exit
        print("load: the generator did not stop in 30 s (after the capture)", file=sys.stderr, flush=True)
        return 0
    f.close()
    if failed:  # after READY a failure is not the exit code's: its error line (with perf_ns) is judged by the session
        # against the typing window (summary.problems), so that a failure after the typing does not fail a capture
        print(f"load: {failed[-1]}", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
