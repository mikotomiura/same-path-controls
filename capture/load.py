"""Background inference load (capture spec section 5a, conditions series.H1_load), in its own process.

    python -m capture.load --model <m> --prompt-file <p.txt> --out load.ndjson   (the session closes stdin to stop)

Generates from the prompt again and again; the output text is discarded (only its token count and times are kept).
A generation that ends before the capture is restarted with the same prompt, and the restarts are counted. Each
generation uses seed = its number ([仮] the spec fixes no seed for the load).
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

    def loop():
        n = 0
        while not stop.is_set():
            t0 = time.perf_counter_ns()
            tokens, done = 0, False
            with ollama.open_stream({"model": a.model, "prompt": prompt, "keep_alive": "10m",
                                     "options": {"temperature": ollama.TEMPERATURE, "seed": n}}) as r:
                for line in r:
                    if stop.is_set():
                        break
                    d = json.loads(line)
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

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    if not first.wait(300):
        raise TimeoutError("the load model produced no token in 300 s")
    print("READY", flush=True)
    sys.stdin.read()
    stop.set()
    t.join(30)
    f.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
