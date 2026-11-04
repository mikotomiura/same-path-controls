"""Write the synthetic schedules and the generic prompt of the smoke test (capture spec section 7) into a folder.

    python -m capture.smoke_series --out capture/smoke/series

Synthetic or generic only: no KUPA prompt, no human series.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from capture import events, ollama  # noqa: E402

SENTENCE = "the quick brown fox jumps over the lazy dog. "


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    text = SENTENCE * 5
    events.write(a.out / "constant.ndjson", events.constant(text, 120.0), "constant", "constant(sentence x 5, 120 ms)")
    events.write(a.out / "lognormal.ndjson", events.lognormal_series(text, 200.0, 0.5, 20261031), "lognormal",
                 "lognormal_series(sentence x 5, median 200 ms, sigma 0.5, seed 20261031, 5% rests >= 2000 ms, "
                 "one Backspace and one Delete)")
    events.write(a.out / "P5.ndjson", events.cycle_letters(1000, 5.0), "P5", "cycle_letters(1000, 5 ms)")
    (a.out / "prompt_smoke.txt").write_text(ollama.SMOKE_PROMPT + "\n", encoding="utf-8")
    (a.out / "prompt_load.txt").write_text(ollama.SMOKE_PROMPT + "\n", encoding="utf-8")
    for p in sorted(a.out.iterdir()):
        print(p.name)
    print(json.dumps({"sentence": SENTENCE, "repeats": 5}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
