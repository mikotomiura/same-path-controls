"""M2: the LLM writes an essay and the gap before each of its characters (capture spec section 4, conditions series.M2).

The plan {"text": ..., "gaps_ms": [...]} is asked with format json and checked: len(gaps_ms) == len(text) (code
points), the first 0, every gap finite and non-negative. At most 3 attempts per prompt and seed; after 3 failures the
failure is counted and nothing is typed (no replacement). The plan is made before typing; the LLM is unloaded before
the plan is typed, and `ollama ps` is recorded before the capture.

[decided by the researcher, 2026-11-04] Attempts 2 and 3 cannot reuse the seed (with the same seed and options the output repeats), so attempt a
(a = 0, 1, 2) uses the option seed `seed + 1000 * a` (returns/questions.md).
[decided by the researcher, 2026-11-04] Characters the schedule cannot hold are skipped and counted, their gap is not added to the next press (the other
presses keep their times, as in M1).
"""

from __future__ import annotations

import json
import math
import time

from capture import events

MAX_ATTEMPTS = 3
JSON_INSTRUCTION = (" Return only a JSON object {\"text\": \"<the essay>\", \"gaps_ms\": [<numbers>]} where gaps_ms has "
                    "one number for each character of text: the pause in milliseconds before typing that character, "
                    "the first 0.")


def plan_prompt(essay_prompt: str) -> str:
    return essay_prompt + JSON_INSTRUCTION


def attempt_seed(seed: int, attempt: int) -> int:
    return seed + 1000 * attempt


def check_plan(raw: str) -> tuple[str, list[float]]:
    """Parse and check one answer. Raises ValueError with the reason."""
    try:
        d = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"not JSON: {e}") from None
    if not isinstance(d, dict) or not isinstance(d.get("text"), str) or not isinstance(d.get("gaps_ms"), list):
        raise ValueError("not an object with a string text and a list gaps_ms")
    text, gaps = d["text"], d["gaps_ms"]
    if len(gaps) != len(text):
        raise ValueError(f"len(gaps_ms) {len(gaps)} != len(text) {len(text)}")
    if not text:
        raise ValueError("empty text")
    out = []
    for g in gaps:
        if isinstance(g, bool) or not isinstance(g, (int, float)) or not math.isfinite(g):
            raise ValueError(f"gap not a finite number: {g!r}")
        if g < 0:
            raise ValueError(f"negative gap: {g}")
        out.append(float(g))
    if out[0] != 0:
        raise ValueError(f"the first gap is {out[0]}, not 0")
    return text, out


def presses_of(text: str, gaps: list[float]) -> tuple[list[dict], int]:
    """The schedule of a checked plan: cumulative times, untypable characters skipped (counted)."""
    t = 0.0
    times, keys, skipped = [], [], 0
    chars = list(text)
    for j, (c, g) in enumerate(zip(chars, gaps)):
        t += g
        if c == "\r" and j + 1 < len(chars) and chars[j + 1] == "\n":
            continue
        if c in "\r\n":
            keys.append("Enter")
        elif events.typable(c):
            keys.append(c)
        else:
            skipped += 1
            continue
        times.append(t)
    return events.make(times, keys), skipped


def make_plan(model: str, essay_prompt: str, seed: int, llm_log, generate=None) -> dict:
    """Ask for the plan up to MAX_ATTEMPTS times. generate(body) -> ollama's answer (injected in tests)."""
    if generate is None:
        from capture import ollama
        generate = ollama.generate
    from capture.ollama import TEMPERATURE
    prompt = plan_prompt(essay_prompt)
    failures = []
    for a in range(MAX_ATTEMPTS):
        body = {"model": model, "prompt": prompt, "format": "json", "keep_alive": "10m",
                "options": {"temperature": TEMPERATURE, "seed": attempt_seed(seed, a)}}
        t0 = time.perf_counter_ns()
        ans = generate(body)
        raw = ans.get("response", "")
        rec = {"type": "attempt", "attempt": a, "body": body, "response": raw, "gen_ns": time.perf_counter_ns() - t0,
               **{k: v for k, v in ans.items() if k not in ("response", "context")}}
        try:
            text, gaps = check_plan(raw)
            presses, skipped = presses_of(text, gaps)
            if not presses:  # a plan of untypable characters only types nothing: a failed attempt (review 2026-11-04)
                raise ValueError(f"no typable character ({skipped} skipped)")
        except ValueError as e:
            failures.append(str(e))
            llm_log.write(json.dumps({**rec, "ok": False, "reason": str(e)}) + "\n")
            continue
        llm_log.write(json.dumps({**rec, "ok": True}) + "\n")
        return {"ok": True, "attempts": a + 1, "failures": failures, "presses": presses, "skipped_chars": skipped}
    return {"ok": False, "attempts": MAX_ATTEMPTS, "failures": failures, "presses": [], "skipped_chars": 0}
