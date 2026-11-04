"""M2: the LLM writes an essay and the gap before each of its characters (capture spec section 4, conditions series.M2).

The plan {"text": ..., "gaps_ms": [...]} is asked with format json and checked: len(gaps_ms) == len(text) (code
points), the first 0, every gap finite and non-negative. At most 3 attempts per prompt and seed; after 3 failures the
failure is counted and nothing is typed (no replacement). The plan is made before typing; the LLM is unloaded before
the plan is typed, and `ollama ps` is recorded before the capture.

Planning only (no typing; the check of decision 3 of .steering/20261104-v7-capture-returns before the production):
    python -m capture.llm_plan --model <m> --prompt-file <task.txt> --seed 1 --out <new folder> [--num-predict N]
writes llm.ndjson (every attempt, as session --llm-plan) and result.json (per attempt: ok, reason, characters of the
answer, ollama's done_reason and eval_count, seconds). --num-predict is for an extra record only: the production
(session --llm-plan) never sets it. The exit code is 0 whether or not the plan passed (result.json says which): a
check, not a capture.

[decided by the researcher, 2026-11-04] Attempts 2 and 3 cannot reuse the seed (with the same seed and options the output repeats), so attempt a
(a = 0, 1, 2) uses the option seed `seed + 1000 * a` (returns/questions.md).
[decided by the researcher, 2026-11-04] Characters the schedule cannot hold are skipped and counted, their gap is not added to the next press (the other
presses keep their times, as in M1).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

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


def make_plan(model: str, essay_prompt: str, seed: int, llm_log, generate=None, extra_options: dict | None = None) -> dict:
    """Ask for the plan up to MAX_ATTEMPTS times. generate(body) -> ollama's answer (injected in tests).
    extra_options (planning-only checks, never the production) are added to ollama's options."""
    if generate is None:
        from capture import ollama
        generate = ollama.generate
    from capture.ollama import TEMPERATURE
    if extra_options and set(extra_options) & {"temperature", "seed"}:
        raise ValueError("extra_options may not set temperature or seed (the conditions fix them)")
    prompt = plan_prompt(essay_prompt)
    failures = []
    for a in range(MAX_ATTEMPTS):
        body = {"model": model, "prompt": prompt, "format": "json", "keep_alive": "10m",
                "options": {"temperature": TEMPERATURE, "seed": attempt_seed(seed, a), **(extra_options or {})}}
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


def summarize_attempts(llm_ndjson: Path) -> list[dict]:
    out = []
    for ln in Path(llm_ndjson).read_text(encoding="utf-8").splitlines():
        r = json.loads(ln)
        if r.get("type") != "attempt":
            continue
        out.append({"attempt": r["attempt"], "seed": r["body"]["options"]["seed"], "ok": r["ok"],
                    "reason": r.get("reason"), "answer_chars": len(r["response"]), "done_reason": r.get("done_reason"),
                    "eval_count": r.get("eval_count"), "seconds": round(r["gen_ns"] / 1e9, 1)})
    return out


def main(argv=None, generate=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m capture.llm_plan")
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-file", type=Path, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--num-predict", type=int)
    a = ap.parse_args(argv)
    from capture import ollama
    a.out.mkdir(parents=True, exist_ok=False)
    task = a.prompt_file.read_text(encoding="utf-8").strip()
    extra = {"num_predict": a.num_predict} if a.num_predict is not None else None
    try:
        with open(a.out / "llm.ndjson", "x", encoding="utf-8", newline="\n") as log:
            model = ollama.model_record(a.model) if generate is None else {"name": a.model}
            log.write(json.dumps({"type": "header", "model": model, "seed": a.seed, "extra_options": extra,
                                  "prompt": plan_prompt(ollama.essay_prompt(task))}) + "\n")
            res = make_plan(a.model, ollama.essay_prompt(task), a.seed, log, generate=generate, extra_options=extra)
    except BaseException:
        if generate is None:
            try:
                ollama.unload(a.model)
            except Exception as e:  # noqa: BLE001 - the error of the plan is the one raised
                print(f"unload failed after the error: {e}", file=sys.stderr)
        raise
    if generate is None:
        ollama.unload(a.model)
    result = {"model": a.model, "seed": a.seed, "extra_options": extra, "ok": res["ok"], "attempts": res["attempts"],
              "n_presses": len(res["presses"]), "skipped_chars": res["skipped_chars"],
              "per_attempt": summarize_attempts(a.out / "llm.ndjson")}
    (a.out / "result.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("model", "ok", "attempts", "n_presses")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
