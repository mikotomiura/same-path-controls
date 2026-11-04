"""M2: the LLM writes an essay and the gap before each of its characters (capture spec section 4, conditions series.M2).

One attempt is two calls to ollama with the same seed (the researcher's decision of 2026-11-04, option A of
returns-v3): (1) the essay, with the essay prompt of M1 and no format; (2) the gaps, with GAPS_PROMPT on the essay
(outer whitespace stripped) and a JSON schema whose gaps_ms has exactly one integer >= 0 per character of the essay
(minItems = maxItems = its code points). The schema makes ollama stop at the last gap: asked in one JSON object with
the essay, the gaps of 7B and 14B ran on as 0, 0, 0... to the length limit (returns-v3/m2-check.md). The gaps are still
checked: len(gaps_ms) == len(text) (code points), the first 0, every gap finite and non-negative.
At most 3 attempts per prompt and seed; after 3 failures the failure is counted and nothing is typed (no replacement).
A read timeout of ollama (PLAN_TIMEOUT_S per call: the model generated for that long without finishing) is a failed
attempt. Any other error of ollama (connection refused, HTTP 4xx/5xx, an answer cut short or not JSON) is an error of
the capture, not of the model: it is raised, the capture ends 'failed' and the runlist redoes it once (the researcher's
decision of 2026-11-04, after the Mac review of returns-v3; deviation D1).
A stop (capture.stop: the hook, hence Esc, starts only after the planning) is looked at before each call and after the
last attempt; once seen, the planning ends with PlanStopped and the capture ends 'interrupted'. A call already running
is not cut (each may last up to PLAN_TIMEOUT_S); a plan that passes while a stop was asked is typed, and the stop then
ends the capture before its first press.
The plan is made before typing; the LLM is unloaded before the plan is typed, and `ollama ps` is recorded before the
capture.

Planning only (no typing; the check of decision 3 of .steering/20261104-v7-capture-returns before the production):
    python -m capture.llm_plan --model <m> --prompt-file <task.txt> --seed 1 --out <new folder> [--num-predict N]
writes llm.ndjson (every attempt, as session --llm-plan) and result.json (per attempt: ok, reason, characters of the
essay and of the gaps answer, ollama's done_reason and eval_count of both calls, seconds). --num-predict is for an
extra record only: the production (session --llm-plan) never sets it. The exit code is 0 whether or not the plan
passed (result.json says which): a check, not a capture.

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
PLAN_TIMEOUT_S = 1800  # per call; the schema bounds the answer, so this only catches an ollama that hangs
GAPS_PROMPT = ("Here is a text that a person will type on a keyboard:\n\n{text}\n\nGive the pause in milliseconds "
               "before typing each of its {n} characters, in order, the first 0. "
               "Return only a JSON object {{\"gaps_ms\": [<integers>]}}.")


def gaps_prompt(text: str) -> str:
    return GAPS_PROMPT.format(text=text, n=len(text))


def gaps_schema(n: int) -> dict:
    return {"type": "object", "required": ["gaps_ms"],
            "properties": {"gaps_ms": {"type": "array", "items": {"type": "integer", "minimum": 0},
                                       "minItems": n, "maxItems": n}}}


def attempt_seed(seed: int, attempt: int) -> int:
    return seed + 1000 * attempt


def check_plan(text: str, raw: str) -> tuple[str, list[float]]:
    """Parse and check the gaps answer for the essay text. Raises ValueError with the reason."""
    try:
        d = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"not JSON: {e}") from None
    if not isinstance(d, dict) or not isinstance(d.get("gaps_ms"), list):
        raise ValueError("not an object with a list gaps_ms")
    gaps = d["gaps_ms"]
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


class PlanStopped(Exception):
    """The planning was stopped (Esc or capture.stop) before a call to ollama."""


class OllamaError(RuntimeError):
    """An error of ollama that is not the model's: the capture fails (and is redone), no attempt is counted.
    args = (message, the call's record); str() is the message only (the record holds the prompt and the essay)."""

    def __str__(self) -> str:
        return str(self.args[0])


def is_read_timeout(e: BaseException) -> bool:
    """A read timeout: the request reached ollama and no answer came within the timeout (the model was still
    generating). urllib raises it bare (socket.timeout is TimeoutError); a timeout while connecting comes wrapped in a
    URLError and is not one (ollama did not answer at all)."""
    import urllib.error
    return isinstance(e, TimeoutError) and not isinstance(e, urllib.error.URLError)


def _call(generate, body: dict) -> dict:
    """One call to ollama, as recorded. A read timeout is kept as the record's error, so that it fails the attempt;
    any other error of ollama raises OllamaError (an answer that is not JSON included), with the record attached."""
    t0 = time.perf_counter_ns()
    try:
        ans, err = generate(body), None
    except Exception as e:  # noqa: BLE001 - sorted below: only a read timeout is the model's
        if not is_read_timeout(e):
            rec = {"body": body, "gen_ns": time.perf_counter_ns() - t0, "error": f"ollama: {type(e).__name__}: {e}"}
            raise OllamaError(rec["error"], rec) from e
        ans, err = {}, f"ollama: {type(e).__name__}: {e}"
    return {"body": body, "response": ans.get("response", ""), "gen_ns": time.perf_counter_ns() - t0,
            **({"error": err} if err else {}),
            **{k: v for k, v in ans.items() if k not in ("response", "context")}}


def make_plan(model: str, essay_prompt: str, seed: int, llm_log, generate=None, extra_options: dict | None = None,
              stop=None) -> dict:
    """Ask for the plan up to MAX_ATTEMPTS times. generate(body) -> ollama's answer (injected in tests).
    extra_options (planning-only checks, never the production) are added to ollama's options. stop (is_set()) is
    asked before each call: once set, PlanStopped. An OllamaError is written to the log as the attempt's line
    ('raised': true) and raised."""
    if generate is None:
        from capture import ollama

        def generate(body):
            return ollama.generate(body, timeout=PLAN_TIMEOUT_S)
    from capture.ollama import TEMPERATURE
    if extra_options and set(extra_options) & {"temperature", "seed"}:
        raise ValueError("extra_options may not set temperature or seed (the conditions fix them)")
    failures = []
    for a in range(MAX_ATTEMPTS):
        options = {"temperature": TEMPERATURE, "seed": attempt_seed(seed, a), **(extra_options or {})}
        rec = {"type": "attempt", "attempt": a}

        def call(key: str, body: dict) -> dict:
            if stop is not None and stop.is_set():
                llm_log.write(json.dumps({**rec, "ok": False, "stopped": True}) + "\n")
                raise PlanStopped(f"stopped before the {key} call of attempt {a}")
            try:
                rec[key] = _call(generate, body)
            except OllamaError as e:
                rec[key] = e.args[1]
                llm_log.write(json.dumps({**rec, "ok": False, "raised": True, "reason": e.args[0]}) + "\n")
                raise
            return rec[key]
        try:
            call("essay", {"model": model, "prompt": essay_prompt, "keep_alive": "10m",
                           "options": options})
            if "error" in rec["essay"]:
                raise ValueError(rec["essay"]["error"])
            text = rec["essay"]["response"].strip()
            if not text:
                raise ValueError("empty essay")
            call("gaps", {"model": model, "prompt": gaps_prompt(text), "format": gaps_schema(len(text)),
                          "keep_alive": "10m", "options": options})
            if "error" in rec["gaps"]:
                raise ValueError(rec["gaps"]["error"])
            text, gaps = check_plan(text, rec["gaps"]["response"])
            presses, skipped = presses_of(text, gaps)
            if not presses:  # a plan of untypable characters only types nothing: a failed attempt (review 2026-11-04)
                raise ValueError(f"no typable character ({skipped} skipped)")
        except ValueError as e:
            rec["gen_ns"] = sum(rec[k]["gen_ns"] for k in ("essay", "gaps") if k in rec)
            failures.append(str(e))
            llm_log.write(json.dumps({**rec, "ok": False, "reason": str(e)}) + "\n")
            continue
        rec["gen_ns"] = rec["essay"]["gen_ns"] + rec["gaps"]["gen_ns"]
        llm_log.write(json.dumps({**rec, "ok": True}) + "\n")
        return {"ok": True, "attempts": a + 1, "failures": failures, "presses": presses, "skipped_chars": skipped}
    if stop is not None and stop.is_set():  # asked during the last call: a stop, not the counted failure of M2
        llm_log.write(json.dumps({"type": "attempt", "attempt": MAX_ATTEMPTS, "ok": False, "stopped": True}) + "\n")
        raise PlanStopped("stopped during the last attempt")
    return {"ok": False, "attempts": MAX_ATTEMPTS, "failures": failures, "presses": [], "skipped_chars": 0}


def summarize_attempts(llm_ndjson: Path) -> list[dict]:
    out = []
    for ln in Path(llm_ndjson).read_text(encoding="utf-8").splitlines():
        r = json.loads(ln)
        if r.get("type") != "attempt" or r.get("stopped") or r.get("raised"):
            continue
        e, g = r.get("essay") or {}, r.get("gaps")
        out.append({"attempt": r["attempt"], "seed": e.get("body", {}).get("options", {}).get("seed"), "ok": r["ok"],
                    "reason": r.get("reason"), "essay_chars": len(e.get("response", "")),
                    "essay_done_reason": e.get("done_reason"), "essay_eval_count": e.get("eval_count"),
                    "gaps_answer_chars": len(g["response"]) if g else None,
                    "gaps_done_reason": g.get("done_reason") if g else None,
                    "gaps_eval_count": g.get("eval_count") if g else None,
                    "seconds": round(r["gen_ns"] / 1e9, 1)})
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
    task = a.prompt_file.read_text(encoding="utf-8-sig").strip()  # a BOM is not part of the task (returns-v3)
    extra = {"num_predict": a.num_predict} if a.num_predict is not None else None
    try:
        with open(a.out / "llm.ndjson", "x", encoding="utf-8", newline="\n") as log:
            model = ollama.model_record(a.model) if generate is None else {"name": a.model}
            log.write(json.dumps({"type": "header", "model": model, "seed": a.seed, "extra_options": extra,
                                  "prompt": ollama.essay_prompt(task), "gaps_prompt": GAPS_PROMPT}) + "\n")
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
