"""Per-capture summary (capture spec section 7): counts on each layer and their drops, gap deviations against the
plan, the injector's lateness, and the LLM and load counts. Counts and summaries only: no key, character or text.

Matching follows conditions matching_and_cleaning: recorded presses are matched to planned presses by order within
the capture; a shortfall is counted as drops and an excess as extras; gap deviations are computed only when the counts
are equal. Presses recorded earlier than the previous one in arrival order are counted before any sort.
Deviation statistics: the median of (layer gap - planned gap), and the 95th percentile and maximum of its absolute
value (ms).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from capture import events

PRESS_KINDS = set(events.COUNTED_KINDS)


def _ndjson(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def dist(x) -> dict:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return {"n": 0}
    return {"n": int(x.size), "median": float(np.median(x)), "p95": float(np.percentile(x, 95)),
            "max": float(np.max(x)), "min": float(np.min(x))}


def deviation(layer_times_ms: np.ndarray, plan_ms: np.ndarray) -> dict:
    if layer_times_ms.size != plan_ms.size or plan_ms.size < 2:
        return {"computed": False}
    d = np.diff(layer_times_ms) - np.diff(plan_ms)
    a = np.abs(d)
    return {"computed": True, "median_ms": float(np.median(d)), "abs_p95_ms": float(np.percentile(a, 95)),
            "abs_max_ms": float(np.max(a))}


def hook_presses(recs: list[dict], t_from_ns: int | None = None, injected: bool = True,
                 t_to_ns: int | None = None) -> list[dict]:
    out = []
    for r in recs:
        if r.get("type") != "key" or not r["down"] or r.get("repeat") or r["kind"] not in PRESS_KINDS:
            continue
        if injected and not (r["injected"] and r["ours"]):
            continue
        if not injected and r["injected"]:
            continue
        if t_from_ns is not None and r["t_ns"] < t_from_ns:
            continue
        if t_to_ns is not None and r["t_ns"] > t_to_ns:
            continue
        out.append(r)
    return out


def page_presses(recs: list[dict]) -> list[dict]:
    return [r for r in recs if r.get("type") == "key" and not r.get("repeat") and r["kind"] in PRESS_KINDS]


def layer(times_ms: np.ndarray, kinds: list[str], plan_ms: np.ndarray, plan_kinds: list[str]) -> dict:
    n, p = len(kinds), len(plan_kinds)
    out = {"n": n, "drops": max(p - n, 0), "extras": max(n - p, 0),
           "earlier_than_previous": int(np.sum(np.diff(times_ms) < 0)) if n > 1 else 0,
           "kinds_equal": kinds == plan_kinds}
    if n == p:
        out["kind_mismatches"] = int(sum(a != b for a, b in zip(kinds, plan_kinds)))
    out["gap_deviation"] = deviation(times_ms, plan_ms)
    if n > 1:
        g = np.diff(times_ms)
        out["zero_gaps"] = int(np.sum(g == 0))
    return out


def uncovered_ms(t0: int, t1: int, spans: list[tuple[int, int]]) -> float:
    """Milliseconds of [t0, t1] (ns) inside no span (ns): the typing time with no background generation running."""
    covered, cur = 0, t0
    for a, b in sorted(spans):
        a, b = max(a, cur), min(b, t1)
        if b > a:
            covered += b - a
            cur = b
    return (t1 - t0 - covered) / 1e6


def covered_spans(load: list[dict], t1: int, stall_ns: int) -> list[tuple[int, int]]:
    """The times (ns) when a background generation was generating: each generation from its start to its end (a
    running one: to t1), less its stalls (gaps of no token longer than load.STALL_MS, written by the load). A running
    generation whose latest token (its progress lines) is more than stall_ns before t1 is counted only up to that token:
    it stalled at the end (Mac review of 2026-11-04). A load that failed covers nothing after its error."""
    gens = {r["n"]: (r["start_perf_ns"], r["end_perf_ns"]) for r in load if r.get("type") == "generation"}
    failed = any(r.get("type") == "error" for r in load)
    last_tok: dict = {}
    for r in load:
        if r.get("type") in ("progress",):
            last_tok[r["n"]] = max(last_tok.get(r["n"], 0), r["last_token_perf_ns"])
        elif r.get("type") == "stall":
            last_tok[r["n"]] = max(last_tok.get(r["n"], 0), r["to_perf_ns"])
    for r in load:
        if r.get("type") == "start" and r["n"] not in gens and not failed:
            end = t1
            lt = max(last_tok.get(r["n"], r["start_perf_ns"]), r["start_perf_ns"])
            if t1 - lt > stall_ns:
                end = lt
            gens[r["n"]] = (r["start_perf_ns"], end)
    stalls: dict = {}
    for r in load:
        if r.get("type") == "stall":
            stalls.setdefault(r["n"], []).append((r["from_perf_ns"], r["to_perf_ns"]))
    out = []
    for n, (a, b) in gens.items():
        cur = a
        for s0, s1 in sorted(stalls.get(n, [])):
            if s0 > cur:
                out.append((cur, min(s0, b)))
            cur = max(cur, s1)
        if b > cur:
            out.append((cur, b))
    return out


# The longest typing time without a background generation that a load capture may have (restarts take a request to
# ollama, a few ms to a few hundred ms). Not in the conditions or the spec: a technical bound (review of 2026-11-04).
LOAD_UNCOVERED_MAX_MS = 1000.0


def foreign_injected_keys(hook: list[dict]) -> int:
    """Key downs that came injected without the injector's mark, modifiers left out (the Alt of --auto-focus): keys
    through a remote desktop come so, and cannot be told from an injection (researcher's decision of 2026-11-04, b)."""
    return sum(1 for r in hook if r.get("type") == "key" and r["down"] and r["injected"] and not r["ours"]
               and r["kind"] != "modifier")


def hook_log(hook: list[dict]) -> dict:
    """Whether hook.ndjson holds every record the hook saw: its footer (written when the hook stops) says
    n_raw == n_written and no error of the writer (Mac review of returns-v3)."""
    ft = hook[-1] if hook and hook[-1].get("type") == "footer" else None
    if ft is None:
        return {"complete": False, "footer": False}
    n_raw, n_written = ft.get("n_raw"), ft.get("n_written")
    ok = (isinstance(n_raw, int) and isinstance(n_written, int) and n_raw == n_written >= 0
          and not ft.get("writer_error"))
    return {"complete": ok, "footer": True, "n_raw": ft.get("n_raw"), "n_written": ft.get("n_written"),
            **({"writer_error": ft["writer_error"]} if ft.get("writer_error") else {})}


def problems(s: dict, meta: dict) -> dict:
    """What makes a finished capture unusable as it stands, so that it ends 'interrupted' and is redone once
    (conditions run_order.logged: 'a capture interrupted or restarted is redone once and counted'). Drops and extras
    are not problems: they are counted results of the path. Empty when there is none."""
    out: dict = {}
    planned = s.get("planned", 0) if meta.get("mode") != "live" else None
    if meta.get("hook_installed", True):
        n = (s.get("hook") or {}).get("n", 0)
        if n == 0 and (planned or meta.get("mode") == "live"):
            out["hook_recorded_nothing"] = True
    n_page = (s.get("page") or {}).get("n", 0)
    if n_page == 0 and (planned or meta.get("mode") == "live"):
        out["page_recorded_nothing"] = True
    if meta.get("hook_installed", True) and s.get("foreign_injected_keys"):
        out["foreign_injected_keys"] = s["foreign_injected_keys"]
    hl = s.get("hook_log")
    if meta.get("hook_installed", True) and hl is not None and not hl.get("complete"):
        out["hook_log_incomplete"] = hl
    pl = meta.get("page_log")
    if pl is not None and (pl.get("sent") != pl.get("saved") or pl.get("batches_failed") or pl.get("write_errors")):
        out["page_log_not_all_saved"] = {"sent": pl.get("sent"), "saved": pl.get("saved"),
                                         "batches_failed": pl.get("batches_failed"),
                                         **({"write_errors": pl["write_errors"]} if pl.get("write_errors") else {})}
    if meta.get("page_opens", 1) > 1:
        out["page_reloaded"] = meta["page_opens"]
    if (s.get("page") or {}).get("not_in_box", 0) > 0:
        out["presses_outside_the_box"] = s["page"]["not_in_box"]
    if meta.get("load"):
        ld = s.get("load")
        if not ld:
            out["load_recorded_nothing"] = True
        else:
            if ld.get("errors_during_typing"):
                out["load_errors"] = ld["errors_during_typing"]
            if ld.get("uncovered_ms", 0.0) > LOAD_UNCOVERED_MAX_MS:
                out["load_uncovered_ms"] = ld["uncovered_ms"]
    return out


def summarize(out: Path) -> dict:
    out = Path(out)
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8")) if (out / "meta.json").exists() else {}
    hook = _ndjson(out / "hook.ndjson")
    page = _ndjson(out / "page.ndjson")
    inj = [r for r in _ndjson(out / "inject.ndjson") if r.get("type") != "header"]
    s: dict = {"mode": meta.get("mode"), "status": meta.get("status")}
    if (out / "hook.ndjson").exists():
        s["hook_log"] = hook_log(hook)
        s["foreign_injected_keys"] = foreign_injected_keys(hook)

    if meta.get("mode") == "live":
        # only the presses between the signal and the end chord (the Enter that readies the page and any press before
        # the signal are not part of L0: review of 2026-11-04); the page records only once armed
        hp = hook_presses(hook, injected=False, t_from_ns=meta.get("typing_start_perf_ns"),
                          t_to_ns=meta.get("typing_end_perf_ns"))
        pp = page_presses(page)
        s["hook"] = {"n": len(hp)}
        s["page"] = {"n": len(pp), "not_in_box": sum(not r.get("in_box", True) for r in pp)}
        s["kinds_equal_hook_page"] = [r["kind"] for r in hp] == [r["kind"] for r in pp]
        return s

    sched = out / ("schedule.ndjson" if (out / "schedule.ndjson").exists() else "events.ndjson")
    if not sched.exists():
        s["planned"] = 0
        return s
    _, plan = events.read(sched)
    plan_ms = np.array([p["t_ms"] for p in plan], dtype=float)
    plan_kinds = events.kinds(plan)
    s["planned"] = len(plan)
    if plan_ms.size > 1:
        s["plan_zero_gap_share"] = float(np.mean(np.diff(plan_ms) == 0))
    sent = [r for r in inj if r.get("ok")]
    s["inject"] = {"attempted": len(inj), "sent": len(sent), "not_sent": len(inj) - len(sent),
                   "late_ms": dist([r.get("late_ns", r.get("wait_ns", 0)) / 1e6 for r in inj])}
    if inj and "queue_len" in inj[0]:
        s["inject"]["queue_max"] = max(r["queue_len"] for r in inj)
    if hook:
        hp = hook_presses(hook)
        s["hook"] = layer(np.array([r["t_ns"] / 1e6 for r in hp]), [r["kind"] for r in hp], plan_ms, plan_kinds)
        s["hook"]["foreign_downs"] = len(hook_presses(hook, injected=False))
        s["hook"]["injected_not_ours"] = sum(1 for r in hook if r.get("type") == "key" and r["down"]
                                             and r["injected"] and not r["ours"])
    else:
        s["hook"] = None
    pp = page_presses(page)
    s["page"] = layer(np.array([r["t_ms"] for r in pp]), [r["kind"] for r in pp], plan_ms, plan_kinds)
    s["page"]["untrusted"] = sum(not r["trusted"] for r in pp)
    s["page"]["composing"] = sum(bool(r.get("composing")) for r in pp)
    s["page"]["not_in_box"] = sum(not r.get("in_box", True) for r in pp)
    s["page"]["named_nonchar"] = sorted({r["name"] for r in page if r.get("type") == "key"
                                         and r.get("kind") == "nonchar" and "name" in r})

    mon = _ndjson(out / "monitor.ndjson")
    t0, t1 = meta.get("typing_start_perf_ns"), meta.get("typing_end_perf_ns")
    if mon and t0 is not None and t1 is not None:
        g = [m["gpu_util"] for m in mon if m.get("src") == "gpu" and "gpu_util" in m and t0 <= m["perf_ns"] <= t1]
        c = [m["cpu_util"] for m in mon if m.get("src") == "cpu" and t0 <= m["perf_ns"] <= t1]
        temp = [m["gpu_temp"] for m in mon if m.get("src") == "gpu" and "gpu_temp" in m and t0 <= m["perf_ns"] <= t1]
        s["monitor_during_typing"] = {"gpu_util_median": float(np.median(g)) if g else None,
                                      "gpu_temp_max": max(temp) if temp else None,
                                      "cpu_util_median": float(np.median(c)) if c else None, "seconds": len(g)}
    load = _ndjson(out / "load.ndjson")
    if load:
        gens = [r for r in load if r.get("type") == "generation"]
        s["load"] = {"generations": len(gens), "restarts": max(len(gens) - 1, 0),
                     "completed": sum(r["completed"] for r in gens),
                     "errors": [r["error"] for r in load if r.get("type") == "error"]}
        if t0 is not None and t1 is not None:  # restarts while presses were typed: generations that began inside the typing
            s["load"]["restarts_during_typing"] = sum(t0 <= r["start_perf_ns"] <= t1 for r in gens)
            # an error at or before the typing's end failed the load while it mattered; a later one is only reported
            s["load"]["errors_during_typing"] = [r["error"] for r in load
                                                 if r.get("type") == "error" and r["perf_ns"] <= t1]
            from capture.load import STALL_MS
            stall_ns = round(STALL_MS * 1e6)
            st = [(max(r["from_perf_ns"], t0), min(r["to_perf_ns"], t1)) for r in load if r.get("type") == "stall"]
            st = [(a, b) for a, b in st if b > a]
            s["load"]["stalls_during_typing"] = {"n": len(st), "ms": sum(b - a for a, b in st) / 1e6,
                                                 "threshold_ms": STALL_MS}
            # typing time with no generation generating: restarts, stalls, a generation stalled at the end
            s["load"]["uncovered_ms"] = uncovered_ms(t0, t1, covered_spans(load, t1, stall_ns))
    for k in ("llm_stream", "llm_plan"):
        if k in meta:
            s[k] = meta[k]
    return s
