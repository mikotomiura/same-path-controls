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


def hook_presses(recs: list[dict], t_from_ns: int | None = None, injected: bool = True) -> list[dict]:
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


def summarize(out: Path) -> dict:
    out = Path(out)
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8")) if (out / "meta.json").exists() else {}
    hook = _ndjson(out / "hook.ndjson")
    page = _ndjson(out / "page.ndjson")
    inj = [r for r in _ndjson(out / "inject.ndjson") if r.get("type") != "header"]
    s: dict = {"mode": meta.get("mode"), "status": meta.get("status")}

    if meta.get("mode") == "live":
        hp = hook_presses(hook, injected=False)
        pp = page_presses(page)
        s["hook"] = {"n": len(hp)}
        s["page"] = {"n": len(pp)}
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
    if mon and t0 and t1:
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
                     "completed": sum(r["completed"] for r in gens)}
        if t0 and t1:  # restarts while presses were typed: generations that began inside the typing
            s["load"]["restarts_during_typing"] = sum(t0 <= r["start_perf_ns"] <= t1 for r in gens)
    for k in ("llm_stream", "llm_plan"):
        if k in meta:
            s[k] = meta[k]
    return s
