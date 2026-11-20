"""Prototype v7t (post hoc, descriptive; run once on the work PC with v7's and v7s's captures).

Conditions: works/conditions/prototype_v7t.json. After v7s's record: whether the page's times in Firefox sit on a step
of about 15.625 ms, for the time the event carries (event.timeStamp, page.ndjson's t_ms) and the time the handler ran
(performance.now(), h_ms), beside Chrome's page on the same PC. Imports v7's analysis code (a6fa5ca) and v7s's
(5508490) unchanged and checks both before anything is read. Writes counts, shares and quantiles only.

Usage (work PC):
    python works/prototype_v7t.py --v7-captures C:\\work\\v7-captures --v7s-captures C:\\work\\v7s-captures
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "works"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

import _common as cm  # noqa: E402
import prototype_v7_run as run7  # noqa: E402
import prototype_v7s as s7  # noqa: E402
import prototype_v7s_plan as splan  # noqa: E402

COND = ROOT / "works" / "conditions" / "prototype_v7t.json"
OUT = ROOT / "results" / "prototype_v7t" / "record.json"
V7_COMMIT = "a6fa5cae45579c2b4cc95ccb938eab6ea622310e"
V7S_COMMIT = "550849003164b74187cda8a36dd6311d46487bc2"
V7_CODE = ["works/prototype_v7.py", "works/prototype_v7_run.py", "works/prototype_v6.py", "works/_common.py"]
V7S_CODE = ["works/prototype_v7s.py", "works/prototype_v7s_plan.py", "works/conditions/prototype_v7s.json",
            "capture/events.py"]
STEP_MS = 15.625          # 1/64 s: the step the reading looks for (conditions.reading.step)
STEP_TOL_MS = 1.0         # a gap within this of k * STEP_MS (k >= 1) is on the step (integer times round to 15/16)
ZERO_TOL_MS = 1e-6
ON_STEP_AT_LEAST = 0.9    # conditions.reading: "on the step"
OFF_STEP_AT_MOST = 0.5    # conditions.reading: "not on the step"
LONG_MS = (18, 50, 100, 200)   # the known gaps longer than the step, where a step shows apart from the gap
N_MODES = 5


# ---------------------------------------------------------------- one capture's two page times


def page_times(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """t_ms and h_ms of the presses v7's read_page counts (same rows, same order); stops if the t_ms differ."""
    t, h = [], []
    for r in run7.read_ndjson(path):
        if r.get("type") != "key" or r.get("repeat") or r.get("kind") not in run7.COUNTED_KINDS:
            continue
        h_ms = r.get("h_ms")
        if not isinstance(h_ms, (int, float)) or isinstance(h_ms, bool) or not math.isfinite(h_ms):
            raise run7.RunStop(f"{path}: a counted press without a finite h_ms")
        t.append(float(r["t_ms"]))
        h.append(float(r["h_ms"]))
    t_, h_ = np.asarray(t, dtype=float), np.asarray(h, dtype=float)
    rel = t_ - t_[0] if t_.size else t_   # read_page's times run from the first counted press
    if not np.array_equal(rel, run7.read_page(path).times):
        raise run7.RunStop(f"{path}: the presses read here are not those v7's read_page counts")
    return t_, h_


def on_step(g: np.ndarray) -> np.ndarray:
    """Per gap: positive and within STEP_TOL_MS of k * STEP_MS for some k >= 1."""
    g = np.asarray(g, dtype=float)
    k = np.maximum(1.0, np.round(g / STEP_MS))
    return (g > ZERO_TOL_MS) & (np.abs(g - k * STEP_MS) <= STEP_TOL_MS)


def gap_measures(x: np.ndarray, modes: bool) -> dict:
    """The gaps of one time series in capture order, nothing dropped."""
    g = np.diff(np.asarray(x, dtype=float))
    if g.size == 0:
        return {"n_gaps": 0}
    pos = g[g > ZERO_TOL_MS]
    out = {"n_gaps": int(g.size), "share_zero": float(np.mean(np.abs(g) <= ZERO_TOL_MS)),
           "share_on_1ms_grid": float(np.mean(np.abs(g - np.round(g)) <= 1e-6)),
           "share_on_step_of_positive": float(np.mean(on_step(pos))) if pos.size else None,
           "smallest_positive_gap_ms": float(pos.min()) if pos.size else None}
    if modes:   # machine captures only: the most frequent positive gaps (0.1 ms bins) and their shares
        if pos.size:
            v, c = np.unique(np.round(pos, 1), return_counts=True)
            top = np.argsort(-c, kind="stable")[:N_MODES]
            out["modes_of_positive"] = [[float(v[i]), float(c[i] / pos.size)] for i in top]
        else:
            out["modes_of_positive"] = []
    return out


def capture_measures(cap: run7.Capture, planned_ms: float | None, machine: bool) -> dict:
    """t_ms and h_ms of one captured item's page, and the handler's lag behind the event's time."""
    if cap.status != "captured" or cap.folder is None:
        return {"status": cap.status}
    t, h = page_times(cap.folder / "page.ndjson")
    lag = h - t
    out = {"t": gap_measures(t, machine), "h": gap_measures(h, machine), "n_presses": int(t.size)}
    if machine and "hook" in cap.layers:   # the injection's own times (QPC), described beside the page's, not read
        out["hook"] = gap_measures(cap.layers["hook"].times, True)
    if machine:
        out["lag_h_minus_t_ms"] = run7.summary(lag, ends=True)
    else:   # a person's capture: its lag enters the series' summary only
        out["_lag_median"] = float(np.median(lag)) if lag.size else None
    if planned_ms is not None:
        out["t"]["path"] = s7.path_measures(t, planned_ms)
        out["h"]["path"] = s7.path_measures(h, planned_ms)
    return out


# ---------------------------------------------------------------- series summaries and readings


def series_summary(rows: list[dict], machine: bool) -> dict:
    """Over a series' captured items: the share of captures 'on the step' and quantiles of the per-capture shares."""
    got = [r for r in rows if "t" in r]
    out: dict = {"captures": len(got), "not_captured": len(rows) - len(got)}
    for side in ("t", "h"):
        for key in ("share_zero", "share_on_step_of_positive", "share_on_1ms_grid"):
            vals = [r[side].get(key) for r in got]
            out[f"{side}/{key}"] = run7.summary(vals, ends=machine)
        steps = [r[side].get("share_on_step_of_positive") for r in got]
        steps = [v for v in steps if v is not None]
        out[f"{side}/share_of_captures_on_step"] = (float(np.mean([v >= ON_STEP_AT_LEAST for v in steps]))
                                                    if len(steps) >= run7.MIN_SUMMARY else None)
    if machine:
        out["lag_median_ms"] = run7.summary([r["lag_h_minus_t_ms"]["median"] for r in got
                                             if r.get("lag_h_minus_t_ms")], ends=True)
    else:
        out["lag_median_ms"] = run7.summary([r.get("_lag_median") for r in got], ends=False)
    return out


def side_reading(shares: list[float | None]) -> str:
    """conditions.reading: all of the long known gaps' shares >= ON_STEP_AT_LEAST: 'on the step'; all <=
    OFF_STEP_AT_MOST: 'not on the step'; else (or any missing) 'not decided'."""
    if len(shares) != len(LONG_MS) or any(s is None for s in shares):
        return "not decided"
    if all(s >= ON_STEP_AT_LEAST for s in shares):
        return "on the step"
    if all(s <= OFF_STEP_AT_MOST for s in shares):
        return "not on the step"
    return "not decided"


def readings(known: dict[str, dict]) -> dict:
    """The three readings fixed in the conditions, from the long known gaps (hook installed)."""
    def shares(prefix: str, side: str) -> list[float | None]:
        return [known.get(f"{prefix}/{ms}", {}).get(side, {}).get("share_on_step_of_positive") for ms in LONG_MS]

    ff_t, ff_h, ch_t = side_reading(shares("F_K", "t")), side_reading(shares("F_K", "h")), \
        side_reading(shares("v7_P_idle", "t"))
    if ff_t != "on the step":
        where = "not decided"
    elif ff_h == "not on the step":
        where = "the event's time only"
    elif ff_h == "on the step":
        where = "the delivery too"
    else:
        where = "not decided"
    return {"firefox_event_time": ff_t, "firefox_handler_time": ff_h, "chrome_event_time": ch_t,
            "where_in_firefox": where,
            "pc_wide": ("no" if ff_t == "on the step" and ch_t == "not on the step" else "not decided")}


# ---------------------------------------------------------------- the run


def base_check(c: dict) -> None:
    """The conditions' base names the commits the code checks against, and v7s's conditions as deposited."""
    b = c["base"]
    if (b["v7_commit"], b["v7s_commit"]) != (V7_COMMIT, V7S_COMMIT):
        raise SystemExit("the conditions' base commits are not the code's")
    if cm.sha256(ROOT / "works" / "conditions" / "prototype_v7s.json") != b["v7s_conditions_sha256"]:
        raise SystemExit("works/conditions/prototype_v7s.json is not the conditions' base")


def code_check() -> None:
    for commit, files in ((V7_COMMIT, V7_CODE), (V7S_COMMIT, V7S_CODE)):
        for f in files:
            old = cm.git("rev-parse", f"{commit}:{f}")
            new = cm.git("hash-object", str(ROOT / f))
            if old != new:
                raise SystemExit(f"{f} differs from {commit[:7]}")


def run(v7_root: Path, v7s_root: Path) -> dict:
    runlog = s7.read_runlogs(v7s_root)
    known: dict[str, dict] = {}
    for ms in splan.K_FIREFOX_MS:
        known[f"F_K/{ms}"] = capture_measures(s7.load_item(v7s_root, runlog, f"F_K/c{ms:03d}"), float(ms), True)
    for ms in splan.K_CHROME_MS:
        known[f"K_chrome/{ms}"] = capture_measures(s7.load_item(v7s_root, runlog, f"K_chrome/c{ms:03d}"), float(ms),
                                                   True)
    for ms in splan.K_NOHOOK_MS:
        for s_ in ("F_K_nohook", "K_chrome_nohook"):
            known[f"{s_}/{ms}"] = capture_measures(s7.load_item(v7s_root, runlog, f"{s_}/c{ms:03d}", hook=False),
                                                   float(ms), True)
    v7caps = run7.load_all(v7_root)
    for cap in v7caps["P_idle"]:
        known[f"v7_P_idle/{cap.unit}"] = capture_measures(cap, float(cap.unit), True)
    series: dict[str, dict] = {}
    for name in (*s7.MC, *s7.F_MACHINE):
        rows = [capture_measures(s7.load_item(v7s_root, runlog, it), None, True)
                for it in sorted(k for k in runlog if k.split("/")[0] == name)]
        series[name] = series_summary(rows, machine=True)
    rows = [capture_measures(s7.load_item(v7s_root, runlog, f"F_H1/h{i:02d}"), None, False)
            for i in range(splan.N_F_H1)]
    series["F_H1"] = series_summary(rows, machine=False)
    return {"known_gaps": known, "series": series, "readings": readings(known)}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v7-captures", required=True, type=Path)
    ap.add_argument("--v7s-captures", required=True, type=Path)
    a = ap.parse_args(argv)
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; refusing to overwrite")
    paths = ["works/conditions/prototype_v7t.json", "works/prototype_v7t.py", *V7_CODE, *V7S_CODE]
    cm.same_as_head([ROOT / p for p in paths])
    state = cm.git_state(paths)
    code_check()
    raw = COND.read_bytes()
    c = json.loads(raw)
    base_check(c)
    if (c["reading"]["step_ms"], c["reading"]["tolerance_ms"], c["reading"]["on_step_at_least"],
            c["reading"]["off_step_at_most"]) != (STEP_MS, STEP_TOL_MS, ON_STEP_AT_LEAST, OFF_STEP_AT_MOST):
        raise SystemExit("the code's constants are not the conditions'")
    rec = run(a.v7_captures, a.v7s_captures)
    rec["provenance"] = {"run_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), **state,
                         "conditions_sha256": hashlib.sha256(raw).hexdigest(),
                         "code_sha256": {p: cm.sha256(ROOT / p) for p in paths[1:]}}
    rec = run7.to_json(rec)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    if OUT.exists():
        tmp.unlink()
        raise SystemExit(f"{OUT} appeared during the run; refusing to overwrite")
    tmp.rename(OUT)
    print(json.dumps(rec["readings"], indent=1))
    print(f"written {OUT} (sha256 {hashlib.sha256(OUT.read_bytes()).hexdigest()})")


if __name__ == "__main__":
    main()
