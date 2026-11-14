"""Prototype v7q (post hoc, descriptive; run once on the work PC with the production captures).

Conditions: works/conditions/prototype_v7q.json. (1) A census of the injected presses the page marked as key repeats;
(2) v7's page-layer readings again with those presses kept; (3) M2's sessions grouped by how D1's judgment moves from
plan to page. Imports v7's analysis code (a6fa5ca) unchanged. Writes counts and summaries only.

Usage (work PC): python works/prototype_v7q.py --captures C:\\work\\v7-captures
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "works"))

import numpy as np  # noqa: E402

import _common as cm  # noqa: E402
import prototype_v6 as v6  # noqa: E402
import prototype_v7 as v7  # noqa: E402
import prototype_v7_run as vr  # noqa: E402

COND = ROOT / "works" / "conditions" / "prototype_v7q.json"
OUT = ROOT / "results" / "prototype_v7q" / "record.json"
KEY = [20261113, 7, 2]
BASE_COMMIT = "a6fa5cae45579c2b4cc95ccb938eab6ea622310e"
BINS = [0.0, 1.0, 5.0, 20.0, 50.0, 150.0, np.inf]
BIN_NAMES = ["0-1", "1-5", "5-20", "20-50", "50-150", "150+"]
V7_CODE = ["works/prototype_v7_run.py", "works/prototype_v7.py", "works/prototype_v6.py", "works/_common.py"]
PAGE_SERIES = ("H1_idle", "H1_load", "H1_rep", "L0", "L1_idle", "L1_load", *vr.MACHINE, "P_idle", "P_load",
               "P_nohook")


# ---------------------------------------------------------------- reading the page with the repeat flag

def page_presses(path: Path) -> tuple[np.ndarray, list[str], np.ndarray]:
    """The page's key lines of a counted kind, in the box and not composing (v7's read_page counts these), in capture
    order, with their repeat flag. Times from the first such line."""
    ts, ks, rep = [], [], []
    for r in vr.read_ndjson(path):
        if r.get("type") in vr.PAGE_OTHER_TYPES:
            continue
        if r.get("type") != "key":
            raise vr.RunStop(f"{path}: a line of unknown type {r.get('type')!r}")
        vr.need_types(path, r, {"t_ms": (int, float), "kind": str, "repeat": bool, "trusted": bool,
                                "composing": bool, "in_box": bool})
        if r["kind"] in vr.COUNTED_KINDS and r["in_box"] and not r["composing"]:
            ts.append(float(r["t_ms"]))
            ks.append(r["kind"])
            rep.append(r["repeat"])
    t = np.array(ts, dtype=float)
    return (t - t[0] if t.size else t), ks, np.array(rep, dtype=bool)


def hook_injected_repeats(path: Path) -> int:
    n = 0
    for r in vr.read_ndjson(path):
        if r.get("type") == "key" and r.get("down") and r.get("repeat") and r.get("injected") and r.get("ours") \
                and r.get("kind") in vr.COUNTED_KINDS:
            n += 1
    return n


def bin_of(x: float) -> str:
    """Closed-left bins; a negative gap (a time inversion in capture order) or a non-finite one is named apart."""
    if not np.isfinite(x):
        return "not finite"
    if x < 0:
        return "negative"
    return BIN_NAMES[int(np.searchsorted(BINS, x, side="right")) - 1]


def tally(d: dict, k: str) -> None:
    d[k] = d.get(k, 0) + 1


def census_series(caps: list) -> dict:
    """repeats_census for one series (the page layer of its captured items)."""
    out = {"items": 0, "items_with_flag": 0, "flagged": 0, "aligned_items": 0, "kind": {}, "previous_kind": {},
           "same_planned_key_as_previous": {}, "planned_gap_before": {}, "recorded_gap_before": {}, "position_tenth": {},
           "first_press_flagged": 0, "hook_flagged_injected": 0}
    for cap in caps:
        if cap.status != "captured" or cap.folder is None:
            continue
        out["items"] += 1
        t, ks, rep = page_presses(cap.folder / "page.ndjson")
        if (cap.folder / "hook.ndjson").exists():
            out["hook_flagged_injected"] += hook_injected_repeats(cap.folder / "hook.ndjson")
        if not rep.any():
            continue
        out["items_with_flag"] += 1
        aligned = (cap.plan is not None and len(cap.plan.kinds) == len(ks) and cap.plan.kinds == ks)
        out["aligned_items"] += aligned
        for j in np.flatnonzero(rep):
            out["flagged"] += 1
            tally(out["kind"], ks[j])
            tally(out["position_tenth"], str(min(9, int(10 * j / len(ks)))))
            if j == 0:
                out["first_press_flagged"] += 1
                continue
            tally(out["previous_kind"], ks[j - 1])
            tally(out["recorded_gap_before"], bin_of(t[j] - t[j - 1]))
            if aligned:
                tally(out["same_planned_key_as_previous"], str(cap.plan.keys[j] == cap.plan.keys[j - 1]).lower())
                tally(out["planned_gap_before"], bin_of(float(cap.plan.times[j] - cap.plan.times[j - 1])))
    return out


# ---------------------------------------------------------------- readings with the flagged presses kept

def kept_capture(cap):
    """The capture with its page layer read again, flagged presses kept as counted presses."""
    if cap.status != "captured" or cap.folder is None or "page" not in cap.layers:
        return cap
    t, ks, _ = page_presses(cap.folder / "page.ndjson")
    layers = dict(cap.layers)
    layers["page"] = vr.Layer(t, ks, cap.layers["page"].not_counted, 0)
    return replace(cap, layers=layers)


def rec_gaps(caps: list, layer: str, drop_ties: bool = True) -> list:
    out = []
    for cap in caps:
        x = vr.recorded(cap, layer, drop_ties)
        out.append(None if x is None else x.gaps)
    return out


def pair_counts(det: str, s0: np.ndarray, s1: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    j0, o0 = vr.judgments(det, s0)
    j1, o1 = vr.judgments(det, s1)
    ok = o0 & o1
    return j0[ok], j1[ok], int((~ok).sum())


def readings_kept(caps_kept: dict, s_h0: dict, sc) -> dict:
    out: dict = {"human": {}, "machine": {}}
    s_h1 = sc.score(rec_gaps(caps_kept["H1_idle"], "page"))
    for d in vr.DETECTORS:
        jb, ja, missing = pair_counts(d, s_h0[d], s_h1[d])
        b, c, n = v7.flips(jb, ja)
        h = v7.headline(b, c, n) if n else None
        out["human"][d] = {"b": b, "c": c, "n": n, "missing": missing, "reading": h["reading"] if h else None}
    out["H1_idle_replay_error"] = vr.replay_error_series(caps_kept["H1_idle"], "page")
    for si, s_ in enumerate(vr.MACHINE):
        caps = caps_kept[s_]
        sp = sc.score([vr.plan_gaps(c) if c.status == "captured" else None for c in caps])
        sr = sc.score(rec_gaps(caps, "page"))
        cell = {}
        for d in vr.DETECTORS:
            jb, ja, missing = pair_counts(d, sp[d], sr[d])
            b, c, n = v7.flips(jb, ja)
            without_plan = sum(1 for cp in caps if cp.status == "no_plan")
            row = {"b": b, "c": c, "n": n, "missing": missing - without_plan, "without_plan": without_plan,
                   "plan_machine": int(jb.sum()),
                   "recorded_machine": int(ja.sum())}
            if n:
                mp = v7.machine_path_reading(jb, ja, np.arange(n), [*KEY, vr.DET_IDX[d], si])
                row["reading"] = mp["reading"]
            cell[d] = row
        out["machine"][s_] = cell
    return out


# ---------------------------------------------------------------- M2's sessions

def m2_groups(caps: list, caps_kept: list) -> dict:
    vals: dict = {}
    for cap, ck in zip(caps, caps_kept):
        if cap.status != "captured":
            continue
        pg = vr.plan_gaps(cap)
        rh, rp = vr.recorded(cap, "hook"), vr.recorded(cap, "page")
        rp_k = vr.recorded(ck, "page")
        rp_t = vr.recorded(cap, "page", drop_ties=False)
        dl = {k: (v7.delta(g) if g is not None and vr._condrey_scorable(g) else None)
              for k, g in (("plan", pg), ("hook", rh.gaps if rh else None), ("page", rp.gaps if rp else None),
                           ("page_repeats_kept", rp_k.gaps if rp_k else None),
                           ("page_equal_times_kept", rp_t.gaps if rp_t else None))}
        if dl["plan"] is None:
            grp = "plan not judged"
        elif dl["plan"] > v7.DELTA_T:
            grp = "stays human"   # the conditions' group: every plan judged 'human', whatever the page
        elif dl["page"] is None:
            grp = "page not judged"
        elif dl["page"] > v7.DELTA_T:
            grp = "flips"
        else:
            grp = "stays machine"
        g = vals.setdefault(grp, {k: [] for k in ("plan_zero_share", "delta_plan", "delta_hook", "delta_page",
                                                  "delta_page_repeats_kept", "delta_page_equal_times_kept",
                                                  "page_dropped_equal_times", "page_repeats",
                                                  "hook_share_below_1ms", "page_share_below_1ms")})
        g["plan_zero_share"].append(float(np.mean(pg == 0)) if pg is not None and pg.size else None)
        for k in ("plan", "hook", "page", "page_repeats_kept", "page_equal_times_kept"):
            g[f"delta_{k}"].append(dl[k])
        g["page_dropped_equal_times"].append(rp.dropped if rp else None)
        g["page_repeats"].append(cap.layers["page"].repeats if "page" in cap.layers else None)
        g["hook_share_below_1ms"].append(float(np.mean(rh.gaps < 1.0)) if rh and rh.gaps.size else None)
        g["page_share_below_1ms"].append(float(np.mean(rp.gaps < 1.0)) if rp and rp.gaps.size else None)
    return {grp: {"sessions": len(v["delta_plan"]), **{k: vr.summary(x) for k, x in v.items()}}
            for grp, v in vals.items()}


# ---------------------------------------------------------------- run

def diagnose(caps: dict, s_h0: dict, sc) -> dict:
    caps_kept = {s_: [kept_capture(c) for c in cs] for s_, cs in caps.items()}
    return {"repeats_census": {s_: census_series(caps[s_]) for s_ in PAGE_SERIES if s_ in caps},
            "repeats_kept": readings_kept(caps_kept, s_h0, sc),
            "m2_sessions": {s_: m2_groups(caps[s_], caps_kept[s_]) for s_ in ("M2_8B", "M2_14B")}}


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2 or argv[0] != "--captures":
        raise SystemExit("usage: python works/prototype_v7q.py --captures <folder of the production captures>")
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; refusing to overwrite")
    paths = ["works/conditions/prototype_v7q.json", "works/prototype_v7q.py", "works/prototype_v7_run.py",
             "works/prototype_v7.py", "works/prototype_v6.py", "works/_common.py", "works/conditions/prototype_v7.json",
             "works/conditions/prototype_v6.json"]
    cm.same_as_head([ROOT / p for p in paths])
    state = cm.git_state(paths)
    if cm.git("diff", "--name-only", BASE_COMMIT, "HEAD", "--", *V7_CODE, "works/conditions/prototype_v7.json"):
        raise SystemExit(f"v7's analysis code or conditions differ from {BASE_COMMIT}")
    raw_q = COND.read_bytes()
    cq = json.loads(raw_q)
    _, c7 = cm.load_conditions(vr.COND)
    if cm.sha256(vr.COND) != cq["base"]["v7_conditions_sha256"]:
        raise SystemExit("v7's conditions are not the bytes this version is based on")
    _, c6 = cm.load_conditions(vr.COND_V6)
    files = {"essay": ("external/kupa-keys/KUPA-KEYS-TASK-2.csv", vr.kp_sha(c7["data"]["kupa"], "essay")),
             "copy": (c6["data"]["kupa"]["files"]["copy"]["file"], c6["data"]["kupa"]["files"]["copy"]["sha256"])}
    for name, (f, sha) in files.items():
        if cm.sha256(ROOT / f) != sha:
            raise SystemExit(f"{f} does not match the SHA-256 of the conditions ({name})")
    essay, seen = v6.read_kupa(ROOT / files["essay"][0])
    copy_, _ = v6.read_kupa(ROOT / files["copy"][0])
    ps = vr.draw_persons(essay, copy_, seen, c7["persons"]["pool"].split("ids_sha256 ")[1].split(",")[0])
    e_gaps = [v7.gaps_of(essay[p]["times"]) for p in ps.e]
    sc = vr.train([e_gaps[i] for i in ps.e_train])
    s_h0 = sc.score([v7.gaps_of(essay[p]["times"]) for p in ps.h])
    caps = vr.load_all(Path(argv[1]))
    rec = diagnose(caps, s_h0, sc)
    rec["persons"] = {"replay_H_sha256": vr.sha_ids(ps.h), "n": len(ps.h)}
    rec["provenance"] = {"run_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), **state,
                         "conditions_sha256": hashlib.sha256(raw_q).hexdigest(),
                         "code_sha256": {p: cm.sha256(ROOT / p) for p in [*V7_CODE, "works/prototype_v7q.py"]},
                         "data_sha256": {name: sha for name, (_, sha) in files.items()}}
    rec = vr.to_json(rec)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    if OUT.exists():
        tmp.unlink()
        raise SystemExit(f"{OUT} appeared during the run; refusing to overwrite")
    tmp.rename(OUT)
    print(f"written {OUT} (sha256 {hashlib.sha256(OUT.read_bytes()).hexdigest()})")


if __name__ == "__main__":
    main()
