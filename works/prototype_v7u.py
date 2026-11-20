"""Prototype v7u (post hoc; run once on the work PC with v7's and v7s's captures, KUPA and v7s's record).

Conditions: works/conditions/prototype_v7u.json. v7s's analysis run twice with one scorer and one set of margins: as
v7s ran it (variant t: the page at the event's time, t_ms, checked against v7s's record) and with every Firefox
capture's page read at the handler's time (variant h: performance.now(), h_ms, as v7t reads it). Everything else
(persons, detectors, cleaning, predictions, reading rules, Chrome, hook) is v7s's, imported unchanged and checked byte
for byte against 5508490 (v7t's page reader against 502f31e). Writes counts, readings and verdicts only.

Usage (work PC):
    python works/prototype_v7u.py --v7-captures C:\\work\\v7-captures --v7s-captures C:\\work\\v7s-captures
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "works"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

import _common as cm  # noqa: E402
import prototype_v6 as v6  # noqa: E402
import prototype_v7_plan as v7plan  # noqa: E402
import prototype_v7_run as run7  # noqa: E402
import prototype_v7s as s7  # noqa: E402
import prototype_v7t as vt  # noqa: E402

COND = ROOT / "works" / "conditions" / "prototype_v7u.json"
OUT = ROOT / "results" / "prototype_v7u" / "record.json"
V7S_RECORD = ROOT / "results" / "prototype_v7s" / "record.json"
V7S_RECORD_SHA256 = "9ff1d37a64ef12f603fc41a144ee14e9d084457cb374640000c09603848c27ce"
V7S_COMMIT = "550849003164b74187cda8a36dd6311d46487bc2"
V7T_COMMIT = "502f31e7cfc325debc0d23c520b725db18b6f422"
V7S_CODE = ["works/prototype_v7s.py", "works/prototype_v7s_plan.py", "works/conditions/prototype_v7s.json",
            "works/prototype_v7_run.py", "works/prototype_v7_plan.py", "works/prototype_v7.py", "works/prototype_v6.py",
            "works/_common.py", "capture/events.py"]
V7T_CODE = ["works/prototype_v7t.py"]
FIREFOX = ("F_K", "F_K_nohook", "F_M0", "F_M2_8B", "F_M2_14B", "F_H1")
F_TESTED = s7.F_MACHINE   # the Firefox series whose cells v7s tested
VARIANTS = ("t", "h")


# ---------------------------------------------------------------- the page at the handler's time


def page_at_handler(cap: run7.Capture) -> run7.Capture:
    """A copy of a captured Firefox item whose page layer holds the handler's times (h_ms, from the first counted
    press) for the presses v7's read_page counts, kinds and counts as they were; other statuses unchanged."""
    if cap.status != "captured" or cap.folder is None or "page" not in cap.layers:
        return cap
    _, h = vt.page_times(cap.folder / "page.ndjson")   # stops unless its presses are read_page's
    la = cap.layers["page"]
    if h.size != la.times.size:
        raise s7.RunStop(f"{cap.folder}: h_ms for {h.size} presses, the page layer has {la.times.size}")
    out = copy.copy(cap)
    out.layers = dict(cap.layers)
    out.layers["page"] = run7.Layer(h - h[0] if h.size else h, list(la.kinds), la.not_counted, la.repeats)
    return out


def handler_loader(load_item):
    """v7s's load_item, with every Firefox item's page at the handler's time."""
    def load(root: Path, runlog: dict[str, str], name: str, hook: bool = True) -> run7.Capture:
        cap = load_item(root, runlog, name, hook=hook)
        return page_at_handler(cap) if name.split("/")[0] in FIREFOX else cap
    return load


# ---------------------------------------------------------------- one variant of v7s's analysis


def variant(name: str, v7caps: dict, v7s_root: Path, h0: list, scorer, valid: dict, margin_state: dict) -> dict:
    """v7s's main from the known gaps to the criterion, for variant t or h. The margins depend only on the plans and
    H0 (not on any recording) and are computed once (margin_state) and shared."""
    real = s7.load_item
    if name == "h":
        s7.load_item = handler_loader(real)
    try:
        runlog = s7.read_runlogs(v7s_root)
        meas, tables = s7.known_tables(v7caps, v7s_root, runlog)
        kinds: dict = {}
        series = s7.read_series(v7caps, v7s_root, runlog, h0, kinds)
    finally:
        s7.load_item = real
    s7.check_one_firefox(meas.get("firefox_version_known_gaps"), kinds.get("firefox_version"))
    scores = s7.score_series(scorer, series)
    if not margin_state:
        sources = s7.margin_sources(series, h0)
        margin_state["margins"], margin_state["counts"] = s7.all_margins(scorer, sources)
    margins, counts = margin_state["margins"], margin_state["counts"]
    crit = {f"{t:g}": s7.criterion_cells(series, scores, margins, counts, tables, valid, t)
            for t in (s7.THETA, *s7.THETA_SENS)}
    human = {f"{t:g}": s7.human_cells(series, scores, margins, counts, tables, t) for t in (s7.THETA, *s7.THETA_SENS)}
    judg = {f"{s_}/page/{d}": run7.judgments(d, scores[s_]["page"][d]) for s_ in (*F_TESTED, "F_H1")
            for d in s7.DETECTORS}
    return {"tables": tables, "crit": crit, "human": human, "judgments": judg, "measures": meas}


# ---------------------------------------------------------------- readings


def firefox_verdict(cr: dict, valid: dict) -> dict:
    """v7s's verdict rule over Firefox's tested cells (F_M0, F_M2_8B, F_M2_14B; both layers) of valid detectors."""
    keys = [k for k, c in cr["cells"].items() if c["valid"] and k.split("/")[0] in F_TESTED]
    return s7.verdict([cr["cells"][k] for k in keys], [p for k in keys for p in cr["sessions"][k]])


def time_choice(cells_t: dict, cells_h: dict, valid: dict) -> dict:
    """conditions.reading.time_choice over Firefox's tested page cells of valid detectors (theta 1): 'changes' if one
    cell's observed readings are decided in both variants and differ; 'does not change' if every such cell is decided
    in both and they agree; else 'not decided'."""
    rows, both, differ = {}, 0, 0
    keys = sorted(k for k in cells_t if k.split("/")[0] in F_TESTED and k.split("/")[1] == "page"
                  and valid[k.split("/")[2]])
    for k in keys:
        a, b = cells_t[k]["observed"], cells_h[k]["observed"]
        rows[k] = {"t": a, "h": b, "t_bc": cells_t[k]["observed_bc"], "h_bc": cells_h[k]["observed_bc"]}
        if s7.decided(a) and s7.decided(b):
            both += 1
            differ += a != b
    if differ:
        reading = "changes"
    elif keys and both == len(keys):
        reading = "does not change"
    else:
        reading = "not decided"
    return {"cells": rows, "cells_decided_in_both": both, "cells_differing": differ, "reading": reading}


def sessions_differing(jt: dict, jh: dict) -> dict:
    """Per Firefox series' page and detector: sessions judged in both variants, and those judged differently."""
    out = {}
    for k in jt:
        (a, oa), (b, ob) = jt[k], jh[k]
        ok = oa & ob
        out[k] = {"judged_in_both": int(ok.sum()), "differing": int(np.sum(ok & (a != b)))}
    return out


def agreement_with_v7s(cells_t: dict, v7s_cells: dict) -> dict:
    """Variant t against v7s's record (theta 1): cells whose observed and predicted b, c, n are not v7s's."""
    keys = sorted(set(cells_t) | set(v7s_cells))
    off = [k for k in keys if k not in cells_t or k not in v7s_cells
           or cells_t[k]["observed_bc"] != v7s_cells[k]["observed_bc"]
           or cells_t[k]["predicted_bc"] != v7s_cells[k]["predicted_bc"]]
    return {"cells": len(keys), "cells_not_as_v7s": len(off), "first_not_as_v7s": off[:10]}


# ---------------------------------------------------------------- the run


def code_check() -> None:
    for commit, files in ((V7S_COMMIT, V7S_CODE), (V7T_COMMIT, V7T_CODE)):
        for f in files:
            if cm.git("rev-parse", f"{commit}:{f}") != cm.git("hash-object", str(ROOT / f)):
                raise SystemExit(f"{f} differs from {commit[:7]}")


def base_check(c: dict) -> None:
    b = c["base"]
    if (b["v7s_commit"], b["v7t_commit"], b["v7s_record_sha256"]) != (V7S_COMMIT, V7T_COMMIT, V7S_RECORD_SHA256):
        raise SystemExit("the conditions' base is not the code's")


def persons_and_scorer():
    """v7s's main from the conditions to the scorer and validity, unchanged in substance (v7s's main is not
    importable in parts); every check v7s makes on the data is made here."""
    _, c7 = cm.load_conditions(run7.COND)
    _, c6 = cm.load_conditions(run7.COND_V6)
    kp = c7["data"]["kupa"]
    files = {"essay": ("external/kupa-keys/KUPA-KEYS-TASK-2.csv", run7.kp_sha(kp, "essay")),
             "copy": (c6["data"]["kupa"]["files"]["copy"]["file"], c6["data"]["kupa"]["files"]["copy"]["sha256"])}
    for name, (f, sha) in files.items():
        if cm.sha256(ROOT / f) != sha:
            raise s7.RunStop(f"{f} does not match the SHA-256 of the conditions ({name})")
    v7rec_path = ROOT / "results" / "prototype_v7" / "record.json"
    if cm.sha256(v7rec_path) != s7.V7_RECORD_SHA256:
        raise s7.RunStop("results/prototype_v7/record.json is not v7's record (its SHA-256)")
    v7rec = json.loads(v7rec_path.read_text(encoding="utf-8"))
    essay, seen = v6.read_kupa(ROOT / files["essay"][0])
    copy_, _ = v6.read_kupa(ROOT / files["copy"][0])
    v6_sha = c7["persons"]["pool"].split("ids_sha256 ")[1].split(",")[0]
    ps = run7.draw_persons(essay, copy_, seen, v6_sha)
    e_train_ids = [ps.e[i] for i in ps.e_train]
    keyed = v7plan.read_kupa_keys(ROOT / files["essay"][0], set(ps.h) | set(e_train_ids))

    def sess_of(pid: str):
        t, k = keyed[pid]
        if not run7.same_times(t, essay[pid]["times"]):
            raise s7.RunStop("the keys' reading of KUPA does not give the persons' times")
        return s7.Sess(t, s7.kinds_of_keys(k))

    h0 = [sess_of(p) for p in ps.h]
    etr = [sess_of(p) for p in e_train_ids]
    scorer = s7.train_all(etr)
    gate = s7.gate_added(etr)
    valid = {**{d: bool(v7rec["validity_gate"]["valid"][d]) for d in run7.DETECTORS},
             **{d: gate[d]["valid"] for d in s7.ADDED}}
    return h0, scorer, valid, files


def run(v7_root: Path, v7s_root: Path, v7s_rec: dict) -> dict:
    h0, scorer, valid, files = persons_and_scorer()
    v7caps = run7.load_all(v7_root)
    state: dict = {}
    res = {v: variant(v, v7caps, v7s_root, h0, scorer, valid, state) for v in VARIANTS}
    th = f"{s7.THETA:g}"
    rec = {"valid": valid,
           "agreement_t_with_v7s": agreement_with_v7s(res["t"]["crit"][th]["cells"],
                                                      v7s_rec["criterion"][th]["cells"]),
           "firefox_page_path_sd": {v: {f"{g:g}": x for g, x in res[v]["tables"]["firefox/page"].items()}
                                    for v in VARIANTS},
           "firefox_verdict": {v: {t: firefox_verdict(res[v]["crit"][t], valid) for t in res[v]["crit"]}
                               for v in VARIANTS},
           "verdict_all": {v: {t: s7.verdicts(res[v]["crit"][t], valid) for t in res[v]["crit"]} for v in VARIANTS},
           "firefox_cells": {v: {k: {x: c[x] for x in ("observed", "predicted", "observed_bc", "predicted_bc",
                                                        "left_out_no_plan", "left_out_other", "extrapolated",
                                                        "valid")}
                                 for k, c in res[v]["crit"][th]["cells"].items() if k.split("/")[0] in F_TESTED}
                             for v in VARIANTS},
           "f_h1": {v: {k: {x: c[x] for x in ("observed_bc", "predicted_bc")}
                        for k, c in res[v]["human"][th].items() if k.startswith("F_H1/")} for v in VARIANTS},
           "time_choice": time_choice(res["t"]["crit"][th]["cells"], res["h"]["crit"][th]["cells"], valid),
           "sessions_differing_t_h": sessions_differing(res["t"]["judgments"], res["h"]["judgments"]),
           "cells_changed_outside_firefox_page": int(sum(
               res["t"]["crit"][th]["cells"][k]["observed_bc"] != res["h"]["crit"][th]["cells"][k]["observed_bc"]
               for k in res["t"]["crit"][th]["cells"]
               if k.split("/")[0] not in F_TESTED or k.split("/")[1] == "hook")),
           "data_sha256": {n: s_ for n, (_, s_) in files.items()}}
    return rec


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v7-captures", required=True, type=Path)
    ap.add_argument("--v7s-captures", required=True, type=Path)
    a = ap.parse_args(argv)
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; refusing to overwrite")
    paths = ["works/conditions/prototype_v7u.json", "works/prototype_v7u.py", *V7S_CODE, *V7T_CODE]
    cm.same_as_head([ROOT / p for p in paths])
    state = cm.git_state(paths)
    code_check()
    raw = COND.read_bytes()
    base_check(json.loads(raw))
    if not V7S_RECORD.exists() or cm.sha256(V7S_RECORD) != V7S_RECORD_SHA256:
        raise SystemExit(f"{V7S_RECORD} is not v7s's record (its SHA-256)")
    s7.code_check()
    s7.check_constants(json.loads(s7.COND.read_text(encoding="utf-8")))   # as v7s's main does
    rec = run(a.v7_captures, a.v7s_captures, json.loads(V7S_RECORD.read_text(encoding="utf-8")))
    rec["provenance"] = {"run_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), **state,
                         "conditions_sha256": hashlib.sha256(raw).hexdigest(),
                         "code_sha256": {p: cm.sha256(ROOT / p) for p in paths[1:]},
                         "v7s_record_sha256": V7S_RECORD_SHA256}
    rec = s7.to_json(rec)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    if OUT.exists():
        tmp.unlink()
        raise SystemExit(f"{OUT} appeared during the run; refusing to overwrite")
    tmp.rename(OUT)
    print(json.dumps({"agreement_t_with_v7s": rec["agreement_t_with_v7s"]["cells_not_as_v7s"],
                      "time_choice": rec["time_choice"]["reading"],
                      "firefox_verdict_theta1": {v: rec["firefox_verdict"][v]["1"]["verdict"] for v in VARIANTS}},
                     indent=1))
    print(f"written {OUT} (sha256 {hashlib.sha256(OUT.read_bytes()).hexdigest()})")


if __name__ == "__main__":
    main()
