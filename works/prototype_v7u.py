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


def load_variant(name: str, v7caps: dict, v7s_root: Path, h0: list) -> dict:
    """v7s's main from the runlogs to the series (every check v7s makes on the recordings), for variant t or h.
    Both variants are loaded before anything is scored, so a check that stops variant h stops in seconds."""
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
    return {"tables": tables, "series": series, "measures": meas}


def same_bases(a: dict, b: dict) -> None:
    """The margins are computed once, from variant t's series: they read only each series' unperturbed side (plans
    or H0) and margins_of, which must therefore be the same in both variants."""
    if set(a) != set(b):
        raise s7.RunStop("the variants hold different series")
    for name in a:
        x, y = a[name], b[name]
        if x.margins_of != y.margins_of or x.status != y.status or len(x.base) != len(y.base):
            raise s7.RunStop(f"{name}: the variants differ outside the recordings")
        for p, q in zip(x.base, y.base):
            if (p is None) != (q is None) or (p is not None and (not np.array_equal(p.times, q.times)
                                                                 or p.kinds != q.kinds)):
                raise s7.RunStop(f"{name}: the variants' unperturbed sides differ")


def score_variant(loaded: dict, h0: list, scorer, valid: dict, margins: dict, counts: dict) -> dict:
    """v7s's main from the scores to the criterion, with the margins of variant t."""
    series, tables = loaded["series"], loaded["tables"]
    scores = s7.score_series(scorer, series)
    crit = {f"{t:g}": s7.criterion_cells(series, scores, margins, counts, tables, valid, t)
            for t in (s7.THETA, *s7.THETA_SENS)}
    human = {f"{t:g}": s7.human_cells(series, scores, margins, counts, tables, t) for t in (s7.THETA, *s7.THETA_SENS)}
    judg = {f"{s_}/page/{d}": run7.judgments(d, scores[s_]["page"][d]) for s_ in (*F_TESTED, "F_H1")
            for d in s7.DETECTORS}
    return {"tables": tables, "crit": crit, "human": human, "judgments": judg}


# ---------------------------------------------------------------- readings


def firefox_verdict(cr: dict, valid: dict) -> dict:
    """v7s's verdict rule over Firefox's tested cells (F_M0, F_M2_8B, F_M2_14B; both layers) of valid detectors."""
    keys = [k for k, c in cr["cells"].items() if c["valid"] and valid[k.split("/")[2]] and k.split("/")[0] in F_TESTED]
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
           or list(cells_t[k]["observed_bc"]) != list(v7s_cells[k]["observed_bc"])
           or list(cells_t[k]["predicted_bc"]) != list(v7s_cells[k]["predicted_bc"])]
    return {"cells": len(keys), "cells_not_as_v7s": len(off), "first_not_as_v7s": off[:10]}


def tables_as_v7s(tables: dict, v7s_tables: dict) -> int:
    """Variant t's path_sd tables against v7s's record: the number of (table, gap) entries that differ."""
    ours = {k: {f"{g:g}": float(v) for g, v in t.items()} for k, t in tables.items()}
    keys = {(k, g) for k, t in ours.items() for g in t} | {(k, g) for k, t in v7s_tables.items() for g in t}
    return sum(ours.get(k, {}).get(g) != v7s_tables.get(k, {}).get(g) for k, g in keys)


def runlogs_as_v7s(v7s_root: Path, v7s_rec: dict) -> None:
    """The captures are those v7s's record was made from: its runlogs' SHA-256 and their runs.json's."""
    pv = v7s_rec["provenance"]
    for f, sha in pv["v7s_runlogs_sha256"].items():
        if cm.sha256(v7s_root / f) != sha:
            raise s7.RunStop(f"{f} is not the runlog v7s's record was made from")
    if run7.read_ndjson(v7s_root / "runlog.b1.ndjson")[0]["runs_sha256"] != pv["v7s_runs_sha256"]:
        raise s7.RunStop("the runlogs name another runs.json than v7s's record")


# ---------------------------------------------------------------- the run


def code_check() -> None:
    for commit, files in ((V7S_COMMIT, V7S_CODE), (V7T_COMMIT, V7T_CODE)):
        for f in files:
            if cm.git("rev-parse", f"{commit}:{f}") != cm.git("hash-object", "--no-filters", str(ROOT / f)):
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

    def sess_of(pid: str) -> s7.Sess:
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


def outside_firefox_page(res: dict) -> int:
    """Cells (machine and human, theta 1) other than Firefox's page whose observed b, c, n differ between t and h."""
    th = f"{s7.THETA:g}"
    n = 0
    for part in ("crit", "human"):
        a = res["t"][part][th]["cells"] if part == "crit" else res["t"][part][th]
        b = res["h"][part][th]["cells"] if part == "crit" else res["h"][part][th]
        for k in a:
            s_, la = k.split("/")[:2]
            if (s_ in (*F_TESTED, "F_H1")) and la == "page":
                continue
            n += list(a[k]["observed_bc"]) != list(b[k]["observed_bc"])
    return n


def record_shape(v7s_rec: dict) -> None:
    """The parts of v7s's record this run reads after scoring are there before anything is scored."""
    th = f"{s7.THETA:g}"
    try:
        ok = (isinstance(v7s_rec["criterion"][th]["cells"], dict) and isinstance(v7s_rec["criterion_human"][th], dict)
              and isinstance(v7s_rec["path"]["path_sd_tables"], dict)
              and isinstance(v7s_rec["provenance"]["v7s_runlogs_sha256"], dict)
              and isinstance(v7s_rec["provenance"]["v7s_runs_sha256"], str))
    except (KeyError, TypeError):
        ok = False
    if not ok:
        raise s7.RunStop("v7s's record does not hold the criterion, criterion_human, path_sd tables and provenance "
                         "this run compares with")


def run(v7_root: Path, v7s_root: Path, v7s_rec: dict) -> dict:
    record_shape(v7s_rec)
    runlogs_as_v7s(v7s_root, v7s_rec)
    h0, scorer, valid, files = persons_and_scorer()
    v7caps = run7.load_all(v7_root)
    loaded = {v: load_variant(v, v7caps, v7s_root, h0) for v in VARIANTS}   # both loaded and checked first
    same_bases(loaded["t"]["series"], loaded["h"]["series"])
    margins, counts = s7.all_margins(scorer, s7.margin_sources(loaded["t"]["series"], h0))   # once, from t
    res = {v: score_variant(loaded[v], h0, scorer, valid, margins, counts) for v in VARIANTS}
    th = f"{s7.THETA:g}"
    rec = {"valid": valid,
           "agreement_t_with_v7s": {**agreement_with_v7s(res["t"]["crit"][th]["cells"],
                                                         v7s_rec["criterion"][th]["cells"]),
                                    "human": agreement_with_v7s(res["t"]["human"][th],
                                                                v7s_rec["criterion_human"][th]),
                                    "path_sd_entries_not_as_v7s": tables_as_v7s(res["t"]["tables"],
                                                                                v7s_rec["path"]["path_sd_tables"])},
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
           "cells_changed_outside_firefox_page": outside_firefox_page(res),
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
