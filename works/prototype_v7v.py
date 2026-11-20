"""Prototype v7v (post hoc, descriptive; run once on the work PC with v7's and v7s's captures, KUPA and the records of
v7s and v7u).

Conditions: works/conditions/prototype_v7v.json. v7u's run up to the scores, unchanged (both variants of the
recordings, one scorer, one set of margins), then v7s's criterion_cells and human_cells in each variant, to count what
v7s's cells were made of: per series, side and valid detector the sessions judged 'machine'; per cell the same within
the set the cell reads; and the distribution of the margins. Every cell is checked against v7s's and v7u's records.
v7u's code is imported unchanged and checked byte for byte against ba4bfc8 (v7s's against 5508490 and v7t's against
502f31e through v7u's own check). Writes counts and quantiles only; no reading.

Usage (work PC):
    python works/prototype_v7v.py --v7-captures C:\\work\\v7-captures --v7s-captures C:\\work\\v7s-captures
"""
from __future__ import annotations

import argparse
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
import prototype_v7 as v7  # noqa: E402
import prototype_v7_run as run7  # noqa: E402
import prototype_v7s as s7  # noqa: E402
import prototype_v7u as vu  # noqa: E402

COND = ROOT / "works" / "conditions" / "prototype_v7v.json"
OUT = ROOT / "results" / "prototype_v7v" / "record.json"
V7S_RECORD = vu.V7S_RECORD
V7S_RECORD_SHA256 = vu.V7S_RECORD_SHA256
V7U_RECORD = ROOT / "results" / "prototype_v7u" / "record.json"
V7U_RECORD_SHA256 = "63e5b89db721154e4e542be0913ab56b59a4554e0260c9388694a5733345a521"
V7U_COMMIT = "ba4bfc8a153c11d56e3c0f52ee291bf99f883268"
V7U_CODE = ["works/prototype_v7u.py", "works/conditions/prototype_v7u.json"]
VALID_IN_V7S = {"D1": True, "D2_lr": True, "D2_rf": True, "D3": True, "D4": False, "D5": False, "D6": True}
MARGIN_SOURCES = {"H0", "M0", "M1_8B", "M1_14B", "M2_8B", "M2_14B", "M3a", "M3b", "MC1", "MC2", "MC3", "MC5"}
SIDES = ("base", *run7.LAYERS)
FIREFOX_SERIES = (*s7.F_MACHINE, "F_H1")
ALL_SERIES = (*s7.MACHINE_SERIES_ALL, *s7.HUMAN_SERIES)
FIRST = 10


# ---------------------------------------------------------------- counts


def judged_counts(scores: dict, valid: dict, only_firefox_page: bool = False) -> dict:
    """Per series, side and valid detector: sessions, sessions with a judgment, sessions judged 'machine'.
    only_firefox_page: Firefox's series' page only (variant h)."""
    out = {}
    for name in ALL_SERIES:
        for side in SIDES:
            if only_firefox_page and (name not in FIREFOX_SERIES or side != "page"):
                continue
            for d in s7.DETECTORS:
                if not valid[d]:
                    continue
                m, ok = run7.judgments(d, scores[name][side][d])
                out[f"{name}/{side}/{d}"] = {"sessions": int(ok.size), "judged": int(ok.sum()),
                                             "machine": int((m & ok).sum())}
    return out


def cells_with_sets(series: dict, scores: dict, margins: dict, counts: dict, tables: dict, valid: dict,
                    theta: float) -> tuple[dict, dict]:
    """v7s's criterion_cells, unchanged, with v7s's cell_reading wrapped to keep the arrays of each cell's first call
    (the observed reading); the wrap returns cell_reading's own result and is removed afterwards."""
    real = s7.cell_reading
    kept: dict[str, tuple] = {}

    def wrapped(series_, layer, det, before, after, keep):
        key = f"{series_}/{layer}/{det}"
        if key not in kept:
            kept[key] = (np.array(before, dtype=float), np.array(after, dtype=float), np.array(keep, dtype=bool))
        return real(series_, layer, det, before, after, keep)

    s7.cell_reading = wrapped
    try:
        crit = s7.criterion_cells(series, scores, margins, counts, tables, valid, theta)
    finally:
        s7.cell_reading = real
    return crit, kept


def in_cell_counts(cells: dict, kept: dict, valid: dict, only_firefox_page: bool = False) -> tuple[dict, int, list]:
    """Within the set each cell's observed reading is made on: n and the sessions judged 'machine' on the plan side
    and on the recorded side, b and c recomputed by v7's flips; and the cells whose recomputed b, c, n are not the
    cell's observed ones."""
    out, off = {}, []
    for key, cell in cells.items():
        s_, la, d = key.split("/")
        if not valid[d] or (only_firefox_page and (s_ not in s7.F_MACHINE or la != "page")):
            continue
        before, after, keep = kept[key]
        jb, ob = run7.judgments(d, before)
        ja, oa = run7.judgments(d, after)
        ok = ob & oa & keep
        b, c, n = v7.flips(jb[ok], ja[ok])
        out[key] = {"n": n, "machine_plan": int(jb[ok].sum()), "machine_recorded": int(ja[ok].sum()), "b": b, "c": c}
        if [b, c, n] != list(cell["observed_bc"]):
            off.append(key)
    return out, len(off), off[:FIRST]


PERSON_SOURCES = {"H0", "M3a", "M3b"}   # H0 is persons; M3a and M3b are built from donors (persons of E, v7)
SPLIT_MIN = 21                          # a person-derived side's quantiles only if both sides hold this many values
MIN_VALUES = run7.MIN_SUMMARY           # below this many values only the count, for every source


def margin_summary(num: list[float], person: bool) -> dict | None:
    """v7's summary: persons' sources without ends; plans with ends. Below MIN_VALUES values the count only."""
    if len(num) < MIN_VALUES:
        return {"n": len(num)} if num else None
    return run7.summary(num, ends=not person)


def margin_table(margins: dict, base_scores: dict, valid: dict) -> dict:
    """Per source and valid detector: sessions, without a margin, 'none', and the numeric margin_sd's distribution
    (plans with ends; H0, M3a and M3b without), in all and by the unperturbed judgment. For a person-derived source
    the two sides' quantiles are written only if both sides hold at least SPLIT_MIN numeric values, else both sides
    give counts only (a quantile of few persons, or the difference with the whole, would give one person's value).
    The sessions whose judgment and margin disagree (a margin without a judgment or none with one) are counted."""
    out = {}
    for name, per_det in margins.items():
        person = name in PERSON_SOURCES
        for d, ms in per_det.items():
            if not valid[d]:
                continue
            jm, ok = run7.judgments(d, base_scores[name][d])
            if len(ms) != ok.size:
                raise s7.RunStop(f"{name}/{d}: {len(ms)} margins for {ok.size} sessions")

            def nums(mask):
                return [float(m) for m, keep in zip(ms, mask) if keep and m is not None and m != "none"]

            def part(mask, dist: bool):
                sel = [m for m, keep in zip(ms, mask) if keep]
                num = nums(mask)
                return {"sessions": len(sel), "no_margin": sum(m is None for m in sel),
                        "none": sum(isinstance(m, str) and m == "none" for m in sel),
                        "margin_sd_ms": margin_summary(num, person) if dist else {"n": len(num)}}
            sides = {"machine": ok & jm, "human": ok & ~jm}
            split = not person or all(len(nums(m)) >= SPLIT_MIN for m in sides.values())
            row = part(np.ones(ok.size, dtype=bool), True)
            row["by_judgment"] = {k: part(m, split) for k, m in sides.items()}
            row["judgment_not_as_margin"] = int(sum((m is None) != (not o) for m, o in zip(ms, ok)))
            out[f"{name}/{d}"] = row
    return out


# ---------------------------------------------------------------- reproduction


def same_cell(a: dict, b: dict, fields: tuple[str, ...]) -> bool:
    for f in fields:
        x, y = a.get(f), b.get(f)
        if f.endswith("_bc"):
            x, y = (None if x is None else list(x)), (None if y is None else list(y))
        if x != y:
            return False
    return True


def compare(ours: dict, theirs: dict, fields: tuple[str, ...]) -> dict:
    keys = sorted(set(ours) | set(theirs))
    off = [k for k in keys if k not in ours or k not in theirs or not same_cell(ours[k], theirs[k], fields)]
    return {"compared": len(keys), "not_as_deposited": len(off), "first": off[:FIRST]}


MACHINE_FIELDS = ("observed", "predicted", "observed_bc", "predicted_bc")
HUMAN_FIELDS = ("observed", "predicted", "observed_bc", "predicted_bc")


def reproduction(res: dict, tables: dict, jitter: dict, v7s_rec: dict, v7u_rec: dict) -> dict:
    """The run's cells against v7s's record (variant t) and v7u's (variants t and h)."""
    out = {}
    for th in res["t"]["crit"]:
        out[f"v7s_cells_{th}"] = compare(res["t"]["crit"][th]["cells"], v7s_rec["criterion"][th]["cells"],
                                         MACHINE_FIELDS)
        out[f"v7s_human_{th}"] = compare(res["t"]["human"][th], v7s_rec["criterion_human"][th], HUMAN_FIELDS)
    ours_t = {k: {f"{g:g}": float(v) for g, v in t.items()} for k, t in tables.items()}
    tk = {(k, g) for k, t in ours_t.items() for g in t} | {(k, g) for k, t in v7s_rec["path"]["path_sd_tables"].items()
                                                           for g in t}
    off_t = sorted(f"{k}/{g}" for k, g in tk
                   if ours_t.get(k, {}).get(g) != v7s_rec["path"]["path_sd_tables"].get(k, {}).get(g))
    out["v7s_path_sd_tables"] = {"compared": len(tk), "not_as_deposited": len(off_t), "first": off_t[:FIRST]}
    js = s7.to_json(jitter)
    out["v7s_perturbation_jitter"] = compare({k: {"row": v} for k, v in js.items()},
                                             {k: {"row": v} for k, v in v7s_rec["perturbation_jitter"].items()},
                                             ("row",))
    th = f"{s7.THETA:g}"
    for v in vu.VARIANTS:
        ff = {k: c for k, c in res[v]["crit"][th]["cells"].items() if k.split("/")[0] in s7.F_MACHINE}
        out[f"v7u_firefox_cells_{v}"] = compare(ff, v7u_rec["firefox_cells"][v], MACHINE_FIELDS)
        fh = {k: c for k, c in res[v]["human"][th].items() if k.startswith("F_H1/")}
        out[f"v7u_f_h1_{v}"] = compare(fh, v7u_rec["f_h1"][v], ("observed_bc", "predicted_bc"))
    out["all_as_deposited"] = all(p["not_as_deposited"] == 0 for p in out.values() if isinstance(p, dict))
    return out


def expected_keys(series, layers, detectors) -> set[str]:
    return {f"{s_}/{la}/{d}" for s_ in series for la in layers for d in detectors}


# ---------------------------------------------------------------- the run


def code_check() -> None:
    for f in V7U_CODE:
        if cm.git("rev-parse", f"{V7U_COMMIT}:{f}") != cm.git("hash-object", "--no-filters", str(ROOT / f)):
            raise SystemExit(f"{f} differs from {V7U_COMMIT[:7]}")
    vu.code_check()   # v7s's code at 5508490 and v7t's at 502f31e


def base_check(c: dict) -> None:
    b = c["base"]
    if (b["v7u_commit"], b["v7s_record_sha256"], b["v7u_record_sha256"]) != (V7U_COMMIT, V7S_RECORD_SHA256,
                                                                              V7U_RECORD_SHA256):
        raise SystemExit("the conditions' base is not the code's")


def v7u_record_shape(v7u_rec: dict) -> None:
    """The parts of v7u's record this run compares with are there before anything is scored."""
    try:
        ok = all(isinstance(v7u_rec[p][v], dict) for p in ("firefox_cells", "f_h1") for v in vu.VARIANTS)
        ok = ok and all(set(v7u_rec["firefox_cells"][v]) == expected_keys(s7.F_MACHINE, run7.LAYERS, s7.DETECTORS)
                        and set(v7u_rec["f_h1"][v]) == expected_keys(("F_H1",), run7.LAYERS, s7.DETECTORS)
                        for v in vu.VARIANTS)
    except (KeyError, TypeError):
        ok = False
    if not ok:
        raise s7.RunStop("v7u's record does not hold firefox_cells and f_h1 for both variants with the keys this run "
                         "makes")


def v7s_record_parts(v7s_rec: dict) -> None:
    """Beyond v7u's own check (theta 1): every theta's cells and human cells and the jitter table are there."""
    try:
        ok = all(isinstance(v7s_rec["criterion"][f"{t:g}"]["cells"], dict)
                 and isinstance(v7s_rec["criterion_human"][f"{t:g}"], dict) for t in (s7.THETA, *s7.THETA_SENS))
        ok = ok and isinstance(v7s_rec["perturbation_jitter"], dict) and v7s_rec["valid"] == VALID_IN_V7S
        ok = ok and all(set(v7s_rec["criterion"][f"{t:g}"]["cells"]) == expected_keys(s7.MACHINE_SERIES_ALL,
                                                                                      run7.LAYERS, s7.DETECTORS)
                        and set(v7s_rec["criterion_human"][f"{t:g}"]) == expected_keys(s7.HUMAN_SERIES, run7.LAYERS,
                                                                                       s7.DETECTORS)
                        for t in (s7.THETA, *s7.THETA_SENS))
        ok = ok and set(v7s_rec["perturbation_jitter"]) == {f"{m}/{d}" for m in MARGIN_SOURCES for d in s7.DETECTORS}
    except (KeyError, TypeError):
        ok = False
    if not ok:
        raise s7.RunStop("v7s's record does not hold every theta's cells, the jitter table and v7s's validity "
                         "with the keys this run makes")


def source_scores(scorer, sources: dict) -> dict:
    """The unperturbed scores of every source of margins (as v7s's all_margins scores them)."""
    return {name: scorer.score(sess) for name, sess in sources.items()}


def score_variant(loaded: dict, scorer, valid: dict, margins: dict, counts: dict) -> dict:
    """v7s's score_series, criterion_cells (theta 1 with the sets kept) and human_cells, unchanged."""
    series, tables = loaded["series"], loaded["tables"]
    scores = s7.score_series(scorer, series)
    crit, kept = {}, {}
    for t in (s7.THETA, *s7.THETA_SENS):
        if t == s7.THETA:
            crit[f"{t:g}"], kept = cells_with_sets(series, scores, margins, counts, tables, valid, t)
        else:
            crit[f"{t:g}"] = s7.criterion_cells(series, scores, margins, counts, tables, valid, t)
    human = {f"{t:g}": s7.human_cells(series, scores, margins, counts, tables, t) for t in (s7.THETA, *s7.THETA_SENS)}
    return {"scores": scores, "crit": crit, "human": human, "kept": kept}


def run(v7_root: Path, v7s_root: Path, v7s_rec: dict, v7u_rec: dict) -> dict:
    vu.record_shape(v7s_rec)
    v7s_record_parts(v7s_rec)
    v7u_record_shape(v7u_rec)
    vu.runlogs_as_v7s(v7s_root, v7s_rec)
    h0, scorer, valid, files = vu.persons_and_scorer()
    if valid != VALID_IN_V7S:
        raise s7.RunStop(f"the validity is not v7s's: {valid}")
    v7caps = run7.load_all(v7_root)
    loaded = {v: vu.load_variant(v, v7caps, v7s_root, h0) for v in vu.VARIANTS}   # both loaded and checked first
    vu.same_bases(loaded["t"]["series"], loaded["h"]["series"])
    sources = s7.margin_sources(loaded["t"]["series"], h0)
    if set(sources) != MARGIN_SOURCES:
        raise s7.RunStop(f"the sources of margins are not the conditions': {sorted(sources)}")
    margins, counts = s7.all_margins(scorer, sources)   # once, from t
    res = {v: score_variant(loaded[v], scorer, valid, margins, counts) for v in vu.VARIANTS}
    th = f"{s7.THETA:g}"
    in_t, off_t, first_t = in_cell_counts(res["t"]["crit"][th]["cells"], res["t"]["kept"], valid)
    in_h, off_h, first_h = in_cell_counts(res["h"]["crit"][th]["cells"], res["h"]["kept"], valid, True)
    base_scores = source_scores(scorer, sources)
    rep = reproduction(res, loaded["t"]["tables"], s7.jitter_table(counts, sources), v7s_rec, v7u_rec)
    return {
        "valid": valid,
        "reproduction": rep,
        "usable": bool(rep["all_as_deposited"] and off_t == 0 and off_h == 0),
        "judged": {"t": judged_counts(res["t"]["scores"], valid),
                   "h": judged_counts(res["h"]["scores"], valid, only_firefox_page=True)},
        "in_cell": {"t": in_t, "h": in_h,
                    "cells_not_as_observed_bc": {"t": off_t, "h": off_h}, "first_not_as_observed_bc":
                        {"t": first_t, "h": first_h}},
        "margins": margin_table(margins, base_scores, valid),
        "data_sha256": {n: s_ for n, (_, s_) in files.items()},
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v7-captures", required=True, type=Path)
    ap.add_argument("--v7s-captures", required=True, type=Path)
    a = ap.parse_args(argv)
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; refusing to overwrite")
    paths = ["works/conditions/prototype_v7v.json", "works/prototype_v7v.py", *V7U_CODE, *vu.V7S_CODE, *vu.V7T_CODE]
    cm.same_as_head([ROOT / p for p in paths])
    state = cm.git_state(paths)
    code_check()
    raw = COND.read_bytes()
    base_check(json.loads(raw))
    for path, sha, who in ((V7S_RECORD, V7S_RECORD_SHA256, "v7s"), (V7U_RECORD, V7U_RECORD_SHA256, "v7u")):
        if not path.exists() or cm.sha256(path) != sha:
            raise SystemExit(f"{path} is not {who}'s record (its SHA-256)")
    s7.code_check()
    s7.check_constants(json.loads(s7.COND.read_text(encoding="utf-8")))   # as v7s's main does
    rec = run(a.v7_captures, a.v7s_captures, json.loads(V7S_RECORD.read_text(encoding="utf-8")),
              json.loads(V7U_RECORD.read_text(encoding="utf-8")))
    rec["provenance"] = {"run_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), **state,
                         "conditions_sha256": hashlib.sha256(raw).hexdigest(),
                         "code_sha256": {p: cm.sha256(ROOT / p) for p in paths[1:]},
                         "v7s_record_sha256": V7S_RECORD_SHA256, "v7u_record_sha256": V7U_RECORD_SHA256}
    rec = s7.to_json(rec)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    if OUT.exists():
        tmp.unlink()
        raise SystemExit(f"{OUT} appeared during the run; refusing to overwrite")
    tmp.rename(OUT)
    print(json.dumps({"usable": rec["usable"], "all_as_deposited": rec["reproduction"]["all_as_deposited"],
                      "cells_not_as_observed_bc": rec["in_cell"]["cells_not_as_observed_bc"]}, indent=1))
    print(f"written {OUT} (sha256 {hashlib.sha256(OUT.read_bytes()).hexdigest()})")


if __name__ == "__main__":
    main()
