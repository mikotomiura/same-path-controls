"""Tests of works/export_v7s_series.py: v7s's loading recorded unchanged, kinds and times only, h_ms on Firefox's page
only, the series and their licences, and the allow-list."""
import csv
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "works"))
sys.path.insert(0, str(ROOT / "tests"))
import export_v7s_series as ex  # noqa: E402
import prototype_v7_run as run7  # noqa: E402
import prototype_v7s as s7  # noqa: E402
import prototype_v7s_plan as splan  # noqa: E402
import prototype_v7u as vu  # noqa: E402

SECRET = "zqxsecretkeysqzx"   # positive control: plan keys that must never reach a file
N_TEXTS = 3


def _cap(name, status="captured", n=4, kinds=None):
    s_ = name.split("/")[0]
    kinds = kinds or ["alnum", "space", "backspace", "enter"][:n]
    t = np.arange(n) * 12.3456789
    plan = None if status == "no_plan" else run7.Plan(t.copy(), list(kinds), [SECRET] * n)
    cap = run7.Capture(s_, 0, status, plan=plan, redone=False)
    h = None
    if status == "captured":
        cap.layers["page"] = run7.Layer(t * 1.04, list(kinds), 0, 0)      # every layer from its first press: 0
        if s_ not in ex.NOHOOK:
            cap.layers["hook"] = run7.Layer(t + 0.000001 * (np.arange(n) > 0), list(kinds), 0, 0)
        if s_ in ex.FIREFOX:
            h = t * 1.02
    return cap, h


def _items():
    """got, runlog and unit_status as load() returns them: every released series, one failed, one redone, one
    no_plan item in the runlog, and one text without an item."""
    got, runlog, unit_status = {}, {}, {}
    for s_ in ex.KNOWN:
        for ms in (0, 5):
            name = f"{s_}/c{ms:03d}"
            got[name], runlog[name] = _cap(name), "done"
    for s_ in ex.RELEASED:
        if s_ in ex.KNOWN:
            continue
        n_units = 2 if s_ == "F_H1" else N_TEXTS
        unit_status[s_] = []
        for u in range(n_units):
            name = f"{s_}/{ex.UNIT_FORM[s_].format(u)}"
            if s_.startswith("MC") and u == 2:      # a text v7 had no plan for: no item
                unit_status[s_].append("no_plan")
                continue
            st = "failed" if (s_, u) == ("F_M2_8B", 1) else "no_plan" if (s_, u) == ("F_M2_14B", 2) else "captured"
            got[name], runlog[name] = _cap(name, status=st), {"failed": "failed", "no_plan": "no_plan"}.get(st, "done")
            unit_status[s_].append(st)
    got["F_M0/t00"][0].redone = True
    return got, runlog, unit_status


def _read(b: bytes) -> list[list[str]]:
    return list(csv.reader(io.StringIO(b.decode("utf-8"))))


def test_constants_are_the_conditions_and_v7us():
    c = json.loads(ex.COND.read_text(encoding="utf-8"))
    ex.base_check(c)
    for key, val in (("v7u_commit", "0" * 40), ("v7s_record_sha256", "0" * 64)):
        bad = json.loads(json.dumps(c))
        bad["base"][key] = val
        with pytest.raises(SystemExit, match="base"):
            ex.base_check(bad)
    bad = json.loads(json.dumps(c))
    bad["series"]["kupa_derived"] = []
    with pytest.raises(SystemExit, match="series"):
        ex.base_check(bad)
    # every series v7s's plan makes is released (M3 is none of them)
    assert set(ex.RELEASED) == {g for g in splan.GROUPS if g not in ("MC_chrome", "F_M2")} | set(s7.MC) | \
        {"F_M2_8B", "F_M2_14B", "K_chrome_nohook", "F_K_nohook"}
    assert not any(s_.startswith("M3") for s_ in ex.RELEASED)
    assert ex.FIREFOX == vu.FIREFOX


def test_files_hold_kinds_and_times_only_and_h_ms_on_firefox_page_only():
    files = ex.build(*_items())
    assert set(files) == {"kupa-derived/F_H1.csv", "status.csv"} | \
        {f"not-kupa-derived/{s_}.csv" for s_ in ex.NOT_KUPA_DERIVED}
    assert SECRET.encode() not in b"".join(files.values())
    rows = _read(files["kupa-derived/F_H1.csv"])
    assert tuple(rows[0]) == ex.COLUMNS
    h00 = [r for r in rows[1:] if r[1] == "h00"]
    assert [r[2] for r in h00] == ["plan"] * 4 + ["hook"] * 4 + ["page"] * 4
    assert [r[4] for r in h00[:4]] == ["alnum", "space", "backspace", "enter"]
    assert h00[1][5:] == ["12.345679", ""] and h00[5][5:] == ["12.345680", ""]
    assert h00[9][5:] == ["12.839506", "12.592592"]                 # page: t_ms and h_ms
    assert h00[8][5:] == ["0.000000", "0.000000"]
    for name in ("not-kupa-derived/K_chrome.csv", "not-kupa-derived/MC1.csv"):
        assert {r[6] for r in _read(files[name])[1:]} == {""}        # Chrome: no h_ms anywhere
    fk = _read(files["not-kupa-derived/F_K_nohook.csv"])[1:]
    assert {r[2] for r in fk} == {"plan", "page"} and all(r[6] for r in fk if r[2] == "page")


def test_status_follows_the_runlog_and_read_series():
    got, runlog, unit_status = _items()
    files = ex.build(got, runlog, unit_status)
    st = _read(files["status.csv"])
    assert st[0] == list(ex.STATUS_COLUMNS)
    assert ["F_M2_8B", "t01", "failed", "false", "yes"] in st
    assert ["F_M2_14B", "t02", "no_plan", "false", "no"] in st
    assert ["MC3", "t02", "no_plan", "false", "no"] in st              # no item: counted as read_series counts it
    assert ["F_M0", "t00", "captured", "true", "yes"] in st
    assert len(st) - 1 == len(runlog) + sum(us.count("no_plan") for s_, us in unit_status.items() if s_.startswith("MC"))
    for s_, us in unit_status.items():                                # the counts are read_series's
        assert [r[2] for r in st[1:] if r[0] == s_] == us
    m2 = _read(files["not-kupa-derived/F_M2_8B.csv"])[1:]
    assert {r[2] for r in m2 if r[1] == "t01"} == {"plan"}             # failed: its plan only
    assert not [r for r in _read(files["not-kupa-derived/MC3.csv"])[1:] if r[1] == "t02"]


def test_disagreements_between_loading_and_counts_stop():
    got, runlog, unit_status = _items()
    unit_status["F_M0"][1] = "failed"
    with pytest.raises(SystemExit, match="read_series counts failed"):
        ex.build(got, runlog, unit_status)
    got, runlog, unit_status = _items()
    unit_status["MC1"][2] = "captured"                                 # counted but never read
    with pytest.raises(SystemExit, match="not read"):
        ex.build(got, runlog, unit_status)
    got, runlog, unit_status = _items()
    got["F_H1/h05"] = _cap("F_H1/h05")                                 # read but outside the counted units
    with pytest.raises(SystemExit, match="not released"):
        ex.build(got, runlog, unit_status)


def test_captures_outside_load_items_forms_stop():
    cases = [("K_chrome/c000", lambda c, h: (setattr(c, "status", "odd"), h)[1], "status 'odd'"),
             ("K_chrome_nohook/c000", lambda c, h: (c.layers.__setitem__("hook", c.layers["page"]), h)[1], "layers"),
             ("F_M0/t00", lambda c, h: None, "handler times missing"),
             ("MC1/t00", lambda c, h: np.zeros(4), "handler times present"),
             ("F_M0/t00", lambda c, h: h[:2], "2 handler times")]
    for name, change, msg in cases:
        got, runlog, unit_status = _items()
        cap, h = got[name]
        got[name] = (cap, change(cap, h))
        with pytest.raises(SystemExit, match=msg):
            ex.build(got, runlog, unit_status)
    cap, _ = _cap("MC1/t00", status="no_plan")
    cap.plan = run7.Plan(np.zeros(2), ["alnum"] * 2, ["a"] * 2)
    with pytest.raises(SystemExit, match="no_plan with a plan"):
        ex.check_capture("MC1/t00", cap, None)
    got, runlog, unit_status = _items()
    got["MC1/t00"][0].layers["page"] = run7.Layer(np.arange(4.0), ["alnum", "modifier", "alnum", "alnum"], 0, 0)
    with pytest.raises(SystemExit, match="outside the counted kinds"):
        ex.build(got, runlog, unit_status)
    got, runlog, unit_status = _items()
    got["F_K/c000"][1][2] = np.inf
    with pytest.raises(SystemExit, match="not finite"):
        ex.build(got, runlog, unit_status)


def test_the_allow_list_stops_any_other_cell():
    """Positive controls placed in copies of valid files: a 60-character string, a key name, a KUPA-like id, a time
    in another form, an h_ms on a Chrome row or a plan row, a Firefox page row without h_ms, a series of another
    file, a KUPA series under CC BY, a series outside the release."""
    files = ex.build(*_items())
    ex.check_files(files)
    for name, bad_row in [("kupa-derived/F_H1.csv", "F_H1,h00,page,0,alnum," + "x" * 60 + ",1.000000"),
                          ("kupa-derived/F_H1.csv", "F_H1,h00,page,0,KeyA,1.000000,1.000000"),
                          ("kupa-derived/F_H1.csv", "F_H1,P0123,page,0,alnum,1.000000,1.000000"),
                          ("kupa-derived/F_H1.csv", "F_H1,h00,page,0,alnum,1.5,1.000000"),
                          ("kupa-derived/F_H1.csv", "F_H1,h00,page,0,alnum,1.000000,"),
                          ("kupa-derived/F_H1.csv", "F_H1,h00,plan,0,alnum,1.000000,1.000000"),
                          ("kupa-derived/F_H1.csv", "F_H1,h00,page,0,alnum,1.000000,1.000000,extra"),
                          ("kupa-derived/F_H1.csv", "F_M0,t00,page,0,alnum,1.000000,1.000000"),
                          ("not-kupa-derived/MC1.csv", "MC1,t00,page,0,alnum,1.000000,1.000000"),
                          ("not-kupa-derived/K_chrome.csv", "K_chrome,t00,page,0,alnum,1.000000,"),
                          ("status.csv", "M3a,t00,captured,false,yes"),
                          ("status.csv", "F_H1,P0123,captured,false,yes"),
                          # one column at a time (code-reviewer M1): series, layer, index, status, redone, plan
                          ("not-kupa-derived/MC2.csv", "MC1,t00,page,0,alnum,1.000000,"),
                          ("kupa-derived/F_H1.csv", "F_H1,h00,keys,0,alnum,1.000000,"),
                          ("kupa-derived/F_H1.csv", "F_H1,h00,page,x,alnum,1.000000,1.000000"),
                          ("kupa-derived/F_H1.csv", "F_H1,h00,page,\u00b2,alnum,1.000000,1.000000"),
                          ("status.csv", "F_H1,h09,odd,false,yes"),
                          ("status.csv", "F_H1,h09,captured,yes,yes"),
                          ("status.csv", "F_H1,h09,captured,false,maybe")]:
        bad = dict(files)
        bad[name] = files[name] + (bad_row + "\n").encode("utf-8")
        with pytest.raises(SystemExit, match="outside the allowed"):
            ex.check_files(bad)
    bad = dict(files)
    bad["not-kupa-derived/F_H1.csv"] = bad.pop("kupa-derived/F_H1.csv")   # a KUPA series under CC BY
    with pytest.raises(SystemExit, match="not the release's"):
        ex.check_files(bad)
    got, runlog, unit_status = _items()   # an item name outside v7s's forms: build itself runs the check
    got["K_chrome/P0123"], runlog["K_chrome/P0123"] = _cap("K_chrome/P0123"), "done"
    with pytest.raises(SystemExit, match="outside the allowed"):
        ex.build(got, runlog, unit_status)


def test_recorder_keeps_what_load_item_returns_and_stops_a_second_read(tmp_path):
    from test_prototype_v7t import _page

    d = tmp_path / "F_M2_8B" / "t00"
    d.mkdir(parents=True)
    _page(d / "page.ndjson", [4280, 4280, 4296], [4290.0, 4291.0, 4302.0])
    cap = run7.Capture("F_M2_8B", 0, "captured", folder=d)
    cap.layers["page"] = run7.read_page(d / "page.ndjson")
    cap.layers["hook"] = run7.Layer(np.array([0.0, 1.0, 2.0]), ["alnum"] * 3, 0)
    calls = []

    def real(root, runlog, name, hook=True):
        calls.append((name, hook))
        return cap if name.startswith("F_") else run7.Capture("MC1", 0, "failed")

    got = {}
    load = ex.recorder(real, got)
    assert load(tmp_path, {}, "F_M2_8B/t00") is cap                    # v7s gets load_item's capture unchanged
    assert np.array_equal(cap.layers["page"].times, [0.0, 0.0, 16.0])
    assert np.array_equal(got["F_M2_8B/t00"][1], [0.0, 1.0, 12.0])     # v7u's handler times kept beside it
    load(tmp_path, {}, "MC1/t00", hook=False)
    assert calls[-1] == ("MC1/t00", False) and got["MC1/t00"][1] is None
    with pytest.raises(SystemExit, match="read twice"):
        load(tmp_path, {}, "MC1/t00")


def test_load_records_v7s_known_tables_and_unit_caps_on_fake_captures(tmp_path, monkeypatch):
    """v7s's own read_runlogs, known_tables, load_item and unit_caps on v7t's fake folders (read_series stubbed with
    v7s's unit_caps: its plan checks need all of v7's captures); the loader is restored."""
    from test_prototype_v7t import _fake_roots, _folder

    v7, v7s = _fake_roots(tmp_path)
    for meta in list(v7s.glob("K_chrome*/*/meta.json")) + list(v7.glob("P_*/*/meta.json")):
        meta.write_text(json.dumps({"browser": {"name": "chrome", "version": s7.CHROME_VERSION}}), encoding="utf-8")
    import prototype_v7t as vt
    from test_prototype_v7t import _page

    for d in list(v7s.glob("*/c000")):   # a 0 ms plan as the work PC records it (as v7u's test does)
        h = 1000.0 + np.floor(np.arange(40) / 2.0)
        firefox = d.parent.name.startswith("F_")
        t = np.round(np.floor(h / vt.STEP_MS) * vt.STEP_MS) if firefox else h + np.arange(40) * 0.01
        (d / "page.ndjson").unlink()
        _page(d / "page.ndjson", t, h)
        if (d / "hook.ndjson").exists():
            rows = [{"type": "header"}] + [{"type": "key", "t_ns": 2_000_000_000 + 1000 * i, "down": True,
                                            "injected": True, "ours": True, "repeat": False, "kind": "alnum"}
                                           for i in range(40)] + [{"type": "footer"}]
            (d / "hook.ndjson").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    for ms in s7.v7.P_NOHOOK_MS:
        _folder(v7 / run7.item_name("P_nohook", int(ms)), 1000.0 + np.arange(40) * ms, False, "chrome", False,
                "P_nohook")
    v7caps = {"P_idle": [run7.load_capture(v7, {run7.item_name("P_idle", int(ms)): "done"}, "P_idle", int(ms))
                         for ms in run7.CONTROLS_MS],
              "P_nohook": [run7.load_capture(v7, {run7.item_name("P_nohook", int(ms)): "done"}, "P_nohook", int(ms),
                                             hook=False) for ms in s7.v7.P_NOHOOK_MS]}
    monkeypatch.setattr(run7, "load_all", lambda root: v7caps)

    def read_series(v7caps_, root, runlog, h0, kinds):
        kinds["firefox_version"] = "1"
        out = {}
        for s_, n, f in [*[(s_, 7, "t{:02d}") for s_ in (*s7.MC, *s7.F_MACHINE)], ("F_H1", splan.N_F_H1, "h{:02d}")]:
            caps = s7.unit_caps(root, runlog, s_, range(n), f)
            out[s_] = s7.Series(s_, [], {}, [c.status for c in caps], s_, "", [])
        return out

    monkeypatch.setattr(s7, "read_series", read_series)
    real = s7.load_item
    got, runlog, unit_status = ex.load(v7, v7s, [])
    assert s7.load_item is real
    assert set(got) == set(runlog)
    assert unit_status["F_M2_14B"][6] == "no_plan" and unit_status["MC1"][6] == "no_plan"   # in the runlog / not
    for name, (cap, h) in got.items():
        firefox = name.split("/")[0] in ex.FIREFOX and cap.status == "captured"
        assert (h is not None) == firefox
        if firefox:
            assert h.size == cap.layers["page"].times.size and h[0] == 0.0
    assert set(got["F_K_nohook/c003"][0].layers) == {"page"}           # known_tables passed hook=False
    files = ex.build(got, runlog, unit_status)
    fm = _read(files["not-kupa-derived/F_M0.csv"])[1:]
    assert all(r[6] for r in fm if r[2] == "page") and not any(r[6] for r in fm if r[2] != "page")


def test_h0_of_is_v7us_copy_of_v7s_main_without_the_scorer():
    """Statement for statement: every statement of h0_of is one of v7u's persons_and_scorer (which v7u's tests check
    against v7s's main), except that KUPA's keys are read for the H persons only."""
    import ast
    import inspect
    import re

    def stmts(f) -> list[str]:
        src = inspect.cleandoc("\n".join(inspect.getsource(f).splitlines()[1:]))
        return [re.sub(r"\bs7\.", "", ast.unparse(n)) for n in ast.parse(src).body
                if not ast.unparse(n).startswith(("'", '"'))]

    theirs = stmts(vu.persons_and_scorer)
    ours = stmts(ex.h0_of)
    keyed = "keyed = v7plan.read_kupa_keys(ROOT / files['essay'][0], set(ps.h))"
    assert keyed in ours
    assert keyed.replace("set(ps.h)", "set(ps.h) | set(e_train_ids)") in theirs
    assert ours[-1] == "return [sess_of(p) for p in ps.h]" and "h0 = [sess_of(p) for p in ps.h]" in theirs
    # what is left out of v7u's copy, named (Codex: a subset check alone passes a deleted statement)
    left_out = ("v7rec_path = ", "if cm.sha256(v7rec_path)", "v7rec = ", "e_train_ids = ", "etr = ", "scorer = ",
                "gate = ", "valid = ", "return (h0, scorer", "h0 = ", "keyed = ")
    kept = [x for x in theirs if not x.startswith(left_out)]
    assert [x for x in ours if x not in (keyed, ours[-1])] == kept
    assert len([x for x in theirs if x.startswith(left_out)]) == len(left_out)


def test_main_refuses_without_arguments():
    with pytest.raises(SystemExit):
        ex.main([])


def _with(files, name, edit):
    bad = dict(files)
    bad[name] = edit(files[name].decode("utf-8")).encode("utf-8")
    return bad


def test_the_structure_stops_absolute_times_gaps_and_layers_status_does_not_allow():
    """code-reviewer M2 and Codex 1: a layer's first time (or h_ms) not 0 (an absolute clock reading), i not 0, 1, 2,
    ..., hook or page rows of a failed unit, plan rows of a unit whose plan is 'no', a missing file, a unit twice in
    status.csv, a unit of a series file missing from status.csv."""
    files = ex.build(*_items())
    first = "F_H1,h00,page,0,alnum,0.000000,0.000000"
    cases = [("kupa-derived/F_H1.csv", lambda t: t.replace(first, "F_H1,h00,page,0,alnum,93512.250000,0.000000"),
              "first time of 0"),
             ("kupa-derived/F_H1.csv", lambda t: t.replace(first, "F_H1,h00,page,0,alnum,0.000000,5012.125000"),
              "first time of 0"),
             ("kupa-derived/F_H1.csv", lambda t: t.replace("F_H1,h00,page,1,", "F_H1,h00,page,2,"), "i = 0, 1"),
             ("not-kupa-derived/F_M2_8B.csv", lambda t: t + "F_M2_8B,t01,page,0,alnum,0.000000,0.000000\n",
              "not what status.csv says"),
             ("status.csv", lambda t: t.replace("F_M0,t00,captured,true,yes", "F_M0,t00,captured,true,no"),
              "not what status.csv says"),
             ("status.csv", lambda t: t.replace("MC3,t02,no_plan,false,no", "MC3,t02,no_plan,false,yes"),
              "plan rows disagree"),
             ("status.csv", lambda t: t + "MC1,t00,captured,false,yes\n", "twice"),
             ("status.csv", lambda t: t.replace("MC1,t01,captured,false,yes\n", ""), "not what status.csv says")]
    for name, edit, msg in cases:
        with pytest.raises(SystemExit, match=msg):
            ex.check_files(_with(files, name, edit))
    bad = dict(files)
    del bad["not-kupa-derived/MC5.csv"]
    with pytest.raises(SystemExit, match="not the release's"):
        ex.check_files(bad)
    ok = _with(files, "kupa-derived/F_H1.csv", lambda t: t.replace(first, "F_H1,h00,page,0,alnum,-0.000000,0.000000"))
    ex.check_files(ok)


def _record(files):
    """A record as v7s writes its counts and path measures, built from the test's status.csv; the measures' keys are
    written as v7s's known_tables makes them (f"{series}/{ms}/{layer}", ms an int)."""
    rows = list(csv.reader(io.StringIO(files["status.csv"].decode("utf-8"))))[1:]
    counts = {}
    for s_ in ex.RELEASED:
        if s_ not in ex.KNOWN:
            mine = [r[2] for r in rows if r[0] == s_]
            counts[s_] = {"sessions": len(mine), **{x: mine.count(x) for x in ex.STATUSES}}
    meas = {}
    for r in rows:
        if r[0] in ex.KNOWN:
            for la in (("page",) if r[0] in ex.NOHOOK else ("hook", "page")):
                meas[f"{r[0]}/{int(r[1][1:])}/{la}"] = {"median_ms": 1.0} if r[2] == "captured" else {"status": r[2]}
    return {"counts": {**counts, "M0": {"sessions": 30}}, "path": {"measures": {**meas, "v7_P_idle/18/page": {}}}}


def test_counts_are_v7s_records_from_status_csv():
    files = ex.build(*_items())
    rec = _record(files)
    assert {"K_chrome/0/hook", "F_K/5/page", "F_K_nohook/0/page"} <= set(rec["path"]["measures"])   # literal keys
    ex.check_counts(files["status.csv"], rec)
    st = files["status.csv"]
    for edit, msg in [(lambda r: r["counts"]["F_M2_8B"].__setitem__("failed", 0), "F_M2_8B"),
                      (lambda r: r["counts"].pop("MC5"), "MC5"),
                      (lambda r: r["path"]["measures"].__setitem__("F_K/5/hook", {"status": "failed"}), "F_K/c005/hook"),
                      (lambda r: r["path"]["measures"].pop("K_chrome_nohook/0/page"), "K_chrome_nohook"),
                      (lambda r: r["path"]["measures"].__setitem__("F_K/18/page", {}), "F_K: the known gaps")]:
        bad = json.loads(json.dumps(rec))
        edit(bad)
        with pytest.raises(SystemExit, match=msg):
            ex.check_counts(st, bad)
    # the input's side missing (Codex): a series or a known gap absent from status.csv
    for drop in ("MC2,", "K_chrome,c005,"):
        lines = st.decode("utf-8").splitlines(keepends=True)
        short = "".join(x for x in lines if not x.startswith(drop)).encode("utf-8")
        with pytest.raises(SystemExit):
            ex.check_counts(short, rec)


def _manifest(files):
    return {"files": ex.ex7.manifest_entries(files),
            "provenance": {"run_at": "2026-11-21T01:02:03+00:00", "git_commit": "a" * 40, "git_dirty": False,
                           "conditions_sha256": "b" * 64, "code_sha256": {p: "c" * 64 for p in ex.code_paths()[1:]},
                           "v7s_record_sha256": ex.V7S_RECORD_SHA256,
                           "v7s_runlogs_sha256": {n: "d" * 64 for n in ex.RUNLOGS}}}


def test_the_manifest_holds_hashes_counts_and_the_time_only():
    files = ex.build(*_items())
    m = _manifest(files)
    ex.check_manifest(m)
    for edit in (lambda m: m["provenance"].__setitem__("user", "someone"),
                 lambda m: m["provenance"].__setitem__("run_at", "C:\\Users\\someone"),
                 lambda m: m["files"]["status.csv"].__setitem__("sha256", "x" * 64),
                 lambda m: m["files"].pop("kupa-derived/F_H1.csv"),
                 lambda m: m["provenance"]["code_sha256"].__setitem__("works/export_v7s_series.py", "KeyA"),
                 lambda m: m["provenance"]["code_sha256"].__setitem__("C:/Users/someone/x.py", "c" * 64),
                 lambda m: m["provenance"]["v7s_runlogs_sha256"].pop("runlog.b3.ndjson"),
                 lambda m: m["provenance"].__setitem__("code_sha256", {})):
        bad = json.loads(json.dumps(m))
        edit(bad)
        with pytest.raises(SystemExit, match="manifest.json"):
            ex.check_manifest(bad)


def test_code_check_passes_on_head_and_stops_on_other_bytes_or_modules(monkeypatch):
    """code-reviewer M4 and M5: v7u's files at ba4bfc8, the files on the way to H0 at 5508490, v7u's and v7s's own
    checks, and every module loaded from its checked file."""
    ex.code_check()
    monkeypatch.setattr(ex, "V7U_COMMIT", "ea634630a29983c41ff905fda738f72abde183e6")   # v7u before its amendments
    with pytest.raises(SystemExit, match="differs from ea63463"):
        ex.code_check()
    monkeypatch.undo()
    monkeypatch.setattr(ex, "V7S_COMMIT", "a6fa5cae45579c2b4cc95ccb938eab6ea622310e")   # pyproject before LightGBM
    with pytest.raises(SystemExit, match="pyproject.toml differs from a6fa5ca"):
        ex.code_check()
    monkeypatch.undo()
    monkeypatch.setattr(vu, "V7S_COMMIT", "668abfdbbf616bb5cc56c028eb7d764ffc1099b5")   # v7s before read_series' fix
    with pytest.raises(SystemExit, match="differs from 668abfd"):
        ex.code_check()
    monkeypatch.undo()
    monkeypatch.setattr(s7, "V7_COMMIT", "fc79abca1ccd0363f5dcc27ebf9966583e803d77")    # v7's run before a6fa5ca
    with pytest.raises(SystemExit, match="differs from fc79abc"):
        ex.code_check()
    monkeypatch.undo()
    monkeypatch.setitem(ex.MODULES, "prototype_v7s", "works/prototype_v7.py")
    with pytest.raises(SystemExit, match="module prototype_v7s"):
        ex.code_check()


def test_main_stops_before_writing_when_a_check_stops(tmp_path, monkeypatch):
    out = tmp_path / "export_v7s_series"
    monkeypatch.setattr(ex, "OUT", out)
    monkeypatch.setattr(ex, "V7U_COMMIT", "ea634630a29983c41ff905fda738f72abde183e6")
    with pytest.raises(SystemExit, match="base is not the code's"):
        ex.main(["--v7-captures", str(tmp_path / "v7"), "--v7s-captures", str(tmp_path / "v7s")])
    assert not out.exists() and not (tmp_path / "export_v7s_series.tmp").exists()
    monkeypatch.undo()
    monkeypatch.setattr(ex, "OUT", out)
    monkeypatch.setattr(ex, "load", lambda *a: (_ for _ in ()).throw(SystemExit("a check of v7s's loading")))
    monkeypatch.setattr(ex, "h0_of", lambda: [])
    monkeypatch.setattr(vu, "runlogs_as_v7s", lambda root, rec: None)
    with pytest.raises(SystemExit, match="v7s's loading"):   # every check up to v7s's loading passes on HEAD
        ex.main(["--v7-captures", str(tmp_path / "v7"), "--v7s-captures", str(tmp_path / "v7s")])
    assert not out.exists()


def test_load_stops_when_an_item_outside_the_runlogs_is_read(tmp_path, monkeypatch):
    """code-reviewer L3: the items read are exactly the runlogs'."""
    monkeypatch.setattr(run7, "load_all", lambda root: {})
    monkeypatch.setattr(s7, "read_runlogs", lambda root: {"F_K/c000": "done"})
    monkeypatch.setattr(s7, "known_tables", lambda v7caps, root, runlog: (
        s7.load_item(root, {"F_K/c000": "no_plan", "F_K/c001": "no_plan"}, "F_K/c000") and
        s7.load_item(root, {"F_K/c001": "no_plan"}, "F_K/c001") and ({}, {})))
    monkeypatch.setattr(s7, "read_series", lambda *a: {})
    monkeypatch.setattr(s7, "check_one_firefox", lambda *a: None)
    with pytest.raises(SystemExit, match="1 items outside the runlogs"):
        ex.load(tmp_path, tmp_path, [])


def test_main_checks_the_counts_and_the_manifest_then_writes(tmp_path, monkeypatch):
    """main from the checks to the files, with v7s's loading and H0 replaced by the test's items (the counts check and
    the manifest check are called with what main builds; the files and manifest are written only after both)."""
    out = tmp_path / "export_v7s_series"
    v7s = tmp_path / "v7s"
    v7s.mkdir()
    for b in range(1, s7.N_BLOCKS + 1):
        (v7s / f"runlog.b{b}.ndjson").write_text("{}\n", encoding="utf-8")
    calls = []
    monkeypatch.setattr(ex, "OUT", out)
    monkeypatch.setattr(ex, "h0_of", lambda: [])
    monkeypatch.setattr(vu, "runlogs_as_v7s", lambda root, rec: None)
    monkeypatch.setattr(ex, "load", lambda v7_root, v7s_root, h0: _items())
    monkeypatch.setattr(ex, "read_record", lambda: {"counts": {}})
    monkeypatch.setattr(ex, "check_counts", lambda st, rec: calls.append(("counts", st == ex.build(*_items())["status.csv"],
                                                                        "counts" in rec)))
    real = ex.check_manifest
    monkeypatch.setattr(ex, "check_manifest", lambda m: (calls.append(("manifest", not out.exists())), real(m)))
    ex.main(["--v7-captures", str(tmp_path / "v7"), "--v7s-captures", str(v7s)])
    assert calls[0] == ("counts", True, True) and calls[1] == ("manifest", True)
    m = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert set(m["files"]) == ex.FILES and set(m["provenance"]["v7s_runlogs_sha256"]) == \
        {f"runlog.b{b}.ndjson" for b in range(1, s7.N_BLOCKS + 1)}
    assert (out / "kupa-derived" / "F_H1.csv").read_bytes() == ex.build(*_items())["kupa-derived/F_H1.csv"]
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        ex.main(["--v7-captures", str(tmp_path / "v7"), "--v7s-captures", str(v7s)])


def test_a_captured_plan_needs_its_layers_and_a_series_without_the_hook_has_none():
    """Codex (second round): a captured unit with a plan has hook and page rows (page only without the hook); hook
    rows in a series without the hook stop."""
    files = ex.build(*_items())
    cut = lambda t, pre: "".join(x for x in t.splitlines(keepends=True) if not x.startswith(pre))  # noqa: E731
    for name, edit, msg in [("kupa-derived/F_H1.csv", lambda t: cut(t, "F_H1,h01,page,"), "layers are not"),
                            ("not-kupa-derived/K_chrome.csv", lambda t: cut(t, "K_chrome,c005,hook,"), "layers are not"),
                            ("not-kupa-derived/F_K_nohook.csv",
                             lambda t: t + "F_K_nohook,c000,hook,0,alnum,0.000000,\n", "layers are not"),
                            ("not-kupa-derived/F_K_nohook.csv",   # a unit not captured: no hook or page rows at all
                             lambda t: t.replace("F_K_nohook,c005,page,", "F_K_nohook,c005,hook,"), "layers are not")]:
        with pytest.raises(SystemExit, match=msg):
            ex.check_files(_with(files, name, edit))


def _write(tmp_path, files):
    out = tmp_path / "export_v7s_series"
    ex.ex7.write_all(out, files, _manifest(files))
    return out


def test_verify_reads_the_returned_folder_as_the_work_pc_checked_it(tmp_path):
    files = ex.build(*_items())
    rec = _record(files)
    out = _write(tmp_path, files)
    assert ex.verify(out, rec)["files"] == ex.ex7.manifest_entries(files)
    p = out / "not-kupa-derived" / "MC1.csv"
    p.write_bytes(p.read_bytes().replace(b"12.839506", b"12.839507"))    # one time changed after the run
    with pytest.raises(SystemExit, match="not the manifest's"):
        ex.verify(out, rec)
    out2 = _write(tmp_path / "b", files)
    (out2 / "keys.txt").write_text("x", encoding="utf-8")                  # a file outside the release
    with pytest.raises(SystemExit, match="not the release's"):
        ex.verify(out2, rec)
    out3 = _write(tmp_path / "c", files)
    bad = json.loads(json.dumps(rec))
    bad["counts"]["F_H1"]["captured"] += 1
    with pytest.raises(SystemExit, match="F_H1"):
        ex.verify(out3, bad)
    with pytest.raises(SystemExit, match="no captures"):
        ex.main(["--verify", str(out3), "--v7-captures", str(tmp_path)])


def test_a_module_of_the_repo_outside_the_checked_list_stops(monkeypatch):
    import types

    m = types.ModuleType("numpy_shadow")
    m.__file__ = str(ex.ROOT / "works" / "numpy_shadow.py")
    monkeypatch.setitem(sys.modules, "numpy_shadow", m)
    with pytest.raises(SystemExit, match="module numpy_shadow was loaded from works/numpy_shadow.py"):
        ex.code_check()
