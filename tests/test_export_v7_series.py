"""Tests of works/export_v7_series.py: kinds and times only, the series released and left out, and the writing."""
import csv
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "works"))
import export_v7_series as ex  # noqa: E402
import prototype_v7_run as vr  # noqa: E402

SECRET = "zqxsecretkeysqzx"   # positive control: plan keys that must never reach a file


def _cap(series, unit, status="captured", n=5, hook=True, kinds=None):
    kinds = kinds or ["alnum", "space", "end", "other", "enter"][:n]
    t = np.arange(n) * 12.3456789
    plan = vr.Plan(t.copy(), list(kinds), [SECRET] * n)
    layers = {} if status != "captured" else {"page": vr.Layer(t + 0.5, list(kinds), 0, 0)}
    if hook and status == "captured":
        layers["hook"] = vr.Layer(t + 0.000001, list(kinds), 0, 0)
    if status == "no_plan":
        plan = None
    return vr.Capture(series, unit, status, plan=plan, layers=layers)


def _caps():
    units = {"H1_idle": [0, 1], "H1_load": [0], "H1_rep": [0], "L0": [0], "L1_idle": [0], "L1_load": [0],
             **{s_: [0, 1] for s_ in vr.MACHINE}, "P_idle": [5], "P_load": [5], "P_nohook": [5]}
    caps = {s_: [_cap(s_, u, hook=s_ != "P_nohook") for u in us] for s_, us in units.items()}
    caps["M2_14B"][1] = _cap("M2_14B", 1, status="failed")
    caps["M1_8B"][1] = _cap("M1_8B", 1, status="no_plan")
    caps["M0"][1].redone = True
    return caps


def _read(b: bytes) -> list[list[str]]:
    return list(csv.reader(io.StringIO(b.decode("utf-8"))))


def test_series_lists_match_the_conditions_and_cover_every_series():
    c = json.loads(ex.COND.read_text(encoding="utf-8"))
    assert tuple(c["series"]["kupa_derived"]) == ex.KUPA_DERIVED
    assert tuple(c["series"]["not_kupa_derived"]) == ex.NOT_KUPA_DERIVED
    assert tuple(c["series"]["not_released"]) == ex.NOT_RELEASED
    every = set(vr.CAPTURE_LAYOUT)
    parts = [set(ex.KUPA_DERIVED), set(ex.NOT_KUPA_DERIVED), set(ex.NOT_RELEASED)]
    assert set().union(*parts) == every and sum(map(len, parts)) == len(every)


def test_files_hold_kinds_and_times_only_and_never_l0_or_l1():
    files = ex.build(_caps())
    assert set(files) == {f"kupa-derived/{s_}.csv" for s_ in ex.KUPA_DERIVED} | \
        {f"not-kupa-derived/{s_}.csv" for s_ in ex.NOT_KUPA_DERIVED} | {"status.csv"}
    blob = b"".join(files.values())
    assert SECRET.encode() not in blob
    for s_ in ex.NOT_RELEASED:
        assert s_.encode() not in blob
    rows = _read(files["kupa-derived/H1_idle.csv"])
    assert tuple(rows[0]) == ex.COLUMNS
    h00 = [r for r in rows[1:] if r[1] == "h00"]
    assert [r[2] for r in h00] == ["plan"] * 5 + ["hook"] * 5 + ["page"] * 5
    assert [r[4] for r in h00[:5]] == ["alnum", "space", "end", "other", "enter"]
    assert [r[3] for r in h00[:5]] == ["0", "1", "2", "3", "4"]
    assert h00[1][5] == "12.345679" and h00[6][5] == "12.345680" and h00[11][5] == "12.845679"


def test_failed_no_plan_and_no_hook():
    files = ex.build(_caps())
    m2 = _read(files["not-kupa-derived/M2_14B.csv"])[1:]
    assert {r[2] for r in m2 if r[1] == "t01"} == {"plan"}          # a failed capture: its plan only
    m1 = _read(files["not-kupa-derived/M1_8B.csv"])[1:]
    assert not [r for r in m1 if r[1] == "t01"]                      # no plan: no row
    pn = _read(files["not-kupa-derived/P_nohook.csv"])[1:]
    assert {r[2] for r in pn} == {"plan", "page"} and {r[1] for r in pn} == {"c005"}
    st = _read(files["status.csv"])
    assert st[0] == ["series", "unit", "status", "redone", "plan"]
    assert ["M2_14B", "t01", "failed", "false", "yes"] in st and ["M1_8B", "t01", "no_plan", "false", "no"] in st
    assert ["M0", "t01", "captured", "true", "yes"] in st and ["M0", "t00", "captured", "false", "yes"] in st
    assert len(st) - 1 == sum(len(_caps()[s_]) for s_ in (*ex.KUPA_DERIVED, *ex.NOT_KUPA_DERIVED))


def test_a_kind_outside_the_counted_kinds_or_a_missing_series_stops():
    caps = _caps()
    caps["M0"][0] = _cap("M0", 0, kinds=["alnum", "modifier", "alnum", "alnum", "alnum"])
    with pytest.raises(SystemExit, match="outside the counted kinds"):
        ex.build(caps)
    caps = _caps()
    del caps["M3b"]
    with pytest.raises(SystemExit, match="missing"):
        ex.build(caps)


def test_manifest_and_writing(tmp_path):
    files = ex.build(_caps())
    m = {"files": ex.manifest_entries(files), "provenance": {}}
    e = m["files"]["kupa-derived/H1_idle.csv"]
    assert e["sha256"] == hashlib.sha256(files["kupa-derived/H1_idle.csv"]).hexdigest()
    assert e["rows"] == 30 and e["presses"] == {"plan": 10, "hook": 10, "page": 10}
    out = tmp_path / "export"
    ex.write_all(out, files, m)
    assert (out / "kupa-derived/H1_idle.csv").read_bytes() == files["kupa-derived/H1_idle.csv"]
    assert json.loads((out / "manifest.json").read_text(encoding="utf-8")) == m
    assert not (tmp_path / "export.tmp").exists()
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        ex.write_all(out, files, m)
    assert not (tmp_path / "export.tmp").exists()   # refused before anything is written
    (tmp_path / "other.tmp").mkdir()
    with pytest.raises(SystemExit, match="move it aside"):
        ex.write_all(tmp_path / "other", files, m)


def test_main_refuses_without_captures_argument():
    with pytest.raises(SystemExit, match="usage"):
        ex.main([])


def _llm(folder: Path, lines: list[dict]) -> None:
    folder.mkdir(parents=True)
    (folder / "llm.ndjson").write_text("".join(json.dumps(x) + "\n" for x in lines), encoding="utf-8")


def test_a_failed_m1_keeps_its_plan_only_when_the_analysis_can_recompute_it(tmp_path):
    """check_plans (v7) sets a failed M1's plan to None when llm.ndjson cannot give the schedule (failed_without_plan);
    the export does the same."""
    caps = _caps()
    bad, good = _cap("M1_14B", 0, status="failed"), _cap("M1_14B", 1, status="failed")
    bad.folder, good.folder = tmp_path / "bad", tmp_path / "good"
    _llm(bad.folder, [{"type": "error", "error": "x"}])
    _llm(good.folder, [{"type": "chunk", "recv_ns": 1_000_000 * (k + 1), "token": "ab", "done": False}
                       for k in range(3)])   # unfinished: no done line, the last token is left out
    caps["M1_14B"] = [bad, good]
    files = ex.build(caps)
    m1 = _read(files["not-kupa-derived/M1_14B.csv"])[1:]
    assert {r[1] for r in m1 if r[2] == "plan"} == {"t01"}
    st = _read(files["status.csv"])
    assert ["M1_14B", "t00", "failed", "false", "no"] in st and ["M1_14B", "t01", "failed", "false", "yes"] in st
    # captured M1 and failed M2 are not touched by the rule
    assert ex.plan_as_analysis(_caps()["M2_14B"][1]) is not None


def test_status_and_layers_outside_load_alls_forms_stop():
    for cap, msg in [(_cap("M0", 0, status="odd"), "status 'odd'"), (_cap("P_nohook", 5), "layers"),
                     (_cap("M0", 0, hook=False), "layers")]:
        caps = _caps()
        caps[cap.series][0] = cap
        with pytest.raises(SystemExit, match=msg):
            ex.build(caps)
    c = _cap("M0", 0, status="no_plan")
    c.plan = vr.Plan(np.zeros(2), ["alnum"] * 2, ["a"] * 2)
    with pytest.raises(SystemExit, match="no_plan with a plan"):
        ex.check_capture(c)


def test_lengths_and_non_finite_times_stop():
    c = _cap("M0", 0)
    c.layers["page"] = vr.Layer(np.arange(3.0), ["alnum"] * 2, 0, 0)
    with pytest.raises(SystemExit, match="3 times but 2 kinds"):
        ex.rows_of(c)
    c = _cap("M0", 0)
    c.layers["page"].times[2] = np.inf
    with pytest.raises(SystemExit, match="not finite"):
        ex.rows_of(c)


def test_the_allow_list_stops_any_other_cell():
    """Positive controls: a 60-character string, a key name, a KUPA-like id and a time in another form, each placed in
    a copy of a valid file, are stopped; the valid files pass."""
    files = ex.build(_caps())
    ex.check_files(files)
    good = files["kupa-derived/H1_idle.csv"].decode("utf-8").splitlines()
    for bad_row in ["H1_idle,h00,page,0,alnum," + "x" * 60, "H1_idle,h00,page,0,KeyA,1.000000",
                    "H1_idle,P0123,page,0,alnum,1.000000", "H1_idle,h00,page,0,alnum,1.5",
                    "H1_idle,h00,page,0,alnum,1.000000,extra", "M0,t00,page,0,alnum,1.000000"]:
        bad = dict(files)
        bad["kupa-derived/H1_idle.csv"] = ("\n".join([*good, bad_row]) + "\n").encode("utf-8")
        with pytest.raises(SystemExit, match="outside the allowed"):
            ex.check_files(bad)
    bad = dict(files)
    bad["not-kupa-derived/H1_idle.csv"] = bad.pop("kupa-derived/H1_idle.csv")   # a KUPA series under CC BY
    with pytest.raises(SystemExit, match="outside the allowed"):
        ex.check_files(bad)
    caps = _caps()
    caps["H1_idle"][1] = _cap("H1_idle", 1000)   # a unit name outside the layout: build itself runs the check
    with pytest.raises(SystemExit, match="outside the allowed"):
        ex.build(caps)
    bad = dict(files)
    bad["status.csv"] = files["status.csv"] + b"L0,e0,captured,false,yes\n"
    with pytest.raises(SystemExit, match="outside the allowed"):
        ex.check_files(bad)


def test_times_from_large_ns_keep_the_ns():
    """A hook time as v7 reads it (t_ns / 1e6, minus the first) keeps whole ns at 6 decimals for a clock of about
    11 days (QPC counts from boot)."""
    t0 = 950_400_000_000_000   # 11 days in ns
    ns = np.array([t0, t0 + 123_456_789, t0 + 987_654_321_987])
    lay = vr._layer(list(ns / 1e6), ["alnum"] * 3, 0)
    c = _cap("M0", 0, n=3, kinds=["alnum"] * 3)
    c.layers["hook"] = lay
    rows = [r for r in ex.rows_of(c) if r[2] == "hook"]
    assert [r[5] for r in rows] == ["0.000000", "123.456789", "987654.321987"]


def test_the_lock_is_a6fa5cas_or_7d02179s_which_only_adds_lightgbm(monkeypatch):
    """uv.lock moved on with v7s (LightGBM): 7d02179's lock only adds lines to a6fa5ca's; the check takes either and
    nothing else."""
    import subprocess

    d = subprocess.run(["git", "diff", "--numstat", ex.LOCK_COMMITS[0], ex.LOCK_COMMITS[1], "--", "uv.lock"],
                       cwd=ex.ROOT, capture_output=True, text=True, check=True).stdout.split()
    assert d[:3] == ["20", "0", "uv.lock"]
    added = subprocess.run(["git", "diff", "-U0", ex.LOCK_COMMITS[0], ex.LOCK_COMMITS[1], "--", "uv.lock"],
                           cwd=ex.ROOT, capture_output=True, text=True, check=True).stdout
    assert "+name = \"lightgbm\"" in added and added.count("\n+[[package]]") == 1
    ex.lock_check()
    monkeypatch.setattr(ex, "LOCK_COMMITS", ("5c34b85", "5c34b85"))
    monkeypatch.setattr(ex, "ROOT", ex.ROOT)
    real = ex.cm.git
    monkeypatch.setattr(ex.cm, "git", lambda *a: "0" * 40 if a[0] == "hash-object" else real(*a))
    with pytest.raises(SystemExit, match="uv.lock is neither"):
        ex.lock_check()
