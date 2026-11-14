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
    assert st[0] == ["series", "unit", "status", "redone"]
    assert ["M2_14B", "t01", "failed", "false"] in st and ["M1_8B", "t01", "no_plan", "false"] in st
    assert ["M0", "t01", "captured", "true"] in st and ["M0", "t00", "captured", "false"] in st
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
