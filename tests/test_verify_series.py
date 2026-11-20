"""Tests of tools/verify_series.py on copies of the released series (skipped where the series are absent)."""
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
SERIES = ROOT / "series" / "v7"
pytestmark = pytest.mark.skipif(not SERIES.is_dir(), reason="the released series are not here")

import verify_series as vs  # noqa: E402


@pytest.fixture
def copy(tmp_path):
    out = tmp_path / "v7"
    shutil.copytree(SERIES, out)
    return out


@pytest.fixture
def record():
    return json.loads(vs.V7_RECORD.read_text(encoding="utf-8"))


def test_the_released_v7_series_pass(copy, record):
    assert vs.verify_v7(copy, record)["provenance"]["git_commit"] == vs.EXPORT_COMMIT


def test_a_changed_time_an_extra_file_or_other_counts_stop(copy, record, tmp_path):
    p = copy / "not-kupa-derived" / "P_idle.csv"
    b = p.read_bytes()
    old = b"P_idle,c005,plan,1,alnum,5.000000\n"
    assert old in b
    p.write_bytes(b.replace(old, b"P_idle,c005,plan,1,alnum,5.000001\n", 1))   # one time changed by 1 ns
    with pytest.raises(SystemExit, match="not the manifest's"):
        vs.verify_v7(copy, record)
    p.write_bytes(b)
    (copy / "notes.txt").write_text("x", encoding="utf-8")
    with pytest.raises(SystemExit, match="not the release's"):
        vs.verify_v7(copy, record)
    (copy / "notes.txt").unlink()
    bad = json.loads(json.dumps(record))
    bad["failures"]["M2_14B"]["failed_capture"] = 0
    with pytest.raises(SystemExit, match="M2_14B"):
        vs.verify_v7(copy, bad)


def test_a_manifest_naming_another_commit_stops(copy, record):
    m = json.loads((copy / "manifest.json").read_text(encoding="utf-8"))
    m["provenance"]["git_commit"] = "0" * 40
    (copy / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(SystemExit, match="export's commit"):
        vs.verify_v7(copy, record)


def test_structure_a_first_time_not_0_or_a_plan_row_status_does_not_allow(copy, record, monkeypatch):
    """The structure is checked after the manifest; the manifest's entries are recomputed here so that the structure
    check is the one that stops."""
    def edit(rel, old, new):
        p = copy / rel
        t = p.read_text(encoding="utf-8")
        assert old in t
        p.write_text(t.replace(old, new, 1), encoding="utf-8")

    monkeypatch.setattr(vs.ex7, "manifest_entries", lambda files: json.loads(
        (copy / "manifest.json").read_text(encoding="utf-8"))["files"])
    edit("not-kupa-derived/P_idle.csv", "P_idle,c005,plan,0,alnum,0.000000", "P_idle,c005,plan,0,alnum,1.000000")
    with pytest.raises(SystemExit, match="first time of 0"):
        vs.verify_v7(copy, record)
    edit("not-kupa-derived/P_idle.csv", "P_idle,c005,plan,0,alnum,1.000000", "P_idle,c005,plan,0,alnum,0.000000")
    edit("status.csv", "P_idle,c005,captured,false,yes", "P_idle,c005,captured,false,no")
    with pytest.raises(SystemExit, match="status.csv"):
        vs.verify_v7(copy, record)


def test_a_unit_twice_in_status_csv_stops(copy, record, monkeypatch):
    monkeypatch.setattr(vs.ex7, "manifest_entries", lambda files: json.loads(
        (copy / "manifest.json").read_text(encoding="utf-8"))["files"])
    p = copy / "status.csv"
    p.write_text(p.read_text(encoding="utf-8") + "P_idle,c005,captured,false,yes\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="twice"):
        vs.verify_v7(copy, record)


def test_main_checks_v7s_record_and_both_series(copy, tmp_path, monkeypatch):
    """main: v7's record by its SHA-256, v7 through verify_v7, v7s through its export's check and its commit."""
    calls = []
    monkeypatch.setattr(vs, "verify_v7", lambda out, rec: calls.append(("v7", out.name)))
    monkeypatch.setattr(vs.ex7s, "read_record", lambda: {})
    monkeypatch.setattr(vs.ex7s, "verify", lambda out, rec: (calls.append(("v7s", out.name)) or
                                                            {"provenance": {"git_commit": vs.EXPORT_COMMIT}}))
    vs.main()
    assert calls == [("v7", "v7"), ("v7s", "v7s")]
    monkeypatch.setattr(vs.ex7s, "verify", lambda out, rec: {"provenance": {"git_commit": "0" * 40}})
    with pytest.raises(SystemExit, match="series/v7s"):
        vs.main()
    other = tmp_path / "record.json"
    other.write_bytes(vs.V7_RECORD.read_bytes() + b" ")
    monkeypatch.setattr(vs, "V7_RECORD", other)
    with pytest.raises(SystemExit, match="not v7's record"):
        vs.main()
