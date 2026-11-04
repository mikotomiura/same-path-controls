"""Tests of capture/runlist.py (order by key, dependencies, one redo, no replacement)."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from capture import runlist  # noqa: E402


def items():
    return [{"name": n, "args": [], "after": d, "block": b} for n, d, b in [
        ("M1", None, 0), ("M1_replay", "M1", 0), ("L0", None, 0), ("L1", "L0", 0), ("P5", None, 0),
        ("P18", None, 0), ("H1", None, 1), ("H1b", None, 1)]]


def test_same_key_same_order_and_other_keys_differ():
    a, _ = runlist.order(items(), [1, 2, 3])
    b, _ = runlist.order(items(), [1, 2, 3])
    assert [x["name"] for x in a] == [x["name"] for x in b]
    assert len({tuple(x["name"] for x in runlist.order(items(), [k])[0]) for k in range(20)}) > 1


def test_dependencies_are_kept_and_blocks_stay_in_order():
    for k in range(50):
        seq, _ = runlist.order(items(), [k])
        pos = {x["name"]: i for i, x in enumerate(seq)}
        assert pos["M1"] < pos["M1_replay"] and pos["L0"] < pos["L1"]
        assert max(pos[n] for n in ("M1", "M1_replay", "L0", "L1", "P5", "P18")) < min(pos["H1"], pos["H1b"])


def test_moves_are_counted():
    moved = [runlist.order(items(), [k])[1] for k in range(50)]
    assert max(moved) >= 1 and min(moved) == 0


def test_bad_lists_are_refused():
    with pytest.raises(ValueError):
        runlist.order(items() + [{"name": "X", "args": [], "after": "H1", "block": 0}], [1])
    with pytest.raises(ValueError):
        runlist.order(items() + [{"name": "Y", "args": [], "after": "nope", "block": 0}], [1])
    with pytest.raises(ValueError):
        runlist.order(items() + [items()[0]], [1])


def test_redo_once_then_missing_and_dependents_not_run(tmp_path):
    seen = []
    status = {"M1": ["interrupted", "failed"], "P5": ["interrupted", "done"]}

    def runner(args, out):
        name = out.name.split(".")[0]
        seen.append(out.name)
        s = status.get(name, ["done", "done"])
        return s[1] if out.name.endswith("redo1") else s[0]

    res = runlist.run(items(), [1, 2, 3], tmp_path / "run", runner=runner)
    by = {r["name"]: r for r in res}
    assert by["M1"]["result"] == "missing_after_redo" and len(by["M1"]["attempts"]) == 2
    assert by["M1_replay"]["result"] == "missing_dependency"
    assert by["P5"]["result"] == "done_after_redo"
    assert by["L1"]["result"] == "done"
    assert not any(n.startswith("M1_replay") for n in seen)
    assert seen.count("M1.redo1") == 1 and "M1.redo2" not in seen  # one redo only
    log = [json.loads(x) for x in (tmp_path / "run" / "runlog.ndjson").read_text().splitlines()]
    assert log[0]["type"] == "header"
    assert log[-1]["counts"]["missing_after_redo"] == 1 and log[-1]["counts"]["missing_dependency"] == 1


def test_out_placeholder_is_resolved(tmp_path):
    its = [{"name": "M1", "args": [], "after": None, "block": 0},
           {"name": "R", "args": ["--events", "{out:M1}/schedule.ndjson"], "after": "M1", "block": 0}]
    got = {}

    def runner(args, out):
        got[out.name] = args
        return "done"

    runlist.run(its, [5], tmp_path / "run", runner=runner)
    assert got["R"] == ["--events", str(tmp_path / "run" / "M1") + "/schedule.ndjson"]


def test_output_folder_is_not_reused(tmp_path):
    (tmp_path / "run").mkdir()
    with pytest.raises(FileExistsError):
        runlist.run(items(), [1], tmp_path / "run", runner=lambda a, o: "done")
