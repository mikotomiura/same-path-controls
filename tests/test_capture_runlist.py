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
    count: dict = {}

    def runner(args, out):
        out.mkdir(parents=True)
        (out / "meta.json").write_text("{}")
        seen.append(str(out.relative_to(tmp_path / "run")))
        k = count[out.name] = count.get(out.name, -1) + 1
        return status.get(out.name, ["done", "done"])[k]

    res = runlist.run(items(), [1, 2, 3], tmp_path / "run", runner=runner)
    by = {r["item"]: r for r in res}
    assert by["M1"]["result"] == "failed" and by["M1"]["detail"] == "missing_after_redo"
    assert [x["folder"] for x in by["M1"]["attempts"]] == ["M1.try1", "M1"]  # the last attempt holds the item's folder
    assert (tmp_path / "run" / "M1.try1").is_dir() and (tmp_path / "run" / "M1").is_dir()
    assert by["M1_replay"]["result"] == "no_plan" and by["M1_replay"]["detail"] == "missing_dependency"
    assert by["P5"]["result"] == "redone" and by["L1"]["result"] == "done"
    assert not any(n.startswith("M1_replay") for n in seen)
    assert seen.count("M1") == 2 and len([n for n in seen if n.startswith("M1")]) == 2  # one redo only
    log = [json.loads(x) for x in (tmp_path / "run" / "runlog.ndjson").read_text().splitlines()]
    assert log[0]["type"] == "header" and log[-1]["type"] == "footer"
    assert all(r["result"] in ("done", "redone", "failed", "no_plan") for r in log if r["type"] == "item")
    assert log[-1]["counts"] == {"done": 5, "redone": 1, "failed": 1, "no_plan": 1}


def test_m2_failed_plan_is_no_plan_and_not_redone(tmp_path):
    calls = []

    def runner(args, out):
        calls.append(out.name)
        return "m2_failed"

    res = runlist.run([{"name": "M2_8B/t00", "args": ["--llm-plan"], "after": None, "block": 0}], [1],
                      tmp_path / "run", runner=runner)
    assert res[0]["result"] == "no_plan" and res[0]["detail"] == "m2_failed" and calls == ["t00"]


def test_m2_redo_types_the_first_plan_and_never_plans_again(tmp_path):
    got = []

    def runner(args, out):
        got.append(list(args))
        out.mkdir(parents=True)
        if "--llm-plan" in args:
            (out / "events.ndjson").write_text("plan")
            return "interrupted"
        return "done"

    its = [{"name": "M2_8B/t03", "args": ["--llm-plan", "--model", "m", "--seed", "1"], "after": None, "block": 0}]
    res = runlist.run(its, [1], tmp_path / "run", runner=runner)
    first = tmp_path / "run" / "M2_8B" / "t03.try1"
    assert got[1] == ["--model", "m", "--seed", "1", "--events", str(first / "events.ndjson")]
    assert res[0]["result"] == "redone" and (first / "events.ndjson").read_text() == "plan"
    assert res[0]["attempts"][0]["plan_kept_for_redo"] == "M2_8B/t03.try1/events.ndjson"


def test_an_item_that_cannot_be_moved_aside_is_failed_and_the_runlist_goes_on(tmp_path, monkeypatch):
    out = tmp_path / "run"

    def runner(args, folder):
        folder.mkdir(parents=True, exist_ok=True)
        return "interrupted" if folder.name == "a" else "done"

    def refuse(src, dst, *a, **k):
        raise PermissionError("held by a closing process")
    monkeypatch.setattr(runlist, "rename", refuse)
    its = [{"name": "a", "args": [], "after": None, "block": 0}, {"name": "b", "args": [], "after": None, "block": 0}]
    res = {r["item"]: r for r in runlist.run(its, [1], out, runner=runner)}
    assert res["a"]["result"] == "failed" and res["a"]["detail"] == "rename_failed"
    assert "PermissionError" in res["a"]["attempts"][0]["rename_error"] and res["b"]["result"] == "done"
    log = [json.loads(x) for x in (out / "runlog.ndjson").read_text().splitlines()]
    assert log[-1]["type"] == "footer"


def test_an_earlier_runs_folder_stops_the_runlist_and_is_never_read(tmp_path):
    """A runlist made again after a stop finds the stopped run's folders: never read as this run's (code-reviewer
    round 3: the old meta.json's 'done' was counted)."""
    for left in ("a", "a.try1"):
        out = tmp_path / left / "run"
        out.mkdir(parents=True)
        (out / left).mkdir()
        (out / left / "meta.json").write_text('{"status": "done"}')
        calls = []
        its = [{"name": "a", "args": [], "after": None, "block": 0}]
        real_mkdir = Path.mkdir
        Path.mkdir = lambda self, *a, **k: None if self == out else real_mkdir(self, *a, **k)
        try:
            with pytest.raises(RuntimeError, match="exists before its capture"):
                runlist.run(its, [1], out, runner=lambda args, folder: calls.append(folder) or "done")
        finally:
            Path.mkdir = real_mkdir
        assert calls == []
        log = [json.loads(x) for x in (out / "runlog.ndjson").read_text().splitlines()]
        assert log[-1] == {"type": "stopped", "item": "a", "reason": "folder_exists", "folders": [left]}


def test_rename_is_retried_then_raised(tmp_path, monkeypatch):
    calls = []

    def flaky(self, dst):
        calls.append(dst)
        if len(calls) < 3:
            raise PermissionError("in use")
        return orig(self, dst)

    orig = Path.rename
    (tmp_path / "a").mkdir()
    monkeypatch.setattr(Path, "rename", flaky)
    runlist.rename(tmp_path / "a", tmp_path / "b", wait_s=0)
    assert len(calls) == 3 and (tmp_path / "b").is_dir()
    monkeypatch.setattr(Path, "rename", lambda self, dst: (_ for _ in ()).throw(PermissionError("in use")))
    with pytest.raises(PermissionError):
        runlist.rename(tmp_path / "b", tmp_path / "c", tries=3, wait_s=0)


def persons_items(n=12):
    return [{"name": f"H1_{r}/h{p:02d}", "args": [], "after": None, "block": 0, "person": p, "role": r}
            for p in range(n) for r in ("idle", "load")] + [{"name": "P_idle/c005", "args": [], "after": None,
                                                              "block": 0}]


def test_each_person_orders_idle_and_load_by_its_own_key():
    import numpy as np
    for key in ([20261031, 7, 6], [3]):
        seq, _ = runlist.order(persons_items(), key)
        names = [x["name"] for x in seq]
        firsts = []
        for p in range(12):
            want = ("idle", "load")[int(np.random.default_rng([20261031, 7, 7, p]).permutation(2)[0])]
            i, ld = names.index(f"H1_idle/h{p:02d}"), names.index(f"H1_load/h{p:02d}")
            assert (i < ld) == (want == "idle")
            firsts.append(want)
        assert len(set(firsts)) == 2  # the keys give both orders among 12 persons
    # the places come from the block's permutation: the set of positions of a person's two items is the plain draw's
    plain = [x["name"] for x in runlist.order([{k: v for k, v in it.items() if k not in ("person", "role")}
                                               for it in persons_items()], [20261031, 7, 6])[0]]
    seq = [x["name"] for x in runlist.order(persons_items(), [20261031, 7, 6])[0]]
    for p in range(12):
        pair = {f"H1_idle/h{p:02d}", f"H1_load/h{p:02d}"}
        assert {i for i, n in enumerate(plain) if n in pair} == {i for i, n in enumerate(seq) if n in pair}


def test_bad_person_items_are_refused():
    its = persons_items(2)
    with pytest.raises(ValueError):
        runlist.order(its[:1] + its[2:], [1])  # person 0 without its load item
    with pytest.raises(ValueError):
        runlist.order([{**its[0], "block": 1}] + its[1:], [1])
    with pytest.raises(ValueError):
        runlist.order([{**its[0], "role": "x"}] + its[1:], [1])


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
