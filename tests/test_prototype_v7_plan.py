"""Tests of works/prototype_v7_plan.py (the production plan) and of its contract with the runlist and the analysis:
the plan made from synthetic persons is run by capture/runlist.py block by block with a fake runner (no Windows, no
LLM), and the plans the captures hold pass prototype_v7_run.check_plans. No KUPA data is read."""

import csv
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "works"))
sys.path.insert(0, str(ROOT))
import prototype_v6 as v6  # noqa: E402
import prototype_v7 as v7  # noqa: E402
import prototype_v7_plan as vp  # noqa: E402
import prototype_v7_run as vr  # noqa: E402
from capture import events as ev  # noqa: E402
from capture import runlist as rl  # noqa: E402

C7 = json.loads((ROOT / "works/conditions/prototype_v7.json").read_text())
N_E = 60
UNTYPABLE = {"é": "alnum", "α": "alnum", "’": "other", "\t": "other"}


def person_keys(n: int, rng: np.random.Generator, odd: bool) -> list[str]:
    pool = list("abcdefghij") + [" ", " ", ".", ",", "Enter", "Backspace", "Delete", "?"]
    keys = [pool[int(rng.integers(len(pool)))] for _ in range(n)]
    if odd:
        for j in rng.choice(n, 3, replace=False):
            keys[int(j)] = list(UNTYPABLE)[int(rng.integers(len(UNTYPABLE)))]
    return keys


def synthetic_persons(rng: np.random.Generator) -> tuple[list[tuple[np.ndarray, list[str]]], list[dict]]:
    hs, h0 = [], []
    for i in range(vr.N_H):
        n = int(rng.integers(30, 60))
        t = np.concatenate([[0.0], np.cumsum(rng.uniform(40, 400, n - 1).round(1))])
        k = person_keys(n, rng, odd=i % 4 == 0)
        hs.append((t, k))
        h0.append({"times": t, "kinds": [v6.kupa_kind(x) for x in k]})
    return hs, h0


def prompts() -> list[str]:
    return [f"Synthetic task {k}." for k in range(1, 11)] + [C7["data"]["prompts_added"][f"P{k}"] for k in range(11, 41)]


@pytest.fixture(scope="module")
def plan(tmp_path_factory):
    rng = np.random.default_rng(7)
    hs, h0 = synthetic_persons(rng)
    e_gaps = [rng.uniform(50, 600, int(rng.integers(40, 90))) for _ in range(N_E)]
    out = tmp_path_factory.mktemp("plan") / "plan"
    man = vp.build(C7, hs, dict(enumerate(e_gaps)), N_E, prompts(), out)
    return {"out": out, "man": man, "h0": h0, "e_gaps": e_gaps, "hs": hs,
            "items": json.loads((out / "runs.json").read_text())}


# ---------------------------------------------------------------- a fake runner (what capture.session writes)


FAIL_KEEP_PLAN = "M1_8B/t03"      # both attempts fail; schedule.ndjson stays: M0, M3, M1_replay still run
FAIL_NO_PLAN = "M1_8B/t05"        # both attempts fail before any schedule: dependants have no plan
FAIL_L0 = "L0/e1"                 # L0 fails: no l1_events, L1 has no plan
M2_FAILED = "M2_14B/t07"          # the plan fails three times


def arg(args: list[str], flag: str) -> str:
    return args[args.index(flag) + 1]


def fake_runner(args: list[str], folder: Path) -> str:
    folder.mkdir(parents=True, exist_ok=False)
    name = "/".join(folder.parts[-2:])
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    meta = {"command": args}
    status = "done"
    if "--events" in args:
        h, presses = ev.read(Path(arg(args, "--events")))
        ev.write(folder / "events.ndjson", presses, h.get("series", ""), h.get("source", ""))
    elif "--llm-stream" in args:
        if name != FAIL_NO_PLAN:
            n = int(rng.integers(20, 50))
            keys = [str(x) for x in rng.choice(list("abcde fgh."), n)] + ["Enter"]
            t = np.maximum.accumulate(np.concatenate([[0.0], np.cumsum(rng.uniform(-5, 60, n))]))
            ev.write(folder / "schedule.ndjson", ev.make(t, keys), "M1", "fake")
            (folder / "llm.ndjson").write_text("{}\n")
        if name in (FAIL_KEEP_PLAN, FAIL_NO_PLAN):
            status = "interrupted"
    elif "--llm-plan" in args:
        if name == M2_FAILED:
            status = "m2_failed"
        else:
            n = int(rng.integers(20, 40))
            ev.write(folder / "events.ndjson", ev.from_gaps("x" * n, np.concatenate([[0], rng.uniform(1, 90, n - 1)])),
                     "M2", "fake")
    elif "--live" in args:
        if name == FAIL_L0:
            status = "interrupted"
        else:
            kinds = [str(x) for x in rng.choice(["alnum", "space", "end", "other", "backspace"], 40)]
            t = np.concatenate([[0.0], np.cumsum(rng.uniform(60, 300, 39))])
            ev.write(folder / "l1_events.ndjson", ev.make(t, [ev.LIVE_KEY[k] for k in kinds]), "L1", "fake")
    meta["status"] = status
    (folder / "meta.json").write_text(json.dumps(meta))
    return status


def in_process_maker(argv: list[str]) -> int:
    assert argv[0] == "works/prototype_v7_plan.py"
    try:
        vp.main(argv[1:])
    except SystemExit as e:
        return 1 if e.code else 0
    return 0


@pytest.fixture(scope="module")
def captured(plan, tmp_path_factory):
    out = tmp_path_factory.mktemp("captures") / "v7"
    shown = []
    for b in range(vp.N_BLOCKS + 1):
        rl.run(plan["items"], [20261031, 7, 6], out, runner=fake_runner, block=b, maker=in_process_maker,
               shower=shown.append)
    return {"out": out, "shown": shown}


def caps_of(root: Path) -> dict[str, list[vr.Capture]]:
    """The plans of every capture, read as load_capture reads them (layers left out: no logger is faked)."""
    runlog = vr.read_runlogs(root)
    units = {"H1_idle": range(vr.N_H), "H1_load": range(vr.N_H), "H1_rep": range(vr.N_REP),
             "L0": range(vr.N_ESSAYS), "L1_idle": range(vr.N_ESSAYS), "L1_load": range(vr.N_ESSAYS),
             **{s_: range(vr.N_TEXTS) for s_ in vr.MACHINE},
             "P_idle": [int(c) for c in vr.CONTROLS_MS], "P_load": [int(c) for c in vr.CONTROLS_MS],
             "P_nohook": [int(c) for c in v7.P_NOHOOK_MS]}
    out = {}
    for s_, us in units.items():
        out[s_] = []
        for u in us:
            name = vr.item_name(s_, u)
            res = runlog[name]
            d = root / name
            if res == "no_plan":
                out[s_].append(vr.Capture(s_, u, "no_plan"))
                continue
            cap = vr.Capture(s_, u, "failed" if res == "failed" else "captured", folder=d)
            f = d / vr.plan_file(s_)
            if s_ != "L0" and f.exists():
                cap.plan = vr.read_events(f)
            out[s_].append(cap)
    return out


@pytest.fixture()
def m1_from_schedule(monkeypatch):
    """check_plans recomputes M1's plan from llm.ndjson (tested in test_prototype_v7_run); here the fake M1 has no
    token stream, so the recomputation returns the schedule the fake wrote."""
    monkeypatch.setattr(vr, "read_llm_stream", lambda path, unfinished=False: (Path(path), [], 0.0, 0))

    def sched(arr, toks, end):
        p = vr.read_events(Path(arr).parent / "schedule.ndjson")
        return p.times, p.times, p.times, p.keys
    monkeypatch.setattr(vr, "m1_schedule", sched)
    monkeypatch.setattr(v7, "plan_shift", lambda w, d: {"share_moved": 0.0, "median_shift": None})


def l0_hook_of(root: Path) -> list:
    out = []
    for e in range(vr.N_ESSAYS):
        f = root / vr.item_name("L0", e) / "l1_events.ndjson"
        if not f.exists():
            out.append(None)
            continue
        p = vr.read_events(f)
        out.append((p.times, p.kinds))
    return out


# ---------------------------------------------------------------- the contract


def test_the_plan_passes_check_plans_after_the_runlist(plan, captured, m1_from_schedule):
    caps = caps_of(captured["out"])
    vr.check_plans(caps, plan["h0"], plan["e_gaps"], l0_hook_of(captured["out"]))
    assert sum(c.plan is not None for c in caps["M3b"]) == vr.N_TEXTS - 1


def test_failures_reach_the_dependants_as_the_conditions_say(captured):
    runlog = vr.read_runlogs(captured["out"])
    assert runlog[FAIL_KEEP_PLAN] == "failed"
    for s_ in ("M0", "M3a", "M3b", "M1_replay_8B"):
        assert runlog[f"{s_}/t03"] == "done"      # B16: made from the failed capture's kept plan
        assert runlog[f"{s_}/t05"] == "no_plan"   # no plan file left
    assert runlog[FAIL_L0] == "failed"
    assert runlog["L1_idle/e1"] == runlog["L1_load/e1"] == "no_plan"
    assert runlog[M2_FAILED] == "no_plan"
    assert sum(r == "done" for r in runlog.values()) == len(runlog) - 3 - 4 - 2 - 1


def test_every_planned_capture_is_in_the_runlog_once(plan, captured):
    runlog = vr.read_runlogs(captured["out"])
    assert sorted(runlog) == sorted(it["name"] for it in plan["items"])
    assert len(runlog) == plan["man"]["n_items"] == 80 * 2 + 10 + 4 + 8 + 40 + 80 + 80 + 80 + 80 + 10 + 3


def test_l0_shows_its_prompt_before_each_essay(plan, captured):
    assert len(captured["shown"]) == vr.N_ESSAYS
    texts = [Path(p).read_text().strip() for p in captured["shown"]]
    ks = vp.l0_prompts()
    assert sorted(texts) == sorted(prompts()[k] for k in ks)
    assert len(set(ks)) == vr.N_ESSAYS and all(0 <= k < 10 for k in ks)


# ---------------------------------------------------------------- the plan's parts


def test_replayed_presses_keep_h0s_times_and_kinds(plan):
    for i, (t, k) in enumerate(plan["hs"]):
        _, presses = ev.read(plan["out"] / f"events/H/h{i:02d}.ndjson")
        keys = [p["key"] for p in presses]
        vp.check_replay(np.array([p["t_ms"] for p in presses]), keys, plan["h0"][i])
        assert [x for x, y in zip(keys, k) if x != y] == [vp.REPRESENTATIVE[UNTYPABLE[y]] for y in k if y in UNTYPABLE]
    n = sum(y in UNTYPABLE for _, k in plan["hs"] for y in k)
    assert plan["man"]["replaced_keys_H"] == {"presses": n, "persons": sum(any(y in UNTYPABLE for y in k)
                                                                            for _, k in plan["hs"])}


def test_check_replay_stops_on_another_kind_or_time(plan):
    t, k = plan["hs"][0]
    keys, _ = vp.replay_keys(k)
    j = next(i for i, x in enumerate(keys) if x == "a")
    with pytest.raises(SystemExit):
        vp.check_replay(t, keys[:j] + ["."] + keys[j + 1:], plan["h0"][0])
    with pytest.raises(SystemExit):
        vp.check_replay(t + 1.0, keys, plan["h0"][0])


def test_an_untypable_sentence_end_or_space_has_no_representative(monkeypatch):
    monkeypatch.setattr(v6, "kupa_kind", lambda k: "space")
    with pytest.raises(SystemExit):
        vp.replay_keys(["　"])


def test_blocks_hold_every_kind_and_l0_alone(plan):
    items = plan["items"]
    assert {it["name"].split("/")[0] for it in items if it["block"] == 0} == {"L0"}
    for s_ in ("H1_idle", "M1_8B", "M1_14B", "M2_8B", "M2_14B", "M0", "M3a"):
        per = [sum(it["block"] == b and it["name"].startswith(s_ + "/") for it in items) for b in range(1, 7)]
        assert max(per) - min(per) <= 1 and sum(per) == (80 if s_ == "H1_idle" else 40)
    by = {it["name"]: it for it in items}
    for it in items:
        if it.get("after"):
            dep = by[it["after"]]
            assert dep["block"] == it["block"] or dep["name"].startswith("L0/")
        if it["name"].startswith("H1_rep/"):
            assert it["block"] == by["H1_idle/" + it["name"].split("/")[1]]["block"]
            assert it["args"][1] == by["H1_idle/" + it["name"].split("/")[1]]["args"][1]


def test_blocks_of_deals_in_turn_from_its_own_key():
    b = vp.blocks_of(0, 80)
    perm = np.random.default_rng([20261031, 7, 14, 0]).permutation(80)
    assert [b[int(g)] for g in perm[:7]] == [1, 2, 3, 4, 5, 6, 1]
    assert vp.blocks_of(1, 40) != vp.blocks_of(2, 40)


def test_loads_use_the_14b_model_and_their_keys(plan):
    m14 = C7["environment"]["models"]["14B_class"]["name"]
    by = {it["name"]: it for it in plan["items"]}
    for name, k in (("H1_load/h03", vp.h_load_prompt(3)), ("L1_load/e2", vp.other_load_prompt(2)),
                    ("P_load/c050", vp.other_load_prompt(4 + 2))):
        a = by[name]["args"]
        assert arg(a, "--load") == m14
        assert Path(arg(a, "--load-prompt-file")).read_text().strip() == prompts()[k]
    assert vp.h_load_prompt(3) == int(np.random.default_rng([20261031, 7, 1, 3]).integers(10))
    assert not any("--load" in by[n]["args"] for n in by if n.split("/")[0] in ("H1_idle", "H1_rep", "P_idle",
                                                                                  "P_nohook", "L1_idle"))


def test_session_arguments(plan):
    m8, m14 = (C7["environment"]["models"][k]["name"] for k in ("8B_class", "14B_class"))
    by = {it["name"]: it for it in plan["items"]}
    assert all("--auto-focus" in it["args"] for it in plan["items"])
    a = by["M1_14B/t39"]["args"]
    assert "--llm-stream" in a and arg(a, "--model") == m14 and arg(a, "--seed") == "1"
    assert Path(arg(a, "--prompt-file")).read_text().strip() == C7["data"]["prompts_added"]["P40"]
    a = by["M2_8B/t00"]["args"]
    assert "--llm-plan" in a and arg(a, "--model") == m8
    assert Path(arg(a, "--prompt-file")).read_text().strip() == "Synthetic task 1."
    assert "--no-hook" in by["P_nohook/c005"]["args"] and "--no-hook" not in by["P_idle/c005"]["args"]
    _, p = ev.read(Path(arg(by["P_idle/c018"]["args"], "--events")))
    assert len(p) == 1000 and np.allclose(np.diff([x["t_ms"] for x in p]), 18.0)
    assert "".join(x["key"] for x in p[:12]) == "abcdefghijab"


def test_manifest_holds_no_id_and_hashes_every_file(plan):
    files = sorted(p.relative_to(plan["out"]).as_posix() for p in plan["out"].rglob("*") if p.is_file())
    assert sorted(plan["man"]["files_sha256"]) == files
    assert "m3_donors.json" in files


def test_derive_m0_and_m3_are_what_check_plans_reads(plan, tmp_path):
    keys = list("hello world.") + ["Enter"] + list("ok")
    sched = tmp_path / "schedule.ndjson"
    ev.write(sched, ev.make(np.arange(len(keys)) * 7.0, keys), "M1", "test")
    m0 = vp.derive("M0", 4, sched, plan["out"])
    assert [p["key"] for p in m0] == keys and np.allclose(np.diff([p["t_ms"] for p in m0]), 55.0)
    want = v7.m3(4, len(keys) - 1, plan["e_gaps"])
    for s_ in ("M3a", "M3b"):
        got = vp.derive(s_, 4, sched, plan["out"])
        assert [p["key"] for p in got] == keys
        assert np.allclose(np.diff([p["t_ms"] for p in got]), want[s_], atol=1e-9)


def test_derive_refuses_a_plan_of_fewer_than_two_presses(plan, tmp_path):
    sched = tmp_path / "schedule.ndjson"
    ev.write(sched, ev.make([0.0], ["a"]), "M1", "test")
    with pytest.raises(SystemExit):
        vp.derive("M0", 0, sched, plan["out"])


def test_read_kupa_keys_follows_v6s_reading(tmp_path):
    f = tmp_path / "essay.csv"
    rows = [("p1", 30, "down", "b", "False"), ("p1", 10, "down", "a", "False"), ("p1", 10, "down", "x", "False"),
            ("p1", 20, "up", "a", "False"), ("p1", 40, "down", "Shift", "False"), ("p1", 50, "down", "é", "False"),
            ("p1", 60, "down", "c", "True"), ("p2", 5, "down", "Enter", "False"), ("p2", 9, "down", "z", "False")]
    with open(f, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "time", "type", "key", "is_repeat"])
        w.writerows(rows)
    got = vp.read_kupa_keys(f, {"p1", "p2"})
    ref, _ = v6.read_kupa(f)
    for p in ("p1", "p2"):
        assert np.array_equal(got[p][0], ref[p]["times"])
        assert [v6.kupa_kind(k) for k in got[p][1]] == ref[p]["kinds"]
    assert got["p1"][1] == ["a", "b", "é"]
    assert set(vp.read_kupa_keys(f, {"p2"})) == {"p2"}


def test_kupa_prompts_by_prompt_id(tmp_path):
    f = tmp_path / "meta.csv"

    def write(rows):
        with open(f, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["id", "prompt_id", "prompt"])
            w.writerows(rows)
    write([(f"r{i}", 10 - i % 10, f"task {10 - i % 10}") for i in range(30)])
    assert vp.kupa_prompts(f) == [f"task {k}" for k in range(1, 11)]
    write([(f"r{i}", 1 + i % 9, f"task {1 + i % 9}") for i in range(30)])
    with pytest.raises(SystemExit):
        vp.kupa_prompts(f)
    write([(f"r{i}", 1 + i % 10, f"task {1 + i % 10}") for i in range(30)] + [("rx", 3, "another")])
    with pytest.raises(SystemExit):
        vp.kupa_prompts(f)


def test_build_never_writes_into_an_existing_folder(plan):
    with pytest.raises(FileExistsError):
        vp.build(C7, plan["hs"], dict(enumerate(plan["e_gaps"])), N_E, prompts(), plan["out"])


# ---------------------------------------------------------------- the runlist's blocks, make and needs


def test_a_block_before_its_earlier_blocks_stops(plan, tmp_path):
    with pytest.raises(ValueError, match="no runlog"):
        rl.run(plan["items"], [20261031, 7, 6], tmp_path / "x", runner=fake_runner, block=2, maker=in_process_maker,
               shower=lambda p: None)


def test_an_unfinished_earlier_block_stops(plan, tmp_path):
    out = tmp_path / "x"
    out.mkdir()
    (out / "runlog.b0.ndjson").write_text(json.dumps({"type": "header"}) + "\n")
    with pytest.raises(ValueError, match="did not finish"):
        rl.run(plan["items"], [20261031, 7, 6], out, runner=fake_runner, block=1, maker=in_process_maker,
               shower=lambda p: None)


def test_a_block_runs_once(plan, captured):
    with pytest.raises(FileExistsError):
        rl.run(plan["items"], [20261031, 7, 6], captured["out"], runner=fake_runner, block=3, maker=in_process_maker,
               shower=lambda p: None)


def test_blocks_keep_the_order_of_the_whole_permutation(plan, captured):
    seq, _ = rl.order(plan["items"], [20261031, 7, 6])
    for b in range(vp.N_BLOCKS + 1):
        rows = [json.loads(x) for x in (captured["out"] / f"runlog.b{b}.ndjson").read_text().splitlines()]
        assert rows[0]["order"] == [it["name"] for it in seq if it["block"] == b]


def test_a_failed_make_is_no_plan(tmp_path):
    items = [{"name": "M1_8B/t00", "args": ["--llm-stream"], "block": 1},
             {"name": "M0/t00", "args": ["--events", "{root}/d.ndjson"], "block": 1, "after": "M1_8B/t00",
              "needs": ["{out:M1_8B/t00}/schedule.ndjson"], "make": ["works/prototype_v7_plan.py", "derive"]}]
    res = rl.run(items, [1], tmp_path / "x", runner=fake_runner, maker=lambda a: 2, shower=lambda p: None)
    assert [(r["item"], r["result"], r["detail"]) for r in res] == [("M1_8B/t00", "done", "done"),
                                                                    ("M0/t00", "no_plan", "make_failed")]


def test_after_without_needs_still_needs_done(tmp_path):
    items = [{"name": "M1_8B/t05", "args": ["--llm-stream"], "block": 1},
             {"name": "x/t00", "args": ["--events", "{out:M1_8B/t05}/schedule.ndjson"], "block": 1,
              "after": "M1_8B/t05"}]
    res = rl.run(items, [1], tmp_path / "x", runner=fake_runner, maker=lambda a: 0, shower=lambda p: None)
    assert res[1]["detail"] == "missing_dependency"


def test_needs_without_after_is_refused():
    with pytest.raises(ValueError):
        rl.check_items([{"name": "a", "args": [], "block": 1, "needs": ["f"]}])


def test_analysis_reads_one_runlog_or_all_seven(captured, tmp_path):
    assert len(vr.read_runlogs(captured["out"])) == 555
    d = tmp_path / "c"
    d.mkdir()
    for b in range(6):
        (d / f"runlog.b{b}.ndjson").write_text((captured["out"] / f"runlog.b{b}.ndjson").read_text())
    with pytest.raises(SystemExit, match="not blocks"):
        vr.read_runlogs(d)
    (d / "runlog.b6.ndjson").write_text((captured["out"] / "runlog.b6.ndjson").read_text())
    assert len(vr.read_runlogs(d)) == 555
    (d / "runlog.ndjson").write_text((captured["out"] / "runlog.b6.ndjson").read_text())
    with pytest.raises(SystemExit, match="one of them"):
        vr.read_runlogs(d)
    (d / "runlog.ndjson").unlink()
    (d / "runlog.b5.ndjson").write_text((captured["out"] / "runlog.b6.ndjson").read_text())
    with pytest.raises(SystemExit, match="two blocks"):
        vr.read_runlogs(d)
