"""Tests of works/prototype_v7_plan.py (the production plan) and of its contract with the runlist and the analysis:
the plan made from synthetic persons is run by capture/runlist.py block by block with a fake runner (no Windows, no
LLM), and the plans the captures hold pass prototype_v7_run.check_plans. No KUPA data is read."""

import csv
import json
import subprocess
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
from capture import hooklog  # noqa: E402
from capture import runlist as rl  # noqa: E402
from test_prototype_v7_run import hook_key, write_layers, write_ndjson  # noqa: E402

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
    man, runs = vp.build(C7, hs, dict(enumerate(e_gaps)), N_E, prompts(), out)
    vp.publish(out, man, runs)
    return {"out": out, "man": man, "h0": h0, "e_gaps": e_gaps, "hs": hs,
            "items": json.loads((out / "runs.json").read_text())}


# ---------------------------------------------------------------- a fake runner (what capture.session writes)


FAIL_KEEP_PLAN = "M1_8B/t03"      # both attempts fail after the generation ended: the plan stays, dependants run
FAIL_NO_PLAN = "M1_8B/t05"        # both attempts fail before any schedule: dependants have no plan
FAIL_ONE_PRESS = "M1_8B/t07"      # fails with a schedule of one press: the analysis reads no plan
FAIL_SHORT_STREAM = "M1_14B/t09"  # fails with one token and no end: the stream cannot be recomputed
FAIL_L0 = "L0/e1"                 # L0 fails after writing l1_events: L1 is not run
M2_FAILED = "M2_14B/t07"          # the plan fails three times
WINDOW = (1_000_000, 10 ** 15)    # L0's typing window on the hook's clock (ns)


def arg(args: list[str], flag: str) -> str:
    return args[args.index(flag) + 1]


def write_m1(folder: Path, rng: np.random.Generator, n_tokens: int, done: bool = True) -> None:
    """A token stream in the capture code's llm.ndjson format, and the schedule the analysis recomputes from it."""
    words = ["the ", "cat", " sat", ".", "\n", "on ", "a", " mat", "\u2019s", ""]
    t = 1_000_000
    rows = [{"type": "header"}, {"type": "request"}]
    for n in range(n_tokens):
        t += int(rng.integers(5, 80)) * 1_000_000
        tok = words[int(rng.integers(len(words)))]
        rows.append({"type": "chunk", "n": n, "recv_ns": t, "token": tok, "n_chars": len(tok), "done": False})
    if done:
        rows.append({"type": "chunk", "n": n_tokens, "recv_ns": t + 9_000_000, "token": "", "n_chars": 0,
                     "done": True})
    write_ndjson(folder / "llm.ndjson", rows)
    try:
        arr, toks, end, _ = vr.read_llm_stream(folder / "llm.ndjson", unfinished=not done)
        _, _, ex, keys = vr.m1_schedule(arr, toks, end)
    except SystemExit:
        return
    ev.write(folder / "schedule.ndjson", ev.make(ex, keys), "M1", "fake")


def fake_runner(args: list[str], folder: Path) -> str:
    folder.mkdir(parents=True, exist_ok=False)
    name = "/".join(folder.parts[-2:])
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    meta = {"command": args}
    status = "done"
    plan_file = None
    if "--events" in args:
        h, presses = ev.read(Path(arg(args, "--events")))
        ev.write(folder / "events.ndjson", presses, h.get("series", ""), h.get("source", ""))
        plan_file = folder / "events.ndjson"
    elif "--llm-stream" in args:
        if name == FAIL_ONE_PRESS:
            write_m1(folder, rng, 30)
            (folder / "schedule.ndjson").unlink()
            ev.write(folder / "schedule.ndjson", ev.make([0.0], ["a"]), "M1", "fake")
        elif name == FAIL_SHORT_STREAM:
            write_m1(folder, rng, 1, done=False)
            ev.write(folder / "schedule.ndjson", ev.make([0.0, 5.0], ["a", "b"]), "M1", "fake")
        elif name != FAIL_NO_PLAN:
            write_m1(folder, rng, int(rng.integers(15, 40)))
            plan_file = folder / "schedule.ndjson"
        if name in (FAIL_KEEP_PLAN, FAIL_NO_PLAN, FAIL_ONE_PRESS, FAIL_SHORT_STREAM):
            status = "interrupted"
    elif "--llm-plan" in args:
        if name == M2_FAILED:
            status = "m2_failed"
        else:
            n = int(rng.integers(20, 40))
            ev.write(folder / "events.ndjson", ev.from_gaps("x" * n, np.concatenate([[0], rng.uniform(1, 90, n - 1)])),
                     "M2", "fake")
            plan_file = folder / "events.ndjson"
    elif "--live" in args:
        meta["typing_start_perf_ns"], meta["typing_end_perf_ns"] = WINDOW
        kinds = [str(x) for x in rng.choice(["alnum", "space", "end", "other", "backspace", "enter"], 60)]
        t = np.concatenate([[2.0], 2.0 + np.cumsum(rng.uniform(60, 300, 59))])
        early = [hook_key(0.5, "alnum", False)]   # before the signal: neither L0 nor L1
        write_layers(folder, t, kinds, rng, injected=False, extra_hook=early)
        tie_on_both_layers(folder)   # a press at the same time as the previous one: dropped by v6's rule
        l1, _ = ev.from_live(hooklog.read(folder / "hook.ndjson"), *WINDOW)
        ev.write(folder / "l1_events.ndjson", l1, "L1", "fake")
        if name == FAIL_L0:   # capture.session: l1_events written, then a stop error in finally makes it failed
            status = "failed"
    if plan_file is not None and status == "done":
        p = vr.read_events(plan_file)
        write_layers(folder, p.times, p.kinds, rng, hook="--no-hook" not in args)
    meta["status"] = status
    (folder / "meta.json").write_text(json.dumps(meta))
    return status


def tie_on_both_layers(folder: Path) -> None:
    """Repeat the 10th counted down at the same time on the hook layer, and its kind on the page layer."""
    rows = hooklog.read(folder / "hook.ndjson")
    downs = [i for i, r in enumerate(rows) if r.get("type") == "key" and r["down"] and not r["repeat"]
             and r["kind"] not in ("modifier", "nonchar") and r["t_ns"] >= WINDOW[0]]
    j = downs[10]
    rows.insert(j + 1, dict(rows[j]))
    write_ndjson(folder / "hook.ndjson", rows)
    page = hooklog.read(folder / "page.ndjson")
    keys = [i for i, r in enumerate(page) if r.get("type") == "key"]
    page.insert(keys[10] + 1, dict(page[keys[10]]))
    write_ndjson(folder / "page.ndjson", page)


def in_process_maker(argv: list[str]) -> int:
    assert argv[0] == "works/prototype_v7_plan.py"
    try:
        vp.main(argv[1:])
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    return 0


@pytest.fixture(scope="module")
def captured(plan, tmp_path_factory):
    out = tmp_path_factory.mktemp("captures") / "v7"
    shown = []
    for b in range(vp.N_BLOCKS + 1):
        rl.run(plan["items"], [20261031, 7, 6], out, runner=fake_runner, block=b, maker=in_process_maker,
               shower=shown.append, runs_sha=runs_file_sha(plan))
    return {"out": out, "shown": shown}


def runs_file_sha(plan) -> str:
    """As capture.runlist main: the SHA-256 of the runs.json file."""
    return vp.sha256(plan["out"] / "runs.json")


# ---------------------------------------------------------------- the contract


def test_the_plan_passes_check_plans_after_the_runlist(plan, captured):
    caps = vr.load_all(captured["out"])   # the analysis's own readers (runlogs, events, hook, page, meta)
    out = vr.check_plans(caps, plan["h0"], plan["e_gaps"], vr.l0_refs(caps))
    assert sum(c.plan is not None for c in caps["M3b"]) == vr.N_TEXTS - 2
    assert out["M1"]["8B"]["sessions_with_plan"] == vr.N_TEXTS - 2
    assert out["M1"]["14B"]["failed_without_plan"] == 1
    assert sum(c.status == "captured" for c in caps["L1_idle"]) == vr.N_ESSAYS - 1


def test_failures_reach_the_dependants_as_the_analysis_reads_them(captured):
    runlog = vr.read_runlogs(captured["out"])
    for m1 in (FAIL_KEEP_PLAN, FAIL_NO_PLAN, FAIL_ONE_PRESS, FAIL_SHORT_STREAM, FAIL_L0):
        assert runlog[m1] == "failed"
    for s_ in ("M0", "M3a", "M3b", "M1_replay_8B"):
        assert runlog[f"{s_}/t03"] == "done"      # B16: made from the failed capture's kept plan
        assert runlog[f"{s_}/t05"] == "no_plan"   # no plan file left
        assert runlog[f"{s_}/t07"] == "no_plan"   # a plan of one press is no plan for the analysis
    assert runlog["M1_replay_14B/t09"] == "no_plan"
    assert runlog["L1_idle/e1"] == runlog["L1_load/e1"] == "no_plan"
    assert runlog[M2_FAILED] == "no_plan"
    assert sum(r == "done" for r in runlog.values()) == len(runlog) - 5 - 12


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
    assert sorted(plan["man"]["files_sha256"]) == sorted(set(files) - {"manifest.json"})
    assert plan["man"]["files_sha256"]["runs.json"] == vp.sha256(plan["out"] / "runs.json")
    assert "m3_donors.json" in files


def m1_folder(tmp_path: Path, status: str = "done", n_tokens: int = 20, seed: int = 3, done: bool = True) -> Path:
    d = tmp_path / f"m1-{status}-{n_tokens}-{done}"
    d.mkdir()
    write_m1(d, np.random.default_rng(seed), n_tokens, done=done)
    (d / "meta.json").write_text(json.dumps({"status": status}))
    return d


def test_derive_m0_and_m3_are_what_check_plans_reads(plan, tmp_path):
    d = m1_folder(tmp_path)
    keys = vr.read_events(d / "schedule.ndjson").keys
    m0 = vp.derive("M0", 4, d, plan["out"])
    assert [p["key"] for p in m0] == keys and np.allclose(np.diff([p["t_ms"] for p in m0]), 55.0)
    want = v7.m3(4, len(keys) - 1, plan["e_gaps"])
    for s_ in ("M3a", "M3b"):
        got = vp.derive(s_, 4, d, plan["out"])
        assert [p["key"] for p in got] == keys
        assert np.allclose(np.diff([p["t_ms"] for p in got]), want[s_], atol=1e-9)


def test_m1_gate_reads_as_the_analysis_reads(tmp_path):
    ok = m1_folder(tmp_path)
    assert vp.m1_plan(ok) == vr.read_events(ok / "schedule.ndjson").keys
    one = m1_folder(tmp_path, "interrupted")
    (one / "schedule.ndjson").unlink()
    with pytest.raises(SystemExit) as e:
        vp.m1_plan(one)
    assert e.value.code == 3
    ev.write(one / "schedule.ndjson", ev.make([0.0], ["a"]), "M1", "t")
    with pytest.raises(SystemExit) as e:
        vp.m1_plan(one)
    assert e.value.code == 3
    bad = m1_folder(tmp_path, "done", 25, seed=4)   # a done M1 whose schedule is not its stream's plan: a fault
    _, presses = ev.read(bad / "schedule.ndjson")
    (bad / "schedule.ndjson").unlink()
    ev.write(bad / "schedule.ndjson", ev.make([p["t_ms"] * 1.01 for p in presses], [p["key"] for p in presses]),
             "M1", "t")
    with pytest.raises(SystemExit) as e:
        vp.m1_plan(bad)
    assert e.value.code not in (0, 3)
    gone = m1_folder(tmp_path, "done", 22, seed=5)
    (gone / "schedule.ndjson").unlink()
    with pytest.raises(SystemExit) as e:
        vp.m1_plan(gone)
    assert e.value.code not in (0, 3)


def test_the_command_line_gives_the_runlists_exit_codes(plan, tmp_path):
    """derive and gate-m1 as the runlist starts them: a new python in the repository (not in this process)."""
    def cli(*a):
        return subprocess.run([sys.executable, "works/prototype_v7_plan.py", *a], cwd=ROOT, capture_output=True,
                              text=True, timeout=300)
    d = m1_folder(tmp_path)
    out = tmp_path / "derived" / "M3b.ndjson"
    r = cli("derive", "--series", "M3b", "--text", "2", "--m1", str(d), "--plan", str(plan["out"]), "--out", str(out))
    assert r.returncode == 0, r.stderr
    _, got = ev.read(out)
    assert np.allclose(np.diff([p["t_ms"] for p in got]), v7.m3(2, len(got) - 1, plan["e_gaps"])["M3b"])
    failed = m1_folder(tmp_path, "interrupted", 1, done=False)   # one token and no end: not recomputable
    ev.write(failed / "schedule.ndjson", ev.make([0.0, 5.0], ["a", "b"]), "M1", "t")
    assert cli("gate-m1", "--m1", str(failed)).returncode == rl.NO_PLAN_EXIT
    (d / "llm.ndjson").write_text('{"type": "error", "error": "x"}\n')
    assert cli("gate-m1", "--m1", str(d)).returncode not in (0, rl.NO_PLAN_EXIT)


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
    with pytest.raises(ValueError, match="has run"):
        rl.run(plan["items"], [20261031, 7, 6], captured["out"], runner=fake_runner, block=3, maker=in_process_maker,
               shower=lambda p: None, runs_sha=runs_file_sha(plan))


def test_blocks_keep_the_order_of_the_whole_permutation(plan, captured):
    seq, _ = rl.order(plan["items"], [20261031, 7, 6])
    for b in range(vp.N_BLOCKS + 1):
        rows = [json.loads(x) for x in (captured["out"] / f"runlog.b{b}.ndjson").read_text().splitlines()]
        assert rows[0]["order"] == [it["name"] for it in seq if it["block"] == b]


def test_make_exit_3_is_no_plan_and_any_other_exit_stops(tmp_path):
    items = [{"name": "M1_8B/t00", "args": ["--llm-stream"], "block": 1},
             {"name": "M0/t00", "args": ["--events", "{root}/d.ndjson"], "block": 1, "after": "M1_8B/t00",
              "needs": ["{out:M1_8B/t00}/schedule.ndjson"], "make": ["works/prototype_v7_plan.py", "derive"]}]
    res = rl.run(items, [1], tmp_path / "x", runner=fake_runner, maker=lambda a: 3, shower=lambda p: None)
    assert [(r["item"], r["result"], r["detail"]) for r in res] == [("M1_8B/t00", "done", "done"),
                                                                    ("M0/t00", "no_plan", "make_no_plan")]
    with pytest.raises(RuntimeError, match="make exited 1"):
        rl.run(items, [1], tmp_path / "y", runner=fake_runner, maker=lambda a: 1, shower=lambda p: None)
    rows = (tmp_path / "y" / "runlog.ndjson").read_text().splitlines()
    assert json.loads(rows[-1])["type"] != "footer"   # the next block refuses an unfinished runlog


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
    rows = (captured["out"] / "runlog.b6.ndjson").read_text().splitlines()
    (d / "runlog.b5.ndjson").write_text("\n".join([json.dumps({**json.loads(rows[0]), "block": 5})] + rows[1:]) + "\n")
    with pytest.raises(SystemExit, match="two blocks"):
        vr.read_runlogs(d)


def blocks_upto(plan, out: Path, last: int, items=None) -> None:
    for b in range(last + 1):
        rl.run(items or plan["items"], [20261031, 7, 6], out, runner=fake_runner, block=b, maker=in_process_maker,
               shower=lambda p: None)


def test_a_later_block_from_another_key_or_runs_stops(plan, tmp_path):
    out = tmp_path / "x"
    blocks_upto(plan, out, 0)
    with pytest.raises(ValueError, match="another key"):
        rl.run(plan["items"], [20261031, 7, 5], out, runner=fake_runner, block=1, maker=in_process_maker,
               shower=lambda p: None)
    other = [dict(it) for it in plan["items"]]
    other[-1] = {**other[-1], "args": other[-1]["args"] + ["--inject", "vk"]}
    with pytest.raises(ValueError, match="another key, runs.json"):
        rl.run(other, [20261031, 7, 6], out, runner=fake_runner, block=1, maker=in_process_maker,
               shower=lambda p: None)


def test_an_earlier_block_must_hold_exactly_its_items(plan, tmp_path):
    out = tmp_path / "x"
    blocks_upto(plan, out, 0)
    f = out / "runlog.b0.ndjson"
    rows = f.read_text().splitlines()
    f.write_text("\n".join(rows[:1] + rows[2:]) + "\n")   # one L0 item left out
    with pytest.raises(ValueError, match="exactly the items"):
        rl.run(plan["items"], [20261031, 7, 6], out, runner=fake_runner, block=1, maker=in_process_maker,
               shower=lambda p: None)


def test_the_analysis_checks_the_runlog_headers(captured, tmp_path):
    heads = vr.runlog_headers(captured["out"])
    assert [h["block"] for h in heads] == list(range(7)) and len({h["runs_sha256"] for h in heads}) == 1
    d = tmp_path / "c"
    d.mkdir()
    for b in range(7):
        (d / f"runlog.b{b}.ndjson").write_text((captured["out"] / f"runlog.b{b}.ndjson").read_text())
    rows = (d / "runlog.b4.ndjson").read_text().splitlines()
    h = json.loads(rows[0])
    for change, msg in (({"key": [20261031, 7, 5]}, "run key"), ({"block": 3}, "block 4"),
                        ({"runs_sha256": "0" * 64}, "different runs.json")):
        (d / "runlog.b4.ndjson").write_text("\n".join([json.dumps({**h, **change})] + rows[1:]) + "\n")
        with pytest.raises(SystemExit, match=msg):
            vr.read_runlogs(d)


def plan_copy(plan, tmp_path: Path) -> tuple[Path, vr.Persons, str]:
    """A copy of the plan with the manifest main_make writes (persons_record, conditions' SHA-256)."""
    import shutil
    d = tmp_path / "plan"
    shutil.copytree(plan["out"], d)
    ps = vr.Persons(pool=["a", "b", "c"], h=["a", "b"], e=["c"], donors=np.array([]), e_train=np.array([]), looked=3)
    cond = vr.conditions_sha256(b"conditions")
    (d / "manifest.json").write_text(json.dumps({**plan["man"], "persons": vp.persons_record(ps),
                                                 "conditions_sha256": cond}))
    return d, ps, cond


def logs_copy(captured, tmp_path: Path) -> Path:
    d = tmp_path / "caps"
    d.mkdir(parents=True)
    for f in captured["out"].glob("runlog*.ndjson"):
        (d / f.name).write_text(f.read_text())
    return d


def test_the_analysis_reads_the_plans_manifest(plan, captured, tmp_path):
    d, ps, cond = plan_copy(plan, tmp_path)
    got = vr.read_plan(d, captured["out"], ps, cond)
    assert got["replaced_keys_H"] == plan["man"]["replaced_keys_H"]
    assert got["runs_sha256"] == vr.runlog_headers(captured["out"])[0]["runs_sha256"] == runs_file_sha(plan)
    with pytest.raises(SystemExit, match="other conditions"):
        vr.read_plan(d, captured["out"], ps, "d" * 64)
    with pytest.raises(SystemExit, match="replay persons"):
        vr.read_plan(d, captured["out"], vr.Persons([], ["z"], [], np.array([]), np.array([]), 0), cond)
    with pytest.raises(SystemExit, match="pool"):
        vr.read_plan(d, captured["out"], vr.Persons(["a", "b"], ["a", "b"], [], np.array([]), np.array([]), 0), cond)
    man = json.loads((d / "manifest.json").read_text())
    (d / "manifest.json").write_text(json.dumps({**man, "n_items": man["n_items"] - 1}))
    with pytest.raises(SystemExit, match="does not count"):
        vr.read_plan(d, captured["out"], ps, cond)
    files = {k: v for k, v in man["files_sha256"].items() if k != "runs.json"}
    (d / "manifest.json").write_text(json.dumps({**man, "files_sha256": files}))
    with pytest.raises(SystemExit, match="names no runs.json"):
        vr.read_plan(d, captured["out"], ps, cond)


def test_a_changed_plan_file_stops_the_analysis(plan, captured, tmp_path):
    d, ps, cond = plan_copy(plan, tmp_path)
    (d / "prompts" / "P3.txt").write_text("Another task.\n")
    with pytest.raises(SystemExit, match="P3.txt"):
        vr.read_plan(d, captured["out"], ps, cond)
    d2, _, _ = plan_copy(plan, tmp_path / "2")
    (d2 / "desktop.ini").write_text("x")
    with pytest.raises(SystemExit, match="extra \\['desktop.ini'\\]"):
        vr.read_plan(d2, captured["out"], ps, cond)


def test_runlogs_of_another_runs_order_or_block_stop_the_analysis(plan, captured, tmp_path):
    d, ps, cond = plan_copy(plan, tmp_path)
    for case in ("sha", "order", "swap", "header_order"):
        caps = logs_copy(captured, tmp_path / case)
        f = caps / "runlog.b3.ndjson"
        rows = f.read_text().splitlines()
        h = json.loads(rows[0])
        if case == "sha":
            for g in caps.glob("runlog*.ndjson"):
                r = g.read_text().splitlines()
                g.write_text("\n".join([json.dumps({**json.loads(r[0]), "runs_sha256": "0" * 64})] + r[1:]) + "\n")
            msg = "another runs.json"
        elif case == "order":
            rows[1], rows[2] = rows[2], rows[1]
            f.write_text("\n".join(rows) + "\n")
            msg = "order"
        elif case == "header_order":   # the lines as drawn, the header naming another order
            f.write_text("\n".join([json.dumps({**h, "order": h["order"][::-1]})] + rows[1:]) + "\n")
            msg = "items or their order"
        else:   # an item moved from block 3 to block 4
            g = caps / "runlog.b4.ndjson"
            r4 = g.read_text().splitlines()
            g.write_text("\n".join(r4[:-1] + [rows[1], r4[-1]]) + "\n")
            f.write_text("\n".join(rows[:1] + rows[2:]) + "\n")
            msg = "items or their order"
        with pytest.raises(SystemExit, match=msg):
            vr.read_plan(d, caps, ps, cond)
    assert h["block"] == 3


def test_a_single_runlog_without_the_runs_sha_stops(captured, tmp_path):
    d = tmp_path / "one"
    d.mkdir()
    rows = (captured["out"] / "runlog.b2.ndjson").read_text().splitlines()
    h = json.loads(rows[0])
    for change in ({"block": None, "runs_sha256": None}, {"block": 2}):
        (d / "runlog.ndjson").write_text("\n".join([json.dumps({**h, **change})] + rows[1:]) + "\n")
        with pytest.raises(SystemExit, match="does not name"):
            vr.runlog_headers(d)


def test_a_done_m1_without_its_schedule_stops_the_runlist(tmp_path):
    def runner(args, folder):
        st = fake_runner(args, folder)
        if "--llm-stream" in args:
            (folder / "schedule.ndjson").unlink()   # a fault: done, and no schedule
        return st
    items = [{"name": "M1_8B/t01", "args": ["--llm-stream"], "block": 1},
             {"name": "M1_replay_8B/t01", "args": ["--events", "{out:M1_8B/t01}/schedule.ndjson"], "block": 1,
              "after": "M1_8B/t01", "needs": ["{out:M1_8B/t01}/schedule.ndjson"],
              "make": ["works/prototype_v7_plan.py", "gate-m1", "--m1", "{out:M1_8B/t01}"]}]
    with pytest.raises(RuntimeError, match="make exited"):
        rl.run(items, [1], tmp_path / "x", runner=runner, maker=in_process_maker, shower=lambda p: None)


def test_a_failed_m1_whose_schedule_is_not_its_streams_plan_is_a_fault(tmp_path):
    d = m1_folder(tmp_path, "interrupted", 25, seed=6)
    _, presses = ev.read(d / "schedule.ndjson")
    (d / "schedule.ndjson").unlink()
    ev.write(d / "schedule.ndjson", ev.make([p["t_ms"] * 1.01 for p in presses], [p["key"] for p in presses]),
             "M1", "t")
    with pytest.raises(SystemExit) as e:
        vp.m1_plan(d)
    assert e.value.code not in (0, 3)


def test_a_done_m1_of_one_press_is_its_plan_but_no_text_for_m0(plan, tmp_path):
    d = m1_folder(tmp_path, "done", 1, seed=8)
    (d / "schedule.ndjson").unlink()
    (d / "llm.ndjson").unlink()
    write_ndjson(d / "llm.ndjson", [{"type": "chunk", "n": 0, "recv_ns": 10 ** 6, "token": "a", "n_chars": 1,
                                     "done": False},
                                    {"type": "chunk", "n": 1, "recv_ns": 3 * 10 ** 6, "token": "", "n_chars": 0,
                                     "done": True}])
    ev.write(d / "schedule.ndjson", ev.make([0.0], ["a"]), "M1", "t")
    assert vp.m1_plan(d) == ["a"]   # done with one press: its plan (load_capture keeps it)
    with pytest.raises(SystemExit) as e:
        vp.derive("M0", 0, d, plan["out"])
    assert e.value.code == 3


def test_every_llm_item_has_seed_1_and_its_classs_model(plan):
    m8, m14 = (C7["environment"]["models"][k]["name"] for k in ("8B_class", "14B_class"))
    n = 0
    for it in plan["items"]:
        a = it["args"]
        if "--llm-stream" in a or "--llm-plan" in a:
            n += 1
            assert arg(a, "--seed") == "1"
            assert arg(a, "--model") == (m8 if "_8B/" in it["name"] else m14)
        if "--load" in a:
            assert arg(a, "--load") == m14
    assert n == 160


def test_pool_sha_is_read_or_stops():
    assert vp.pool_v6_sha(C7) == "a062fddbc4192956f52c6422a81301af4145834bb1c0a37a5aab4dee73f41042"
    with pytest.raises(SystemExit, match="ids_sha256"):
        vp.pool_v6_sha({"persons": {"pool": "v6's 200"}})


def test_l0s_tie_is_dropped_on_both_sides(captured):
    caps = vr.load_all(captured["out"])
    refs = vr.l0_refs(caps)
    l0 = caps["L0"][0]
    assert refs[0][0].size == l0.layers["hook"].times.size - 1
    assert caps["L1_idle"][0].plan.times.size == refs[0][0].size


def test_a_later_block_may_use_an_earlier_blocks_failed_plan(tmp_path):
    items = [{"name": "M1_8B/t03", "args": ["--llm-stream"], "block": 0},
             {"name": "M1_replay_8B/t03", "args": ["--events", "{out:M1_8B/t03}/schedule.ndjson"], "block": 1,
              "after": "M1_8B/t03", "needs": ["{out:M1_8B/t03}/schedule.ndjson"]}]
    out = tmp_path / "x"
    for b in (0, 1):
        res = rl.run(items, [1], out, runner=fake_runner, block=b, maker=lambda a: 0, shower=lambda p: None)
    assert res[0]["result"] == "done"
    assert json.loads((out / "runlog.b0.ndjson").read_text().splitlines()[1])["result"] == "failed"


def test_main_names_the_runs_files_sha256(tmp_path, monkeypatch):
    f = tmp_path / "runs.json"
    f.write_text(json.dumps([{"name": "P_idle/c005", "args": [], "block": 1}], indent=1))
    seen = {}
    monkeypatch.setattr(rl, "run", lambda items, key, out, **kw: seen.update(kw) or [])
    monkeypatch.setattr(sys, "argv", ["runlist", "--list", str(f), "--key", "1", "--out", str(tmp_path / "o"),
                                      "--block", "1"])
    rl.main()
    assert seen == {"block": 1, "runs_sha": vp.sha256(f)}


C6_COPY = json.loads((ROOT / "works/conditions/prototype_v6.json").read_text())["data"]["kupa"]["files"]["copy"]


def fake_kupa(plan, tmp_path: Path, monkeypatch) -> vr.Persons:
    """main_make's KUPA reading, its checks and the git state replaced by synthetic ones (no KUPA on this machine);
    the SHA-256 checks see the conditions' values."""
    ps = vr.Persons(pool=[f"p{i}" for i in range(vr.N_H + N_E)], h=[f"p{i}" for i in range(vr.N_H)],
                    e=[f"p{vr.N_H + i}" for i in range(N_E)], donors=np.array([]), e_train=np.array([]), looked=999)
    essay = {pid: h for pid, h in zip(ps.h, plan["h0"])}
    essay.update({pid: {"times": np.concatenate([[0.0], np.cumsum(g)])} for pid, g in zip(ps.e, plan["e_gaps"])})
    monkeypatch.setattr(vp, "git_clean_commit", lambda: "f" * 40)
    monkeypatch.setattr(vp, "ROOT", tmp_path)
    for f in ("external/kupa-keys/KUPA-KEYS-TASK-2.csv", "external/kupa-keys/KUPA-KEYS-META.csv", C6_COPY["file"]):
        (tmp_path / f).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / f).write_text("")
    kp = C7["data"]["kupa"]
    want = {"KUPA-KEYS-TASK-2.csv": vr.kp_sha(kp, "essay"), "KUPA-KEYS-META.csv": vr.kp_sha(kp, "meta"),
            Path(C6_COPY["file"]).name: C6_COPY["sha256"]}
    monkeypatch.setattr(vr.cm, "sha256", lambda path: want[Path(path).name])
    monkeypatch.setattr(vr.cm, "load_conditions", lambda path: (b"conditions", C7) if path == vr.COND
                        else (b"v6", json.loads(Path(path).read_text())))
    monkeypatch.setattr(v6, "read_kupa", lambda path: (essay, set(essay)))
    monkeypatch.setattr(vr, "draw_persons", lambda e, c, seen, sha: ps)
    monkeypatch.setattr(vp, "read_kupa_keys", lambda path, ids: {pid: hk for pid, hk in zip(ps.h, plan["hs"])})
    monkeypatch.setattr(vp, "kupa_prompts", lambda path: prompts()[:10])
    return ps


def test_main_make_writes_the_manifest_the_analysis_reads(plan, captured, tmp_path, monkeypatch):
    import hashlib
    ps = fake_kupa(plan, tmp_path, monkeypatch)
    out = tmp_path / "made"
    vp.main_make(out)
    got = vr.read_plan(out, logs_with_runs_sha(captured, tmp_path, vp.sha256(out / "runs.json")), ps,
                       vr.conditions_sha256(b"conditions"))
    assert got["replaced_keys_H"] == plan["man"]["replaced_keys_H"]
    man = json.loads((out / "manifest.json").read_text())
    assert man["git_commit"] == "f" * 40 and man["persons"]["ids_looked_at"] == 999
    assert not any(f'"{pid}"' in (out / "manifest.json").read_text() for pid in ps.pool)


def logs_with_runs_sha(captured, tmp_path: Path, sha: str) -> Path:
    d = logs_copy(captured, tmp_path / "logs-sha")
    for g in d.glob("runlog*.ndjson"):
        r = g.read_text().splitlines()
        g.write_text("\n".join([json.dumps({**json.loads(r[0]), "runs_sha256": sha})] + r[1:]) + "\n")
    return d


def test_main_make_leaves_no_folder_when_it_fails(plan, tmp_path, monkeypatch):
    fake_kupa(plan, tmp_path, monkeypatch)
    monkeypatch.setattr(vp, "persons_record", lambda ps: (_ for _ in ()).throw(RuntimeError("disk full")))
    out = tmp_path / "made"
    with pytest.raises(RuntimeError, match="disk full"):
        vp.main_make(out)
    assert not out.exists()


def test_main_make_stops_on_a_kupa_file_of_another_sha(plan, tmp_path, monkeypatch):
    fake_kupa(plan, tmp_path, monkeypatch)
    monkeypatch.setattr(vr.cm, "sha256", lambda path: "0" * 64)
    with pytest.raises(SystemExit, match="does not match"):
        vp.main_make(tmp_path / "made")
    assert not (tmp_path / "made").exists()


def test_one_runlog_of_all_blocks_passes_the_analysis(plan, tmp_path):
    caps = tmp_path / "one"
    rl.run(plan["items"], [20261031, 7, 6], caps, runner=fake_runner, maker=in_process_maker, shower=lambda p: None,
           runs_sha=runs_file_sha(plan))
    d, ps, cond = plan_copy(plan, tmp_path)
    assert vr.read_plan(d, caps, ps, cond)["runs_sha256"] == runs_file_sha(plan)
    f = caps / "runlog.ndjson"
    rows = f.read_text().splitlines()
    rows[1], rows[2] = rows[2], rows[1]
    f.write_text("\n".join(rows) + "\n")
    with pytest.raises(SystemExit, match="items or their order"):
        vr.read_plan(d, caps, ps, cond)


def test_a_plan_without_runs_json_cannot_be_run(plan, tmp_path):
    out = tmp_path / "p"
    man, runs = vp.build(C7, plan["hs"], dict(enumerate(plan["e_gaps"])), N_E, prompts(), out)
    assert not (out / "runs.json").exists() and not (out / "manifest.json").exists()
    vp.publish(out, man, runs)
    assert vp.sha256(out / "runs.json") == man["files_sha256"]["runs.json"]
