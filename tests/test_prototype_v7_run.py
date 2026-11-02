"""Tests of works/prototype_v7_run.py: the run end to end on synthetic persons and a synthetic capture folder (no
KUPA, no capture of the work PC). Sizes are reduced (persons, texts, bootstrap resamples) so that the test runs in
minutes; the conditions' sizes are checked apart."""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "works"))
import prototype_v7 as v7  # noqa: E402
import prototype_v7_run as vr  # noqa: E402

C7 = json.loads((ROOT / "works/conditions/prototype_v7.json").read_text())
C4 = json.loads((ROOT / "works/conditions/prototype_v4.json").read_text())
SIZES = {k: getattr(vr, k) for k in ("N_H", "N_REP", "N_TEXTS", "N_POOL", "MIN_PRESSES", "N_ESSAYS")}   # before any patch


def human_like(n: int, rng: np.random.Generator) -> np.ndarray:
    mu = math.log(rng.uniform(120, 300))
    e = np.empty(n)
    e[0] = rng.normal(0, 0.5)
    for i in range(1, n):
        e[i] = 0.8 * e[i - 1] + rng.normal(0, 0.3)
    g = np.exp(mu + e)
    rest = rng.random(n) < 0.05
    g[rest] = rng.uniform(800, 3000, rest.sum())
    return g


def text_keys(n: int, rng: np.random.Generator) -> list[str]:
    """Words of letters, spaces, a sentence end every few words, an Enter and a Backspace now and then."""
    out: list[str] = []
    while len(out) < n:
        for _ in range(int(rng.integers(4, 9))):
            out += list("abcdefgh"[: int(rng.integers(2, 8))]) + [" "]
        out += [".", " "]
        if rng.random() < 0.2:
            out.append("Enter")
        if rng.random() < 0.2:
            out.append("Backspace")
    return out[:n]


def write_ndjson(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def write_events(path: Path, times: np.ndarray, keys: list[str], series: str) -> None:
    write_ndjson(path, [{"kind": "header", "series": series, "source": "test", "created_utc": "-", "n": len(keys)}]
                 + [{"i": i, "t_ms": float(t), "key": k} for i, (t, k) in enumerate(zip(times, keys))])


def write_layers(d: Path, times: np.ndarray, kinds: list[str], rng: np.random.Generator, injected: bool = True,
                 hook: bool = True, page_grid: float = 0.1) -> None:
    """The recorded presses: the planned times plus a small offset and jitter, a Shift down on the hook layer before
    every 50th press (not counted), the page on a 0.1 ms grid."""
    th = times + 3.0 + rng.normal(0, 0.3, times.size)
    tp = np.round((times + 8.0 + rng.normal(0, 0.8, times.size)) / page_grid) * page_grid
    if hook:
        rows = []
        for i, (t, k) in enumerate(zip(th, kinds)):
            if i % 50 == 0:
                rows.append({"t_ns": int((t - 0.1) * 1e6), "down": True, "injected": injected, "kind": "modifier"})
            rows.append({"t_ns": int(t * 1e6), "down": True, "injected": injected, "kind": k})
            rows.append({"t_ns": int((t + 40) * 1e6), "down": False, "injected": injected, "kind": k})
        write_ndjson(d / "hook.ndjson", rows)
    write_ndjson(d / "page.ndjson", [{"type": "clock", "time_origin": 0}]
                 + [{"type": "keydown", "t_ms": float(t), "kind": k, "repeat": False, "trusted": True}
                    for t, k in zip(tp, kinds)])


@pytest.fixture(scope="module")
def small(tmp_path_factory):
    """Reduced sizes: 12 replay persons, 4 repeats, 8 texts, 70 persons in E; N_BOOT 60."""
    mp = pytest.MonkeyPatch()
    mp.setattr(vr, "N_H", 12)
    mp.setattr(vr, "N_REP", 4)
    mp.setattr(vr, "N_TEXTS", 8)
    mp.setattr(v7, "N_BOOT", 60)
    rng = np.random.default_rng(7)
    h0 = []
    for _ in range(vr.N_H):
        n = int(rng.integers(320, 420))
        keys = text_keys(n, rng)
        t = np.concatenate([[0.0], np.cumsum(human_like(n - 1, rng))])
        h0.append({"times": t, "kinds": [v7.LOGGER_KIND_TO_V6[vr.key_kind(k)] for k in keys], "keys": keys})
    e_gaps = [human_like(int(rng.integers(300, 420)), rng) for _ in range(70)]
    root = tmp_path_factory.mktemp("captures")
    runlog = []

    def item(series, unit, result="done"):
        runlog.append({"item": vr.item_name(series, unit), "result": result})
        return root / vr.item_name(series, unit)

    for s_ in ("H1_idle", "H1_load", "H1_rep"):
        for i in range(vr.N_REP if s_ == "H1_rep" else vr.N_H):
            if s_ == "H1_load" and i == 3:
                item(s_, i, "failed")
                continue
            d = item(s_, i, "redone" if (s_, i) == ("H1_idle", 5) else "done")
            x = h0[i]
            write_events(d / "events.ndjson", x["times"], x["keys"], s_)
            write_layers(d, x["times"], [vr.key_kind(k) for k in x["keys"]], rng)
    l0_hook = []
    for i in range(vr.N_ESSAYS):
        n = 400
        keys = text_keys(n, rng)
        t = np.concatenate([[0.0], np.cumsum(human_like(n - 1, rng))])
        d = item("L0", i)
        write_layers(d, t, [vr.key_kind(k) for k in keys], rng, injected=False)
        hk = [r for r in vr.read_ndjson(d / "hook.ndjson") if r["down"] and r["kind"] != "modifier"]
        th = np.array([r["t_ns"] / 1e6 for r in hk])
        l0_hook.append((th - th[0], keys))
    for s_ in ("L1_idle", "L1_load"):
        for i, (t, keys) in enumerate(l0_hook):
            d = item(s_, i)
            write_events(d / "events.ndjson", t, keys, s_)
            write_layers(d, t, [vr.key_kind(k) for k in keys], rng)
    m1_plans = {}
    for model in ("8B", "14B"):
        for t_ in range(vr.N_TEXTS):
            s_ = f"M1_{model}"
            if (model, t_) == ("14B", 6):
                item(s_, t_, "no_plan")
                item(f"M1_replay_{model}", t_, "no_plan")
                continue
            failed_up = (model, t_) == ("8B", 1)   # typing failed twice; the plan files are kept
            d = item(s_, t_, "failed" if failed_up else "done")
            ntok = int(rng.integers(150, 220))
            gaps = rng.exponential(25.0, ntok)
            arr = 500.0 + np.cumsum(gaps)
            toks = ["".join(rng.choice(list("abcdefg"), int(rng.integers(1, 5)))) + (" " if rng.random() < 0.7 else "")
                    for _ in range(ntok)]
            toks[5] = ""   # an empty token, skipped and counted
            end = arr[-1] + 30.0
            write_ndjson(d / "llm.ndjson", [{"recv_ns": int(a * 1e6), "token": tk, "done": False}
                                            for a, tk in zip(arr, toks)] + [{"recv_ns": int(end * 1e6), "token": "",
                                                                             "done": True}])
            keep = [i for i, tk in enumerate(toks) if tk]
            arr_r = np.array([int(arr[i] * 1e6) / 1e6 for i in keep])
            wished, det = v7.m1_wished_times(arr_r, np.array([len(toks[i]) for i in keep]), int(end * 1e6) / 1e6)
            plan = v7.executable_plan(wished, det)
            plan = plan - plan[0]
            keys = list("".join(toks[i] for i in keep))
            m1_plans[(model, t_)] = (plan, keys)
            write_events(d / "events.ndjson", plan, keys, s_)
            write_events(d / "schedule.ndjson", plan, keys, s_)
            if failed_up:
                d = item(f"M1_replay_{model}", t_)
                write_events(d / "events.ndjson", plan, keys, f"M1_replay_{model}")
                write_layers(d, plan, [vr.key_kind(k) for k in keys], rng)
                continue
            write_layers(d, plan, [vr.key_kind(k) for k in keys], rng)
            write_ndjson(d / "inject.ndjson", [{"i": i, "planned_ns": int(p * 1e6), "sent_ns": int((p + 0.2) * 1e6),
                                                "returned_ns": int((p + 0.3) * 1e6), "ret": 2, "queue": 0}
                                               for i, p in enumerate(plan)])
            d = item(f"M1_replay_{model}", t_)
            write_events(d / "events.ndjson", plan, keys, f"M1_replay_{model}")
            write_layers(d, plan, [vr.key_kind(k) for k in keys], rng)
    for t_ in range(vr.N_TEXTS):
        plan8, keys = m1_plans[("8B", t_)]
        t0 = np.arange(len(keys)) * vr.M0_GAP_MS
        d = item("M0", t_)
        write_events(d / "events.ndjson", t0, keys, "M0")
        write_layers(d, t0, [vr.key_kind(k) for k in keys], rng)
        drawn = v7.m3(t_, len(keys) - 1, e_gaps)
        for s_ in ("M3a", "M3b"):
            tt = np.concatenate([[0.0], np.cumsum(drawn[s_])])
            d = item(s_, t_, "failed" if (s_, t_) == ("M3b", 2) else "done")
            if (s_, t_) == ("M3b", 2):
                continue
            write_events(d / "events.ndjson", tt, keys, s_)
            write_layers(d, tt, [vr.key_kind(k) for k in keys], rng)
        for model in ("8B", "14B"):
            s_ = f"M2_{model}"
            d = item(s_, t_)
            g = np.concatenate([[0.0], rng.uniform(40, 260, len(keys) - 1)])
            tt = np.cumsum(g)
            write_events(d / "events.ndjson", tt, keys, s_)
            write_layers(d, tt, [vr.key_kind(k) for k in keys], rng)
    for s_ in ("P_idle", "P_load", "P_nohook"):
        for c in (vr.CONTROLS_MS if s_ != "P_nohook" else v7.P_NOHOOK_MS):
            d = item(s_, int(c))
            keys = [chr(ord("a") + i % 10) for i in range(vr.P_PRESSES)]
            tt = np.arange(vr.P_PRESSES) * c
            write_events(d / "events.ndjson", tt, keys, s_)
            write_layers(d, tt, ["alnum"] * vr.P_PRESSES, rng, hook=s_ != "P_nohook",
                         page_grid=10.0 if c == 5.0 else 0.1)   # 5 ms: presses batched in pairs
    write_ndjson(root / "runlog.ndjson", runlog)
    donors, train = v7.e_split(len(e_gaps))
    yield {"root": root, "h0": h0, "e_gaps": e_gaps, "train": train, "runlog": runlog}
    mp.undo()


@pytest.fixture(scope="module")
def record(small):
    caps = vr.load_all(small["root"])
    rec = vr.run(C7, small["h0"], small["e_gaps"], small["train"], caps, C4["mapping"], C4["relief"])
    return vr.to_json(rec)


def numeric_lists(x, path=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from numeric_lists(v, f"{path}/{k}")
    elif isinstance(x, list):
        if x and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in x):
            yield path, len(x)
        for i, v in enumerate(x):
            yield from numeric_lists(v, f"{path}[{i}]")


def test_run_end_to_end_writes_json_without_nan(record):
    text = json.dumps(record, allow_nan=False)
    assert len(text) > 1000
    for key in ("validity_gate", "headline", "machine_path", "paired_rows", "replay_error", "replay_repeat",
                "controls", "c1", "sensitivity_quantisation", "g1", "failures", "design_table", "predictions"):
        assert key in record


def test_record_holds_no_series(record):
    """record: counts and summaries only. No list of numbers longer than a summary or an interval (a series of
    press times, gaps or per-person values would be one)."""
    long = [(p, n) for p, n in numeric_lists(record) if n > 2]
    assert long == [], long[:5]


def test_predictions_are_all_evaluated(record):
    want = {"P1", "P2", "P3", "P4", "P5", "R1", "V1", "Hd1", "Hd2", "Hd3", "Hm1", "Hm2", "Dm1", "Dm2", "C1a", "C1b",
            "C1c", "G1"}
    assert set(record["predictions"]) == want
    assert all(v in (True, False, None) for v in record["predictions"].values())
    assert record["predictions"]["P3"] is True   # only the 5 ms control is batched on the page


def test_failures_count_on_the_planned_sessions(record):
    f = record["failures"]
    assert f["H1_load"] == {"planned": 12, "no_plan": 0, "failed_capture": 1, "redone": 0}
    assert f["H1_idle"]["redone"] == 1
    assert f["M1_14B"]["no_plan"] == 1 and f["M1_replay_14B"]["no_plan"] == 1
    assert f["M3b"]["failed_capture"] == 1


def test_missing_pairs_and_lost_clusters_kept_apart(record):
    """Amendment B14: a session without a plan lowers the clusters; a failed capture is a missing pair."""
    mp = record["machine_path"]["main"]
    x = mp["D1/hook/M1_14B"]
    assert x["without_plan"] == 1 and x["missing_failed_capture"] == 0 and x["n_clusters"] <= 7
    y = mp["D1/hook/M3b"]
    assert y["without_plan"] == 0 and y["missing_failed_capture"] == 1 and y["missing_pairs"] >= 1
    h = record["paired_rows"]["main"]["H0-H1_load/hook"]["D1"]
    assert h["missing_failed_capture"] == 1 and h["n"] + h["missing_pairs"] == 12
    g = record["headline"]["main"]["machine_general"]
    assert "M3b" in g["kept_out"]


def test_headline_cells_on_every_detector_and_layer(record):
    cells = record["headline"]["main"]["cells"]
    assert set(cells) == {f"{d}/{la}" for d in vr.DETECTORS for la in vr.LAYERS}
    for c in cells.values():
        assert c["grid_note"] in ("note", "no note", "unknown")
        assert c["enters_general"] is False   # 12 persons: fewer than 35 complete pairs


def test_plans_recomputed_for_m1(record):
    m1 = record["plans"]["M1"]
    assert m1["8B"]["sessions_with_plan"] == 8 and m1["14B"]["sessions_with_plan"] == 7   # 8B t01 failed, plan kept
    assert m1["8B"]["empty_tokens_skipped"] == 8
    assert m1["8B"]["wait_ms"]["median"] == pytest.approx(0.2, abs=1e-3)


def test_modifier_presses_not_counted(small):
    caps = vr.load_all(small["root"])
    la = caps["H1_idle"][0].layers["hook"]
    assert la.not_counted > 0 and la.times.size == small["h0"][0]["times"].size


def _copy_tree(src: Path, dst: Path) -> Path:
    import shutil
    shutil.copytree(src, dst)
    return dst


def test_a_plan_that_is_not_h0_stops_the_run(small, tmp_path):
    root = _copy_tree(small["root"], tmp_path / "c")
    p = root / vr.item_name("H1_idle", 2) / "events.ndjson"
    rows = vr.read_ndjson(p)
    rows[10]["t_ms"] += 1.0
    rows[11]["t_ms"] += 1.0
    write_ndjson(p, rows)
    caps = vr.load_all(root)
    with pytest.raises(vr.RunStop, match="not H0's presses"):
        vr.check_plans(caps, small["h0"], small["e_gaps"], [None] * vr.N_ESSAYS)


def test_m3_plan_must_be_m3s_draw(small, tmp_path):
    root = _copy_tree(small["root"], tmp_path / "c")
    p = root / vr.item_name("M3a", 1) / "events.ndjson"
    rows = vr.read_ndjson(p)
    for r in rows[5:]:
        r["t_ms"] += 0.5
    write_ndjson(p, rows)
    caps = vr.load_all(root)
    l0 = [(lambda t: (t[0], [cp.layers["hook"].kinds[i] for i in t[1]]))(vr.v7.clean_times(cp.layers["hook"].times))
          for cp in caps["L0"]]
    with pytest.raises(vr.RunStop, match="m3"):
        vr.check_plans(caps, small["h0"], small["e_gaps"], l0)


def test_m1_schedule_must_be_the_executable_plan(small, tmp_path):
    root = _copy_tree(small["root"], tmp_path / "c")
    for f in ("schedule.ndjson", "events.ndjson"):
        p = root / vr.item_name("M1_8B", 0) / f
        rows = vr.read_ndjson(p)
        rows[3]["t_ms"] = rows[2]["t_ms"]   # still non-decreasing, but not B13's plan
        write_ndjson(p, rows)
    caps = vr.load_all(root)
    l0 = [(lambda t: (t[0], [cp.layers["hook"].kinds[i] for i in t[1]]))(vr.v7.clean_times(cp.layers["hook"].times))
          for cp in caps["L0"]]
    with pytest.raises(vr.RunStop, match="executable plan"):
        vr.check_plans(caps, small["h0"], small["e_gaps"], l0)


def test_an_item_missing_from_the_runlog_stops(small, tmp_path):
    root = _copy_tree(small["root"], tmp_path / "c")
    rows = [r for r in small["runlog"] if r["item"] != vr.item_name("M0", 3)]
    write_ndjson(root / "runlog.ndjson", rows)
    with pytest.raises(vr.RunStop, match="not in the runlog"):
        vr.load_all(root)


def test_a_stray_runlog_item_stops(small, tmp_path):
    root = _copy_tree(small["root"], tmp_path / "c")
    write_ndjson(root / "runlog.ndjson", small["runlog"] + [{"item": "M0/t99", "result": "done"}])
    with pytest.raises(vr.RunStop, match="outside the conditions"):
        vr.load_all(root)


def test_a_done_item_without_folder_stops(small, tmp_path):
    root = _copy_tree(small["root"], tmp_path / "c")
    import shutil
    shutil.rmtree(root / vr.item_name("M2_8B", 4))
    with pytest.raises(vr.RunStop, match="folder is missing"):
        vr.load_all(root)


def test_key_kind_follows_v6():
    assert [vr.key_kind(k) for k in ("a", "Z", "7", " ", ".", "!", "?", ",", "Enter", "Backspace", "Delete")] == [
        "alnum", "alnum", "alnum", "space", "end", "end", "end", "other", "enter", "backspace", "delete"]
    with pytest.raises(vr.RunStop):
        vr.key_kind("Tab")


def test_judgments_follow_the_thresholds():
    m, ok = vr.judgments("D1", np.array([0.269, 0.2691, np.nan]))
    assert m.tolist() == [True, False, False] and ok.tolist() == [True, True, False]
    m, ok = vr.judgments("D3", np.array([0.5, 0.5001]))
    assert m.tolist() == [False, True]


def test_conditions_sizes():
    """The constants that the reduced test patches are the conditions' sizes."""
    s = C7["series"]
    assert "80 x 2 (+ H1_rep 10)" in s["counts_of_captures"] and (SIZES["N_H"], SIZES["N_REP"]) == (80, 10)
    assert "40 prompts (P1-P40)" in s["M1_8B and M1_14B"] and SIZES["N_TEXTS"] == 40
    assert "first 300 eligible" in C7["persons"]["pool"] and SIZES["N_POOL"] == 300
    assert "at least 300 counted presses" in C7["persons"]["replay_H"] and SIZES["MIN_PRESSES"] == 300
    assert "constant gap 55 ms" in s["M0"] and vr.M0_GAP_MS == 55.0
    assert "5, 18, 50, 100, 200 ms, 1000 presses" in s["P"] and vr.CONTROLS_MS == (5.0, 18.0, 50.0, 100.0, 200.0)
    assert "4 essays" in s["L0"] and SIZES["N_ESSAYS"] == 4


def test_new_keys_give_distinct_streams():
    """The keys of the run (C1's pair index 0..7; the count's series 0..8; the jitter of H0 for 80 sessions) give
    no two equal streams, among themselves and with the other keys of the conditions."""
    def first(key):
        return tuple(np.random.default_rng(key).integers(0, 2 ** 62, 4))
    keys = [[vr.S, 7, 4, la, p, st] for la in range(2) for p in range(len(vr.C1_PAIRS)) for st in range(3)]
    keys += [[vr.S, 7, 5, la, d, se] for la in range(2) for d in range(5) for se in range(len(vr.MACHINE))]
    keys += [[vr.S, 7, 10, la, se, ss, sd] for la in range(2) for se in range(10) for ss in range(80)
             for sd in range(2)]
    keys += [[vr.S, 7, 9, la, d, p] for la in range(2) for d in range(5) for p in range(7)]
    keys += [[vr.S, 7, 11, la, d, se] for la in range(2) for d in range(5) for se in range(9)]
    keys += [[vr.S, 7, 12, la, d] for la in range(2) for d in range(5)]
    keys += [[vr.S, 7, 3, t] for t in range(40)] + [[vr.S, 7, 13], [vr.S, 7, 8], [vr.S, 7, 6], [vr.S, 7, 2]]
    seen = {}
    for k in keys:
        f = first(k)
        assert f not in seen, (seen.get(f), k)
        seen[f] = k


def test_p3_needs_5_ms_and_only_5_ms():
    base = {(s_, c): False for s_ in ("P_idle", "P_load") for c in (5, 18, 50, 100, 200)}
    assert vr.p3(base | {("P_idle", 5): True}) is True
    assert vr.p3(base | {("P_load", 5): True}) is True
    assert vr.p3(base) is False                                              # 5 ms not batched
    assert vr.p3(base | {("P_idle", 5): True, ("P_load", 18): True}) is False   # another control too
    assert vr.p3(base | {("P_idle", 5): None}) is None


def test_a_failed_upstream_capture_keeps_its_plan(record):
    """Both reviews (2026-11-02): M1_8B t01's typing failed but its plan exists; M0, M3 and M1_replay of that text were
    captured. The run goes on; the failure is one missing pair of M1_8B."""
    x = record["machine_path"]["main"]["D1/hook/M1_8B"]
    assert x["missing_failed_capture"] == 1 and x["without_plan"] == 0
    assert record["machine_path"]["main"]["D1/hook/M0"]["missing_pairs"] == 0
    assert record["failures"]["M1_8B"]["failed_capture"] == 1


def test_machine_cells_carry_bounds_saturation_and_gap_sets(record):
    x = record["machine_path"]["main"]["D2_lr/page/M3b"]
    assert set(x["bounds"]) == {"complete", "missing_as_no_flip", "missing_as_b", "missing_as_c"}
    assert set(x["saturation"]) == {"plan", "recorded"} and 0 <= x["used_gaps_differ_share"] <= 1


def test_rows_never_read_carry_no_reading(record):
    for name in ("H1_idle-H1_rep/hook", "L0-L1_idle/page"):
        cell = record["paired_rows"]["main"][name]["D1"]
        assert "headline" not in cell and "bounds" not in cell and "enters_general" not in cell
        assert len(cell["tango_95"]) == 2 and cell["read"] is False


def test_persons_distributions_hold_no_end_points(record):
    for d, x in record["margin_on_H0"].items():
        assert "min" not in x and "max" not in x, d
    assert vr.summary([1.0, 2.0]) == {"n": 2}
    assert "max" in vr.summary([1.0, 2.0], ends=True)


def test_a_nan_in_the_record_stops():
    with pytest.raises(vr.RunStop):
        vr.to_json({"a": [1.0, float("nan")]})


def test_assumed_fields_must_have_their_types(tmp_path):
    p = tmp_path / "hook.ndjson"
    write_ndjson(p, [{"t_ns": 1, "down": "false", "injected": True, "kind": "alnum"}])
    with pytest.raises(vr.RunStop, match="down"):
        vr.read_hook(p, injected=True)
    write_ndjson(p, [{"t_ns": 1.5, "down": True, "injected": True, "kind": "alnum"}])
    with pytest.raises(vr.RunStop, match="t_ns"):
        vr.read_hook(p, injected=True)


def test_m1_schedule_keys_must_be_the_generated_text(small, tmp_path):
    root = _copy_tree(small["root"], tmp_path / "c")
    for f in ("schedule.ndjson", "events.ndjson"):
        p = root / vr.item_name("M1_8B", 2) / f
        rows = vr.read_ndjson(p)
        rows[4]["key"] = "z" if rows[4]["key"] != "z" else "y"
        write_ndjson(p, rows)
    caps = vr.load_all(root)
    l0 = [(lambda t: (t[0], [cp.layers["hook"].kinds[i] for i in t[1]]))(vr.v7.clean_times(cp.layers["hook"].times))
          for cp in caps["L0"]]
    with pytest.raises(vr.RunStop, match="generated text|not the text of M1_8B"):
        vr.check_plans(caps, small["h0"], small["e_gaps"], l0)
