"""Tests of works/prototype_v7t.py: the step, the page's two times read as v7 reads the presses, and the readings."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "works"))
import prototype_v7_run as run7  # noqa: E402
import prototype_v7t as vt  # noqa: E402


def _page(path: Path, t, h, extra=()):
    rows = [{"type": "open"}]
    rows += [{"type": "key", "t_ms": float(a), "h_ms": float(b), "kind": "alnum", "repeat": False, "trusted": True,
              "composing": False, "in_box": True} for a, b in zip(t, h)]
    rows += list(extra)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_constants_are_the_conditions():
    c = json.loads(vt.COND.read_text(encoding="utf-8"))["reading"]
    assert (c["step_ms"], c["tolerance_ms"], c["on_step_at_least"], c["off_step_at_most"]) == \
        (vt.STEP_MS, vt.STEP_TOL_MS, vt.ON_STEP_AT_LEAST, vt.OFF_STEP_AT_MOST)


def test_on_step_takes_integer_multiples_and_leaves_the_planned_gaps():
    on = np.array([15, 16, 31, 32, 46, 47, 62, 63, 15.625, 93.75])
    off = np.array([0, 1, 5, 8, 18, 50, 100, 200, 24])
    assert vt.on_step(on).all()
    assert not vt.on_step(off).any()


def test_page_times_are_read_pages_presses(tmp_path):
    p = tmp_path / "page.ndjson"
    rep = {"type": "key", "t_ms": 5.0, "h_ms": 5.5, "kind": "alnum", "repeat": True, "trusted": True,
           "composing": False, "in_box": True}
    other = {"type": "key", "t_ms": 7.0, "h_ms": 7.5, "kind": "nonchar", "repeat": False, "trusted": True,
             "composing": False, "in_box": True}
    _page(p, [4280, 4296, 4312], [4280.3, 4296.4, 4312.2], extra=[rep, other])
    t, h = vt.page_times(p)
    assert np.array_equal(t, [4280, 4296, 4312])   # the page's own times, not shifted to the first press
    assert np.array_equal(t - t[0], run7.read_page(p).times)
    assert np.array_equal(h, [4280.3, 4296.4, 4312.2])


def test_a_press_without_h_ms_stops(tmp_path):
    p = tmp_path / "page.ndjson"
    p.write_text(json.dumps({"type": "key", "t_ms": 1.0, "kind": "alnum", "repeat": False, "trusted": True,
                             "composing": False, "in_box": True}) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        vt.page_times(p)


def test_gap_measures_on_a_stepped_series():
    t = np.cumsum([0, 0, 16, 15, 16, 0, 31])   # a 1 ms plan seen on a 1/64 s step: zeros and 15/16
    m = vt.gap_measures(t, modes=True)
    assert m["n_gaps"] == 6
    assert m["share_zero"] == pytest.approx(2 / 6)
    assert m["share_on_step_of_positive"] == 1.0
    assert m["smallest_positive_gap_ms"] == 15.0
    assert m["modes_of_positive"][0] == [16.0, 0.5]
    assert "modes_of_positive" not in vt.gap_measures(t, modes=False)


def _known(ff_t, ff_h, ch_t):
    out = {}
    for ms, a, b, c in zip(vt.LONG_MS, ff_t, ff_h, ch_t):
        out[f"F_K/{ms}"] = {"t": {"share_on_step_of_positive": a}, "h": {"share_on_step_of_positive": b}}
        out[f"v7_P_idle/{ms}"] = {"t": {"share_on_step_of_positive": c}}
    return out


def test_readings():
    r = vt.readings(_known([1, 1, .95, .9], [.1, .2, 0, .5], [0, .1, .13, .2]))
    assert r == {"firefox_event_time": "on the step", "firefox_handler_time": "not on the step",
                 "chrome_event_time": "not on the step", "where_in_firefox": "the event's time only", "pc_wide": "no"}
    r = vt.readings(_known([1, 1, 1, 1], [1, 1, .95, .92], [1, 1, 1, 1]))
    assert r["where_in_firefox"] == "the delivery too" and r["pc_wide"] == "not decided"
    r = vt.readings(_known([1, 1, .89, 1], [0, 0, 0, 0], [0, 0, 0, 0]))
    assert r["firefox_event_time"] == "not decided" and r["where_in_firefox"] == "not decided"
    r = vt.readings(_known([1, 1, 1, None], [0, 0, 0, 0], [0, 0, 0, 0]))
    assert r["firefox_event_time"] == "not decided"
    r = vt.readings(_known([1, 1, 1, 1], [.6, 0, 0, 0], [0, 0, 0, 0]))
    assert r["firefox_handler_time"] == "not decided" and r["where_in_firefox"] == "not decided"


def test_person_summaries_give_no_single_value():
    rows = [{"t": {"share_zero": i / 10, "share_on_step_of_positive": 1.0, "share_on_1ms_grid": 1.0},
             "h": {"share_zero": 0.0, "share_on_step_of_positive": 0.0, "share_on_1ms_grid": 0.0},
             "_lag_median": 1.0 + i} for i in range(10)]
    s = vt.series_summary(rows, machine=False)
    assert "min" not in s["t/share_zero"] and "max" not in s["t/share_zero"] and "min" not in s["lag_median_ms"]
    assert s["t/share_of_captures_on_step"] == 1.0 and s["h/share_of_captures_on_step"] == 0.0
    few = vt.series_summary(rows[:4], machine=False)
    assert few["t/share_zero"] == {"n": 4} and few["t/share_of_captures_on_step"] is None
    assert "_lag_median" not in json.dumps(s)


def test_presses_other_than_read_pages_stop(tmp_path, monkeypatch):
    p = tmp_path / "page.ndjson"
    _page(p, [4280, 4296, 4312], [4280.3, 4296.4, 4312.2])
    real = run7.read_page

    def shifted(path):
        layer = real(path)
        layer.times = layer.times + 1.0
        return layer

    monkeypatch.setattr(run7, "read_page", shifted)
    with pytest.raises(SystemExit):
        vt.page_times(p)


def test_run_end_to_end_on_fake_captures(tmp_path, monkeypatch):
    """run() as main calls it, every item a captured page: Firefox's event time on the step, its handler time and
    Chrome's not."""
    import prototype_v7s as s7
    import prototype_v7s_plan as splan

    rng = np.random.default_rng(3)

    def cap(series, unit, ms, stepped):
        d = tmp_path / f"{series}-{unit}"
        d.mkdir()
        h = 1000.0 + np.arange(60) * ms + rng.uniform(0, 0.3, 60)
        t = np.floor(h / vt.STEP_MS) * vt.STEP_MS if stepped else h - 0.2
        _page(d / "page.ndjson", np.round(t) if stepped else t, h)
        return run7.Capture(series, unit, "captured", folder=d)

    items = {}
    for ms in splan.K_FIREFOX_MS:
        items[f"F_K/c{ms:03d}"] = (ms, True)
    for ms in splan.K_CHROME_MS:
        items[f"K_chrome/c{ms:03d}"] = (ms, False)
    for ms in splan.K_NOHOOK_MS:
        items[f"F_K_nohook/c{ms:03d}"] = (ms, True)
        items[f"K_chrome_nohook/c{ms:03d}"] = (ms, False)
    for s_ in (*s7.MC, *s7.F_MACHINE):
        for i in range(6):
            items[f"{s_}/t{i:02d}"] = (37.0, s_.startswith("F_"))
    for i in range(splan.N_F_H1):
        items[f"F_H1/h{i:02d}"] = (120.0, True)
    caps = {k: cap(*k.split("/"), ms, st) for k, (ms, st) in items.items()}
    p_idle = [cap("P_idle", int(ms), ms, False) for ms in run7.CONTROLS_MS]
    monkeypatch.setattr(s7, "read_runlogs", lambda root: {k: "done" for k in items})
    monkeypatch.setattr(s7, "load_item", lambda root, runlog, name, hook=True: caps[name])
    monkeypatch.setattr(run7, "load_all", lambda root: {"P_idle": p_idle})
    rec = vt.run(tmp_path, tmp_path)
    assert rec["readings"] == {"firefox_event_time": "on the step", "firefox_handler_time": "not on the step",
                               "chrome_event_time": "not on the step", "where_in_firefox": "the event's time only",
                               "pc_wide": "no"}
    assert set(rec["series"]) == {*s7.MC, *s7.F_MACHINE, "F_H1"}
    assert rec["series"]["F_M0"]["captures"] == 6 and rec["series"]["F_H1"]["captures"] == 10
    assert "min" not in rec["series"]["F_H1"]["t/share_zero"]
    assert rec["known_gaps"]["F_K/50"]["t"]["path"]["path_sd"] is not None
    json.dumps(run7.to_json(rec), allow_nan=False)
    assert "_lag_median" not in json.dumps(run7.to_json(rec))


# ---------------------------------------------------------------- v7s's and v7's own loaders on fake folders


def _folder(d: Path, h_ms, stepped: bool, browser: str, hook: bool, events_series: str):
    """One captured item as the capture writes it: events.ndjson, meta.json, page.ndjson and (unless nohook) hook."""
    from capture import events as ev

    d.mkdir(parents=True)
    n = len(h_ms)
    plan = np.round(h_ms - h_ms[0])   # the plan as whole ms: non-decreasing for every gap here (0 ms included)
    ev.write(d / "events.ndjson", ev.make(list(np.maximum.accumulate(plan)), ["a"] * n), events_series, "test")
    (d / "meta.json").write_text(json.dumps({"browser": {"name": browser, "version": "1"}}), encoding="utf-8")
    t = np.round(np.floor(h_ms / vt.STEP_MS) * vt.STEP_MS) if stepped else h_ms - 0.2
    _page(d / "page.ndjson", t, h_ms)
    if hook:
        rows = [{"type": "header"}]
        rows += [{"type": "key", "t_ns": int(x * 1e6), "down": True, "injected": True, "ours": True,
                  "repeat": False, "kind": "alnum"} for x in h_ms]
        rows += [{"type": "footer"}]
        (d / "hook.ndjson").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _fake_roots(tmp_path):
    import prototype_v7s as s7
    import prototype_v7s_plan as splan

    rng = np.random.default_rng(7)
    h = lambda ms: 1000.0 + np.arange(40) * ms + rng.uniform(0, 0.3, 40)  # noqa: E731
    v7s = tmp_path / "v7s"
    items = {}
    for ms in splan.K_FIREFOX_MS:
        items[f"F_K/c{ms:03d}"] = (ms, True, "firefox", True)
    for ms in splan.K_CHROME_MS:
        items[f"K_chrome/c{ms:03d}"] = (ms, False, "chrome", True)
    for ms in splan.K_NOHOOK_MS:
        items[f"F_K_nohook/c{ms:03d}"] = (ms, True, "firefox", False)   # no hook.ndjson: hook=False must be passed
        items[f"K_chrome_nohook/c{ms:03d}"] = (ms, False, "chrome", False)
    for s_ in (*s7.MC, *s7.F_MACHINE):
        for i in range(6):
            items[f"{s_}/t{i:02d}"] = (37.0, s_.startswith("F_"), "firefox" if s_.startswith("F_") else "chrome", True)
    for i in range(splan.N_F_H1):
        items[f"F_H1/h{i:02d}"] = (120.0, True, "firefox", True)
    for name, (ms, st, br, hk) in items.items():
        _folder(v7s / name, h(max(ms, 0.0)), st, br, hk, name.split("/")[0])
    results = {name: "done" for name in items}
    results["F_M2_14B/t06"] = "no_plan"   # a no_plan item as v7s's runlogs keep it
    names = list(results)
    for b in range(1, s7.N_BLOCKS + 1):
        rows = [{"type": "header", "key": list(splan.RUNLIST_KEY), "block": b, "runs_sha256": "r"}]
        rows += [{"type": "item", "item": nm, "result": results[nm]} for nm in names[b - 1::s7.N_BLOCKS]]
        rows += [{"type": "footer"}]
        (v7s / f"runlog.b{b}.ndjson").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    v7 = tmp_path / "v7"
    for ms in run7.CONTROLS_MS:
        _folder(v7 / run7.item_name("P_idle", int(ms)), h(ms), False, "chrome", True, "P_idle")
    return v7, v7s


def _real_p_idle(root):
    """v7's load_all reads every series of v7; only P_idle is faked, through v7's own load_capture."""
    runlog = {run7.item_name("P_idle", int(ms)): "done" for ms in run7.CONTROLS_MS}
    return {"P_idle": [run7.load_capture(root, runlog, "P_idle", int(ms)) for ms in run7.CONTROLS_MS]}


def test_run_with_v7s_and_v7_loaders(tmp_path, monkeypatch):
    v7, v7s = _fake_roots(tmp_path)
    monkeypatch.setattr(run7, "load_all", _real_p_idle)
    rec = vt.run(v7, v7s)
    assert rec["readings"]["where_in_firefox"] == "the event's time only" and rec["readings"]["pc_wide"] == "no"
    assert rec["series"]["F_M2_14B"]["captures"] == 6 and rec["series"]["F_M2_14B"]["not_captured"] == 1
    assert rec["known_gaps"]["F_K/50"]["hook"]["share_on_step_of_positive"] == 0.0
    assert "hook" not in rec["known_gaps"]["F_K_nohook/3"]
    assert rec["known_gaps"]["v7_P_idle/200"]["t"]["share_on_step_of_positive"] == 0.0


def test_main_writes_once(tmp_path, monkeypatch):
    import _common as cm

    v7, v7s = _fake_roots(tmp_path)
    monkeypatch.setattr(run7, "load_all", _real_p_idle)
    monkeypatch.setattr(cm, "same_as_head", lambda paths: None)   # the files under test are not committed yet
    out = tmp_path / "out" / "record.json"
    monkeypatch.setattr(vt, "OUT", out)
    vt.main(["--v7-captures", str(v7), "--v7s-captures", str(v7s)])
    rec = json.loads(out.read_text(encoding="utf-8"))
    assert rec["readings"]["firefox_event_time"] == "on the step"
    assert set(rec["provenance"]["code_sha256"]) >= {"works/prototype_v7t.py", "works/prototype_v7s.py"}
    with pytest.raises(SystemExit):
        vt.main(["--v7-captures", str(v7), "--v7s-captures", str(v7s)])


def test_base_check_stops_on_other_commits():
    c = json.loads(vt.COND.read_text(encoding="utf-8"))
    vt.base_check(c)
    c["base"]["v7s_commit"] = "0" * 40
    with pytest.raises(SystemExit):
        vt.base_check(c)


def test_on_step_edges():
    assert vt.on_step(np.array([14.625, 16.625, 30.25, 32.25])).all()
    assert not vt.on_step(np.array([14.6, 16.7, 30.2, 32.3])).any()


def test_share_of_captures_at_exactly_the_bound():
    rows = [{"t": {"share_zero": 0.0, "share_on_step_of_positive": v, "share_on_1ms_grid": 1.0},
             "h": {"share_zero": 0.0, "share_on_step_of_positive": None, "share_on_1ms_grid": 1.0},
             "lag_h_minus_t_ms": {"median": 0.1}} for v in (0.9, 0.9, 0.9, 0.9, 0.89)]
    s = vt.series_summary(rows, machine=True)
    assert s["t/share_of_captures_on_step"] == pytest.approx(0.8)
    assert s["h/share_of_captures_on_step"] is None   # no capture with a positive gap on that side


def test_a_press_with_a_nan_h_ms_stops(tmp_path):
    p = tmp_path / "page.ndjson"
    p.write_text(json.dumps({"type": "key", "t_ms": 1.0, "h_ms": float("nan"), "kind": "alnum", "repeat": False,
                             "trusted": True, "composing": False, "in_box": True}) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        vt.page_times(p)


def test_main_stops_when_the_base_is_not_the_codes(tmp_path, monkeypatch):
    import _common as cm

    c = json.loads(vt.COND.read_text(encoding="utf-8"))
    c["base"]["v7_commit"] = "0" * 40
    cond = tmp_path / "cond.json"
    cond.write_text(json.dumps(c), encoding="utf-8")
    monkeypatch.setattr(cm, "same_as_head", lambda paths: None)
    monkeypatch.setattr(vt, "COND", cond)
    monkeypatch.setattr(vt, "OUT", tmp_path / "out" / "record.json")
    monkeypatch.setattr(vt, "run", lambda *a: pytest.fail("run() reached with a wrong base"))
    with pytest.raises(SystemExit, match="base commits"):
        vt.main(["--v7-captures", str(tmp_path), "--v7s-captures", str(tmp_path)])
