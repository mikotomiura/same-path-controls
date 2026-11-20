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
