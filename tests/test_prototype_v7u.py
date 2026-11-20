"""Tests of works/prototype_v7u.py: the page at the handler's time, the loader swap, and the readings."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "works"))
sys.path.insert(0, str(ROOT / "tests"))
import prototype_v7_run as run7  # noqa: E402
import prototype_v7s as s7  # noqa: E402
import prototype_v7u as vu  # noqa: E402
from test_prototype_v7t import _fake_roots, _page  # noqa: E402


def _cap(tmp_path, t, h, name="F_M2_8B"):
    d = tmp_path / name
    d.mkdir(parents=True)
    _page(d / "page.ndjson", t, h)
    cap = run7.Capture(name, 0, "captured", folder=d)
    cap.layers["page"] = run7.read_page(d / "page.ndjson")
    cap.layers["hook"] = run7.Layer(np.array([0.0, 1.0, 2.0]), ["alnum"] * 3, 0)
    return cap


def test_constants_are_the_conditions():
    c = json.loads(vu.COND.read_text(encoding="utf-8"))
    vu.base_check(c)
    c["base"]["v7t_commit"] = "0" * 40
    with pytest.raises(SystemExit):
        vu.base_check(c)


def test_page_at_handler_replaces_only_the_page_times(tmp_path):
    cap = _cap(tmp_path, [4280, 4280, 4296], [4290.0, 4291.0, 4302.0])
    out = vu.page_at_handler(cap)
    assert np.array_equal(out.layers["page"].times, [0.0, 1.0, 12.0])
    assert out.layers["page"].kinds == cap.layers["page"].kinds
    assert out.layers["hook"] is cap.layers["hook"]
    assert np.array_equal(cap.layers["page"].times, [0.0, 0.0, 16.0])   # the original capture is not changed
    failed = run7.Capture("F_M2_8B", 0, "failed")
    assert vu.page_at_handler(failed) is failed


def test_handler_loader_swaps_firefox_items_only(tmp_path):
    caps = {"F_M2_8B/t00": _cap(tmp_path, [4280, 4280, 4296], [4290.0, 4291.0, 4302.0]),
            "MC1/t00": _cap(tmp_path, [10.0, 11.0, 12.5], [10.1, 11.1, 12.6], name="MC1")}
    seen = []

    def load_item(root, runlog, name, hook=True):
        seen.append((name, hook))
        return caps[name]

    load = vu.handler_loader(load_item)
    assert np.array_equal(load(tmp_path, {}, "F_M2_8B/t00").layers["page"].times, [0.0, 1.0, 12.0])
    assert load(tmp_path, {}, "MC1/t00") is caps["MC1/t00"]
    load(tmp_path, {}, "F_M2_8B/t00", hook=False)
    assert seen[-1] == ("F_M2_8B/t00", False)


def test_variant_h_reads_firefox_pages_at_the_handlers_time_and_restores_the_loader(tmp_path, monkeypatch):
    """variant() through v7s's own read_runlogs and load_item on v7t's fake folders; v7s's later steps stubbed."""
    v7, v7s = _fake_roots(tmp_path)
    seen = {}

    def known_tables(v7caps, root, runlog):
        cap = s7.load_item(root, runlog, "F_K/c050")
        seen.setdefault("F_K", []).append(cap.layers["page"].times[:3].copy())
        cap = s7.load_item(root, runlog, "K_chrome/c005")
        seen.setdefault("K", []).append(cap.layers["page"].times[:3].copy())
        return {"firefox_version_known_gaps": "1"}, {"firefox/page": {0.0: 1.0}}

    def read_series(v7caps, root, runlog, h0, kinds):
        kinds["firefox_version"] = "1"
        return {}

    monkeypatch.setattr(s7, "known_tables", known_tables)
    monkeypatch.setattr(s7, "read_series", read_series)
    monkeypatch.setattr(s7, "score_series", lambda scorer, series: {s_: {"page": {d: np.array([0.0]) for d in
                                                                                     s7.DETECTORS}}
                                                                     for s_ in (*vu.F_TESTED, "F_H1")})
    monkeypatch.setattr(s7, "margin_sources", lambda series, h0: {})
    monkeypatch.setattr(s7, "all_margins", lambda scorer, src: ({}, {}))
    monkeypatch.setattr(s7, "criterion_cells", lambda *a: {"cells": {}, "sessions": {}})
    monkeypatch.setattr(s7, "human_cells", lambda *a: {})
    real = s7.load_item
    state: dict = {}
    for v in vu.VARIANTS:
        vu.variant(v, {}, v7s, [], None, {}, state)
        assert s7.load_item is real
    t_ff, h_ff = seen["F_K"]
    t_k, h_k = seen["K"]
    assert not np.array_equal(t_ff, h_ff)      # Firefox: event time in t, handler time in h
    assert np.array_equal(t_k, h_k)            # Chrome: the same in both
    assert np.allclose(np.diff(h_ff), 50.0, atol=0.5)


def test_loader_is_restored_when_a_variant_stops(tmp_path, monkeypatch):
    real = s7.load_item
    monkeypatch.setattr(s7, "read_runlogs", lambda root: (_ for _ in ()).throw(s7.RunStop("stop")))
    with pytest.raises(SystemExit):
        vu.variant("h", {}, tmp_path, [], None, {}, {})
    assert s7.load_item is real


def _cells(**obs):
    return {k: {"observed": v, "observed_bc": [0, 0, 1]} for k, v in obs.items()}


def test_time_choice():
    valid = {d: True for d in s7.DETECTORS} | {"D4": False}
    k1, k2, k4 = "F_M2_8B/page/D1", "F_M0/page/D3", "F_M0/page/D4"
    t = _cells(**{k1: "not changed", k2: "not changed", k4: "responds", "F_M2_8B/hook/D1": "not changed",
                  "MC1/page/D1": "responds"})
    h = _cells(**{k1: "responds", k2: "not changed", k4: "not changed", "F_M2_8B/hook/D1": "responds",
                  "MC1/page/D1": "not changed"})
    r = vu.time_choice(t, h, valid)
    assert r["reading"] == "changes" and r["cells_differing"] == 1 and set(r["cells"]) == {k1, k2}
    h[k1]["observed"] = "not changed"
    assert vu.time_choice(t, h, valid)["reading"] == "does not change"
    h[k1]["observed"] = "undecided"
    assert vu.time_choice(t, h, valid)["reading"] == "not decided"


def test_agreement_and_sessions():
    a = {"X/page/D1": {"observed_bc": [0, 1, 5], "predicted_bc": [0, 1, 5]},
         "Y/page/D1": {"observed_bc": [0, 0, 5], "predicted_bc": [0, 0, 5]}}
    b = {**a, "Y/page/D1": {"observed_bc": [0, 2, 5], "predicted_bc": [0, 0, 5]}}
    assert vu.agreement_with_v7s(a, a)["cells_not_as_v7s"] == 0
    assert vu.agreement_with_v7s(a, b)["first_not_as_v7s"] == ["Y/page/D1"]
    jt = {"F/page/D1": (np.array([True, False, True]), np.array([True, True, False]))}
    jh = {"F/page/D1": (np.array([False, False, True]), np.array([True, True, True]))}
    assert vu.sessions_differing(jt, jh) == {"F/page/D1": {"judged_in_both": 2, "differing": 1}}


def test_firefox_verdict_reads_firefox_cells_of_valid_detectors_only():
    valid = {d: True for d in s7.DETECTORS} | {"D4": False}
    cells = {"F_M2_8B/page/D1": {"predicted": "responds", "observed": "not changed", "valid": True},
             "F_M0/page/D4": {"predicted": "responds", "observed": "not changed", "valid": False},
             "MC1/page/D1": {"predicted": "responds", "observed": "not changed", "valid": True}}
    cr = {"cells": cells, "sessions": {k: [] for k in cells}}
    v = vu.firefox_verdict(cr, valid)
    assert v["cells_with_two_decided_readings"] == 1 and v["cells_disagreeing"] == 1


def _crit(obs):
    cells, sessions = {}, {}
    for s_ in (*s7.MC, *vu.F_TESTED):
        for la in run7.LAYERS:
            for d in s7.DETECTORS:
                k = f"{s_}/{la}/{d}"
                o = obs if s_ in vu.F_TESTED and la == "page" else "not changed"
                cells[k] = {"observed": o, "predicted": "not changed", "observed_bc": [0, int(o == "responds"), 40],
                            "predicted_bc": [0, 0, 40], "without_prediction": 0, "left_out_no_plan": 0,
                            "left_out_other": 0, "extrapolated": 0, "valid": d != "D4", "tested": True,
                            "sessions_differing": 0}
                sessions[k] = [(False, o == "responds")] * 40
    return {"cells": cells, "sessions": sessions}


def test_run_assembles_the_record(monkeypatch):
    valid = {d: d != "D4" for d in s7.DETECTORS}
    monkeypatch.setattr(vu, "persons_and_scorer", lambda: ([], None, valid, {"essay": ("f", "e"), "copy": ("g", "c")}))
    monkeypatch.setattr(run7, "load_all", lambda root: {})
    j = (np.array([True, False]), np.array([True, True]))

    def variant(v, *a):
        obs = "not changed" if v == "t" else "responds"
        th = {f"{t:g}": _crit(obs) for t in (s7.THETA, *s7.THETA_SENS)}
        return {"tables": {"firefox/page": {1.0: 0.4, 50.0: 0.9}}, "crit": th,
                "human": {t: {f"F_H1/page/{d}": {"observed_bc": [0, 0, 10], "predicted_bc": [0, 0, 10]}
                              for d in s7.DETECTORS} for t in th},
                "judgments": {f"{s_}/page/{d}": j for s_ in (*vu.F_TESTED, "F_H1") for d in s7.DETECTORS},
                "measures": {}}

    monkeypatch.setattr(vu, "variant", variant)
    v7s_rec = {"criterion": {"1": _crit("not changed")}}
    rec = vu.run(Path("."), Path("."), v7s_rec)
    json.dumps(s7.to_json(rec), allow_nan=False)
    assert rec["agreement_t_with_v7s"]["cells_not_as_v7s"] == 0
    assert rec["time_choice"]["reading"] == "changes"
    assert rec["firefox_verdict"]["t"]["1"]["verdict"] == "held"
    assert rec["firefox_verdict"]["h"]["1"]["verdict"] == "failed"
    assert rec["cells_changed_outside_firefox_page"] == 0
    assert rec["firefox_page_path_sd"]["h"] == {"1": 0.4, "50": 0.9}
    assert set(rec["f_h1"]["h"]) == {f"F_H1/page/{d}" for d in s7.DETECTORS}


def test_main_stops_without_v7s_record(tmp_path, monkeypatch):
    import _common as cm

    monkeypatch.setattr(cm, "same_as_head", lambda paths: None)
    monkeypatch.setattr(vu, "OUT", tmp_path / "out" / "record.json")
    monkeypatch.setattr(vu, "V7S_RECORD", tmp_path / "missing.json")
    monkeypatch.setattr(vu, "run", lambda *a: pytest.fail("run() reached without v7s's record"))
    with pytest.raises(SystemExit, match="v7s's record"):
        vu.main(["--v7-captures", str(tmp_path), "--v7s-captures", str(tmp_path)])
