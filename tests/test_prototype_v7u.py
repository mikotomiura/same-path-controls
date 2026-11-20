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
    assert (out.layers["page"].not_counted, out.layers["page"].repeats) == \
        (cap.layers["page"].not_counted, cap.layers["page"].repeats)
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


def test_load_variant_h_through_v7s_known_tables_and_restores_the_loader(tmp_path, monkeypatch):
    """v7s's own read_runlogs, known_tables, load_item and unit_caps on v7t's fake folders (read_series stubbed:
    its plan checks need all of v7's captures)."""
    from test_prototype_v7t import _folder

    v7, v7s = _fake_roots(tmp_path)
    for meta in list(v7s.glob("K_chrome*/*/meta.json")) + list(v7.glob("P_*/*/meta.json")):
        meta.write_text(json.dumps({"browser": {"name": "chrome", "version": s7.CHROME_VERSION}}), encoding="utf-8")
    # a 0 ms plan as the work PC records it (v7t): the hook's times apart by microseconds, the handler's partly a
    # millisecond apart; the plan (events.ndjson) stays 0 ms
    for d in list(v7s.glob("*/c000")):
        h = 1000.0 + np.floor(np.arange(40) / 2.0)   # 1000 to 1019 ms: the event time crosses one step
        firefox = d.parent.name.startswith("F_")
        t = np.round(np.floor(h / vt_step()) * vt_step()) if firefox else h + np.arange(40) * 0.01
        (d / "page.ndjson").unlink()
        _page(d / "page.ndjson", t, h)
        if (d / "hook.ndjson").exists():
            rows = [{"type": "header"}] + [{"type": "key", "t_ns": 2_000_000_000 + 1000 * i, "down": True,
                                            "injected": True, "ours": True, "repeat": False, "kind": "alnum"}
                                           for i in range(40)] + [{"type": "footer"}]
            (d / "hook.ndjson").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    for ms in s7.v7.P_NOHOOK_MS:
        h = 1000.0 + np.arange(40) * ms
        _folder(v7 / run7.item_name("P_nohook", int(ms)), h, False, "chrome", False, "P_nohook")
    p_idle = [run7.load_capture(v7, {run7.item_name("P_idle", int(ms)): "done"}, "P_idle", int(ms))
              for ms in run7.CONTROLS_MS]
    p_nohook = [run7.load_capture(v7, {run7.item_name("P_nohook", int(ms)): "done"}, "P_nohook", int(ms), hook=False)
                for ms in s7.v7.P_NOHOOK_MS]
    v7caps = {"P_idle": p_idle, "P_nohook": p_nohook}

    def read_series(v7caps, root, runlog, h0, kinds):
        kinds["firefox_version"] = "1"
        return {}

    monkeypatch.setattr(s7, "read_series", read_series)
    real = s7.load_item
    out = {v: vu.load_variant(v, v7caps, v7s, []) for v in vu.VARIANTS}
    assert s7.load_item is real
    ft, fh = out["t"]["tables"]["firefox/page"], out["h"]["tables"]["firefox/page"]
    assert set(ft) == set(fh) == {float(x) for x in s7.KNOWN_POINTS["firefox"]}
    assert ft[50.0] > 1.0 and fh[50.0] < 1.0                     # stepped at the event's time, not at the handler's
    for k in ("chrome/page", "chrome/hook", "firefox/hook"):
        assert out["t"]["tables"][k] == out["h"]["tables"][k]    # Chrome and every hook read the same way


def test_loader_is_restored_when_a_variant_stops(tmp_path, monkeypatch):
    real = s7.load_item
    monkeypatch.setattr(s7, "read_runlogs", lambda root: (_ for _ in ()).throw(s7.RunStop("stop")))
    with pytest.raises(SystemExit):
        vu.load_variant("h", {}, tmp_path, [])
    assert s7.load_item is real


def test_same_bases():
    S = s7.Series
    a = {"F_M0": S("F_M0", [s7.Sess([0.0, 5.0], ["alnum"] * 2), None], {}, ["captured", "no_plan"], "M0", "firefox",
                   [5.0, None])}
    b = {"F_M0": S("F_M0", [s7.Sess([0.0, 5.0], ["alnum"] * 2), None], {"page": []}, ["captured", "no_plan"], "M0",
                   "firefox", [5.0, None])}
    vu.same_bases(a, b)
    b["F_M0"].margins_of = "M2_8B"
    with pytest.raises(SystemExit):
        vu.same_bases(a, b)
    b["F_M0"].margins_of = "M0"
    b["F_M0"].status = ["captured", "failed"]
    with pytest.raises(SystemExit):
        vu.same_bases(a, b)
    b["F_M0"].status = ["captured", "no_plan"]
    b["F_M0"].base.append(None)
    with pytest.raises(SystemExit):
        vu.same_bases(a, b)
    b["F_M0"].base.pop()
    b["F_M0"].base[0] = s7.Sess([0.0, 6.0], ["alnum"] * 2)
    with pytest.raises(SystemExit):
        vu.same_bases(a, b)


def test_persons_and_scorer_is_v7s_mains_copy():
    """The copied part of v7s's main, statement for statement (names qualified with s7. and the return aside)."""
    import ast
    import inspect
    import re

    def stmts(src: str) -> list[str]:
        return [re.sub(r"\bs7\.", "", ast.unparse(n)) for n in ast.parse(src).body]

    main = ast.parse(inspect.getsource(s7.main)).body[0].body
    start = next(i for i, n in enumerate(main) if "load_conditions(run7.COND)" in ast.unparse(n))
    end = next(i for i, n in enumerate(main) if ast.unparse(n).startswith("valid = "))
    theirs = [ast.unparse(n) for n in main[start:end + 1]]
    ours = stmts(inspect.cleandoc("\n".join(inspect.getsource(vu.persons_and_scorer).splitlines()[1:])))
    ours = [x for x in ours if not x.startswith(("return ", "'", '"'))]
    assert ours == theirs


def test_runlogs_and_tables_as_v7s(tmp_path):
    for b in range(1, 3):
        (tmp_path / f"runlog.b{b}.ndjson").write_text(json.dumps({"type": "header", "runs_sha256": "r"}) + "\n")
    import _common as cm
    pv = {"v7s_runlogs_sha256": {f"runlog.b{b}.ndjson": cm.sha256(tmp_path / f"runlog.b{b}.ndjson")
                                 for b in range(1, 3)}, "v7s_runs_sha256": "r"}
    vu.runlogs_as_v7s(tmp_path, {"provenance": pv})
    with pytest.raises(SystemExit):
        vu.runlogs_as_v7s(tmp_path, {"provenance": {**pv, "v7s_runs_sha256": "other"}})
    (tmp_path / "runlog.b2.ndjson").write_text("{}\n")
    with pytest.raises(SystemExit):
        vu.runlogs_as_v7s(tmp_path, {"provenance": pv})
    assert vu.tables_as_v7s({"firefox/page": {1.0: 0.5, 50.0: 0.9}}, {"firefox/page": {"1": 0.5, "50": 0.9}}) == 0
    assert vu.tables_as_v7s({"firefox/page": {1.0: 0.5}}, {"firefox/page": {"1": 0.5, "50": 0.9}}) == 1


def vt_step():
    import prototype_v7t as vt
    return vt.STEP_MS


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
    h[k2]["observed"] = "responds"   # one decided difference with an undecided cell beside it: 'changes'
    assert vu.time_choice(t, h, valid)["reading"] == "changes"


def test_agreement_and_sessions():
    a = {"X/page/D1": {"observed_bc": [0, 1, 5], "predicted_bc": [0, 1, 5]},
         "Y/page/D1": {"observed_bc": [0, 0, 5], "predicted_bc": [0, 0, 5]}}
    b = {**a, "Y/page/D1": {"observed_bc": [0, 2, 5], "predicted_bc": [0, 0, 5]}}
    assert vu.agreement_with_v7s(a, a)["cells_not_as_v7s"] == 0
    assert vu.agreement_with_v7s(a, b)["first_not_as_v7s"] == ["Y/page/D1"]
    c = {**a, "X/page/D1": {"observed_bc": [0, 1, 5], "predicted_bc": [1, 1, 5]}}
    assert vu.agreement_with_v7s(a, c)["first_not_as_v7s"] == ["X/page/D1"]
    assert vu.agreement_with_v7s(a, {"X/page/D1": a["X/page/D1"]})["first_not_as_v7s"] == ["Y/page/D1"]
    jt = {"F/page/D1": (np.array([True, False, True]), np.array([True, True, False]))}
    jh = {"F/page/D1": (np.array([False, False, True]), np.array([True, True, True]))}
    assert vu.sessions_differing(jt, jh) == {"F/page/D1": {"judged_in_both": 2, "differing": 1}}


def test_firefox_verdict_reads_firefox_cells_of_valid_detectors_only():
    valid = {d: True for d in s7.DETECTORS} | {"D4": False}
    cells = {"F_M2_8B/page/D1": {"predicted": "responds", "observed": "not changed", "valid": True},
             "F_M0/page/D4": {"predicted": "responds", "observed": "not changed", "valid": False},
             "MC1/page/D1": {"predicted": "responds", "observed": "not changed", "valid": True}}
    cr = {"cells": cells, "sessions": {k: [(False, True)] for k in cells}}
    v = vu.firefox_verdict(cr, valid)
    assert v["cells_with_two_decided_readings"] == 1 and v["cells_disagreeing"] == 1 and v["sessions"] == 1


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
    calls = []
    monkeypatch.setattr(vu, "runlogs_as_v7s", lambda root, rec: calls.append(("runlogs",)))
    monkeypatch.setattr(run7, "load_all", lambda root: {})
    monkeypatch.setattr(vu, "load_variant", lambda v, *a: calls.append(("load", v)) or
                        {"series": f"series-{v}", "tables": {}, "measures": {}})
    monkeypatch.setattr(vu, "same_bases", lambda a, b: calls.append(("bases", a, b)))
    monkeypatch.setattr(s7, "margin_sources", lambda series, h0: calls.append(("sources", series)) or {})
    monkeypatch.setattr(s7, "all_margins", lambda scorer, src: calls.append(("margins",)) or ({}, {}))
    j = (np.array([True, False]), np.array([True, True]))

    def score_variant(loaded, *a):
        v = loaded["series"][-1]
        obs = "not changed" if v == "t" else "responds"
        th = {f"{t:g}": _crit(obs) for t in (s7.THETA, *s7.THETA_SENS)}
        human = {f"F_H1/{la}/{d}": {"observed_bc": [0, 0, 10], "predicted_bc": [0, 0, 10]}
                 for la in run7.LAYERS for d in s7.DETECTORS}
        if v == "h":
            human["F_H1/page/D1"] = {"observed_bc": [0, 3, 10], "predicted_bc": [0, 0, 10]}   # inside: not counted
        return {"tables": {"firefox/page": {1.0: 0.4, 50.0: 0.9}, "chrome/page": {1.0: 0.3}}, "crit": th,
                "human": {t: dict(human) for t in th},
                "judgments": {f"{s_}/page/{d}": j for s_ in (*vu.F_TESTED, "F_H1") for d in s7.DETECTORS}}

    monkeypatch.setattr(vu, "score_variant", score_variant)
    v7s_rec = {"criterion": {"1": _crit("not changed")},
               "criterion_human": {"1": {f"F_H1/{la}/{d}": {"observed_bc": [0, 0, 10], "predicted_bc": [0, 0, 10]}
                                         for la in run7.LAYERS for d in s7.DETECTORS}},
               "path": {"path_sd_tables": {"firefox/page": {"1": 0.4, "50": 0.9}, "chrome/page": {"1": 0.3}}},
               "provenance": {"v7s_runlogs_sha256": {}, "v7s_runs_sha256": "r"}}
    rec = vu.run(Path("."), Path("."), v7s_rec)
    json.dumps(s7.to_json(rec), allow_nan=False)
    # both variants loaded and their bases compared before anything is scored; the margins once, from variant t
    assert calls == [("runlogs",), ("load", "t"), ("load", "h"), ("bases", "series-t", "series-h"), ("sources", "series-t"),
                     ("margins",)]
    assert rec["agreement_t_with_v7s"]["cells_not_as_v7s"] == 0
    assert rec["agreement_t_with_v7s"]["human"]["cells_not_as_v7s"] == 0
    assert rec["agreement_t_with_v7s"]["path_sd_entries_not_as_v7s"] == 0
    assert rec["time_choice"]["reading"] == "changes"
    assert rec["firefox_verdict"]["t"]["1"]["verdict"] == "held"
    assert rec["firefox_verdict"]["h"]["1"]["verdict"] == "failed"
    assert rec["cells_changed_outside_firefox_page"] == 0
    assert rec["firefox_page_path_sd"]["h"] == {"1": 0.4, "50": 0.9}


def test_outside_firefox_page_counts_hook_and_human_cells():
    def res(hook_obs, human_obs):
        crit = {"1": {"cells": {"F_M0/hook/D1": {"observed_bc": hook_obs}, "F_M0/page/D1": {"observed_bc": [0, 9, 9]},
                                "MC1/page/D1": {"observed_bc": [0, 0, 9]}}}}
        human = {"1": {"F_H1/hook/D1": {"observed_bc": human_obs}, "F_H1/page/D1": {"observed_bc": [0, 9, 9]}}}
        return {"crit": crit, "human": human}
    t = res([0, 0, 9], [0, 0, 10])
    assert vu.outside_firefox_page({"t": t, "h": res([0, 0, 9], [0, 0, 10])}) == 0
    assert vu.outside_firefox_page({"t": t, "h": res([0, 1, 9], [0, 0, 10])}) == 1
    assert vu.outside_firefox_page({"t": t, "h": res([0, 0, 9], [1, 0, 10])}) == 1


def test_record_shape_stops_before_scoring(monkeypatch):
    monkeypatch.setattr(vu, "runlogs_as_v7s", lambda *a: pytest.fail("went on with a record of another shape"))
    with pytest.raises(SystemExit, match="does not hold"):
        vu.run(Path("."), Path("."), {"criterion": {"1": {"cells": {}}}, "path": {"path_sd_tables": {}}})


def test_main_stops_without_v7s_record(tmp_path, monkeypatch):
    import _common as cm

    monkeypatch.setattr(cm, "same_as_head", lambda paths: None)
    monkeypatch.setattr(vu, "OUT", tmp_path / "out" / "record.json")
    monkeypatch.setattr(vu, "V7S_RECORD", tmp_path / "missing.json")
    monkeypatch.setattr(vu, "run", lambda *a: pytest.fail("run() reached without v7s's record"))
    with pytest.raises(SystemExit, match="v7s's record"):
        vu.main(["--v7-captures", str(tmp_path), "--v7s-captures", str(tmp_path)])
