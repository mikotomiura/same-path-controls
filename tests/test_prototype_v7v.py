"""Tests of works/prototype_v7v.py: the counts behind v7s's cells, the margins' table and the reproduction."""
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
import prototype_v7v as vv  # noqa: E402

VALID = dict(vv.VALID_IN_V7S)


def test_constants_are_the_conditions():
    c = json.loads(vv.COND.read_text(encoding="utf-8"))
    vv.base_check(c)
    for k in ("v7u_commit", "v7s_record_sha256", "v7u_record_sha256"):
        bad = json.loads(json.dumps(c))
        bad["base"][k] = "0" * 64
        with pytest.raises(SystemExit):
            vv.base_check(bad)
    assert vv.V7S_RECORD_SHA256 == c["base"]["v7s_record_sha256"] == vu.V7S_RECORD_SHA256


def test_code_check_passes_on_the_committed_code():
    vv.code_check()


# ---------------------------------------------------------------- a small series through v7s's criterion_cells


N = 12


def _series():
    base = [s7.Sess(np.arange(0.0, 50.0, 1.0), ["alnum"] * 50) for _ in range(N)]
    status = ["captured"] * (N - 1) + ["no_plan"]
    return {"MC1": s7.Series("MC1", base, {la: list(base) for la in run7.LAYERS}, status, "MC1", "chrome",
                             [1.0] * N)}


def _scores():
    rng = np.random.default_rng(0)
    sc = {"base": {}, "hook": {}, "page": {}}
    for d in s7.DETECTORS:
        if d == "D1":
            sc["base"][d] = np.full(N, 0.1)                           # all 'machine' at the plan
            sc["hook"][d] = np.full(N, 0.1)
            sc["page"][d] = np.where(np.arange(N) < 5, 0.5, 0.1)      # five turn 'human' on the page
            sc["page"][d][7] = np.nan                                  # one without a judgment
        else:
            sc["base"][d] = rng.uniform(0.0, 0.4, N)                  # all 'human'
            sc["hook"][d] = sc["base"][d].copy()
            sc["page"][d] = np.where(np.arange(N) == 2, 0.9, sc["base"][d])   # one turns 'machine'
    return {"MC1": sc}


def _margins_counts():
    margins = {"MC1": {d: [0.5] * N for d in s7.DETECTORS}}
    cnt = np.zeros((N, len(s7.SIGMAS)), dtype=int)
    cnt[:, 10:] = 10
    counts = {"MC1": {d: cnt.copy() for d in s7.DETECTORS}}
    tables = {"chrome/hook": {0.0: 0.1, 5.0: 0.1}, "chrome/page": {0.0: 0.8, 5.0: 0.1}}
    return margins, counts, tables


def test_cells_with_sets_keeps_the_observed_call_and_restores_v7s(monkeypatch):
    monkeypatch.setattr(s7, "MACHINE_SERIES_ALL", ("MC1",))
    real = s7.cell_reading
    margins, counts, tables = _margins_counts()
    series, scores = _series(), _scores()
    crit, kept = vv.cells_with_sets(series, scores, margins, counts, tables, VALID, 1.0)
    assert s7.cell_reading is real
    plain = s7.criterion_cells(series, scores, margins, counts, tables, VALID, 1.0)
    assert s7.to_json(crit) == s7.to_json(plain)                      # the wrap changes nothing
    before, after, keep = kept["MC1/page/D1"]
    assert np.array_equal(before, scores["MC1"]["base"]["D1"])
    assert np.array_equal(after, scores["MC1"]["page"]["D1"], equal_nan=True)   # the observed call, not predicted
    assert not keep[N - 1]                                             # no_plan is outside the set
    ic, off, first = vv.in_cell_counts(crit["cells"], kept, VALID)
    assert off == 0 and first == []
    row = ic["MC1/page/D1"]
    assert [row["b"], row["c"], row["n"]] == crit["cells"]["MC1/page/D1"]["observed_bc"]
    assert row["machine_plan"] == row["n"] and row["machine_recorded"] == row["n"] - row["c"] + row["b"]
    assert row["c"] == 5 and row["n"] == N - 2                        # no_plan and the NaN left out
    d6 = ic["MC1/page/D6"]
    assert d6["machine_plan"] == 0 and d6["machine_recorded"] == d6["b"] == 1
    assert not any(k.endswith(("/D4", "/D5")) for k in ic)            # not valid: not counted


def test_cells_with_sets_restores_v7s_when_it_stops(monkeypatch):
    real = s7.cell_reading
    monkeypatch.setattr(s7, "criterion_cells", lambda *a: (_ for _ in ()).throw(s7.RunStop("stop")))
    with pytest.raises(SystemExit):
        vv.cells_with_sets({}, {}, {}, {}, {}, VALID, 1.0)
    assert s7.cell_reading is real


def test_in_cell_counts_reports_a_cell_not_as_its_observed_bc(monkeypatch):
    monkeypatch.setattr(s7, "MACHINE_SERIES_ALL", ("MC1",))
    margins, counts, tables = _margins_counts()
    crit, kept = vv.cells_with_sets(_series(), _scores(), margins, counts, tables, VALID, 1.0)
    crit["cells"]["MC1/hook/D3"]["observed_bc"] = [9, 9, 9]
    _, off, first = vv.in_cell_counts(crit["cells"], kept, VALID)
    assert off == 1 and first == ["MC1/hook/D3"]


def test_in_cell_counts_firefox_page_only():
    kept = {k: (np.array([0.9]), np.array([0.9]), np.array([True])) for k in ("F_M0/page/D3", "F_M0/hook/D3",
                                                                             "MC1/page/D3", "F_H1/page/D3")}
    cells = {k: {"observed_bc": [0, 0, 1]} for k in kept}
    ic, off, _ = vv.in_cell_counts(cells, kept, VALID, only_firefox_page=True)
    assert set(ic) == {"F_M0/page/D3"} and off == 0


# ---------------------------------------------------------------- judged counts and margins


def _all_scores(value):
    return {name: {side: {d: np.array(value, dtype=float) for d in s7.DETECTORS} for side in vv.SIDES}
            for name in vv.ALL_SERIES}


def test_judged_counts():
    sc = _all_scores([0.9, 0.2, np.nan])
    sc["M0"]["base"]["D1"] = np.array([0.1, 0.1, 0.5])               # D1: delta at or below 0.269 is 'machine'
    out = vv.judged_counts(sc, VALID)
    assert out["M0/base/D1"] == {"sessions": 3, "judged": 3, "machine": 2}
    assert out["M0/base/D6"] == {"sessions": 3, "judged": 2, "machine": 1}
    assert not any(k.endswith(("/D4", "/D5")) for k in out)
    assert {k.split("/")[0] for k in out} == set(vv.ALL_SERIES) and {k.split("/")[1] for k in out} == set(vv.SIDES)
    h = vv.judged_counts(sc, VALID, only_firefox_page=True)
    assert {k.split("/")[0] for k in h} == set(vv.FIREFOX_SERIES) and {k.split("/")[1] for k in h} == {"page"}


def test_margin_table():
    plan_m = [0.1 * (i + 1) for i in range(8)] + ["none", None]
    margins = {"M0": {d: plan_m for d in s7.DETECTORS}, "H0": {d: [1.0, 2.0, 3.0, "none"] for d in s7.DETECTORS}}
    base = {"M0": {d: np.array([0.9] * 6 + [0.1] * 3 + [np.nan]) for d in s7.DETECTORS},
            "H0": {d: np.array([0.9, 0.1, 0.1, 0.1]) for d in s7.DETECTORS}}
    out = vv.margin_table(margins, base, VALID)
    m0 = out["M0/D6"]
    assert (m0["sessions"], m0["no_margin"], m0["none"]) == (10, 1, 1)
    assert m0["margin_sd_ms"]["n"] == 8 and m0["margin_sd_ms"]["min"] == pytest.approx(0.1)   # plans: with ends
    assert m0["by_judgment"]["machine"]["margin_sd_ms"]["n"] == 6
    assert m0["by_judgment"]["human"]["sessions"] == 3 and m0["by_judgment"]["human"]["none"] == 1
    h0 = out["H0/D6"]
    assert h0["margin_sd_ms"] == {"n": 3}                               # persons: the count only below 5
    assert "M0/D4" not in out and "M0/D5" not in out
    big = {"H0": {"D6": [float(i) for i in range(1, 21)]}}
    hb = vv.margin_table(big, {"H0": {"D6": np.full(20, 0.9)}}, {"D6": True})["H0/D6"]["margin_sd_ms"]
    assert "min" not in hb and "max" not in hb and hb["n"] == 20         # no person's minimum or maximum
    with pytest.raises(SystemExit):
        vv.margin_table({"M0": {"D6": [0.1]}}, {"M0": {"D6": np.array([0.9, 0.9])}}, {"D6": True})


# ---------------------------------------------------------------- reproduction


def _cell(obs="not changed", bc=(0, 0, 40)):
    return {"observed": obs, "predicted": "not changed", "observed_bc": list(bc), "predicted_bc": [0, 0, 40]}


def _res_and_records():
    ths = [f"{t:g}" for t in (s7.THETA, *s7.THETA_SENS)]
    cells = {"MC1/page/D1": _cell(), "F_M0/page/D6": _cell(), "F_M2_8B/page/D1": _cell()}
    human = {"H1_idle/page/D1": {"observed_bc": [0, 0, 80], "predicted_bc": [0, 0, 80], "observed": "x",
                                 "predicted": "x"},
             "F_H1/page/D1": {"observed_bc": [0, 1, 10], "predicted_bc": [0, 0, 10]}}
    res = {v: {"crit": {th: {"cells": json.loads(json.dumps(cells))} for th in ths},
               "human": {th: json.loads(json.dumps(human)) for th in ths}} for v in vu.VARIANTS}
    tables = {"chrome/page": {1.0: 0.4, 50.0: 0.1}}
    jitter = {"M0/D1": {"n": 40, "share_by_sigma": [0.0, 0.5], "first_sigma": {"3_of_n": 0.05, "0.10": "not reached"}}}
    v7s_rec = {"criterion": {th: {"cells": json.loads(json.dumps(cells))} for th in ths},
               "criterion_human": {th: json.loads(json.dumps(human)) for th in ths},
               "path": {"path_sd_tables": {"chrome/page": {"1": 0.4, "50": 0.1}}},
               "perturbation_jitter": json.loads(json.dumps(jitter))}
    ff = {k: v for k, v in cells.items() if k.startswith("F_")}
    v7u_rec = {"firefox_cells": {v: json.loads(json.dumps(ff)) for v in vu.VARIANTS},
               "f_h1": {v: {"F_H1/page/D1": {"observed_bc": [0, 1, 10], "predicted_bc": [0, 0, 10]}}
                        for v in vu.VARIANTS}}
    return res, tables, jitter, v7s_rec, v7u_rec


def test_reproduction_all_as_deposited():
    out = vv.reproduction(*_res_and_records())
    assert out["all_as_deposited"] is True
    assert out["v7s_cells_1"]["compared"] == 3 and out["v7u_firefox_cells_h"]["compared"] == 2


@pytest.mark.parametrize("change, part", [
    (lambda r, t, j, s, u: r["t"]["crit"]["0.5"]["cells"]["MC1/page/D1"].update(observed="responds"), "v7s_cells_0.5"),
    (lambda r, t, j, s, u: r["t"]["crit"]["1"]["cells"]["MC1/page/D1"].update(predicted_bc=[1, 0, 40]),
     "v7s_cells_1"),
    (lambda r, t, j, s, u: r["t"]["human"]["2"]["H1_idle/page/D1"].update(observed="y"), "v7s_human_2"),
    (lambda r, t, j, s, u: t["chrome/page"].update({50.0: 0.2}), "v7s_path_sd_tables"),
    (lambda r, t, j, s, u: j["M0/D1"]["first_sigma"].update({"0.10": 0.07}), "v7s_perturbation_jitter"),
    (lambda r, t, j, s, u: r["h"]["crit"]["1"]["cells"]["F_M0/page/D6"].update(observed_bc=[0, 1, 40]),
     "v7u_firefox_cells_h"),
    (lambda r, t, j, s, u: r["h"]["human"]["1"]["F_H1/page/D1"].update(observed_bc=[0, 2, 10]), "v7u_f_h1_h"),
    (lambda r, t, j, s, u: s["criterion"]["1"]["cells"].pop("MC1/page/D1"), "v7s_cells_1"),
])
def test_reproduction_finds_each_difference(change, part):
    args = _res_and_records()
    change(*args)
    out = vv.reproduction(*args)
    assert out["all_as_deposited"] is False
    assert out[part]["not_as_deposited"] == 1 and len(out[part]["first"]) == 1
    assert sum(p["not_as_deposited"] for p in out.values() if isinstance(p, dict)) == 1


def test_record_parts_stop_before_scoring():
    s = json.loads(vv.V7S_RECORD.read_text(encoding="utf-8")) if vv.V7S_RECORD.exists() else None
    if s is not None:
        vv.v7s_record_parts(s)
    with pytest.raises(SystemExit, match="every theta"):
        vv.v7s_record_parts({"criterion": {"1": {"cells": {}}}, "criterion_human": {"1": {}},
                             "perturbation_jitter": {}, "valid": VALID})
    full = {"criterion": {th: {"cells": {}} for th in ("1", "0.5", "2")},
            "criterion_human": {th: {} for th in ("1", "0.5", "2")}, "perturbation_jitter": {}, "valid": VALID}
    vv.v7s_record_parts(full)
    with pytest.raises(SystemExit):
        vv.v7s_record_parts({**full, "valid": {**VALID, "D5": True}})
    vv.v7u_record_shape({"firefox_cells": {"t": {}, "h": {}}, "f_h1": {"t": {}, "h": {}}})
    with pytest.raises(SystemExit, match="v7u's record"):
        vv.v7u_record_shape({"firefox_cells": {"t": {}}, "f_h1": {"t": {}, "h": {}}})


# ---------------------------------------------------------------- the run's order


def _run_stubs(monkeypatch, valid=VALID, sources=None):
    calls = []
    monkeypatch.setattr(vu, "record_shape", lambda rec: calls.append("v7s_shape"))
    monkeypatch.setattr(vv, "v7s_record_parts", lambda rec: calls.append("v7s_parts"))
    monkeypatch.setattr(vv, "v7u_record_shape", lambda rec: calls.append("v7u_shape"))
    monkeypatch.setattr(vu, "runlogs_as_v7s", lambda root, rec: calls.append("runlogs"))
    monkeypatch.setattr(vu, "persons_and_scorer", lambda: calls.append("persons") or
                        ([], None, valid, {"essay": ("f", "e"), "copy": ("g", "c")}))
    monkeypatch.setattr(run7, "load_all", lambda root: {})
    monkeypatch.setattr(vu, "load_variant", lambda v, *a: calls.append(f"load_{v}") or {"series": v, "tables": {}})
    monkeypatch.setattr(vu, "same_bases", lambda a, b: calls.append("bases"))
    src = {k: [] for k in (vv.MARGIN_SOURCES if sources is None else sources)}
    monkeypatch.setattr(s7, "margin_sources", lambda series, h0: calls.append(f"sources_{series}") or src)
    monkeypatch.setattr(s7, "all_margins", lambda scorer, s: calls.append("margins") or ({}, {}))
    monkeypatch.setattr(vv, "score_variant", lambda loaded, *a: calls.append(f"score_{loaded['series']}") or
                        {"scores": {}, "crit": {"1": {"cells": {}}}, "human": {}, "kept": {}})
    monkeypatch.setattr(vv, "reproduction", lambda *a: {"all_as_deposited": True})
    monkeypatch.setattr(vv, "judged_counts", lambda *a, **k: {})
    monkeypatch.setattr(vv, "margin_table", lambda *a: {})
    monkeypatch.setattr(vv, "source_scores", lambda *a: calls.append("source_scores") or {})
    monkeypatch.setattr(s7, "jitter_table", lambda *a: {})
    return calls


def test_run_order(monkeypatch):
    calls = _run_stubs(monkeypatch)
    rec = vv.run(Path("."), Path("."), {}, {})
    assert calls == ["v7s_shape", "v7s_parts", "v7u_shape", "runlogs", "persons", "load_t", "load_h", "bases",
                     "sources_t", "margins", "score_t", "score_h", "source_scores"]
    assert set(rec) == {"valid", "reproduction", "judged", "in_cell", "margins", "data_sha256"}


def test_run_stops_on_another_validity_before_loading(monkeypatch):
    calls = _run_stubs(monkeypatch, valid={**VALID, "D5": True})
    with pytest.raises(SystemExit, match="validity"):
        vv.run(Path("."), Path("."), {}, {})
    assert "load_t" not in calls


def test_run_stops_on_other_sources_before_the_margins(monkeypatch):
    calls = _run_stubs(monkeypatch, sources=vv.MARGIN_SOURCES - {"MC5"})
    with pytest.raises(SystemExit, match="sources"):
        vv.run(Path("."), Path("."), {}, {})
    assert "margins" not in calls


@pytest.mark.parametrize("which", ["v7s", "v7u"])
def test_main_stops_without_a_record(tmp_path, monkeypatch, which):
    import _common as cm

    monkeypatch.setattr(cm, "same_as_head", lambda paths: None)
    monkeypatch.setattr(vv, "code_check", lambda: None)
    monkeypatch.setattr(vv, "OUT", tmp_path / "out" / "record.json")
    good = tmp_path / "good.json"
    good.write_text("{}")
    monkeypatch.setattr(vv, "V7S_RECORD", tmp_path / "missing.json" if which == "v7s" else vv.V7S_RECORD)
    monkeypatch.setattr(vv, "V7U_RECORD", tmp_path / "missing.json" if which == "v7u" else vv.V7U_RECORD)
    if which == "v7u" and not vv.V7S_RECORD.exists():
        pytest.skip("v7s's record is not in this checkout")
    monkeypatch.setattr(vv, "run", lambda *a: pytest.fail("run() reached without the record"))
    with pytest.raises(SystemExit, match=f"{which}'s record"):
        vv.main(["--v7-captures", str(tmp_path), "--v7s-captures", str(tmp_path)])


def test_main_refuses_to_overwrite(tmp_path, monkeypatch):
    out = tmp_path / "record.json"
    out.write_text("{}")
    monkeypatch.setattr(vv, "OUT", out)
    monkeypatch.setattr(vv, "run", lambda *a: pytest.fail("ran over an existing record"))
    with pytest.raises(SystemExit, match="refusing"):
        vv.main(["--v7-captures", str(tmp_path), "--v7s-captures", str(tmp_path)])


def test_main_checks_the_conditions_base_before_the_records(tmp_path, monkeypatch):
    import _common as cm

    monkeypatch.setattr(cm, "same_as_head", lambda paths: None)
    monkeypatch.setattr(vv, "code_check", lambda: None)
    monkeypatch.setattr(vv, "OUT", tmp_path / "out" / "record.json")
    monkeypatch.setattr(vv, "base_check", lambda c: (_ for _ in ()).throw(SystemExit("base checked")))
    monkeypatch.setattr(vv, "run", lambda *a: pytest.fail("run() reached without the base check"))
    with pytest.raises(SystemExit, match="base checked"):
        vv.main(["--v7-captures", str(tmp_path), "--v7s-captures", str(tmp_path)])
