"""Prototype v7s: the analysis's pieces and the plan maker, on synthetic data only (no KUPA, no capture)."""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "works"))
sys.path.insert(0, str(ROOT))

import prototype_v7 as v7  # noqa: E402
import prototype_v7_run as run7  # noqa: E402
import prototype_v7s as s7  # noqa: E402
import prototype_v7s_plan as sp  # noqa: E402
from capture import events as ev  # noqa: E402

COND = json.loads((ROOT / "works" / "conditions" / "prototype_v7s.json").read_text(encoding="utf-8"))


def sess(gaps, kinds=None) -> s7.Sess:
    g = np.asarray(gaps, dtype=float)
    t = np.concatenate([[0.0], np.cumsum(g)])
    return s7.Sess(t, kinds if kinds is not None else ["alnum"] * t.size)


def human(n: int, rng: np.random.Generator) -> s7.Sess:
    g = rng.lognormal(np.log(180), 0.6, n - 1)
    kinds = list(rng.choice(["alnum", "alnum", "alnum", "space", "end", "backspace", "other"], n))
    return sess(g, kinds)


# ---------------------------------------------------------------- sessions


def test_sess_rejects_mismatch_and_unknown_kind():
    with pytest.raises(ValueError):
        s7.Sess(np.array([0.0, 1.0]), ["alnum"])
    with pytest.raises(ValueError):
        s7.Sess(np.array([0.0, 1.0]), ["alnum", "modifier"])


def test_perturbed_keeps_kinds_with_their_presses():
    s = s7.Sess(np.array([0.0, 10.0, 10.0, 30.0]), ["alnum", "space", "end", "other"])
    p = s7.perturbed(s, "grid", 1.0)
    assert p.times.tolist() == [0.0, 10.0, 30.0]          # the equal time dropped (v6's rule)
    assert p.kinds == ["alnum", "space", "other"]


# ---------------------------------------------------------------- D5


def test_d5_windows_gaps_and_codes():
    kinds = [s7.KINDS[i % 7] for i in range(66)]
    s = sess(np.arange(1, 66, dtype=float), kinds)        # 65 gaps -> 2 windows, 5 dropped
    x = s7.d5_windows(s)
    assert x.shape == (2, 60)
    assert x[0, :30].tolist() == list(np.arange(1, 31, dtype=float))
    assert x[1, 30:].tolist() == [float(s7.KIND_CODE[k]) for k in kinds[31:61]]   # kinds of the ending presses
    assert s7.d5_windows(sess(np.ones(29))).shape == (0, 60)


def test_d5_synthetic_copies_kinds_draws_gaps():
    rows = np.hstack([np.full((3, 30), 100.0), np.full((3, 30), 4.0)])
    kde = v7.KDE1(np.array([5.0, 6.0]))
    out = s7.d5_synthetic(rows, kde, np.random.default_rng(0))
    assert np.all(out[:, 30:] == 4.0)
    assert np.all(out[:, :30] < 20.0) and np.all(out[:, :30] >= 0.0)


# ---------------------------------------------------------------- D6


@pytest.mark.parametrize("n,want", [(99, []), (100, [(0, 100)]), (999, [(0, 999)]), (1000, [(0, 1000)]),
                                    (1700, [(0, 1000), (700, 1700)]), (2399, [(0, 1000), (700, 1700)]),
                                    (2400, [(0, 1000), (700, 1700), (1400, 2400)])])
def test_d6_spans(n, want):
    assert s7.d6_spans(n) == want


def test_d6_features_pairs_at_least_twice():
    kinds = ["alnum", "space"] * 60                       # pairs alnum->space and space->alnum only
    g = np.where(np.arange(119) % 2 == 0, 100.0, 300.0)   # alnum->space 100, space->alnum 300
    x = s7.d6_features(sess(g, kinds))
    assert x.shape == (1, 49)
    assert x[0, s7.PAIR_IDX[("alnum", "space")]] == 100.0
    assert x[0, s7.PAIR_IDX[("space", "alnum")]] == 300.0
    assert np.isnan(x[0, s7.PAIR_IDX[("alnum", "alnum")]])
    once = ["alnum", "space"] * 60
    once[-1] = "end"                                      # space->end... no: the last press is end once
    y = s7.d6_features(sess(g, once))
    assert np.isnan(y[0, s7.PAIR_IDX[("alnum", "end")]])  # occurs once: missing


class FixedRng:
    def __init__(self, u):
        self.u = u

    def random(self):
        return self.u


def test_with_backspaces_counter_rule():
    k = ["alnum"] * 13
    out, n = s7.with_backspaces(k, FixedRng(0.0))         # always below 0.8: after the 6th and the 12th
    assert n == 2 and out[6] == "backspace" and out[13] == "backspace" and len(out) == 15
    out, n = s7.with_backspaces(k, FixedRng(0.9))
    assert n == 0 and out == k
    out, n = s7.with_backspaces(["alnum"] * 5 + ["backspace"] + ["alnum"] * 6, FixedRng(0.0))
    assert n == 1 and out[-1] == "backspace"              # the typed backspace reset the counter


def test_at_p_forge_positive_gaps_and_fallback():
    rng = np.random.default_rng(1)
    hs = [human(300, rng) for _ in range(5)]
    st = s7.at_p_stats(hs)
    f = s7.at_p_forge(hs[0], st, np.random.default_rng(2))
    assert len(f.kinds) >= len(hs[0].kinds) and np.all(np.diff(f.times) > 0)
    st2 = s7.AtP(np.zeros(49), np.zeros(49), np.zeros(49, dtype=bool), 50.0, 0.0)
    f2 = s7.at_p_forge(sess(np.ones(9)), st2, np.random.default_rng(0))
    assert np.allclose(np.diff(f2.times), 50.0)           # global fallback, SD 0 kept constant


def test_trained_detectors_give_probabilities():
    rng = np.random.default_rng(3)
    hs = [human(400, rng) for _ in range(12)]
    d5 = s7.train_d5(hs, np.random.default_rng(4))
    d6 = s7.train_d6(hs, np.random.default_rng(5))
    test = [human(400, rng), sess(np.full(399, 55.0)), sess(np.ones(10)), None]
    for p in (d5.proba(test), d6.proba(test)):
        assert np.all((p[:2] >= 0) & (p[:2] <= 1)) and np.isnan(p[2]) and np.isnan(p[3])


# ---------------------------------------------------------------- path


def test_path_measures_grid_and_cleaning():
    raw = np.array([0.0, 1.0, 1.0, 2.0, 3.0, 4.0])       # one shared stamp
    m = s7.path_measures(raw, 1.0)
    assert m["share_on_1ms_grid"] == 1.0
    assert m["dropped_share"] == pytest.approx(1 / 6)
    assert m["path_sd"] == pytest.approx(0.0)             # cleaned gaps 1, 1, 1, 1
    assert m["raw_sd"] > 0
    m2 = s7.path_measures(np.array([0.0, 1.05, 2.05, 3.05]), 1.0)
    assert m2["share_on_1ms_grid"] == pytest.approx(2 / 3)


def test_path_sd_at():
    t = {0.0: 0.4, 1.0: 0.2, 5.0: 0.1, 200.0: 0.05}
    assert s7.path_sd_at(t, 0.5) == (pytest.approx(0.3), False)
    assert s7.path_sd_at(t, 3.0) == (pytest.approx(0.15), False)
    assert s7.path_sd_at(t, 200.0) == (0.05, False)
    assert s7.path_sd_at(t, 900.0) == (0.05, True)
    with pytest.raises(ValueError):
        s7.path_sd_at({1.0: 0.1}, 0.5)


# ---------------------------------------------------------------- margins, predictions, verdict


def test_margin_sd():
    row = np.zeros(len(s7.SIGMAS), dtype=int)
    assert s7.margin_sd(row) == "none"
    row[3] = 4
    row[5] = 5
    assert s7.margin_sd(row) == pytest.approx(math.sqrt(2) * s7.SIGMAS[5])
    assert s7.margin_sd(np.full(len(s7.SIGMAS), -1)) is None


def test_predict():
    assert s7.predict(1.0, 1.0) is True
    assert s7.predict(0.99, 1.0) is False
    assert s7.predict(1.0, 2.0, theta=0.5) is True
    assert s7.predict(None, 1.0) is None and s7.predict(1.0, None) is None
    assert s7.predict(50.0, "none") is False
    assert s7.predict(95.0, "none") is None               # beyond sqrt(2) * 60.9 ms


def test_flip_prob():
    row = np.zeros(len(s7.SIGMAS), dtype=int)
    row[4:] = 10
    row[3] = 4
    sig3 = s7.SIGMAS[3]
    assert s7.flip_prob(row, math.sqrt(2) * sig3) == pytest.approx(0.4)
    assert s7.flip_prob(row, math.sqrt(2) * sig3 / 2) == pytest.approx(0.0)          # below: 0 (rows 0-2 are 0)
    assert s7.flip_prob(row, 2 * math.sqrt(2) * sig3, theta=2.0) == pytest.approx(0.4)
    assert s7.flip_prob(row, math.sqrt(2) * 61.0) is None and s7.flip_prob(np.full(42, -1), 1.0) is None
    assert s7.flip_prob(row, None) is None


def test_expected_judgments_turns_the_most_probable():
    before = np.array([0.1, 0.1, 0.1, 0.5, 0.5, np.nan])   # D1: three 'machine', two 'human', one unjudged
    s, keep = s7.expected_judgments(before, "D1", [0.9, 0.2, 0.3, 0.6, None, 0.9])
    j, ok = run7.judgments("D1", s)
    # machine side: 0.9 + 0.2 + 0.3 = 1.4 -> 1 turned (the 0.9); human side: 0.6 -> 1 turned
    assert j[:5].tolist() == [False, True, True, True, False] and not ok[5]
    assert keep.tolist() == [True, True, True, True, False, False]
    p, _ = s7.expected_judgments(np.array([0.9, 0.2]), "D3", [0.4, 0.5])
    assert run7.judgments("D3", p)[0].tolist() == [True, True]                      # 0.4 -> 0; 0.5 -> 1 (half up)


def test_verdict_words_and_boundaries():
    cells = [{"predicted": "responds", "observed": "responds"}, {"predicted": "not changed", "observed": "undecided"}]
    assert s7.verdict(cells, [(True, True)] * 9 + [(True, False)])["verdict"] == "held"   # share 0.10: held
    assert s7.verdict(cells, [(True, True)] * 8 + [(True, False)] * 2)["verdict"] == "failed"
    bad = cells + [{"predicted": "not changed", "observed": "responds"}]
    assert s7.verdict(bad, [(True, True)])["verdict"] == "failed"
    assert s7.verdict([{"predicted": "undecided", "observed": "responds"}], [])["verdict"] == "not decided"


class D1Only:
    """A scorer standing in for all detectors: D1's delta; the others 'machine' exactly when delta <= 0.269."""

    def score(self, sessions):
        d = np.array([np.nan if s is None else v7.delta(s.gaps) for s in sessions])
        p = np.where(np.isfinite(d), (d <= v7.DELTA_T).astype(float), np.nan)
        return {k: (d if k == "D1" else p) for k in s7.DETECTORS}


def test_margin_of_a_constant_plan_follows_delta():
    """A constant 10 ms plan: D1 flips once the gap SD (sqrt 2 sigma) passes 0.269 x 10 ms, i.e. sigma 2 ms."""
    s = sess(np.full(400, 10.0))
    base = D1Only().score([s])
    cnt = s7.flip_counts(D1Only(), [s], 99, base)
    assert math.sqrt(2) * s7.SIGMAS[21] <= s7.margin_sd(cnt["D1"][0]) <= math.sqrt(2) * s7.SIGMAS[22]   # sigma 1.90 ms
    none = s7.flip_counts(D1Only(), [None], 99, D1Only().score([None]))
    assert s7.margin_sd(none["D1"][0]) is None


# ---------------------------------------------------------------- constants against the conditions


def test_constants_match_conditions():
    s7.check_constants(COND)
    cap = COND["captures"]
    for ms in sp.K_CHROME_MS:
        assert f" {ms}" in cap["K_chrome"] or f"{ms}," in cap["K_chrome"]
    assert "0, 1, 2, 3, 5, 18, 50, 100 and 200 ms" in cap["firefox"]["series"]
    idx = COND["criterion"]["series_indices"]
    for name in ("MC1", "MC2", "MC3", "MC5", "F_M0", "F_M2_8B", "F_M2_14B"):
        assert f"{s7.SERIES_IDX[name]} {name}" in idx
    assert "(H0, the unperturbed side of persons, 16)" in COND["criterion"]["cell_prediction"]
    assert "theta = 1" in COND["criterion"]["statement"] and s7.THETA == 1.0
    assert "at least 5 of the 10 repeats" in COND["criterion"]["margin_sd"]
    assert s7.SIGMAS[0] == 0.05 and s7.SIGMAS[-1] == pytest.approx(60.89, abs=0.01)
    assert (s7.FLIP_MIN, s7.REPEATS) == (5, 10)
    d6 = COND["detectors_added"]["D6"]
    assert "1000 presses starting at press 0, 700" in d6["windows"] and "fewer than 100 presses" in d6["windows"]
    assert "random_state 6" in d6["classifier"] and "random_state 28" in COND["detectors_added"]["D5"]["classifier"]
    assert "[20261115, 7, 1]" in cap["order"] and list(sp.RUNLIST_KEY) == [20261115, 7, 1]


def test_v7_code_is_a6fa5ca():
    s7.code_check()


def test_group_blocks_is_a_permutation():
    b = sp.group_blocks()
    assert sorted(b.values()) == list(range(1, 7)) and set(b) == set(sp.GROUPS)


# ---------------------------------------------------------------- the plan maker


def write_runlog(root: Path, items: dict[str, str]) -> None:
    rows = [{"type": "header", "key": run7.RUN_KEY, "block": None, "runs_sha256": "x"}]
    rows += [{"type": "item", "item": k, "result": v} for k, v in items.items()]
    rows += [{"type": "footer"}]
    (root / "runlog.ndjson").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def fake_v7(tmp: Path, m0_gap: float = 55.0, no_plan: tuple[str, ...] = ()) -> tuple[Path, Path]:
    plan, cap = tmp / "plan", tmp / "cap"
    (plan / "events" / "H").mkdir(parents=True)
    cap.mkdir()
    shas = {}
    for i in range(sp.N_F_H1):
        f = plan / "events" / "H" / f"h{i:02d}.ndjson"
        ev.write(f, ev.make([0.0, 120.0, 300.0], ["a", " ", "b"]), "H1", "test")
        shas[f"events/H/h{i:02d}.ndjson"] = hashlib.sha256(f.read_bytes()).hexdigest()
    (plan / "manifest.json").write_text(json.dumps({"files_sha256": shas}), encoding="utf-8")
    items = {}
    for s_ in ("M0", "M2_8B", "M2_14B"):
        for t in range(run7.N_TEXTS):
            name = run7.item_name(s_, t)
            if name in no_plan:
                items[name] = "no_plan"
                continue
            items[name] = "done"
            d = cap / name
            d.mkdir(parents=True)
            gap = m0_gap if s_ == "M0" else 1.0
            ev.write(d / "events.ndjson", ev.make([k * gap for k in range(4)], ["H", "i", ".", "Enter"]), s_, "t")
    write_runlog(cap, items)
    return plan, cap


def test_plan_build_counts_blocks_and_copies(tmp_path):
    plan, cap = fake_v7(tmp_path, no_plan=("M0/t03", "M2_14B/t07"))
    man, runs = sp.build(plan, cap, tmp_path / "out")
    items = json.loads(runs)
    per = man["items_per_series"]
    assert per["K_chrome"] == 5 and per["K_chrome_nohook"] == 5 and per["F_K"] == 9 and per["F_K_nohook"] == 5
    assert all(per[f"MC{g}"] == 39 for g in sp.MC_MS) and per["F_M0"] == 39
    assert per["F_M2_8B"] == 40 and per["F_M2_14B"] == 39 and per["F_H1"] == 10
    assert man["v7_without_plan"] == {"M0": 1, "M2_14B": 1}
    blocks = sp.group_blocks()
    for it in items:
        s_ = it["name"].split("/")[0]
        group = {"K_chrome": "K_chrome", "K_chrome_nohook": "K_chrome", "F_K": "F_K", "F_K_nohook": "F_K",
                 "F_M0": "F_M0", "F_M2_8B": "F_M2", "F_M2_14B": "F_M2", "F_H1": "F_H1"}.get(s_, "MC_chrome")
        assert it["block"] == blocks[group]
        assert ("firefox" in it["args"]) == s_.startswith("F_")
        assert ("--no-hook" in it["args"]) == s_.endswith("nohook")
    src = (cap / "M2_8B" / "t05" / "events.ndjson").read_bytes()
    assert (tmp_path / "out" / "events" / "F_M2_8B" / "t05.ndjson").read_bytes() == src
    mc = run7.read_events(tmp_path / "out" / "events" / "MC" / "g2" / "t00.ndjson")
    assert np.allclose(np.diff(mc.times), 2.0) and mc.keys == ["H", "i", ".", "Enter"]
    k0 = run7.read_events(tmp_path / "out" / "events" / "K" / "c000.ndjson")
    assert k0.times.size == 1000 and np.all(k0.times == 0.0)


def test_plan_stops_on_a_wrong_m0_or_h(tmp_path):
    plan, cap = fake_v7(tmp_path / "a", m0_gap=50.0)
    with pytest.raises(sp.PlanStop, match="not a constant 55"):
        sp.build(plan, cap, tmp_path / "a" / "out")
    plan, cap = fake_v7(tmp_path / "b")
    f = plan / "events" / "H" / "h03.ndjson"
    f.write_bytes(f.read_bytes() + b"\n")
    with pytest.raises(sp.PlanStop, match="does not match v7's plan manifest"):
        sp.build(plan, cap, tmp_path / "b" / "out")


def test_plan_stops_on_a_missing_plan_file(tmp_path):
    plan, cap = fake_v7(tmp_path)
    (cap / "M2_8B" / "t01" / "events.ndjson").unlink()
    with pytest.raises(sp.PlanStop):
        sp.build(plan, cap, tmp_path / "out")


# ---------------------------------------------------------------- v7s's runlogs


def test_read_runlogs_v7s(tmp_path):
    for b in range(1, 7):
        rows = [{"type": "header", "key": list(sp.RUNLIST_KEY), "block": b, "runs_sha256": "r"},
                {"type": "item", "item": f"X/{b}", "result": "done"}, {"type": "footer"}]
        (tmp_path / f"runlog.b{b}.ndjson").write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert len(s7.read_runlogs(tmp_path)) == 6
    f = tmp_path / "runlog.b2.ndjson"
    f.write_text(f.read_text().replace('"key": [20261115, 7, 1]', '"key": [20261031, 7, 6]'))
    with pytest.raises(s7.RunStop):
        s7.read_runlogs(tmp_path)


# ---------------------------------------------------------------- Firefox's launch


def test_firefox_args_and_user_js():
    from capture import browser as br
    b = object.__new__(br.Browser)
    b.name, b.url, b.incognito, b.exe, b.profile = "firefox", "http://127.0.0.1:1/", False, Path("ff.exe"), Path("P")
    assert b.args() == ["ff.exe", "-profile", "P", "-no-remote", "-kiosk", "http://127.0.0.1:1/"]
    b.incognito = True
    with pytest.raises(ValueError):
        b.args()
    js = br.user_js()
    assert 'user_pref("network.proxy.no_proxies_on", "127.0.0.1");' in js
    assert "reduceTimerPrecision" not in js and "resistFingerprinting" not in js   # the timer prefs untouched
    c = object.__new__(br.Browser)
    c.name, c.url, c.incognito, c.exe, c.profile = "chrome", "u", False, Path("c.exe"), Path("P")
    assert c.args()[-1] == "--app=u"                      # Chrome unchanged


def test_criterion_end_to_end_on_synthetic_series():
    """Constant plans recorded with Gaussian error of a known SD: the criterion predicts D1's flips at 1 ms (error SD
    0.5 ms, delta far above 0.269) and none at 5 ms on the page; the cells and the verdict are built for every series."""
    rng = np.random.default_rng(7)
    n_s, n_p = 36, 120

    def series(name, gap, rec_sd, browser="chrome", margins_of=None):
        base = [sess(np.full(n_p - 1, float(gap))) for _ in range(n_s)]
        rec = {la: [s7.Sess(np.sort(b.times + rng.normal(0, rec_sd if la == "page" else 0.01, b.times.size)),
                            b.kinds) for b in base] for la in run7.LAYERS}
        return s7.Series(name, base, rec, ["captured"] * n_s, margins_of or name, browser,
                         [s7.median_gap(b) for b in base])

    gaps = {"M0": 55, "M1_8B": 20, "M1_14B": 20, "M2_8B": 30, "M2_14B": 30, "M3a": 150, "M3b": 150,
            "MC1": 1, "MC2": 2, "MC3": 3, "MC5": 5}
    ser = {k: series(k, g, 0.5 / math.sqrt(2)) for k, g in gaps.items()}
    ser["M1_replay_8B"] = series("M1_replay_8B", 20, 0.3, margins_of="M1_8B")
    ser["M1_replay_14B"] = series("M1_replay_14B", 20, 0.3, margins_of="M1_14B")
    for f_, src in (("F_M0", "M0"), ("F_M2_8B", "M2_8B"), ("F_M2_14B", "M2_14B")):
        ser[f_] = series(f_, gaps[src], 0.5, browser="firefox", margins_of=src)
    sc = D1Only()
    scores = {k: {"base": sc.score(se.base), **{la: sc.score(se.rec[la]) for la in run7.LAYERS}}
              for k, se in ser.items()}
    sources = {k: se.base for k, se in ser.items() if se.margins_of == k}
    margins, counts = s7.all_margins(sc, sources)
    tab = {0.0: 0.5, 1.0: 0.5, 5.0: 0.5, 200.0: 0.5}
    tables = {f"{b}/{la}": (tab if la == "page" else {0.0: 0.01, 200.0: 0.01})
              for b in ("chrome", "firefox") for la in run7.LAYERS}
    valid = {d: d in ("D1",) for d in s7.DETECTORS}
    crit = s7.criterion_cells(ser, scores, margins, counts, tables, valid, 1.0)
    c1 = crit["cells"]["MC1/page/D1"]
    c5 = crit["cells"]["MC5/page/D1"]
    assert c1["predicted"] == "responds" and c1["observed"] == "responds"
    assert c5["predicted"] == "not changed" and c5["observed"] == "not changed"
    assert crit["cells"]["MC1/hook/D1"]["predicted"] == "not changed"
    v = s7.verdicts(crit, valid)
    assert set(v) == {"all", "D1"} and v["all"]["verdict"] == "held"
    jt = s7.jitter_table(counts, sources)
    assert jt["MC5/D1"]["first_sigma"]["0.10"] <= 2.0



# ---------------------------------------------------------------- after the code review (decision 3)


def test_failed_recordings_leave_both_readings():
    """R1: a failed capture (no recording) has a plan and a prediction but enters neither reading."""
    n = 40
    base = [sess(np.full(99, 1.0)) for _ in range(n)]
    rec = [s7.Sess(np.sort(b.times + np.random.default_rng(i).normal(0, 0.5, b.times.size)), b.kinds)
           if i >= 3 else None for i, b in enumerate(base)]
    se = s7.Series("MC1", base, {"hook": rec, "page": rec}, ["failed"] * 3 + ["captured"] * (n - 3), "MC1",
                   "chrome", [1.0] * n)
    sc = D1Only()
    scores = {"MC1": {"base": sc.score(base), **{la: sc.score(rec) for la in run7.LAYERS}}}
    cnt = np.zeros((n, len(s7.SIGMAS)), dtype=int)
    cnt[:, 10:] = 10                                     # every session flips from sigma index 10
    counts = {"MC1": {d: cnt for d in s7.DETECTORS}}
    margins = {"MC1": {d: [s7.margin_sd(r) for r in cnt] for d in s7.DETECTORS}}
    tab = {f"chrome/{la}": {0.0: 5.0, 200.0: 5.0} for la in run7.LAYERS}
    old = s7.MACHINE_SERIES_ALL
    s7.MACHINE_SERIES_ALL = ("MC1",)
    try:
        crit = s7.criterion_cells({"MC1": se}, scores, margins, counts, tab, {d: True for d in s7.DETECTORS}, 1.0)
    finally:
        s7.MACHINE_SERIES_ALL = old
    c = crit["cells"]["MC1/page/D1"]
    assert c["observed_bc"][2] == c["predicted_bc"][2] == n - 3 and c["left_out_other"] == 3
    assert c["left_out_no_plan"] == 0


def test_check_known_stops_on_a_missing_point():
    full = {f"{b}/{la}": {x: 0.1 for x in pts} for b, pts in s7.KNOWN_POINTS.items() for la in run7.LAYERS}
    s7.check_known(full)
    full["firefox/page"].pop(1.0)
    with pytest.raises(s7.RunStop, match="known gaps \\[1.0\\] missing"):
        s7.check_known(full)


def cap_on(browser, version, status="captured"):
    c = run7.Capture("X", 0, status)
    c.inject = {"browser": browser, "browser_version": version}
    return c


def test_check_browsers():
    assert s7.check_browsers([cap_on("chrome", s7.CHROME_VERSION)] * 2, "chrome") == s7.CHROME_VERSION
    assert s7.check_browsers([cap_on("firefox", "160.0.0.1"), cap_on("x", "y", "failed")], "firefox") == "160.0.0.1"
    for caps, b in (([cap_on("chrome", "156.0.8078.74")], "chrome"), ([cap_on("chrome", s7.CHROME_VERSION)], "firefox"),
                    ([cap_on("firefox", "160.0.0.1"), cap_on("firefox", "160.0.1.1")], "firefox")):
        with pytest.raises(s7.RunStop):
            s7.check_browsers(caps, b)


class P:
    def __init__(self, plan):
        self.plan = plan


def test_expected_items_follow_v7s_plans():
    v7caps = {"M0": [P(object()) if t != 4 else P(None) for t in range(run7.N_TEXTS)],
              "M2_8B": [P(object())] * run7.N_TEXTS, "M2_14B": [P(None)] * run7.N_TEXTS}
    names = s7.expected_items(v7caps)
    assert "MC1/t04" not in names and "F_M0/t04" not in names and "MC5/t03" in names
    assert "F_M2_8B/t39" in names and not any(n.startswith("F_M2_14B") for n in names)
    assert len([n for n in names if n.startswith(("K_", "F_K"))]) == 24 and "F_H1/h09" in names


def test_plan_copy_must_be_its_own_source(tmp_path, monkeypatch):
    plan, cap = fake_v7(tmp_path)
    (cap / "M2_8B" / "t05" / "events.ndjson").write_bytes((cap / "M2_8B" / "t05" / "events.ndjson").read_bytes())
    real = sp.shutil.copyfile

    def swapped(src, dst):   # t03's plan put in t05's place
        src = Path(str(src).replace("t05", "t03"))
        return real(src, dst)
    m2 = cap / "M2_8B"
    (m2 / "t03" / "events.ndjson").write_bytes((m2 / "t03" / "events.ndjson").read_bytes().replace(b'"t"', b'"u"'))
    monkeypatch.setattr(sp.shutil, "copyfile", swapped)
    with pytest.raises(sp.PlanStop, match="the copy differs from its source"):
        sp.build(plan, cap, tmp_path / "out")


def test_firefox_user_js_keeps_alt_off_the_menu():
    from capture import browser as br
    assert 'user_pref("ui.key.menuAccessKeyFocuses", false);' in br.user_js()


def test_firefox_user_js_takes_its_window_as_active():
    from capture import browser as br
    js = br.user_js()
    assert 'user_pref("focusmanager.testmode", true);' in js     # R19
    # no pref of time, events or fingerprinting resistance (the page runs at Firefox's default precision)
    touched = [k for k in br.FIREFOX_PREFS if k.startswith(("privacy.", "dom.event", "dom.timer"))
               or "timer" in k.lower() or "precision" in k.lower() or "fingerprint" in k.lower()]
    assert touched == []
    cond = json.loads((ROOT / "works" / "conditions" / "prototype_v7s.json").read_text(encoding="utf-8"))
    assert any(r.startswith("R19 ") and "focusmanager.testmode" in r for r in cond["amendments_before_any_value"])
    assert "R19" in cond["fixed_before_running"]


def test_firefox_user_js_keeps_the_welcome_modal_closed():
    from capture import browser as br
    js = br.user_js()
    assert 'user_pref("browser.preonboarding.enabled", false);' in js                          # R20
    assert 'user_pref("termsofuse.bypassNotification", true);' in js
    assert 'user_pref("datareporting.policy.dataSubmissionPolicyBypassNotification", true);' in js
    cond = json.loads((ROOT / "works" / "conditions" / "prototype_v7s.json").read_text(encoding="utf-8"))
    assert any(r.startswith("R20 ") and "browser.preonboarding.enabled" in r for r in cond["amendments_before_any_value"])
    assert "R20" in cond["fixed_before_running"]


def test_user_js_recorded_for_firefox_and_not_written_for_chrome(monkeypatch, tmp_path):
    import hashlib
    from capture import browser as br
    monkeypatch.setattr(br, "file_version", lambda exe: "0.0.0.0")
    monkeypatch.setattr(br.tempfile, "tempdir", str(tmp_path))
    ff = br.Browser("firefox", "http://127.0.0.1:1/")
    assert br.user_js_sha256(ff.profile) == hashlib.sha256(br.user_js().encode("utf-8")).hexdigest()
    ch = br.Browser("chrome", "http://127.0.0.1:1/")
    assert not (ch.profile / "user.js").exists()                 # Chrome gets no user.js, so no testmode
    src = (ROOT / "capture" / "session.py").read_text(encoding="utf-8")
    assert 'if a.browser == "firefox":' in src and 'meta["browser"]["user_js_sha256"] = br.user_js_sha256(b.profile)' in src


# ---------------------------------------------------------------- after the independent review (decision 4)


def test_d5_d6_label_direction():
    """'machine' is the synthetic class: held-out synthetic sessions of each detector's own generator score higher
    than held-out humans (a flipped label would put them lower)."""
    rng = np.random.default_rng(11)

    def structured(n):   # the gap ending a space is long, others short: a structure iid draws do not keep
        kinds = list(rng.choice(["alnum", "alnum", "alnum", "space"], n))
        g = np.array([rng.normal(600, 30) if k == "space" else rng.normal(120, 10) for k in kinds[1:]])
        return sess(g, kinds)
    train, test = [structured(300) for _ in range(60)], [structured(300) for _ in range(8)]   # LightGBM: >= 20 rows a leaf
    d5 = s7.train_d5(train, np.random.default_rng(1))
    d6 = s7.train_d6(train, np.random.default_rng(2))
    r = np.random.default_rng(3)
    syn5 = [s7.Sess(np.concatenate([[0.0], np.cumsum(d5.kde.draw(s.gaps.size, r))]), s.kinds) for s in test]
    syn6 = [s7.at_p_forge(s, d6.atp, r) for s in test]
    assert np.mean(d5.proba(syn5)) > np.mean(d5.proba(test)) + 0.05
    assert np.mean(d6.proba(syn6)) > np.mean(d6.proba(test)) + 0.05


def test_verdicts_count_only_tested_cells_of_valid_detectors():
    agree = {"predicted": "responds", "observed": "responds"}
    disagree = {"predicted": "not changed", "observed": "responds"}
    crit = {"cells": {"MC1/page/D1": {**agree, "tested": True, "valid": True},
                      "M2_8B/page/D1": {**disagree, "tested": False, "valid": True},
                      "MC1/page/D4": {**disagree, "tested": True, "valid": False}},
            "sessions": {"MC1/page/D1": [(True, True)], "M2_8B/page/D1": [(False, True)] * 9,
                         "MC1/page/D4": [(False, True)] * 9}}
    v = s7.verdicts(crit, {d: d != "D4" for d in s7.DETECTORS})
    assert v["all"]["verdict"] == "held" and v["all"]["sessions"] == 1 and "D4" not in v


def test_human_cells_direction_and_n():
    rng = np.random.default_rng(5)
    base = [human(300, rng) for _ in range(40)]
    se = s7.Series("H1_idle", base, {la: base for la in run7.LAYERS}, ["captured"] * 40, "H0", "chrome",
                   [s7.median_gap(b) for b in base])
    sc = D1Only()
    scores = {"H1_idle": {"base": sc.score(base), **{la: sc.score(base) for la in run7.LAYERS}}}
    cnt = np.full((40, len(s7.SIGMAS)), 10, dtype=int)   # every person flips at every sigma: probability 1
    margins = {"H0": {d: [s7.margin_sd(r) for r in cnt] for d in s7.DETECTORS}}
    tab = {f"chrome/{la}": {0.0: 1.0, 200.0: 1.0} for la in run7.LAYERS}
    old = s7.HUMAN_SERIES
    s7.HUMAN_SERIES = ("H1_idle",)
    try:
        out = s7.human_cells({"H1_idle": se}, scores, margins, {"H0": {d: cnt for d in s7.DETECTORS}}, tab, 1.0)
    finally:
        s7.HUMAN_SERIES = old
    row = out["H1_idle/page/D1"]
    assert row["predicted_bc"] == [40, 0, 40] and row["observed_bc"] == [0, 0, 40]   # human -> machine is b
    assert row["predicted"] == "responds" and row["observed"] == "not changed"
    se.status = ["captured"] * 30 + ["no_plan"] * 10
    s7.HUMAN_SERIES = ("H1_idle",)
    try:
        out = s7.human_cells({"H1_idle": se}, scores, margins, {"H0": {d: cnt for d in s7.DETECTORS}}, tab, 1.0)
    finally:
        s7.HUMAN_SERIES = old
    assert "observed" not in out["H1_idle/page/D1"]                                   # n = 30 < 35: no reading


def test_flip_prob_starts_from_zero():
    row = np.zeros(len(s7.SIGMAS), dtype=int)
    row[0] = 10
    assert s7.flip_prob(row, math.sqrt(2) * 0.025) == pytest.approx(0.5)


def cap_with(kinds_plan, kinds_layer):
    c = run7.Capture("X", 0, "captured")
    c.plan = run7.Plan(np.arange(len(kinds_plan), dtype=float), list(kinds_plan), ["a"] * len(kinds_plan))
    c.layers = {la: run7.Layer(np.arange(len(kinds_layer), dtype=float), list(kinds_layer), 0)
                for la in run7.LAYERS}
    return c


def test_check_kinds_stops_on_lost_presses():
    plan = ["alnum", "space"] * 50
    ok = s7.check_kinds("X", [cap_with(plan, plan[:-1])])           # 1 of 100 lost: 0.01
    assert ok["page"]["edit_share"] == pytest.approx(0.01)
    with pytest.raises(s7.RunStop, match="lost or changed in kind"):
        s7.check_kinds("X", [cap_with(plan, [k for k in plan if k != "alnum"][:-3] + ["space"] * 3)])


def test_read_series_stops_on_an_mc_plan_not_m0s(monkeypatch):
    n = run7.N_TEXTS
    none = lambda s_, k: [run7.Capture(s_, i, "no_plan") for i in range(k)]   # noqa: E731
    v7caps = {s_: none(s_, n) for s_ in run7.MACHINE}
    v7caps["H1_idle"] = none("H1_idle", run7.N_H)
    m0 = run7.Capture("M0", 0, "captured")
    m0.plan = run7.Plan(np.arange(4) * 55.0, ["alnum"] * 4, ["a", "b", "c", "d"])
    v7caps["M0"][0] = m0
    runlog = {nm: "done" for nm in s7.expected_items(v7caps)}

    def fake(root, rl, series, units, fmt):
        out = []
        for u in units:
            c = run7.Capture(series, u, "no_plan")
            if series.startswith("MC") and u == 0:
                c = run7.Capture(series, 0, "captured")
                g = float(series[2:])
                c.plan = run7.Plan(np.arange(4) * g, ["alnum"] * 4, ["a", "b", "c", "x"])   # one key differs
            out.append(c)
        return out
    monkeypatch.setattr(s7, "unit_caps", fake)
    with pytest.raises(s7.RunStop, match="not M0's keys"):
        s7.read_series(v7caps, Path("."), runlog, [sess(np.ones(3))] * run7.N_H)
    with pytest.raises(s7.RunStop, match="items outside the plan"):
        s7.read_series(v7caps, Path("."), {**runlog, "MC1/t07": "done"}, [sess(np.ones(3))] * run7.N_H)



# ---------------------------------------------------------------- 2nd cross-review (decision 5)


def test_one_firefox_version():
    s7.check_one_firefox("160.0.0.1", "160.0.0.1")
    for k, v in (("160.0.0.1", "160.0.1.1"), (None, "160.0.0.1"), ("160.0.0.1", None)):
        with pytest.raises(s7.RunStop, match="not one version"):
            s7.check_one_firefox(k, v)


def test_grid_share_summary_needs_five():
    assert s7.grid_share_summary([1.0, 0.98, None, 1.0]) is None
    assert s7.grid_share_summary([1.0, 0.98, 1.0, 1.0, 0.995]) == pytest.approx(0.8)


def test_check_kinds_per_capture_for_known_gaps():
    plan = ["alnum"] * 100
    caps = [cap_with(plan, plan) for _ in range(13)] + [cap_with(plan, plan[:80])]   # one capture loses 20%
    assert s7.check_kinds("K", caps)["page"]["edit_share"] < 0.02                     # pooled: passes
    with pytest.raises(s7.RunStop):
        s7.check_kinds("K", caps, per_capture=True)


def test_read_series_runs_r8(monkeypatch):
    n = run7.N_TEXTS
    none = lambda s_, k: [run7.Capture(s_, i, "no_plan") for i in range(k)]   # noqa: E731
    v7caps = {s_: none(s_, n) for s_ in run7.MACHINE}
    v7caps["H1_idle"] = none("H1_idle", run7.N_H)
    v7caps["M0"][0] = cap_with(["alnum"] * 100, ["alnum"] * 100)
    v7caps["M0"][0].plan = run7.Plan(np.arange(100) * 55.0, ["alnum"] * 100, ["a"] * 100)
    runlog = {nm: "done" for nm in s7.expected_items(v7caps)}

    def fake(root, rl, series, units, fmt):
        out = []
        for u in units:
            c = run7.Capture(series, u, "no_plan")
            if series.startswith("MC") and u == 0:
                c = cap_with(["alnum"] * 100, ["alnum"] * 90)                # 10% lost on both layers
                c.plan = run7.Plan(np.arange(100) * float(series[2:]), ["alnum"] * 100, ["a"] * 100)
                c.inject = {"browser": "chrome", "browser_version": s7.CHROME_VERSION}
            out.append(c)
        return out
    monkeypatch.setattr(s7, "unit_caps", fake)
    with pytest.raises(s7.RunStop, match="lost or changed in kind"):
        s7.read_series(v7caps, Path("."), runlog, [sess(np.ones(3))] * run7.N_H, {})
