"""Tests of works/prototype_v7.py on synthetic series only (no KUPA, no capture)."""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "works"))
import prototype_v7 as v7  # noqa: E402

C7 = json.loads((ROOT / "works/conditions/prototype_v7.json").read_text())


def human_like(n: int, rng: np.random.Generator) -> np.ndarray:
    """A lognormal typing series with a person-specific speed, lag-1 dependence (AR(1) on ln gap) and occasional
    rests (synthetic; delta far above 0.269). An i.i.d. series would be indistinguishable from D3's resampling
    generators by construction."""
    mu = math.log(rng.uniform(120, 300))
    e = np.empty(n)
    e[0] = rng.normal(0, 0.5)
    for i in range(1, n):
        e[i] = 0.8 * e[i - 1] + rng.normal(0, 0.3)
    g = np.exp(mu + e)
    rest = rng.random(n) < 0.05
    g[rest] = rng.uniform(800, 3000, rest.sum())
    return g


def humans(k: int, seed: int = 0, lo: int = 300, hi: int = 600) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [human_like(int(rng.integers(lo, hi)), rng) for _ in range(k)]


# ---------------------------------------------------------------- constants against the conditions


def test_constants_follow_the_conditions():
    d = C7["detectors_C2"]
    assert "delta > 0.269" in d["D1"] and v7.DELTA_T == 0.269
    assert "10 x the mean" in d["D1"] and v7.TRIM_FACTOR == 10.0
    assert "Uniform(30, 80)" in d["D2"] and v7.UNIFORM_MS == (30.0, 80.0)
    assert "seed 42" in d["D2"] and "100 trees" in d["D2"] and (v7.D2_SEED, v7.D2_TREES) == (42, 100)
    assert "seed 123" in d["D3"] and "300 trees" in d["D3"] and (v7.D3_SEED, v7.D3_TREES) == (123, 300)
    assert "windows of 70" in d["D3"] and v7.WINDOW == 70
    assert "50-bin histogram on [0, 2000] ms" in d["D2"] and (v7.HIST_BINS, v7.HIST_MAX_MS) == (50, 2000.0)
    assert "> 500 ms" in d["D2"] and "< 150 ms" in d["D2"]
    assert "below 0.90" in d["validity_gate"] and "exceeds 0.10" in d["validity_gate"]
    assert "[-0.10, 0.10]" in d["headline"] and v7.MARGIN == 0.10
    assert "8 empirical-quantile bins" in C7["series"]["M3b"] and v7.N_BINS == 8
    assert "2000 bootstrap" in C7["stages_C1"] and v7.N_BOOT == 2000
    assert (v7.PAUSE_MS, v7.BURST_MS) == (500.0, 150.0)
    assert (v7.GATE_AUC, v7.GATE_HUMAN_SHARE) == (0.90, 0.10)
    assert "pooled 1st and 99th percentiles" in d["D3"]


# ---------------------------------------------------------------- cleaning


def test_clean_times_sorts_and_drops_equal():
    t, idx, dropped = v7.clean_times(np.array([10.0, 5.0, 10.0, 20.0]))
    assert list(t) == [0.0, 5.0, 15.0] and dropped == 1 and list(idx) == [1, 0, 3]
    with pytest.raises(ValueError):
        v7.clean_times(np.array([0.0, np.nan]))


# ---------------------------------------------------------------- D1 and D2


def test_delta_by_hand_and_trim():
    g = np.array([100.0, 200.0, 300.0])
    assert v7.delta(g) == pytest.approx(np.std(g) / np.mean(g))
    # one huge gap among few stays (mean of all = (9*100 + 100000)/10 = 10090; 100000 <= 100900)
    g2 = np.array([100.0] * 9 + [100000.0])
    assert v7.trimmed(g2).size == 10
    g3 = np.array([100.0] * 99 + [100000.0])  # mean 1099; 100000 > 10990 -> removed
    assert v7.trimmed(g3).size == 99 and v7.delta(g3) == 0.0


def test_d1_positive_and_negative_controls():
    rng = np.random.default_rng(1)
    assert v7.d1_machine(rng.uniform(30, 80, 500))          # delta about 0.17 -> machine
    assert not v7.d1_machine(human_like(500, rng))           # human
    assert v7.d1_machine(np.full(100, 55.0))                 # constant -> delta 0


def test_mean_burst_run_and_entropy():
    g = np.array([100, 100, 200, 100, 600, 100, 100, 100], dtype=float)
    assert v7.mean_burst_run(g) == pytest.approx((2 + 1 + 3) / 3)
    assert v7.mean_burst_run(np.array([200.0, 300.0])) == 0.0
    h, out = v7.hist_entropy(np.array([10.0, 10.0, 50.0, 50.0, 3000.0]))
    assert out == 1 and h == pytest.approx(math.log(2))      # 10 and 50 fall in two bins of width 40
    assert v7.hist_entropy(np.array([5000.0]))[0] == 0.0


def test_condrey_features_order_and_values():
    g = np.array([100.0, 600.0, 100.0, 120.0, 2100.0] + [100.0] * 95 + [1e6])
    f, out = v7.condrey_features(g)
    _, per = v7.feature_matrix([g, np.r_[g[:-1], 1e6], np.full(50, 100.0)])
    assert list(per) == [1, 1, 0]
    g = g[:-1]                                   # the 1e6 gap is trimmed; 2100 stays and is left out of the histogram
    assert f.shape == (7,) and out == 1
    assert f[5] == pytest.approx(v7.hist_entropy(g)[0]) and f[5] > 0
    assert f[0] == pytest.approx(g.std() / g.mean()) and f[1] == pytest.approx(g.mean())
    assert f[2] == pytest.approx(g.var()) and f[3] == pytest.approx(2 / 100)
    assert f[4] == pytest.approx(v7.mean_burst_run(g)) and f[4] == pytest.approx((1 + 2 + 95) / 3) and f[6] == pytest.approx(np.diff(g).std())


def test_d2_separates_uniform_from_humans():
    h = humans(30)
    d2 = v7.train_d2(h, np.random.default_rng(42))
    test_h = humans(10, seed=5)
    p = d2.proba(test_h + v7.uniform_sessions([400] * 10, np.random.default_rng(7)))
    for name in ("D2_lr", "D2_rf"):
        assert (p[name][:10] < 0.5).all() and (p[name][10:] > 0.5).all()


def test_d2_is_reproducible():
    h = humans(12)
    a = v7.train_d2(h, np.random.default_rng(42)).proba(h[:3])
    b = v7.train_d2(h, np.random.default_rng(42)).proba(h[:3])
    assert all(np.array_equal(a[k], b[k]) for k in a)


# ---------------------------------------------------------------- D3


def test_windows_shape_and_j():
    g = np.arange(150, dtype=float)
    w = v7.windows_of(g)
    assert w.shape == (2, 140)
    assert w[0, 70] == 0.0 and (w[0, 71:] == 1.0).all() and w[1, 0] == 70.0
    wd = v7.windows_of(np.arange(70, 0, -1, dtype=float))
    assert (wd[0, 71:] == 1.0).all()                 # |gap_t - gap_{t-1}|, not the signed difference
    assert v7.windows_of(np.arange(69, dtype=float)).shape == (0, 140)


def test_stack_windows_weights_sum_to_one_per_session():
    x, w, sid = v7.stack_windows([np.ones(140), np.ones(70), np.ones(10), np.ones(210)])
    assert len(x) == 6
    for s in (0, 1, 3):
        assert w[sid == s].sum() == pytest.approx(1.0)
    assert 2 not in sid


def test_generators_fitted_as_the_conditions_say():
    h = humans(9)
    gen = v7.fit_generators(h)
    pooled = np.concatenate(h)
    assert gen.lo == np.percentile(pooled, 1) and gen.hi == np.percentile(pooled, 99)


def test_d3_synthetic_windows_in_thirds_and_per_session():
    rng = np.random.default_rng(5)
    h = [human_like(n, rng) for n in (75, 700, 1500, 140, 30, 2200)]   # unequal lengths and one below WINDOW
    gen = v7.fit_generators(h)
    syn, lab = v7.d3_synthetic(h, gen, np.random.default_rng(0), with_labels=True)
    nwin = [g.size // 70 for g in h]
    assert [len(x) for x in lab] == nwin
    assert [s.size // 70 for s in syn] == nwin and syn[4].size == 30
    counts = np.bincount(np.concatenate(lab), minlength=3)
    assert counts.max() - counts.min() <= 1 and counts.sum() == sum(nwin)
    # windows from the uniform generator lie in [lo, hi]; most others do not all
    w = np.concatenate([v7.windows_of(s)[:, :70] for s in syn if s.size >= 70])
    flat = np.concatenate(lab)
    uni = w[flat == 0]
    assert ((uni >= gen.lo) & (uni <= gen.hi)).all()


def test_condbins_keeps_lag_one_dependence():
    rng = np.random.default_rng(3)
    # alternating slow/fast with strong lag-1 negative dependence
    s = np.where(np.arange(4000) % 2 == 0, rng.uniform(400, 500, 4000), rng.uniform(50, 60, 4000))
    cb = v7.CondBins([s])
    out, fb = cb.draw(4000, rng)
    r_orig = np.corrcoef(s[:-1], s[1:])[0, 1]
    r_cb = np.corrcoef(out[:-1], out[1:])[0, 1]
    r_iid = np.corrcoef(*(lambda x: (x[:-1], x[1:]))(rng.choice(s, 4000)))[0, 1]
    assert r_orig < -0.9 and r_cb < -0.9 and abs(r_iid) < 0.1 and fb == 0


def test_condbins_empty_bin_falls_back_and_counts():
    cb = v7.CondBins([np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])])
    # the top bin holds only the last gap, which has no successor
    assert cb.successors[-1].size == 0
    out, fb = cb.draw(2000, np.random.default_rng(0))
    assert fb > 0 and set(out) <= set(range(1, 9))


def test_condbins_successors_do_not_cross_sessions():
    cb = v7.CondBins([np.array([1.0, 1000.0]), np.array([5.0, 6.0])])
    allsucc = np.concatenate(cb.successors)
    assert sorted(allsucc) == [6.0, 1000.0]


def test_train_d3_passes_session_weights(monkeypatch):
    seen = {}
    real = v7.RandomForestClassifier.fit

    def spy(self, x, y, sample_weight=None):
        seen["w"], seen["y"] = sample_weight, y
        return real(self, x, y, sample_weight=sample_weight)

    monkeypatch.setattr(v7.RandomForestClassifier, "fit", spy)
    h = humans(6, lo=150, hi=400)
    v7.train_d3(h, np.random.default_rng(123))
    w, y = seen["w"], seen["y"]
    assert w is not None
    assert w[y == 0].sum() == pytest.approx(len(h)) and w[y == 1].sum() == pytest.approx(len(h))
    assert (y == 0).sum() == (y == 1).sum() == sum(g.size // 70 for g in h)


def test_d3_separates_and_short_sessions_get_nan():
    h = humans(20)
    d3 = v7.train_d3(h, np.random.default_rng(123))
    gen = v7.fit_generators(h)
    uni = np.random.default_rng(2).uniform(gen.lo, gen.hi, 300)
    p = d3.proba([humans(1, seed=9)[0], uni, np.ones(50)])
    assert p[0] < 0.5 and p[1] > 0.5 and math.isnan(p[2])
    # a constant series lies outside the training generators: D3 need not call it 'machine' (seen 0.46 here); the
    # conditions predict D1, not D3, on M0
    assert 0.0 <= d3.proba([np.full(300, 55.0)])[0] <= 1.0


# ---------------------------------------------------------------- validity gate


def test_validity_gate_on_synthetic_humans():
    out = v7.validity_gate(humans(25, seed=11), key=[1, 2, 3])
    for name in ("D1", "D2_lr", "D2_rf", "D3"):
        assert out[name]["valid"], (name, out[name])
        assert out[name]["n_human"] == 25
    assert out["D3"]["no_score"] == 0


def test_validity_gate_never_trains_on_held_out_persons(monkeypatch):
    h = humans(15, seed=3)
    trained: list[set] = []
    real2, real3 = v7.train_d2, v7.train_d3

    def spy2(human, rng):
        trained.append({id(g) for g in human})
        return real2(human, rng)

    def spy3(human, rng):
        trained.append({id(g) for g in human})
        return real3(human, rng)

    monkeypatch.setattr(v7, "train_d2", spy2)
    monkeypatch.setattr(v7, "train_d3", spy3)
    v7.validity_gate(h, key=[1, 2, 3])
    fold = v7.person_folds(len(h), [1, 2, 3])
    for k in range(5):
        held = {id(g) for g, f in zip(h, fold) if f == k}
        for t in trained[2 * k: 2 * k + 2]:
            assert not (t & held) and len(t) == len(h) - len(held)


def test_validity_gate_fails_when_humans_look_like_machines():
    rng = np.random.default_rng(4)
    machine_like = [rng.uniform(40, 70, 400) for _ in range(25)]
    out = v7.validity_gate(machine_like, key=[1, 2, 3])
    assert not out["D1"]["valid"]           # D1 calls these humans 'machine'


def test_person_folds_cover_all():
    f = v7.person_folds(23, [20261031, 7, 8])
    assert sorted(np.bincount(f)) == [4, 4, 5, 5, 5]
    assert np.array_equal(f, v7.person_folds(23, [20261031, 7, 8]))


def test_rank_auc():
    assert v7.rank_auc(np.array([0.1, 0.2, 0.9, 0.8]), np.array([0, 0, 1, 1])) == 1.0
    assert v7.rank_auc(np.array([0.5, 0.5]), np.array([0, 1])) == 0.5
    assert v7.rank_auc(np.array([0.9, 0.1]), np.array([0, 1])) == 0.0
    with pytest.raises(ValueError):
        v7.rank_auc(np.array([np.nan, 1.0]), np.array([0, 1]))


# ---------------------------------------------------------------- intervals and the headline


def test_tango_reproduces_the_design_calculation():
    # conditions, detectors_C2.headline and decision 10 of .steering/20261102-v7-conditions
    lo, hi = v7.tango_interval(0, 0, 40)
    assert (round(lo, 3), round(hi, 3)) == (-0.088, 0.088)
    lo, hi = v7.tango_interval(1, 0, 40)
    assert (round(lo, 3), round(hi, 3)) == (-0.065, 0.129)
    lo, hi = v7.tango_interval(1, 1, 40)
    assert (round(lo, 3), round(hi, 3)) == (-0.108, 0.108)


def test_tango_is_symmetric_and_bounded():
    for b, c, n in ((3, 1, 40), (0, 7, 40), (10, 10, 20), (40, 0, 40)):
        lo, hi = v7.tango_interval(b, c, n)
        lo2, hi2 = v7.tango_interval(c, b, n)
        assert lo == pytest.approx(-hi2, abs=1e-9) and hi == pytest.approx(-lo2, abs=1e-9)
        assert -1 <= lo <= (b - c) / n <= hi <= 1
        assert abs(v7.tango_z(lo, b, c, n)) == pytest.approx(v7.Z95, abs=1e-6) or lo == -1.0


def test_tango_z_matches_wald_like_limit():
    # at delta = 0 the score statistic is McNemar's (b - c) / sqrt(b + c)
    assert v7.tango_z(0.0, 9, 4, 50) == pytest.approx(5 / math.sqrt(13))


def test_headline_readings():
    assert v7.headline(0, 0, 40)["reading"] == "not changed"
    assert v7.headline(1, 0, 40)["reading"] == "undecided"
    assert v7.headline(12, 0, 40)["reading"] == "responds"
    assert v7.headline(0, 12, 40)["reading"] == "responds"
    h = v7.headline(3, 2, 40)
    assert h["reading"] == "undecided" and (h["b"], h["c"]) == (3, 2)


def test_flips():
    b, c, n = v7.flips(np.array([0, 0, 1, 1, 0], dtype=bool), np.array([1, 0, 0, 1, 1], dtype=bool))
    assert (b, c, n) == (2, 1, 5)
    with pytest.raises(TypeError):
        v7.flips(np.array([np.nan, 0.0]), np.array([0.0, 0.0]))
    before, after, removed = v7.judged_pairs(np.array([np.nan, 0.2, 0.7, 0.4]), np.array([0.9, 0.6, 0.1, np.nan]))
    assert removed == 2 and v7.flips(before, after) == (1, 1, 2)


def test_across():
    r = {("D1", "hook"): "not changed", ("D1", "page"): "not changed",
         ("D2_lr", "hook"): "not changed", ("D2_lr", "page"): "responds",
         ("D3", "hook"): "not changed", ("D3", "page"): "not changed"}
    dets = ("D1", "D2_lr", "D3")
    a = v7.across(r, {"D1": True, "D2_lr": True, "D3": False}, detectors=dets)
    assert a["not_changed_on"] == {"D1": ["hook", "page"], "D2_lr": ["hook"]} and not a["general_not_moved"]
    assert a["responds"] == [["D2_lr", "page"]] and a["not_valid_here"] == ["D3"]
    only_d1 = {k: v for k, v in r.items() if k[0] in ("D1", "D3")}
    two = ("D1", "D3")
    assert v7.across(only_d1, {"D1": True, "D3": False}, detectors=two)["general_not_moved"]
    # an invalid detector does not make the statement; neither does no valid detector
    assert not v7.across(only_d1, {"D1": False, "D3": False}, detectors=two)["general_not_moved"]
    with pytest.raises(ValueError):                  # a missing layer cannot pass
        v7.across({("D1", "hook"): "not changed"}, {"D1": True}, detectors=("D1",))
    with pytest.raises(ValueError):                  # a valid detector without readings cannot pass
        v7.across({("D1", "hook"): "not changed", ("D1", "page"): "not changed"}, {"D1": True, "D3": True},
                  detectors=two)
    with pytest.raises(ValueError):                  # by default all four detectors must be judged
        v7.across({("D1", la): "not changed" for la in v7.LAYERS}, {"D1": True})


def test_clopper_pearson_closed_forms():
    lo, hi = v7.clopper_pearson(0, 40)
    assert lo == 0.0 and hi == pytest.approx(1 - 0.025 ** (1 / 40))
    lo, hi = v7.clopper_pearson(40, 40)
    assert hi == 1.0 and lo == pytest.approx(0.025 ** (1 / 40))
    # interior: the ends solve the binomial tail equations
    from scipy import stats
    lo, hi = v7.clopper_pearson(5, 20)
    assert stats.binom.sf(4, 20, lo) == pytest.approx(0.025) and stats.binom.cdf(5, 20, hi) == pytest.approx(0.025)


def test_cluster_bootstrap_share():
    machine = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=bool)
    cluster = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    lo, hi, k = v7.cluster_bootstrap_share(machine, cluster, [1])
    assert k == 4 and 0.0 <= lo < 0.5 < hi <= 1.0
    lo, hi, _ = v7.cluster_bootstrap_share(np.ones(8, dtype=bool), cluster, [1])
    assert lo == hi == 1.0
    # two clusters of 10 identical sessions: resampling clusters gives [0, 1]; resampling sessions would not
    lo, hi, k = v7.cluster_bootstrap_share(np.repeat([True, False], 10), np.repeat([0, 1], 10), [2])
    assert (lo, hi, k) == (0.0, 1.0, 2)
    with pytest.raises(TypeError):                   # a NaN must not become 'machine'
        v7.cluster_bootstrap_share(np.array([np.nan, 0.0]), np.array([0, 1]), [1])


def test_paired_prob_change():
    out = v7.paired_prob_change(np.array([0.1, 0.2, np.nan, 0.3]), np.array([0.2, 0.3, 0.5, 0.4]), [1])
    assert out["mean"] == pytest.approx(0.1) and out["n"] == 3 and out["removed"] == 1
    assert out["ci95"][0] == pytest.approx(0.1) and out["ci95"][1] == pytest.approx(0.1)
    d = np.arange(20, dtype=float) / 100
    out = v7.paired_prob_change(np.zeros(20), d, [3])
    m = d[np.random.default_rng([3]).integers(0, 20, (2000, 20))].mean(axis=1)
    assert out["ci95"] == [np.percentile(m, 2.5), np.percentile(m, 97.5)]
    with pytest.raises(ValueError):
        v7.paired_prob_change(np.zeros((20, 1)), d, [3])
    with pytest.raises(ValueError):
        v7.paired_prob_change(np.zeros((10, 2)), np.zeros((10, 2)), [3])


# ---------------------------------------------------------------- M3


def test_m3_uses_the_conditions_key_and_one_donor():
    donors = humans(45, seed=2, lo=80, hi=120)
    out = v7.m3(5, 300, donors)
    rng = np.random.default_rng([20261031, 7, 3, 5])     # the gap stream of text 5
    assert out["donor"] == int(np.random.default_rng([20261031, 7, 13]).permutation(45)[5])   # amendments B2, B10
    g = donors[out["donor"]]
    assert set(out["M3a"]) <= set(g) and set(out["M3b"]) <= set(g)
    assert out["M3a"].size == out["M3b"].size == 300
    a = rng.choice(g, 300, replace=True)
    b, _ = v7.CondBins([g]).draw(300, rng)
    assert np.array_equal(out["M3a"], a) and np.array_equal(out["M3b"], b)   # one generator, in this order
    again = v7.m3(5, 300, donors)
    assert np.array_equal(out["M3a"], again["M3a"]) and np.array_equal(out["M3b"], again["M3b"])


# ---------------------------------------------------------------- C1


def test_components_join_shared_tokens():
    tok = [{("person", 1)}, {("person", 1)}, {("text", 3)}, {("text", 3), ("person", 2)}, {("person", 2)},
           {("text", 9)}]
    assert list(v7.components(tok)) == [0, 0, 1, 1, 1, 2]


def test_group_folds_never_split_a_component():
    comp = v7.components([{("p", i // 2)} for i in range(30)])
    f = v7.group_folds(comp, np.random.default_rng(0))
    for c in np.unique(comp):
        assert np.unique(f[comp == c]).size == 1
    assert np.bincount(f).min() >= 6
    with pytest.raises(ValueError):
        v7.group_folds(np.array([0, 0, 1, 2, 3]), np.random.default_rng(0))


def test_c1_detects_a_shift_and_not_a_null():
    rng = np.random.default_rng(0)
    n = 40
    tok = [{("person", i)} for i in range(n)] + [{("text", i)} for i in range(n)]
    lab = np.array([0] * n + [1] * n)
    x_null = rng.normal(size=(2 * n, 7))
    x_shift = x_null + lab[:, None] * 1.5
    out = v7.c1(x_shift, lab, tok, [20261031, 7, 4, 0, 0, 0])
    assert out["reading"] == {"detection": "detectable", "size": "above 0.60"}
    null = v7.c1(x_null, lab, tok, [20261031, 7, 4, 0, 0, 0])
    assert null["reading"]["size"] != "above 0.60"


def test_c1_folds_keep_related_series_together(monkeypatch):
    n = 40
    tok = [{("person", i)} for i in range(n)] * 2
    comp = v7.components(tok)
    f = v7.group_folds(comp, np.random.default_rng(0))
    assert all(f[i] == f[i + n] for i in range(n))
    with pytest.raises(ValueError):
        v7.check_folds(tok, np.r_[np.arange(n) % 5, (np.arange(n) + 1) % 5])
    # a fold assignment that splits a component is refused inside c1
    monkeypatch.setattr(v7, "group_folds", lambda comp, rng: np.r_[np.arange(n) % 5, (np.arange(n) + 1) % 5])
    with pytest.raises(ValueError):
        v7.c1(np.random.default_rng(1).normal(size=(2 * n, 7)), np.array([0] * n + [1] * n), tok, [1])


def test_c1_bootstrap_resamples_components():
    # 4 components only: the interval must be wide (resampling series would make it narrow)
    rng = np.random.default_rng(2)
    n = 40
    tok = [{("text", i % 4)} for i in range(n)] + [{("text", 4 + i % 4)} for i in range(n)]
    lab = np.array([0] * n + [1] * n)
    x = rng.normal(size=(2 * n, 3)) + lab[:, None] * 0.6
    out = v7.c1(x, lab, tok, [1])
    assert out["n_components"] == 8 and out["rank_auc_ci"][1] - out["rank_auc_ci"][0] > 0.2


def test_stage_vectors_on_a_synthetic_text():
    c4 = json.loads((ROOT / "works/conditions/prototype_v4.json").read_text())
    rng = np.random.default_rng(0)
    kinds = (["alnum"] * 5 + ["space"]) * 6 + ["end", "space"]
    kinds = kinds * 8
    times = np.concatenate([[0.0], np.cumsum(human_like(len(kinds) - 1, rng))])
    v = v7.stage_vectors(times, kinds, c4["mapping"], c4["relief"])
    assert v["S0"].shape == (7,) and v["S1"].shape == (4,) and v["S2"].shape == (4,)
    v_enter = v7.stage_vectors(times, [("enter" if k == "space" else k) for k in kinds], c4["mapping"], c4["relief"])
    assert all(np.array_equal(v[s], v_enter[s]) for s in v)


# ---------------------------------------------------------------- replay error


def test_replay_error_aligned():
    p = np.arange(0, 2000, 100, dtype=float)
    q = p + 3.0 + np.arange(20) * 0.5          # deviation 3.0 .. 12.5
    out = v7.replay_error(p, q, ["a"] * 20, ["a"] * 20)
    d = q - p
    assert out["matched"] and out["offset_ms"] == np.median(d) and out["drift_ms"] == 12.5
    assert out["jitter_p95_ms"] == np.percentile(np.abs(d - np.median(d)), 95)
    assert out["jitter_mad_ms"] == np.median(np.abs(d - np.median(d))) and out["kind_edit_distance"] == 0
    assert v7.replay_error(p, q)["matched"]                   # no kinds: counts only


def test_replay_error_not_evaluable_when_counts_differ():
    p = np.arange(10) * 100.0
    k = list("abcabcabca")
    out = v7.replay_error(p, np.delete(p, 3), k, k[:3] + k[4:])
    assert out["drops"] == 1 and not out["counts_equal"] and not out["matched"]
    assert out["offset_ms"] is None and out["drift_ms"] is None and out["kind_edit_distance"] == 1
    extra = v7.replay_error(p, np.append(p, 2000.0))
    assert extra["extras"] == 1 and extra["jitter_p95_ms"] is None
    # a drop and an extra that cancel: counts equal, kinds differ -> not evaluable
    swap = v7.replay_error(p, p, k, k[:3] + k[4:] + ["c"])
    assert swap["counts_equal"] and not swap["matched"] and swap["offset_ms"] is None
    # an adjacent swap of two kinds
    sw = k[:]
    sw[1], sw[2] = sw[2], sw[1]
    assert v7.replay_error(p, p, k, sw)["kind_edit_distance"] == 2


def test_kind_edit_distance_against_a_plain_implementation():
    def plain(a, b):
        d = [[i + j if i * j == 0 else 0 for j in range(len(b) + 1)] for i in range(len(a) + 1)]
        for i in range(1, len(a) + 1):
            for j in range(1, len(b) + 1):
                d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
        return d[-1][-1]
    rng = np.random.default_rng(0)
    for _ in range(200):
        a = list(rng.choice(list("xyz"), rng.integers(0, 12)))
        b = list(rng.choice(list("xyz"), rng.integers(0, 12)))
        assert v7.kind_edit_distance(a, b) == plain(a, b)


# ---------------------------------------------------------------- sensitivity and resolution


def test_quantise_then_clean():
    t = np.array([0.0, 3.0, 9.0, 17.0, 31.0])
    q, idx, dropped = v7.sensitivity_variant(t, "grid", 16.0)
    assert list(q) == [0.0, 16.0, 32.0] and dropped == 2 and list(idx) == [0, 2, 4]
    kinds = ["a", "b", "c", "d", "e"]
    tj = np.array([0.0, 1.0, 50.0])                       # jitter SD 8 reorders the first two
    j, idx, _ = v7.sensitivity_variant(tj, "jitter", 8.0, np.random.default_rng(1))
    assert (np.diff(j) > 0).all() and sorted(idx) == [0, 1, 2]
    moved = tj + np.random.default_rng(1).normal(0, 8.0, 3)
    assert list(idx) == list(np.argsort(moved, kind="stable"))
    assert [kinds[i] for i in idx][-1] == "c"
    with pytest.raises(ValueError):
        v7.sensitivity_variant(t, "jitter", 2.0)


def test_resolution_recovers_a_grid():
    rng = np.random.default_rng(0)
    g = rng.choice(np.arange(1, 15) * 15.6, 3000) + rng.normal(0, 0.5, 3000)
    out = v7.resolution(np.abs(g), 0.0, 250.0, 0.1, bw_ms=1.0)
    assert out["resolution_ms"] == pytest.approx(15.6, abs=0.5) and out["stable"]
    # negative control: continuous gaps have no grid; the estimate must not be read as one
    smooth = np.random.default_rng(1).lognormal(math.log(150), 0.4, 3000)
    neg = v7.resolution(smooth, 0.0, 250.0, 0.1, bw_ms=1.0)
    assert not neg["stable"]


def test_rank_auc_mixed_ties():
    # pairs: (0.5 vs 0.5) tie, (0.5 vs 0.2) win, (0.9 vs 0.5) win, (0.9 vs 0.2) win -> (0.5 + 3) / 4
    assert v7.rank_auc(np.array([0.5, 0.2, 0.5, 0.9]), np.array([0, 0, 1, 1])) == pytest.approx(3.5 / 4)


def _tango_brute(b, c, n, grid=np.linspace(-0.999, 0.999, 19981)):
    """An independent Tango interval: the nuisance p21 by maximising the restricted multinomial likelihood
    numerically, and the interval by scanning delta on a 0.0001 grid."""
    from scipy import optimize
    x12, x21, rest = b, c, n - b - c

    def z(d):
        lo, hi = max(0.0, -d), min(1.0, (1.0 - d) / 2.0)
        if hi <= lo:
            p21 = lo
        else:
            def nll(p):
                p12, p11 = p + d, 1.0 - 2.0 * p - d
                tiny = 1e-300
                return -(x12 * math.log(max(p12, tiny)) + x21 * math.log(max(p, tiny))
                         + rest * math.log(max(p11, tiny)))
            p21 = optimize.minimize_scalar(nll, bounds=(lo, hi), method="bounded",
                                           options={"xatol": 1e-12}).x
        var = n * (2.0 * p21 + d * (1.0 - d))
        return (b - c - n * d) / math.sqrt(var) if var > 0 else math.copysign(math.inf, b - c - n * d)

    inside = [d for d in grid if abs(z(d)) <= v7.Z95]
    return min(inside), max(inside)


@pytest.mark.parametrize("b,c", [(0, 0), (1, 0), (1, 1), (3, 0), (2, 5), (6, 1), (10, 3), (0, 9)])
def test_tango_against_an_independent_implementation(b, c):
    lo, hi = v7.tango_interval(b, c, 40)
    blo, bhi = _tango_brute(b, c, 40, grid=np.linspace(-0.6, 0.6, 12001))
    assert lo == pytest.approx(blo, abs=2e-4) and hi == pytest.approx(bhi, abs=2e-4)


def test_model_seeds_and_sizes():
    h = humans(8)
    d2 = v7.train_d2(h, np.random.default_rng(42))
    rf2 = d2.rf.steps[-1][1]
    lr = d2.lr.steps[-1][1]
    assert (rf2.random_state, rf2.n_estimators, lr.C) == (42, 100, 1.0)
    rf3 = v7.train_d3(h, np.random.default_rng(123)).rf
    assert (rf3.random_state, rf3.n_estimators) == (123, 300)


def test_component_bootstrap_resamples_whole_components():
    # class 1: one component scored high, one low; class 0: two components in the middle. Resampling components
    # often draws only the high or only the low one (AUC 1 or 0); resampling rows stays near 0.5.
    score = np.r_[np.full(10, 0.5), np.full(10, 0.5), np.full(10, 0.9), np.full(10, 0.1)]
    label = np.r_[np.zeros(20, dtype=int), np.ones(20, dtype=int)]
    comp = np.repeat(np.arange(4), 10)
    lo, hi, skipped = v7.component_bootstrap_auc(score, label, comp, np.random.default_rng(0))
    assert lo == 0.0 and hi == 1.0 and skipped > 0


# ---------------------------------------------------------------- amendments A2-A5 (2026-11-02, before any value)


def test_conditions_name_the_amendments():
    h = C7["detectors_C2"]["headline"]
    assert "discordance (b + c) / n at most 0.10" in h and "lower end of the discordance's Wilson interval above 0.10" in h
    assert "machine_path" in C7["pairs"] and "cluster bootstrap" in C7["detectors_C2"]["count"]
    assert "[20261031, 7, 11, layer, detector, series]" in C7["detectors_C2"]["count"]
    assert "no equivalence reading" in C7["pairs"]["replay_repeat"]


def test_wilson_interval_known_values():
    lo, hi = v7.wilson_interval(0, 40)
    z2 = v7.Z95 ** 2
    assert lo == 0.0 and hi == pytest.approx(z2 / (40 + z2))
    assert round(hi, 3) == 0.088   # the value written in the conditions
    lo, hi = v7.wilson_interval(10, 40)
    assert (round(lo, 3), round(hi, 3)) == (0.142, 0.402)   # by hand: mid 0.2719, half 0.1300
    lo2, hi2 = v7.wilson_interval(30, 40)
    assert lo2 == pytest.approx(1 - hi) and hi2 == pytest.approx(1 - lo)
    with pytest.raises(ValueError):
        v7.wilson_interval(5, 4)


def test_headline_discordance_route():
    # b = c = 5: the difference is 0 but a quarter of the persons swap (R3 F5 of attack-me round 1)
    h = v7.headline(5, 5, 40)
    assert h["difference"] == 0 and h["reading"] == "responds" and h["wilson_95"][0] > v7.MARGIN
    # one flip must not become 'responds' through the discordance (its upper end 0.129 > 0.10, lower end small)
    assert v7.headline(1, 0, 40)["reading"] == "undecided"
    assert v7.headline(2, 2, 40)["reading"] == "undecided"
    assert v7.headline(0, 0, 40)["reading"] == "not changed"


def test_read_rule_requires_both_for_not_changed():
    assert v7.read_rule((-0.05, 0.05), (0.0, 0.08)) == "not changed"
    assert v7.read_rule((-0.05, 0.05), (0.0, 0.12)) == "undecided"
    assert v7.read_rule((0.02, 0.15), (0.03, 0.2)) == "undecided"   # amendment B3: not the whole interval beyond
    assert v7.read_rule((0.11, 0.3), (0.03, 0.2)) == "responds"
    assert v7.read_rule((-0.3, -0.11), (0.03, 0.2)) == "responds"
    assert v7.read_rule((-0.05, 0.05), (0.11, 0.3)) == "responds"
    assert v7.read_rule((-0.15, 0.05), (0.0, 0.3)) == "undecided"


def test_machine_path_reading():
    n = 40
    before = np.ones(n, dtype=bool)
    cluster = np.arange(n) // 4
    out = v7.machine_path_reading(before, before.copy(), cluster, [1])
    assert (out["b"], out["c"], out["n"], out["n_clusters"]) == (0, 0, 40, 10)
    # no flip in 10 clusters: the bootstrap interval degenerates to [0, 0], the effective-n interval (n = 10) does not
    assert out["bootstrap_95"]["difference"] == [0.0, 0.0] and out["reading"] == "undecided"
    out40 = v7.machine_path_reading(before, before.copy(), np.arange(n), [1])
    assert out40["reading"] == "not changed"   # 40 independent sessions: as the human side with b = c = 0
    after2 = before.copy()
    after2[:20] = False          # half of the machine sessions now judged human (c = 20)
    out2 = v7.machine_path_reading(before, after2, np.arange(n), [1])
    assert out2["c"] == 20 and out2["difference"] == -0.5 and out2["reading"] == "responds"
    assert out2["bootstrap_95"]["difference"][1] < 0 and out2["route"] == "both"
    with pytest.raises(TypeError):
        v7.machine_path_reading(before.astype(float), before, cluster, [1])
    with pytest.raises(ValueError):
        v7.machine_path_reading(before, before, cluster[:-1], [1])


def test_machine_path_one_flip_is_not_easier_than_the_human_side():
    # code review HIGH (2026-11-02): with the bootstrap alone one flip read 'not changed'
    n = 40
    before = np.ones(n, dtype=bool)
    after = before.copy()
    after[0] = False
    for cluster in (np.arange(n), np.arange(n) // 4):
        out = v7.machine_path_reading(before, after, cluster, [1])
        assert out["reading"] == v7.headline(0, 1, 40)["reading"] == "undecided"


def test_machine_path_few_clusters_cannot_decide():
    n = 40
    before = np.ones(n, dtype=bool)
    after = before.copy()
    after[:10] = False
    one = v7.machine_path_reading(before, after, np.zeros(n, dtype=int), [1])
    eff = one["effective_n_95"]["difference"]
    assert one["n_clusters"] == 1 and eff[1] - eff[0] > 1.5 and one["reading"] == "undecided"
    assert one["bootstrap_95"]["reading"] == "responds"   # the degenerate bootstrap alone would have read 'responds'
    with pytest.raises(TypeError):
        v7.machine_path_reading(before, after, np.where(np.arange(n) < 5, np.nan, 1.0), [1])
    two = v7.machine_path_reading(before, after, np.arange(n) % 2, [1])
    assert two["reading"] == "undecided"


def test_route_and_closed_bounds():
    assert v7.headline(12, 0, 40)["route"] in ("difference", "both")
    assert v7.headline(5, 5, 40)["route"] == "discordance"
    assert v7.headline(0, 0, 40)["route"] == ""
    # an end equal to the margin up to rounding is inside (closed bounds)
    assert v7.read_rule((-0.1, 0.1), (0.0, 0.1 + 1e-15)) == "not changed"
    assert v7.read_rule((-0.05, 0.05), (0.1, 0.3)) == "undecided"
    assert v7.read_rule((-0.05, 0.05), (0.1 + 1e-9, 0.3)) == "responds"


def test_headline_symmetry_and_conditions_numbers():
    for b, c in ((3, 0), (5, 5), (12, 1), (0, 0)):
        h, g = v7.headline(b, c, 40), v7.headline(c, b, 40)
        assert h["reading"] == g["reading"] and h["wilson_95"] == g["wilson_95"]
    h1 = v7.headline(1, 0, 40)
    assert [round(x, 3) for x in h1["tango_95"]] == [-0.065, 0.129] and h1["reading"] == "undecided"
    assert [round(x, 3) for x in v7.tango_interval(1, 0, 40, float(v7.stats.norm.ppf(0.95)))] == [-0.040, 0.105]
    # amendment B11: the conditions now state the n = 80 design (the 90% check: three flips of 80)
    assert [round(x, 3) for x in v7.tango_interval(3, 0, 80, float(v7.stats.norm.ppf(0.95)))] == [0.004, 0.090]
    assert "[0.004, 0.090]" in C7["detectors_C2"]["headline"]


def test_threshold_margin_and_floor():
    m = v7.threshold_margin("D1", np.array([1.0, 0.269, np.nan]))
    assert m[0] == pytest.approx(0.731) and m[1] == 0 and np.isnan(m[2])
    assert v7.threshold_margin("D3", np.array([0.2]))[0] == pytest.approx(0.3)
    with pytest.raises(ValueError):
        v7.threshold_margin("D9", np.array([0.2]))
    h0 = np.linspace(0.1, 0.4, 10)
    h1 = h0 + 0.2
    rep = h1 + 0.001
    assert v7.above_floor(h0, h1, rep, [5])["above_floor"]
    assert not v7.above_floor(h0, h0 + 0.001, h0 + 0.2, [5])["above_floor"]
    # mean of the difference positive but its interval crossing 0: not above the floor (the interval decides)
    ch = np.array([0.30, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    rep_ch = np.array([0.0, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02])
    out = v7.above_floor(np.zeros(10), ch, ch + rep_ch, [5])
    assert out["mean"] > 0 and out["ci95"][0] <= 0 and not out["above_floor"]


def test_not_changed_probability_matches_the_closed_form():
    # with n = 40 only b = c = 0 reads 'not changed', so the probability is (1 - rate) ** 40
    for r in (0.0, 0.005, 0.01, 0.02, 0.05):
        assert v7.not_changed_probability(r, 40) == pytest.approx((1 - r) ** 40)
    assert round(v7.not_changed_probability(0.02, 40), 3) == 0.446
    assert v7.not_changed_probability(1.0, 40) == 0.0


def test_not_changed_probability_brute_force_other_n():
    # a direct sum over (b, c) of the headline reading, independent of the closed form
    n, r = 120, 0.01
    want = 0.0
    for b in range(n + 1):
        for c in range(n - b + 1):
            if v7.headline(b, c, n)["reading"] == "not changed":
                want += (math.comb(n, b) * math.comb(n - b, c) * (r / 2) ** (b + c) * (1 - r) ** (n - b - c))
    assert v7.not_changed_probability(r, n) == pytest.approx(want, rel=1e-9)


def test_machine_path_disagreeing_intervals_read_undecided():
    # Codex round 2: the two pairs of intervals are read separately and only a common reading is taken.
    # 10 flips all in one cluster of 10 sessions, 30 other singleton clusters: the effective-n interval reads
    # 'responds' while the cluster bootstrap (the big cluster is missing in about a third of the resamples) does not.
    n = 40
    before = np.zeros(n, dtype=bool)
    after = before.copy()
    after[:10] = True
    cluster = np.concatenate([np.zeros(10, dtype=int), np.arange(1, 31)])
    out = v7.machine_path_reading(before, after, cluster, [1])
    assert out["effective_n_95"]["reading"] == "responds"
    assert out["bootstrap_95"]["reading"] != "responds"
    assert out["reading"] == "undecided" and out["route"] == ""


def test_threshold_equality_semantics():
    # D1: 'human' needs delta > 0.269, so a margin of 0 is 'machine'; D2/D3: 'machine' needs p > 0.5, margin 0 is human
    assert v7.threshold_margin("D1", np.array([v7.DELTA_T]))[0] == 0 and v7.d1_machine is not None
    assert v7.threshold_margin("D2_lr", np.array([0.5]))[0] == 0
    j_before, _, _ = v7.judged_pairs(np.array([0.5]), np.array([0.5]))
    assert not j_before[0]       # p = 0.5 is judged human


def test_floor_needs_enough_persons():
    h0 = np.array([0.1, 0.2, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
    out = v7.above_floor(h0, h0 + 0.3, h0 + 0.3001, [5])
    assert out["n"] == 2 and out["above_floor"] is None


# ---------------------------------------------------------------- amendments B (2026-11-02, after attack-me round 2)


def test_b3_two_steps_b4_c0():
    # R3 N3 / Codex 2 of round 2: b = 4, c = 0 read 'responds' before B3 without showing a change beyond 0.10
    h = v7.headline(4, 0, 40)
    assert h["tango_95"][0] > 0 and h["tango_95"][0] < v7.MARGIN
    assert h["reading"] == "undecided" and h["changed_from_0"] is True
    assert v7.headline(0, 0, 40)["changed_from_0"] is False
    big = v7.headline(20, 0, 40)
    assert big["tango_95"][0] > v7.MARGIN and big["reading"] == "responds" and big["route"] in ("difference", "both")


def test_b3_narrower_interval_cannot_lose_responds():
    # the same share of flips with more persons: once 'responds' it stays (monotone in n for a fixed share)
    seen = False
    for n in range(20, 401, 20):
        r = v7.headline(n // 2, 0, n)["reading"]
        seen = seen or r == "responds"
        assert not (seen and r != "responds")
    assert seen


def test_b8_zero_flip_min_clusters_is_35():
    assert v7.zero_flip_min_clusters() == 35
    t = {row["n_eff"]: row for row in v7.zero_flip_table()}
    assert t[34]["reading"] == "undecided" and round(t[34]["tango_95"][1], 3) == 0.102
    assert t[35]["reading"] == "not changed" and t[40]["reading"] == "not changed"
    assert v7.enters_machine_general(35) and not v7.enters_machine_general(34)


def test_b2_donors_without_replacement_and_old_draw_had_34():
    d = v7.m3_donors(160)
    assert d.size == 40 and np.unique(d).size == 40 and d.max() < 160
    old = {int(np.random.default_rng([20261031, 7, 3, t]).integers(0, 160)) for t in range(40)}
    assert len(old) == 34    # the reason of B2: 34 clusters cannot read 'not changed'
    with pytest.raises(ValueError):
        v7.m3_donors(39)
    with pytest.raises(ValueError):
        v7.m3(40, 100, humans(45, seed=2, lo=80, hi=120))


def test_machine_path_changed_from_0_needs_both_intervals():
    n = 40
    before = np.ones(n, dtype=bool)
    after = before.copy()
    after[:4] = False
    out = v7.machine_path_reading(before, after, np.arange(n), [1])
    assert out["reading"] == "undecided" and out["changed_from_0"] is True
    out0 = v7.machine_path_reading(before, before.copy(), np.arange(n), [1])
    assert out0["changed_from_0"] is False and out0["reading"] == "not changed"


def test_d4_features_kde_and_training():
    rng = np.random.default_rng(5)
    h = humans(30, seed=3)
    assert v7.d4_features(np.ones(29)) is None and v7.d4_features(np.arange(40.0)).tolist() == list(range(30))
    kde = v7.fit_kde1(h)
    assert kde.pool.size == 30 * 30
    x = kde.draw(5000, rng)
    assert x.size == 5000 and x.min() >= 0
    zero = v7.KDE1(np.zeros(10))
    z = zero.draw(1000, np.random.default_rng(1))
    assert z.min() >= 0 and zero.redrawn > 0      # draws below 0 are drawn again and counted
    m = v7.train_d4(humans(150, seed=3), np.random.default_rng(v7.D4_SEED))
    # held out: new human-like persons against new draws of the same KDE (the paper's Universal bots); a uniform
    # series would be outside the training distribution and is not what D4 learns
    held = humans(50, seed=8)
    machine = v7.d4_synthetic(50, m.kde, np.random.default_rng(99))
    p = m.proba(held + machine + [np.ones(10)])
    assert np.isnan(p[-1]) and np.all(np.isfinite(p[:-1]))
    assert v7.rank_auc(p[:-1], np.array([0] * 50 + [1] * 50)) > 0.75   # 0.86 on these seeds (lag-1 dependence)


def test_d4_is_trained_only_on_its_own_humans():
    h = humans(20, seed=4)
    m1 = v7.train_d4(h, np.random.default_rng(28))
    m2 = v7.train_d4(h, np.random.default_rng(28))
    probe = humans(5, seed=9)
    assert np.array_equal(m1.proba(probe), m2.proba(probe))    # deterministic with the conditions' seeds
    assert np.array_equal(m1.kde.pool, np.concatenate([g[:30] for g in h]))


def test_b5_gridded_gate_is_reported_and_does_not_decide():
    out = v7.validity_gate(humans(25, seed=11), key=[1, 2, 3], grid_ms=16.0)
    for name in ("D1_uniform", "D1_d3gen", "D2_lr", "D2_rf", "D3", "D4"):
        g = out[name]["gridded"]
        assert g["grid_ms"] == 16.0 and 0 <= g["auc"] <= 1 and 0 <= g["human_machine_share"] <= 1
        # the held-out humans are not gridded, so their 'machine' share is the same
        assert g["human_machine_share"] == pytest.approx(out[name]["human_machine_share"])
    assert "no_score" in out["D4"] and "valid" in out["D4"]
    # the gridded version is computed on gridded sessions (a 16 ms grid moves at least one AUC)
    assert any(out[k]["gridded"]["auc"] != out[k]["auc"] for k in ("D1_uniform", "D1_d3gen", "D2_lr", "D2_rf", "D3", "D4"))


def test_gridded_rounds_times_not_gaps():
    g = np.array([10.4, 10.4, 10.4])
    assert v7.gridded(g, 1.0).tolist() == [10.0, 11.0, 10.0]   # times 0, 10.4, 20.8, 31.2 -> 0, 10, 21, 31
    assert v7.gridded(np.array([0.3, 0.3]), 1.0).tolist() == [1.0]   # equal times after rounding: dropped


def test_saturation_and_logit():
    assert v7.saturation_share(np.array([0.0, 0.005, 0.5, 0.995, 1.0, np.nan])) == pytest.approx(0.8)
    with pytest.raises(ValueError):
        v7.saturation_share(np.array([np.nan]))
    h = humans(15, seed=6)
    d2 = v7.train_d2(h, np.random.default_rng(42))
    p = d2.proba(h[:3])
    assert np.allclose(1 / (1 + np.exp(-p["D2_lr_logit"])), p["D2_lr"])


def test_d4_in_detectors_and_margin():
    assert "D4" in v7.DETECTORS
    assert v7.threshold_margin("D4", np.array([0.2]))[0] == pytest.approx(0.3)


def test_conditions_record_amendments_b():
    assert "amendment B1" in C7["data"]["prompts"] and len(C7["data"]["prompts_added"]) == 30
    assert "40 prompts (P1-P40) x 1 seed" in C7["series"]["M1_8B and M1_14B"]
    assert "permutation(|E|)[:40]" in C7["series"]["M3a"]
    assert "D4" in C7["detectors_C2"] and "responds beyond 0.10" in C7["detectors_C2"]["headline"]
    assert "35 or more" in C7["detectors_C2"]["headline"]


def test_machine_path_changed_from_0_is_common_to_both_intervals():
    # 40 sessions in 4 clusters, one flip in each: every bootstrap resample gives 0.10 (excludes 0), while the
    # effective-n interval (n_eff = 4, b_eff = 0.4) contains 0; 'changed from 0' must need both
    n = 40
    before = np.ones(n, dtype=bool)
    after = before.copy()
    after[[0, 10, 20, 30]] = False
    out = v7.machine_path_reading(before, after, np.arange(n) // 10, [1])
    assert out["bootstrap_95"]["difference"][1] < 0 and out["effective_n_95"]["difference"][1] > 0
    assert out["changed_from_0"] is False


# ---------------------------------------------------------------- amendment B10 and the reviews of B (2026-11-02)


def _all_keys(fold_shift: int = 1) -> list:
    S = 20261031
    k = [[S, 7, 1, p] for p in range(40)] + [[S, 7, 2]] + [[S, 7, 3, t] for t in range(40)] + [[S, 7, 13]]
    k += [[S, 7, 4, la, p, st] for la in range(2) for p in range(12) for st in range(3)]
    k += [[S, 7, 5, la, d, se] for la in range(2) for d in range(5) for se in range(12)]
    k += [[S, 7, 6]] + [[S, 7, 7, p] for p in range(40)] + [[S, 7, 8]]
    k += [[S, 7, 9, la, d, p] for la in range(2) for d in range(5) for p in range(6)]
    k += [[S, 7, 10, la, se, ss, sd] for la in range(2) for se in range(10) for ss in range(40) for sd in range(2)]
    k += [[S, 7, 11, la, d, se] for la in range(2) for d in range(5) for se in range(9)]
    k += [[S, 7, 12, la, d] for la in range(2) for d in range(5)]
    for seed in (v7.D2_SEED, v7.D3_SEED, v7.D4_SEED):
        k.append(seed)
        for f in range(v7.N_FOLDS):
            k += [[seed, f + fold_shift], [seed, f + fold_shift, 1]]
    return k


def test_b10_no_two_keys_give_one_stream():
    def first(key):
        return tuple(np.random.default_rng(key).integers(0, 2 ** 62, 4))
    assert first(28) == first([28, 0])          # the reason of B10: a trailing 0 is absent for SeedSequence
    seen = {}
    for key in _all_keys(1):
        f = first(key)
        assert f not in seen, (seen.get(f), key)
        seen[f] = key
    old = [first(key) for key in _all_keys(0)]
    assert len(set(old)) < len(old)            # the keys before B10 collided


def test_b10_gate_uses_shifted_fold_keys(monkeypatch):
    seen = []
    real = v7.train_d4
    monkeypatch.setattr(v7, "train_d4", lambda h, rng: (seen.append(rng.bit_generator.seed_seq.entropy), real(h, rng))[1])
    v7.validity_gate(humans(25, seed=11), key=[1, 2, 3])
    assert seen == [[v7.D4_SEED, k + 1] for k in range(v7.N_FOLDS)]


def test_gate_kde_and_training_use_the_training_folds_only(monkeypatch):
    h = humans(25, seed=11)
    fold = v7.person_folds(len(h), [1, 2, 3])
    pools = []
    real = v7.fit_kde1
    monkeypatch.setattr(v7, "fit_kde1", lambda hs: (pools.append([id(g) for g in hs]), real(hs))[1])
    v7.validity_gate(h, key=[1, 2, 3])
    ids = [id(g) for g in h]
    # train_d4 and the held-out generator each fit once per fold, on the persons outside the fold
    assert len(pools) == 2 * v7.N_FOLDS
    for i, used in enumerate(pools):
        k = i // 2
        assert set(used) == {ids[j] for j in range(len(h)) if fold[j] != k}


def test_b5_validity_is_the_same_whatever_the_grid():
    h = humans(25, seed=11)
    a = v7.validity_gate(h, key=[1, 2, 3], grid_ms=1.0)
    b = v7.validity_gate(h, key=[1, 2, 3], grid_ms=16.0)
    for name in ("D1", "D2_lr", "D2_rf", "D3", "D4"):
        assert a[name]["valid"] == b[name]["valid"] and a[name]["auc"] == b[name]["auc"]
        assert a[name]["human_machine_share"] == b[name]["human_machine_share"]
    assert set(b["D4"]["redrawn"]) == {"train", "held_out"}
    assert "no_score" in b["D3"]["gridded"] and "n_machine" in b["D4"]["gridded"]


def test_gridded_lost_presses_are_counted_not_hidden():
    # D4 synthetic sessions are exactly 30 gaps; a coarse grid merges presses, so some lose their D4 score
    h = humans(25, seed=11)
    out = v7.validity_gate(h, key=[1, 2, 3], grid_ms=200.0)
    g = out["D4"]["gridded"]
    assert g["no_score"] > 0 and g["n_machine"] + g["no_score"] >= sum(1 for x in h if x.size >= 30)


def test_d4_held_out_synthetic_matches_eligible_humans():
    h = humans(20, seed=12) + [np.full(10, 100.0) for _ in range(5)]    # five held-out-able humans too short for D4
    out = v7.validity_gate(h, key=[1, 2, 3])
    assert out["D4"]["no_score"] == 5 and out["D4"]["n_human"] == 20 and out["D4"]["n_machine"] == 20


def test_b3_edges_on_the_responds_side():
    assert v7.read_rule((v7.MARGIN, 0.3), (0.0, 0.05)) == "undecided"          # an end equal to 0.10 is inside
    assert v7.read_rule((v7.MARGIN + 1e-9, 0.3), (0.0, 0.05)) == "responds"
    assert v7.read_rule((-0.3, -v7.MARGIN), (0.0, 0.05)) == "undecided"
    assert v7.read_rule((-0.3, -v7.MARGIN - 1e-9), (0.0, 0.05)) == "responds"
    assert v7.changed_from_0((0.0, 0.2)) is False and v7.changed_from_0((1e-12, 0.2)) is True


def test_saturation_edges_inclusive():
    assert v7.saturation_share(np.array([0.01, 0.99, 0.0101, 0.9899])) == 0.5


def test_d4_rf_seed_and_trees():
    m = v7.train_d4(humans(20, seed=4), np.random.default_rng(28))
    assert m.rf.random_state == 28 and m.rf.n_estimators == 100


def test_b8_machine_general_uses_the_clusters_left_per_cell():
    valid = {"D1": True, "D4": True, "D3": False}
    series = ["M0", "M2_8B", "M3a"]
    cells = [(d, la, s) for d in ("D1", "D4") for la in ("hook", "page") for s in series]
    r = {c: "not changed" for c in cells}
    n = {c: 40 for c in cells}
    n[("D1", "hook", "M2_8B")] = 34                  # one cell short keeps the whole series out
    n[("D4", "page", "M3a")] = 35                    # 35 is enough
    r[("D4", "hook", "M2_8B")] = "responds"
    out = v7.machine_general(r, valid, n, series)
    assert out["entering"] == ["M0", "M3a"] and out["kept_out"] == ["M2_8B"] and out["general_not_moved"] is True
    assert out["responds_kept_out"] == [["D4", "hook", "M2_8B"]] and out["responds"] == []
    r[("D4", "page", "M3a")] = "undecided"
    assert v7.machine_general(r, valid, n, series)["general_not_moved"] is False
    del n[("D1", "page", "M0")]
    with pytest.raises(ValueError):
        v7.machine_general(r, valid, n, series)

def test_b5_gridded_version_never_changes_validity(monkeypatch):
    # make the gridded synthetic sessions look human (one held-out-like person repeated): the gridded AUC falls below
    # the gate, and validity must still follow the ungridded AUC and share only
    template = humans(1, seed=99)[0]
    monkeypatch.setattr(v7, "gridded", lambda g, grid_ms: template.copy())
    out = v7.validity_gate(humans(25, seed=11), key=[1, 2, 3], grid_ms=1.0)
    assert any(out[n]["gridded"]["auc"] < v7.GATE_AUC for n in ("D2_lr", "D2_rf", "D3", "D4"))
    for n in ("D1_uniform", "D1_d3gen", "D2_lr", "D2_rf", "D3", "D4"):
        o = out[n]
        assert o["valid"] == bool(o["auc"] >= v7.GATE_AUC and o["human_machine_share"] <= v7.GATE_HUMAN_SHARE)
    assert out["D2_lr"]["valid"] or out["D4"]["valid"]     # something valid that a gridded decision would flip



def test_gate_survives_gridded_sessions_that_vanish(monkeypatch):
    # every gridded synthetic session collapses to fewer than 2 gaps: counted, AUC None, validity unchanged
    h = humans(25, seed=11)
    base = v7.validity_gate(h, key=[1, 2, 3])
    monkeypatch.setattr(v7, "gridded", lambda g, grid_ms: np.array([5.0]))
    out = v7.validity_gate(h, key=[1, 2, 3], grid_ms=1.0)
    for n in ("D1_uniform", "D1_d3gen", "D2_lr", "D2_rf", "D3", "D4"):
        assert out[n]["gridded"]["auc"] is None and out[n]["gridded"]["no_score"] > 0
        assert out[n]["valid"] == base[n]["valid"] and out[n]["auc"] == base[n]["auc"]


def test_d4_fold_without_eligible_humans_stops_the_run():
    with pytest.raises(ValueError):
        v7.train_d4([np.full(10, 100.0) for _ in range(8)], np.random.default_rng(28))


def test_gate_uses_exactly_the_conditions_keys(monkeypatch):
    used = []
    real = np.random.default_rng

    def spy(seed=None):
        used.append(int(seed) if np.ndim(seed) == 0 else [int(x) for x in seed])
        return real(seed)
    h = humans(25, seed=11)                     # made before the spy (numpy is one module)
    monkeypatch.setattr(v7.np.random, "default_rng", spy)
    v7.validity_gate(h, key=[1, 2, 3])
    want = [[s, k + 1] for s in (v7.D2_SEED, v7.D3_SEED, v7.D4_SEED) for k in range(v7.N_FOLDS)]
    want += [[s, k + 1, 1] for s in (v7.D2_SEED, v7.D3_SEED, v7.D4_SEED) for k in range(v7.N_FOLDS)]
    got = [tuple(x) if isinstance(x, list) else (x,) for x in used if x != [1, 2, 3]]
    assert sorted(got) == sorted(map(tuple, want)), [g for g in got if len(g) == 1]


# ---------------------------------------------------------------- amendment B11 (after attack-me round 3)


def test_b11_reading_probabilities_sum_and_known_values():
    for n in (40, 80):
        for r in (0.0, 0.02, 0.10):
            pr = v7.reading_probabilities(r, n)
            assert sum(pr.values()) == pytest.approx(1.0, abs=1e-9)
            assert pr["not changed"] == pytest.approx(v7.not_changed_probability(r, n), abs=1e-9)
    assert round(v7.reading_probabilities(0.02, 80)["not changed"], 2) == 0.78
    assert round(v7.reading_probabilities(0.10, 40)["responds"], 3) == 0.042


def test_b11_design_numbers_in_the_conditions():
    h = C7["detectors_C2"]["headline"]
    assert "[-0.046, 0.046]" in h and "[-0.022, 0.087]" in h and "[-0.010, 0.105]" in h
    assert v7.headline(2, 0, 80)["reading"] == "not changed" and v7.headline(3, 0, 80)["reading"] == "undecided"
    assert [round(x, 3) for x in v7.headline(0, 0, 80)["tango_95"]] == [-0.046, 0.046]


def test_b11_boundary_readings():
    out = v7.boundary_readings(0, 0, 78, 2)
    assert out["complete"] == "not changed" and out["missing_as_no_flip"] == "not changed"
    assert out["missing_as_b"] == "not changed"             # 2 of 80 one way still reads 'not changed'
    assert v7.boundary_readings(0, 0, 77, 3)["missing_as_b"] == "undecided"
    assert v7.boundary_readings(0, 0, 77, 3)["missing_as_c"] == "undecided"
    assert v7.boundary_readings(0, 0, 77, 3)["missing_as_no_flip"] == "not changed"
    z = v7.boundary_readings(1, 0, 80, 0)
    assert z["complete"] == z["missing_as_no_flip"] == z["missing_as_b"] == z["missing_as_c"] == "not changed"
    allmiss = v7.boundary_readings(0, 0, 0, 80)
    assert allmiss["complete"] is None and allmiss["missing_as_no_flip"] == "not changed"
    with pytest.raises(ValueError):
        v7.boundary_readings(0, 0, 10, -1)


def test_b11_grid_note_only_from_the_auc():
    g = {"D1_uniform": {"gridded": {"auc": 0.89}}, "D1_d3gen": {"gridded": {"auc": 0.99}},
         "D2_lr": {"gridded": {"auc": 0.90}}, "D3": {"gridded": {"auc": None}}, "D4": {"gridded": {"auc": float("nan")}}}
    assert v7.grid_note(g, "D1") == "note"          # D1 is read on D1_uniform, not D1_d3gen
    assert v7.grid_note(g, "D2_lr") == "no note"    # 0.90 is not below the gate
    assert v7.grid_note(g, "D3") == "unknown" and v7.grid_note(g, "D4") == "unknown"


def test_b11_executable_plan_is_monotone_and_keeps_order():
    w = np.array([0.0, 10.0, 25.0, 18.0, 30.0, 29.0, 40.0])     # token arrived early: wished times go back
    e = v7.executable_plan(w)
    assert e.tolist() == [0, 10, 25, 25, 30, 30, 40] and np.all(np.diff(e) >= 0)
    assert v7.check_gaps(np.diff(e)).min() == 0               # gaps are never negative
    with pytest.raises(ValueError):
        v7.check_gaps(np.diff(w))                            # the wished times would have stopped the analysis


def test_b11_floor_on_absolute_changes():
    before = np.zeros(10)
    after = np.array([0.05, -0.05] * 5)                      # signed mean 0, absolute mean 0.05
    out = v7.floor_abs_change(before, after, [1])
    assert out["bias"]["mean"] == pytest.approx(0.0) and out["mean_abs"] == pytest.approx(0.05) and out["passes"] is False
    assert v7.floor_abs_change(before, before + 0.01, [1])["passes"] is True
    assert v7.floor_abs_change(before, before + 0.02, [1])["passes"] is True        # the limit is inside
    assert v7.floor_abs_change(before, before + 0.0201, [1])["passes"] is False
    gap = after.copy()
    gap[:3] = np.nan                                                               # 7 complete: not read, counted
    o = v7.floor_abs_change(before, gap, [1])
    assert o["passes"] is None and o["n"] == 7 and o["removed"] == 3
    with pytest.raises(ValueError):
        v7.floor_abs_change(before, np.full(10, np.nan), [1])


def test_b11_e_split_holds_the_donors_out():
    donors, train = v7.e_split(220)
    assert donors.size == 40 and train.size == 180
    assert not set(donors) & set(train) and set(donors) | set(train) == set(range(220))
    assert np.array_equal(donors, v7.m3_donors(220))


def test_b11_conditions_record_the_amendments():
    assert "first 80 of the pool" in C7["persons"]["replay_H"] and "first 300 eligible" in C7["persons"]["pool"]
    assert "E_train" in C7["persons"]["estimation_E"] and "P_nohook" in C7["series"]
    assert "executable plan" in C7["series"]["M1_8B and M1_14B"] and "mean over persons of the absolute" in C7["prediction"]


def test_b11_design_numbers_computed():
    r = lambda h: [round(x, 3) for x in h["tango_95"]]
    assert r(v7.headline(2, 0, 80)) == [-0.022, 0.087] and r(v7.headline(3, 0, 80)) == [-0.010, 0.105]
    assert r(v7.headline(1, 1, 80)) == [-0.057, 0.057] and r(v7.headline(1, 0, 80)) == [-0.034, 0.067]
    assert [round(x, 3) for x in v7.headline(1, 1, 80)["wilson_95"]] == [0.007, 0.087]
    assert round(v7.reading_probabilities(0.10, 80)["undecided"], 2) == 0.96


def test_b11_plan_shift_bounds_and_p5():
    out = v7.plan_shift(np.array([0.0, 10.0, 25.0, 18.0, 30.0, 29.0, 40.0]))
    assert out["share_moved"] == pytest.approx(2 / 7) and out["median_shift"] == pytest.approx(4.0)
    assert v7.plan_shift(np.arange(5.0))["median_shift"] is None
    b = v7.general_probability_bounds([0.95] * 10)
    assert b["independent_example"] == pytest.approx(0.95 ** 10) and b["upper"] == 0.95
    assert b["lower"] == pytest.approx(0.5)            # Frechet: 10 x 0.95 - 9
    with pytest.raises(ValueError):
        v7.general_probability_bounds([1.2])
    assert v7.p5_passes(18.0, 19.0) and not v7.p5_passes(18.0, 19.01)


# ---------------------------------------------------------------- amendment B12 (after attack-me round 4)


def test_b12_general_cell_ok_needs_every_bound():
    assert v7.general_cell_ok(0, 0, 80, 0)
    assert v7.general_cell_ok(0, 0, 78, 2)                  # 2 missing as b still 'not changed'
    assert not v7.general_cell_ok(0, 0, 77, 3)              # 3 missing as b -> 'undecided': split, kept out
    assert v7.boundary_readings(0, 0, 77, 3)["complete"] == "not changed"   # the cell's own reading stays


def test_b12_gate_without_a_resolution_is_unknown():
    out = v7.validity_gate(humans(25, seed=11), key=[1, 2, 3], grid_ms=None)
    for det in ("D1", "D2_lr", "D2_rf", "D3", "D4"):
        assert v7.grid_note(out, det) == "unknown"
    base = v7.validity_gate(humans(25, seed=11), key=[1, 2, 3], grid_ms=1.0)
    assert all(out[n]["valid"] == base[n]["valid"] for n in ("D2_lr", "D2_rf", "D3", "D4"))


def test_b12_ties_kept_sensitivity():
    t = np.array([0.0, 10.0, 10.0, 25.0])
    kept, idx, dropped = v7.clean_times(t, drop_ties=False)
    assert dropped == 0 and v7.gaps_of(kept).tolist() == [10.0, 0.0, 15.0]
    main, _, d2 = v7.clean_times(t)
    assert d2 == 1 and v7.gaps_of(main).tolist() == [10.0, 15.0]


def test_b12_token1_planned_no_earlier_than_determined():
    wished = np.array([0.0, 10.0, 20.0, 30.0, 35.0])       # token 1 (3 chars) spread over a_2 - a_1 = 30 (pre-B13 rule)
    det = np.array([30.0, 30.0, 30.0, 30.0, 30.0])         # token 1 known only at a_2 = 30
    e = v7.executable_plan(wished, det)
    assert e.tolist() == [30.0, 30.0, 30.0, 30.0, 35.0] and np.all(e >= det)
    sh = v7.plan_shift(wished, det)
    assert sh["share_moved"] == pytest.approx(3 / 5) and sh["median_shift"] == pytest.approx(20.0)
    with pytest.raises(ValueError):
        v7.executable_plan(wished, det[:3])


def test_b12_one_way_table():
    pr = v7.reading_probabilities(0.02, 80, one_way=True)
    assert sum(pr.values()) == pytest.approx(1.0)
    assert pr["not changed"] == pytest.approx(sum(math.comb(80, b) * 0.02 ** b * 0.98 ** (80 - b) for b in range(3)))


def test_b13_m1_schedule_delayed_by_one_token():
    # tokens arrive at 0, 30, 40, 100 (relative), with 3, 2, 1, 2 characters; generation ends at 120
    a = np.array([0.0, 30.0, 40.0, 100.0])
    n = np.array([3, 2, 1, 2])
    w, det = v7.m1_wished_times(a, n, 120.0)
    assert w.tolist() == [30.0, 40.0, 50.0, 40.0, 45.0, 100.0, 120.0, 130.0]
    assert np.all(w >= det) and det.tolist() == [30, 30, 30, 40, 40, 100, 120, 120]
    e = v7.executable_plan(w, det)
    assert e.tolist() == [30, 40, 50, 50, 50, 100, 120, 130]      # an early token overlaps: the running maximum
    sh = v7.plan_shift(w, det)
    assert sh["share_moved"] == pytest.approx(2 / 8)
    with pytest.raises(ValueError):
        v7.m1_wished_times(a, np.array([3, 0, 1, 2]), 120.0)
    with pytest.raises(ValueError):
        v7.m1_wished_times(a, n, 90.0)                          # end before the last arrival


def test_b12_review_fixes():
    assert v7.validity_gate.__defaults__[-1] is None                # no 1 ms default
    out = v7.validity_gate(humans(25, seed=11), key=[1, 2, 3])
    assert out["D4"]["gridded"]["human_machine_share"] is None and v7.grid_note(out, "D4") == "unknown"
    with pytest.raises(ValueError):
        v7.general_probability_bounds([0.9, float("nan")])
    assert v7.general_probability_bounds([1.0])["lower"] == 1.0
    assert not v7.general_cell_ok(0, 0, 34, 0)                    # fewer than 35 complete pairs
    t = np.array([0.0, 10.0, 10.0, 20.0])
    assert v7.sensitivity_variant(t, "grid", 1.0, drop_ties=False)[2] == 0
    assert v7.sensitivity_variant(t, "grid", 1.0)[2] == 1
    p0 = v7.reading_probabilities(0.0, 80, one_way=True)
    assert p0["not changed"] == 1.0
    one, two = v7.reading_probabilities(0.10, 80, one_way=True), v7.reading_probabilities(0.10, 80)
    assert one["responds"] > two["responds"]                      # one way, the difference reaches the margin sooner
