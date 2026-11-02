"""Prototype v7: the analysis functions of the path trap (works/conditions/prototype_v7.json, commit 041082c), written
before any capture exists and tested on synthetic series only (.steering/20261102-v7-analysis-code).

This file holds no run() and no main(): the format of the work PC's returns is not fixed yet, and no value of v7 has
been taken. Every function follows the wording of the conditions; where the conditions leave a detail open, the choice
is stated in the docstring and in decisions.md of .steering/20261102-v7-analysis-code.

Sections: series and cleaning; detectors D1, D2, D3 (detectors_C2) and their validity gate; counts and intervals
(Clopper-Pearson, cluster bootstrap, Tango's paired score interval, the headline); M3 generators; C1 (stages_C1:
grouping by relation, out-of-fold scores, rank_auc and its reading); replay error; timing sensitivity; resolution.
"""

from __future__ import annotations

import math
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if not sys.pycache_prefix:
    sys.pycache_prefix = tempfile.mkdtemp(prefix="pycache-")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "works"))

import numpy as np  # noqa: E402
from scipy import optimize, stats  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

import prototype_v6 as v6  # noqa: E402

SEED = 20261031
VERSION = 7
DELTA_T = 0.269          # Condrey's threshold on delta
TRIM_FACTOR = 10.0       # one pass: gaps above 10 x the mean of all gaps removed
PAUSE_MS, BURST_MS = 500.0, 150.0
HIST_BINS, HIST_MAX_MS = 50, 2000.0
UNIFORM_MS = (30.0, 80.0)   # Condrey's automated baseline
D2_SEED, D3_SEED = 42, 123
D2_TREES, D3_TREES = 100, 300
WINDOW = 70              # QUACK!-type window of consecutive gaps
N_BINS = 8               # M3b and D3's conditional generator
D4_SEED, D4_TREES = 28, 100   # amendment B4: the reduced BeCAPTCHA-Type
D4_L = 30                # first 30 press-press gaps of a session
D4_BW_MS = 1.0           # KDE bandwidth (the paper's sigma = 1.0, unit not stated)
M3_DONORS = 40           # amendment B2: M3's donors drawn without replacement, one per M0 text
GATE_AUC, GATE_HUMAN_SHARE = 0.90, 0.10
MARGIN = 0.10            # headline equivalence margin
HD3_LIMIT = 0.02         # amendment B11: Hd3's limit on the mean absolute paired change
P5_LIMIT_MS = 1.0        # amendment B11: P5, page-layer median with and without the hook
P_NOHOOK_MS = (5.0, 18.0, 50.0)   # amendment B14 (decision 38): P_nohook's controls without load (5 ms added)
H0Q_SHOWN = (("jitter", 2.0), ("jitter", 8.0))   # amendment B14 (decision 37): H0q versions beside 'responds'
N_BOOT = 2000
N_FOLDS = 5
Z95 = float(stats.norm.ppf(0.975))
LOGGER_KIND_TO_V6 = {"alnum": "alnum", "space": "space", "end": "end", "enter": "space", "backspace": "other",
                     "delete": "other", "other": "other"}


# ---------------------------------------------------------------- series and cleaning


def check_gaps(g: np.ndarray) -> np.ndarray:
    g = np.asarray(g, dtype=float)
    if g.ndim != 1 or not np.all(np.isfinite(g)) or np.any(g < 0):
        raise ValueError("gaps must be a finite, non-negative 1-d array")
    return g


def clean_times(times: np.ndarray, drop_ties: bool = True) -> tuple[np.ndarray, np.ndarray, int]:
    """v6's rule (matching_and_cleaning): stable sort, a press at the same time as the previous one dropped and
    counted; times from the first press. Returns the times, the original index of each kept press (so that kinds can
    follow: kinds[i] for i in index) and the number dropped."""
    t = np.asarray(times, dtype=float)
    if not np.all(np.isfinite(t)):
        raise ValueError("a press time is not finite")
    order = np.argsort(t, kind="stable")
    t = t[order]
    keep = np.concatenate([[True], np.diff(t) > 0]) if t.size else np.array([], dtype=bool)
    if not drop_ties:   # amendment B12 (decision 32): the sensitivity with ties kept (zero gaps retained), none dropped
        keep = np.ones(t.size, dtype=bool)
    t, idx = t[keep], order[keep]
    return (t - t[0] if t.size else t), idx, int((~keep).sum())


def gaps_of(times: np.ndarray) -> np.ndarray:
    return np.diff(np.asarray(times, dtype=float))


# ---------------------------------------------------------------- D1 and D2 (Condrey)


def trimmed(g: np.ndarray) -> np.ndarray:
    """One pass: gaps above TRIM_FACTOR x the mean of all gaps removed (D1's trim; D2 uses the same)."""
    g = check_gaps(g)
    if g.size == 0:
        raise ValueError("no gap")
    return g[g <= TRIM_FACTOR * g.mean()]


def delta(g: np.ndarray) -> float:
    """Condrey's delta: SD (ddof 0) / mean of the trimmed gaps."""
    t = trimmed(g)
    m = t.mean()
    if m <= 0:
        raise ValueError("mean gap is 0")
    return float(t.std(ddof=0) / m)


def d1_machine(g: np.ndarray) -> bool:
    """D1: 'human' if delta > 0.269, else 'machine'."""
    return not delta(g) > DELTA_T


def mean_burst_run(g: np.ndarray) -> float:
    """Mean length (in gaps) of maximal runs of consecutive gaps below BURST_MS; 0 when there is no such gap."""
    fast = g < BURST_MS
    runs, n = [], 0
    for f in fast:
        if f:
            n += 1
        elif n:
            runs.append(n)
            n = 0
    if n:
        runs.append(n)
    return float(np.mean(runs)) if runs else 0.0


def hist_entropy(g: np.ndarray) -> tuple[float, int]:
    """Shannon entropy (natural log) of a HIST_BINS-bin histogram on [0, HIST_MAX_MS]; gaps above HIST_MAX_MS are left
    out and their number returned. 0 log 0 = 0; no gap in range gives entropy 0."""
    inside = g[g <= HIST_MAX_MS]
    counts, _ = np.histogram(inside, bins=HIST_BINS, range=(0.0, HIST_MAX_MS))
    out = int(g.size - inside.size)
    if counts.sum() == 0:
        return 0.0, out
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log(p)).sum()), out


D2_FEATURES = ("delta", "mean_gap", "gap_variance", "share_above_500", "mean_burst_run", "entropy", "sd_of_diffs")


def condrey_features(g: np.ndarray) -> tuple[np.ndarray, int]:
    """D2's seven features on the trimmed gaps (order D2_FEATURES), and the number of trimmed gaps above
    HIST_MAX_MS left out of the histogram (counted, as the conditions ask)."""
    t = trimmed(g)
    if t.size < 2:
        raise ValueError("fewer than 2 gaps after the trim")
    h, out = hist_entropy(t)
    return np.array([t.std(ddof=0) / t.mean(), t.mean(), t.var(ddof=0), float(np.mean(t > PAUSE_MS)),
                     mean_burst_run(t), h, float(np.diff(t).std(ddof=0))]), out


def feature_matrix(sessions: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Features of each session and, per session in the same order, its gaps left out of the histogram."""
    rows = [condrey_features(g) for g in sessions]
    return np.array([r[0] for r in rows]), np.array([r[1] for r in rows], dtype=int)


def uniform_sessions(lengths: list[int], rng: np.random.Generator) -> list[np.ndarray]:
    """One synthetic session per length, gaps ~ Uniform(30, 80) ms (Condrey's baseline, no path)."""
    return [rng.uniform(*UNIFORM_MS, n) for n in lengths]


@dataclass
class D2:
    lr: object
    rf: object
    train_above_hist: np.ndarray   # per training session (humans, then synthetic), gaps left out of the histogram

    def proba(self, sessions: list[np.ndarray]) -> dict[str, np.ndarray]:
        """'machine' probabilities of both models; 'above_hist' = per session, the gaps left out of its histogram."""
        x, out = feature_matrix(sessions)
        return {"D2_lr": self.lr.predict_proba(x)[:, 1], "D2_rf": self.rf.predict_proba(x)[:, 1], "above_hist": out,
                "D2_lr_logit": self.lr.decision_function(x)}   # amendment B9: the paired change also on the logit


def train_d2(human: list[np.ndarray], rng: np.random.Generator) -> D2:
    """Human sessions (class 0) against as many Uniform(30, 80) sessions of matched length (class 1, 'machine').
    Both models standardised; LR with C 1; RF with D2_TREES trees; random_state D2_SEED. The synthetic sessions are
    drawn from rng (the caller passes numpy.random.default_rng(D2_SEED) or a fold's key)."""
    synth = uniform_sessions([g.size for g in human], rng)
    x, above = feature_matrix(human + synth)
    y = np.array([0] * len(human) + [1] * len(synth))
    lr = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000)).fit(x, y)
    rf = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=D2_TREES, random_state=D2_SEED)).fit(x, y)
    return D2(lr, rf, above)


# ---------------------------------------------------------------- D3 (QUACK!-type, hold omitted)


def windows_of(g: np.ndarray) -> np.ndarray:
    """Non-overlapping windows of WINDOW consecutive gaps from the start; features per window the WINDOW gaps and
    J_t = |gap_t - gap_{t-1}| (J_0 = 0), flattened. Shape (n_windows, 2 * WINDOW); a remainder below WINDOW is dropped."""
    g = check_gaps(g)
    n = g.size // WINDOW
    w = g[: n * WINDOW].reshape(n, WINDOW)
    j = np.concatenate([np.zeros((n, 1)), np.abs(np.diff(w, axis=1))], axis=1)
    return np.concatenate([w, j], axis=1)


@dataclass
class Generators:
    """D3's three generators fitted on pooled human gaps without any path."""
    lo: float
    hi: float
    pooled: np.ndarray
    cond: "CondBins"

    def session(self, which: int, n: int, rng: np.random.Generator) -> np.ndarray:
        if which == 0:
            return rng.uniform(self.lo, self.hi, n)
        if which == 1:
            return rng.choice(self.pooled, n, replace=True)
        return self.cond.draw(n, rng)[0]


class CondBins:
    """Second-order resampling (M3b; D3's third generator when pooled): N_BINS bins at the empirical quantiles of the
    gaps; the next gap drawn from the successors of gaps in the bin of the previous drawn gap. Successor pairs are taken
    within each session only. An empty bin falls back to the marginal and is counted."""

    def __init__(self, sessions: list[np.ndarray]):
        sessions = [check_gaps(s) for s in sessions]
        self.marginal = np.concatenate(sessions)
        if self.marginal.size == 0:
            raise ValueError("no gap")
        self.edges = np.quantile(self.marginal, np.arange(1, N_BINS) / N_BINS, method="linear")
        succ: list[list[float]] = [[] for _ in range(N_BINS)]
        for s in sessions:
            for b, nxt in zip(self.bin(s[:-1]), s[1:]):
                succ[int(b)].append(float(nxt))
        self.successors = [np.array(x) for x in succ]

    def bin(self, x) -> np.ndarray:
        return np.searchsorted(self.edges, np.asarray(x, dtype=float), side="right")

    def draw(self, n: int, rng: np.random.Generator) -> tuple[np.ndarray, int]:
        out = np.empty(n)
        fallback = 0
        for i in range(n):
            if i == 0:
                out[i] = rng.choice(self.marginal)
                continue
            s = self.successors[int(self.bin(out[i - 1]))]
            if s.size == 0:
                fallback += 1
                out[i] = rng.choice(self.marginal)
            else:
                out[i] = rng.choice(s)
        return out, fallback


def fit_generators(human: list[np.ndarray]) -> Generators:
    pooled = np.concatenate([check_gaps(g) for g in human])
    lo, hi = np.percentile(pooled, [1, 99], method="linear")
    return Generators(float(lo), float(hi), pooled, CondBins(human))


def d3_synthetic(human: list[np.ndarray], gen: Generators, rng: np.random.Generator,
                 with_labels: bool = False):
    """Equally many synthetic windows as the human sessions hold, one third from each generator ('one third each' of
    the windows): the N windows are permuted and cut into thirds, and each window is WINDOW gaps drawn afresh from its
    generator (the conditional generator starts each window from the marginal). The windows are grouped into one
    synthetic session per human session, holding as many windows as it (so each synthetic session weighs as its human
    session). A human session with no full window gets a synthetic session of its length from generator i % 3 (it
    enters no training window; it serves D1's comparison only). with_labels also returns the generator of each
    window, session by session."""
    nwin = [g.size // WINDOW for g in human]
    total = int(sum(nwin))
    which = np.empty(total, dtype=int)
    for k, part in enumerate(np.array_split(rng.permutation(total), 3)):
        which[part] = k
    out, labels, at = [], [], 0
    for i, (g, w) in enumerate(zip(human, nwin)):
        if w == 0:
            out.append(gen.session(i % 3, g.size, rng))
            labels.append(np.array([], dtype=int))
            continue
        lab = which[at: at + w]
        at += w
        out.append(np.concatenate([gen.session(int(k), WINDOW, rng) for k in lab]))
        labels.append(lab)
    return (out, labels) if with_labels else out


def stack_windows(sessions: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Windows of all sessions, their weights (1 / the session's number of windows) and their session index. Sessions
    with no full window contribute nothing."""
    xs, ws, sid = [], [], []
    for i, g in enumerate(sessions):
        w = windows_of(g)
        if len(w):
            xs.append(w)
            ws.append(np.full(len(w), 1.0 / len(w)))
            sid.append(np.full(len(w), i))
    if not xs:
        raise ValueError("no session has a full window")
    return np.concatenate(xs), np.concatenate(ws), np.concatenate(sid)


@dataclass
class D3:
    rf: object

    def proba(self, sessions: list[np.ndarray]) -> np.ndarray:
        """Session score: the mean 'machine' probability over its windows; NaN for a session with fewer than WINDOW
        gaps (no score, counted by the caller)."""
        out = np.full(len(sessions), np.nan)
        for i, g in enumerate(sessions):
            w = windows_of(g)
            if len(w):
                out[i] = float(self.rf.predict_proba(w)[:, 1].mean())
        return out


def train_d3(human: list[np.ndarray], rng: np.random.Generator) -> D3:
    """Human windows (class 0) against the windows of matched synthetic sessions (class 1); each session weighted
    equally; RF of D3_TREES trees, scikit-learn defaults otherwise, random_state D3_SEED."""
    gen = fit_generators(human)
    synth = d3_synthetic(human, gen, rng)
    xh, wh, _ = stack_windows(human)
    xm, wm, _ = stack_windows(synth)
    x = np.concatenate([xh, xm])
    y = np.concatenate([np.zeros(len(xh), dtype=int), np.ones(len(xm), dtype=int)])
    rf = RandomForestClassifier(n_estimators=D3_TREES, random_state=D3_SEED)
    rf.fit(x, y, sample_weight=np.concatenate([wh, wm]))
    return D3(rf)


# ---------------------------------------------------------------- D4 (reduced BeCAPTCHA-Type, amendment B4)


def d4_features(g: np.ndarray) -> np.ndarray | None:
    """The first D4_L press-press gaps as they are (no scaling); None for a session with fewer (no D4 score)."""
    g = check_gaps(g)
    return g[:D4_L].copy() if g.size >= D4_L else None


@dataclass
class KDE1:
    """The paper's Universal model on one feature: a Gaussian KDE (bandwidth D4_BW_MS) of pooled gaps; each draw is a
    pooled gap plus N(0, bandwidth); a draw below 0 is drawn again and counted."""
    pool: np.ndarray
    redrawn: int = 0

    def draw(self, n: int, rng: np.random.Generator) -> np.ndarray:
        out = np.empty(n)
        filled = 0
        while filled < n:
            k = n - filled
            x = self.pool[rng.integers(0, self.pool.size, k)] + rng.normal(0.0, D4_BW_MS, k)
            ok = x[x >= 0]
            self.redrawn += int(k - ok.size)
            out[filled:filled + ok.size] = ok
            filled += ok.size
        return out


def fit_kde1(human: list[np.ndarray]) -> KDE1:
    """Pooled first D4_L gaps of every human session with at least D4_L gaps (the segment D4 scores). With no such
    session it raises: the run stops rather than reporting D4 on nothing."""
    rows = [f for f in (d4_features(g) for g in human) if f is not None]
    if not rows:
        raise ValueError(f"no human session has {D4_L} gaps")
    return KDE1(np.concatenate(rows))


@dataclass
class D4:
    rf: object
    kde: KDE1

    def proba(self, sessions: list[np.ndarray]) -> np.ndarray:
        """'machine' probability per session; NaN for a session with fewer than D4_L gaps (counted by the caller)."""
        out = np.full(len(sessions), np.nan)
        rows = [(i, f) for i, f in enumerate(d4_features(g) for g in sessions) if f is not None]
        if rows:
            out[[i for i, _ in rows]] = self.rf.predict_proba(np.array([f for _, f in rows]))[:, 1]
        return out


def d4_synthetic(n_sessions: int, kde: KDE1, rng: np.random.Generator) -> list[np.ndarray]:
    return [kde.draw(D4_L, rng) for _ in range(n_sessions)]


def train_d4(human: list[np.ndarray], rng: np.random.Generator) -> D4:
    """Human sessions with at least D4_L gaps (class 0, their first D4_L gaps) against as many synthetic sessions of
    D4_L gaps from the KDE fitted on them (class 1); RF of D4_TREES trees, scikit-learn defaults otherwise,
    random_state D4_SEED. The caller passes numpy.random.default_rng(D4_SEED) or a fold's key."""
    kde = fit_kde1(human)
    xh = np.array([f for f in (d4_features(g) for g in human) if f is not None])
    xm = np.array(d4_synthetic(len(xh), kde, rng))
    x = np.concatenate([xh, xm])
    y = np.concatenate([np.zeros(len(xh), dtype=int), np.ones(len(xm), dtype=int)])
    rf = RandomForestClassifier(n_estimators=D4_TREES, random_state=D4_SEED).fit(x, y)
    return D4(rf, kde)


# ---------------------------------------------------------------- validity gate


def rank_auc(score: np.ndarray, label: np.ndarray) -> float:
    """P(score of a class-1 item > score of a class-0 item), ties 1/2. NaN scores are an error."""
    score, label = np.asarray(score, dtype=float), np.asarray(label, dtype=int)
    if not np.all(np.isfinite(score)):
        raise ValueError("a score is not finite")
    pos, neg = score[label == 1], score[label == 0]
    if pos.size == 0 or neg.size == 0:
        raise ValueError("one class is empty")
    r = stats.rankdata(np.concatenate([pos, neg]))
    return float((r[: pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def person_folds(n: int, key: list[int]) -> np.ndarray:
    """Fold of each person: a permutation from key, cut into N_FOLDS near-equal parts."""
    fold = np.empty(n, dtype=int)
    for k, part in enumerate(np.array_split(np.random.default_rng(key).permutation(n), N_FOLDS)):
        fold[part] = k
    return fold


def gridded(g: np.ndarray, grid_ms: float) -> np.ndarray:
    """A gap series with its press times (from 0) rounded to grid_ms, cleaned by v6's rule, gaps taken again
    (amendment B5)."""
    t = np.concatenate([[0.0], np.cumsum(check_gaps(g))])
    return gaps_of(clean_times(quantise(t, grid_ms))[0])


def validity_gate(human: list[np.ndarray], key: list[int] | None = None, grid_ms: float | None = None) -> dict:
    """Within E: N_FOLDS folds by person; per fold D2 and D3 trained on the other folds with their synthetic sessions
    drawn there; the held-out humans scored against held-out synthetic sessions of matched length from the same
    generators (D2: Uniform(30, 80); D3: the three generators fitted on the training folds). D1 has no training and is
    scored (machine score = -delta) against both synthetic sets. Held-out scores are pooled over folds into one AUC
    and one share of held-out humans judged 'machine' per detector. A detector is 'valid' if AUC >= GATE_AUC and the
    share <= GATE_HUMAN_SHARE.

    Random streams (analysis_details.gate_streams): the folds from key [20261031, 7, 8]; in fold k the training
    synthetic sessions of D2 from default_rng([D2_SEED, k + 1]) and of D3 from default_rng([D3_SEED, k + 1]), the
    held-out synthetic sessions of D2 from default_rng([D2_SEED, k + 1, 1]) and of D3 from
    default_rng([D3_SEED, k + 1, 1]) (amendment B10: [seed, 0] would repeat the full training stream of seed).

    D4 (amendment B4) the same way: training synthetic from default_rng([D4_SEED, k + 1]), held-out from
    default_rng([D4_SEED, k + 1, 1]), its KDE fitted on the training folds only, one held-out synthetic session per
    held-out human session with at least D4_L gaps. out['D4']['no_score'] counts held-out humans with too few gaps;
    out['D4']['redrawn'] the KDE draws below 0 drawn again, training and held-out apart.

    Amendment B5 (descriptive): out[name]['gridded'] holds the AUC and the held-out human 'machine' share when the
    same trained detectors score the held-out synthetic sessions with their press times rounded to grid_ms (KUPA's
    effective resolution estimated from H0 of E). Amendment B12: with grid_ms None (the default; no resolution read)
    no gridded version is computed, its AUC and share are None and grid_note reads 'unknown'; never 1 ms by default.
    It does not decide validity."""
    key = key or [SEED, VERSION, 8]
    fold = person_folds(len(human), key)
    pooled: dict[str, dict[str, list]] = {d: {"score": [], "label": []}
                                          for d in ("D1_uniform", "D1_d3gen", "D2_lr", "D2_rf", "D3", "D4")}
    pooled_g = {d: {"score": [], "label": []} for d in pooled}
    d3_short = d4_short = 0
    no_score_g = {d: 0 for d in ("D1_uniform", "D1_d3gen", "D2_lr", "D2_rf", "D3", "D4")}
    redrawn = {"train": 0, "held_out": 0}
    for k in range(N_FOLDS):
        train = [g for g, f in zip(human, fold) if f != k]
        test = [g for g, f in zip(human, fold) if f == k]
        d2 = train_d2(train, np.random.default_rng([D2_SEED, k + 1]))
        d3 = train_d3(train, np.random.default_rng([D3_SEED, k + 1]))
        syn2 = uniform_sessions([g.size for g in test], np.random.default_rng([D2_SEED, k + 1, 1]))
        syn3 = d3_synthetic(test, fit_generators(train), np.random.default_rng([D3_SEED, k + 1, 1]))
        d4 = train_d4(train, np.random.default_rng([D4_SEED, k + 1]))
        kde_out = fit_kde1(train)
        syn4 = d4_synthetic(sum(g.size >= D4_L for g in test), kde_out, np.random.default_rng([D4_SEED, k + 1, 1]))
        redrawn["train"] += d4.kde.redrawn
        redrawn["held_out"] += kde_out.redrawn
        lab = [0] * len(test) + [1] * len(test)
        # amendment B12 (decision 30): grid_ms None (no resolution read) -> no gridded version, its AUC 'unknown'
        g2, g3, g4 = ([gridded(g, grid_ms) for g in ss] if grid_ms is not None else [] for ss in (syn2, syn3, syn4))
        # a gridded session left with fewer than 2 gaps has no D1 or D2 score: drop it and count it (all detectors)
        for name_, gs in (("D1_uniform", g2), ("D2_lr", g2), ("D2_rf", g2), ("D1_d3gen", g3), ("D3", g3), ("D4", g4)):
            no_score_g[name_] += sum(g.size < 2 for g in gs)
        g2, g3, g4 = ([g for g in gs if g.size >= 2] for gs in (g2, g3, g4))
        for pool, s2, s3, s4 in ((pooled, syn2, syn3, syn4), (pooled_g, g2, g3, g4)):
            for name, ss in (("D1_uniform", s2), ("D1_d3gen", s3)):
                pool[name]["score"] += [-delta(g) for g in test + ss]
                pool[name]["label"] += [0] * len(test) + [1] * len(ss)
            p2 = d2.proba(test + s2)
            for name in ("D2_lr", "D2_rf"):
                pool[name]["score"] += list(p2[name])
                pool[name]["label"] += [0] * len(test) + [1] * len(s2)
            for name, model, ss in (("D3", d3, s3), ("D4", d4, s4)):
                p = model.proba(test + ss)
                lab_m = [0] * len(test) + [1] * len(ss)
                ok = np.isfinite(p)
                if pool is pooled:
                    if name == "D3":
                        d3_short += int((~ok).sum())
                    else:
                        d4_short += int((~ok).sum())
                else:
                    no_score_g[name] += int((~ok[len(test):]).sum())   # gridded synthetic below the window (humans: no_score)
                pool[name]["score"] += list(p[ok])
                pool[name]["label"] += list(np.array(lab_m)[ok])
    out = {}
    for name, v in pooled.items():
        s, y = np.array(v["score"]), np.array(v["label"])
        auc = rank_auc(s, y)
        if name.startswith("D1"):
            machine = s[y == 0] >= -DELTA_T          # -delta >= -0.269  <=>  delta <= 0.269
        else:
            machine = s[y == 0] > 0.5
        share = float(machine.mean())
        out[name] = {"auc": auc, "human_machine_share": share, "n_human": int((y == 0).sum()),
                     "n_machine": int((y == 1).sum()),
                     "valid": bool(auc >= GATE_AUC and share <= GATE_HUMAN_SHARE)}
        sg, yg = np.array(pooled_g[name]["score"]), np.array(pooled_g[name]["label"])
        mg = sg[yg == 0] >= -DELTA_T if name.startswith("D1") else sg[yg == 0] > 0.5
        both = (yg == 0).any() and (yg == 1).any()
        out[name]["gridded"] = {"grid_ms": grid_ms, "auc": rank_auc(sg, yg) if both else None,   # None: one class empty
                                "human_machine_share": float(mg.mean()) if mg.size and grid_ms is not None else None,
                                "n_human": int((yg == 0).sum()), "n_machine": int((yg == 1).sum()),
                                "no_score": no_score_g[name]}   # gridded synthetic sessions without a score; B5: never decides
    out["D3"]["no_score"] = d3_short
    out["D4"]["no_score"] = d4_short
    out["D4"]["redrawn"] = redrawn
    out["D1"] = dict(out["D1_uniform"], against="Uniform(30, 80) (Condrey's baseline); D1_d3gen reported beside")
    return out


# ---------------------------------------------------------------- counts and intervals


def clopper_pearson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    if n <= 0 or not 0 <= k <= n:
        raise ValueError("need 0 <= k <= n, n > 0")
    a = (1 - level) / 2
    lo = 0.0 if k == 0 else float(stats.beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1 - a, k + 1, n - k))
    return lo, hi


def cluster_bootstrap_share(machine: np.ndarray, cluster: np.ndarray, key: list[int]) -> tuple[float, float, int]:
    """95% percentile interval of the share of sessions judged 'machine' over N_BOOT resamples of clusters (texts or
    donors) with replacement; every session of a drawn cluster enters. Returns (lo, hi, n_clusters)."""
    machine, cluster = np.asarray(machine), np.asarray(cluster)
    if machine.dtype != bool or machine.ndim != 1 or cluster.shape != machine.shape:
        raise TypeError("judgments must be a boolean 1-d array with one cluster each (remove missing ones first)")
    ids = np.unique(cluster)
    by = [machine[cluster == c] for c in ids]
    rng = np.random.default_rng(key)
    shares = np.empty(N_BOOT)
    for b in range(N_BOOT):
        pick = rng.integers(0, ids.size, ids.size)
        x = np.concatenate([by[i] for i in pick])
        shares[b] = x.mean()
    return float(np.percentile(shares, 2.5)), float(np.percentile(shares, 97.5)), int(ids.size)


def tango_z(delta_: float, b: int, c: int, n: int) -> float:
    """Tango's (1998) score statistic for H0: p_b - p_c = delta_, with b = discordant pairs 0 -> 1 (human -> machine)
    and c = 1 -> 0; the nuisance p_c by its restricted maximum likelihood."""
    a_ = 2.0 * n
    b_ = -b - c + (2.0 * n - b + c) * delta_
    c_ = -c * delta_ * (1.0 - delta_)
    disc = max(b_ * b_ - 4.0 * a_ * c_, 0.0)
    p21 = (math.sqrt(disc) - b_) / (2.0 * a_)
    var = n * (2.0 * p21 + delta_ * (1.0 - delta_))
    num = b - c - n * delta_
    if var <= 0:
        return 0.0 if num == 0 else math.copysign(math.inf, num)
    return num / math.sqrt(var)


def tango_interval(b: int, c: int, n: int, z: float = Z95) -> tuple[float, float]:
    """The set of delta in [-1, 1] with |Z(delta)| <= z; Z decreases in delta, so each end is one root."""
    if n <= 0 or b < 0 or c < 0 or b + c > n:
        raise ValueError("need b, c >= 0 and b + c <= n")
    d = (b - c) / n
    eps = 1e-12
    lo = -1.0 if tango_z(-1.0 + eps, b, c, n) <= z else optimize.brentq(
        lambda x: tango_z(x, b, c, n) - z, -1.0 + eps, d, xtol=1e-12)
    hi = 1.0 if tango_z(1.0 - eps, b, c, n) >= -z else optimize.brentq(
        lambda x: tango_z(x, b, c, n) + z, d, 1.0 - eps, xtol=1e-12)
    return float(lo), float(hi)


def flips(before: np.ndarray, after: np.ndarray) -> tuple[int, int, int]:
    """Paired 'machine' judgments (True = machine) of the same persons: b = human -> machine, c = machine -> human,
    n = pairs. Only boolean arrays are accepted (a NaN or None would otherwise become a judgment silently); pairs
    without a judgment are removed before, e.g. by judged_pairs."""
    before, after = np.asarray(before), np.asarray(after)
    if before.dtype != bool or after.dtype != bool:
        raise TypeError("judgments must be boolean arrays")
    if before.shape != after.shape:
        raise ValueError("unpaired")
    return int((~before & after).sum()), int((before & ~after).sum()), int(before.size)


def judged_pairs(p_before: np.ndarray, p_after: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """'machine' probabilities of the same persons -> paired judgments (probability > 0.5), the pairs where either
    side has no score (NaN: D3's sessions below WINDOW gaps) removed and counted."""
    a, b = np.asarray(p_before, dtype=float), np.asarray(p_after, dtype=float)
    if a.shape != b.shape:
        raise ValueError("unpaired")
    ok = np.isfinite(a) & np.isfinite(b)
    return a[ok] > 0.5, b[ok] > 0.5, int((~ok).sum())


def wilson_interval(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson's score interval of a proportion k / n (amendment A5: the discordance (b + c) / n)."""
    if n <= 0 or not 0 <= k <= n:
        raise ValueError("need 0 <= k <= n, n > 0")
    p = k / n   # k and n may be non-integer (machine_path_reading's effective counts)
    den = 1.0 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, mid - half), min(1.0, mid + half)


def read_rule(diff_ci: tuple[float, float], disc_ci: tuple[float, float]) -> str:
    """The reading of detectors_C2.headline (decision 10, amendment A5), shared by the human rows (Tango and Wilson)
    and the machine_path rows (cluster bootstrap): 'not changed' needs the difference's interval inside
    [-MARGIN, MARGIN] and the discordance's upper end at most MARGIN; 'responds' (amendment B3: 'responds beyond 0.10')
    if the whole of the difference's interval is beyond MARGIN on one side, or the discordance's lower end is above
    MARGIN; else 'undecided'. 'changed from 0' is reported beside by changed_from_0."""
    return read_rule_route(diff_ci, disc_ci)[0]


EPS_EDGE = 1e-12   # the bounds are closed: an interval end equal to MARGIN (up to rounding) is inside


def read_rule_route(diff_ci: tuple[float, float], disc_ci: tuple[float, float]) -> tuple[str, str]:
    """read_rule with the route of a 'responds': 'difference' (Tango's or the bootstrap interval of (b - c) / n),
    'discordance' (the lower end of (b + c) / n above MARGIN) or 'both'; '' for the other readings."""
    lo, hi = diff_ci
    dlo, dhi = disc_ci
    if lo >= -MARGIN - EPS_EDGE and hi <= MARGIN + EPS_EDGE and dhi <= MARGIN + EPS_EDGE:
        return "not changed", ""
    by_diff = lo > MARGIN + EPS_EDGE or hi < -MARGIN - EPS_EDGE   # amendment B3: the whole interval beyond the margin
    by_disc = dlo > MARGIN + EPS_EDGE
    if by_diff or by_disc:
        return "responds", "both" if by_diff and by_disc else ("difference" if by_diff else "discordance")
    return "undecided", ""


def changed_from_0(diff_ci: tuple[float, float]) -> bool:
    """Amendment B3, the first step: the difference's interval excludes 0 (it can stand with 'undecided')."""
    return bool(diff_ci[0] > 0 or diff_ci[1] < 0)


def headline(b: int, c: int, n: int) -> dict:
    """Two-sided equivalence (decision 10) with the discordance as a second route (amendment A5): Tango's interval of
    (b - c) / n and Wilson's interval of (b + c) / n, read by read_rule. Flips are reported beside."""
    lo, hi = tango_interval(b, c, n)
    dlo, dhi = wilson_interval(b + c, n)
    reading, route = read_rule_route((lo, hi), (dlo, dhi))
    return {"b": b, "c": c, "n": n, "difference": (b - c) / n, "tango_95": [lo, hi],
            "discordance": (b + c) / n, "wilson_95": [dlo, dhi], "reading": reading, "route": route,
            "changed_from_0": changed_from_0((lo, hi))}


def machine_path_reading(before: np.ndarray, after: np.ndarray, cluster: np.ndarray, key: list[int]) -> dict:
    """The machine_path rows (amendment A2): paired 'machine' judgments of the same machine sessions, planned (before)
    and recorded (after), with their clusters (texts, prompts or donors). b, c, n; the difference (b - c) / n and the
    discordance (b + c) / n with 95% percentile intervals over N_BOOT resamples of clusters with replacement (every
    session of a drawn cluster enters, so sessions are weighted equally), and the human side's intervals (Tango's for
    the difference, Wilson's for the discordance) computed with the number of clusters as the effective n and b, c
    scaled to it. The row is read by read_rule on each pair of intervals separately; their common reading is the
    row's reading, and if they disagree the row reads 'undecided' (code review and Codex, 2026-11-02: the bootstrap
    alone let one flip read 'not changed', and a hull of the two could create a 'responds' that neither reads). One
    flip of 40 reads 'undecided', and one cluster gives an effective-n interval wider than 1.5. Cluster ids must be
    finite (a NaN id would drop rows from the bootstrap silently). Only boolean
    judgments are accepted (remove missing ones first, e.g. by judged_pairs). The key is [20261031, 7, 11, layer,
    detector, series] with layer, detector as in analysis_details.keys and series 0 M0, 1 M1_8B, 2 M1_14B,
    3 M1_replay of 8B, 4 M1_replay of 14B, 5 M2_8B, 6 M2_14B, 7 M3a, 8 M3b."""
    before, after, cluster = np.asarray(before), np.asarray(after), np.asarray(cluster)
    if before.dtype != bool or after.dtype != bool:
        raise TypeError("judgments must be boolean arrays")
    if cluster.dtype.kind not in "iu":
        raise TypeError("cluster ids must be integers (no NaN or missing id)")
    if before.ndim != 1 or before.shape != after.shape or cluster.shape != before.shape:
        raise ValueError("before, after and cluster must be 1-d arrays of the same sessions")
    b, c, n = flips(before, after)
    if n == 0:
        raise ValueError("no pair")
    up = (~before & after).astype(float)
    down = (before & ~after).astype(float)
    ids = np.unique(cluster)
    rows = [np.flatnonzero(cluster == i) for i in ids]
    rng = np.random.default_rng(key)
    diff = np.empty(N_BOOT)
    disc = np.empty(N_BOOT)
    for k in range(N_BOOT):
        idx = np.concatenate([rows[i] for i in rng.integers(0, ids.size, ids.size)])
        diff[k] = (up[idx] - down[idx]).mean()
        disc[k] = (up[idx] + down[idx]).mean()
    boot_d = (float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5)))
    boot_s = (float(np.percentile(disc, 2.5)), float(np.percentile(disc, 97.5)))
    n_eff = float(ids.size)
    t_lo, t_hi = tango_interval(b * n_eff / n, c * n_eff / n, n_eff)
    w_lo, w_hi = wilson_interval((b + c) * n_eff / n, n_eff)
    r_boot, route_boot = read_rule_route(boot_d, boot_s)
    r_eff, route_eff = read_rule_route((t_lo, t_hi), (w_lo, w_hi))
    reading = r_boot if r_boot == r_eff else "undecided"
    route = route_eff if reading == "responds" else ""
    return {"b": b, "c": c, "n": n, "n_clusters": int(ids.size), "difference": (b - c) / n,
            "discordance": (b + c) / n,
            "bootstrap_95": {"difference": list(boot_d), "discordance": list(boot_s), "reading": r_boot},
            "effective_n_95": {"difference": [t_lo, t_hi], "discordance": [w_lo, w_hi], "reading": r_eff},
            "reading": reading, "route": route,
            "changed_from_0": changed_from_0(boot_d) and changed_from_0((t_lo, t_hi))}   # common to both, as the reading


def zero_flip_min_clusters(n_max: int = 1000) -> int:
    """Amendment B8: the smallest effective number of clusters at which no flip (b = c = 0) reads 'not changed'."""
    for n in range(1, n_max + 1):
        if headline(0, 0, n)["reading"] == "not changed":
            return n
    raise ValueError("no n reads 'not changed'")


def zero_flip_table(ns: range = range(10, 41)) -> list[dict]:
    """Amendment B8, the design table of the supplement: per effective n, the zero-flip intervals and reading."""
    return [{"n_eff": n, "tango_95": list(tango_interval(0, 0, n)), "wilson_95": list(wilson_interval(0, n)),
             "reading": headline(0, 0, n)["reading"]} for n in ns]


def enters_machine_general(n_clusters: int) -> bool:
    """Amendment B8: a machine series enters a machine-side general statement only with enough clusters left."""
    return n_clusters >= zero_flip_min_clusters()


def machine_general(readings: dict[tuple[str, str, str], str], valid: dict[str, bool],
                    n_clusters: dict[tuple[str, str, str], int], series: list[str],
                    layers: tuple[str, ...] = ("hook", "page"), *,
                    missing_pairs: dict[tuple[str, str, str], int]) -> dict:
    """Amendment B8: the machine-side general statement. readings[(detector, layer, series)] -> the machine_path
    reading; valid[detector] from the gate; n_clusters[(detector, layer, series)] the clusters of that cell's complete
    pairs. Amendment B14 (decision 35): missing_pairs[(detector, layer, series)] counts the pairs of that cell whose
    plan exists but whose capture failed or that have no judgment (e.g. D4's sessions below 30 gaps); sessions without
    a plan are not pairs and only lower the clusters. A series enters only if every valid detector on every layer has
    a reading, enough clusters (enters_machine_general) and no missing pair in its cell (one missing pair placed as a
    flip reads 'undecided'); a cell without a count raises. The statement holds if every detector is valid
    (analysis_details.across) and every entering series reads 'not changed' in every such cell. Series kept out are listed apart, with their 'responds' cells, so that
    the narrowing of the statement stays visible."""
    vd = [d for d in valid if valid[d]]
    cells = {s_: [(d, la, s_) for d in vd for la in layers] for s_ in series}
    missing = [list(c) for s_ in series for c in cells[s_]
               if c not in readings or c not in n_clusters or c not in missing_pairs]
    if missing:
        raise ValueError(f"readings, cluster counts or missing-pair counts missing: {missing}")
    if any(missing_pairs[c] < 0 for s_ in series for c in cells[s_]):
        raise ValueError("a missing-pair count is negative")
    enter = [s_ for s_ in series
             if all(enters_machine_general(n_clusters[c]) and missing_pairs[c] == 0 for c in cells[s_])]
    kept = [s_ for s_ in series if s_ not in enter]
    # analysis_details.across (amendment B11): the machine side's general statement also needs all detectors valid
    general = (len(vd) == len(valid) and bool(vd) and bool(enter)
               and all(readings[c] == "not changed" for s_ in enter for c in cells[s_]))
    responds = [list(c) for s_ in enter for c in cells[s_] if readings[c] == "responds"]
    responds_kept = [list(c) for s_ in kept for c in cells[s_] if readings[c] == "responds"]
    return {"entering": enter, "kept_out": kept, "general_not_moved": general, "responds": responds,
            "responds_kept_out": responds_kept}


def reading_probabilities(rate: float, n: int, one_way: bool = False) -> dict[str, float]:
    """Amendment B11 (decision 19): the probabilities of the three readings of one cell when each person flips with
    probability `rate`, half each way (or, amendment B12, all one way, b only), by enumerating every (b, c).
    not_changed_probability is the first of them (half each way)."""
    if not 0 <= rate < 1 or n <= 0:
        raise ValueError("need 0 <= rate < 1 and n > 0")
    out = {"not changed": 0.0, "responds": 0.0, "undecided": 0.0}
    if one_way:
        for b in range(n + 1):
            pr = math.comb(n, b) * rate ** b * (1 - rate) ** (n - b)
            out[headline(b, 0, n)["reading"]] += pr
        return out
    half = rate / 2
    for b in range(n + 1):
        for c in range(n - b + 1):
            if half == 0:
                pr = float(b == c == 0)
            else:
                pr = math.exp(math.lgamma(n + 1) - math.lgamma(b + 1) - math.lgamma(c + 1) - math.lgamma(n - b - c + 1)
                              + (b + c) * math.log(half) + (n - b - c) * math.log1p(-rate))
            out[headline(b, c, n)["reading"]] += pr
    return out


def boundary_readings(b: int, c: int, n: int, missing: int) -> dict:
    """Amendment B11 (decision 23): the reading of the complete pairs beside three readings that put the `missing`
    pairs back as no flips, all as flips b (human -> machine) and all as flips c. With no complete pair the complete
    reading is None and the three are still given."""
    if missing < 0 or n < 0 or n + missing == 0:
        raise ValueError("need missing >= 0, n >= 0 and at least one pair")
    return {"complete": headline(b, c, n)["reading"] if n > 0 else None,
            "missing_as_no_flip": headline(b, c, n + missing)["reading"],
            "missing_as_b": headline(b + missing, c, n + missing)["reading"],
            "missing_as_c": headline(b, c + missing, n + missing)["reading"]}


GATE_ENTRY = {"D1": "D1_uniform", "D2_lr": "D2_lr", "D2_rf": "D2_rf", "D3": "D3", "D4": "D4"}   # D1's validity: Uniform


def grid_note(gate: dict, detector: str) -> str:
    """Amendments B11, B12 (decisions 25, 30): 'note' if the detector's gridded validity version has an AUC below GATE_AUC
    (the note 'possible dependence on the recording grid' goes on every cell of the detector, whatever its reading;
    the H0 to H0q flips are shown beside each human-side 'responds' cell by the reporting step), 'no note' if not,
    'unknown' (on every cell too) if the
    gridded AUC could not be computed (None or NaN: reported as such, never read as 'no dependence'). D1 is read on
    D1_uniform (analysis_details.D1_gate). The held-out humans are not gridded, so only the AUC can raise the note."""
    auc = gate[GATE_ENTRY[detector]]["gridded"]["auc"]
    if auc is None or not np.isfinite(auc):
        return "unknown"
    return "note" if auc < GATE_AUC else "no note"


def executable_plan(wished: np.ndarray, determined: np.ndarray | None = None) -> np.ndarray:
    """Amendment B11 (decision 22): M1's wished times can go backwards; the executable plan is their running maximum
    (the character order kept, equal times allowed). Each wished time is first raised to the time it is determined
    (`determined`; under B13, m1_wished_times gives a_{k+1} for token k, so the raise changes nothing there), so no
    press is planned before it is known. It is the machine side's plan and is not cleaned (amendment B14, decision 36:
    its zero gaps are kept and counted by plan_zero_gap_share); the difference is reported (plan_shift)."""
    w = np.asarray(wished, dtype=float)
    if w.ndim != 1 or not np.all(np.isfinite(w)):
        raise ValueError("wished times must be a finite 1-d array")
    if determined is not None:
        dt = np.asarray(determined, dtype=float)
        if dt.shape != w.shape or not np.all(np.isfinite(dt)):
            raise ValueError("determined times must match the wished times")
        w = np.maximum(w, dt)
    return np.maximum.accumulate(w) if w.size else w


def m1_wished_times(arrivals: np.ndarray, n_chars: np.ndarray, end: float) -> tuple[np.ndarray, np.ndarray]:
    """Amendment B13 (decision 33): M1's schedule delayed by one token. Token k (arrival a_k, n_k characters; empty
    tokens skipped by the caller and counted) is spread from a_{k+1} over a_{k+1} - a_k: wished times
    a_{k+1} + j (a_{k+1} - a_k) / n_k, j = 0..n_k-1, with a_{K+1} = end (the end of generation). Times relative to a_1.
    Returns the wished times and the times they are determined (a_{k+1}); every wished time is >= its determined time."""
    a = np.asarray(arrivals, dtype=float)
    n = np.asarray(n_chars, dtype=int)
    if a.ndim != 1 or a.shape != n.shape or a.size == 0 or np.any(n < 1) or not np.all(np.isfinite(a)):
        raise ValueError("need matching 1-d arrivals and positive character counts")
    nxt = np.append(a[1:], float(end))
    if np.any(np.diff(np.append(a, float(end))) < 0):
        raise ValueError("arrivals and end must not go backwards")
    wished = np.concatenate([nx + np.arange(k) * (nx - ak) / k for ak, nx, k in zip(a, nxt, n)]) - a[0]
    det = np.repeat(nxt, n) - a[0]
    return wished, det


def plan_shift(wished: np.ndarray, determined: np.ndarray | None = None) -> dict:
    """Amendment B11: what the executable plan changed, reported apart: the share of presses moved and the median shift
    (executable minus wished) over the moved presses (None if none moved)."""
    w = np.asarray(wished, dtype=float)
    d = executable_plan(w, determined) - w
    moved = d > 0
    return {"share_moved": float(moved.mean()) if w.size else 0.0,
            "median_shift": float(np.median(d[moved])) if moved.any() else None}


def general_probability_bounds(cell_probs: list[float]) -> dict:
    """Amendment B12: the Frechet bounds on the probability that every cell reads 'not changed', from the per-cell
    probabilities p_i: lower max(0, sum p_i - (m - 1)), upper min p_i; the product (cells independent) is an example
    of a dependence structure, not a bound."""
    pr = np.asarray(cell_probs, dtype=float)
    if pr.ndim != 1 or pr.size == 0 or not np.all(np.isfinite(pr)) or np.any((pr < 0) | (pr > 1)):
        raise ValueError("need a 1-d array of finite probabilities in [0, 1]")
    return {"lower": float(max(0.0, pr.sum() - (pr.size - 1))), "upper": float(pr.min()),
            "independent_example": float(np.prod(pr))}


def general_cell_ok(b: int, c: int, n: int, missing: int) -> bool:
    """Amendment B12 (decision 29): a cell enters the general statement (and the abstract's 'not changed') only if its
    complete-pair reading and all three missing-data bounds read 'not changed'."""
    if n < zero_flip_min_clusters():   # fewer than 35 complete pairs: kept out (headline B11), stated explicitly
        return False
    r = boundary_readings(b, c, n, missing)
    return all(v == "not changed" for v in r.values())


def p5_passes(median_with_hook_ms: float, median_without_hook_ms: float) -> bool:
    """Amendment B11, prediction P5: the page layer's median recorded gap without the hook within P5_LIMIT_MS of that
    with the hook (P_NOHOOK_MS: 5, 18 and 50 ms controls without load, amendment B14; each checked on its own)."""
    return abs(median_with_hook_ms - median_without_hook_ms) <= P5_LIMIT_MS + EPS_EDGE


def floor_abs_change(before: np.ndarray, after: np.ndarray, key: list[int]) -> dict:
    """Amendment B11 (decision 24), Hd3: the mean over persons of |after - before| decides (at most HD3_LIMIT); the
    signed mean is the bias, with paired_prob_change's percentile interval, and does not decide."""
    before, after = np.asarray(before, dtype=float), np.asarray(after, dtype=float)
    bias = paired_prob_change(before, after, key)   # raises if no complete pair
    ok = np.isfinite(before) & np.isfinite(after)
    mabs = float(np.abs(after[ok] - before[ok]).mean())
    # the pairs with a missing score are counted (bias['removed']); fewer than FLOOR_MIN_PERSONS: not read
    read = bias["n"] >= FLOOR_MIN_PERSONS
    return {"mean_abs": mabs, "passes": (mabs <= HD3_LIMIT + EPS_EDGE) if read else None, "bias": bias,
            "n": bias["n"], "removed": bias["removed"]}



def e_split(n_e: int) -> tuple[np.ndarray, np.ndarray]:
    """Amendment B11 (decision 20): E's indices split into the 40 M3 donors (m3_donors, text order) and E_train
    (every other index, ascending), on which D2, D3, D4 and their validity gate are trained."""
    donors = m3_donors(n_e)
    train = np.setdiff1d(np.arange(n_e), donors)
    return donors, train


def saturation_share(p: np.ndarray) -> float:
    """Amendment B9: the share of scored sessions with a 'machine' probability at most 0.01 or at least 0.99."""
    p = np.asarray(p, dtype=float)
    p = p[np.isfinite(p)]
    if p.size == 0:
        raise ValueError("no score")
    return float(((p <= 0.01) | (p >= 0.99)).mean())


def not_changed_probability(rate: float, n: int) -> float:
    """Design table of amendment A5: the probability that one cell reads 'not changed' when each person flips with
    probability `rate`, half each way (b and c multinomial), computed by enumerating every (b, c)."""
    if not 0 <= rate <= 1 or n <= 0:
        raise ValueError("need 0 <= rate <= 1 and n > 0")
    if rate == 1:   # every person flips (b + c = n > 0): log1p(-1) is not defined, the cells are counted directly
        return float(sum(math.comb(n, b) * 0.5 ** n for b in range(n + 1) if headline(b, n - b, n)["reading"] == "not changed"))
    half = rate / 2
    total = 0.0
    for b in range(n + 1):
        for c in range(n - b + 1):
            if headline(b, c, n)["reading"] != "not changed":
                continue
            total += math.exp(math.lgamma(n + 1) - math.lgamma(b + 1) - math.lgamma(c + 1) - math.lgamma(n - b - c + 1)
                              + (b + c) * math.log(half) + (n - b - c) * math.log1p(-rate)) if half > 0 else float(b == c == 0)
    return total


def threshold_margin(detector: str, scores: np.ndarray) -> np.ndarray:
    """Each session's margin to its detector's threshold on the 'human' side (amendment A4): D1 delta - 0.269, D2 and
    D3 0.5 - 'machine' probability; NaN kept (no score). Positive = judged human. A margin of exactly 0 means 'machine'
    for D1 ('human' needs delta > 0.269) and 'human' for D2 and D3 ('machine' needs probability > 0.5)."""
    s = np.asarray(scores, dtype=float)
    if detector == "D1":
        return s - DELTA_T
    if detector in ("D2_lr", "D2_rf", "D3", "D4"):
        return 0.5 - s
    raise ValueError(f"unknown detector: {detector}")


def above_floor(h0: np.ndarray, h1: np.ndarray, h1_rep: np.ndarray, key: list[int]) -> dict:
    """The floor of pairs.replay_repeat (amendment A3, made explicit after Codex's point 4): on the same persons (the
    H1_rep ones), the per-person absolute score change H0 -> H1_idle against H1_idle -> H1_rep; 'above the floor' if
    the 95% bootstrap interval of the mean of |H1 - H0| - |H1_rep - H1| lies above 0."""
    h0, h1, h1_rep = (np.asarray(x, dtype=float) for x in (h0, h1, h1_rep))
    if not h0.shape == h1.shape == h1_rep.shape or h0.ndim != 1:
        raise ValueError("h0, h1 and h1_rep must be 1-d arrays of the same persons")
    out = paired_prob_change(np.abs(h1_rep - h1), np.abs(h1 - h0), key)
    out["above_floor"] = out["ci95"][0] > 0 if out["n"] >= FLOOR_MIN_PERSONS else None
    return out


FLOOR_MIN_PERSONS = 8   # complete triplets (H0, H1_idle, H1_rep) needed to read the floor; else None (not read)


LAYERS = ("hook", "page")
DETECTORS = ("D1", "D2_lr", "D2_rf", "D3", "D4")   # D4 by amendment B4


def across(readings: dict[tuple[str, str], str], valid: dict[str, bool], layers: tuple[str, ...] = LAYERS,
           detectors: tuple[str, ...] = DETECTORS) -> dict:
    """readings[(detector, layer)] -> headline reading; valid[detector] from the validity gate (every detector
    named). Every detector in valid must have a reading on every layer, or this raises (a missing layer must not let
    the general statement pass). 'not changed' per valid detector only on its layers that read so; the general
    statement requires every valid detector on every layer to read 'not changed'; any valid 'responds' is listed;
    invalid detectors are listed apart and their readings are not read as robustness."""
    if set(valid) != set(detectors):
        raise ValueError(f"the validity of every detector is needed: {sorted(valid)} vs {sorted(detectors)}")
    missing = [[d, la] for d in valid for la in layers if (d, la) not in readings]
    if missing:
        raise ValueError(f"readings missing: {missing}")
    stray = [k for k in readings if k[0] not in valid or k[1] not in layers]
    if stray:
        raise ValueError(f"readings of an unknown detector or layer: {stray}")
    vd = [d for d in valid if valid[d]]
    per = {d: [la for la in layers if readings[(d, la)] == "not changed"] for d in vd}
    # amendment B11 (decision 26): the general statement requires every detector to be valid; with any detector
    # 'not valid here' no general statement is made (the valid ones' cells are still read one by one)
    general = len(vd) == len(valid) and all(readings[(d, la)] == "not changed" for d in vd for la in layers)
    responds = [[d, la] for d in vd for la in layers if readings[(d, la)] == "responds"]
    return {"not_changed_on": per, "valid": vd, "not_valid_here": [d for d in valid if not valid[d]],
            "general_not_moved": general, "responds": responds}


def paired_prob_change(before: np.ndarray, after: np.ndarray, key: list[int]) -> dict:
    """Mean change of a score (after - before) with a 95% percentile interval over N_BOOT resamples of persons (pairs
    with a NaN removed and counted). Used for the 'machine' probability of D2 and D3 and, by amendment A4, for D1's
    delta and for the floor H1_idle against H1_rep (amendment A3)."""
    before, after = np.asarray(before, dtype=float), np.asarray(after, dtype=float)
    if before.ndim != 1 or before.shape != after.shape:
        raise ValueError("before and after must be 1-d arrays of the same persons")
    ok = np.isfinite(before) & np.isfinite(after)
    d = after[ok] - before[ok]
    if d.size == 0:
        raise ValueError("no pair")
    rng = np.random.default_rng(key)
    m = d[rng.integers(0, d.size, (N_BOOT, d.size))].mean(axis=1)
    return {"mean": float(d.mean()), "ci95": [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))],
            "n": int(d.size), "removed": int((~ok).sum())}


# ---------------------------------------------------------------- M3 generators


def m3_donors(n_e: int) -> np.ndarray:
    """Amendment B2 (key changed by B10): the M3_DONORS donors of the M0 texts, drawn from E without replacement as
    default_rng([20261031, 7, 13]).permutation(|E|)[:40]; text t gets the t-th. The caller passes E in the draw
    order of persons.pool (the index refers to that order). B10: the key [20261031, 7, 3] of B2 gave the same
    stream as text 0's [20261031, 7, 3, 0] (numpy's SeedSequence treats a trailing 0 as absent)."""
    if n_e < M3_DONORS:
        raise ValueError(f"E holds {n_e} persons, fewer than {M3_DONORS} donors")
    return np.random.default_rng([SEED, VERSION, 13]).permutation(n_e)[:M3_DONORS]


def m3(text: int, n_gaps: int, donors: list[np.ndarray]) -> dict:
    """M3a and M3b for M0 text `text` (0 to 39): its donor from m3_donors (amendment B2); with the generator
    default_rng([20261031, 7, 3, text]), M3a's gaps (independent draws with replacement from the donor's gaps) and then
    M3b's (CondBins on the donor's one session). Returns the donor index, both gap series and M3b's fallbacks."""
    if not 0 <= text < M3_DONORS:
        raise ValueError(f"text must be in 0..{M3_DONORS - 1}")
    donor = int(m3_donors(len(donors))[text])
    rng = np.random.default_rng([SEED, VERSION, 3, text])
    g = check_gaps(donors[donor])
    a = rng.choice(g, n_gaps, replace=True)
    bsr, fb = CondBins([g]).draw(n_gaps, rng)
    return {"donor": donor, "M3a": a, "M3b": bsr, "M3b_fallback": fb}


# ---------------------------------------------------------------- C1


def components(tokens: list[set]) -> np.ndarray:
    """Connected components of the relation 'shares a token' (a person, a text, a prompt and seed): component index
    of each series, numbered in order of first appearance."""
    parent = list(range(len(tokens)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen: dict = {}
    for i, ts in enumerate(tokens):
        for t in ts:
            if t in seen:
                ra, rb = find(i), find(seen[t])
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
            else:
                seen[t] = i
    roots = [find(i) for i in range(len(tokens))]
    order: dict[int, int] = {}
    return np.array([order.setdefault(r, len(order)) for r in roots])


def group_folds(comp: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """N_FOLDS folds that never split a component: components in a random order, each to the fold holding the fewest
    series so far (ties to the lowest fold)."""
    ids = np.unique(comp)
    if ids.size < N_FOLDS:
        raise ValueError("fewer components than folds")
    size = np.zeros(N_FOLDS, dtype=int)
    fold_of = {}
    for cid in rng.permutation(ids):
        f = int(np.argmin(size))
        fold_of[int(cid)] = f
        size[f] += int((comp == cid).sum())
    return np.array([fold_of[int(c)] for c in comp])


def check_folds(tokens: list[set], fold: np.ndarray) -> None:
    """No token (person, text, prompt and seed) may appear in two folds."""
    where: dict = {}
    for ts, f in zip(tokens, fold):
        for t in ts:
            if where.setdefault(t, int(f)) != int(f):
                raise ValueError(f"related series in two folds: {t}")


def c1(x: np.ndarray, label: np.ndarray, tokens: list[set], key: list[int]) -> dict:
    """Human (0) against machine or replayed (1) on one stage: out-of-fold 'class 1' probabilities of a standardised
    logistic regression (C 1) over N_FOLDS folds grouped by relation, rank_auc of the scores and its 95% percentile
    interval over N_BOOT resamples of components; read on v6's two axes. One generator from key serves the folds
    first and then the bootstrap. A resample with one class only is skipped and counted."""
    x, label = np.asarray(x, dtype=float), np.asarray(label, dtype=int)
    if not np.all(np.isfinite(x)):
        raise ValueError("a feature is not finite")
    if len(tokens) != label.size or any(not t for t in tokens):
        raise ValueError("every series needs its relation tokens")
    comp = components(tokens)
    rng = np.random.default_rng(key)
    fold = group_folds(comp, rng)
    check_folds(tokens, fold)
    score = np.full(label.size, np.nan)
    for k in range(N_FOLDS):
        tr, te = fold != k, fold == k
        if np.unique(label[tr]).size < 2:
            raise ValueError(f"fold {k}: the training part holds one class")
        m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000)).fit(x[tr], label[tr])
        score[te] = m.predict_proba(x[te])[:, 1]
    auc = rank_auc(score, label)
    lo, hi, skipped = component_bootstrap_auc(score, label, comp, rng)
    return {"rank_auc": auc, "rank_auc_ci": [lo, hi], "reading": v6.reading(lo, hi), "n": int(label.size),
            "n_components": int(np.unique(comp).size), "bootstrap_skipped": skipped}


def component_bootstrap_auc(score: np.ndarray, label: np.ndarray, comp: np.ndarray, rng: np.random.Generator
                            ) -> tuple[float, float, int]:
    """95% percentile interval of rank_auc over N_BOOT resamples of components (the conditions' 'persons/texts':
    a component joins every series that shares a person, text or prompt and seed, so it is resampled whole; the
    scores are fixed out-of-fold scores, not refitted). A resample with one class only is skipped and counted."""
    ids = np.unique(comp)
    members = [np.flatnonzero(comp == c) for c in ids]
    boots, skipped = [], 0
    for _ in range(N_BOOT):
        idx = np.concatenate([members[i] for i in rng.integers(0, ids.size, ids.size)])
        if np.unique(label[idx]).size < 2:
            skipped += 1
            continue
        boots.append(rank_auc(score[idx], label[idx]))
    if not boots:
        raise ValueError("every bootstrap resample held one class")
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), skipped


def stage_vectors(times: np.ndarray, kinds: list[str], m: dict, r: dict, rest_ms: float = 2000.0) -> dict:
    """v6's S0, S1 and S2 of one series (sentence tiles, counting 'before'). Logger kinds are mapped to v6's kinds
    (enter -> space; backspace and delete -> other), as v6 maps KUPA's Enter, Backspace and Delete."""
    k6 = [LOGGER_KIND_TO_V6[k] for k in kinds]
    s = v6.Series(np.asarray(times, dtype=float), v6.kupa_starts(k6), v6.kupa_word_starts(k6))
    return v6.Tiles(s, "sentence", "before", m, r, rest_ms).all()


# ---------------------------------------------------------------- replay error


def kind_edit_distance(a: list[str], b: list[str]) -> int:
    """Levenshtein distance between two kind sequences (substitution, insertion and deletion cost 1)."""
    codes = {k: i for i, k in enumerate(sorted(set(a) | set(b)))}
    x = np.array([codes[k] for k in a], dtype=np.int64)
    y = np.array([codes[k] for k in b], dtype=np.int64)
    row = np.arange(y.size + 1, dtype=np.int64)
    ar = np.arange(y.size + 1, dtype=np.int64)
    for i in range(1, x.size + 1):
        cand = np.empty(y.size + 1, dtype=np.int64)
        cand[0] = i
        cand[1:] = np.minimum(row[:-1] + (y != x[i - 1]), row[1:] + 1)
        row = np.minimum.accumulate(cand - ar) + ar          # insertions along the row
    return int(row[-1])


def replay_error(planned: np.ndarray, recorded: np.ndarray, planned_kinds: list[str] | None = None,
                 recorded_kinds: list[str] | None = None) -> dict:
    """Recorded minus planned, matched by order (matching_and_cleaning), every recorded event kept. Times must be in
    one reference (the caller aligns them; relative to the first press when the clocks cannot be joined).
    drops = planned presses not recorded (shortfall), extras = excess. kind_edit_distance (always, given kinds) is the
    Levenshtein distance between the planned and recorded kinds: it counts the conditions' 'order inversions' with the
    drops and extras (an adjacent swap of two kinds counts 2; a swap, drop or extra within one kind is not seen).
    The deviations are computed only when the counts are equal and, if kinds are given, the kinds agree position by
    position ('matched'); a drop and an extra of the same kind cancel and cannot be seen (a limit). Then:
    offset = median deviation; jitter = MAD and 95th percentile of |deviation - offset|; drift = deviation of the last
    press. Otherwise these are None ('not evaluable'): a press missing in the middle would shift every later pair."""
    p, q = np.asarray(planned, dtype=float), np.asarray(recorded, dtype=float)
    if p.size == 0:
        raise ValueError("nothing planned")
    out = {"n_planned": int(p.size), "n_recorded": int(q.size), "drops": int(max(p.size - q.size, 0)),
           "extras": int(max(q.size - p.size, 0)), "counts_equal": bool(p.size == q.size), "kind_edit_distance": None,
           "matched": False, "offset_ms": None, "jitter_mad_ms": None, "jitter_p95_ms": None, "drift_ms": None}
    if planned_kinds is not None and recorded_kinds is not None:
        if len(planned_kinds) != p.size or len(recorded_kinds) != q.size:
            raise ValueError("kinds and times differ in length")
        out["kind_edit_distance"] = kind_edit_distance(list(planned_kinds), list(recorded_kinds))
        same = out["counts_equal"] and list(planned_kinds) == list(recorded_kinds)
    else:
        same = out["counts_equal"]
    if not same:
        return out
    d = q - p
    off = float(np.median(d))
    a = np.abs(d - off)
    out.update(matched=True, offset_ms=off, jitter_mad_ms=float(np.median(a)),
               jitter_p95_ms=float(np.percentile(a, 95)), drift_ms=float(d[-1]))
    return out


# ---------------------------------------------------------------- timing sensitivity and resolution


def entry_probability(rate: float, n_planned: int, missing: int, one_way: bool = False) -> float:
    """Amendment B14 (decision 35): the design table's probability that one human-side cell enters the general
    statement (general_cell_ok) when `missing` of `n_planned` pairs are missing and each complete pair flips with
    probability `rate`, half each way (or all one way, b only). Missing pairs are pairs with an original but no
    capture or no judgment; sessions without a plan are not pairs (they lower n_planned)."""
    if not 0 <= rate < 1 or n_planned <= 0 or not 0 <= missing <= n_planned:
        raise ValueError("need 0 <= rate < 1, n_planned > 0 and 0 <= missing <= n_planned")
    n = n_planned - missing
    total = 0.0
    if one_way or rate == 0:
        for b in range(n + 1):
            pr = math.comb(n, b) * rate ** b * (1 - rate) ** (n - b)
            if pr and general_cell_ok(b, 0, n, missing):
                total += pr
        return total
    half = rate / 2
    for b in range(n + 1):
        for c in range(n - b + 1):
            pr = math.exp(math.lgamma(n + 1) - math.lgamma(b + 1) - math.lgamma(c + 1) - math.lgamma(n - b - c + 1)
                          + (b + c) * math.log(half) + (n - b - c) * math.log1p(-rate))
            if general_cell_ok(b, c, n, missing):
                total += pr
    return total


def plan_zero_gap_share(plan: np.ndarray) -> float:
    """Amendment B14 (decision 36): the share of planned presses whose gap to the previous press is 0 in the uncleaned
    executable plan (series.M1); the denominator is the presses after the first (n - 1), stated in the table's note. The plan must be non-decreasing (executable_plan's output)."""
    p = np.asarray(plan, dtype=float)
    if p.ndim != 1 or p.size < 2 or not np.all(np.isfinite(p)):
        raise ValueError("need a finite 1-d plan of at least 2 presses")
    g = np.diff(p)
    if np.any(g < 0):
        raise ValueError("the executable plan must not go backwards")
    return float((g == 0).mean())


def capture_inversions(times: np.ndarray) -> int:
    """Amendment B14 (decision 39, matching_and_cleaning): the number of presses whose recorded time is earlier than
    the previous press's in capture order (the previous press, not the running maximum), counted before the stable
    sort of clean_times."""
    t = np.asarray(times, dtype=float)
    if t.ndim != 1 or not np.all(np.isfinite(t)):
        raise ValueError("need finite 1-d times in capture order")
    return int((np.diff(t) < 0).sum()) if t.size > 1 else 0


def used_gaps_mask(detector: str, g: np.ndarray) -> np.ndarray:
    """Amendment B14 (decision 39): which gaps of a session a detector uses: D1 and D2 the gaps kept by the trim, D3 the
    gaps inside whole windows of WINDOW, D4 the first D4_L gaps."""
    g = check_gaps(g)
    m = np.zeros(g.size, dtype=bool)
    if detector in ("D1", "D2_lr", "D2_rf"):
        if g.size == 0:
            raise ValueError("no gap")
        m = g <= TRIM_FACTOR * g.mean()
    elif detector == "D3":
        m[: (g.size // WINDOW) * WINDOW] = True
    elif detector == "D4":
        if g.size >= D4_L:   # d4_features: no D4 score below D4_L gaps, so no gap is used
            m[:D4_L] = True
    else:
        raise ValueError(detector)
    return m


def used_set_differs(detector: str, before: np.ndarray, after: np.ndarray, kept_before: np.ndarray | None = None,
                     kept_after: np.ndarray | None = None) -> bool:
    """Amendment B14 (decision 39): whether the set of gaps the detector uses differs between the two sides of a pair.
    Compared by position; a pair whose numbers of gaps differ counts as differing, and so does a pair whose kept press
    indices (clean_times' second output, given for both sides) differ, since equal counts can hide different presses
    dropped. For D3 and D4 equal counts and equal kept presses mean the same set. Reported as a share, not used to read."""
    a, b = check_gaps(before), check_gaps(after)
    if (kept_before is None) != (kept_after is None):
        raise ValueError("give the kept press indices of both sides or of neither")
    if a.size != b.size:
        return True
    if kept_before is not None:
        kb, ka = np.asarray(kept_before), np.asarray(kept_after)
        if kb.shape != (a.size + 1,) or ka.shape != (b.size + 1,):
            raise ValueError("kept press indices must have one more entry than the gaps")
        if not np.array_equal(kb, ka):
            return True
    return bool(np.any(used_gaps_mask(detector, a) != used_gaps_mask(detector, b)))


def quantise(times: np.ndarray, grid_ms: float) -> np.ndarray:
    return np.round(np.asarray(times, dtype=float) / grid_ms) * grid_ms


def jitter(times: np.ndarray, sd_ms: float, rng: np.random.Generator) -> np.ndarray:
    t = np.asarray(times, dtype=float)
    return t + rng.normal(0.0, sd_ms, t.size)


def sensitivity_variant(times: np.ndarray, how: str, value: float, rng: np.random.Generator | None = None,
                        drop_ties: bool = True) -> tuple[np.ndarray, np.ndarray, int]:
    """sensitivity_quantisation: the times changed by one assumption ('grid' ms or 'jitter' SD ms), then v6's
    cleaning (presses made equal or reordered are sorted, equal ones dropped and counted); gaps are taken after.
    Returns clean_times' result: times, the original index of each kept press (kinds follow by it), dropped."""
    if how == "grid":
        t = quantise(times, value)
    elif how == "jitter":
        if rng is None:
            raise ValueError("jitter needs a generator")
        t = jitter(times, value, rng)
    else:
        raise ValueError(how)
    return clean_times(t, drop_ties)   # amendment B12: drop_ties False for the ties-kept sensitivity


def kde_modes_spacing(g: np.ndarray, lo_ms: float, hi_ms: float, step_ms: float, bw_ms: float
                      ) -> tuple[float | None, int]:
    """Mean distance between neighbouring modes of a Gaussian KDE of g with a kernel SD of bw_ms (in ms: scipy's
    bw_method is a factor of the data's SD, so it is converted here)."""
    sd = float(g.std(ddof=1))
    if sd == 0:
        return None, 1
    kde = stats.gaussian_kde(g, bw_method=bw_ms / sd)
    x = np.arange(lo_ms, hi_ms + step_ms / 2, step_ms)
    y = kde(x)
    peaks = np.flatnonzero((y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:])) + 1
    if peaks.size < 2:
        return None, int(peaks.size)
    return float(np.diff(x[peaks]).mean()), int(peaks.size)


def resolution(gaps: np.ndarray, lo_ms: float, hi_ms: float, step_ms: float, bw_ms: float,
               stable_within: float = 0.10) -> dict:
    """Monaco 2016's method (the conditions' 'spacing of peaks of the press-press histogram'; Monaco takes the mean
    distance between modes of a Gaussian KDE, verify.md of .steering/20261102-paper-b-literature). The range, grid and
    bandwidth are arguments: the conditions do not fix them (they are fixed in the conditions before a value is
    taken). The estimate is also taken at bw_ms / 2 and 2 bw_ms; it is 'stable' only if both are within
    stable_within (relative) of it, and an unstable estimate is not read as a resolution."""
    g = check_gaps(gaps)
    g = g[(g >= lo_ms) & (g <= hi_ms)]
    if g.size < 2:
        return {"resolution_ms": None, "n_modes": int(g.size > 0), "stable": False}
    est, nm = kde_modes_spacing(g, lo_ms, hi_ms, step_ms, bw_ms)
    half, _ = kde_modes_spacing(g, lo_ms, hi_ms, step_ms, bw_ms / 2)
    double, _ = kde_modes_spacing(g, lo_ms, hi_ms, step_ms, bw_ms * 2)
    stable = (est is not None and half is not None and double is not None
              and abs(half - est) <= stable_within * est and abs(double - est) <= stable_within * est)
    return {"resolution_ms": est, "n_modes": nm, "at_half_bw": half, "at_double_bw": double, "stable": bool(stable)}
