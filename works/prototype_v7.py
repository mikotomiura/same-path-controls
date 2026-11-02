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
GATE_AUC, GATE_HUMAN_SHARE = 0.90, 0.10
MARGIN = 0.10            # headline equivalence margin
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


def clean_times(times: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """v6's rule (matching_and_cleaning): stable sort, a press at the same time as the previous one dropped and
    counted; times from the first press. Returns the times, the original index of each kept press (so that kinds can
    follow: kinds[i] for i in index) and the number dropped."""
    t = np.asarray(times, dtype=float)
    if not np.all(np.isfinite(t)):
        raise ValueError("a press time is not finite")
    order = np.argsort(t, kind="stable")
    t = t[order]
    keep = np.concatenate([[True], np.diff(t) > 0]) if t.size else np.array([], dtype=bool)
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
        return {"D2_lr": self.lr.predict_proba(x)[:, 1], "D2_rf": self.rf.predict_proba(x)[:, 1], "above_hist": out}


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


def validity_gate(human: list[np.ndarray], key: list[int] | None = None) -> dict:
    """Within E: N_FOLDS folds by person; per fold D2 and D3 trained on the other folds with their synthetic sessions
    drawn there; the held-out humans scored against held-out synthetic sessions of matched length from the same
    generators (D2: Uniform(30, 80); D3: the three generators fitted on the training folds). D1 has no training and is
    scored (machine score = -delta) against both synthetic sets. Held-out scores are pooled over folds into one AUC
    and one share of held-out humans judged 'machine' per detector. A detector is 'valid' if AUC >= GATE_AUC and the
    share <= GATE_HUMAN_SHARE.

    Random streams (choices of this file, not in the conditions): the folds from key [20261031, 7, 8]; in fold k the
    training synthetic sessions of D2 from default_rng([D2_SEED, k]) and of D3 from default_rng([D3_SEED, k]), the
    held-out synthetic sessions of D2 from default_rng([D2_SEED, k, 1]) and of D3 from default_rng([D3_SEED, k, 1])."""
    key = key or [SEED, VERSION, 8]
    fold = person_folds(len(human), key)
    pooled: dict[str, dict[str, list]] = {d: {"score": [], "label": []}
                                          for d in ("D1_uniform", "D1_d3gen", "D2_lr", "D2_rf", "D3")}
    d3_short = 0
    for k in range(N_FOLDS):
        train = [g for g, f in zip(human, fold) if f != k]
        test = [g for g, f in zip(human, fold) if f == k]
        d2 = train_d2(train, np.random.default_rng([D2_SEED, k]))
        d3 = train_d3(train, np.random.default_rng([D3_SEED, k]))
        syn2 = uniform_sessions([g.size for g in test], np.random.default_rng([D2_SEED, k, 1]))
        syn3 = d3_synthetic(test, fit_generators(train), np.random.default_rng([D3_SEED, k, 1]))
        lab = [0] * len(test) + [1] * len(test)
        for name, sess in (("D1_uniform", test + syn2), ("D1_d3gen", test + syn3)):
            pooled[name]["score"] += [-delta(g) for g in sess]
            pooled[name]["label"] += lab
        p2 = d2.proba(test + syn2)
        for name in ("D2_lr", "D2_rf"):
            pooled[name]["score"] += list(p2[name])
            pooled[name]["label"] += lab
        p3 = d3.proba(test + syn3)
        ok = np.isfinite(p3)
        d3_short += int((~ok).sum())
        pooled["D3"]["score"] += list(p3[ok])
        pooled["D3"]["label"] += list(np.array(lab)[ok])
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
                     "valid": bool(auc >= GATE_AUC and share <= GATE_HUMAN_SHARE)}
    out["D3"]["no_score"] = d3_short
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


def headline(b: int, c: int, n: int) -> dict:
    """Two-sided equivalence (decision 10): Tango's interval inside [-MARGIN, MARGIN] -> 'not changed'; an end outside
    with the interval excluding 0 -> 'responds'; else 'undecided'. Flips are reported beside."""
    lo, hi = tango_interval(b, c, n)
    if lo >= -MARGIN and hi <= MARGIN:
        r = "not changed"
    elif (lo < -MARGIN or hi > MARGIN) and (lo > 0 or hi < 0):
        r = "responds"
    else:
        r = "undecided"
    return {"b": b, "c": c, "n": n, "difference": (b - c) / n, "tango_95": [lo, hi], "reading": r}


LAYERS = ("hook", "page")
DETECTORS = ("D1", "D2_lr", "D2_rf", "D3")


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
    general = bool(vd) and all(readings[(d, la)] == "not changed" for d in vd for la in layers)
    responds = [[d, la] for d in vd for la in layers if readings[(d, la)] == "responds"]
    return {"not_changed_on": per, "valid": vd, "not_valid_here": [d for d in valid if not valid[d]],
            "general_not_moved": general, "responds": responds}


def paired_prob_change(before: np.ndarray, after: np.ndarray, key: list[int]) -> dict:
    """Mean change of the 'machine' probability (after - before) with a 95% percentile interval over N_BOOT resamples
    of persons (pairs with a NaN removed and counted)."""
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


def m3(text: int, n_gaps: int, donors: list[np.ndarray]) -> dict:
    """M3a and M3b for M0 text `text`: one donor drawn from E by default_rng([20261031, 7, 3, text]); with the same
    generator, M3a's gaps (independent draws with replacement from the donor's gaps) and then M3b's (CondBins on the
    donor's one session). Returns the donor index, both gap series and M3b's number of fallbacks."""
    rng = np.random.default_rng([SEED, VERSION, 3, text])
    donor = int(rng.integers(0, len(donors)))
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


def quantise(times: np.ndarray, grid_ms: float) -> np.ndarray:
    return np.round(np.asarray(times, dtype=float) / grid_ms) * grid_ms


def jitter(times: np.ndarray, sd_ms: float, rng: np.random.Generator) -> np.ndarray:
    t = np.asarray(times, dtype=float)
    return t + rng.normal(0.0, sd_ms, t.size)


def sensitivity_variant(times: np.ndarray, how: str, value: float, rng: np.random.Generator | None = None
                        ) -> tuple[np.ndarray, np.ndarray, int]:
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
    return clean_times(t)


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
