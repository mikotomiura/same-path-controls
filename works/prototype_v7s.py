"""Prototype v7s: the criterion tested on new captures, two detectors from peer-reviewed papers (D5, D6), and Firefox.

Conditions: works/conditions/prototype_v7s.json. Runs on the work PC (KUPA, v7's and v7s's captures stay there):

    python works/prototype_v7s.py --v7-captures <v7's capture folder> --v7s-captures <v7s's capture folder>

Imports v7's analysis code unchanged (checked byte-equal to a6fa5ca before anything is read) for persons, E_train,
D1-D4, cleaning, judgments and the reading rules. Writes results/prototype_v7s/record.json (counts, shares, quantiles,
readings only); refuses to overwrite.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "works"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402

import _common as cm  # noqa: E402
import prototype_v6 as v6  # noqa: E402
import prototype_v7 as v7  # noqa: E402
import prototype_v7_plan as v7plan  # noqa: E402
import prototype_v7_run as run7  # noqa: E402
import prototype_v7s_plan as splan  # noqa: E402
from capture import events as ev  # noqa: E402

COND = ROOT / "works" / "conditions" / "prototype_v7s.json"
OUT = ROOT / "results" / "prototype_v7s" / "record.json"
V7_COMMIT = "a6fa5cae45579c2b4cc95ccb938eab6ea622310e"
V7_FILES = ("works/prototype_v7.py", "works/prototype_v7_run.py")
KEY = splan.KEY                                   # 20261115
RUN_KEY = list(splan.RUNLIST_KEY)                 # captures.order
N_BLOCKS = len(splan.GROUPS)

KINDS = ev.COUNTED_KINDS                          # the seven counted kinds; code = index
KIND_CODE = {k: i for i, k in enumerate(KINDS)}
DETECTORS = (*run7.DETECTORS, "D5", "D6")
DET_IDX = {**run7.DET_IDX, "D5": 5, "D6": 6}
ADDED = ("D5", "D6")
SERIES_IDX = {**run7.MS_IDX, "MC1": 9, "MC2": 10, "MC3": 11, "MC5": 12, "F_M0": 13, "F_M2_8B": 14, "F_M2_14B": 15,
              "H0": 16}
MC = tuple(f"MC{g}" for g in splan.MC_MS)
F_MACHINE = ("F_M0", "F_M2_8B", "F_M2_14B")
TESTED = (*MC, *F_MACHINE)
# whose plan a series types (its margins are those of that plan: the same plan, checked)
PLAN_OF = {"M1_replay_8B": "M1_8B", "M1_replay_14B": "M1_14B", "F_M0": "M0", "F_M2_8B": "M2_8B",
           "F_M2_14B": "M2_14B"}

# criterion
SIGMAS = tuple(0.05 * 2.0 ** (k / 4) for k in range(42))   # criterion.margin_sd: 0.05 ms x 2^(k/4), k = 0..41
REPEATS = 10
FLIP_MIN = 5
THETA = 1.0
THETA_SENS = (0.5, 2.0)
GRID_MS = (0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
VERDICT_SESSION_SHARE = 0.10
P_IDLE_CHROME_MS = (18, 50, 100, 200)   # v7's P_idle above K_chrome's range (criterion.path_sd)
CHROME_VERSION = "156.0.8078.80"         # machine (v7's Chrome)
V7_RECORD_SHA256 = "53c03bf7bbf20310b3b98fc7f4e203668169cfb0c200ed75fb340c6d28889879"
KIND_LOSS_MAX = 0.02                      # amendment R8: share of planned presses whose kind is lost or changed
KNOWN_POINTS = {"chrome": tuple(float(x) for x in (*splan.K_CHROME_MS, *P_IDLE_CHROME_MS)),
                "firefox": tuple(float(x) for x in splan.K_FIREFOX_MS)}

# D5 (BeCAPTCHA-Type, whole session)
D5_L = 30
D5_TREES = 100
D5_RF_SEED = 28
# D6 (Mehta et al., timing features)
D6_WINDOW = 1000
D6_STEP = 700
D6_MIN_PRESSES = 100
D6_MIN_PAIR = 2
D6_LGBM_SEED = 6
BS_EVERY = 6
BS_P = 0.8
REDRAW_MAX = 100
PAIRS = tuple((a, b) for a in KINDS for b in KINDS)
PAIR_IDX = {p: i for i, p in enumerate(PAIRS)}


class RunStop(SystemExit):
    pass


# ---------------------------------------------------------------- sessions


@dataclass
class Sess:
    """A session as the detectors see it: press times from 0 (non-decreasing) and the kind of every press."""
    times: np.ndarray
    kinds: list[str]

    def __post_init__(self):
        self.times = np.asarray(self.times, dtype=float)
        if self.times.size != len(self.kinds):
            raise ValueError(f"{self.times.size} times for {len(self.kinds)} kinds")
        if any(k not in KIND_CODE for k in self.kinds):
            raise ValueError("a kind outside the counted kinds")

    @property
    def gaps(self) -> np.ndarray:
        return v7.gaps_of(self.times)


def perturbed(s: Sess, how: str, value: float, rng: np.random.Generator | None = None) -> Sess:
    """v7.sensitivity_variant (times changed, then v6's cleaning) with the kinds following the kept presses."""
    t, idx, _ = v7.sensitivity_variant(s.times, how, value, rng)
    return Sess(t, [s.kinds[i] for i in idx])


def from_plan(plan: run7.Plan | None) -> Sess | None:
    return None if plan is None or plan.times.size < 2 else Sess(plan.times, list(plan.kinds))


def from_recorded(rec: run7.Recorded | None) -> Sess | None:
    return None if rec is None or rec.times.size < 2 else Sess(rec.times, list(rec.kinds))


# ---------------------------------------------------------------- D5: BeCAPTCHA-Type over the whole session


def d5_windows(s: Sess) -> np.ndarray:
    """Non-overlapping windows of D5_L gaps from the start (the remainder dropped): per window the D5_L gaps and the
    kind codes of the presses that end them. Shape (windows, 2 * D5_L); no row for fewer than D5_L gaps."""
    g = v7.check_gaps(s.gaps)
    codes = np.array([KIND_CODE[k] for k in s.kinds[1:]], dtype=float)
    n = g.size // D5_L
    if n == 0:
        return np.empty((0, 2 * D5_L))
    return np.hstack([g[: n * D5_L].reshape(n, D5_L), codes[: n * D5_L].reshape(n, D5_L)])


def stack(sessions: list[Sess], featurise) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rows of all sessions, their weights (1 / the session's rows) and the session index of each row."""
    xs, ws, sid = [], [], []
    for i, s in enumerate(sessions):
        x = featurise(s)
        if len(x):
            xs.append(x)
            ws.append(np.full(len(x), 1.0 / len(x)))
            sid.append(np.full(len(x), i))
    if not xs:
        raise ValueError("no session gives a window")
    return np.concatenate(xs), np.concatenate(ws), np.concatenate(sid)


def d5_synthetic(human_rows: np.ndarray, kde: v7.KDE1, rng: np.random.Generator) -> np.ndarray:
    """One synthetic window per human window: D5_L gaps from the KDE, the kind codes of that human window."""
    out = human_rows.copy()
    for j in range(len(out)):
        out[j, :D5_L] = kde.draw(D5_L, rng)
    return out


@dataclass
class D5:
    rf: object
    kde: v7.KDE1

    def proba(self, sessions: list[Sess | None]) -> np.ndarray:
        out = np.full(len(sessions), np.nan)
        for i, s in enumerate(sessions):
            if s is None:
                continue
            x = d5_windows(s)
            if len(x):
                out[i] = float(self.rf.predict_proba(x)[:, 1].mean())
        return out


def train_d5(human: list[Sess], rng: np.random.Generator) -> D5:
    xh, wh, _ = stack(human, d5_windows)
    kde = v7.KDE1(xh[:, :D5_L].ravel().copy())
    xm = d5_synthetic(xh, kde, rng)
    x = np.concatenate([xh, xm])
    y = np.concatenate([np.zeros(len(xh), dtype=int), np.ones(len(xm), dtype=int)])
    rf = RandomForestClassifier(n_estimators=D5_TREES, random_state=D5_RF_SEED)
    rf.fit(x, y, sample_weight=np.concatenate([wh, wh]))
    return D5(rf, kde)


# ---------------------------------------------------------------- D6: Mehta et al.'s timing features


def d6_spans(n_presses: int) -> list[tuple[int, int]]:
    """Windows of D6_WINDOW presses starting at 0, D6_STEP, ... while they fit; one window of the whole session below
    D6_WINDOW; none below D6_MIN_PRESSES."""
    if n_presses < D6_MIN_PRESSES:
        return []
    if n_presses < D6_WINDOW:
        return [(0, n_presses)]
    return [(a, a + D6_WINDOW) for a in range(0, n_presses - D6_WINDOW + 1, D6_STEP)]


def d6_features(s: Sess) -> np.ndarray:
    """Per window: for each ordered pair of kinds, the mean gap from a press of the first kind to the next press when
    it is of the second, if the pair occurs at least D6_MIN_PAIR times in the window, else NaN."""
    g = v7.check_gaps(s.gaps)
    codes = np.array([KIND_CODE[k] for k in s.kinds], dtype=int)
    rows = []
    for a, b in d6_spans(len(s.kinds)):
        pair = codes[a:b - 1] * len(KINDS) + codes[a + 1:b]
        gg = g[a:b - 1]
        sums = np.bincount(pair, weights=gg, minlength=len(PAIRS))
        cnt = np.bincount(pair, minlength=len(PAIRS))
        row = np.full(len(PAIRS), np.nan)
        ok = cnt >= D6_MIN_PAIR
        row[ok] = sums[ok] / cnt[ok]
        rows.append(row)
    return np.array(rows) if rows else np.empty((0, len(PAIRS)))


@dataclass
class AtP:
    """At-P: pooled per-pair mean and SD (pairs with >= 2 gaps) and the global mean and SD."""
    mean: np.ndarray
    sd: np.ndarray
    has: np.ndarray
    g_mean: float
    g_sd: float
    redrawn: int = 0
    fell_back_to_mean: int = 0
    inserted_backspaces: int = 0


def at_p_stats(sessions: list[Sess]) -> AtP:
    allg, allp = [], []
    for s in sessions:
        codes = np.array([KIND_CODE[k] for k in s.kinds], dtype=int)
        allg.append(v7.check_gaps(s.gaps))
        allp.append(codes[:-1] * len(KINDS) + codes[1:])
    g, p = np.concatenate(allg), np.concatenate(allp)
    mean, sd = np.zeros(len(PAIRS)), np.zeros(len(PAIRS))
    has = np.zeros(len(PAIRS), dtype=bool)
    for j in range(len(PAIRS)):
        x = g[p == j]
        if x.size >= 2:
            mean[j], sd[j], has[j] = x.mean(), x.std(), True
    return AtP(mean, sd, has, float(g.mean()), float(g.std()))


def with_backspaces(kinds: list[str], rng: np.random.Generator) -> tuple[list[str], int]:
    """At-P's editing: a counter of non-backspace presses since the last backspace (or the start); after a press, if
    it is at least BS_EVERY, a backspace is inserted with probability BS_P and the counter reset."""
    out, c, n = [], 0, 0
    for k in kinds:
        out.append(k)
        if k == "backspace":
            c = 0
            continue
        c += 1
        if c >= BS_EVERY and rng.random() < BS_P:
            out.append("backspace")
            c, n = 0, n + 1
    return out, n


def at_p_forge(s: Sess, st: AtP, rng: np.random.Generator) -> Sess:
    kinds, n_bs = with_backspaces(s.kinds, rng)
    st.inserted_backspaces += n_bs
    gaps = np.empty(len(kinds) - 1)
    for i in range(len(kinds) - 1):
        j = PAIR_IDX[(kinds[i], kinds[i + 1])]
        mu, sd = (st.mean[j], st.sd[j]) if st.has[j] else (st.g_mean, st.g_sd)
        for _ in range(REDRAW_MAX):
            x = rng.normal(mu, sd) if sd > 0 else mu
            if x > 0:
                break
            st.redrawn += 1
        else:
            x = mu
            st.fell_back_to_mean += 1
        gaps[i] = x
    return Sess(np.concatenate([[0.0], np.cumsum(gaps)]), kinds)


@dataclass
class D6:
    model: object
    atp: AtP

    def proba(self, sessions: list[Sess | None]) -> np.ndarray:
        out = np.full(len(sessions), np.nan)
        for i, s in enumerate(sessions):
            if s is None:
                continue
            x = d6_features(s)
            if len(x):   # label 1 = human (the paper's); 'machine' = 1 - P(human)
                out[i] = float(1.0 - self.model.predict_proba(x)[:, 1].mean())
        return out


def lgbm():
    import lightgbm
    return lightgbm.LGBMClassifier(random_state=D6_LGBM_SEED, deterministic=True, n_jobs=1, verbose=-1)


def train_d6(human: list[Sess], rng: np.random.Generator) -> D6:
    atp = at_p_stats(human)
    forged = [at_p_forge(s, atp, rng) for s in human]
    xh, wh, _ = stack(human, d6_features)
    xm, wm, _ = stack(forged, d6_features)
    x = np.concatenate([xh, xm])
    y = np.concatenate([np.ones(len(xh), dtype=int), np.zeros(len(xm), dtype=int)])
    model = lgbm().fit(x, y, sample_weight=np.concatenate([wh, wm]))
    return D6(model, atp)


# ---------------------------------------------------------------- all detectors


@dataclass
class Scorer:
    s7: run7.Scorer
    d5: D5
    d6: D6

    def score(self, sessions: list[Sess | None]) -> dict[str, np.ndarray]:
        out = self.s7.score([None if s is None else s.gaps for s in sessions])
        out["D5"] = self.d5.proba(sessions)
        out["D6"] = self.d6.proba(sessions)
        return {d: out[d] for d in DETECTORS}


def train_all(e_train: list[Sess]) -> Scorer:
    return Scorer(run7.train([s.gaps for s in e_train]),
                  train_d5(e_train, np.random.default_rng([KEY, 7, 5])),
                  train_d6(e_train, np.random.default_rng([KEY, 7, 6])))


def gate_added(human: list[Sess]) -> dict:
    """v7's validity gate for D5 and D6: v7's folds of E_train; per fold trained on the other folds, held-out humans
    against held-out synthetic sessions from the training folds' generator (one per held-out human with a score)."""
    fold = v7.person_folds(len(human), [v7.SEED, v7.VERSION, 8])
    pooled = {d: {"score": [], "label": []} for d in ADDED}
    no_score = {d: 0 for d in ADDED}
    for k in range(v7.N_FOLDS):
        train = [s for s, f in zip(human, fold) if f != k]
        test = [s for s, f in zip(human, fold) if f == k]
        d5 = train_d5(train, np.random.default_rng([KEY, 7, 5, k + 1]))
        d6 = train_d6(train, np.random.default_rng([KEY, 7, 6, k + 1]))
        rng5 = np.random.default_rng([KEY, 7, 5, k + 1, 1])
        syn5 = []
        for s in test:   # a synthetic session: the human's kinds, gaps from the training folds' KDE
            if s.gaps.size >= D5_L:
                syn5.append(Sess(np.concatenate([[0.0], np.cumsum(d5.kde.draw(s.gaps.size, rng5))]), s.kinds))
        rng6 = np.random.default_rng([KEY, 7, 6, k + 1, 1])
        syn6 = [at_p_forge(s, d6.atp, rng6) for s in test if d6_spans(len(s.kinds))]
        for name, model, syn in (("D5", d5, syn5), ("D6", d6, syn6)):
            p = model.proba(test + syn)
            lab = np.array([0] * len(test) + [1] * len(syn))
            ok = np.isfinite(p)
            no_score[name] += int((~ok[: len(test)]).sum())
            pooled[name]["score"] += list(p[ok])
            pooled[name]["label"] += list(lab[ok])
    out = {}
    for name, v in pooled.items():
        s, y = np.array(v["score"]), np.array(v["label"])
        auc = v7.rank_auc(s, y)
        share = float((s[y == 0] > 0.5).mean())
        out[name] = {"auc": auc, "human_machine_share": share, "n_human": int((y == 0).sum()),
                     "n_machine": int((y == 1).sum()), "no_score": no_score[name],
                     "valid": bool(auc >= v7.GATE_AUC and share <= v7.GATE_HUMAN_SHARE),
                     "gridded": {"grid_ms": None, "auc": None, "note": "unknown (v7 read no resolution)"}}
    return out


# ---------------------------------------------------------------- captures of v7s


def read_runlogs(root: Path) -> dict[str, str]:
    """runlog.b1 .. b6 of one runs.json and the key of captures.order (as v7's checks, for v7s's blocks)."""
    files = [root / f"runlog.b{b}.ndjson" for b in range(1, N_BLOCKS + 1)]
    got = sorted(root.glob("runlog*.ndjson"))
    if got != sorted(files):
        raise RunStop(f"the runlogs are not blocks 1-{N_BLOCKS}: {[f.name for f in got]}")
    heads, out = [], {}
    for b, f in enumerate(files, start=1):
        rows = run7.read_ndjson(f)
        h = rows[0] if rows else {}
        if h.get("type") != "header" or h.get("key") != RUN_KEY or h.get("block") != b \
                or not isinstance(h.get("runs_sha256"), str):
            raise RunStop(f"{f}: the header does not name key {RUN_KEY}, block {b} and its runs.json")
        heads.append(h["runs_sha256"])
        for item, res in run7.read_runlog(f).items():
            if item in out:
                raise RunStop(f"runlog: {item} appears in two blocks")
            out[item] = res
    if len(set(heads)) != 1:
        raise RunStop("the blocks were run from different runs.json files")
    return out


def load_item(root: Path, runlog: dict[str, str], name: str, hook: bool = True) -> run7.Capture:
    """One v7s item as v7's load_capture reads one (no live capture in v7s)."""
    series, unit = name.split("/")
    if name not in runlog:
        raise RunStop(f"{name}: not in the runlog")
    res, d = runlog[name], root / name
    if res == "no_plan":
        return run7.Capture(series, 0, res)
    if res == "failed":
        cap = run7.Capture(series, 0, "failed", folder=d if d.is_dir() else None)
        if d.is_dir() and (d / "events.ndjson").exists():
            cap.plan = run7.read_events(d / "events.ndjson")
        return cap
    if not d.is_dir():
        raise RunStop(f"{name}: the runlog says {res} but the folder is missing")
    cap = run7.Capture(series, 0, "captured", redone=res == "redone", folder=d)
    cap.plan = run7.read_events(d / "events.ndjson")
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    br = meta.get("browser") if isinstance(meta.get("browser"), dict) else {}
    cap.inject = {"browser": br.get("name"), "browser_version": br.get("version")}   # read back by check_browsers
    if hook:
        cap.layers["hook"] = run7.read_hook(d / "hook.ndjson", injected=True)
    cap.layers["page"] = run7.read_page(d / "page.ndjson")
    for la, layer in cap.layers.items():
        if cap.plan.times.size and layer.times.size == 0:
            raise RunStop(f"{name}: {la}.ndjson holds no counted press")
    return cap


# ---------------------------------------------------------------- path measures and path_sd


def path_measures(raw: np.ndarray, planned_ms: float) -> dict:
    """A known-gap capture on one layer: raw gaps (capture order, nothing dropped) and after v6's cleaning."""
    g = np.diff(np.asarray(raw, dtype=float))
    t, _, dropped = v7.clean_times(raw)
    gc = v7.gaps_of(t)
    med = float(np.median(g))
    a = np.abs(g - med)
    on_ms = np.abs(g - np.round(g)) <= 1e-6
    return {"median_ms": med, "median_minus_planned_ms": med - planned_ms, "mad_ms": float(np.median(a)),
            "p95_abs_dev_ms": float(np.percentile(a, 95)), "share_below_1ms": float(np.mean(g < 1.0)),
            "dropped_share": dropped / raw.size if raw.size else None, "raw_sd": float(g.std()),
            "path_sd": float(gc.std()) if gc.size else None, "share_on_1ms_grid": float(on_ms.mean()),
            "n_gaps": int(g.size)}


def path_sd_at(table: dict[float, float], g: float) -> tuple[float, bool]:
    """criterion.path_sd: linear interpolation in the gap between known gaps; above the largest, the largest's value
    (extrapolated True). Below the smallest (0 is always known) is an error."""
    xs = sorted(table)
    if g < xs[0]:
        raise ValueError(f"gap {g} below the known gaps")
    if g >= xs[-1]:
        return float(table[xs[-1]]), g > xs[-1]
    return float(np.interp(g, xs, [table[x] for x in xs])), False


# ---------------------------------------------------------------- margins and predictions


def flip_counts(scorer: Scorer, sessions: list[Sess | None], series_idx: int, base: dict[str, np.ndarray]
                ) -> dict[str, np.ndarray]:
    """For every detector: (sessions, sigmas) counts over REPEATS of a judgment differing from the unperturbed one
    (criterion.margin_sd). A session without an unperturbed judgment gets -1 (no margin)."""
    n = len(sessions)
    counts = {d: np.zeros((n, len(SIGMAS)), dtype=int) for d in DETECTORS}
    j0 = {d: run7.judgments(d, base[d]) for d in DETECTORS}
    for k, sig in enumerate(SIGMAS):
        for r in range(REPEATS):
            pert = [None if s is None else
                    perturbed(s, "jitter", sig, np.random.default_rng([KEY, 7, 2, series_idx, i, k, r]))
                    for i, s in enumerate(sessions)]
            sc = scorer.score(pert)
            for d in DETECTORS:
                j1, o1 = run7.judgments(d, sc[d])
                m0, o0 = j0[d]
                counts[d][:, k] += (o0 & o1 & (m0 != j1)).astype(int)
    for d in DETECTORS:
        counts[d][~j0[d][1], :] = -1
    return counts


def margin_sd(row: np.ndarray) -> float | str | None:
    """sqrt(2) * the smallest sigma at which the session flips in >= FLIP_MIN of REPEATS; 'none' if at no sigma;
    None for a session without an unperturbed judgment (row of -1)."""
    if np.any(row < 0):
        return None
    hit = np.flatnonzero(row >= FLIP_MIN)
    return math.sqrt(2.0) * SIGMAS[int(hit[0])] if hit.size else "none"


def predict(path_sd: float | None, margin: float | str | None, theta: float = THETA) -> bool | None:
    """criterion.statement: flip if path_sd / margin_sd >= theta; with margin 'none' not a flip while path_sd stays
    below theta * sqrt(2) * the largest sigma, else no prediction; None when either term is missing."""
    if path_sd is None or margin is None:
        return None
    if margin == "none":
        return False if path_sd < theta * math.sqrt(2.0) * SIGMAS[-1] else None
    return bool(path_sd / float(margin) >= theta)


def flip_prob(row: np.ndarray, path_sd: float | None, theta: float = THETA) -> float | None:
    """criterion.cell_prediction: the session's share of the REPEATS flipping at the jitter that stands for the path,
    sigma* = path_sd / (sqrt(2) * theta), linear in sigma between the grid's points and from (0, 0); None without an
    unperturbed judgment, without path_sd, or above the largest sigma."""
    if path_sd is None or np.any(row < 0):
        return None
    sig = path_sd / (math.sqrt(2.0) * theta)
    if sig > SIGMAS[-1]:
        return None
    return float(np.interp(sig, [0.0, *SIGMAS], [0.0, *(row / REPEATS)]))


def expected_judgments(before: np.ndarray, det: str, prob: list[float | None]) -> tuple[np.ndarray, np.ndarray]:
    """criterion.cell_prediction: in each direction round(sum of the flip probabilities) sessions are turned, those of
    the largest probability first (ties in session order); returned as scores standing for the judgments (D1: delta 0
    'machine' or 1 'human'; others probability 1 or 0) and the mask of sessions with a probability."""
    jb, ob = run7.judgments(det, before)
    keep = np.array([q is not None for q in prob]) & ob
    q = np.array([0.0 if x is None else x for x in prob])
    mach = jb.copy()
    for side in (False, True):
        idx = np.flatnonzero(keep & (jb == side))
        k = int(math.floor(q[idx].sum() + 0.5))
        order = idx[np.argsort(-q[idx], kind="stable")][:k]
        mach[order] = not side
    s = np.where(mach, 0.0, 1.0) if det == "D1" else np.where(mach, 1.0, 0.0)
    return np.where(ob, s, np.nan), keep


def cell_reading(series: str, layer: str, det: str, before: np.ndarray, after: np.ndarray, keep: np.ndarray) -> dict:
    """v7's machine_path rule on the sessions kept (one cluster per text): b, c, n and the reading (None with no
    pair)."""
    jb, ob = run7.judgments(det, before)
    ja, oa = run7.judgments(det, after)
    ok = ob & oa & keep
    b, c, n = v7.flips(jb[ok], ja[ok])
    out = {"b": int(b), "c": int(c), "n": int(n)}
    if n:
        key = [v7.SEED, v7.VERSION, 11, run7.LAYER_IDX[layer], DET_IDX[det], SERIES_IDX[series]]
        out.update(v7.machine_path_reading(jb[ok], ja[ok], np.flatnonzero(ok).astype(int), key))
    else:
        out["reading"] = None
    return out


def decided(reading: str | None) -> bool:
    return reading in ("not changed", "responds")


def verdict(cells: list[dict], sessions: list[tuple[bool, bool]]) -> dict:
    """criterion.verdict: cells hold 'predicted' and 'observed' readings; sessions (predicted flip, observed flip)."""
    both = [c for c in cells if decided(c["predicted"]) and decided(c["observed"])]
    disagree = sum(c["predicted"] != c["observed"] for c in both)
    n = len(sessions)
    differ = sum(p != o for p, o in sessions)
    share = differ / n if n else None
    if not both:
        word = "not decided"
    elif disagree == 0 and share is not None and share <= VERDICT_SESSION_SHARE:
        word = "held"
    else:
        word = "failed"
    return {"cells_with_two_decided_readings": len(both), "cells_disagreeing": disagree, "sessions": n,
            "sessions_differing": differ, "sessions_differing_share": share, "verdict": word}


# ---------------------------------------------------------------- the run on the work PC


def code_check() -> None:
    for f in V7_FILES:
        old = subprocess.run(["git", "show", f"{V7_COMMIT}:{f}"], cwd=ROOT, capture_output=True, check=True).stdout
        if (ROOT / f).read_bytes() != old:
            raise RunStop(f"{f} differs from {V7_COMMIT[:7]} (base.code_check)")


def kinds_of_keys(keys: list[str]) -> list[str]:
    """H0's kinds as replayed (series.H1_idle of v7, amendment B16): the kind of each replay key."""
    return [run7.key_kind(k) for k in v7plan.replay_keys(keys)[0]]


def same_plan(a: run7.Plan | None, b: run7.Plan | None) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return run7.same_times(a.times, b.times) and list(a.kinds) == list(b.kinds)


@dataclass
class Series:
    """Sessions of one series in unit order: the unperturbed side, the recordings per layer, the statuses, and the
    plan series whose margins it takes (and the browser whose path it ran on)."""
    name: str
    base: list[Sess | None]
    rec: dict[str, list[Sess | None]]
    status: list[str]
    margins_of: str
    browser: str
    median_gap: list[float | None]
    page_grid: list[float | None] | None = None   # share of page gaps on the 1 ms grid, per captured session


def median_gap(s: Sess | None) -> float | None:
    return None if s is None or s.gaps.size == 0 else float(np.median(s.gaps))


def grid_share(cap: run7.Capture) -> float | None:
    if cap.status != "captured" or "page" not in cap.layers or cap.layers["page"].times.size < 2:
        return None
    g = np.diff(cap.layers["page"].times)
    return float(np.mean(np.abs(g - np.round(g)) <= 1e-6))


def series_from_caps(name: str, caps: list[run7.Capture], browser: str, base_override: list[Sess | None] | None = None
                     ) -> Series:
    base = base_override if base_override is not None else [from_plan(c.plan) for c in caps]
    rec = {la: [from_recorded(run7.recorded(c, la)) for c in caps] for la in run7.LAYERS}
    return Series(name, base, rec, [c.status for c in caps], PLAN_OF.get(name, name), browser,
                  [median_gap(s) for s in base], [grid_share(c) for c in caps])


def kind_loss(cap: run7.Capture, layer: str) -> tuple[int, int] | None:
    """amendment R8: the edit distance between the plan's kinds and the layer's counted kinds in capture order
    (before cleaning), and the plan's length; None for a capture that has no such layer."""
    if cap.status != "captured" or cap.plan is None or layer not in cap.layers:
        return None
    return v7.kind_edit_distance(list(cap.plan.kinds), list(cap.layers[layer].kinds)), int(cap.plan.times.size)


def check_kinds(name: str, caps: list[run7.Capture]) -> dict:
    """amendment R8: per layer, the share of planned presses lost or changed in kind over the series' captures; the
    run stops above KIND_LOSS_MAX (e.g. a browser that gives an injected character no key name drops it from the
    counted presses, and the detectors would judge a shorter series)."""
    out = {}
    for la in run7.LAYERS:
        got = [x for x in (kind_loss(c, la) for c in caps) if x is not None]
        if not got:
            continue
        share = sum(d for d, _ in got) / max(1, sum(n for _, n in got))
        out[la] = {"captures": len(got), "edit_share": share,
                   "not_counted": int(sum(c.layers[la].not_counted for c in caps
                                          if c.status == "captured" and la in c.layers))}
        if share > KIND_LOSS_MAX:
            raise RunStop(f"{name}/{la}: {share:.3f} of the planned presses lost or changed in kind "
                          f"(above {KIND_LOSS_MAX}; amendment R8)")
    return out


def check_browsers(caps: list[run7.Capture], browser: str) -> str | None:
    """Every captured item ran on the series' browser, Chrome on v7's version, Firefox on one version (a capture on
    another version is a deviation: the analysis stops so that it is written down first). Returns the version."""
    seen = {(c.inject or {}).get("browser_version") for c in caps if c.status == "captured"}
    names = {(c.inject or {}).get("browser") for c in caps if c.status == "captured"}
    if None in seen or None in names:
        raise RunStop(f"a captured {browser} item's meta.json names no browser or version")
    if names - {browser}:
        raise RunStop(f"a {browser} item was captured in {sorted(map(str, names - {browser}))}")
    if browser == "chrome" and seen - {CHROME_VERSION}:
        raise RunStop(f"Chrome items on {sorted(map(str, seen - {CHROME_VERSION}))}, not {CHROME_VERSION}")
    if len(seen) > 1:
        raise RunStop(f"{browser} items on more than one version: {sorted(map(str, seen))}")
    return next(iter(seen), None)


def unit_caps(root: Path, runlog: dict[str, str], series: str, units: range, fmt: str) -> list[run7.Capture]:
    """Captures of a v7s series by unit; a unit runs.json does not hold (v7 had no plan) is no_plan."""
    out = []
    for u in units:
        name = f"{series}/{fmt.format(u)}"
        out.append(load_item(root, runlog, name) if name in runlog else run7.Capture(series, u, "no_plan"))
    return out


def expected_items(v7caps: dict) -> set[str]:
    """The items v7s's plan makes from v7 (prototype_v7s_plan.build): a text has MC and F_M0 items iff v7's M0 left a
    plan, F_M2 items iff v7's M2 did; every known gap and F_H1 person always."""
    names = {f"K_chrome/c{ms:03d}" for ms in splan.K_CHROME_MS} | {f"F_K/c{ms:03d}" for ms in splan.K_FIREFOX_MS}
    names |= {f"{s_}/c{ms:03d}" for s_ in ("K_chrome_nohook", "F_K_nohook") for ms in splan.K_NOHOOK_MS}
    names |= {f"F_H1/h{i:02d}" for i in range(splan.N_F_H1)}
    for t in range(run7.N_TEXTS):
        if v7caps["M0"][t].plan is not None:
            names |= {f"{s_}/t{t:02d}" for s_ in (*MC, "F_M0")}
        for m in ("8B", "14B"):
            if v7caps[f"M2_{m}"][t].plan is not None:
                names.add(f"F_M2_{m}/t{t:02d}")
    return names


def read_series(v7caps: dict, v7s_root: Path, runlog: dict[str, str], h0: list[Sess],
                kinds_out: dict | None = None) -> dict[str, Series]:
    want = expected_items(v7caps)
    if set(runlog) != want:
        raise RunStop(f"v7s's runlogs hold {len(set(runlog) - want)} items outside the plan and miss "
                      f"{len(want - set(runlog))} of it")
    out: dict[str, Series] = {}
    for s_ in run7.MACHINE:
        out[s_] = series_from_caps(s_, v7caps[s_], "chrome")
    for s_, src in (("M1_replay_8B", "M1_8B"), ("M1_replay_14B", "M1_14B")):   # the plans it takes margins from
        for t, (c, m) in enumerate(zip(v7caps[s_], v7caps[src])):
            if c.plan is not None and not same_plan(c.plan, m.plan):
                raise RunStop(f"{s_}/t{t:02d}: not the plan of {src}")
    out["H1_idle"] = series_from_caps("H1_idle", v7caps["H1_idle"], "chrome", base_override=h0)
    out["H1_idle"].margins_of = "H0"
    texts = range(run7.N_TEXTS)
    versions = {}
    for s_ in MC:
        caps = unit_caps(v7s_root, runlog, s_, texts, "t{:02d}")
        g = float(s_[2:])
        for t, c in enumerate(caps):   # M0's keys of the same text at the constant (prototype_v7s_plan.build)
            m0 = v7caps["M0"][t].plan
            if c.plan is not None and (m0 is None or list(c.plan.keys) != list(m0.keys)
                                       or not run7.same_times(c.plan.times, g * np.arange(c.plan.times.size))):
                raise RunStop(f"{s_}/t{t:02d}: not M0's keys at a constant {g:g} ms")
        versions[s_] = check_browsers(caps, "chrome")
        if kinds_out is not None:
            kinds_out[s_] = check_kinds(s_, caps)
        out[s_] = series_from_caps(s_, caps, "chrome")
    for s_ in F_MACHINE:
        caps = unit_caps(v7s_root, runlog, s_, texts, "t{:02d}")
        src = v7caps[PLAN_OF[s_]]
        for t, c in enumerate(caps):   # the same plan as v7's, text by text (a text v7 had no plan for has none)
            if c.status != "no_plan" and not same_plan(c.plan, src[t].plan):
                raise RunStop(f"{s_}/t{t:02d}: the plan is not v7's {PLAN_OF[s_]} plan of the same text")
        versions[s_] = check_browsers(caps, "firefox")
        if kinds_out is not None:
            kinds_out[s_] = check_kinds(s_, caps)
        out[s_] = series_from_caps(s_, caps, "firefox")
    caps = unit_caps(v7s_root, runlog, "F_H1", range(splan.N_F_H1), "h{:02d}")
    for c, h in zip(caps, h0):   # F_H1 types v7's events/H: H0's times and kinds
        if c.plan is not None and (not run7.same_times(c.plan.times, h.times) or list(c.plan.kinds) != h.kinds):
            raise RunStop("F_H1: a schedule is not H0's")
    versions["F_H1"] = check_browsers(caps, "firefox")
    if kinds_out is not None:
        kinds_out["F_H1"] = check_kinds("F_H1", caps)
        kinds_out["firefox_version"] = next(iter({v for k, v in versions.items() if k.startswith("F_") and v}), None)
    ff = {v for k, v in versions.items() if k.startswith("F_") and v is not None}
    if len(ff) > 1:
        raise RunStop(f"Firefox series on more than one version: {sorted(ff)}")
    out["F_H1"] = series_from_caps("F_H1", caps, "firefox", base_override=h0[: len(caps)])
    out["F_H1"].margins_of = "H0"
    return out


def known_tables(v7caps: dict, v7s_root: Path, runlog: dict[str, str]) -> tuple[dict, dict]:
    """path_sd per browser and layer at each known gap (hook installed), and every path measure."""
    meas: dict[str, dict] = {}
    table: dict[tuple[str, str], dict[float, float]] = {}
    k_caps: dict[str, list[run7.Capture]] = {"chrome": [], "firefox": []}

    def add(label: str, browser: str, layer: str, cap: run7.Capture, ms: float, use: bool) -> None:
        if cap.plan is not None and label.startswith(("K_", "F_K")) and \
                not run7.same_times(cap.plan.times, ms * np.arange(cap.plan.times.size)):
            raise RunStop(f"{label}: the plan is not a constant {ms:g} ms")
        if label.startswith(("K_", "F_K")) and layer == "page":
            k_caps["firefox" if label.startswith("F_K") else "chrome"].append(cap)
        if cap.status != "captured" or layer not in cap.layers:
            meas[f"{label}/{layer}"] = {"status": cap.status}
            return
        m = path_measures(cap.layers[layer].times, ms)
        meas[f"{label}/{layer}"] = m
        if use and m["path_sd"] is not None:
            table.setdefault((browser, layer), {})[float(ms)] = m["path_sd"]

    for ms in splan.K_CHROME_MS:
        cap = load_item(v7s_root, runlog, f"K_chrome/c{ms:03d}")
        for la in run7.LAYERS:
            add(f"K_chrome/{ms}", "chrome", la, cap, ms, True)
    for ms in splan.K_NOHOOK_MS:
        add(f"K_chrome_nohook/{ms}", "chrome", "page", load_item(v7s_root, runlog, f"K_chrome_nohook/c{ms:03d}",
                                                                 hook=False), ms, False)
    for cap in v7caps["P_idle"]:
        ms = float(cap.unit)
        for la in run7.LAYERS:
            add(f"v7_P_idle/{cap.unit}", "chrome", la, cap, ms, cap.unit in P_IDLE_CHROME_MS)
    for cap in v7caps["P_nohook"]:
        add(f"v7_P_nohook/{cap.unit}", "chrome", "page", cap, float(cap.unit), False)
    for ms in splan.K_FIREFOX_MS:
        cap = load_item(v7s_root, runlog, f"F_K/c{ms:03d}")
        for la in run7.LAYERS:
            add(f"F_K/{ms}", "firefox", la, cap, ms, True)
    for ms in splan.K_NOHOOK_MS:
        add(f"F_K_nohook/{ms}", "firefox", "page", load_item(v7s_root, runlog, f"F_K_nohook/c{ms:03d}", hook=False),
            ms, False)
    for b, cs in k_caps.items():
        check_browsers(cs, b)
        meas[f"kinds/{b}"] = check_kinds(f"known gaps in {b}", cs)
    tables = {f"{b}/{la}": t for (b, la), t in table.items()}
    check_known(tables)
    return meas, tables


def check_known(tables: dict[str, dict[float, float]]) -> None:
    """amendment R2: every known gap of each browser gives a path_sd on both layers, else the run stops."""
    for b, pts in KNOWN_POINTS.items():
        for la in run7.LAYERS:
            got = set(tables.get(f"{b}/{la}", {}))
            if got != set(pts):
                raise RunStop(f"path_sd for {b}/{la}: known gaps {sorted(set(pts) - got)} missing (a known-gap "
                              "capture failed or its gaps are empty); the analysis stops rather than interpolate "
                              "over other points")


# ---------------------------------------------------------------- readings, predictions, verdicts


MACHINE_SERIES_ALL = (*run7.MACHINE, *MC, *F_MACHINE)
HUMAN_SERIES = ("H1_idle", "F_H1")


def score_series(scorer: Scorer, series: dict[str, Series]) -> dict[str, dict]:
    """Per series: the unperturbed side's scores and the recorded scores per layer."""
    out = {}
    for name, se in series.items():
        out[name] = {"base": scorer.score(se.base), **{la: scorer.score(se.rec[la]) for la in run7.LAYERS}}
    return out


def margin_sources(series: dict[str, Series], h0: list[Sess]) -> dict[str, list[Sess | None]]:
    src = {"H0": h0}
    for name, se in series.items():
        if se.margins_of == name and name not in HUMAN_SERIES:
            src[name] = se.base
    return src


def all_margins(scorer: Scorer, sources: dict[str, list[Sess | None]]) -> tuple[dict, dict]:
    """margin_sd of every session of every source, and the flip counts (kept for the descriptive jitter table)."""
    margins, counts = {}, {}
    for name, sess in sources.items():
        base = scorer.score(sess)
        cnt = flip_counts(scorer, sess, SERIES_IDX[name], base)
        counts[name] = cnt
        margins[name] = {d: [margin_sd(row) for row in cnt[d]] for d in DETECTORS}
    return margins, counts


def session_predictions(se: Series, layer: str, det: str, margins: dict, counts: dict, tables: dict, theta: float
                        ) -> tuple[list[bool | None], list[float | None], int]:
    """Per session: the predicted flip and the flip probability (None: no prediction), and the number of sessions
    marked extrapolated. A session counts as predicted only if it has both."""
    table = tables.get(f"{se.browser}/{layer}")
    m = margins[se.margins_of][det]
    cnt = counts[se.margins_of][det]
    flips_, probs, extra = [], [], 0
    for i, g in enumerate(se.median_gap):
        if g is None or table is None or se.status[i] == "no_plan":
            flips_.append(None)
            probs.append(None)
            continue
        psd, ex = path_sd_at(table, g)
        extra += ex
        f, q = predict(psd, m[i], theta), flip_prob(cnt[i], psd, theta)
        flips_.append(f if q is not None else None)
        probs.append(q if f is not None else None)
    return flips_, probs, extra


def criterion_cells(series: dict[str, Series], scores: dict, margins: dict, counts: dict, tables: dict,
                    valid: dict[str, bool], theta: float) -> dict:
    """Predicted and observed readings of every machine cell, and the per-session pairs, for one theta."""
    cells, sessions = {}, {}
    for name in MACHINE_SERIES_ALL:
        se = series[name]
        for la in run7.LAYERS:
            for d in DETECTORS:
                flip, prob, extra = session_predictions(se, la, d, margins, counts, tables, theta)
                before, after = scores[name]["base"][d], scores[name][la][d]
                jb, ob = run7.judgments(d, before)
                ja, oa = run7.judgments(d, after)
                # one set of sessions for both readings: a plan, a judgment on both sides of the pair, a prediction
                # (review 1 of decision 3: a failed capture had a predicted flip but no observed judgment)
                common = np.array([st_ != "no_plan" for st_ in se.status]) & ob & oa \
                    & np.array([f is not None and q is not None for f, q in zip(flip, prob)])
                s_pred, keep = expected_judgments(before, d, [q if common[i] else None for i, q in enumerate(prob)])
                has = common & keep
                obs = cell_reading(name, la, d, before, after, has)
                prd = cell_reading(name, la, d, before, s_pred, has)
                pairs = [(bool(flip[i]), bool(jb[i] != ja[i])) for i in range(len(flip))
                         if has[i] and ob[i] and oa[i]]
                key = f"{name}/{la}/{d}"
                cells[key] = {"observed": obs.get("reading"), "predicted": prd.get("reading"),
                              "observed_bc": [obs["b"], obs["c"], obs["n"]],
                              "predicted_bc": [prd["b"], prd["c"], prd["n"]],
                              "without_prediction": int(sum(f is None for f in flip)),
                              "left_out": int(len(flip) - has.sum()),
                              "extrapolated": extra, "valid": valid[d], "tested": name in TESTED,
                              "sessions_differing": int(sum(p_ != o_ for p_, o_ in pairs))}
                sessions[key] = pairs
    return {"cells": cells, "sessions": sessions}


def human_cells(series: dict[str, Series], scores: dict, margins: dict, counts: dict, tables: dict, theta: float
                ) -> dict:
    """H0 against H1_idle (v7, Chrome, post hoc: v7's headline rule, n = 80) and against F_H1 (Firefox, 10 persons:
    flips only)."""
    out = {}
    for name in HUMAN_SERIES:
        se = series[name]
        for la in run7.LAYERS:
            for d in DETECTORS:
                flip, prob, extra = session_predictions(se, la, d, margins, counts, tables, theta)
                jb, ob = run7.judgments(d, scores[name]["base"][d])
                ja, oa = run7.judgments(d, scores[name][la][d])
                ok = ob & oa & np.array([f is not None for f in flip])
                b, c, n = v7.flips(jb[ok], ja[ok])
                sp_, _ = expected_judgments(scores[name]["base"][d], d, [prob[i] if ok[i] else None
                                                                      for i in range(len(prob))])
                jp, _ = run7.judgments(d, sp_)
                pb = int(np.sum(ok & ~jb & jp))
                pc = int(np.sum(ok & jb & ~jp))
                row = {"observed_bc": [int(b), int(c), int(n)], "predicted_bc": [pb, pc, int(n)],
                       "without_prediction": int(sum(f is None for f in flip)), "extrapolated": extra,
                       "sessions_differing": int(sum(bool(flip[i]) != bool(jb[i] != ja[i])
                                                     for i in np.flatnonzero(ok)))}
                if name == "H1_idle" and n >= 35:   # criterion.cell_prediction: human rows with n < 35: no reading
                    row["observed"] = v7.headline(int(b), int(c), int(n))["reading"]
                    row["predicted"] = v7.headline(pb, pc, int(n))["reading"]
                out[f"{name}/{la}/{d}"] = row
    return out


def verdicts(crit: dict, valid: dict[str, bool]) -> dict:
    """criterion.verdict over the tested cells of valid detectors: in all and per detector."""
    def pick(det: str | None):
        keys = [k for k, c in crit["cells"].items() if c["tested"] and c["valid"]
                and (det is None or k.endswith("/" + det))]
        cells = [crit["cells"][k] for k in keys]
        sess = [p for k in keys for p in crit["sessions"][k]]
        return verdict(cells, sess)
    return {"all": pick(None), **{d: pick(d) for d in DETECTORS if valid[d]}}


def jitter_table(counts: dict, sources: dict) -> dict:
    """perturbation_grid (jitter, v7p's form): per source and detector the mean flip share at each sigma and the
    smallest sigma reaching 3/n and 0.10."""
    out = {}
    for name, cnt in counts.items():
        for d in DETECTORS:
            c = cnt[d]
            judged = c[:, 0] >= 0
            n = int(judged.sum())
            if not n:
                out[f"{name}/{d}"] = {"n": 0}
                continue
            if n < run7.MIN_SUMMARY:
                out[f"{name}/{d}"] = {"n": n}
                continue
            share = c[judged].sum(axis=0) / (n * REPEATS)
            first = {}
            for lab, lev in (("3_of_n", 3 / n), ("0.10", 0.10)):
                hit = np.flatnonzero(share >= lev)
                first[lab] = SIGMAS[int(hit[0])] if hit.size else "not reached"
            out[f"{name}/{d}"] = {"n": n, "share_by_sigma": [float(x) for x in share], "first_sigma": first}
    return out


def grid_table(scorer: Scorer, sources: dict) -> dict:
    out = {}
    for name, sess in sources.items():
        base = scorer.score(sess)
        for g in GRID_MS:
            sc = scorer.score([None if s is None else perturbed(s, "grid", g) for s in sess])
            for d in DETECTORS:
                j0, o0 = run7.judgments(d, base[d])
                j1, o1 = run7.judgments(d, sc[d])
                ok = o0 & o1
                b, c, n = v7.flips(j0[ok], j1[ok])
                out[f"{name}/{g}/{d}"] = {"b": int(b), "c": int(c), "n": int(n),
                                          "share": (b + c) / n if n else None}
    return out


def readings_added(series: dict[str, Series], scores: dict, v7caps: dict, scorer: Scorer) -> dict:
    """detectors_added.readings: D5 and D6 on v7's rows by v7's rules (H0 against H1_idle and H1_load, n = 80; the
    floor H1_rep in floor_added) and machine_path on every machine series."""
    out = {}
    load = {la: scorer.score([from_recorded(run7.recorded(c, la)) for c in v7caps["H1_load"]]) for la in run7.LAYERS}
    for d in ADDED:
        for la in run7.LAYERS:
            for row, after in (("H0-H1_idle", scores["H1_idle"][la][d]), ("H0-H1_load", load[la][d])):
                jb, ob = run7.judgments(d, scores["H1_idle"]["base"][d])
                ja, oa = run7.judgments(d, after)
                ok = ob & oa
                b, c, n = v7.flips(jb[ok], ja[ok])
                out[f"{row}/{la}/{d}"] = v7.headline(int(b), int(c), int(n)) if n else {"n": 0}
            for name in MACHINE_SERIES_ALL:
                se = series[name]
                has = np.array([st_ != "no_plan" for st_ in se.status])
                out[f"{name}/{la}/{d}"] = cell_reading(name, la, d, scores[name]["base"][d], scores[name][la][d], has)
    return out


def floor_added(v7caps: dict, scores: dict, scorer: Scorer) -> dict:
    """H1_idle against H1_rep for D5 and D6 by v7's rule (pairs.replay_repeat): flips as counts, and the floor read by
    v7.above_floor on the complete triplets (H0, H1_idle, H1_rep) with v7's key [20261031, 7, 12, layer, detector]."""
    out = {}
    for la in run7.LAYERS:
        rep = scorer.score([from_recorded(run7.recorded(c, la)) for c in v7caps["H1_rep"]])
        for d in ADDED:
            k = len(rep[d])
            h0, a, r = scores["H1_idle"]["base"][d][:k], scores["H1_idle"][la][d][:k], rep[d]
            ok = np.isfinite(h0) & np.isfinite(a) & np.isfinite(r)
            ja, _ = run7.judgments(d, a)
            jr, _ = run7.judgments(d, r)
            b, c, n = v7.flips(ja[ok], jr[ok])
            row = {"b": int(b), "c": int(c), "n": int(n)}
            if n:
                key = [v7.SEED, v7.VERSION, 12, run7.LAYER_IDX[la], DET_IDX[d]]
                fl = v7.above_floor(h0[ok], a[ok], r[ok], key)
                row["above_floor"] = fl["above_floor"]
                row["triplets"] = fl["n"]
                if fl["n"] >= run7.MIN_SUMMARY:
                    row["mean_difference_ci95"] = fl.get("ci95")
            out[f"{la}/{d}"] = row
    return out


def firefox_vs_chrome(scores: dict) -> dict:
    """Firefox's recording against Chrome's of the same plans: per layer and detector, flips between the two
    recordings and the mean absolute score change (persons: the first 10)."""
    out = {}
    for f_, c_ in (("F_M0", "M0"), ("F_M2_8B", "M2_8B"), ("F_M2_14B", "M2_14B"), ("F_H1", "H1_idle")):
        for la in run7.LAYERS:
            for d in DETECTORS:
                a, b_ = scores[c_][la][d], scores[f_][la][d]
                a = a[: b_.size]
                ja, oa = run7.judgments(d, a)
                jb, ob = run7.judgments(d, b_)
                ok = oa & ob
                b, c, n = v7.flips(ja[ok], jb[ok])
                out[f"{f_}/{la}/{d}"] = {"b": int(b), "c": int(c), "n": int(n),
                                         "mean_abs_change": float(np.abs(a[ok] - b_[ok]).mean())
                                         if n >= run7.MIN_SUMMARY else None}
    return out


def to_json(x):
    if isinstance(x, dict):
        return {str(k): to_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_json(v) for v in x]
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def check_constants(c: dict) -> None:
    """The constants of this file are the conditions' (a drift stops the run)."""
    if "0.05 * 2^(k / 4) ms, k = 0, 1, ..., 41" not in c["criterion"]["margin_sd"] or len(SIGMAS) != 42:
        raise RunStop("the sigma grid differs from the conditions' margin_sd")
    if c["perturbation_grid"]["grid_ms"] != list(GRID_MS):
        raise RunStop("perturbation_grid.grid_ms differs from the code")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v7-captures", required=True, type=Path)
    ap.add_argument("--v7s-captures", required=True, type=Path)
    a = ap.parse_args(argv)
    if OUT.exists():
        raise RunStop(f"{OUT} exists; refusing to overwrite")
    code_check()
    c = json.loads(COND.read_text(encoding="utf-8"))
    check_constants(c)
    _, c7 = cm.load_conditions(run7.COND)
    _, c6 = cm.load_conditions(run7.COND_V6)
    kp = c7["data"]["kupa"]
    files = {"essay": ("external/kupa-keys/KUPA-KEYS-TASK-2.csv", run7.kp_sha(kp, "essay")),
             "copy": (c6["data"]["kupa"]["files"]["copy"]["file"], c6["data"]["kupa"]["files"]["copy"]["sha256"])}
    for name, (f, sha) in files.items():
        if cm.sha256(ROOT / f) != sha:
            raise RunStop(f"{f} does not match the SHA-256 of the conditions ({name})")
    v7rec_path = ROOT / "results" / "prototype_v7" / "record.json"
    if cm.sha256(v7rec_path) != V7_RECORD_SHA256:
        raise RunStop("results/prototype_v7/record.json is not v7's record (its SHA-256)")
    v7rec = json.loads(v7rec_path.read_text(encoding="utf-8"))
    essay, seen = v6.read_kupa(ROOT / files["essay"][0])
    copy_, _ = v6.read_kupa(ROOT / files["copy"][0])
    v6_sha = c7["persons"]["pool"].split("ids_sha256 ")[1].split(",")[0]
    ps = run7.draw_persons(essay, copy_, seen, v6_sha)
    e_train_ids = [ps.e[i] for i in ps.e_train]
    keyed = v7plan.read_kupa_keys(ROOT / files["essay"][0], set(ps.h) | set(e_train_ids))

    def sess_of(pid: str) -> Sess:
        t, k = keyed[pid]
        if not run7.same_times(t, essay[pid]["times"]):
            raise RunStop("the keys' reading of KUPA does not give the persons' times")
        return Sess(t, kinds_of_keys(k))

    h0 = [sess_of(p) for p in ps.h]
    etr = [sess_of(p) for p in e_train_ids]
    scorer = train_all(etr)
    gate = gate_added(etr)
    valid = {**{d: bool(v7rec["validity_gate"]["valid"][d]) for d in run7.DETECTORS},
             **{d: gate[d]["valid"] for d in ADDED}}
    v7caps = run7.load_all(a.v7_captures)
    runlog = read_runlogs(a.v7s_captures)
    meas, tables = known_tables(v7caps, a.v7s_captures, runlog)
    kinds: dict = {}
    series = read_series(v7caps, a.v7s_captures, runlog, h0, kinds)
    scores = score_series(scorer, series)
    sources = margin_sources(series, h0)
    margins, counts = all_margins(scorer, sources)
    crit = {f"{t:g}": criterion_cells(series, scores, margins, counts, tables, valid, t)
            for t in (THETA, *THETA_SENS)}
    rec = {
        "validity_gate_added": gate, "valid": valid,
        "path": {"measures": meas, "path_sd_tables": {k: {f"{g:g}": v for g, v in t.items()} for k, t in tables.items()}},
        "criterion": {th: {"cells": cr["cells"], "verdict": verdicts(cr, valid)} for th, cr in crit.items()},
        "criterion_human": {f"{t:g}": human_cells(series, scores, margins, counts, tables, t)
                            for t in (THETA, *THETA_SENS)},
        "readings_added": readings_added(series, scores, v7caps, scorer),
        "kinds": kinds,
        "firefox_page_grid": {name: {"n": sum(x is not None for x in series[name].page_grid),
                                     "min": (None if name == "F_H1" else   # one person's value is not written
                                             min((x for x in series[name].page_grid if x is not None), default=None)),
                                     "share_of_captures_at_least_0.99": (
                                         float(np.mean([x >= 0.99 for x in series[name].page_grid if x is not None]))
                                         if any(x is not None for x in series[name].page_grid) else None)}
                              for name in (*F_MACHINE, "F_H1")},
        "chrome_page_grid_min": {name: min((x for x in series[name].page_grid if x is not None), default=None)
                                 for name in MC},
        "floor_added": floor_added(v7caps, scores, scorer),
        "perturbation_jitter": jitter_table(counts, sources),
        "perturbation_grid": grid_table(scorer, sources),
        "firefox_vs_chrome": firefox_vs_chrome(scores),
        "d6_forgery": {k: getattr(scorer.d6.atp, k) for k in ("redrawn", "fell_back_to_mean", "inserted_backspaces")},
        "d5_kde_redrawn": scorer.d5.kde.redrawn,
        "counts": {name: {"sessions": len(se.status), **{st_: se.status.count(st_)
                                                          for st_ in ("captured", "failed", "no_plan")}}
                   for name, se in series.items()},
    }
    paths = ["works/conditions/prototype_v7s.json", "works/prototype_v7s.py", "works/prototype_v7s_plan.py",
             "works/prototype_v7_run.py", "works/prototype_v7_plan.py", "works/prototype_v7.py",
             "works/prototype_v6.py", "works/_common.py", "capture/events.py", "capture/browser.py",
             "capture/session.py", "capture/runlist.py", "capture/chain-v7s/chain.ps1", "capture/chain-v7s/guard.ps1",
             "capture/chain-v7s/minimize.ps1", "pyproject.toml", "uv.lock"]
    rec["provenance"] = {"run_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                         **cm.git_state(paths),
                         "conditions_sha256": hashlib.sha256(COND.read_bytes()).hexdigest(),
                         "code_sha256": {p_: cm.sha256(ROOT / p_) for p_ in paths},
                         "data_sha256": {n: s_ for n, (_, s_) in files.items()},
                         "v7_record_sha256": cm.sha256(v7rec_path),
                         "v7s_runs_sha256": run7.read_ndjson(a.v7s_captures / "runlog.b1.ndjson")[0]["runs_sha256"],
                         "v7s_runlogs_sha256": {f"runlog.b{b}.ndjson": cm.sha256(a.v7s_captures / f"runlog.b{b}.ndjson")
                                                for b in range(1, N_BLOCKS + 1)}}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(to_json(rec), indent=1) + "\n", encoding="utf-8")
    tmp.replace(OUT)
    print(json.dumps(to_json({th: r["verdict"] for th, r in rec["criterion"].items()}), indent=1))


if __name__ == "__main__":
    main()
