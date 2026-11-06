"""Prototype v7: the run that applies the analysis functions of works/prototype_v7.py to KUPA's persons and to the
captures of the work PC (works/conditions/prototype_v7.json; .steering/20261102-v7-analysis-run).

Written before any capture exists and tested on synthetic persons and synthetic capture folders only. It runs on the
work PC, where KUPA and the captures are; what leaves the work PC is results/prototype_v7/record.json (counts and
summaries only: no press time, gap, key, text, LLM output, per-person value or person id).

The readers of the capture files (section 'readers') follow capture-spec-v2 where it fixes a format (events.ndjson)
and assume field names where it does not (hook.ndjson, page.ndjson, inject.ndjson, llm.ndjson, runlog.ndjson). Each
assumption is stated in ASSUMED and in one function, to be checked against the work PC's code when its bundle comes
back; nothing else in this file reads a capture file.

Layout of the captures (CAPTURE_LAYOUT; written here first and to be followed by the runlist of the production
captures): <captures>/runlog.ndjson and one folder per item, <captures>/<series>/<unit>/, holding the files of
capture-spec-v2 section 5.
"""

from __future__ import annotations

import atexit
import datetime as dt
import hashlib
import json
import math
import shutil
import sys
import tempfile
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if not sys.pycache_prefix:
    sys.pycache_prefix = tempfile.mkdtemp(prefix="pycache-")
    atexit.register(shutil.rmtree, sys.pycache_prefix, ignore_errors=True)
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "works"))

import numpy as np  # noqa: E402

import _common as cm  # noqa: E402
import prototype_v6 as v6  # noqa: E402
import prototype_v7 as v7  # noqa: E402

COND = ROOT / "works" / "conditions" / "prototype_v7.json"
COND_V6 = ROOT / "works" / "conditions" / "prototype_v6.json"
OUT_DIR = ROOT / "results" / "prototype_v7"
CODE = ["works/prototype_v7_run.py", "works/prototype_v7_plan.py", "capture/runlist.py", "capture/events.py",
        "works/prototype_v7.py", "works/prototype_v6.py", "works/prototype_v5.py",
        "works/prototype_v4.py", "works/_common.py", "src/keystroke.py", "uv.lock"]
S = v7.SEED

N_POOL, N_V6, N_H, MIN_PRESSES = 300, 200, 80, 300   # persons.pool, persons.replay_H
N_TEXTS, N_REP, N_ESSAYS = 40, 10, 4                  # M1/M2 prompts and M0/M3 texts; H1_rep; L0
M0_GAP_MS = 55.0
CONTROLS_MS = (5.0, 18.0, 50.0, 100.0, 200.0)
P_PRESSES = 1000
LAYERS = v7.LAYERS
LAYER_IDX = {"hook": 0, "page": 1}
DETECTORS = v7.DETECTORS
DET_IDX = {d: i for i, d in enumerate(DETECTORS)}
PROB_DETECTORS = ("D2_lr", "D2_rf", "D3", "D4")
# machine_path's series index (detectors_C2.count, amendment A2); the count's cluster bootstrap uses the same index
MACHINE = ("M0", "M1_8B", "M1_14B", "M1_replay_8B", "M1_replay_14B", "M2_8B", "M2_14B", "M3a", "M3b")
MS_IDX = {s_: i for i, s_ in enumerate(MACHINE)}
# sensitivity_quantisation's jitter key (analysis_details.keys): 0 H0, then the machine series in MACHINE's order
JITTER_IDX = {"H0": 0, **{s_: i + 1 for i, s_ in enumerate(MACHINE)}}
# analysis_details.keys: the paired rows of detectors_C2.count
PAIR_IDX = {"H0-H1_idle": 0, "H0-H1_load": 1, "H1_idle-H1_rep": 2, "H0-H0q_jitter2": 3, "L0-L1_idle": 4,
            "L0-L1_load": 5, "H0-H0q_jitter8": 6}
# stages_C1's pair index (not fixed by the conditions; decisions.md of .steering/20261102-v7-analysis-run)
C1_PAIRS = ("H0-H1_idle", "H1_idle-M0", "H1_idle-M1_8B", "H1_idle-M1_14B", "H1_idle-M2_8B", "H1_idle-M2_14B",
            "H1_idle-M3a", "H1_idle-M3b")
C1_IDX = {p: i for i, p in enumerate(C1_PAIRS)}
STAGES = ("S0", "S1", "S2")
H0Q_VERSIONS = (("grid", 1.0), ("grid", 4.0), ("grid", 16.0), ("jitter", 2.0), ("jitter", 8.0), ("resolution", None))
RES_ARGS = (0.0, 300.0, 0.1, 1.0)   # analysis_details.resolution: range [0, 300] ms, 0.1 ms grid, kernel SD 1.0 ms
RATES = (0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30)   # detectors_C2.headline, amendment B11
COUNTED_KINDS = ("alnum", "space", "end", "enter", "backspace", "delete", "other")

CAPTURE_LAYOUT = {
    "H1_idle": "h{i:02d}", "H1_load": "h{i:02d}", "H1_rep": "h{i:02d}",   # i: index in replay_H's draw order
    "L0": "e{i}", "L1_idle": "e{i}", "L1_load": "e{i}",                    # i: essay 0..3
    **{s_: "t{i:02d}" for s_ in MACHINE},                                 # i: text t = prompt P(t+1)
    "P_idle": "c{i:03d}", "P_load": "c{i:03d}", "P_nohook": "c{i:03d}",    # i: the control's gap in ms
}
ASSUMED = {
    # Adapted on 2026-11-04 to the files of the work PC's capture code (feature/v7-capture after the review fixes,
    # .steering/20261104-v7-capture-returns); the conditions are unchanged (analysis_details.run.captures).
    "events.ndjson": "capture-spec-v2 section 1 (fixed): a header line {kind: header, ...}, then {i, t_ms, key}. M1 "
                     "has no events.ndjson: its plan is schedule.ndjson (the same format)",
    "hook.ndjson": "a header line and a footer line (type header / footer), then one line per event {type: key, t_ns, "
                   "down, injected, ours, repeat, kind}; kind one of COUNTED_KINDS or modifier / nonchar (not "
                   "counted); auto-repeated downs are left out",
    "page.ndjson": "one line per keydown {type: key, t_ms (event.timeStamp), h_ms, kind, repeat, trusted, composing, "
                   "in_box}; other lines (open, status, ready, armed, log) have another type",
    "inject.ndjson": "replays: a header line {type: header, ...}, then one line per press {i, planned_ns, sent_ns, "
                     "returned_ns, late_ns, sendinput_returned, n_inputs, ok}; M1: no header, one line per press "
                     "{i, token, wished_ms, plan_ms, planned_ns, sent_ns, returned_ns, wait_ns, queue_len, "
                     "sendinput_returned, n_inputs, ok}. A failed press is ok false",
    "llm.ndjson (M1)": "lines of type header, request, chunk {n, recv_ns, token, n_chars, done}, error and final; the "
                       "chunk with done true ends the generation (its recv_ns is a_{K+1}) and its text, if any, is a "
                       "last token arriving at a_{K+1}. Characters outside printable ASCII and the \\r of \\r\\n are "
                       "skipped: their slots stay empty and the running maximum is taken over the typed characters "
                       "(the researcher's decision E-5 of 2026-11-04); \\n is Enter",
    "runlog.ndjson": "a header line, one line per item {type: item, item: '<series>/<unit>', result: done | redone | "
                     "failed | no_plan, detail, attempts} and a footer line; the folder <series>/<unit> holds the "
                     "last attempt (a first interrupted attempt is <unit>.try1, not read); a failed item keeps its "
                     "plan files (events.ndjson; M1 llm.ndjson and, when the stream ended without an error, "
                     "schedule.ndjson); a failed plan of fewer than 2 presses is no plan",
    "L1 events": "L0's hook-layer counted presses between the signal and the end chord (meta.json typing_start_perf_ns "
                 "to typing_end_perf_ns), auto-repeats left out, stable-sorted with equal times dropped (v6's rule), "
                 "times from the first press, keys the kinds' representatives (capture code events.from_live)",
    "meta.json": "typing_start_perf_ns and typing_end_perf_ns (the same clock as the hook's t_ns): L0's hook layer is "
                 "read between them",
}


class RunStop(SystemExit):
    """A contract of the conditions is broken: the run stops (nothing is written)."""


# ---------------------------------------------------------------- readers (the only code that reads a capture file)


def read_ndjson(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise RunStop(f"{path}: line {n} is not JSON ({e})")
    return rows


def key_kind(key: str) -> str:
    """The kind of a planned key (capture-spec-v2 section 1: one character, or Enter, Backspace, Delete), by v6's rule
    (kupa_kind) with Enter, Backspace and Delete kept apart as the loggers keep them."""
    named = {"Enter": "enter", "Backspace": "backspace", "Delete": "delete"}
    if key in named:
        return named[key]
    if len(key) != 1:
        raise RunStop(f"a planned key is neither one character nor a named key: {key!r}")
    if key in ".!?":
        return "end"
    if key == " ":
        return "space"
    return "alnum" if key.isalnum() else "other"   # v6's kupa_kind (H0's kinds are v6's)


@dataclass
class Plan:
    times: np.ndarray     # t_ms, from the first press, non-decreasing (the plan is not cleaned: series.M1, B14)
    kinds: list[str]
    keys: list[str]       # stays on the work PC (never written to the record)


def read_events(path: Path) -> Plan:
    rows = read_ndjson(path)
    if not rows or rows[0].get("kind") != "header":
        raise RunStop(f"{path}: no header line")
    body = rows[1:]
    if rows[0].get("n") is not None and rows[0]["n"] != len(body):
        raise RunStop(f"{path}: the header's n is {rows[0]['n']}, the file holds {len(body)} presses")
    if [r.get("i") for r in body] != list(range(len(body))):
        raise RunStop(f"{path}: i is not 0, 1, 2, ...")
    t = np.array([float(r["t_ms"]) for r in body])
    if not np.all(np.isfinite(t)) or np.any(np.diff(t) < 0):
        raise RunStop(f"{path}: planned times are not finite and non-decreasing")
    keys = [r["key"] for r in body]
    return Plan(t - t[0] if t.size else t, [key_kind(k) for k in keys], keys)


@dataclass
class Layer:
    times: np.ndarray     # ms, in capture order, from the first counted press (not sorted, not cleaned)
    kinds: list[str]
    not_counted: int      # events of a kind outside COUNTED_KINDS (modifiers, the stop key), left out and counted
    repeats: int = 0      # downs of a counted kind marked repeat, left out (v6's reading) and reported per series
    # (cleaning.repeats_left_out; the researcher's decision of 2026-11-06: returns-v3.5 saw Chrome mark one injected
    # press of 1558 as repeat on the page, none on the hook)


def _layer(times: list[float], kinds: list[str], not_counted: int, repeats: int = 0) -> Layer:
    t = np.array(times, dtype=float)
    if not np.all(np.isfinite(t)):
        raise RunStop("a recorded time is not finite")
    return Layer(t - t[0] if t.size else t, kinds, not_counted, repeats)


def need_types(path: Path, r: dict, types: dict) -> None:
    """The assumed fields exist with the assumed types (a string 'false' would otherwise read as true)."""
    for k, t in types.items():
        v = r.get(k)
        if v is None or not isinstance(v, t) or (t is not bool and isinstance(v, bool)):
            raise RunStop(f"{path}: field {k!r} missing or not {t} in {sorted(r)}")


def read_hook(path: Path, injected: bool, window_ns: tuple[int, int] | None = None) -> Layer:
    """ASSUMED['hook.ndjson']. Key downs only, auto-repeats left out; injected presses for a replay, not-injected ones
    for live typing (L0), which are read inside window_ns (the signal to the end chord) only."""
    ts, ks_, other, reps = [], [], 0, 0
    rows = read_ndjson(path)
    if not rows or rows[0].get("type") != "header" or rows[-1].get("type") != "footer":
        raise RunStop(f"{path}: no header and footer line (the hook did not finish writing)")
    for r in rows:
        if r.get("type") in ("header", "footer"):
            continue
        if r.get("type") != "key":
            raise RunStop(f"{path}: a line of unknown type {r.get('type')!r}")
        need_types(path, r, {"t_ns": int, "down": bool, "injected": bool, "ours": bool, "repeat": bool, "kind": str})
        if not r["down"]:
            continue
        # a replay's presses are those the capture injected (injected and ours, as the work PC's summary counts them);
        # live typing's are the not-injected ones
        mine = r["injected"] and r["ours"]
        if (injected and not mine) or (not injected and r["injected"]):
            continue
        if window_ns is not None and not window_ns[0] <= r["t_ns"] <= window_ns[1]:
            continue
        if r["repeat"]:   # left out as before; counted when of a counted kind (the selection above left unchanged)
            reps += r["kind"] in COUNTED_KINDS
            continue
        if r["kind"] in COUNTED_KINDS:
            ts.append(r["t_ns"] / 1e6)
            ks_.append(r["kind"])
        else:
            other += 1
    return _layer(ts, ks_, other, reps)


PAGE_OTHER_TYPES = ("open", "status", "ready", "armed", "log")   # capture/serve.py's other lines


def read_page(path: Path) -> Layer:
    """ASSUMED['page.ndjson']. Keydowns that are not repeats (the repeats of a counted kind are counted)."""
    ts, ks_, other, reps = [], [], 0, 0
    for r in read_ndjson(path):
        if r.get("type") in PAGE_OTHER_TYPES:
            continue
        if r.get("type") != "key":
            raise RunStop(f"{path}: a line of unknown type {r.get('type')!r}")
        need_types(path, r, {"t_ms": (int, float), "kind": str, "repeat": bool, "trusted": bool, "composing": bool,
                             "in_box": bool})
        if r["repeat"]:   # counted as presses are: of a counted kind, in the box and not composing (code-reviewer)
            reps += r["kind"] in COUNTED_KINDS and r["in_box"] and not r["composing"]
            continue
        if r["kind"] in COUNTED_KINDS and (not r["in_box"] or r["composing"]):
            raise RunStop(f"{path}: a counted press outside the box or while composing (the work PC's checks make "
                          "such a capture interrupted)")
        if r["kind"] in COUNTED_KINDS:
            ts.append(float(r["t_ms"]))
            ks_.append(r["kind"])
        else:
            other += 1
    return _layer(ts, ks_, other, reps)


def read_inject(path: Path) -> dict:
    """ASSUMED['inject.ndjson']: the waits (sent minus planned, ms), the queue lengths and the failed SendInput calls."""
    rows = [r for r in read_ndjson(path) if r.get("type") != "header"]
    if not rows:
        raise RunStop(f"{path}: no press")
    for r in rows:
        need_types(path, r, {"planned_ns": int, "sent_ns": int, "ok": bool})
    wait = np.array([(r["sent_ns"] - r["planned_ns"]) / 1e6 for r in rows])
    queue = None
    if any("queue_len" in r for r in rows):   # M1's typist only, then on every row
        for r in rows:
            need_types(path, r, {"queue_len": int})
        queue = np.array([r["queue_len"] for r in rows])
    return {"wait_ms": wait, "queue": queue, "failed": int(sum(1 for r in rows if not r["ok"]))}


def read_llm_stream(path: Path, unfinished: bool = False) -> tuple[np.ndarray, list[str], float, int]:
    """ASSUMED['llm.ndjson (M1)']: the arrivals (ms) and texts (NFC) of the non-empty tokens, the end of generation
    and the number of empty tokens skipped (series.M1). The done chunk's text, if any, is a last token arriving at the
    end. The texts stay on the work PC (used only to recompute the schedule and check its keys).
    unfinished (a failed capture whose typing stopped before the generation ended): with no line ending the
    generation, the last token was never released by the work PC's planner, so it is left out and its arrival is
    the end of the tokens before it."""
    arr, toks, empty, end = [], [], 0, None
    for r in read_ndjson(path):
        if r.get("type") == "error":
            raise RunStop(f"{path}: the generation failed ({r.get('error')})")
        if r.get("type") != "chunk":
            continue
        if end is not None:
            raise RunStop(f"{path}: a chunk after the one that ends the generation")
        need_types(path, r, {"recv_ns": int, "token": str, "done": bool})
        tok = unicodedata.normalize("NFC", r["token"])
        if r["done"]:
            end = r["recv_ns"] / 1e6
            if not tok:
                continue
        elif not tok:
            empty += 1
            continue
        arr.append(r["recv_ns"] / 1e6)
        toks.append(tok)
    if end is None:
        if not unfinished or len(arr) < 2:
            raise RunStop(f"{path}: no line ends the generation")
        end = arr.pop()
        toks.pop()
    return np.array(arr), toks, float(end), empty


def typed_key(text: str, j: int) -> str | None:
    """The key typed for the j-th code point of a token (capture code llm_stream.token_keys): \\n is Enter, the \\r of
    \\r\\n and characters outside printable ASCII are skipped (None; the researcher's decision E-5 of 2026-11-04)."""
    c = text[j]
    if c == "\n":
        return "Enter"
    if c == "\r":
        return None
    return c if 0x20 <= ord(c) <= 0x7E else None


def m1_schedule(arr: np.ndarray, toks: list[str], end: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """M1's schedule recomputed as the work PC types it: B13's wished times over every code point (v7.m1_wished_times),
    skipped code points removed (their slots stay empty), then the running maximum over the typed ones
    (v7.executable_plan). Returns the typed presses' wished and determined times, the executable plan (from the first
    typed press) and the keys."""
    wished, det = v7.m1_wished_times(arr, np.array([len(x) for x in toks]), end)
    keys = [typed_key(x, j) for x in toks for j in range(len(x))]
    keep = np.array([k is not None for k in keys], dtype=bool)
    if not keep.any():
        raise RunStop("M1: no typable character in the generated text")
    w, d = wished[keep], det[keep]
    ex = v7.executable_plan(w, d)
    return w, d, ex - ex[0], [k for k in keys if k is not None]


def read_runlog(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    rows = read_ndjson(path)
    if not rows or rows[0].get("type") != "header" or rows[-1].get("type") != "footer":
        raise RunStop(f"{path}: no header and footer line (the runlist did not finish)")
    for r in rows[1:-1]:
        if r.get("type") != "item":
            raise RunStop(f"runlog: a line of unknown type {r.get('type')!r}")
        need_types(path, r, {"item": str, "result": str})
        if r["item"] in out:
            raise RunStop(f"runlog: {r['item']} appears twice")
        if r.get("result") not in ("done", "redone", "failed", "no_plan"):
            raise RunStop(f"runlog: unknown result {r['result']!r}")
        out[r["item"]] = r["result"]
    return out


N_RUN_BLOCKS = 7   # run_order (amendment B16): block 0 (L0) and blocks 1-6


RUN_KEY = [v7.SEED, 7, 6]   # run_order.randomised


def runlog_files(root: Path) -> list[Path]:
    """One runlog.ndjson (every block in one run), or runlog.b0.ndjson to runlog.b6.ndjson (one run per block, capture
    runlist --block; amendment B16), never both."""
    blocks = sorted(root.glob("runlog.b*.ndjson"))
    single = root / "runlog.ndjson"
    if not blocks:
        return [single]
    if single.exists():
        raise RunStop(f"{root} holds runlog.ndjson and per-block runlogs: one of them is not of this run")
    want = [root / f"runlog.b{b}.ndjson" for b in range(N_RUN_BLOCKS)]
    if blocks != want:
        raise RunStop(f"the per-block runlogs are not blocks 0-{N_RUN_BLOCKS - 1}: {[b.name for b in blocks]}")
    return want


def runlog_headers(root: Path) -> list[dict]:
    """The header line of each runlog: the run key of the conditions, the block of its file name, and (per-block
    runs) one runs.json for all blocks (its SHA-256 in every header)."""
    files = runlog_files(root)
    heads = []
    for b, f in enumerate(files):
        rows = read_ndjson(f)
        if not rows or rows[0].get("type") != "header":
            raise RunStop(f"{f}: no header line")
        h = rows[0]
        if h.get("key") != RUN_KEY:
            raise RunStop(f"{f}: the run key is {h.get('key')}, not {RUN_KEY} (run_order.randomised)")
        want_block = b if len(files) > 1 else None
        if h.get("block") != want_block or not isinstance(h.get("runs_sha256"), str):
            raise RunStop(f"{f}: the header does not name block {want_block} and the runs.json it ran")
        heads.append(h)
    if len({h.get("runs_sha256") for h in heads}) != 1:
        raise RunStop("the blocks were run from different runs.json files")
    return heads


def read_runlogs(root: Path) -> dict[str, str]:
    """Every item's result over runlog_files(); an item appears once over all of them."""
    runlog_headers(root)
    out: dict[str, str] = {}
    for f in runlog_files(root):
        for item, res in read_runlog(f).items():
            if item in out:
                raise RunStop(f"runlog: {item} appears in two blocks")
            out[item] = res
    return out


# ---------------------------------------------------------------- captures


@dataclass
class Capture:
    """One item of the runlist. status: 'captured', 'failed' (a missing pair: the plan or original exists, the
    capture failed twice) or 'no_plan' (outside the pairs: lowers the clusters, amendment B14)."""
    series: str
    unit: int
    status: str
    plan: Plan | None = None
    layers: dict[str, Layer] = field(default_factory=dict)
    inject: dict | None = None
    redone: bool = False
    folder: Path | None = None


def item_name(series: str, unit: int) -> str:
    return f"{series}/{CAPTURE_LAYOUT[series].format(i=unit)}"


def plan_file(series: str) -> str:
    """M1 writes its plan as it types (schedule.ndjson); every other series types events.ndjson."""
    return "schedule.ndjson" if series in ("M1_8B", "M1_14B") else "events.ndjson"


def typing_window(d: Path) -> tuple[int, int]:
    """ASSUMED['meta.json']: the live capture's typing window on the hook's clock."""
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    need_types(d / "meta.json", meta, {"typing_start_perf_ns": int, "typing_end_perf_ns": int})
    if meta["typing_end_perf_ns"] < meta["typing_start_perf_ns"]:
        raise RunStop(f"{d}/meta.json: the typing window ends before it starts")
    return meta["typing_start_perf_ns"], meta["typing_end_perf_ns"]


def load_capture(root: Path, runlog: dict[str, str], series: str, unit: int, *, live: bool = False,
                 hook: bool = True) -> Capture:
    name = item_name(series, unit)
    if name not in runlog:
        raise RunStop(f"{name}: not in the runlog (every planned capture must be accounted for)")
    res = runlog[name]
    d = root / name
    if res == "no_plan":
        return Capture(series, unit, res)
    if res == "failed":
        # a failed capture is a missing pair whose plan exists (amendment B14): the runlist keeps its plan files
        # (events.ndjson; for M1 also llm.ndjson and schedule.ndjson) so that the plan side and its dependants
        # (M0, M3, M1_replay) can still be checked and scored
        cap = Capture(series, unit, "failed", folder=d if d.is_dir() else None)
        if not live and d.is_dir() and (d / plan_file(series)).exists():
            cap.plan = read_events(d / plan_file(series))
            if cap.plan.times.size < 2:   # e.g. M1 with nothing typable: no plan to check or to place back
                cap.plan = None
        return cap
    if not d.is_dir():
        raise RunStop(f"{name}: the runlog says {res} but the folder is missing")
    cap = Capture(series, unit, "captured", redone=res == "redone", folder=d)
    cap.plan = None if live else read_events(d / plan_file(series))
    if hook:
        cap.layers["hook"] = read_hook(d / "hook.ndjson", injected=not live, window_ns=typing_window(d) if live else None)
    cap.layers["page"] = read_page(d / "page.ndjson")
    if live and hook and cap.layers["hook"].kinds != cap.layers["page"].kinds:
        raise RunStop(f"{name}: L0's hook and page layers do not hold the same presses (the page records from the "
                      "signal, the hook is read from it: a press on one logger only)")
    planned = live or (cap.plan is not None and cap.plan.times.size > 0)
    for la, layer in cap.layers.items():   # a logger whose lines were all read as something else is not 'no press'
        if planned and layer.times.size == 0:
            raise RunStop(f"{name}: {la}.ndjson holds no counted press (a format other than ASSUMED, or the logger "
                          "recorded nothing although the runlist said done)")
    if not live and (d / "inject.ndjson").exists():
        cap.inject = read_inject(d / "inject.ndjson")
    return cap


def load_all(root: Path) -> dict[str, list[Capture]]:
    """Every capture of series.counts_of_captures, by series in unit order."""
    runlog = read_runlogs(root)
    units = {"H1_idle": range(N_H), "H1_load": range(N_H), "H1_rep": range(N_REP), "L0": range(N_ESSAYS),
             "L1_idle": range(N_ESSAYS), "L1_load": range(N_ESSAYS), **{s_: range(N_TEXTS) for s_ in MACHINE},
             "P_idle": [int(c) for c in CONTROLS_MS], "P_load": [int(c) for c in CONTROLS_MS],
             "P_nohook": [int(c) for c in v7.P_NOHOOK_MS]}
    out = {s_: [load_capture(root, runlog, s_, u, live=s_ == "L0", hook=s_ != "P_nohook") for u in us]
           for s_, us in units.items()}
    known = {item_name(s_, u) for s_, us in units.items() for u in us}
    stray = sorted(set(runlog) - known)
    if stray:
        raise RunStop(f"runlog items outside the conditions: {stray[:5]}")
    return out


# ---------------------------------------------------------------- persons (persons.pool, replay_H, estimation_E)


def eligible_v6(essay: dict, copy_: dict, pid: str) -> bool:
    """participants.draw of prototype_v6.json: both tasks eligible as v6's run() checks them."""
    for x in (copy_.get(pid), essay.get(pid)):
        if x is None or x["times"].size < 2:
            return False
        st = v6.kupa_starts(x["kinds"])
        s = v6.Series(x["times"], st, v6.kupa_word_starts(x["kinds"]))
        if len(st) < 5 or int(np.floor(s.times[-1] / v6.WINDOW_MS)) < 5 or not v6.every_sentence_holds_a_gap(s):
            return False
    return True


def sha_ids(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


@dataclass
class Persons:
    pool: list[str]
    h: list[str]           # replay_H in draw order (unit i of H1 is h[i])
    e: list[str]           # estimation_E in the pool's draw order
    donors: np.ndarray     # index into e, text order (series.M3a)
    e_train: np.ndarray    # index into e
    looked: int


def draw_persons(essay: dict, copy_: dict, seen_essay: set[str], v6_sha: str | None) -> Persons:
    """persons.pool: v6's walk (default_rng([20261031, 6, 0]).permutation of the sorted ids) continued past v6's 200
    to 300 eligible persons; v6's 200 must be its first 200 (their SHA-256 checked when v6_sha is given). replay_H:
    the first 80 of the pool with at least 300 counted essay presses. E: the rest of the pool, in draw order."""
    pool, looked = [], 0
    for pid in v6.draw_order(sorted(seen_essay), 0):
        looked += 1
        if eligible_v6(essay, copy_, pid):
            pool.append(pid)
            if len(pool) == N_POOL:
                break
    if len(pool) < N_POOL:
        raise RunStop(f"KUPA: only {len(pool)} eligible persons for the pool of {N_POOL}")
    if v6_sha is not None and sha_ids(pool[:N_V6]) != v6_sha:
        raise RunStop("the first 200 of the pool are not v6's 200")
    h = [p for p in pool if essay[p]["times"].size >= MIN_PRESSES][:N_H]
    if len(h) < N_H:
        raise RunStop(f"only {len(h)} pool persons hold {MIN_PRESSES} counted presses")
    hs = set(h)
    e = [p for p in pool if p not in hs]
    donors, train = v7.e_split(len(e))
    return Persons(pool, h, e, donors, train, looked)


# ---------------------------------------------------------------- plans (checked against what they must be)


TOL_MS = 1e-3   # planned times are written in ms as floats; ns rounding on the work PC stays far below this


def same_times(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and bool(np.all(np.abs(a - b) <= TOL_MS))


def l0_refs(caps: dict[str, list[Capture]]) -> list:
    """What L1 must replay: each captured L0's hook layer, cleaned as v6 (equal times dropped); None for an L0 not
    captured."""
    out = []
    for cp in caps["L0"]:
        if cp.status != "captured":
            out.append(None)
            continue
        t_, idx, _ = v7.clean_times(cp.layers["hook"].times)
        out.append((t_, [cp.layers["hook"].kinds[i] for i in idx]))
    return out


def check_plans(caps: dict[str, list[Capture]], h0: list[dict], e_gaps: list[np.ndarray],
                l0_hook: list[np.ndarray | None]) -> dict:
    """The plan of every capture is what the conditions say it is, or the run stops:
    H1_idle, H1_load, H1_rep = H0's counted presses (times and kinds); L1 = L0's hook-layer times; M0 = a constant
    55 ms on the text of M1_8B; M1_replay = its M1's executable plan; M3a, M3b = m3() of the donors (keys of M0's
    text); M1 = the executable plan recomputed from the token arrivals; P = the constant controls. M2's plan is
    the LLM's validated plan (only the events format is checked here; the JSON validation is the work PC's). A
    failed capture keeps its plan (load_capture), so its dependants are checked against it. Returns M1's plan shift, zero-gap shares and empty tokens."""
    out: dict = {"M1": {}}
    for s_ in ("H1_idle", "H1_load", "H1_rep"):
        for cap in caps[s_]:
            if cap.plan is None:
                continue
            x = h0[cap.unit]
            if not same_times(cap.plan.times, x["times"]) or [v7.LOGGER_KIND_TO_V6[k] for k in cap.plan.kinds] != list(
                    x["kinds"]):
                raise RunStop(f"{item_name(s_, cap.unit)}: the plan is not H0's presses")
    for s_ in ("L1_idle", "L1_load"):
        for cap in caps[s_]:
            if cap.plan is None:
                continue
            ref = l0_hook[cap.unit]
            if ref is None:
                raise RunStop(f"{item_name(s_, cap.unit)}: captured although L0 has no hook layer")
            if not same_times(cap.plan.times, ref[0]) or cap.plan.kinds != ref[1]:
                raise RunStop(f"{item_name(s_, cap.unit)}: the plan is not L0's hook-layer presses")
    for model in ("8B", "14B"):
        m1 = caps[f"M1_{model}"]
        plan_shift, zero, empty, failed_no_plan = [], [], 0, 0
        for cap in m1:
            if cap.plan is None:
                continue
            try:
                arr, toks, end, emp = read_llm_stream(cap.folder / "llm.ndjson", unfinished=cap.status == "failed")
                wished, det, ex, keys = m1_schedule(arr, toks, end)
            except (RunStop, ValueError):
                if cap.status != "failed":
                    raise
                cap.plan = None   # a failed capture with too little generated to recompute: no plan (counted)
                failed_no_plan += 1
                continue
            empty += emp
            if not same_times(cap.plan.times, ex):
                raise RunStop(f"{item_name(cap.series, cap.unit)}: the schedule is not the executable plan of B13")
            if cap.plan.keys != keys:
                raise RunStop(f"{item_name(cap.series, cap.unit)}: the schedule's keys are not the generated text")
            plan_shift.append(v7.plan_shift(wished, det))
            zero.append(v7.plan_zero_gap_share(cap.plan.times))
        out["M1"][model] = {
            "sessions_with_plan": len(zero), "empty_tokens_skipped": empty, "failed_without_plan": failed_no_plan,
            "share_moved_median": float(np.median([p["share_moved"] for p in plan_shift])) if plan_shift else None,
            "median_shift_ms_median": _median_or_none([p["median_shift"] for p in plan_shift]),
            "zero_gap_share_median": float(np.median(zero)) if zero else None,
            "zero_gap_share_max": float(np.max(zero)) if zero else None}
        for rep in caps[f"M1_replay_{model}"]:
            src = m1[rep.unit]
            if rep.plan is None:
                continue
            if src.plan is None:
                raise RunStop(f"{item_name(rep.series, rep.unit)}: captured although its M1 has no plan")
            if not same_times(rep.plan.times, src.plan.times) or rep.plan.keys != src.plan.keys:
                raise RunStop(f"{item_name(rep.series, rep.unit)}: not the schedule of its M1")
    for cap in caps["M0"]:
        if cap.plan is None:
            continue
        src = caps["M1_8B"][cap.unit]
        if src.plan is None or cap.plan.keys != src.plan.keys:
            raise RunStop(f"{item_name('M0', cap.unit)}: not the text of M1_8B")
        if not np.all(np.abs(np.diff(cap.plan.times) - M0_GAP_MS) <= TOL_MS):
            raise RunStop(f"{item_name('M0', cap.unit)}: the gaps are not {M0_GAP_MS} ms")
    for t in range(N_TEXTS):
        m0 = caps["M0"][t]
        drawn = v7.m3(t, len(m0.plan.keys) - 1, e_gaps) if m0.plan is not None else None
        for s_ in ("M3a", "M3b"):
            cap = caps[s_][t]
            if cap.plan is None:
                continue
            if drawn is None or cap.plan.keys != m0.plan.keys:
                raise RunStop(f"{item_name(s_, t)}: not the keys of M0's text")
            if not same_times(np.diff(cap.plan.times), drawn[s_]):
                raise RunStop(f"{item_name(s_, t)}: the gaps are not m3()'s draw")
    for s_ in ("P_idle", "P_load", "P_nohook"):
        for cap in caps[s_]:
            if cap.plan is None:
                continue
            if cap.plan.times.size != P_PRESSES or not np.all(np.abs(np.diff(cap.plan.times) - cap.unit) <= TOL_MS):
                raise RunStop(f"{item_name(s_, cap.unit)}: not {P_PRESSES} presses at {cap.unit} ms")
    return out


def _median_or_none(v: list) -> float | None:
    v = [x for x in v if x is not None]
    return float(np.median(v)) if v else None


# ---------------------------------------------------------------- series on a layer


@dataclass
class Recorded:
    """One capture on one layer after matching_and_cleaning (v6's rule, or ties kept)."""
    gaps: np.ndarray
    times: np.ndarray
    kinds: list[str]
    kept: np.ndarray
    dropped: int
    inversions: int


def recorded(cap: Capture, layer: str, drop_ties: bool = True) -> Recorded | None:
    if cap.status != "captured" or layer not in cap.layers:
        return None
    la = cap.layers[layer]
    inv = v7.capture_inversions(la.times)   # amendment B14: counted in capture order, before the stable sort
    t, idx, dropped = v7.clean_times(la.times, drop_ties)
    return Recorded(v7.gaps_of(t), t, [la.kinds[i] for i in idx], idx, dropped, inv)


def plan_gaps(cap: Capture) -> np.ndarray | None:
    """The machine side's plan, not cleaned (series.M1, amendment B14): its gaps as planned, zero gaps kept."""
    return None if cap.plan is None else v7.gaps_of(cap.plan.times)


# ---------------------------------------------------------------- detectors and scores


@dataclass
class Scorer:
    d2: v7.D2
    d3: v7.D3
    d4: v7.D4

    def score(self, sessions: list[np.ndarray | None]) -> dict[str, np.ndarray]:
        """Per session: D1's delta, the 'machine' probabilities of D2_lr, D2_rf, D3 and D4, D2_lr's logit and the
        gaps left out of D2's histogram; NaN (no judgment) for a missing session or one too short for a detector
        (D1 and D2 need 2 gaps after the trim and a mean above 0, D3 70 gaps, D4 30 gaps)."""
        n = len(sessions)
        out = {k: np.full(n, np.nan) for k in (*DETECTORS, "D2_lr_logit")}
        above = np.zeros(n, dtype=int)
        ok12 = [i for i, g in enumerate(sessions) if g is not None and _condrey_scorable(g)]
        if ok12:
            sub = [sessions[i] for i in ok12]
            out["D1"][ok12] = [v7.delta(g) for g in sub]
            p = self.d2.proba(sub)
            for k in ("D2_lr", "D2_rf", "D2_lr_logit"):
                out[k][ok12] = p[k]
            above[ok12] = p["above_hist"]
        present = [i for i, g in enumerate(sessions) if g is not None]
        if present:
            sub = [sessions[i] for i in present]
            out["D3"][present] = self.d3.proba(sub)
            out["D4"][present] = self.d4.proba(sub)
        out["above_hist"] = above
        return out


def _condrey_scorable(g: np.ndarray) -> bool:
    if g.size == 0 or g.mean() <= 0:
        return False
    t = v7.trimmed(g)
    return t.size >= 2 and t.mean() > 0


def train(e_train: list[np.ndarray]) -> Scorer:
    """D2, D3 and D4 trained on H0 of E_train with their training streams (analysis_details.D2_models, D3_windows;
    detectors_C2.D4)."""
    return Scorer(v7.train_d2(e_train, np.random.default_rng(v7.D2_SEED)),
                  v7.train_d3(e_train, np.random.default_rng(v7.D3_SEED)),
                  v7.train_d4(e_train, np.random.default_rng(v7.D4_SEED)))


def judgments(det: str, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """'machine' judgments and the mask of sessions that have one (D1: delta <= 0.269; others probability > 0.5)."""
    s = np.asarray(s, dtype=float)
    ok = np.isfinite(s)
    machine = np.zeros(s.size, dtype=bool)
    machine[ok] = (s[ok] <= v7.DELTA_T) if det == "D1" else (s[ok] > 0.5)
    return machine, ok


MIN_SUMMARY = 5   # below this many values a distribution of persons is given as its count only


def summary(x, ends: bool = False) -> dict | None:
    """Quantiles of a distribution. ends False (persons' values): p05 to p95 only and nothing but the count below
    MIN_SUMMARY values, since a minimum or maximum, or a quantile of one value, is one person's value. ends True for
    the replay system's measurements (replay error, waits), whose maxima the predictions R1 and Hd3 read."""
    x = np.asarray([v for v in x if v is not None], dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return None
    if not ends and x.size < MIN_SUMMARY:
        return {"n": int(x.size)}
    names, qs = ("p05", "p25", "median", "p75", "p95"), [5, 25, 50, 75, 95]
    if ends:
        names, qs = ("min", *names, "max"), [0, *qs, 100]
    return {"n": int(x.size), **{k: float(v) for k, v in zip(names, np.percentile(x, qs))}}


# ---------------------------------------------------------------- rows of detectors_C2


def paired_human_row(name: str, layer: str, before: dict, after: dict, failed_after: np.ndarray,
                     gaps_before: list, gaps_after: list, kept_before: list | None = None,
                     kept_after: list | None = None, read: bool = True) -> dict:
    """A paired row of detectors_C2.count (persons in unit order): per detector b, c, n, the missing pairs (failed
    capture and no judgment apart), Tango's and Wilson's intervals and the reading (read False: reported only, never
    read, amendment B9), the three missing-data bounds, whether the cell can enter a general statement, the mean
    paired score change with its bootstrap over persons (analysis_details.keys), D2_lr's logit beside it, saturation
    on both sides and the share of pairs whose used gap sets differ (amendment B14)."""
    out = {}
    for det in DETECTORS:
        jb, okb = judgments(det, before[det])
        ja, oka = judgments(det, after[det])
        ok = okb & oka
        b, c, n = v7.flips(jb[ok], ja[ok])
        missing = int((~ok).sum())
        cell: dict = {"b": b, "c": c, "n": n, "missing_pairs": missing,
                      "missing_failed_capture": int((~ok & failed_after).sum()),
                      "missing_no_judgment": int((~ok & ~failed_after).sum()), "read": read}
        if not read:   # amendment B9: Tango's interval reported only, never read (no reading, bounds or entry)
            if n > 0:
                cell["tango_95"] = list(v7.tango_interval(b, c, n))
        else:
            if n > 0:
                cell["headline"] = v7.headline(b, c, n)
            if n + missing > 0:
                cell["bounds"] = v7.boundary_readings(b, c, n, missing)
            cell["enters_general"] = bool(n > 0 and v7.general_cell_ok(b, c, n, missing))
        key = [S, v7.VERSION, 9, LAYER_IDX[layer], DET_IDX[det], PAIR_IDX[name]]
        if n > 0:
            cell["score_change"] = v7.paired_prob_change(before[det], after[det], key)
            if det == "D2_lr":   # amendment B9: the logit beside the probability (the same resamples of persons)
                cell["score_change_logit"] = v7.paired_prob_change(before["D2_lr_logit"], after["D2_lr_logit"], key)
        if det in PROB_DETECTORS:
            cell["saturation"] = {side: (v7.saturation_share(s_[det]) if np.isfinite(s_[det]).any() else None)
                                  for side, s_ in (("before", before), ("after", after))}
        both = [i for i in range(len(gaps_before)) if gaps_before[i] is not None and gaps_after[i] is not None]
        if both:
            diff = []
            for i in both:
                kb = None if kept_before is None else kept_before[i]
                ka = None if kept_after is None else kept_after[i]
                diff.append(v7.used_set_differs(det, gaps_before[i], gaps_after[i], *((kb, ka) if kb is not None
                                                                                       and ka is not None else ())))
            cell["used_gaps_differ_share"] = float(np.mean(diff))
        out[det] = cell
    return out


def cluster_mean_change(before: np.ndarray, after: np.ndarray, cluster: np.ndarray, key: list[int]) -> dict:
    """The machine_path rows' paired score change (amendment A2): the mean of after - before with a 95% percentile
    interval over N_BOOT resamples of clusters, drawn as machine_path_reading draws them (the same key gives the same
    resamples)."""
    d = np.asarray(after, dtype=float) - np.asarray(before, dtype=float)
    ids = np.unique(cluster)
    rows = [np.flatnonzero(cluster == i) for i in ids]
    rng = np.random.default_rng(key)
    m = np.empty(v7.N_BOOT)
    for k in range(v7.N_BOOT):
        m[k] = d[np.concatenate([rows[i] for i in rng.integers(0, ids.size, ids.size)])].mean()
    return {"mean": float(d.mean()), "ci95": [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))],
            "n": int(d.size)}


def machine_path_cell(series: str, layer: str, det: str, plan_s: np.ndarray, rec_s: np.ndarray,
                      status: list[str], plan_g: list | None = None, rec_g: list | None = None) -> dict:
    """One cell of pairs.machine_path: the plan's judgment against the recorded one, sessions in text order.
    Sessions without a plan are outside the pairs (counted as lost clusters); failed captures and sessions without a
    judgment are missing pairs (amendment B14)."""
    has_plan = np.array([st_ != "no_plan" for st_ in status])
    jb, okb = judgments(det, plan_s)
    ja, oka = judgments(det, rec_s)
    ok = okb & oka & has_plan
    failed = np.array([st_ == "failed" for st_ in status])
    cell: dict = {"sessions_planned": len(status), "without_plan": int((~has_plan).sum()),
                  "missing_pairs": int((has_plan & ~ok).sum()), "missing_failed_capture": int((has_plan & failed).sum()),
                  "missing_no_judgment": int((has_plan & ~failed & ~ok).sum())}
    clusters = np.flatnonzero(ok)   # the cluster of a session is its text (or prompt, or donor: one per text)
    cell["n_clusters"] = int(clusters.size)
    if clusters.size:
        key = [S, v7.VERSION, 11, LAYER_IDX[layer], DET_IDX[det], MS_IDX[series]]
        cell.update(v7.machine_path_reading(jb[ok], ja[ok], clusters.astype(int), key))
        cell["score_change"] = cluster_mean_change(plan_s[ok], rec_s[ok], clusters.astype(int), key)
        cell["plan_machine"] = int(jb[ok].sum())
        cell["recorded_machine"] = int(ja[ok].sum())
    else:
        cell["reading"] = None
    # detectors_C2.headline (B11, B14): the three missing-data bounds beside every reading. One session per text, so
    # the effective n equals n and b, c need no scaling here
    n_ = int(ok.sum())
    if n_ + cell["missing_pairs"] > 0:
        b_, c_ = (cell["b"], cell["c"]) if n_ else (0, 0)
        cell["bounds"] = v7.boundary_readings(b_, c_, n_, cell["missing_pairs"])
    if det in PROB_DETECTORS:   # amendment B9: saturation in every row
        cell["saturation"] = {side: (v7.saturation_share(x[has_plan]) if np.isfinite(x[has_plan]).any() else None)
                              for side, x in (("plan", plan_s), ("recorded", rec_s))}
    if plan_g is not None and rec_g is not None:   # amendment B14: share of pairs whose used gap sets differ
        both = [i for i in range(len(status)) if has_plan[i] and plan_g[i] is not None and rec_g[i] is not None]
        if both:
            cell["used_gaps_differ_share"] = float(np.mean([v7.used_set_differs(det, plan_g[i], rec_g[i])
                                                            for i in both]))
    return cell


def count_cell(det: str, s_: np.ndarray, layer: str | None, series: str, status: list[str] | None = None,
               interval: bool = True) -> dict:
    """detectors_C2.count: the share of sessions judged 'machine' (sessions without a judgment removed and counted);
    Clopper-Pearson for independent sessions, the cluster bootstrap over texts for machine series on a layer; the
    machine side's plan (no layer, no key in the conditions) is counted with no interval (interval False)."""
    j, ok = judgments(det, s_)
    if status is not None:
        ok = ok & np.array([st_ != "no_plan" for st_ in status])
    k, n = int(j[ok].sum()), int(ok.sum())
    out: dict = {"machine": k, "n": n, "removed": int(s_.size - n), "share": k / n if n else None}
    if n == 0:
        return out
    if not interval:
        pass
    elif series in MS_IDX and layer is not None:
        key = [S, v7.VERSION, 5, LAYER_IDX[layer], DET_IDX[det], MS_IDX[series]]
        lo, hi, ncl = v7.cluster_bootstrap_share(j[ok], np.flatnonzero(ok), key)
        out.update(cluster_bootstrap_95=[lo, hi], n_clusters=ncl)
    elif series not in MS_IDX:
        out["clopper_pearson_95"] = list(v7.clopper_pearson(k, n))
    if det in PROB_DETECTORS:
        out["saturation"] = v7.saturation_share(s_[ok])
    return out


# ---------------------------------------------------------------- replay error and the controls


def replay_error_series(caps: list[Capture], layer: str, planned: list[np.ndarray | None] | None = None,
                        planned_kinds: list[list[str] | None] | None = None) -> dict:
    """pairs.replay_error on every capture of a series on one layer (analysis_details.replay_error: every recorded
    event kept, times from the first press; deviations only when 'matched'), summarised over captures."""
    per = []
    for i, cap in enumerate(caps):
        if cap.status != "captured" or layer not in cap.layers:
            continue
        p = cap.plan.times if planned is None else planned[i]
        pk = cap.plan.kinds if planned_kinds is None else planned_kinds[i]
        if p is None:
            continue
        la = cap.layers[layer]
        per.append(v7.replay_error(p, la.times, pk, la.kinds))
    if not per:
        return {"captures": 0}
    m = [x for x in per if x["matched"]]
    return {"captures": len(per), "matched": len(m), "share_matched": len(m) / len(per),
            "drops": int(sum(x["drops"] for x in per)), "extras": int(sum(x["extras"] for x in per)),
            "kind_edit_distance": int(sum(x["kind_edit_distance"] or 0 for x in per)),
            "offset_ms": summary([x["offset_ms"] for x in m], ends=True), "jitter_mad_ms": summary([x["jitter_mad_ms"] for x in m], ends=True),
            "jitter_p95_ms": summary([x["jitter_p95_ms"] for x in m], ends=True),
            "abs_drift_ms": summary([abs(x["drift_ms"]) for x in m], ends=True),
            "all_matched": len(m) == len(per)}


def raw_gaps(cap: Capture, layer: str) -> np.ndarray | None:
    """Gaps in capture order with nothing dropped (the controls' medians and their sub-millisecond share: cleaning
    would remove the equal times that batching makes)."""
    if cap.status != "captured" or layer not in cap.layers:
        return None
    return np.diff(cap.layers[layer].times)


def control_stats(g: np.ndarray, planned_ms: float) -> dict:
    med = float(np.median(g))
    a = np.abs(g - med)
    return {"median_ms": med, "median_minus_planned_ms": med - planned_ms, "mad_ms": float(np.median(a)),
            "p95_abs_dev_ms": float(np.percentile(a, 95)), "share_below_1ms": float(np.mean(g < 1.0)), "n_gaps": int(g.size)}


# ---------------------------------------------------------------- the run


def run(c: dict, h0: list[dict], e_gaps: list[np.ndarray], e_train_idx: np.ndarray, caps: dict[str, list[Capture]],
        m: dict, r: dict) -> dict:
    """h0: the replay persons' counted essay presses ({times, kinds}) in replay_H's order; e_gaps: H0 gaps of E in
    the pool's order (donors index into it); caps: load_all()'s captures; m, r: v4's mapping and relief (C1's
    stages, as v6)."""
    rec: dict = {"assumed_formats": ASSUMED, "capture_layout": CAPTURE_LAYOUT}
    nh = len(h0)
    h0_gaps = [v7.gaps_of(x["times"]) for x in h0]
    e_train = [e_gaps[i] for i in e_train_idx]

    # plans
    rec["plans"] = check_plans(caps, h0, e_gaps, l0_refs(caps))

    # resolution (analysis_details.resolution)
    res = {"H0_E": v7.resolution(np.concatenate(e_gaps), *RES_ARGS),
           "H0_H": v7.resolution(np.concatenate(h0_gaps), *RES_ARGS)}
    for la in LAYERS:
        for s_ in ("H1_idle", "P_idle"):
            g = [x.gaps for x in (recorded(cp, la) for cp in caps[s_]) if x is not None]
            res[f"{s_}_{la}"] = v7.resolution(np.concatenate(g), *RES_ARGS) if g else None
    rec["resolution"] = res
    grid_e = res["H0_E"]["resolution_ms"] if res["H0_E"]["stable"] else None
    grid_h = res["H0_H"]["resolution_ms"] if res["H0_H"]["stable"] else None

    # detectors and the validity gate
    gate = v7.validity_gate(e_train, grid_ms=grid_e)
    valid = {d: gate[v7.GATE_ENTRY[d]]["valid"] for d in DETECTORS}
    if set(valid) != set(DETECTORS) or len(DETECTORS) != 5:
        raise RunStop("the general statements need the validity of all five detectors")
    rec["validity_gate"] = {"E_train": len(e_train), "grid_ms": grid_e, "gate": gate, "valid": valid,
                            "grid_note": {d: v7.grid_note(gate, d) for d in DETECTORS}}
    sc = train(e_train)
    rec["training"] = {"D2_above_hist_train": int(sc.d2.train_above_hist.sum()), "D4_kde_redrawn": sc.d4.kde.redrawn}

    # sessions and scores
    def rec_list(series: str, la: str, drop_ties: bool = True) -> list[Recorded | None]:
        return [recorded(cp, la, drop_ties) for cp in caps[series]]

    scored: dict = {}

    def scores_of(tag, gaps: list) -> dict:
        if tag not in scored:
            scored[tag] = sc.score(gaps)
        return scored[tag]

    s_h0 = scores_of(("H0",), h0_gaps)
    rec["margin_on_H0"] = {d: summary(v7.threshold_margin(d, s_h0[d])) for d in DETECTORS}
    rec["cleaning"] = {"ties_kept_note": "on the human side only the recorded series keep equal times; H0 is KUPA's "
                                         "counted presses, which drop them by definition (data.counted_press), and "
                                         "H1 replays only those"}
    rows: dict = {}
    counts: dict = {"H0": {d: count_cell(d, s_h0[d], None, "H0") for d in DETECTORS}}
    above_hist: dict = {"H0": int(s_h0["above_hist"].sum())}
    for ties in (True, False):
        tag = "main" if ties else "ties_kept"
        rows[tag] = {}
        for la in LAYERS:
            hs = {s_: rec_list(s_, la, ties) for s_ in ("H1_idle", "H1_load", "H1_rep", "L0", "L1_idle", "L1_load")}
            sh = {s_: scores_of((s_, la, tag), [x.gaps if x else None for x in v]) for s_, v in hs.items()}
            failed = {s_: np.array([cp.status == "failed" for cp in caps[s_]]) for s_ in hs}
            g0 = list(h0_gaps)
            k0 = [np.arange(g.size + 1) for g in h0_gaps]   # H0 is KUPA's counted presses, already cleaned
            for name, s_ in (("H0-H1_idle", "H1_idle"), ("H0-H1_load", "H1_load")):
                rows[tag][(name, la)] = paired_human_row(name, la, s_h0, sh[s_], failed[s_], g0,
                                                         [x.gaps if x else None for x in hs[s_]], k0,
                                                         [x.kept if x else None for x in hs[s_]])
            l0 = sh["L0"]
            for name, s_ in (("L0-L1_idle", "L1_idle"), ("L0-L1_load", "L1_load")):
                rows[tag][(name, la)] = paired_human_row(
                    name, la, l0, sh[s_], failed[s_] | np.array([cp.status != "captured" for cp in caps["L0"]]),
                    [x.gaps if x else None for x in hs["L0"]], [x.gaps if x else None for x in hs[s_]], read=False)
            if tag == "main":
                rep = {d: v[:N_REP] for d, v in sh["H1_idle"].items()}
                rows[tag][("H1_idle-H1_rep", la)] = paired_human_row(
                    "H1_idle-H1_rep", la, rep, sh["H1_rep"], failed["H1_rep"],
                    [x.gaps if x else None for x in hs["H1_idle"][:N_REP]],
                    [x.gaps if x else None for x in hs["H1_rep"]],
                    [x.kept if x else None for x in hs["H1_idle"][:N_REP]],
                    [x.kept if x else None for x in hs["H1_rep"]], read=False)
                for s_, v in hs.items():
                    counts.setdefault(s_, {})[la] = {d: count_cell(d, sh[s_][d], la, s_) for d in DETECTORS}
                    above_hist.setdefault(s_, {})[la] = int(sh[s_]["above_hist"].sum())
            for s_ in (*hs, *MACHINE, "P_idle", "P_load"):
                v = rec_list(s_, la, ties)
                rec["cleaning"].setdefault(tag, {}).setdefault(s_, {})[la] = {
                    "dropped_equal_times": int(sum(x.dropped for x in v if x)),
                    "inversions": int(sum(x.inversions for x in v if x)),
                    "repeats_left_out": int(sum(cp.layers[la].repeats for cp in caps[s_]
                                                if cp.status == "captured" and la in cp.layers))}

    # machine_path
    mp: dict = {}
    for ties in (True, False):
        tag = "main" if ties else "ties_kept"
        mp[tag] = {}
        for s_ in MACHINE:
            status = [cp.status for cp in caps[s_]]
            pg = [plan_gaps(cp) for cp in caps[s_]]
            ps = scores_of((s_, "plan"), pg)
            for la in LAYERS:
                rg = [x.gaps if x else None for x in rec_list(s_, la, ties)]
                rs = scores_of((s_, la, tag), rg)
                for d in DETECTORS:
                    mp[tag][(d, la, s_)] = machine_path_cell(s_, la, d, ps[d], rs[d], status, pg, rg)
                if tag == "main":
                    counts.setdefault(s_, {})[la] = {d: count_cell(d, rs[d], la, s_, status) for d in DETECTORS}
                    above_hist.setdefault(s_, {})[la] = int(rs["above_hist"].sum())
            if tag == "main":
                counts[s_]["plan"] = {d: count_cell(d, ps[d], None, s_, status, interval=False) for d in DETECTORS}
                above_hist[s_]["plan"] = int(ps["above_hist"].sum())

    # quantisation only: H0 against H0q (per layer: the jitter key holds the layer)
    h0q: dict = {}
    for la in LAYERS:
        for how, val in H0Q_VERSIONS:
            vname = f"{how}{'' if val is None else int(val)}"
            if how == "resolution":
                if grid_h is None:
                    h0q[(vname, la)] = "unknown: no resolution read on H0"
                    continue
                how_, val_ = "grid", grid_h
            else:
                how_, val_ = how, val
            gq = []
            for i, x in enumerate(h0):
                rng = (np.random.default_rng([S, v7.VERSION, 10, LAYER_IDX[la], JITTER_IDX["H0"], i,
                                              0 if val == 2.0 else 1]) if how == "jitter" else None)
                gq.append(v7.gaps_of(v7.sensitivity_variant(x["times"], how_, val_, rng)[0]))
            sq = scores_of(("H0q", vname, la), gq)
            pname = {"jitter2": "H0-H0q_jitter2", "jitter8": "H0-H0q_jitter8"}.get(vname)
            if pname:
                h0q[(vname, la)] = paired_human_row(pname, la, s_h0, sq, np.zeros(nh, dtype=bool), h0_gaps, gq)
            else:
                h0q[(vname, la)] = {d: dict(zip(("b", "c", "n"), v7.flips(*_judged(d, s_h0[d], sq[d]))))
                                    for d in DETECTORS}

    # headline and the general statements
    head: dict = {}
    for tag in ("main", "ties_kept"):
        readings_g = {}
        cells = {}
        for la in LAYERS:
            for d in DETECTORS:
                cell = rows[tag][("H0-H1_idle", la)][d]
                hr = cell.get("headline", {}).get("reading")
                cells[f"{d}/{la}"] = x = {"reading": hr, "route": cell.get("headline", {}).get("route"),
                                          "changed_from_0": cell.get("headline", {}).get("changed_from_0"),
                                          "b": cell["b"], "c": cell["c"], "n": cell["n"],
                                          "bounds": cell.get("bounds"), "enters_general": cell["enters_general"],
                                          "grid_note": rec["validity_gate"]["grid_note"][d],
                                          "split_by_missing_data": bool(cell.get("bounds") and len(set(
                                              cell["bounds"].values()) - {None}) > 1)}
                if hr == "responds":   # amendment B14: the jitter versions of H0q beside it
                    x["H0q_beside"] = {v: {k: h0q[(v, la)][d][k] for k in ("b", "c", "n")}
                                       for v in ("jitter2", "jitter8")}
                readings_g[(d, la)] = ("not changed" if cell["enters_general"]
                                       else ("not entered" if hr == "not changed" else (hr or "no pair")))
        head[tag] = {"cells": cells, "across": v7.across(readings_g, valid)}
        mreadings = {k: v["reading"] for k, v in mp[tag].items()}
        ncl = {k: v["n_clusters"] for k, v in mp[tag].items()}
        miss = {k: v["missing_pairs"] for k, v in mp[tag].items()}
        head[tag]["machine_general"] = v7.machine_general(
            {k: (v if v is not None else "no pair") for k, v in mreadings.items()}, valid, ncl, list(MACHINE),
            missing_pairs=miss)

    # replay error, repeat floor, controls, hook
    rerr: dict = {}
    for la in LAYERS:
        for s_ in ("H1_idle", "H1_load", "H1_rep", "L1_idle", "L1_load", "M1_replay_8B", "M1_replay_14B", "M2_8B",
                   "M2_14B", "M3a", "M3b", "P_idle", "P_load"):
            rerr[f"{s_}/{la}"] = replay_error_series(caps[s_], la)
    rerr["P_nohook/page"] = replay_error_series(caps["P_nohook"], "page")
    for model in ("8B", "14B"):
        w = [cp.inject for cp in caps[f"M1_{model}"] if cp.inject is not None]
        rec["plans"]["M1"][model]["wait_ms"] = summary(np.concatenate([x["wait_ms"] for x in w]), ends=True) if w else None
        q = [x["queue"] for x in w if x["queue"] is not None]
        rec["plans"]["M1"][model]["queue_max"] = int(max(x.max() for x in q)) if q else None
    repeat: dict = {}
    for la in LAYERS:
        h1 = caps["H1_idle"][:N_REP]
        planned = [cp.layers[la].times if cp.status == "captured" and la in cp.layers else None for cp in h1]
        pk = [cp.layers[la].kinds if cp.status == "captured" and la in cp.layers else None for cp in h1]
        cell = {"gap_deviation": replay_error_series(caps["H1_rep"], la, planned, pk), "detectors": {}}
        dg = []   # |gap(H1_rep) - gap(H1_idle)| over matched pairs: what Hd3 reads (the researcher)
        for cp, p_, k_ in zip(caps["H1_rep"], planned, pk):
            if p_ is None or cp.status != "captured" or la not in cp.layers:
                continue
            q_ = cp.layers[la]
            if q_.times.size == p_.size and q_.kinds == k_:
                dg.append(np.abs(np.diff(q_.times) - np.diff(p_)))
        cell["abs_gap_difference_p95_ms"] = (float(np.percentile(np.concatenate(dg), 95)) if dg else None)
        s1 = scores_of(("H1_idle", la, "main"), [x.gaps if x else None for x in rec_list("H1_idle", la)])
        s2 = scores_of(("H1_rep", la, "main"), [x.gaps if x else None for x in rec_list("H1_rep", la)])
        for d in DETECTORS:
            a0, a1, a2 = s_h0[d][:N_REP], s1[d][:N_REP], s2[d]
            x: dict = {"flips": rows["main"][("H1_idle-H1_rep", la)][d]}
            if (np.isfinite(a1) & np.isfinite(a2)).any():
                x["hd3"] = v7.floor_abs_change(a1, a2, [S, v7.VERSION, 9, LAYER_IDX[la], DET_IDX[d], 2])
            if (np.isfinite(a0) & np.isfinite(a1) & np.isfinite(a2)).any():
                x["floor"] = v7.above_floor(a0, a1, a2, [S, v7.VERSION, 12, LAYER_IDX[la], DET_IDX[d]])
            cell["detectors"][d] = x
        repeat[la] = cell
    controls: dict = {}
    for s_ in ("P_idle", "P_load", "P_nohook"):
        for cp in caps[s_]:
            for la in (("page",) if s_ == "P_nohook" else LAYERS):
                g = raw_gaps(cp, la)
                controls[f"{s_}/{cp.unit}/{la}"] = control_stats(g, cp.unit) if g is not None and g.size else None

    # C1 (stages_C1) and its sensitivity
    c1_rows = c1_all(caps, h0, m, r, sensitivity=None)
    sens: dict = {"versions": [f"{h}{'' if v is None else int(v)}" for h, v in H0Q_VERSIONS], "machine_path": {},
                  "c1": {}}
    for how, val in H0Q_VERSIONS:
        vname = f"{how}{'' if val is None else int(val)}"
        if how == "resolution" and grid_h is None:
            sens["machine_path"][vname] = sens["c1"][vname] = "unknown: no resolution read on H0"
            continue
        changed, total = [], 0
        for s_ in MACHINE:
            status = [cp.status for cp in caps[s_]]
            ps = scores_of((s_, "plan"), [plan_gaps(cp) for cp in caps[s_]])
            for la in LAYERS:
                gq = [(lambda x: x.gaps if x else None)(variant_recorded(cp, la, how, val, grid_h, s_))
                      for cp in caps[s_]]
                rs = sc.score(gq)
                for d in DETECTORS:
                    new = machine_path_cell(s_, la, d, ps[d], rs[d], status)["reading"]
                    total += 1
                    if new != mp["main"][(d, la, s_)]["reading"]:
                        changed.append([d, la, s_, mp["main"][(d, la, s_)]["reading"], new])
        sens["machine_path"][vname] = {"cells": total, "changed": changed}
        c1v = c1_all(caps, h0, m, r, sensitivity=(how, val, grid_h))
        sens["c1"][vname] = {"changed": [[k, _c1_state(c1_rows.get(k)), _c1_state(v)] for k, v in c1v.items()
                                         if "/" in k and _c1_state(v) != _c1_state(c1_rows.get(k))]}

    # G1
    g1 = {}
    for la in LAYERS:
        hs = rec_list("H1_idle", la)
        ix = [i for i, x in enumerate(hs) if x is not None]
        if not ix:
            g1[la] = None
            continue
        a = _bin_shares(np.concatenate([h0_gaps[i] for i in ix]))
        b = _bin_shares(np.concatenate([hs[i].gaps for i in ix]))
        diff = b - a
        g1[la] = {"persons": len(ix), "first_bin_diff": float(diff[0]),
                  "other_bins_max_abs_diff": float(np.abs(diff[1:]).max()),
                  "passes": bool(abs(diff[0]) > 0.01 and np.all(np.abs(diff[1:]) <= 0.01))}

    # failures (run_order.failures): planned sessions as the denominator
    fails = {}
    for s_, v in caps.items():
        fails[s_] = {"planned": len(v), "no_plan": sum(cp.status == "no_plan" for cp in v),
                     "failed_capture": sum(cp.status == "failed" for cp in v), "redone": sum(cp.redone for cp in v)}

    rec.update({
        "counts": _keys_to_str(counts), "d2_above_hist": above_hist,
        "paired_rows": {tag: _keys_to_str(v) for tag, v in rows.items()},
        "machine_path": {tag: _keys_to_str(v) for tag, v in mp.items()}, "h0q": _keys_to_str(h0q),
        "headline": head, "replay_error": rerr, "replay_repeat": repeat, "controls": controls,
        "c1": c1_rows, "sensitivity_quantisation": sens, "g1": g1, "failures": fails,
        "design_table": design_table()})
    rec["predictions"] = predictions(rec, valid)
    return rec


def _c1_state(x) -> object:
    """A C1 cell's reading, or its 'not computed' reason (a change between the two is a change too)."""
    return x["reading"] if isinstance(x, dict) else x


def _judged(det: str, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ja, oka = judgments(det, a)
    jb, okb = judgments(det, b)
    ok = oka & okb
    return ja[ok], jb[ok]


def variant_recorded(cp: Capture, la: str, how: str, val: float | None, grid_h: float | None,
                     series: str) -> Recorded | None:
    """sensitivity_quantisation on a machine series' recorded times (capture order), then v6's cleaning; the kinds
    follow their presses. Jitter key [20261031, 7, 10, layer, series, session, sd] (analysis_details.keys)."""
    if cp.status != "captured" or la not in cp.layers:
        return None
    la_ = cp.layers[la]
    if how == "jitter":
        rng = np.random.default_rng([S, v7.VERSION, 10, LAYER_IDX[la], JITTER_IDX[series], cp.unit,
                                     0 if val == 2.0 else 1])
        t, idx, dropped = v7.sensitivity_variant(la_.times, "jitter", val, rng)
    else:
        t, idx, dropped = v7.sensitivity_variant(la_.times, "grid", grid_h if how == "resolution" else val)
    return Recorded(v7.gaps_of(t), t, [la_.kinds[i] for i in idx], idx, dropped, 0)


def _bin_shares(g: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(g[g <= v7.HIST_MAX_MS], bins=v7.HIST_BINS, range=(0.0, v7.HIST_MAX_MS))
    return counts / g.size


def _keys_to_str(d: dict) -> dict:
    return {("/".join(k) if isinstance(k, tuple) else k): v for k, v in d.items()}


def c1_all(caps: dict, h0: list[dict], m: dict, r: dict, sensitivity: tuple | None) -> dict:
    """stages_C1 on the path row and the typer rows, per layer and stage. sensitivity (how, value, grid_h) changes
    the machine series' recorded times (the typer rows; the path row is human on both sides and is not changed)."""
    out: dict = {}
    h0_vec, fail = [], 0
    for x in h0:
        try:
            h0_vec.append(v7.stage_vectors(x["times"], list(x["kinds"]), m, r))
        except ValueError:
            h0_vec.append(None)
            fail += 1
    out["stage_failures_H0"] = fail
    for la in LAYERS:
        vecs: dict = {}
        for s_ in ("H1_idle", *[p.split("-")[1] for p in C1_PAIRS[1:]]):
            vv, f = [], 0
            for cp in caps[s_]:
                if sensitivity is not None and s_ != "H1_idle":
                    x = variant_recorded(cp, la, *sensitivity, s_)
                else:
                    x = recorded(cp, la)
                if x is None:
                    vv.append(None)
                    continue
                try:
                    vv.append(v7.stage_vectors(x.times, x.kinds, m, r))
                except ValueError:
                    vv.append(None)
                    f += 1
            vecs[s_] = vv
            out.setdefault("stage_failures", {}).setdefault(la, {})[s_] = f
        for pair in C1_PAIRS:
            a_name, b_name = pair.split("-")
            for st, stage in enumerate(STAGES):
                xa = h0_vec if a_name == "H0" else vecs[a_name]
                xb = vecs[b_name]
                rows_x, lab, tok = [], [], []
                for side, xs, label in ((a_name, xa, 0), (b_name, xb, 1)):
                    for u, v in enumerate(xs):
                        if v is None:
                            continue
                        rows_x.append(v[stage])
                        lab.append(label)
                        human = side in ("H0", "H1_idle")
                        tok.append({("person", u)} if human else {("text", u)})
                key = [S, v7.VERSION, 4, LAYER_IDX[la], C1_IDX[pair], st]
                try:
                    out[f"{pair}/{la}/{stage}"] = v7.c1(np.array(rows_x), np.array(lab), tok, key)
                except ValueError as e:
                    if "two folds" in str(e):   # C1_bootstrap: a fold assignment splitting related series stops
                        raise RunStop(f"C1 {pair} {la} {stage}: {e}")
                    out[f"{pair}/{la}/{stage}"] = f"not computed: {e}"
    return out


def design_table() -> dict:
    """detectors_C2.headline (amendments B8, B11, B12, B14): computed from the design, no data."""
    n = N_H
    two = {str(rt): v7.reading_probabilities(rt, n) for rt in RATES}
    one = {str(rt): v7.reading_probabilities(rt, n, one_way=True) for rt in RATES}
    entry = {str(rt): {str(mm): v7.entry_probability(rt, n, mm) for mm in (0, 1, 2)} for rt in RATES}
    general = {str(rt): v7.general_probability_bounds([two[str(rt)]["not changed"]] * 10) for rt in RATES}
    return {"n": n, "readings_half_each_way": two, "readings_one_way": one, "entry_probability": entry,
            "human_general_bounds_10_cells": general, "zero_flip_table": v7.zero_flip_table(),
            "zero_flip_min_clusters": v7.zero_flip_min_clusters()}


# ---------------------------------------------------------------- predictions (each passes or fails on its own)


def predictions(rec: dict, valid: dict[str, bool]) -> dict:
    """The conditions' prediction, item by item: True (passes), False (fails) or None (not evaluable, with the reason
    beside). Not being able to read 'not changed' is not support for 'harmless' (the conditions)."""
    out: dict = {}
    ctl = rec["controls"]

    def ctl_ok(series: str, ms: int, la: str):
        return ctl.get(f"{series}/{ms}/{la}")

    def all_or_none(vals):
        """False if any item fails; None (not evaluable) if none fails and any is missing; else True."""
        vals = list(vals)
        if any(v is False for v in vals):
            return False
        if any(v is None for v in vals):
            return None
        return True

    out["P1"] = all_or_none([None if ctl_ok("P_idle", int(c), "hook") is None
                             else abs(ctl_ok("P_idle", int(c), "hook")["median_minus_planned_ms"]) <= 1.0 + v7.EPS_EDGE
                             for c in CONTROLS_MS])
    out["P2"] = all_or_none([None if ctl_ok("P_idle", int(c), "page") is None
                             else abs(ctl_ok("P_idle", int(c), "page")["median_minus_planned_ms"]) <= 2.0 + v7.EPS_EDGE
                             for c in CONTROLS_MS if c != 5.0])
    above = {(s_, int(c)): (None if ctl_ok(s_, int(c), "page") is None else ctl_ok(s_, int(c), "page")["share_below_1ms"] > 0.05)
             for s_ in ("P_idle", "P_load") for c in CONTROLS_MS}
    out["P3"] = p3(above)
    p4 = []
    for c in CONTROLS_MS:
        for la in LAYERS:
            a, b = ctl_ok("P_idle", int(c), la), ctl_ok("P_load", int(c), la)
            p4.append(None if a is None or b is None else abs(a["median_ms"] - b["median_ms"]) <= 2.0 + v7.EPS_EDGE)
    out["P4"] = all_or_none(p4)

    r1 = []
    for la, lim in (("hook", 5.0), ("page", 10.0)):
        e = rec["replay_error"][f"H1_idle/{la}"]
        if not e.get("captures") or not e.get("all_matched"):
            r1.append(None)
        else:
            r1.append(e["jitter_p95_ms"]["max"] < lim and e["abs_drift_ms"]["max"] < 50.0)
    out["R1"] = all_or_none(r1)
    out["V1"] = all(valid.values())

    cells = rec["headline"]["main"]["cells"]
    def nc(x):
        return None if x is None else x == "not changed"
    out["Hd1"] = all_or_none(nc(cells[f"D1/{la}"]["reading"]) for la in LAYERS)
    out["Hd2"] = all_or_none(nc(cells[f"{d}/{la}"]["reading"]) for d in PROB_DETECTORS for la in LAYERS)

    hd3 = []
    for la, lim in (("hook", 2.0), ("page", 5.0)):
        rp = rec["replay_repeat"][la]
        g = rp["gap_deviation"]
        # the researcher (2026-11-02): Hd3's 'absolute gap deviation' is |gap(H1_rep) - gap(H1_idle)| over matched
        # pairs (a slow drift inflates the time deviation, not the gaps); the jitter of replay_error is reported beside
        hd3.append(None if not g.get("captures") or not g.get("all_matched") or rp["abs_gap_difference_p95_ms"] is None
                   else rp["abs_gap_difference_p95_ms"] < lim)
        for d in DETECTORS:
            x = rp["detectors"][d].get("hd3")
            hd3.append(None if x is None else x["passes"])   # None also when fewer than 8 complete pairs (B12)
    out["Hd3"] = all_or_none(hd3)

    mp = rec["machine_path"]["main"]
    hm1 = []
    for la in LAYERS:
        x = mp[f"D1/{la}/M0"]
        if x.get("n") != N_TEXTS:   # fewer than 40 complete pairs: '40 of 40' cannot be evaluated
            hm1.append(None if x.get("plan_machine") == x.get("n") == x.get("recorded_machine") else False)
        else:
            hm1.append(x["plan_machine"] == N_TEXTS and x["recorded_machine"] == N_TEXTS)
    out["Hm1"] = all_or_none(hm1)
    out["Hm2"] = all_or_none([all(valid.values())] + [nc(mp[f"{d}/{la}/{s_}"].get("reading"))
                                                       for d in DETECTORS for la in LAYERS for s_ in ("M3a", "M3b")])

    cnt = rec["counts"]
    dm1 = []
    for la in LAYERS:
        def k_of(s_, d="D1"):
            x = cnt[s_][la][d]
            return x["machine"], x["n"]
        # counts out of 40: decided as soon as the missing sessions cannot change the outcome, else None
        k, n = k_of("M0")
        dm1.append(False if k < n else (True if n == N_TEXTS else None))
        for s_ in ("M3a", "M3b"):
            k, n = k_of(s_)
            dm1.append(True if n - k >= 38 else (False if n - k + (N_TEXTS - n) < 38 else None))
        for s_ in ("M1_8B", "M1_14B"):
            k, n = k_of(s_)
            dm1.append(True if k >= 30 else (False if k + (N_TEXTS - n) < 30 else None))
    out["Dm1"] = all_or_none(dm1)
    out["Dm2"] = all_or_none([None if cnt["M3a"][la]["D3"]["n"] == 0 or cnt["M3b"][la]["D3"]["n"] == 0 else
                              (cnt["M3b"][la]["D3"]["n"] - cnt["M3b"][la]["D3"]["machine"])
                              > (cnt["M3a"][la]["D3"]["n"] - cnt["M3a"][la]["D3"]["machine"]) for la in LAYERS])

    c1 = rec["c1"]

    def size(pair, la, stage):
        x = c1.get(f"{pair}/{la}/{stage}")
        return x["reading"]["size"] if isinstance(x, dict) else None

    out["C1a"] = all_or_none([None if size("H0-H1_idle", la, "S0") is None
                              else size("H0-H1_idle", la, "S0") in ("below 0.60", "undecided") for la in LAYERS])
    out["C1b"] = all_or_none([None if size(p, la, "S0") is None else size(p, la, "S0") == "above 0.60"
                              for p in ("H1_idle-M0", "H1_idle-M1_8B", "H1_idle-M1_14B") for la in LAYERS])
    out["C1c"] = all_or_none([None if size(p, la, st) is None else size(p, la, st) != "above 0.60"
                              for p in ("H1_idle-M3a", "H1_idle-M3b") for la in LAYERS for st in ("S1", "S2")])
    out["G1"] = all_or_none([None if rec["g1"][la] is None else rec["g1"][la]["passes"] for la in LAYERS])

    p5 = []
    for c in v7.P_NOHOOK_MS:
        a, b = ctl_ok("P_idle", int(c), "page"), ctl_ok("P_nohook", int(c), "page")
        if a is None or b is None:
            p5.append(None)
            continue
        p5.append(v7.p5_passes(a["median_ms"], b["median_ms"]) and abs(a["mad_ms"] - b["mad_ms"]) <= 1.0 + v7.EPS_EDGE
                  and abs(a["p95_abs_dev_ms"] - b["p95_abs_dev_ms"]) <= 1.0 + v7.EPS_EDGE
                  and abs(a["share_below_1ms"] - b["share_below_1ms"]) <= 0.01 + v7.EPS_EDGE)
    out["P5"] = all_or_none(p5)
    return out


def p3(above: dict[tuple[str, int], bool | None]) -> bool | None:
    """P3: the 5 ms control is the only control whose share of recorded page gaps below 1 ms is above 0.05. above maps
    (P_idle or P_load, control ms) to that comparison; a control counts if it is above with or without load (a
    reading of 'the only control', decisions.md of .steering/20261102-v7-analysis-run); None if any is missing."""
    if any(v is None for v in above.values()):
        return None
    return {c for (_, c), v in above.items() if v} == {5}


# ---------------------------------------------------------------- main


def to_json(x):
    if isinstance(x, dict):
        return {str(k): to_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_json(v) for v in x]
    if isinstance(x, np.ndarray):
        return [to_json(v) for v in x.tolist()]
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (float, np.floating)):
        if not math.isfinite(float(x)):   # a NaN here is a fault of the computation, not a missing judgment
            raise RunStop("a non-finite number reached the record")
        return float(x)
    return x


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 4 or argv[0] != "--captures" or argv[2] != "--plan":
        raise SystemExit("usage: python works/prototype_v7_run.py --captures <folder of the production captures> "
                         "--plan <folder of the plan (works/prototype_v7_plan.py make)>")
    captures, plan_dir = Path(argv[1]), Path(argv[3])
    OUT_DIR.parent.mkdir(exist_ok=True)
    if OUT_DIR.exists():
        raise SystemExit(f"{OUT_DIR} exists; refusing to overwrite")
    raw, c = cm.load_conditions(COND)
    if "environment" not in c:
        raise SystemExit("the work PC's items (models, digests, injection, browser and driver) are not in the "
                         "conditions yet (fixed_before_running): no value is taken before that commit")
    raw6, c6 = cm.load_conditions(COND_V6)
    m4 = c6["mapping"]["v4_conditions"]
    raw4 = cm.check_file(ROOT / m4["file"], m4["sha256"], "v4's conditions")
    c4 = json.loads(raw4)
    paths = [COND.relative_to(ROOT).as_posix(), COND_V6.relative_to(ROOT).as_posix(), m4["file"], *CODE,
             DEVIATIONS.relative_to(ROOT).as_posix()]   # the deviations as committed (amendment B17)
    cm.same_as_head([ROOT / p for p in paths])
    state = cm.git_state(paths)
    kp = c["data"]["kupa"]
    files = {"essay": ("external/kupa-keys/KUPA-KEYS-TASK-2.csv", kp_sha(kp, "essay")),
             "meta": ("external/kupa-keys/KUPA-KEYS-META.csv", kp_sha(kp, "meta")),
             "copy": (c6["data"]["kupa"]["files"]["copy"]["file"], c6["data"]["kupa"]["files"]["copy"]["sha256"])}
    for name, (f, sha) in files.items():
        if cm.sha256(ROOT / f) != sha:
            raise SystemExit(f"{f} does not match the SHA-256 of the conditions ({name})")
    essay, seen = v6.read_kupa(ROOT / files["essay"][0])
    copy_, _ = v6.read_kupa(ROOT / files["copy"][0])
    v6_sha = c["persons"]["pool"].split("ids_sha256 ")[1].split(",")[0]   # persons.pool: v6's 200, must match
    ps = draw_persons(essay, copy_, seen, v6_sha)
    h0 = [{"times": essay[p]["times"], "kinds": essay[p]["kinds"]} for p in ps.h]
    e_gaps = [v7.gaps_of(essay[p]["times"]) for p in ps.e]
    caps = load_all(captures)
    plan_rec = read_plan(plan_dir, captures, ps, conditions_sha256(raw))
    dev_raw = DEVIATIONS.read_bytes()
    plan_rec["deviations"] = {"file": DEVIATIONS.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(dev_raw).hexdigest(),
                              "git": cm.git_state([DEVIATIONS.relative_to(ROOT).as_posix()])}
    tmp = Path(tempfile.mkdtemp(prefix="prototype_v7-", dir=OUT_DIR.parent))
    try:
        rec = run(c, h0, e_gaps, ps.e_train, caps, c4["mapping"], c4["relief"])
    except BaseException:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    rec["persons"] = {"pool_sha256": sha_ids(ps.pool), "ids_looked_at": ps.looked, "replay_H_sha256": sha_ids(ps.h),
                      "E": len(ps.e), "E_train": int(ps.e_train.size), "donors": int(ps.donors.size),
                      "v6_200_match": True}
    rec["plan"] = plan_rec
    rec["provenance"] = {"run_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), **state,
                         "conditions_sha256": hashlib.sha256(raw).hexdigest(),
                         "v6_conditions_sha256": hashlib.sha256(raw6).hexdigest(),
                         "code_sha256": {p: cm.sha256(ROOT / p) for p in CODE},
                         "data_sha256": {n: s_ for n, (_, s_) in files.items()},
                         "versions": cm.versions("numpy", "scipy", "scikit-learn"),
                         "bit_generator": type(np.random.default_rng(0).bit_generator).__name__}
    (tmp / "record.json").write_text(json.dumps(to_json(rec), indent=1, ensure_ascii=False, allow_nan=False) + "\n")
    if OUT_DIR.exists():
        raise SystemExit(f"{OUT_DIR} appeared during the run; the result is left in {tmp}")
    tmp.rename(OUT_DIR)
    print(json.dumps(to_json(rec["predictions"])))


DEVIATIONS = ROOT / "works" / "conditions" / "prototype_v7_deviations.md"   # record (amendment B17)


def conditions_sha256(raw: bytes) -> str:
    """The conditions' SHA-256 as the plan's manifest and the analysis both name it (record, amendment B17)."""
    return hashlib.sha256(raw).hexdigest()


def read_plan(plan_dir: Path, captures: Path, ps: Persons, conditions_sha: str) -> dict:
    """The plan the captures ran (amendment B16), checked before any value: every file of the plan is the one of its
    manifest.json; the manifest was made from these conditions and these replay persons; every runlog header names
    its runs.json; and each runlog holds exactly the items of its block (all blocks for one runlog.ndjson), in the
    order capture.runlist.order draws with the run key. The number of replaced keys of H1 (series.H1_idle, B16) is
    recorded from the manifest."""
    sys.path.insert(0, str(ROOT))
    from capture import runlist as rl
    man = json.loads((plan_dir / "manifest.json").read_text(encoding="utf-8"))
    files = sorted(p.relative_to(plan_dir).as_posix() for p in plan_dir.rglob("*") if p.is_file())
    if "runs.json" not in man["files_sha256"]:
        raise RunStop("the plan's manifest names no runs.json")
    extra = sorted(set(files) - set(man["files_sha256"]) - {"manifest.json"})
    missing = sorted(set(man["files_sha256"]) - set(files))
    if extra or missing:
        raise RunStop(f"the plan's files are not the ones of its manifest: extra {extra[:5]}, missing {missing[:5]}")
    for rel, sha in man["files_sha256"].items():
        if hashlib.sha256((plan_dir / rel).read_bytes()).hexdigest() != sha:
            raise RunStop(f"the plan's {rel} is not the one of its manifest")
    if man.get("conditions_sha256") != conditions_sha:
        raise RunStop("the plan was made from other conditions than these")
    if man["persons"]["replay_H_sha256"] != sha_ids(ps.h) or man["persons"]["pool_sha256"] != sha_ids(ps.pool):
        raise RunStop("the plan's replay persons or pool are not this run's")
    runs_sha = man["files_sha256"]["runs.json"]
    if {h["runs_sha256"] for h in runlog_headers(captures)} != {runs_sha}:
        raise RunStop("the captures were run from another runs.json than this plan's")
    items = json.loads((plan_dir / "runs.json").read_text(encoding="utf-8"))
    if man["n_items"] != len(items) or man["items_per_block"] != {
            str(b): sum(it["block"] == b for it in items) for b in range(N_RUN_BLOCKS)}:
        raise RunStop("the plan's manifest does not count runs.json's items")
    seq, _ = rl.order(items, RUN_KEY)
    logs = runlog_files(captures)
    for f in logs:
        rows = read_ndjson(f)
        block = rows[0].get("block")
        want = [it["name"] for it in seq if block is None or it["block"] == block]
        got = [r.get("item") for r in rows[1:-1]]
        if rows[0].get("order") != want or got != want:
            raise RunStop(f"{f.name}: its items or their order are not the plan's {'blocks' if block is None else 'block ' + str(block)}")
    return {"runs_sha256": runs_sha, "conditions_sha256": conditions_sha, "replaced_keys_H": man["replaced_keys_H"],
            "n_items": man["n_items"], "items_per_block": man["items_per_block"],
            "plan_git_commit": man.get("git_commit"), "plan_code_sha256": man.get("plan_code_sha256")}


def kp_sha(kp: dict, which: str) -> str:
    """data.kupa states the SHA-256 of the essay (KUPA-KEYS-TASK-2.csv) and meta (KUPA-KEYS-META.csv) files."""
    text = kp if isinstance(kp, str) else json.dumps(kp)
    name = {"essay": "KUPA-KEYS-TASK-2.csv ", "meta": "KUPA-KEYS-META.csv "}[which]
    return text.split(name)[1].split()[0].strip(",;")


if __name__ == "__main__":
    main()
