"""The schedule of presses (capture spec section 1): NDJSON, a header line and then one press per line.

    {"kind": "header", "series": "<name>", "source": "<how it was made>", "created_utc": "...", "n": 123}
    {"i": 0, "t_ms": 0.0, "key": "a"}

t_ms is the planned time from the first press (ms, finite, non-decreasing; equal times allowed: amendment B14 keeps
the zero gaps of a plan). key is one printable character or one of Enter, Backspace, Delete.

The kinds are shared by events, the hook and the page: alnum (one ASCII letter or digit), space, end (. ! ?), enter,
backspace, delete, other (any other single printable character). The loggers add two kinds that are never counted
presses (data.counted_press): 'modifier' (Shift, Ctrl, Alt, Win alone) and 'nonchar' (any other key that types no
character: arrows, Esc, F-keys, a key pressed with Ctrl or Alt held, an IME 'Process' key, 'Unidentified').
[仮] 'modifier' and 'nonchar' are not in the spec's list of seven; they are added so that the shift presses of the
virtual-key method and the stop chord can be told apart from the counted presses (returns/questions.md).
"""

from __future__ import annotations

import json
import math
import string
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

NAMED_KEYS = ("Enter", "Backspace", "Delete")
COUNTED_KINDS = ("alnum", "space", "end", "enter", "backspace", "delete", "other")
KINDS = COUNTED_KINDS + ("modifier", "nonchar")
ALNUM = frozenset(string.ascii_letters + string.digits)
END = frozenset(".!?")
# from_live: the key that stands for each kind (the researcher's decision of 2026-11-02; L0's characters are not kept)
LIVE_KEY = {"alnum": "e", "space": " ", "end": ".", "enter": "Enter", "backspace": "Backspace", "delete": "Delete",
            "other": ","}
CYCLE = "abcdefghij"


def kind_of_char(c: str) -> str:
    """Kind of one typed character."""
    if len(c) != 1:
        raise ValueError(f"not one character: {c!r}")
    if c in ALNUM:
        return "alnum"
    if c == " ":
        return "space"
    if c in END:
        return "end"
    if c in "\r\n":
        return "enter"
    return "other"


def kind_of_key(key: str) -> str:
    """Kind of a schedule key (one character or a named key)."""
    if key == "Enter":
        return "enter"
    if key == "Backspace":
        return "backspace"
    if key == "Delete":
        return "delete"
    return kind_of_char(key)


def typable(c: str) -> bool:
    """A character the schedule can hold: printable ASCII (0x20-0x7E). [decided by the researcher, 2026-11-04] Other characters (curly quotes, dashes,
    letters outside ASCII) are skipped and counted by the makers that read text (llm_stream, llm_plan), so that the
    same characters are typed under either injection method (returns/questions.md)."""
    return len(c) == 1 and 0x20 <= ord(c) <= 0x7E


def check_key(key: str) -> None:
    if key in NAMED_KEYS:
        return
    if not (isinstance(key, str) and typable(key)):
        raise ValueError(f"key must be one printable ASCII character or one of {NAMED_KEYS}: {key!r}")


def validate(presses: list[dict]) -> None:
    prev = -math.inf
    for j, p in enumerate(presses):
        if p.get("i") != j:
            raise ValueError(f"press {j}: i is {p.get('i')!r}, not {j}")
        t = p.get("t_ms")
        if not isinstance(t, (int, float)) or isinstance(t, bool) or not math.isfinite(t):
            raise ValueError(f"press {j}: t_ms is not a finite number: {t!r}")
        if t < prev:
            raise ValueError(f"press {j}: t_ms {t} is earlier than the previous {prev}")
        if j == 0 and t != 0:
            raise ValueError(f"press 0 must be at t_ms 0, not {t}")
        check_key(p.get("key"))
        prev = t


def make(times_ms, keys: list[str]) -> list[dict]:
    times = [float(t) for t in times_ms]
    if len(times) != len(keys):
        raise ValueError(f"{len(times)} times for {len(keys)} keys")
    if times:
        t0 = times[0]
        times = [t - t0 for t in times]
    out = [{"i": i, "t_ms": t, "key": k} for i, (t, k) in enumerate(zip(times, keys))]
    validate(out)
    return out


def write(path: Path, presses: list[dict], series: str, source: str, **extra) -> None:
    validate(presses)
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists; schedules are not overwritten")
    header = {"kind": "header", "series": series, "source": source,
              "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "n": len(presses), **extra}
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(header, ensure_ascii=True) + "\n")
        for p in presses:
            f.write(json.dumps(p, ensure_ascii=True) + "\n")


def read(path: Path) -> tuple[dict, list[dict]]:
    lines = [ln for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"{path} is empty")
    header = json.loads(lines[0])
    if header.get("kind") != "header":
        raise ValueError(f"{path}: the first line is not a header")
    presses = [json.loads(ln) for ln in lines[1:]]
    if header.get("n") != len(presses):
        raise ValueError(f"{path}: header says {header.get('n')} presses, the file holds {len(presses)}")
    validate(presses)
    return header, presses


# ---------------------------------------------------------------- makers


def text_keys(text: str) -> list[str]:
    """One key per character; a line break is Enter (\\r\\n counts once)."""
    keys = []
    for c in text.replace("\r\n", "\n"):
        keys.append("Enter" if c in "\r\n" else c)
    return keys


def constant(text: str, interval_ms: float) -> list[dict]:
    """M0: every gap the same."""
    if not (math.isfinite(interval_ms) and interval_ms >= 0):
        raise ValueError(f"interval must be finite and non-negative: {interval_ms}")
    keys = text_keys(text)
    return make([i * float(interval_ms) for i in range(len(keys))], keys)


def from_gaps(text_or_keys, gaps_ms) -> list[dict]:
    """A schedule from the gap before each press (the first gap must be 0): synthetic series, M2, M3."""
    keys = text_keys(text_or_keys) if isinstance(text_or_keys, str) else list(text_or_keys)
    g = np.asarray(gaps_ms, dtype=float)
    if g.ndim != 1 or g.size != len(keys):
        raise ValueError(f"{g.size} gaps for {len(keys)} keys")
    if g.size and (not np.all(np.isfinite(g)) or np.any(g < 0) or g[0] != 0):
        raise ValueError("gaps must be finite and non-negative, the first 0")
    return make(np.cumsum(g), keys)


def cycle_letters(n: int, interval_ms: float) -> list[dict]:
    """Positive control P: a-j in turn, n presses, one gap."""
    keys = [CYCLE[i % len(CYCLE)] for i in range(n)]
    return make([i * float(interval_ms) for i in range(n)], keys)


def lognormal_series(text: str, median_ms: float, sigma: float, seed: int, rest_share: float = 0.05,
                     rest_ms: float = 2000.0, edits: bool = True) -> list[dict]:
    """Smoke series 2 (spec section 7): gaps lognormal(median, sigma); a share of the gaps replaced by rests of at least
    rest_ms (rest_ms x (1 + Exp(1)), [仮] the length of a rest is not fixed by the spec); one Backspace and one Delete
    inserted (the Backspace after the first word, the Delete after the second; [仮] places not fixed by the spec)."""
    keys = text_keys(text)
    if edits:
        spaces = [i for i, k in enumerate(keys) if k == " "]
        if len(spaces) < 2:
            raise ValueError("the text needs two spaces to place Backspace and Delete")
        keys = keys[: spaces[1] + 1] + ["Delete"] + keys[spaces[1] + 1:]
        keys = keys[: spaces[0] + 1] + ["Backspace"] + keys[spaces[0] + 1:]
    rng = np.random.default_rng(seed)
    g = median_ms * np.exp(sigma * rng.standard_normal(len(keys)))
    rest = rng.random(len(keys)) < rest_share
    g[rest] = rest_ms * (1.0 + rng.exponential(1.0, int(rest.sum())))
    g[0] = 0.0
    return from_gaps(keys, g)


def from_live(hook_records: list[dict]) -> tuple[list[dict], dict]:
    """L1 from L0's hook records: the non-injected, non-repeated key downs of a counted kind, each typed as the key that
    stands for its kind (LIVE_KEY). Returns the schedule and counts of what was left out."""
    times, keys = [], []
    left = {"injected": 0, "repeat": 0, "up": 0, "modifier": 0, "nonchar": 0}
    for r in hook_records:
        if r.get("type") != "key":
            continue
        if not r["down"]:
            left["up"] += 1
        elif r["injected"]:
            left["injected"] += 1
        elif r.get("repeat"):
            left["repeat"] += 1
        elif r["kind"] in ("modifier", "nonchar"):
            left[r["kind"]] += 1
        else:
            times.append(r["t_ns"] / 1e6)
            keys.append(LIVE_KEY[r["kind"]])
    order = np.argsort(np.asarray(times), kind="stable")  # the hook writes in arrival order; the plan needs time order
    left["reordered"] = int(np.sum(order != np.arange(len(order))))
    times = [times[i] for i in order]
    keys = [keys[i] for i in order]
    return make(times, keys), left


def gaps(presses: list[dict]) -> np.ndarray:
    return np.diff(np.array([p["t_ms"] for p in presses], dtype=float))


def kinds(presses: list[dict]) -> list[str]:
    return [kind_of_key(p["key"]) for p in presses]
