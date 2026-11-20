"""Prototype v7s: the capture plan (runs.json and the planned series), made on the work PC from v7's plan and captures.

    python works/prototype_v7s_plan.py make --v7-plan <v7's plan folder> --v7-captures <v7's capture folder> --out <new>

Conditions: works/conditions/prototype_v7s.json (captures). Writes into a new folder:
  runs.json                    the items of capture/runlist.py (names below), one block per group (captures.order)
  events/K/cNNN.ndjson         known constant gaps (a-j cycled, 1000 presses; NNN the gap in ms)
  events/MC/gG/tNN.ndjson      M0's keys of text NN at a constant G ms (MC1, MC2, MC3, MC5)
  events/F_M0/tNN.ndjson       v7's M0 plan of text NN, byte for byte
  events/F_M2_8B|14B/tNN.ndjson  v7's validated M2 plan of text NN, byte for byte
  events/F_H1/hNN.ndjson       v7's H1 schedule of replay person NN (NN < 10), byte for byte
  manifest.json                counts, every file's SHA-256 and the SHA-256 of every v7 file read (never a key or time)
Names: K_chrome/cNNN, K_chrome_nohook/cNNN, MC1|MC2|MC3|MC5/tNN, F_K/cNNN, F_K_nohook/cNNN, F_M0/tNN,
F_M2_8B/tNN, F_M2_14B/tNN, F_H1/hNN. A v7 text without a plan (no_plan, or no plan file) gets no item and is counted.
Every file read from v7 is checked: H's schedules against v7's plan manifest, M0's constant 55 ms, M0's and M2's
files present for every item v7's runlogs did not mark no_plan.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "works"))
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402

import prototype_v7_run as vr  # noqa: E402
from capture import events as ev  # noqa: E402

KEY = 20261115
K_CHROME_MS = (0, 1, 2, 3, 5)
K_FIREFOX_MS = (0, 1, 2, 3, 5, 18, 50, 100, 200)
K_NOHOOK_MS = (0, 1, 2, 3, 5)
MC_MS = (1, 2, 3, 5)
M0_MS = 55.0
N_TEXTS = vr.N_TEXTS
N_F_H1 = 10
P_PRESSES = vr.P_PRESSES
GROUPS = ("K_chrome", "MC_chrome", "F_K", "F_M0", "F_M2", "F_H1")
RUNLIST_KEY = (KEY, 7, 1)
SESSION = ["--auto-focus"]


class PlanStop(SystemExit):
    pass


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def group_blocks() -> dict[str, int]:
    """captures.order: block of each group = 1 + its place in default_rng([20261115, 7, 0]).permutation(6)."""
    perm = np.random.default_rng([KEY, 7, 0]).permutation(len(GROUPS))
    return {GROUPS[int(g)]: b + 1 for b, g in enumerate(perm)}


def v7_plan_file(v7cap: Path, runlog: dict[str, str], series: str, t: int) -> Path | None:
    """The events.ndjson v7 typed for series/text t, or None when v7 had no plan (no_plan, or a failed capture that
    kept no plan file). A captured item without its file is a fault."""
    name = vr.item_name(series, t)
    if name not in runlog:
        raise PlanStop(f"{name}: not in v7's runlogs")
    f = v7cap / name / "events.ndjson"
    if runlog[name] == "no_plan":
        return None
    if f.exists():
        return f
    if runlog[name] == "failed":
        return None
    raise PlanStop(f"{name}: v7's runlog says {runlog[name]} but {f} is missing")


def check_m0(plan: vr.Plan, name: str) -> None:
    g = np.diff(plan.times)
    if plan.times.size < 2 or not np.allclose(g, M0_MS, atol=1e-6):
        raise PlanStop(f"{name}: not a constant {M0_MS} ms plan")


def constant(keys: list[str], gap_ms: float) -> list[dict]:
    return ev.make([i * float(gap_ms) for i in range(len(keys))], keys)


def build(v7plan: Path, v7cap: Path, out: Path) -> tuple[dict, bytes]:
    out.mkdir(parents=True, exist_ok=False)
    out = out.resolve()
    runlog = vr.read_runlogs(v7cap)
    v7man = json.loads((v7plan / "manifest.json").read_text(encoding="utf-8"))
    blocks = group_blocks()
    items: list[dict] = []
    read: dict[str, str] = {}
    lost: dict[str, int] = {}
    copies: dict[Path, str] = {}   # each copy and the SHA-256 of its own source

    def add(name: str, events: Path, group: str, browser: str, nohook: bool = False) -> None:
        args = ["--events", str(events), "--browser", browser, *(["--no-hook"] if nohook else []), *SESSION]
        items.append({"name": name, "args": args, "block": blocks[group]})

    def new_file(rel: str) -> Path:
        p = out / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    # known gaps
    for ms in sorted(set(K_CHROME_MS) | set(K_FIREFOX_MS)):
        f = new_file(f"events/K/c{ms:03d}.ndjson")
        ev.write(f, ev.cycle_letters(P_PRESSES, float(ms)), "K", f"constant {ms} ms")
        if ms in K_CHROME_MS:
            add(f"K_chrome/c{ms:03d}", f, "K_chrome", "chrome")
        if ms in K_NOHOOK_MS:
            add(f"K_chrome_nohook/c{ms:03d}", f, "K_chrome", "chrome", nohook=True)
        if ms in K_FIREFOX_MS:
            add(f"F_K/c{ms:03d}", f, "F_K", "firefox")
        if ms in K_NOHOOK_MS:
            add(f"F_K_nohook/c{ms:03d}", f, "F_K", "firefox", nohook=True)

    # M0's texts: MC and F_M0
    for t in range(N_TEXTS):
        src = v7_plan_file(v7cap, runlog, "M0", t)
        if src is None:
            lost["M0"] = lost.get("M0", 0) + 1
            continue
        name = vr.item_name("M0", t)
        plan = vr.read_events(src)
        check_m0(plan, name)
        read[name + "/events.ndjson"] = sha256(src)
        for g in MC_MS:
            f = new_file(f"events/MC/g{g}/t{t:02d}.ndjson")
            ev.write(f, constant(plan.keys, g), f"MC{g}", f"M0's keys of text {t} at a constant {g} ms")
            add(f"MC{g}/t{t:02d}", f, "MC_chrome", "chrome")
        f = new_file(f"events/F_M0/t{t:02d}.ndjson")
        shutil.copyfile(src, f)
        copies[f] = read[name + "/events.ndjson"]
        add(f"F_M0/t{t:02d}", f, "F_M0", "firefox")

    # M2's validated plans
    for model in ("8B", "14B"):
        s7 = f"M2_{model}"
        for t in range(N_TEXTS):
            src = v7_plan_file(v7cap, runlog, s7, t)
            if src is None:
                lost[s7] = lost.get(s7, 0) + 1
                continue
            vr.read_events(src)   # v7's own check of a plan file (header, order, finite, non-decreasing)
            read[vr.item_name(s7, t) + "/events.ndjson"] = sha256(src)
            f = new_file(f"events/F_{s7}/t{t:02d}.ndjson")
            shutil.copyfile(src, f)
            copies[f] = read[vr.item_name(s7, t) + "/events.ndjson"]
            add(f"F_{s7}/t{t:02d}", f, "F_M2", "firefox")

    # H1's schedules of the first 10 replay persons
    want = v7man["files_sha256"]
    for i in range(N_F_H1):
        rel = f"events/H/h{i:02d}.ndjson"
        src = v7plan / rel
        if rel not in want or sha256(src) != want[rel]:
            raise PlanStop(f"{rel}: does not match v7's plan manifest")
        read["v7-plan/" + rel] = want[rel]
        f = new_file(f"events/F_H1/h{i:02d}.ndjson")
        shutil.copyfile(src, f)
        copies[f] = want[rel]
        add(f"F_H1/h{i:02d}", f, "F_H1", "firefox")

    for f, want_sha in copies.items():   # each copy is its own source (the same text or person), byte for byte
        if sha256(f) != want_sha:
            raise PlanStop(f"{f.relative_to(out).as_posix()}: the copy differs from its source")
    if sorted(copies) != sorted(p for p in out.rglob("*") if p.is_file() and
                                p.relative_to(out).as_posix().startswith("events/F_")):
        raise PlanStop("a file under events/F_ is not a copy made here")
    runs = (json.dumps(items, indent=1) + "\n").encode("utf-8")
    files = sorted(p for p in out.rglob("*") if p.is_file())
    series = list(dict.fromkeys(it["name"].split("/")[0] for it in items))
    manifest = {"n_items": len(items),
                "items_per_series": {s_: sum(it["name"].split("/")[0] == s_ for it in items) for s_ in series},
                "items_per_block": {str(b): sum(it["block"] == b for it in items) for b in range(1, len(GROUPS) + 1)},
                "group_blocks": blocks, "v7_without_plan": lost,
                "v7_files_read_sha256": read,
                "v7_plan_manifest_sha256": sha256(v7plan / "manifest.json"),
                "files_sha256": {**{p.relative_to(out).as_posix(): sha256(p) for p in files},
                                 "runs.json": hashlib.sha256(runs).hexdigest()}}
    return manifest, runs


def publish(out: Path, manifest: dict, runs: bytes) -> None:
    with open(out / "runs.json", "xb") as f:
        f.write(runs)
    with open(out / "manifest.json", "x", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=1)
        f.write("\n")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    mk = sub.add_parser("make")
    mk.add_argument("--v7-plan", required=True, type=Path)
    mk.add_argument("--v7-captures", required=True, type=Path)
    mk.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    manifest, runs = build(a.v7_plan, a.v7_captures, a.out)
    publish(a.out.resolve(), manifest, runs)
    print(json.dumps({k: manifest[k] for k in ("n_items", "items_per_series", "items_per_block", "group_blocks",
                                               "v7_without_plan")}, indent=1))


if __name__ == "__main__":
    main()
