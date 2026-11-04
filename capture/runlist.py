"""Run several captures in a random order (capture spec section 6, conditions run_order).

    python -m capture.runlist --list <runs.json> --key 20261031 7 6 --out <dir>

runs.json: [{"name": ..., "args": [session arguments without --out], "after": <name or null>, "block": <day>}, ...]
An argument "{out:NAME}" is replaced by the output folder of the finished capture NAME (M1_replay types
"{out:M1}/schedule.ndjson").

Order: one generator numpy.random.default_rng(key); blocks in ascending order, each block put in the order of
rng.permutation(len(block)) [仮: one generator for all blocks, drawn block by block]. Then dependencies: an item whose
dependency comes later is moved to just after it (repeated until none is left), and the moves are counted.
A dependency must be in the same block or an earlier one.

Each capture is run once; a capture that does not end 'done' (interrupted or failed) is redone once in a new folder
(NAME.redo1); a second failure is counted and the series is missing (never replaced). An item whose dependency is
missing is not run and is counted as missing. Every result goes to runlog.ndjson.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def check_items(items: list[dict]) -> None:
    names = [it["name"] for it in items]
    if len(set(names)) != len(names):
        raise ValueError("names must be unique")
    block = {it["name"]: it["block"] for it in items}
    for it in items:
        d = it.get("after")
        if d is None:
            continue
        if d not in block:
            raise ValueError(f"{it['name']} depends on an unknown item {d}")
        if block[d] > it["block"]:
            raise ValueError(f"{it['name']} depends on {d} of a later block")
        if d == it["name"]:
            raise ValueError(f"{d} depends on itself")


def order(items: list[dict], key: list[int]) -> tuple[list[dict], int]:
    """The run order and the number of moves made for dependencies."""
    check_items(items)
    rng = np.random.default_rng(key)
    out: list[dict] = []
    for b in sorted({it["block"] for it in items}):
        blk = [it for it in items if it["block"] == b]
        out += [blk[i] for i in rng.permutation(len(blk))]
    moves = 0
    for _ in range(len(out) ** 2 + 1):
        pos = {it["name"]: i for i, it in enumerate(out)}
        bad = next((it for it in out if it.get("after") and pos[it["after"]] > pos[it["name"]]), None)
        if bad is None:
            return out, moves
        out.remove(bad)
        out.insert(out.index(next(x for x in out if x["name"] == bad["after"])) + 1, bad)
        moves += 1
    raise ValueError("the dependencies form a cycle")


def resolve(args: list[str], done: dict[str, Path]) -> list[str]:
    out = []
    for x in args:
        for name, p in done.items():
            x = x.replace("{out:" + name + "}", str(p))
        if "{out:" in x:
            raise ValueError(f"unresolved argument {x}")
        out.append(x)
    return out


def default_runner(args: list[str], out: Path) -> str:
    """Run one capture; returns the status written in its meta.json ('done', 'interrupted', 'failed', ...)."""
    r = subprocess.run([sys.executable, "-m", "capture.session", *args, "--out", str(out)], cwd=ROOT)
    meta = out / "meta.json"
    if not meta.exists():
        return f"failed (exit {r.returncode}, no meta.json)"
    return json.loads(meta.read_text(encoding="utf-8")).get("status", f"failed (exit {r.returncode})")


OK = ("done", "m2_failed")  # an M2 plan failing three times is a counted result of the series, not a failed capture


def run(items: list[dict], key: list[int], out: Path, runner=default_runner) -> list[dict]:
    out.mkdir(parents=True, exist_ok=False)
    seq, moves = order(items, key)
    log = open(out / "runlog.ndjson", "x", encoding="utf-8", newline="\n")

    def write(rec: dict) -> None:
        log.write(json.dumps(rec) + "\n")
        log.flush()

    write({"type": "header", "key": key, "order": [it["name"] for it in seq], "dependency_moves": moves,
           "start_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    done: dict[str, Path] = {}
    results = []
    for it in seq:
        dep = it.get("after")
        if dep and dep not in done:
            rec = {"type": "item", "name": it["name"], "result": "missing_dependency", "dependency": dep}
            write(rec)
            results.append(rec)
            continue
        attempts = []
        for a in range(2):
            folder = out / (it["name"] if a == 0 else f"{it['name']}.redo{a}")
            t0 = time.time()
            status = runner(resolve(it["args"], done), folder)
            attempts.append({"folder": folder.name, "status": status, "seconds": round(time.time() - t0, 1)})
            if status in OK:
                done[it["name"]] = folder
                break
        if it["name"] in done:
            result = "done" if len(attempts) == 1 else "done_after_redo"
        else:
            result = "missing_after_redo"
        rec = {"type": "item", "name": it["name"], "result": result, "attempts": attempts}
        write(rec)
        results.append(rec)
    write({"type": "footer", "end_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "counts": {r: sum(x["result"] == r for x in results)
                      for r in ("done", "done_after_redo", "missing_after_redo", "missing_dependency")}})
    log.close()
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", type=Path, required=True)
    ap.add_argument("--key", type=int, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    items = json.loads(a.list.read_text(encoding="utf-8"))
    res = run(items, a.key, a.out)
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
