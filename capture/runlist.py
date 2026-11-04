"""Run several captures in a random order (capture spec section 6, conditions run_order).

    python -m capture.runlist --list <runs.json> --key 20261031 7 6 --out <dir>

runs.json: [{"name": "<series>/<unit>", "args": [session arguments without --out], "after": <name or null>,
"block": <day>, "person": <replay person's unit, H1_idle and H1_load only>, "role": "idle" | "load"}, ...]
The name is the folder of the item under --out, CAPTURE_LAYOUT of works/prototype_v7_run.py (H1_idle/h00, L0/e0,
M1_8B/t00, P_idle/c005, ...).
An argument "{out:NAME}" is replaced by the output folder of the finished capture NAME (M1_replay types
"{out:M1}/schedule.ndjson").

Order: one generator numpy.random.default_rng(key); blocks in ascending order, each block put in the order of
rng.permutation(len(block)) [仮: one generator for all blocks, drawn block by block]. Then dependencies: an item whose
dependency comes later is moved to just after it (repeated until none is left), and the moves are counted.
A dependency must be in the same block or an earlier one.
Persons (conditions run_order: 'for each replay person the order of idle and load is drawn with key [20261031, 7, 7,
person]'): the two items of a person keep the two places the block's permutation gave them, and which of them comes
first is numpy.random.default_rng([20261031, 7, 7, person]).permutation(2) applied to (idle, load) (review of
2026-11-04; the reading that the person's draw decides only who goes first is Claude's, recorded in decisions.md).

Each capture is run once; a capture that does not end 'done' (interrupted or failed) is moved to NAME.try1 and redone
once in NAME, so that the folder NAME always holds the last attempt; a second failure is counted and the series is
missing (never replaced). An M2 capture whose plan was made is redone by typing that plan (--events of the first
attempt's events.ndjson): its plan is never made again (review of 2026-11-04). An M2 plan failing three times is not
redone. An item whose dependency is missing is not run.
runlog.ndjson: a header line, one line per item {type: item, item: NAME, result, detail, attempts} and a footer.
result is the word of conditions analysis_details.run.captures: done, redone (done at the second attempt), failed
(both attempts failed: a missing pair; the folder keeps the plan files) or no_plan (the M2 plan failed three times, or
the dependency that gives the plan is missing); detail keeps the runlist's own word.
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


PERSON_KEY = (20261031, 7, 7)
ROLES = ("idle", "load")


def check_persons(items: list[dict]) -> dict[int, dict[str, dict]]:
    by: dict[int, dict[str, dict]] = {}
    for it in items:
        if "person" not in it and "role" not in it:
            continue
        p, r = it.get("person"), it.get("role")
        if not isinstance(p, int) or isinstance(p, bool) or p < 0 or r not in ROLES:
            raise ValueError(f"{it['name']}: person must be an int >= 0 and role one of {ROLES}")
        if r in by.setdefault(p, {}):
            raise ValueError(f"person {p} has two {r} items")
        by[p][r] = it
    for p, d in by.items():
        if set(d) != set(ROLES):
            raise ValueError(f"person {p} needs one idle and one load item")
        if d["idle"]["block"] != d["load"]["block"]:
            raise ValueError(f"person {p}: idle and load must be in the same block")
        if d["idle"].get("after") or d["load"].get("after"):
            raise ValueError(f"person {p}: idle and load items have no dependency")
    return by


def person_first(person: int) -> str:
    """The role typed first for this person (conditions run_order, key [20261031, 7, 7, person])."""
    return ROLES[int(np.random.default_rng([*PERSON_KEY, person]).permutation(2)[0])]


def order(items: list[dict], key: list[int]) -> tuple[list[dict], int]:
    """The run order and the number of moves made for dependencies."""
    check_items(items)
    persons = check_persons(items)
    rng = np.random.default_rng(key)
    out: list[dict] = []
    for b in sorted({it["block"] for it in items}):
        blk = [it for it in items if it["block"] == b]
        out += [blk[i] for i in rng.permutation(len(blk))]
    for p, d in persons.items():  # the person's two places stay; the person's key decides which comes first
        i, j = sorted((out.index(d["idle"]), out.index(d["load"])))
        first = person_first(p)
        out[i], out[j] = d[first], d[ROLES[1 - ROLES.index(first)]]
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


NO_PLAN = "m2_failed"  # an M2 plan failing three times: a counted result of the series, nothing typed, not redone
RESULT = {"done": "done", "done_after_redo": "redone", "missing_after_redo": "failed", "rename_failed": "failed",
          "m2_failed": "no_plan", "missing_dependency": "no_plan"}


def rename(src: Path, dst: Path, tries: int = 10, wait_s: float = 1.0) -> None:
    """Windows refuses a rename while a closing process still holds a file: retried, then raised (the runlist stops)."""
    for i in range(tries):
        try:
            src.rename(dst)
            return
        except PermissionError:
            if i == tries - 1:
                raise
            time.sleep(wait_s)


def redo_args(args: list[str], first: Path) -> list[str]:
    """An M2 capture is redone by typing the plan of its first attempt (never made again)."""
    if "--llm-plan" not in args or not (first / "events.ndjson").exists():
        return args
    return [x for x in args if x != "--llm-plan"] + ["--events", str(first / "events.ndjson")]


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
        name = it["name"]
        dep = it.get("after")
        if dep and dep not in done:
            rec = {"type": "item", "item": name, "result": RESULT["missing_dependency"],
                   "detail": "missing_dependency", "dependency": dep}
            write(rec)
            results.append(rec)
            continue
        attempts = []
        folder = out / name
        args = resolve(it["args"], done)
        detail = "missing_after_redo"
        for a in range(2):
            t0 = time.time()
            status = runner(args, folder)
            attempts.append({"folder": str(folder.relative_to(out)), "status": status,
                             "seconds": round(time.time() - t0, 1)})
            if status == "done":
                done[name] = folder
                detail = "done" if a == 0 else "done_after_redo"
                break
            if status == NO_PLAN:
                detail = NO_PLAN
                break
            if a == 0:  # the interrupted attempt is kept beside, and the redo takes the item's folder
                first = folder.with_name(folder.name + ".try1")
                try:
                    if first.exists():
                        raise FileExistsError(f"{first} exists: the runlist never writes over a capture")
                    if folder.exists():
                        rename(folder, first)
                        attempts[-1]["folder"] = str(first.relative_to(out))
                except OSError as e:  # the item cannot be redone in its folder: failed, and the runlist goes on
                    attempts[-1]["rename_error"] = f"{type(e).__name__}: {e}"
                    detail = "rename_failed"
                    break
                new_args = redo_args(args, first)
                if new_args != args:
                    attempts[-1]["plan_kept_for_redo"] = str((first / "events.ndjson").relative_to(out))
                args = new_args
        rec = {"type": "item", "item": name, "result": RESULT[detail], "detail": detail, "attempts": attempts}
        write(rec)
        results.append(rec)
    write({"type": "footer", "end_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "counts": {r: sum(x["result"] == r for x in results) for r in ("done", "redone", "failed", "no_plan")},
           "details": {r: sum(x["detail"] == r for x in results) for r in RESULT}})
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
