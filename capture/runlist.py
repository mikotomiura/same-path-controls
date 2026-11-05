"""Run several captures in a random order (capture spec section 6, conditions run_order).

    python -m capture.runlist --list <runs.json> --key 20261031 7 6 --out <dir> [--block <b>]

runs.json: [{"name": "<series>/<unit>", "args": [session arguments without --out], "after": <name or null>,
"block": <day>, "person": <replay person's unit, H1_idle and H1_load only>, "role": "idle" | "load",
"needs": [files the item reads from its dependency], "make": [a script and its arguments, run before the capture],
"show": <a file shown before the capture>}, ...] (needs, make and show added for the production plan of
works/prototype_v7_plan.py, amendment B16; .steering/20261104-v7-production-plan)
The name is the folder of the item under --out, CAPTURE_LAYOUT of works/prototype_v7_run.py (H1_idle/h00, L0/e0,
M1_8B/t00, P_idle/c005, ...).
An argument "{out:NAME}" is replaced by the output folder of the finished capture NAME (M1_replay types
"{out:M1}/schedule.ndjson"); "{root}" by the --out folder.
Dependencies: an item with "after" and no "needs" runs only after its dependency ended done. An item with "needs" runs
after its dependency ended done or failed (both attempts failed: its last folder keeps the plan files) when every file
of "needs" exists: the analysis checks a failed capture's plan and its dependants against it (prototype_v7_run
check_plans). Otherwise the item is no_plan (detail missing_dependency or missing_plan_file).
"make" runs once before the item's first attempt, before "needs" is looked at ([python, *make] in the repository): exit 0 goes on, exit 3
(NO_PLAN_EXIT: the dependency left no plan the analysis reads, e.g. works/prototype_v7_plan.py gate-m1) is no_plan
(detail make_no_plan), and any other exit stops the runlist (a fault of the code or the machine is not a missing
plan; the runlog is left without its footer and the next block refuses to start). "show" prints the file and waits
for Enter (L0: the prompt the researcher writes on).
--block b runs only block b, in its place of the whole order (the order of all blocks is drawn as without --block),
into the same --out folder, with runlog.b<b>.ndjson; every earlier block must have a finished runlog there, run from
the same runs.json (its SHA-256 in every header) and key, holding exactly that block's items; they are the
dependencies of block b. Without --block, every block runs in one runlog.ndjson. --out is made absolute first.

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
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def check_items(items: list[dict]) -> None:
    for it in items:
        for f in ("needs", "make"):
            if f in it and not (isinstance(it[f], list) and it[f] and all(isinstance(x, str) for x in it[f])):
                raise ValueError(f"{it.get('name')}: {f} must be a non-empty list of strings")
        if "show" in it and not isinstance(it["show"], str):
            raise ValueError(f"{it.get('name')}: show must be a string")
        if "needs" in it and not it.get("after"):
            raise ValueError(f"{it.get('name')}: needs goes with after")
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


def resolve(args: list[str], done: dict[str, Path], root: Path | None = None) -> list[str]:
    out = []
    for x in args:
        if root is not None:
            x = x.replace("{root}", str(root))
        for name, p in done.items():
            x = x.replace("{out:" + name + "}", str(p))
        if "{out:" in x or "{root}" in x:
            raise ValueError(f"unresolved argument {x}")
        out.append(x)
    return out


REMOTE_HOST = "remote_host"  # the session's status when a remote-desktop host stopped a live capture: the runlist stops


def default_runner(args: list[str], out: Path) -> str:
    """Run one capture; returns the status written in its meta.json ('done', 'interrupted', 'failed', ...)."""
    from capture import remote
    r = subprocess.run([sys.executable, "-m", "capture.session", *args, "--out", str(out)], cwd=ROOT)
    meta = out / "meta.json"
    if r.returncode == remote.REMOTE_HOST_EXIT:  # with a meta.json or not: the runlist stops (Codex, round 3)
        return REMOTE_HOST
    if not meta.exists():
        return f"failed (exit {r.returncode}, no meta.json)"
    return json.loads(meta.read_text(encoding="utf-8")).get("status", f"failed (exit {r.returncode})")


NO_PLAN_EXIT = 3   # works/prototype_v7_plan.py NoPlan


def runs_sha256(items: list[dict]) -> str:
    """The SHA-256 of the runs.json file when main reads one; of the items' canonical JSON otherwise (tests)."""
    return hashlib.sha256(json.dumps(items, sort_keys=True).encode()).hexdigest()


def default_maker(argv: list[str]) -> int:
    """Make an item's plan before its capture (works/prototype_v7_plan.py derive); returns the exit code."""
    return subprocess.run([sys.executable, *argv], cwd=ROOT).returncode


def default_shower(path: str) -> None:
    print("\n" + Path(path).read_text(encoding="utf-8").strip() + "\n", flush=True)
    input("Press Enter when ready to start (the countdown follows). ")


NO_PLAN = "m2_failed"  # an M2 plan failing three times: a counted result of the series, nothing typed, not redone
RESULT = {"done": "done", "done_after_redo": "redone", "missing_after_redo": "failed", "rename_failed": "failed",
          "m2_failed": "no_plan", "missing_dependency": "no_plan", "missing_plan_file": "no_plan",
          "make_no_plan": "no_plan"}


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


def earlier_blocks(out: Path, items: list[dict], block: int, key: list[int],
                   runs_sha: str) -> tuple[dict[str, Path], dict[str, Path]]:
    """The items of the finished runlogs of the blocks before `block`: (done, attempted) as run() keeps them. Each
    must be of the same runs.json and key, and hold exactly its block's items."""
    done: dict[str, Path] = {}
    attempted: dict[str, Path] = {}
    for b in sorted({it["block"] for it in items}):
        if b >= block:
            continue
        f = out / f"runlog.b{b}.ndjson"
        if not f.exists():
            raise ValueError(f"block {b} comes before block {block} and has no runlog in {out}")
        rows = [json.loads(ln) for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not rows or rows[0].get("type") != "header" or rows[-1].get("type") != "footer":
            raise ValueError(f"{f} did not finish (no header or footer line): ask before going on")
        h = rows[0]
        if h.get("key") != key or h.get("runs_sha256") != runs_sha or h.get("block") != b:
            raise ValueError(f"{f} was run with another key, runs.json or block: ask before going on")
        names = sorted(r.get("item") for r in rows[1:-1])
        if names != sorted(it["name"] for it in items if it["block"] == b):
            raise ValueError(f"{f} does not hold exactly the items of block {b}: ask before going on")
        for r in rows[1:-1]:
            if r.get("result") in ("done", "redone"):
                done[r["item"]] = attempted[r["item"]] = out / r["item"]
            elif r.get("result") == "failed" and (out / r["item"]).is_dir():
                attempted[r["item"]] = out / r["item"]
    return done, attempted


def default_remote_check() -> list[str]:
    from capture import remote
    return remote.running()


def run(items: list[dict], key: list[int], out: Path, runner=default_runner, block: int | None = None,
        maker=default_maker, shower=default_shower, runs_sha: str | None = None,
        remote_check=default_remote_check) -> list[dict]:
    out = Path(out).resolve()
    runs_sha = runs_sha or runs_sha256(items)
    seq, moves = order(items, key)
    in_run = seq if block is None else [it for it in seq if it["block"] == block]
    if any("--live" in it["args"] for it in in_run):  # before anything is written: the run can simply be made again
        hosts = remote_check()                         # (decision 5; before the mkdir, code-reviewer round 3)
        if hosts:
            from capture import remote
            raise RuntimeError(remote.message(hosts) + " The runlist did not start.")
    if block is None:
        out.mkdir(parents=True, exist_ok=False)
        log_name, done, attempted = "runlog.ndjson", {}, {}
    else:
        blocks = sorted({it["block"] for it in items})
        if block not in blocks:
            raise ValueError(f"no item in block {block}")
        out.mkdir(parents=True, exist_ok=True)
        if (out / "runlog.ndjson").exists():
            raise ValueError(f"{out} holds a runlog of all blocks: --block is not mixed with it")
        done, attempted = earlier_blocks(out, items, block, list(key), runs_sha)
        seq = [it for it in seq if it["block"] == block]
        log_name = f"runlog.b{block}.ndjson"
    if (out / log_name).exists():
        raise ValueError(f"{out / log_name} exists: block {block} has run (or stopped) before; ask before going on")
    log = open(out / log_name, "x", encoding="utf-8", newline="\n")

    def write(rec: dict) -> None:
        log.write(json.dumps(rec) + "\n")
        log.flush()

    write({"type": "header", "key": list(key), "block": block, "runs_sha256": runs_sha,
           "order": [it["name"] for it in seq],
           "dependency_moves": moves, "start_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    results = []

    def no_plan(name: str, detail: str, **extra) -> None:
        rec = {"type": "item", "item": name, "result": RESULT[detail], "detail": detail, **extra}
        write(rec)
        results.append(rec)

    for it in seq:
        name = it["name"]
        dep = it.get("after")
        if dep and dep not in (attempted if "needs" in it else done):
            no_plan(name, "missing_dependency", dependency=dep)
            continue
        if "make" in it:   # before needs: the make (gate-m1, derive) tells a missing plan from a fault
            rc = maker(resolve(it["make"], attempted, out))
            if rc == NO_PLAN_EXIT:
                no_plan(name, "make_no_plan", exit=rc)
                continue
            if rc != 0:
                log.close()
                raise RuntimeError(f"{name}: make exited {rc} (not a missing plan): the runlist stops; ask before "
                                   "going on")
        if "needs" in it:
            missing = [f for f in resolve(it["needs"], attempted, out) if not Path(f).exists()]
            if missing:
                no_plan(name, "missing_plan_file", dependency=dep, missing=missing)
                continue
        if "show" in it:
            shower(it["show"])
        attempts = []
        folder = out / name
        leftover = [f for f in (folder, folder.with_name(folder.name + ".try1")) if f.exists()]
        if leftover:  # an earlier run's capture (a runlist made again after a stop): never read as this run's
            write({"type": "stopped", "item": name, "reason": "folder_exists",
                   "folders": [f.relative_to(out).as_posix() for f in leftover]})
            log.close()
            raise RuntimeError(f"{name}: {', '.join(map(str, leftover))} exists before its capture: an earlier run's "
                               "folder is never read as this run's (code-reviewer round 3). Ask before going on.")
        args = resolve(it["args"], attempted, out)
        detail = "missing_after_redo"
        for a in range(2):
            t0 = time.time()
            status = runner(args, folder)
            attempts.append({"folder": folder.relative_to(out).as_posix(), "status": status,
                             "seconds": round(time.time() - t0, 1)})
            if status == "done":
                done[name] = attempted[name] = folder
                detail = "done" if a == 0 else "done_after_redo"
                break
            if status == NO_PLAN:
                detail = NO_PLAN
                break
            if status == REMOTE_HOST:  # a host started after the check above: stop, never a failed item (decision 5)
                from capture import remote
                write({"type": "stopped", "item": name, "reason": "remote_host"})
                log.close()
                raise RuntimeError(f"{name}: a remote-desktop host is running, or the check of the hosts failed (the "
                                   "session's message above); the runlist stops here (the item is not counted). "
                                   f"{remote.message(list(remote.REMOTE_HOSTS))} Ask before going on.")
            if a == 0:  # the interrupted attempt is kept beside, and the redo takes the item's folder
                first = folder.with_name(folder.name + ".try1")
                try:
                    if first.exists():
                        raise FileExistsError(f"{first} exists: the runlist never writes over a capture")
                    if folder.exists():
                        rename(folder, first)
                        attempts[-1]["folder"] = first.relative_to(out).as_posix()
                except OSError as e:  # the item cannot be redone in its folder: failed, and the runlist goes on
                    attempts[-1]["rename_error"] = f"{type(e).__name__}: {e}"
                    detail = "rename_failed"
                    if folder.is_dir():   # the first attempt stays in the folder: its plan files as any failed one
                        attempted[name] = folder
                    break
                new_args = redo_args(args, first)
                if new_args != args:
                    attempts[-1]["plan_kept_for_redo"] = (first / "events.ndjson").relative_to(out).as_posix()
                args = new_args
        if detail == "missing_after_redo":   # both attempts failed: the last folder keeps the plan files
            attempted[name] = folder
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
    ap.add_argument("--block", type=int)
    a = ap.parse_args()
    raw = a.list.read_bytes()
    items = json.loads(raw.decode("utf-8"))
    res = run(items, a.key, a.out, block=a.block, runs_sha=hashlib.sha256(raw).hexdigest())
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
