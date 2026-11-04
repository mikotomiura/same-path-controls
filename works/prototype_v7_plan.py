"""Prototype v7: the production plan (the runlist and the planned series), made on the work PC from KUPA.

    python works/prototype_v7_plan.py make --out <plan folder>
    python works/prototype_v7_plan.py derive --series M0|M3a|M3b --text <t> --m1 <M1_8B's capture folder>
                                             --plan <plan folder> --out <events.ndjson>
    python works/prototype_v7_plan.py gate-m1 --m1 <an M1's capture folder>

make: checks KUPA against the SHA-256 of the conditions (data.kupa) and draws the persons with the analysis's own
function (prototype_v7_run.draw_persons), then writes into a new folder:
  runs.json                 the items of capture/runlist.py (names of prototype_v7_run.CAPTURE_LAYOUT)
  events/H/hNN.ndjson       H0's counted presses of replay person NN (H1_idle, H1_load; H1_rep for NN < 10), with the
                            original keys; a key outside printable ASCII is its v6 kind's representative (amendment B16)
  events/P/cNNN.ndjson      the positive controls: a-j cycled, 1000 presses, NNN ms
  prompts/Pkk.txt           the 40 prompts of M1 and M2 (P1-P10 KUPA's, by prompt_id; P11-P40 data.prompts_added)
  prompts/L0_eN.txt         the 4 prompts of L0 (series.L0's draw); shown to the researcher before each essay
  load/<series>_<unit>.txt  the background prompt of each capture under load (series.H1_load, amendment B16)
  m3_donors.json            the gaps of M3's 40 donors only (derived from KUPA: it stays on the work PC)
  manifest.json             counts, SHA-256 of every file, the persons' SHA-256 (never an id), the conditions' SHA-256
derive: at run time (runlist 'make'), M0's plan (55 ms on M1_8B's keys) and M3a's and M3b's (prototype_v7.m3 on
M1_8B's keys = M0's keys) from M1_8B's executable plan of the same text (amendment B16).
gate-m1: at run time, before M1_replay: whether the M1 capture left the plan the analysis reads (m1_plan).
Exit codes: 0 done; 3 (NoPlan, capture/runlist.py NO_PLAN_EXIT) the dependency left no plan: the item is no_plan;
anything else is a fault and stops the runlist.

Everything written here is checked again by the analysis (prototype_v7_run.check_plans): this script and the analysis
share draw_persons, m3 and the constants, and the contract test (tests/test_prototype_v7_plan.py) runs the plan through
the runlist and check_plans.
"""

from __future__ import annotations

import argparse
import atexit
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if not sys.pycache_prefix:   # as prototype_v7_run: no __pycache__ left in the repository (derive runs 120 times)
    sys.pycache_prefix = tempfile.mkdtemp(prefix="pycache-")
    atexit.register(shutil.rmtree, sys.pycache_prefix, ignore_errors=True)
import numpy as np  # noqa: E402

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "works"))
sys.path.insert(0, str(ROOT))
import prototype_v6 as v6  # noqa: E402
import prototype_v7 as v7  # noqa: E402
import prototype_v7_run as vr  # noqa: E402
from capture import events as ev  # noqa: E402

S = v7.SEED
N_BLOCKS = 6                                     # run_order (amendment B16): blocks 1-6 besides L0's block 0
BLOCK_KEY = (S, 7, 14)                           # + group kind
L0_KEY = (S, 7, 2)                               # series.L0
H_LOAD_KEY = (S, 7, 1)                           # + person (series.H1_load)
OTHER_LOAD_KEY = (S, 7, 15)                      # + i (amendment B16: L1_load i = essay, P_load i = 4 + control place)
SEED_LLM = 1                                     # series.M1 and M2: seed 1
REPRESENTATIVE = {"alnum": "e", "other": ","}   # amendment B16: = capture events.LIVE_KEY of these kinds
GROUP_KINDS = ("person", "text_8B", "text_14B", "M2_8B", "M2_14B", "control", "l1_essay")
SESSION = ["--auto-focus"]                      # the researcher's decision E-7: production starts unattended
N_PROMPTS_KUPA = 10

assert all(ev.LIVE_KEY[k] == r for k, r in REPRESENTATIVE.items())


class PlanStop(SystemExit):
    pass


class NoPlan(SystemExit):
    """The dependency left no plan the analysis reads: the dependant is no_plan (exit 3)."""

    def __init__(self, msg: str):
        print(f"no plan: {msg}", file=sys.stderr)
        super().__init__(3)


# ---------------------------------------------------------------- keys of the replayed persons


def read_kupa_keys(path: Path, ids: set[str]) -> dict[str, tuple[np.ndarray, list[str]]]:
    """The counted presses of the given ids as v6.read_kupa reads them (same rows, same stable sort, same drop of a
    press at the same time as the previous one, times from 0), with the key of each press kept."""
    raw: dict[str, tuple[list[float], list[str]]] = {}
    csv.field_size_limit(sys.maxsize)
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["id"] not in ids or row["type"] != "down" or row["is_repeat"] == "True":
                continue
            if v6.kupa_kind(row["key"]) is None:
                continue
            ts, ks = raw.setdefault(row["id"], ([], []))
            ts.append(float(row["time"]))
            ks.append(row["key"])
    out = {}
    for pid, (ts, ks) in raw.items():
        t, k, _ = v6.clean(ts, ks)   # clean() carries any list beside the times
        out[pid] = (t, k)
    return out


def replay_keys(keys: list[str]) -> tuple[list[str], int]:
    """Amendment B16: a key outside printable ASCII (and not Enter, Backspace, Delete) becomes its v6 kind's
    representative, so that the press keeps H0's kind. Returns the keys and the number replaced."""
    out, n = [], 0
    for k in keys:
        if k in ev.NAMED_KEYS or ev.typable(k):
            out.append(k)
            continue
        kind = v6.kupa_kind(k)
        if kind not in REPRESENTATIVE:
            raise PlanStop(f"an untypable key of kind {kind!r} has no representative (amendment B16 gives alnum, other)")
        out.append(REPRESENTATIVE[kind])
        n += 1
    return out, n


def check_replay(times: np.ndarray, keys: list[str], h0: dict) -> None:
    """The replayed presses are H0's: the same times, and the same v6 kinds (as check_plans reads them)."""
    if times.shape != h0["times"].shape or not np.array_equal(times, h0["times"]):
        raise PlanStop("the keys' reading of KUPA does not give H0's times")
    if [v7.LOGGER_KIND_TO_V6[vr.key_kind(k)] for k in keys] != list(h0["kinds"]):
        raise PlanStop("a replayed key does not keep H0's kind")


# ---------------------------------------------------------------- prompts


def kupa_prompts(meta_path: Path) -> list[str]:
    """data.prompts: P1-P10, the 'prompt' column of KUPA-KEYS-META.csv in the order of 'prompt_id'."""
    by: dict[int, str] = {}
    csv.field_size_limit(sys.maxsize)
    with open(meta_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row.get("prompt_id", "").strip():
                continue
            pid, text = int(row["prompt_id"]), row["prompt"].strip()
            if by.setdefault(pid, text) != text:
                raise PlanStop(f"KUPA's prompt_id {pid} holds two prompts")
    if len(by) != N_PROMPTS_KUPA:
        raise PlanStop(f"KUPA-KEYS-META.csv holds {len(by)} prompts, not {N_PROMPTS_KUPA}")
    return [by[k] for k in sorted(by)]


def all_prompts(c: dict, kupa: list[str]) -> list[str]:
    added = c["data"]["prompts_added"]
    extra = [added[f"P{k}"] for k in range(N_PROMPTS_KUPA + 1, vr.N_TEXTS + 1)]
    out = list(kupa) + extra
    if len(out) != vr.N_TEXTS or len(set(out)) != len(out):
        raise PlanStop("the 40 prompts are not 40 distinct prompts")
    return out


def l0_prompts() -> list[int]:
    return [int(i) for i in np.random.default_rng(list(L0_KEY)).choice(N_PROMPTS_KUPA, vr.N_ESSAYS, replace=False)]


def h_load_prompt(person: int) -> int:
    return int(np.random.default_rng([*H_LOAD_KEY, person]).integers(N_PROMPTS_KUPA))


def other_load_prompt(i: int) -> int:
    return int(np.random.default_rng([*OTHER_LOAD_KEY, i]).integers(N_PROMPTS_KUPA))


# ---------------------------------------------------------------- blocks


def blocks_of(kind: int, n_groups: int) -> list[int]:
    """Amendment B16: the groups of one kind in the order of default_rng([20261031, 7, 14, kind]).permutation, dealt to
    blocks 1..6 in turn. Returns the block of each group (index = group)."""
    perm = np.random.default_rng([*BLOCK_KEY, kind]).permutation(n_groups)
    out = [0] * n_groups
    for j, g in enumerate(perm):
        out[int(g)] = 1 + j % N_BLOCKS
    return out


# ---------------------------------------------------------------- the plan


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8", newline="\n") as f:
        f.write(text.strip() + "\n")


def models(c: dict) -> tuple[str, str]:
    m = c["environment"]["models"]
    return m["8B_class"]["name"], m["14B_class"]["name"]


def build(c: dict, h_series: list[tuple[np.ndarray, list[str]]], donors: dict[int, np.ndarray], n_e: int,
          prompts: list[str], out: Path) -> tuple[dict, bytes]:
    """Write the plan into the new folder `out` (see the module's text) but runs.json, and return the manifest and
    runs.json's bytes (written by publish). h_series: the replay
    persons' (times, original keys) in replay_H's draw order; donors: M3's donor gaps by E index; n_e: |E|."""
    if len(h_series) != vr.N_H:
        raise PlanStop(f"{len(h_series)} replay persons, not {vr.N_H}")
    out.mkdir(parents=True, exist_ok=False)
    out = out.resolve()
    m8, m14 = models(c)
    items: list[dict] = []
    replaced = {"presses": 0, "persons": 0}

    def ev_file(rel: str, presses: list[dict], series: str, source: str, **extra) -> str:
        p = out / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        ev.write(p, presses, series, source, **extra)
        return str(p)

    def prompt_file(rel: str, text: str) -> str:
        write_text(out / rel, text)
        return str(out / rel)

    def load_args(rel: str, k: int) -> list[str]:
        return ["--load", m14, "--load-prompt-file", prompt_file(rel, prompts[k])]

    def add(series: str, unit: int, args: list[str], block: int, **extra) -> None:
        items.append({"name": vr.item_name(series, unit), "args": [*args, *SESSION], "block": block, **extra})

    # block 0: L0; L1 in blocks 1-6 (one group per essay)
    l0 = l0_prompts()
    l1_block = blocks_of(GROUP_KINDS.index("l1_essay"), vr.N_ESSAYS)
    for e in range(vr.N_ESSAYS):
        add("L0", e, ["--live"], 0, show=prompt_file(f"prompts/L0_e{e}.txt", prompts[l0[e]]))
        src = "{out:" + vr.item_name("L0", e) + "}/l1_events.ndjson"
        dep = {"after": vr.item_name("L0", e)}   # L0 done (a failed L0 is not captured: the analysis has no L1 plan)
        add("L1_idle", e, ["--events", src], l1_block[e], **dep)
        add("L1_load", e, ["--events", src, *load_args(f"load/L1_load_e{e}.txt", other_load_prompt(e))],
            l1_block[e], **dep)

    # replay persons
    p_block = blocks_of(GROUP_KINDS.index("person"), vr.N_H)
    for i, (times, keys) in enumerate(h_series):
        k2, n = replay_keys(keys)
        replaced["presses"] += n
        replaced["persons"] += n > 0
        f = ev_file(f"events/H/h{i:02d}.ndjson", ev.make(times, k2), "H1", "KUPA essay of replay person "
                    f"{i} (replay_H's draw order)", replaced=n)
        add("H1_idle", i, ["--events", f], p_block[i], person=i, role="idle")
        add("H1_load", i, ["--events", f, *load_args(f"load/H1_load_h{i:02d}.txt", h_load_prompt(i))], p_block[i],
            person=i, role="load")
        if i < vr.N_REP:
            add("H1_rep", i, ["--events", f], p_block[i])

    # texts: M1 and its dependants
    pfiles = [prompt_file(f"prompts/P{t + 1}.txt", prompts[t]) for t in range(vr.N_TEXTS)]
    for model, name, kind in (("8B", m8, "text_8B"), ("14B", m14, "text_14B")):
        blk = blocks_of(GROUP_KINDS.index(kind), vr.N_TEXTS)
        for t in range(vr.N_TEXTS):
            m1 = vr.item_name(f"M1_{model}", t)
            add(f"M1_{model}", t, ["--llm-stream", "--model", name, "--prompt-file", pfiles[t], "--seed",
                                   str(SEED_LLM)], blk[t])
            m1_dir = "{out:" + m1 + "}"
            sched = m1_dir + "/schedule.ndjson"
            dep = {"after": m1, "needs": [sched]}
            add(f"M1_replay_{model}", t, ["--events", sched], blk[t],
                make=["works/prototype_v7_plan.py", "gate-m1", "--m1", m1_dir], **dep)
            if model != "8B":
                continue
            for s_ in ("M0", "M3a", "M3b"):
                derived = "{root}/_derived/" + s_ + f"/t{t:02d}.ndjson"
                make = ["works/prototype_v7_plan.py", "derive", "--series", s_, "--text", str(t), "--m1",
                        m1_dir, "--plan", str(out), "--out", derived]
                add(s_, t, ["--events", derived], blk[t], make=make, **dep)
    for model, name in (("8B", m8), ("14B", m14)):
        blk = blocks_of(GROUP_KINDS.index(f"M2_{model}"), vr.N_TEXTS)
        for t in range(vr.N_TEXTS):
            add(f"M2_{model}", t, ["--llm-plan", "--model", name, "--prompt-file", pfiles[t], "--seed",
                                   str(SEED_LLM)], blk[t])

    # controls
    c_block = blocks_of(GROUP_KINDS.index("control"), len(vr.CONTROLS_MS))
    for j, ms in enumerate(vr.CONTROLS_MS):
        u = int(ms)
        f = ev_file(f"events/P/c{u:03d}.ndjson", ev.cycle_letters(vr.P_PRESSES, ms), "P", f"constant {ms} ms")
        add("P_idle", u, ["--events", f], c_block[j])
        add("P_load", u, ["--events", f, *load_args(f"load/P_load_c{u:03d}.txt", other_load_prompt(4 + j))],
            c_block[j])
        if ms in v7.P_NOHOOK_MS:
            add("P_nohook", u, ["--events", f, "--no-hook"], c_block[j])

    # M3's donors (the only KUPA gaps kept beside the replay schedules; on the work PC only)
    donor_idx = v7.m3_donors(n_e)
    don = {"n_e": n_e, "donors": {str(t): {"index": int(donor_idx[t]),
                                            "gaps": [float(x) for x in donors[int(donor_idx[t])]]}
                                  for t in range(vr.N_TEXTS)}}
    with open(out / "m3_donors.json", "x", encoding="utf-8", newline="\n") as f:
        json.dump(don, f)
    runs = (json.dumps(items, indent=1) + "\n").encode("utf-8")
    files = sorted(p for p in out.rglob("*") if p.is_file())
    manifest = {"n_items": len(items),
                "items_per_series": {s_: sum(it["name"].split("/")[0] == s_ for it in items)
                                     for s_ in dict.fromkeys(it["name"].split("/")[0] for it in items)},
                "items_per_block": {str(b): sum(it["block"] == b for it in items) for b in range(N_BLOCKS + 1)},
                "replaced_keys_H": replaced, "l0_prompts": [f"P{k + 1}" for k in l0],
                "models": {"8B_class": m8, "14B_class": m14},
                "files_sha256": {**{p.relative_to(out).as_posix(): sha256(p) for p in files},
                                 "runs.json": hashlib.sha256(runs).hexdigest()}}
    return manifest, runs


def publish(out: Path, manifest: dict, runs: bytes) -> None:
    """manifest.json, then runs.json last: a plan folder without runs.json cannot be run, and one with runs.json has
    its manifest (an interrupted make leaves no runnable half-plan, even when the clean-up itself cannot run)."""
    with open(out / "manifest.json", "x", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=1)
    with open(out / "runs.json", "xb") as f:
        f.write(runs)


def persons_record(ps: "vr.Persons") -> dict:
    """The manifest's persons (read back by prototype_v7_run.read_plan): SHA-256 of sorted ids, never an id."""
    return {"pool_sha256": vr.sha_ids(ps.pool), "replay_H_sha256": vr.sha_ids(ps.h), "ids_looked_at": ps.looked,
            "E": len(ps.e), "v6_200_match": True}


def pool_v6_sha(c: dict) -> str:
    """persons.pool names v6's 200 by 'ids_sha256 <hex>'."""
    import re
    m = re.search(r"ids_sha256 ([0-9a-f]{64})", c["persons"]["pool"])
    if m is None:
        raise PlanStop("persons.pool does not name v6's ids_sha256")
    return m.group(1)


def git_clean_commit() -> str:
    """The plan is made from a committed state only (its code and the conditions are named by the commit)."""
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True,
                           text=True, check=True).stdout.strip()   # untracked files too (ignored ones are not listed)
    if dirty:
        raise PlanStop("the repository has uncommitted changes: commit or ask before making the plan")
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()


def main_make(out: Path) -> None:
    commit = git_clean_commit()
    raw, c = vr.cm.load_conditions(vr.COND)
    if "environment" not in c:
        raise PlanStop("the conditions hold no environment items yet")
    raw6, c6 = vr.cm.load_conditions(vr.COND_V6)
    kp = c["data"]["kupa"]
    files = {"essay": ("external/kupa-keys/KUPA-KEYS-TASK-2.csv", vr.kp_sha(kp, "essay")),
             "meta": ("external/kupa-keys/KUPA-KEYS-META.csv", vr.kp_sha(kp, "meta")),
             "copy": (c6["data"]["kupa"]["files"]["copy"]["file"], c6["data"]["kupa"]["files"]["copy"]["sha256"])}
    for name, (f, sha) in files.items():
        if not (ROOT / f).exists():
            raise PlanStop(f"{f} is missing (download KUPA-KEYS again, data.kupa)")
        if vr.cm.sha256(ROOT / f) != sha:
            raise PlanStop(f"{f} does not match the SHA-256 of the conditions ({name}): the run stops (data.kupa)")
    essay, seen = v6.read_kupa(ROOT / files["essay"][0])
    copy_, _ = v6.read_kupa(ROOT / files["copy"][0])
    ps = vr.draw_persons(essay, copy_, seen, pool_v6_sha(c))
    keyed = read_kupa_keys(ROOT / files["essay"][0], set(ps.h))
    h_series = []
    for p in ps.h:
        t, k = keyed[p]
        check_replay(t, replay_keys(k)[0], essay[p])
        h_series.append((t, k))
    e_gaps = {i: v7.gaps_of(essay[p]["times"]) for i, p in enumerate(ps.e)}
    prompts = all_prompts(c, kupa_prompts(ROOT / files["meta"][0]))
    if out.exists():
        raise PlanStop(f"{out} exists; a plan is never written over")
    try:
        man, runs = build(c, h_series, e_gaps, len(ps.e), prompts, out)
        man.update({"conditions_sha256": vr.conditions_sha256(raw), "git_commit": commit, "git_dirty": False,
                    "data_sha256": {n: s_ for n, (_, s_) in files.items()}, "persons": persons_record(ps),
                    "plan_code_sha256": sha256(Path(__file__))})
        publish(out, man, runs)
    except BaseException:
        shutil.rmtree(out, ignore_errors=True)   # no half-written plan (or one without its manifest) is left
        raise
    print(json.dumps({k: man[k] for k in ("n_items", "items_per_block", "replaced_keys_H", "l0_prompts")}))


# ---------------------------------------------------------------- derive (run time)


def m1_plan(m1: Path) -> list[str]:
    """The keys of the plan an M1 capture left, read as prototype_v7_run (load_capture and check_plans) reads it: the
    capture is 'captured' when its meta.json status is done, else 'failed'; a failed one with no schedule.ndjson, a
    plan of fewer than 2 presses or a token stream too short to recompute has no plan (NoPlan); a plan that is not
    the executable plan recomputed from llm.ndjson is a fault (PlanStop), as it stops the analysis."""
    m1 = Path(m1)
    meta = json.loads((m1 / "meta.json").read_text(encoding="utf-8")) if (m1 / "meta.json").exists() else {}
    failed = meta.get("status") != "done"
    if not (m1 / "schedule.ndjson").exists():
        if failed:
            raise NoPlan(f"{m1} left no schedule.ndjson")
        raise PlanStop(f"{m1} is done but holds no schedule.ndjson")
    plan = vr.read_events(m1 / "schedule.ndjson")
    if failed and plan.times.size < 2:   # load_capture: a failed plan of fewer than 2 presses is no plan
        raise NoPlan(f"{m1}: the failed capture's plan holds fewer than 2 presses")
    try:
        arr, toks, end, _ = vr.read_llm_stream(m1 / "llm.ndjson", unfinished=failed)
        _, _, ex, keys = vr.m1_schedule(arr, toks, end)
    except (vr.RunStop, ValueError) as e:
        if failed:
            raise NoPlan(f"{m1}: the token stream cannot be recomputed ({e})")
        raise PlanStop(f"{m1} is done but its token stream cannot be recomputed: {e}")
    if not vr.same_times(plan.times, ex) or plan.keys != keys:
        raise PlanStop(f"{m1}: schedule.ndjson is not the executable plan of its token stream (B13)")
    return plan.keys


def derive(series: str, text: int, m1: Path, plan: Path) -> list[dict]:
    """M0, M3a or M3b of text `text` from M1_8B's executable plan (amendment B16; check_plans reads them back)."""
    if not 0 <= text < vr.N_TEXTS:
        raise PlanStop(f"text must be in 0..{vr.N_TEXTS - 1}")
    keys = m1_plan(m1)
    if len(keys) < 2:
        raise NoPlan(f"{m1}: M1_8B's plan holds fewer than 2 presses: no text for M0 and M3")
    if series == "M0":
        return ev.make([i * vr.M0_GAP_MS for i in range(len(keys))], keys)
    if series not in ("M3a", "M3b"):
        raise PlanStop(f"unknown series {series}")
    don = json.loads((Path(plan) / "m3_donors.json").read_text(encoding="utf-8"))
    d = don["donors"][str(text)]
    gaps: list = [None] * int(don["n_e"])
    gaps[int(d["index"])] = np.asarray(d["gaps"], dtype=float)
    drawn = v7.m3(text, len(keys) - 1, gaps)
    if drawn["donor"] != int(d["index"]):
        raise PlanStop("m3_donors.json does not hold the donor m3() draws")
    return ev.from_gaps(keys, np.concatenate([[0.0], drawn[series]]))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python works/prototype_v7_plan.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    mk = sub.add_parser("make")
    mk.add_argument("--out", type=Path, required=True)
    dv = sub.add_parser("derive")
    dv.add_argument("--series", required=True, choices=["M0", "M3a", "M3b"])
    dv.add_argument("--text", type=int, required=True)
    dv.add_argument("--m1", type=Path, required=True)
    dv.add_argument("--plan", type=Path, required=True)
    dv.add_argument("--out", type=Path, required=True)
    gt = sub.add_parser("gate-m1")
    gt.add_argument("--m1", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "make":
        main_make(a.out)
        return
    if a.cmd == "gate-m1":
        m1_plan(a.m1)
        return
    presses = derive(a.series, a.text, a.m1, a.plan)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    ev.write(a.out, presses, a.series, f"derived from {a.m1}/schedule.ndjson (amendment B16)", text=a.text)


if __name__ == "__main__":
    main()
