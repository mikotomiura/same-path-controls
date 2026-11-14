"""Prototype v7p (post hoc, descriptive): the jitter and grid at which v7's detectors flip the 80 replayed persons.

Conditions: works/conditions/prototype_v7p.json. Imports v7's analysis code (a6fa5ca) unchanged: the same persons,
the same training on E_train with v7's seeds, the same scoring and judgments, and v7's sensitivity_variant.
Usage: python works/prototype_v7p.py   (writes results/prototype_v7p/record.json; refuses to overwrite)
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "works"))

import numpy as np  # noqa: E402

import _common as cm  # noqa: E402
import prototype_v6 as v6  # noqa: E402
import prototype_v7 as v7  # noqa: E402
import prototype_v7_run as run7  # noqa: E402

COND = ROOT / "works" / "conditions" / "prototype_v7p.json"
OUT = ROOT / "results" / "prototype_v7p" / "record.json"
KEY = [20261113, 7, 1]


def flips_of(det: str, s0: np.ndarray, s1: np.ndarray) -> dict:
    j0, o0 = run7.judgments(det, s0)
    j1, o1 = run7.judgments(det, s1)
    ok = o0 & o1
    b, c, n = v7.flips(j0[ok], j1[ok])
    d = np.abs(s1[ok] - s0[ok])
    return {"b": int(b), "c": int(c), "n": int(n), "mean_abs_score_change": float(d.mean()) if d.size else None}


def first_reach(rows: list[tuple[float, float]], level: float):
    for x, share in rows:
        if share >= level:
            return x
    return "not reached"


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; refusing to overwrite")
    c = json.loads(COND.read_text())
    _, c7 = cm.load_conditions(run7.COND)
    _, c6 = cm.load_conditions(run7.COND_V6)
    kp = c7["data"]["kupa"]
    files = {"essay": ("external/kupa-keys/KUPA-KEYS-TASK-2.csv", run7.kp_sha(kp, "essay")),
             "copy": (c6["data"]["kupa"]["files"]["copy"]["file"], c6["data"]["kupa"]["files"]["copy"]["sha256"])}
    for name, (f, sha) in files.items():
        if cm.sha256(ROOT / f) != sha:
            raise SystemExit(f"{f} does not match the SHA-256 of the conditions ({name})")
    essay, seen = v6.read_kupa(ROOT / files["essay"][0])
    copy_, _ = v6.read_kupa(ROOT / files["copy"][0])
    v6_sha = c7["persons"]["pool"].split("ids_sha256 ")[1].split(",")[0]
    ps = run7.draw_persons(essay, copy_, seen, v6_sha)
    h0 = [essay[p]["times"] for p in ps.h]
    e_gaps = [v7.gaps_of(essay[p]["times"]) for p in ps.e]
    sc = run7.train([e_gaps[i] for i in ps.e_train])
    s0 = sc.score([v7.gaps_of(t) for t in h0])
    dets = list(run7.DETECTORS)
    ver = c["versions"]
    rec: dict = {"persons": {"replay_H_sha256": run7.sha_ids(ps.h), "n": len(h0), "E_train": int(ps.e_train.size)},
                 "H0_machine": {d: int(run7.judgments(d, s0[d])[0][run7.judgments(d, s0[d])[1]].sum()) for d in dets},
                 "grid": {}, "jitter": {}}
    for g in ver["grid_ms"]:
        s1 = sc.score([v7.gaps_of(v7.sensitivity_variant(t, "grid", float(g))[0]) for t in h0])
        rec["grid"][str(g)] = {d: flips_of(d, s0[d], s1[d]) for d in dets}
    for k, sd in enumerate(ver["jitter_sd_ms"]):
        reps = []
        for r in range(ver["repeats_for_jitter"]):
            gq = [v7.gaps_of(v7.sensitivity_variant(t, "jitter", float(sd),
                                                    np.random.default_rng([*KEY, k, r, i]))[0]) for i, t in enumerate(h0)]
            s1 = sc.score(gq)
            reps.append({d: flips_of(d, s0[d], s1[d]) for d in dets})
        out = {}
        for d in dets:
            sh = [(x[d]["b"] + x[d]["c"]) / x[d]["n"] for x in reps if x[d]["n"]]
            mc = [x[d]["mean_abs_score_change"] for x in reps if x[d]["mean_abs_score_change"] is not None]
            out[d] = {"flip_share_mean": float(np.mean(sh)), "flip_share_min": float(min(sh)),
                      "flip_share_max": float(max(sh)), "mean_abs_score_change_mean": float(np.mean(mc)),
                      "n": reps[0][d]["n"]}
        rec["jitter"][str(sd)] = out
    summary = {}
    for d in dets:
        jr = [(float(sd), rec["jitter"][str(sd)][d]["flip_share_mean"]) for sd in ver["jitter_sd_ms"]]
        gr = [(float(g), (rec["grid"][str(g)][d]["b"] + rec["grid"][str(g)][d]["c"]) / rec["grid"][str(g)][d]["n"])
              for g in ver["grid_ms"]]
        summary[d] = {"jitter_sd_ms_at_3_of_80": first_reach(jr, 3 / 80), "jitter_sd_ms_at_0.10": first_reach(jr, 0.10),
                      "grid_ms_at_3_of_80": first_reach(gr, 3 / 80), "grid_ms_at_0.10": first_reach(gr, 0.10)}
    rec["summary"] = summary
    paths = ["works/conditions/prototype_v7p.json", "works/prototype_v7p.py", "works/prototype_v7_run.py",
             "works/prototype_v7.py", "works/prototype_v6.py", "works/_common.py"]
    rec["provenance"] = {"run_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                         **cm.git_state(paths),
                         "conditions_sha256": hashlib.sha256(COND.read_bytes()).hexdigest()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
