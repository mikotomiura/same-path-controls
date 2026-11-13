"""Prototype v7r (post hoc, descriptive; run once on the work PC with the production captures).

Conditions: works/conditions/prototype_v7r.json. For M2's sessions in v7q's groups, the size of the planned and
recorded gaps. Imports v7's analysis code (a6fa5ca) and v7q (b40bfb8) unchanged. Writes counts and summaries only.

Usage (work PC): python works/prototype_v7r.py --captures C:\\work\\v7-captures
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
import prototype_v7 as v7  # noqa: E402
import prototype_v7_run as vr  # noqa: E402

COND = ROOT / "works" / "conditions" / "prototype_v7r.json"
OUT = ROOT / "results" / "prototype_v7r" / "record.json"
BASE_COMMIT = "a6fa5cae45579c2b4cc95ccb938eab6ea622310e"
V7Q_COMMIT = "b40bfb88d05dea40a6a192201d77ea195e7fda50"
CODE = ["works/prototype_v7_run.py", "works/prototype_v7.py", "works/prototype_v6.py", "works/_common.py",
        "works/prototype_v7q.py"]
LIMITS_MS = (1.0, 2.0, 5.0, 10.0)


def _delta(g):
    return v7.delta(g) if g is not None and vr._condrey_scorable(g) else None


def group_of(cap) -> str:
    """v7q's m2_groups rule: plan not judged / stays human / page not judged / flips / stays machine."""
    rp = vr.recorded(cap, "page")
    d_plan, d_page = _delta(vr.plan_gaps(cap)), _delta(rp.gaps if rp else None)
    if d_plan is None:
        return "plan not judged"
    if d_plan > v7.DELTA_T:
        return "stays human"
    if d_page is None:
        return "page not judged"
    return "flips" if d_page > v7.DELTA_T else "stays machine"


def sizes(cap) -> dict:
    pg = vr.plan_gaps(cap)
    pos = pg[pg > 0] if pg is not None else np.array([])
    med = float(np.median(pos)) if pos.size else None
    rh, rp = vr.recorded(cap, "hook"), vr.recorded(cap, "page")
    return {"plan_median_positive_gap_ms": med,
            "plan_largest_gap_ms": float(pg.max()) if pg is not None and pg.size else None,
            "plan_share_at_median": (float(np.mean(np.abs(pg - med) <= 1e-9)) if med is not None else None),
            "hook_median_gap_ms": float(np.median(rh.gaps)) if rh and rh.gaps.size else None,
            "page_median_gap_ms": float(np.median(rp.gaps)) if rp and rp.gaps.size else None}


def m2_sizes(caps: list) -> dict:
    vals: dict = {}
    for cap in caps:
        if cap.status != "captured":
            continue
        g = vals.setdefault(group_of(cap), [])
        g.append(sizes(cap))
    out = {}
    for grp, rows in vals.items():
        med = [r["plan_median_positive_gap_ms"] for r in rows if r["plan_median_positive_gap_ms"] is not None]
        out[grp] = {"sessions": len(rows),
                    **{k: vr.summary([r[k] for r in rows]) for k in rows[0]},
                    "plan_median_positive_gap_min_max_ms": [min(med), max(med)] if med else None,
                    "sessions_with_median_planned_gap_below_ms": {str(lim): int(sum(m < lim for m in med))
                                                                  for lim in LIMITS_MS}}
    return out


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2 or argv[0] != "--captures":
        raise SystemExit("usage: python works/prototype_v7r.py --captures <folder of the production captures>")
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; refusing to overwrite")
    paths = ["works/conditions/prototype_v7r.json", "works/prototype_v7r.py", *CODE, "works/conditions/prototype_v7.json"]
    cm.same_as_head([ROOT / p for p in paths])
    state = cm.git_state(paths)
    if cm.git("diff", "--name-only", BASE_COMMIT, "HEAD", "--", *CODE[:4], "works/conditions/prototype_v7.json"):
        raise SystemExit(f"v7's analysis code or conditions differ from {BASE_COMMIT}")
    if cm.git("diff", "--name-only", V7Q_COMMIT, "HEAD", "--", "works/prototype_v7q.py"):
        raise SystemExit(f"v7q differs from {V7Q_COMMIT}")
    raw = COND.read_bytes()
    caps = vr.load_all(Path(argv[1]))
    rec = {"m2_sizes": {s_: m2_sizes(caps[s_]) for s_ in ("M2_8B", "M2_14B")},
           "provenance": {"run_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), **state,
                          "conditions_sha256": hashlib.sha256(raw).hexdigest(),
                          "code_sha256": {p: cm.sha256(ROOT / p) for p in [*CODE, "works/prototype_v7r.py"]}}}
    rec = vr.to_json(rec)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    if OUT.exists():
        tmp.unlink()
        raise SystemExit(f"{OUT} appeared during the run; refusing to overwrite")
    tmp.rename(OUT)
    print(f"written {OUT} (sha256 {hashlib.sha256(OUT.read_bytes()).hexdigest()})")


if __name__ == "__main__":
    main()
