"""The release of paper B's series (not an analysis; run once on the work PC with the production captures).

Conditions: works/conditions/export_v7_series.json. For every released series, the kinds and times of the planned
presses and of those recorded on the hook and the page, as v7's analysis reads them (prototype_v7_run.load_all at
a6fa5ca, unchanged). Keys, text and identifiers are never written; L0 and L1 (the first author's typing) are not
written.

Usage (work PC): python works/export_v7_series.py --captures C:\\work\\v7-captures
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import io
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "works"))

import _common as cm  # noqa: E402
import prototype_v7_run as vr  # noqa: E402

COND = ROOT / "works" / "conditions" / "export_v7_series.json"
OUT = ROOT / "results" / "export_v7_series"
BASE_COMMIT = "a6fa5cae45579c2b4cc95ccb938eab6ea622310e"
# uv.lock: a6fa5ca's, or 7d02179's (v7s's conditions: LightGBM 4.7.0 and nothing else added, no line removed or changed;
# no module v7's analysis imports changes). Amended before any capture was read (2026-11-20).
LOCK_COMMITS = (BASE_COMMIT, "7d02179ac8eba2429a337045029df17be8cf0aaf")


def lock_check() -> None:
    have = cm.git("hash-object", "--no-filters", str(ROOT / "uv.lock"))
    if have not in {cm.git("rev-parse", f"{c}:uv.lock") for c in LOCK_COMMITS}:
        raise SystemExit(f"uv.lock is neither {LOCK_COMMITS[0][:7]}'s nor {LOCK_COMMITS[1][:7]}'s")
CODE = ["works/prototype_v7_run.py", "works/prototype_v7.py", "works/prototype_v6.py", "works/_common.py"]
KUPA_DERIVED = ("H1_idle", "H1_load", "H1_rep", "M3a", "M3b")
NOT_KUPA_DERIVED = ("M0", "M1_8B", "M1_14B", "M1_replay_8B", "M1_replay_14B", "M2_8B", "M2_14B",
                    "P_idle", "P_load", "P_nohook")
NOT_RELEASED = ("L0", "L1_idle", "L1_load")
COLUMNS = ("series", "unit", "layer", "i", "kind", "t_ms")
STATUS_COLUMNS = ("series", "unit", "status", "redone", "plan")
STATUSES = ("captured", "failed", "no_plan")
T_MS = re.compile(r"-?[0-9]+\.[0-9]{6}")


def unit_name(series: str, unit: int) -> str:
    return vr.CAPTURE_LAYOUT[series].format(i=unit)


def plan_as_analysis(cap):
    """The plan as v7's analysis keeps it: check_plans sets a failed M1 capture's plan to None when its schedule
    cannot be recomputed from llm.ndjson (counted as failed_without_plan); the same rule is applied here."""
    if cap.plan is None or cap.status != "failed" or cap.series not in ("M1_8B", "M1_14B"):
        return cap.plan
    try:
        arr, toks, end, _ = vr.read_llm_stream(cap.folder / "llm.ndjson", unfinished=True)
        vr.m1_schedule(arr, toks, end)
    except (vr.RunStop, ValueError):
        return None
    return cap.plan


def check_capture(cap) -> None:
    """The status and layers load_all gives (fail-closed if v7's reader ever gives something else)."""
    if cap.status not in STATUSES:
        raise SystemExit(f"{cap.series}/{cap.unit}: status {cap.status!r}")
    want = (set() if cap.status != "captured" else {"page"} if cap.series == "P_nohook" else {"hook", "page"})
    if set(cap.layers) != want:
        raise SystemExit(f"{cap.series}/{cap.unit}: layers {sorted(cap.layers)} for status {cap.status}")
    if cap.status == "no_plan" and cap.plan is not None:
        raise SystemExit(f"{cap.series}/{cap.unit}: no_plan with a plan")


def rows_of(cap) -> list[tuple]:
    """The rows of one capture: plan, hook and page, kinds and times only (Plan.keys is never read here)."""
    u = unit_name(cap.series, cap.unit)
    out = []
    layers = []
    plan = plan_as_analysis(cap)
    if plan is not None:
        layers.append(("plan", plan.times, plan.kinds))
    for la in ("hook", "page"):
        if la in cap.layers:
            layers.append((la, cap.layers[la].times, cap.layers[la].kinds))
    for la, times, kinds in layers:
        if len(times) != len(kinds):
            raise SystemExit(f"{cap.series}/{u} {la}: {len(times)} times but {len(kinds)} kinds")
        for i, (t, k) in enumerate(zip(times, kinds)):
            if k not in vr.COUNTED_KINDS:
                raise SystemExit(f"{cap.series}/{u} {la}: kind {k!r} outside the counted kinds")
            if not math.isfinite(float(t)):
                raise SystemExit(f"{cap.series}/{u} {la}: a time is not finite")
            out.append((cap.series, u, la, i, k, f"{float(t):.6f}"))
    return out


def csv_bytes(rows: list[tuple], columns: tuple[str, ...]) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(columns)
    w.writerows(rows)
    return buf.getvalue().encode("utf-8")


def build(caps: dict) -> dict[str, bytes]:
    """Relative path -> bytes of every file but the manifest."""
    missing = [s_ for s_ in (*KUPA_DERIVED, *NOT_KUPA_DERIVED) if s_ not in caps]
    if missing:
        raise SystemExit(f"series missing from the captures: {missing}")
    files, status = {}, []
    for group, series in (("kupa-derived", KUPA_DERIVED), ("not-kupa-derived", NOT_KUPA_DERIVED)):
        for s_ in series:
            rows = []
            for cap in caps[s_]:
                check_capture(cap)
                r = rows_of(cap)
                has_plan = "yes" if any(x[2] == "plan" for x in r) else "no"
                status.append((s_, unit_name(s_, cap.unit), cap.status, "true" if cap.redone else "false", has_plan))
                rows.extend(r)
            files[f"{group}/{s_}.csv"] = csv_bytes(rows, COLUMNS)
    files["status.csv"] = csv_bytes(status, STATUS_COLUMNS)
    check_files(files)
    return files


def check_files(files: dict[str, bytes]) -> None:
    """Every cell of every file is one of the allowed values or forms (an allow-list, not a search for what must not
    be there): series names of the release, unit names of the layout, layers, integers, kinds, times."""
    released = (*KUPA_DERIVED, *NOT_KUPA_DERIVED)
    units = {s_: re.compile(re.escape(vr.CAPTURE_LAYOUT[s_]).replace(r"\{i:02d\}", "[0-9]{2}")
                            .replace(r"\{i:03d\}", "[0-9]{3}")) for s_ in released}
    for name, b in files.items():
        rows = list(csv.reader(io.StringIO(b.decode("utf-8"))))
        head, body = tuple(rows[0]), rows[1:]
        if name == "status.csv":
            ok = head == STATUS_COLUMNS and all(
                len(r) == 5 and r[0] in released and units[r[0]].fullmatch(r[1]) and r[2] in STATUSES
                and r[3] in ("true", "false") and r[4] in ("yes", "no") for r in body)
        else:
            s_ = name.split("/")[-1].removesuffix(".csv")
            want_dir = "kupa-derived" if s_ in KUPA_DERIVED else "not-kupa-derived"
            ok = name == f"{want_dir}/{s_}.csv" and s_ in released and head == COLUMNS and all(
                len(r) == 6 and r[0] == s_ and units[s_].fullmatch(r[1]) and r[2] in ("plan", "hook", "page")
                and r[3].isdigit() and r[4] in vr.COUNTED_KINDS and T_MS.fullmatch(r[5]) for r in body)
        if not ok:
            raise SystemExit(f"{name}: a cell outside the allowed values or forms")


def manifest_entries(files: dict[str, bytes]) -> dict:
    out = {}
    for name, b in sorted(files.items()):
        rows = list(csv.reader(io.StringIO(b.decode("utf-8"))))[1:]
        entry = {"sha256": hashlib.sha256(b).hexdigest(), "rows": len(rows)}
        if name != "status.csv":
            entry["presses"] = {la: sum(r[2] == la for r in rows) for la in ("plan", "hook", "page")}
        out[name] = entry
    return out


def write_all(out: Path, files: dict[str, bytes], manifest: dict) -> None:
    """Into a temporary folder first, renamed to out at the end; refuses to overwrite."""
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    tmp = out.with_name(out.name + ".tmp")
    try:
        tmp.mkdir(parents=True, exist_ok=False)   # taken exclusively: a second run stops here
    except FileExistsError:
        raise SystemExit(f"{tmp} exists (a run stopped half-way?); move it aside first") from None
    for name, b in files.items():
        p = tmp / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b)
    (tmp / "manifest.json").write_text(json.dumps(manifest, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    if out.exists():
        raise SystemExit(f"{out} appeared during the run; refusing to overwrite")
    tmp.rename(out)


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2 or argv[0] != "--captures":
        raise SystemExit("usage: python works/export_v7_series.py --captures <folder of the production captures>")
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; refusing to overwrite")
    paths = ["works/conditions/export_v7_series.json", "works/export_v7_series.py",
             *dict.fromkeys([*CODE, *vr.CODE]), "works/conditions/prototype_v7.json"]
    cm.same_as_head([ROOT / p for p in paths])
    state = cm.git_state(paths)
    if cm.git("diff", "--name-only", BASE_COMMIT, "HEAD", "--", *[p for p in vr.CODE if p != "uv.lock"],
              "works/conditions/prototype_v7.json"):
        raise SystemExit(f"v7's analysis code or conditions differ from {BASE_COMMIT}")
    lock_check()
    raw = COND.read_bytes()
    if cm.sha256(vr.COND) != json.loads(raw)["base"]["v7_conditions_sha256"]:
        raise SystemExit("v7's conditions are not the bytes this export is based on")
    caps = vr.load_all(Path(argv[1]))
    files = build(caps)
    manifest = {"files": manifest_entries(files),
                "provenance": {"run_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), **state,
                               "conditions_sha256": hashlib.sha256(raw).hexdigest(),
                               "code_sha256": {p: cm.sha256(ROOT / p)
                                               for p in [*dict.fromkeys([*CODE, *vr.CODE]), "works/export_v7_series.py"]}}}
    write_all(OUT, files, manifest)
    for name, e in manifest["files"].items():
        print(f"{e['sha256']}  {name}  rows {e['rows']}")
    print(f"written {OUT}")


if __name__ == "__main__":
    main()
