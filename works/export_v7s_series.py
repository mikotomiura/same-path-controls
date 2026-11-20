"""The release of the series of v7s's captures (not an analysis; run once on the work PC with v7's and v7s's captures).

Conditions: works/conditions/export_v7s_series.json. v7s's loading runs unchanged (prototype_v7s at 5508490:
read_runlogs, known_tables, read_series, check_one_firefox), and every capture v7s's load_item returns is recorded:
the kinds and times of the planned presses and of those recorded on the hook and the page; for Firefox's page also the
handler's time (h_ms) as v7u's page_at_handler gives it (ba4bfc8). Keys, text and identifiers are never written.

Usage (work PC):
    python works/export_v7s_series.py --v7-captures C:\\work\\v7-captures --v7s-captures C:\\work\\v7s-captures
"""
from __future__ import annotations

import argparse
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
sys.path.insert(0, str(ROOT))

import _common as cm  # noqa: E402
import export_v7_series as ex7  # noqa: E402
import prototype_v6 as v6  # noqa: E402
import prototype_v7_plan as v7plan  # noqa: E402
import prototype_v7_run as run7  # noqa: E402
import prototype_v7s as s7  # noqa: E402
import prototype_v7u as vu  # noqa: E402

COND = ROOT / "works" / "conditions" / "export_v7s_series.json"
OUT = ROOT / "results" / "export_v7s_series"
V7S_COMMIT = "550849003164b74187cda8a36dd6311d46487bc2"
V7U_COMMIT = "ba4bfc8a153c11d56e3c0f52ee291bf99f883268"
V7T_COMMIT = "502f31e7cfc325debc0d23c520b725db18b6f422"
V7U_CODE = ["works/prototype_v7u.py", "works/conditions/prototype_v7u.json"]
V7S_RECORD = ROOT / "results" / "prototype_v7s" / "record.json"
V7S_RECORD_SHA256 = "9ff1d37a64ef12f603fc41a144ee14e9d084457cb374640000c09603848c27ce"

KUPA_DERIVED = ("F_H1",)
NOT_KUPA_DERIVED = ("K_chrome", "K_chrome_nohook", "MC1", "MC2", "MC3", "MC5", "F_M0", "F_M2_8B", "F_M2_14B", "F_K",
                    "F_K_nohook")
RELEASED = (*KUPA_DERIVED, *NOT_KUPA_DERIVED)
FIREFOX = ("F_K", "F_K_nohook", "F_M0", "F_M2_8B", "F_M2_14B", "F_H1")
KNOWN = ("K_chrome", "K_chrome_nohook", "F_K", "F_K_nohook")   # loaded by known_tables, one item per gap
NOHOOK = ("K_chrome_nohook", "F_K_nohook")
UNIT_FORM = {**{s_: "c{:03d}" for s_ in KNOWN}, **{s_: "t{:02d}" for s_ in (*s7.MC, *s7.F_MACHINE)},
             "F_H1": "h{:02d}"}                                 # the formats of v7s's items (expected_items, unit_caps)
UNIT_RE = {s_: re.compile({"c{:03d}": "c[0-9]{3}", "t{:02d}": "t[0-9]{2}", "h{:02d}": "h[0-9]{2}"}[f])
           for s_, f in UNIT_FORM.items()}
COLUMNS = ("series", "unit", "layer", "i", "kind", "t_ms", "h_ms")
STATUS_COLUMNS = ex7.STATUS_COLUMNS
STATUSES = ex7.STATUSES
T_MS = ex7.T_MS


# ---------------------------------------------------------------- reading as v7s reads


def recorder(real, got: dict):
    """v7s's load_item, unchanged in what it returns; every item it reads is kept in got (name -> (capture, h)), h
    being the page's handler times (v7u's page_at_handler) for a captured Firefox item and None otherwise."""
    def load(root: Path, runlog: dict[str, str], name: str, hook: bool = True) -> run7.Capture:
        cap = real(root, runlog, name, hook=hook)
        if name in got:
            raise SystemExit(f"{name}: read twice by v7s's loading")
        h = None
        if name.split("/")[0] in FIREFOX and cap.status == "captured" and "page" in cap.layers:
            h = vu.page_at_handler(cap).layers["page"].times
        got[name] = (cap, h)
        return cap
    return load


def h0_of() -> list[s7.Sess]:
    """H0 as v7s's main makes it (and v7u's persons_and_scorer), without the scorer: the persons' times and kinds.
    KUPA's keys are read only for their kinds; nothing of them is kept."""
    _, c7 = cm.load_conditions(run7.COND)
    _, c6 = cm.load_conditions(run7.COND_V6)
    kp = c7["data"]["kupa"]
    files = {"essay": ("external/kupa-keys/KUPA-KEYS-TASK-2.csv", run7.kp_sha(kp, "essay")),
             "copy": (c6["data"]["kupa"]["files"]["copy"]["file"], c6["data"]["kupa"]["files"]["copy"]["sha256"])}
    for name, (f, sha) in files.items():
        if cm.sha256(ROOT / f) != sha:
            raise s7.RunStop(f"{f} does not match the SHA-256 of the conditions ({name})")
    essay, seen = v6.read_kupa(ROOT / files["essay"][0])
    copy_, _ = v6.read_kupa(ROOT / files["copy"][0])
    v6_sha = c7["persons"]["pool"].split("ids_sha256 ")[1].split(",")[0]
    ps = run7.draw_persons(essay, copy_, seen, v6_sha)
    keyed = v7plan.read_kupa_keys(ROOT / files["essay"][0], set(ps.h))

    def sess_of(pid: str) -> s7.Sess:
        t, k = keyed[pid]
        if not run7.same_times(t, essay[pid]["times"]):
            raise s7.RunStop("the keys' reading of KUPA does not give the persons' times")
        return s7.Sess(t, s7.kinds_of_keys(k))

    return [sess_of(p) for p in ps.h]


def load(v7_root: Path, v7s_root: Path, h0: list[s7.Sess]) -> tuple[dict, dict[str, str], dict[str, list[str]]]:
    """v7s's main from v7's captures and v7s's runlogs to the series and the Firefox version check; returns the items
    read (name -> (capture, h)), the runlog and the statuses read_series gives per unit."""
    v7caps = run7.load_all(v7_root)
    got: dict = {}
    real = s7.load_item
    s7.load_item = recorder(real, got)
    try:
        runlog = s7.read_runlogs(v7s_root)
        meas, _ = s7.known_tables(v7caps, v7s_root, runlog)
        kinds: dict = {}
        series = s7.read_series(v7caps, v7s_root, runlog, h0, kinds)
    finally:
        s7.load_item = real
    s7.check_one_firefox(meas.get("firefox_version_known_gaps"), kinds.get("firefox_version"))
    if set(got) != set(runlog):
        raise SystemExit(f"v7s's loading read {len(set(got) - set(runlog))} items outside the runlogs and left "
                         f"{len(set(runlog) - set(got))} of them unread")
    return got, runlog, {s_: list(series[s_].status) for s_ in RELEASED if s_ not in KNOWN}


# ---------------------------------------------------------------- the files


def check_capture(name: str, cap: run7.Capture, h) -> None:
    """The status and layers v7s's load_item gives (fail-closed if it ever gives something else)."""
    s_ = name.split("/")[0]
    if cap.status not in STATUSES:
        raise SystemExit(f"{name}: status {cap.status!r}")
    want = set() if cap.status != "captured" else {"page"} if s_ in NOHOOK else {"hook", "page"}
    if set(cap.layers) != want:
        raise SystemExit(f"{name}: layers {sorted(cap.layers)} for status {cap.status}")
    if cap.status == "no_plan" and cap.plan is not None:
        raise SystemExit(f"{name}: no_plan with a plan")
    if (h is not None) != (s_ in FIREFOX and cap.status == "captured"):
        raise SystemExit(f"{name}: handler times {'present' if h is not None else 'missing'}")
    if h is not None and len(h) != len(cap.layers["page"].times):
        raise SystemExit(f"{name}: {len(h)} handler times for {len(cap.layers['page'].times)} page presses")


def fmt(t) -> str:
    if not math.isfinite(float(t)):
        raise SystemExit("a time is not finite")
    return f"{float(t):.6f}"


def rows_of(name: str, cap: run7.Capture, h) -> list[tuple]:
    """The rows of one item: plan, hook and page, kinds and times only (Plan.keys is never read here)."""
    s_, u = name.split("/")
    layers = []
    if cap.plan is not None:
        layers.append(("plan", cap.plan.times, cap.plan.kinds, None))
    for la in ("hook", "page"):
        if la in cap.layers:
            layers.append((la, cap.layers[la].times, cap.layers[la].kinds, h if la == "page" else None))
    out = []
    for la, times, kinds, hs in layers:
        if len(times) != len(kinds):
            raise SystemExit(f"{name} {la}: {len(times)} times but {len(kinds)} kinds")
        for i, (t, k) in enumerate(zip(times, kinds)):
            if k not in run7.COUNTED_KINDS:
                raise SystemExit(f"{name} {la}: kind {k!r} outside the counted kinds")
            out.append((s_, u, la, i, k, fmt(t), "" if hs is None else fmt(hs[i])))
    return out


def items_of(series: str, runlog: dict[str, str], unit_status: dict[str, list[str]]) -> list[tuple[str, str | None]]:
    """(item name, status read_series gave or None) in unit order: a known gap's items from the runlog; the other
    series' units as read_series counts them."""
    if series in KNOWN:
        return [(n, None) for n in sorted(n for n in runlog if n.split("/")[0] == series)]
    return [(f"{series}/{UNIT_FORM[series].format(u)}", st) for u, st in enumerate(unit_status[series])]


def build(got: dict, runlog: dict[str, str], unit_status: dict[str, list[str]]) -> dict[str, bytes]:
    """Relative path -> bytes of every file but the manifest."""
    files, status, used = {}, [], set()
    for group, series in (("kupa-derived", KUPA_DERIVED), ("not-kupa-derived", NOT_KUPA_DERIVED)):
        for s_ in series:
            rows = []
            items = items_of(s_, runlog, unit_status)
            if not items:
                raise SystemExit(f"{s_}: no item")
            for name, st in items:
                u = name.split("/")[1]
                if name in got:
                    cap, h = got[name]
                    if st is not None and st != cap.status:
                        raise SystemExit(f"{name}: read_series counts {st}, the item is {cap.status}")
                    used.add(name)
                else:
                    if st != "no_plan" or name in runlog:
                        raise SystemExit(f"{name}: not read but counted {st}")
                    cap, h = run7.Capture(s_, 0, "no_plan"), None
                check_capture(name, cap, h)
                r = rows_of(name, cap, h)
                has_plan = "yes" if any(x[2] == "plan" for x in r) else "no"
                status.append((s_, u, cap.status, "true" if cap.redone else "false", has_plan))
                rows.extend(r)
            files[f"{group}/{s_}.csv"] = ex7.csv_bytes(rows, COLUMNS)
    if used != set(got):
        raise SystemExit(f"items read but not released: {sorted(set(got) - used)[:5]}")
    files["status.csv"] = ex7.csv_bytes(status, STATUS_COLUMNS)
    check_files(files)
    return files


def check_files(files: dict[str, bytes]) -> None:
    """Every cell of every file is one of the allowed values or forms (an allow-list, not a search for what must not
    be there): series names of the release, unit forms of v7s's items, layers, integers, kinds, times, and h_ms only
    on a Firefox page row (and there always)."""
    for name, b in files.items():
        rows = list(csv.reader(io.StringIO(b.decode("utf-8"))))
        head, body = tuple(rows[0]), rows[1:]
        if name == "status.csv":
            ok = head == STATUS_COLUMNS and all(
                len(r) == 5 and r[0] in RELEASED and UNIT_RE[r[0]].fullmatch(r[1]) and r[2] in STATUSES
                and r[3] in ("true", "false") and r[4] in ("yes", "no") for r in body)
        else:
            s_ = name.split("/")[-1].removesuffix(".csv")
            want_dir = "kupa-derived" if s_ in KUPA_DERIVED else "not-kupa-derived"
            ok = name == f"{want_dir}/{s_}.csv" and s_ in RELEASED and head == COLUMNS and all(
                len(r) == 7 and r[0] == s_ and UNIT_RE[s_].fullmatch(r[1]) and r[2] in ("plan", "hook", "page")
                and r[3].isdigit() and r[4] in run7.COUNTED_KINDS and T_MS.fullmatch(r[5])
                and (T_MS.fullmatch(r[6]) if (s_ in FIREFOX and r[2] == "page") else r[6] == "") for r in body)
        if not ok:
            raise SystemExit(f"{name}: a cell outside the allowed values or forms")


# ---------------------------------------------------------------- the run


def base_check(c: dict) -> None:
    b, se = c["base"], c["series"]
    if (b["v7s_commit"], b["v7u_commit"], b["v7t_commit"], b["v7s_record_sha256"]) != \
            (V7S_COMMIT, V7U_COMMIT, V7T_COMMIT, V7S_RECORD_SHA256):
        raise SystemExit("the conditions' base is not the code's")
    if (tuple(se["kupa_derived"]), tuple(se["not_kupa_derived"]), tuple(se["firefox"])) != \
            (KUPA_DERIVED, NOT_KUPA_DERIVED, FIREFOX):
        raise SystemExit("the conditions' series are not the code's")
    if (vu.V7S_COMMIT, vu.V7T_COMMIT, vu.V7S_RECORD_SHA256, vu.FIREFOX) != \
            (V7S_COMMIT, V7T_COMMIT, V7S_RECORD_SHA256, FIREFOX):
        raise SystemExit("v7u's pins are not this export's")


def code_check() -> None:
    """v7u's files are the bytes of ba4bfc8; v7u's own check pins v7s's code (5508490) and v7t's (502f31e); v7s's
    pins v7's (a6fa5ca)."""
    for f in V7U_CODE:
        if cm.git("rev-parse", f"{V7U_COMMIT}:{f}") != cm.git("hash-object", "--no-filters", str(ROOT / f)):
            raise SystemExit(f"{f} differs from {V7U_COMMIT[:7]}")
    vu.code_check()
    s7.code_check()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v7-captures", required=True, type=Path)
    ap.add_argument("--v7s-captures", required=True, type=Path)
    a = ap.parse_args(argv)
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; refusing to overwrite")
    paths = ["works/conditions/export_v7s_series.json", "works/export_v7s_series.py", "works/export_v7_series.py",
             *V7U_CODE, *vu.V7S_CODE, *vu.V7T_CODE]
    cm.same_as_head([ROOT / p for p in paths])
    state = cm.git_state(paths)
    raw = COND.read_bytes()
    base_check(json.loads(raw))
    code_check()
    s7.check_constants(json.loads(s7.COND.read_text(encoding="utf-8")))   # as v7s's main does
    if not V7S_RECORD.exists() or cm.sha256(V7S_RECORD) != V7S_RECORD_SHA256:
        raise SystemExit(f"{V7S_RECORD} is not v7s's record (its SHA-256)")
    vu.runlogs_as_v7s(a.v7s_captures, json.loads(V7S_RECORD.read_text(encoding="utf-8")))
    h0 = h0_of()
    got, runlog, unit_status = load(a.v7_captures, a.v7s_captures, h0)
    files = build(got, runlog, unit_status)
    manifest = {"files": ex7.manifest_entries(files),
                "provenance": {"run_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), **state,
                               "conditions_sha256": hashlib.sha256(raw).hexdigest(),
                               "code_sha256": {p: cm.sha256(ROOT / p) for p in paths[1:]},
                               "v7s_record_sha256": V7S_RECORD_SHA256,
                               "v7s_runlogs_sha256": {f"runlog.b{b}.ndjson": cm.sha256(a.v7s_captures /
                                                                                     f"runlog.b{b}.ndjson")
                                                      for b in range(1, s7.N_BLOCKS + 1)}}}
    ex7.write_all(OUT, files, manifest)
    for name, e in manifest["files"].items():
        print(f"{e['sha256']}  {name}  rows {e['rows']}")
    print(f"written {OUT}")


if __name__ == "__main__":
    main()
