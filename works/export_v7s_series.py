"""The release of the series of v7s's captures (not an analysis; run once on the work PC with v7's and v7s's captures).

Conditions: works/conditions/export_v7s_series.json. v7s's loading runs unchanged (prototype_v7s at 5508490:
read_runlogs, known_tables, read_series, check_one_firefox), and every capture v7s's load_item returns is recorded:
the kinds and times of the planned presses and of those recorded on the hook and the page; for Firefox's page also the
handler's time (h_ms) as v7u's page_at_handler gives it (ba4bfc8). Keys, text and identifiers are never written.

Usage (work PC):
    uv run --offline --frozen python works/export_v7s_series.py --v7-captures C:\\work\\v7-captures --v7s-captures C:\\work\\v7s-captures
After the return (Mac):
    uv run --offline --frozen python works/export_v7s_series.py --verify <the returned export_v7s_series folder>
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
# read on the way to H0 (draw_persons, v6's Series) or by the environment, outside v7u's and v7s's own checks: the
# bytes v7s ran with (5508490)
V7S_MORE = ["src/keystroke.py", "works/prototype_v4.py", "works/prototype_v5.py", "works/conditions/prototype_v7.json",
            "works/conditions/prototype_v6.json", "pyproject.toml", "uv.lock"]
# every module the run imports from the repo, and the file it must have been loaded from
MODULES = {"_common": "works/_common.py", "export_v7_series": "works/export_v7_series.py",
           "prototype_v4": "works/prototype_v4.py", "prototype_v5": "works/prototype_v5.py",
           "prototype_v6": "works/prototype_v6.py", "prototype_v7": "works/prototype_v7.py",
           "prototype_v7_plan": "works/prototype_v7_plan.py", "prototype_v7_run": "works/prototype_v7_run.py",
           "prototype_v7s": "works/prototype_v7s.py", "prototype_v7s_plan": "works/prototype_v7s_plan.py",
           "prototype_v7t": "works/prototype_v7t.py", "prototype_v7u": "works/prototype_v7u.py",
           "keystroke": "src/keystroke.py", "capture": "capture/__init__.py", "capture.events": "capture/events.py"}
INDEX = re.compile(r"[0-9]+")
ZERO = ("0.000000", "-0.000000")
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


FILES = frozenset({"status.csv", *(f"kupa-derived/{s_}.csv" for s_ in KUPA_DERIVED),
                   *(f"not-kupa-derived/{s_}.csv" for s_ in NOT_KUPA_DERIVED)})


def cell_ok(s_: str, r: list[str]) -> bool:
    """One row of a series file: every cell an allowed value or form."""
    return (len(r) == 7 and r[0] == s_ and bool(UNIT_RE[s_].fullmatch(r[1])) and r[2] in ("plan", "hook", "page")
            and bool(INDEX.fullmatch(r[3])) and r[4] in run7.COUNTED_KINDS and bool(T_MS.fullmatch(r[5]))
            and bool(T_MS.fullmatch(r[6]) if (s_ in FIREFOX and r[2] == "page") else r[6] == ""))


def check_files(files: dict[str, bytes]) -> None:
    """Every cell of every file is one of the allowed values or forms (an allow-list, not a search for what must not
    be there): series names of the release, unit forms of v7s's items, layers, integers, kinds, times, and h_ms only
    on a Firefox page row (and there always). Then the structure: exactly the release's files, each unit once in
    status.csv, each layer's i = 0, 1, 2, ... with its first time (and h_ms) at 0 (times run from the layer's first
    counted press: an absolute clock reading cannot pass), and the layers a unit has are those its status allows."""
    if set(files) != FILES:
        raise SystemExit(f"the files are not the release's: {sorted(set(files) ^ FILES)}")
    st = list(csv.reader(io.StringIO(files["status.csv"].decode("utf-8"))))
    if tuple(st[0]) != STATUS_COLUMNS or not all(
            len(r) == 5 and r[0] in RELEASED and UNIT_RE[r[0]].fullmatch(r[1]) and r[2] in STATUSES
            and r[3] in ("true", "false") and r[4] in ("yes", "no") for r in st[1:]):
        raise SystemExit("status.csv: a cell outside the allowed values or forms")
    status = {(r[0], r[1]): r for r in st[1:]}
    if len(status) != len(st) - 1:
        raise SystemExit("status.csv: a unit appears twice")
    for name, b in files.items():
        if name == "status.csv":
            continue
        s_ = name.split("/")[-1].removesuffix(".csv")
        rows = list(csv.reader(io.StringIO(b.decode("utf-8"))))
        if tuple(rows[0]) != COLUMNS or not all(cell_ok(s_, r) for r in rows[1:]):
            raise SystemExit(f"{name}: a cell outside the allowed values or forms")
        layers: dict[tuple[str, str], list[list[str]]] = {}
        for r in rows[1:]:
            layers.setdefault((r[1], r[2]), []).append(r)
        for (u, la), rs in layers.items():
            if [int(r[3]) for r in rs] != list(range(len(rs))) or rs[0][5] not in ZERO \
                    or (rs[0][6] != "" and rs[0][6] not in ZERO):
                raise SystemExit(f"{name}: {u}/{la} is not i = 0, 1, ... from a first time of 0")
            srow = status.get((s_, u))
            if srow is None or (la == "plan" and srow[4] != "yes") or (la != "plan" and srow[2] != "captured"):
                raise SystemExit(f"{name}: {u}/{la} is not what status.csv says of the unit")
        for (s2, u), srow in status.items():
            if s2 != s_:
                continue
            if (srow[4] == "yes") != ((u, "plan") in layers):
                raise SystemExit(f"{name}: {u}'s plan rows disagree with status.csv")
            if srow[2] == "captured" and srow[4] == "yes":   # a captured plan was recorded on every layer it has
                need = {"page"} if s_ in NOHOOK else {"hook", "page"}
                if {la for (u2, la) in layers if u2 == u and la != "plan"} != need:
                    raise SystemExit(f"{name}: {u} is captured but its layers are not {sorted(need)}")


def check_counts(status_csv: bytes, record: dict) -> None:
    """From status.csv, so that the work PC (before writing) and the Mac (after the return) check the same: for MC,
    F_M and F_H1 the units per status are v7s's record's counts; for each known gap (not in the record's counts) the
    units are those of v7s's path measures, captured exactly where the measures hold values (not {'status': ...})."""
    rows = list(csv.reader(io.StringIO(status_csv.decode("utf-8"))))[1:]
    meas = record["path"]["measures"]
    for s_ in RELEASED:
        mine = [r for r in rows if r[0] == s_]
        if s_ not in KNOWN:
            want = record["counts"].get(s_)
            have = {"sessions": len(mine), **{x: sum(r[2] == x for r in mine) for x in STATUSES}}
            if want != have:
                raise SystemExit(f"{s_}: {have} is not v7s's record's counts {want}")
            continue
        layers = ("page",) if s_ in NOHOOK else run7.LAYERS
        gaps = {k.split("/")[1] for k in meas if k.split("/")[0] == s_}
        if {str(int(r[1][1:])) for r in mine} != gaps or len(mine) != len(gaps):
            raise SystemExit(f"{s_}: the known gaps are not those of v7s's record's path measures")
        for r in mine:
            for la in layers:
                entry = meas.get(f"{s_}/{int(r[1][1:])}/{la}")
                ok = entry is not None and (("status" not in entry) if r[2] == "captured"
                                            else entry == {"status": r[2]})
                if not ok:
                    raise SystemExit(f"{s_}/{r[1]}/{la}: {r[2]}, not as v7s's record's path measures")


HEX64, HEX40 = re.compile(r"[0-9a-f]{64}"), re.compile(r"[0-9a-f]{40}")


RUNLOGS = frozenset(f"runlog.b{b}.ndjson" for b in range(1, s7.N_BLOCKS + 1))


def code_paths() -> list[str]:
    """Every file the run checks against HEAD (the conditions first, then the code)."""
    return ["works/conditions/export_v7s_series.json", "works/export_v7s_series.py",
            *dict.fromkeys([*MODULES.values(), *V7U_CODE, *vu.V7S_CODE, *vu.V7T_CODE, *V7S_MORE])]


def check_manifest(m: dict) -> None:
    """The manifest's keys and value forms, nested keys included (it holds hashes, counts and the run's time only)."""
    pv = m["provenance"]
    ok = set(m) == {"files", "provenance"} and set(m["files"]) == FILES and all(
        set(e) == ({"sha256", "rows"} if n == "status.csv" else {"sha256", "rows", "presses"})
        and HEX64.fullmatch(e["sha256"]) and isinstance(e["rows"], int)
        and (n == "status.csv" or (set(e["presses"]) == {"plan", "hook", "page"}
                                   and all(isinstance(v, int) for v in e["presses"].values())))
        for n, e in m["files"].items())
    ok = ok and set(pv) == {"run_at", "git_commit", "git_dirty", "conditions_sha256", "code_sha256", "v7s_record_sha256",
                            "v7s_runlogs_sha256"} \
        and re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\+00:00", pv["run_at"]) \
        and HEX40.fullmatch(pv["git_commit"]) and isinstance(pv["git_dirty"], bool) \
        and set(pv["code_sha256"]) == set(code_paths()[1:]) and set(pv["v7s_runlogs_sha256"]) == RUNLOGS \
        and all(HEX64.fullmatch(v) for v in (pv["conditions_sha256"], pv["v7s_record_sha256"],
                                             *pv["code_sha256"].values(), *pv["v7s_runlogs_sha256"].values()))
    if not ok:
        raise SystemExit("manifest.json: a key or value outside the allowed forms")


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
    """v7u's files are the bytes of ba4bfc8 and the files read on the way to H0 those of 5508490; v7u's own check pins
    v7s's code (5508490) and v7t's (502f31e); v7s's pins v7's (a6fa5ca). Every module was loaded from its checked
    file (a same-named file earlier on sys.path would otherwise run unchecked)."""
    for commit, files in ((V7U_COMMIT, V7U_CODE), (V7S_COMMIT, V7S_MORE)):
        for f in files:
            if cm.git("rev-parse", f"{commit}:{f}") != cm.git("hash-object", "--no-filters", str(ROOT / f)):
                raise SystemExit(f"{f} differs from {commit[:7]}")
    vu.code_check()
    s7.code_check()
    for mod, f in MODULES.items():
        m = sys.modules.get(mod)
        if m is None or Path(m.__file__).resolve() != (ROOT / f).resolve():
            raise SystemExit(f"module {mod} was not loaded from {f}")
    root, tests, me = ROOT.resolve(), (ROOT / "tests").resolve(), Path(__file__).resolve()
    for name, m in list(sys.modules.items()):   # nothing else of the repo runs (e.g. an untracked works/numpy.py)
        f = getattr(m, "__file__", None)
        if not f:
            continue
        pf = Path(f).resolve()
        if pf.is_relative_to(root) and not pf.is_relative_to(tests) and pf != me and name not in MODULES \
                and not pf.is_relative_to(root / ".venv"):
            raise SystemExit(f"module {name} was loaded from {pf.relative_to(root)}, which the run does not check")


def read_record() -> dict:
    if not V7S_RECORD.exists() or cm.sha256(V7S_RECORD) != V7S_RECORD_SHA256:
        raise SystemExit(f"{V7S_RECORD} is not v7s's record (its SHA-256)")
    return json.loads(V7S_RECORD.read_text(encoding="utf-8"))


def verify(out: Path, record: dict) -> dict:
    """After the return (the Mac): the files as written are exactly the release's, their SHA-256, rows and presses are
    the manifest's, the manifest's forms, the allow-list and structure, and the counts (the checks of the work PC,
    from the files alone). Returns the manifest."""
    m = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    on_disk = {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()}
    if on_disk != FILES | {"manifest.json"}:
        raise SystemExit(f"{out}: the files are not the release's: {sorted(on_disk ^ (FILES | {'manifest.json'}))}")
    files = {n: (out / n).read_bytes() for n in FILES}
    check_manifest(m)
    if ex7.manifest_entries(files) != m["files"]:
        raise SystemExit(f"{out}: the files' SHA-256, rows or presses are not the manifest's")
    check_files(files)
    check_counts(files["status.csv"], record)
    return m


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v7-captures", type=Path)
    ap.add_argument("--v7s-captures", type=Path)
    ap.add_argument("--verify", type=Path, help="a returned results/export_v7s_series folder (the Mac)")
    a = ap.parse_args(argv)
    if a.verify is not None:
        if a.v7_captures is not None or a.v7s_captures is not None:
            raise SystemExit("--verify takes no captures")
        verify(a.verify, read_record())
        print(f"verified {a.verify}")
        return
    if a.v7_captures is None or a.v7s_captures is None:
        raise SystemExit("usage: --v7-captures <folder> --v7s-captures <folder>, or --verify <returned folder>")
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; refusing to overwrite")
    paths = code_paths()
    cm.same_as_head([ROOT / p for p in paths])
    state = cm.git_state(paths)
    raw = COND.read_bytes()
    base_check(json.loads(raw))
    code_check()
    s7.check_constants(json.loads(s7.COND.read_text(encoding="utf-8")))   # as v7s's main does
    record = read_record()
    vu.runlogs_as_v7s(a.v7s_captures, record)
    h0 = h0_of()
    got, runlog, unit_status = load(a.v7_captures, a.v7s_captures, h0)
    files = build(got, runlog, unit_status)
    check_counts(files["status.csv"], record)
    manifest = {"files": ex7.manifest_entries(files),
                "provenance": {"run_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), **state,
                               "conditions_sha256": hashlib.sha256(raw).hexdigest(),
                               "code_sha256": {p: cm.sha256(ROOT / p) for p in paths[1:]},
                               "v7s_record_sha256": V7S_RECORD_SHA256,
                               "v7s_runlogs_sha256": {f"runlog.b{b}.ndjson": cm.sha256(a.v7s_captures /
                                                                                     f"runlog.b{b}.ndjson")
                                                      for b in range(1, s7.N_BLOCKS + 1)}}}
    check_manifest(manifest)
    ex7.write_all(OUT, files, manifest)
    for name, e in manifest["files"].items():
        print(f"{e['sha256']}  {name}  rows {e['rows']}")
    print(f"written {OUT}")


if __name__ == "__main__":
    main()
