"""Check the released series against their manifests, their exports' allow-lists and the records of their runs.

series/v7s: works/export_v7s_series.py's own check (--verify: the files, the manifest's SHA-256, rows, presses and
forms, the allow-list and structure, and the counts of v7s's record).
series/v7: the same with the functions of works/export_v7_series.py, which has no such check of its own: the files
are exactly the release's, the manifest is the files' (SHA-256, rows, presses), every cell passes the export's
allow-list, every layer runs i = 0, 1, ... from a first time of 0, a unit has plan rows exactly where status.csv says
and recorded rows only when captured, the counts per series and status are v7's record's (failures), and the files were
written by the commit that wrote v7s's (the export's code at fe25fd1 of the working repository). Neither export's code
is changed.

Usage: uv run --frozen python tools/verify_series.py
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "works"))
sys.path.insert(0, str(ROOT))

import export_v7_series as ex7  # noqa: E402
import export_v7s_series as ex7s  # noqa: E402

V7_RECORD = ROOT / "results" / "prototype_v7" / "record.json"
V7_RECORD_SHA256 = "53c03bf7bbf20310b3b98fc7f4e203668169cfb0c200ed75fb340c6d28889879"
EXPORT_COMMIT = "fe25fd1592427a62f2d8570bdce2155500d45339"
FILES = frozenset({"status.csv", *(f"kupa-derived/{s}.csv" for s in ex7.KUPA_DERIVED),
                   *(f"not-kupa-derived/{s}.csv" for s in ex7.NOT_KUPA_DERIVED)})
ZERO = ("0.000000", "-0.000000")


def rows_of(b: bytes) -> list[list[str]]:
    return list(csv.reader(io.StringIO(b.decode("utf-8"))))


def verify_v7(out: Path, record: dict) -> dict:
    m = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    on_disk = {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()}
    if on_disk != FILES | {"manifest.json"}:
        raise SystemExit(f"{out}: the files are not the release's: {sorted(on_disk ^ (FILES | {'manifest.json'}))}")
    files = {n: (out / n).read_bytes() for n in FILES}
    if ex7.manifest_entries(files) != m["files"]:
        raise SystemExit(f"{out}: the files' SHA-256, rows or presses are not the manifest's")
    if m["provenance"].get("git_commit") != EXPORT_COMMIT or m["provenance"].get("git_dirty") is not False:
        raise SystemExit(f"{out}: not written by the export's commit {EXPORT_COMMIT[:7]} on a clean tree")
    ex7.check_files(files)
    st_rows = rows_of(files["status.csv"])[1:]
    status = {(r[0], r[1]): r for r in st_rows}
    if len(status) != len(st_rows):
        raise SystemExit(f"{out}: a unit appears twice in status.csv")
    for name, b in files.items():
        if name == "status.csv":
            continue
        s = name.split("/")[-1].removesuffix(".csv")
        layers: dict[tuple[str, str], list[list[str]]] = {}
        for r in rows_of(b)[1:]:
            layers.setdefault((r[1], r[2]), []).append(r)
        for (u, la), rs in layers.items():
            if [int(r[3]) for r in rs] != list(range(len(rs))) or rs[0][5] not in ZERO:
                raise SystemExit(f"{name}: {u}/{la} is not i = 0, 1, ... from a first time of 0")
            st = status.get((s, u))
            if st is None or (la == "plan" and st[4] != "yes") or (la != "plan" and st[2] != "captured"):
                raise SystemExit(f"{name}: {u}/{la} is not what status.csv says of the unit")
        for (s2, u), st in status.items():
            if s2 == s and (st[4] == "yes") != ((u, "plan") in layers):
                raise SystemExit(f"{name}: {u}'s plan rows disagree with status.csv")
    failures = record["failures"]
    for s in (*ex7.KUPA_DERIVED, *ex7.NOT_KUPA_DERIVED):
        mine = [r for (s2, _), r in status.items() if s2 == s]
        have = {"planned": len(mine), "no_plan": sum(r[2] == "no_plan" for r in mine),
                "failed_capture": sum(r[2] == "failed" for r in mine), "redone": sum(r[3] == "true" for r in mine)}
        if have != failures.get(s):
            raise SystemExit(f"{out}: {s}: {have} is not v7's record's {failures.get(s)}")
    return m


def main() -> None:
    if hashlib.sha256(V7_RECORD.read_bytes()).hexdigest() != V7_RECORD_SHA256:
        raise SystemExit(f"{V7_RECORD} is not v7's record (its SHA-256)")
    verify_v7(ROOT / "series" / "v7", json.loads(V7_RECORD.read_text(encoding="utf-8")))
    print("verified series/v7")
    m = ex7s.verify(ROOT / "series" / "v7s", ex7s.read_record())
    if m["provenance"].get("git_commit") != EXPORT_COMMIT:
        raise SystemExit(f"series/v7s: not written by the export's commit {EXPORT_COMMIT[:7]}")
    print("verified series/v7s")


if __name__ == "__main__":
    main()
