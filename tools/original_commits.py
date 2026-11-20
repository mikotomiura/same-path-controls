"""Let the original commits of the working repository resolve in this archive (git replace refs from commit-map.tsv).

The analyses and their tests check that the code and conditions are the bytes of named commits (for example
`git rev-parse a6fa5ca...:works/prototype_v7.py`). Those are commits of the working repository; files were removed
from every commit of this archive (README, History), so its commits have other hashes. This tool writes, for every
row of commit-map.tsv whose hashes differ, the ref refs/replace/<original> -> <archived>, so that git reads the
archived commit wherever the original is named. No file is changed. Replace refs are local: run this once after
cloning (or after unpacking the bundle), before the tests.

Every row is checked first (two full lowercase hashes, no original twice, the archived commit present here). A replace
ref that already points elsewhere stops the run. The refs are written, or removed, in one transaction of
`git update-ref --stdin`: all or none.

Usage:
  python tools/original_commits.py            # write the refs and check that each original resolves
  python tools/original_commits.py --remove   # delete them (only those that point where the map says)
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAP = ROOT / "commit-map.tsv"
HASH = re.compile(r"[0-9a-f]{40}")
ZERO = "0" * 40


def git(*args: str, stdin: str | None = None, root: Path | None = None) -> str:
    r = subprocess.run(["git", *args], cwd=root or ROOT, capture_output=True, text=True, input=stdin)
    if r.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed: {r.stderr.strip()}")
    return r.stdout.strip()


def rows(path: Path = MAP) -> list[tuple[str, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != "original\tarchived":
        raise SystemExit(f"{path}: not the commit map (header original<TAB>archived)")
    out, seen = [], set()
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) != 2 or not all(HASH.fullmatch(x) for x in parts):
            raise SystemExit(f"{path}: a row is not two full hashes: {line!r}")
        if parts[0] in seen:
            raise SystemExit(f"{path}: {parts[0]} appears twice")
        seen.add(parts[0])
        out.append((parts[0], parts[1]))
    return out


def current(ref: str, root: Path) -> str | None:
    r = subprocess.run(["git", "--no-replace-objects", "rev-parse", "--verify", "-q", ref], cwd=root,
                       capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def write(pairs: list[tuple[str, str]], root: Path) -> None:
    for a, b in pairs:   # every row checked before any ref is written
        r = subprocess.run(["git", "--no-replace-objects", "cat-file", "-t", b], cwd=root, capture_output=True,
                           text=True)
        if r.returncode != 0 or r.stdout.strip() != "commit":
            raise SystemExit(f"{b} (archived commit of {a}) is not a commit here")
    cmds, before = [], {}
    for a, b in pairs:
        have = current(f"refs/replace/{a}", root)
        if have is not None and have != b:
            raise SystemExit(f"refs/replace/{a} already points to {have}, not {b}; nothing written")
        before[a] = have
        cmds.append(f"update refs/replace/{a} {b} {have or ZERO}")
    if not cmds:
        return
    git("update-ref", "--stdin", stdin="start\n" + "\n".join(cmds) + "\nprepare\ncommit\n", root=root)
    for a, b in pairs:   # each original now reads as its archived commit
        if git("rev-parse", f"{a}^{{tree}}", root=root) != git("rev-parse", f"{b}^{{tree}}", root=root):
            undo = [f"delete refs/replace/{x} {y}" for x, y in pairs if before[x] is None]
            if undo:
                git("update-ref", "--stdin", stdin="start\n" + "\n".join(undo) + "\nprepare\ncommit\n", root=root)
            raise SystemExit(f"{a} does not resolve to {b}; the refs written by this run were removed again")


def remove(pairs: list[tuple[str, str]], root: Path) -> int:
    cmds = []
    for a, b in pairs:
        have = current(f"refs/replace/{a}", root)
        if have is None:
            continue
        if have != b:
            raise SystemExit(f"refs/replace/{a} points to {have}, not {b} as the map says; nothing removed")
        cmds.append(f"delete refs/replace/{a} {b}")
    if cmds:
        git("update-ref", "--stdin", stdin="start\n" + "\n".join(cmds) + "\nprepare\ncommit\n", root=root)
    return len(cmds)


def main(argv: list[str], root: Path = ROOT, path: Path = MAP) -> None:
    all_rows = rows(path)
    pairs = [(a, b) for a, b in all_rows if a != b]
    if argv == ["--remove"]:
        print(f"removed the replace refs of {remove(pairs, root)} original commits")
        return
    if argv:
        raise SystemExit(__doc__)
    write(pairs, root)
    print(f"{len(pairs)} original commits now resolve to their archived commits "
          f"({len(all_rows) - len(pairs)} rows have the same hash in both)")


if __name__ == "__main__":
    main(sys.argv[1:])
