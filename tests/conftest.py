"""Two things this archive does not hold, said before and after the tests.

1. The records of runs v6k to v6s are not in this archive (README, History); they are rebuilt with
   tools/rebuild_records.py. The modules test_prototype_v6n to v6s read the records of v6k and v6l when they are
   imported, and the fixture `whole` of test_prototype_v6m reads them; with the records absent those modules are not
   collected and those tests are skipped, with the reason.
2. The analyses and their tests name commits of the working repository (for example the commit v7's analyses pin).
   In this archive they resolve through git replace refs that tools/original_commits.py writes from commit-map.tsv;
   without them the tests that check pinned commits fail with git errors, and the note below says what to run.
"""

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NEEDED = [ROOT / "results" / f"prototype_{r}" / "record.json" for r in ("v6k", "v6l")]
MISSING = [p.relative_to(ROOT).as_posix() for p in NEEDED if not p.exists()]
REASON = f"needs {', '.join(MISSING)} (not archived; rebuild with tools/rebuild_records.py)"
IMPORT_READS = [f"test_prototype_{r}.py" for r in ("v6n", "v6o", "v6p", "v6q", "v6r", "v6s")]

collect_ignore = IMPORT_READS if MISSING else []

PINNED = "a6fa5cae45579c2b4cc95ccb938eab6ea622310e"   # the commit v7's analyses pin
# rev-parse follows replace refs (git cat-file -e does not, for an object absent from the archive)
ORIGINALS_MISSING = subprocess.run(["git", "rev-parse", "--verify", "-q", f"{PINNED}^{{tree}}"], cwd=ROOT,
                                   capture_output=True).returncode != 0
ORIGINALS_NOTE = ("the original commits do not resolve: run `python tools/original_commits.py` once (README, "
                  "Reproduce), else the tests that check pinned commits fail")


def notes() -> list[str]:
    out = []
    if MISSING:
        out.append(f"not collected: {', '.join(IMPORT_READS)}; skipped: the tests of test_prototype_v6m.py that "
                   f"use 'whole' -- {REASON}")
    if ORIGINALS_MISSING:
        out.append(ORIGINALS_NOTE)
    return out


def pytest_collection_modifyitems(config, items):
    if not MISSING:
        return
    for item in items:
        if item.path.name == "test_prototype_v6m.py" and "whole" in getattr(item, "fixturenames", ()):
            item.add_marker(pytest.mark.skip(reason=REASON))


def pytest_report_header(config):
    return notes()


def pytest_terminal_summary(terminalreporter):
    # shown also with -q, where the header is not printed
    for line in notes():
        terminalreporter.write_line(line)
