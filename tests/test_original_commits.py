"""Tests of tools/original_commits.py on temporary repositories: the originals are commits of another repository,
absent here, as in the archive."""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import original_commits as oc  # noqa: E402

ENV = ["-c", "user.name=t", "-c", "user.email=t@example.invalid", "-c", "commit.gpgsign=false"]


def git(root: Path, *args: str) -> str:
    return subprocess.run(["git", *ENV, *args], cwd=root, capture_output=True, text=True, check=True).stdout.strip()


def repo(path: Path, contents: list[str], date: str) -> list[str]:
    """One commit per content of file f; returns the hashes."""
    path.mkdir()
    git(path, "init", "-q")
    out = []
    for c in contents:
        (path / "f").write_text(c, encoding="utf-8")
        git(path, "add", "f")
        subprocess.run(["git", *ENV, "commit", "-q", "-m", c], cwd=path, check=True,
                       env={"GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date, "PATH": "/usr/bin:/bin:/usr/local/bin"})
        out.append(git(path, "rev-parse", "HEAD"))
    return out


@pytest.fixture
def pair(tmp_path):
    """Originals (another repository) and their archived commits here: same trees, other hashes."""
    orig = repo(tmp_path / "work", ["a", "b", "c"], "2026-01-01T00:00:00+00:00")
    arch = repo(tmp_path / "archive", ["a", "b", "c"], "2026-02-02T00:00:00+00:00")
    assert not set(orig) & set(arch)
    root = tmp_path / "archive"
    path = tmp_path / "map.tsv"
    path.write_text("original\tarchived\n" + "".join(f"{a}\t{b}\n" for a, b in zip(orig, arch)), encoding="utf-8")
    return root, path, orig, arch


def refs(root: Path) -> list[str]:
    return git(root, "for-each-ref", "--format=%(refname)", "refs/replace").split()


def test_the_originals_resolve_and_a_second_run_changes_nothing(pair):
    root, path, orig, arch = pair
    with pytest.raises(subprocess.CalledProcessError):
        git(root, "rev-parse", "--verify", f"{orig[0]}^{{tree}}")
    oc.main([], root, path)
    for a, b in zip(orig, arch):
        assert git(root, "rev-parse", f"{a}^{{tree}}") == git(root, "rev-parse", f"{b}^{{tree}}")
        assert git(root, "show", f"{a}:f") == git(root, "show", f"{b}:f")
    oc.main([], root, path)
    assert len(refs(root)) == 3


def test_a_ref_pointing_elsewhere_stops_before_anything_is_written(pair):
    root, path, orig, arch = pair
    git(root, "update-ref", f"refs/replace/{orig[1]}", arch[2])
    with pytest.raises(SystemExit, match="already points to"):
        oc.main([], root, path)
    assert refs(root) == [f"refs/replace/{orig[1]}"]


def test_a_missing_archived_commit_or_a_bad_row_stops_before_anything_is_written(pair, tmp_path):
    root, path, orig, arch = pair
    text = path.read_text(encoding="utf-8")
    bad = tmp_path / "bad.tsv"
    bad.write_text(text + f"{'e' * 40}\t{'f' * 40}\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="not a commit here"):
        oc.main([], root, bad)
    assert refs(root) == []
    for row, msg in ((f"{orig[0]}\t{arch[0]}\n", "twice"), ("abc\tdef\n", "two full hashes"),
                     (f"{orig[0].upper()}x\t{arch[0]}\n", "two full hashes")):
        bad.write_text(text + row, encoding="utf-8")
        with pytest.raises(SystemExit, match=msg):
            oc.main([], root, bad)
    assert refs(root) == []
    bad.write_text("from\tto\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="not the commit map"):
        oc.main([], root, bad)


def test_remove_deletes_only_what_the_map_says(pair):
    root, path, orig, arch = pair
    oc.main([], root, path)
    oc.main(["--remove"], root, path)
    assert refs(root) == []
    oc.main([], root, path)
    git(root, "update-ref", f"refs/replace/{orig[0]}", arch[1])
    with pytest.raises(SystemExit, match="nothing removed"):
        oc.main(["--remove"], root, path)
    assert len(refs(root)) == 3


def test_the_archive_map_has_full_unique_rows():
    if (ROOT / "commit-map.tsv").exists():
        rows = oc.rows()
        assert rows and len({a for a, _ in rows}) == len(rows)


def test_a_written_ref_that_does_not_give_the_archived_tree_stops(pair, monkeypatch):
    """The check after writing: an original whose tree is not its archived commit's (git's answer replaced here)."""
    root, path, orig, arch = pair
    real = oc.git

    def git_(*args, **kw):
        if args[:1] == ("rev-parse",) and args[1] == f"{orig[2]}^{{tree}}":
            return "0" * 40
        return real(*args, **kw)

    monkeypatch.setattr(oc, "git", git_)
    with pytest.raises(SystemExit, match="does not resolve to"):
        oc.main([], root, path)
    assert refs(root) == []          # written by the run, removed again


def test_a_map_without_differing_rows_writes_nothing_and_succeeds(pair, tmp_path):
    root, path, orig, arch = pair
    same = tmp_path / "same.tsv"
    same.write_text("original\tarchived\n" + "".join(f"{b}\t{b}\n" for b in arch), encoding="utf-8")
    oc.main([], root, same)
    oc.main(["--remove"], root, same)
    same.write_text("original\tarchived\n", encoding="utf-8")
    oc.main([], root, same)
    assert refs(root) == []
