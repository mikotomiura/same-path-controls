"""Tests of capture/events.py (schedule format, kinds, makers). Synthetic only."""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from capture import events  # noqa: E402


def test_write_read_round_trip(tmp_path):
    p = events.from_gaps(["a", " ", "Enter", "Backspace", "Delete", "."], [0, 10.5, 0, 3, 7, 1000])
    events.write(tmp_path / "s.ndjson", p, "x", "test", extra=1)
    h, q = events.read(tmp_path / "s.ndjson")
    assert q == p
    assert h["n"] == 6 and h["series"] == "x" and h["extra"] == 1


def test_write_refuses_to_overwrite(tmp_path):
    p = events.constant("ab", 10)
    events.write(tmp_path / "s.ndjson", p, "x", "t")
    with pytest.raises(FileExistsError):
        events.write(tmp_path / "s.ndjson", p, "x", "t")


@pytest.mark.parametrize("bad", [
    [{"i": 0, "t_ms": 0.0, "key": "a"}, {"i": 1, "t_ms": -1.0, "key": "a"}],     # backwards
    [{"i": 0, "t_ms": 0.0, "key": "a"}, {"i": 1, "t_ms": math.inf, "key": "a"}],  # not finite
    [{"i": 0, "t_ms": 0.0, "key": "a"}, {"i": 1, "t_ms": math.nan, "key": "a"}],
    [{"i": 0, "t_ms": 0.0, "key": "a"}, {"i": 2, "t_ms": 1.0, "key": "a"}],       # i not consecutive
    [{"i": 0, "t_ms": 0.0, "key": "ab"}],                                          # two characters
    [{"i": 0, "t_ms": 0.0, "key": "Tab"}],                                         # not a named key
    [{"i": 0, "t_ms": 0.0, "key": "é"}],                                      # outside ASCII
    [{"i": 0, "t_ms": 0.0, "key": "\n"}],                                          # control
    [{"i": 0, "t_ms": 5.0, "key": "a"}],                                           # first not at 0
    [{"i": 0, "t_ms": True, "key": "a"}],
])
def test_validate_rejects(bad):
    with pytest.raises(ValueError):
        events.validate(bad)


def test_read_rejects_wrong_count(tmp_path):
    f = tmp_path / "s.ndjson"
    f.write_text(json.dumps({"kind": "header", "n": 2}) + "\n" + json.dumps({"i": 0, "t_ms": 0, "key": "a"}) + "\n")
    with pytest.raises(ValueError):
        events.read(f)


def test_equal_times_are_kept():
    p = events.from_gaps("abc", [0, 0, 0])
    assert [x["t_ms"] for x in p] == [0, 0, 0]


def test_constant():
    p = events.constant("ab\ncd", 120)
    assert [x["key"] for x in p] == ["a", "b", "Enter", "c", "d"]
    assert np.allclose(events.gaps(p), 120)


def test_from_gaps_times_and_checks():
    p = events.from_gaps("abc", [0, 5, 7.5])
    assert [x["t_ms"] for x in p] == [0, 5, 12.5]
    for g in ([1, 5, 7], [0, -1, 2], [0, np.nan, 1], [0, 1]):
        with pytest.raises(ValueError):
            events.from_gaps("abc", g)


def test_cycle_letters():
    p = events.cycle_letters(25, 5)
    assert "".join(x["key"] for x in p) == ("abcdefghij" * 3)[:25]
    assert p[-1]["t_ms"] == 120.0


def test_lognormal_series_holds_one_backspace_one_delete_and_rests():
    text = "the quick brown fox jumps over the lazy dog. " * 5
    p = events.lognormal_series(text, 200, 0.5, 20261031)
    keys = [x["key"] for x in p]
    assert keys.count("Backspace") == 1 and keys.count("Delete") == 1
    assert len(p) == len(text) + 2
    assert np.sum(events.gaps(p) >= 2000) >= 1
    assert events.lognormal_series(text, 200, 0.5, 20261031) == p  # same seed, same series


def test_kinds():
    keys = ["a", "Z", "7", " ", ".", "!", "?", ",", "'", "Enter", "Backspace", "Delete"]
    assert events.kinds(events.from_gaps(keys, [0] * 12)) == [
        "alnum", "alnum", "alnum", "space", "end", "end", "end", "other", "other", "enter", "backspace", "delete"]


def hook_rec(t_ms, kind, down=True, injected=False, repeat=False):
    return {"type": "key", "t_ns": int(t_ms * 1e6), "down": down, "injected": injected, "ours": injected,
            "repeat": repeat, "kind": kind}


def test_from_live_keys_times_and_what_is_left_out():
    recs = [{"type": "header"}, hook_rec(1000, "alnum"), hook_rec(1050, "alnum", down=False),
            hook_rec(1100, "modifier"), hook_rec(1200, "other"), hook_rec(1300, "space"),
            hook_rec(1350, "space", repeat=True), hook_rec(1400, "end"), hook_rec(1500, "enter"),
            hook_rec(1600, "backspace"), hook_rec(1700, "delete"), hook_rec(1800, "alnum", injected=True),
            hook_rec(1900, "nonchar")]
    p, left = events.from_live(recs)
    assert [x["key"] for x in p] == ["e", ",", " ", ".", "Enter", "Backspace", "Delete"]
    assert [x["t_ms"] for x in p] == [0, 200, 300, 400, 500, 600, 700]
    assert left == {"injected": 1, "repeat": 1, "up": 1, "modifier": 1, "nonchar": 1, "reordered": 0}


def test_from_live_keeps_no_character():
    recs = [hook_rec(i * 10, k) for i, k in enumerate(["alnum"] * 5 + ["other"] * 3)]
    p, _ = events.from_live(recs)
    assert {x["key"] for x in p} == {"e", ","}
