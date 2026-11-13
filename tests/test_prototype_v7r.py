"""Tests of works/prototype_v7r.py: v7q's groups written again, and the sizes of planned and recorded gaps."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "works"))
sys.path.insert(0, str(ROOT / "tests"))
import prototype_v7_run as vr  # noqa: E402
import prototype_v7q as vq  # noqa: E402
import prototype_v7r as vrr  # noqa: E402
from test_prototype_v7q import _cap  # noqa: E402


def _caps():
    rng = np.random.default_rng(5)
    var = lambda: np.cumsum(np.concatenate([[0.0], rng.uniform(1, 60, 199)]))  # noqa: E731
    one_ms, twenty = np.arange(200) * 1.0, np.arange(200) * 20.0
    return [_cap(one_ms, var()), _cap(twenty, twenty.copy()), _cap(np.zeros(200), twenty.copy()),
            _cap(var(), var()), _cap(twenty, np.zeros(1))]


def test_groups_equal_v7qs():
    caps = _caps()
    mine = {}
    for c in caps:
        g = vrr.group_of(c)
        mine[g] = mine.get(g, 0) + 1
    assert mine == {g: v["sessions"] for g, v in vq.m2_groups(caps, caps).items()}
    assert set(mine) == {"flips", "stays machine", "plan not judged", "stays human", "page not judged"}


def test_sizes_of_planned_gaps_by_group():
    out = vrr.m2_sizes(_caps())
    assert out["flips"]["plan_median_positive_gap_min_max_ms"] == [1.0, 1.0]
    assert out["stays machine"]["plan_median_positive_gap_min_max_ms"] == [20.0, 20.0]
    assert out["flips"]["sessions_with_median_planned_gap_below_ms"] == {"1.0": 0, "2.0": 1, "5.0": 1, "10.0": 1}
    assert out["plan not judged"]["plan_median_positive_gap_min_max_ms"] is None
    assert out["flips"]["plan_share_at_median"] == {"n": 1}   # fewer than 5 sessions: the count only
