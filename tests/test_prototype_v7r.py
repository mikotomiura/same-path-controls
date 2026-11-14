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


def test_groups_equal_v7qs_session_by_session():
    caps = _caps()
    # a constant plan recorded on a page of constant gaps with one long pause (page delta from one outlier)
    caps.append(_cap(np.arange(200) * 20.0, np.cumsum(np.concatenate([[0.0], np.full(198, 20.0), [600.0]]))))
    for c in caps:
        theirs = list(vq.m2_groups([c], [c]))
        assert theirs == [vrr.group_of(c)]
    assert {vrr.group_of(c) for c in caps} == {"flips", "stays machine", "plan not judged", "stays human",
                                               "page not judged"}


def test_sessions_not_captured_are_left_out():
    c = _cap(np.arange(200) * 20.0, np.arange(200) * 20.0)
    c.status = "failed"
    assert vrr.m2_sizes([c]) == {} and vq.m2_groups([c], [c]) == {}


def test_sizes_of_planned_gaps_by_group():
    out = vrr.m2_sizes(_caps())
    assert out["flips"]["plan_median_positive_gap_min_max_ms"] == [1.0, 1.0]
    assert out["stays machine"]["plan_median_positive_gap_min_max_ms"] == [20.0, 20.0]
    assert out["flips"]["sessions_with_median_planned_gap_below_ms"] == {"1.0": 0, "2.0": 1, "5.0": 1, "10.0": 1}
    assert out["plan not judged"]["plan_median_positive_gap_min_max_ms"] is None
    assert out["flips"]["plan_share_at_median"] == {"n": 1}   # fewer than 5 sessions: the count only


def test_share_at_median_counts_zeros_in_the_denominator_and_drops_are_reported():
    plan = np.concatenate([[0.0, 0.0], np.arange(1, 199) * 1.0])  # two equal times, then 1 ms gaps
    page = np.cumsum(np.concatenate([[0.0], np.full(199, 1.0)]))
    sz = vrr.sizes(_cap(plan, page))
    assert sz["plan_median_positive_gap_ms"] == 1.0
    assert abs(sz["plan_share_at_median"] - 198 / 199) < 1e-12
    assert sz["hook_dropped_equal_times"] == 1 and sz["page_dropped_equal_times"] == 0
