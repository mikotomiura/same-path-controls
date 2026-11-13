"""Tests of works/prototype_v7q.py on the synthetic capture folder of test_prototype_v7_run (no KUPA, no capture of
the work PC): the census of page repeats, the readings with flagged presses kept, and M2's groups."""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "works"))
sys.path.insert(0, str(ROOT / "tests"))
import prototype_v7 as v7  # noqa: E402
import prototype_v7_run as vr  # noqa: E402
import prototype_v7q as vq  # noqa: E402
from test_prototype_v7_run import small, write_ndjson  # noqa: E402,F401  (small is a fixture)


def flag_page(folder: Path, idx: list[int]) -> None:
    """Mark the counted page lines at the given positions as repeats, as Chrome did on the work PC."""
    rows = [json.loads(x) for x in (folder / "page.ndjson").read_text().splitlines() if x.strip()]
    keys = [i for i, r in enumerate(rows) if r.get("type") == "key"]
    for j in idx:
        rows[keys[j]]["repeat"] = True
    write_ndjson(folder / "page.ndjson", rows)


@pytest.fixture(scope="module")
def flagged(small, tmp_path_factory):
    root = tmp_path_factory.mktemp("v7q") / "caps"
    shutil.copytree(small["root"], root)
    flag_page(root / "H1_idle" / "h00", [5, 6, 40])
    flag_page(root / "M2_8B" / "t00", [10])
    caps = vr.load_all(root)
    sc = vr.train([small["e_gaps"][i] for i in small["train"]])
    s_h0 = sc.score([v7.gaps_of(x["times"]) for x in small["h0"]])
    return {"caps": caps, "rec": vr.to_json(vq.diagnose(caps, s_h0, sc)), "root": root}


def test_census_counts_the_flagged_presses(flagged):
    c = flagged["rec"]["repeats_census"]["H1_idle"]
    assert c["flagged"] == 3 and c["items_with_flag"] == 1 and c["aligned_items"] == 1
    assert sum(c["kind"].values()) == 3 and sum(c["same_planned_key_as_previous"].values()) == 3
    assert sum(c["recorded_gap_before"].values()) == 3
    assert flagged["rec"]["repeats_census"]["M2_8B"]["flagged"] == 1
    assert flagged["rec"]["repeats_census"]["M0"]["flagged"] == 0


def test_main_reading_drops_and_kept_reading_restores(flagged):
    cap = flagged["caps"]["H1_idle"][0]
    assert cap.layers["page"].repeats == 3 and cap.layers["page"].times.size == len(cap.plan.kinds) - 3
    kept = vq.kept_capture(cap)
    assert kept.layers["page"].times.size == len(cap.plan.kinds) and kept.layers["page"].kinds == cap.plan.kinds
    assert cap.layers["page"].times.size == len(cap.plan.kinds) - 3   # the original is not changed
    re = flagged["rec"]["repeats_kept"]["H1_idle_replay_error"]
    assert re["drops"] == 0 and re["all_matched"]


def test_kept_readings_cover_every_detector_and_series(flagged):
    rk = flagged["rec"]["repeats_kept"]
    assert set(rk["human"]) == set(vr.DETECTORS)
    assert set(rk["machine"]) == set(vr.MACHINE)
    assert all(rk["human"][d]["reading"] in ("not changed", "undecided", "responds") for d in vr.DETECTORS)


def test_m2_groups_hold_every_captured_session(flagged):
    for s_ in ("M2_8B", "M2_14B"):
        n = sum(1 for c in flagged["caps"][s_] if c.status == "captured")
        assert sum(g["sessions"] for g in flagged["rec"]["m2_sessions"][s_].values()) == n


def test_record_holds_no_series(flagged):
    def walk(x):
        if isinstance(x, dict):
            for v in x.values():
                yield from walk(v)
        elif isinstance(x, list):
            yield x
            for v in x:
                yield from walk(v)
    assert all(len(lst) <= 8 for lst in walk(flagged["rec"]))
    text = json.dumps(flagged["rec"])
    assert "\"key\"" not in text and "NaN" not in text


def test_bins_are_closed_left():
    assert vq.bin_of(0.0) == "0-1" and vq.bin_of(1.0) == "1-5" and vq.bin_of(149.9) == "50-150"
    assert vq.bin_of(150.0) == "150+"


def _cap(plan_t: np.ndarray, page_t: np.ndarray) -> vr.Capture:
    n = plan_t.size
    kinds = ["alnum"] * n
    plan = vr.Plan(plan_t, kinds, ["a"] * n)
    layers = {"hook": vr.Layer(plan_t.copy(), kinds, 0, 0), "page": vr.Layer(page_t, ["alnum"] * page_t.size, 0, 0)}
    return vr.Capture("M2_8B", 0, "captured", plan=plan, layers=layers)


def test_m2_groups_tell_flips_from_stays():
    """A constant plan (delta 0, 'machine') recorded on a page whose gaps vary (delta above 0.269) flips; one
    recorded as planned stays 'machine'; an all-zero plan is not judged."""
    rng = np.random.default_rng(3)
    plan = np.arange(200) * 20.0
    page_var = np.cumsum(np.concatenate([[0.0], rng.uniform(1, 60, 199)]))
    caps = [_cap(plan, page_var), _cap(plan, plan.copy()), _cap(np.zeros(200), plan.copy())]
    g = vq.m2_groups(caps, caps)
    assert g["flips"]["sessions"] == 1 and g["stays machine"]["sessions"] == 1
    assert g["plan not judged"]["sessions"] == 1
