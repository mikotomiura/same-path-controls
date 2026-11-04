"""Tests of the M1 causal plan (capture/llm_stream.py, amendment B13) and the M2 plan check (capture/llm_plan.py)."""

import io
import json
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from capture import llm_plan, llm_stream  # noqa: E402

MS = 1_000_000


def test_b13_each_token_is_spread_from_the_next_arrival():
    # tokens at 0, 100, 300 ms, generation ends at 400 ms
    plan, p = llm_stream.plan_offline([(0, "ab"), (100 * MS, "c"), (300 * MS, "de")], 400 * MS)
    assert [x["key"] for x in plan] == list("abcde")
    # token 1 (2 chars) from a2 = 100 over a2 - a1 = 100: 100, 150; token 2 from a3 = 300 over 200: 300;
    # token 3 from the end 400 over 100: 400, 450
    assert [x["wished_ms"] for x in plan] == [100, 150, 300, 400, 450]
    assert [x["plan_ms"] for x in plan] == [100, 150, 300, 400, 450]
    assert p.k == 3 and p.moved == 0


def test_every_character_is_wished_no_earlier_than_it_is_known():
    plan, _ = llm_stream.plan_offline([(0, "Hello"), (10 * MS, ","), (11 * MS, " world"), (40 * MS, ".")], 60 * MS)
    known = {1: 10, 2: 11, 3: 40, 4: 60}  # token k is complete when token k+1 arrives (or generation ends)
    for x in plan:
        assert x["wished_ms"] >= known[x["token"]]


def test_running_maximum_keeps_order_and_equal_times():
    # a long token followed by quick ones: wished times go backwards, the plan holds at the running maximum
    plan, p = llm_stream.plan_offline([(0, "abcd"), (100 * MS, "e"), (101 * MS, "f")], 102 * MS)
    assert [x["wished_ms"] for x in plan] == [100, 125, 150, 175, 101, 102]
    assert [x["plan_ms"] for x in plan] == [100, 125, 150, 175, 175, 175]
    assert p.moved == 2 and p.shifts_ms == [74, 73]


def test_nfc_code_points_and_empty_tokens():
    assert llm_stream.n_chars("é") == 1  # e + combining acute is one code point after NFC
    plan, p = llm_stream.plan_offline([(0, "éx"), (5 * MS, ""), (10 * MS, "y"), (12 * MS, "")], 20 * MS)
    assert p.empty_tokens == 2
    # token 1 has 2 code points spread from 10 over 10: the accented e is skipped (not ASCII), x at 15
    assert [x["key"] for x in plan] == ["x", "y"]
    assert [x["wished_ms"] for x in plan] == [15, 20]
    assert p.skipped_chars == 1


def test_line_breaks_become_enter_and_crlf_is_one_press():
    keys, skipped = llm_stream.token_keys("a\r\nb\n")
    assert keys == ["a", None, "Enter", "b", "Enter"] and skipped == 0


class FakeInjector:
    def __init__(self, cost_ns=0, foreground=1):
        self.cost = cost_ns

        class U:
            @staticmethod
            def GetForegroundWindow():
                return foreground
        self.user32 = U()

    def send(self, key):
        t = time.perf_counter_ns()
        while time.perf_counter_ns() < t + self.cost:
            pass
        return t, time.perf_counter_ns(), 2, 2


def test_typist_waits_in_a_queue_and_never_runs_ahead():
    # five presses planned at the same time, each send takes 3 ms: each waits for the one before
    ty = llm_stream.Typist(FakeInjector(cost_ns=3 * MS), guard_hwnd=1, stop=None)
    ty.a1_ns = time.perf_counter_ns() + 20 * MS
    for _ in range(5):
        ty.q.put({"key": "a", "wished_ms": 0.0, "plan_ms": 0.0, "token": 1})
    ty.q.put(None)
    ty.start()
    ty.join(5)
    r = ty.records
    assert len(r) == 5
    assert all(x["sent_ns"] >= x["planned_ns"] for x in r)
    assert all(b["sent_ns"] >= a["returned_ns"] for a, b in zip(r, r[1:]))
    assert r[-1]["wait_ns"] >= 10 * MS
    assert r[0]["queue_len"] >= 4


def test_typist_stops_when_focus_is_lost():
    ty = llm_stream.Typist(FakeInjector(foreground=1), guard_hwnd=2, stop=None)
    ty.a1_ns = time.perf_counter_ns()
    ty.q.put({"key": "a", "wished_ms": 0.0, "plan_ms": 0.0, "token": 1})
    ty.q.put(None)
    ty.start()
    ty.join(5)
    assert ty.records == [] and ty.stopped_focus_lost


@pytest.mark.parametrize("raw,reason", [
    ('{"gaps_ms": [0, 1]}', "len"),
    ('{"gaps_ms": [0, -1, 2]}', "negative"),
    ('{"gaps_ms": [0, 1e999, 2]}', "finite"),
    ('{"gaps_ms": [5, 1, 2]}', "first"),
    ('{"gaps_ms": [0, "1", 2]}', "finite"),
    ('{"text": "abc"}', "object"),
    ("not json", "JSON"),
])
def test_m2_check_rejects(raw, reason):
    with pytest.raises(ValueError, match=reason):
        llm_plan.check_plan("abc", raw)


def test_m2_check_accepts_and_builds_the_schedule():
    text, gaps = llm_plan.check_plan("a b\u2014c", '{"gaps_ms": [0, 100, 50, 30, 20]}')
    p, skipped = llm_plan.presses_of(text, gaps)
    assert skipped == 1  # the em dash
    assert [x["key"] for x in p] == ["a", " ", "b", "c"]
    assert [x["t_ms"] for x in p] == [0, 100, 150, 200]


def two_calls(pairs, bodies=None):
    """A fake ollama for M2: each attempt is (essay answer, gaps answer); an answer that is an exception is raised."""
    answers = iter([x for pair in pairs for x in pair])

    def gen(body):
        if bodies is not None:
            bodies.append(body)
        a = next(answers)
        if isinstance(a, BaseException):
            raise a
        return a if isinstance(a, dict) else {"response": a}
    return gen


def test_m2_asks_the_essay_then_its_gaps_with_a_schema_of_its_length():
    bodies = []
    r = llm_plan.make_plan("m", "Write.", 1, io.StringIO(), generate=two_calls([("  ab \n", '{"gaps_ms": [0, 80]}')], bodies))
    assert r["ok"] and [x["t_ms"] for x in r["presses"]] == [0, 80]
    essay, gaps = bodies
    assert essay["prompt"] == "Write." and "format" not in essay
    assert gaps["prompt"] == llm_plan.gaps_prompt("ab") and "\n\nab\n\n" in gaps["prompt"] and " 2 characters" in gaps["prompt"]
    items = gaps["format"]["properties"]["gaps_ms"]
    assert items["minItems"] == items["maxItems"] == 2 and items["items"] == {"type": "integer", "minimum": 0}
    assert essay["options"] == gaps["options"] == {"temperature": 0.7, "seed": 1}


def test_m2_three_attempts_then_counted_failure():
    bodies = []
    log = io.StringIO()
    r = llm_plan.make_plan("m", "Write.", 7, log, generate=two_calls([("ab", '{"gaps_ms": [0]}')] * 3, bodies))
    assert not r["ok"] and r["attempts"] == 3 and r["presses"] == []
    assert [b["options"]["seed"] for b in bodies] == [7, 7, 1007, 1007, 2007, 2007]
    assert len(log.getvalue().splitlines()) == 3


def test_m2_stops_at_the_first_valid_plan():
    r = llm_plan.make_plan("m", "Write.", 1, io.StringIO(),
                           generate=two_calls([("ab", '{"gaps_ms": [0]}'), ("ab", '{"gaps_ms": [0, 80]}')]))
    assert r["ok"] and r["attempts"] == 2 and [x["t_ms"] for x in r["presses"]] == [0, 80]


def test_m2_plan_of_untypable_characters_only_is_a_failed_attempt():
    log = io.StringIO()
    r = llm_plan.make_plan("m", "Write.", 1, log, generate=two_calls([("\u2014\u2019", '{"gaps_ms": [0, 10]}'),
                                                                        ("ab", '{"gaps_ms": [0, 80]}')]))
    assert r["ok"] and r["attempts"] == 2 and r["failures"] == ["no typable character (2 skipped)"]
    assert '"ok": false' in log.getvalue().splitlines()[0]


def test_m2_a_timeout_of_ollama_is_a_failed_attempt_not_an_error():
    # a read timeout only (the decision of 2026-11-04 on returns-v3): other errors of ollama raise
    # (tests/test_capture_v3_review.py)
    import json
    log = io.StringIO()
    r = llm_plan.make_plan("m", "Write.", 1, log, generate=two_calls([(TimeoutError("timed out"),),
                                                                        ("ab", TimeoutError("timed out")),
                                                                        ("ab", '{"gaps_ms": [0, 80]}')]))
    assert r["ok"] and r["attempts"] == 3
    assert r["failures"] == ["ollama: TimeoutError: timed out", "ollama: TimeoutError: timed out"]
    first, second = (json.loads(x) for x in log.getvalue().splitlines()[:2])
    assert "gaps" not in first and second["essay"]["response"] == "ab" and "error" in second["gaps"]


def test_m2_an_empty_essay_is_a_failed_attempt():
    bodies = []
    r = llm_plan.make_plan("m", "Write.", 1, io.StringIO(),
                           generate=two_calls([(" \n",), ("ab", '{"gaps_ms": [0, 80]}')], bodies))
    assert r["ok"] and r["failures"] == ["empty essay"] and len(bodies) == 3


class FakeStream:
    def __init__(self, lines):
        self.lines = lines

    def __enter__(self):
        return iter(self.lines)

    def __exit__(self, *a):
        return False


def run_m1(monkeypatch, lines, sent=None, wait_s=0.0):
    """llm_stream.run with a fake injector module and a fake ollama stream (no Windows, no model). sent collects the
    keys typed; wait_s is how long each wait for a planned time takes."""
    import json
    import types
    fake = types.ModuleType("capture.inject")

    class Inj(FakeInjector):
        def __init__(self, method):
            super().__init__()

        def send(self, key):
            if sent is not None:
                sent.append(key)
            return super().send(key)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False
    fake.Injector = Inj
    fake.wait_until = lambda t, stop=None: time.sleep(wait_s) or False
    monkeypatch.setitem(sys.modules, "capture.inject", fake)
    from capture import ollama
    monkeypatch.setattr(ollama, "open_stream", lambda body: FakeStream([json.dumps(x).encode() for x in lines]))
    log = io.StringIO()
    return llm_stream.run("m", "Write.", 1, "unicode", None, None, log), log.getvalue()


def test_m1_final_line_with_text_is_typed(monkeypatch):
    res, _ = run_m1(monkeypatch, [{"response": "ab", "done": False}, {"response": "c", "done": True}])
    assert [x["key"] for x in res["plan"]] == ["a", "b", "c"]
    assert res["counts"]["generation_done"] and res["counts"]["typed"] == 3


def test_m1_ollama_error_line_fails_the_capture(monkeypatch):
    with pytest.raises(RuntimeError, match="ollama: model failed"):
        run_m1(monkeypatch, [{"response": "ab", "done": False}, {"error": "model failed"}])


def test_m1_only_text_on_the_done_line_is_typed(monkeypatch):
    res, _ = run_m1(monkeypatch, [{"response": "", "done": False}, {"response": "hi", "done": True}])
    assert [x["key"] for x in res["plan"]] == ["h", "i"] and res["counts"]["typed"] == 2
    assert res["counts"]["not_sent"] == 0


def test_m1_stream_failure_stops_the_typist_before_raising(monkeypatch):
    sent = []
    with pytest.raises(RuntimeError, match="ollama: gone"):
        # 'def' releases 'abc' to the typist, whose wait takes 0.3 s; the error line comes at once
        run_m1(monkeypatch, [{"response": "abc", "done": False}, {"response": "def", "done": False},
                             {"error": "gone"}], sent=sent, wait_s=0.3)
    time.sleep(1.2)  # long enough for a typist that was not stopped to type the three queued presses
    assert sent == []


def test_planning_only_command_records_each_attempt(tmp_path):
    task = tmp_path / "task.txt"
    task.write_text("Write about a season.")
    bodies = []
    gen = two_calls([({"response": "ab", "done_reason": "stop", "eval_count": 3},
                      {"response": '{"gaps_ms": [0]}', "done_reason": "length", "eval_count": 9}),
                     ({"response": "ab", "done_reason": "stop", "eval_count": 3},
                      {"response": '{"gaps_ms": [0, 40]}', "done_reason": "stop", "eval_count": 7})], bodies)
    llm_plan.main(["--model", "m", "--prompt-file", str(task), "--seed", "1", "--out", str(tmp_path / "o"),
                   "--num-predict", "4096"], generate=gen)
    r = json.loads((tmp_path / "o" / "result.json").read_text())
    assert r["ok"] and r["attempts"] == 2 and r["n_presses"] == 2
    assert [(x["ok"], x["seed"], x["gaps_done_reason"], x["essay_chars"]) for x in r["per_attempt"]] == [
        (False, 1, "length", 2), (True, 1001, "stop", 2)]
    assert all(b["options"]["num_predict"] == 4096 for b in bodies)


def test_the_production_plan_sets_no_num_predict():
    bodies = []
    llm_plan.make_plan("m", "Write.", 1, io.StringIO(), generate=two_calls([("a", '{"gaps_ms": [0]}')], bodies))
    assert all(set(b["options"]) == {"temperature", "seed"} for b in bodies) and len(bodies) == 2


@pytest.mark.parametrize("key", ["seed", "temperature"])
def test_extra_options_may_not_set_the_seed_or_temperature(key):
    with pytest.raises(ValueError):
        llm_plan.make_plan("m", "Write.", 1, io.StringIO(), generate=lambda b: {}, extra_options={key: 5})
