"""Tests of the fixes of the Mac review of returns-v3 (2026-11-04): M2's errors of ollama sorted from the model's
failures, a stop during M2's planning, the hook's footer checked, keys injected through a remote desktop, the page's
batch written whole, the load's stall written before its stop, and a BOM left out of the task. Each test fails with the
fix taken out."""

import http.client
import io
import json
import sys
import threading
import time
import urllib.error
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from capture import llm_plan, load, serve, session, summary  # noqa: E402
from tests.test_capture_llm import two_calls  # noqa: E402


# --- M2: only a read timeout is the model's failed attempt (the researcher's decision, D1) ---

@pytest.mark.parametrize("err", [
    OSError("connection reset"),
    ConnectionRefusedError("refused"),
    urllib.error.URLError(TimeoutError("timed out")),  # a timeout while connecting: ollama did not answer at all
    urllib.error.HTTPError("http://x", 500, "llama runner process has terminated", {}, None),
    urllib.error.HTTPError("http://x", 404, "model not found", {}, None),
    http.client.IncompleteRead(b"{\"resp"),
    json.JSONDecodeError("Expecting value", "", 0),  # an answer of ollama cut short: not JSON
])
@pytest.mark.parametrize("call", ["essay", "gaps"])
def test_m2_an_error_of_ollama_raises_and_is_logged_not_counted(err, call):
    log = io.StringIO()
    pairs = [(err,)] if call == "essay" else [("ab", err)]
    with pytest.raises(llm_plan.OllamaError):
        llm_plan.make_plan("m", "Write.", 1, log, generate=two_calls(pairs))
    rows = [json.loads(x) for x in log.getvalue().splitlines()]
    assert len(rows) == 1 and rows[0]["raised"] is True and rows[0]["ok"] is False
    assert "error" in rows[0][call] and rows[0]["reason"].startswith("ollama: ")


def test_m2_a_read_timeout_is_a_failed_attempt():
    assert llm_plan.is_read_timeout(TimeoutError("timed out"))
    assert not llm_plan.is_read_timeout(urllib.error.URLError(TimeoutError("timed out")))
    r = llm_plan.make_plan("m", "Write.", 1, io.StringIO(),
                           generate=two_calls([("ab", TimeoutError("timed out")), ("ab", '{"gaps_ms": [0, 80]}')]))
    assert r["ok"] and r["attempts"] == 2 and r["failures"] == ["ollama: TimeoutError: timed out"]


class Flag:
    def __init__(self, after_calls=None):
        self.n, self.after = 0, after_calls

    def is_set(self):
        self.n += 1
        return self.after is not None and self.n > self.after


@pytest.mark.parametrize("after", [0, 1, 3])  # before the essay; before the gaps; before attempt 2's essay
def test_m2_a_stop_ends_the_planning_before_the_next_call(after):
    bodies, log = [], io.StringIO()
    gen = two_calls([("ab", '{"gaps_ms": [0]}')] * 3, bodies)
    with pytest.raises(llm_plan.PlanStopped):
        llm_plan.make_plan("m", "Write.", 1, log, generate=gen, stop=Flag(after))
    assert len(bodies) == after
    assert json.loads(log.getvalue().splitlines()[-1])["stopped"] is True


def test_m2_a_bom_is_not_part_of_the_task(tmp_path):
    (tmp_path / "t.txt").write_bytes("﻿Write about rain.\r\n".encode("utf-8"))
    bodies = []
    llm_plan.main(["--model", "m", "--prompt-file", str(tmp_path / "t.txt"), "--seed", "1", "--out",
                   str(tmp_path / "o")], generate=two_calls([("ab", '{"gaps_ms": [0, 80]}')], bodies))
    header = json.loads((tmp_path / "o" / "llm.ndjson").read_text(encoding="utf-8").splitlines()[0])
    assert "﻿" not in header["prompt"] and header["prompt"].endswith("Task: Write about rain.")
    assert "﻿" not in bodies[0]["prompt"]
    src = (ROOT / "capture" / "session.py").read_text(encoding="utf-8")
    assert 'a.prompt_file.read_text(encoding="utf-8-sig")' in src  # the session reads the task the same way


# --- the hook's footer: every record the hook saw is in the file ---

def _hook(n_raw, n_written, footer=True, writer_error=None):
    rows = [{"type": "header"}] + [{"type": "key", "t_ns": i, "down": True, "injected": True, "ours": True,
                                    "repeat": False, "kind": "alnum"} for i in range(n_written)]
    if footer:
        rows.append({"type": "footer", "n_raw": n_raw, "n_written": n_written,
                     **({"writer_error": writer_error} if writer_error else {})})
    return rows


def test_a_hook_log_that_lost_records_makes_the_capture_interrupted():
    meta = {"mode": "events", "hook_installed": True}
    base = {"planned": 3, "hook": {"n": 3}, "page": {"n": 3}}
    assert summary.problems({**base, "hook_log": summary.hook_log(_hook(6, 6))}, meta) == {}
    for h in (_hook(6, 4), _hook(6, 6, footer=False), _hook(6, 6, writer_error=["TypeError: x"])):
        p = summary.problems({**base, "hook_log": summary.hook_log(h)}, meta)
        assert p["hook_log_incomplete"]["complete"] is False


def test_the_hook_writer_keeps_its_error_for_the_footer():
    src = (ROOT / "capture" / "hooklog.py").read_text(encoding="utf-8")
    writer = src[src.index("    def writer():"):src.index("    threading.Thread(target=reporter")]
    assert "except Exception" in writer and "writer_error.append" in writer
    assert '"writer_error": writer_error' in src


# --- keys through a remote desktop come injected without our mark (the researcher's decision, a b c) ---

def _key(kind, injected, ours, down=True):
    return {"type": "key", "t_ns": 1, "down": down, "injected": injected, "ours": ours, "repeat": False, "kind": kind}


def test_foreign_injected_keys_are_counted_without_modifiers(tmp_path):
    hook = [_key("alnum", True, True), _key("modifier", True, False), _key("modifier", True, False, down=False),
            _key("alnum", True, False), _key("enter", True, False), _key("alnum", True, False, down=False),
            _key("alnum", False, False)]
    assert summary.foreign_injected_keys(hook) == 2
    base = {"planned": 1, "hook": {"n": 1}, "page": {"n": 1}}
    meta = {"mode": "events", "hook_installed": True}
    assert summary.problems({**base, "foreign_injected_keys": 2}, meta)["foreign_injected_keys"] == 2
    assert summary.problems({**base, "foreign_injected_keys": 0}, meta) == {}
    # the summary of a live capture counts them too
    d = tmp_path / "l0"
    d.mkdir()
    (d / "meta.json").write_text(json.dumps({"mode": "live", "typing_start_perf_ns": 0, "typing_end_perf_ns": 9}))
    (d / "hook.ndjson").write_text("".join(json.dumps(r) + "\n" for r in [*hook, {"type": "footer", "n_raw": 7,
                                                                                 "n_written": 7}]))
    assert summary.summarize(d)["foreign_injected_keys"] == 2


def test_the_hook_reports_a_foreign_injected_key_from_its_callback_without_output():
    src = (ROOT / "capture" / "hooklog.py").read_text(encoding="utf-8")
    body = src[src.index("    def proc(code"):src.index("    def reporter")]
    assert "k.dwExtraInfo != EXTRA_MARK and vk not in MODIFIERS" in body and "foreign_at.append(t)" in body
    assert "print(" not in body and ".write(" not in body
    rep = src[src.index("    def reporter"):src.index("    cb = HOOKPROC(proc)")]
    assert 'print(f"FOREIGN' in rep
    ses = (ROOT / "capture" / "session.py").read_text(encoding="utf-8")
    assert 'w[0] == "FOREIGN"' in ses and "hook.foreign.is_set()" in ses


def test_a_remote_desktop_host_is_found_in_the_process_list():
    listing = ('"System Idle Process","0","Services","0","8 K"\n'
               '"remoting_host.exe","5124","Console","1","21,340 K"\n"chrome.exe","9","Console","1","9 K"\n')
    assert session.remote_hosts_running(listing) == ["remoting_host.exe"]
    assert session.remote_hosts_running(listing.replace("remoting_host", "notepad")) == []
    assert session.remote_hosts_running('"REMOTING_HOST.EXE","1","Console","1","1 K"') == ["remoting_host.exe"]


# --- the page's batch is written whole or not at all ---

def test_a_batch_that_cannot_be_written_whole_writes_nothing(tmp_path):
    st = serve.State(tmp_path / "page.ndjson")
    bad = {"t_ms": 1.0, "kind": "alnum", "x": {1}}  # not JSON: the batch fails while it is built
    good = {"t_ms": 1.0, "kind": "alnum"}
    with pytest.raises(TypeError):
        st.keep("log", {"flush": 1, "batch": 0, "entries": [good, bad]})
    assert st.keep("log", {"flush": 1, "batch": 0, "entries": [good, good]}) == {"saved": 0}
    st.f.close()
    recs = [json.loads(x) for x in (tmp_path / "page.ndjson").read_text().splitlines()]
    assert [r["type"] for r in recs] == ["log", "key", "key"] and st.saved_entries[1] == 2


# --- the load: lines written by its loop ---

class SlowStream:
    """ollama's stream: (seconds to wait before the line, the line); stop is set before the given line."""

    def __init__(self, items, stop=None, stop_before=None):
        self.items, self.stop, self.stop_before = items, stop, stop_before

    def __enter__(self):
        def it():
            for i, (wait, line) in enumerate(self.items):
                if i == self.stop_before:
                    self.stop.set()
                time.sleep(wait)
                yield json.dumps(line).encode()
        return it()

    def __exit__(self, *a):
        return False


def _run_loop(items, stop_before=None):
    f, stop, first, failed = io.StringIO(), threading.Event(), threading.Event(), []
    streams = [SlowStream(items, stop, stop_before)]

    def open_stream(body):
        if not streams:
            stop.set()
            return SlowStream([])
        return streams.pop(0)
    load.loop(f, open_stream, lambda n: {"seed": n}, stop, first, failed)
    return [json.loads(x) for x in f.getvalue().splitlines()], failed


def test_the_load_writes_a_stall_and_progress_lines():
    tok = {"response": "a", "done": False}
    rows, failed = _run_loop([(0.0, tok), (0.3, tok), (0.7, tok), (0.0, {"response": "", "done": True})])
    assert not failed
    kinds = [r["type"] for r in rows]
    assert kinds[0] == "start" and "generation" in kinds
    stalls = [r for r in rows if r["type"] == "stall"]
    assert len(stalls) == 1 and stalls[0]["to_perf_ns"] - stalls[0]["from_perf_ns"] > 500e6
    assert any(r["type"] == "progress" for r in rows)  # 0.3 s after the first token: more than PROGRESS_S
    gen = next(r for r in rows if r["type"] == "generation")
    assert gen["completed"] is True and gen["tokens"] == 3


def test_a_stall_cut_by_the_stop_is_still_written():
    tok = {"response": "a", "done": False}
    # the session stops the load while the generation is stalled; the next line comes after the stop
    rows, _ = _run_loop([(0.0, tok), (0.7, tok), (0.0, tok)], stop_before=1)
    stalls = [r for r in rows if r["type"] == "stall"]
    gen = next(r for r in rows if r["type"] == "generation")
    assert len(stalls) == 1 and gen["completed"] is False
    covered = sum(b - a for a, b in summary.covered_spans(rows, gen["end_perf_ns"], 10**12))
    assert gen["end_perf_ns"] - gen["start_perf_ns"] > 600e6 and covered < 100e6  # the stall is not covered


# --- capture.stop says when there is no capture to stop (Windows only) ---

@pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
def test_capture_stop_finds_no_capture_when_none_holds_the_event():
    from capture import stop
    name = "Local\\v7-capture-stop-test-none"
    assert stop.signal(name) is False  # nobody holds it: an event made here would vanish and stop nothing
    s = stop.StopSignal(None, name=name)
    try:
        assert stop.signal(name) is True and s.is_set() and s.by == "capture.stop"
    finally:
        s.close()
