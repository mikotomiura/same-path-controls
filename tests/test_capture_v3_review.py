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
from capture import llm_plan, load, remote, runlist, serve, session, summary  # noqa: E402
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


@pytest.mark.parametrize("after", [0, 1, 3])  # before the essay; before the gaps; before attempt 2's gaps
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
    assert remote.hosts_in(listing) == ["remoting_host.exe"]
    assert remote.hosts_in(listing.replace("remoting_host", "notepad")) == []
    assert remote.hosts_in('"REMOTING_HOST.EXE","1","Console","1","1 K"') == ["remoting_host.exe"]
    assert session.remote_stop_message(lambda: []) is None
    assert "remoting_host.exe" in session.remote_stop_message(lambda: ["remoting_host.exe"])

    def broken():
        raise RuntimeError("tasklist failed (1)")
    assert "could not be made" in session.remote_stop_message(broken)  # fail-closed: a failing check stops


class _Stopper:
    def __init__(self, by=None, close_error=None):
        self.by, self.esc, self.closed, self.close_error = by, None, 0, close_error

    def is_set(self):
        return self.by is not None

    def close(self):
        self.closed += 1
        if self.close_error:
            raise OSError(self.close_error)


def _fake_windows(monkeypatch, plan=None, unload_error=None, stopper=None):
    """session.main up to the typing, on the Mac: the machine, git, the stop signal and ollama replaced."""
    from capture import ollama, stop
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(session, "machine", lambda: {"boot_time_utc": None, "ollama_processes_start_utc": None})
    monkeypatch.setattr(session, "git_state", lambda: {})
    monkeypatch.setattr(stop, "StopSignal", lambda esc=None: stopper or _Stopper())
    monkeypatch.setattr(ollama, "model_record", lambda m: {"name": m})
    monkeypatch.setattr(ollama, "version", lambda: "0")
    monkeypatch.setattr(ollama, "ps", lambda: [])

    def unload(m):
        if unload_error:
            raise unload_error
    monkeypatch.setattr(ollama, "unload", unload)
    monkeypatch.setattr(session.time, "sleep", lambda s: None)
    if plan is not None:
        monkeypatch.setattr(llm_plan, "make_plan", plan)


def _m2_args(tmp_path):
    (tmp_path / "task.txt").write_text("\ufeffwrite", encoding="utf-8")
    return ["--llm-plan", "--model", "m", "--prompt-file", str(tmp_path / "task.txt"), "--seed", "1",
            "--out", str(tmp_path / "c")]


def test_a_live_capture_does_not_start_while_a_host_runs_or_the_check_fails(tmp_path, monkeypatch):
    _fake_windows(monkeypatch)
    for i, running in enumerate((lambda: ["remoting_host.exe"], lambda: 1 / 0)):
        monkeypatch.setattr(remote, "running", running)
        out = tmp_path / f"L0_{i}"
        assert session.main(["--live", "--out", str(out)]) == remote.REMOTE_HOST_EXIT
        assert not out.exists()  # before the folder: nothing to clear before the runlist is run again


def test_m2_stopped_during_planning_is_interrupted_with_its_end_written(tmp_path, monkeypatch):
    def plan(*a, **k):
        raise llm_plan.PlanStopped("stopped")
    _fake_windows(monkeypatch, plan=plan, stopper=_Stopper(by="capture.stop"))
    assert session.main(_m2_args(tmp_path)) == session.INTERRUPTED
    meta = json.loads((tmp_path / "c" / "meta.json").read_text(encoding="utf-8"))
    assert meta["status"] == "interrupted" and meta["interrupted"] == {"during": "M2 planning"} and meta["end_utc"]


def test_m2_an_error_whose_cleanup_fails_still_writes_the_end(tmp_path, monkeypatch):
    def plan(*a, **k):
        raise llm_plan.OllamaError("ollama: HTTPError: 500", {"body": {"prompt": "secret"}})
    _fake_windows(monkeypatch, plan=plan, stopper=_Stopper(close_error="handle"))
    with pytest.raises(llm_plan.OllamaError):
        session.main(_m2_args(tmp_path))
    meta = json.loads((tmp_path / "c" / "meta.json").read_text(encoding="utf-8"))
    assert meta["status"] == "failed" and meta["end_utc"] and "secret" not in json.dumps(meta)
    assert meta["stop_errors"] == ["stop signal: OSError: handle"]


def test_m2_a_plan_made_is_kept_for_the_redo_when_the_unload_fails(tmp_path, monkeypatch):
    presses = [{"i": 0, "t_ms": 0.0, "key": "a"}]

    def plan(*a, **k):
        return {"ok": True, "attempts": 1, "failures": [], "skipped_chars": 0, "presses": presses}
    _fake_windows(monkeypatch, plan=plan, unload_error=ConnectionResetError("reset"))
    with pytest.raises(ConnectionResetError):
        session.main(_m2_args(tmp_path))
    first = tmp_path / "c"
    assert json.loads((first / "meta.json").read_text(encoding="utf-8"))["status"] == "failed"
    args = ["--llm-plan", "--model", "m"]
    assert runlist.redo_args(args, first) == ["--model", "m", "--events", str(first / "events.ndjson")]


def test_the_session_s_exit_5_stops_the_runlist_with_or_without_a_meta(tmp_path, monkeypatch):
    rc = {"v": 0}
    monkeypatch.setattr(runlist.subprocess, "run", lambda *a, **k: type("R", (), {"returncode": rc["v"]})())
    out = tmp_path / "c"
    rc["v"] = remote.REMOTE_HOST_EXIT
    assert runlist.default_runner([], out) == runlist.REMOTE_HOST
    out.mkdir()
    (out / "meta.json").write_text('{"status": "interrupted"}')
    assert runlist.default_runner([], out) == runlist.REMOTE_HOST  # an L0 ended by a host: stops, not redone
    rc["v"] = 3
    assert runlist.default_runner([], out) == "interrupted"
    assert runlist.default_runner([], tmp_path / "none").startswith("failed (exit 3, no meta.json)")


def _live_items():
    return [{"name": "L0/e0", "args": ["--live"], "after": None, "block": 0},
            {"name": "L1_idle/e0", "args": ["--events", "x"], "after": "L0/e0", "block": 0}]


def test_a_runlist_with_a_live_item_does_not_start_while_a_remote_host_runs(tmp_path):
    calls = []
    with pytest.raises(RuntimeError, match="remote-desktop host"):
        runlist.run(_live_items(), [1], tmp_path / "run", runner=lambda a, o: calls.append(o) or "done",
                    remote_check=lambda: ["remoting_host.exe"])
    assert calls == [] and not (tmp_path / "run").exists()  # nothing written: run it again
    runlist.run(_live_items(), [1], tmp_path / "run", runner=lambda a, o: "done", remote_check=lambda: [])
    # without a live item, no check
    runlist.run([{"name": "P5/c0", "args": [], "after": None, "block": 0}], [1], tmp_path / "run2",
                runner=lambda a, o: "done", remote_check=lambda: 1 / 0)


def test_a_remote_host_found_by_the_session_stops_the_runlist_and_counts_nothing(tmp_path):
    calls = []

    def runner(args, out):
        calls.append(out.name)
        return runlist.REMOTE_HOST
    with pytest.raises(RuntimeError, match="the runlist stops here"):
        runlist.run(_live_items(), [1], tmp_path / "run", runner=runner, remote_check=lambda: [])
    rows = [json.loads(x) for x in (tmp_path / "run" / "runlog.ndjson").read_text().splitlines()]
    assert calls == ["e0"] and [r["type"] for r in rows] == ["header", "stopped"]  # no item line, no redo


def _live_hook(end=False, foreign=False, dead=False):
    h = type("H", (), {})()
    h.end, h.foreign = threading.Event(), threading.Event()
    if end:
        h.end.set()
    if foreign:
        h.foreign.set()
    h.p = type("P", (), {"returncode": 1, "poll": lambda self: 1 if dead else None})()
    return h


def test_the_live_wait_stops_on_capture_stop_and_on_a_dead_hook():
    assert session.wait_live(_live_hook(end=True), _Stopper(), 0) is None
    assert session.wait_live(_live_hook(foreign=True), _Stopper(), 0) == {"foreign_injected_key": True}
    assert session.wait_live(_live_hook(), _Stopper(by="capture.stop"), 0) == {"stopped_by": "capture.stop"}
    with pytest.raises(RuntimeError, match="the hook ended"):
        session.wait_live(_live_hook(dead=True), _Stopper(), 0)
    # the end seen with a foreign key or a stop: the interruption is not hidden (Codex, round 3)
    assert session.wait_live(_live_hook(end=True, foreign=True), _Stopper(), 0) == {"foreign_injected_key": True}
    assert session.wait_live(_live_hook(end=True), _Stopper(by="capture.stop"), 0) == {"stopped_by": "capture.stop"}
    # an end with a hook that has ended after writing it: the end, not a failure
    assert session.wait_live(_live_hook(end=True, dead=True), _Stopper(), 0) is None


def test_the_hook_reader_sets_foreign_from_the_hook_s_line():
    h = session.Hook.__new__(session.Hook)
    h.esc, h.end, h.foreign, h.done = (threading.Event() for _ in range(4))
    h.p = type("P", (), {"stdout": io.StringIO("ESC 1\nFOREIGN 2\nDONE\n")})()
    h._read()
    assert h.esc.is_set() and h.foreign.is_set() and h.done.is_set() and not h.end.is_set()


def test_the_end_before_typing_tells_a_stop_from_an_error():
    st = session.ended_before_typing(llm_plan.PlanStopped("x"), "capture.stop")
    assert st == {"status": "interrupted", "stopped_by": "capture.stop", "interrupted": {"during": "M2 planning"}}
    er = session.ended_before_typing(llm_plan.OllamaError("ollama: HTTPError: 500", {"body": {"prompt": "secret"}}),
                                     None)
    assert er == {"status": "failed", "error": "OllamaError: ollama: HTTPError: 500"}  # the record is not in meta
    assert session.ended_before_typing(ConnectionRefusedError("no"), None)["status"] == "failed"


def test_m2_a_stop_during_the_last_attempt_is_a_stop_not_the_counted_failure():
    class LateFlag:
        def __init__(self):
            self.n = 0

        def is_set(self):  # not set before any of the 6 calls, set after the last
            self.n += 1
            return self.n > 6
    log = io.StringIO()
    with pytest.raises(llm_plan.PlanStopped):
        llm_plan.make_plan("m", "Write.", 1, log, generate=two_calls([("ab", '{"gaps_ms": [0]}')] * 3),
                           stop=LateFlag())
    assert json.loads(log.getvalue().splitlines()[-1])["stopped"] is True


def test_m2_summary_of_attempts_skips_stopped_and_raised_lines(tmp_path):
    f = tmp_path / "llm.ndjson"
    rows = [{"type": "header"}, {"type": "attempt", "attempt": 0, "ok": False, "stopped": True},
            {"type": "attempt", "attempt": 0, "ok": False, "raised": True, "reason": "x", "essay": {"error": "x"}}]
    f.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert llm_plan.summarize_attempts(f) == []


def test_ollama_error_reads_as_its_message():
    assert str(llm_plan.OllamaError("ollama: X", {"body": {"prompt": "long"}})) == "ollama: X"


def test_a_footer_without_counts_is_not_complete():
    assert summary.hook_log([{"type": "footer"}])["complete"] is False
    assert summary.hook_log([{"type": "footer", "n_raw": 3, "n_written": 3}])["complete"] is True
    assert summary.hook_log([{"type": "footer", "n_raw": "3", "n_written": "3"}])["complete"] is False


def test_a_write_error_of_the_page_log_interrupts_the_capture(tmp_path):
    st = serve.State(tmp_path / "page.ndjson")
    real = st.f

    class Broken:
        def write(self, t):
            raise OSError("disk full")

        def flush(self):
            pass
    st.f = Broken()
    with pytest.raises(OSError):
        st.keep("log", {"flush": 1, "batch": 0, "entries": [{"t_ms": 1.0}]})
    st.f = real
    st.keep("flushed", {"flush": 1, "sent": 1})
    assert st.last["flushed"]["write_errors"] == 1
    real.close()
    s = {"planned": 1, "hook": {"n": 1}, "page": {"n": 1}}
    p = summary.problems(s, {"mode": "events", "page_log": {"sent": 1, "saved": 1, "batches_failed": 0,
                                                            "write_errors": 1}})
    assert p["page_log_not_all_saved"]["write_errors"] == 1
    src = (ROOT / "capture" / "session.py").read_text(encoding="utf-8")
    assert '"duplicates_dropped", "write_errors")' in src


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


def test_a_stopped_block_is_set_aside_and_run_again(tmp_path):
    """Decision 12: the stopped block's runlog and folders are moved (never deleted), and the block runs again."""
    its = [{"name": "L0/e0", "args": ["--live"], "after": None, "block": 0},
           {"name": "P5/c0", "args": [], "after": None, "block": 1},
           {"name": "L0/e1", "args": ["--live"], "after": None, "block": 1},
           {"name": "P5/c1", "args": [], "after": None, "block": 1}]
    out = tmp_path / "run"
    sha = runlist.runs_sha256(its)

    def ok(args, folder):
        folder.mkdir(parents=True)
        return "done"
    runlist.run(its, [1], out, runner=ok, block=0, remote_check=lambda: [])

    def host_at_l0(args, folder):
        folder.mkdir(parents=True)
        return runlist.REMOTE_HOST if folder.name == "e1" else "done"
    with pytest.raises(RuntimeError, match="the runlist stops here"):
        runlist.run(its, [1], out, runner=host_at_l0, block=1, remote_check=lambda: [])
    with pytest.raises(ValueError, match="has run"):  # as before: not run over
        runlist.run(its, [1], out, runner=ok, block=1, remote_check=lambda: [])
    with pytest.raises(ValueError, match="finished"):  # a finished block is never set aside
        runlist.set_aside(its, out, 0, sha)
    with pytest.raises(ValueError, match="another runs.json"):
        runlist.set_aside(its, out, 1, "0" * 64)
    dest = runlist.set_aside(its, out, 1, sha, now="T")
    rec = json.loads((dest / "set_aside.json").read_text(encoding="utf-8"))
    assert set(rec["moved"]) == {"runlog.b1.ndjson", "P5/c0", "L0/e1"} and rec["stopped_runlog"][-1]["type"] == "stopped"
    assert (dest / "L0" / "e1").is_dir() and (out / "L0" / "e0").is_dir()  # block 0 stays
    res = runlist.run(its, [1], out, runner=ok, block=1, remote_check=lambda: [])
    assert [r["result"] for r in res] == ["done", "done", "done"]
