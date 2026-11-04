"""Tests of the fixes of the third handoff (2026-11-04, the Mac review's findings that need Windows): the browser's
flags and stale profiles, a stop during a long wait, the page log's acknowledged batches, and the hook's records
written while it runs. Each test fails with the fix taken out."""

import json
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from capture import inject, llm_stream, summary  # noqa: E402

MS = 1_000_000
win = pytest.mark.skipif(sys.platform != "win32", reason="Windows only")


class Flag:
    def __init__(self):
        self.e = threading.Event()

    def is_set(self):
        return self.e.is_set()

    def set_after(self, s):
        threading.Timer(s, self.e.set).start()


# --- 1. the browser's flags and stale profiles ---

@win
def test_browser_opens_the_page_as_an_app_offline_and_incognito_only_live():
    from capture import browser as br
    for live in (False, True):
        b = br.Browser("chrome", "http://127.0.0.1:1234/", incognito=live)
        try:
            a = b.args()
            assert a[-1] == "--app=http://127.0.0.1:1234/"
            assert "--new-window" not in a and "http://127.0.0.1:1234/" not in a
            assert "--host-resolver-rules=MAP * ~NOTFOUND , EXCLUDE 127.0.0.1" in a
            assert ("--incognito" in a) == live
        finally:
            b.profile.rmdir()


def test_stale_profiles_are_deleted_before_a_capture(tmp_path):
    from capture import browser as br
    old = tmp_path / (br.PROFILE_PREFIX + "abc")
    (old / "Default").mkdir(parents=True)
    (old / "Default" / "History").write_text("x")
    other = tmp_path / "keep-me"
    other.mkdir()
    res = br.remove_stale_profiles(tmp_path)
    assert res["found"] == 1 and res["deleted"] == [str(old)] and not res["failed"]
    assert not old.exists() and other.exists()


# --- 3. a stop during a long wait ---

def test_wait_until_returns_at_once_when_stopped_during_a_long_wait():
    stop = Flag()
    stop.set_after(0.1)
    t0 = time.perf_counter()
    assert inject.wait_until(time.perf_counter_ns() + 3000 * MS, stop) is True
    assert time.perf_counter() - t0 < 0.5


def test_wait_until_without_stop_waits_to_the_target():
    t0 = time.perf_counter_ns()
    assert inject.wait_until(t0 + 30 * MS, None) is False
    assert time.perf_counter_ns() - t0 >= 30 * MS


class FakeInj:
    sent: list = []

    def __init__(self, method):
        self.method = method

        class U:
            @staticmethod
            def GetForegroundWindow():
                return 1
        self.user32 = U()

    def inputs_for(self, key):
        return [0, 0]

    def is_fallback(self, key):
        return False

    def send(self, key):
        t = time.perf_counter_ns()
        FakeInj.sent.append(key)
        return t, t, 2, 2

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_type_schedule_sends_nothing_after_a_stop_during_a_rest(monkeypatch):
    monkeypatch.setattr(inject, "Injector", FakeInj)
    FakeInj.sent = []
    stop = Flag()
    stop.set_after(0.1)
    t0 = time.perf_counter()
    presses = [{"i": 0, "key": "a", "t_ms": 0.0}, {"i": 1, "key": "b", "t_ms": 3000.0}]
    recs, summ = inject.type_schedule(presses, "unicode", stop=stop, guard_hwnd=1)
    assert FakeInj.sent == ["a"] and summ["stopped_by_esc"] and summ["attempted"] == 1
    assert time.perf_counter() - t0 < 0.5


def test_typist_sends_nothing_after_a_stop_during_a_rest():
    FakeInj.sent = []
    stop = Flag()
    ty = llm_stream.Typist(FakeInj("unicode"), guard_hwnd=1, stop=stop)
    ty.a1_ns = time.perf_counter_ns()
    ty.q.put({"key": "a", "wished_ms": 0.0, "plan_ms": 0.0, "token": 1})
    ty.q.put({"key": "b", "wished_ms": 3000.0, "plan_ms": 3000.0, "token": 2})
    ty.q.put(None)
    stop.set_after(0.1)
    t0 = time.perf_counter()
    ty.start()
    ty.join(5)
    assert FakeInj.sent == ["a"] and ty.stopped_by_esc
    assert time.perf_counter() - t0 < 0.5


@win
def test_capture_stop_sets_the_named_event_the_session_watches():
    from capture import stop
    name = "Local\\v7-capture-stop-test"
    s = stop.StopSignal(None, name=name)
    try:
        assert not s.is_set()
        subprocess.run([sys.executable, "-c", f"from capture import stop; stop.signal({name!r})"], cwd=ROOT,
                       check=True)
        assert s.is_set() and s.by == "capture.stop"
    finally:
        s.close()
    esc = threading.Event()
    s = stop.StopSignal(esc, name=name)  # a new session resets the event: an earlier stop does not stop it
    try:
        assert not s.is_set()
        esc.set()
        assert s.is_set() and s.by == "esc"
    finally:
        s.close()


# --- 5. the page log's acknowledged batches ---

def _post(base, path, body):
    req = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=5).read() or b"{}")


def test_server_acknowledges_batches_and_drops_one_sent_again(tmp_path):
    p = subprocess.Popen([sys.executable, "-m", "capture.serve", "--out", str(tmp_path / "page.ndjson")], cwd=ROOT,
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        base = f"http://127.0.0.1:{int(p.stdout.readline().split()[1])}"
        e = {"t_ms": 1.5, "kind": "alnum", "repeat": False, "trusted": True}
        assert _post(base, "/log", {"flush": 1, "batch": 0, "n_batches": 2, "entries": [e, e]}) == {"saved": 0}
        # batch 0 again (its acknowledgement was lost): acknowledged, not written twice
        assert _post(base, "/log", {"flush": 1, "batch": 0, "n_batches": 2, "entries": [e, e]})["saved"] == 0
        assert _post(base, "/log", {"flush": 1, "batch": 1, "n_batches": 2, "entries": [e]}) == {"saved": 1}
        _post(base, "/flushed", {"flush": 1, "n_batches": 2, "sent": 3, "batches_failed": 0, "batches_resent": 1})
        st = json.loads(urllib.request.urlopen(base + "/state", timeout=5).read())
        fl = st["last"]["flushed"]
        assert fl["sent"] == 3 and fl["saved"] == 3 and fl["duplicates_dropped"] == 1
    finally:
        p.stdin.close()
        p.wait(10)
    recs = [json.loads(x) for x in (tmp_path / "page.ndjson").read_text().splitlines()]
    assert [r["type"] for r in recs] == ["log", "key", "key", "log", "key"]  # 'flushed' is not a line of the file


def test_a_page_log_not_all_saved_interrupts_the_capture():
    s = {"planned": 3, "hook": {"n": 3}, "page": {"n": 2}}
    assert summary.problems(s, {"mode": "events", "page_log": {"sent": 3, "saved": 3, "batches_failed": 0}}) == {}
    p = summary.problems(s, {"mode": "events", "page_log": {"sent": 3, "saved": 2, "batches_failed": 1}})
    assert p["page_log_not_all_saved"] == {"sent": 3, "saved": 2, "batches_failed": 1}


# --- 4. the hook's records are written while it runs ---

@win
def test_hook_writes_records_while_it_runs_and_keeps_its_callback_free_of_output(tmp_path):
    import ctypes
    out = tmp_path / "hook.ndjson"
    p = subprocess.Popen([sys.executable, "-m", "capture.hooklog", "--out", str(out)], cwd=ROOT,
                         stdout=subprocess.PIPE, text=True)
    try:
        tid = int(p.stdout.readline().split()[1])
        u = ctypes.WinDLL("user32")
        VK_F24 = 0x87  # a key that types nothing and that no application of this PC uses
        u.keybd_event(VK_F24, 0, 0, 0)
        u.keybd_event(VK_F24, 0, 2, 0)
        time.sleep(1.5)  # more than the writer's period
        rows = [json.loads(x) for x in out.read_text().splitlines() if x.strip()]
        assert rows[0]["type"] == "header"
        assert sum(r.get("type") == "key" for r in rows) >= 2  # on disk before the hook stops
        assert rows[-1].get("type") != "footer"
    finally:
        ctypes.WinDLL("user32").PostThreadMessageW(tid, 0x0012, 0, 0)
        p.wait(10)
    rows = [json.loads(x) for x in out.read_text().splitlines() if x.strip()]
    assert rows[-1]["type"] == "footer" and rows[-1]["n_raw"] == rows[-1]["n_written"] >= 2
    src = (ROOT / "capture" / "hooklog.py").read_text(encoding="utf-8")
    body = src[src.index("    def proc(code"):src.index("    def reporter")]
    assert "print(" not in body and ".write(" not in body


# --- 7. a stalled background generation does not cover the typing ---

def _load_summary(tmp_path, load_rows, t0, t1):
    from capture import events
    d = tmp_path / "cap"
    d.mkdir(parents=True)
    events.write(d / "events.ndjson", events.cycle_letters(3, 10.0), "P", "test")
    (d / "meta.json").write_text(json.dumps({"mode": "events", "typing_start_perf_ns": t0, "typing_end_perf_ns": t1,
                                             "load": {"model": "m"}}))
    (d / "load.ndjson").write_text("".join(json.dumps(r) + "\n" for r in load_rows))
    return summary.summarize(d)["load"]


def test_a_stall_inside_a_generation_is_not_covered(tmp_path):
    S = 1_000_000_000
    rows = [{"type": "header"}, {"type": "start", "n": 0, "start_perf_ns": 0},
            {"type": "stall", "n": 0, "from_perf_ns": 2 * S, "to_perf_ns": 5 * S},
            {"type": "generation", "n": 0, "start_perf_ns": 0, "end_perf_ns": 10 * S, "tokens": 9, "completed": True}]
    ld = _load_summary(tmp_path, rows, 1 * S, 9 * S)
    assert ld["uncovered_ms"] == pytest.approx(3000.0)
    assert ld["stalls_during_typing"]["n"] == 1
    assert summary.problems({"planned": 0, "load": ld}, {"mode": "events", "load": {"model": "m"},
                                                          "hook_installed": False})["load_uncovered_ms"] > 1000


def test_a_generation_stalled_at_the_end_of_the_typing_is_not_covered(tmp_path):
    S = 1_000_000_000
    # still running at the end of the typing (no generation line), its last token at 4 s, typing to 9 s
    rows = [{"type": "header"}, {"type": "start", "n": 0, "start_perf_ns": 0},
            {"type": "progress", "n": 0, "last_token_perf_ns": 4 * S, "tokens": 50}]
    ld = _load_summary(tmp_path, rows, 1 * S, 9 * S)
    assert ld["uncovered_ms"] == pytest.approx(5000.0)
    rows[-1]["last_token_perf_ns"] = 9 * S - 100_000_000  # tokens up to 0.1 s before the end: covered
    ld = _load_summary(tmp_path / "b", rows, 1 * S, 9 * S)
    assert ld["uncovered_ms"] == pytest.approx(0.0)
