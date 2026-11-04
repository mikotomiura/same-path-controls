"""Tests of the background load (capture/load.py) with a fake ollama: a dead generator fails the load process."""

import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from capture import load, ollama  # noqa: E402


class FakeStream:
    def __init__(self, lines):
        self.lines = [json.dumps(x).encode() for x in lines]

    def __enter__(self):
        return iter(self.lines)

    def __exit__(self, *a):
        return False


def run_load(monkeypatch, tmp_path, streams, capsys):
    calls = iter(streams)

    def open_stream(body):
        s = next(calls)
        if isinstance(s, BaseException):
            raise s
        return FakeStream(s)
    monkeypatch.setattr(ollama, "open_stream", open_stream)
    monkeypatch.setattr(ollama, "model_record", lambda m: {"name": m})
    (tmp_path / "p.txt").write_text("Task.", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["load", "--model", "m", "--prompt-file", str(tmp_path / "p.txt"),
                                      "--out", str(tmp_path / "load.ndjson")])
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))  # the session closes stdin: stop at once
    rc = load.main()
    recs = [json.loads(x) for x in (tmp_path / "load.ndjson").read_text().splitlines()]
    return rc, recs, capsys.readouterr().out


def test_load_error_line_before_any_token_fails_at_start(monkeypatch, tmp_path, capsys):
    rc, recs, out = run_load(monkeypatch, tmp_path, [[{"error": "no model"}]], capsys)
    assert rc == 1 and out.startswith("FAIL load: RuntimeError: ollama: no model")
    assert recs[-1]["type"] == "error"


def test_load_dead_after_ready_is_reported(monkeypatch, tmp_path, capsys):
    import threading
    gate = threading.Event()
    ok = [{"response": "a", "done": False}, {"response": "", "done": True}]

    class Slow:
        def __enter__(self):
            gate.wait(5)
            raise ConnectionError("gone")

        def __exit__(self, *a):
            return False
    calls = iter([FakeStream(ok), Slow()])
    monkeypatch.setattr(ollama, "open_stream", lambda body: next(calls))
    monkeypatch.setattr(ollama, "model_record", lambda m: {"name": m})
    (tmp_path / "p.txt").write_text("Task.", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["load", "--model", "m", "--prompt-file", str(tmp_path / "p.txt"),
                                      "--out", str(tmp_path / "load.ndjson")])

    class Stdin:
        def read(self):
            gate.set()  # the generator dies while the capture runs
            import time
            time.sleep(0.3)
            return ""
    monkeypatch.setattr(sys, "stdin", Stdin())
    rc = load.main()
    recs = [json.loads(x) for x in (tmp_path / "load.ndjson").read_text().splitlines()]
    # after READY the exit code is 0: the error line, with its time, is judged by the session against the typing
    assert rc == 0 and recs[-1]["type"] == "error" and "ConnectionError" in recs[-1]["error"]
    assert isinstance(recs[-1]["perf_ns"], int)
    assert [r["type"] for r in recs if r.get("type") in ("start", "generation")][:3] == ["start", "generation",
                                                                                         "start"]
