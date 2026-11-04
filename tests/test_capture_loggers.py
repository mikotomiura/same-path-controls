"""Tests of the two loggers' records: kinds, reading, and that no character is kept (capture/hooklog.py,
capture/serve.py, capture/page/index.html, capture/summary.py). The Windows-only part types into the capture page."""

import json
import re
import subprocess
import sys
import threading
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from capture import events, hooklog, mouseguard, serve, summary  # noqa: E402

WIN = sys.platform == "win32"
# a US-like layout: a-z at 0x41.., digits at 0x30.., space 0x20, '.' 0xBE, ',' 0xBC, '!' shift+1, '?' shift+0xBF
TABLE = {**{(0x41 + i, False): chr(0x61 + i) for i in range(26)}, **{(0x41 + i, True): chr(0x41 + i) for i in range(26)},
         **{(0x30 + i, False): str(i) for i in range(10)}, (0x20, False): " ", (0xBE, False): ".",
         (0xBC, False): ",", (0x31, True): "!", (0xBF, True): "?"}


def raw(vk, down=True, flags=0, scan=0, t=0):
    return (t, vk, scan, flags, down, 0)


def test_hook_kinds_from_virtual_keys_and_shift():
    ev = [raw(0x41), raw(0x41, False), raw(0xA0), raw(0x31), raw(0x31, False), raw(0xA0, False), raw(0x31),
          raw(0x20), raw(0xBE), raw(0xBC), raw(0x0D), raw(0x08), raw(0x2E), raw(0x1B), raw(0x25)]
    k = [r["kind"] for r in hooklog.classify(ev, TABLE) if r["down"]]
    assert k == ["alnum", "modifier", "end", "alnum", "space", "end", "other", "enter", "backspace", "delete",
                 "nonchar", "nonchar"]


def test_hook_kinds_of_vk_packet_and_ctrl_combinations():
    ev = [raw(0xE7, scan=ord("q")), raw(0xE7, scan=ord("?")), raw(0xE7, scan=ord(" ")),
          raw(0xA2), raw(0x43), raw(0x43, False), raw(0xA2, False)]
    k = [r["kind"] for r in hooklog.classify(ev, TABLE) if r["down"]]
    assert k == ["alnum", "end", "space", "modifier", "nonchar"]


def test_hook_repeats_and_injected_flag():
    r = hooklog.classify([raw(0x41), raw(0x41), raw(0x41, False), raw(0x41, flags=0x10)], TABLE)
    assert [x["repeat"] for x in r] == [False, True, False, False]
    assert [x["injected"] for x in r] == [False, False, False, True]


def test_hook_record_holds_no_character():
    ev = [raw(0x41 + i) for i in range(26)] + [raw(0xE7, scan=ord(c)) for c in "secret words"]
    for r in hooklog.classify(ev, TABLE):
        assert set(r) <= {"type", "t_ns", "down", "injected", "ours", "repeat", "kind", "vk", "packet"}
        assert "vk" not in r  # letters keep no virtual key
        assert r["kind"] in events.KINDS


def test_page_kinds_match_events():
    js = (ROOT / "capture/page/index.html").read_text(encoding="utf-8")
    assert '/^[A-Za-z0-9]$/' in js and 'k === "." || k === "!" || k === "?"' in js
    # the page sends kinds and key names only: the entry is built from these fields
    entry = re.search(r"const r = \{(.*?)\};", js, re.S).group(1)
    assert "e.key" not in entry.replace("kindOf(e.key)", "")
    assert "ta.value" not in re.search(r'cmd === "flush"(.*?)\}', js, re.S).group(1).replace('ta.value = ""', "")


def post(base, path, body):
    req = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=5).read()


def test_server_keeps_page_messages_and_passes_commands(tmp_path):
    p = subprocess.Popen([sys.executable, "-m", "capture.serve", "--out", str(tmp_path / "page.ndjson")], cwd=ROOT,
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        base = f"http://127.0.0.1:{int(p.stdout.readline().split()[1])}"
        assert b"textarea" in urllib.request.urlopen(base + "/", timeout=5).read()
        got = {}
        t = threading.Thread(target=lambda: got.update(json.loads(urllib.request.urlopen(base + "/cmd").read())))
        t.start()
        post(base, "/control", {"cmd": "arm"})
        t.join(5)
        assert got == {"cmd": "arm"}
        post(base, "/open", {"date_now_ms": 1})
        post(base, "/log", {"entries": [{"t_ms": 1.5, "kind": "alnum", "repeat": False, "trusted": True}],
                            "date_now_ms": 2})
        st = json.loads(urllib.request.urlopen(base + "/state", timeout=5).read())
        assert st["counts"]["open"] == 1 and st["counts"]["log"] == 1
    finally:
        p.stdin.close()
        p.wait(10)
    recs = [json.loads(x) for x in (tmp_path / "page.ndjson").read_text().splitlines()]
    assert [r["type"] for r in recs] == ["open", "log", "key"]
    assert all("server_perf_ns" in r for r in recs[:2])
    keys = summary.page_presses(recs)
    assert len(keys) == 1 and keys[0]["kind"] == "alnum"


def test_summary_counts_drops_and_deviation():
    plan_ms = summary.np.array([0.0, 100, 200, 300])
    kinds = ["alnum"] * 4
    full = summary.layer(summary.np.array([5.0, 106, 204, 305]), kinds, plan_ms, kinds)
    assert full["drops"] == 0 and full["gap_deviation"]["computed"]
    assert full["gap_deviation"]["abs_max_ms"] == pytest.approx(2.0)
    short = summary.layer(summary.np.array([5.0, 106, 204]), kinds[:3], plan_ms, kinds)
    assert short["drops"] == 1 and not short["gap_deviation"]["computed"]
    back = summary.layer(summary.np.array([5.0, 3, 204, 305]), kinds, plan_ms, kinds)
    assert back["earlier_than_previous"] == 1


def test_mouse_guard_holds_buttons_and_wheel_not_movement_or_injected():
    for msg in (0x0201, 0x0202, 0x0204, 0x0205, 0x0207, 0x0208, 0x020A, 0x020B, 0x020E):  # buttons, wheel
        assert mouseguard.swallow(msg, 0)
        assert not mouseguard.swallow(msg, mouseguard.LLMHF_INJECTED)
    assert not mouseguard.swallow(mouseguard.WM_MOUSEMOVE, 0)


@pytest.mark.skipif(not WIN, reason="SendInput and WH_KEYBOARD_LL need Windows")
def test_real_press_seen_by_hook_as_injected(tmp_path):
    """The smallest real press: one character, Enter and Delete typed into the capture page (Chrome, auto focus);
    the hook sees each with LLKHF_INJECTED and the injector's mark, the page sees the same kinds."""
    from capture import browser
    if not browser.EXE["chrome"].exists():
        pytest.skip("Chrome is not installed")
    ev = tmp_path / "e.ndjson"
    events.write(ev, events.from_gaps(["x", "Enter", "Delete"], [0, 80, 80]), "t", "test")
    r = subprocess.run([sys.executable, "-m", "capture.session", "--events", str(ev), "--out", str(tmp_path / "o"),
                        "--auto-focus"], cwd=ROOT, capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, r.stdout + r.stderr
    hook = hooklog.read(tmp_path / "o" / "hook.ndjson")
    downs = [x for x in hook if x.get("type") == "key" and x["down"] and x["injected"] and x["ours"]]
    assert [x["kind"] for x in downs] == ["alnum", "enter", "delete"]
    assert [x.get("vk") for x in downs] == [None, 0x0D, 0x2E]
    s = json.loads((tmp_path / "o" / "summary.json").read_text())
    assert s["page"]["n"] == 3 and s["page"]["kinds_equal"] and s["hook"]["kinds_equal"]
    meta = json.loads((tmp_path / "o" / "meta.json").read_text(encoding="utf-8"))
    assert meta["mouse_guard"] == {"used": True, "swallowed": 0}
    text = (tmp_path / "o" / "hook.ndjson").read_text() + (tmp_path / "o" / "page.ndjson").read_text()
    assert '"x"' not in text
