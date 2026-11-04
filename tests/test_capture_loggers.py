"""Tests of the two loggers' records: kinds, reading, and that no character is kept (capture/hooklog.py,
capture/serve.py, capture/page/index.html, capture/summary.py). The Windows-only part types into the capture page."""

import json
import re
import shutil
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
    # Ctrl held: Backspace, Enter and Delete keep their kind (the page gives the same: test_page_kind_of_runs...)
    ev = [raw(0xA2), raw(0x08), raw(0x0D), raw(0x2E), raw(0xA2, False)]
    assert [r["kind"] for r in hooklog.classify(ev, TABLE) if r["down"]] == ["modifier", "backspace", "enter",
                                                                             "delete"]


def test_hook_repeats_and_injected_flag():
    r = hooklog.classify([raw(0x41), raw(0x41), raw(0x41, False), raw(0x41, flags=0x10)], TABLE)
    assert [x["repeat"] for x in r] == [False, True, False, False]
    assert [x["injected"] for x in r] == [False, False, False, True]


def test_hook_numpad_kinds_match_the_page():
    ev = [raw(0x60 + d) for d in range(10)] + [raw(0x6E), raw(0x6B), raw(0x6F)]
    kinds = [r["kind"] for r in hooklog.classify(ev, TABLE) if r["down"]]
    assert kinds == ["alnum"] * 10 + ["end", "other", "other"]
    assert all("vk" not in r for r in hooklog.classify(ev, TABLE))  # digits keep no virtual key


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
    assert "e.key" not in entry
    assert "r.name = e.key" in js and "if (KEPT_NAMES.has(e.key)) r.name = e.key;" in js  # names only from the list
    assert "ta.value" not in re.search(r'cmd === "flush"(.*?)\}', js, re.S).group(1).replace('ta.value = ""', "")


def page_js_kinds(cases: list[dict]) -> list[list]:
    """Run the page's KEPT_NAMES and kindOf in node on cases {key, ctrlKey, altKey, metaKey}: [kind, name kept]."""
    js = (ROOT / "capture/page/index.html").read_text(encoding="utf-8")
    names = re.search(r"const KEPT_NAMES = new Set\(\[.*?\]\);", js, re.S).group(0)
    kind_of = re.search(r"function kindOf\(e\) \{.*?\n\}", js, re.S).group(0)
    prog = (names + "\n" + kind_of + "\nconst cases = JSON.parse(process.argv[1]);\n"
            "console.log(JSON.stringify(cases.map(e => [kindOf(e), KEPT_NAMES.has(e.key)])));")
    r = subprocess.run(["node", "-e", prog, json.dumps(cases)], capture_output=True, text=True, check=True)
    return json.loads(r.stdout)


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_page_kind_of_runs_and_keeps_no_character():
    c = lambda k, **m: {"key": k, "ctrlKey": False, "altKey": False, "metaKey": False, **m}  # noqa: E731
    cases = [c("a"), c("Z"), c("7"), c(" "), c("."), c("?"), c(","), c("Enter"), c("Backspace"), c("Delete"),
             c("Shift"), c("Control"), c("Process"), c("Unidentified"), c("ArrowLeft"),
             c("z", ctrlKey=True), c("Enter", ctrlKey=True), c("a", altKey=True), c("v", metaKey=True),
             c("\U0001F600"), c("\u00e9"), c("\u2019"), c("Backspace", ctrlKey=True), c("Delete", altKey=True)]
    got = page_js_kinds(cases)
    # Enter, Backspace and Delete keep their kind with Ctrl/Alt held, as the hook gives them (Ctrl+Backspace is one
    # Backspace press: test_hook_kinds_of_vk_packet_and_ctrl_combinations)
    assert [g[0] for g in got] == ["alnum", "alnum", "alnum", "space", "end", "end", "other", "enter", "backspace",
                                   "delete", "modifier", "modifier", "nonchar", "nonchar", "nonchar",
                                   "nonchar", "enter", "nonchar", "nonchar", "nonchar", "other", "other",
                                   "backspace", "delete"]
    kept = [k["key"] for k, g in zip(cases, got) if g[1]]
    assert kept == ["Enter", "Backspace", "Delete", "Shift", "Control", "Process", "Unidentified", "ArrowLeft",
                    "Enter", "Backspace", "Delete"]  # never a character, not even an emoji of length 2


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


def test_uncovered_ms_counts_typing_time_with_no_generation():
    ms = 1_000_000
    assert summary.uncovered_ms(0, 100 * ms, [(-5 * ms, 40 * ms), (45 * ms, 200 * ms)]) == 5.0
    assert summary.uncovered_ms(0, 100 * ms, []) == 100.0
    assert summary.uncovered_ms(0, 100 * ms, [(-1, 50 * ms), (10 * ms, 60 * ms)]) == 40.0  # overlaps counted once


def write_capture(d, meta, hook=None, page=None, load=None, plan_n=3):
    import json as js
    d.mkdir()
    (d / "meta.json").write_text(js.dumps(meta), encoding="utf-8")
    events.write(d / "events.ndjson", events.make([0.0, 100.0, 200.0][:plan_n], ["a", "b", "c"][:plan_n]), "t", "t")
    for name, recs in (("hook.ndjson", hook), ("page.ndjson", page), ("load.ndjson", load)):
        if recs is not None:
            (d / name).write_text("".join(js.dumps(r) + "\n" for r in recs), encoding="utf-8")


def hk(t_ms, injected=True, ours=True, kind="alnum"):
    return {"type": "key", "t_ns": int(t_ms * 1e6), "down": True, "injected": injected, "ours": ours,
            "repeat": False, "kind": kind}


def pg(t_ms, in_box=True):
    return {"type": "key", "t_ms": t_ms, "h_ms": t_ms, "kind": "alnum", "repeat": False, "trusted": True,
            "composing": False, "in_box": in_box}


def test_problems_after_a_capture(tmp_path):
    base = {"mode": "events", "hook_installed": True, "page_opens": 1, "typing_start_perf_ns": 0,
            "typing_end_perf_ns": int(300e6)}
    ok_h, ok_p = [hk(0), hk(100), hk(200)], [pg(0), pg(100), pg(200)]

    def probs(name, meta=None, **kw):
        d = tmp_path / name
        m = {**base, **(meta or {})}
        write_capture(d, m, **{"hook": ok_h, "page": ok_p, **kw})
        return summary.problems(summary.summarize(d), m)

    assert probs("ok") == {}
    assert probs("drops", hook=[hk(0)], page=[pg(0), pg(100)]) == {}  # drops are results, not problems
    assert probs("hook0", hook=[]) == {"hook_recorded_nothing": True}
    assert probs("nohook", meta={"hook_installed": False}, hook=None) == {}
    assert probs("page0", page=[]) == {"page_recorded_nothing": True}
    assert probs("reload", meta={"page_opens": 2}) == {"page_reloaded": 2}
    assert probs("box", page=[pg(0), pg(100, in_box=False), pg(200)]) == {"presses_outside_the_box": 1}
    ld = {"load": {"model": "m"}}
    gen = lambda n, a, b: {"type": "generation", "n": n, "start_perf_ns": int(a * 1e6),  # noqa: E731
                           "end_perf_ns": int(b * 1e6), "tokens": 1, "completed": True}
    start = lambda n, a: {"type": "start", "n": n, "start_perf_ns": int(a * 1e6)}  # noqa: E731
    assert probs("load_ok", meta=ld, load=[start(0, -50), gen(0, -50, 120), start(1, 121)]) == {}
    long_ = {**ld, "typing_end_perf_ns": int(3000e6)}
    assert probs("load_gap", meta=long_, load=[start(0, -50), gen(0, -50, 100)])["load_uncovered_ms"] == 2900.0
    assert probs("load_short_gap", meta=long_, load=[start(0, -50), gen(0, -50, 100), start(1, 900)]) == {}
    assert probs("load_none", meta=ld, load=None) == {"load_recorded_nothing": True}
    err = {"type": "error", "n": 1, "perf_ns": 0, "error": "X: y"}
    assert probs("load_err", meta=ld, load=[start(0, -50), gen(0, -50, 400), err])["load_errors"] == ["X: y"]
    late = {"type": "error", "n": 1, "perf_ns": int(350e6), "error": "X: late"}  # after the typing's end (300 ms)
    assert probs("load_err_late", meta=ld, load=[start(0, -50), gen(0, -50, 400), late]) == {}
    # a generation started before the typing and not ended yet covers the typing (its start line is written at once)
    assert probs("load_running", meta=ld, load=[start(0, -50)]) == {}


def test_live_summary_counts_only_presses_between_signal_and_end(tmp_path):
    d = tmp_path / "live"
    m = {"mode": "live", "hook_installed": True, "page_opens": 1, "typing_start_perf_ns": int(100e6),
         "typing_end_perf_ns": int(500e6)}
    hook = [hk(50, injected=False, ours=False, kind="enter"), hk(200, injected=False, ours=False),
            hk(300, injected=False, ours=False), hk(600, injected=False, ours=False)]
    write_capture(d, m, hook=hook, page=[pg(200), pg(300)])
    s = summary.summarize(d)
    assert s["hook"]["n"] == 2 and s["kinds_equal_hook_page"]


def test_live_presses_outside_the_box_make_the_capture_interrupted(tmp_path):
    d = tmp_path / "live_out"
    m = {"mode": "live", "hook_installed": True, "page_opens": 1, "typing_start_perf_ns": int(100e6),
         "typing_end_perf_ns": int(500e6)}
    write_capture(d, m, hook=[hk(200, injected=False, ours=False), hk(300, injected=False, ours=False)],
                  page=[pg(200), pg(300, in_box=False)])
    s = summary.summarize(d)
    assert s["page"]["not_in_box"] == 1
    assert summary.problems(s, m) == {"presses_outside_the_box": 1}
