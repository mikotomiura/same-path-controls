"""One capture (capture spec section 5).

    python -m capture.session --events <plan.ndjson> --out <dir>
    python -m capture.session --llm-stream --model <m> --prompt-file <p.txt> --seed <s> --out <dir>
    python -m capture.session --llm-plan  --model <m> --prompt-file <p.txt> --seed <s> --out <dir>
    python -m capture.session --live --out <dir>
  common: --load <model> --load-prompt-file <p.txt>   background inference load
          --no-hook                                   the browser page only (P_nohook, amendments B11 and B14)
          --browser chrome|edge  --inject unicode|vk  --auto-focus
          --no-mouse-guard                            do not hold the mouse while typing (held by default; held in
                                                      every production capture: the researcher, 2026-11-04)

Order: (M2: the plan is made and the model unloaded first) monitor -> load -> server -> browser on the page -> hook ->
'click in the box and press Enter' (or --auto-focus: the session brings the page to the front and focuses the box)
-> checks (the page window in front, the box focused, the IME closed) -> the page is armed -> 3 s signal -> typing ->
1 s -> the page sends its log -> everything stopped. The output folder must not exist.

Stops: a non-injected Esc seen by the hook, or the page window leaving the foreground (checked before every press).
Both end the capture as 'interrupted' (exit code 3); presses not typed are counted, never skipped silently.
Live (L0): nothing is typed; the end signal is Ctrl+Alt+End (seen by the hook, non-injected).
"""

from __future__ import annotations

import argparse
import ctypes
import json
import platform
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from capture import events, hooklog, summary  # noqa: E402

PY = sys.executable
INTERRUPTED = 3
COUNTDOWN_S = 3
# The session's interpreter switch interval (sys.setswitchinterval), set before anything is typed. In M1 the typist
# thread waits for the interpreter lock while the stream reader holds it; Python's default lets it wait up to 5 ms.
# Compared on the work PC (M1 smoke, qwen2.5:3b, two captures each, 2026-11-04, session notes v3): the wait of presses
# that were not queued behind an earlier send (queue length 0) had a median of 0.007 ms and a 95th percentile of
# 0.06-0.08 ms with both 5 ms and 0.5 ms, and the longer waits came from the queue (plan collisions), not from the
# lock. 0.5 ms gave no smaller wait, so Python's default (5 ms) is kept, and written to meta.json.
SWITCH_INTERVAL_MS = 5.0
AFTER_S = 1.0


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def run_text(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout.strip()
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"unavailable: {e}"


def git_state() -> dict:
    commit = run_text(["git", "-C", str(ROOT), "rev-parse", "HEAD"])
    dirty = run_text(["git", "-C", str(ROOT), "status", "--porcelain"])
    return {"git_commit": commit, "git_dirty": bool(dirty), "git_branch":
            run_text(["git", "-C", str(ROOT), "rev-parse", "--abbrev-ref", "HEAD"])}


def boot_time_utc() -> str:
    ms = ctypes.windll.kernel32.GetTickCount64
    ms.restype = ctypes.c_ulonglong
    t = time.time() - ms() / 1000.0
    return datetime.fromtimestamp(t, timezone.utc).isoformat(timespec="seconds")


def machine() -> dict:
    import numpy
    smi = run_text(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,temperature.gpu,utilization.gpu",
                    "--format=csv,noheader"])
    head = run_text(["nvidia-smi"])
    cuda = next((ln.split("CUDA Version:")[1].strip(" |") for ln in head.splitlines() if "CUDA Version:" in ln), None)

    class MEM(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + \
                   [(n, ctypes.c_ulonglong) for n in ("total", "avail", "tpage", "apage", "tvirt", "avirt", "aext")]

    m = MEM()
    m.dwLength = ctypes.sizeof(MEM)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    ollama_start = run_text(["powershell", "-NoProfile", "-Command",
                             "Get-Process ollama* -ErrorAction SilentlyContinue | ForEach-Object "
                             "{ $_.ProcessName + ' ' + $_.StartTime.ToUniversalTime().ToString('o') }"])
    return {"computer": platform.node(), "os": platform.platform(), "os_version": platform.version(),
            "cpu": run_text(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"]),
            "ram_bytes": m.total, "python": platform.python_version(), "numpy": numpy.__version__,
            "gpu": smi, "cuda": cuda, "boot_time_utc": boot_time_utc(), "ollama_processes_start_utc": ollama_start}


class Child:
    """A child process of the capture (server, hook, monitor, load) with its first line read."""

    def __init__(self, args: list[str], name: str):
        self.name = name
        self.p = subprocess.Popen([PY, "-m", *args], cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  text=True, encoding="utf-8")
        self.first = self.p.stdout.readline().strip()
        if not self.first or self.first.startswith("FAIL"):
            self.p.kill()
            raise RuntimeError(f"{name} did not start: {self.first!r}")

    def stop(self, timeout: float = 30) -> None:
        if self.p.poll() is None:
            try:
                self.p.stdin.close()
            except OSError:
                pass
            try:
                self.p.wait(timeout)
            except subprocess.TimeoutExpired:
                self.p.kill()
                raise RuntimeError(f"{self.name} did not stop in {timeout} s") from None
        if self.p.returncode not in (0, None):
            raise RuntimeError(f"{self.name} ended with {self.p.returncode}")


class Hook(Child):
    def __init__(self, out: Path):
        super().__init__(["capture.hooklog", "--out", str(out)], "hook")
        self.tid = int(self.first.split()[1])
        self.esc = threading.Event()
        self.end = threading.Event()
        self.done = threading.Event()
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        for line in self.p.stdout:
            w = line.split()
            if not w:
                continue
            if w[0] == "ESC":
                self.esc.set()
            elif w[0] == "END":
                self.end.set()
            elif w[0] == "DONE":
                self.done.set()

    def stop(self, timeout: float = 30) -> None:
        ctypes.WinDLL("user32").PostThreadMessageW(self.tid, 0x0012, 0, 0)  # WM_QUIT
        if not self.done.wait(timeout):
            self.p.kill()
            raise RuntimeError("the hook did not write its records")
        self.p.wait(timeout)


class MouseGuard(Child):
    """capture.mouseguard: holds mouse buttons and wheel while presses are typed (not in the live capture)."""

    def __init__(self):
        super().__init__(["capture.mouseguard"], "mouseguard")
        self.tid = int(self.first.split()[1])
        self.swallowed: int | None = None

    def stop(self, timeout: float = 30) -> None:
        if self.p.poll() is not None:
            raise RuntimeError(f"the mouse guard ended early ({self.p.returncode})")
        ctypes.WinDLL("user32").PostThreadMessageW(self.tid, 0x0012, 0, 0)  # WM_QUIT
        out, _ = self.p.communicate(timeout=timeout)
        w = out.split()
        if len(w) < 2 or w[0] != "DONE":
            raise RuntimeError(f"the mouse guard did not report: {out!r}")
        self.swallowed = int(w[1])


class Page:
    def __init__(self, port: int):
        self.base = f"http://127.0.0.1:{port}"

    def cmd(self, c: str) -> None:
        req = urllib.request.Request(self.base + "/control", json.dumps({"cmd": c}).encode(),
                                     {"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10).read()

    def state(self) -> dict:
        return json.loads(urllib.request.urlopen(self.base + "/state", timeout=10).read())

    def wait(self, what: str, at_least: int, timeout: float) -> dict:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            s = self.state()
            if s["counts"][what] >= at_least:
                return s
            time.sleep(0.05)
        raise TimeoutError(f"the page sent no '{what}' in {timeout} s")


def parse(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="python -m capture.session")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--events", type=Path)
    mode.add_argument("--llm-stream", action="store_true")
    mode.add_argument("--llm-plan", action="store_true")
    mode.add_argument("--live", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model")
    ap.add_argument("--prompt-file", type=Path)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--load")
    ap.add_argument("--load-prompt-file", type=Path)
    ap.add_argument("--no-hook", action="store_true")
    ap.add_argument("--browser", default="chrome", choices=["chrome", "edge"])  # Chrome, decided by the researcher 2026-11-04
    ap.add_argument("--inject", default="unicode", choices=["unicode", "vk"])
    ap.add_argument("--auto-focus", action="store_true")  # the start used in production (researcher, 2026-11-04)
    ap.add_argument("--no-mouse-guard", action="store_true")
    ap.add_argument("--ready-timeout", type=float, default=300.0)
    ap.add_argument("--switch-interval-ms", type=float, default=SWITCH_INTERVAL_MS)
    a = ap.parse_args(argv)
    if (a.llm_stream or a.llm_plan) and not (a.model and a.prompt_file and a.seed is not None):
        ap.error("--llm-stream and --llm-plan need --model, --prompt-file and --seed")
    if bool(a.load) != bool(a.load_prompt_file):
        ap.error("--load and --load-prompt-file go together")
    if a.live and a.no_hook:
        ap.error("--live needs the hook (its end signal and its times)")
    return a


def main(argv=None) -> int:
    a = parse(argv)
    if sys.platform != "win32":
        raise SystemExit("the capture runs on Windows only")
    from capture import browser as br
    from capture import inject, llm_plan, llm_stream, ollama
    from capture.stop import StopSignal

    out: Path = a.out
    out.mkdir(parents=True, exist_ok=False)  # never over an earlier capture
    sys.setswitchinterval(a.switch_interval_ms / 1000.0)
    mode = "live" if a.live else "llm_stream" if a.llm_stream else "llm_plan" if a.llm_plan else "events"
    meta: dict = {"mode": mode, "command": [PY, "-m", "capture.session", *(argv if argv is not None else sys.argv[1:])],
                  "start_utc": now_utc(), "machine": machine(), **git_state(), "status": "running",
                  "injection": None if a.live else {"method": a.inject, "named_keys": "virtual keys (VK_RETURN, "
                                                    "VK_BACK, VK_DELETE)", "one_press": "down and up in one SendInput"},
                  "switch_interval_s": sys.getswitchinterval(), "hook_installed": not a.no_hook, "load": {"model": a.load} if a.load else None,
                  "restarts": {"browser": "started fresh for this capture (new temporary profile)",
                               "page": None, "ollama": None, "machine_boot_utc": None}}
    meta["restarts"]["machine_boot_utc"] = meta["machine"]["boot_time_utc"]
    meta["restarts"]["ollama"] = meta["machine"]["ollama_processes_start_utc"]

    def save_meta():
        (out / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")

    save_meta()
    presses: list[dict] | None = None
    llm_log = None
    if a.events:
        header, presses = events.read(a.events)
        events.write(out / "events.ndjson", presses, header.get("series", ""), header.get("source", ""),
                     copied_from=str(a.events))
        src_meta = Path(a.events).parent / "meta.json"
        if src_meta.exists():  # an M2 redone by the runlist: the plan's record of the first attempt (review 2026-11-04)
            sm = json.loads(src_meta.read_text(encoding="utf-8"))
            if "llm_plan" in sm:
                meta["plan_from"] = {"folder": str(Path(a.events).parent), "mode": sm.get("mode"),
                                     "llm_plan": sm["llm_plan"],
                                     "ollama_ps_before_capture": sm.get("ollama_ps_before_capture")}
    if a.llm_stream or a.llm_plan:
        task = a.prompt_file.read_text(encoding="utf-8").strip()
        llm_log = open(out / "llm.ndjson", "x", encoding="utf-8", newline="\n")
        llm_log.write(json.dumps({"type": "header", "model": ollama.model_record(a.model), "ollama": ollama.version(),
                                  "temperature": ollama.TEMPERATURE, "seed": a.seed,
                                  "essay_prompt": ollama.essay_prompt(task),
                                  "prompt": (llm_plan.plan_prompt(ollama.essay_prompt(task)) if a.llm_plan
                                             else ollama.essay_prompt(task))}) + "\n")
        meta["ollama"] = ollama.version()
    if a.llm_plan:
        res = llm_plan.make_plan(a.model, ollama.essay_prompt(task), a.seed, llm_log)
        meta["llm_plan"] = {"ok": res["ok"], "attempts": res["attempts"], "failures": res["failures"],
                            "skipped_chars": res["skipped_chars"]}
        ollama.unload(a.model)
        time.sleep(1.0)
        meta["ollama_ps_before_capture"] = ollama.ps()
        if not res["ok"]:
            meta["status"] = "m2_failed"
            meta["end_utc"] = now_utc()
            llm_log.close()
            save_meta()
            print("M2: the plan failed 3 times; counted, nothing typed")
            return 0
        presses = res["presses"]
        events.write(out / "events.ndjson", presses, "M2", f"llm_plan {a.model} seed {a.seed}")
        save_meta()

    kids: list = []
    b = None
    hook = None
    guard = None
    meta["mouse_guard"] = {"used": not (a.live or a.no_mouse_guard)}
    try:
        mon = Child(["capture.monitor", "--out", str(out / "monitor.ndjson")], "monitor")
        kids.append(mon)
        if a.load:
            load = Child(["capture.load", "--model", a.load, "--prompt-file", str(a.load_prompt_file),
                          "--out", str(out / "load.ndjson")], "load")
            kids.append(load)
        srv = Child(["capture.serve", "--out", str(out / "page.ndjson")], "server")
        kids.append(srv)
        page = Page(int(srv.first.split()[1]))
        stale = br.remove_stale_profiles()  # an earlier capture's profile is deleted before this one starts
        if stale["failed"]:
            raise RuntimeError(f"an earlier capture's browser profile could not be deleted: {stale['failed']}")
        b = br.Browser(a.browser, page.base + "/", incognito=a.live)
        meta["browser"] = {"name": a.browser, "version": b.version, "exe": str(b.exe), "args": b.args()[1:],
                           "stale_profiles_deleted": stale["deleted"]}
        hwnd = b.start()
        opened = page.wait("open", 1, 30)["last"]["open"]
        # the GPU that draws the page (WebGL's unmasked renderer, as chrome://gpu shows it): the display is on the
        # integrated Intel GPU and the LLM on the RTX; recorded per capture so that a change is seen afterwards
        meta["browser"]["gl_renderer"] = opened.get("gl_renderer")
        meta["browser"]["gl_vendor"] = opened.get("gl_vendor")
        meta["browser"]["user_agent"] = opened.get("user_agent")
        if not a.no_hook:
            hook = Hook(out / "hook.ndjson")
        if a.auto_focus:
            br.foreground(hwnd)
            page.cmd("focus")
            page.wait("status", 1, 10)
        else:
            print("Click in the box of the page and press Enter.", flush=True)
            br.foreground(hwnd)
            page.wait("ready", 1, a.ready_timeout)
        ime_before, ime_after = br.ime_close(hwnd)
        n_status = page.state()["counts"]["status"]
        page.cmd("status")
        st = page.wait("status", n_status + 1, 10)["last"]["status"]
        fg = br._user32().GetForegroundWindow() == hwnd
        meta["checks"] = {"foreground": fg, "page_has_focus": st["has_focus"], "box_active": st["box_active"],
                          "ime_open_before": ime_before, "ime_open_after": ime_after}
        if not (fg and st["has_focus"] and st["box_active"] and ime_after == 0):
            raise RuntimeError(f"the page is not ready to receive presses: {meta['checks']}")
        def arm() -> None:
            n_armed = page.state()["counts"]["armed"]
            page.cmd("arm")
            page.wait("armed", n_armed + 1, 10)

        if not a.live:
            arm()
        if meta["mouse_guard"]["used"]:
            guard = MouseGuard()
        for k in range(COUNTDOWN_S, 0, -1):
            print(f"{k}...", flush=True)
            time.sleep(1.0)
        if a.live:  # L0: the page records from the signal, as the hook layer is read from it (review of 2026-11-04)
            arm()
        # Esc seen by the hook, or 'python -m capture.stop' from another console (the only way with --no-hook)
        stop = stopper = StopSignal(hook.esc if hook else None)
        if not a.live:
            print("To stop: Esc" + (" (not seen without the hook)" if hook is None else "")
                  + ", or 'uv run python -m capture.stop' in another console.", flush=True)
        meta["typing_start_utc"] = now_utc()
        meta["typing_start_perf_ns"] = time.perf_counter_ns()
        interrupted = None
        if a.live:
            print("Type now. End with Ctrl+Alt+End.", flush=True)
            hook.end.wait()
        elif a.llm_stream:
            res = llm_stream.run(a.model, ollama.essay_prompt(task), a.seed, a.inject, hwnd, stop, llm_log)
            c = res["counts"]
            meta["llm_stream"] = c
            plan = events.make([x["plan_ms"] for x in res["plan"]], [x["key"] for x in res["plan"]])
            events.write(out / "schedule.ndjson", plan, "M1", f"llm_stream {a.model} seed {a.seed}",
                         rule="amendment B13: token k from a_{k+1} over a_{k+1}-a_k; running maximum (B11, B14)")
            with open(out / "inject.ndjson", "x", encoding="utf-8", newline="\n") as f:
                for r in res["records"]:
                    f.write(json.dumps(r) + "\n")
            if c["stopped_by_esc"] or c["stopped_focus_lost"] or c["typed"] < c["planned"] or c["not_sent"] or \
                    c["planned"] == 0:  # nothing typable: no capture (as M2's plan of no typable character)
                interrupted = {k: c[k] for k in ("stopped_by_esc", "stopped_focus_lost", "typed", "planned",
                                                 "not_sent")}
            if not c["generation_done"]:
                interrupted = {**(interrupted or {}), "generation_done": False}
        else:
            recs, summ = inject.type_schedule(presses, a.inject, stop=stop, guard_hwnd=hwnd)
            meta["injection"]["fallback_unicode"] = summ["fallback_unicode"]
            meta["inject_summary"] = summ
            with open(out / "inject.ndjson", "x", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps({"type": "header", **summ}) + "\n")
                for r in recs:
                    f.write(json.dumps(r) + "\n")
            if summ["stopped_by_esc"] or summ["stopped_focus_lost"] or summ["not_sent"] or \
                    summ["attempted"] < summ["planned"]:
                interrupted = summ
        meta["typing_end_perf_ns"] = time.perf_counter_ns()
        meta["stopped_by"] = stopper.by
        stopper.close()
        meta["typing_end_utc"] = now_utc()
        if guard is not None:
            guard.stop()
            meta["mouse_guard"]["swallowed"] = guard.swallowed
            guard = None
        time.sleep(AFTER_S)
        n_flushed = page.state()["counts"]["flushed"]
        page.cmd("flush")
        fl = page.wait("flushed", n_flushed + 1, 60)["last"]["flushed"]
        # what the page sent against what the server wrote (acknowledged batches; a difference interrupts)
        meta["page_log"] = {k: fl.get(k) for k in ("sent", "saved", "n_batches", "batches_failed", "batches_resent",
                                                   "duplicates_dropped")}
        meta["page_opens"] = page.state()["counts"]["open"]
        meta["restarts"]["page"] = "reloaded during the capture" if meta["page_opens"] > 1 else "not reloaded"
        if hook is not None:  # the hook writes hook.ndjson only when it stops: stopped here, before the checks read it
            h, hook = hook, None
            h.stop()
        # checks after the capture (review of 2026-11-04): a logger that recorded nothing, a reloaded page, presses
        # outside the box or a load that stopped make the capture 'interrupted' (redone once by the runlist). The load
        # file is read while the load still runs: its finished generations are flushed line by line.
        save_meta()
        after = summary.problems(summary.summarize(out), meta)
        meta["checks_after"] = after
        if after:
            interrupted = {**(interrupted or {}), "checks_after": after}
        meta["status"] = "interrupted" if interrupted else "done"
        if interrupted:
            meta["interrupted"] = interrupted
        if a.live and meta["status"] == "done":  # L1's plan: runs.json gives L1 --events {out:L0/eN}/l1_events.ndjson
            l1, left = events.from_live(hooklog.read(out / "hook.ndjson"), meta["typing_start_perf_ns"],
                                        meta["typing_end_perf_ns"])
            events.write(out / "l1_events.ndjson", l1, "L1", "from_live of this L0's hook.ndjson", left_out=left)
            meta["l1_events"] = {"n": len(l1), "left_out": left}
    except BaseException as e:
        meta["status"] = "failed"
        meta["error"] = f"{type(e).__name__}: {e}"
        raise
    finally:
        errors = []
        if guard is not None:  # an exception during typing: release the mouse first
            try:
                guard.stop()
                meta["mouse_guard"]["swallowed"] = guard.swallowed
            except Exception as e:  # noqa: BLE001 - collected and raised below
                errors.append(f"mouseguard: {e}")
        if hook is not None:
            try:
                hook.stop()
            except Exception as e:  # noqa: BLE001 - collected and raised below
                errors.append(f"hook: {e}")
        if b is not None:
            try:
                page.cmd("quit")
            except Exception:  # noqa: BLE001 - the page may be gone already; the browser is closed next
                pass
            try:
                b.close()
            except Exception as e:  # noqa: BLE001
                errors.append(f"browser: {e}")
        for k in reversed(kids):
            try:
                k.stop()
            except Exception as e:  # noqa: BLE001
                errors.append(f"{k.name}: {e}")
        if llm_log is not None and not llm_log.closed:
            llm_log.close()
        meta["end_utc"] = now_utc()
        if errors:
            meta["stop_errors"] = errors
            if meta["status"] in ("done", "interrupted"):
                meta["status"] = "failed"
        save_meta()
    s = summary.summarize(out)
    (out / "summary.json").write_text(json.dumps(s, indent=1), encoding="utf-8")
    print(json.dumps(s, indent=1))
    if meta["status"] == "failed":
        return 1
    return INTERRUPTED if meta["status"] == "interrupted" else 0


if __name__ == "__main__":
    raise SystemExit(main())
