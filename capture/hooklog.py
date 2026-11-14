"""The OS layer (capture spec section 3a): WH_KEYBOARD_LL in its own process, key downs and ups with perf_counter_ns.

Run as a child of the session:  python -m capture.hooklog --out hook.ndjson
It prints 'READY <thread id>' once the hook is installed, 'ESC <ns>' when a non-injected Esc goes down (the stop
flag of the injector), 'END <ns>' on a non-injected Ctrl+Alt+End (the end signal of the live capture L0) and
'FOREIGN <ns>' on an injected key down without the injector's mark that is not a modifier (keys through a remote desktop
come injected: returns-v3); all are printed by a thread of their own, never from the callback. The records are appended to the file in batches while the
hook runs; the session stops it with PostThreadMessageW(thread id, WM_QUIT), and the last batch and the footer are
written then.

[仮] The spec asks for a dedicated thread; a dedicated process is used instead, so that the injector's waiting and the
LLM reader of the session never hold the interpreter lock that the hook's callback needs (both use perf_counter_ns,
which is one clock for the whole machine).

What is kept per event: the time, down or up, LLKHF_INJECTED, whether dwExtraInfo carries the injector's mark, whether
the down repeats a key already down, and the kind. The character is never kept: vkCode is written only for Enter,
Backspace, Delete, Esc and modifiers, and VK_PACKET's Unicode (in scanCode) is used for the kind and dropped.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
import time
from ctypes import wintypes
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from capture import events  # noqa: E402

WH_KEYBOARD_LL = 13
WM_KEYDOWN, WM_KEYUP, WM_SYSKEYDOWN, WM_SYSKEYUP = 0x0100, 0x0101, 0x0104, 0x0105
WM_QUIT = 0x0012
LLKHF_INJECTED = 0x10
VK_PACKET = 0xE7
VK_ESCAPE, VK_END = 0x1B, 0x23
VK_RETURN, VK_BACK, VK_DELETE = 0x0D, 0x08, 0x2E
SHIFT = {0x10, 0xA0, 0xA1}
CTRL = {0x11, 0xA2, 0xA3}
ALT = {0x12, 0xA4, 0xA5}
WIN = {0x5B, 0x5C}
MODIFIERS = SHIFT | CTRL | ALT | WIN | {0x14}  # Caps Lock counted as a modifier
KEEP_VK = {VK_RETURN, VK_BACK, VK_DELETE, VK_ESCAPE} | MODIFIERS
# The numeric keypad (NumLock on): VkKeyScanW maps characters to the main keys only, so these are added here so that
# the hook gives the kind the page gives ('7' alnum, '.' end, '+' other: review of 2026-11-04). VK_DECIMAL is '.' on
# the layouts in use (Japanese, US).
NUMPAD = {**{0x60 + d: str(d) for d in range(10)}, 0x6A: "*", 0x6B: "+", 0x6D: "-", 0x6E: ".", 0x6F: "/"}
EXTRA_MARK = 0x7A7A0007  # inject.EXTRA_MARK (not imported: inject loads Windows-only libraries at import)


def layout_table(vk_scan) -> dict[tuple[int, bool], str]:
    """(virtual key, shift) -> printable ASCII character on the current layout, from VkKeyScanW (Ctrl/Alt
    combinations left out)."""
    table = {}
    for code in range(0x20, 0x7F):
        c = chr(code)
        r = vk_scan(c)
        if r == -1:
            continue
        vk, sh = r & 0xFF, (r >> 8) & 0xFF
        if sh & ~1:
            continue
        table.setdefault((vk, bool(sh & 1)), c)
    return table


class Classifier:
    """Kinds and repeats of raw events, one event at a time in arrival order (the writer thread classifies each batch
    as it is written). Modifier state and repeats are followed through the events themselves."""

    def __init__(self, table: dict[tuple[int, bool], str]):
        self.table = table
        self.held: set[int] = set()

    def one(self, t, vk, scan, flags, down, extra) -> dict:
        held = self.held
        repeat = bool(down and vk in held and vk != VK_PACKET)
        shift = bool(held & SHIFT)
        ctrl_alt = bool(held & (CTRL | ALT))
        if vk in MODIFIERS:
            kind = "modifier"
        elif vk == VK_RETURN:
            kind = "enter"
        elif vk == VK_BACK:
            kind = "backspace"
        elif vk == VK_DELETE:
            kind = "delete"
        elif vk == VK_PACKET:
            c = chr(scan)
            kind = events.kind_of_char(c) if c.isprintable() or c in "\r\n" else "nonchar"
        elif ctrl_alt:
            kind = "nonchar"
        elif vk in NUMPAD:
            kind = events.kind_of_char(NUMPAD[vk])
        else:
            c = self.table.get((vk, shift))
            if c is None and shift and self.table.get((vk, False)) == " ":
                # Shift with Space types a space and the page counts it as one; VkKeyScanW maps ' ' to Space without
                # Shift only (Dev.3 of v7: Space typed with Shift held under Caps Lock was 'nonchar' on the hook)
                c = " "
            kind = events.kind_of_char(c) if c is not None else "nonchar"
        if down:
            held.add(vk)
        else:
            held.discard(vk)
        rec = {"type": "key", "t_ns": t, "down": bool(down), "injected": bool(flags & LLKHF_INJECTED),
               "ours": extra == EXTRA_MARK, "repeat": repeat, "kind": kind}
        if vk in KEEP_VK:
            rec["vk"] = vk
        if vk == VK_PACKET:
            rec["packet"] = True
        return rec


def classify(raw: list[tuple], table: dict[tuple[int, bool], str]) -> list[dict]:
    """Kinds and repeats of raw events in arrival order. raw: (t_ns, vk, scan, flags, down, extra)."""
    c = Classifier(table)
    return [c.one(*r) for r in raw]


class KBDLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [("vkCode", wintypes.DWORD), ("scanCode", wintypes.DWORD), ("flags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t)]


# Records are appended to hook.ndjson in batches by a writer thread, never from the hook's callback (the callback only
# stores the raw event and, for Esc and the end chord, sets a flag): a crash loses at most the last batch, and the
# callback does no I/O (Mac review of 2026-11-04). The writer wakes every WRITE_EVERY_S and writes what has come; it
# holds the interpreter lock only for the short time a batch takes to be classified and written.
WRITE_EVERY_S = 0.5
# The hook process's interpreter switch interval: a callback waiting for the lock while the writer holds it waits at
# most about this long (Python's default is 5 ms). Compared with the P 18 ms smoke (session notes v3).
SWITCH_INTERVAL_S = 0.0005


def run(out: Path) -> int:
    import collections
    import threading
    sys.setswitchinterval(SWITCH_INTERVAL_S)
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    LRESULT = ctypes.c_ssize_t
    HOOKPROC = ctypes.WINFUNCTYPE(LRESULT, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)
    user32.SetWindowsHookExW.argtypes = [ctypes.c_int, HOOKPROC, wintypes.HINSTANCE, wintypes.DWORD]
    user32.SetWindowsHookExW.restype = wintypes.HHOOK
    user32.CallNextHookEx.argtypes = [wintypes.HHOOK, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM]
    user32.CallNextHookEx.restype = LRESULT
    user32.GetMessageW.argtypes = [ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT]
    user32.VkKeyScanW.argtypes = [wintypes.WCHAR]
    user32.VkKeyScanW.restype = ctypes.c_short
    user32.GetKeyboardLayout.restype = ctypes.c_void_p
    kernel32.GetModuleHandleW.restype = wintypes.HMODULE

    raw: collections.deque = collections.deque()   # append and popleft are atomic: no lock in the callback
    append = raw.append
    perf = time.perf_counter_ns
    held: set[int] = set()
    n_raw = [0]
    esc_at: list[int] = []
    end_at: list[int] = []
    foreign_at: list[int] = []
    signal = threading.Event()

    def proc(code, wparam, lparam):
        if code == 0:
            t = perf()
            k = ctypes.cast(lparam, ctypes.POINTER(KBDLLHOOKSTRUCT)).contents
            down = wparam in (WM_KEYDOWN, WM_SYSKEYDOWN)
            vk, flags = k.vkCode, k.flags
            append((t, vk, k.scanCode, flags, down, k.dwExtraInfo))
            n_raw[0] += 1
            if down:
                held.add(vk)
                if flags & LLKHF_INJECTED:
                    # injected but not by us, and not a modifier (the Alt of --auto-focus): keys typed through a
                    # remote desktop, for instance, which the hook cannot tell from an injection (returns-v3)
                    if k.dwExtraInfo != EXTRA_MARK and vk not in MODIFIERS:
                        foreign_at.append(t)
                        signal.set()
                else:
                    if vk == VK_ESCAPE:
                        esc_at.append(t)
                        signal.set()
                    elif vk == VK_END and held & CTRL and held & ALT:
                        end_at.append(t)
                        signal.set()
            else:
                held.discard(vk)
        return user32.CallNextHookEx(None, code, wparam, lparam)

    def reporter():  # prints Esc and the end chord for the session, outside the callback
        sent_esc = sent_end = sent_foreign = 0
        while True:
            signal.wait()
            signal.clear()
            while sent_esc < len(esc_at):
                print(f"ESC {esc_at[sent_esc]}", flush=True)
                sent_esc += 1
            while sent_end < len(end_at):
                print(f"END {end_at[sent_end]}", flush=True)
                sent_end += 1
            if sent_foreign < len(foreign_at):  # the first of a run is enough for the session
                print(f"FOREIGN {foreign_at[sent_foreign]}", flush=True)
                sent_foreign = len(foreign_at)

    cb = HOOKPROC(proc)
    start = {"type": "header", "pid": __import__("os").getpid(), "time_ns": time.time_ns(), "perf_ns": perf(),
             "keyboard_layout": hex(user32.GetKeyboardLayout(0) or 0), "write_every_s": WRITE_EVERY_S,
             "switch_interval_s": sys.getswitchinterval()}
    f = open(out, "x", encoding="utf-8", newline="\n")
    f.write(json.dumps(start) + "\n")
    f.flush()
    classifier = Classifier(layout_table(user32.VkKeyScanW))
    written = [0]
    writer_error: list[str] = []
    stop_writer = threading.Event()

    def drain():
        lines = []
        while raw:
            lines.append(json.dumps(classifier.one(*raw.popleft())))
        if lines:
            f.write("\n".join(lines) + "\n")
            f.flush()
            written[0] += len(lines)

    def writer():
        # an error here is kept for the footer (n_raw != n_written then): the session's check sees the loss (Mac
        # review of returns-v3), instead of a writer thread dying silently while the capture ends 'done'
        try:
            while not stop_writer.wait(WRITE_EVERY_S):
                drain()
        except Exception as e:  # noqa: BLE001 - recorded in the footer
            writer_error.append(f"{type(e).__name__}: {e}")

    threading.Thread(target=reporter, daemon=True).start()
    wt = threading.Thread(target=writer, daemon=True)
    wt.start()
    hook = user32.SetWindowsHookExW(WH_KEYBOARD_LL, cb, kernel32.GetModuleHandleW(None), 0)
    if not hook:
        print(f"FAIL SetWindowsHookExW {ctypes.get_last_error()}", flush=True)
        return 2
    print(f"READY {kernel32.GetCurrentThreadId()}", flush=True)
    msg = wintypes.MSG()
    while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
        user32.TranslateMessage(ctypes.byref(msg))
        user32.DispatchMessageW(ctypes.byref(msg))
    user32.UnhookWindowsHookEx(hook)
    stop_writer.set()
    wt.join()
    try:
        drain()
    except Exception as e:  # noqa: BLE001 - recorded in the footer
        writer_error.append(f"{type(e).__name__}: {e}")
    stop = {"type": "footer", "time_ns": time.time_ns(), "perf_ns": perf(), "n_raw": n_raw[0],
            "n_written": written[0], **({"writer_error": writer_error} if writer_error else {})}
    f.write(json.dumps(stop) + "\n")
    f.close()
    print("DONE", flush=True)
    return 0


def read(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    return run(a.out)


if __name__ == "__main__":
    raise SystemExit(main())
