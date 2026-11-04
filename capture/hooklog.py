"""The OS layer (capture spec section 3a): WH_KEYBOARD_LL in its own process, key downs and ups with perf_counter_ns.

Run as a child of the session:  python -m capture.hooklog --out hook.ndjson
It prints 'READY <thread id>' once the hook is installed, 'ESC <ns>' when a non-injected Esc goes down (the stop
flag of the injector) and 'END <ns>' on a non-injected Ctrl+Alt+End (the end signal of the live capture L0). The
session stops it with PostThreadMessageW(thread id, WM_QUIT); the records are written then.

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


def classify(raw: list[tuple], table: dict[tuple[int, bool], str]) -> list[dict]:
    """Kinds and repeats of raw events in arrival order. raw: (t_ns, vk, scan, flags, down, extra).
    Modifier state and repeats are followed through the events themselves."""
    held: set[int] = set()
    out = []
    for t, vk, scan, flags, down, extra in raw:
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
        else:
            c = table.get((vk, shift))
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
        out.append(rec)
    return out


class KBDLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [("vkCode", wintypes.DWORD), ("scanCode", wintypes.DWORD), ("flags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t)]


def run(out: Path) -> int:
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

    raw: list[tuple] = []
    append = raw.append
    perf = time.perf_counter_ns
    held: set[int] = set()

    def proc(code, wparam, lparam):
        if code == 0:
            t = perf()
            k = ctypes.cast(lparam, ctypes.POINTER(KBDLLHOOKSTRUCT)).contents
            down = wparam in (WM_KEYDOWN, WM_SYSKEYDOWN)
            vk, flags = k.vkCode, k.flags
            append((t, vk, k.scanCode, flags, down, k.dwExtraInfo))
            if down:
                held.add(vk)
                if not flags & LLKHF_INJECTED:
                    if vk == VK_ESCAPE:
                        print(f"ESC {t}", flush=True)
                    elif vk == VK_END and held & CTRL and held & ALT:
                        print(f"END {t}", flush=True)
            else:
                held.discard(vk)
        return user32.CallNextHookEx(None, code, wparam, lparam)

    cb = HOOKPROC(proc)
    start = {"type": "header", "pid": __import__("os").getpid(), "time_ns": time.time_ns(), "perf_ns": perf(),
             "keyboard_layout": hex(user32.GetKeyboardLayout(0) or 0)}
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
    stop = {"type": "footer", "time_ns": time.time_ns(), "perf_ns": perf(), "n_raw": len(raw)}
    recs = classify(raw, layout_table(user32.VkKeyScanW))
    with open(out, "x", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(start) + "\n")
        for r in recs:
            f.write(json.dumps(r) + "\n")
        f.write(json.dumps(stop) + "\n")
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
