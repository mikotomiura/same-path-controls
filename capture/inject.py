"""Type a schedule through SendInput (capture spec section 2; Windows only).

Two methods for characters (meta.json 'injection'):
- 'unicode': KEYEVENTF_UNICODE (the system turns it into VK_PACKET);
- 'vk': the character's virtual key from VkKeyScanW on the current keyboard layout, with Shift pressed around it
  when the layout needs it (Ctrl and Alt combinations are refused). A character the layout cannot type is sent as
  unicode and counted ('fallback_unicode').
Enter, Backspace and Delete are always virtual keys (VK_RETURN, VK_BACK, VK_DELETE). One press is its down and up in a
single SendInput call (holds are not measured); in the vk method a shifted character is Shift down, key down, key up,
Shift up in that one call.

Timing: the start T0 is perf_counter_ns(); each press waits for T0 + t_ms with time.sleep until 2 ms remain and then
spins (time.sleep(0) in the spin, so that other threads of the process are not kept from the interpreter lock).
timeBeginPeriod(1) is set for the capture. A press that is late is sent at once and its lateness kept (no catching
up faster than the schedule, no skipping).
"""

from __future__ import annotations

import ctypes
import sys
import time
from ctypes import wintypes

INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004
VK_SHIFT, VK_RETURN, VK_BACK, VK_DELETE = 0x10, 0x0D, 0x08, 0x2E
NAMED_VK = {"Enter": VK_RETURN, "Backspace": VK_BACK, "Delete": VK_DELETE}
SPIN_NS = 2_000_000
METHODS = ("unicode", "vk")

ULONG_PTR = ctypes.c_size_t


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD), ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR)]


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR)]


class _U(ctypes.Union):
    _fields_ = [("ki", KEYBDINPUT), ("mi", MOUSEINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("u", _U)]


# a mark in dwExtraInfo so that our own presses can be told apart in the hook (besides LLKHF_INJECTED)
EXTRA_MARK = 0x7A7A0007


def _user32():
    u = ctypes.WinDLL("user32", use_last_error=True)
    u.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int]
    u.SendInput.restype = wintypes.UINT
    u.VkKeyScanW.argtypes = [wintypes.WCHAR]
    u.VkKeyScanW.restype = ctypes.c_short
    u.GetForegroundWindow.restype = wintypes.HWND
    return u


def _kb(vk: int = 0, scan: int = 0, flags: int = 0) -> INPUT:
    return INPUT(type=INPUT_KEYBOARD, u=_U(ki=KEYBDINPUT(wVk=vk, wScan=scan, dwFlags=flags, time=0,
                                                       dwExtraInfo=EXTRA_MARK)))


class Injector:
    def __init__(self, method: str):
        if sys.platform != "win32":
            raise RuntimeError("SendInput needs Windows")
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        self.method = method
        self.user32 = _user32()
        self.winmm = ctypes.WinDLL("winmm")
        self.fallback_unicode = 0
        self._cache: dict[str, ctypes.Array] = {}

    def inputs_for(self, key: str) -> ctypes.Array:
        """The INPUT array of one press (built once per key)."""
        arr = self._cache.get(key)
        if arr is not None:
            return arr
        if key in NAMED_VK:
            seq = [_kb(NAMED_VK[key]), _kb(NAMED_VK[key], flags=KEYEVENTF_KEYUP)]
        elif self.method == "vk":
            r = self.user32.VkKeyScanW(key)
            vk, shift = r & 0xFF, (r >> 8) & 0xFF
            if r == -1 or shift & ~1:
                seq = None  # not on the layout, or needs Ctrl/Alt: sent as unicode and counted below
            else:
                seq = [_kb(vk), _kb(vk, flags=KEYEVENTF_KEYUP)]
                if shift & 1:
                    seq = [_kb(VK_SHIFT)] + seq + [_kb(VK_SHIFT, flags=KEYEVENTF_KEYUP)]
        else:
            seq = None
        if seq is None:
            if len(key) != 1:
                raise ValueError(f"cannot type {key!r}")
            seq = [_kb(scan=ord(key), flags=KEYEVENTF_UNICODE),
                   _kb(scan=ord(key), flags=KEYEVENTF_UNICODE | KEYEVENTF_KEYUP)]
        arr = (INPUT * len(seq))(*seq)
        self._cache[key] = arr
        return arr

    def is_fallback(self, key: str) -> bool:
        if key in NAMED_VK or self.method == "unicode":
            return False
        r = self.user32.VkKeyScanW(key)
        return r == -1 or bool(((r >> 8) & 0xFF) & ~1)

    def send(self, key: str) -> tuple[int, int, int, int]:
        """Send one press now. Returns (sent_ns, returned_ns, SendInput's return, the number of INPUTs)."""
        arr = self.inputs_for(key)
        n = len(arr)
        t0 = time.perf_counter_ns()
        r = self.user32.SendInput(n, arr, ctypes.sizeof(INPUT))
        t1 = time.perf_counter_ns()
        return t0, t1, int(r), n

    def __enter__(self):
        self.winmm.timeBeginPeriod(1)
        return self

    def __exit__(self, *exc):
        self.winmm.timeEndPeriod(1)
        return False


# The longest single sleep of a wait: a stop (Esc, or capture.stop) asked during a long wait (L1's or H1's rests of
# several seconds) is seen within this time, and the next press is not sent (Mac review of 2026-11-04).
STOP_POLL_NS = 20_000_000


def wait_until(target_ns: int, stop=None) -> bool:
    """Wait for target_ns (sleep until 2 ms remain, then spin). Returns True when stop (an object with is_set()) was
    set during the wait; the caller then sends nothing. The stop is asked at least every STOP_POLL_NS while sleeping
    and once more at the end; never inside the last 2 ms spin, so that the spin stays as it was."""
    while True:
        left = target_ns - time.perf_counter_ns()
        if left <= SPIN_NS:
            break
        if stop is not None and stop.is_set():
            return True
        time.sleep(min(left - SPIN_NS, STOP_POLL_NS) / 1e9)
    while time.perf_counter_ns() < target_ns:
        time.sleep(0)
    return stop is not None and stop.is_set()


def type_schedule(presses: list[dict], method: str, stop=None, t0_ns: int | None = None,
                  guard_hwnd: int | None = None) -> tuple[list[dict], dict]:
    """Type the presses at T0 + t_ms. stop: an object with is_set() (the hook's Esc or capture.stop), checked before
    each press, during each wait and again just before each press is sent.
    guard_hwnd: the page's window; before each press the foreground window must be it, otherwise typing stops at once
    and the capture is interrupted (redone once by the runlist: the researcher, 2026-11-04)
    (a press must never reach another window: in the first probe on the work PC, with the IME turned on, the presses
    went to another window although the page had reported focus just before). Returns one record per press attempted
    and a summary."""
    recs = []
    inj = Injector(method)
    for p in presses:  # build every INPUT array before T0
        inj.inputs_for(p["key"])
    fallbacks = sum(inj.is_fallback(p["key"]) for p in presses)
    stopped = False
    focus_lost = False
    fg = inj.user32.GetForegroundWindow
    with inj:
        T0 = time.perf_counter_ns() if t0_ns is None else t0_ns
        for p in presses:
            if stop is not None and stop.is_set():
                stopped = True
                break
            planned = T0 + round(p["t_ms"] * 1e6)
            if wait_until(planned, stop):  # a stop during the wait: this press is not sent (checked just before it)
                stopped = True
                break
            if guard_hwnd is not None and fg() != guard_hwnd:
                focus_lost = True
                break
            sent, ret_ns, r, n = inj.send(p["key"])
            recs.append({"i": p["i"], "planned_ns": planned, "sent_ns": sent, "returned_ns": ret_ns,
                         "late_ns": sent - planned, "sendinput_returned": r, "n_inputs": n, "ok": r == n})
    summary = {"method": method, "t0_ns": T0, "planned": len(presses), "attempted": len(recs),
               "not_sent": sum(not x["ok"] for x in recs), "stopped_by_esc": stopped,
               "stopped_focus_lost": focus_lost,
               "fallback_unicode": int(fallbacks)}
    return recs, summary
