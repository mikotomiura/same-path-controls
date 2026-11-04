"""Hold the mouse while presses are typed (a safety measure asked by the researcher on 2026-11-04), in its own process.

    python -m capture.mouseguard      (prints 'READY <thread id>'; stopped with PostThreadMessageW(WM_QUIT))

A WH_MOUSE_LL hook swallows the non-injected button and wheel events (clicks cannot move the focus away from the page
while it is typed into); pointer movement passes. Touch promoted to mouse input arrives at the same hook [not checked:
the work PC has no touch screen known to this session]. The keyboard is not held: holding it would need a second
keyboard hook in front of the logger, which would change the path that is measured, and Esc must stay usable. The
keyboard side is guarded instead by the foreground check before every press (inject.type_schedule, llm_stream.Typist).
Windows removes a low-level hook that stops answering, and Ctrl+Alt+Del is never held, so the mouse cannot stay held.
On exit it prints the number of events swallowed.
"""

from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes

WH_MOUSE_LL = 14
WM_MOUSEMOVE = 0x0200
LLMHF_INJECTED = 0x01


class MSLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [("pt", wintypes.POINT), ("mouseData", wintypes.DWORD), ("flags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t)]


def swallow(msg: int, flags: int) -> bool:
    """Swallow every non-injected mouse message except movement."""
    return msg != WM_MOUSEMOVE and not flags & LLMHF_INJECTED


def main() -> int:
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    LRESULT = ctypes.c_ssize_t
    HOOKPROC = ctypes.WINFUNCTYPE(LRESULT, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)
    user32.SetWindowsHookExW.argtypes = [ctypes.c_int, HOOKPROC, wintypes.HINSTANCE, wintypes.DWORD]
    user32.SetWindowsHookExW.restype = wintypes.HHOOK
    user32.CallNextHookEx.argtypes = [wintypes.HHOOK, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM]
    user32.CallNextHookEx.restype = LRESULT
    user32.GetMessageW.argtypes = [ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT]
    kernel32.GetModuleHandleW.restype = wintypes.HMODULE
    held = [0]

    def proc(code, wparam, lparam):
        if code == 0:
            m = ctypes.cast(lparam, ctypes.POINTER(MSLLHOOKSTRUCT)).contents
            if swallow(wparam, m.flags):
                held[0] += 1
                return 1
        return user32.CallNextHookEx(None, code, wparam, lparam)

    cb = HOOKPROC(proc)
    hook = user32.SetWindowsHookExW(WH_MOUSE_LL, cb, kernel32.GetModuleHandleW(None), 0)
    if not hook:
        print(f"FAIL SetWindowsHookExW {ctypes.get_last_error()}", flush=True)
        return 2
    print(f"READY {kernel32.GetCurrentThreadId()}", flush=True)
    msg = wintypes.MSG()
    while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
        user32.TranslateMessage(ctypes.byref(msg))
        user32.DispatchMessageW(ctypes.byref(msg))
    user32.UnhookWindowsHookEx(hook)
    print(f"DONE {held[0]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
