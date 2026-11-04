"""Stop a capture that is typing, from outside it (Mac review of 2026-11-04).

    python -m capture.stop        (from another console; sets the named event the session watches)

With the hook installed, a non-injected Esc stops the capture (the hook reports it). With --no-hook (P_nohook) there is
no hook to see Esc, so the session also watches a named Windows event, 'Local\\v7-capture-stop', which this command
sets. The session resets the event when it starts, so that an earlier stop never stops a later capture. Ctrl+C in the
session's console also ends it, but as 'failed' (the children share the console and stop too); this command ends it as
'interrupted', like Esc.
"""

from __future__ import annotations

import ctypes
import sys
import threading
from ctypes import wintypes

EVENT_NAME = "Local\\v7-capture-stop"
WAIT_OBJECT_0 = 0


def _kernel32():
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.CreateEventW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.BOOL, wintypes.LPCWSTR]
    k.CreateEventW.restype = wintypes.HANDLE
    k.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    k.WaitForSingleObject.restype = wintypes.DWORD
    k.SetEvent.argtypes = [wintypes.HANDLE]
    k.ResetEvent.argtypes = [wintypes.HANDLE]
    return k


class StopSignal:
    """is_set() is true once the hook saw Esc (esc: a threading.Event, or None without the hook) or once the named
    event was set by 'python -m capture.stop'. Cheap enough to be asked before every press and during waits."""

    def __init__(self, esc: threading.Event | None = None, name: str = EVENT_NAME):
        self.esc = esc
        self.k = _kernel32()
        self.h = self.k.CreateEventW(None, True, False, name)  # manual reset, not signalled
        if not self.h:
            raise OSError(f"CreateEventW failed: {ctypes.get_last_error()}")
        self.k.ResetEvent(self.h)
        self.by: str | None = None

    def is_set(self) -> bool:
        if self.esc is not None and self.esc.is_set():
            self.by = self.by or "esc"
            return True
        if self.k.WaitForSingleObject(self.h, 0) == WAIT_OBJECT_0:
            self.by = self.by or "capture.stop"
            return True
        return False

    def close(self) -> None:
        if self.h:
            self.k.CloseHandle(self.h)
            self.h = None


def signal(name: str = EVENT_NAME) -> None:
    k = _kernel32()
    h = k.CreateEventW(None, True, False, name)
    if not h:
        raise OSError(f"CreateEventW failed: {ctypes.get_last_error()}")
    k.SetEvent(h)
    k.CloseHandle(h)


def main() -> int:
    if sys.platform != "win32":
        raise SystemExit("Windows only")
    signal()
    print("stop signalled")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
