"""Start the browser on the page, bring it to the front and turn the IME off (Windows only).

The browser runs in a profile made for the capture (a new temporary folder), so that no extension, history or
restored session of the user's profile enters it, and the profile is deleted afterwards (the page empties its box,
and no typed text is left on disk by the browser either).
"""

from __future__ import annotations

import ctypes
import shutil
import subprocess
import tempfile
import time
from ctypes import wintypes
from pathlib import Path

EXE = {
    "chrome": Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    "edge": Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    # v7s (works/conditions/prototype_v7s.json, captures.firefox): the second path, the page layer only differs
    "firefox": Path(r"C:\Program Files\Mozilla Firefox\firefox.exe"),
}
TITLE = "v7 capture page"
WM_IME_CONTROL = 0x0283
IMC_GETOPENSTATUS, IMC_SETOPENSTATUS = 0x0005, 0x0006


def file_version(path: Path) -> str:
    v = ctypes.WinDLL("version")
    size = v.GetFileVersionInfoSizeW(str(path), None)
    if not size:
        raise OSError(f"no version information in {path}")
    buf = ctypes.create_string_buffer(size)
    v.GetFileVersionInfoW(str(path), 0, size, buf)
    p, n = ctypes.c_void_p(), wintypes.UINT()
    v.VerQueryValueW(buf, "\\", ctypes.byref(p), ctypes.byref(n))

    class VS_FIXEDFILEINFO(ctypes.Structure):
        _fields_ = [(f, wintypes.DWORD) for f in ("sig", "strucver", "ms", "ls", "pms", "pls", "mask", "flags", "os",
                                                    "type", "subtype", "dms", "dls")]

    fi = ctypes.cast(p, ctypes.POINTER(VS_FIXEDFILEINFO)).contents
    return f"{fi.pms >> 16}.{fi.pms & 0xFFFF}.{fi.pls >> 16}.{fi.pls & 0xFFFF}"


def _user32():
    u = ctypes.WinDLL("user32", use_last_error=True)
    u.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    u.SendMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    u.SendMessageW.restype = ctypes.c_ssize_t
    u.GetForegroundWindow.restype = wintypes.HWND
    return u


def find_window(title: str = TITLE) -> int | None:
    u = _user32()
    found = []
    PROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def cb(hwnd, _):
        if u.IsWindowVisible(hwnd):
            buf = ctypes.create_unicode_buffer(512)
            u.GetWindowTextW(hwnd, buf, 512)
            if buf.value.startswith(title):
                found.append(hwnd)
        return True

    u.EnumWindows(PROC(cb), 0)
    return found[0] if found else None


def foreground(hwnd: int) -> bool:
    """Bring the window to the front. Windows lets a process do this only in some states; an Alt tap (a modifier,
    recorded by the hook as 'modifier' before the capture is armed) unlocks it, as is common practice."""
    u = _user32()
    if u.GetForegroundWindow() == hwnd:
        return True
    u.keybd_event(0x12, 0, 0, 0)
    u.keybd_event(0x12, 0, 2, 0)
    u.ShowWindow(hwnd, 9)  # SW_RESTORE
    u.SetForegroundWindow(hwnd)
    time.sleep(0.3)
    return u.GetForegroundWindow() == hwnd


def ime_open(hwnd: int) -> int:
    """The IME open status of the window (1 open = Japanese input, 0 closed = direct input)."""
    imm = ctypes.WinDLL("imm32")
    imm.ImmGetDefaultIMEWnd.restype = wintypes.HWND
    ime = imm.ImmGetDefaultIMEWnd(hwnd)
    return int(_user32().SendMessageW(ime, WM_IME_CONTROL, IMC_GETOPENSTATUS, 0))


def ime_close(hwnd: int) -> tuple[int, int]:
    """Close the IME of the window (direct input). Returns the status before and after."""
    imm = ctypes.WinDLL("imm32")
    imm.ImmGetDefaultIMEWnd.restype = wintypes.HWND
    ime = imm.ImmGetDefaultIMEWnd(hwnd)
    u = _user32()
    before = int(u.SendMessageW(ime, WM_IME_CONTROL, IMC_GETOPENSTATUS, 0))
    u.SendMessageW(ime, WM_IME_CONTROL, IMC_SETOPENSTATUS, 0)
    after = int(u.SendMessageW(ime, WM_IME_CONTROL, IMC_GETOPENSTATUS, 0))
    return before, after


PROFILE_PREFIX = "v7-capture-profile-"
# No name outside the machine is resolved: the page (127.0.0.1) opens and every other host name fails (Mac review of
# 2026-11-04; checked on the work PC, session notes v3). A literal IP address other than 127.0.0.1 is not covered by
# this rule; the background services that would reach out are turned off by the flags below.
HOST_RULES = "MAP * ~NOTFOUND , EXCLUDE 127.0.0.1"
OFFLINE_FLAGS = [f"--host-resolver-rules={HOST_RULES}", "--disable-background-networking",
                 "--disable-component-update", "--disable-domain-reliability", "--no-pings"]


# Firefox (v7s): no command-line flags for these, so the new profile gets a user.js. Nothing here touches the timer
# prefs (privacy.reduceTimerPrecision and its kin): the page runs at Firefox's default precision (captures.firefox).
# The proxy sends every request but those to 127.0.0.1 to a port where nothing listens (localhost is not proxied:
# network.proxy.allow_hijacking_localhost false), as Chrome's host rules leave only 127.0.0.1.
FIREFOX_PREFS = {
    "browser.shell.checkDefaultBrowser": False,
    "browser.aboutwelcome.enabled": False,
    "browser.startup.homepage_override.mstone": "ignore",
    "startup.homepage_welcome_url": "",
    "startup.homepage_welcome_url.additional": "",
    "datareporting.policy.dataSubmissionEnabled": False,
    "datareporting.healthreport.uploadEnabled": False,
    "toolkit.telemetry.enabled": False,
    "toolkit.telemetry.unified": False,
    "app.normandy.enabled": False,
    "app.shield.optoutstudies.enabled": False,
    "app.update.auto": False,
    "browser.safebrowsing.malware.enabled": False,
    "browser.safebrowsing.phishing.enabled": False,
    "browser.safebrowsing.downloads.enabled": False,
    "network.captive-portal-service.enabled": False,
    "network.connectivity-service.enabled": False,
    "extensions.update.enabled": False,
    "browser.search.update": False,
    "network.proxy.type": 1,
    "network.proxy.http": "127.0.0.1",
    "network.proxy.http_port": 9,
    "network.proxy.ssl": "127.0.0.1",
    "network.proxy.ssl_port": 9,
    "network.proxy.share_proxy_settings": True,
    "network.proxy.no_proxies_on": "127.0.0.1",
    "network.proxy.allow_hijacking_localhost": False,
    "network.dns.disablePrefetch": True,
    "network.prefetch-next": False,
    # foreground() taps Alt to bring the window to the front; in Firefox on Windows a lone Alt would move the focus to
    # the menu bar and the next letters would be read as its access keys (code review of 2026-11-15)
    "ui.key.menuAccessKeyFocuses": False,
    "datareporting.policy.firstRunURL": "",
    "toolkit.telemetry.reportingpolicy.firstRun": False,
    "browser.startup.page": 0,
    "browser.sessionstore.resume_from_crash": False,
}


def user_js(prefs: dict = FIREFOX_PREFS) -> str:
    """The text of user.js: one user_pref line per pref (JSON's quoting is valid for user.js)."""
    import json
    return "".join(f"user_pref({json.dumps(k)}, {json.dumps(v)});\n" for k, v in prefs.items())


def stale_profiles(tmp: Path | None = None) -> list[Path]:
    """Temporary profiles left by an earlier capture (a capture that crashed before deleting its profile)."""
    tmp = Path(tempfile.gettempdir()) if tmp is None else Path(tmp)
    return sorted(p for p in tmp.glob(PROFILE_PREFIX + "*") if p.is_dir())


def remove_stale_profiles(tmp: Path | None = None) -> dict:
    """Delete the profiles of earlier captures before a new one starts (their typed text must not stay on disk, and
    no capture may start from a used profile). Returns what was found and deleted, for meta.json."""
    found = stale_profiles(tmp)
    deleted, failed = [], []
    for p in found:
        try:
            shutil.rmtree(p)
            deleted.append(str(p))
        except OSError as e:
            failed.append(f"{p}: {e}")
    return {"found": len(found), "deleted": deleted, "failed": failed}


class Browser:
    def __init__(self, name: str, url: str, incognito: bool = False):
        if name not in EXE:
            raise ValueError(f"browser must be one of {sorted(EXE)}")
        self.name, self.url, self.incognito = name, url, incognito
        self.exe = EXE[name]
        self.version = file_version(self.exe)
        self.profile = Path(tempfile.mkdtemp(prefix=PROFILE_PREFIX))
        if name == "firefox":
            (self.profile / "user.js").write_text(user_js(), encoding="utf-8", newline="\n")
        self.proc: subprocess.Popen | None = None
        self.started_ns: int | None = None

    def args(self) -> list[str]:
        """--app opens the page in a window with no address bar and no tabs, so that no press can reach an address
        bar (Mac review of 2026-11-04). --incognito in the live capture (L0). The list is written to meta.json as
        it is passed. Firefox (v7s) has no --app: -kiosk opens the page full screen with no address bar or tabs."""
        if self.name == "firefox":
            if self.incognito:
                raise ValueError("the live capture (L0) is not taken in Firefox")
            return [str(self.exe), "-profile", str(self.profile), "-no-remote", "-kiosk", self.url]
        return ([str(self.exe), f"--user-data-dir={self.profile}", "--no-first-run", "--no-default-browser-check",
                 "--disable-extensions", "--disable-sync", "--disable-translate", *OFFLINE_FLAGS]
                + (["--incognito"] if self.incognito else []) + [f"--app={self.url}"])

    def start(self, timeout_s: float = 20.0) -> int:
        self.started_ns = time.time_ns()
        self.proc = subprocess.Popen(self.args())
        end = time.monotonic() + timeout_s
        while time.monotonic() < end:
            hwnd = find_window()
            if hwnd:
                return hwnd
            time.sleep(0.2)
        raise TimeoutError(f"the page window did not appear in {timeout_s} s")

    def close(self) -> None:
        if self.proc is not None:
            # the profile's browser processes are found by the profile path on their command line
            subprocess.run(["powershell", "-NoProfile", "-Command",
                            "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*"
                            + str(self.profile).replace("'", "''") + "*' } | "
                            "ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"],
                           capture_output=True)
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        for _ in range(20):
            try:
                shutil.rmtree(self.profile)
                return
            except OSError:
                time.sleep(0.5)
        raise OSError(f"could not delete the browser profile {self.profile}")
