"""Remote-desktop hosts on the work PC (returns-v3): keys typed through Chrome Remote Desktop come injected, so a live
capture (L0) through it cannot be told from an injection, and its end chord is not seen. The live capture is typed on
the PC's own keyboard; while a host runs, neither the session's live capture nor a runlist holding one starts (the
researcher's decisions 2 and 5 of .steering/20261104-v7-prep-v3-returns)."""

from __future__ import annotations

import subprocess
import sys

REMOTE_HOSTS = ("remoting_host.exe",)  # Chrome Remote Desktop's host (the service 'chromoting' keeps it running)
# the session's exit code when a remote-desktop host stops a live capture before its folder is made: the runlist
# stops on it (never a failed item)
REMOTE_HOST_EXIT = 5


def hosts_in(tasklist_csv: str) -> list[str]:
    """The remote-desktop hosts in the output of 'tasklist /FO CSV /NH' (one quoted CSV row per process)."""
    names = {ln.split('","')[0].strip('"').lower() for ln in tasklist_csv.splitlines() if ln.startswith('"')}
    return [h for h in REMOTE_HOSTS if h in names]


def running() -> list[str]:
    """The remote-desktop hosts running now (Windows; [] elsewhere). Raises when the process list cannot be read."""
    if sys.platform != "win32":
        return []
    r = subprocess.run(["tasklist", "/FO", "CSV", "/NH"], capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        raise RuntimeError(f"tasklist failed ({r.returncode}): the remote-desktop check cannot be made")
    return hosts_in(r.stdout)


def message(hosts: list[str]) -> str:
    return (f"a remote-desktop host is running ({', '.join(hosts)}): its keys come injected and L0 cannot be told from "
            "an injection. Stop it (the service 'chromoting', as administrator) and type on this PC's own keyboard.")
