# Minimizes every visible top-level window of the given processes (default: VS Code and Edge), so that none can cover the
# capture Chrome window when it opens behind them (block 3's hidden pages, 2026-11-10; Edge added 2026-11-11 after block 5).
# -All (added 2026-11-11 for block 6 after the network cut): every visible, uncloaked window of 600x600 or more of any
# process except the capture Chrome (user-data-dir v7-capture-profile) and Windows' shell (explorer, TextInputHost,
# ApplicationFrameHost, ShellExperienceHost, StartMenuExperienceHost, SearchHost). Prints only counts and process names.
param([string[]]$Names = @('Code', 'msedge'), [switch]$All)
Add-Type @'
using System; using System.Runtime.InteropServices;
public class MW { public delegate bool P(IntPtr h, IntPtr l);
[DllImport("user32.dll")] public static extern bool EnumWindows(P p, IntPtr l);
[DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr h);
[DllImport("user32.dll")] public static extern bool IsIconic(IntPtr h);
[DllImport("user32.dll")] public static extern int GetWindowThreadProcessId(IntPtr h, out int pid);
[DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out R r);
[DllImport("dwmapi.dll")] public static extern int DwmGetWindowAttribute(IntPtr h, int a, out int v, int s);
public struct R { public int L,T,Rr,B; }
[DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int c); }
'@
$shell = @('explorer', 'TextInputHost', 'ApplicationFrameHost', 'ShellExperienceHost', 'StartMenuExperienceHost', 'SearchHost')
$ids = @(Get-Process -Name $Names -EA SilentlyContinue | ForEach-Object Id)
$capture = @(Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'chrome.exe' -and $_.CommandLine -match 'v7-capture-profile' } | ForEach-Object ProcessId)
$script:n = 0; $script:who = @{}
$cb = [MW+P]{ param($h, $l)
    if (-not [MW]::IsWindowVisible($h) -or [MW]::IsIconic($h)) { return $true }
    $p = 0; [void][MW]::GetWindowThreadProcessId($h, [ref]$p)
    if ($All) {
        if ($capture -contains $p) { return $true }
        $c = 0; [void][MW]::DwmGetWindowAttribute($h, 14, [ref]$c, 4); if ($c) { return $true }
        $r = New-Object MW+R; [void][MW]::GetWindowRect($h, [ref]$r)
        if (($r.Rr - $r.L) -lt 600 -or ($r.B - $r.T) -lt 600) { return $true }
        $nm = (Get-Process -Id $p -EA SilentlyContinue).Name
        if (-not $nm -or $shell -contains $nm) { return $true }
    } elseif ($ids -notcontains $p) { return $true } else { $nm = (Get-Process -Id $p -EA SilentlyContinue).Name }
    [void][MW]::ShowWindow($h, 6); $script:n++; $script:who[$nm] = 1
    return $true }
[void][MW]::EnumWindows($cb, [IntPtr]::Zero)
"minimized $script:n window(s)$(if ($script:n) { ' of ' + (($script:who.Keys | Sort-Object) -join ',') })"
