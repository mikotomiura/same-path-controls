# Diagnostic for block 3's hidden pages (2026-11-10): every 2 s, the windows above the capture Chrome window in z-order
# (process, rectangle, TOP = topmost, LAY = layered, CLOAK = DWM cloak value), the foreground process, the input idle
# time and the console display state is not read. A line is written only when it changes. Names and rectangles only
# (no window titles, no keys, no text).
$logf = 'C:\work\v7-chain\zwatch.log'
Add-Type @'
using System; using System.Runtime.InteropServices;
public class ZW { public delegate bool P(IntPtr h, IntPtr l);
[DllImport("user32.dll")] public static extern bool EnumWindows(P p, IntPtr l);
[DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr h);
[DllImport("user32.dll")] public static extern bool IsIconic(IntPtr h);
[DllImport("user32.dll")] public static extern int GetWindowThreadProcessId(IntPtr h, out int pid);
[DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out R r);
[DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
[DllImport("user32.dll")] public static extern int GetWindowLong(IntPtr h, int i);
[DllImport("dwmapi.dll")] public static extern int DwmGetWindowAttribute(IntPtr h, int a, out int v, int s);
public struct R { public int L,T,Rr,B; }
[StructLayout(LayoutKind.Sequential)] public struct LI { public uint cb; public uint t; }
[DllImport("user32.dll")] public static extern bool GetLastInputInfo(ref LI l);
public static uint IdleMs() { LI l = new LI(); l.cb = 8; GetLastInputInfo(ref l); return (uint)Environment.TickCount - l.t; } }
'@
$names = @{}
function PName($procId) { if (-not $names.ContainsKey($procId)) { $names[$procId] = (Get-Process -Id $procId -EA SilentlyContinue).Name }; $names[$procId] }
$last = ''
while ($true) {
    $script:above = New-Object System.Collections.ArrayList; $script:found = $false; $script:cap = ''
    $fg = [ZW]::GetForegroundWindow(); $fp = 0; [void][ZW]::GetWindowThreadProcessId($fg, [ref]$fp)
    $cb = [ZW+P]{ param($h, $l)
        if ($script:found) { return $true }
        if (-not [ZW]::IsWindowVisible($h)) { return $true }
        $r = New-Object ZW+R; [void][ZW]::GetWindowRect($h, [ref]$r)
        $w = $r.Rr - $r.L; $hh = $r.B - $r.T
        if ($w -lt 20 -or $hh -lt 20) { return $true }
        $p = 0; [void][ZW]::GetWindowThreadProcessId($h, [ref]$p); $n = PName $p
        $c = 0; [void][ZW]::DwmGetWindowAttribute($h, 14, [ref]$c, 4)
        $ex = [ZW]::GetWindowLong($h, -20)
        $d = "{0}[{1},{2} {3}x{4}{5}{6}{7}{8}]" -f $n, $r.L, $r.T, $w, $hh, $(if ($ex -band 8) { ' TOP' } else { '' }),
            $(if ($ex -band 0x80000) { ' LAY' } else { '' }), $(if ($c) { " CLOAK=$c" } else { '' }), $(if ([ZW]::IsIconic($h)) { ' MIN' } else { '' })
        if ($n -eq 'chrome' -and $w -gt 600 -and $hh -gt 600) { $script:found = $true; $script:cap = $d; return $true }
        [void]$script:above.Add($d)
        return $true }
    [void][ZW]::EnumWindows($cb, [IntPtr]::Zero)
    $idle = [ZW]::IdleMs() / 1000
    $state = "fg=$(PName $fp) capture=$(if ($script:found) { $script:cap } else { 'none' }) above: $($script:above -join ' ')"
    if ($state -ne $last) {
        Add-Content -Encoding utf8 $logf ("{0}Z idle={1:n0}s {2}" -f (Get-Date).ToUniversalTime().ToString('MM-ddTHH:mm:ss'), $idle, $state)
        $last = $state
    }
    Start-Sleep -Seconds 2
}
