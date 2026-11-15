# v7s copy (2026-11-15): Chrome or Firefox capture browsers, v7s's folders.
# Guard for the v7 chain (added 2026-11-10 after block 3's hidden pages): stops the chain at the first capture whose
# page recorded nothing (meta.json checks_after.page_recorded_nothing), or when two capture browsers run at once.
# Stopping = killing chain.ps1's process tree and any capture browser; the block's runlog is then left without its footer
# (set aside by hand, as in the handoff's step 5). Reads meta.json and process lists only (no press, key or text).
$logf = 'C:\work\v7s-chain\chain.log'
$out = 'C:\work\v7s-captures'
function Log($m) { $l = "$((Get-Date).ToUniversalTime().ToString('o')) $m"; Add-Content -Encoding utf8 $logf $l }
function StopAll($why) {
    Log "STOP guard: $why"
    Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'powershell.exe' -and $_.CommandLine -match 'chain\.ps1' } |
        ForEach-Object { & taskkill /PID $_.ProcessId /T /F | Out-Null }
    Get-CimInstance Win32_Process | Where-Object { $_.Name -in @('chrome.exe', 'firefox.exe') -and $_.CommandLine -match 'v7-capture-profile' } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -EA SilentlyContinue }
    Log 'guard: chain and capture browsers stopped; the running block has no footer (set aside by hand)'
    exit 1
}
# -NoStop (added 2026-11-11, the researcher's decision for block 6 after the network cut): a page that recorded nothing is
# logged once and the chain is not stopped (the runlist runs that item again itself); minimizing goes on.
$NoStop = $args -contains '-NoStop'
$seen = @{}
$since = Get-Date
Log "guard start$(if ($NoStop) { ' (-NoStop: log a page that recorded nothing, do not stop)' })"
while ($true) {
    $dirs = @(Get-CimInstance Win32_Process | Where-Object { $_.Name -in @('chrome.exe', 'firefox.exe') -and $_.CommandLine -match 'v7-capture-profile' } |
              ForEach-Object { [regex]::Match($_.CommandLine, 'v7-capture-profile-\w+').Value } | Sort-Object -Unique)
    if ($dirs.Count -gt 1) { StopAll "$($dirs.Count) capture browsers at once" }
    Get-ChildItem $out -Recurse -Filter meta.json -EA SilentlyContinue |
        Where-Object { $_.FullName -notmatch '\\stopped-' -and $_.LastWriteTime -gt $since } | ForEach-Object {
            try { $m = Get-Content $_.FullName -Raw | ConvertFrom-Json } catch { return }
            if ($m.checks_after.page_recorded_nothing) {
                $rel = $_.Directory.FullName.Substring($out.Length + 1)
                if ($NoStop) { if (-not $seen.ContainsKey("$rel $($_.LastWriteTime.Ticks)")) { $seen["$rel $($_.LastWriteTime.Ticks)"] = 1; Log "guard: page recorded nothing in $rel (status=$($m.status)); not stopped (-NoStop)" }; return }
                StopAll "page recorded nothing in $rel (status=$($m.status))"
            }
        }
    if (-not (Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'powershell.exe' -and $_.CommandLine -match 'chain\.ps1' })) {
        Log 'guard end: no chain.ps1 process'; exit 0
    }
    # a VS Code window over the place of the capture window made the page hidden (2026-11-10): keep them minimized
    $mn = & 'C:\work\v7s-chain\minimize.ps1' -All  # -All added 2026-11-11: every large window but the capture Chrome
    if ($mn -notmatch '^minimized 0 ') { Log "guard: $mn" }
    Start-Sleep -Seconds 15
}
