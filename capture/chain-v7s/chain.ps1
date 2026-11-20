# v7s: run blocks 1..6 of works/prototype_v7s_plan.py's runs.json one after another on this PC alone (no network needed).
# Written 2026-11-15 from v7's chain.ps1 (capture/chain): the same footer, set-aside and check steps; v7s's folders and
# key; no LLM (the ollama checks are dropped); Chrome's and Firefox's versions and update stops are checked before
# every block (works/conditions/prototype_v7s.json, captures).
# Writes counts and times only to C:\work\v7s-chain\chain.log (no press, key, text or per-item seconds).
param([Parameter(Mandatory = $true)][string]$FirefoxVersion, [switch]$CheckOnly)
$ErrorActionPreference = 'Stop'
$env:UV_OFFLINE = '1'
$repo = 'C:\work\affective-tactility-keystroke'
$out = 'C:\work\v7s-captures'
$logf = 'C:\work\v7s-chain\chain.log'
$rl = @('-m', 'capture.runlist', '--list', 'C:\work\v7s-plan\runs.json', '--key', '20261115', '7', '1', '--out', $out)
$ff = 'C:\Program Files\Mozilla Firefox'
Set-Location $repo

function Log($m) { $l = "$((Get-Date).ToUniversalTime().ToString('o')) $m"; Add-Content -Encoding utf8 $logf $l; Write-Host $l }
function RunlistAlive { (Get-CimInstance Win32_Process | Where-Object { ($_.Name -in @('python.exe', 'uv.exe')) -and $_.CommandLine -match '-m capture\.runlist' } | Measure-Object).Count -gt 0 }
function Footer($b) {
    $f = Join-Path $out "runlog.b$b.ndjson"
    if (-not (Test-Path $f)) { return $null }
    $l = @(Get-Content $f | ForEach-Object { $_ | ConvertFrom-Json })
    if ($l.Count -gt 0 -and $l[-1].type -eq 'footer') { return $l }
    return $null
}
function Summary($b) {
    $f = Join-Path $out "runlog.b$b.ndjson"
    if (-not (Test-Path $f)) { return "block ${b}: no runlog" }
    $l = @(Get-Content $f | ForEach-Object { $_ | ConvertFrom-Json })
    $items = $l | Where-Object type -eq item | Group-Object result | ForEach-Object { "$($_.Name)=$($_.Count)" }
    $end = if ($l[-1].type -eq 'footer') { "end=$($l[-1].end_utc) counts=$($l[-1].counts | ConvertTo-Json -Compress)" } else { 'NO FOOTER' }
    "block ${b}: start=$($l[0].start_utc) $end items: $($items -join ' ')"
}
function Checks {
    $bad = @()
    if ((Get-Item 'C:\Program Files\Google\Chrome\Application\chrome.exe').VersionInfo.ProductVersion -ne '156.0.8078.80') { $bad += 'chrome version' }
    Get-Service GoogleUpdater* | ForEach-Object { if ($_.StartType -ne 'Disabled' -or $_.Status -ne 'Stopped') { $bad += "$($_.Name)" } }
    if ((Get-Item "$ff\firefox.exe").VersionInfo.ProductVersion -ne $FirefoxVersion) { $bad += 'firefox version' }
    $pol = Join-Path $ff 'distribution\policies.json'
    if (-not (Test-Path $pol)) { $bad += 'firefox policies.json missing' }
    elseif (-not ((Get-Content $pol -Raw | ConvertFrom-Json).policies.DisableAppUpdate -eq $true)) { $bad += 'firefox DisableAppUpdate not true' }
    $s = Get-Service chromoting -ErrorAction SilentlyContinue
    if ($s -and ($s.Status -ne 'Stopped' -or $s.StartType -ne 'Manual')) { $bad += 'chromoting' }
    if (Get-Process remoting_host -ErrorAction SilentlyContinue) { $bad += 'remoting_host running' }
    if ((nvidia-smi --query-gpu=driver_version --format=csv,noheader).Trim() -ne '552.74') { $bad += 'driver' }
    if ((git status --short) -ne $null) { $bad += 'git status not empty' }
    $junk = Get-ChildItem C:\work\v7s-plan -Recurse -Force -File | Where-Object { $_.Name -in 'Thumbs.db', 'desktop.ini' }
    if ($junk) { $bad += 'plan folder has Thumbs.db/desktop.ini' }
    if (Get-CimInstance Win32_Process | Where-Object { $_.Name -in @('chrome.exe', 'firefox.exe') -and $_.CommandLine -match 'v7-capture-profile' }) { $bad += 'a capture browser is running before the block' }
    return $bad
}
function RunBlock($b) {
    Log "block $b start (runlist)"
    & uv run python @rl --block $b
    Log "block $b runlist exit=$LASTEXITCODE"
}

if ($CheckOnly) { $bad = Checks; "checks differ: [$($bad -join ', ')] (empty = all equal)"; "RunlistAlive=$(RunlistAlive)"; exit 0 }

Log "chain start (firefox $FirefoxVersion)"
if (RunlistAlive) { Log 'STOP: a runlist is already running'; exit 1 }
foreach ($b in 1..6) {
    if (Footer $b) { Log "block $b already has a footer; skipped"; continue }
    $bad = Checks
    if ($bad.Count) { Log "STOP before block ${b}: checks differ: $($bad -join ', ')"; exit 1 }
    Log "checks before block ${b}: all equal"
    RunBlock $b
    Log (Summary $b)
    if (-not (Footer $b)) {
        Log "block $b has no footer: set aside once (--set-aside) and run again once"
        & uv run python @rl --block $b --set-aside
        Log "set-aside exit=$LASTEXITCODE"
        if ($LASTEXITCODE -ne 0) { Log "STOP: set-aside of block $b failed"; exit 1 }
        $sa = Get-ChildItem $out -Directory -Filter 'stopped-*' | Sort-Object Name | Select-Object -Last 1
        $j = Get-Content (Join-Path $sa.FullName 'set_aside.json') -Raw | ConvertFrom-Json
        Log "set aside in $($sa.Name): to_move=$(@($j.to_move).Count) moved=$(@($j.moved).Count) cut_item=$($j.cut_item) origin=$($j.origin | ConvertTo-Json -Compress)"
        if (@($j.to_move).Count -ne @($j.moved).Count) { Log 'STOP: moved differs from to_move'; exit 1 }
        RunBlock $b
        Log (Summary $b)
        if (-not (Footer $b)) { Log "STOP: block $b has no footer after one run again"; exit 1 }
    }
}
Log 'chain done: blocks 1-6 have footers'
