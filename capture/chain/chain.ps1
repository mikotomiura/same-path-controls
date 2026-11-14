# v7 production: run blocks 2..6 one after another on this PC alone (no network needed).
# Waits for the running block 1 to end. After each block: footer -> environment checks -> next block.
# A block that ends without a footer is set aside once with the runlist's --set-aside and run again once;
# if it still has no footer, or a check differs, the chain stops (nothing further runs).
# Writes counts and times only to C:\work\v7-chain\chain.log (no press, key, text or per-item seconds).
$ErrorActionPreference = 'Stop'
$env:UV_OFFLINE = '1'
$repo = 'C:\work\affective-tactility-keystroke'
$out = 'C:\work\v7-captures'
$logf = 'C:\work\v7-chain\chain.log'
$rl = @('-m', 'capture.runlist', '--list', 'C:\work\v7-plan\runs.json', '--key', '20261031', '7', '6', '--out', $out)
Set-Location $repo

function Log($m) { $l = "$((Get-Date).ToUniversalTime().ToString('o')) $m"; Add-Content -Encoding utf8 $logf $l; Write-Host $l }
# only a real runlist (python or uv), never a console window whose command line names it (fixed 2026-11-09)
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
    $s = Get-Service chromoting
    if ($s.Status -ne 'Stopped' -or $s.StartType -ne 'Manual') { $bad += 'chromoting' }
    if (Get-Process remoting_host -ErrorAction SilentlyContinue) { $bad += 'remoting_host running' }
    if ((ollama --version) -notmatch '0\.40\.13') { $bad += 'ollama version' }
    $ol = (ollama list) -join "`n"
    if ($ol -notmatch 'qwen2\.5:7b-instruct-q4_K_M\s+845dbda0ea48') { $bad += '7b id' }
    if ($ol -notmatch 'qwen2\.5:14b-instruct-q4_K_M\s+7cdf5a0187d5') { $bad += '14b id' }
    if ((nvidia-smi --query-gpu=driver_version --format=csv,noheader).Trim() -ne '552.74') { $bad += 'driver' }
    $want = @{ 'KUPA-KEYS-META.csv' = '4CE485492F37C0F4B63F27AD0518B7C27E67A31447684EEFDF9066653423EAEE';
               'KUPA-KEYS-TASK-1.csv' = 'B26AD0FCA7E34F8F4E87DCCD0C04B95ADC082CC8DB4F87DC6BE8D6224F5F10F2';
               'KUPA-KEYS-TASK-2.csv' = '4AD24CBF78879F2FB45C1334EB827F2C8C2FADA09A957252A4C86B8E20F28AEF' }
    foreach ($k in $want.Keys) { if ((Get-FileHash "external\kupa-keys\$k" -Algorithm SHA256).Hash -ne $want[$k]) { $bad += $k } }
    if ((git status --short) -ne $null) { $bad += 'git status not empty' }
    $junk = Get-ChildItem C:\work\v7-plan -Recurse -Force -File | Where-Object { $_.Name -in 'Thumbs.db', 'desktop.ini' }
    if ($junk) { $bad += 'plan folder has Thumbs.db/desktop.ini' }
    # added 2026-11-10: a capture browser left from an earlier capture (block 3's hidden pages)
    if (Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'chrome.exe' -and $_.CommandLine -match 'v7-capture-profile' }) { $bad += 'a capture browser is running before the block' }
    return $bad
}
function RunBlock($b) {
    Log "block $b start (runlist)"
    & uv run python @rl --block $b
    Log "block $b runlist exit=$LASTEXITCODE"
}

if ($args -contains '-CheckOnly') { $bad = Checks; "checks differ: [$($bad -join ', ')] (empty = all equal)"; Summary 1; "RunlistAlive=$(RunlistAlive)"; exit 0 }

Log "chain start; waiting for the running block to end"
while (RunlistAlive) { Start-Sleep -Seconds 60 }
Log (Summary 1)
if (-not (Footer 1)) { Log 'STOP: block 1 has no footer (left for the researcher; not set aside by the chain)'; exit 1 }

foreach ($b in 2..6) {
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
