# After the chain (2026-11-11, the researcher's decision: "自動で解析まで"; the network is cut from 19:00Z):
# waits for chain.ps1 to end; only if chain.log says 'chain done', checks the 7 runlog footers, an empty git status
# and that results\prototype_v7 does not exist, then runs step 7's analysis once. Its whole output goes to
# C:\work\v7-chain\analysis.log; on a non-zero exit the output is appended unchanged to questions.md (no code or
# capture is fixed). Nothing else runs (step 8 and the gaps count are done by Claude after the network is back).
param([int]$ChainPid)
$ErrorActionPreference = 'Continue'
$env:UV_OFFLINE = '1'
$repo = 'C:\work\affective-tactility-keystroke'
$out = 'C:\work\v7-captures'
$logf = 'C:\work\v7-chain\chain.log'
$alog = 'C:\work\v7-chain\analysis.log'
$q = 'C:\work\v4-notes\questions.md'
function Log($m) { Add-Content -Encoding utf8 $logf "$((Get-Date).ToUniversalTime().ToString('o')) $m" }
Log "post start; waiting for chain.ps1 pid $ChainPid"
while (Get-Process -Id $ChainPid -EA SilentlyContinue) { Start-Sleep -Seconds 60 }
$last = @(Get-Content $logf | Where-Object { $_ -notmatch ' (post|guard)[ :]' })[-1]
if ($last -notmatch 'chain done') { Log "post: chain did not end with 'chain done'; analysis not run"; exit 1 }
$bad = @()
foreach ($b in 0..6) {
    $f = Join-Path $out "runlog.b$b.ndjson"
    if (-not (Test-Path $f)) { $bad += "runlog.b$b missing"; continue }
    $l = Get-Content $f -Tail 1 | ConvertFrom-Json
    if ($l.type -ne 'footer') { $bad += "runlog.b$b no footer" }
}
Set-Location $repo
if ((git status --short) -ne $null) { $bad += 'git status not empty' }
if (Test-Path (Join-Path $repo 'results\prototype_v7')) { $bad += 'results\prototype_v7 exists' }
if ($bad.Count) { Log "post: checks differ: $($bad -join ', '); analysis not run"; exit 1 }
Log 'post: 7 footers, git status empty, results\prototype_v7 absent; analysis start'
& uv run python works/prototype_v7_run.py --captures C:\work\v7-captures --plan C:\work\v7-plan *>&1 |
    ForEach-Object { "$_" } | Set-Content -Encoding utf8 $alog
$code = $LASTEXITCODE
Log "post: analysis exit=$code"
if ($code -ne 0) {
    $t = (Get-Content $alog -Raw)
    Add-Content -Encoding utf8 $q "`n## step 7 の解析が止まった（$((Get-Date).ToUniversalTime().ToString('o'))、exit=$code、post.ps1 が自動で追記）`n`n``````n$t`n```````n"
}
