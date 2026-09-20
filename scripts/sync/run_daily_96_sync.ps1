<#
Daily 96-point DB synchronization entry point.

The project owner only needs to run the crawler/update job that makes the
remote database current.  This wrapper then pulls the two required 96-point
tables from DB, uses a 7-day overlap for late backfills, and writes an atomic
local mirror plus a manifest.  It deliberately does not read or rotate
browser cookies.

Register once in Windows Task Scheduler, for example:
  schtasks /Create /SC DAILY /ST 02:30 /TN "EFM3-96-DB-Sync" /TR `
    "powershell.exe -NoProfile -ExecutionPolicy Bypass -File <full-path>\run_daily_96_sync.ps1"
#>
[CmdletBinding()]
param(
    [int]$OverlapDays = 7,
    [switch]$IncludeExtended
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\")).Path
$Python = if ($env:EPF_PYTHON -and (Test-Path -LiteralPath $env:EPF_PYTHON)) {
    $env:EPF_PYTHON
} elseif (Test-Path -LiteralPath "D:\computer_download\environment\conda\epf-2\python.exe") {
    "D:\computer_download\environment\conda\epf-2\python.exe"
} else {
    "python"
}
$LogRoot = Join-Path $ProjectRoot "outputs\96\sync\logs"
New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogPath = Join-Path $LogRoot "daily_db_sync_$stamp.log"
$LockPath = Join-Path $LogRoot "daily_db_sync.lock"

try {
    $lock = [System.IO.File]::Open($LockPath, [System.IO.FileMode]::CreateNew, [System.IO.FileAccess]::Write, [System.IO.FileShare]::None)
} catch {
    Write-Error "Another 96-point sync is already running: $LockPath"
    exit 2
}

try {
    "[$(Get-Date -Format o)] Starting 96-point DB sync" | Tee-Object -FilePath $LogPath
    "[$(Get-Date -Format o)] ProjectRoot=$ProjectRoot" | Tee-Object -FilePath $LogPath -Append
    "[$(Get-Date -Format o)] Source=db Mode=incremental OverlapDays=$OverlapDays" | Tee-Object -FilePath $LogPath -Append
    $args = @(
        "-u", "main.py", "--pipeline", "sync_dataset", "--resolution", "15min",
        "--sync-source", "db", "--sync-mode", "incremental",
        "--sync-overlap-days", "$OverlapDays"
    )
    if ($IncludeExtended) { $args += "--include-extended" }
    Push-Location $ProjectRoot
    try {
        # Python logging defaults to stderr. Piping a native command directly
        # through Windows PowerShell 5.1 makes stderr look like
        # NativeCommandError and can abort under ErrorAction=Stop. Capture
        # both streams with Start-Process, then append them to the single
        # diagnostic log; the process exit code is authoritative.
        $stdoutPath = "$LogPath.stdout.partial"
        $stderrPath = "$LogPath.stderr.partial"
        Remove-Item -LiteralPath $stdoutPath, $stderrPath -Force -ErrorAction SilentlyContinue
        $process = Start-Process -FilePath $Python -ArgumentList $args `
            -WorkingDirectory $ProjectRoot -Wait -PassThru -NoNewWindow `
            -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
        if (Test-Path -LiteralPath $stdoutPath) {
            Get-Content -LiteralPath $stdoutPath | Tee-Object -FilePath $LogPath -Append
        }
        if (Test-Path -LiteralPath $stderrPath) {
            Get-Content -LiteralPath $stderrPath | Tee-Object -FilePath $LogPath -Append
        }
        $exitCode = $process.ExitCode
        Remove-Item -LiteralPath $stdoutPath, $stderrPath -Force -ErrorAction SilentlyContinue
    } finally {
        Pop-Location
    }
    "[$(Get-Date -Format o)] Finished exit_code=$exitCode" | Tee-Object -FilePath $LogPath -Append
    exit $exitCode
} finally {
    if ($lock) { $lock.Dispose() }
    Remove-Item -LiteralPath $LockPath -Force -ErrorAction SilentlyContinue
}
