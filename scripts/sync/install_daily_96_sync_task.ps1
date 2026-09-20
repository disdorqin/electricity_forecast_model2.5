<#
Install or remove the per-user Windows Task Scheduler entry for the 96-point
database mirror.  The task invokes run_daily_96_sync.ps1; it never stores a
database password or browser cookie in the task definition.
#>
[CmdletBinding(SupportsShouldProcess)]
param(
    [string]$TaskName = "EFM3-96-DB-Sync",
    [ValidatePattern("^([01]\d|2[0-3]):[0-5]\d$")]
    [string]$At = "02:30",
    [switch]$Unregister
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Runner = Join-Path $ProjectRoot "scripts\sync\run_daily_96_sync.ps1"
if (-not (Test-Path -LiteralPath $Runner)) {
    throw "Daily sync runner not found: $Runner"
}

if ($Unregister) {
    if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
        if ($PSCmdlet.ShouldProcess($TaskName, "Unregister scheduled task")) {
            Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
            Write-Output "UNREGISTERED: $TaskName"
        }
    } else {
        Write-Output "NOT_FOUND: $TaskName"
    }
    exit 0
}

$powershell = Join-Path $env:SystemRoot "System32\WindowsPowerShell\v1.0\powershell.exe"
$actionArgs = "-NoProfile -ExecutionPolicy Bypass -File `"$Runner`""
$action = New-ScheduledTaskAction -Execute $powershell -Argument $actionArgs -WorkingDirectory $ProjectRoot
$trigger = New-ScheduledTaskTrigger -Daily -At ([datetime]::ParseExact($At, "HH:mm", $null))
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited

if ($PSCmdlet.ShouldProcess($TaskName, "Register daily 96-point DB sync at $At")) {
    Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
        -Settings $settings -Principal $principal -Description `
        "EFM3 96-point DB mirror; invokes run_daily_96_sync.ps1; no cookies or credentials stored." -Force | Out-Null
}

$task = Get-ScheduledTask -TaskName $TaskName -ErrorAction Stop
$taskAction = $task.Actions | Select-Object -First 1
Write-Output ("REGISTERED: {0}; State={1}; User={2}; Execute={3}; Arguments={4}" -f `
    $TaskName, $task.State, $principal.UserId, $taskAction.Execute, $taskAction.Arguments)
