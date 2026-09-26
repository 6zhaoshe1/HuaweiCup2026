# Install NVIDIA 581.80 (last Windows 10 compatible notebook DCH package) with a rollback path.
# ASCII-only on purpose (Windows PowerShell 5.1 reads UTF-8-without-BOM as GBK).
# Run elevated:  powershell -NoProfile -ExecutionPolicy Bypass -File <this file>
$ErrorActionPreference = 'Continue'

$dir = Join-Path $env:USERPROFILE 'Downloads\drivers'
$exe = Join-Path $dir '581.80-notebook-win10-win11-64bit-international-dch-whql.exe'
$log = Join-Path $dir 'install-nvidia-581.80.log'

if (-not (Test-Path $exe)) { Write-Host "installer not found: $exe"; exit 2 }

Start-Transcript -Path $log -Append -Force | Out-Null
Write-Host "=== install NVIDIA 581.80 started $(Get-Date -Format s) ==="

Write-Host "`n--- [1/4] record current display drivers ---"
Get-CimInstance Win32_VideoController |
  Select-Object Name, DriverVersion, DriverDate, CurrentHorizontalResolution, CurrentVerticalResolution |
  Format-Table -AutoSize | Out-String | Write-Host

Write-Host "`n--- [2/4] export all third-party driver packages (rollback source) ---"
$bak = Join-Path $dir 'backup-driverstore'
New-Item -ItemType Directory -Force -Path $bak | Out-Null
pnputil /export-driver * $bak
Write-Host ("exported packages: {0}" -f (Get-ChildItem $bak -Directory -ErrorAction SilentlyContinue).Count)

Write-Host "`n--- [3/4] system restore point ---"
try {
  Enable-ComputerRestore -Drive 'C:\' -ErrorAction SilentlyContinue
  Checkpoint-Computer -Description 'before-nvidia-581.80' -RestorePointType MODIFY_SETTINGS
  Write-Host "restore point created"
} catch {
  Write-Host ("restore point skipped: " + $_.Exception.Message)
}

Write-Host "`n--- [4/4] silent clean install ---"
$p = Start-Process -FilePath $exe -ArgumentList '-s', '-clean', '-noreboot' -Wait -PassThru
Write-Host ("installer exit code: {0}" -f $p.ExitCode)

Write-Host "`n--- result ---"
Get-CimInstance Win32_VideoController |
  Select-Object Name, DriverVersion, DriverDate, CurrentHorizontalResolution, CurrentVerticalResolution |
  Format-Table -AutoSize | Out-String | Write-Host
Write-Host "=== done $(Get-Date -Format s) : reboot required ==="
Stop-Transcript | Out-Null
