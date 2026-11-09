# Read-only Windows resource report. Run in the normal host PowerShell session.
$ErrorActionPreference = 'Stop'
$registration = Get-ChildItem -LiteralPath 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Lxss' |
    ForEach-Object { Get-ItemProperty -LiteralPath $_.PSPath } |
    Where-Object DistributionName -EQ 'Ubuntu-24.04'
if (@($registration).Count -ne 1) {
    throw 'Expected one registered Ubuntu-24.04 distribution.'
}
$basePath = $registration.BasePath -replace '^\\\?\\', ''
$vhdName = if ($registration.VhdFileName) { $registration.VhdFileName } else { 'ext4.vhdx' }
$vhdFile = Get-Item -LiteralPath (Join-Path $basePath $vhdName)
$volume = Get-Volume -FilePath $vhdFile.FullName
$computer = Get-CimInstance Win32_ComputerSystem
$os = Get-CimInstance Win32_OperatingSystem
[ordered]@{
    captured_at = (Get-Date).ToString('o')
    distribution = $registration.DistributionName
    wsl_version = $registration.Version
    vhdx_path = $vhdFile.FullName
    vhdx_file_length_bytes = $vhdFile.Length
    backing_volume = $volume.DriveLetter
    volume_size_bytes = $volume.Size
    volume_free_bytes = $volume.SizeRemaining
    host_physical_memory_bytes = $computer.TotalPhysicalMemory
    host_free_physical_memory_kib = $os.FreePhysicalMemory
} | ConvertTo-Json
