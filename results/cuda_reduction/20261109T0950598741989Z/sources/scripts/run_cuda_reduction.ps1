param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$Distro = 'Ubuntu-24.04',
    [string]$LinuxProjectRoot = '/home/ysjun5656/dev/accelflow',
    [string]$BuildRoot = (Join-Path $env:LOCALAPPDATA 'AccelFlow\cuda-reduction'),
    [int]$Trials = 3
)
$ErrorActionPreference = 'Stop'
if ($Trials -lt 1 -or $Trials -gt 10) { throw 'Trials must be 1..10' }
$stamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffffffZ')
$outputRoot = Join-Path $ProjectRoot "results\cuda_reduction\$stamp"
$linuxOutput = "$LinuxProjectRoot/results/cuda_reduction/$stamp"
New-Item -ItemType Directory -Path $outputRoot,$BuildRoot -Force | Out-Null

& wsl.exe -d $Distro -- python3 "$LinuxProjectRoot/tools/report/export_reduction_fixtures.py" --output $linuxOutput 2>&1 |
    Tee-Object -FilePath (Join-Path $outputRoot 'fixture-export.log')
if ($LASTEXITCODE -ne 0) { throw 'NPU fixture export/validation failed' }

$nvcc = (Get-Command nvcc.exe -ErrorAction Stop).Source
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$visualStudio = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $visualStudio) { throw 'MSVC x64 Build Tools were not found' }
$developerCommand = Join-Path $visualStudio 'Common7\Tools\VsDevCmd.bat'
$source = Join-Path $BuildRoot 'reduction.cu'
$binary = Join-Path $BuildRoot 'reduction.exe'
Copy-Item -LiteralPath (Join-Path $ProjectRoot 'cuda\reduction\reduction.cu') -Destination $source
$buildCommand = Join-Path $BuildRoot 'build.cmd'
@"
@echo off
call "$developerCommand" -arch=x64 -host_arch=x64
if errorlevel 1 exit /b 1
cd /d "$BuildRoot"
"$nvcc" -std=c++17 -O3 -arch=sm_61 -lineinfo -Xcompiler=/W4,/utf-8 "$source" -o "$binary"
if errorlevel 1 exit /b 1
"@ | Set-Content -LiteralPath $buildCommand -Encoding ascii
& $env:ComSpec /d /c call $buildCommand > (Join-Path $outputRoot 'cuda-build.log') 2>&1
if ($LASTEXITCODE -ne 0) { throw "CUDA build failed: $outputRoot" }
Copy-Item -LiteralPath $buildCommand -Destination (Join-Path $outputRoot 'build-command.txt')
& $nvcc --version | Set-Content -LiteralPath (Join-Path $outputRoot 'nvcc-version.txt')
& wsl.exe -d $Distro -- nvidia-smi | Set-Content -LiteralPath (Join-Path $outputRoot 'nvidia-smi.txt')
$localFixtures = Join-Path $BuildRoot 'fixtures'
New-Item -ItemType Directory -Path $localFixtures -Force | Out-Null
Get-ChildItem -LiteralPath (Join-Path $outputRoot 'scratch') -Filter '*.bin' | Copy-Item -Destination $localFixtures

# Keep the exact source and hashes used for this execution beside the raw reports.
$sources = @('cuda\reduction\reduction.cu','npu\reduction\reduction.cpp','scripts\run_cuda_reduction.ps1',
             'tools\report\export_reduction_fixtures.py','tools\report\benchmark_cuda_reduction.py')
$hashes = @{}
foreach ($relative in $sources) {
    $original = Join-Path $ProjectRoot $relative
    $destination = Join-Path (Join-Path $outputRoot 'sources') $relative
    New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force | Out-Null
    Copy-Item -LiteralPath $original -Destination $destination
    $hashes[$relative] = (Get-FileHash -LiteralPath $original).Hash.ToLower()
}
$hashes | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $outputRoot 'source-sha256.json') -Encoding utf8
& python.exe (Join-Path $ProjectRoot 'tools\report\benchmark_cuda_reduction.py') --output $outputRoot --executable $binary --local-fixtures $localFixtures --trials $Trials 2>&1 |
    Tee-Object -FilePath (Join-Path $outputRoot 'cuda-runner.log')
if ($LASTEXITCODE -ne 0) { throw 'CUDA reduction validation or benchmark failed' }
Write-Output "RESULTS=$outputRoot"
