param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$Distro = 'Ubuntu-24.04',
    [string]$LinuxProjectRoot = '',
    [string]$BuildRoot = (Join-Path $env:LOCALAPPDATA 'AccelFlow\cuda-rmsnorm'),
    [string]$ResultsName = '',
    [int]$Trials = 3
)
$ErrorActionPreference = 'Stop'
if (-not $LinuxProjectRoot) {
    if ($ProjectRoot -match '^\\\\(?:wsl\.localhost|wsl\$)\\([^\\]+)(\\.*)$') {
        $Distro = $Matches[1]
        $LinuxProjectRoot = $Matches[2].Replace('\', '/')
    } else { throw 'Set -LinuxProjectRoot to the WSL project directory' }
}
if (-not $ResultsName) { $ResultsName = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffffffZ') }
if ($ResultsName -notmatch '^[a-zA-Z0-9_-]+$') { throw 'ResultsName must be a single directory name' }
$outputRoot = Join-Path $ProjectRoot "results\rmsnorm\$ResultsName"
$linuxOutput = "$LinuxProjectRoot/results/rmsnorm/$ResultsName"
New-Item -ItemType Directory -Path $outputRoot,$BuildRoot -Force | Out-Null
if (Test-Path -LiteralPath (Join-Path $outputRoot 'cuda-summary.json')) { throw 'CUDA results already exist; choose a new ResultsName' }
if (-not (Test-Path -LiteralPath (Join-Path $outputRoot 'npu-summary.json'))) {
    & wsl.exe -d $Distro -- python3 "$LinuxProjectRoot/tools/report/run_rmsnorm.py" --output $linuxOutput 2>&1 |
        Tee-Object -FilePath (Join-Path $outputRoot 'npu-runner.log')
    if ($LASTEXITCODE -ne 0) { throw 'NPU RMSNorm validation failed' }
}
$npu = Get-Content -Raw -LiteralPath (Join-Path $outputRoot 'npu-summary.json') | ConvertFrom-Json
if (-not $npu.passed) { throw 'NPU summary did not pass' }
$nvcc = (Get-Command nvcc.exe -ErrorAction Stop).Source
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$vs = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) { throw 'MSVC x64 Build Tools not found' }
$developer = Join-Path $vs 'Common7\Tools\VsDevCmd.bat'
$source = Join-Path $BuildRoot 'rmsnorm.cu'; $binary = Join-Path $BuildRoot 'rmsnorm.exe'
Copy-Item -LiteralPath (Join-Path $ProjectRoot 'cuda\rmsnorm\rmsnorm.cu') -Destination $source
$buildCommand = Join-Path $BuildRoot 'build.cmd'
@"
@echo off
call "$developer" -arch=x64 -host_arch=x64
if errorlevel 1 exit /b 1
cd /d "$BuildRoot"
"$nvcc" -std=c++17 -O3 -arch=sm_61 -lineinfo -Xcompiler=/W4,/utf-8 "$source" -o "$binary"
if errorlevel 1 exit /b 1
"@ | Set-Content -LiteralPath $buildCommand -Encoding ascii
& $env:ComSpec /d /c call $buildCommand > (Join-Path $outputRoot 'cuda-build.log') 2>&1
if ($LASTEXITCODE -ne 0) { throw 'CUDA RMSNorm build failed' }
Copy-Item $buildCommand (Join-Path $outputRoot 'cuda-build-command.txt')
& $nvcc --version | Set-Content (Join-Path $outputRoot 'nvcc-version.txt')
& wsl.exe -d $Distro -- nvidia-smi | Set-Content (Join-Path $outputRoot 'nvidia-smi-before.txt')
$localFixtures = Join-Path $BuildRoot 'fixtures'
New-Item -ItemType Directory -Path $localFixtures -Force | Out-Null
Get-ChildItem -LiteralPath (Join-Path $outputRoot 'scratch') -Filter '*.bin' | Copy-Item -Destination $localFixtures
$sources = @('cuda\rmsnorm\rmsnorm.cu','scripts\run_cuda_rmsnorm.ps1','tools\report\benchmark_cuda_rmsnorm.py')
$hashes = @{}
foreach ($relative in $sources) {
    $original = Join-Path $ProjectRoot $relative
    $destination = Join-Path (Join-Path $outputRoot 'cuda-sources') $relative
    New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force | Out-Null
    Copy-Item -LiteralPath $original -Destination $destination
    $hashes[$relative] = (Get-FileHash -LiteralPath $original).Hash.ToLower()
}
$hashes | ConvertTo-Json | Set-Content (Join-Path $outputRoot 'cuda-source-sha256.json') -Encoding utf8
& python.exe (Join-Path $ProjectRoot 'tools\report\benchmark_cuda_rmsnorm.py') --output $outputRoot --executable $binary --local-fixtures $localFixtures --trials $Trials 2>&1 |
    Tee-Object -FilePath (Join-Path $outputRoot 'cuda-runner.log')
if ($LASTEXITCODE -ne 0) { throw 'CUDA RMSNorm tests failed' }
& wsl.exe -d $Distro -- python3 "$LinuxProjectRoot/tools/report/audit_rmsnorm.py" --output $linuxOutput
if ($LASTEXITCODE -ne 0) { throw 'Cross-backend RMSNorm audit failed' }
Write-Output "RESULTS=$outputRoot"
