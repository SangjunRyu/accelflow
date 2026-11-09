param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$Distro = 'Ubuntu-24.04',
    [string]$LinuxProjectRoot = '',
    [string]$BuildRoot = (Join-Path $env:LOCALAPPDATA 'AccelFlow\cuda-matmul'),
    [string]$ReuseFixturesFrom = '',
    [string[]]$Variants = @('naive','tiled'),
    [int]$Trials = 3
)
$ErrorActionPreference = 'Stop'
if (-not $LinuxProjectRoot) {
    if ($projectRoot -match '^\\\\(?:wsl\.localhost|wsl\$)\\([^\\]+)(\\.*)$') {
        $Distro = $Matches[1]
        $LinuxProjectRoot = $Matches[2].Replace('\', '/')
    } else {
        throw 'Set -LinuxProjectRoot to the WSL project directory.'
    }
}
$stamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffffffZ')
$outputRoot = Join-Path $projectRoot "results\cuda_matmul\$stamp"
$linuxOutput = "$LinuxProjectRoot/results/cuda_matmul/$stamp"
New-Item -ItemType Directory -Path $outputRoot,$BuildRoot -Force | Out-Null

# The NPU producer exports the actual BF16 inputs, so C++ RNG differences cannot change the test.
if ($ReuseFixturesFrom) {
    $manifest = Get-Content -Raw -LiteralPath (Join-Path $ReuseFixturesFrom 'fixtures.json') | ConvertFrom-Json
    if (-not $manifest.passed) { throw 'The reused NPU fixtures did not pass.' }
    Copy-Item -LiteralPath (Join-Path $ReuseFixturesFrom 'fixtures.json') -Destination $outputRoot
    Copy-Item -LiteralPath (Join-Path $ReuseFixturesFrom 'scratch') -Destination $outputRoot -Recurse
    foreach ($fixture in $manifest.fixtures) {
        Copy-Item -LiteralPath (Join-Path $ReuseFixturesFrom $fixture.npu_report) -Destination $outputRoot
    }
    $ReuseFixturesFrom | Set-Content -LiteralPath (Join-Path $outputRoot 'fixture-source.txt')
} else {
    & wsl.exe -d $Distro -- python3 "$LinuxProjectRoot/tools/report/export_matmul_fixtures.py" --output $linuxOutput 2>&1 |
        Tee-Object -FilePath (Join-Path $outputRoot 'fixture-export.log')
    if ($LASTEXITCODE -ne 0) { throw 'NPU input export or validation failed.' }
}

$nvcc = (Get-Command nvcc.exe -ErrorAction Stop).Source
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$visualStudio = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $visualStudio) { throw 'MSVC x64 Build Tools were not found.' }
$developerCommand = Join-Path $visualStudio 'Common7\Tools\VsDevCmd.bat'
$source = Join-Path $BuildRoot 'matmul.cu'
$binary = Join-Path $BuildRoot 'matmul.exe'
Copy-Item -LiteralPath (Join-Path $projectRoot 'cuda\matmul\matmul.cu') -Destination $source
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
if ($LASTEXITCODE -ne 0) { throw "CUDA build failed; see $outputRoot\cuda-build.log" }
Copy-Item -LiteralPath $buildCommand -Destination (Join-Path $outputRoot 'build-command.txt')
& $nvcc --version | Set-Content -LiteralPath (Join-Path $outputRoot 'nvcc-version.txt')
& wsl.exe -d $Distro -- nvidia-smi | Set-Content -LiteralPath (Join-Path $outputRoot 'nvidia-smi.txt')

$localFixtures = Join-Path $BuildRoot 'fixtures'
New-Item -ItemType Directory -Path $localFixtures -Force | Out-Null
Get-ChildItem -LiteralPath (Join-Path $outputRoot 'scratch') -Filter '*.bin' |
    Copy-Item -Destination $localFixtures
& python.exe (Join-Path $projectRoot 'tools\report\benchmark_cuda_matmul.py') --output $outputRoot --executable $binary --local-fixtures $localFixtures --trials $Trials --variants $Variants 2>&1 |
    Tee-Object -FilePath (Join-Path $outputRoot 'cuda-runner.log')
if ($LASTEXITCODE -ne 0) { throw 'CUDA validation or benchmarking failed.' }
Write-Output "RESULTS=$outputRoot"
