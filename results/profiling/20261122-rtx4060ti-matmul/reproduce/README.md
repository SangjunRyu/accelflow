# 캡처 다시 실행

Windows에서 CUDA Toolkit 13.5, Nsight Systems 2026.5.2, Nsight Compute 2026.4.0과 NumPy가 설치된 64비트 Python을 사용한다. 이 파일이 있는 디렉터리에서 실행한다. 다른 경로에 설치한 경우 변수와 profile_matmul.py의 CUDA 경로를 맞춘다.

```powershell
$py = 'python'
$nsys = 'C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.5.2\target-windows-x64\nsys.exe'
$ncu = 'C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.4.0\target\windows-desktop-win7-x64\ncu.exe'
& $py profile_matmul.py --compile ../../../../cuda/matmul/matmul.cu --cubin reproduce-sm89.cubin

$sections = @('SpeedOfLight', 'LaunchStats', 'Occupancy', 'MemoryWorkloadAnalysis', 'MemoryWorkloadAnalysis_Tables', 'SchedulerStats', 'WarpStateStats')
$sectionArgs = @()
foreach ($section in $sections) { $sectionArgs += @('--section', $section) }
$bankMetrics = 'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum'

foreach ($variant in @('naive', 'tiled')) {
  $app = @('profile_matmul.py', '--cubin', 'reproduce-sm89.cubin', '--variant', $variant)
  & $nsys profile --trace=cuda --sample=none --cpuctxsw=none --output "reproduce-systems-$variant" $py @app --output "reproduce-systems-$variant.json" --samples 2 --pipeline 4
  & $ncu --target-processes all --clock-control none --launch-skip 20 --launch-count 1 @sectionArgs --metrics $bankMetrics --export "reproduce-compute-$variant" $py @app --output "reproduce-compute-$variant.json" --samples 1 --pipeline 0
}
```
