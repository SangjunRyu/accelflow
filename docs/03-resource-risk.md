# 03. 자원·위험 보고서 — Phase 0-B / 0-C

후속 업데이트: 자원 재확인과 실제 감시를 통해 예제 target 빌드를 완료했습니다.
빌드 후 C: 여유는 약 30.99GiB, checkout 사용량은 약 3.3GiB였습니다.
아래 보류 판정은 이전 계획 시점의 기록입니다. [실제 실행 결과](06-tt-metal-build-validation.md).

상태: 호스트 자원 확인·최소 의존성 설치 완료; 후속 제한된 target 빌드도 완료.
[진행 순서](README.md) · 이전: [02. 초기 설치 계획](02-installation-plan.md) · 다음: [04. TTSim 설치](04-ttsim-installation.md)

## Phase 0-C refresh

C: free space increased to 35,717,005,312 bytes (33.26 GiB).
WSL available RAM is about 3.23 GiB, swap is 1 GiB. Host available RAM at the later
snapshot was 873,732 KiB (0.83 GiB), with unchanged 7.89 GiB physical RAM.
The official TTSim binary was safely installed (227,544 bytes), with standalone
validation peak RSS 146,188 KiB. TT-Metal source and build were not started.
The new [TT-Metal plan](05-tt-metal-stage-plan.md) calculates a 15–34 GiB additional
planning budget plus a 10 GiB host reserve. Its conservative 44 GiB free-space
criterion is not an upstream minimum and is not met. Actual build usage is unknown.
The following section preserves the earlier pre-dependency-installation snapshot.

Windows registry, file metadata, Get-Volume, and Win32 CIM queries were read.
Recorded values: [host-resources JSON](../results/setup/host-resources-2026-10-06.json).
Reinspect with `scripts/check_host_resources.ps1` in host PowerShell.

| Item | Observed result |
| --- | --- |
| Ubuntu distribution | Ubuntu-24.04, WSL version 2 |
| Actual backing file | `C:\Users\ysjun5656\AppData\Local\wsl\{903dd91c-8ce5-45d9-b15f-9bbfce2d29dc}\ext4.vhdx` |
| Backing file length | 33,411,825,664 bytes (31.12 GiB); file length, not allocated-cluster measurement |
| Backing volume | C: |
| Volume capacity | 1,023,000,178,688 bytes (952.74 GiB) |
| Actual volume free space | 13,134,454,784 bytes (12.23 GiB) at inspection |
| Host physical RAM | 8,476,680,192 bytes (7.89 GiB, nominal 8 GB system) |
| Host available RAM | 1,587,136 KiB (1.51 GiB), transient snapshot |
| WSL allocation | About 3.8 GiB RAM and 1 GiB swap |

The Linux ext4 filesystem's 931 GiB logical free space does not represent available
host storage. This Ubuntu VHDX grows on C:, so external checkouts under Linux home
also consume C: space.

## Decision

Minimal dependency installation may proceed after apt's proposed download and disk
usage are reviewed. Retain at least 5 GiB host free space for this small installation;
this is a local precaution, not an upstream requirement. No TT-Metal build is allowed
in this step. The current host memory pressure and 12.23 GiB disk headroom do not
pass the resource review for a full clone/submodule/build workflow.

Apt's reviewed proposal: 24.1 MB download, 74.8 MB additional disk usage,
23 new packages and 6 updates (curl/OpenSSL dependencies), no removals.
This passes the minimal-install review. Git, build-essential, Python development/venv,
certificates, and wget were already present; apt preserves them at their current versions.

Before a later build, aim for at least 40–50 GiB physical free space for sources,
dependencies, artifacts, and host headroom. This is a conservative planning budget,
not a measured upstream minimum. Check the actual checkout/build usage and allow
more if needed. Close memory-heavy applications and start with one build job.

## Recommended WSL settings (proposal only)

For an 8 GB host, retain a 4 GB WSL memory ceiling rather than raising it to 6–8 GB.
Keep swap at 1 GB while disk space is tight. After freeing disk space, 2 GB swap is a
reasonable starting point for a one-job build; heavy swapping is not a guarantee
that a large build will complete. A host RAM upgrade or larger remote Linux machine
may be needed if the build exceeds this budget.

Suggested entries to merge into the existing `[wsl2]` section later:

```ini
memory=4GB
processors=4
swap=2GB
```

Preserve the existing custom `kernel` entry. CPU limit 4 is optional to leave host
capacity; build parallelism should still be 1. This is a recommendation based on the
observed host, not a change applied to `.wslconfig`.

[Microsoft's configuration reference](https://learn.microsoft.com/en-us/windows/wsl/wsl-config)
documents these settings and their scope across WSL2 distributions. Applying changes
requires a WSL restart; no shutdown or restart was performed for this task.

## Windows CMake versus Ubuntu CMake

The user's Windows installation is CMake 3.27.6 at
`C:\Program Files\CMake\bin\cmake.exe`. TT-Metal's native Linux build needs a Linux
CMake, Ninja, and Linux compiler. Windows CMake does not satisfy that dependency.
The minimal apt CMake is a baseline; the selected TT-Metal tag's toolchain requirements
must be checked separately before a future build.
The actual `v0.78.0` top-level source was fetched from GitHub inside WSL and contains
`cmake_minimum_required(VERSION 3.24...4.2)`. Ubuntu CMake 3.28.3 meets that minimum;
the upper range is a policy compatibility declaration, not a demand to install 4.2.
This does not establish requirements of all submodules or prove full build compatibility.

## Scope

Install only baseline build dependencies. Prefer the pinned official prebuilt Wormhole
TTSim library for the subsequent binary stage. No source build of TTSim, TT-Metal
clone/build, physical-device drivers/tools, Docker setup, or WSL configuration edits.
