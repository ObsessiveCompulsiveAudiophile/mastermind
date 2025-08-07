# Mastermind: Weighted-Entropy Heuristic Implementations

This repository contains three standalone programs that implement a **weighted-entropy heuristic** for the classic *Mastermind* game, achieving an average of **4.3488** guesses for the full game set.

| Program | Purpose | Hardware |
|---------|---------|----------|
| `kernel.cu` | CUDA-accelerated optimiser that computes stage-based weights | NVIDIA GPU |
| `treeGenFixedWeights.cpp` | CPU-based generator using **fixed** weights | any x86-64 CPU |
| `treeGenStageWeights.cpp` | CPU-based generator using **stage-based** weights | any x86-64 CPU |

> See the author’s paper for a full description of the algorithm and benchmarks.

---

## 🚀 Quick start (most users)

Pre-compiled binaries are attached to every [GitHub release](https://github.com/obsessivecompulsiveaudiophile/mastermind/releases/latest):

| Archive | Contents |
|---------|----------|
| `mastermind-windows.zip` | Windows 10/11 64-bit executables (CUDA 12.4, built for compute ≥ 5.2) |
| `mastermind-linux.tar.gz` | Linux x86-64 executables (CUDA 11.8, glibc ≥ 2.35) |

1. Download & extract.
2. Run `mastermind_cuda_windows.exe` (Win) or `mastermind_cuda_linux` (Linux).

---

## 🛠️ Building from source

### Prerequisites

| OS | Required tool-chain | GPU driver |
|--|--|--|
| **Windows** | [CUDA Toolkit **12.4**](https://developer.nvidia.com/cuda-downloads) + Visual Studio 2022 (MSVC 14.4x) | NVIDIA driver ≥ 531 |
| **Linux** | [CUDA Toolkit **11.8**](https://developer.nvidia.com/cuda-11-8-0-download-archive) + GCC 11+ | NVIDIA driver ≥ 470 |

### Clone

```bash
git clone https://github.com/obsessivecompulsiveaudiophile/mastermind.git
cd mastermind

Windows (Visual Studio Developer Prompt)

cmd
:: CUDA program
nvcc -arch=sm_52 -O3 -o mastermind_cuda_windows.exe kernel.cu
Replace -arch=sm_52 with -arch=sm_86 (or your GPU’s compute capability) for maximum performance on newer cards.

:: CPU programs
cl /EHsc /O2 treeGenFixedWeights.cpp   /Fe:treeGenFixedWeights_windows.exe
cl /EHsc /O2 treeGenStageWeights.cpp   /Fe:treeGenStageWeights_windows.exe

Linux

bash
# CUDA program
nvcc -arch=sm_52 -O3 -o mastermind_cuda_linux kernel.cu

# CPU programs
g++ -O3 -std=c++17 treeGenFixedWeights.cpp   -o treeGenFixedWeights_linux
g++ -O3 -std=c++17 treeGenStageWeights.cpp   -o treeGenStageWeights_linux
