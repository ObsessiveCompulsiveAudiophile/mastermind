# Mastermind: Weighted Entropy Heuristic Implementations

This repository contains three standalone programs implementing a weighted entropy-based heuristic for the Mastermind game, achieving an expected number of guesses below 3.573:
- `kernel.cu`: CUDA-accelerated optimizer for NVIDIA GPUs.
- `tree_gen1.cpp`: CPU-based decision tree generator using optimized weights (Strategy 1).
- `tree_gen2.cpp`: CPU-based decision tree generator using optimized weights (Strategy 2).

See the [paper](link-to-your-latex-paper) for details on the heuristic and its performance.

## Prerequisites
- **Windows**:
  - NVIDIA CUDA Toolkit 11.8 (download from [NVIDIA](https://developer.nvidia.com/cuda-11-8-0-download-archive)).
  - Visual Studio 2022 Community with C++ Desktop Development workload (for `cl.exe`).
  - NVIDIA GPU with compute capability 6.1 or higher (e.g., GTX 1080, RTX 2080, RTX 3090).
- **Linux**:
  - NVIDIA CUDA Toolkit 11.8 (install via `apt` or NVIDIA’s `.run` installer).
  - `g++` (install via `sudo apt-get install g++`).
  - NVIDIA GPU with compute capability 6.1 or higher.

## Building the Programs

### Windows
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/mastermind.git
   cd mastermind
