# hipBLASLt Agent Guide

hipBLASLt is AMD's GEMM library for ML workloads, targeting the HIP/ROCm stack. It provides high-performance batched GEMM, grouped GEMM, and epilogue fusion for FP32/FP16/BF16/INT8 data types.

## Repository Layout

- `library/` — core library source, including kernel generation, solution logic, and the public API (`hipblaslt.h`).
- `clients/` — sample clients, benchmarks, and gtest-based tests.
- `cmake/` — CMake modules and dependency discovery.
- `deps/` — external dependency scripts.
- `docker/` — container build files.
- `docs/` — Sphinx/readthedocs documentation.
- `install.sh` — convenience install script.

## Build Commands

### Install script (recommended)

```bash
./install.sh -id
```

Flags:
- `-i` — install after build.
- `-d` — build dependencies.
- `-c` — clean build.
- `-h` — show all options.

### CMake build

```bash
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=/opt/rocm ..
make -j$(nproc)
make install
```

## Test Commands

```bash
# Run the gtest suite from the clients directory
./build/clients/staging/hipblaslt-test

# Benchmark a specific GEMM shape
./build/clients/staging/hipblaslt-bench --trans_a N --trans_b T -m 4096 -n 4096 -k 4096
```

## Lint / Code Style

- C/C++ sources use the `.clang-format` config; run `git clang-format` before committing.
- Python helper scripts use `.style.yapf` formatting.
- Keep kernel logic consistent with existing solution naming (`_*_gt_*`, `_*_pk_*`).

## Key Conventions

- Public API is C-compatible with a `hipblasLt` prefix.
- Tuning config files live under `library/src/amd_detail/rocblaslt/src/kernels/`.
- Grouped GEMM and epilogue fusion are the main differentiators from plain hipBLAS.

## Common Gotchas

- A full ROCm stack (hipcc, rocblas, etc.) is required to build.
- Some kernels are architecture-specific; unsupported GPUs may fall back to reference paths.
- `install.sh` downloads/builds dependencies on first run; subsequent runs are much faster.
