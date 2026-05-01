# Zinc+: SNARKs for Polynomial Rings

[![Docs](https://github.com/NethermindEth/zinc-plus/actions/workflows/docs.yml/badge.svg)](https://github.com/NethermindEth/zinc-plus/actions/workflows/docs.yml)
[![Lint](https://github.com/NethermindEth/zinc-plus/actions/workflows/linter.yml/badge.svg)](https://github.com/NethermindEth/zinc-plus/actions/workflows/linter.yml)
[![Build](https://github.com/NethermindEth/zinc-plus/actions/workflows/build-and-test.yml/badge.svg)](https://github.com/NethermindEth/zinc-plus/actions/workflows/build-and-test.yml)
[![Dependencies](https://github.com/NethermindEth/zinc-plus/actions/workflows/dependency-audit.yml/badge.svg)](https://github.com/NethermindEth/zinc-plus/actions/workflows/dependency-audit.yml)
[![UB](https://github.com/NethermindEth/zinc-plus/actions/workflows/ub-detection.yml/badge.svg)](https://github.com/NethermindEth/zinc-plus/actions/workflows/ub-detection.yml)
[![Coverage](https://github.com/NethermindEth/zinc-plus/actions/workflows/coverage.yml/badge.svg)](https://github.com/NethermindEth/zinc-plus/actions/workflows/coverage.yml)
![Rust](https://img.shields.io/badge/rust-stable-orange.svg)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

> [!WARNING]
> This project hasn't undergone an audit. It is intended for research and educational purposes only. The code is provided "AS IS" basis without any representation or warranty.

Zinc+ is a performance-oriented SNARK framework designed to handle computations across multiple mathematical domains without the "witness inflation" overhead typical of traditional arithmetization.  

Traditional SNARKs are restricted to a single finite field, forcing complex bitwise logic and non-native modular arithmetic to be emulated, which often bloats the witness by orders of magnitude. 
Zinc+ resolves this by allowing constraints to be expressed directly and simultaneously over rational ($\mathbb{Q}[X]$), integer ($\mathbb{Z}[X]$), and multiple finite field ($\mathbb{F}_{q_i}[X]$) polynomial rings.

> [!IMPORTANT]
> For benchmark results of Sha256 and ECDSA, see [README](https://github.com/NethermindEth/zinc-plus/blob/main-beta/README.md) in [main-beta](https://github.com/NethermindEth/zinc-plus/tree/main-beta) branch.

## Benchmarks

Available benchmarks:

| Benchmark          | What it measures                                                                                                                       |
|--------------------|----------------------------------------------------------------------------------------------------------------------------------------|
| `zip_benches`      | PCS-level operations (encode, Merkle tree, commit, prove, verify) using **scalar** evaluations with IPRS codes. Uses `i32` evaluations |
| `zip_plus_benches` | Same PCS-level operations using **polynomial** evaluations (degree 32 & 64) with both RAA and IPRS codes. Uses `{0,1}^D` evaluations.  |
| `e2e`              | Full Zinc+ SNARK prove & verify on several test AIRs (NoMult, BinaryDecomposition, BigLinear, BigLinearPI) at varying sizes.           |

To run benchmarks, use
```bash
RUSTFLAGS="-C target-cpu=native" cargo bench \
  --features "simd parallel unchecked" \
  --bench BENCH_NAME
```

### Flags & features

| Flag / Feature         | What it does                                                                                                                                                     |
|------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `-C target-cpu=native` | Lets the compiler emit platform-specific instructions (NEON, AVX-512, etc.). Required for `simd`.                                                                |
| `simd`                 | Bit-packs binary polynomials into `u64`s and uses hand-written NEON / AVX-512 intrinsics for key operations (widening, inner products).                          |
| `parallel`             | Enables [rayon](https://docs.rs/rayon)-based multi-threaded execution across the whole stack (sumcheck, encoding, commitment, etc.).                             |
| `unchecked`            | Replaces `checked_add` / `checked_mul` with plain arithmetic, removing overflow guards. Only affects integer-typed computations; field arithmetic is unaffected. |

## AI tools disclosure

The content published here may have been refined/augmented by the use of large language models (LLM), computer programs designed to comprehend and generate human language. However, any output refined/generated with the assistance of such programs has been reviewed, edited and revised by Nethermind.

## License

Apache 2.0

## Would like to contribute?

see [Contributing](./CONTRIBUTING.md).
