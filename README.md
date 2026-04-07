# Zinc+

WIP implementation and playground on Nethermind's improvement to Zinc SNARK .

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

## License

Apache 2.0

## Would like to contribute?

see [Contributing](./CONTRIBUTING.md).
