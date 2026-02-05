# SIMD Optimizations for IPRS Encoding

This document describes the SIMD optimizations implemented to accelerate the IPRS (Integer Pseudo Reed-Solomon) encoding in the Zip+ protocol.

## Overview

The IPRS encoder uses a Pseudo Number Theoretic Transform (PNTT) with radix-8 recursion. The most expensive operation is the **base layer multiplication**, where each input polynomial is multiplied by twiddle factors and accumulated into output polynomials.

Two key optimizations were implemented:
1. **SIMD-accelerated polynomial addition** - Vectorized `+=` for `DensePolynomial<i64, N>`
2. **Fused multiply-add** - Combines `acc += poly × twiddle` into a single operation, eliminating intermediate allocations

## Performance Results

Benchmarks on Apple M-series (aarch64 with NEON):

| Configuration | Regular | Fused | Speedup |
|--------------|---------|-------|---------|
| IPRS rate 1/2 (2^16 poly) | 21.2 ms | 12.1 ms | **~43%** |
| IPRS rate 1/4 (2^16 poly) | 45.5 ms | 24.1 ms | **~47%** |

## Architecture

### 1. SIMD Polynomial Addition

**File**: `poly/src/univariate/dense.rs`

Added `add_assign_simd<N>` function with platform-specific implementations:

```rust
// NEON (aarch64): Process 8 i64s per iteration using 4×int64x2_t
#[cfg(all(feature = "simd", target_arch = "aarch64"))]
pub unsafe fn add_assign_simd<const N: usize>(lhs: &mut [i64; N], rhs: &[i64; N])

// AVX-512 (x86_64): Process 8 i64s per iteration using __m512i
#[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"))]
pub unsafe fn add_assign_simd<const N: usize>(lhs: &mut [i64; N], rhs: &[i64; N])
```

### 2. Fused Multiply-Add Trait

**File**: `poly/src/univariate/binary_u64.rs`

Added `FusedMulAdd` trait that computes `self += poly × scalar` without intermediate allocation:

```rust
pub trait FusedMulAdd<Poly, Scalar> {
    fn fused_mul_add(&mut self, poly: &Poly, scalar: Scalar);
}

// Implementation for BinaryU64Poly<N> × i64 -> DensePolynomial<i64, N>
impl<const N: usize> FusedMulAdd<BinaryU64Poly<N>, i64> for DensePolynomial<i64, N>
```

**Key insight**: When multiplying a binary polynomial by a scalar, each output coefficient is either 0 or the scalar value. The fused operation:
1. Iterates through bit positions
2. For each set bit, adds the (possibly negated) scalar directly to the accumulator
3. Uses SIMD to accelerate the conditional additions

### 3. PNTT Integration

**File**: `zip-plus/src/code/iprs/pntt/radix8/mul_by_twiddle.rs`

Added `FusedMulAddByTwiddle` trait and `FusedWideningMulByTwiddle` struct:

```rust
pub trait FusedMulAddByTwiddle<Acc, Lhs, Twiddle>: Send + Sync {
    fn fused_mul_add(acc: &mut Acc, lhs: &Lhs, twiddle: Twiddle);
}
```

**File**: `zip-plus/src/code/iprs/pntt/radix8.rs`

Added `pntt_fused` and `base_multiply_into_output_fused` functions that use the fused operations in the base layer:

```rust
pub fn pntt_fused<In, Out, Twiddle, M, F, MCombine>(
    input: &[In],
    params: &Radix8PnttParams<C>,
) -> Vec<Out>
where
    F: FusedMulAddByTwiddle<Out, In, Twiddle>,
    // ... other bounds
```

### 4. IprsCode API

**File**: `zip-plus/src/code/iprs.rs`

Added `encode_fused` method on `IprsCode`:

```rust
#[cfg(feature = "simd")]
impl<Zt, Config, MT> IprsCode<Zt, Config, MT>
where
    // ... bounds including FusedMulAdd ...
{
    pub fn encode_fused(&self, row: &[Zt::Eval]) -> Vec<Zt::Cw>
}
```

## Files Changed

| File | Changes |
|------|---------|
| `poly/src/univariate/dense.rs` | Added `add_assign_simd` with NEON/AVX-512 |
| `poly/src/univariate/binary_u64.rs` | Added `FusedMulAdd` trait and implementations |
| `poly/src/univariate.rs` | Added exports for new functions/traits |
| `zip-plus/src/code/iprs/pntt/radix8/mul_by_twiddle.rs` | Added `FusedMulAddByTwiddle` trait |
| `zip-plus/src/code/iprs/pntt/radix8.rs` | Added `pntt_fused`, `base_multiply_into_output_fused` |
| `zip-plus/src/code/iprs.rs` | Added `encode_fused` method |
| `zip-plus/benches/zip_common.rs` | Added `encode_rows_fused` benchmark |
| `zip-plus/benches/zip_plus_benches.rs` | Added `IprsFusedEncode` implementations |

## Usage

### Enabling SIMD

The optimizations require the `simd` feature flag:

```bash
cargo build --features "simd parallel"
```

### Using Fused Encoding

```rust
use zip_plus::code::iprs::IprsCode;

// Create IPRS code
let iprs_code: IprsCode<_, _, _> = LinearCode::new(poly_size);

// Regular encoding
let codeword = iprs_code.encode(&row);

// Fused encoding (faster, requires "simd" feature)
#[cfg(feature = "simd")]
let codeword = iprs_code.encode_fused(&row);
```

### Running Benchmarks

```bash
# Run both regular and fused benchmarks
cargo bench --bench zip_plus_benches --features "simd parallel" -- "EncodeRows"

# Quick benchmark
cargo bench --bench zip_plus_benches --features "simd parallel" -- "EncodeRows" --quick
```

## Implementation Details

### Why Fused Operations Help

The original base layer loop:
```rust
for col in 0..base_dim {
    let term = M::mul_by_twiddle(&input[idx], bm_row_col);  // Allocates DensePolynomial
    acc += &term;  // Another operation
}
```

The fused version:
```rust
for col in 0..base_dim {
    F::fused_mul_add(&mut acc, &input[idx], bm_row_col);  // No allocation, direct accumulation
}
```

Benefits:
1. **No intermediate allocation** - Eliminates temporary `DensePolynomial` per multiplication
2. **Better cache utilization** - Accumulator stays in registers/L1 cache
3. **SIMD vectorization** - The addition loop is vectorized with NEON/AVX-512

### Platform Support

| Platform | SIMD Instructions | Status |
|----------|------------------|--------|
| aarch64 (Apple M1/M2/M3) | NEON | ✅ Implemented |
| x86_64 with AVX-512 | AVX-512F | ✅ Implemented |
| x86_64 without AVX-512 | Fallback | ✅ Scalar fallback |

## Testing

The correctness of fused operations is verified by a test that compares results:

```bash
cargo test -p zip-plus --lib --features "simd parallel" -- pntt_fused_matches_pntt
```

This test generates random inputs and verifies that `pntt_fused` produces identical results to the regular `pntt` function.

## Future Improvements

1. **AVX2 fallback** - Add AVX2 implementation for x86_64 systems without AVX-512
2. **Butterfly stage fusion** - Apply similar fusion to the combine/butterfly stages
3. **Prefetching** - Add explicit prefetch hints for the twiddle factor arrays
4. **Memory layout** - Consider AoS vs SoA layout for better SIMD utilization
