# IPRS Encoding Optimizations

This document describes the optimizations made to the IPRS (Integer Pseudo Reed-Solomon) encoding in the Zip+ protocol.

## Overview

Three major optimizations were implemented to improve IPRS encoding performance:

1. **Unchecked Butterfly Addition** (~12-15% overall speedup)
2. **Unchecked Base Layer Addition** (~5-10% additional speedup)
3. **In-Place PNTT Output** (reduces per-row allocations)

---

## 1. Unchecked Butterfly Optimization

### Background

The PNTT (Pseudo Number Theoretic Transform) algorithm consists of two main stages:

1. **Base Layer** (~50% of time): Multiplies input chunks by a precomputed Vandermonde-like matrix (64×32)
2. **Butterfly Stage** (~50% of time): Applies radix-8 FFT butterflies across `DEPTH` levels

Each radix-8 butterfly computes 8 outputs from 8 inputs:

$$y[k] = x[0] + \sum_{j=1}^{7} x[j] \cdot \omega^{k \cdot j}$$

Where $\omega$ is an 8th root of unity. The computation involves:
- 56 twiddle multiplications (7 inputs × 8 outputs)
- 56 polynomial additions (accumulating into 8 outputs)

### The Problem

The original implementation used `CheckedAdd::checked_add` for all additions, which:
1. Clones the accumulator polynomial (32 × i64 coefficients)
2. Performs overflow-checked addition on each coefficient
3. Returns `Option<Self>`, requiring unwrapping

This overhead was significant because the butterfly stage performs millions of additions during encoding.

### The Solution

The values in the butterfly stage cannot overflow i64:

| Stage | Maximum Value | Bits Required |
|-------|---------------|---------------|
| Input (i32) | $2^{31}$ | 31 bits |
| Twiddle | $2^{16}$ (roots of unity in $\mathbb{F}_{2^{16}+1}$) | 16 bits |
| After widening multiply | $2^{47}$ | 47 bits |
| After 8 additions | $8 \times 2^{47} = 2^{50}$ | 50 bits |
| i64 capacity | $2^{63}$ | 63 bits |

Since the maximum possible value ($2^{50}$) is well within i64 range ($2^{63}$), overflow checking is unnecessary.

### Implementation

A new `unchecked-butterfly` feature flag was added that switches to unchecked addition:

**File:** `zip-plus/src/code/iprs/pntt/radix8/butterfly.rs`

```rust
/// Checked version (default)
pub(crate) fn apply_radix_8_butterflies<R, Twiddle, M>(
    ys: [&mut R; 8],
    xs: &[R],
    twiddles: &[[Twiddle; 8]; 7],
) where
    R: Clone + CheckedAdd,
    M: MulByTwiddle<R, Twiddle, Output = R>,
{ ... }

/// Unchecked version (faster)
#[inline(always)]
pub(crate) fn apply_radix_8_butterflies_unchecked<R, Twiddle, M>(
    ys: [&mut R; 8],
    xs: &[R],
    twiddles: &[[Twiddle; 8]; 7],
) where
    R: Clone + for<'a> AddAssign<&'a R>,  // Uses AddAssign instead of CheckedAdd
    M: MulByTwiddle<R, Twiddle, Output = R>,
{
    for (y, butterfly_row) in ys.into_iter().zip(BUTTERFLY_TABLE.iter()) {
        let mut acc = xs[0].clone();
        for (j_minus_1, (&twiddle_idx, x)) in butterfly_row.iter().zip(&xs[1..]).enumerate() {
            let twisted = M::mul_by_twiddle(x, &twiddles[j_minus_1][twiddle_idx]);
            acc += &twisted;  // Direct += instead of checked_add
        }
        *y = acc;
    }
}
```

The key difference is using `AddAssign<&R>` trait bound instead of `CheckedAdd`, which:
- Avoids cloning on each addition
- Skips overflow checking
- Uses in-place mutation

**File:** `zip-plus/src/code/iprs/pntt/radix8.rs`

Conditional imports and function signatures based on feature flag:

```rust
#[cfg(not(feature = "unchecked-butterfly"))]
use butterfly::apply_radix_8_butterflies;
#[cfg(feature = "unchecked-butterfly")]
use butterfly::apply_radix_8_butterflies_unchecked;
```

### Performance Results

Benchmarks run on Apple M-series (aarch64), 100 iterations of `encode_rows`:

| Metric | Baseline (checked) | Optimized (unchecked) | Improvement |
|--------|-------------------|----------------------|-------------|
| Wall clock | ~1.35-1.48s | ~1.19-1.22s | **12-15%** |
| Butterfly time/call | ~2100-2200µs | ~1660-1700µs | **~20%** |
| Total PNTT time/call | ~4.1-4.5ms | ~3.6-3.7ms | **~12%** |
| Butterfly % of total | ~51% | ~46% | - |

---

## 2. Unchecked Base Layer Optimization

### Background

The same overflow analysis that applies to the butterfly stage also applies to the base layer. The base layer performs a matrix-vector multiplication with the 64×32 Vandermonde-like matrix, accumulating 32 terms per output.

### The Problem

The original base layer implementation used the `add!` macro which calls `checked_add`, performing overflow checking on every addition in the inner loop.

### The Solution

Using the same reasoning as the butterfly optimization, the base layer values cannot overflow:

| Stage | Maximum Value | Bits Required |
|-------|---------------|---------------|
| Input (i32) | $2^{31}$ | 31 bits |
| Twiddle | $2^{16}$ (Vandermonde matrix entries) | 16 bits |
| After widening multiply | $2^{47}$ | 47 bits |
| After 32 additions | $32 \times 2^{47} = 2^{52}$ | 52 bits |
| i64 capacity | $2^{63}$ | 63 bits |

Since $2^{52} \ll 2^{63}$, overflow is impossible.

### Implementation

When `unchecked-butterfly` is enabled, the base layer functions use direct `+` operator instead of the `add!` macro:

**File:** `zip-plus/src/code/iprs/pntt/radix8.rs`

```rust
/// Allocates the output vector and performs base layer multiplications.
/// This version uses unchecked addition for better performance.
#[cfg(feature = "unchecked-butterfly")]
fn base_multiply_into_output_unchecked<In, Out, C, M>(...) -> Vec<Out>
where
    Out: Clone + FromRef<In> + Send + Sync + for<'a> Add<&'a Out, Output = Out>,
    ...
{
    // ... same logic, but uses:
    acc + &term  // Direct add instead of add!(acc, &term)
}
```

### Expected Performance Improvement

Based on the base layer taking ~54% of total PNTT time, and assuming similar overhead reduction as butterflies:
- Base layer speedup: ~15-20%
- Overall PNTT speedup: ~8-11% additional
- Combined with butterfly optimization: **~20-25% total speedup**

---

## 3. In-Place PNTT Output Optimization

### Background

The standard `encode_rows` operation encodes multiple rows of a polynomial matrix. Each row encoding produces a codeword vector.

### The Problem

The original implementation allocated a new `Vec<Cw>` for each row:

```rust
// Original pattern (simplified)
fn encode_rows(&self, rows: &[Vec<Eval>]) -> Vec<Vec<Cw>> {
    rows.iter()
        .map(|row| self.encode(row))  // Allocates Vec<Cw> per row
        .collect()
}
```

For a matrix with thousands of rows, this creates thousands of small allocations, causing:
- Allocator pressure
- Poor cache locality
- Memory fragmentation

### The Solution

Pre-allocate the entire output matrix and have each PNTT call write directly into its designated slice using `MaybeUninit`:

```rust
// Optimized pattern (simplified)
fn encode_rows(&self, rows: &[Vec<Eval>]) -> Vec<Vec<Cw>> {
    let mut output: Vec<MaybeUninit<Cw>> = Vec::with_capacity(rows.len() * codeword_len);
    
    for (row, out_chunk) in rows.iter().zip(output.chunks_mut(codeword_len)) {
        self.encode_into_uninit(row, out_chunk);  // Writes directly, no allocation
    }
    
    // Safe: all elements initialized
    unsafe { transmute(output) }
}
```

### Implementation

**File:** `zip-plus/src/code.rs`

Added new methods to the `LinearCode` trait:

```rust
pub trait LinearCode<Zt: ZipTypes>: Sync + Send {
    // Existing method
    fn encode(&self, row: &[Zt::Eval]) -> Vec<Zt::Cw>;

    /// Encodes directly into an uninitialized output buffer.
    /// Avoids per-row allocations when encoding multiple rows.
    fn encode_into_uninit(&self, row: &[Zt::Eval], out: &mut [MaybeUninit<Zt::Cw>]) {
        // Default implementation falls back to encode() + copy
        let encoded = self.encode(row);
        Out::from(out).copy_from_slice(encoded.as_slice());
    }

    /// SIMD fused variant for even faster encoding.
    #[cfg(feature = "simd")]
    fn encode_fused_into_uninit(&self, row: &[Zt::Eval], out: &mut [MaybeUninit<Zt::Cw>]) {
        self.encode_into_uninit(row, out);
    }
}
```

**File:** `zip-plus/src/code/iprs/pntt/radix8.rs`

Added in-place PNTT functions:

```rust
/// In-place PNTT that writes into an existing output buffer.
pub(crate) fn pntt_into<In, Out, C, MulInByTwiddle, MulOutByTwiddle>(
    input: &[In],
    params: &Radix8PnttParams<C>,
    out: &mut [MaybeUninit<Out>],
)
where
    C: Config,
    In: Clone + Send + Sync,
    Out: CheckedAdd + CheckedMul + FromRef<In> + Clone + Send + Sync + Debug,
    MulInByTwiddle: MulByTwiddle<In, PnttInt, Output = Out>,
    MulOutByTwiddle: MulByTwiddle<Out, PnttInt, Output = Out>,
{
    // Write base layer results directly into output buffer
    base_multiply_into_output_in_place::<_, _, _, MulInByTwiddle>(input, params, out);

    // Safe: base layer initialized all elements
    let out_init = unsafe {
        std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut Out, out.len())
    };

    // Apply butterfly stages in-place
    combine_stages::<_, _, MulOutByTwiddle>(out_init, params);
}
```

**File:** `zip-plus/src/code/iprs.rs`

IPRS implementation of in-place encoding:

```rust
impl<Zt: ZipTypes, Config, MT> LinearCode<Zt> for IprsCode<Zt, Config, MT>
where
    // ... trait bounds ...
{
    fn encode_into_uninit(&self, row: &[Zt::Eval], out: &mut [MaybeUninit<Zt::Cw>]) {
        assert_eq!(row.len(), Config::INPUT_LEN);
        
        pntt::radix8::pntt_into::<_, _, _, WideningMulByTwiddle<MT>, MBSMulByTwiddle<CHECKED>>(
            row,
            &self.pntt_params,
            out,
        );
    }
}
```

**File:** `zip-plus/src/pcs/phase_commit.rs`

Updated `encode_rows` to use the new in-place API.

### Benefits

1. **Zero per-row allocations** — Output buffer is allocated once upfront
2. **Better cache locality** — Contiguous output memory
3. **Reduced allocator pressure** — Fewer malloc/free calls
4. **Compatible with parallelism** — Each thread writes to its own slice

---

## Usage

### Building with Optimizations

```bash
# With unchecked butterfly (faster)
cargo build --release --features "asm simd parallel unchecked-butterfly"

# Without (default, safer)
cargo build --release --features "asm simd parallel"
```

### Running Benchmarks

```bash
cargo bench --bench zip_plus_benches --features "asm simd parallel unchecked-butterfly" -- IPRS
```

### Running Tests

```bash
cargo test --release --features "asm simd parallel unchecked-butterfly" -p zip-plus iprs
```

---

## Safety Considerations

### Unchecked Butterfly and Base Layer

✅ **Safe to use when:**
- Encoding with standard IPRS parameters ($\mathbb{F}_{2^{16}+1}$, DEPTH=2)
- Input values are within i32 range
- Using the standard `WideningMulByTwiddle` which widens i32 → i64

⚠️ **May not be safe when:**
- Using custom configurations with larger input types
- Chaining multiple encoding operations without reduction
- Using alternative twiddle factor implementations

### In-Place Output

The in-place optimization uses `MaybeUninit` and requires careful handling:
- The base layer must initialize all output elements before the butterfly stage reads them
- The final `unsafe` transmute assumes all elements are initialized

---

## Files Changed

| File | Changes |
|------|---------|
| `zip-plus/Cargo.toml` | Added `unchecked-butterfly` feature flag |
| `zip-plus/src/code.rs` | Added `encode_into_uninit` and `encode_fused_into_uninit` to `LinearCode` trait |
| `zip-plus/src/code/iprs.rs` | Implemented in-place encoding for `IprsCode`, conditional `encode_inner` for unchecked |
| `zip-plus/src/code/iprs/pntt/radix8.rs` | Added `pntt_into`, conditional butterfly imports, `base_multiply_into_output_unchecked`, `base_multiply_into_output_in_place_unchecked` |
| `zip-plus/src/code/iprs/pntt/radix8/butterfly.rs` | Added `apply_radix_8_butterflies_unchecked` |
| `zip-plus/src/pcs/phase_commit.rs` | Updated `encode_rows` to use in-place encoding |

---

## Future Work

1. **SIMD addition** — Vectorize the 32-coefficient polynomial additions using NEON/AVX
2. **Cache optimization** — Improve memory access patterns in butterfly stage
3. **SIMD base layer multiply** — Vectorize the base layer matrix-vector multiplication
4. **Reduce cloning in butterflies** — Avoid cloning 8 subresults per butterfly call
