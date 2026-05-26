//! `GF(2^128)`: prototype "Binius64-style" 128-bit binary extension field.
//!
//! Built as the direct quotient
//!
//! ```text
//! F_2[X] / <X^128 + X^7 + X^2 + X + 1>
//! ```
//!
//! (the GHASH / AES-GCM irreducible pentanomial), **not** the recursive
//! Binius tower `τ_7 = τ_6[α_7]/(α_7² + α_7·α_6 + 1)`. Picking the GHASH
//! polynomial gives us:
//!
//! - 16-byte storage (`[u64; 2]`) — 1.5× smaller than [`super::binary_gf192::BinaryFieldGF192`],
//! - direct hardware multiplication via PMULL on aarch64 (`vmull_p64`) and
//!   PCLMUL on x86_64 (`_mm_clmulepi64_si128`),
//! - a well-known, easily verifiable reduction routine.
//!
//! Trade-off vs a strict Binius tower: this representation has no clean
//! sub-field projection / Frobenius structure exposed at the surface. If
//! the protocol ever depends on those (tower-aware folding, sub-field
//! commitments), the right move is to swap the *internal* mul for a
//! Karatsuba-over-`GF(2^64)` recursion while keeping the public API the
//! same — flagged inline at `mul`.
//!
//! Current scope: **benchmark-only**. The intent is to measure the
//! mul / square / inverse throughput of a 128-bit Binius-flavoured field
//! against the existing GF(2^192) baseline so we can quantify the savings
//! from a future protocol-wide migration. Pre-migration we still owe the
//! soundness write-up — see the TODO immediately below.
//!
//! # TODO — soundness / degree-budget analysis (deferred work item 1)
//!
//! Before any code path inside the protocol switches to `BinaryFieldGF128`,
//! pin down the following and write it up alongside this module:
//!
//! 1. **Per-challenge soundness.** ε ≈ `d / 2^128` per Schwartz–Zippel
//!    application, where `d` is the relevant polynomial degree. Sum the
//!    error across all F_2-open Fiat-Shamir draws (γ batching, coeffs
//!    proximity, opened-column sampling, α projection); confirm the total
//!    clears the target security bound (≥ 100 bits with repetition, or
//!    use a soundness-amplifying repetition factor).
//! 2. **Lift-discharge degree budget.** The current open carries
//!    `BinaryF2Poly<W>` widths up to W=10 (≈ 640-bit accumulators) over
//!    GF(2^192). At 128-bit projection the same accumulator widths would
//!    overflow the embedding — either raise the γ count (split the
//!    column batch into ⌈W·128/192⌉ slices) or recut the proof shape.
//!    Either way the open code needs new widths once we migrate.
//! 3. **Proximity gap.** Re-validate the encoding-consistency check's
//!    proximity parameter for the rate-1/4 RAA code in the smaller field.
//! 4. **Comparison with the integer protocol's lifted-eval recipe.** That
//!    path projects to a prime field; the F_2 protocol is the side
//!    actually constrained by the binary-extension size, so the migration
//!    only affects the F_2-native call sites.
//!
//! Until the analysis above lands, this module is exposed for benches /
//! experiments only; production code should keep using GF(2^192).

use core::{
    fmt::{Debug, Display, Formatter, Result as FmtResult},
    ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign},
};
use num_traits::{One, Zero};

use crate::univariate::binary::BinaryPoly;

/// Low bits of the reduction polynomial — `g(X) = X^7 + X^2 + X + 1`,
/// stored as `0x87` (bits 0, 1, 2, 7 set). The `X^128` term is implicit
/// in the reduction routine: `X^128 ≡ g(X) mod f`.
pub const REDUCTION_LOW_GF128: u64 = 0x87;

/// An element of `GF(2^128) = F_2[X] / <X^128 + X^7 + X^2 + X + 1>`.
///
/// Stored as `[u64; 2]` with little-endian word ordering: bit `64·w + b`
/// of `words[w]` carries the coefficient of `X^{64·w + b}`. All bits at
/// or above 128 are zero in any reduced element.
#[derive(Clone, Copy, Debug, Default, Hash, PartialEq, Eq)]
#[repr(transparent)]
pub struct BinaryFieldGF128 {
    words: [u64; 2],
}

impl BinaryFieldGF128 {
    #[inline]
    pub const fn zero() -> Self {
        Self { words: [0, 0] }
    }

    #[inline]
    pub const fn one() -> Self {
        Self { words: [1, 0] }
    }

    /// Construct from raw bit-packed words; caller guarantees reduced form
    /// (the natural invariant since `[u64; 2]` only carries 128 bits).
    #[inline]
    pub const fn from_words(words: [u64; 2]) -> Self {
        Self { words }
    }

    #[inline]
    pub const fn words(&self) -> &[u64; 2] {
        &self.words
    }

    #[inline]
    pub fn is_zero(&self) -> bool {
        self.words[0] == 0 && self.words[1] == 0
    }

    /// `a²` via carryless square + reduction.
    ///
    /// Squaring in characteristic 2 is the bit-expansion permutation
    /// (`bit i` → `bit 2i`). We could specialise this to a fixed
    /// PSHUFB/TBL lookup, but for benchmark purposes the generic
    /// `clmul_128x128(self, self)` path is good enough.
    #[inline]
    pub fn square(&self) -> Self {
        let prod = clmul_128x128(&self.words, &self.words);
        Self::from_words(reduce_256_to_128(prod))
    }

    /// `a^{-1}` via Fermat: `a^{-1} = a^{2^128 - 2}` for `a ≠ 0`.
    ///
    /// Same naive chain as [`super::binary_gf192::BinaryFieldGF192::inverse`]
    /// — `127 squarings + 126 mults`. Fine for benchmark coverage; a
    /// production-grade replacement is described in the TODO below.
    ///
    /// # TODO — Itoh–Tsujii addition chain (deferred optimisation)
    ///
    /// For `GF(2^n)`, Itoh–Tsujii computes `a^{-1} = (a^{2^n - 2}) = (a^r)^{2}`
    /// where `r = (2^{n-1} - 1)`, building `a^r` via the recurrence
    /// `c_{k+m} = (c_k)^{2^m} · c_m`. The total cost is
    /// `O(log n) mults + (n - 1) squarings` — for `GF(2^128)` that's
    /// ~7 multiplications + 127 squarings, versus the naive 126 mults
    /// + 127 squarings (an ~18× drop in multiplications). Same shape
    /// applies to [`super::binary_gf192::BinaryFieldGF192::inverse`]
    /// (~8 mults + 191 squarings via Itoh–Tsujii instead of 190 + 191).
    ///
    /// Worth doing once the PIOP starts hitting inverses on a hot path
    /// (currently the F_2 prove path doesn't; it's mostly mul + add).
    ///
    /// Panics if `self.is_zero()`.
    pub fn inverse(&self) -> Self {
        assert!(!self.is_zero(), "GF(2^128): zero has no inverse");
        let mut c = *self; // c = a^{2^1 - 1} = a
        for _ in 1..127 {
            c = c.square();
            c *= &*self;
        }
        // c = a^{2^127 - 1}. One more squaring → a^{2 · (2^127 - 1)} = a^{2^128 - 2}.
        c.square()
    }
}

impl Zero for BinaryFieldGF128 {
    fn zero() -> Self {
        Self::zero()
    }
    fn is_zero(&self) -> bool {
        Self::is_zero(self)
    }
}

impl One for BinaryFieldGF128 {
    fn one() -> Self {
        Self::one()
    }
    fn is_one(&self) -> bool {
        self.words[0] == 1 && self.words[1] == 0
    }
}

impl Display for BinaryFieldGF128 {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "GF128[{:016x}_{:016x}]", self.words[1], self.words[0])
    }
}

// -- additive group (XOR) --------------------------------------------

impl AddAssign<&Self> for BinaryFieldGF128 {
    #[inline]
    fn add_assign(&mut self, rhs: &Self) {
        self.words[0] ^= rhs.words[0];
        self.words[1] ^= rhs.words[1];
    }
}

impl AddAssign for BinaryFieldGF128 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self += &rhs;
    }
}

impl Add for BinaryFieldGF128 {
    type Output = Self;
    #[inline]
    fn add(mut self, rhs: Self) -> Self {
        self += &rhs;
        self
    }
}

impl Add<&Self> for BinaryFieldGF128 {
    type Output = Self;
    #[inline]
    fn add(mut self, rhs: &Self) -> Self {
        self += rhs;
        self
    }
}

impl SubAssign<&Self> for BinaryFieldGF128 {
    #[inline]
    fn sub_assign(&mut self, rhs: &Self) {
        *self += rhs; // char 2: sub = add
    }
}

impl SubAssign for BinaryFieldGF128 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self += &rhs;
    }
}

impl Sub for BinaryFieldGF128 {
    type Output = Self;
    #[inline]
    fn sub(mut self, rhs: Self) -> Self {
        self += &rhs;
        self
    }
}

// -- multiplicative group --------------------------------------------

impl MulAssign<&Self> for BinaryFieldGF128 {
    #[inline]
    fn mul_assign(&mut self, rhs: &Self) {
        let prod = clmul_128x128(&self.words, &rhs.words);
        self.words = reduce_256_to_128(prod);
    }
}

impl MulAssign for BinaryFieldGF128 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self *= &rhs;
    }
}

impl Mul for BinaryFieldGF128 {
    type Output = Self;
    #[inline]
    fn mul(mut self, rhs: Self) -> Self {
        self *= &rhs;
        self
    }
}

impl Mul<&Self> for BinaryFieldGF128 {
    type Output = Self;
    #[inline]
    fn mul(mut self, rhs: &Self) -> Self {
        self *= rhs;
        self
    }
}

// -- carryless multiplication and reduction --------------------------
//
// 128×128 → 256-bit carryless product, computed as four 64×64 clmuls
// composed via Karatsuba (3 inner muls instead of 4). The 64×64 base is
// hardware-accelerated where available:
//   - aarch64: `vmull_p64` (PMULL/PMULL2)
//   - x86_64 + pclmulqdq: `_mm_clmulepi64_si128`
//   - else: a scalar fallback (bit-by-bit shift+XOR).

/// 128×128 → 256-bit carryless product.
///
/// Karatsuba layout (let `a = (a0, a1)`, `b = (b0, b1)`):
///   `lo`  = a0 · b0
///   `hi`  = a1 · b1
///   `mid` = (a0 ^ a1) · (b0 ^ b1) ^ lo ^ hi   = a0·b1 ^ a1·b0
/// Then the 256-bit product is `lo | (mid << 64) | (hi << 128)`.
#[inline]
fn clmul_128x128(a: &[u64; 2], b: &[u64; 2]) -> [u64; 4] {
    let lo = clmul_64x64(a[0], b[0]);
    let hi = clmul_64x64(a[1], b[1]);
    let mid_raw = clmul_64x64(a[0] ^ a[1], b[0] ^ b[1]);
    let mid = [mid_raw[0] ^ lo[0] ^ hi[0], mid_raw[1] ^ lo[1] ^ hi[1]];

    // Compose: word 0 = lo[0], word 1 = lo[1] ^ mid[0], word 2 = hi[0] ^ mid[1], word 3 = hi[1].
    [lo[0], lo[1] ^ mid[0], hi[0] ^ mid[1], hi[1]]
}

/// 64×64 → 128-bit carryless multiplication.
///
/// Dispatches to the hardware primitive when available, falls back to a
/// scalar bit-by-bit loop otherwise (used on platforms without PMULL /
/// PCLMUL; correctness mirror of the hardware path so tests stay
/// deterministic).
#[inline]
pub(crate) fn clmul_64x64(a: u64, b: u64) -> [u64; 2] {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        // SAFETY: NEON+CRYPTO are enabled at compile time. `vmull_p64`
        // returns a 128-bit polynomial product as a `poly128_t`, which
        // we transmute to `[u64; 2]` (little-endian word order).
        unsafe {
            use core::arch::aarch64::vmull_p64;
            // `vmull_p64` takes `poly64_t` operands — alias for `u64`
            // for ABI purposes.
            let prod: u128 = core::mem::transmute(vmull_p64(a, b));
            [prod as u64, (prod >> 64) as u64]
        }
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
    {
        // SAFETY: pclmulqdq enabled at compile time.
        unsafe {
            use core::arch::x86_64::{__m128i, _mm_clmulepi64_si128, _mm_set_epi64x};
            let av = _mm_set_epi64x(0, a as i64);
            let bv = _mm_set_epi64x(0, b as i64);
            let prod: __m128i = _mm_clmulepi64_si128(av, bv, 0x00);
            let bytes: [u64; 2] = core::mem::transmute(prod);
            bytes
        }
    }
    #[cfg(not(any(
        all(target_arch = "aarch64", target_feature = "neon"),
        all(target_arch = "x86_64", target_feature = "pclmulqdq"),
    )))]
    {
        clmul_64x64_scalar(a, b)
    }
}

/// Portable scalar fallback for 64×64 carryless mul.
///
/// Iterates set bits of `a`, XORing `b << shift` into the 128-bit
/// accumulator. Quadratic in the bit count of `a`, but correctness-only
/// — never invoked on the targets we benchmark on.
#[inline]
#[allow(dead_code, clippy::arithmetic_side_effects)]
fn clmul_64x64_scalar(a: u64, b: u64) -> [u64; 2] {
    let mut a_word = a;
    let mut acc_lo: u64 = 0;
    let mut acc_hi: u64 = 0;
    while a_word != 0 {
        let lo = a_word.trailing_zeros();
        if lo == 0 {
            acc_lo ^= b;
        } else if lo < 64 {
            acc_lo ^= b << lo;
            acc_hi ^= b >> (64 - lo);
        }
        a_word &= a_word - 1;
    }
    [acc_lo, acc_hi]
}

/// Reduce a 256-bit `F_2[X]` polynomial modulo
/// `f(X) = X^128 + X^7 + X^2 + X + 1`.
///
/// For each bit `i ≥ 128` of the input, `X^i ≡ X^{i - 128} · g(X)`
/// (mod `f`), with `g(X) = X^7 + X^2 + X + 1`. Word-at-a-time:
///   - word 3 (bits 192..255) folds into `r[1]` (high half) and into
///     `hi2` for re-reduction;
///   - word 2 (bits 128..191) folds into `r[0]` and `r[1]`;
///   - `hi2`'s overflow (≤ 7 high bits → bits 128..134 globally)
///     re-folds once more into `r[0]`.
#[inline]
fn reduce_256_to_128(prod: [u64; 4]) -> [u64; 2] {
    let mut r = [prod[0], prod[1]];
    let mut hi2 = prod[2];

    // Step A: word 3 (bits 192..255). X^{192+k} ≡ X^{64+k} · g.
    let w = prod[3];
    let (lo, hi) = word_times_g_gf128(w);
    r[1] ^= lo;
    hi2 ^= hi;

    // Step B: word 2 (bits 128..191) — updated by step A's `hi`.
    // X^{128+k} ≡ X^k · g, so this lands in `r[0]` (bits 0..63) and
    // `r[1]` (bits 64..70 from the `<< 7` overflow).
    let w = hi2;
    let (lo, hi) = word_times_g_gf128(w);
    r[0] ^= lo;
    r[1] ^= hi;

    r
}

/// `w · g(X)` where `g(X) = X^7 + X^2 + X + 1`. Mirrors the helper in
/// `binary_gf192`, kept local so the GF128 module compiles in isolation.
#[inline]
#[allow(clippy::arithmetic_side_effects)]
fn word_times_g_gf128(w: u64) -> (u64, u64) {
    let lo = w ^ (w << 1) ^ (w << 2) ^ (w << 7);
    let hi = (w >> 63) ^ (w >> 62) ^ (w >> 57);
    (lo, hi)
}

// -- α-projection helpers --------------------------------------------
//
// Mirror of [`super::binary_gf192::alpha_powers`] /
// `eval_f2_poly_d_at_with_powers`, but targeting `BinaryFieldGF128`.
// The α-projection kernel is the prove-side hot loop in the F_2 PIOP
// (~`num_cols × 2^num_vars` per-cell evaluations); porting it lets us
// measure the end-to-end win from a smaller embedding field independent
// of the rest of the prover.

/// Precompute the powers `α^0, α^1, ..., α^{d-1}` in `GF(2^128)`.
///
/// `d − 1` field multiplications. Each subsequent per-cell evaluation
/// becomes XOR-only (no inner-loop multiplications).
pub fn alpha_powers(alpha: &BinaryFieldGF128, d: usize) -> Vec<BinaryFieldGF128> {
    let mut powers = Vec::with_capacity(d);
    if d == 0 {
        return powers;
    }
    powers.push(BinaryFieldGF128::one());
    for _ in 1..d {
        let mut next = powers[powers.len() - 1];
        next *= &*alpha;
        powers.push(next);
    }
    powers
}

/// Evaluate an `F_2[X]<D>` cell at α using precomputed powers
/// (`alpha_powers[i] = α^i`). Inner loop is XOR-only: for each set bit
/// `i` of the cell, the accumulator XORs in `alpha_powers[i]`.
///
/// Caller must ensure `alpha_powers.len() >= D`.
#[inline]
pub fn eval_f2_poly_d_at_with_powers<const D: usize>(
    p: &BinaryPoly<D>,
    alpha_powers: &[BinaryFieldGF128],
) -> BinaryFieldGF128 {
    debug_assert!(
        alpha_powers.len() >= D,
        "eval_f2_poly_d_at_with_powers: powers slice ({}) shorter than D ({D})",
        alpha_powers.len(),
    );
    let mut acc = BinaryFieldGF128::zero();
    for (i, c) in p.iter().enumerate() {
        if c.inner() {
            acc += &alpha_powers[i];
        }
    }
    acc
}

/// Branchless variant: for each bit `i`, mask `alpha_powers[i]` by
/// `-(bit) as u64` and unconditionally XOR into the accumulator. Avoids
/// the ~50% branch-misprediction tax on the bit-set test for random
/// cells. Same `O(D)` work as the branchy version but consistent
/// pipelining and no misprediction stalls.
///
/// Caller must ensure `alpha_powers.len() >= D`.
#[inline]
pub fn eval_f2_poly_d_at_with_powers_branchless<const D: usize>(
    p: &BinaryPoly<D>,
    alpha_powers: &[BinaryFieldGF128],
) -> BinaryFieldGF128 {
    debug_assert!(
        alpha_powers.len() >= D,
        "eval_f2_poly_d_at_with_powers_branchless: powers slice ({}) shorter than D ({D})",
        alpha_powers.len(),
    );
    let mut acc_lo: u64 = 0;
    let mut acc_hi: u64 = 0;
    for (i, c) in p.iter().enumerate() {
        let mask: u64 = 0u64.wrapping_sub(c.inner() as u64);
        let pw = alpha_powers[i].words();
        acc_lo ^= pw[0] & mask;
        acc_hi ^= pw[1] & mask;
    }
    BinaryFieldGF128::from_words([acc_lo, acc_hi])
}

// -- tests -----------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng, rngs::StdRng};

    fn rand_elt(rng: &mut StdRng) -> BinaryFieldGF128 {
        BinaryFieldGF128::from_words([rng.random(), rng.random()])
    }

    #[test]
    fn add_is_xor_and_self_inverse() {
        let mut rng = StdRng::seed_from_u64(0xC0FFEE_128);
        for _ in 0..256 {
            let a = rand_elt(&mut rng);
            let b = rand_elt(&mut rng);
            assert_eq!((a + b) + b, a); // char 2: x + x = 0
            assert_eq!(a + BinaryFieldGF128::zero(), a);
        }
    }

    #[test]
    fn one_is_mul_identity_and_zero_annihilates() {
        let mut rng = StdRng::seed_from_u64(0xDEAD_128);
        let one = BinaryFieldGF128::one();
        let zero = BinaryFieldGF128::zero();
        for _ in 0..256 {
            let a = rand_elt(&mut rng);
            assert_eq!(a * one, a);
            assert_eq!(a * zero, zero);
        }
    }

    #[test]
    fn square_matches_self_multiply() {
        let mut rng = StdRng::seed_from_u64(0xBEEF_128);
        for _ in 0..256 {
            let a = rand_elt(&mut rng);
            assert_eq!(a.square(), a * a);
        }
    }

    #[test]
    fn inverse_satisfies_a_times_inv_equals_one() {
        let mut rng = StdRng::seed_from_u64(0xF00D_128);
        let one = BinaryFieldGF128::one();
        for _ in 0..64 {
            let mut a = rand_elt(&mut rng);
            if a.is_zero() {
                a = BinaryFieldGF128::one();
            }
            let inv = a.inverse();
            assert_eq!(a * inv, one);
        }
    }

    #[test]
    fn frobenius_squaring_is_linear() {
        // In characteristic 2 the Frobenius `x → x²` is additive:
        // `(x + y)² = x² + y²`. Sanity that the clmul-based square
        // respects this.
        let mut rng = StdRng::seed_from_u64(0xBAAD_128);
        for _ in 0..256 {
            let a = rand_elt(&mut rng);
            let b = rand_elt(&mut rng);
            assert_eq!((a + b).square(), a.square() + b.square());
        }
    }

    #[test]
    fn karatsuba_clmul_matches_scalar_clmul() {
        // The Karatsuba `clmul_128x128` composes three 64×64 clmuls;
        // cross-check against a naive 4-mul layout via the scalar path,
        // so the Karatsuba bookkeeping (the XOR mixes) is right.
        let mut rng = StdRng::seed_from_u64(0xABCD_128);
        for _ in 0..64 {
            let a: [u64; 2] = [rng.random(), rng.random()];
            let b: [u64; 2] = [rng.random(), rng.random()];
            let karatsuba = clmul_128x128(&a, &b);

            // Naive 4-mul reference using the scalar 64x64.
            let m00 = clmul_64x64_scalar(a[0], b[0]);
            let m01 = clmul_64x64_scalar(a[0], b[1]);
            let m10 = clmul_64x64_scalar(a[1], b[0]);
            let m11 = clmul_64x64_scalar(a[1], b[1]);
            let naive = [
                m00[0],
                m00[1] ^ m01[0] ^ m10[0],
                m11[0] ^ m01[1] ^ m10[1],
                m11[1],
            ];
            assert_eq!(karatsuba, naive);
        }
    }
}
