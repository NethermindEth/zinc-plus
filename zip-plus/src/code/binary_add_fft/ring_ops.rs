//! Arithmetic in `Z[X] / f̃` with `f̃ = X^32 + X^7 + X^3 + X^2 + 1`.
//!
//! Elements are represented as length-32 arrays of `Int<N>` coefficients
//! and are *always* kept in reduced form (degree `< 32`). The
//! multiplication routines here expand to a length-63 intermediate
//! buffer (since `deg(a) + deg(b) ≤ 62`) and then fold the high
//! monomials back down using `X^32 ≡ X^7 + X^3 + X^2 + 1 (mod f̃)`.
//!
//! Coefficient growth is **not** tracked here — callers must size `N`
//! to absorb the worst-case growth across the entire FFT depth. The
//! `<CHECK: bool>` const generic threaded through the IPRS code is not
//! yet wired in here; we use plain `+` and `*` and rely on `Int<N>`'s
//! limb width being large enough to avoid wraparound. Adding the
//! checked-mode plumbing is a follow-up once the FFT kernel is in.

use crypto_primitives::crypto_bigint_int::Int;
use num_traits::ConstZero;
use std::ops::Add;
use zinc_poly::univariate::dense::DensePolynomial;

use super::basis::{D, F_TILDE_LOWER_DEGREES, Gf2_32, lift_to_int_array};

/// Buffer length used by the unreduced intermediate of a product:
/// `deg < 2D - 1 = 63`.
pub const PRODUCT_LEN: usize = 2 * D - 1;

/// Length of a reduced element of `Z[X] / f̃`: `D = 32`.
pub const REDUCED_LEN: usize = D;

/// Trait alias for coefficient types this module operates on. Any type
/// that's `Copy`, has a const additive identity, supports owned-addition,
/// and can be compared for equality (used by the zero-skip optimisation
/// inside `reduce_mod_ftilde`). Satisfied by `Int<N>` (via
/// `crypto-primitives`) and by `i64`, `i128`, etc. (via std). Monomorphisation
/// gives each instantiation its own optimised code, so genericity has no
/// runtime cost — but lets us point the FFT kernel at native primitive
/// types where the bit-budget allows.
pub trait FftCoeff: Copy + ConstZero + Add<Self, Output = Self> + PartialEq {}
impl<T: Copy + ConstZero + Add<Self, Output = Self> + PartialEq> FftCoeff for T {}

/// Reduce a length-`63` buffer (degree `< 63`) to a length-`32`
/// element of `Z[X] / f̃`.
///
/// Folds each high monomial `X^j` for `j ≥ 32` into the lower
/// coefficients using `X^j ≡ X^(j-32) · (X^7 + X^3 + X^2 + 1)`. The
/// folding proceeds from the top down, so a single sweep suffices
/// (the new contributions land in `[j-32, j-25] ⊆ [0, 30]`, all
/// strictly below the current `j`).
#[inline]
pub fn reduce_mod_ftilde<C: FftCoeff>(buf: &[C; PRODUCT_LEN]) -> [C; REDUCED_LEN] {
    let mut local: [C; PRODUCT_LEN] = *buf;
    for j in (D..PRODUCT_LEN).rev() {
        let c = local[j];
        if c == C::ZERO {
            continue;
        }
        let base = j - D;
        for &shift in &F_TILDE_LOWER_DEGREES {
            local[base + shift] = local[base + shift] + c;
        }
    }
    let mut out = [C::ZERO; REDUCED_LEN];
    out.copy_from_slice(&local[..REDUCED_LEN]);
    out
}

/// Schoolbook multiplication of two length-32 polynomials with `Int<N>`
/// coefficients into a length-63 buffer. **No reduction** is applied.
///
/// Caller is responsible for sizing `N` so the coefficient sums do not
/// overflow: the largest output coefficient is the sum of up to `32`
/// products `a_i · b_j`.
#[inline]
pub fn schoolbook_mul<const N: usize>(
    a: &[Int<N>; REDUCED_LEN],
    b: &[Int<N>; REDUCED_LEN],
) -> [Int<N>; PRODUCT_LEN] {
    let mut out = [Int::<N>::ZERO; PRODUCT_LEN];
    for i in 0..REDUCED_LEN {
        let ai = a[i];
        if ai == Int::<N>::ZERO {
            continue;
        }
        for j in 0..REDUCED_LEN {
            // out[i+j] += a[i] * b[j]
            out[i + j] = out[i + j] + ai * b[j];
        }
    }
    out
}

/// Multiply two reduced elements of `Z[X] / f̃`.
#[inline]
pub fn mul_mod_ftilde<const N: usize>(
    a: &[Int<N>; REDUCED_LEN],
    b: &[Int<N>; REDUCED_LEN],
) -> [Int<N>; REDUCED_LEN] {
    let prod = schoolbook_mul(a, b);
    reduce_mod_ftilde(&prod)
}

/// Lift a `Gf2_32` to a reduced `Z[X] / f̃` element.
#[inline]
pub fn lift_gf32<const N: usize>(g: Gf2_32) -> [Int<N>; REDUCED_LEN] {
    lift_to_int_array(g)
}

/// Multiply a reduced element `x` of `Z[X] / f̃` by a *lifted* `Gf2_32`
/// twiddle.
///
/// Since the twiddle's coefficients live in `{0, 1} ⊂ Z`, the product
/// reduces to a sparse shift-and-add: `x · X^i` (a left shift of the
/// coefficient array by `i`) is added to the accumulator whenever bit
/// `i` of `twiddle` is set. No coefficient multiplications are
/// performed — the inner loop is purely `Int<N>` additions.
///
/// This is the *only* multiplication primitive used by the additive
/// FFT kernel: every butterfly twiddle is itself a lifted `Gf2_32`
/// element, even though the "values" being multiplied can have
/// arbitrary `Int<N>` coefficients after several stages.
#[inline]
pub fn mul_by_lifted_gf32<C: FftCoeff>(
    x: &[C; REDUCED_LEN],
    twiddle: Gf2_32,
) -> [C; REDUCED_LEN] {
    let mut buf = [C::ZERO; PRODUCT_LEN];
    for i in 0..D {
        if twiddle.coeff(i) {
            for j in 0..REDUCED_LEN {
                buf[j + i] = buf[j + i] + x[j];
            }
        }
    }
    reduce_mod_ftilde(&buf)
}

/// Add two reduced elements coefficient-wise.
#[inline]
pub fn add_in_z<C: FftCoeff>(a: &[C; REDUCED_LEN], b: &[C; REDUCED_LEN]) -> [C; REDUCED_LEN] {
    let mut out = [C::ZERO; REDUCED_LEN];
    for i in 0..REDUCED_LEN {
        out[i] = a[i] + b[i];
    }
    out
}

/// Reduce a `Z[X] / f̃` element mod 2, recovering a `Gf2_32` element.
/// Used by tests to verify that the lifted arithmetic agrees with the
/// underlying `GF(2^32)` arithmetic.
#[inline]
pub fn reduce_to_gf32<const N: usize>(a: &[Int<N>; REDUCED_LEN]) -> Gf2_32 {
    let mut bits = 0u32;
    for (i, c) in a.iter().enumerate() {
        // Take the low bit of c. Int<N> exposes a `is_odd` predicate
        // via `BitsConstant`; fall back to a portable "modulo 2"
        // expressed as `c - 2 · (c / 2)`. Simplest: use the public
        // `bit(0)` accessor if available, otherwise round-trip via
        // string / decimal. We use a direct subtraction approach to
        // stay generic over `N`.
        if is_odd(c) {
            bits |= 1u32 << i;
        }
    }
    Gf2_32(bits)
}

/// Whether the low bit of `c` is set. Computed as `c != 2 · (c / 2)`,
/// which works for any `Int<N>`.
#[inline]
fn is_odd<const N: usize>(c: &Int<N>) -> bool {
    let halved = *c >> 1;
    let doubled = halved + halved;
    *c != doubled
}

/// The arithmetic interface the additive-FFT kernel needs from a
/// `Z[X] / f̃` element. Implementors live in `Z[X] / f̃` (or, for the
/// reference implementation used in tests, in `GF(2^32)`).
///
/// This trait lets the FFT kernel be polymorphic over the concrete
/// "ring element" type while still being efficient: implementors are
/// expected to be `Clone` and to provide a fast multiplication by a
/// *lifted* `Gf2_32` twiddle (the only multiplication primitive the
/// kernel uses).
///
/// `Send + Sync` are required so the kernel can parallelize the
/// butterfly loop across blocks within each stage (via rayon under
/// the `parallel` feature).
pub trait FftRingElement: Sized + Clone + Send + Sync {
    /// Additive identity.
    fn fft_zero() -> Self;

    /// `self + other`.
    fn fft_add(&self, other: &Self) -> Self;

    /// Multiply by a `Gf2_32` twiddle, structurally lifted to
    /// `Z[X] / f̃`. The twiddle's coefficients are taken to be
    /// `{0, 1} ⊂ Z`, matching the literal lift of the
    /// `GF(2^32) = F_2[X] / f` element.
    fn fft_mul_lifted_gf32(&self, twiddle: Gf2_32) -> Self;
}

/// Lifted-`Z[X]/f̃` ring element (the natural shape for the additive FFT
/// over the lifted ring). Multiplication by a twiddle delegates to
/// [`mul_by_lifted_gf32`]; addition is coefficient-wise.
impl<const N: usize> FftRingElement for DensePolynomial<Int<N>, REDUCED_LEN> {
    #[inline]
    fn fft_zero() -> Self {
        Self {
            coeffs: [Int::<N>::ZERO; REDUCED_LEN],
        }
    }

    #[inline]
    fn fft_add(&self, other: &Self) -> Self {
        let mut coeffs = [Int::<N>::ZERO; REDUCED_LEN];
        for i in 0..REDUCED_LEN {
            coeffs[i] = self.coeffs[i] + other.coeffs[i];
        }
        Self { coeffs }
    }

    #[inline]
    fn fft_mul_lifted_gf32(&self, twiddle: Gf2_32) -> Self {
        let coeffs = mul_by_lifted_gf32(&self.coeffs, twiddle);
        Self { coeffs }
    }
}

/// Native-`i128`-coefficient variant. Same algebra as the
/// `DensePolynomial<Int<N>, 32>` impl above, but the coefficient
/// arithmetic compiles to native CPU 128-bit adds instead of
/// `crypto-bigint`'s multi-limb generic adds. Valid for any `row_len`
/// such that the coefficient growth `33^m · max(|input|)` stays below
/// `2^127` — e.g. `row_len ≤ 2^20` with `BinaryPoly<32>` inputs
/// (m ≤ 22).
///
/// We delegate to the generic [`mul_by_lifted_gf32`] kernel rather
/// than the split-coefficient SIMD variant
/// ([`mul_by_lifted_gf32_i128_split`]) — on Apple M4 the scalar
/// `i128` ADD/ADC dual-issue is already at the throughput ceiling
/// (~265ms commit at `poly_size=2^14, batch=11`), and the
/// split/recombine overhead in the SIMD variant *slows* the kernel
/// down by ~30%. The split kernel is kept for platforms where it
/// might pay (AVX-512 hosts that can SIMD 4× `i128` adds in 512-bit
/// vectors); see [`mul_by_lifted_gf32_i128_split`].
impl FftRingElement for DensePolynomial<i128, REDUCED_LEN> {
    #[inline]
    fn fft_zero() -> Self {
        Self {
            coeffs: [0i128; REDUCED_LEN],
        }
    }

    #[inline]
    fn fft_add(&self, other: &Self) -> Self {
        let mut coeffs = [0i128; REDUCED_LEN];
        for i in 0..REDUCED_LEN {
            coeffs[i] = self.coeffs[i] + other.coeffs[i];
        }
        Self { coeffs }
    }

    #[inline]
    fn fft_mul_lifted_gf32(&self, twiddle: Gf2_32) -> Self {
        let coeffs = mul_by_lifted_gf32(&self.coeffs, twiddle);
        Self { coeffs }
    }
}

/// SIMD-friendly variant of [`mul_by_lifted_gf32`] specialised to
/// `i128`. The 32-element `i128` array is unpacked into separate
/// `[u64; 32]` (low limbs) and `[i64; 32]` (high limbs) — both
/// natively SIMD types — then the inner loops over `u64` adds and
/// `i64` adds compile to NEON / AVX vector ops via auto-vectorization.
/// Carry between lo and hi is collected once per inner iteration into
/// a `[u64; 32]` carries scratch and applied in a second pass; this
/// breaks the lo→hi data dependency that would otherwise prevent
/// vectorization.
///
/// Algorithmically identical to [`mul_by_lifted_gf32`] (verified by
/// the `lifted_fft_mod_2_matches_gf32_fft_i128` soundness witness).
///
/// **Currently unused on Apple Silicon** — the scalar `i128` path
/// outperforms this on Apple M4 because the split/recombine overhead
/// dominates the modest SIMD savings on NEON's 128-bit registers.
/// May be worth swapping in on AVX-512 hosts where 512-bit vectors
/// can SIMD 4× `i128` adds at once. See the FftRingElement impl above
/// for the trade-off context.
#[inline]
#[allow(dead_code)]
pub fn mul_by_lifted_gf32_i128_split(x: &[i128; REDUCED_LEN], twiddle: Gf2_32) -> [i128; REDUCED_LEN] {
    // Split inputs: lo (u64) and hi (i64). The compiler can SIMD-load
    // these via NEON LDR.2D / LDR.16B.
    let mut x_lo = [0u64; REDUCED_LEN];
    let mut x_hi = [0i64; REDUCED_LEN];
    for j in 0..REDUCED_LEN {
        x_lo[j] = x[j] as u64;
        x_hi[j] = (x[j] >> 64) as i64;
    }

    // Unreduced product buffer (degree < 63).
    let mut buf_lo = [0u64; PRODUCT_LEN];
    let mut buf_hi = [0i64; PRODUCT_LEN];

    for i in 0..D {
        if !twiddle.coeff(i) {
            continue;
        }
        // Pass 1: lo additions, collecting carries.
        // Both this and pass 2 are SIMD-vectorizable.
        let mut carries = [0u64; REDUCED_LEN];
        for j in 0..REDUCED_LEN {
            let (s, c) = buf_lo[j + i].overflowing_add(x_lo[j]);
            buf_lo[j + i] = s;
            carries[j] = c as u64;
        }
        // Pass 2: hi additions, incorporating carries.
        for j in 0..REDUCED_LEN {
            buf_hi[j + i] = buf_hi[j + i]
                .wrapping_add(x_hi[j])
                .wrapping_add(carries[j] as i64);
        }
    }

    // Reduce mod f̃ on the split representation.
    // Cascade from high to low so single-pass suffices.
    for j in (D..PRODUCT_LEN).rev() {
        let lo = buf_lo[j];
        let hi = buf_hi[j];
        if lo == 0 && hi == 0 {
            continue;
        }
        let base = j - D;
        for &shift in &F_TILDE_LOWER_DEGREES {
            let (s, c) = buf_lo[base + shift].overflowing_add(lo);
            buf_lo[base + shift] = s;
            buf_hi[base + shift] = buf_hi[base + shift]
                .wrapping_add(hi)
                .wrapping_add(c as i64);
        }
    }

    // Recombine into i128 output.
    let mut out = [0i128; REDUCED_LEN];
    for j in 0..REDUCED_LEN {
        out[j] = (buf_lo[j] as i128) | ((buf_hi[j] as i128) << 64);
    }
    out
}

/// Native-`i32`-coefficient variant. Tightest "single primitive" type
/// for the FFT output of a `BinaryPoly<32>` codeword. Bit-budget
/// analysis: the codeword `P(v_k) = ∑_i c_i · X_i(v_k)` is a sum of
/// at most `codeword_len` products of `{0,1}`-coefficient lifted GF
/// elements. Each product in `Z[X]/f̃` has coefficients bounded by a
/// small constant (~32×5 after reduction). So the worst-case
/// magnitude scales linearly with `codeword_len`, not exponentially
/// with `m`. Concretely: at `codeword_len = 2^16`, worst-case ≈
/// `2^16 × ~200 ≈ 2^24`, well within `i32`'s `±2^31` range. The
/// empirical max for random `BinaryPoly<32>` inputs at
/// `poly_size = 2^14` is `≈ 2^21` (see `column_size_report` test).
impl FftRingElement for DensePolynomial<i32, REDUCED_LEN> {
    #[inline]
    fn fft_zero() -> Self {
        Self {
            coeffs: [0i32; REDUCED_LEN],
        }
    }

    #[inline]
    fn fft_add(&self, other: &Self) -> Self {
        let mut coeffs = [0i32; REDUCED_LEN];
        for i in 0..REDUCED_LEN {
            coeffs[i] = self.coeffs[i] + other.coeffs[i];
        }
        Self { coeffs }
    }

    #[inline]
    fn fft_mul_lifted_gf32(&self, twiddle: Gf2_32) -> Self {
        let coeffs = mul_by_lifted_gf32(&self.coeffs, twiddle);
        Self { coeffs }
    }
}

/// Reference `GF(2^32)` element. Used in tests to verify that the
/// lifted FFT, reduced mod 2, matches the underlying `GF(2^32)` FFT.
/// "Multiplication by a lifted twiddle" here is just `GF(2^32)`
/// multiplication, since the twiddle and the value live in the same
/// field.
impl FftRingElement for Gf2_32 {
    #[inline]
    fn fft_zero() -> Self {
        Gf2_32::ZERO
    }

    #[inline]
    fn fft_add(&self, other: &Self) -> Self {
        Gf2_32::add(*self, *other)
    }

    #[inline]
    fn fft_mul_lifted_gf32(&self, twiddle: Gf2_32) -> Self {
        Gf2_32::mul(*self, twiddle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::ConstOne;

    /// Helper: build a reduced element with a single nonzero coefficient
    /// `coeff` at position `degree`.
    fn monomial<const N: usize>(coeff: i64, degree: usize) -> [Int<N>; REDUCED_LEN] {
        assert!(degree < REDUCED_LEN);
        let mut out = [Int::<N>::ZERO; REDUCED_LEN];
        out[degree] = Int::<N>::from(coeff);
        out
    }

    /// Helper: build a length-63 buffer with a single nonzero coefficient.
    fn monomial_unreduced<const N: usize>(coeff: i64, degree: usize) -> [Int<N>; PRODUCT_LEN] {
        assert!(degree < PRODUCT_LEN);
        let mut out = [Int::<N>::ZERO; PRODUCT_LEN];
        out[degree] = Int::<N>::from(coeff);
        out
    }

    #[test]
    fn reduce_mod_ftilde_passes_through_low_degrees() {
        // A buffer with only [0..32) entries set should be returned unchanged.
        let mut buf = [Int::<2>::ZERO; PRODUCT_LEN];
        for (i, slot) in buf.iter_mut().enumerate().take(REDUCED_LEN) {
            *slot = Int::<2>::from((i as i64) + 1);
        }
        let reduced = reduce_mod_ftilde(&buf);
        for i in 0..REDUCED_LEN {
            assert_eq!(reduced[i], Int::<2>::from((i as i64) + 1));
        }
    }

    #[test]
    fn reduce_mod_ftilde_folds_x32_correctly() {
        // X^32 ≡ X^7 + X^3 + X^2 + 1.
        let buf = monomial_unreduced::<2>(1, 32);
        let reduced = reduce_mod_ftilde(&buf);
        let mut expected = [Int::<2>::ZERO; REDUCED_LEN];
        expected[0] = Int::<2>::ONE;
        expected[2] = Int::<2>::ONE;
        expected[3] = Int::<2>::ONE;
        expected[7] = Int::<2>::ONE;
        assert_eq!(reduced, expected);
    }

    #[test]
    fn reduce_mod_ftilde_folds_x33_correctly() {
        // X^33 ≡ X · X^32 ≡ X^8 + X^4 + X^3 + X.
        let buf = monomial_unreduced::<2>(1, 33);
        let reduced = reduce_mod_ftilde(&buf);
        let mut expected = [Int::<2>::ZERO; REDUCED_LEN];
        expected[1] = Int::<2>::ONE;
        expected[3] = Int::<2>::ONE;
        expected[4] = Int::<2>::ONE;
        expected[8] = Int::<2>::ONE;
        assert_eq!(reduced, expected);
    }

    #[test]
    fn reduce_mod_ftilde_cascades_when_low_terms_themselves_overflow() {
        // X^62 is the highest unreduced monomial. Folding it once produces
        // X^30 contributions at offset 30+{0,2,3,7} = {30,32,33,37}, all
        // below 62 — so the sweep handles those when it reaches them.
        let buf = monomial_unreduced::<3>(1, 62);
        let reduced = reduce_mod_ftilde(&buf);
        // Sanity: result should not panic and should be in [0, 32).
        // Compute the expected expansion by hand for one check:
        // X^62 = X^30 · X^32 ≡ X^30 · (X^7 + X^3 + X^2 + 1)
        //      = X^37 + X^33 + X^32 + X^30.
        // X^37 ≡ X^5 · X^32 = X^12 + X^8 + X^7 + X^5.
        // X^33 ≡ X^8 + X^4 + X^3 + X.
        // X^32 ≡ X^7 + X^3 + X^2 + 1.
        // X^30 stays.
        // Sum (over Z): coeffs at {0,1,2,3,3,4,5,7,7,8,8,12,30} =
        //   {0:1, 1:1, 2:1, 3:2, 4:1, 5:1, 7:2, 8:2, 12:1, 30:1}.
        let mut expected = [Int::<3>::ZERO; REDUCED_LEN];
        expected[0] = Int::<3>::ONE;
        expected[1] = Int::<3>::ONE;
        expected[2] = Int::<3>::ONE;
        expected[3] = Int::<3>::from(2);
        expected[4] = Int::<3>::ONE;
        expected[5] = Int::<3>::ONE;
        expected[7] = Int::<3>::from(2);
        expected[8] = Int::<3>::from(2);
        expected[12] = Int::<3>::ONE;
        expected[30] = Int::<3>::ONE;
        assert_eq!(reduced, expected);
    }

    #[test]
    fn schoolbook_mul_of_monomials() {
        // X^3 · X^5 = X^8 (length-63 buffer, no reduction needed).
        let a = monomial::<2>(1, 3);
        let b = monomial::<2>(1, 5);
        let prod = schoolbook_mul(&a, &b);
        let mut expected = [Int::<2>::ZERO; PRODUCT_LEN];
        expected[8] = Int::<2>::ONE;
        assert_eq!(prod, expected);
    }

    #[test]
    fn mul_mod_ftilde_x16_times_x16_is_x32_reduced() {
        // X^16 · X^16 = X^32 ≡ X^7 + X^3 + X^2 + 1.
        let a = monomial::<2>(1, 16);
        let prod = mul_mod_ftilde(&a, &a);
        let mut expected = [Int::<2>::ZERO; REDUCED_LEN];
        expected[0] = Int::<2>::ONE;
        expected[2] = Int::<2>::ONE;
        expected[3] = Int::<2>::ONE;
        expected[7] = Int::<2>::ONE;
        assert_eq!(prod, expected);
    }

    #[test]
    fn lift_then_reduce_recovers_gf32() {
        for raw in [0u32, 1, 0xff, 0xdead_beef, 0x8000_0000] {
            let g = Gf2_32(raw);
            let lifted: [Int<2>; REDUCED_LEN] = lift_gf32(g);
            let back = reduce_to_gf32(&lifted);
            assert_eq!(back, g, "round-trip failed for {raw:#010x}");
        }
    }

    /// Core soundness sanity check: lifted multiplication, reduced mod 2,
    /// agrees with `Gf2_32::mul`.
    #[test]
    fn lifted_mul_commutes_with_gf32_mul() {
        let cases = [
            (Gf2_32(0x1), Gf2_32(0x1)),
            (Gf2_32(0x2), Gf2_32(0x8000_0000)),
            (Gf2_32(0xcafe_babe), Gf2_32(0xdead_beef)),
            (Gf2_32(0xffff_ffff), Gf2_32(0xffff_ffff)),
        ];
        for (a, b) in cases {
            let la: [Int<4>; REDUCED_LEN] = lift_gf32(a);
            let lb: [Int<4>; REDUCED_LEN] = lift_gf32(b);
            let lifted_prod = mul_mod_ftilde(&la, &lb);
            let reduced_prod = reduce_to_gf32(&lifted_prod);
            assert_eq!(reduced_prod, a.mul(b), "for a={a:?}, b={b:?}");
        }
    }

    #[test]
    fn lifted_add_commutes_with_gf32_add() {
        let a = Gf2_32(0xcafe_babe);
        let b = Gf2_32(0xdead_beef);
        let la: [Int<2>; REDUCED_LEN] = lift_gf32(a);
        let lb: [Int<2>; REDUCED_LEN] = lift_gf32(b);
        let lifted_sum = add_in_z(&la, &lb);
        assert_eq!(reduce_to_gf32(&lifted_sum), a.add(b));
    }

    /// `mul_by_lifted_gf32(lift(a), b)` must agree, after reduction mod 2,
    /// with `Gf2_32::mul(a, b)` — and must agree at the integer level with
    /// the full schoolbook multiplication `lift(a) · lift(b)`.
    #[test]
    fn mul_by_lifted_gf32_agrees_with_gf32_mul() {
        let cases = [
            (Gf2_32(0x1), Gf2_32(0x1)),
            (Gf2_32(0x2), Gf2_32(0x8000_0000)),
            (Gf2_32(0xcafe_babe), Gf2_32(0xdead_beef)),
            (Gf2_32(0xffff_ffff), Gf2_32(0xffff_ffff)),
            (Gf2_32(0x5555_5555), Gf2_32(0xaaaa_aaaa)),
        ];
        for (a, b) in cases {
            let la: [Int<4>; REDUCED_LEN] = lift_gf32(a);
            let via_lifted_twiddle = mul_by_lifted_gf32(&la, b);
            assert_eq!(
                reduce_to_gf32(&via_lifted_twiddle),
                a.mul(b),
                "for a={a:?}, b={b:?}"
            );
            // Also agree with full schoolbook mul at the integer level
            // (both have {0,1} factor inputs, so no overflow concerns here).
            let lb: [Int<4>; REDUCED_LEN] = lift_gf32(b);
            let via_schoolbook = mul_mod_ftilde(&la, &lb);
            assert_eq!(via_lifted_twiddle, via_schoolbook, "integer-level disagreement");
        }
    }

    /// Multiplication by `Gf2_32::ZERO` annihilates; multiplication by
    /// `Gf2_32::ONE` is the identity.
    #[test]
    fn mul_by_lifted_gf32_special_values() {
        let mut x = [Int::<2>::ZERO; REDUCED_LEN];
        x[0] = Int::<2>::from(7);
        x[5] = Int::<2>::from(-3);
        x[31] = Int::<2>::from(2);
        assert_eq!(mul_by_lifted_gf32(&x, Gf2_32::ZERO), [Int::<2>::ZERO; REDUCED_LEN]);
        assert_eq!(mul_by_lifted_gf32(&x, Gf2_32::ONE), x);
    }
}
