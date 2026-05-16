//! Arithmetic in `Z[X] / f̃` with `f̃ = X^16 + X^5 + X^3 + X^2 + 1`.
//!
//! Elements are represented as length-16 arrays of `Int<N>` coefficients
//! and are *always* kept in reduced form (degree `< 16`). The
//! multiplication routines expand to a length-31 intermediate buffer
//! (since `deg(a) + deg(b) ≤ 30`) and then fold the high monomials
//! back down using `X^16 ≡ X^5 + X^3 + X^2 + 1 (mod f̃)`.
//!
//! Mirrors `binary_add_fft_8/ring_ops.rs` but for D = 16. `i64`,
//! `i128`, and `Int<N>` are provided, plus a narrow `i32` variant.
//! The `i32` element is *not* generally safe — for large codewords the
//! lifted coefficients exceed 32 bits — so it is used only on the
//! encode fast path, gated by a sound per-instance bound computed in
//! `BinaryAddFft16Code::new` (see [`I32FftConvert`]).

use crypto_primitives::crypto_bigint_int::Int;
use num_traits::ConstZero;
use std::ops::Add;
use zinc_poly::univariate::dense::DensePolynomial;

use super::basis::{D_16, F_TILDE_LOWER_DEGREES_16, Gf2_16, lift_to_int_array_16};

/// Buffer length used by the unreduced intermediate of a product:
/// `deg < 2D - 1 = 31`.
pub const PRODUCT_LEN_16: usize = 2 * D_16 - 1;

/// Length of a reduced element of `Z[X] / f̃`: `D = 16`.
pub const REDUCED_LEN_16: usize = D_16;

/// Trait alias for coefficient types this module operates on.
pub trait FftCoeff16: Copy + ConstZero + Add<Self, Output = Self> + PartialEq {}
impl<T: Copy + ConstZero + Add<Self, Output = Self> + PartialEq> FftCoeff16 for T {}

/// Reduce a length-`31` buffer (degree `< 31`) to a length-`16`
/// element of `Z[X] / f̃`.
#[inline]
pub fn reduce_mod_ftilde_16<C: FftCoeff16>(buf: &[C; PRODUCT_LEN_16]) -> [C; REDUCED_LEN_16] {
    let mut local: [C; PRODUCT_LEN_16] = *buf;
    for j in (D_16..PRODUCT_LEN_16).rev() {
        let c = local[j];
        if c == C::ZERO {
            continue;
        }
        let base = j - D_16;
        for &shift in &F_TILDE_LOWER_DEGREES_16 {
            local[base + shift] = local[base + shift] + c;
        }
    }
    let mut out = [C::ZERO; REDUCED_LEN_16];
    out.copy_from_slice(&local[..REDUCED_LEN_16]);
    out
}

/// Schoolbook multiplication of two length-16 polynomials with `Int<N>`
/// coefficients into a length-31 buffer. **No reduction** is applied.
#[inline]
pub fn schoolbook_mul_16<const N: usize>(
    a: &[Int<N>; REDUCED_LEN_16],
    b: &[Int<N>; REDUCED_LEN_16],
) -> [Int<N>; PRODUCT_LEN_16] {
    let mut out = [Int::<N>::ZERO; PRODUCT_LEN_16];
    for i in 0..REDUCED_LEN_16 {
        let ai = a[i];
        if ai == Int::<N>::ZERO {
            continue;
        }
        for j in 0..REDUCED_LEN_16 {
            out[i + j] = out[i + j] + ai * b[j];
        }
    }
    out
}

/// Multiply two reduced elements of `Z[X] / f̃`.
#[inline]
pub fn mul_mod_ftilde_16<const N: usize>(
    a: &[Int<N>; REDUCED_LEN_16],
    b: &[Int<N>; REDUCED_LEN_16],
) -> [Int<N>; REDUCED_LEN_16] {
    let prod = schoolbook_mul_16(a, b);
    reduce_mod_ftilde_16(&prod)
}

/// Lift a `Gf2_16` to a reduced `Z[X] / f̃` element.
#[inline]
pub fn lift_gf16<const N: usize>(g: Gf2_16) -> [Int<N>; REDUCED_LEN_16] {
    lift_to_int_array_16(g)
}

/// Multiply a reduced element `x` of `Z[X] / f̃` by a *lifted* `Gf2_16`
/// twiddle.
///
/// Since the twiddle's coefficients live in `{0, 1} ⊂ Z`, the product
/// reduces to a sparse shift-and-add: `x · X^i` (a left shift of the
/// coefficient array by `i`) is added to the accumulator whenever bit
/// `i` of `twiddle` is set.
#[inline]
pub fn mul_by_lifted_gf16<C: FftCoeff16>(
    x: &[C; REDUCED_LEN_16],
    twiddle: Gf2_16,
) -> [C; REDUCED_LEN_16] {
    let mut buf = [C::ZERO; PRODUCT_LEN_16];
    // Iterate the set bits of the twiddle directly (`trailing_zeros` +
    // clear-lowest-bit) rather than branching on all 16 bit positions:
    // the twiddle bits are pseudo-random, so a per-bit `if` mispredicts
    // ~half the time. The slice form lets the add autovectorise.
    let mut bits = twiddle.0;
    while bits != 0 {
        let i = bits.trailing_zeros() as usize;
        bits &= bits - 1;
        let dst = &mut buf[i..i + REDUCED_LEN_16];
        for (d, &xj) in dst.iter_mut().zip(x.iter()) {
            *d = *d + xj;
        }
    }
    reduce_mod_ftilde_16(&buf)
}

/// Add two reduced elements coefficient-wise.
#[inline]
pub fn add_in_z_16<C: FftCoeff16>(
    a: &[C; REDUCED_LEN_16],
    b: &[C; REDUCED_LEN_16],
) -> [C; REDUCED_LEN_16] {
    let mut out = [C::ZERO; REDUCED_LEN_16];
    for i in 0..REDUCED_LEN_16 {
        out[i] = a[i] + b[i];
    }
    out
}

/// Reduce a `Z[X] / f̃` element mod 2, recovering a `Gf2_16` element.
/// Used by tests to verify that the lifted arithmetic agrees with the
/// underlying `GF(2^16)` arithmetic.
#[inline]
pub fn reduce_to_gf16<const N: usize>(a: &[Int<N>; REDUCED_LEN_16]) -> Gf2_16 {
    let mut bits = 0u16;
    for (i, c) in a.iter().enumerate() {
        if is_odd(c) {
            bits |= 1u16 << i;
        }
    }
    Gf2_16(bits)
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
/// `Z[X] / f̃` element (D = 16 version).
pub trait FftRingElement16: Sized + Clone + Send + Sync {
    /// Additive identity.
    fn fft_zero() -> Self;

    /// `self + other`.
    fn fft_add(&self, other: &Self) -> Self;

    /// Multiply by a `Gf2_16` twiddle, structurally lifted to `Z[X] / f̃`.
    fn fft_mul_lifted_gf16(&self, twiddle: Gf2_16) -> Self;

    /// Length-`(2D-1) = 31` unreduced accumulator type.
    type UnreducedAcc: Send + Sync + Clone;

    /// Initialize an empty unreduced accumulator (all zeros).
    fn fft_unreduced_zero() -> Self::UnreducedAcc;

    /// Add `input × twiddle` (lifted, unreduced) into `acc`.
    fn fft_acc_mul_lifted_gf16(
        input: &Self,
        twiddle: Gf2_16,
        acc: &mut Self::UnreducedAcc,
    );

    /// Reduce an unreduced accumulator mod `f̃` and produce a finished
    /// ring element.
    fn fft_finalize_acc(acc: Self::UnreducedAcc) -> Self;
}

/// Lifted-`Z[X]/f̃` ring element backed by `Int<N>` coefficients.
impl<const N: usize> FftRingElement16 for DensePolynomial<Int<N>, REDUCED_LEN_16> {
    #[inline]
    fn fft_zero() -> Self {
        Self {
            coeffs: [Int::<N>::ZERO; REDUCED_LEN_16],
        }
    }

    #[inline]
    fn fft_add(&self, other: &Self) -> Self {
        let mut coeffs = [Int::<N>::ZERO; REDUCED_LEN_16];
        for i in 0..REDUCED_LEN_16 {
            coeffs[i] = self.coeffs[i] + other.coeffs[i];
        }
        Self { coeffs }
    }

    #[inline]
    fn fft_mul_lifted_gf16(&self, twiddle: Gf2_16) -> Self {
        let coeffs = mul_by_lifted_gf16(&self.coeffs, twiddle);
        Self { coeffs }
    }

    type UnreducedAcc = [Int<N>; PRODUCT_LEN_16];

    #[inline]
    fn fft_unreduced_zero() -> Self::UnreducedAcc {
        [Int::<N>::ZERO; PRODUCT_LEN_16]
    }

    #[inline]
    fn fft_acc_mul_lifted_gf16(input: &Self, twiddle: Gf2_16, acc: &mut Self::UnreducedAcc) {
        let x = &input.coeffs;
        let mut bits = twiddle.0;
        while bits != 0 {
            let i = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            let dst = &mut acc[i..i + REDUCED_LEN_16];
            for (d, &xj) in dst.iter_mut().zip(x.iter()) {
                *d = *d + xj;
            }
        }
    }

    #[inline]
    fn fft_finalize_acc(acc: Self::UnreducedAcc) -> Self {
        let coeffs = reduce_mod_ftilde_16(&acc);
        Self { coeffs }
    }
}

/// Native-`i128`-coefficient variant.
impl FftRingElement16 for DensePolynomial<i128, REDUCED_LEN_16> {
    #[inline]
    fn fft_zero() -> Self {
        Self {
            coeffs: [0i128; REDUCED_LEN_16],
        }
    }

    #[inline]
    fn fft_add(&self, other: &Self) -> Self {
        let mut coeffs = [0i128; REDUCED_LEN_16];
        for i in 0..REDUCED_LEN_16 {
            coeffs[i] = self.coeffs[i] + other.coeffs[i];
        }
        Self { coeffs }
    }

    #[inline]
    fn fft_mul_lifted_gf16(&self, twiddle: Gf2_16) -> Self {
        let coeffs = mul_by_lifted_gf16(&self.coeffs, twiddle);
        Self { coeffs }
    }

    type UnreducedAcc = [i128; PRODUCT_LEN_16];
    #[inline]
    fn fft_unreduced_zero() -> Self::UnreducedAcc {
        [0i128; PRODUCT_LEN_16]
    }
    #[inline]
    fn fft_acc_mul_lifted_gf16(input: &Self, twiddle: Gf2_16, acc: &mut Self::UnreducedAcc) {
        let x = &input.coeffs;
        let mut bits = twiddle.0;
        while bits != 0 {
            let i = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            let dst = &mut acc[i..i + REDUCED_LEN_16];
            for (d, &xj) in dst.iter_mut().zip(x.iter()) {
                *d = d.wrapping_add(xj);
            }
        }
    }
    #[inline]
    fn fft_finalize_acc(acc: Self::UnreducedAcc) -> Self {
        let coeffs = reduce_mod_ftilde_16(&acc);
        Self { coeffs }
    }
}

/// Native-`i64` variant. This is the default coefficient type for the
/// D=16 codeword polynomial.
impl FftRingElement16 for DensePolynomial<i64, REDUCED_LEN_16> {
    #[inline]
    fn fft_zero() -> Self {
        Self {
            coeffs: [0i64; REDUCED_LEN_16],
        }
    }

    #[inline]
    fn fft_add(&self, other: &Self) -> Self {
        let mut coeffs = [0i64; REDUCED_LEN_16];
        for i in 0..REDUCED_LEN_16 {
            coeffs[i] = self.coeffs[i] + other.coeffs[i];
        }
        Self { coeffs }
    }

    #[inline]
    fn fft_mul_lifted_gf16(&self, twiddle: Gf2_16) -> Self {
        let coeffs = mul_by_lifted_gf16(&self.coeffs, twiddle);
        Self { coeffs }
    }

    type UnreducedAcc = [i64; PRODUCT_LEN_16];
    #[inline]
    fn fft_unreduced_zero() -> Self::UnreducedAcc {
        [0i64; PRODUCT_LEN_16]
    }
    #[inline]
    fn fft_acc_mul_lifted_gf16(input: &Self, twiddle: Gf2_16, acc: &mut Self::UnreducedAcc) {
        let x = &input.coeffs;
        let mut bits = twiddle.0;
        while bits != 0 {
            let i = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            let dst = &mut acc[i..i + REDUCED_LEN_16];
            for (d, &xj) in dst.iter_mut().zip(x.iter()) {
                *d = d.wrapping_add(xj);
            }
        }
    }
    #[inline]
    fn fft_finalize_acc(acc: Self::UnreducedAcc) -> Self {
        let coeffs = reduce_mod_ftilde_16(&acc);
        Self { coeffs }
    }
}

/// Narrow native-`i32` variant. Only sound when the whole transform
/// provably stays within 32 bits; selected on the encode fast path via
/// [`I32FftConvert`] after `BinaryAddFft16Code::new` proves the bound.
impl FftRingElement16 for DensePolynomial<i32, REDUCED_LEN_16> {
    #[inline]
    fn fft_zero() -> Self {
        Self {
            coeffs: [0i32; REDUCED_LEN_16],
        }
    }

    #[inline]
    fn fft_add(&self, other: &Self) -> Self {
        let mut coeffs = [0i32; REDUCED_LEN_16];
        for i in 0..REDUCED_LEN_16 {
            coeffs[i] = self.coeffs[i] + other.coeffs[i];
        }
        Self { coeffs }
    }

    #[inline]
    fn fft_mul_lifted_gf16(&self, twiddle: Gf2_16) -> Self {
        let coeffs = mul_by_lifted_gf16(&self.coeffs, twiddle);
        Self { coeffs }
    }

    type UnreducedAcc = [i32; PRODUCT_LEN_16];
    #[inline]
    fn fft_unreduced_zero() -> Self::UnreducedAcc {
        [0i32; PRODUCT_LEN_16]
    }
    #[inline]
    fn fft_acc_mul_lifted_gf16(input: &Self, twiddle: Gf2_16, acc: &mut Self::UnreducedAcc) {
        let x = &input.coeffs;
        let mut bits = twiddle.0;
        while bits != 0 {
            let i = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            let dst = &mut acc[i..i + REDUCED_LEN_16];
            for (d, &xj) in dst.iter_mut().zip(x.iter()) {
                *d = d.wrapping_add(xj);
            }
        }
    }
    #[inline]
    fn fft_finalize_acc(acc: Self::UnreducedAcc) -> Self {
        let coeffs = reduce_mod_ftilde_16(&acc);
        Self { coeffs }
    }
}

/// Conversion between a codeword element type (`Zt::Cw`) and the
/// `i32`-backed FFT working buffer used by the narrow-integer encode
/// fast path.
///
/// `BinaryAddFft16Code::new` runs a sound worst-case probe; when it
/// proves the whole transform stays within `i32`, `encode` lifts inputs
/// via `narrow_to_i32_fft`, runs the FFT on 64-byte `i32` elements
/// instead of 128-byte `i64` ones (halving encode memory traffic), and
/// widens the result with `widen_from_i32_fft`. The produced codeword
/// is bit-identical to the `i64` path, so the proof and verifier are
/// unaffected.
pub trait I32FftConvert {
    /// Narrow a freshly-lifted codeword element to the `i32` FFT buffer.
    /// Lossless whenever the fast path is selected.
    fn narrow_to_i32_fft(&self) -> DensePolynomial<i32, REDUCED_LEN_16>;
    /// Widen a finished `i32` FFT element back to the codeword type.
    fn widen_from_i32_fft(p: &DensePolynomial<i32, REDUCED_LEN_16>) -> Self;
}

impl I32FftConvert for DensePolynomial<i64, REDUCED_LEN_16> {
    #[inline]
    fn narrow_to_i32_fft(&self) -> DensePolynomial<i32, REDUCED_LEN_16> {
        DensePolynomial {
            coeffs: self.coeffs.map(|c| c as i32),
        }
    }
    #[inline]
    fn widen_from_i32_fft(p: &DensePolynomial<i32, REDUCED_LEN_16>) -> Self {
        DensePolynomial {
            coeffs: p.coeffs.map(i64::from),
        }
    }
}

/// Reference `GF(2^16)` element. Used in tests to verify that the
/// lifted FFT, reduced mod 2, matches the underlying `GF(2^16)` FFT.
impl FftRingElement16 for Gf2_16 {
    #[inline]
    fn fft_zero() -> Self {
        Gf2_16::ZERO
    }

    #[inline]
    fn fft_add(&self, other: &Self) -> Self {
        Gf2_16::add(*self, *other)
    }

    #[inline]
    fn fft_mul_lifted_gf16(&self, twiddle: Gf2_16) -> Self {
        Gf2_16::mul(*self, twiddle)
    }

    type UnreducedAcc = Self;
    #[inline]
    fn fft_unreduced_zero() -> Self::UnreducedAcc {
        Gf2_16::ZERO
    }
    #[inline]
    fn fft_acc_mul_lifted_gf16(input: &Self, twiddle: Gf2_16, acc: &mut Self::UnreducedAcc) {
        *acc = acc.add(input.mul(twiddle));
    }
    #[inline]
    fn fft_finalize_acc(acc: Self::UnreducedAcc) -> Self {
        acc
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::ConstOne;

    /// Helper: build a reduced element with a single nonzero coefficient.
    fn monomial<const N: usize>(coeff: i64, degree: usize) -> [Int<N>; REDUCED_LEN_16] {
        assert!(degree < REDUCED_LEN_16);
        let mut out = [Int::<N>::ZERO; REDUCED_LEN_16];
        out[degree] = Int::<N>::from(coeff);
        out
    }

    /// Helper: build a length-31 buffer with a single nonzero coefficient.
    fn monomial_unreduced<const N: usize>(coeff: i64, degree: usize) -> [Int<N>; PRODUCT_LEN_16] {
        assert!(degree < PRODUCT_LEN_16);
        let mut out = [Int::<N>::ZERO; PRODUCT_LEN_16];
        out[degree] = Int::<N>::from(coeff);
        out
    }

    #[test]
    fn reduce_mod_ftilde_passes_through_low_degrees() {
        let mut buf = [Int::<2>::ZERO; PRODUCT_LEN_16];
        for (i, slot) in buf.iter_mut().enumerate().take(REDUCED_LEN_16) {
            *slot = Int::<2>::from((i as i64) + 1);
        }
        let reduced = reduce_mod_ftilde_16(&buf);
        for i in 0..REDUCED_LEN_16 {
            assert_eq!(reduced[i], Int::<2>::from((i as i64) + 1));
        }
    }

    #[test]
    fn reduce_mod_ftilde_folds_x16_correctly() {
        // X^16 ≡ X^5 + X^3 + X^2 + 1.
        let buf = monomial_unreduced::<2>(1, 16);
        let reduced = reduce_mod_ftilde_16(&buf);
        let mut expected = [Int::<2>::ZERO; REDUCED_LEN_16];
        expected[0] = Int::<2>::ONE;
        expected[2] = Int::<2>::ONE;
        expected[3] = Int::<2>::ONE;
        expected[5] = Int::<2>::ONE;
        assert_eq!(reduced, expected);
    }

    #[test]
    fn reduce_mod_ftilde_folds_x17_correctly() {
        // X^17 ≡ X · X^16 ≡ X^6 + X^4 + X^3 + X.
        let buf = monomial_unreduced::<2>(1, 17);
        let reduced = reduce_mod_ftilde_16(&buf);
        let mut expected = [Int::<2>::ZERO; REDUCED_LEN_16];
        expected[1] = Int::<2>::ONE;
        expected[3] = Int::<2>::ONE;
        expected[4] = Int::<2>::ONE;
        expected[6] = Int::<2>::ONE;
        assert_eq!(reduced, expected);
    }

    #[test]
    fn schoolbook_mul_of_monomials() {
        // X^7 · X^8 = X^15 (length-31 buffer, no reduction needed).
        let a = monomial::<2>(1, 7);
        let b = monomial::<2>(1, 8);
        let prod = schoolbook_mul_16(&a, &b);
        let mut expected = [Int::<2>::ZERO; PRODUCT_LEN_16];
        expected[15] = Int::<2>::ONE;
        assert_eq!(prod, expected);
    }

    #[test]
    fn mul_mod_ftilde_x8_times_x8_is_x16_reduced() {
        // X^8 · X^8 = X^16 ≡ X^5 + X^3 + X^2 + 1.
        let a = monomial::<2>(1, 8);
        let prod = mul_mod_ftilde_16(&a, &a);
        let mut expected = [Int::<2>::ZERO; REDUCED_LEN_16];
        expected[0] = Int::<2>::ONE;
        expected[2] = Int::<2>::ONE;
        expected[3] = Int::<2>::ONE;
        expected[5] = Int::<2>::ONE;
        assert_eq!(prod, expected);
    }

    #[test]
    fn lift_then_reduce_recovers_gf16() {
        for raw in [0u16, 1, 0xffff, 0xa5a5, 0x8000, 0x1234] {
            let g = Gf2_16(raw);
            let lifted: [Int<2>; REDUCED_LEN_16] = lift_gf16(g);
            let back = reduce_to_gf16(&lifted);
            assert_eq!(back, g, "round-trip failed for {raw:#06x}");
        }
    }

    /// Core soundness sanity check: lifted multiplication, reduced mod 2,
    /// agrees with `Gf2_16::mul`.
    #[test]
    fn lifted_mul_commutes_with_gf16_mul() {
        let cases = [
            (Gf2_16(0x1), Gf2_16(0x1)),
            (Gf2_16(0x2), Gf2_16(0x8000)),
            (Gf2_16(0xcafe), Gf2_16(0xdead)),
            (Gf2_16(0xffff), Gf2_16(0xffff)),
            (Gf2_16(0xa5a5), Gf2_16(0x5a5a)),
        ];
        for (a, b) in cases {
            let la: [Int<3>; REDUCED_LEN_16] = lift_gf16(a);
            let lb: [Int<3>; REDUCED_LEN_16] = lift_gf16(b);
            let lifted_prod = mul_mod_ftilde_16(&la, &lb);
            let reduced_prod = reduce_to_gf16(&lifted_prod);
            assert_eq!(reduced_prod, a.mul(b), "for a={a:?}, b={b:?}");
        }
    }

    #[test]
    fn lifted_add_commutes_with_gf16_add() {
        let a = Gf2_16(0xcafe);
        let b = Gf2_16(0xdead);
        let la: [Int<2>; REDUCED_LEN_16] = lift_gf16(a);
        let lb: [Int<2>; REDUCED_LEN_16] = lift_gf16(b);
        let lifted_sum = add_in_z_16(&la, &lb);
        assert_eq!(reduce_to_gf16(&lifted_sum), a.add(b));
    }

    #[test]
    fn mul_by_lifted_gf16_agrees_with_gf16_mul() {
        let cases = [
            (Gf2_16(0x1), Gf2_16(0x1)),
            (Gf2_16(0x2), Gf2_16(0x8000)),
            (Gf2_16(0xcafe), Gf2_16(0xdead)),
            (Gf2_16(0xffff), Gf2_16(0xffff)),
            (Gf2_16(0x55aa), Gf2_16(0xaa55)),
        ];
        for (a, b) in cases {
            let la: [Int<3>; REDUCED_LEN_16] = lift_gf16(a);
            let via_lifted_twiddle = mul_by_lifted_gf16(&la, b);
            assert_eq!(
                reduce_to_gf16(&via_lifted_twiddle),
                a.mul(b),
                "for a={a:?}, b={b:?}"
            );
            let lb: [Int<3>; REDUCED_LEN_16] = lift_gf16(b);
            let via_schoolbook = mul_mod_ftilde_16(&la, &lb);
            assert_eq!(via_lifted_twiddle, via_schoolbook, "integer-level disagreement");
        }
    }

    #[test]
    fn mul_by_lifted_gf16_special_values() {
        let mut x = [Int::<2>::ZERO; REDUCED_LEN_16];
        x[0] = Int::<2>::from(7);
        x[3] = Int::<2>::from(-3);
        x[15] = Int::<2>::from(2);
        assert_eq!(
            mul_by_lifted_gf16(&x, Gf2_16::ZERO),
            [Int::<2>::ZERO; REDUCED_LEN_16]
        );
        assert_eq!(mul_by_lifted_gf16(&x, Gf2_16::ONE), x);
    }
}
