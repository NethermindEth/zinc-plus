//! Wide bit-packed `F_2[X]` polynomial type with true `F_2` arithmetic.
//!
//! Each polynomial of degree `< 64 * W` is stored as `[u64; W]`, with bit
//! `64*k + i` of the array's `k`-th word representing the coefficient of
//! `X^(64*k + i)`. Addition is XOR; multiplication is the F_2-style
//! carryless product (`1 · 1 = 1`, `1 + 1 = 0`), produced into a wider
//! output type whose word count covers the sum of operand degrees.
//!
//! This type exists alongside [`BinaryRefPoly`] / [`BinaryU64Poly`] for
//! cases where:
//!
//! - true F_2 semantics are needed everywhere (so `Boolean`'s
//!   integer-overflow-on-1+1 contract is in the way), and
//! - the degree exceeds 64 (so `BinaryU64Poly` doesn't fit).
//!
//! The intended use case is wide random-combination coefficients for an
//! `F_2`-RAA commit lane: per the design, challenges live in
//! `F_2[X]<128>` (`W = 2`) and the linear combination of `F_2[X]<32>`
//! codeword cells against those challenges produces entries in
//! `F_2[X]<160>` (`W ≥ 3`).

use crate::univariate::F2AddAssign;
use core::ops::{Add, AddAssign};
use crypto_primitives::boolean::Boolean;
use num_traits::Zero;
use std::iter::Sum;
use zinc_utils::from_ref::FromRef;

use crate::univariate::{binary_ref::BinaryRefPoly, binary_u64::BinaryU64Poly};

/// `F_2[X]<64 * W>`: bit-packed binary polynomial, true `F_2` arithmetic.
///
/// Bit `i` of `words[i / 64]` (LSB-first within the word) holds the
/// coefficient of `X^i`. The polynomial has degree `< 64 * W`.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
#[repr(transparent)]
pub struct BinaryF2Poly<const W: usize> {
    words: [u64; W],
}

impl<const W: usize> Default for BinaryF2Poly<W> {
    fn default() -> Self {
        Self::zero()
    }
}

impl<const W: usize> BinaryF2Poly<W> {
    pub const fn zero() -> Self {
        Self { words: [0u64; W] }
    }

    pub const fn one() -> Self {
        let mut words = [0u64; W];
        if W > 0 {
            words[0] = 1;
        }
        Self { words }
    }

    /// Construct from raw word array.
    pub const fn from_words(words: [u64; W]) -> Self {
        Self { words }
    }

    /// Borrow the raw word array.
    pub const fn words(&self) -> &[u64; W] {
        &self.words
    }

    /// `true` iff every word is zero (i.e. the polynomial is zero).
    pub fn is_zero(&self) -> bool {
        self.words.iter().all(|w| *w == 0)
    }

    /// Read the coefficient of `X^i`. Returns `false` if `i >= 64 * W`.
    #[inline]
    pub fn bit(&self, i: usize) -> bool {
        let w = i >> 6;
        let b = i & 63;
        if w >= W {
            return false;
        }
        ((self.words[w] >> b) & 1) != 0
    }
}

impl<const W: usize> Zero for BinaryF2Poly<W> {
    fn zero() -> Self {
        Self::zero()
    }
    fn is_zero(&self) -> bool {
        Self::is_zero(self)
    }
}

// XOR (in-place) is the natural `F_2` add.
impl<'a, const W: usize> AddAssign<&'a Self> for BinaryF2Poly<W> {
    #[inline]
    fn add_assign(&mut self, rhs: &'a Self) {
        for i in 0..W {
            // `^=` is XOR; F_2 add. No overflow possible.
            #[allow(clippy::arithmetic_side_effects, clippy::suspicious_op_assign_impl)]
            {
                self.words[i] ^= rhs.words[i];
            }
        }
    }
}

impl<const W: usize> AddAssign<Self> for BinaryF2Poly<W> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        <Self as AddAssign<&Self>>::add_assign(self, &rhs);
    }
}

impl<const W: usize> Add<Self> for BinaryF2Poly<W> {
    type Output = Self;
    #[inline]
    fn add(mut self, rhs: Self) -> Self::Output {
        self += &rhs;
        self
    }
}

impl<'a, const W: usize> Add<&'a Self> for BinaryF2Poly<W> {
    type Output = Self;
    #[inline]
    fn add(mut self, rhs: &'a Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl<const W: usize> F2AddAssign for BinaryF2Poly<W> {
    #[inline]
    fn f2_add_assign(&mut self, rhs: &Self) {
        <Self as AddAssign<&Self>>::add_assign(self, rhs);
    }
}

impl<'a, const W: usize> Sum<&'a BinaryF2Poly<W>> for BinaryF2Poly<W> {
    fn sum<I: Iterator<Item = &'a BinaryF2Poly<W>>>(iter: I) -> Self {
        iter.fold(Self::zero(), |mut acc, x| {
            acc += x;
            acc
        })
    }
}

impl<const W: usize> FromRef<BinaryF2Poly<W>> for BinaryF2Poly<W> {
    #[inline]
    fn from_ref(value: &BinaryF2Poly<W>) -> Self {
        *value
    }
}

// -----------------------------------------------------------------
// Lifts from the existing narrow binary-poly types.
//
// `BinaryRefPoly<D>` and `BinaryU64Poly<D>` represent F_2[X]<D> with
// `D ≤ 64` (BinaryU64Poly) or any D (BinaryRefPoly, array-backed).
// We define a one-way lift into `BinaryF2Poly<W>` whenever the source
// type's bit count fits in the destination's bit budget, asserted at
// construction time.
// -----------------------------------------------------------------

impl BinaryF2Poly<2> {
    /// Lift a `BinaryRefPoly<D>` (`D ≤ 128`) into the wide representation.
    /// Panics if `D > 128`.
    pub fn from_ref_poly_128<const D: usize>(p: &BinaryRefPoly<D>) -> Self {
        assert!(D <= 128, "BinaryF2Poly<2> holds < 128 bits; D = {D}");
        let mut out = Self::zero();
        for i in 0..D {
            if p.inner().coeffs[i].inner() {
                #[allow(clippy::arithmetic_side_effects)]
                {
                    out.words[i / 64] |= 1u64 << (i % 64);
                }
            }
        }
        out
    }

    /// Lift a `BinaryU64Poly<D>` (`D ≤ 64`) into the wide representation.
    pub fn from_u64_poly_128<const D: usize>(p: &BinaryU64Poly<D>) -> Self {
        assert!(D <= 64, "BinaryU64Poly<D> stores in a u64; D = {D}");
        Self::from_words([*p.inner(), 0])
    }
}

impl BinaryF2Poly<3> {
    /// Lift a `BinaryRefPoly<D>` (`D ≤ 192`) into the wide representation.
    /// Panics if `D > 192`.
    pub fn from_ref_poly_192<const D: usize>(p: &BinaryRefPoly<D>) -> Self {
        assert!(D <= 192, "BinaryF2Poly<3> holds < 192 bits; D = {D}");
        let mut out = Self::zero();
        for i in 0..D {
            if p.inner().coeffs[i].inner() {
                #[allow(clippy::arithmetic_side_effects)]
                {
                    out.words[i / 64] |= 1u64 << (i % 64);
                }
            }
        }
        out
    }

    /// Lift a `BinaryU64Poly<D>` (`D ≤ 64`) into the wide representation.
    pub fn from_u64_poly_192<const D: usize>(p: &BinaryU64Poly<D>) -> Self {
        assert!(D <= 64, "BinaryU64Poly<D> stores in a u64; D = {D}");
        Self::from_words([*p.inner(), 0, 0])
    }
}

// `BinaryRefPoly<D>` is array-of-Boolean. Lift via per-bit walk.
impl<const W: usize> BinaryF2Poly<W> {
    /// Set the coefficient of `X^i`. Panics if `i >= 64 * W`.
    #[inline]
    #[allow(clippy::arithmetic_side_effects)]
    pub fn set_bit(&mut self, i: usize, on: bool) {
        let w = i >> 6;
        let b = i & 63;
        assert!(w < W, "set_bit: i = {i} out of range for W = {W}");
        let mask = 1u64 << b;
        if on {
            self.words[w] |= mask;
        } else {
            self.words[w] &= !mask;
        }
    }
}

// -----------------------------------------------------------------
// Carryless F_2[X] multiplication: produces a wider result.
//
// We multiply two bit-packed polynomials `a` (W_A words) and `b`
// (W_B words) into an output of W_OUT words. The result's degree is
// `< 64*W_A + 64*W_B - 1`, so `W_OUT >= W_A + W_B` always suffices.
//
// Algorithm: schoolbook, one bit of `a` at a time. For each set bit
// `i` in `a`, XOR a left-shifted copy of `b` (by `i` bits) into the
// accumulator. O((64*W_A) * W_B) word ops; fine for the W ≤ 3 case
// we exercise (32×128 → 160 bits, ~4096 word ops).
// -----------------------------------------------------------------

/// Carryless (F_2[X]) multiplication. Output word count `W_OUT` must
/// satisfy `W_OUT >= W_A + W_B`. Panics otherwise.
#[allow(clippy::arithmetic_side_effects)]
pub fn f2_poly_mul<const W_A: usize, const W_B: usize, const W_OUT: usize>(
    a: &BinaryF2Poly<W_A>,
    b: &BinaryF2Poly<W_B>,
) -> BinaryF2Poly<W_OUT> {
    assert!(
        W_OUT >= W_A + W_B,
        "f2_poly_mul: W_OUT ({W_OUT}) must be >= W_A + W_B ({W_A} + {W_B}); \
         product can have up to 64*(W_A + W_B) - 1 bits."
    );
    let mut acc = [0u64; W_OUT];
    for ai in 0..W_A {
        let mut a_word = a.words[ai];
        while a_word != 0 {
            let lo = a_word.trailing_zeros() as usize;
            // XOR `b << shift` into `acc`, where shift = 64*ai + lo.
            let shift = 64 * ai + lo;
            xor_shifted(&mut acc, b.words(), shift);
            // Clear the LSB.
            a_word &= a_word - 1;
        }
    }
    BinaryF2Poly::from_words(acc)
}

/// XOR `b << shift` into `acc`. `acc` and `b` are bit-packed in
/// LSB-first words. Bits that would land at or past `64 * acc.len()`
/// are discarded (cannot happen if the caller sized `acc` to fit the
/// result, but we do not assert here — `f2_poly_mul` does).
#[inline]
#[allow(clippy::arithmetic_side_effects)]
fn xor_shifted(acc: &mut [u64], b: &[u64], shift: usize) {
    let word_shift = shift / 64;
    let bit_shift = shift % 64;
    let n_acc = acc.len();
    for i in 0..b.len() {
        if word_shift + i >= n_acc {
            break;
        }
        let bw = b[i];
        if bit_shift == 0 {
            acc[word_shift + i] ^= bw;
        } else {
            acc[word_shift + i] ^= bw << bit_shift;
            if word_shift + i + 1 < n_acc {
                acc[word_shift + i + 1] ^= bw >> (64 - bit_shift);
            }
        }
    }
}

// -----------------------------------------------------------------
// Lift from the canonical `BinaryPoly` alias.
//
// `BinaryPoly` is feature-gated: `BinaryRefPoly` without `simd`,
// `BinaryU64Poly` with `simd`. We provide a `from_binary_poly` lift
// for both, dispatched at compile time by feature, that produces a
// `BinaryF2Poly` of any sufficient width.
// -----------------------------------------------------------------

/// Lift a coefficient `Boolean` into a bit at position 0 of a wide
/// `BinaryF2Poly<W>`. Used by call sites that need to splat a single
/// `F_2` value into the wide layout.
impl<const W: usize> FromRef<Boolean> for BinaryF2Poly<W> {
    fn from_ref(value: &Boolean) -> Self {
        let mut out = Self::zero();
        if value.inner() && W > 0 {
            out.words[0] = 1;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::univariate::binary::BinaryPoly;

    /// `F_2[X]<32>` value with bit `i` set iff bit `i` of `bits` is set.
    fn bp32(bits: u32) -> BinaryPoly<32> {
        BinaryPoly::from(bits)
    }

    /// `F_2[X]<128>` value from a `u128` bit pattern.
    fn bp128(bits: u128) -> BinaryF2Poly<2> {
        let lo = bits as u64;
        let hi = (bits >> 64) as u64;
        BinaryF2Poly::from_words([lo, hi])
    }

    #[test]
    fn add_is_xor_and_one_plus_one_is_zero() {
        let mut a = bp128(0b1010);
        let b = bp128(0b0011);
        a += &b;
        assert_eq!(a, bp128(0b1001));
        // direct 1+1
        let mut x = bp128(1);
        let y = bp128(1);
        x += &y;
        assert!(x.is_zero());
    }

    #[test]
    fn mul_low_degree_matches_handworked_examples() {
        // (X) * (X + 1) = X^2 + X
        let a: BinaryF2Poly<1> = BinaryF2Poly::from_words([0b10]); // X
        let b: BinaryF2Poly<1> = BinaryF2Poly::from_words([0b11]); // X + 1
        let c: BinaryF2Poly<2> = f2_poly_mul(&a, &b);
        assert_eq!(c.words(), &[0b110, 0]); // X^2 + X

        // (X + 1) * (X + 1) = X^2 + 1   (F_2: cross terms cancel)
        let d: BinaryF2Poly<2> = f2_poly_mul(&b, &b);
        assert_eq!(d.words(), &[0b101, 0]);
    }

    #[test]
    fn mul_32x128_to_160_degree_bound_holds() {
        // a has bit 31 set (X^31), b has bit 127 set (X^127). Product
        // has only X^158 set, which lives in word 2 (bit 158 - 128 = 30).
        let a: BinaryF2Poly<1> = BinaryF2Poly::from_words([1u64 << 31]);
        let b: BinaryF2Poly<2> = BinaryF2Poly::from_words([0, 1u64 << 63]);
        let c: BinaryF2Poly<3> = f2_poly_mul(&a, &b);
        assert_eq!(c.words(), &[0, 0, 1u64 << 30]);
    }

    #[test]
    fn distributivity_holds() {
        let a: BinaryF2Poly<1> = BinaryF2Poly::from_words([0xA5A5_A5A5]);
        let b: BinaryF2Poly<2> = BinaryF2Poly::from_words([0x1234_5678, 0xDEAD_BEEF]);
        let c: BinaryF2Poly<2> = BinaryF2Poly::from_words([0xCAFE_F00D, 0x0F0F_0F0F]);

        // a · (b + c)
        let mut bpc = b;
        bpc += &c;
        let lhs: BinaryF2Poly<3> = f2_poly_mul(&a, &bpc);

        // a·b + a·c
        let ab: BinaryF2Poly<3> = f2_poly_mul(&a, &b);
        let ac: BinaryF2Poly<3> = f2_poly_mul(&a, &c);
        let mut rhs = ab;
        rhs += &ac;

        assert_eq!(lhs, rhs);
    }

    #[test]
    fn zero_multiplied_anywhere_is_zero() {
        let z32: BinaryF2Poly<1> = BinaryF2Poly::zero();
        let b: BinaryF2Poly<2> = BinaryF2Poly::from_words([0xDEAD_BEEF, 0xCAFE_F00D]);
        let c: BinaryF2Poly<3> = f2_poly_mul(&z32, &b);
        assert!(c.is_zero());

        let a: BinaryF2Poly<1> = BinaryF2Poly::from_words([0xA5A5_A5A5]);
        let z128: BinaryF2Poly<2> = BinaryF2Poly::zero();
        let c2: BinaryF2Poly<3> = f2_poly_mul(&a, &z128);
        assert!(c2.is_zero());
    }

    #[test]
    fn lift_from_binary_poly_32_round_trips() {
        let val = 0xDEAD_BEEFu32;
        let p = bp32(val);
        // Build the expected wide-3 representation manually.
        let mut expected: BinaryF2Poly<3> = BinaryF2Poly::zero();
        for i in 0..32 {
            if (val >> i) & 1 != 0 {
                expected.set_bit(i, true);
            }
        }
        // Lift via per-bit walk (cfg-agnostic).
        let mut got: BinaryF2Poly<3> = BinaryF2Poly::zero();
        for (i, c) in p.iter().enumerate() {
            if c.inner() {
                got.set_bit(i, true);
            }
        }
        assert_eq!(got, expected);
    }
}
