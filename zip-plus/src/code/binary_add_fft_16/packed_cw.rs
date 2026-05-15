//! `PackedI64Poly16` — `DensePolynomial<i64, 16>` with a packed
//! Transcribable encoding that uses `BITS_PER_COEF` bits per coefficient
//! instead of a full `i64` (64 bits).
//!
//! Mirrors `binary_add_fft_8/packed_cw.rs` but for D = 16 with an
//! `i64` backing (the D=8 module's i32 backing was found to overflow
//! at codeword_len ≥ 256, and D=16 amplifies further so i64 is used
//! by default).

use core::{
    fmt::{self, Debug, Display},
    iter::{Product, Sum},
    ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign},
};

use crypto_primitives::{FromWithConfig, Semiring};
use num_traits::{CheckedAdd, CheckedMul, CheckedSub, ConstOne, ConstZero, One, Zero};
use zinc_poly::{ConstCoeffBitWidth, Polynomial, univariate::{binary::BinaryPoly, dense::DensePolynomial}};
use zinc_transcript::traits::{ConstTranscribable, GenTranscribable};
use zinc_utils::{from_ref::FromRef, mul_by_scalar::MulByScalar, named::Named};

use super::basis::Gf2_16;
use super::ext_field::{Bit4Poly16, Gf2_16Ext};
use super::ring_ops::FftRingElement16;

/// Bit-packed wrapper around `DensePolynomial<i64, 16>`. The
/// Transcribable encoding writes each coefficient in `BITS_PER_COEF`
/// bits (signed, two's complement), packed contiguously into
/// `(16 * BITS_PER_COEF) / 8` bytes total.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(transparent)]
pub struct PackedI64Poly16<const BITS_PER_COEF: u32>(pub DensePolynomial<i64, 16>);

impl<const B: u32> PackedI64Poly16<B> {
    pub const fn from_inner(inner: DensePolynomial<i64, 16>) -> Self {
        Self(inner)
    }
}

// --- Formatting ---

impl<const B: u32> Debug for PackedI64Poly16<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PackedI64Poly16<{B}>(")?;
        Debug::fmt(&self.0, f)?;
        write!(f, ")")
    }
}
impl<const B: u32> Display for PackedI64Poly16<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PackedI64Poly16<{B}>(...)")
    }
}

// --- Zero / One ---

impl<const B: u32> Zero for PackedI64Poly16<B> {
    fn zero() -> Self { Self(DensePolynomial::zero()) }
    fn is_zero(&self) -> bool { self.0.is_zero() }
}
impl<const B: u32> ConstZero for PackedI64Poly16<B> {
    const ZERO: Self = Self(DensePolynomial { coeffs: [0i64; 16] });
}
impl<const B: u32> One for PackedI64Poly16<B> {
    fn one() -> Self { Self(DensePolynomial::one()) }
}
impl<const B: u32> ConstOne for PackedI64Poly16<B> {
    const ONE: Self = {
        let mut c = [0i64; 16];
        c[0] = 1;
        Self(DensePolynomial { coeffs: c })
    };
}

// --- Arithmetic, delegated to inner ---

macro_rules! delegate_binop {
    ($trait:ident, $method:ident) => {
        impl<const B: u32> $trait for PackedI64Poly16<B> {
            type Output = Self;
            fn $method(self, rhs: Self) -> Self { Self(self.0.$method(rhs.0)) }
        }
        impl<const B: u32> $trait<&Self> for PackedI64Poly16<B> {
            type Output = Self;
            fn $method(self, rhs: &Self) -> Self { Self(self.0.$method(&rhs.0)) }
        }
    };
}
macro_rules! delegate_assign {
    ($trait:ident, $method:ident) => {
        impl<const B: u32> $trait for PackedI64Poly16<B> {
            fn $method(&mut self, rhs: Self) { self.0.$method(rhs.0); }
        }
        impl<const B: u32> $trait<&Self> for PackedI64Poly16<B> {
            fn $method(&mut self, rhs: &Self) { self.0.$method(&rhs.0); }
        }
    };
}
delegate_binop!(Add, add);
delegate_binop!(Sub, sub);
delegate_binop!(Mul, mul);
delegate_assign!(AddAssign, add_assign);
delegate_assign!(SubAssign, sub_assign);
delegate_assign!(MulAssign, mul_assign);

impl<const B: u32> CheckedAdd for PackedI64Poly16<B> {
    fn checked_add(&self, rhs: &Self) -> Option<Self> {
        self.0.checked_add(&rhs.0).map(Self)
    }
}
impl<const B: u32> CheckedSub for PackedI64Poly16<B> {
    fn checked_sub(&self, rhs: &Self) -> Option<Self> {
        self.0.checked_sub(&rhs.0).map(Self)
    }
}
impl<const B: u32> CheckedMul for PackedI64Poly16<B> {
    fn checked_mul(&self, rhs: &Self) -> Option<Self> {
        self.0.checked_mul(&rhs.0).map(Self)
    }
}

impl<const B: u32> Sum for PackedI64Poly16<B> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        Self(iter.map(|x| x.0).sum())
    }
}
impl<'a, const B: u32> Sum<&'a Self> for PackedI64Poly16<B> {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        Self(iter.map(|x| &x.0).sum())
    }
}
impl<const B: u32> Product for PackedI64Poly16<B> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        Self(iter.map(|x| x.0).product())
    }
}
impl<'a, const B: u32> Product<&'a Self> for PackedI64Poly16<B> {
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        Self(iter.map(|x| &x.0).product())
    }
}

// --- Semiring / Ring ---

impl<const B: u32> Semiring for PackedI64Poly16<B> {}

// --- ConstCoeffBitWidth ---

impl<const B: u32> ConstCoeffBitWidth for PackedI64Poly16<B> {
    const COEFF_BIT_WIDTH: usize = B as usize;
}

// --- Polynomial<Self> with DEGREE_BOUND = 0 ---

impl<const B: u32> Polynomial<Self> for PackedI64Poly16<B> {
    const DEGREE_BOUND: usize = 0;
}

// --- Conversions ---

impl<const B1: u32, const B2: u32> FromRef<PackedI64Poly16<B1>> for PackedI64Poly16<B2> {
    fn from_ref(value: &PackedI64Poly16<B1>) -> Self {
        Self(value.0)
    }
}

impl<const B: u32> FromRef<BinaryPoly<16>> for PackedI64Poly16<B> {
    fn from_ref(value: &BinaryPoly<16>) -> Self {
        Self(<DensePolynomial<i64, 16> as FromRef<BinaryPoly<16>>>::from_ref(value))
    }
}

impl<const B: u32> Named for PackedI64Poly16<B> {
    fn type_name() -> String {
        format!("Packed16<i64, {B}b>")
    }
}

// --- MulByScalar<&Bit4Poly16> ---

impl<const B: u32> MulByScalar<&Bit4Poly16> for PackedI64Poly16<B> {
    fn mul_by_scalar<const CHECK: bool>(&self, rhs: &Bit4Poly16) -> Option<Self> {
        self.0.mul_by_scalar::<CHECK>(rhs).map(Self)
    }
}

// --- FromWithConfig<&PackedI64Poly16> for the verifier field ---

impl<const P: u64, const B: u32> FromWithConfig<&PackedI64Poly16<B>> for Gf2_16Ext<P> {
    fn from_with_cfg(value: &PackedI64Poly16<B>, cfg: &Self::Config) -> Self {
        <Self as FromWithConfig<&DensePolynomial<i64, 16>>>::from_with_cfg(&value.0, cfg)
    }
}

// --- FftRingElement16 ---

impl<const B: u32> FftRingElement16 for PackedI64Poly16<B> {
    fn fft_zero() -> Self {
        Self(<DensePolynomial<i64, 16> as FftRingElement16>::fft_zero())
    }
    fn fft_add(&self, other: &Self) -> Self {
        Self(self.0.fft_add(&other.0))
    }
    fn fft_mul_lifted_gf16(&self, twiddle: Gf2_16) -> Self {
        Self(self.0.fft_mul_lifted_gf16(twiddle))
    }

    type UnreducedAcc = <DensePolynomial<i64, 16> as FftRingElement16>::UnreducedAcc;
    fn fft_unreduced_zero() -> Self::UnreducedAcc {
        <DensePolynomial<i64, 16> as FftRingElement16>::fft_unreduced_zero()
    }
    fn fft_acc_mul_lifted_gf16(input: &Self, twiddle: Gf2_16, acc: &mut Self::UnreducedAcc) {
        <DensePolynomial<i64, 16> as FftRingElement16>::fft_acc_mul_lifted_gf16(
            &input.0, twiddle, acc,
        );
    }
    fn fft_finalize_acc(acc: Self::UnreducedAcc) -> Self {
        Self(<DensePolynomial<i64, 16> as FftRingElement16>::fft_finalize_acc(acc))
    }
}

// --- Packed Transcribable encoding ---

#[inline]
const fn num_bytes_for(b: u32) -> usize {
    let bits = 16 * b as usize;
    bits.div_ceil(8)
}

fn write_packed<const B: u32>(coeffs: &[i64; 16], out: &mut [u8]) {
    debug_assert_eq!(out.len(), num_bytes_for(B));
    out.fill(0);
    let mask: u64 = if B >= 64 { u64::MAX } else { (1u64 << B) - 1 };
    let mut bit_pos: usize = 0;
    for &c in coeffs {
        let v: u64 = (c as u64) & mask;
        let lo_byte = bit_pos / 8;
        let lo_off = bit_pos % 8;
        // Use u128 to hold up to 64-bit value shifted by up to 7 bits.
        let shifted: u128 = (v as u128) << lo_off;
        let span_bytes = ((B as usize + lo_off) + 7) / 8;
        for k in 0..span_bytes {
            out[lo_byte + k] |= (shifted >> (8 * k)) as u8;
        }
        bit_pos += B as usize;
    }
}

fn read_packed<const B: u32>(input: &[u8]) -> [i64; 16] {
    debug_assert_eq!(input.len(), num_bytes_for(B));
    let mask: u64 = if B >= 64 { u64::MAX } else { (1u64 << B) - 1 };
    let sign_bit: u64 = if B >= 64 { 0 } else { 1u64 << (B - 1) };
    let sign_ext: u64 = !mask;
    let mut out = [0i64; 16];
    let mut bit_pos: usize = 0;
    for slot in out.iter_mut() {
        let lo_byte = bit_pos / 8;
        let lo_off = bit_pos % 8;
        let span_bytes = ((B as usize + lo_off) + 7) / 8;
        // Use u128 to hold up to 64-bit value shifted by up to 7 bits.
        let mut acc: u128 = 0;
        for k in 0..span_bytes {
            acc |= (input[lo_byte + k] as u128) << (8 * k);
        }
        let v_low: u64 = ((acc >> lo_off) as u64) & mask;
        let v = if v_low & sign_bit != 0 { v_low | sign_ext } else { v_low };
        *slot = v as i64;
        bit_pos += B as usize;
    }
    out
}

impl<const B: u32> GenTranscribable for PackedI64Poly16<B> {
    fn read_transcription_bytes_exact(bytes: &[u8]) -> Self {
        let coeffs = read_packed::<B>(bytes);
        Self(DensePolynomial { coeffs })
    }
    fn write_transcription_bytes_exact(&self, buf: &mut [u8]) {
        write_packed::<B>(&self.0.coeffs, buf);
    }
}

impl<const B: u32> ConstTranscribable for PackedI64Poly16<B> {
    const NUM_BYTES: usize = num_bytes_for(B);
    const NUM_BITS: usize = 16 * (B as usize);
}

// Tests
#[cfg(test)]
mod tests {
    use super::*;
    use rand::{prelude::*, rngs::StdRng};

    fn round_trip<const B: u32>(coeffs: [i64; 16]) {
        let p = PackedI64Poly16::<B>(DensePolynomial { coeffs });
        let mut buf = vec![0u8; num_bytes_for(B)];
        p.write_transcription_bytes_exact(&mut buf);
        let p2: PackedI64Poly16<B> = PackedI64Poly16::<B>::read_transcription_bytes_exact(&buf);
        assert_eq!(p, p2, "round-trip mismatch with B={B}");
    }

    #[test]
    fn round_trip_24bit() {
        let mut rng = StdRng::seed_from_u64(0x1234);
        let mut coeffs = [0i64; 16];
        for c in coeffs.iter_mut() {
            *c = rng.random_range(-(1 << 23)..(1 << 23));
        }
        round_trip::<24>(coeffs);
    }

    #[test]
    fn round_trip_40bit() {
        let mut rng = StdRng::seed_from_u64(0x4321);
        let mut coeffs = [0i64; 16];
        for c in coeffs.iter_mut() {
            *c = rng.random_range(-(1i64 << 39)..(1i64 << 39));
        }
        round_trip::<40>(coeffs);
    }

    #[test]
    fn round_trip_32bit() {
        let mut coeffs = [0i64; 16];
        for (i, c) in coeffs.iter_mut().enumerate() {
            *c = (i as i64) - 8;
        }
        round_trip::<32>(coeffs);
    }

    #[test]
    fn boundary_values() {
        // Only test B < 64 since `1i64 << 63` overflows in debug mode.
        for &b in &[24u32, 40] {
            let mut coeffs = [0i64; 16];
            let max: i64 = (1i64 << (b - 1)) - 1;
            let min: i64 = -(1i64 << (b - 1));
            coeffs[0] = max;
            coeffs[1] = min;
            coeffs[2] = 0;
            coeffs[3] = -1;
            match b {
                24 => round_trip::<24>(coeffs),
                40 => round_trip::<40>(coeffs),
                _ => unreachable!(),
            }
        }
    }
}
