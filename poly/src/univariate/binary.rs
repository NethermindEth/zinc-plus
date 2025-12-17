use crate::{
    ConstCoeffBitWidth, EvaluatablePolynomial, EvaluationError, Polynomial,
    univariate::dense::DensePolynomial,
};
use crypto_primitives::{Semiring, semiring::boolean::Boolean};
use num_traits::{CheckedAdd, CheckedMul, CheckedSub, ConstZero, One, Zero};
use rand::{distr::StandardUniform, prelude::*};
use std::{
    fmt::Display,
    hash::Hash,
    iter::{Product, Sum},
    ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign},
};
use zinc_transcript::traits::ConstTranscribable;
use zinc_utils::{add, from_ref::FromRef, mul, named::Named};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(transparent)]
pub struct BinaryPoly<const DEGREE_PLUS_ONE: usize>(DensePolynomial<Boolean, DEGREE_PLUS_ONE>);

impl<const DEGREE_PLUS_ONE: usize> AsRef<DensePolynomial<Boolean, DEGREE_PLUS_ONE>>
    for BinaryPoly<DEGREE_PLUS_ONE>
{
    #[inline(always)]
    fn as_ref(&self) -> &DensePolynomial<Boolean, DEGREE_PLUS_ONE> {
        &self.0
    }
}

impl<const DEGREE_PLUS_ONE: usize> From<DensePolynomial<Boolean, DEGREE_PLUS_ONE>>
    for BinaryPoly<DEGREE_PLUS_ONE>
{
    #[inline(always)]
    fn from(dense_polynomial: DensePolynomial<Boolean, DEGREE_PLUS_ONE>) -> Self {
        Self(dense_polynomial)
    }
}

impl<const DEGREE_PLUS_ONE: usize> From<BinaryPoly<DEGREE_PLUS_ONE>>
    for DensePolynomial<Boolean, DEGREE_PLUS_ONE>
{
    #[inline(always)]
    fn from(binary_poly: BinaryPoly<DEGREE_PLUS_ONE>) -> Self {
        binary_poly.0
    }
}

impl<const DEGREE_PLUS_ONE: usize> BinaryPoly<DEGREE_PLUS_ONE> {
    /// Create a new polynomial with the given coefficients.
    /// If the input has fewer than N+1 coefficients, the remaining slots will
    /// be filled with zeros. If the input has more than N+1 coefficients,
    /// it will panic.
    #[inline(always)]
    pub fn new(coeffs: impl AsRef<[Boolean]>) -> Self {
        Self(DensePolynomial::new(coeffs))
    }
}

impl<const DEGREE_PLUS_ONE: usize> BinaryPoly<DEGREE_PLUS_ONE> {
    /// Create a new polynomial with the given coefficients.
    /// If the input has fewer than N+1 coefficients, the remaining slots will
    /// be filled with zeros. If the input has more than N+1 coefficients,
    /// it will panic.
    #[inline(always)]
    pub fn new_padded(coeffs: impl AsRef<[Boolean]>) -> Self {
        Self(DensePolynomial::new_with_zero(coeffs, Boolean::ZERO))
    }
}

impl<const DEGREE_PLUS_ONE: usize> Display for BinaryPoly<DEGREE_PLUS_ONE> {
    #[inline(always)]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl<const DEGREE_PLUS_ONE: usize> Hash for BinaryPoly<DEGREE_PLUS_ONE> {
    #[inline(always)]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl<const DEGREE_PLUS_ONE: usize> Zero for BinaryPoly<DEGREE_PLUS_ONE> {
    #[inline(always)]
    fn zero() -> Self {
        Self(DensePolynomial::zero())
    }

    #[inline(always)]
    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}

impl<const DEGREE_PLUS_ONE: usize> One for BinaryPoly<DEGREE_PLUS_ONE> {
    #[inline(always)]
    fn one() -> Self {
        Self(DensePolynomial::one())
    }
}

impl<const DEGREE_PLUS_ONE: usize> Add for BinaryPoly<DEGREE_PLUS_ONE> {
    type Output = Self;

    #[allow(clippy::arithmetic_side_effects)]
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl<'a, const DEGREE_PLUS_ONE: usize> Add<&'a Self> for BinaryPoly<DEGREE_PLUS_ONE> {
    type Output = Self;

    #[allow(clippy::arithmetic_side_effects)]
    #[inline(always)]
    fn add(self, rhs: &'a Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl<const DEGREE_PLUS_ONE: usize> Sub for BinaryPoly<DEGREE_PLUS_ONE> {
    type Output = Self;

    #[allow(clippy::arithmetic_side_effects)]
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl<'a, const DEGREE_PLUS_ONE: usize> Sub<&'a Self> for BinaryPoly<DEGREE_PLUS_ONE> {
    type Output = Self;

    #[allow(clippy::arithmetic_side_effects)]
    #[inline(always)]
    fn sub(self, rhs: &'a Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl<const DEGREE_PLUS_ONE: usize> Mul for BinaryPoly<DEGREE_PLUS_ONE> {
    type Output = Self;

    #[allow(clippy::arithmetic_side_effects)]
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        Self(self.0 * rhs.0)
    }
}

impl<'a, const DEGREE_PLUS_ONE: usize> Mul<&'a Self> for BinaryPoly<DEGREE_PLUS_ONE> {
    type Output = Self;

    #[allow(clippy::arithmetic_side_effects)]
    #[inline(always)]
    fn mul(self, rhs: &'a Self) -> Self::Output {
        Self(self.0 * rhs.0)
    }
}

impl<const DEGREE_PLUS_ONE: usize> AddAssign for BinaryPoly<DEGREE_PLUS_ONE> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        self.0.add_assign(&rhs.0);
    }
}

impl<'a, const DEGREE_PLUS_ONE: usize> AddAssign<&'a Self> for BinaryPoly<DEGREE_PLUS_ONE> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: &'a Self) {
        self.0.add_assign(&rhs.0);
    }
}

impl<const DEGREE_PLUS_ONE: usize> SubAssign for BinaryPoly<DEGREE_PLUS_ONE> {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        self.0.sub_assign(&rhs.0);
    }
}

impl<'a, const DEGREE_PLUS_ONE: usize> SubAssign<&'a Self> for BinaryPoly<DEGREE_PLUS_ONE> {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: &'a Self) {
        self.0.sub_assign(&rhs.0);
    }
}

impl<const DEGREE_PLUS_ONE: usize> MulAssign for BinaryPoly<DEGREE_PLUS_ONE> {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        self.0.mul_assign(&rhs.0);
    }
}

impl<'a, const DEGREE_PLUS_ONE: usize> MulAssign<&'a Self> for BinaryPoly<DEGREE_PLUS_ONE> {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: &'a Self) {
        self.0.mul_assign(&rhs.0);
    }
}

impl<const DEGREE_PLUS_ONE: usize> CheckedAdd for BinaryPoly<DEGREE_PLUS_ONE> {
    #[inline(always)]
    fn checked_add(&self, other: &Self) -> Option<Self> {
        Some(Self(self.0.checked_add(&other.0)?))
    }
}

impl<const DEGREE_PLUS_ONE: usize> CheckedSub for BinaryPoly<DEGREE_PLUS_ONE> {
    #[inline(always)]
    fn checked_sub(&self, other: &Self) -> Option<Self> {
        Some(Self(self.0.checked_sub(&other.0)?))
    }
}

impl<const DEGREE_PLUS_ONE: usize> CheckedMul for BinaryPoly<DEGREE_PLUS_ONE> {
    #[inline(always)]
    fn checked_mul(&self, other: &Self) -> Option<Self> {
        Some(Self(self.0.checked_mul(&other.0)?))
    }
}

impl<const DEGREE_PLUS_ONE: usize> Sum for BinaryPoly<DEGREE_PLUS_ONE> {
    #[inline(always)]
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        Self(iter.map(|x| x.0).sum())
    }
}

impl<'a, const DEGREE_PLUS_ONE: usize> Sum<&'a Self> for BinaryPoly<DEGREE_PLUS_ONE> {
    #[inline(always)]
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        Self(iter.map(|x| &x.0).sum())
    }
}

impl<const DEGREE_PLUS_ONE: usize> Product for BinaryPoly<DEGREE_PLUS_ONE> {
    #[inline(always)]
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        Self(iter.map(|x| x.0).product())
    }
}

impl<'a, const DEGREE_PLUS_ONE: usize> Product<&'a Self> for BinaryPoly<DEGREE_PLUS_ONE> {
    #[inline(always)]
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        Self(iter.map(|x| &x.0).product())
    }
}

impl<const DEGREE_PLUS_ONE: usize> Semiring for BinaryPoly<DEGREE_PLUS_ONE> {}

impl<const DEGREE_PLUS_ONE: usize> Distribution<BinaryPoly<DEGREE_PLUS_ONE>> for StandardUniform {
    #[inline(always)]
    fn sample<Gen: Rng + ?Sized>(&self, rng: &mut Gen) -> BinaryPoly<DEGREE_PLUS_ONE> {
        let coeffs: [Boolean; DEGREE_PLUS_ONE] = rng.random();

        // I didn't manage to delegate this one to
        // `DensePolynomial::sample` because of unsatisfied
        // traits.

        BinaryPoly(DensePolynomial::new(coeffs))
    }
}

//
// Zip-specific traits
//
impl<const DEGREE_PLUS_ONE: usize> Polynomial<Boolean> for BinaryPoly<DEGREE_PLUS_ONE> {
    const DEGREE_BOUND: usize = DensePolynomial::<Boolean, DEGREE_PLUS_ONE>::DEGREE_BOUND;
}

impl<R: Clone + Zero + One + CheckedAdd + CheckedMul, const DEGREE_PLUS_ONE: usize>
    EvaluatablePolynomial<Boolean, R> for BinaryPoly<DEGREE_PLUS_ONE>
{
    type EvaluationPoint = R;

    fn evaluate_at_point(&self, point: &R) -> Result<R, EvaluationError> {
        if DEGREE_PLUS_ONE.is_one() {
            return Ok(R::zero());
        }

        let result = self.0.coeffs[1..]
            .iter()
            .fold(
                (
                    if self.0.coeffs[0].inner() {
                        R::one()
                    } else {
                        R::zero()
                    },
                    R::one(),
                ),
                |(mut acc, mut pow), coeff| {
                    pow = mul!(pow, point);

                    if coeff.inner() {
                        acc = add!(acc, &pow);
                    }

                    (acc, pow)
                },
            )
            .0;

        Ok(result)
    }
}

impl<const DEGREE_PLUS_ONE: usize> ConstCoeffBitWidth for BinaryPoly<DEGREE_PLUS_ONE> {
    const COEFF_BIT_WIDTH: usize = DensePolynomial::<Boolean, DEGREE_PLUS_ONE>::COEFF_BIT_WIDTH;
}

impl<const DEGREE_PLUS_ONE: usize> Named for BinaryPoly<DEGREE_PLUS_ONE> {
    fn type_name() -> String {
        format!("BPoly<{}>", Self::DEGREE_BOUND)
    }
}

impl<const DEGREE_PLUS_ONE: usize> ConstTranscribable for BinaryPoly<DEGREE_PLUS_ONE> {
    const NUM_BYTES: usize = DensePolynomial::<Boolean, DEGREE_PLUS_ONE>::NUM_BYTES;

    #[inline(always)]
    fn read_transcription_bytes(bytes: &[u8]) -> Self {
        Self(DensePolynomial::read_transcription_bytes(bytes))
    }

    #[inline(always)]
    fn write_transcription_bytes(&self, buf: &mut [u8]) {
        self.0.write_transcription_bytes(buf);
    }
}

impl<const DEGREE_PLUS_ONE: usize> FromRef<BinaryPoly<DEGREE_PLUS_ONE>>
    for BinaryPoly<DEGREE_PLUS_ONE>
{
    #[inline(always)]
    fn from_ref(poly: &BinaryPoly<DEGREE_PLUS_ONE>) -> Self {
        *poly
    }
}

impl<const DEGREE_PLUS_ONE: usize> From<&BinaryPoly<DEGREE_PLUS_ONE>>
    for BinaryPoly<DEGREE_PLUS_ONE>
{
    #[inline(always)]
    fn from(value: &BinaryPoly<DEGREE_PLUS_ONE>) -> Self {
        Self::from_ref(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evaluate_is_correct() {
        for i in 0..16 {
            let poly = BinaryPoly::<4>::new([
                (i & 0b0001 != 0).into(),
                (i & 0b0010 != 0).into(),
                (i & 0b0100 != 0).into(),
                (i & 0b1000 != 0).into(),
            ]);

            let result = poly.evaluate_at_point(&2).unwrap();

            assert_eq!(result, i);
        }
    }
}
