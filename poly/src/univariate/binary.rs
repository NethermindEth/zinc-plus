use crate::{
    ConstCoeffBitWidth, EvaluatablePolynomial, EvaluationError, Polynomial,
    univariate::dense::DensePolynomial,
};
use crypto_primitives::{PrimeField, Semiring, semiring::boolean::Boolean};
use derive_more::{
    Add, AddAssign, AsRef, Display, From, Mul, MulAssign, Product, Sub, SubAssign, Sum,
};
use num_traits::{CheckedAdd, CheckedMul, CheckedSub, ConstZero, One, Zero};
use rand::{distr::StandardUniform, prelude::*};
use std::{
    array,
    hash::Hash,
    iter::{Product, Sum},
    marker::PhantomData,
    ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign},
};
use zinc_transcript::traits::ConstTranscribable;
use zinc_utils::{
    from_ref::FromRef,
    inner_product::{BooleanInnerProductUncheckedAdd, InnerProduct, InnerProductError},
    mul_by_scalar::WideningMulByScalar,
    named::Named,
    projectable_to_field::ProjectableToField,
};

#[derive(
    Add,
    AddAssign,
    AsRef,
    Clone,
    Debug,
    From,
    Default,
    Display,
    Hash,
    PartialEq,
    Eq,
    Mul,
    MulAssign,
    Sub,
    SubAssign,
    Sum,
    Product,
)]
#[repr(transparent)]
pub struct BinaryPoly<const DEGREE_PLUS_ONE: usize>(DensePolynomial<Boolean, DEGREE_PLUS_ONE>);

impl<const DEGREE_PLUS_ONE: usize> BinaryPoly<DEGREE_PLUS_ONE> {
    #[inline(always)]
    pub const fn inner(&self) -> &DensePolynomial<Boolean, DEGREE_PLUS_ONE> {
        &self.0
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

impl From<u32> for BinaryPoly<32> {
    fn from(value: u32) -> Self {
        Self(DensePolynomial {
            coeffs: array::from_fn(|i| Boolean::new(value & (1 << i) != 0)),
        })
    }
}

impl From<u64> for BinaryPoly<64> {
    fn from(value: u64) -> Self {
        Self(DensePolynomial {
            coeffs: array::from_fn(|i| Boolean::new(value & (1 << i) != 0)),
        })
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

    /// Create a new polynomial with the given coefficients.
    /// If the input has fewer than N+1 coefficients, the remaining slots will
    /// be filled with zeros. If the input has more than N+1 coefficients,
    /// it will panic.
    #[inline(always)]
    pub fn new_padded(coeffs: impl AsRef<[Boolean]>) -> Self {
        Self(DensePolynomial::new_with_zero(coeffs, Boolean::ZERO))
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

impl<'a, const DEGREE_PLUS_ONE: usize> Add<&'a Self> for BinaryPoly<DEGREE_PLUS_ONE> {
    type Output = Self;

    #[allow(clippy::arithmetic_side_effects)]
    #[inline(always)]
    fn add(self, rhs: &'a Self) -> Self::Output {
        Self(self.0 + rhs.0)
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

impl<'a, const DEGREE_PLUS_ONE: usize> AddAssign<&'a Self> for BinaryPoly<DEGREE_PLUS_ONE> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: &'a Self) {
        self.0.add_assign(&rhs.0);
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

impl<'a, const DEGREE_PLUS_ONE: usize> Sum<&'a Self> for BinaryPoly<DEGREE_PLUS_ONE> {
    #[inline(always)]
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        Self(iter.map(|x| &x.0).sum())
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
            .try_fold(
                (self.0.coeffs[0].widen::<R>(), R::one()),
                |(mut acc, mut pow), coeff| {
                    pow = pow.checked_mul(point).ok_or(EvaluationError::Overflow)?;

                    if coeff.inner() {
                        acc = acc.checked_add(&pow).ok_or(EvaluationError::Overflow)?;
                    }

                    Ok((acc, pow))
                },
            )?
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
        poly.clone()
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

pub struct BinaryPolyInnerProduct<R, I, const DEGREE_PLUS_ONE: usize>(PhantomData<(R, I)>);

impl<Rhs, I, Out, const DEGREE_PLUS_ONE: usize> InnerProduct<BinaryPoly<DEGREE_PLUS_ONE>, Rhs, Out>
    for BinaryPolyInnerProduct<Rhs, I, DEGREE_PLUS_ONE>
where
    I: InnerProduct<[Boolean], Rhs, Out>,
{
    #[inline(always)]
    fn inner_product(
        lhs: &BinaryPoly<DEGREE_PLUS_ONE>,
        rhs: &[Rhs],
        zero: Out,
    ) -> Result<Out, InnerProductError> {
        I::inner_product(&lhs.0.coeffs, rhs, zero)
    }
}

impl<F, const DEGREE_PLUS_ONE: usize> ProjectableToField<F> for BinaryPoly<DEGREE_PLUS_ONE>
where
    F: PrimeField + FromRef<F> + 'static,
{
    #![allow(clippy::arithmetic_side_effects)] // False alert, field operations are safe
    fn prepare_projection(sampled_value: &F) -> impl Fn(&Self) -> F + Send + Sync + 'static {
        let field_cfg = sampled_value.cfg().clone();
        let r_powers = {
            // Preprocess powers prior to inner product.
            let mut r_powers = Vec::with_capacity(DEGREE_PLUS_ONE);

            let mut curr = F::one_with_cfg(&field_cfg);
            r_powers.push(curr.clone());

            for _ in 1..DEGREE_PLUS_ONE {
                curr *= sampled_value;
                r_powers.push(curr.clone());
            }

            r_powers
        };

        move |poly: &BinaryPoly<DEGREE_PLUS_ONE>| {
            BinaryPolyInnerProduct::<_, BooleanInnerProductUncheckedAdd, _>::inner_product(
                poly,
                &r_powers,
                F::zero_with_cfg(&field_cfg),
            )
            .expect("Failed to evaluate polynomial")
        }
    }
}

#[derive(Clone, Copy, Default)]
pub struct BinaryPolyWideningMulByScalar<Output>(PhantomData<Output>);

impl<Rhs, Output, const DEGREE_PLUS_ONE: usize>
    WideningMulByScalar<BinaryPoly<DEGREE_PLUS_ONE>, Rhs> for BinaryPolyWideningMulByScalar<Output>
where
    Rhs: Copy,
    Output: From<Rhs> + Send + Sync + Default + Copy + Zero,
{
    type Output = DensePolynomial<Output, DEGREE_PLUS_ONE>;

    fn mul_by_scalar_widen(lhs: &BinaryPoly<DEGREE_PLUS_ONE>, rhs: &Rhs) -> Self::Output {
        let mut coeffs: [Output; DEGREE_PLUS_ONE] = [Output::zero(); DEGREE_PLUS_ONE];

        coeffs.iter_mut().enumerate().for_each(|(i, out)| {
            if lhs.0.coeffs[i].inner() {
                *out = (*rhs).into();
            }
        });

        DensePolynomial { coeffs }
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
