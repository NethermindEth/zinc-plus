use crate::{
    ConstCoeffBitWidth, EvaluatablePolynomial, EvaluationError, Polynomial,
    univariate::binary::BinaryPoly,
};
use core::slice;
use crypto_primitives::{
    FixedSemiring, FromWithConfig, IntoWithConfig, PrimeField, Ring, Semiring, boolean::Boolean,
};
use itertools::Itertools;
use num_traits::{CheckedAdd, CheckedMul, CheckedNeg, CheckedSub, One, Zero};
use rand::{distr::StandardUniform, prelude::*};
use std::{
    array,
    fmt::Display,
    hash::Hash,
    iter::{Product, Sum},
    marker::PhantomData,
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};
use zinc_transcript::traits::ConstTranscribable;
use zinc_utils::{
    from_ref::FromRef,
    inner_product::{InnerProduct, InnerProductError},
    mul_by_scalar::MulByScalar,
    named::Named,
    projectable_to_field::ProjectableToField,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DensePolynomial<R, const DEGREE_PLUS_ONE: usize> {
    /// Coefficients of the polynomial, lowest degree first
    pub coeffs: [R; DEGREE_PLUS_ONE],
}

impl<R: Semiring + Zero, const DEGREE_PLUS_ONE: usize> DensePolynomial<R, DEGREE_PLUS_ONE> {
    /// Create a new polynomial with the given coefficients.
    /// If the input has fewer than N+1 coefficients, the remaining slots will
    /// be filled with zeros. If the input has more than N+1 coefficients,
    /// it will panic.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn new(coeffs: impl AsRef<[R]>) -> Self {
        let coeffs = coeffs.as_ref();
        assert!(
            coeffs.len() <= DEGREE_PLUS_ONE,
            "Too many coefficients provided: expected at most {}, got {}",
            DEGREE_PLUS_ONE,
            coeffs.len()
        );

        if coeffs.is_empty() {
            return Self::zero();
        }

        let mut coeffs = coeffs.to_vec();
        coeffs.resize(DEGREE_PLUS_ONE, R::zero());
        let coeffs = coeffs.try_into().expect("unreachable");

        DensePolynomial { coeffs }
    }
}

impl<R: Semiring, const DEGREE_PLUS_ONE: usize> DensePolynomial<R, DEGREE_PLUS_ONE> {
    /// Create a new polynomial with the given coefficients.
    /// If the input has fewer than N+1 coefficients, the remaining slots will
    /// be filled with zeros. If the input has more than N+1 coefficients,
    /// it will panic.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn new_with_zero(coeffs: impl AsRef<[R]>, zero: R) -> Self {
        let coeffs = coeffs.as_ref();
        assert!(
            coeffs.len() <= DEGREE_PLUS_ONE,
            "Too many coefficients provided: expected at most {}, got {}",
            DEGREE_PLUS_ONE,
            coeffs.len()
        );

        let mut coeffs = coeffs.to_vec();
        coeffs.resize(DEGREE_PLUS_ONE, zero);
        let coeffs = coeffs.try_into().expect("unreachable");

        DensePolynomial { coeffs }
    }
}

impl<R: Copy, const DEGREE_PLUS_ONE: usize> Copy for DensePolynomial<R, DEGREE_PLUS_ONE> {}

impl<R: Default, const DEGREE_PLUS_ONE: usize> Default for DensePolynomial<R, DEGREE_PLUS_ONE> {
    fn default() -> Self {
        DensePolynomial {
            coeffs: array::from_fn::<_, DEGREE_PLUS_ONE, _>(|_| R::default()),
        }
    }
}

impl<R: Display, const DEGREE_PLUS_ONE: usize> Display for DensePolynomial<R, DEGREE_PLUS_ONE> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        let mut first = true;
        for coeff in self.coeffs.iter() {
            if first {
                first = false;
            } else {
                write!(f, ", ")?;
            }
            write!(f, "{}", coeff)?;
        }
        write!(f, "]")?;
        Ok(())
    }
}

impl<R: Hash, const DEGREE_PLUS_ONE: usize> Hash for DensePolynomial<R, DEGREE_PLUS_ONE> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        for coeff in self.coeffs.iter() {
            coeff.hash(state);
        }
    }
}

impl<R: Semiring + Zero, const DEGREE_PLUS_ONE: usize> Zero
    for DensePolynomial<R, DEGREE_PLUS_ONE>
{
    fn zero() -> Self {
        Self {
            coeffs: array::from_fn::<_, DEGREE_PLUS_ONE, _>(|_| R::zero()),
        }
    }

    fn is_zero(&self) -> bool {
        self.coeffs.iter().all(|c| c.is_zero())
    }
}

impl<R: Semiring + Zero + One, const DEGREE_PLUS_ONE: usize> One
    for DensePolynomial<R, DEGREE_PLUS_ONE>
{
    fn one() -> Self {
        Self::from(R::one())
    }
}

impl<R: Ring + Neg<Output = R>, const DEGREE_PLUS_ONE: usize> Neg
    for DensePolynomial<R, DEGREE_PLUS_ONE>
{
    type Output = Self;

    #[allow(clippy::arithmetic_side_effects)] // By design
    fn neg(mut self) -> Self::Output {
        self.coeffs.iter_mut().for_each(|c| *c = -c.clone());
        self
    }
}

impl<R: Semiring, const DEGREE_PLUS_ONE: usize> Add for DensePolynomial<R, DEGREE_PLUS_ONE> {
    type Output = Self;

    #[allow(clippy::arithmetic_side_effects, clippy::op_ref)]
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        self + &rhs
    }
}

impl<'a, R: Semiring, const DEGREE_PLUS_ONE: usize> Add<&'a Self>
    for DensePolynomial<R, DEGREE_PLUS_ONE>
{
    type Output = Self;

    #[allow(clippy::arithmetic_side_effects)]
    #[inline(always)]
    fn add(mut self, rhs: &'a Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl<R: Semiring, const DEGREE_PLUS_ONE: usize> Sub for DensePolynomial<R, DEGREE_PLUS_ONE> {
    type Output = Self;

    #[allow(clippy::arithmetic_side_effects, clippy::op_ref)]
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        self - &rhs
    }
}

impl<'a, R: Semiring, const DEGREE_PLUS_ONE: usize> Sub<&'a Self>
    for DensePolynomial<R, DEGREE_PLUS_ONE>
{
    type Output = Self;

    #[allow(clippy::arithmetic_side_effects)]
    #[inline(always)]
    fn sub(mut self, rhs: &'a Self) -> Self::Output {
        self -= rhs;
        self
    }
}

impl<R: Semiring, const DEGREE_PLUS_ONE: usize> Mul for DensePolynomial<R, DEGREE_PLUS_ONE> {
    type Output = Self;

    #[allow(clippy::arithmetic_side_effects, clippy::op_ref)]
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        self * &rhs
    }
}

impl<'a, R: Semiring, const DEGREE_PLUS_ONE: usize> Mul<&'a Self>
    for DensePolynomial<R, DEGREE_PLUS_ONE>
{
    type Output = Self;

    fn mul(self, _rhs: &'a Self) -> Self::Output {
        unimplemented!("Polynomial multiplication is not implemented")
    }
}

impl<R: Semiring, const DEGREE_PLUS_ONE: usize> AddAssign for DensePolynomial<R, DEGREE_PLUS_ONE> {
    #[allow(clippy::arithmetic_side_effects)]
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self += &rhs;
    }
}

impl<'a, R: Semiring, const DEGREE_PLUS_ONE: usize> AddAssign<&'a Self>
    for DensePolynomial<R, DEGREE_PLUS_ONE>
{
    #[allow(clippy::arithmetic_side_effects)]
    #[inline(always)]
    fn add_assign(&mut self, rhs: &'a Self) {
        for i in 0..DEGREE_PLUS_ONE {
            self.coeffs[i] += &rhs.coeffs[i];
        }
    }
}

impl<R: Semiring, const DEGREE_PLUS_ONE: usize> SubAssign for DensePolynomial<R, DEGREE_PLUS_ONE> {
    #[allow(clippy::arithmetic_side_effects)]
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self -= &rhs;
    }
}

impl<'a, R: Semiring, const DEGREE_PLUS_ONE: usize> SubAssign<&'a Self>
    for DensePolynomial<R, DEGREE_PLUS_ONE>
{
    #[allow(clippy::arithmetic_side_effects)]
    #[inline(always)]
    fn sub_assign(&mut self, rhs: &'a Self) {
        for i in 0..DEGREE_PLUS_ONE {
            self.coeffs[i] -= &rhs.coeffs[i];
        }
    }
}

impl<R: Semiring, const DEGREE_PLUS_ONE: usize> MulAssign for DensePolynomial<R, DEGREE_PLUS_ONE> {
    #[allow(clippy::arithmetic_side_effects)]
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self *= &rhs;
    }
}

impl<'a, R: Semiring, const DEGREE_PLUS_ONE: usize> MulAssign<&'a Self>
    for DensePolynomial<R, DEGREE_PLUS_ONE>
{
    fn mul_assign(&mut self, _rhs: &'a Self) {
        unimplemented!("Polynomial multiplication is not implemented")
    }
}

impl<R: Ring + Zero, const DEGREE_PLUS_ONE: usize> CheckedNeg
    for DensePolynomial<R, DEGREE_PLUS_ONE>
{
    fn checked_neg(&self) -> Option<Self> {
        let mut coeffs = self.coeffs.clone();

        coeffs
            .iter_mut()
            .filter(|coeff| !coeff.is_zero())
            .try_for_each(|x| {
                *x = x.checked_neg()?;
                Some(())
            })?;

        Some(Self { coeffs })
    }
}

impl<R: Semiring, const DEGREE_PLUS_ONE: usize> CheckedAdd for DensePolynomial<R, DEGREE_PLUS_ONE> {
    fn checked_add(&self, other: &Self) -> Option<Self> {
        let mut coeffs = self.coeffs.clone();

        coeffs.iter_mut().zip(other).try_for_each(|(a, b)| {
            *a = a.checked_add(b)?;
            Some(())
        })?;

        Some(Self { coeffs })
    }
}

impl<R: Semiring, const DEGREE_PLUS_ONE: usize> CheckedSub for DensePolynomial<R, DEGREE_PLUS_ONE> {
    fn checked_sub(&self, other: &Self) -> Option<Self> {
        let mut coeffs = self.coeffs.clone();

        coeffs.iter_mut().zip(other).try_for_each(|(a, b)| {
            *a = a.checked_sub(b)?;
            Some(())
        })?;

        Some(Self { coeffs })
    }
}

impl<R: Semiring, const DEGREE_PLUS_ONE: usize> CheckedMul for DensePolynomial<R, DEGREE_PLUS_ONE> {
    fn checked_mul(&self, _other: &Self) -> Option<Self> {
        unimplemented!("Polynomial multiplication is not implemented")
    }
}

impl<R: FixedSemiring, const DEGREE_PLUS_ONE: usize> Sum for DensePolynomial<R, DEGREE_PLUS_ONE> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, x| {
            acc.checked_add(&x).expect("overflow in sum")
        })
    }
}

impl<'a, R: FixedSemiring, const DEGREE_PLUS_ONE: usize> Sum<&'a Self>
    for DensePolynomial<R, DEGREE_PLUS_ONE>
{
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, x| {
            acc.checked_add(x).expect("overflow in sum")
        })
    }
}

impl<R: FixedSemiring, const DEGREE_PLUS_ONE: usize> Product
    for DensePolynomial<R, DEGREE_PLUS_ONE>
{
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |acc, x| {
            acc.checked_mul(&x).expect("overflow in product")
        })
    }
}

impl<'a, R: FixedSemiring, const DEGREE_PLUS_ONE: usize> Product<&'a Self>
    for DensePolynomial<R, DEGREE_PLUS_ONE>
{
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |acc, x| {
            acc.checked_mul(x).expect("overflow in product")
        })
    }
}

impl<R: Semiring, const DEGREE_PLUS_ONE: usize> Semiring for DensePolynomial<R, DEGREE_PLUS_ONE> {}

impl<R: Ring + FixedSemiring, const DEGREE_PLUS_ONE: usize> Ring
    for DensePolynomial<R, DEGREE_PLUS_ONE>
{
}

impl<R, const DEGREE_PLUS_ONE: usize> Distribution<DensePolynomial<R, DEGREE_PLUS_ONE>>
    for StandardUniform
where
    StandardUniform: Distribution<R>,
    StandardUniform: Distribution<[R; DEGREE_PLUS_ONE]>, // This one we get for free
{
    fn sample<Gen: Rng + ?Sized>(&self, rng: &mut Gen) -> DensePolynomial<R, DEGREE_PLUS_ONE> {
        let coeffs: [R; DEGREE_PLUS_ONE] = rng.random();
        DensePolynomial { coeffs }
    }
}

//
// Zip-specific traits
//
impl<R: Semiring, const DEGREE_PLUS_ONE: usize> Polynomial<R>
    for DensePolynomial<R, DEGREE_PLUS_ONE>
{
    const DEGREE_BOUND: usize = DEGREE_PLUS_ONE - 1;
}

impl<R: Semiring, const DEGREE_PLUS_ONE: usize> EvaluatablePolynomial<R, R>
    for DensePolynomial<R, DEGREE_PLUS_ONE>
{
    type EvaluationPoint = R;

    fn evaluate_at_point(&self, point: &R) -> Result<R, EvaluationError> {
        // Horner's method.
        let mut result = self
            .coeffs
            .last()
            .ok_or(EvaluationError::EmptyPolynomial)?
            .clone();

        for coeff in self.coeffs.iter().rev().skip(1) {
            let term = result.checked_mul(point).ok_or(EvaluationError::Overflow)?;
            result = term.checked_add(coeff).ok_or(EvaluationError::Overflow)?;
        }

        Ok(result)
    }
}

impl<R: Semiring + ConstTranscribable, const DEGREE_PLUS_ONE: usize> ConstCoeffBitWidth
    for DensePolynomial<R, DEGREE_PLUS_ONE>
{
    const COEFF_BIT_WIDTH: usize = R::NUM_BITS;
}

impl<R: Semiring + Named, const DEGREE_PLUS_ONE: usize> Named
    for DensePolynomial<R, DEGREE_PLUS_ONE>
{
    fn type_name() -> String {
        format!("Poly<{}, {}>", R::type_name(), Self::DEGREE_BOUND)
    }
}

impl<R: ConstTranscribable + Default, const DEGREE_PLUS_ONE: usize> ConstTranscribable
    for DensePolynomial<R, DEGREE_PLUS_ONE>
{
    const NUM_BYTES: usize = R::NUM_BYTES * DEGREE_PLUS_ONE;

    #[allow(clippy::arithmetic_side_effects)]
    fn read_transcription_bytes(bytes: &[u8]) -> Self {
        assert_eq!(
            bytes.len(),
            R::NUM_BYTES * DEGREE_PLUS_ONE,
            "Invalid byte length for DensePolynomial: expected {}, got {}",
            R::NUM_BYTES * DEGREE_PLUS_ONE,
            bytes.len()
        );

        // Can't use as_chunks because generic parameters may not be used in const
        // operations.
        let coeffs = bytes
            .chunks_exact(R::NUM_BYTES)
            .map(R::read_transcription_bytes)
            .collect_array()
            .expect("Unreachable");
        Self { coeffs }
    }

    fn write_transcription_bytes(&self, buf: &mut [u8]) {
        for (chunk, coeff) in buf.chunks_exact_mut(R::NUM_BYTES).zip(self.coeffs.iter()) {
            coeff.write_transcription_bytes(chunk);
        }
    }
}

// Conversions.

impl<R, S, const DEGREE_PLUS_ONE: usize> FromRef<DensePolynomial<S, DEGREE_PLUS_ONE>>
    for DensePolynomial<R, DEGREE_PLUS_ONE>
where
    R: Semiring + FromRef<S> + Default,
{
    fn from_ref(value: &DensePolynomial<S, DEGREE_PLUS_ONE>) -> Self {
        let mut coeffs = array::from_fn::<_, DEGREE_PLUS_ONE, _>(|_| R::default());
        coeffs
            .iter_mut()
            .zip(value.coeffs.iter())
            .for_each(|(coeff, other_coeff)| {
                *coeff = R::from_ref(other_coeff);
            });
        DensePolynomial { coeffs }
    }
}

impl<R, const DEGREE_PLUS_ONE: usize> FromRef<BinaryPoly<DEGREE_PLUS_ONE>>
    for DensePolynomial<R, DEGREE_PLUS_ONE>
where
    R: Semiring + FromRef<Boolean> + Default,
{
    #[inline(always)]
    fn from_ref(value: &BinaryPoly<DEGREE_PLUS_ONE>) -> Self {
        Self::from_ref(value.inner())
    }
}

impl<R, S, const DEGREE_PLUS_ONE: usize> From<&DensePolynomial<S, DEGREE_PLUS_ONE>>
    for DensePolynomial<R, DEGREE_PLUS_ONE>
where
    R: Semiring + FromRef<S> + Default,
{
    fn from(value: &DensePolynomial<S, DEGREE_PLUS_ONE>) -> Self {
        Self::from_ref(value)
    }
}

impl<R: Zero, const DEGREE_PLUS_ONE: usize> From<R> for DensePolynomial<R, DEGREE_PLUS_ONE> {
    fn from(value: R) -> Self {
        let mut coeffs = array::from_fn(|_| R::zero());
        coeffs[0] = value;
        Self { coeffs }
    }
}

impl<const DEGREE_PLUS_ONE: usize> FromRef<i64> for DensePolynomial<i128, DEGREE_PLUS_ONE> {
    fn from_ref(value: &i64) -> Self {
        Self::from(i128::from(*value))
    }
}

impl<'a, R, S, const DEGREE_PLUS_ONE: usize> MulByScalar<&'a S>
    for DensePolynomial<R, DEGREE_PLUS_ONE>
where
    R: FixedSemiring + MulByScalar<&'a S>,
{
    fn mul_by_scalar(&self, rhs: &'a S) -> Option<Self> {
        let mut coeffs = self.coeffs.clone();

        coeffs
            .iter_mut()
            .filter(|coeff| !coeff.is_zero())
            .try_for_each(|x| {
                *x = x.mul_by_scalar(rhs)?;
                Some(())
            })?;

        Some(Self { coeffs })
    }
}

impl<R, F, const DEGREE_PLUS_ONE: usize> ProjectableToField<F>
    for DensePolynomial<R, DEGREE_PLUS_ONE>
where
    R: Semiring,
    F: PrimeField + for<'a> FromWithConfig<&'a R> + for<'a> MulByScalar<&'a F> + 'static,
{
    #![allow(clippy::arithmetic_side_effects)] // False alert, field operations are safe
    fn prepare_projection(
        sampled_value: &F,
    ) -> impl Fn(&DensePolynomial<R, DEGREE_PLUS_ONE>) -> F + 'static {
        let sampled_value = sampled_value.clone();
        let field_cfg = sampled_value.cfg().clone();

        move |poly: &DensePolynomial<R, DEGREE_PLUS_ONE>| {
            let coeffs: [F; DEGREE_PLUS_ONE] = poly
                .coeffs
                .iter()
                .map(|v| v.into_with_cfg(&field_cfg))
                .collect_array()
                .expect("unreachable");

            let poly2 = DensePolynomial { coeffs };
            poly2
                .evaluate_at_point(&sampled_value)
                .expect("Failed to evaluate polynomial at point")
        }
    }
}

impl<'a, R, const DEGREE_PLUS_ONE: usize> IntoIterator for &'a DensePolynomial<R, DEGREE_PLUS_ONE> {
    type Item = &'a R;

    type IntoIter = slice::Iter<'a, R>;

    fn into_iter(self) -> Self::IntoIter {
        self.coeffs.iter()
    }
}

impl<R, const DEGREE_PLUS_ONE: usize> AsRef<[R]> for DensePolynomial<R, DEGREE_PLUS_ONE> {
    fn as_ref(&self) -> &[R] {
        self.coeffs.as_slice()
    }
}

pub struct DensePolyInnerProduct<
    R,
    Rhs,
    Out,
    I: InnerProduct<[R], Rhs, Out>,
    const DEGREE_PLUS_ONE: usize,
>(PhantomData<(I, R, Rhs, Out)>);

impl<R, Rhs, Out, I, const DEGREE_PLUS_ONE: usize>
    InnerProduct<DensePolynomial<R, DEGREE_PLUS_ONE>, Rhs, Out>
    for DensePolyInnerProduct<R, Rhs, Out, I, DEGREE_PLUS_ONE>
where
    I: InnerProduct<[R], Rhs, Out>,
{
    #[inline(always)]
    fn inner_product(
        lhs: &DensePolynomial<R, DEGREE_PLUS_ONE>,
        rhs: &[Rhs],
        zero: Out,
    ) -> Result<Out, InnerProductError> {
        I::inner_product(&lhs.coeffs, rhs, zero)
    }
}
