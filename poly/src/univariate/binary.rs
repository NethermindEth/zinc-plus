use crate::{
    ConstCoeffBitWidth, EvaluatablePolynomial, EvaluationError, Polynomial,
    univariate::dense::DensePolynomial,
};
use crypto_primitives::{PrimeField, Semiring, semiring::boolean::Boolean};
use derive_more::{AsRef, Display, From};
use num_traits::{CheckedAdd, CheckedMul, CheckedSub, One, Zero};
use rand::{distr::StandardUniform, prelude::*};
use std::{
    hash::Hash,
    iter::{Product, Sum},
    marker::PhantomData,
    ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign},
};
use zinc_transcript::traits::ConstTranscribable;
use zinc_utils::{
    from_ref::FromRef,
    inner_product::{
        BooleanInnerProductCheckedAdd, BooleanInnerProductUncheckedAdd, InnerProduct,
        InnerProductError,
    },
    mul_by_scalar::WideningMulByScalar,
    named::Named,
    projectable_to_field::ProjectableToField,
};

const DEGREE_PLUS_ONE: usize = 32;

#[derive(
    AsRef,
    Clone,
    Copy,
    Debug,
    From,
    Default,
    Display,
    Hash,
    PartialEq,
    Eq,
)]
#[repr(transparent)]
pub struct BinaryPoly(u32);

// impl BinaryPoly {
//     #[inline(always)]
//     pub const fn inner(&self) -> &DensePolynomial<Boolean, DEGREE_PLUS_ONE> {
//         &self.0
//     }
// }

impl From<BinaryPoly>
    for DensePolynomial<Boolean, DEGREE_PLUS_ONE>
{
    #[inline(always)]
    fn from(binary_poly: BinaryPoly) -> Self {
        DensePolynomial {
            coeffs: std::array::from_fn(|i| Boolean::from(((binary_poly.0 >> i) & 1) == 1)),
        }
    }
}

impl BinaryPoly {
    /// Create a new polynomial with the given coefficients.
    /// If the input has fewer than N+1 coefficients, the remaining slots will
    /// be filled with zeros. If the input has more than N+1 coefficients,
    /// it will panic.
    #[inline(always)]
    pub fn new(coeffs: impl AsRef<[Boolean]>) -> Self {
        // Self(DensePolynomial::new(coeffs))
        let coeffs = coeffs.as_ref();
        assert!(
            coeffs.len() <= DEGREE_PLUS_ONE,
            "Too many coefficients provided: expected at most {}, got {}",
            DEGREE_PLUS_ONE,
            coeffs.len()
        );

        let mut bits = 0u32;
        for (i, coeff) in coeffs.iter().enumerate() {
            if coeff.into_inner() {
                bits |= 1u32 << i;
            }
        }

        Self(bits)
    }

    /// Create a new polynomial with the given coefficients.
    /// If the input has fewer than N+1 coefficients, the remaining slots will
    /// be filled with zeros. If the input has more than N+1 coefficients,
    /// it will panic.
    #[inline(always)]
    pub fn new_padded(coeffs: impl AsRef<[Boolean]>) -> Self {
        // Self(DensePolynomial::new_with_zero(coeffs, Boolean::ZERO))
        Self::new(coeffs)
    }
}

impl Add for BinaryPoly {
    type Output = Self;

    #[allow(clippy::arithmetic_side_effects, clippy::op_ref)]
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        self + &rhs
    }
}

impl Zero for BinaryPoly {
    #[inline(always)]
    fn zero() -> Self {
        Self(0)
    }

    #[inline(always)]
    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}

impl One for BinaryPoly {
    #[inline(always)]
    fn one() -> Self {
        Self(1)
    }
}

impl<'a> Add<&'a Self> for BinaryPoly {
    type Output = Self;

    #[allow(clippy::arithmetic_side_effects)]
    #[inline(always)]
    fn add(self, rhs: &'a Self) -> Self::Output {
        Self(self.0 ^ rhs.0)
    }
}

impl Sub for BinaryPoly {
    type Output = Self;

    #[allow(clippy::arithmetic_side_effects, clippy::op_ref)]
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        self - &rhs
    }
}

impl<'a> Sub<&'a Self> for BinaryPoly {
    type Output = Self;

    #[allow(clippy::arithmetic_side_effects)]
    #[inline(always)]
    fn sub(self, rhs: &'a Self) -> Self::Output {
        Self(self.0 ^ rhs.0)
    }
}

impl Mul for BinaryPoly {
    type Output = Self;

    #[allow(clippy::arithmetic_side_effects)]
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        self * &rhs
    }
}

impl<'a> Mul<&'a Self> for BinaryPoly {
    type Output = Self;

    #[allow(clippy::arithmetic_side_effects)]
    #[inline(always)]
    fn mul(self, rhs: &'a Self) -> Self::Output {
        let mut acc = 0u32;
        let mut rhs_bits = rhs.0;

        while rhs_bits != 0 {
            let bit = rhs_bits.trailing_zeros();
            acc ^= self.0.wrapping_shl(bit);
            rhs_bits &= rhs_bits - 1;
        }

        Self(acc)
    }
}

impl AddAssign for BinaryPoly {
    #[allow(clippy::arithmetic_side_effects, clippy::op_ref)]
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self += &rhs;
    }
}

impl<'a> AddAssign<&'a Self> for BinaryPoly {
    #[inline(always)]
    fn add_assign(&mut self, rhs: &'a Self) {
        self.0 ^= rhs.0;
    }
}

impl SubAssign for BinaryPoly {
    #[allow(clippy::arithmetic_side_effects, clippy::op_ref)]
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self -= &rhs;
    }
}

impl<'a> SubAssign<&'a Self> for BinaryPoly {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: &'a Self) {
        self.0 ^= rhs.0;
    }
}

impl MulAssign for BinaryPoly {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self *= &rhs;
    }
}

impl<'a> MulAssign<&'a Self> for BinaryPoly {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: &'a Self) {
        let mut acc = 0u32;
        let mut rhs_bits = rhs.0;

        while rhs_bits != 0 {
            let bit = rhs_bits.trailing_zeros();
            acc ^= self.0.wrapping_shl(bit);
            rhs_bits &= rhs_bits - 1;
        }

        self.0 = acc;
    }
}

impl CheckedAdd for BinaryPoly {
    #[inline(always)]
    fn checked_add(&self, other: &Self) -> Option<Self> {
        Some(Self(self.0 ^ other.0))
    }
}

impl CheckedSub for BinaryPoly {
    #[inline(always)]
    fn checked_sub(&self, other: &Self) -> Option<Self> {
        Some(Self(self.0 ^ other.0))
    }
}

impl CheckedMul for BinaryPoly {
    #[inline(always)]
    fn checked_mul(&self, other: &Self) -> Option<Self> {
        Some((*self) * other)
    }
}

impl<'a> Sum<&'a Self> for BinaryPoly {
    #[inline(always)]
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, x| acc + x)
    }
}

impl<'a> Product<&'a Self> for BinaryPoly {
    #[inline(always)]
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |acc, x| acc * x)
    }
}

impl Semiring for BinaryPoly {}

impl Distribution<BinaryPoly> for StandardUniform {
    #[inline(always)]
    fn sample<Gen: Rng + ?Sized>(&self, rng: &mut Gen) -> BinaryPoly {
        // let coeffs: [Boolean; DEGREE_PLUS_ONE] = rng.random();

        // I didn't manage to delegate this one to
        // `DensePolynomial::sample` because of unsatisfied
        // traits.

        BinaryPoly(rng.random())
    }
}

//
// Zip-specific traits
//
impl Polynomial<Boolean> for BinaryPoly {
    const DEGREE_BOUND: usize = DEGREE_PLUS_ONE - 1;
}

impl<R: Clone + Zero + One + CheckedAdd + CheckedMul>
    EvaluatablePolynomial<Boolean, R> for BinaryPoly
{
    type EvaluationPoint = R;

    fn evaluate_at_point(&self, point: &R) -> Result<R, EvaluationError> {
        // if DEGREE_PLUS_ONE.is_one() {
        //     return Ok(R::zero());
        // }
        //
        // let result = self.0.coeffs[1..]
        //     .iter()
        //     .try_fold(
        //         (self.0.coeffs[0].widen::<R>(), R::one()),
        //         |(mut acc, mut pow), coeff| {
        //             pow = pow.checked_mul(point).ok_or(EvaluationError::Overflow)?;
        //
        //             if coeff.inner() {
        //                 acc = acc.checked_add(&pow).ok_or(EvaluationError::Overflow)?;
        //             }
        //
        //             Ok((acc, pow))
        //         },
        //     )?
        //     .0;
        //
        // Ok(result)
        if self.0 == 0 {
            return Ok(R::zero());
        }

        let max_bit = (31u32 - self.0.leading_zeros()) as usize;

        let mut acc = if (self.0 & 1) == 1 {
            R::one()
        } else {
            R::zero()
        };

        let mut pow = R::one();
        for i in 1..=max_bit {
            pow = pow.checked_mul(point).ok_or(EvaluationError::Overflow)?;
            if ((self.0 >> i) & 1) == 1 {
                acc = acc.checked_add(&pow).ok_or(EvaluationError::Overflow)?;
            }
        }

        Ok(acc)
    }
}

impl ConstCoeffBitWidth for BinaryPoly {
    const COEFF_BIT_WIDTH: usize = DensePolynomial::<Boolean, DEGREE_PLUS_ONE>::COEFF_BIT_WIDTH;
}

impl Named for BinaryPoly {
    fn type_name() -> String {
        format!("BPoly<{}>", Self::DEGREE_BOUND)
    }
}

impl ConstTranscribable for BinaryPoly {
    const NUM_BYTES: usize = <u32 as ConstTranscribable>::NUM_BYTES;

    #[inline(always)]
    fn read_transcription_bytes(bytes: &[u8]) -> Self {
        Self(<u32 as ConstTranscribable>::read_transcription_bytes(bytes))
    }

    #[inline(always)]
    fn write_transcription_bytes(&self, buf: &mut [u8]) {
        self.0.write_transcription_bytes(buf);
    }
}

impl FromRef<BinaryPoly>
    for BinaryPoly
{
    #[inline(always)]
    fn from_ref(poly: &BinaryPoly) -> Self {
        poly.clone()
    }
}

impl From<&BinaryPoly>
    for BinaryPoly
{
    #[inline(always)]
    fn from(value: &BinaryPoly) -> Self {
        Self::from_ref(value)
    }
}

pub struct BinaryPolyInnerProduct<R, I>(PhantomData<(R, I)>);

impl<Rhs, Out> InnerProduct<BinaryPoly, Rhs, Out>
    for BinaryPolyInnerProduct<Rhs, BooleanInnerProductUncheckedAdd>
where
    Rhs: Clone,
    Out: FromRef<Rhs> + for<'a> Add<&'a Out, Output = Out>,
{
    #[inline(always)]
    #[allow(clippy::arithmetic_side_effects)]
    fn inner_product(
        lhs: &BinaryPoly,
        rhs: &[Rhs],
        zero: Out,
    ) -> Result<Out, InnerProductError> {
        if rhs.len() != DEGREE_PLUS_ONE {
            return Err(InnerProductError::LengthMismatch {
                lhs: DEGREE_PLUS_ONE,
                rhs: rhs.len(),
            });
        }

        let mut acc = zero;
        let mut bits = lhs.0;
        while bits != 0 {
            let i = bits.trailing_zeros() as usize;
            acc = acc + &Out::from_ref(&rhs[i]);
            bits &= bits - 1;
        }

        Ok(acc)
    }
}

impl<Rhs, Out> InnerProduct<BinaryPoly, Rhs, Out>
    for BinaryPolyInnerProduct<Rhs, BooleanInnerProductCheckedAdd>
where
    Rhs: Clone,
    Out: From<Rhs> + CheckedAdd,
{
    #[inline(always)]
    fn inner_product(
        lhs: &BinaryPoly,
        rhs: &[Rhs],
        zero: Out,
    ) -> Result<Out, InnerProductError> {
        if rhs.len() != DEGREE_PLUS_ONE {
            return Err(InnerProductError::LengthMismatch {
                lhs: DEGREE_PLUS_ONE,
                rhs: rhs.len(),
            });
        }

        let mut acc = zero;
        let mut bits = lhs.0;
        while bits != 0 {
            let i = bits.trailing_zeros() as usize;
            acc = acc
                .checked_add(&Out::from(rhs[i].clone()))
                .ok_or(InnerProductError::Overflow)?;
            bits &= bits - 1;
        }

        Ok(acc)
    }
}

impl<F> ProjectableToField<F> for BinaryPoly
where
    F: PrimeField + FromRef<F> + 'static,
{
    #![allow(clippy::arithmetic_side_effects)] // False alert, field operations are safe
    fn prepare_projection(sampled_value: &F) -> impl Fn(&Self) -> F + 'static {
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

        move |poly: &BinaryPoly| {
            BinaryPolyInnerProduct::<_, BooleanInnerProductUncheckedAdd>::inner_product(
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

impl<Rhs, Output>
    WideningMulByScalar<BinaryPoly, Rhs> for BinaryPolyWideningMulByScalar<Output>
where
    Rhs: Copy,
    Output: From<Rhs> + Send + Sync + Default + Copy + Zero,
{
    type Output = DensePolynomial<Output, DEGREE_PLUS_ONE>;

    fn mul_by_scalar_widen(lhs: &BinaryPoly, rhs: &Rhs) -> Self::Output {
        let mut coeffs: [Output; DEGREE_PLUS_ONE] = [Output::zero(); DEGREE_PLUS_ONE];

        // coeffs.iter_mut().enumerate().for_each(|(i, out)| {
        //     if lhs.0.coeffs[i].inner() {
        //         *out = (*rhs).into();
        //     }
        // });
        let rhs = *rhs;
        let mut bits = lhs.0;
        while bits != 0 {
            let i = bits.trailing_zeros() as usize;
            coeffs[i] = rhs.into();
            bits &= bits - 1;
        }

        DensePolynomial { coeffs }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evaluate_is_correct() {
        for i in 0..16 {
            let poly = BinaryPoly::new([
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
