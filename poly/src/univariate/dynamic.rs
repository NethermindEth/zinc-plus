mod multiplication;

use crypto_primitives::{
    FromWithConfig, IntoWithConfig, PrimeField, Ring, Semiring, boolean::Boolean,
};
use derive_more::From;
use itertools::Itertools;
use num_traits::{CheckedAdd, CheckedMul, CheckedNeg, CheckedSub, ConstZero, One, Zero};
use std::{
    fmt::Display,
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};
use zinc_utils::{mul_by_scalar::MulByScalar, projectable_to_field::ProjectableToField};

use crate::{
    EvaluatablePolynomial, EvaluationError, Polynomial,
    univariate::{
        binary::BinaryPoly,
        dense::DensePolynomial,
        dynamic::multiplication::{mul_schoolbook_checked, mul_schoolbook_unchecked},
    },
};

/// Polynomials of dynamic degree. To be used
/// in UAIR and PIOP where ZIP+ degree bound
/// is not observed anymore.
///
/// Note that operations involving dynamic polynomials
/// do not trim leading zeros meaning
/// one can end up with unequal objects of the type
/// `DynamicPoly<R>` that represent equal polynomials,
/// therefore `trim` or `trim_with_zero` has to be called
/// before checking equality.
#[derive(Debug, Clone, From, Hash, PartialEq, Eq)]
pub struct DynamicPolynomial<R> {
    pub coeffs: Vec<R>,
}

impl<R: Zero + Clone> DynamicPolynomial<R> {
    /// Create a new polynomial with the given coefficients.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn new_trimmed(coeffs: impl AsRef<[R]>) -> Self {
        let coeffs = coeffs.as_ref();

        if let Some((non_zero, _)) = coeffs.iter().rev().find_position(|coeff| !coeff.is_zero()) {
            let deg_plus_one = coeffs.len() - non_zero;

            Self {
                coeffs: coeffs.iter().take(deg_plus_one).cloned().collect(),
            }
        } else {
            Self::zero()
        }
    }

    pub fn new(coeffs: impl AsRef<[R]>) -> Self {
        Self {
            coeffs: Vec::from(coeffs.as_ref()),
        }
    }

    #[allow(clippy::arithmetic_side_effects)]
    pub fn degree(&self) -> Option<usize> {
        if let Some((non_zero, _)) = self
            .coeffs
            .iter()
            .rev()
            .find_position(|coeff| !coeff.is_zero())
        {
            Some(self.coeffs.len() - non_zero - 1)
        } else {
            None
        }
    }

    #[allow(clippy::arithmetic_side_effects)]
    pub fn trim(&mut self) {
        self.coeffs
            .truncate(self.degree().map_or(0, |degree| degree + 1));
    }
}

impl<R> DynamicPolynomial<R> {
    pub const fn zero() -> Self {
        Self { coeffs: Vec::new() }
    }
}

impl<R: Eq> DynamicPolynomial<R> {
    #[allow(clippy::arithmetic_side_effects)]
    pub fn degree_with_zero(&self, zero: &R) -> Option<usize> {
        if let Some((non_zero, _)) = self
            .coeffs
            .iter()
            .rev()
            .find_position(|coeff| *coeff != zero)
        {
            Some(self.coeffs.len() - non_zero - 1)
        } else {
            None
        }
    }

    #[allow(clippy::arithmetic_side_effects)]
    pub fn trim_with_zero(&mut self, zero: &R) {
        self.coeffs
            .truncate(self.degree_with_zero(zero).map_or(0, |degree| degree + 1));
    }
}

impl<R> Default for DynamicPolynomial<R> {
    fn default() -> Self {
        Self {
            coeffs: Default::default(),
        }
    }
}

impl<R: Display> Display for DynamicPolynomial<R> {
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

impl<R: Semiring> Zero for DynamicPolynomial<R> {
    fn zero() -> Self {
        DynamicPolynomial::zero()
    }

    fn is_zero(&self) -> bool {
        self.coeffs.is_empty()
    }
}

impl<R: Semiring> ConstZero for DynamicPolynomial<R> {
    const ZERO: Self = DynamicPolynomial::zero();
}

impl<R: Semiring + One> One for DynamicPolynomial<R> {
    fn one() -> Self {
        Self {
            coeffs: vec![R::one()],
        }
    }
}

impl<R: Semiring> Add for DynamicPolynomial<R> {
    type Output = Self;

    #[allow(clippy::arithmetic_side_effects)]
    fn add(mut self, mut rhs: Self) -> Self::Output {
        if self.coeffs.len() < rhs.coeffs.len() {
            // Let self contain the polynomial with the
            // largest degree.
            std::mem::swap(&mut self, &mut rhs);
        }

        self.coeffs
            .iter_mut()
            .zip(rhs.coeffs.iter())
            .for_each(|(lhs, rhs)| {
                *lhs += rhs;
            });

        self
    }
}

impl<R: Semiring> Add<&Self> for DynamicPolynomial<R> {
    type Output = Self;

    #[allow(clippy::arithmetic_side_effects)]
    fn add(mut self, rhs: &Self) -> Self::Output {
        if self.coeffs.len() >= rhs.coeffs.len() {
            // If the degree of self is greater than
            // that of rhs we don't need to allocate
            // new vector of coeffs as we own enough memory.
            self.coeffs
                .iter_mut()
                .zip(rhs.coeffs.iter())
                .for_each(|(lhs, rhs)| {
                    *lhs += rhs;
                });

            self
        } else {
            self + rhs.clone()
        }
    }
}

impl<R: Ring> Sub for DynamicPolynomial<R> {
    type Output = Self;

    #[allow(clippy::arithmetic_side_effects)]
    fn sub(mut self, mut rhs: Self) -> Self::Output {
        if self.coeffs.len() < rhs.coeffs.len() {
            let lhs_len = self.coeffs.len();

            rhs.coeffs
                .iter_mut()
                .zip(self.coeffs)
                .for_each(|(rhs, lhs)| {
                    *rhs = lhs - &*rhs;
                });

            rhs.coeffs.iter_mut().skip(lhs_len).for_each(|rhs| {
                *rhs = rhs.clone().neg();
            });

            rhs
        } else {
            self.coeffs
                .iter_mut()
                .zip(rhs.coeffs.iter())
                .for_each(|(lhs, rhs)| {
                    *lhs -= rhs;
                });

            self
        }
    }
}

impl<R: Ring> Sub<&Self> for DynamicPolynomial<R> {
    type Output = Self;

    #[allow(clippy::arithmetic_side_effects)]
    fn sub(mut self, rhs: &Self) -> Self::Output {
        if self.coeffs.len() >= rhs.coeffs.len() {
            // If the degree of self is greater than
            // that of rhs we don't need to allocate
            // new vector of coeffs as we own enough memory.
            self.coeffs
                .iter_mut()
                .zip(rhs.coeffs.iter())
                .for_each(|(lhs, rhs)| {
                    *lhs -= rhs;
                });

            self
        } else {
            self - rhs.clone()
        }
    }
}

impl<R: Semiring> Mul for DynamicPolynomial<R> {
    type Output = Self;

    #[allow(clippy::arithmetic_side_effects)]
    fn mul(self, rhs: Self) -> Self::Output {
        &self * &rhs
    }
}

impl<R: Semiring> Mul<&Self> for DynamicPolynomial<R> {
    type Output = Self;

    #[allow(clippy::arithmetic_side_effects)]
    fn mul(self, rhs: &Self) -> Self::Output {
        &self * rhs
    }
}

impl<'a, R: Semiring> Mul<&'a DynamicPolynomial<R>> for &'a DynamicPolynomial<R> {
    type Output = DynamicPolynomial<R>;

    #[allow(clippy::arithmetic_side_effects)]
    fn mul(self, rhs: Self) -> Self::Output {
        if self.is_zero() || rhs.is_zero() {
            return Self::Output::zero();
        }

        let degree = (self.coeffs.len() - 1) + (rhs.coeffs.len() - 1);
        let mut coeffs = Vec::with_capacity(degree + 1);

        mul_schoolbook_unchecked(&self.coeffs, &rhs.coeffs, coeffs.spare_capacity_mut());

        // Safety: the multiplication algorithm should fill in the entire spare
        // capacity.
        unsafe {
            coeffs.set_len(degree + 1);
        }

        Self::Output { coeffs }
    }
}

impl<R: Semiring> AddAssign for DynamicPolynomial<R> {
    #[allow(clippy::arithmetic_side_effects)] // by definition
    fn add_assign(&mut self, rhs: Self) {
        if self.coeffs.len() < rhs.coeffs.len() {
            let res = rhs.add(&*self);

            *self = res;
        } else {
            self.coeffs
                .iter_mut()
                .zip(rhs.coeffs.iter())
                .for_each(|(lhs, rhs)| {
                    *lhs += rhs;
                });
        }
    }
}

impl<R: Semiring> AddAssign<&Self> for DynamicPolynomial<R> {
    #[allow(clippy::arithmetic_side_effects)] // by definition
    fn add_assign(&mut self, rhs: &Self) {
        if self.coeffs.len() < rhs.coeffs.len() {
            self.add_assign(rhs.clone());
        } else {
            self.coeffs
                .iter_mut()
                .zip(rhs.coeffs.iter())
                .for_each(|(lhs, rhs)| {
                    *lhs += rhs;
                });
        }
    }
}

impl<R: Ring> SubAssign<&Self> for DynamicPolynomial<R> {
    #[allow(clippy::arithmetic_side_effects)] // by definition
    fn sub_assign(&mut self, rhs: &Self) {
        if self.coeffs.len() < rhs.coeffs.len() {
            self.sub_assign(rhs.clone());
        } else {
            self.coeffs
                .iter_mut()
                .zip(rhs.coeffs.iter())
                .for_each(|(lhs, rhs)| {
                    *lhs -= rhs;
                });
        }
    }
}

impl<R: Ring> SubAssign for DynamicPolynomial<R> {
    #[allow(clippy::arithmetic_side_effects)] // by definition
    fn sub_assign(&mut self, rhs: Self) {
        if self.coeffs.len() < rhs.coeffs.len() {
            let lhs_len = self.coeffs.len();
            let mut rhs = rhs;

            rhs.coeffs
                .iter_mut()
                .zip(self.coeffs.iter())
                .for_each(|(rhs, lhs)| {
                    *rhs = lhs.clone() - &*rhs;
                });

            rhs.coeffs.iter_mut().skip(lhs_len).for_each(|rhs| {
                *rhs = rhs.clone().neg();
            });

            *self = rhs;
        } else {
            self.coeffs
                .iter_mut()
                .zip(rhs.coeffs.iter())
                .for_each(|(lhs, rhs)| {
                    *lhs -= rhs;
                });
        }
    }
}

impl<R: Semiring> MulAssign for DynamicPolynomial<R> {
    #[allow(clippy::arithmetic_side_effects)] // by definition
    fn mul_assign(&mut self, rhs: Self) {
        let res = rhs * &*self;

        *self = res
    }
}

impl<R: Semiring> MulAssign<&Self> for DynamicPolynomial<R> {
    #[allow(clippy::arithmetic_side_effects)] // by definition
    fn mul_assign(&mut self, rhs: &Self) {
        let res = &*self * rhs;

        *self = res;
    }
}

impl<R: Semiring> CheckedAdd for DynamicPolynomial<R> {
    fn checked_add(&self, rhs: &Self) -> Option<Self> {
        if self.coeffs.len() < rhs.coeffs.len() {
            let mut res = rhs.clone();

            res.coeffs
                .iter_mut()
                .zip(self.coeffs.iter())
                .try_for_each(|(lhs, rhs)| {
                    *lhs = lhs.checked_add(rhs)?;
                    Some(())
                })?;

            Some(res)
        } else {
            let mut res = self.clone();

            res.coeffs
                .iter_mut()
                .zip(rhs.coeffs.iter())
                .try_for_each(|(lhs, rhs)| {
                    *lhs = lhs.checked_add(rhs)?;
                    Some(())
                })?;

            Some(res)
        }
    }
}

impl<R: Ring> CheckedSub for DynamicPolynomial<R> {
    fn checked_sub(&self, rhs: &Self) -> Option<Self> {
        if self.coeffs.len() < rhs.coeffs.len() {
            let mut res = rhs.clone();

            res.coeffs
                .iter_mut()
                .zip(self.coeffs.iter())
                .try_for_each(|(rhs, lhs)| {
                    *rhs = lhs.checked_sub(rhs)?;
                    Some(())
                })?;

            res.coeffs
                .iter_mut()
                .skip(self.coeffs.len())
                .try_for_each(|res| {
                    *res = res.checked_neg()?;
                    Some(())
                })?;

            Some(res)
        } else {
            let mut res = self.clone();

            res.coeffs
                .iter_mut()
                .zip(rhs.coeffs.iter())
                .try_for_each(|(lhs, rhs)| {
                    *lhs = lhs.checked_sub(rhs)?;
                    Some(())
                })?;

            Some(res)
        }
    }
}

impl<R: Semiring> CheckedMul for DynamicPolynomial<R> {
    #[allow(clippy::arithmetic_side_effects)] // degrees normally shouldn't be that large
    fn checked_mul(&self, rhs: &Self) -> Option<Self> {
        if self.is_zero() || rhs.is_zero() {
            return Some(Self::zero());
        }

        let degree = (self.coeffs.len() - 1) + (rhs.coeffs.len() - 1);
        let mut coeffs = Vec::with_capacity(degree + 1);

        mul_schoolbook_checked(&self.coeffs, &rhs.coeffs, coeffs.spare_capacity_mut())?;

        // Safety: the multiplication algorithm should fill in the entire spare
        // capacity.
        unsafe {
            coeffs.set_len(degree + 1);
        }

        Some(Self::Output { coeffs })
    }
}

impl<R: Ring> Neg for DynamicPolynomial<R> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            coeffs: self.coeffs.into_iter().map(|coeff| coeff.neg()).collect(),
        }
    }
}

impl<R: Ring> CheckedNeg for DynamicPolynomial<R> {
    fn checked_neg(&self) -> Option<Self> {
        Some(Self {
            coeffs: self
                .coeffs
                .iter()
                .map(|coeff| coeff.checked_neg())
                .collect::<Option<Vec<_>>>()?,
        })
    }
}

impl<R: Ring> Semiring for DynamicPolynomial<R> {}
impl<R: Ring> Ring for DynamicPolynomial<R> {}

impl<R: Semiring, const DEGREE_PLUS_ONE: usize> From<DensePolynomial<R, DEGREE_PLUS_ONE>>
    for DynamicPolynomial<R>
{
    fn from(dense_poly: DensePolynomial<R, DEGREE_PLUS_ONE>) -> Self {
        Self {
            coeffs: Vec::from(dense_poly.coeffs),
        }
    }
}

impl<const DEGREE_PLUS_ONE: usize> From<BinaryPoly<DEGREE_PLUS_ONE>>
    for DynamicPolynomial<Boolean>
{
    fn from(binary_poly: BinaryPoly<DEGREE_PLUS_ONE>) -> Self {
        Self::from(DensePolynomial::from(binary_poly))
    }
}

impl<R: Semiring> Polynomial<R> for DynamicPolynomial<R> {
    const DEGREE_BOUND: usize = usize::MAX;
}

impl<R: Semiring> EvaluatablePolynomial<R, R> for DynamicPolynomial<R> {
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

impl<R, F> ProjectableToField<F> for DynamicPolynomial<R>
where
    R: Semiring,
    F: PrimeField + for<'a> FromWithConfig<&'a R> + for<'a> MulByScalar<&'a F> + 'static,
{
    #![allow(clippy::arithmetic_side_effects)] // False alert, field operations are safe
    fn prepare_projection(
        sampled_value: &F,
    ) -> impl Fn(&DynamicPolynomial<R>) -> F + Send + Sync + 'static {
        let sampled_value = sampled_value.clone();
        let field_cfg = sampled_value.cfg().clone();

        move |poly: &DynamicPolynomial<R>| {
            let coeffs: Vec<F> = poly
                .coeffs
                .iter()
                .map(|v| v.into_with_cfg(&field_cfg))
                .collect();

            let poly2 = DynamicPolynomial { coeffs };
            poly2
                .evaluate_at_point(&sampled_value)
                .expect("Failed to evaluate polynomial at point")
        }
    }
}

#[cfg(test)]
mod tests {
    use crypto_primitives::crypto_bigint_int::Int;

    use super::*;

    #[test]
    fn new_creates_correctly() {
        assert_eq!(
            DynamicPolynomial::new_trimmed([1i32, 2i32, 3i32, 0, 0]),
            DynamicPolynomial {
                coeffs: vec![1, 2, 3]
            }
        );
    }

    fn get_2_test_polynomial() -> (DynamicPolynomial<Int<4>>, DynamicPolynomial<Int<4>>) {
        let x = DynamicPolynomial::new(vec![
            Int::from_i8(2),
            Int::from_i8(0),
            Int::from_i8(2),
            Int::from_i8(0),
            Int::from_i8(0),
        ]);

        let y = DynamicPolynomial::new(vec![Int::from_i8(1), Int::from_i8(2), Int::from_i8(3)]);

        (x, y)
    }

    #[test]
    fn add_zero() {
        assert_eq!(
            DynamicPolynomial::<Int<4>>::ZERO + DynamicPolynomial::ZERO,
            DynamicPolynomial::ZERO
        );

        let x = DynamicPolynomial::new(vec![
            Int::<4>::from_i8(2),
            Int::from_i8(0),
            Int::from_i8(2),
            Int::from_i8(0),
            Int::from_i8(0),
        ]);

        assert_eq!(x.clone() + &DynamicPolynomial::ZERO, x);
        assert_eq!(DynamicPolynomial::ZERO + x.clone(), x);
        assert_eq!(DynamicPolynomial::ZERO + &x, x);

        let mut y = x.clone();

        y += DynamicPolynomial::ZERO;

        assert_eq!(y, x);

        y += &DynamicPolynomial::ZERO;

        assert_eq!(y, x);
    }

    #[test]
    fn addition_is_correct() {
        let (x, y) = get_2_test_polynomial();

        let res = DynamicPolynomial::new(vec![
            Int::<4>::from_i8(3),
            Int::from_i8(2),
            Int::from_i8(5),
            Int::ZERO,
            Int::ZERO,
        ]);

        assert_eq!(x.clone() + &y, res);
        assert_eq!(y.clone() + &x, res);
        assert_eq!(x.clone() + y.clone(), res);
        assert_eq!(x.checked_add(&y), Some(res.clone()));

        let mut z = x.clone();
        z += y.clone();
        assert_eq!(z, res);

        let mut z = x.clone();
        z += &y;
        assert_eq!(z, res);

        let mut z = y.clone();
        z += x.clone();
        assert_eq!(z, res);

        let mut z = y.clone();
        z += &x;
        assert_eq!(z, res);
    }

    #[test]
    fn subtraction_is_correct() {
        let (x, y) = get_2_test_polynomial();

        let res = DynamicPolynomial::new(vec![
            Int::<4>::from_i8(1),
            Int::from_i8(-2),
            Int::from_i8(-1),
            Int::ZERO,
            Int::ZERO,
        ]);

        assert_eq!(x.clone() - &y, res);
        assert_eq!(x.clone() - y.clone(), res);
        assert_eq!(x.checked_sub(&y), Some(res.clone()));

        let mut z = x.clone();
        z -= y.clone();
        assert_eq!(z, res);

        let mut z = x.clone();
        z -= &y;
        assert_eq!(z, res);

        let x = y;
        let y = DynamicPolynomial::new(vec![
            Int::from_i8(2),
            Int::from_i8(0),
            Int::from_i8(2),
            Int::from_i8(-1),
            Int::ZERO,
        ]);

        let res = DynamicPolynomial::new(vec![
            Int::<4>::from_i8(-1),
            Int::from_i8(2),
            Int::from_i8(1),
            Int::from_i8(1),
            Int::ZERO,
        ]);

        assert_eq!(x.clone() - &y, res);
        assert_eq!(x.clone() - y.clone(), res);
        assert_eq!(x.checked_sub(&y), Some(res.clone()));

        let mut z = x.clone();
        z -= y.clone();
        assert_eq!(z, res);

        let mut z = x.clone();
        z -= &y;
        assert_eq!(z, res);
    }

    #[test]
    fn mul_zero() {
        assert_eq!(
            DynamicPolynomial::<Int<4>>::ZERO * DynamicPolynomial::ZERO,
            DynamicPolynomial::ZERO
        );

        let x = DynamicPolynomial::new(vec![
            Int::<4>::from_i8(2),
            Int::from_i8(0),
            Int::from_i8(2),
            Int::from_i8(0),
            Int::from_i8(0),
        ]);

        assert_eq!(
            x.clone() * &DynamicPolynomial::ZERO,
            DynamicPolynomial::ZERO
        );
        assert_eq!(DynamicPolynomial::ZERO * x.clone(), DynamicPolynomial::ZERO);
        assert_eq!(DynamicPolynomial::ZERO * &x, DynamicPolynomial::ZERO);

        let mut y = x.clone();

        y *= DynamicPolynomial::ZERO;

        assert_eq!(y, DynamicPolynomial::ZERO);

        y *= &DynamicPolynomial::ZERO;

        assert_eq!(y, DynamicPolynomial::ZERO);
    }

    #[test]
    fn multiplication_is_correct() {
        let (x, y) = get_2_test_polynomial();

        let res = DynamicPolynomial::new(vec![
            Int::<4>::from_i8(2),
            Int::from_i8(4),
            Int::from_i8(8),
            Int::from_i8(4),
            Int::from_i8(6),
            Int::ZERO,
            Int::ZERO,
        ]);

        assert_eq!(x.clone() * y.clone(), res);
        assert_eq!(x.clone() * &y, res);
        assert_eq!(&x * &y, res);
        assert_eq!(y.clone() * x.clone(), res);
        assert_eq!(y.clone() * &x, res);
        assert_eq!(&y * &x, res);
        assert_eq!(x.checked_mul(&y), Some(res.clone()));
        assert_eq!(y.checked_mul(&x), Some(res.clone()));

        let mut z = x.clone();
        z *= y.clone();
        assert_eq!(z, res);

        let mut z = x.clone();
        z *= &y;
        assert_eq!(z, res);

        let mut z = y.clone();
        z *= x.clone();
        assert_eq!(z, res);

        let mut z = y.clone();
        z *= &x;
        assert_eq!(z, res);
    }

    #[test]
    fn test_trim() {
        let mut x = DynamicPolynomial::<Int<4>>::new([
            Int::ZERO,
            Int::ZERO,
            Int::ZERO,
            Int::ZERO,
            Int::ZERO,
        ]);
        x.trim();
        assert_eq!(x, DynamicPolynomial::ZERO);

        let mut x = DynamicPolynomial::<Int<4>>::new([
            Int::from_i8(2),
            Int::from_i8(3),
            Int::ZERO,
            Int::ZERO,
            Int::ZERO,
        ]);
        x.trim();
        assert_eq!(
            x,
            DynamicPolynomial::<Int<4>>::new([Int::from_i8(2), Int::from_i8(3),])
        );

        let mut x = DynamicPolynomial::<Int<4>>::new([
            Int::ZERO,
            Int::ZERO,
            Int::ZERO,
            Int::ZERO,
            Int::ZERO,
        ]);
        x.trim_with_zero(&Int::ZERO);
        assert_eq!(x, DynamicPolynomial::ZERO);

        let mut x = DynamicPolynomial::<Int<4>>::new([
            Int::from_i8(2),
            Int::from_i8(3),
            Int::ZERO,
            Int::ZERO,
            Int::ZERO,
        ]);
        x.trim_with_zero(&Int::ZERO);
        assert_eq!(
            x,
            DynamicPolynomial::<Int<4>>::new([Int::from_i8(2), Int::from_i8(3),])
        );
    }
}
