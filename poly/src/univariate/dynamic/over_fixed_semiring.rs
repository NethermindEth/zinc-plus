use crypto_primitives::{
    FixedSemiring, FromWithConfig, IntoWithConfig, PrimeField, Ring, Semiring, boolean::Boolean,
};
use derive_more::From;
use itertools::Itertools;
use num_traits::{CheckedAdd, CheckedMul, CheckedNeg, CheckedSub, ConstZero, One, Zero};
use std::{
    fmt::Display,
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};
use zinc_utils::{
    CHECKED, UNCHECKED, mul_by_scalar::MulByScalar, projectable_to_field::ProjectableToField,
};

use crate::{
    EvaluatablePolynomial, EvaluationError, Polynomial,
    univariate::{
        binary::BinaryPoly,
        dense::DensePolynomial,
        dynamic::{multiplication::mul_schoolbook, over_field::DynamicPolynomialF},
    },
};

/// Polynomials of dynamic degree. The implementation
/// is tailored to work with `FixedSemiring`s. To be used
/// in UAIR and PIOP where ZIP+ degree bound
/// is not observed anymore.
///
/// Note that operations involving dynamic polynomials
/// do not trim leading zeros meaning
/// one can end up with unequal objects of the type
/// `DynamicPoly<R>` that represent equal polynomials,
/// therefore `trim` has to be called before checking
/// equality.
#[derive(Debug, Default, Clone, From, Hash, PartialEq, Eq)]
pub struct DynamicPolynomialFS<R: FixedSemiring> {
    pub coeffs: Vec<R>,
}

impl<R: FixedSemiring> DynamicPolynomialFS<R> {
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

impl<R: FixedSemiring> Display for DynamicPolynomialFS<R> {
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

impl<R: FixedSemiring> Zero for DynamicPolynomialFS<R> {
    #[inline(always)]
    fn zero() -> Self {
        Default::default()
    }

    fn is_zero(&self) -> bool {
        self.coeffs.iter().all(|coeff| coeff.is_zero())
    }
}

impl<R: FixedSemiring> ConstZero for DynamicPolynomialFS<R> {
    const ZERO: Self = Self { coeffs: Vec::new() };
}

impl<R: FixedSemiring> One for DynamicPolynomialFS<R> {
    fn one() -> Self {
        Self {
            coeffs: vec![R::one()],
        }
    }
}

impl<R: FixedSemiring + Ring> Neg for DynamicPolynomialFS<R> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            coeffs: self.coeffs.into_iter().map(|coeff| coeff.neg()).collect(),
        }
    }
}

impl<R: FixedSemiring> Add for DynamicPolynomialFS<R> {
    type Output = Self;

    #[allow(clippy::arithmetic_side_effects)]
    fn add(self, rhs: Self) -> Self::Output {
        self.add(&rhs)
    }
}

impl<R: FixedSemiring> Add<&Self> for DynamicPolynomialFS<R> {
    type Output = Self;

    #[allow(clippy::arithmetic_side_effects)]
    fn add(mut self, rhs: &Self) -> Self::Output {
        if self.coeffs.len() < rhs.coeffs.len() {
            self.coeffs.resize(rhs.coeffs.len(), R::zero());
        }

        self.coeffs
            .iter_mut()
            .zip(&rhs.coeffs)
            .for_each(|(lhs, rhs)| {
                *lhs += rhs;
            });

        self
    }
}

impl<R: FixedSemiring> Sub for DynamicPolynomialFS<R> {
    type Output = Self;

    #[allow(clippy::arithmetic_side_effects)]
    fn sub(self, rhs: Self) -> Self::Output {
        self.sub(&rhs)
    }
}

impl<R: FixedSemiring> Sub<&Self> for DynamicPolynomialFS<R> {
    type Output = Self;

    #[allow(clippy::arithmetic_side_effects)]
    fn sub(mut self, rhs: &Self) -> Self::Output {
        if self.coeffs.len() < rhs.coeffs.len() {
            self.coeffs.resize(rhs.coeffs.len(), R::zero());
        }

        self.coeffs
            .iter_mut()
            .zip(&rhs.coeffs)
            .for_each(|(lhs, rhs)| {
                *lhs -= rhs;
            });

        self
    }
}

impl<R: FixedSemiring> Mul for DynamicPolynomialFS<R> {
    type Output = Self;

    #[allow(clippy::arithmetic_side_effects)]
    fn mul(self, rhs: Self) -> Self::Output {
        &self * &rhs
    }
}

impl<R: FixedSemiring> Mul<&Self> for DynamicPolynomialFS<R> {
    type Output = Self;

    #[allow(clippy::arithmetic_side_effects)]
    fn mul(self, rhs: &Self) -> Self::Output {
        &self * rhs
    }
}

impl<'a, R: FixedSemiring> Mul<&'a DynamicPolynomialFS<R>> for &'a DynamicPolynomialFS<R> {
    type Output = DynamicPolynomialFS<R>;

    #[allow(clippy::arithmetic_side_effects)]
    fn mul(self, rhs: Self) -> Self::Output {
        if self.is_zero() || rhs.is_zero() {
            return Self::Output::zero();
        }

        let degree = (self.coeffs.len() - 1) + (rhs.coeffs.len() - 1);
        let mut coeffs = Vec::with_capacity(degree + 1);

        mul_schoolbook::<_, UNCHECKED>(&self.coeffs, &rhs.coeffs, coeffs.spare_capacity_mut());

        // Safety: the multiplication algorithm should fill in the entire spare
        // capacity.
        unsafe {
            coeffs.set_len(degree + 1);
        }

        Self::Output { coeffs }
    }
}

impl<R: FixedSemiring> CheckedAdd for DynamicPolynomialFS<R> {
    fn checked_add(&self, rhs: &Self) -> Option<Self> {
        let mut res = self.clone();

        if self.coeffs.len() < rhs.coeffs.len() {
            res.coeffs.resize(rhs.coeffs.len(), R::zero());
        }

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

impl<R: FixedSemiring> CheckedSub for DynamicPolynomialFS<R> {
    fn checked_sub(&self, rhs: &Self) -> Option<Self> {
        let mut res = self.clone();

        if self.coeffs.len() < rhs.coeffs.len() {
            res.coeffs.resize(rhs.coeffs.len(), R::zero());
        }

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

impl<R: FixedSemiring> CheckedMul for DynamicPolynomialFS<R> {
    #[allow(clippy::arithmetic_side_effects)] // degrees normally shouldn't be that large
    fn checked_mul(&self, rhs: &Self) -> Option<Self> {
        if self.is_zero() || rhs.is_zero() {
            return Some(Self::zero());
        }

        let degree = (self.coeffs.len() - 1) + (rhs.coeffs.len() - 1);
        let mut coeffs = Vec::with_capacity(degree + 1);

        mul_schoolbook::<_, CHECKED>(&self.coeffs, &rhs.coeffs, coeffs.spare_capacity_mut())?;

        // Safety: the multiplication algorithm should fill in the entire spare
        // capacity.
        unsafe {
            coeffs.set_len(degree + 1);
        }

        Some(Self::Output { coeffs })
    }
}

impl<R: FixedSemiring + Ring> CheckedNeg for DynamicPolynomialFS<R> {
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

impl<R: FixedSemiring> AddAssign for DynamicPolynomialFS<R> {
    #[allow(clippy::arithmetic_side_effects)] // by definition
    fn add_assign(&mut self, rhs: Self) {
        self.add_assign(&rhs);
    }
}

impl<R: FixedSemiring> AddAssign<&Self> for DynamicPolynomialFS<R> {
    #[allow(clippy::arithmetic_side_effects)] // by definition
    fn add_assign(&mut self, rhs: &Self) {
        if self.coeffs.len() < rhs.coeffs.len() {
            self.coeffs.resize(rhs.coeffs.len(), R::zero());
        }

        self.coeffs
            .iter_mut()
            .zip(&rhs.coeffs)
            .for_each(|(lhs, rhs)| {
                *lhs += rhs;
            });
    }
}

impl<R: FixedSemiring> SubAssign for DynamicPolynomialFS<R> {
    #[allow(clippy::arithmetic_side_effects)] // by definition
    fn sub_assign(&mut self, rhs: Self) {
        self.sub_assign(&rhs);
    }
}

impl<R: FixedSemiring> SubAssign<&Self> for DynamicPolynomialFS<R> {
    #[allow(clippy::arithmetic_side_effects)] // by definition
    fn sub_assign(&mut self, rhs: &Self) {
        if self.coeffs.len() < rhs.coeffs.len() {
            self.coeffs.resize(rhs.coeffs.len(), R::zero());
        }

        self.coeffs
            .iter_mut()
            .zip(rhs.coeffs.iter())
            .for_each(|(lhs, rhs)| {
                *lhs -= rhs;
            });
    }
}

impl<R: FixedSemiring> MulAssign for DynamicPolynomialFS<R> {
    #[allow(clippy::arithmetic_side_effects)] // by definition
    fn mul_assign(&mut self, rhs: Self) {
        let res = rhs * &*self;

        *self = res
    }
}

impl<R: FixedSemiring> MulAssign<&Self> for DynamicPolynomialFS<R> {
    #[allow(clippy::arithmetic_side_effects)] // by definition
    fn mul_assign(&mut self, rhs: &Self) {
        let res = &*self * rhs;

        *self = res;
    }
}

impl<R: FixedSemiring> Semiring for DynamicPolynomialFS<R> {}
impl<R: FixedSemiring + Ring> Ring for DynamicPolynomialFS<R> {}

impl<R: FixedSemiring, const DEGREE_PLUS_ONE: usize> From<DensePolynomial<R, DEGREE_PLUS_ONE>>
    for DynamicPolynomialFS<R>
{
    fn from(dense_poly: DensePolynomial<R, DEGREE_PLUS_ONE>) -> Self {
        Self {
            coeffs: Vec::from(dense_poly.coeffs),
        }
    }
}

impl<const DEGREE_PLUS_ONE: usize> From<BinaryPoly<DEGREE_PLUS_ONE>>
    for DynamicPolynomialFS<Boolean>
{
    fn from(binary_poly: BinaryPoly<DEGREE_PLUS_ONE>) -> Self {
        Self::from(DensePolynomial::from(binary_poly))
    }
}

impl<R: FixedSemiring> Polynomial<R> for DynamicPolynomialFS<R> {
    const DEGREE_BOUND: usize = usize::MAX;
}

impl<R: FixedSemiring> EvaluatablePolynomial<R, R> for DynamicPolynomialFS<R> {
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

impl<R, F> ProjectableToField<F> for DynamicPolynomialFS<R>
where
    R: FixedSemiring,
    F: PrimeField + for<'a> FromWithConfig<&'a R> + for<'a> MulByScalar<&'a F> + 'static,
{
    #![allow(clippy::arithmetic_side_effects)] // False alert, field operations are safe
    fn prepare_projection(
        sampled_value: &F,
    ) -> impl Fn(&DynamicPolynomialFS<R>) -> F + Send + Sync + 'static {
        let sampled_value = sampled_value.clone();
        let field_cfg = sampled_value.cfg().clone();

        move |poly: &DynamicPolynomialFS<R>| {
            let coeffs: Vec<F> = poly
                .coeffs
                .iter()
                .map(|v| v.into_with_cfg(&field_cfg))
                .collect();

            let poly2 = DynamicPolynomialF { coeffs };
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
            DynamicPolynomialFS::new_trimmed([1i32, 2i32, 3i32, 0, 0]),
            DynamicPolynomialFS {
                coeffs: vec![1, 2, 3]
            }
        );
    }

    fn get_2_test_polynomial() -> (DynamicPolynomialFS<Int<4>>, DynamicPolynomialFS<Int<4>>) {
        let x = DynamicPolynomialFS::new(vec![
            Int::from_i8(2),
            Int::from_i8(0),
            Int::from_i8(2),
            Int::from_i8(0),
            Int::from_i8(0),
        ]);

        let y = DynamicPolynomialFS::new(vec![Int::from_i8(1), Int::from_i8(2), Int::from_i8(3)]);

        (x, y)
    }

    #[test]
    fn add_zero() {
        assert_eq!(
            DynamicPolynomialFS::<Int<4>>::ZERO + DynamicPolynomialFS::ZERO,
            DynamicPolynomialFS::ZERO
        );

        let x = DynamicPolynomialFS::new(vec![
            Int::<4>::from_i8(2),
            Int::from_i8(0),
            Int::from_i8(2),
            Int::from_i8(0),
            Int::from_i8(0),
        ]);

        assert_eq!(x.clone() + &DynamicPolynomialFS::ZERO, x);
        assert_eq!(DynamicPolynomialFS::ZERO + x.clone(), x);
        assert_eq!(DynamicPolynomialFS::ZERO + &x, x);

        let mut y = x.clone();

        y += DynamicPolynomialFS::ZERO;

        assert_eq!(y, x);

        y += &DynamicPolynomialFS::ZERO;

        assert_eq!(y, x);
    }

    #[test]
    fn addition_is_correct() {
        let (x, y) = get_2_test_polynomial();

        let res = DynamicPolynomialFS::new(vec![
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

        let res = DynamicPolynomialFS::new(vec![
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
        let y = DynamicPolynomialFS::new(vec![
            Int::from_i8(2),
            Int::from_i8(0),
            Int::from_i8(2),
            Int::from_i8(-1),
            Int::ZERO,
        ]);

        let res = DynamicPolynomialFS::new(vec![
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
            DynamicPolynomialFS::<Int<4>>::ZERO * DynamicPolynomialFS::ZERO,
            DynamicPolynomialFS::ZERO
        );

        let x = DynamicPolynomialFS::new(vec![
            Int::<4>::from_i8(2),
            Int::from_i8(0),
            Int::from_i8(2),
            Int::from_i8(0),
            Int::from_i8(0),
        ]);

        assert_eq!(
            x.clone() * &DynamicPolynomialFS::ZERO,
            DynamicPolynomialFS::ZERO
        );
        assert_eq!(
            DynamicPolynomialFS::ZERO * x.clone(),
            DynamicPolynomialFS::ZERO
        );
        assert_eq!(DynamicPolynomialFS::ZERO * &x, DynamicPolynomialFS::ZERO);

        let mut y = x.clone();

        y *= DynamicPolynomialFS::ZERO;

        assert_eq!(y, DynamicPolynomialFS::ZERO);

        y *= &DynamicPolynomialFS::ZERO;

        assert_eq!(y, DynamicPolynomialFS::ZERO);
    }

    #[test]
    fn multiplication_is_correct() {
        let (x, y) = get_2_test_polynomial();

        let res = DynamicPolynomialFS::new(vec![
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
        let mut x = DynamicPolynomialFS::<Int<4>>::new([
            Int::ZERO,
            Int::ZERO,
            Int::ZERO,
            Int::ZERO,
            Int::ZERO,
        ]);
        x.trim();
        assert_eq!(x, DynamicPolynomialFS::ZERO);

        let mut x = DynamicPolynomialFS::<Int<4>>::new([
            Int::from_i8(2),
            Int::from_i8(3),
            Int::ZERO,
            Int::ZERO,
            Int::ZERO,
        ]);
        x.trim();
        assert_eq!(
            x,
            DynamicPolynomialFS::<Int<4>>::new([Int::from_i8(2), Int::from_i8(3),])
        );
    }
}
