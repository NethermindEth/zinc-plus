use std::{
    fmt::{Debug, Display},
    ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign},
};

use crate::{ConstraintBuilder, TraceRow, Uair, ideal::ImpossibleIdeal};
use crypto_primitives::FixedConfig;
use num_traits::{CheckedAdd, CheckedMul, CheckedSub, One, Pow, Zero};
use zinc_utils::add;

/// Compute the maximum number of multiplicands
/// in products of witness elements in the UAIR `U`.
pub fn count_max_degree<U: Uair>() -> usize {
    count_constraint_degrees_flattened::<U>()
        .into_iter()
        .max()
        .unwrap_or(0)
}

/// Compute the degree of each individual constraint in the UAIR `U`.
///
/// Returns a `Vec<usize>` where the i-th element is the degree of the i-th
/// emitted constraint, in emission order. Under the
/// [`crate::ConstraintBuilder::assert_in_fq_ideal`] ordering convention, all
/// $Q[X]$ degrees appear first, followed by all
/// $F_{q_i}[X]$ degrees.
pub fn count_constraint_degrees_flattened<U: Uair>() -> Vec<usize> {
    let split = count_constraint_degrees::<U>();
    let mut all = split.q_degrees;
    all.extend(split.fq_degrees.into_iter().flatten());
    all
}

/// Compute the per-family degrees of each constraint in `U`.
pub fn count_constraint_degrees<U: Uair>() -> ConstraintDegreeCollector {
    let mut dc = ConstraintDegreeCollector::default();

    let sig = U::signature();
    let (up_dummy, down_dummy) = sig.dummy_rows(DegreeCountingSemiring::var());
    let up_row = TraceRow::from_slice_with_layout(&up_dummy, sig.total_cols().as_column_layout());
    let down_row =
        TraceRow::from_slice_with_layout(&down_dummy, sig.down_cols().as_column_layout());

    U::constrain_general(
        &mut dc,
        &FixedConfig::<DegreeCountingSemiring>::default(),
        up_row,
        down_row,
        |_| DegreeCountingSemiring::scalar(),
        |x, _| Some(*x),
        |_| ImpossibleIdeal,
        |_| ImpossibleIdeal,
    );

    dc
}

/// Collects the degree of each constraint in a UAIR by implementing the
/// `ConstraintBuilder` trait.
#[derive(Debug, Default)]
pub struct ConstraintDegreeCollector {
    pub q_degrees: Vec<usize>,
    pub fq_degrees: Vec<Vec<usize>>,
}

impl ConstraintBuilder for ConstraintDegreeCollector {
    type Expr = DegreeCountingSemiring;
    type Ideal = ImpossibleIdeal;
    type FqIdeal = ImpossibleIdeal;

    fn assert_in_ideal(&mut self, expr: Self::Expr, _ideal: &Self::Ideal) {
        self.q_degrees.push(expr.0);
    }

    fn assert_zero(&mut self, expr: Self::Expr) {
        self.q_degrees.push(expr.0);
    }

    fn assert_in_fq_ideal(&mut self, prime_idx: usize, expr: Self::Expr, _ideal: &Self::FqIdeal) {
        if self.fq_degrees.len() <= prime_idx {
            self.fq_degrees.resize(add!(prime_idx, 1), Vec::new());
        }
        self.fq_degrees[prime_idx].push(expr.0);
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct DegreeCountingSemiring(usize);

impl DegreeCountingSemiring {
    pub fn var() -> Self {
        DegreeCountingSemiring(1)
    }

    pub fn scalar() -> Self {
        DegreeCountingSemiring(0)
    }
}

impl Display for DegreeCountingSemiring {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(&self, f)
    }
}

macro_rules! impl_binary_additive_op {
    ($trait:ident, $op:ident) => {
        impl $trait<&DegreeCountingSemiring> for DegreeCountingSemiring {
            type Output = Self;

            #[inline(always)]
            fn $op(self, rhs: &DegreeCountingSemiring) -> Self::Output {
                DegreeCountingSemiring(std::cmp::max(self.0, rhs.0))
            }
        }

        impl $trait<DegreeCountingSemiring> for DegreeCountingSemiring {
            type Output = Self;

            #[inline(always)]
            fn $op(self, rhs: DegreeCountingSemiring) -> Self::Output {
                self.$op(&rhs)
            }
        }
    };
}

impl_binary_additive_op!(Add, add);
impl_binary_additive_op!(Sub, sub);

impl Mul<&Self> for DegreeCountingSemiring {
    type Output = Self;

    #[allow(clippy::arithmetic_side_effects, clippy::suspicious_arithmetic_impl)]
    #[inline(always)]
    fn mul(self, rhs: &Self) -> Self::Output {
        DegreeCountingSemiring(self.0 + rhs.0)
    }
}

impl Mul<Self> for DegreeCountingSemiring {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        self.mul(&rhs)
    }
}

macro_rules! impl_additive_op_assign {
    ($trait:ident, $op:ident) => {
        impl $trait<&DegreeCountingSemiring> for DegreeCountingSemiring {
            #[inline(always)]
            fn $op(&mut self, rhs: &DegreeCountingSemiring) {
                self.0 = std::cmp::max(self.0, rhs.0);
            }
        }

        impl $trait<DegreeCountingSemiring> for DegreeCountingSemiring {
            #[inline(always)]
            fn $op(&mut self, rhs: DegreeCountingSemiring) {
                self.$op(&rhs);
            }
        }
    };
}

impl_additive_op_assign!(AddAssign, add_assign);
impl_additive_op_assign!(SubAssign, sub_assign);

impl MulAssign<&Self> for DegreeCountingSemiring {
    #[allow(clippy::arithmetic_side_effects, clippy::suspicious_op_assign_impl)]
    #[inline(always)]
    fn mul_assign(&mut self, rhs: &Self) {
        self.0 += rhs.0;
    }
}

impl MulAssign<Self> for DegreeCountingSemiring {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        self.add_assign(&rhs);
    }
}

macro_rules! impl_checked_additive_op {
    ($trait:ident, $op:ident) => {
        impl $trait for DegreeCountingSemiring {
            #[inline(always)]
            fn $op(&self, rhs: &DegreeCountingSemiring) -> Option<Self::Output> {
                Some(DegreeCountingSemiring(std::cmp::max(self.0, rhs.0)))
            }
        }
    };
}

impl_checked_additive_op!(CheckedAdd, checked_add);
impl_checked_additive_op!(CheckedSub, checked_sub);

impl CheckedMul for DegreeCountingSemiring {
    #[inline(always)]
    fn checked_mul(&self, rhs: &Self) -> Option<Self> {
        Some(DegreeCountingSemiring(self.0.checked_add(rhs.0)?))
    }
}

impl Zero for DegreeCountingSemiring {
    #[inline(always)]
    fn zero() -> Self {
        Self::scalar()
    }

    #[inline(always)]
    fn is_zero(&self) -> bool {
        self.0 == 0
    }
}

impl One for DegreeCountingSemiring {
    #[inline(always)]
    fn one() -> Self {
        Self::scalar()
    }
}

impl From<bool> for DegreeCountingSemiring {
    #[inline(always)]
    fn from(_value: bool) -> Self {
        Self::scalar()
    }
}

impl Pow<u32> for DegreeCountingSemiring {
    type Output = Self;

    #[allow(clippy::arithmetic_side_effects)] // degrees are small
    #[inline(always)]
    fn pow(self, exp: u32) -> Self {
        DegreeCountingSemiring(self.0 * exp as usize)
    }
}

impl std::iter::Sum for DegreeCountingSemiring {
    #[allow(clippy::arithmetic_side_effects)] // degrees are small
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::scalar(), |acc, x| acc + x)
    }
}

impl<'a> std::iter::Sum<&'a Self> for DegreeCountingSemiring {
    #[allow(clippy::arithmetic_side_effects)] // degrees are small
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::scalar(), |acc, x| acc + x)
    }
}

impl std::iter::Product for DegreeCountingSemiring {
    #[allow(clippy::arithmetic_side_effects)] // degrees are small
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::scalar(), |acc, x| acc * x)
    }
}

impl<'a> std::iter::Product<&'a Self> for DegreeCountingSemiring {
    #[allow(clippy::arithmetic_side_effects)] // degrees are small
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::scalar(), |acc, x| acc * x)
    }
}
