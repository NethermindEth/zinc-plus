use std::{
    fmt::{Debug, Display},
    ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign},
};

use crypto_primitives::Semiring;
use num_traits::{CheckedAdd, CheckedMul, CheckedSub};

use crate::{ConstraintBuilder, TraceRow, Uair, ideal::ImpossibleIdeal};

/// Compute the maximum number of multiplicands
/// in products of witness elements in the UAIR `U`.
pub fn count_max_degree<U: Uair>() -> usize {
    let mut dc = DegreeCounter::new();

    let up_and_down = vec![DegreeCountingSemiring::var(); U::signature().max_cols()];

    let trace_row = TraceRow {
        binary_poly: &up_and_down,
        arbitrary_poly: &up_and_down,
        int: &up_and_down,
    };

    U::constrain_general(
        &mut dc,
        trace_row,
        trace_row,
        |_| DegreeCountingSemiring::scalar(),
        |x, _| Some(*x),
        |_| ImpossibleIdeal,
    );

    dc.0
}

pub(crate) struct DegreeCounter(usize);

impl DegreeCounter {
    pub fn new() -> Self {
        Self(0)
    }
}

impl ConstraintBuilder for DegreeCounter {
    type Expr = DegreeCountingSemiring;

    type Ideal = ImpossibleIdeal;

    fn assert_in_ideal(&mut self, expr: Self::Expr, _ideal: &Self::Ideal) {
        self.0 = std::cmp::max(self.0, expr.0);
    }

    fn assert_zero(&mut self, expr: Self::Expr) {
        self.0 = std::cmp::max(self.0, expr.0);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct DegreeCountingSemiring(usize);

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

impl Semiring for DegreeCountingSemiring {}
