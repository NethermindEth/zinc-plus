use std::{
    fmt::{Debug, Display},
    iter::{Product, Sum},
    ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign},
};

use crypto_primitives::Semiring;
use num_traits::{CheckedAdd, CheckedMul, CheckedSub, ConstOne, ConstZero, One, Zero};
use zinc_utils::{from_ref::FromRef, mul_by_scalar::MulByScalar};

/// A dummy type implementing `FixedSemiring` trait.
/// Used for `ConstraintCounter` to have something
/// that implements `FixedSemiring` but has zero-cost
/// operations. Can be used in other contexts
/// where operations on expression should be ignored.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq, Hash)]
pub struct DummySemiring;

impl Display for DummySemiring {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(&self, f)
    }
}

macro_rules! impl_binary_op {
    ($trait:ident, $op:ident) => {
        impl $trait<&DummySemiring> for DummySemiring {
            type Output = Self;

            #[inline(always)]
            fn $op(self, _rhs: &DummySemiring) -> Self::Output {
                DummySemiring
            }
        }

        impl $trait<DummySemiring> for DummySemiring {
            type Output = Self;

            #[inline(always)]
            fn $op(self, _rhs: DummySemiring) -> Self::Output {
                DummySemiring
            }
        }
    };
}

impl_binary_op!(Add, add);
impl_binary_op!(Sub, sub);
impl_binary_op!(Mul, mul);

macro_rules! impl_op_assing {
    ($trait:ident, $op:ident) => {
        impl $trait<&DummySemiring> for DummySemiring {
            #[inline(always)]
            fn $op(&mut self, _rhs: &DummySemiring) {}
        }

        impl $trait<DummySemiring> for DummySemiring {
            #[inline(always)]
            fn $op(&mut self, _rhs: DummySemiring) {}
        }
    };
}

impl_op_assing!(AddAssign, add_assign);
impl_op_assing!(SubAssign, sub_assign);
impl_op_assing!(MulAssign, mul_assign);

macro_rules! impl_checked_op {
    ($trait:ident, $op:ident) => {
        impl $trait for DummySemiring {
            #[inline(always)]
            fn $op(&self, _rhs: &Self) -> Option<Self> {
                Some(DummySemiring)
            }
        }
    };
}

impl_checked_op!(CheckedAdd, checked_add);
impl_checked_op!(CheckedSub, checked_sub);
impl_checked_op!(CheckedMul, checked_mul);

impl Sum for DummySemiring {
    #[inline(always)]
    fn sum<I: Iterator<Item = Self>>(_iter: I) -> Self {
        DummySemiring
    }
}

impl Product for DummySemiring {
    #[inline(always)]
    fn product<I: Iterator<Item = Self>>(_iter: I) -> Self {
        DummySemiring
    }
}

impl<'a> Sum<&'a DummySemiring> for DummySemiring {
    #[inline(always)]
    fn sum<I: Iterator<Item = &'a DummySemiring>>(_iter: I) -> Self {
        DummySemiring
    }
}

impl<'a> Product<&'a DummySemiring> for DummySemiring {
    #[inline(always)]
    fn product<I: Iterator<Item = &'a DummySemiring>>(_iter: I) -> Self {
        DummySemiring
    }
}

impl Zero for DummySemiring {
    fn zero() -> Self {
        DummySemiring
    }

    fn is_zero(&self) -> bool {
        true
    }
}

impl One for DummySemiring {
    fn one() -> Self {
        DummySemiring
    }
}

impl ConstZero for DummySemiring {
    const ZERO: Self = DummySemiring;
}

impl ConstOne for DummySemiring {
    const ONE: Self = DummySemiring;
}

impl Semiring for DummySemiring {}

impl<T> FromRef<T> for DummySemiring {
    #[inline(always)]
    fn from_ref(_value: &T) -> Self {
        DummySemiring
    }
}

impl<T> MulByScalar<&T> for DummySemiring {
    #[inline(always)]
    fn mul_by_scalar(&self, _rhs: &T) -> Option<Self> {
        Some(DummySemiring)
    }
}
