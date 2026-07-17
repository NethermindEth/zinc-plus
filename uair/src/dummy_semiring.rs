use std::{
    fmt::{Debug, Display},
    iter::{Product, Sum},
    ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign},
};

use crypto_primitives::FixedConfig;
use num_traits::{CheckedAdd, CheckedMul, CheckedSub, ConstOne, ConstZero, One, Pow, Zero};

/// A dummy type implementing the `Semiring` trait.
/// Used for `ConstraintCounter` to have something
/// that implements `Semiring` (and hence bridges to a `SemiringConfig` via
/// `FixedConfig`) but has zero-cost operations. Can be used in other contexts
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

macro_rules! impl_op_assign {
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

impl_op_assign!(AddAssign, add_assign);
impl_op_assign!(SubAssign, sub_assign);
impl_op_assign!(MulAssign, mul_assign);

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

impl Pow<u32> for DummySemiring {
    type Output = Self;

    fn pow(self, _rhs: u32) -> Self::Output {
        DummySemiring
    }
}

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

impl From<bool> for DummySemiring {
    fn from(_value: bool) -> Self {
        DummySemiring
    }
}

pub type DummySemiringConfig = FixedConfig<DummySemiring>;
pub static DUMMY_SEMIRING_CONFIG: FixedConfig<DummySemiring> = FixedConfig::const_default();
