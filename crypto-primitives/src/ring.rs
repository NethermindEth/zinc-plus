#[cfg(feature = "crypto_bigint")]
pub mod crypto_bigint_int;

use crate::{ConstIntSemiring, FixedSemiring, IntSemiring, IntSemiringWithShifts, Semiring};
use core::ops::{Neg, Rem, RemAssign};
use num_traits::{CheckedNeg, CheckedRem};

/// A ring is a semiring with subtraction, meaning it has an additive inverse
/// for every element.
pub trait Ring: Semiring + Neg<Output = Self> + CheckedNeg {}

/// Ring whose zero and one elements can be constructed from the type alone.
pub trait FixedRing: Ring + FixedSemiring {}
impl<T> FixedRing for T where T: Ring + FixedSemiring {}

pub trait ConstRing: FixedRing + FixedSemiring {}
impl<T> ConstRing for T where T: FixedRing + FixedSemiring {}

/// Ring of integers, usually denoted as `Z`.
pub trait IntRing: Ring + IntSemiring {
    /// Checked absolute value. Computes `self.abs()`, returning `None` if `self
    /// == MIN`.
    fn checked_abs(&self) -> Option<Self>;

    fn is_positive(&self) -> bool;

    fn is_negative(&self) -> bool;
}

pub trait IntRingWithRem:
    IntRing + CheckedRem + RemAssign + for<'a> Rem<&'a Self> + for<'a> RemAssign<&'a Self>
{
}
impl<T> IntRingWithRem for T where
    T: IntRing + CheckedRem + RemAssign + for<'a> Rem<&'a Self> + for<'a> RemAssign<&'a Self>
{
}

pub trait IntRingWithShifts: IntRing + IntSemiringWithShifts {}
impl<T> IntRingWithShifts for T where T: IntRing + IntSemiringWithShifts {}

pub trait ConstIntRing: IntRing + ConstIntSemiring + From<i8> {}
impl<T> ConstIntRing for T where T: IntRing + ConstIntSemiring + From<i8> {}

macro_rules! primitive_int_ring {
    ($t:ident) => {
        impl Ring for $t {}
        impl IntRing for $t {
            fn checked_abs(&self) -> Option<Self> {
                $t::checked_abs(*self)
            }

            fn is_positive(&self) -> bool {
                *self > 0
            }

            fn is_negative(&self) -> bool {
                *self < 0
            }
        }
    };
}

primitive_int_ring!(i8);
primitive_int_ring!(i16);
primitive_int_ring!(i32);
primitive_int_ring!(i64);
primitive_int_ring!(i128);
