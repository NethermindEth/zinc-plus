#[cfg(feature = "crypto_bigint")]
pub mod crypto_bigint_int;

use core::{
    fmt::{Debug, Display},
    hash::Hash,
    iter::{Product, Sum},
    ops::{
        Add, AddAssign, Mul, MulAssign, Rem, RemAssign, Shl, ShlAssign, Shr, ShrAssign, Sub,
        SubAssign,
    },
    str::FromStr,
};
use num_traits::{
    CheckedAdd, CheckedMul, CheckedNeg, CheckedRem, CheckedSub, ConstOne, ConstZero, One, Pow, Zero,
};

/// A ring is like a field without a multiplicative inverse or division.
pub trait Ring:
    // Core traits
    Sized
    + Debug
    + Display
    + Clone
    + PartialEq
    + Eq
    + Sync
    + Send
    + Hash
    // Arithmetic operations consuming rhs
    + CheckedNeg
    + CheckedAdd
    + CheckedSub
    + CheckedMul
    + AddAssign
    + SubAssign
    + MulAssign
    + Sum
    + Product
    // Arithmetic operations with rhs reference
    + for<'a> Add<&'a Self, Output=Self>
    + for<'a> Sub<&'a Self, Output=Self>
    + for<'a> Mul<&'a Self, Output=Self>
    + for<'a> AddAssign<&'a Self>
    + for<'a> SubAssign<&'a Self>
    + for<'a> MulAssign<&'a Self>
    + for<'a> Sum<&'a Self>
    + for<'a> Product<&'a Self>
    {}

/// Ring whose zero and one elements can be constructed from the type alone.
pub trait FixedRing: Ring + Default + Zero + One {}
impl<T> FixedRing for T where T: Ring + Default + Zero + One {}

pub trait ConstRing: FixedRing + ConstZero + ConstOne + From<bool> + From<i8> {}
impl<T> ConstRing for T where T: FixedRing + ConstZero + ConstOne + From<bool> + From<i8> {}

/// Ring of integers, usually denoted as `Z`.
pub trait IntRing: Ring + PartialOrd + Pow<u32> {
    fn is_odd(&self) -> bool;

    fn is_even(&self) -> bool;

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

pub trait IntRingWithShifts:
    IntRing + Shl<u32> + Shr<u32> + ShlAssign<u32> + ShrAssign<u32>
{
}
impl<T> IntRingWithShifts for T where
    T: IntRing + Shl<u32> + Shr<u32> + ShlAssign<u32> + ShrAssign<u32>
{
}

pub trait ConstIntRing: IntRing + ConstRing + FromStr {}
impl<T> ConstIntRing for T where T: IntRing + ConstRing + FromStr {}

macro_rules! primitive_int_ring {
    ($t:ident) => {
        impl Ring for $t {}
        impl IntRing for $t {
            fn is_odd(&self) -> bool {
                *self & 1 == 1
            }

            fn is_even(&self) -> bool {
                *self & 1 == 0
            }

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
