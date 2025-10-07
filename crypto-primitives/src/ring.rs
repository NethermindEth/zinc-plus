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
    + Default
    + Sync
    + Send
    + Zero
    + One
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

pub trait ConstRing: Ring + ConstZero + ConstOne {}
impl<T> ConstRing for T where T: Ring + ConstZero + ConstOne {}

/// Ring of integers, usually denoted as `Z`.
pub trait IntRing: Ring + Ord + Pow<u32> + From<bool> + From<i8> {}

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

pub trait ConstIntRing: IntRing + ConstRing {}
impl<T> ConstIntRing for T where T: IntRing + ConstRing {}

macro_rules! primitive_int_ring {
    ($t:ident) => {
        impl Ring for $t {}
        impl IntRing for $t {}
    };
}

primitive_int_ring!(i8);
primitive_int_ring!(i16);
primitive_int_ring!(i32);
primitive_int_ring!(i64);
primitive_int_ring!(i128);
