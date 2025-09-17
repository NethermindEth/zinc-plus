#[cfg(feature = "crypto_bigint")]
pub mod crypto_bigint_int;

use core::{
    fmt::{Debug, Display},
    iter::{Product, Sum},
    ops::{Add, AddAssign, Mul, MulAssign, Rem, RemAssign, Sub, SubAssign},
};
use num_traits::{CheckedAdd, CheckedMul, CheckedNeg, CheckedRem, CheckedShl, CheckedShr, CheckedSub, ConstOne, ConstZero, One, Pow, Zero};

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
    // Arithmetic operations consuming rhs
    + CheckedNeg
    + CheckedAdd
    + CheckedSub
    + CheckedMul
    + CheckedShl
    + CheckedShr
    + Pow<u32>
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

pub trait ConstRing:
    Ring
    + ConstZero
    + ConstOne
    {}

/// Ring of integers, usually denoted as `Z`.
pub trait IntRing:
    ConstRing
    + PartialOrd
    // Arithmetic operations consuming rhs
    + CheckedRem
    + RemAssign
    // Arithmetic operations with rhs reference
    + for<'a> Rem<&'a Self>
    + for<'a> RemAssign<&'a Self>
    {}

macro_rules! primitive_int_ring {
    ($t:ident) => {
        impl Ring for $t {}
        impl ConstRing for $t {}
        impl IntRing for $t {}
    };
}

primitive_int_ring!(i32);
primitive_int_ring!(i64);
primitive_int_ring!(i128);
