pub mod boolean;
#[cfg(feature = "crypto_bigint")]
pub mod crypto_bigint_uint;

use core::{
    fmt::{Debug, Display},
    hash::Hash,
    iter::{Product, Sum},
    ops::{Add, AddAssign, Mul, MulAssign, Shl, ShlAssign, Shr, ShrAssign, Sub, SubAssign},
    str::FromStr,
};
use num_traits::{CheckedAdd, CheckedMul, CheckedSub, ConstOne, ConstZero, One, Pow, Zero};

/// A semiring is a mathematical structure that consists of a set equipped with
/// two binary operations: addition and multiplication. The addition operation
/// is commutative and associative, has an identity element (zero), and every
/// element has an additive inverse. The multiplication operation is
/// associative, has an identity element (one), and distributes over addition.
/// However, unlike a ring, a semiring does not require the existence of
/// additive inverses for all elements.
/// Even though subtraction is not required, we include it here for convenience.
pub trait Semiring:
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
pub trait FixedSemiring: Semiring + Default + Zero + One {}
impl<T> FixedSemiring for T where T: Semiring + Default + Zero + One {}

pub trait ConstSemiring: FixedSemiring + ConstZero + ConstOne + From<bool> {}
impl<T> ConstSemiring for T where T: FixedSemiring + ConstZero + ConstOne + From<bool> {}

/// Semiring of integers
pub trait IntSemiring: Semiring + PartialOrd + Pow<u32> {
    fn is_odd(&self) -> bool;

    fn is_even(&self) -> bool;
}

pub trait IntSemiringWithShifts:
    IntSemiring + Shl<u32> + Shr<u32> + ShlAssign<u32> + ShrAssign<u32>
{
}
impl<T> IntSemiringWithShifts for T where
    T: IntSemiring + Shl<u32> + Shr<u32> + ShlAssign<u32> + ShrAssign<u32>
{
}

pub trait ConstIntSemiring: IntSemiring + ConstSemiring + FromStr {}
impl<T> ConstIntSemiring for T where T: IntSemiring + ConstSemiring + FromStr {}

macro_rules! primitive_int_semiring {
    ($t:ident) => {
        impl Semiring for $t {}
        impl IntSemiring for $t {
            fn is_odd(&self) -> bool {
                *self & 1 == 1
            }

            fn is_even(&self) -> bool {
                *self & 1 == 0
            }
        }
    };
}

primitive_int_semiring!(i8);
primitive_int_semiring!(i16);
primitive_int_semiring!(i32);
primitive_int_semiring!(i64);
primitive_int_semiring!(i128);
primitive_int_semiring!(u8);
primitive_int_semiring!(u16);
primitive_int_semiring!(u32);
primitive_int_semiring!(u64);
primitive_int_semiring!(u128);
