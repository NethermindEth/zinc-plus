// use crypto_bigint::Random;
// use num_traits::{ConstOne, ConstZero, One, Zero};
use crate::{Limb, PrimeField};
use std::ops::{Rem, RemAssign};
use std::{
    // UniformRand,
    fmt::Debug,
    iter::{Product, Sum},
    ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign},
};

/// A ring is like a field without a multiplicative inverse or division.
pub trait Ring:
    // Core traits
    Sized
    + Debug
    + Clone
    + PartialEq
    + Eq
    + Default
    + Sync
    + Send
    // + Zero
    // + One
    // Arithmetic operations consuming rhs
    + Neg<Output = Self>
    + Add<Self, Output = Self>
    + Sub<Self, Output = Self>
    + Mul<Self, Output = Self>
    + Div<Self, Output = Self> // TODO: ???
    + AddAssign
    + MulAssign
    + SubAssign
    + Sum
    + Product
    // Arithmetic operations with rhs reference
    + for<'a> Add<&'a Self, Output = Self>
    + for<'a> Sub<&'a Self, Output = Self>
    + for<'a> Mul<&'a Self, Output = Self>
    + for<'a> Div<&'a Self, Output = Self>
    + for<'a> AddAssign<&'a Self>
    + for<'a> MulAssign<&'a Self>
    + for<'a> Sum<&'a Self>
    + for<'a> Product<&'a Self>
    + for<'a> From<&'a Self>
    // + Random
    // + UniformRand
    {}

/// Ring of integers, usually denoted as `Z`.
pub trait IntRing:
Ring
    + From<Limb>
    + for<'a> TryFrom<&'a [Limb]>
    // Arithmetic operations consuming rhs
    + Rem
    + Rem<u64>
    + RemAssign
    + RemAssign<u64>
    // Arithmetic operations with rhs reference
    + for<'a> Rem<&'a Self>
    + for<'a> RemAssign<&'a Self>
{
    /// Number of u64 limbs used to represent this integer type
    fn num_limbs() -> usize;

    fn limbs(&self) -> &[Limb];

    // Cannot implement it as From trait since both participants are of non-local types
    fn to_field<F: PrimeField>(&self) -> F {
        F::from(self.limbs())
    }
}
