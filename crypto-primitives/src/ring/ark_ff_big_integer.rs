//! Note that this is not a real ring as it does not have an additive inverse.

use super::*;
use crate::impl_pow_via_repeated_squaring;
use alloc::vec::Vec;
use ark_ff::{BigInt, BigInteger as InnerBigInteger};
use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, Read, SerializationError, Valid, Validate,
    Write,
};
use core::{
    fmt::{Debug, Display, Formatter, LowerHex, Result as FmtResult, UpperHex},
    hash::Hasher,
    iter::{Product, Sum},
    ops::{
        Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Mul,
        MulAssign, Shl, Shr, Sub, SubAssign,
    },
    str::FromStr,
};
use num_traits::{
    CheckedAdd, CheckedMul, CheckedNeg, CheckedSub, ConstOne, ConstZero, One, Pow, Zero,
};
use pastey::paste;

#[cfg(feature = "rand")]
use ark_std::{UniformRand, rand::prelude::*};
use num_bigint::BigUint;
#[cfg(feature = "rand")]
use rand::distr::StandardUniform;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct BigInteger<B: InnerBigInteger>(B);

impl<B: InnerBigInteger> BigInteger<B> {
    /// Wraps a given value into this wrapper type
    #[inline(always)]
    pub const fn new(value: B) -> Self {
        Self(value)
    }

    /// Get the reference to the wrapped value
    #[inline(always)]
    pub const fn inner(&self) -> &B {
        &self.0
    }

    /// Get the wrapped value, consuming self
    #[inline(always)]
    pub const fn into_inner(self) -> B {
        self.0
    }
}

//
// Core traits
//

impl<B: InnerBigInteger> Debug for BigInteger<B> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        Debug::fmt(&self.0, f)
    }
}

impl<B: InnerBigInteger> Display for BigInteger<B> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        Display::fmt(&self.0, f)
    }
}

impl<B: InnerBigInteger> Default for BigInteger<B> {
    #[inline(always)]
    fn default() -> Self {
        Self::zero()
    }
}

impl<B: InnerBigInteger + LowerHex> LowerHex for BigInteger<B> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        LowerHex::fmt(&self.0, f)
    }
}

impl<B: InnerBigInteger + UpperHex> UpperHex for BigInteger<B> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        UpperHex::fmt(&self.0, f)
    }
}

impl<B: InnerBigInteger> Hash for BigInteger<B> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.as_ref().hash(state)
    }
}

impl<B: InnerBigInteger> FromStr for BigInteger<B> {
    type Err = B::Err;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        B::from_str(s).map(Self)
    }
}

impl<B: InnerBigInteger> AsRef<[u64]> for BigInteger<B> {
    #[inline(always)]
    fn as_ref(&self) -> &[u64] {
        self.0.as_ref()
    }
}

impl<B: InnerBigInteger> AsMut<[u64]> for BigInteger<B> {
    #[inline(always)]
    fn as_mut(&mut self) -> &mut [u64] {
        self.0.as_mut()
    }
}

//
// Zero and One traits
//

impl<B: InnerBigInteger> Zero for BigInteger<B> {
    #[inline(always)]
    fn zero() -> Self {
        Self(B::default())
    }

    #[inline(always)]
    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}

impl<B: InnerBigInteger> One for BigInteger<B> {
    #[inline(always)]
    fn one() -> Self {
        Self::from(1_u32)
    }
}

impl<const N: usize> ConstZero for BigInteger<BigInt<N>> {
    const ZERO: Self = Self(BigInt!("0"));
}

impl<const N: usize> ConstOne for BigInteger<BigInt<N>> {
    const ONE: Self = Self(BigInt!("1"));
}

//
// Basic arithmetic operations
//

// Basic operations are calling checked operations and panic on overflow.
macro_rules! impl_basic_op_panic_on_overflow {
    ($trait_name:tt, $trait_op:tt) => {
        impl<B: InnerBigInteger> $trait_name for BigInteger<B> {
            type Output = Self;

            #[inline(always)]
            fn $trait_op(self, rhs: Self) -> Self::Output {
                self.$trait_op(&rhs)
            }
        }

        impl<'a, B: InnerBigInteger> $trait_name<&'a Self> for BigInteger<B> {
            type Output = Self;

            #[inline(always)]
            fn $trait_op(self, rhs: &'a Self) -> Self::Output {
                paste! {
                self.[<checked_ $trait_op>](rhs).expect(concat!(
                    stringify!($trait_name),
                    " overflow"
                ))
                }
            }
        }
    };
}

impl_basic_op_panic_on_overflow!(Add, add);
impl_basic_op_panic_on_overflow!(Sub, sub);
impl_basic_op_panic_on_overflow!(Mul, mul);

macro_rules! impl_basic_op_delegate {
    ($trait_name:tt, $trait_op:tt) => {
        impl<B: InnerBigInteger> $trait_name for BigInteger<B> {
            type Output = Self;

            #[inline(always)]
            fn $trait_op(self, rhs: Self) -> Self::Output {
                self.$trait_op(&rhs)
            }
        }

        impl<'a, B: InnerBigInteger> $trait_name<&'a Self> for BigInteger<B> {
            type Output = Self;

            #[inline(always)]
            fn $trait_op(self, rhs: &'a Self) -> Self::Output {
                Self(self.0.$trait_op(&rhs.0))
            }
        }
    };
}

impl_basic_op_delegate!(BitOr, bitor);
impl_basic_op_delegate!(BitAnd, bitand);
impl_basic_op_delegate!(BitXor, bitxor);

impl<B: InnerBigInteger> Shl<u32> for BigInteger<B> {
    type Output = Self;

    #[inline(always)]
    fn shl(self, rhs: u32) -> Self::Output {
        Self(self.0.shl(rhs))
    }
}

impl<B: InnerBigInteger> Shr<u32> for BigInteger<B> {
    type Output = Self;

    #[inline(always)]
    fn shr(self, rhs: u32) -> Self::Output {
        Self(self.0.shr(rhs))
    }
}

impl<B: InnerBigInteger> Pow<u32> for BigInteger<B> {
    type Output = Self;

    impl_pow_via_repeated_squaring!();
}

//
// Checked arithmetic operations
//

impl<B: InnerBigInteger> CheckedNeg for BigInteger<B> {
    fn checked_neg(&self) -> Option<Self> {
        Self::zero().checked_sub(self)
    }
}

impl<B: InnerBigInteger> CheckedAdd for BigInteger<B> {
    #[inline(always)]
    fn checked_add(&self, other: &Self) -> Option<Self> {
        let mut result = *self;
        let carry = result.0.add_with_carry(&other.0);
        if carry {
            None // Overflow occurred
        } else {
            Some(result)
        }
    }
}

impl<B: InnerBigInteger> CheckedSub for BigInteger<B> {
    #[inline(always)]
    fn checked_sub(&self, other: &Self) -> Option<Self> {
        let mut result = *self;
        let borrow = result.0.sub_with_borrow(&other.0);
        if borrow {
            None // Overflow occurred
        } else {
            Some(result)
        }
    }
}

impl<B: InnerBigInteger> CheckedMul for BigInteger<B> {
    #[inline(always)]
    fn checked_mul(&self, other: &Self) -> Option<Self> {
        let (lo, hi) = self.0.mul(&other.0);
        if !hi.is_zero() {
            None // Overflow occurred
        } else {
            Some(Self(lo))
        }
    }
}

//
// Arithmetic assign operations
//

macro_rules! impl_assign_op {
    ($trait_name:tt, $trait_op:tt, $regular_op:tt) => {
        impl<B: InnerBigInteger> $trait_name<Self> for BigInteger<B> {
            #[inline(always)]
            fn $trait_op(&mut self, rhs: Self) {
                self.$trait_op(&rhs);
            }
        }

        impl<'a, B: InnerBigInteger> $trait_name<&'a Self> for BigInteger<B> {
            #[inline(always)]
            fn $trait_op(&mut self, rhs: &'a Self) {
                self.0 = self.$regular_op(rhs).0;
            }
        }
    };
}

impl_assign_op!(AddAssign, add_assign, add);
impl_assign_op!(SubAssign, sub_assign, sub);
impl_assign_op!(MulAssign, mul_assign, mul);
impl_assign_op!(BitOrAssign, bitor_assign, bitor);
impl_assign_op!(BitAndAssign, bitand_assign, bitand);
impl_assign_op!(BitXorAssign, bitxor_assign, bitxor);

macro_rules! impl_assign_op_primitive {
    ($trait_name:tt, $trait_op:tt, $regular_op:tt, $rhs_type:tt) => {
        impl<B: InnerBigInteger> $trait_name<$rhs_type> for BigInteger<B> {
            #[inline(always)]
            fn $trait_op(&mut self, rhs: $rhs_type) {
                self.0 = self.$regular_op(rhs).0;
            }
        }

        impl<'a, B: InnerBigInteger> $trait_name<&'a $rhs_type> for BigInteger<B> {
            #[inline(always)]
            fn $trait_op(&mut self, rhs: &'a $rhs_type) {
                self.$trait_op(*rhs);
            }
        }
    };
}

impl_assign_op_primitive!(ShlAssign, shl_assign, shl, u32);
impl_assign_op_primitive!(ShrAssign, shr_assign, shr, u32);

//
// Aggregate operations
//

impl<B: InnerBigInteger> Sum for BigInteger<B> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, x| {
            acc.checked_add(&x).expect("overflow in sum")
        })
    }
}

impl<'a, B: InnerBigInteger> Sum<&'a Self> for BigInteger<B> {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, x| {
            acc.checked_add(x).expect("overflow in sum")
        })
    }
}

impl<B: InnerBigInteger> Product for BigInteger<B> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |acc, x| {
            acc.checked_mul(&x).expect("overflow in product")
        })
    }
}

impl<'a, B: InnerBigInteger> Product<&'a Self> for BigInteger<B> {
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |acc, x| {
            acc.checked_mul(x).expect("overflow in product")
        })
    }
}

//
// Conversions
//

impl<const N: usize> From<BigInt<N>> for BigInteger<BigInt<N>> {
    #[inline(always)]
    fn from(value: BigInt<N>) -> Self {
        Self(value)
    }
}

impl<const N: usize> From<BigInteger<BigInt<N>>> for BigInt<N> {
    #[inline(always)]
    fn from(value: BigInteger<BigInt<N>>) -> Self {
        value.0
    }
}

macro_rules! impl_from_primitive {
    ($($t:ty),+) => {
        $(
            impl<B: InnerBigInteger> From<$t> for BigInteger<B> {
                fn from(value: $t) -> Self {
                    Self(B::from(value))
                }
            }

            impl<'a, B: InnerBigInteger> From<&'a $t> for BigInteger<B> {
                #[inline(always)]
                fn from(value: &$t) -> Self {
                    Self::from(*value)
                }
            }
        )+
    };
}

impl_from_primitive!(u8, u16, u32, u64);

impl<B: InnerBigInteger> From<bool> for BigInteger<B> {
    fn from(value: bool) -> Self {
        Self(B::from(u64::from(value)))
    }
}

impl<B: InnerBigInteger> TryFrom<BigUint> for BigInteger<B> {
    type Error = <B as TryFrom<BigUint>>::Error;

    fn try_from(value: BigUint) -> Result<Self, Self::Error> {
        B::try_from(value).map(Self)
    }
}

impl<B: InnerBigInteger> From<BigInteger<B>> for BigUint {
    fn from(value: BigInteger<B>) -> Self {
        value.0.into()
    }
}

//
// Ring and IntRing
//

impl<B: InnerBigInteger> Ring for BigInteger<B> {}

impl<B: InnerBigInteger> IntRing for BigInteger<B> {}

//
// RNG
//

#[cfg(feature = "rand")]
impl<B: InnerBigInteger> Distribution<BigInteger<B>> for StandardUniform {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> BigInteger<B> {
        BigInteger(UniformRand::rand(rng))
    }
}

#[cfg(feature = "rand")]
impl<B: InnerBigInteger> UniformRand for BigInteger<B> {
    fn rand<R: ark_std::rand::Rng + ?Sized>(rng: &mut R) -> Self {
        Self(B::rand(rng))
    }
}

//
// Traits from ark-ff
//

impl<B: InnerBigInteger> CanonicalSerialize for BigInteger<B> {
    fn serialize_with_mode<W: Write>(
        &self,
        writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.0.serialize_with_mode(writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.0.serialized_size(compress)
    }
}

impl<B: InnerBigInteger> CanonicalDeserialize for BigInteger<B> {
    fn deserialize_with_mode<R: Read>(
        reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        B::deserialize_with_mode(reader, compress, validate).map(Self)
    }
}

impl<B: InnerBigInteger> Valid for BigInteger<B> {
    fn check(&self) -> Result<(), SerializationError> {
        self.0.check()
    }
}

impl<B: InnerBigInteger> zeroize::Zeroize for BigInteger<B> {
    fn zeroize(&mut self) {
        self.0.zeroize()
    }
}

impl<B: InnerBigInteger> InnerBigInteger for BigInteger<B> {
    const NUM_LIMBS: usize = B::NUM_LIMBS;

    fn add_with_carry(&mut self, other: &Self) -> bool {
        self.0.add_with_carry(&other.0)
    }

    fn sub_with_borrow(&mut self, other: &Self) -> bool {
        self.0.sub_with_borrow(&other.0)
    }

    fn mul2(&mut self) -> bool {
        self.0.mul2()
    }

    #[allow(clippy::arithmetic_side_effects)] // False alert
    fn muln(&mut self, amt: u32) {
        self.0 <<= amt;
    }

    fn mul_low(&self, other: &Self) -> Self {
        Self(self.0.mul_low(&other.0))
    }

    fn mul_high(&self, other: &Self) -> Self {
        Self(self.0.mul_high(&other.0))
    }

    fn mul(&self, other: &Self) -> (Self, Self) {
        let (lo, hi) = self.0.mul(&other.0);
        (Self(lo), Self(hi))
    }

    fn div2(&mut self) {
        self.0.div2()
    }

    #[allow(clippy::arithmetic_side_effects)] // False alert
    fn divn(&mut self, amt: u32) {
        self.0 >>= amt;
    }

    fn is_odd(&self) -> bool {
        self.0.is_odd()
    }

    fn is_even(&self) -> bool {
        self.0.is_even()
    }

    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }

    fn num_bits(&self) -> u32 {
        self.0.num_bits()
    }

    fn get_bit(&self, i: usize) -> bool {
        self.0.get_bit(i)
    }

    fn from_bits_be(bits: &[bool]) -> Self {
        Self(B::from_bits_be(bits))
    }

    fn from_bits_le(bits: &[bool]) -> Self {
        Self(B::from_bits_le(bits))
    }

    fn to_bytes_be(&self) -> Vec<u8> {
        self.0.to_bytes_be()
    }

    fn to_bytes_le(&self) -> Vec<u8> {
        self.0.to_bytes_le()
    }
}
