use crate::{
    field::{
        biginteger::{BigInt, Words},
        uint::Uint,
    },
    pcs::utils::AsWords,
    traits::Integer,
};
use ark_std::{
    cmp::Ordering,
    iter::Sum,
    ops::{Add, AddAssign, Mul, Neg, RemAssign, Sub},
    rand::RngCore,
};
use crypto_bigint::{
    Int as CryptoInt, NonZero, Random, Word,
    subtle::{Choice, ConstantTimeEq},
};
use num_traits::{ConstOne, ConstZero, One, Zero};
use p3_field::Packable;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(transparent)]
pub struct Int<const N: usize>(pub(crate) CryptoInt<N>);

impl<const N: usize> Zero for Int<N> {
    #[inline]
    fn zero() -> Self {
        Self(CryptoInt::zero())
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}

impl<const N: usize> Add<Self> for Int<N> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl<const N: usize> ConstZero for Int<N> {
    const ZERO: Self = Self(CryptoInt::ZERO);
}

impl<const N: usize> One for Int<N> {
    #[inline]
    fn one() -> Self {
        Self(CryptoInt::one())
    }
}

impl<const N: usize> ConstOne for Int<N> {
    const ONE: Self = Self(CryptoInt::ONE);
}

impl<const N: usize> Neg for Int<N> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        Self(self.0.checked_neg().unwrap())
    }
}

impl<const N: usize> Mul<Self> for Int<N> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self(self.0 * rhs.0)
    }
}

impl<const N: usize> ConstantTimeEq for Int<N> {
    #[inline]
    fn ct_eq(&self, other: &Self) -> Choice {
        self.0.ct_eq(&other.0)
    }
}

impl<const N: usize> PartialOrd for Int<N> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl<const N: usize> RemAssign<Self> for Int<N> {
    #[inline]
    fn rem_assign(&mut self, rhs: Self) {
        self.0.rem_assign(
            NonZero::new(rhs.0).expect("Cannot reduce modulo zero: field modulus is zero"),
        );
    }
}

impl<'a, const N: usize> Add<&'a Self> for Int<N> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: &'a Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl<'a, const N: usize> Mul<&'a Self> for Int<N> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: &'a Self) -> Self::Output {
        Self(self.0 * rhs.0)
    }
}

impl<const N: usize> AddAssign<Self> for Int<N> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.0 += &rhs.0;
    }
}

impl<'a, const N: usize> AddAssign<&'a Self> for Int<N> {
    #[inline]
    fn add_assign(&mut self, rhs: &'a Self) {
        self.0 += &rhs.0;
    }
}

impl<'a, const N: usize> Sub<&'a Self> for Int<N> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: &'a Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl<const N: usize> From<[u64; N]> for Int<N> {
    #[inline]
    fn from(value: [u64; N]) -> Self {
        Self(CryptoInt::from_words(value))
    }
}

impl<const N: usize> From<i64> for Int<N> {
    #[inline]
    fn from(value: i64) -> Self {
        Self(CryptoInt::from(value))
    }
}

impl<const N: usize> From<i32> for Int<N> {
    #[inline]
    fn from(value: i32) -> Self {
        Self(CryptoInt::from(value))
    }
}

impl<const N: usize> From<i8> for Int<N> {
    #[inline]
    fn from(value: i8) -> Self {
        Self(CryptoInt::from(value))
    }
}

impl<const N: usize> From<u8> for Int<N> {
    #[inline]
    fn from(value: u8) -> Self {
        Self(CryptoInt::from(value as i8))
    }
}

impl<const N: usize> Default for Int<N> {
    #[inline]
    fn default() -> Self {
        Self(CryptoInt::default())
    }
}

impl<const N: usize> Random for Int<N> {
    #[inline]
    fn random(rng: &mut (impl RngCore + ?Sized)) -> Self {
        Self(CryptoInt::random(rng))
    }
}

impl<'a, const N: usize, const M: usize> From<&'a Int<M>> for Int<N> {
    #[inline]
    fn from(value: &'a Int<M>) -> Self {
        Self(CryptoInt::from(&value.0))
    }
}

impl<const N: usize> AsWords for Int<N> {
    fn as_words(&self) -> &[Word] {
        self.0.as_words()
    }
}

impl<const N: usize> Sum for Int<N> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |mut acc, x| {
            acc.add_assign(&x);
            acc
        })
    }
}

impl<const N: usize> Packable for Int<N> {}

impl<const N: usize> Integer for Int<N> {
    type I = BigInt<N>;
    type Uint = Uint<N>;
    type W = Words<N>;

    fn from_words(words: Words<N>) -> Self {
        Self(CryptoInt::from_words(words.0))
    }

    fn from_i64(value: i64) -> Self {
        Self(CryptoInt::from_i64(value))
    }

    fn abs(&self) -> Self::Uint {
        Uint(self.0.abs())
    }
}

/// Defines a wrapper type suitable for implementing the `ZipTypes` trait with
/// const generics.
///
/// # Usage
/// - `define_random_field_zip_types!();` Expands to `pub struct
///   RandomFieldZipTypes<const N: usize>();`
///
/// - `define_random_field_zip_types!(CustomName);` Expands to `pub struct
///   CustomName<const N: usize>();`
///
/// This macro allows downstream crates to define their own local wrapper types
/// for implementing traits, in compliance with Rust's orphan rule.
#[macro_export]
macro_rules! define_random_field_zip_types {
    () => {
        define_random_field_zip_types!(RandomFieldZipTypes);
    };

    ($name:ident) => {
        #[allow(dead_code)]
        #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name<const N: usize>();
    };
}

/// Implements the `ZipTypes` trait for a wrapper type and a specific const
/// parameter.
///
/// # Usage
/// - `implement_random_field_zip_types!(N);` Implements `ZipTypes` for
///   `RandomFieldZipTypes<N>`.
///
/// - `implement_random_field_zip_types!(TypeName, N);` Implements `ZipTypes`
///   for `TypeName<N>`.
///
/// This macro reduces boilerplate and ensures consistent associated type
/// definitions for each implementation.
#[macro_export]
macro_rules! implement_random_field_zip_types {
    ($N:expr) => {
        implement_random_field_zip_types!(RandomFieldZipTypes, $N);
    };

    ($name:ident, $N:expr) => {
        impl $crate::traits::ZipTypes for $name<$N> {
            type K = $crate::field::Int<{ 4 * $N }>;
            type L = $crate::field::Int<{ 2 * $N }>;
            type M = $crate::field::Int<{ 8 * $N }>;
            type N = $crate::field::Int<$N>;
        }
    };
}
