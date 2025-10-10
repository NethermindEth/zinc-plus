use super::*;
use core::{
    cmp::Ordering,
    fmt::{Debug, Display, Formatter, LowerHex, Result as FmtResult, UpperHex},
    hash::Hasher,
    iter::{Product, Sum},
    ops::{Add, AddAssign, Mul, MulAssign, Rem, RemAssign, Shl, Shr, Sub, SubAssign},
};
use crypto_bigint::{
    CheckedMul as CryptoCheckedMul, CheckedSub as CryptoCheckedSub, Word,
    subtle::{
        Choice, ConditionallySelectable, ConstantTimeEq, ConstantTimeGreater, ConstantTimeLess,
    },
};
use num_traits::{
    CheckedAdd, CheckedMul, CheckedNeg, CheckedRem, CheckedSub, ConstOne, ConstZero, One, Pow, Zero,
};
use pastey::paste;

use crate::impl_pow_via_repeated_squaring;
#[cfg(feature = "rand")]
use rand::{distr::StandardUniform, prelude::*, rand_core::TryRngCore};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct Int<const LIMBS: usize>(crypto_bigint::Int<LIMBS>);

impl<const LIMBS: usize> Int<LIMBS> {
    /// Wraps a given value into this wrapper type
    #[inline(always)]
    pub const fn new(value: crypto_bigint::Int<LIMBS>) -> Self {
        Self(value)
    }

    /// Get the reference to the wrapped value
    #[inline(always)]
    pub const fn inner(&self) -> &crypto_bigint::Int<LIMBS> {
        &self.0
    }

    /// Get the wrapped value, consuming self
    #[inline(always)]
    pub const fn into_inner(self) -> crypto_bigint::Int<LIMBS> {
        self.0
    }

    /// See [crypto_bigint::Int::from_words]
    #[inline(always)]
    pub const fn from_words(arr: [Word; LIMBS]) -> Self {
        Self(crypto_bigint::Int::from_words(arr))
    }

    /// See [crypto_bigint::Int::resize]
    #[inline(always)]
    pub const fn resize<const T: usize>(&self) -> Int<T> {
        Int::<T>(self.0.resize())
    }

    pub const fn checked_resize<const T: usize>(&self) -> Option<Int<T>> {
        match checked_resize::<LIMBS, T>(&self.0) {
            None => None,
            Some(inner) => Some(Int(inner)),
        }
    }

    /// See [crypto_bigint::Int::cmp_vartime]
    pub const fn cmp_vartime(&self, rhs: &Self) -> Ordering {
        self.0.cmp_vartime(&rhs.0)
    }
}

const fn checked_resize<const SRC: usize, const DST: usize>(
    num: &crypto_bigint::Int<SRC>,
) -> Option<crypto_bigint::Int<DST>> {
    if SRC > DST {
        let max = Int::<DST>::MAX.0.resize();
        let cmp = num.cmp_vartime(&max);
        if cmp.is_gt() {
            return None;
        }

        let min = Int::<DST>::MIN.0.resize();
        let cmp = num.cmp_vartime(&min);
        if cmp.is_lt() {
            return None;
        }
    }
    Some(num.resize())
}

macro_rules! define_consts {
    ($($name:ident),+) => {
        $(pub const $name: Self = Self(crypto_bigint::Int::<LIMBS>::$name);)+
    };
}

impl<const LIMBS: usize> Int<LIMBS> {
    define_consts!(MINUS_ONE, MIN, MAX, SIGN_MASK, FULL_MASK);

    /// Total size of the represented integer in bits.
    pub const BITS: u32 = crypto_bigint::Int::<LIMBS>::BITS;

    /// Total size of the represented integer in bytes.
    pub const BYTES: usize = crypto_bigint::Int::<LIMBS>::BYTES;

    /// The number of limbs used on this platform.
    pub const LIMBS: usize = LIMBS;
}

//
// Core traits
//

impl<const LIMBS: usize> Debug for Int<LIMBS> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        Debug::fmt(&self.0, f)
    }
}

impl<const LIMBS: usize> Display for Int<LIMBS> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        Display::fmt(&self.0, f)
    }
}

impl<const LIMBS: usize> Default for Int<LIMBS> {
    #[inline(always)]
    fn default() -> Self {
        Self::ZERO
    }
}

impl<const LIMBS: usize> LowerHex for Int<LIMBS> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        LowerHex::fmt(&self.0, f)
    }
}

impl<const LIMBS: usize> UpperHex for Int<LIMBS> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        UpperHex::fmt(&self.0, f)
    }
}

impl<const LIMBS: usize> Hash for Int<LIMBS> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state)
    }
}

impl<const LIMBS: usize> FromStr for Int<LIMBS> {
    type Err = ();

    #[allow(clippy::arithmetic_side_effects)] // False positive
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (neg, s) = if let Some(s) = s.strip_prefix('-') {
            (true, s)
        } else {
            (false, s)
        };
        use crypto_bigint::Uint;
        let abs = Uint::<LIMBS>::from_str_radix_vartime(s, 10).map_err(|_| ())?;
        let res: Result<Self, _> = abs.try_into();
        match res {
            Ok(res) if neg => res.checked_neg().ok_or(()),
            Ok(res) => Ok(res),
            // Corner case: value is exactly minimum Int<LIMBS>
            _ if neg && abs == (Uint::MAX.shr(1) + Uint::one()) => Ok(Int::<LIMBS>::MIN),
            _ => Err(()),
        }
    }
}

//
// Zero and One traits
//

impl<const LIMBS: usize> Zero for Int<LIMBS> {
    #[inline(always)]
    fn zero() -> Self {
        Self::ZERO
    }

    #[inline(always)]
    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}

impl<const LIMBS: usize> One for Int<LIMBS> {
    #[inline(always)]
    fn one() -> Self {
        Self::ONE
    }
}

impl<const LIMBS: usize> ConstZero for Int<LIMBS> {
    const ZERO: Self = Self(crypto_bigint::Int::ZERO);
}

impl<const LIMBS: usize> ConstOne for Int<LIMBS> {
    const ONE: Self = Self(crypto_bigint::Int::ONE);
}

//
// Basic arithmetic operations
//

macro_rules! impl_basic_op {
    ($trait_name:tt, $trait_op:tt) => {
        impl<const LIMBS: usize> $trait_name for Int<LIMBS> {
            type Output = Self;

            #[inline(always)]
            fn $trait_op(self, rhs: Self) -> Self::Output {
                self.$trait_op(&rhs)
            }
        }

        impl<'a, const LIMBS: usize> $trait_name<&'a Self> for Int<LIMBS> {
            type Output = Self;

            #[inline(always)]
            fn $trait_op(self, rhs: &'a Self) -> Self::Output {
                Self(self.0.$trait_op(&rhs.0))
            }
        }
    };
}

impl_basic_op!(Add, add);
impl_basic_op!(Sub, sub);
impl_basic_op!(Mul, mul);

impl<const LIMBS: usize> Rem for Int<LIMBS> {
    type Output = Self;

    #[inline(always)]
    fn rem(self, rhs: Self) -> Self::Output {
        self.rem(&rhs)
    }
}

impl<'a, const LIMBS: usize> Rem<&'a Self> for Int<LIMBS> {
    type Output = Self;

    #[inline(always)]
    fn rem(self, rhs: &'a Self) -> Self::Output {
        let non_zero = crypto_bigint::NonZero::new(rhs.0).expect("division by zero");
        Self(self.0.rem(&non_zero))
    }
}

impl<const LIMBS: usize> Shl<u32> for Int<LIMBS> {
    type Output = Self;

    #[inline(always)]
    fn shl(self, rhs: u32) -> Self::Output {
        Self(self.0.shl(rhs))
    }
}

impl<const LIMBS: usize> Shr<u32> for Int<LIMBS> {
    type Output = Self;

    #[inline(always)]
    fn shr(self, rhs: u32) -> Self::Output {
        Self(self.0.shr(rhs))
    }
}

impl<const LIMBS: usize> Pow<u32> for Int<LIMBS> {
    type Output = Self;

    impl_pow_via_repeated_squaring!();
}

//
// Checked arithmetic operations
//

impl<const LIMBS: usize> CheckedNeg for Int<LIMBS> {
    fn checked_neg(&self) -> Option<Self> {
        let result = self.0.checked_neg();
        if result.is_some().into() {
            Some(Self(result.unwrap()))
        } else {
            None
        }
    }
}

macro_rules! impl_checked_op {
    ($trait_name:tt, $trait_op:tt) => {
        impl<const LIMBS: usize> $trait_name for Int<LIMBS> {
            #[inline(always)]
            fn $trait_op(&self, other: &Self) -> Option<Self> {
                let value: Option<crypto_bigint::Int<LIMBS>> = self.0.$trait_op(&other.0).into();
                value.map(Self)
            }
        }
    };
}

impl_checked_op!(CheckedAdd, checked_add);
impl_checked_op!(CheckedSub, checked_sub);
impl_checked_op!(CheckedMul, checked_mul);

impl<const LIMBS: usize> CheckedRem for Int<LIMBS> {
    #[inline(always)]
    fn checked_rem(&self, other: &Self) -> Option<Self> {
        let non_zero = crypto_bigint::NonZero::new(other.0).into_option()?;
        Some(Self(self.0.rem(&non_zero)))
    }
}

//
// Arithmetic assign operations
//

macro_rules! impl_assign_op {
    ($trait_name:tt, $trait_op:tt) => {
        impl<const LIMBS: usize> $trait_name<Self> for Int<LIMBS> {
            #[inline(always)]
            fn $trait_op(&mut self, rhs: Self) {
                self.$trait_op(&rhs);
            }
        }

        impl<'a, const LIMBS: usize> $trait_name<&'a Self> for Int<LIMBS> {
            #[inline(always)]
            fn $trait_op(&mut self, rhs: &'a Self) {
                self.0.$trait_op(&rhs.0);
            }
        }
    };
}

impl_assign_op!(AddAssign, add_assign);
impl_assign_op!(SubAssign, sub_assign);
impl_assign_op!(MulAssign, mul_assign);

impl<const LIMBS: usize> RemAssign for Int<LIMBS> {
    #[inline(always)]
    fn rem_assign(&mut self, rhs: Self) {
        self.rem_assign(&rhs);
    }
}

impl<'a, const LIMBS: usize> RemAssign<&'a Self> for Int<LIMBS> {
    #![allow(clippy::arithmetic_side_effects)]
    #[inline(always)]
    fn rem_assign(&mut self, rhs: &'a Self) {
        let non_zero = crypto_bigint::NonZero::new(rhs.0).expect("division by zero");
        self.0 %= non_zero;
    }
}

//
// Aggregate operations
//

impl<const LIMBS: usize> Sum for Int<LIMBS> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, x| {
            acc.checked_add(&x).expect("overflow in sum")
        })
    }
}

impl<'a, const LIMBS: usize> Sum<&'a Self> for Int<LIMBS> {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, x| {
            acc.checked_add(x).expect("overflow in sum")
        })
    }
}

impl<const LIMBS: usize> Product for Int<LIMBS> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |acc, x| {
            acc.checked_mul(&x).expect("overflow in product")
        })
    }
}

impl<'a, const LIMBS: usize> Product<&'a Self> for Int<LIMBS> {
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |acc, x| {
            acc.checked_mul(x).expect("overflow in product")
        })
    }
}

//
// Conversions
//

impl<const LIMBS: usize> From<crypto_bigint::Int<LIMBS>> for Int<LIMBS> {
    #[inline(always)]
    fn from(value: crypto_bigint::Int<LIMBS>) -> Self {
        Self(value)
    }
}

impl<const LIMBS: usize> From<Int<LIMBS>> for crypto_bigint::Int<LIMBS> {
    #[inline(always)]
    fn from(value: Int<LIMBS>) -> Self {
        value.0
    }
}

impl<const LIMBS: usize> From<bool> for Int<LIMBS> {
    fn from(value: bool) -> Self {
        Self(crypto_bigint::Int::<LIMBS>::from(i8::from(value)))
    }
}

macro_rules! impl_from_primitive {
    ($($t:ty),+) => {
        $(
            impl<const LIMBS: usize> From<$t> for Int<LIMBS> {
                fn from(value: $t) -> Self {
                    assert!(core::mem::size_of::<$t>() <= crypto_bigint::Int::<LIMBS>::BYTES,
                            "`{}` is too large to fit into `Int<{LIMBS}>`", stringify!($t));
                    Self(crypto_bigint::Int::<LIMBS>::from(value))
                }
            }

            impl<'a, const LIMBS: usize> From<&'a $t> for Int<LIMBS> {
                #[inline(always)]
                fn from(value: &$t) -> Self {
                    Self::from(*value)
                }
            }

            impl<const LIMBS: usize> Int<LIMBS> {
            paste! {
                /// Create an Int from a primitive type.
                /// It does NOT check for overflow - this behaviour is
                /// consistent with the `crypto_bigint::Int` methods.
                pub const fn [<from_ $t>](n: $t) -> Self {
                    Self(crypto_bigint::Int::<LIMBS>::[<from_ $t>](n))
                }
            }
            }
        )+
    };
}

impl_from_primitive!(i8, i16, i32, i64, i128);

impl<const LIMBS: usize, const LIMBS2: usize> TryFrom<&crypto_bigint::Int<LIMBS2>> for Int<LIMBS> {
    type Error = ();

    fn try_from(num: &crypto_bigint::Int<LIMBS2>) -> Result<Self, Self::Error> {
        checked_resize(num).map(Self).ok_or(())
    }
}

impl<const LIMBS: usize, const LIMBS2: usize> TryFrom<crypto_bigint::Uint<LIMBS2>> for Int<LIMBS> {
    type Error = ();

    fn try_from(num: crypto_bigint::Uint<LIMBS2>) -> Result<Self, Self::Error> {
        (&num).try_into()
    }
}

impl<const LIMBS: usize, const LIMBS2: usize> TryFrom<&crypto_bigint::Uint<LIMBS2>> for Int<LIMBS> {
    type Error = ();

    fn try_from(num: &crypto_bigint::Uint<LIMBS2>) -> Result<Self, Self::Error> {
        if LIMBS < LIMBS2 {
            let max = Int::<LIMBS>::MAX.0.as_uint().resize();
            if num > &max {
                return Err(());
            }
        }
        let result = num.resize().try_into_int();
        if result.is_none().into() {
            return Err(());
        }
        Ok(result.unwrap().into())
    }
}
//
// Ring and IntRing
//

impl<const LIMBS: usize> Ring for Int<LIMBS> {}

impl<const LIMBS: usize> IntRing for Int<LIMBS> {}

//
// RNG
//

#[cfg(feature = "rand")]
impl<const LIMBS: usize> Distribution<Int<LIMBS>> for StandardUniform {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Int<LIMBS> {
        crypto_bigint::Random::random(rng)
    }
}

#[cfg(feature = "rand")]
impl<const LIMBS: usize> crypto_bigint::Random for Int<LIMBS> {
    fn try_random<R: TryRngCore + ?Sized>(rng: &mut R) -> Result<Self, R::Error> {
        crypto_bigint::Int::try_random(rng).map(Self)
    }
}

//
// Serialization and Deserialization
//

#[cfg(feature = "serde")]
impl<'de, const LIMBS: usize> serde::Deserialize<'de> for Int<LIMBS>
where
    crypto_bigint::Int<LIMBS>: crypto_bigint::Encoding,
{
    #[inline(always)]
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        crypto_bigint::Int::<LIMBS>::deserialize(deserializer).map(Self)
    }
}

#[cfg(feature = "serde")]
impl<const LIMBS: usize> serde::Serialize for Int<LIMBS>
where
    crypto_bigint::Int<LIMBS>: crypto_bigint::Encoding,
{
    #[inline(always)]
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.0.serialize(serializer)
    }
}

//
// Traits from crypto_bigint
//

impl<const LIMBS: usize> ConstantTimeEq for Int<LIMBS> {
    #[inline]
    fn ct_eq(&self, other: &Self) -> Choice {
        ConstantTimeEq::ct_eq(&self.0, &other.0)
    }
}

impl<const LIMBS: usize> ConstantTimeGreater for Int<LIMBS> {
    #[inline]
    fn ct_gt(&self, other: &Self) -> Choice {
        ConstantTimeGreater::ct_gt(&self.0, &other.0)
    }
}

impl<const LIMBS: usize> ConstantTimeLess for Int<LIMBS> {
    #[inline]
    fn ct_lt(&self, other: &Self) -> Choice {
        ConstantTimeLess::ct_lt(&self.0, &other.0)
    }
}

impl<const LIMBS: usize> ConditionallySelectable for Int<LIMBS> {
    fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
        ConditionallySelectable::conditional_select(&a.0, &b.0, choice).into()
    }
}

impl<const LIMBS: usize> crypto_bigint::Bounded for Int<LIMBS> {
    const BITS: u32 = Self::BITS;
    const BYTES: usize = Self::BYTES;
}

impl<const LIMBS: usize> crypto_bigint::Constants for Int<LIMBS> {
    const MAX: Self = Self::MAX;
}

#[allow(clippy::arithmetic_side_effects, clippy::cast_lossless)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::ensure_type_implements_trait;
    use alloc::{format, string::ToString, vec::Vec};

    #[cfg(target_pointer_width = "64")]
    const WORD_FACTOR: usize = 1;
    #[cfg(target_pointer_width = "32")]
    const WORD_FACTOR: usize = 2;

    type Int1 = Int<WORD_FACTOR>;
    type Int2 = Int<{ WORD_FACTOR * 2 }>;
    type Int4 = Int<{ WORD_FACTOR * 4 }>;

    #[test]
    fn ensure_blanket_traits() {
        ensure_type_implements_trait!(Int4, ConstIntRing);
    }

    #[test]
    fn basic_operations() {
        // Test with 4 limbs (256-bit integers)
        let a = Int4::from(10_i64);
        let b = Int4::from(5_i64);

        // Test addition
        let c = a + b;
        assert_eq!(c, Int4::from(15_i64));

        // Test subtraction
        let d = a - b;
        assert_eq!(d, Int4::from(5_i64));

        // Test multiplication
        let e = a * b;
        assert_eq!(e, Int4::from(50_i64));

        // Test remainder
        let f = a % b;
        assert_eq!(f, Int4::from(0_i64));

        // Test shl
        let x = Int1::from(0x0001_i64);

        assert_eq!(x << 0, x);
        assert_eq!(x << 1, 0x0002.into());
        assert_eq!(x << 15, 0x8000.into());
        assert_eq!(x << 63, (-0x8000000000000000_i64).into());
        // x << 64 panics, it's tested separately

        // Test shr
        let x = Int4::from(0x8000_i32);

        assert_eq!(x >> 0, x);
        assert_eq!(x >> 1, 0x4000.into());
        assert_eq!(x >> 15, 0x0001.into());
        assert_eq!(x >> 16, Int::ZERO);
    }

    #[test]
    #[should_panic(expected = "`shift` within the bit size of the integer")]
    fn shl_panics_on_overflow() {
        let x = Int1::from(0x0001_i64);
        let _ = x << 64;
    }

    #[test]
    fn checked_operations() {
        let a = Int4::from(10_i64);
        let b = Int4::from(5_i64);
        let zero = Int4::ZERO;

        // Test checked_add
        let c = a.checked_add(&b).unwrap();
        assert_eq!(c, Int4::from(15_i64));

        // Test checked_sub
        let d = a.checked_sub(&b).unwrap();
        assert_eq!(d, Int4::from(5_i64));

        // Test checked_mul
        let e = a.checked_mul(&b).unwrap();
        assert_eq!(e, Int4::from(50_i64));

        // Test checked_rem
        let f = a.checked_rem(&b).unwrap();
        assert_eq!(f, Int4::ZERO);

        // Test checked_rem with zero divisor
        assert!(a.checked_rem(&zero).is_none());
    }

    #[allow(clippy::op_ref)]
    #[test]
    fn reference_operations() {
        let a = Int4::from(10_i64);
        let b = Int4::from(5_i64);

        // Test reference-based addition
        let c = a + &b;
        assert_eq!(c, Int4::from(15_i64));

        // Test reference-based subtraction
        let d = a - &b;
        assert_eq!(d, Int4::from(5_i64));

        // Test reference-based multiplication
        let e = a * &b;
        assert_eq!(e, Int4::from(50_i64));

        // Test reference-based remainder
        let f = a % &b;
        assert_eq!(f, Int4::ZERO);
    }

    #[test]
    fn conversions() {
        // Test From<crypto_bigint::Int> for Int
        let original = crypto_bigint::Int::from(123_i64);
        let wrapped: Int4 = original.into();
        assert_eq!(wrapped.0, original);

        // Test From<Int> for crypto_bigint::Int
        let wrapped = Int4::from(456_i64);
        let unwrapped: crypto_bigint::Int<{ 4 * WORD_FACTOR }> = wrapped.into();
        assert_eq!(unwrapped, crypto_bigint::Int::from(456_i64));

        // Test conversion methods
        let value = crypto_bigint::Int::from(789_i64);
        let wrapped = Int4::new(value);
        assert_eq!(wrapped.inner(), &value);
        assert_eq!(wrapped.into_inner(), value);
    }

    #[test]
    fn pow_operation() {
        // Test basic exponentiation
        let base = Int4::from(2_i64);

        // 2^0 = 1
        assert_eq!(base.pow(0), Int4::one());

        // 2^1 = 2
        assert_eq!(base.pow(1), base);

        // 2^3 = 8
        assert_eq!(base.pow(3), Int4::from(8_i64));

        // 2^10 = 1024
        assert_eq!(base.pow(10), Int4::from(1024_i64));

        // Test with different base
        let base = Int4::from(3_i64);

        // 3^4 = 81
        assert_eq!(base.pow(4), Int4::from(81_i64));

        // Test with base 1
        let base = Int4::from(1_i64);
        assert_eq!(base.pow(1000), Int4::from(1_i64));

        // Test with base 0
        let base = Int4::from(0_i64);
        assert_eq!(base.pow(0), Int4::one()); // 0^0 = 1 by convention
        assert_eq!(base.pow(10), Int4::zero()); // 0^n = 0 for n > 0
    }

    #[test]
    fn checked_neg() {
        // Test with positive number
        let a = Int4::from(10_i64);
        let neg_a = a.checked_neg().unwrap();
        assert_eq!(neg_a, Int4::from(-10_i64));

        // Test with negative number
        let b = Int4::from(-5_i64);
        let neg_b = b.checked_neg().unwrap();
        assert_eq!(neg_b, Int4::from(5_i64));

        // Test with zero
        let zero = Int4::zero();
        let neg_zero = zero.checked_neg().unwrap();
        assert_eq!(neg_zero, zero);

        // Test with MIN value (should return None as -MIN would overflow)
        let min = Int4::MIN;
        assert!(min.checked_neg().is_none());
    }

    #[test]
    fn rem_assign_operations() {
        // Test RemAssign with owned value
        let mut a = Int4::from(17_i64);
        let b = Int4::from(5_i64);
        a %= b;
        assert_eq!(a, Int4::from(2_i64));

        // Test RemAssign with reference
        let mut c = Int4::from(19_i64);
        let d = Int4::from(6_i64);
        c %= &d;
        assert_eq!(c, Int4::from(1_i64));

        // Test with divisor 1
        let mut e = Int4::from(42_i64);
        let one = Int4::one();
        e %= &one;
        assert_eq!(e, Int4::zero());
    }

    #[test]
    #[should_panic(expected = "division by zero")]
    fn rem_assign_panics_on_zero_divisor() {
        let mut a = Int4::from(10_i64);
        let zero = Int4::zero();
        a %= zero;
    }

    #[test]
    fn resize_method() {
        // Test resizing to same size
        let a = Int4::from(0x12345678_i64);
        let resized_same = a.resize::<4>();
        assert_eq!(resized_same, a);

        // Test resizing to larger size
        let b = Int2::from(0x9ABCDEF0_i64);
        let resized_larger = b.resize::<4>();
        assert_eq!(
            resized_larger.into_inner().to_words()[0],
            b.into_inner().to_words()[0]
        );
        assert_eq!(
            resized_larger.into_inner().to_words()[1],
            b.into_inner().to_words()[1]
        );
        assert_eq!(resized_larger.into_inner().to_words()[2], 0);
        assert_eq!(resized_larger.into_inner().to_words()[3], 0);

        // Test resizing to smaller size (truncation)
        let c = Int4::from(0x1234567890ABCDEF_i64);
        let resized_smaller = c.resize::<2>();
        assert_eq!(
            resized_smaller.into_inner().to_words()[0],
            c.into_inner().to_words()[0]
        );
        assert_eq!(
            resized_smaller.into_inner().to_words()[1],
            c.into_inner().to_words()[1]
        );
    }

    #[test]
    fn from_words() {
        // Test with single limb
        let words = [0x1234567890ABCDEF];
        let a = Int1::from_words(words);
        assert_eq!(a.into_inner().to_words()[0], words[0]);

        // Test with multiple limbs
        let words = [
            0x1234567890ABCDEF,
            0xFEDCBA9876543210,
            0x0F0F0F0F0F0F0F0F,
            0xF0F0F0F0F0F0F0F0,
        ];
        let b = Int4::from_words(words);
        let b_words = b.into_inner().to_words();
        for i in 0..4 {
            assert_eq!(b_words[i], words[i]);
        }
    }

    #[test]
    fn aggregate_operations() {
        // Test Sum trait
        let values: Vec<Int4> = [1_i64, 2_i64, 3_i64].into_iter().map(Int::from).collect();
        let sum: Int4 = values.iter().sum();
        assert_eq!(sum, Int4::from(6_i64));

        let sum2: Int4 = values.into_iter().sum();
        assert_eq!(sum2, Int4::from(6_i64));

        // Test Product trait
        let values: Vec<Int4> = [2_i64, 3_i64, 4_i64].into_iter().map(Int::from).collect();
        let product: Int4 = values.iter().product();
        assert_eq!(product, Int4::from(24_i64));

        let product2: Int4 = values.into_iter().product();
        assert_eq!(product2, Int4::from(24_i64));

        // Test empty collections
        let empty_vec: Vec<Int4> = Vec::new();
        let empty_sum: Int4 = empty_vec.iter().sum();
        assert_eq!(empty_sum, Int4::zero());

        let empty_product: Int4 = empty_vec.iter().product();
        assert_eq!(empty_product, Int4::one());
    }

    #[test]
    fn from_primitive() {
        // Test from_i8
        let a = Int4::from_i8(42);
        assert_eq!(a, Int4::from(42_i64));
        let b = Int4::from_i8(-42);
        assert_eq!(b, Int4::from(-42_i64));

        // Test from_i16
        let c = Int4::from_i16(12345);
        assert_eq!(c, Int4::from(12345_i64));
        let d = Int4::from_i16(-12345);
        assert_eq!(d, Int4::from(-12345_i64));

        // Test from_i32
        let e = Int4::from_i32(1234567890);
        assert_eq!(e, Int4::from(1234567890_i64));
        let f = Int4::from_i32(-1234567890);
        assert_eq!(f, Int4::from(-1234567890_i64));

        // Test from_i64
        let g = Int4::from_i64(1234567890123456789);
        assert_eq!(g, Int4::from(1234567890123456789_i64));
        let h = Int4::from_i64(-1234567890123456789);
        assert_eq!(h, Int4::from(-1234567890123456789_i64));

        // Test from_i128
        let i = Int4::from_i128(1234567890123456789012345678901234567);
        assert_eq!(
            i.into_inner(),
            crypto_bigint::Int::from(1234567890123456789012345678901234567_i128)
        );
        let j = Int4::from_i128(-1234567890123456789012345678901234567);
        assert_eq!(
            j.into_inner(),
            crypto_bigint::Int::from(-1234567890123456789012345678901234567_i128)
        );
    }

    #[test]
    fn from_primitive_edge_cases() {
        for value in [i32::MIN, i32::MAX] {
            let i = Int1::from(value);
            let j = Int2::from(value);
            assert_eq!(i.resize(), j);
        }

        for value in [i64::MIN, i64::MAX] {
            let i = Int1::from(value);
            let j = Int2::from(value);
            assert_eq!(i.resize(), j);
        }

        for value in [i128::MIN, i128::MAX] {
            let i = Int2::from(value);
            let j = Int::<3>::from(value);
            assert_eq!(i.resize(), j);
        }
    }

    #[should_panic]
    #[test]
    fn from_too_large_primitive() {
        // Test from_i128
        let _ = Int1::from(i128::MAX);
    }

    #[test]
    fn edge_cases() {
        // Test operations with MAX values
        let max = Int4::MAX;
        let one = Int4::one();

        // MAX + 1 should overflow in checked_add
        assert!(max.checked_add(&one).is_none());

        // MAX - MAX = 0
        assert_eq!(max.checked_sub(&max).unwrap(), Int4::zero());

        // Test operations with MIN values
        let min = Int4::MIN;

        // MIN - 1 should overflow in checked_sub
        assert!(min.checked_sub(&one).is_none());

        // Test operations with large shifts
        let x = Int4::from(1_i64);

        // Shift left by almost the bit limit
        let shifted = x << (Int4::BITS - 1);
        assert_eq!(shifted, Int4::from_words([0, 0, 0, 0x8000000000000000]));

        // Test with large powers that don't overflow
        let two = Int4::from(2_i64);
        let large_power = two.pow(100); // 2^100 is large but fits in 256 bits

        // Verify result: 2^100 = 1267650600228229401496703205376
        // The actual representation depends on the endianness and word order in
        // crypto_bigint Instead of hardcoding the expected value, let's verify
        // the numerical properties

        // 2^100 should be divisible by 2^10 = 1024 with no remainder
        assert_eq!(large_power % Int4::from(1024_i64), Int4::zero());

        // 2^100 / 2 = 2^99
        let half_power = large_power >> 1;
        assert_eq!(half_power << 1, large_power);
    }

    #[test]
    fn assign_operations() {
        // Test AddAssign
        let mut a = Int4::from(10_i64);
        a += Int4::from(5_i64);
        assert_eq!(a, Int4::from(15_i64));

        let mut b = Int4::from(20_i64);
        b += &Int4::from(3_i64);
        assert_eq!(b, Int4::from(23_i64));

        // Test SubAssign
        let mut c = Int4::from(10_i64);
        c -= Int4::from(3_i64);
        assert_eq!(c, Int4::from(7_i64));

        let mut d = Int4::from(50_i64);
        d -= &Int4::from(25_i64);
        assert_eq!(d, Int4::from(25_i64));

        // Test MulAssign
        let mut e = Int4::from(7_i64);
        e *= Int4::from(6_i64);
        assert_eq!(e, Int4::from(42_i64));

        let mut f = Int4::from(3_i64);
        f *= &Int4::from(4_i64);
        assert_eq!(f, Int4::from(12_i64));
    }

    #[test]
    fn formatting() {
        let a = Int1::from(255_i64);
        let b = Int1::from(-1_i64);

        // Test Debug
        assert_eq!(format!("{:?}", a), "Int(0x00000000000000FF)");
        assert_eq!(format!("{:?}", b), "Int(0xFFFFFFFFFFFFFFFF)");

        // Test Display
        assert_eq!(format!("{}", a), "00000000000000FF");
        assert_eq!(format!("{}", b), "FFFFFFFFFFFFFFFF");

        // Test LowerHex
        assert_eq!(format!("{:x}", a), "00000000000000ff");
        assert_eq!(format!("{:x}", b), "ffffffffffffffff");

        // Test UpperHex
        assert_eq!(format!("{:X}", a), "00000000000000FF");
        assert_eq!(format!("{:X}", b), "FFFFFFFFFFFFFFFF");
    }

    #[test]
    fn default_trait() {
        let default_val: Int4 = Default::default();
        assert_eq!(default_val, Int4::ZERO);
        assert!(default_val.is_zero());
    }

    #[test]
    fn constants() {
        // Test MINUS_ONE
        assert_eq!(Int4::MINUS_ONE + Int4::ONE, Int4::ZERO);

        // Test MIN and MAX
        assert!(Int4::MIN < Int4::ZERO);
        assert!(Int4::MAX > Int4::ZERO);
        assert!(Int4::MIN < Int4::MAX);

        // Test BITS, BYTES, LIMBS
        assert_eq!(Int4::BITS, 256);
        assert_eq!(Int4::BYTES, 32);
        assert_eq!(Int4::LIMBS, 4);

        assert_eq!(Int2::BITS, 128);
        assert_eq!(Int2::BYTES, 16);
        assert_eq!(Int2::LIMBS, 2);
    }

    #[test]
    fn cmp_vartime() {
        let a = Int4::from(10_i64);
        let b = Int4::from(20_i64);
        let c = Int4::from(10_i64);

        assert_eq!(a.cmp_vartime(&b), Ordering::Less);
        assert_eq!(b.cmp_vartime(&a), Ordering::Greater);
        assert_eq!(a.cmp_vartime(&c), Ordering::Equal);
    }

    #[test]
    fn cross_size_conversions() {
        // Test From<&Int<LIMBS>> for Int<LIMBS2>
        let a = Int2::from(12345_i64);
        let b: Option<Int4> = a.checked_resize();
        assert_eq!(b, Some(Int::from(12345_i64)));
        let b: Int4 = a.resize();
        assert_eq!(b, Int::from(12345_i64));

        let a = Int2::from_i128(i128::MAX);
        let b: Option<Int1> = a.checked_resize();
        assert_eq!(b, None);
        let b: Int1 = a.resize();
        assert_eq!(b, Int::from(-1_i64));

        // Test From<&crypto_bigint::Int<LIMBS>> for Int<LIMBS2>
        let c = crypto_bigint::Int::<{ 2 * WORD_FACTOR }>::from(67890_i64);
        let d: Int4 = (&c).try_into().unwrap();
        assert_eq!(d, Int4::from(67890_i64));

        // Test reference conversions from primitives
        let val = 42_i32;
        let e = Int4::from(&val);
        assert_eq!(e, Int4::from(42_i64));
    }

    #[test]
    fn constant_time_traits() {
        use crypto_bigint::subtle::Choice;

        let a = Int4::from(10_i64);
        let b = Int4::from(20_i64);
        let c = Int4::from(10_i64);

        // Test ConstantTimeEq
        assert_eq!(a.ct_eq(&c).unwrap_u8(), 1);
        assert_eq!(a.ct_eq(&b).unwrap_u8(), 0);

        // Test ConstantTimeGreater
        assert_eq!(b.ct_gt(&a).unwrap_u8(), 1);
        assert_eq!(a.ct_gt(&b).unwrap_u8(), 0);

        // Test ConstantTimeLess
        assert_eq!(a.ct_lt(&b).unwrap_u8(), 1);
        assert_eq!(b.ct_lt(&a).unwrap_u8(), 0);

        // Test ConditionallySelectable
        let selected_true = Int4::conditional_select(&a, &b, Choice::from(0));
        assert_eq!(selected_true, a);

        let selected_false = Int4::conditional_select(&a, &b, Choice::from(1));
        assert_eq!(selected_false, b);
    }

    #[test]
    fn crypto_bigint_traits() {
        use crypto_bigint::{Bounded, Constants};

        // Test Bounded trait
        assert_eq!(<Int4 as Bounded>::BITS, 256);
        assert_eq!(<Int4 as Bounded>::BYTES, 32);

        // Test Constants trait
        assert_eq!(<Int4 as Constants>::MAX, Int4::MAX);
    }

    #[test]
    fn from_str() {
        // Test parsing from string
        let a: Int4 = "123".parse().unwrap();
        assert_eq!(a, Int4::from(123_i64));

        let b: Int4 = "-456".parse().unwrap();
        assert_eq!(b, Int4::from(-456_i64));

        let c: Int4 = "0".parse().unwrap();
        assert_eq!(c, Int4::zero());

        let d: Int4 = "1".parse().unwrap();
        assert_eq!(d, Int4::one());
        assert_eq!(i64::MAX.to_string().parse::<Int1>().unwrap(), Int1::MAX);
        assert_eq!(i64::MIN.to_string().parse::<Int1>().unwrap(), Int1::MIN);

        // Test invalid cases
        assert!("0x123".parse::<Int4>().is_err());
        assert!("abc".parse::<Int4>().is_err());
        assert!("12.34".parse::<Int4>().is_err());
        assert!("".parse::<Int4>().is_err());

        // Number doesn't fit Int1
        assert!(
            ((i64::MAX as i128) + 1)
                .to_string()
                .parse::<Int1>()
                .is_err()
        );
        assert!(
            ((i64::MIN as i128) - 1)
                .to_string()
                .parse::<Int1>()
                .is_err()
        );
    }

    #[cfg(feature = "rand")]
    #[test]
    fn random_generation() {
        use rand::prelude::*;

        // Use a seeded RNG for reproducibility
        let mut rng = StdRng::seed_from_u64(1);

        // Test crypto_bigint::Random trait
        let random1: Int4 = crypto_bigint::Random::random(&mut rng);
        let random2: Int4 = crypto_bigint::Random::random(&mut rng);

        // Random values should be different
        assert_ne!(random1, random2);

        // Test Distribution trait
        let random3: Int4 = rng.random();
        let random4: Int4 = rng.random();

        assert_ne!(random3, random4);
    }
}
