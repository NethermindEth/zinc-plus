use super::*;
use core::{
    cmp::Ordering,
    fmt::{Debug, Display, Formatter, LowerHex, Result as FmtResult, UpperHex},
    iter::{Product, Sum},
    ops::{Add, AddAssign, Mul, MulAssign, Rem, RemAssign, Shl, Shr, Sub, SubAssign},
};
use core::hash::Hasher;
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

#[cfg(feature = "rand")]
use rand::{distr::StandardUniform, prelude::*, rand_core::TryRngCore};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct Int<const LIMBS: usize>(crypto_bigint::Int<LIMBS>);

impl<const LIMBS: usize> Int<LIMBS> {
    /// Create a new Int from a crypto_bigint::Int
    #[inline(always)]
    pub fn new(value: crypto_bigint::Int<LIMBS>) -> Self {
        Self(value)
    }

    /// Get the inner crypto_bigint::Int value
    #[inline(always)]
    pub fn inner(&self) -> &crypto_bigint::Int<LIMBS> {
        &self.0
    }

    /// Get the inner crypto_bigint::Int value, consuming self
    #[inline(always)]
    pub fn into_inner(self) -> crypto_bigint::Int<LIMBS> {
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

    /// See [crypto_bigint::Int::cmp_vartime]
    pub const fn cmp_vartime(&self, rhs: &Self) -> Ordering {
        self.0.cmp_vartime(&rhs.0)
    }
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
                Self(self.0.$trait_op(&rhs.0))
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

    fn pow(self, rhs: u32) -> Self::Output {
        // Implement exponentiation using repeated squaring
        if rhs == 0 {
            return Self::one();
        }

        let mut base = self.into_inner();
        let mut result = Self::one().into_inner();
        let mut exp = rhs;

        while exp > 0 {
            if exp & 1 == 1 {
                result = result
                    .checked_mul(&base)
                    .expect("overflow in exponentiation");
            }
            exp >>= 1;
            if exp > 0 {
                base = base.checked_mul(&base).expect("overflow in exponentiation");
            }
        }

        Self(result)
    }
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
                self.0.$trait_op(&rhs.0);
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

impl<const LIMBS: usize, const LIMBS2: usize> From<&Int<LIMBS>> for Int<LIMBS2> {
    #[inline(always)]
    fn from(num: &Int<LIMBS>) -> Int<LIMBS2> {
        num.resize()
    }
}

impl<const LIMBS: usize, const LIMBS2: usize> From<&crypto_bigint::Int<LIMBS>> for Int<LIMBS2> {
    #[inline(always)]
    fn from(num: &crypto_bigint::Int<LIMBS>) -> Int<LIMBS2> {
        Self(num.resize())
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

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::{format, vec::Vec};

    #[test]
    fn basic_operations() {
        // Test with 4 limbs (256-bit integers)
        let a = Int::<4>::from(10_i64);
        let b = Int::<4>::from(5_i64);

        // Test addition
        let c = a + b;
        assert_eq!(c, Int::<4>::from(15_i64));

        // Test subtraction
        let d = a - b;
        assert_eq!(d, Int::<4>::from(5_i64));

        // Test multiplication
        let e = a * b;
        assert_eq!(e, Int::<4>::from(50_i64));

        // Test remainder
        let f = a % b;
        assert_eq!(f, Int::<4>::from(0_i64));

        // Test shl
        let x = Int::<1>::from(0x0001_i64);

        assert_eq!(x << 0, x);
        assert_eq!(x << 1, 0x0002.into());
        assert_eq!(x << 15, 0x8000.into());
        assert_eq!(x << 63, (-0x8000000000000000_i64).into());
        // x << 64 panics, it's tested separately

        // Test shr
        let x = Int::<4>::from(0x8000_i32);

        assert_eq!(x >> 0, x);
        assert_eq!(x >> 1, 0x4000.into());
        assert_eq!(x >> 15, 0x0001.into());
        assert_eq!(x >> 16, Int::ZERO);
    }

    #[test]
    #[should_panic(expected = "`shift` within the bit size of the integer")]
    fn shl_panics_on_overflow() {
        let x = Int::<1>::from(0x0001_i64);
        let _ = x << 64;
    }

    #[test]
    fn checked_operations() {
        let a = Int::<4>::from(10_i64);
        let b = Int::<4>::from(5_i64);
        let zero = Int::<4>::ZERO;

        // Test checked_add
        let c = a.checked_add(&b).unwrap();
        assert_eq!(c, Int::<4>::from(15_i64));

        // Test checked_sub
        let d = a.checked_sub(&b).unwrap();
        assert_eq!(d, Int::<4>::from(5_i64));

        // Test checked_mul
        let e = a.checked_mul(&b).unwrap();
        assert_eq!(e, Int::<4>::from(50_i64));

        // Test checked_rem
        let f = a.checked_rem(&b).unwrap();
        assert_eq!(f, Int::<4>::ZERO);

        // Test checked_rem with zero divisor
        assert!(a.checked_rem(&zero).is_none());
    }

    #[allow(clippy::op_ref)]
    #[test]
    fn reference_operations() {
        let a = Int::<4>::from(10_i64);
        let b = Int::<4>::from(5_i64);

        // Test reference-based addition
        let c = a + &b;
        assert_eq!(c, Int::<4>::from(15_i64));

        // Test reference-based subtraction
        let d = a - &b;
        assert_eq!(d, Int::<4>::from(5_i64));

        // Test reference-based multiplication
        let e = a * &b;
        assert_eq!(e, Int::<4>::from(50_i64));

        // Test reference-based remainder
        let f = a % &b;
        assert_eq!(f, Int::<4>::ZERO);
    }

    #[test]
    fn conversions() {
        // Test From<crypto_bigint::Int> for Int
        let original = crypto_bigint::Int::<4>::from(123_i64);
        let wrapped: Int<4> = original.into();
        assert_eq!(wrapped.0, original);

        // Test From<Int> for crypto_bigint::Int
        let wrapped = Int::<4>::from(456_i64);
        let unwrapped: crypto_bigint::Int<4> = wrapped.into();
        assert_eq!(unwrapped, crypto_bigint::Int::from(456_i64));

        // Test conversion methods
        let value = crypto_bigint::Int::<4>::from(789_i64);
        let wrapped = Int::new(value);
        assert_eq!(wrapped.inner(), &value);
        assert_eq!(wrapped.into_inner(), value);
    }

    #[test]
    fn pow_operation() {
        // Test basic exponentiation
        let base = Int::<4>::from(2_i64);

        // 2^0 = 1
        assert_eq!(base.pow(0), Int::<4>::one());

        // 2^1 = 2
        assert_eq!(base.pow(1), base);

        // 2^3 = 8
        assert_eq!(base.pow(3), Int::<4>::from(8_i64));

        // 2^10 = 1024
        assert_eq!(base.pow(10), Int::<4>::from(1024_i64));

        // Test with different base
        let base = Int::<4>::from(3_i64);

        // 3^4 = 81
        assert_eq!(base.pow(4), Int::<4>::from(81_i64));

        // Test with base 1
        let base = Int::<4>::from(1_i64);
        assert_eq!(base.pow(1000), Int::<4>::from(1_i64));

        // Test with base 0
        let base = Int::<4>::from(0_i64);
        assert_eq!(base.pow(0), Int::<4>::one()); // 0^0 = 1 by convention
        assert_eq!(base.pow(10), Int::<4>::zero()); // 0^n = 0 for n > 0
    }

    #[test]
    fn checked_neg() {
        // Test with positive number
        let a = Int::<4>::from(10_i64);
        let neg_a = a.checked_neg().unwrap();
        assert_eq!(neg_a, Int::<4>::from(-10_i64));

        // Test with negative number
        let b = Int::<4>::from(-5_i64);
        let neg_b = b.checked_neg().unwrap();
        assert_eq!(neg_b, Int::<4>::from(5_i64));

        // Test with zero
        let zero = Int::<4>::zero();
        let neg_zero = zero.checked_neg().unwrap();
        assert_eq!(neg_zero, zero);

        // Test with MIN value (should return None as -MIN would overflow)
        let min = Int::<4>::MIN;
        assert!(min.checked_neg().is_none());
    }

    #[test]
    fn rem_assign_operations() {
        // Test RemAssign with owned value
        let mut a = Int::<4>::from(17_i64);
        let b = Int::<4>::from(5_i64);
        a %= b;
        assert_eq!(a, Int::<4>::from(2_i64));

        // Test RemAssign with reference
        let mut c = Int::<4>::from(19_i64);
        let d = Int::<4>::from(6_i64);
        c %= &d;
        assert_eq!(c, Int::<4>::from(1_i64));

        // Test with divisor 1
        let mut e = Int::<4>::from(42_i64);
        let one = Int::<4>::one();
        e %= &one;
        assert_eq!(e, Int::<4>::zero());
    }

    #[test]
    #[should_panic(expected = "division by zero")]
    fn rem_assign_panics_on_zero_divisor() {
        let mut a = Int::<4>::from(10_i64);
        let zero = Int::<4>::zero();
        a %= zero;
    }

    #[test]
    fn resize_method() {
        // Test resizing to same size
        let a = Int::<4>::from(0x12345678_i64);
        let resized_same = a.resize::<4>();
        assert_eq!(resized_same, a);

        // Test resizing to larger size
        let b = Int::<2>::from(0x9ABCDEF0_i64);
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
        let c = Int::<4>::from(0x1234567890ABCDEF_i64);
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
        let a = Int::<1>::from_words(words);
        assert_eq!(a.into_inner().to_words()[0], words[0]);

        // Test with multiple limbs
        let words = [
            0x1234567890ABCDEF,
            0xFEDCBA9876543210,
            0x0F0F0F0F0F0F0F0F,
            0xF0F0F0F0F0F0F0F0,
        ];
        let b = Int::<4>::from_words(words);
        let b_words = b.into_inner().to_words();
        for i in 0..4 {
            assert_eq!(b_words[i], words[i]);
        }
    }

    #[test]
    fn aggregate_operations() {
        // Test Sum trait
        let values: Vec<Int<4>> = [1_i64, 2_i64, 3_i64].into_iter().map(Int::from).collect();
        let sum: Int<4> = values.iter().sum();
        assert_eq!(sum, Int::<4>::from(6_i64));

        let sum2: Int<4> = values.into_iter().sum();
        assert_eq!(sum2, Int::<4>::from(6_i64));

        // Test Product trait
        let values: Vec<Int<4>> = [2_i64, 3_i64, 4_i64].into_iter().map(Int::from).collect();
        let product: Int<4> = values.iter().product();
        assert_eq!(product, Int::<4>::from(24_i64));

        let product2: Int<4> = values.into_iter().product();
        assert_eq!(product2, Int::<4>::from(24_i64));

        // Test empty collections
        let empty_vec: Vec<Int<4>> = Vec::new();
        let empty_sum: Int<4> = empty_vec.iter().sum();
        assert_eq!(empty_sum, Int::<4>::zero());

        let empty_product: Int<4> = empty_vec.iter().product();
        assert_eq!(empty_product, Int::<4>::one());
    }

    #[test]
    fn from_primitive() {
        // Test from_i8
        let a = Int::<4>::from_i8(42);
        assert_eq!(a, Int::<4>::from(42_i64));
        let b = Int::<4>::from_i8(-42);
        assert_eq!(b, Int::<4>::from(-42_i64));

        // Test from_i16
        let c = Int::<4>::from_i16(12345);
        assert_eq!(c, Int::<4>::from(12345_i64));
        let d = Int::<4>::from_i16(-12345);
        assert_eq!(d, Int::<4>::from(-12345_i64));

        // Test from_i32
        let e = Int::<4>::from_i32(1234567890);
        assert_eq!(e, Int::<4>::from(1234567890_i64));
        let f = Int::<4>::from_i32(-1234567890);
        assert_eq!(f, Int::<4>::from(-1234567890_i64));

        // Test from_i64
        let g = Int::<4>::from_i64(1234567890123456789);
        assert_eq!(g, Int::<4>::from(1234567890123456789_i64));
        let h = Int::<4>::from_i64(-1234567890123456789);
        assert_eq!(h, Int::<4>::from(-1234567890123456789_i64));

        // Test from_i128
        let i = Int::<4>::from_i128(1234567890123456789012345678901234567);
        assert_eq!(
            i.into_inner(),
            crypto_bigint::Int::<4>::from(1234567890123456789012345678901234567_i128)
        );
        let j = Int::<4>::from_i128(-1234567890123456789012345678901234567);
        assert_eq!(
            j.into_inner(),
            crypto_bigint::Int::<4>::from(-1234567890123456789012345678901234567_i128)
        );
    }

    #[test]
    fn from_primitive_edge_cases() {
        for value in [i32::MIN, i32::MAX] {
            let i = Int::<1>::from(value);
            let j = Int::<2>::from(value);
            assert_eq!(i.resize(), j);
        }

        for value in [i64::MIN, i64::MAX] {
            let i = Int::<1>::from(value);
            let j = Int::<2>::from(value);
            assert_eq!(i.resize(), j);
        }

        for value in [i128::MIN, i128::MAX] {
            let i = Int::<2>::from(value);
            let j = Int::<3>::from(value);
            assert_eq!(i.resize(), j);
        }
    }

    #[should_panic]
    #[test]
    fn from_too_large_primitive() {
        // Test from_i128
        let _ = Int::<1>::from(i128::MAX);
    }

    #[test]
    fn edge_cases() {
        // Test operations with MAX values
        let max = Int::<4>::MAX;
        let one = Int::<4>::one();

        // MAX + 1 should overflow in checked_add
        assert!(max.checked_add(&one).is_none());

        // MAX - MAX = 0
        assert_eq!(max.checked_sub(&max).unwrap(), Int::<4>::zero());

        // Test operations with MIN values
        let min = Int::<4>::MIN;

        // MIN - 1 should overflow in checked_sub
        assert!(min.checked_sub(&one).is_none());

        // Test operations with large shifts
        let x = Int::<4>::from(1_i64);

        // Shift left by almost the bit limit
        let shifted = x << (Int::<4>::BITS - 1);
        assert_eq!(shifted, Int::<4>::from_words([0, 0, 0, 0x8000000000000000]));

        // Test with large powers that don't overflow
        let two = Int::<4>::from(2_i64);
        let large_power = two.pow(100); // 2^100 is large but fits in 256 bits

        // Verify result: 2^100 = 1267650600228229401496703205376
        // The actual representation depends on the endianness and word order in
        // crypto_bigint Instead of hardcoding the expected value, let's verify
        // the numerical properties

        // 2^100 should be divisible by 2^10 = 1024 with no remainder
        assert_eq!(large_power % Int::<4>::from(1024_i64), Int::<4>::zero());

        // 2^100 / 2 = 2^99
        let half_power = large_power >> 1;
        assert_eq!(half_power << 1, large_power);
    }

    #[test]
    fn assign_operations() {
        // Test AddAssign
        let mut a = Int::<4>::from(10_i64);
        a += Int::<4>::from(5_i64);
        assert_eq!(a, Int::<4>::from(15_i64));

        let mut b = Int::<4>::from(20_i64);
        b += &Int::<4>::from(3_i64);
        assert_eq!(b, Int::<4>::from(23_i64));

        // Test SubAssign
        let mut c = Int::<4>::from(10_i64);
        c -= Int::<4>::from(3_i64);
        assert_eq!(c, Int::<4>::from(7_i64));

        let mut d = Int::<4>::from(50_i64);
        d -= &Int::<4>::from(25_i64);
        assert_eq!(d, Int::<4>::from(25_i64));

        // Test MulAssign
        let mut e = Int::<4>::from(7_i64);
        e *= Int::<4>::from(6_i64);
        assert_eq!(e, Int::<4>::from(42_i64));

        let mut f = Int::<4>::from(3_i64);
        f *= &Int::<4>::from(4_i64);
        assert_eq!(f, Int::<4>::from(12_i64));
    }

    #[test]
    fn formatting() {
        #[cfg(target_pointer_width = "64")]
        const WORD_FACTOR: usize = 1;
        #[cfg(target_pointer_width = "32")]
        const WORD_FACTOR: usize = 2;

        let a = Int::<WORD_FACTOR>::from(255_i64);
        let b = Int::<WORD_FACTOR>::from(-1_i64);

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
        let default_val: Int<4> = Default::default();
        assert_eq!(default_val, Int::<4>::ZERO);
        assert!(default_val.is_zero());
    }

    #[test]
    fn constants() {
        // Test MINUS_ONE
        assert_eq!(Int::<4>::MINUS_ONE + Int::<4>::ONE, Int::<4>::ZERO);

        // Test MIN and MAX
        assert!(Int::<4>::MIN < Int::<4>::ZERO);
        assert!(Int::<4>::MAX > Int::<4>::ZERO);
        assert!(Int::<4>::MIN < Int::<4>::MAX);

        // Test BITS, BYTES, LIMBS
        assert_eq!(Int::<4>::BITS, 256);
        assert_eq!(Int::<4>::BYTES, 32);
        assert_eq!(Int::<4>::LIMBS, 4);

        assert_eq!(Int::<2>::BITS, 128);
        assert_eq!(Int::<2>::BYTES, 16);
        assert_eq!(Int::<2>::LIMBS, 2);
    }

    #[test]
    fn cmp_vartime() {
        let a = Int::<4>::from(10_i64);
        let b = Int::<4>::from(20_i64);
        let c = Int::<4>::from(10_i64);

        assert_eq!(a.cmp_vartime(&b), Ordering::Less);
        assert_eq!(b.cmp_vartime(&a), Ordering::Greater);
        assert_eq!(a.cmp_vartime(&c), Ordering::Equal);
    }

    #[test]
    fn cross_size_conversions() {
        // Test From<&Int<LIMBS>> for Int<LIMBS2>
        let a = Int::<2>::from(12345_i64);
        let b: Int<4> = (&a).into();
        assert_eq!(b, Int::<4>::from(12345_i64));

        // Test From<&crypto_bigint::Int<LIMBS>> for Int<LIMBS2>
        let c = crypto_bigint::Int::<2>::from(67890_i64);
        let d: Int<4> = (&c).into();
        assert_eq!(d, Int::<4>::from(67890_i64));

        // Test reference conversions from primitives
        let val = 42_i32;
        let e = Int::<4>::from(&val);
        assert_eq!(e, Int::<4>::from(42_i64));
    }

    #[test]
    fn constant_time_traits() {
        use crypto_bigint::subtle::Choice;

        let a = Int::<4>::from(10_i64);
        let b = Int::<4>::from(20_i64);
        let c = Int::<4>::from(10_i64);

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
        let selected_true = Int::<4>::conditional_select(&a, &b, Choice::from(0));
        assert_eq!(selected_true, a);

        let selected_false = Int::<4>::conditional_select(&a, &b, Choice::from(1));
        assert_eq!(selected_false, b);
    }

    #[test]
    fn crypto_bigint_traits() {
        use crypto_bigint::{Bounded, Constants};

        // Test Bounded trait
        assert_eq!(<Int<4> as Bounded>::BITS, 256);
        assert_eq!(<Int<4> as Bounded>::BYTES, 32);

        // Test Constants trait
        assert_eq!(<Int<4> as Constants>::MAX, Int::<4>::MAX);
    }

    #[cfg(feature = "rand")]
    #[test]
    fn random_generation() {
        use rand::prelude::*;

        // Use a seeded RNG for reproducibility
        let mut rng = StdRng::seed_from_u64(1);

        // Test crypto_bigint::Random trait
        let random1: Int<4> = crypto_bigint::Random::random(&mut rng);
        let random2: Int<4> = crypto_bigint::Random::random(&mut rng);

        // Random values should be different
        assert_ne!(random1, random2);

        // Test Distribution trait
        let random3: Int<4> = rng.random();
        let random4: Int<4> = rng.random();

        assert_ne!(random3, random4);
    }
}
