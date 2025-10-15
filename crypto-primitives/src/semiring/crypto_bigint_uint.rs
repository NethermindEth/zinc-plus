use super::*;
use crate::{crypto_bigint_int::Int, impl_pow_via_repeated_squaring};
use core::{
    cmp::Ordering,
    fmt::{Debug, Display, Formatter, LowerHex, Result as FmtResult, UpperHex},
    hash::{Hash, Hasher},
    iter::{Product, Sum},
    ops::{
        Add, AddAssign, Mul, MulAssign, Rem, RemAssign, Shl, ShlAssign, Shr, ShrAssign, Sub,
        SubAssign,
    },
    str::FromStr,
};
use crypto_bigint::{Integer, Limb, Word};
use num_traits::{
    CheckedAdd, CheckedMul, CheckedRem, CheckedSub, ConstOne, ConstZero, One, Pow, Zero,
};
use pastey::paste;

#[cfg(feature = "rand")]
use rand::{distr::StandardUniform, prelude::*, rand_core::TryRngCore};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct Uint<const LIMBS: usize>(crypto_bigint::Uint<LIMBS>);

impl<const LIMBS: usize> Uint<LIMBS> {
    /// Wraps a given value into this wrapper type
    #[inline(always)]
    pub const fn new(value: crypto_bigint::Uint<LIMBS>) -> Self {
        Self(value)
    }

    #[inline(always)]
    pub const fn new_ref(value: &crypto_bigint::Uint<LIMBS>) -> &Self {
        // Safety: Uint<LIMBS> is #[repr(transparent)] and is guaranteed to have the
        // same memory layout as crypto_bigint::Uint
        unsafe { &*(value as *const crypto_bigint::Uint<LIMBS> as *const Self) }
    }

    #[inline(always)]
    pub const fn new_ref_mut(value: &mut crypto_bigint::Uint<LIMBS>) -> &mut Self {
        // Safety: Uint<LIMBS> is #[repr(transparent)] and is guaranteed to have the
        // same memory layout as crypto_bigint::Uint
        unsafe { &mut *(value as *mut crypto_bigint::Uint<LIMBS> as *mut Self) }
    }

    /// Get the reference to the wrapped value
    #[inline(always)]
    pub const fn inner(&self) -> &crypto_bigint::Uint<LIMBS> {
        &self.0
    }

    /// Get the wrapped value, consuming self
    #[inline(always)]
    pub const fn into_inner(self) -> crypto_bigint::Uint<LIMBS> {
        self.0
    }

    /// See [crypto_bigint::Uint::from_words]
    #[inline(always)]
    pub const fn from_words(arr: [Word; LIMBS]) -> Self {
        Self(crypto_bigint::Uint::from_words(arr))
    }

    /// See [crypto_bigint::Uint::to_words]
    #[inline]
    pub const fn to_words(self) -> [Word; LIMBS] {
        self.0.to_words()
    }

    /// See [crypto_bigint::Uint::as_words]
    pub const fn as_words(&self) -> &[Word; LIMBS] {
        self.0.as_words()
    }

    /// See [crypto_bigint::Uint::as_mut_words]
    pub const fn as_mut_words(&mut self) -> &mut [Word; LIMBS] {
        self.0.as_mut_words()
    }

    /// See [crypto_bigint::Uint::as_limbs]
    pub const fn as_limbs(&self) -> &[Limb; LIMBS] {
        self.0.as_limbs()
    }

    /// See [crypto_bigint::Uint::as_mut_limbs]
    pub const fn as_mut_limbs(&mut self) -> &mut [Limb; LIMBS] {
        self.0.as_mut_limbs()
    }

    /// See [crypto_bigint::Uint::to_limbs]
    pub const fn to_limbs(self) -> [Limb; LIMBS] {
        self.0.to_limbs()
    }

    /// See [crypto_bigint::Uint::resize]
    #[inline(always)]
    pub const fn resize<const T: usize>(&self) -> Uint<T> {
        Uint::<T>(self.0.resize())
    }

    pub const fn checked_resize<const T: usize>(&self) -> Option<Uint<T>> {
        match checked_resize::<LIMBS, T>(&self.0) {
            None => None,
            Some(inner) => Some(Uint(inner)),
        }
    }

    /// See [crypto_bigint::Uint::cmp_vartime]
    pub const fn cmp_vartime(&self, rhs: &Self) -> Ordering {
        self.0.cmp_vartime(&rhs.0)
    }

    /// See [crypto_bigint::Uint::as_int]
    pub const fn as_int(&self) -> &Int<LIMBS> {
        Int::new_ref(self.0.as_int())
    }
}

const fn checked_resize<const SRC: usize, const DST: usize>(
    num: &crypto_bigint::Uint<SRC>,
) -> Option<crypto_bigint::Uint<DST>> {
    if SRC > DST {
        let max = Uint::<DST>::MAX.0.resize();
        let cmp = num.cmp_vartime(&max);
        if cmp.is_gt() {
            return None;
        }
    }
    Some(num.resize())
}

macro_rules! define_consts {
    ($($name:ident),+) => {
        $(pub const $name: Self = Self(crypto_bigint::Uint::<LIMBS>::$name);)+
    };
}

impl<const LIMBS: usize> Uint<LIMBS> {
    define_consts!(MAX);

    /// Total size of the represented integer in bits.
    pub const BITS: u32 = crypto_bigint::Uint::<LIMBS>::BITS;

    /// Total size of the represented integer in bytes.
    pub const BYTES: usize = crypto_bigint::Uint::<LIMBS>::BYTES;

    /// The number of limbs used on this platform.
    pub const LIMBS: usize = LIMBS;
}

//
// Core traits
//

impl<const LIMBS: usize> Debug for Uint<LIMBS> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        Debug::fmt(&self.0, f)
    }
}

impl<const LIMBS: usize> Display for Uint<LIMBS> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        Display::fmt(&self.0, f)
    }
}

impl<const LIMBS: usize> Default for Uint<LIMBS> {
    #[inline(always)]
    fn default() -> Self {
        Self::ZERO
    }
}

impl<const LIMBS: usize> LowerHex for Uint<LIMBS> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        LowerHex::fmt(&self.0, f)
    }
}

impl<const LIMBS: usize> UpperHex for Uint<LIMBS> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        UpperHex::fmt(&self.0, f)
    }
}

impl<const LIMBS: usize> Hash for Uint<LIMBS> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state)
    }
}

impl<const LIMBS: usize> FromStr for Uint<LIMBS> {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let uint = crypto_bigint::Uint::<LIMBS>::from_str_radix_vartime(s, 10).map_err(|_| ())?;
        Ok(Self(uint))
    }
}

//
// Zero and One traits
//

impl<const LIMBS: usize> Zero for Uint<LIMBS> {
    #[inline(always)]
    fn zero() -> Self {
        Self::ZERO
    }

    #[inline(always)]
    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}

impl<const LIMBS: usize> One for Uint<LIMBS> {
    #[inline(always)]
    fn one() -> Self {
        Self::ONE
    }
}

impl<const LIMBS: usize> ConstZero for Uint<LIMBS> {
    const ZERO: Self = Self(crypto_bigint::Uint::ZERO);
}

impl<const LIMBS: usize> ConstOne for Uint<LIMBS> {
    const ONE: Self = Self(crypto_bigint::Uint::ONE);
}

//
// Basic arithmetic operations
//

macro_rules! impl_basic_op {
    ($trait_name:tt, $trait_op:tt) => {
        impl<const LIMBS: usize> $trait_name for Uint<LIMBS> {
            type Output = Self;

            #[inline(always)]
            fn $trait_op(self, rhs: Self) -> Self::Output {
                self.$trait_op(&rhs)
            }
        }

        impl<'a, const LIMBS: usize> $trait_name<&'a Self> for Uint<LIMBS> {
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

impl<const LIMBS: usize> Rem for Uint<LIMBS> {
    type Output = Self;

    #[inline(always)]
    fn rem(self, rhs: Self) -> Self::Output {
        self.rem(&rhs)
    }
}

impl<'a, const LIMBS: usize> Rem<&'a Self> for Uint<LIMBS> {
    type Output = Self;

    fn rem(self, rhs: &'a Self) -> Self::Output {
        let non_zero = crypto_bigint::NonZero::new(rhs.0).expect("division by zero");
        Self(self.0.rem(&non_zero))
    }
}

impl<const LIMBS: usize> Shl<u32> for Uint<LIMBS> {
    type Output = Self;

    #[inline(always)]
    fn shl(self, rhs: u32) -> Self::Output {
        Self(self.0.shl(rhs))
    }
}

impl<const LIMBS: usize> Shr<u32> for Uint<LIMBS> {
    type Output = Self;

    #[inline(always)]
    fn shr(self, rhs: u32) -> Self::Output {
        Self(self.0.shr(rhs))
    }
}

impl<const LIMBS: usize> Pow<u32> for Uint<LIMBS> {
    type Output = Self;

    impl_pow_via_repeated_squaring!();
}

//
// Checked arithmetic operations
//

impl<const LIMBS: usize> CheckedAdd for Uint<LIMBS> {
    fn checked_add(&self, other: &Self) -> Option<Self> {
        let (result, overflow) = self.0.carrying_add(&other.0, crypto_bigint::Limb::ZERO);
        if overflow.0 != 0 {
            None
        } else {
            Some(Self(result))
        }
    }
}

impl<const LIMBS: usize> CheckedSub for Uint<LIMBS> {
    fn checked_sub(&self, other: &Self) -> Option<Self> {
        let (result, borrow) = self.0.borrowing_sub(&other.0, crypto_bigint::Limb::ZERO);
        if borrow.0 != 0 {
            None
        } else {
            Some(Self(result))
        }
    }
}

impl<const LIMBS: usize> CheckedMul for Uint<LIMBS> {
    fn checked_mul(&self, other: &Self) -> Option<Self> {
        // Use widening_mul which returns (lo, hi)
        let (lo, hi) = self.0.widening_mul(&other.0);
        if hi.is_zero() { Some(Self(lo)) } else { None }
    }
}

impl<const LIMBS: usize> CheckedRem for Uint<LIMBS> {
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
        impl<const LIMBS: usize> $trait_name<Self> for Uint<LIMBS> {
            #[inline(always)]
            fn $trait_op(&mut self, rhs: Self) {
                self.$trait_op(&rhs);
            }
        }

        impl<'a, const LIMBS: usize> $trait_name<&'a Self> for Uint<LIMBS> {
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

impl<const LIMBS: usize> RemAssign for Uint<LIMBS> {
    #[inline(always)]
    fn rem_assign(&mut self, rhs: Self) {
        self.rem_assign(&rhs);
    }
}

impl<'a, const LIMBS: usize> RemAssign<&'a Self> for Uint<LIMBS> {
    #![allow(clippy::arithmetic_side_effects)]
    fn rem_assign(&mut self, rhs: &'a Self) {
        let non_zero = crypto_bigint::NonZero::new(rhs.0).expect("division by zero");
        self.0 %= non_zero;
    }
}

impl<const LIMBS: usize> ShlAssign<u32> for Uint<LIMBS> {
    #[inline(always)]
    fn shl_assign(&mut self, rhs: u32) {
        self.0.shl_assign(rhs);
    }
}

impl<const LIMBS: usize> ShrAssign<u32> for Uint<LIMBS> {
    #[inline(always)]
    fn shr_assign(&mut self, rhs: u32) {
        self.0.shr_assign(rhs);
    }
}

//
// Aggregate operations
//

impl<const LIMBS: usize> Sum for Uint<LIMBS> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, x| {
            acc.checked_add(&x).expect("overflow in sum")
        })
    }
}

impl<'a, const LIMBS: usize> Sum<&'a Self> for Uint<LIMBS> {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, x| {
            acc.checked_add(x).expect("overflow in sum")
        })
    }
}

impl<const LIMBS: usize> Product for Uint<LIMBS> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |acc, x| {
            acc.checked_mul(&x).expect("overflow in product")
        })
    }
}

impl<'a, const LIMBS: usize> Product<&'a Self> for Uint<LIMBS> {
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |acc, x| {
            acc.checked_mul(x).expect("overflow in product")
        })
    }
}

//
// Conversions
//

impl<const LIMBS: usize> From<crypto_bigint::Uint<LIMBS>> for Uint<LIMBS> {
    #[inline(always)]
    fn from(value: crypto_bigint::Uint<LIMBS>) -> Self {
        Self(value)
    }
}

impl<const LIMBS: usize> From<Uint<LIMBS>> for crypto_bigint::Uint<LIMBS> {
    #[inline(always)]
    fn from(value: Uint<LIMBS>) -> Self {
        value.0
    }
}

impl<const LIMBS: usize> From<bool> for Uint<LIMBS> {
    #[inline(always)]
    fn from(value: bool) -> Self {
        Self(crypto_bigint::Uint::<LIMBS>::from(u8::from(value)))
    }
}

macro_rules! impl_from_primitive {
    ($($t:ty),+) => {
        $(
            impl<const LIMBS: usize> From<$t> for Uint<LIMBS> {
                fn from(value: $t) -> Self {
                    assert!(core::mem::size_of::<$t>() <= crypto_bigint::Uint::<LIMBS>::BYTES,
                            "`{}` is too large to fit into `Uint<{LIMBS}>`", stringify!($t));
                    Self(crypto_bigint::Uint::<LIMBS>::from(value))
                }
            }

            impl<'a, const LIMBS: usize> From<&'a $t> for Uint<LIMBS> {
                #[inline(always)]
                fn from(value: &$t) -> Self {
                    Self::from(*value)
                }
            }

            impl<const LIMBS: usize> Uint<LIMBS> {
            paste! {
                /// Create a Uint from a primitive type.
                pub const fn [<from_ $t>](n: $t) -> Self {
                    Self(crypto_bigint::Uint::<LIMBS>::[<from_ $t>](n))
                }
            }
            }
        )+
    };
}

impl_from_primitive!(u8, u16, u32, u64, u128);

impl<const LIMBS: usize, const LIMBS2: usize> TryFrom<&crypto_bigint::Uint<LIMBS2>>
    for Uint<LIMBS>
{
    type Error = ();

    fn try_from(num: &crypto_bigint::Uint<LIMBS2>) -> Result<Self, Self::Error> {
        checked_resize(num).map(Self).ok_or(())
    }
}

//
// Semiring
//

impl<const LIMBS: usize> Semiring for Uint<LIMBS> {}

impl<const LIMBS: usize> IntSemiring for Uint<LIMBS> {
    fn is_odd(&self) -> bool {
        self.0.is_odd().into()
    }

    fn is_even(&self) -> bool {
        self.0.is_even().into()
    }
}

//
// RNG
//

#[cfg(feature = "rand")]
impl<const LIMBS: usize> Distribution<Uint<LIMBS>> for StandardUniform {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Uint<LIMBS> {
        crypto_bigint::Random::random(rng)
    }
}

#[cfg(feature = "rand")]
impl<const LIMBS: usize> crypto_bigint::Random for Uint<LIMBS> {
    fn try_random<R: TryRngCore + ?Sized>(rng: &mut R) -> Result<Self, R::Error> {
        crypto_bigint::Uint::try_random(rng).map(Self)
    }
}

//
// Serialization and Deserialization
//

#[cfg(feature = "serde")]
impl<'de, const LIMBS: usize> serde::Deserialize<'de> for Uint<LIMBS>
where
    crypto_bigint::Uint<LIMBS>: crypto_bigint::Encoding,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        crypto_bigint::Uint::<LIMBS>::deserialize(deserializer).map(Self)
    }
}

#[cfg(feature = "serde")]
impl<const LIMBS: usize> serde::Serialize for Uint<LIMBS>
where
    crypto_bigint::Uint<LIMBS>: crypto_bigint::Encoding,
{
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

impl<const LIMBS: usize> crypto_bigint::subtle::ConstantTimeEq for Uint<LIMBS> {
    #[inline]
    fn ct_eq(&self, other: &Self) -> crypto_bigint::subtle::Choice {
        crypto_bigint::subtle::ConstantTimeEq::ct_eq(&self.0, &other.0)
    }
}

impl<const LIMBS: usize> crypto_bigint::subtle::ConstantTimeGreater for Uint<LIMBS> {
    #[inline]
    fn ct_gt(&self, other: &Self) -> crypto_bigint::subtle::Choice {
        crypto_bigint::subtle::ConstantTimeGreater::ct_gt(&self.0, &other.0)
    }
}

impl<const LIMBS: usize> crypto_bigint::subtle::ConstantTimeLess for Uint<LIMBS> {
    #[inline]
    fn ct_lt(&self, other: &Self) -> crypto_bigint::subtle::Choice {
        crypto_bigint::subtle::ConstantTimeLess::ct_lt(&self.0, &other.0)
    }
}

impl<const LIMBS: usize> crypto_bigint::subtle::ConditionallySelectable for Uint<LIMBS> {
    fn conditional_select(a: &Self, b: &Self, choice: crypto_bigint::subtle::Choice) -> Self {
        crypto_bigint::subtle::ConditionallySelectable::conditional_select(&a.0, &b.0, choice)
            .into()
    }
}

impl<const LIMBS: usize> crypto_bigint::Bounded for Uint<LIMBS> {
    const BITS: u32 = Self::BITS;
    const BYTES: usize = Self::BYTES;
}

impl<const LIMBS: usize> crypto_bigint::Constants for Uint<LIMBS> {
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

    type Uint1 = Uint<WORD_FACTOR>;
    type Uint2 = Uint<{ WORD_FACTOR * 2 }>;
    type Uint4 = Uint<{ WORD_FACTOR * 4 }>;

    #[test]
    fn ensure_blanket_traits() {
        ensure_type_implements_trait!(Uint4, ConstIntSemiring);
        ensure_type_implements_trait!(Uint4, IntSemiringWithShifts);
    }

    #[test]
    fn basic_operations() {
        let a = Uint4::from(10_u64);
        let b = Uint4::from(5_u64);

        // Test addition
        assert_eq!(a + b, Uint4::from(15_u64));

        // Test subtraction
        assert_eq!(a - b, Uint4::from(5_u64));

        // Test multiplication
        assert_eq!(a * b, Uint4::from(50_u64));

        // Test remainder
        assert_eq!(a % b, Uint4::ZERO);

        // Test shl
        let x = Uint1::from(0x0001_u64);
        assert_eq!(x << 0, x);
        assert_eq!(x << 1, 0x0002_u64.into());
        assert_eq!(x << 15, 0x8000_u64.into());

        // Test shr
        let x = Uint4::from(0x8000_u32);
        assert_eq!(x >> 0, x);
        assert_eq!(x >> 1, 0x4000_u64.into());
        assert_eq!(x >> 15, 0x0001_u64.into());
    }

    #[test]
    #[should_panic(expected = "`shift` within the bit size of the integer")]
    fn shl_panics_on_overflow() {
        let x = Uint1::from(0x0001_u64);
        let _ = x << 64;
    }

    #[test]
    fn checked_operations() {
        let a = Uint4::from(10_u64);
        let b = Uint4::from(5_u64);

        assert_eq!(a.checked_add(&b), Some(Uint4::from(15_u64)));
        assert_eq!(a.checked_sub(&b), Some(Uint4::from(5_u64)));
        assert_eq!(a.checked_mul(&b), Some(Uint4::from(50_u64)));
        assert_eq!(a.checked_rem(&b), Some(Uint4::ZERO));

        // Test underflow
        assert!(b.checked_sub(&a).is_none());
    }

    #[allow(clippy::op_ref)]
    #[test]
    fn reference_operations() {
        let a = Uint4::from(10_u64);
        let b = Uint4::from(5_u64);

        // Test reference-based addition
        let c = a + &b;
        assert_eq!(c, Uint4::from(15_u64));

        // Test reference-based subtraction
        let d = a - &b;
        assert_eq!(d, Uint4::from(5_u64));

        // Test reference-based multiplication
        let e = a * &b;
        assert_eq!(e, Uint4::from(50_u64));

        // Test reference-based remainder
        let f = a % &b;
        assert_eq!(f, Uint4::ZERO);
    }

    #[test]
    fn conversions() {
        // Test From<crypto_bigint::Uint> for Uint
        let original = crypto_bigint::Uint::from(123_u64);
        let wrapped: Uint4 = original.into();
        assert_eq!(wrapped.0, original);

        // Test From<Uint> for crypto_bigint::Uint
        let wrapped = Uint4::from(456_u64);
        let unwrapped: crypto_bigint::Uint<{ 4 * WORD_FACTOR }> = wrapped.into();
        assert_eq!(unwrapped, crypto_bigint::Uint::from(456_u64));

        // Test conversion methods
        let value = crypto_bigint::Uint::from(789_u64);
        let wrapped = Uint4::new(value);
        assert_eq!(wrapped.inner(), &value);
        assert_eq!(wrapped.into_inner(), value);
    }

    #[test]
    fn pow_operation() {
        // Test basic exponentiation
        let base = Uint4::from(2_u64);

        // 2^0 = 1
        assert_eq!(base.pow(0), Uint4::one());

        // 2^1 = 2
        assert_eq!(base.pow(1), base);

        // 2^3 = 8
        assert_eq!(base.pow(3), Uint4::from(8_u64));

        // 2^10 = 1024
        assert_eq!(base.pow(10), Uint4::from(1024_u64));

        // Test with different base
        let base = Uint4::from(3_u64);

        // 3^4 = 81
        assert_eq!(base.pow(4), Uint4::from(81_u64));

        // Test with base 1
        let base = Uint4::from(1_u64);
        assert_eq!(base.pow(1000), Uint4::from(1_u64));

        // Test with base 0
        let base = Uint4::from(0_u64);
        assert_eq!(base.pow(0), Uint4::one()); // 0^0 = 1 by convention
        assert_eq!(base.pow(10), Uint4::zero()); // 0^n = 0 for n > 0
    }

    #[test]
    fn rem_assign_operations() {
        // Test RemAssign with owned value
        let mut a = Uint4::from(17_u64);
        let b = Uint4::from(5_u64);
        a %= b;
        assert_eq!(a, Uint4::from(2_u64));

        // Test RemAssign with reference
        let mut c = Uint4::from(19_u64);
        let d = Uint4::from(6_u64);
        c %= &d;
        assert_eq!(c, Uint4::from(1_u64));

        // Test with divisor 1
        let mut e = Uint4::from(42_u64);
        let one = Uint4::one();
        e %= &one;
        assert_eq!(e, Uint4::zero());
    }

    #[test]
    #[should_panic(expected = "division by zero")]
    fn rem_assign_panics_on_zero_divisor() {
        let mut a = Uint4::from(10_u64);
        let zero = Uint4::zero();
        a %= zero;
    }

    #[test]
    fn resize_method() {
        // Test resizing to same size
        let a = Uint4::from(0x12345678_u64);
        let resized_same = a.resize::<4>();
        assert_eq!(resized_same, a);

        // Test resizing to larger size
        let b = Uint2::from(0x9ABCDEF0_u64);
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
        let c = Uint4::from(0x1234567890ABCDEF_u64);
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
        let a = Uint1::from_words(words);
        assert_eq!(a.into_inner().to_words()[0], words[0]);

        // Test with multiple limbs
        let words = [
            0x1234567890ABCDEF,
            0xFEDCBA9876543210,
            0x0F0F0F0F0F0F0F0F,
            0xF0F0F0F0F0F0F0F0,
        ];
        let b = Uint4::from_words(words);
        let b_words = b.into_inner().to_words();
        for i in 0..4 {
            assert_eq!(b_words[i], words[i]);
        }
    }

    #[test]
    fn aggregate_operations() {
        let values: Vec<Uint4> = [1_u64, 2_u64, 3_u64].into_iter().map(Uint::from).collect();
        assert_eq!(values.iter().sum::<Uint4>(), Uint4::from(6_u64));
        assert_eq!(values.into_iter().sum::<Uint4>(), Uint4::from(6_u64));

        let values: Vec<Uint4> = [2_u64, 3_u64, 4_u64].into_iter().map(Uint::from).collect();
        assert_eq!(values.iter().product::<Uint4>(), Uint4::from(24_u64));
    }

    #[test]
    fn from_primitive() {
        // Test from_u8
        let a = Uint4::from_u8(42);
        assert_eq!(a, Uint4::from(42_u64));

        // Test from_u16
        let c = Uint4::from_u16(12345);
        assert_eq!(c, Uint4::from(12345_u64));

        // Test from_u32
        let e = Uint4::from_u32(1234567890);
        assert_eq!(e, Uint4::from(1234567890_u64));

        // Test from_u64
        let g = Uint4::from_u64(1234567890123456789);
        assert_eq!(g, Uint4::from(1234567890123456789_u64));

        // Test from_u128
        let i = Uint4::from_u128(1234567890123456789012345678901234567);
        assert_eq!(
            i.into_inner(),
            crypto_bigint::Uint::from(1234567890123456789012345678901234567_u128)
        );
    }

    #[test]
    fn from_primitive_edge_cases() {
        for value in [u32::MIN, u32::MAX] {
            let i = Uint1::from(value);
            let j = Uint2::from(value);
            assert_eq!(i.resize(), j);
        }

        for value in [u64::MIN, u64::MAX] {
            let i = Uint1::from(value);
            let j = Uint2::from(value);
            assert_eq!(i.resize(), j);
        }

        for value in [u128::MIN, u128::MAX] {
            let i = Uint2::from(value);
            let j = Uint::<3>::from(value);
            assert_eq!(i.resize(), j);
        }
    }

    #[should_panic]
    #[test]
    fn from_too_large_primitive() {
        // Test from_u128
        let _ = Uint1::from(u128::MAX);
    }

    #[test]
    fn edge_cases() {
        // Test operations with MAX values
        let max = Uint4::MAX;
        let one = Uint4::one();

        // MAX + 1 should overflow in checked_add
        assert!(max.checked_add(&one).is_none());

        // MAX - MAX = 0
        assert_eq!(max.checked_sub(&max).unwrap(), Uint4::zero());

        // Test operations with MIN values (0 for unsigned)
        let min = Uint4::ZERO;

        // MIN - 1 should overflow in checked_sub
        assert!(min.checked_sub(&one).is_none());

        // Test operations with large shifts
        let x = Uint4::from(1_u64);

        // Shift left by almost the bit limit
        let shifted = x << (Uint4::BITS - 1);
        assert_eq!(shifted, Uint4::from_words([0, 0, 0, 0x8000000000000000]));

        // Test with large powers that don't overflow
        let two = Uint4::from(2_u64);
        let large_power = two.pow(100); // 2^100 is large but fits in 256 bits

        // 2^100 should be divisible by 2^10 = 1024 with no remainder
        assert_eq!(large_power % Uint4::from(1024_u64), Uint4::zero());

        // 2^100 / 2 = 2^99
        let half_power = large_power >> 1;
        assert_eq!(half_power << 1, large_power);
    }

    #[test]
    fn assign_operations() {
        // Test AddAssign
        let mut a = Uint4::from(10_u64);
        a += Uint4::from(5_u64);
        assert_eq!(a, Uint4::from(15_u64));

        let mut b = Uint4::from(20_u64);
        b += &Uint4::from(3_u64);
        assert_eq!(b, Uint4::from(23_u64));

        // Test SubAssign
        let mut c = Uint4::from(10_u64);
        c -= Uint4::from(3_u64);
        assert_eq!(c, Uint4::from(7_u64));

        let mut d = Uint4::from(50_u64);
        d -= &Uint4::from(25_u64);
        assert_eq!(d, Uint4::from(25_u64));

        // Test MulAssign
        let mut e = Uint4::from(7_u64);
        e *= Uint4::from(6_u64);
        assert_eq!(e, Uint4::from(42_u64));

        let mut f = Uint4::from(3_u64);
        f *= &Uint4::from(4_u64);
        assert_eq!(f, Uint4::from(12_u64));

        let mut f = Uint1::from(2_u64);
        f <<= 2;
        assert_eq!(f, Uint1::from(8_u64)); // 2 << 2 = 8
        f <<= 61;
        assert_eq!(f, Uint1::ZERO);

        let mut f = Uint1::from(3_u64);
        f >>= 1;
        assert_eq!(f, Uint1::from(1_u64)); // 3 >> 1 = 1
        f >>= 1;
        assert_eq!(f, Uint1::ZERO);
    }

    #[test]
    fn formatting() {
        let a = Uint1::from(255_u64);
        let b = Uint1::MAX;

        // Test Debug
        assert_eq!(format!("{:?}", a), "Uint(0x00000000000000FF)");
        assert_eq!(format!("{:?}", b), "Uint(0xFFFFFFFFFFFFFFFF)");

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
        let default_val: Uint4 = Default::default();
        assert_eq!(default_val, Uint4::ZERO);
        assert!(default_val.is_zero());
    }

    #[test]
    fn constants() {
        // Test MAX
        assert!(Uint4::MAX > Uint4::ZERO);

        // Test BITS, BYTES, LIMBS
        assert_eq!(Uint4::BITS, 256);
        assert_eq!(Uint4::BYTES, 32);
        assert_eq!(Uint4::LIMBS, 4);

        assert_eq!(Uint2::BITS, 128);
        assert_eq!(Uint2::BYTES, 16);
        assert_eq!(Uint2::LIMBS, 2);
    }

    #[test]
    fn cmp_vartime() {
        let a = Uint4::from(10_u64);
        let b = Uint4::from(20_u64);
        let c = Uint4::from(10_u64);

        assert_eq!(a.cmp_vartime(&b), Ordering::Less);
        assert_eq!(b.cmp_vartime(&a), Ordering::Greater);
        assert_eq!(a.cmp_vartime(&c), Ordering::Equal);
    }

    #[test]
    fn cross_size_conversions() {
        // Test resize methods
        let a = Uint2::from(12345_u64);
        let b: Option<Uint4> = a.checked_resize();
        assert_eq!(b, Some(Uint::from(12345_u64)));
        let b: Uint4 = a.resize();
        assert_eq!(b, Uint::from(12345_u64));

        let a = Uint2::from_u128(u128::MAX);
        let b: Option<Uint1> = a.checked_resize();
        assert_eq!(b, None);
        let b: Uint1 = a.resize();
        assert_eq!(b, Uint::MAX);

        // Test From<&crypto_bigint::Uint<LIMBS>> for Uint<LIMBS2>
        let c = crypto_bigint::Uint::<{ 2 * WORD_FACTOR }>::from(67890_u64);
        let d: Uint4 = (&c).try_into().unwrap();
        assert_eq!(d, Uint4::from(67890_u64));

        // Test reference conversions from primitives
        let val = 42_u32;
        let e = Uint4::from(&val);
        assert_eq!(e, Uint4::from(42_u64));
    }

    #[test]
    fn constant_time_traits() {
        use crypto_bigint::subtle::{
            Choice, ConditionallySelectable, ConstantTimeEq, ConstantTimeGreater, ConstantTimeLess,
        };

        let a = Uint4::from(10_u64);
        let b = Uint4::from(20_u64);
        let c = Uint4::from(10_u64);

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
        let selected_true = Uint4::conditional_select(&a, &b, Choice::from(0));
        assert_eq!(selected_true, a);

        let selected_false = Uint4::conditional_select(&a, &b, Choice::from(1));
        assert_eq!(selected_false, b);
    }

    #[test]
    fn crypto_bigint_traits() {
        use crypto_bigint::{Bounded, Constants};

        // Test Bounded trait
        assert_eq!(<Uint4 as Bounded>::BITS, 256);
        assert_eq!(<Uint4 as Bounded>::BYTES, 32);

        // Test Constants trait
        assert_eq!(<Uint4 as Constants>::MAX, Uint4::MAX);
    }

    #[test]
    fn from_str() {
        // Test parsing from string
        let a: Uint4 = "123".parse().unwrap();
        assert_eq!(a, Uint4::from(123_u64));

        let c: Uint4 = "0".parse().unwrap();
        assert_eq!(c, Uint4::zero());

        let d: Uint4 = "1".parse().unwrap();
        assert_eq!(d, Uint4::one());
        assert_eq!(u64::MAX.to_string().parse::<Uint1>().unwrap(), Uint1::MAX);

        // Test invalid cases
        assert!("0x123".parse::<Uint4>().is_err());
        assert!("abc".parse::<Uint4>().is_err());
        assert!("12.34".parse::<Uint4>().is_err());
        assert!("".parse::<Uint4>().is_err());
        assert!("-456".parse::<Uint4>().is_err()); // Negative not allowed for unsigned

        // Number doesn't fit Uint1
        assert!(
            ((u64::MAX as u128) + 1)
                .to_string()
                .parse::<Uint1>()
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
        let random1: Uint4 = crypto_bigint::Random::random(&mut rng);
        let random2: Uint4 = crypto_bigint::Random::random(&mut rng);

        // Random values should be different
        assert_ne!(random1, random2);

        // Test Distribution trait
        let random3: Uint4 = rng.random();
        let random4: Uint4 = rng.random();

        assert_ne!(random3, random4);
    }
}
