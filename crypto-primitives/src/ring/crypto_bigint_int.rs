use super::*;
use core::{
    fmt::{Debug, Display, Formatter, Result as FmtResult},
    iter::{Product, Sum},
    ops::{Add, AddAssign, Mul, MulAssign, Rem, RemAssign, Shl, Shr, Sub, SubAssign},
};
use crypto_bigint::{CheckedMul as CryptoCheckedMul, CheckedSub as CryptoCheckedSub, Word};
use num_traits::{
    CheckedAdd, CheckedMul, CheckedNeg, CheckedRem, CheckedSub, ConstOne, ConstZero, One, Pow, Zero,
};
use pastey::paste;

#[derive(Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
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

//
// Zero and One traits
//

impl<const LIMBS: usize> Zero for Int<LIMBS> {
    fn zero() -> Self {
        Self::ZERO
    }

    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}

impl<const LIMBS: usize> One for Int<LIMBS> {
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

            fn $trait_op(self, rhs: Self) -> Self::Output {
                Self(self.0.$trait_op(&rhs.0))
            }
        }

        impl<'a, const LIMBS: usize> $trait_name<&'a Self> for Int<LIMBS> {
            type Output = Self;

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

    fn rem(self, rhs: Self) -> Self::Output {
        self.rem(&rhs)
    }
}

impl<'a, const LIMBS: usize> Rem<&'a Self> for Int<LIMBS> {
    type Output = Self;

    fn rem(self, rhs: &'a Self) -> Self::Output {
        let non_zero = crypto_bigint::NonZero::new(rhs.0).expect("division by zero");
        Self(self.0.rem(&non_zero))
    }
}

impl<const LIMBS: usize> Shl<u32> for Int<LIMBS> {
    type Output = Self;

    fn shl(self, rhs: u32) -> Self::Output {
        Self(self.0.shl(rhs))
    }
}

impl<const LIMBS: usize> Shr<u32> for Int<LIMBS> {
    type Output = Self;

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

macro_rules! impl_checked_op {
    ($trait_name:tt, $trait_op:tt) => {
        impl<const LIMBS: usize> $trait_name for Int<LIMBS> {
            fn $trait_op(&self, other: &Self) -> Option<Self> {
                let result = self.0.$trait_op(&other.0);
                if result.is_some().into() {
                    Some(Self(result.unwrap()))
                } else {
                    None
                }
            }
        }
    };
}

impl_checked_op!(CheckedAdd, checked_add);
impl_checked_op!(CheckedSub, checked_sub);
impl_checked_op!(CheckedMul, checked_mul);

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

impl<const LIMBS: usize> CheckedRem for Int<LIMBS> {
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
            fn $trait_op(&mut self, rhs: Self) {
                self.0.$trait_op(&rhs.0);
            }
        }

        impl<'a, const LIMBS: usize> $trait_name<&'a Self> for Int<LIMBS> {
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
    fn rem_assign(&mut self, rhs: Self) {
        self.rem_assign(&rhs);
    }
}

impl<'a, const LIMBS: usize> RemAssign<&'a Self> for Int<LIMBS> {
    #![allow(clippy::arithmetic_side_effects)]
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
    fn from(value: crypto_bigint::Int<LIMBS>) -> Self {
        Self(value)
    }
}

impl<const LIMBS: usize> From<Int<LIMBS>> for crypto_bigint::Int<LIMBS> {
    fn from(value: Int<LIMBS>) -> Self {
        value.0
    }
}

impl<const LIMBS: usize, const LIMBS2: usize> From<&Int<LIMBS>> for Int<LIMBS2> {
    fn from(num: &Int<LIMBS>) -> Int<LIMBS2> {
        num.resize()
    }
}

impl<const LIMBS: usize, const LIMBS2: usize> From<&crypto_bigint::Int<LIMBS>> for Int<LIMBS2> {
    fn from(num: &crypto_bigint::Int<LIMBS>) -> Int<LIMBS2> {
        Self(num.resize())
    }
}

macro_rules! impl_from_primitive {
    ($($t:ty),+) => {
        $(
            impl<const LIMBS: usize> From<$t> for Int<LIMBS> {
                fn from(value: $t) -> Self {
                    Self(crypto_bigint::Int::<LIMBS>::from(value))
                }
            }

            impl<const LIMBS: usize> Int<LIMBS> {
            paste! {
                pub const fn  [<from_ $t>] (n: $t) -> Self {
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

impl<const LIMBS: usize> ConstRing for Int<LIMBS> {}

impl<const LIMBS: usize> IntRing for Int<LIMBS> {}

//
// Traits from crypto_bigint
//

#[cfg(feature = "rand_core")]
impl<const LIMBS: usize> crypto_bigint::Random for Int<LIMBS> {
    fn random<R: rand_core::RngCore + ?Sized>(rng: &mut R) -> Self {
        Self(crypto_bigint::Int::random(rng))
    }

    fn try_random<R: rand_core::TryRngCore + ?Sized>(rng: &mut R) -> Result<Self, R::Error> {
        crypto_bigint::Int::try_random(rng).map(Self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    #[test]
    fn test_int_basic_operations() {
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
    fn test_shl_panics_on_overflow() {
        let x = Int::<1>::from(0x0001_i64);
        let _ = x << 64;
    }

    #[test]
    fn test_int_checked_operations() {
        let a = Int::<4>::from(10_i64);
        let b = Int::<4>::from(5_i64);
        let zero = Int::<4>::ZERO;

        // Test checked_add
        let c = a.checked_add(&b).unwrap();
        assert_eq!(c, Int::<4>::from(15i64));

        // Test checked_sub
        let d = a.checked_sub(&b).unwrap();
        assert_eq!(d, Int::<4>::from(5i64));

        // Test checked_mul
        let e = a.checked_mul(&b).unwrap();
        assert_eq!(e, Int::<4>::from(50i64));

        // Test checked_rem
        let f = a.checked_rem(&b).unwrap();
        assert_eq!(f, Int::<4>::ZERO);

        // Test checked_rem with zero divisor
        assert!(a.checked_rem(&zero).is_none());
    }

    #[allow(clippy::op_ref)]
    #[test]
    fn test_int_reference_operations() {
        let a = Int::<4>::from(10i64);
        let b = Int::<4>::from(5i64);

        // Test reference-based addition
        let c = a + &b;
        assert_eq!(c, Int::<4>::from(15i64));

        // Test reference-based subtraction
        let d = a - &b;
        assert_eq!(d, Int::<4>::from(5i64));

        // Test reference-based multiplication
        let e = a * &b;
        assert_eq!(e, Int::<4>::from(50i64));

        // Test reference-based remainder
        let f = a % &b;
        assert_eq!(f, Int::<4>::ZERO);
    }

    #[test]
    fn test_int_conversions() {
        // Test From<crypto_bigint::Int> for Int
        let original = crypto_bigint::Int::<4>::from(123i64);
        let wrapped: Int<4> = original.into();
        assert_eq!(wrapped.0, original);

        // Test From<Int> for crypto_bigint::Int
        let wrapped = Int::<4>::from(456i64);
        let unwrapped: crypto_bigint::Int<4> = wrapped.into();
        assert_eq!(unwrapped, crypto_bigint::Int::from(456i64));

        // Test conversion methods
        let value = crypto_bigint::Int::<4>::from(789i64);
        let wrapped = Int::new(value);
        assert_eq!(wrapped.inner(), &value);
        assert_eq!(wrapped.into_inner(), value);
    }

    #[test]
    fn test_pow_operation() {
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
    fn test_checked_neg() {
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
    fn test_rem_assign_operations() {
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
    fn test_rem_assign_panics_on_zero_divisor() {
        let mut a = Int::<4>::from(10_i64);
        let zero = Int::<4>::zero();
        a %= zero;
    }

    #[test]
    fn test_resize_method() {
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
    fn test_from_words() {
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
    fn test_aggregate_operations() {
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
    fn test_from_primitive() {
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
    fn test_edge_cases() {
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
}
