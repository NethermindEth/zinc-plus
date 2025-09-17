use super::*;
use core::fmt::{Debug, Display, Formatter, Result as FmtResult};
use core::ops::{Add, AddAssign, Mul, MulAssign, Rem, RemAssign, Shl, Shr, Sub, SubAssign};
use core::iter::{Product, Sum};
use num_traits::{CheckedAdd, CheckedMul, CheckedNeg, CheckedRem, CheckedShl, CheckedShr, CheckedSub, ConstOne, ConstZero, One, Pow, Zero};
use crypto_bigint::{CheckedSub as CryptoCheckedSub, CheckedMul as CryptoCheckedMul, Word};
use paste::paste;

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

        let mut base = self;
        let mut result = Self::one();
        let mut exp = rhs;

        while exp > 0 {
            if exp & 1 == 1 {
                result = result * base;
            }
            exp >>= 1;
            if exp > 0 {
                base = base.clone() * base;
            }
        }

        result
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

impl<const LIMBS: usize> CheckedShl for Int<LIMBS> {
    fn checked_shl(&self, rhs: u32) -> Option<Self> {
        // crypto_bigint::Int implements Shl<u32>
        // Always succeeds for u32 shift amounts
        Some(Self(self.0.shl(rhs)))
    }
}

impl<const LIMBS: usize> CheckedShr for Int<LIMBS> {
    fn checked_shr(&self, rhs: u32) -> Option<Self> {
        // crypto_bigint::Int implements Shr<u32>
        // Always succeeds for u32 shift amounts
        Some(Self(self.0.shr(rhs)))
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
        iter.fold(Self::zero(), |acc, x| acc + x)
    }
}

impl<'a, const LIMBS: usize> Sum<&'a Self> for Int<LIMBS> {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, x| acc + x)
    }
}

impl<const LIMBS: usize> Product for Int<LIMBS> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |acc, x| acc * x)
    }
}

impl<'a, const LIMBS: usize> Product<&'a Self> for Int<LIMBS> {
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |acc, x| acc * x)
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

    #[test]
    fn test_int_basic_operations() {
        // Test with 4 limbs (256-bit integers)
        let a = Int::<4>(crypto_bigint::Int::from(10i64));
        let b = Int::<4>(crypto_bigint::Int::from(5i64));
        
        // Test addition
        let c = a + b;
        assert_eq!(c, Int::<4>(crypto_bigint::Int::from(15i64)));
        
        // Test subtraction
        let d = a - b;
        assert_eq!(d, Int::<4>(crypto_bigint::Int::from(5i64)));
        
        // Test multiplication
        let e = a * b;
        assert_eq!(e, Int::<4>(crypto_bigint::Int::from(50i64)));
        
        // Test remainder
        let f = a % b;
        assert_eq!(f, Int::<4>(crypto_bigint::Int::from(0i64)));
    }
    
    #[test]
    fn test_int_checked_operations() {
        let a = Int::<4>(crypto_bigint::Int::from(10i64));
        let b = Int::<4>(crypto_bigint::Int::from(5i64));
        let zero = Int::<4>(crypto_bigint::Int::ZERO);
        
        // Test checked_add
        let c = a.checked_add(&b).unwrap();
        assert_eq!(c, Int::<4>(crypto_bigint::Int::from(15i64)));
        
        // Test checked_sub
        let d = a.checked_sub(&b).unwrap();
        assert_eq!(d, Int::<4>(crypto_bigint::Int::from(5i64)));
        
        // Test checked_mul
        let e = a.checked_mul(&b).unwrap();
        assert_eq!(e, Int::<4>(crypto_bigint::Int::from(50i64)));
        
        // Test checked_rem
        let f = a.checked_rem(&b).unwrap();
        assert_eq!(f, Int::<4>(crypto_bigint::Int::ZERO));
        
        // Test checked_rem with zero divisor
        assert!(a.checked_rem(&zero).is_none());
    }
    
    #[test]
    fn test_int_reference_operations() {
        let a = Int::<4>(crypto_bigint::Int::from(10i64));
        let b = Int::<4>(crypto_bigint::Int::from(5i64));
        
        // Test reference-based addition
        let c = a.clone() + &b;
        assert_eq!(c, Int::<4>(crypto_bigint::Int::from(15i64)));
        
        // Test reference-based subtraction
        let d = a.clone() - &b;
        assert_eq!(d, Int::<4>(crypto_bigint::Int::from(5i64)));
        
        // Test reference-based multiplication
        let e = a.clone() * &b;
        assert_eq!(e, Int::<4>(crypto_bigint::Int::from(50i64)));
        
        // Test reference-based remainder
        let f = a.clone() % &b;
        assert_eq!(f, Int::<4>(crypto_bigint::Int::ZERO));
    }
    
    #[test]
    fn test_int_conversions() {
        // Test From<crypto_bigint::Int> for Int
        let original = crypto_bigint::Int::<4>::from(123i64);
        let wrapped: Int<4> = original.clone().into();
        assert_eq!(wrapped.0, original);
        
        // Test From<Int> for crypto_bigint::Int
        let wrapped = Int::<4>(crypto_bigint::Int::from(456i64));
        let unwrapped: crypto_bigint::Int<4> = wrapped.into();
        assert_eq!(unwrapped, crypto_bigint::Int::from(456i64));
        
        // Test conversion methods
        let value = crypto_bigint::Int::<4>::from(789i64);
        let wrapped = Int::new(value.clone());
        assert_eq!(wrapped.inner(), &value);
        assert_eq!(wrapped.into_inner(), value);
    }
}
