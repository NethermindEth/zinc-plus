use crate::{IntSemiring, Semiring};
use core::{
    fmt::{Debug, Display, Formatter, Result as FmtResult},
    hash::{Hash, Hasher},
    iter::{Product, Sum},
    ops::{Add, AddAssign, Deref, Mul, MulAssign, Sub, SubAssign},
    str::{FromStr, ParseBoolError},
};
use num_traits::{CheckedAdd, CheckedMul, CheckedSub, ConstOne, ConstZero, One, Pow, Zero};

#[cfg(feature = "rand")]
use rand::{distr::StandardUniform, prelude::*};

/// A boolean semiring where true represents 1 and false represents 0.
/// Arithmetic operations behave like modulo-2 arithmetic:
/// - In debug mode: overflow panics (e.g., true + true panics)
/// - In release mode: overflow wraps (e.g., true + true = false)
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct Boolean(bool);

impl Boolean {
    pub const FALSE: Self = Self(false);
    pub const TRUE: Self = Self(true);

    /// Creates a new Boolean from a bool value
    #[inline(always)]
    pub const fn new(value: bool) -> Self {
        Self(value)
    }

    /// Get the inner bool value
    #[inline(always)]
    pub const fn inner(&self) -> bool {
        self.0
    }

    /// Convert to the inner bool value
    #[inline(always)]
    pub const fn into_inner(self) -> bool {
        self.0
    }

    /// Convert to u8 (0 or 1)
    #[inline(always)]
    pub const fn to_u8(&self) -> u8 {
        self.0 as u8
    }

    /// Create from u8 (0 is false, non-zero is true)
    #[inline(always)]
    pub const fn from_u8(value: u8) -> Self {
        Self(value != 0)
    }
}

//
// Core traits
//

impl Debug for Boolean {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        Debug::fmt(&self.0, f)
    }
}

impl Display for Boolean {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        Display::fmt(&self.0, f)
    }
}

impl Default for Boolean {
    #[inline(always)]
    fn default() -> Self {
        Self::FALSE
    }
}

impl Deref for Boolean {
    type Target = bool;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Hash for Boolean {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state)
    }
}

impl FromStr for Boolean {
    type Err = ParseBoolError;

    /// In addition to "true" and "false", also supports "1" and "0".
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "1" => Ok(Self::TRUE),
            "0" => Ok(Self::FALSE),
            _ => bool::from_str(s).map(Self),
        }
    }
}

//
// Zero and One traits
//

impl Zero for Boolean {
    #[inline(always)]
    fn zero() -> Self {
        Self::FALSE
    }

    #[inline(always)]
    fn is_zero(&self) -> bool {
        !self.0
    }
}

impl One for Boolean {
    #[inline(always)]
    fn one() -> Self {
        Self::TRUE
    }
}

impl ConstZero for Boolean {
    const ZERO: Self = Self::FALSE;
}

impl ConstOne for Boolean {
    const ONE: Self = Self::TRUE;
}

//
// From implementations
//

impl From<bool> for Boolean {
    #[inline(always)]
    fn from(value: bool) -> Self {
        Self(value)
    }
}

impl From<Boolean> for bool {
    #[inline(always)]
    fn from(value: Boolean) -> Self {
        value.0
    }
}

//
// Basic arithmetic operations
//

impl Add for Boolean {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        self.add(&rhs)
    }
}

impl<'a> Add<&'a Boolean> for Boolean {
    type Output = Self;

    fn add(self, rhs: &'a Self) -> Self::Output {
        // In debug mode, panic on overflow (when both are true)
        debug_assert!(!(self.0 && rhs.0), "attempt to add with overflow");

        // Addition is XOR in modulo 2
        Self(self.0 ^ rhs.0)
    }
}

impl Sub for Boolean {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        self.sub(&rhs)
    }
}

impl<'a> Sub<&'a Boolean> for Boolean {
    type Output = Self;

    fn sub(self, rhs: &'a Self) -> Self::Output {
        // In debug mode, panic on underflow
        debug_assert!(self.0 || !rhs.0, "attempt to subtract with overflow");

        // Otherwise, subtraction is exactly like addition, XOR in modulo 2
        Self(self.0 ^ rhs.0)
    }
}

impl Mul for Boolean {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        self.mul(&rhs)
    }
}

impl<'a> Mul<&'a Boolean> for Boolean {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: &'a Self) -> Self::Output {
        // Boolean multiplication is AND
        Self(self.0 && rhs.0)
    }
}

//
// Arithmetic assign operations
//

macro_rules! impl_assign_op {
    ($trait_name:tt, $trait_op:tt, $op_method:ident) => {
        #[allow(clippy::arithmetic_side_effects)]
        impl $trait_name for Boolean {
            #[inline(always)]
            fn $trait_op(&mut self, rhs: Self) {
                self.$trait_op(&rhs);
            }
        }

        #[allow(clippy::arithmetic_side_effects)]
        impl $trait_name<&Boolean> for Boolean {
            #[inline(always)]
            fn $trait_op(&mut self, rhs: &Self) {
                *self = self.$op_method(rhs);
            }
        }
    };
}

impl_assign_op!(AddAssign, add_assign, add);
impl_assign_op!(SubAssign, sub_assign, sub);
impl_assign_op!(MulAssign, mul_assign, mul);

//
// Checked arithmetic operations
//

impl CheckedAdd for Boolean {
    fn checked_add(&self, rhs: &Self) -> Option<Self> {
        // Overflow when both are true
        if self.0 && rhs.0 {
            None
        } else {
            Some(Self(self.0 ^ rhs.0))
        }
    }
}

impl CheckedSub for Boolean {
    fn checked_sub(&self, rhs: &Self) -> Option<Self> {
        // Underflow when false - true
        if !self.0 && rhs.0 {
            None
        } else {
            Some(Self(self.0 ^ rhs.0))
        }
    }
}

impl CheckedMul for Boolean {
    #[inline(always)]
    fn checked_mul(&self, rhs: &Self) -> Option<Self> {
        // Multiplication never overflows
        Some(Self(self.0 && rhs.0))
    }
}

//
// Aggregate operations
//

#[allow(clippy::arithmetic_side_effects)]
impl Sum for Boolean {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::FALSE, |acc, x| acc + x)
    }
}

#[allow(clippy::arithmetic_side_effects)]
impl<'a> Sum<&'a Boolean> for Boolean {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::FALSE, |acc, x| acc + x)
    }
}

#[allow(clippy::arithmetic_side_effects)]
impl Product for Boolean {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::TRUE, |acc, x| acc * x)
    }
}

#[allow(clippy::arithmetic_side_effects)]
impl<'a> Product<&'a Boolean> for Boolean {
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::TRUE, |acc, x| acc * x)
    }
}

//
// Power operation
//

impl Pow<u32> for Boolean {
    type Output = Self;

    fn pow(self, rhs: u32) -> Self::Output {
        // 0^0 = 1 (by convention)
        // 0^n = 0 for n > 0
        // 1^n = 1 for all n
        if rhs == 0 || self.0 {
            Self::TRUE
        } else {
            Self::FALSE
        }
    }
}

//
// Semiring
//

impl Semiring for Boolean {}

impl IntSemiring for Boolean {
    fn is_odd(&self) -> bool {
        self.0
    }

    fn is_even(&self) -> bool {
        !self.0
    }
}

//
// RNG
//

#[cfg(feature = "rand")]
impl Distribution<Boolean> for StandardUniform {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Boolean {
        Boolean::new(rng.random())
    }
}

//
// Tests
//

#[allow(
    clippy::arithmetic_side_effects,
    clippy::cast_lossless,
    clippy::op_ref,
    clippy::bool_assert_comparison
)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ConstIntSemiring, ensure_type_implements_trait};
    use alloc::{vec, vec::Vec};

    #[test]
    fn ensure_blanket_traits() {
        ensure_type_implements_trait!(Boolean, ConstIntSemiring);
    }

    #[test]
    fn constants() {
        assert_eq!(Boolean::FALSE, Boolean(false));
        assert_eq!(Boolean::TRUE, Boolean(true));
        assert_eq!(Boolean::ZERO, Boolean(false));
        assert_eq!(Boolean::ONE, Boolean(true));
    }

    #[test]
    fn basic_operations_no_overflow() {
        let f = Boolean::FALSE;
        let t = Boolean::TRUE;

        // Addition without overflow
        assert_eq!(f + f, f);
        assert_eq!(f + t, t);
        assert_eq!(t + f, t);

        // Subtraction without underflow
        assert_eq!(f - f, f);
        assert_eq!(t - f, t);
        assert_eq!(t - t, f);

        // Multiplication (never overflows)
        assert_eq!(f * f, f);
        assert_eq!(f * t, f);
        assert_eq!(t * f, f);
        assert_eq!(t * t, t);
    }

    #[test]
    #[cfg_attr(
        debug_assertions,
        should_panic(expected = "attempt to add with overflow")
    )]
    fn add_overflow_behavior() {
        let t = Boolean::TRUE;
        let _result = t + t;

        // In release mode, this wraps to false
        #[cfg(not(debug_assertions))]
        assert_eq!(_result, Boolean::FALSE);
    }

    #[test]
    #[cfg_attr(
        debug_assertions,
        should_panic(expected = "attempt to subtract with overflow")
    )]
    fn sub_underflow_behavior() {
        let f = Boolean::FALSE;
        let t = Boolean::TRUE;
        let _result = f - t;

        // In release mode, this wraps to true (255 & 1 = 1)
        #[cfg(not(debug_assertions))]
        assert_eq!(_result, Boolean::TRUE);
    }

    #[test]
    fn checked_operations() {
        let f = Boolean::FALSE;
        let t = Boolean::TRUE;

        // Checked add
        assert_eq!(f.checked_add(&f), Some(f));
        assert_eq!(f.checked_add(&t), Some(t));
        assert_eq!(t.checked_add(&f), Some(t));
        assert_eq!(t.checked_add(&t), None); // Overflow

        // Checked sub
        assert_eq!(f.checked_sub(&f), Some(f));
        assert_eq!(t.checked_sub(&f), Some(t));
        assert_eq!(t.checked_sub(&t), Some(f));
        assert_eq!(f.checked_sub(&t), None); // Underflow

        // Checked mul (never fails)
        assert_eq!(f.checked_mul(&f), Some(f));
        assert_eq!(f.checked_mul(&t), Some(f));
        assert_eq!(t.checked_mul(&f), Some(f));
        assert_eq!(t.checked_mul(&t), Some(t));
    }

    #[test]
    fn reference_operations() {
        let f = Boolean::FALSE;
        let t = Boolean::TRUE;

        // Test reference-based addition
        assert_eq!(f + &f, f);
        assert_eq!(f + &t, t);

        // Test reference-based subtraction
        assert_eq!(t - &f, t);
        assert_eq!(t - &t, f);

        // Test reference-based multiplication
        assert_eq!(f * &t, f);
        assert_eq!(t * &t, t);
    }

    #[test]
    fn assign_operations() {
        let f = Boolean::FALSE;
        let t = Boolean::TRUE;

        // AddAssign
        let mut a = f;
        a += t;
        assert_eq!(a, t);

        // SubAssign
        let mut b = t;
        b -= t;
        assert_eq!(b, f);

        // MulAssign
        let mut c = t;
        c *= t;
        assert_eq!(c, t);

        // Reference assign operations
        let mut d = f;
        d += &t;
        assert_eq!(d, t);

        let mut e = t;
        e -= &f;
        assert_eq!(e, t);

        let mut g = t;
        g *= &t;
        assert_eq!(g, t);
    }

    #[test]
    fn conversions() {
        // From bool
        assert_eq!(Boolean::from(true), Boolean::TRUE);
        assert_eq!(Boolean::from(false), Boolean::FALSE);

        // To bool
        assert_eq!(bool::from(Boolean::TRUE), true);
        assert_eq!(bool::from(Boolean::FALSE), false);

        // Methods
        assert_eq!(Boolean::new(true).inner(), true);
        assert_eq!(Boolean::new(false).into_inner(), false);
        assert_eq!(Boolean::TRUE.to_u8(), 1);
        assert_eq!(Boolean::FALSE.to_u8(), 0);
        assert_eq!(Boolean::from_u8(0), Boolean::FALSE);
        assert_eq!(Boolean::from_u8(1), Boolean::TRUE);
        assert_eq!(Boolean::from_u8(2), Boolean::TRUE);
    }

    #[test]
    fn from_str() {
        assert_eq!("true".parse::<Boolean>(), Ok(Boolean::TRUE));
        assert_eq!("false".parse::<Boolean>(), Ok(Boolean::FALSE));
        assert_eq!("1".parse::<Boolean>(), Ok(Boolean::TRUE));
        assert_eq!("0".parse::<Boolean>(), Ok(Boolean::FALSE));
        assert!("invalid".parse::<Boolean>().is_err());
        assert!("2".parse::<Boolean>().is_err());
    }

    #[test]
    fn pow_operation() {
        let f = Boolean::FALSE;
        let t = Boolean::TRUE;

        // 0^0 = 1 by convention
        assert_eq!(f.pow(0), t);

        // 0^n = 0 for n > 0
        assert_eq!(f.pow(1), f);
        assert_eq!(f.pow(2), f);
        assert_eq!(f.pow(100), f);

        // 1^n = 1 for all n
        assert_eq!(t.pow(0), t);
        assert_eq!(t.pow(1), t);
        assert_eq!(t.pow(2), t);
        assert_eq!(t.pow(100), t);
    }

    #[test]
    fn int_semiring_traits() {
        let f = Boolean::FALSE;
        let t = Boolean::TRUE;

        // is_odd
        assert!(!f.is_odd());
        assert!(t.is_odd());

        // is_even
        assert!(f.is_even());
        assert!(!t.is_even());
    }

    #[test]
    fn zero_and_one_traits() {
        assert_eq!(Boolean::zero(), Boolean::FALSE);
        assert_eq!(Boolean::one(), Boolean::ONE);

        assert!(Boolean::FALSE.is_zero());
        assert!(!Boolean::TRUE.is_zero());
    }

    #[test]
    fn aggregate_operations() {
        // Sum
        let values: Vec<Boolean> = vec![Boolean::FALSE, Boolean::TRUE, Boolean::FALSE];
        let sum: Boolean = values.iter().sum();
        assert_eq!(sum, Boolean::TRUE);

        let sum2: Boolean = values.into_iter().sum();
        assert_eq!(sum2, Boolean::TRUE);

        // Product
        let values: Vec<Boolean> = vec![Boolean::TRUE, Boolean::TRUE, Boolean::TRUE];
        let product: Boolean = values.iter().product();
        assert_eq!(product, Boolean::TRUE);

        let values_with_false: Vec<Boolean> = vec![Boolean::TRUE, Boolean::FALSE, Boolean::TRUE];
        let product2: Boolean = values_with_false.into_iter().product();
        assert_eq!(product2, Boolean::FALSE);

        // Empty collections
        let empty: Vec<Boolean> = vec![];
        let empty_sum: Boolean = empty.iter().sum();
        assert_eq!(empty_sum, Boolean::ZERO);

        let empty: Vec<Boolean> = vec![];
        let empty_product: Boolean = empty.into_iter().product();
        assert_eq!(empty_product, Boolean::ONE);
    }

    #[test]
    fn ordering() {
        let f = Boolean::FALSE;
        let t = Boolean::TRUE;

        assert!(f < t);
        assert!(f <= t);
        assert!(f <= f);
        assert!(t > f);
        assert!(t >= f);
        assert!(t >= t);
        assert_eq!(f, f);
        assert_eq!(t, t);
        assert_ne!(f, t);
    }

    #[test]
    fn debug_and_display() {
        use alloc::format;

        assert_eq!(format!("{:?}", Boolean::TRUE), "true");
        assert_eq!(format!("{:?}", Boolean::FALSE), "false");
        assert_eq!(format!("{}", Boolean::TRUE), "true");
        assert_eq!(format!("{}", Boolean::FALSE), "false");
    }

    #[test]
    fn hash() {
        use core::hash::{Hash, Hasher};

        // Simple hash implementation for testing
        struct SimpleHasher(u64);

        impl Hasher for SimpleHasher {
            fn finish(&self) -> u64 {
                self.0
            }

            fn write(&mut self, bytes: &[u8]) {
                for &byte in bytes {
                    self.0 = self.0.wrapping_mul(31).wrapping_add(byte as u64);
                }
            }
        }

        let mut hasher1 = SimpleHasher(0);
        Boolean::TRUE.hash(&mut hasher1);
        let hash1 = hasher1.finish();

        let mut hasher2 = SimpleHasher(0);
        Boolean::TRUE.hash(&mut hasher2);
        let hash2 = hasher2.finish();

        assert_eq!(hash1, hash2);

        let mut hasher3 = SimpleHasher(0);
        Boolean::FALSE.hash(&mut hasher3);
        let hash3 = hasher3.finish();

        assert_ne!(hash1, hash3);
    }

    #[test]
    fn default() {
        assert_eq!(Boolean::default(), Boolean::FALSE);
    }
}
