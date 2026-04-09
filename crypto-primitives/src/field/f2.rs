use crate::{ConstPrimeField, ConstSemiring, Field, Ring, Semiring, boolean::Boolean};
use core::{
    fmt::{Debug, Display, Formatter, Result as FmtResult},
    hash::Hash,
    iter::{Product, Sum},
    ops::{Add, AddAssign, Deref, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
    str::FromStr,
};
use crypto_primitives_proc_macros::InfallibleCheckedOp;
use num_traits::{
    CheckedAdd, CheckedDiv, CheckedMul, CheckedNeg, CheckedSub, ConstOne, ConstZero, Inv, One, Pow,
    Zero,
};

#[cfg(feature = "rand")]
use rand::{distr::StandardUniform, prelude::*};

/// The field with two elements, GF(2). Elements are {0, 1} represented as
/// {false, true}. Arithmetic is modulo 2: addition and subtraction are XOR,
/// multiplication is AND.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, InfallibleCheckedOp)]
#[infallible_checked_unary_op((CheckedNeg, neg))]
#[infallible_checked_binary_op((CheckedAdd, add), (CheckedSub, sub), (CheckedMul, mul))]
#[repr(transparent)]
pub struct F2(bool);

impl F2 {
    /// Creates a new F2 element from a bool value.
    #[inline(always)]
    pub const fn new(value: bool) -> Self {
        Self(value)
    }

    /// Get the inner bool value.
    #[inline(always)]
    pub const fn inner(&self) -> &bool {
        &self.0
    }

    /// Convert to the inner bool value.
    #[inline(always)]
    pub const fn into_inner(self) -> bool {
        self.0
    }
}

//
// Core traits
//

impl Display for F2 {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "{} (mod 2)", u8::from(self.0))
    }
}

impl Default for F2 {
    #[inline(always)]
    fn default() -> Self {
        Self::ZERO
    }
}

impl Deref for F2 {
    type Target = bool;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl FromStr for F2 {
    type Err = <Boolean as FromStr>::Err;

    /// See [`Boolean::from_str`] for accepted formats.
    #[inline(always)]
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Boolean::from_str(s).map(|b| Self(*b))
    }
}

//
// Zero and One traits
//

impl Zero for F2 {
    #[inline(always)]
    fn zero() -> Self {
        Self::ZERO
    }

    #[inline(always)]
    fn is_zero(&self) -> bool {
        !self.0
    }
}

impl One for F2 {
    #[inline(always)]
    fn one() -> Self {
        Self::ONE
    }
}

impl ConstZero for F2 {
    const ZERO: Self = Self(false);
}

impl ConstOne for F2 {
    const ONE: Self = Self(true);
}

//
// Basic arithmetic operations
//

impl Neg for F2 {
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self::Output {
        // In GF(2), -x = x
        self
    }
}

macro_rules! impl_basic_op_forward_to_assign {
    ($trait:ident, $method:ident, $assign_method:ident) => {
        impl $trait for F2 {
            type Output = F2;

            #[inline(always)]
            fn $method(self, rhs: F2) -> Self::Output {
                self.$method(&rhs)
            }
        }

        impl $trait<&Self> for F2 {
            type Output = F2;

            #[inline(always)]
            fn $method(mut self, rhs: &F2) -> Self::Output {
                self.$assign_method(rhs);
                self
            }
        }

        impl $trait<F2> for &F2 {
            type Output = F2;

            #[inline(always)]
            fn $method(self, rhs: F2) -> Self::Output {
                (*self).$method(&rhs)
            }
        }

        impl $trait for &F2 {
            type Output = F2;

            #[inline(always)]
            fn $method(self, rhs: &F2) -> Self::Output {
                (*self).$method(rhs)
            }
        }
    };
}

impl_basic_op_forward_to_assign!(Add, add, add_assign);
impl_basic_op_forward_to_assign!(Sub, sub, sub_assign);
impl_basic_op_forward_to_assign!(Mul, mul, mul_assign);
impl_basic_op_forward_to_assign!(Div, div, div_assign);

impl Pow<u32> for F2 {
    type Output = Self;

    fn pow(self, rhs: u32) -> Self::Output {
        // 0^0 = 1 (by convention)
        // 0^n = 0 for n > 0
        // 1^n = 1 for all n
        if rhs == 0 || self.0 {
            Self::ONE
        } else {
            Self::ZERO
        }
    }
}

impl Inv for F2 {
    type Output = Option<Self>;

    #[inline(always)]
    fn inv(self) -> Self::Output {
        // 0 has no inverse, 1 is its own inverse
        if self.0 { Some(self) } else { None }
    }
}

//
// Checked arithmetic operations
//

impl CheckedDiv for F2 {
    #[allow(clippy::arithmetic_side_effects)] // False alert
    fn checked_div(&self, rhs: &Self) -> Option<Self> {
        Some(*self * rhs.inv()?)
    }
}

//
// Arithmetic assign operations
//

macro_rules! impl_op_assign_boilerplate {
    ($trait:ident, $method:ident) => {
        impl $trait for F2 {
            #[inline(always)]
            fn $method(&mut self, rhs: F2) {
                self.$method(&rhs);
            }
        }
    };
}

impl_op_assign_boilerplate!(AddAssign, add_assign);
impl_op_assign_boilerplate!(SubAssign, sub_assign);
impl_op_assign_boilerplate!(MulAssign, mul_assign);
impl_op_assign_boilerplate!(DivAssign, div_assign);

impl AddAssign<&Self> for F2 {
    #[allow(clippy::suspicious_op_assign_impl)] // False alert
    #[inline(always)]
    fn add_assign(&mut self, rhs: &Self) {
        self.0 ^= rhs.0;
    }
}

impl SubAssign<&Self> for F2 {
    #[allow(clippy::suspicious_op_assign_impl)] // False alert
    #[inline(always)]
    fn sub_assign(&mut self, rhs: &Self) {
        // In GF(2), subtraction = addition = XOR
        self.0 ^= rhs.0;
    }
}

impl MulAssign<&Self> for F2 {
    #[allow(clippy::suspicious_op_assign_impl)] // False alert
    #[inline(always)]
    fn mul_assign(&mut self, rhs: &Self) {
        self.0 &= rhs.0;
    }
}

impl DivAssign<&Self> for F2 {
    #[allow(clippy::arithmetic_side_effects, clippy::suspicious_op_assign_impl)] // False alert
    #[inline(always)]
    fn div_assign(&mut self, rhs: &Self) {
        *self *= rhs.inv().expect("Division by zero");
    }
}

//
// Aggregate operations
//

impl Sum for F2 {
    #[allow(clippy::arithmetic_side_effects)] // False alert
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |acc, x| acc + x)
    }
}

impl<'a> Sum<&'a Self> for F2 {
    #[allow(clippy::arithmetic_side_effects)] // False alert
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |acc, x| acc + x)
    }
}

impl Product for F2 {
    #[allow(clippy::arithmetic_side_effects)] // False alert
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ONE, |acc, x| acc * x)
    }
}

impl<'a> Product<&'a Self> for F2 {
    #[allow(clippy::arithmetic_side_effects)] // False alert
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::ONE, |acc, x| acc * x)
    }
}

//
// Conversions
//

impl From<&Self> for F2 {
    fn from(value: &Self) -> Self {
        *value
    }
}

impl From<bool> for F2 {
    #[inline(always)]
    fn from(value: bool) -> Self {
        Self(value)
    }
}

impl From<Boolean> for F2 {
    #[inline(always)]
    fn from(value: Boolean) -> Self {
        Self(*value)
    }
}

impl From<&Boolean> for F2 {
    #[inline(always)]
    fn from(value: &Boolean) -> Self {
        Self(**value)
    }
}

macro_rules! impl_from_unsigned {
    ($($t:ty),* $(,)?) => {
        $(
            impl From<$t> for F2 {
                #[inline(always)]
                fn from(value: $t) -> Self {
                    Self(value % 2 != 0)
                }
            }

            impl From<&$t> for F2 {
                #[inline(always)]
                fn from(value: &$t) -> Self {
                    Self::from(*value)
                }
            }
        )*
    };
}

macro_rules! impl_from_signed {
    ($($t:ty),* $(,)?) => {
        $(
            impl From<$t> for F2 {
                #[inline(always)]
                fn from(value: $t) -> Self {
                    // In F2, -1 = 1, so only parity matters
                    Self(value.unsigned_abs() % 2 != 0)
                }
            }

            impl From<&$t> for F2 {
                #[inline(always)]
                fn from(value: &$t) -> Self {
                    Self::from(*value)
                }
            }
        )*
    };
}

impl_from_unsigned!(u8, u16, u32, u64, u128);
impl_from_signed!(i8, i16, i32, i64, i128);

//
// Semiring, Ring and Field
//

impl Semiring for F2 {}

impl ConstSemiring for F2 {
    const MAX: Self = Self::ONE;
    const MIN: Self = Self::ZERO;
}

impl Ring for F2 {}

impl Field for F2 {
    type Inner = bool;
    type Modulus = u8;

    #[inline(always)]
    fn inner(&self) -> &Self::Inner {
        &self.0
    }

    #[inline(always)]
    fn inner_mut(&mut self) -> &mut Self::Inner {
        &mut self.0
    }

    #[inline(always)]
    fn into_inner(self) -> Self::Inner {
        self.0
    }
}

impl ConstPrimeField for F2 {
    const MODULUS: Self::Modulus = 2;
    const MODULUS_MINUS_ONE_DIV_TWO: Self::Inner = false;

    #[inline(always)]
    fn new(inner: Self::Inner) -> Self {
        Self(inner)
    }

    #[inline(always)]
    fn new_unchecked(inner: Self::Inner) -> Self {
        Self(inner)
    }
}

//
// RNG
//

#[cfg(feature = "rand")]
impl Distribution<F2> for StandardUniform {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> F2 {
        F2::new(rng.random())
    }
}

//
// Zeroize
//

#[cfg(feature = "zeroize")]
impl zeroize::DefaultIsZeroes for F2 {}

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
    use crate::{
        ConstPrimeField, FromPrimitiveWithConfig, PrimeField, ensure_type_implements_trait,
    };
    use alloc::format;
    use num_traits::{One, Zero};

    const V0: F2 = F2::ZERO;
    const V1: F2 = F2::ONE;

    #[test]
    fn ensure_blanket_traits() {
        // NB: this ensures `PrimeField` implementation too!
        ensure_type_implements_trait!(F2, FromPrimitiveWithConfig);
    }

    #[test]
    fn constants() {
        assert_eq!(V0, F2(false));
        assert_eq!(V1, F2(true));
        assert_eq!(V0, F2::zero());
        assert_eq!(V1, F2::one());
    }

    #[test]
    fn zero_one_basics() {
        assert!(V0.is_zero());
        assert!(PrimeField::is_zero(&V0));
        assert!(!V1.is_zero());
        assert!(!PrimeField::is_zero(&V1));
        assert_ne!(V0, V1);
    }

    #[test]
    fn basic_ops() {
        // Addition (XOR)
        assert_eq!(V0 + V0, V0);
        assert_eq!(V0 + V1, V1);
        assert_eq!(V1 + V0, V1);
        assert_eq!(V1 + V1, V0);

        // Subtraction (same as addition)
        assert_eq!(V0 - V0, V0);
        assert_eq!(V1 - V0, V1);
        assert_eq!(V0 - V1, V1);
        assert_eq!(V1 - V1, V0);

        // Multiplication (AND)
        assert_eq!(V0 * V0, V0);
        assert_eq!(V0 * V1, V0);
        assert_eq!(V1 * V0, V0);
        assert_eq!(V1 * V1, V1);

        // Division
        assert_eq!(V0 / V1, V0);
        assert_eq!(V1 / V1, V1);
    }

    #[test]
    fn negation_properties() {
        // In char 2, -x = x
        assert_eq!(-V0, V0);
        assert_eq!(-V1, V1);
        assert_eq!(V1 + (-V1), V0);
        assert_eq!(-(-V1), V1);
    }

    #[test]
    fn inversion_properties() {
        let inv = V1.inv().expect("1 should be invertible");
        assert_eq!(V1 * inv, F2::one());
        assert!(V0.inv().is_none());
    }

    #[test]
    #[should_panic]
    fn div_by_zero_panics() {
        let _ = V1 / V0;
    }

    #[test]
    fn assign_ops_with_refs_and_values() {
        let mut x = V1;
        x += V1;
        assert_eq!(x, V0);

        let mut x = V1;
        x += &V1;
        assert_eq!(x, V0);

        let mut x = V1;
        x -= V1;
        assert_eq!(x, V0);

        let mut x = V1;
        x -= &V1;
        assert_eq!(x, V0);

        let mut x = V1;
        x *= V1;
        assert_eq!(x, V1);

        let mut x = V1;
        x *= &V1;
        assert_eq!(x, V1);

        let mut x = V1;
        x /= V1;
        assert_eq!(x, V1);

        let mut x = V1;
        x /= &V1;
        assert_eq!(x, V1);
    }

    #[test]
    fn ref_and_value_combinations_add_sub_mul() {
        let a = V1;
        let b = V0;

        // Add: all ref/value combinations
        assert_eq!(a + b, a);
        assert_eq!(a + &b, a);
        assert_eq!(&a + b, a);
        assert_eq!(&a + &b, a);

        // Sub
        assert_eq!(a - b, a);
        assert_eq!(a - &b, a);
        assert_eq!(&a - b, a);
        assert_eq!(&a - &b, a);

        // Mul
        assert_eq!(a * b, V0);
        assert_eq!(a * &b, V0);
        assert_eq!(&a * b, V0);
        assert_eq!(&a * &b, V0);
    }

    #[test]
    fn checked_operations() {
        // Checked neg
        assert_eq!(V0.checked_neg(), Some(V0));
        assert_eq!(V1.checked_neg(), Some(V1));

        // Checked add
        assert_eq!(V0.checked_add(&V0), Some(V0));
        assert_eq!(V0.checked_add(&V1), Some(V1));
        assert_eq!(V1.checked_add(&V0), Some(V1));
        assert_eq!(V1.checked_add(&V1), Some(V0));

        // Checked sub
        assert_eq!(V0.checked_sub(&V0), Some(V0));
        assert_eq!(V1.checked_sub(&V0), Some(V1));
        assert_eq!(V0.checked_sub(&V1), Some(V1));
        assert_eq!(V1.checked_sub(&V1), Some(V0));

        // Checked mul
        assert_eq!(V0.checked_mul(&V0), Some(V0));
        assert_eq!(V0.checked_mul(&V1), Some(V0));
        assert_eq!(V1.checked_mul(&V0), Some(V0));
        assert_eq!(V1.checked_mul(&V1), Some(V1));

        // Checked div (fails for division by zero)
        assert_eq!(V0.checked_div(&V1), Some(V0));
        assert_eq!(V1.checked_div(&V1), Some(V1));
        assert_eq!(V0.checked_div(&V0), None);
        assert_eq!(V1.checked_div(&V0), None);
    }

    #[test]
    fn pow_operation() {
        // 0^0 = 1 by convention
        assert_eq!(V0.pow(0), V1);

        // 0^n = 0 for n > 0
        assert_eq!(V0.pow(1), V0);
        assert_eq!(V0.pow(2), V0);
        assert_eq!(V0.pow(100), V0);

        // 1^n = 1 for all n
        assert_eq!(V1.pow(0), V1);
        assert_eq!(V1.pow(1), V1);
        assert_eq!(V1.pow(2), V1);
        assert_eq!(V1.pow(100), V1);
    }

    #[test]
    fn sum_and_product_trait_basic() {
        let v = [V1, V0, V1, V1];

        // Sum: 1 + 0 + 1 + 1 = 1 (XOR fold)
        let sum1: F2 = v.iter().cloned().sum();
        let sum2: F2 = v.iter().sum();
        let sum3: F2 = v.into_iter().sum();
        assert_eq!(sum1, V1);
        assert_eq!(sum2, V1);
        assert_eq!(sum3, V1);

        // Product: 1 * 0 * 1 * 1 = 0 (AND fold)
        let prod1: F2 = v.iter().product();
        let prod2: F2 = v.into_iter().product();
        assert_eq!(prod1, V0);
        assert_eq!(prod2, V0);

        // Empty iterators: neutral elements
        let empty: [F2; 0] = [];
        let sum_empty: F2 = empty.iter().cloned().sum();
        assert_eq!(sum_empty, V0);
        let prod_empty: F2 = empty.iter().product();
        assert_eq!(prod_empty, V1);
    }

    #[test]
    fn conversions() {
        // Self-reference conversion
        assert_eq!(F2::from(&V1), V1);

        // bool conversion
        assert_eq!(F2::from(true), V1);
        assert_eq!(F2::from(false), V0);

        let t: F2 = true.into();
        let f: F2 = false.into();
        assert_eq!(t, V1);
        assert_eq!(f, V0);

        // Boolean conversion
        assert_eq!(F2::from(Boolean::from(true)), V1);
        assert_eq!(F2::from(Boolean::from(false)), V0);
        assert_eq!(F2::from(&Boolean::from(true)), V1);

        // Integer conversions
        assert_eq!(F2::from(0_u64), V0);
        assert_eq!(F2::from(1_u32), V1);
        assert_eq!(F2::from(2_u64), V0);
        assert_eq!(F2::from(3_u64), V1);
        assert_eq!(F2::from(255_u8), V1);

        assert_eq!(F2::from(0_i64), V0);
        assert_eq!(F2::from(1_i32), V1);
        assert_eq!(F2::from(-1_i32), V1);
        assert_eq!(F2::from(-2_i64), V0);
        assert_eq!(F2::from(-3_i64), V1);

        // Integer reference conversions
        assert_eq!(F2::from(&42_u32), V0);
        assert_eq!(F2::from(&-100_i64), V0);
    }

    #[test]
    fn const_prime_field() {
        assert_eq!(<F2 as ConstPrimeField>::MODULUS, 2_u8);
        assert_eq!(<F2 as ConstPrimeField>::MODULUS_MINUS_ONE_DIV_TWO, false);

        assert_eq!(F2::new(true), V1);
        assert_eq!(F2::new(false), V0);
        assert_eq!(F2::new_unchecked(true), V1);
        assert_eq!(F2::new_unchecked(false), V0);
    }

    #[test]
    fn prime_field_methods() {
        assert_eq!(V1.modulus(), 2_u8);
        assert_eq!(V1.modulus_minus_one_div_two(), false);
        assert_eq!(F2::make_cfg(&2_u8), Ok(()));
        assert!(F2::make_cfg(&0_u8).is_err());
        assert!(F2::make_cfg(&3_u8).is_err());
    }

    #[test]
    fn formatting_traits() {
        assert_eq!(format!("{:?}", V1), "F2(true)");
        assert_eq!(format!("{:?}", V0), "F2(false)");
        assert_eq!(format!("{}", V1), "1 (mod 2)");
        assert_eq!(format!("{}", V0), "0 (mod 2)");
    }

    #[test]
    fn default_trait() {
        assert_eq!(F2::default(), V0);
    }

    #[test]
    fn ord_trait() {
        assert!(V0 < V1);
        assert!(V0 <= V1);
        assert!(V0 <= V0);
        assert!(V1 > V0);
        assert!(V1 >= V0);
        assert!(V1 >= V1);
        assert_eq!(V0, V0);
        assert_eq!(V1, V1);
        assert_ne!(V0, V1);
    }

    #[test]
    fn from_str() {
        assert_eq!("false".parse::<F2>(), Ok(V0));
        assert_eq!("true".parse::<F2>(), Ok(V1));

        assert_eq!("0".parse::<F2>(), Ok(V0));
        assert_eq!("0000".parse::<F2>(), Ok(V0));
        assert_eq!("0x0".parse::<F2>(), Ok(V0));
        assert_eq!("0x0000".parse::<F2>(), Ok(V0));

        assert_eq!("1".parse::<F2>(), Ok(V1));
        assert_eq!("0001".parse::<F2>(), Ok(V1));
        assert_eq!("0x1".parse::<F2>(), Ok(V1));
        assert_eq!("0x0001".parse::<F2>(), Ok(V1));

        assert!("invalid".parse::<F2>().is_err());
        assert!("2".parse::<F2>().is_err());
        assert!("-1".parse::<F2>().is_err());
        assert!("".parse::<F2>().is_err());
    }

    #[test]
    fn field_inner() {
        assert_eq!(*V1.inner(), true);
        assert_eq!(*V0.inner(), false);
        assert_eq!(V1.into_inner(), true);
        assert_eq!(V0.into_inner(), false);

        // Field::inner
        assert_eq!(*Field::inner(&V1), true);
        assert_eq!(*Field::inner(&V0), false);

        // Field::inner_mut
        let mut x = V0;
        *Field::inner_mut(&mut x) = true;
        assert_eq!(x, V1);
    }
}
