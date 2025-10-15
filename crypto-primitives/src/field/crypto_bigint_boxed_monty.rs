use super::*;
use crate::{IntRing, IntSemiring, Semiring, boolean::Boolean, crypto_bigint_int::Int};
use core::{
    cmp::Ordering,
    fmt::{Debug, Display, Formatter, Result as FmtResult},
    hash::{Hash, Hasher},
    iter::{Product, Sum},
    ops::{Add, AddAssign, DivAssign, Mul, MulAssign, Rem, RemAssign, Sub, SubAssign},
};
use crypto_bigint::{
    BoxedUint, Integer, NonZero, Odd, Resize,
    modular::{BoxedMontyForm, BoxedMontyParams},
};
use crypto_primitives_proc_macros::InfallibleCheckedOp;
use num_traits::{CheckedAdd, CheckedDiv, CheckedMul, CheckedNeg, CheckedRem, CheckedSub, Pow};

#[derive(Clone, PartialEq, Eq, InfallibleCheckedOp)]
#[infallible_checked_unary_op((CheckedNeg, neg))]
#[infallible_checked_binary_op((CheckedAdd, add), (CheckedSub, sub), (CheckedMul, mul))]
#[repr(transparent)]
pub struct BoxedMontyField(BoxedMontyForm);

impl BoxedMontyField {
    /// Creates a new `BoxedMontyField` from a `BoxedMontyForm`.
    pub const fn new(form: BoxedMontyForm) -> Self {
        Self(form)
    }
}

//
// Core traits
//

impl Debug for BoxedMontyField {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        Debug::fmt(&self.0, f)
    }
}

impl Display for BoxedMontyField {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(
            f,
            "{} (mod {})",
            self.0.retrieve(),
            self.0.params().modulus()
        )
    }
}

impl PartialOrd for BoxedMontyField {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.modulus() != other.modulus() {
            return None;
        }
        Some(Ord::cmp(self.0.as_montgomery(), other.0.as_montgomery()))
    }
}

impl Hash for BoxedMontyField {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.as_montgomery().hash(state)
    }
}

//
// Basic arithmetic operations
//

impl Neg for BoxedMontyField {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self(self.0.neg())
    }
}

macro_rules! impl_basic_op {
    ($trait:ident, $method:ident) => {
        impl $trait for BoxedMontyField {
            type Output = Self;

            #[inline(always)]
            fn $method(self, rhs: Self) -> Self::Output {
                (&self).$method(&rhs)
            }
        }

        impl $trait<&Self> for BoxedMontyField {
            type Output = Self;

            #[inline(always)]
            fn $method(self, rhs: &Self) -> Self::Output {
                (&self).$method(rhs)
            }
        }

        impl $trait for &BoxedMontyField {
            type Output = BoxedMontyField;

            #[inline(always)]
            fn $method(self, rhs: Self) -> Self::Output {
                BoxedMontyField(BoxedMontyForm::$method(&self.0, &rhs.0))
            }
        }

        impl $trait<BoxedMontyField> for &BoxedMontyField {
            type Output = BoxedMontyField;

            #[inline(always)]
            fn $method(self, rhs: BoxedMontyField) -> Self::Output {
                self.$method(&rhs)
            }
        }
    };
}

impl_basic_op!(Add, add);
impl_basic_op!(Sub, sub);
impl_basic_op!(Mul, mul);

impl Div for BoxedMontyField {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        self.div(&rhs)
    }
}

impl Div<&Self> for BoxedMontyField {
    type Output = Self;

    fn div(self, rhs: &Self) -> Self::Output {
        self.checked_div(rhs).expect("Division by zero")
    }
}

impl Div for &BoxedMontyField {
    type Output = BoxedMontyField;

    fn div(self, rhs: Self) -> Self::Output {
        self.checked_div(rhs).expect("Division by zero")
    }
}

impl Div<BoxedMontyField> for &BoxedMontyField {
    type Output = BoxedMontyField;

    fn div(self, rhs: BoxedMontyField) -> Self::Output {
        self.div(&rhs)
    }
}

impl Rem for BoxedMontyField {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        self.rem(&rhs)
    }
}

impl Rem<&Self> for BoxedMontyField {
    type Output = Self;

    fn rem(self, rhs: &Self) -> Self::Output {
        self.checked_rem(rhs).expect("Division by zero")
    }
}

impl Rem for &BoxedMontyField {
    type Output = BoxedMontyField;

    fn rem(self, rhs: Self) -> Self::Output {
        self.checked_rem(rhs).expect("Division by zero")
    }
}

impl Rem<BoxedMontyField> for &BoxedMontyField {
    type Output = BoxedMontyField;

    fn rem(self, rhs: BoxedMontyField) -> Self::Output {
        self.rem(&rhs)
    }
}

impl Pow<u32> for BoxedMontyField {
    type Output = Self;

    fn pow(self, rhs: u32) -> Self::Output {
        Self(self.0.pow(&BoxedUint::from(rhs)))
    }
}

impl Inv for BoxedMontyField {
    type Output = Option<Self>;

    fn inv(self) -> Self::Output {
        let result = self.0.invert_vartime();
        if result.is_some().into() {
            Some(Self(result.unwrap()))
        } else {
            None
        }
    }
}

//
// Checked arithmetic operations
// (Note: Field operations do not overflow)
//

impl CheckedDiv for BoxedMontyField {
    #[allow(clippy::arithmetic_side_effects)] // False alert
    fn checked_div(&self, rhs: &Self) -> Option<Self> {
        let inv = rhs.0.invert();
        if inv.is_none().into() {
            return None; // Division by zero
        }
        // Safe to unwrap since we checked for None above
        let inv = inv.unwrap();
        Some(Self(BoxedMontyForm::mul(&self.0, &inv)))
    }
}

impl CheckedRem for BoxedMontyField {
    fn checked_rem(&self, v: &Self) -> Option<Self> {
        let rhs = NonZero::new(v.0.retrieve()).into_option()?;
        Some(Self(BoxedMontyForm::new(
            self.0.retrieve().rem(&rhs),
            self.0.params().clone(),
        )))
    }
}

//
// Arithmetic assign operations
//

macro_rules! impl_field_op_assign {
    ($trait:ident, $method:ident) => {
        impl $trait for BoxedMontyField {
            fn $method(&mut self, rhs: Self) {
                self.0.$method(&rhs.0);
            }
        }
        impl $trait<&Self> for BoxedMontyField {
            fn $method(&mut self, rhs: &Self) {
                self.0.$method(&rhs.0);
            }
        }
    };
}

impl_field_op_assign!(AddAssign, add_assign);
impl_field_op_assign!(SubAssign, sub_assign);
impl_field_op_assign!(MulAssign, mul_assign);

impl DivAssign for BoxedMontyField {
    fn div_assign(&mut self, rhs: Self) {
        self.div_assign(&rhs);
    }
}

impl DivAssign<&Self> for BoxedMontyField {
    fn div_assign(&mut self, rhs: &Self) {
        self.0.mul_assign(rhs.0.invert().expect("Division by zero"))
    }
}

impl RemAssign for BoxedMontyField {
    fn rem_assign(&mut self, rhs: Self) {
        self.rem_assign(&rhs);
    }
}

impl RemAssign<&Self> for BoxedMontyField {
    fn rem_assign(&mut self, rhs: &Self) {
        *self = self.checked_rem(rhs).expect("Division by zero");
    }
}

//
// Aggregate operations
//

impl Sum for BoxedMontyField {
    fn sum<I: Iterator<Item = Self>>(mut iter: I) -> Self {
        let Some(BoxedMontyField(first)) = iter.next() else {
            panic!("Sum of an empty iterator is not defined for BoxedMontyField");
        };
        Self(iter.fold(first, |acc, x| BoxedMontyForm::add(&acc, &x.0)))
    }
}

impl<'a> Sum<&'a Self> for BoxedMontyField {
    fn sum<I: Iterator<Item = &'a Self>>(mut iter: I) -> Self {
        let Some(BoxedMontyField(first)) = iter.next() else {
            panic!("Sum of an empty iterator is not defined for BoxedMontyField");
        };
        Self(iter.fold(first.clone(), |acc, x| BoxedMontyForm::add(&acc, &x.0)))
    }
}

impl Product for BoxedMontyField {
    fn product<I: Iterator<Item = Self>>(mut iter: I) -> Self {
        let Some(BoxedMontyField(first)) = iter.next() else {
            panic!("Product of an empty iterator is not defined for BoxedMontyField");
        };
        Self(iter.fold(first, |acc, x| BoxedMontyForm::mul(&acc, &x.0)))
    }
}

impl<'a> Product<&'a Self> for BoxedMontyField {
    #[allow(clippy::arithmetic_side_effects)] // False alert
    fn product<I: Iterator<Item = &'a Self>>(mut iter: I) -> Self {
        let Some(BoxedMontyField(first)) = iter.next() else {
            panic!("Product of an empty iterator is not defined for BoxedMontyField");
        };
        Self(iter.fold(first.clone(), |acc, x| BoxedMontyForm::mul(&acc, &x.0)))
    }
}

//
// Conversions
//

impl From<BoxedMontyForm> for BoxedMontyField {
    #[inline(always)]
    fn from(value: BoxedMontyForm) -> Self {
        Self(value)
    }
}

impl From<BoxedMontyField> for BoxedMontyForm {
    #[inline(always)]
    fn from(value: BoxedMontyField) -> Self {
        value.0
    }
}

impl From<&BoxedMontyField> for BoxedMontyField {
    fn from(value: &Self) -> Self {
        value.clone()
    }
}

macro_rules! impl_from_unsigned {
    ($($t:ty),* $(,)?) => {
        $(
            impl FromWithConfig<$t> for BoxedMontyField {
                fn from_with_cfg(value: $t, cfg: &Self::Config) -> Self {
                    let abs: BoxedUint = value.into();
                    let abs = abs.resize(cfg.modulus().bits_precision());
                    Self(BoxedMontyForm::new(abs, cfg.clone()))
                }
            }

            impl FromWithConfig<&$t> for BoxedMontyField {
                fn from_with_cfg(value: &$t, cfg: &Self::Config) -> Self {
                    Self::from_with_cfg(*value, cfg)
                }
            }
        )*
    };
}

macro_rules! impl_from_signed {
    ($($t:ty),* $(,)?) => {
        $(
            #[allow(clippy::arithmetic_side_effects)] // False alert
            impl FromWithConfig<$t> for BoxedMontyField {
                fn from_with_cfg(value: $t, cfg: &Self::Config) -> Self {
                    let abs: u128 = if value == <$t>::MIN {
                        <u128 as TryFrom<$t>>::try_from(<$t>::MAX).expect("unreachable") + 1
                    } else {
                        value.abs().try_into().expect("unreachable")
                    };
                    let abs: BoxedUint = abs.into();
                    let abs = abs.resize(cfg.modulus().bits_precision());
                    let result = Self(BoxedMontyForm::new(abs, cfg.clone()));
                    if value.is_negative() { -result } else { result }
                }
            }

            impl FromWithConfig<&$t> for BoxedMontyField {
                fn from_with_cfg(value: &$t, cfg: &Self::Config) -> Self {
                    Self::from_with_cfg(*value, cfg)
                }
            }
        )*
    };
}

impl_from_unsigned!(u8, u16, u32, u64, u128);
impl_from_signed!(i8, i16, i32, i64, i128);

impl FromWithConfig<bool> for BoxedMontyField {
    fn from_with_cfg(value: bool, cfg: &Self::Config) -> Self {
        let abs: BoxedUint = if value {
            BoxedUint::one()
        } else {
            BoxedUint::zero()
        };
        let abs = abs.resize(cfg.modulus().bits_precision());
        Self(BoxedMontyForm::new(abs, cfg.clone()))
    }
}

impl FromWithConfig<&bool> for BoxedMontyField {
    fn from_with_cfg(value: &bool, cfg: &Self::Config) -> Self {
        Self::from_with_cfg(*value, cfg)
    }
}

impl FromWithConfig<Boolean> for BoxedMontyField {
    fn from_with_cfg(value: Boolean, cfg: &Self::Config) -> Self {
        Self::from_with_cfg(*value, cfg)
    }
}

impl FromWithConfig<&Boolean> for BoxedMontyField {
    fn from_with_cfg(value: &Boolean, cfg: &Self::Config) -> Self {
        Self::from_with_cfg(*value, cfg)
    }
}

impl<const LIMBS: usize> FromWithConfig<Int<LIMBS>> for BoxedMontyField {
    fn from_with_cfg(value: Int<LIMBS>, cfg: &Self::Config) -> Self {
        Self::from_with_cfg(&value, cfg)
    }
}

impl<const LIMBS: usize> FromWithConfig<&Int<LIMBS>> for BoxedMontyField {
    #[allow(clippy::arithmetic_side_effects)] // False alert
    fn from_with_cfg(value: &Int<LIMBS>, cfg: &Self::Config) -> Self {
        let abs: BoxedUint = value.inner().abs().into();
        let abs = abs.resize(cfg.modulus().bits_precision());

        let result = Self(BoxedMontyForm::new(abs, cfg.clone()));

        if value.is_negative() { -result } else { result }
    }
}

impl FromWithConfig<BoxedUint> for BoxedMontyField {
    fn from_with_cfg(value: BoxedUint, cfg: &Self::Config) -> Self {
        Self::from_with_cfg(&value, cfg)
    }
}

impl FromWithConfig<&BoxedUint> for BoxedMontyField {
    fn from_with_cfg(value: &BoxedUint, cfg: &Self::Config) -> Self {
        let value = value.resize(cfg.modulus().bits_precision());
        Self(BoxedMontyForm::new(value, cfg.clone()))
    }
}

impl<const LIMBS: usize> FromWithConfig<crypto_bigint::Uint<LIMBS>> for BoxedMontyField {
    fn from_with_cfg(value: crypto_bigint::Uint<LIMBS>, cfg: &Self::Config) -> Self {
        Self::from_with_cfg(&value, cfg)
    }
}

impl<const LIMBS: usize> FromWithConfig<&crypto_bigint::Uint<LIMBS>> for BoxedMontyField {
    #[allow(clippy::arithmetic_side_effects)] // False alert
    fn from_with_cfg(value: &crypto_bigint::Uint<LIMBS>, cfg: &Self::Config) -> Self {
        let value: BoxedUint = value.into();
        let value = value.resize(cfg.modulus().bits_precision());
        Self(BoxedMontyForm::new(value, cfg.clone()))
    }
}

//
// Semiring, Ring and Field
//

impl Semiring for BoxedMontyField {}

impl Ring for BoxedMontyField {}

impl IntSemiring for BoxedMontyField {
    fn is_odd(&self) -> bool {
        // Sadly there's no efficient way to implement this for Montgomery form
        self.0.retrieve().is_odd().into()
    }

    fn is_even(&self) -> bool {
        // Sadly there's no efficient way to implement this for Montgomery form
        self.0.retrieve().is_even().into()
    }
}

impl IntRing for BoxedMontyField {
    fn checked_abs(&self) -> Option<Self> {
        Some(self.clone())
    }

    fn is_positive(&self) -> bool {
        self.0.is_nonzero().into()
    }

    fn is_negative(&self) -> bool {
        false
    }
}

impl Field for BoxedMontyField {
    type Inner = BoxedUint;

    #[inline(always)]
    fn inner(&self) -> &Self::Inner {
        self.0.as_montgomery()
    }
}

impl PrimeField for BoxedMontyField {
    type Config = BoxedMontyParams;

    fn cfg(&self) -> &Self::Config {
        self.0.params()
    }

    fn modulus(&self) -> BoxedUint {
        self.0.params().modulus().clone().get()
    }

    #[allow(clippy::arithmetic_side_effects)] // False alert
    fn modulus_minus_one_div_two(&self) -> BoxedUint {
        let value = self.0.params().modulus().clone().get();
        (value - BoxedUint::one()) / NonZero::new(BoxedUint::from(2_u8)).unwrap()
    }

    fn make_cfg(modulus: &Self::Inner) -> Result<Self::Config, FieldError> {
        let Some(modulus) = Odd::new(modulus.clone()).into_option() else {
            return Err(FieldError::InvalidModulus);
        };
        Ok(BoxedMontyParams::new(modulus))
    }

    fn new_unchecked_with_cfg(inner: Self::Inner, cfg: &Self::Config) -> Self {
        Self(BoxedMontyForm::from_montgomery(inner, cfg.clone()))
    }

    fn zero_with_cfg(cfg: &Self::Config) -> Self {
        Self(BoxedMontyForm::zero(cfg.clone()))
    }

    fn is_zero_with_cfg(&self, _cfg: &Self::Config) -> bool {
        self.0.is_zero().into()
    }

    fn one_with_cfg(cfg: &Self::Config) -> Self {
        Self(BoxedMontyForm::one(cfg.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{IntRingWithRem, ensure_type_implements_trait};
    use alloc::vec;
    use crypto_bigint::BoxedUint;
    use num_traits::Pow;

    type F = BoxedMontyField;

    //
    // Test helpers
    //
    fn test_config() -> BoxedMontyParams {
        // Using a 256-bit prime
        let modulus = BoxedUint::from_be_hex(
            "00dca94d8a1ecce3b6e8755d8999787d0524d8ca1ea755e7af84fb646fa31f27",
            256,
        )
        .unwrap();
        let modulus = Odd::new(modulus).expect("modulus should be odd");
        BoxedMontyParams::new(modulus)
    }

    fn from_u64(value: u64) -> F {
        F::from_with_cfg(value, &test_config())
    }

    fn from_i64(value: i64) -> F {
        F::from_with_cfg(value, &test_config())
    }

    fn zero() -> F {
        F::zero_with_cfg(&test_config())
    }

    fn one() -> F {
        F::one_with_cfg(&test_config())
    }

    #[test]
    fn ensure_blanket_traits() {
        ensure_type_implements_trait!(F, IntRingWithRem);
    }

    #[test]
    fn zero_one_basics() {
        let z = zero();
        assert!(z.is_zero_with_cfg(&test_config()));
        let o = one();
        assert!(!o.is_zero_with_cfg(&test_config()));
        assert_ne!(z, o);
    }

    #[test]
    fn basic_operations() {
        // Negation
        let a = from_u64(9);
        let neg_a = -a.clone();
        assert_eq!(&a + &neg_a, zero());

        let a = from_u64(10);
        let b = from_u64(5);

        // Addition
        let c = a.clone() + b.clone();
        assert_eq!(c, from_u64(15));

        // Subtraction
        let d = a.clone() - b.clone();
        assert_eq!(d, from_u64(5));

        // Multiplication
        let e = a.clone() * b.clone();
        assert_eq!(e, from_u64(50));

        // Division
        let num = from_u64(11);
        let den = from_u64(5);
        let q = num.clone() / den.clone();
        assert_eq!(&q * &den, num);
    }

    #[test]
    fn add_wrapping() {
        let a = from_i64(-100);
        let b = from_u64(105);
        let c = a + b;
        let d = from_u64(5);
        assert_eq!(c, d);
    }

    #[allow(clippy::op_ref)]
    #[test]
    fn reference_operations() {
        let a = from_u64(10);
        let b = from_u64(5);

        // Addition
        let c = &a + &b;
        assert_eq!(c, from_u64(15));

        // Subtraction
        let d = &a - &b;
        assert_eq!(d, from_u64(5));

        // Multiplication
        let e = &a * &b;
        assert_eq!(e, from_u64(50));

        // Division
        let num = from_u64(11);
        let den = from_u64(5);
        let q = &num / &den;
        assert_eq!(&q * &den, num);
    }

    #[test]
    fn from_unsigned_and_signed() {
        let cfg = test_config();
        assert_eq!(F::from_with_cfg(0_u64, &cfg), zero());
        assert_eq!(F::from_with_cfg(1_u32, &cfg), one());
        assert_eq!(from_i64(-1) + one(), zero());
        assert_eq!(from_i64(-5) + from_u64(5), zero());
    }

    #[test]
    fn from_bool() {
        let cfg = test_config();
        assert_eq!(F::from_with_cfg(true, &cfg), one());
        assert_eq!(F::from_with_cfg(false, &cfg), zero());

        let t = F::from_with_cfg(true, &cfg);
        let f = F::from_with_cfg(false, &cfg);
        assert_eq!(t, one());
        assert_eq!(f, zero());
    }

    #[test]
    fn assign_operations() {
        // Addition
        let mut a = from_u64(5);
        a += from_u64(6);
        assert_eq!(a, from_u64(11));

        // Subtraction
        let mut a = from_u64(20);
        a -= from_u64(7);
        assert_eq!(a, from_u64(13));

        // Multiplication
        let mut a = from_u64(11);
        a *= from_u64(3);
        assert_eq!(a, from_u64(33));

        // Division
        let mut a = from_u64(20);
        let b = from_u64(4);
        a /= b;
        assert_eq!(&a * &from_u64(4), from_u64(20));
    }

    #[test]
    #[should_panic(expected = "Division by zero")]
    fn div_by_zero_panics() {
        let a = from_u64(7);
        let zero = zero();
        let _ = a / zero;
    }

    #[test]
    fn pow_operation() {
        // Test basic exponentiation
        let base = from_u64(2);

        // 2^0 = 1
        assert_eq!(base.clone().pow(0), one());

        // 2^1 = 2
        assert_eq!(base.clone().pow(1), base);

        // 2^3 = 8
        assert_eq!(base.clone().pow(3), from_u64(8));

        // 2^10 = 1024
        assert_eq!(base.clone().pow(10), from_u64(1024));

        // Test with different base
        let base = from_u64(3);

        // 3^4 = 81
        assert_eq!(base.clone().pow(4), from_u64(81));

        // Test with base 1
        let base = one();
        assert_eq!(base.clone().pow(1000), one());

        // Test with base 0
        let base = zero();
        assert_eq!(base.clone().pow(0), one()); // 0^0 = 1 by convention
        assert_eq!(base.clone().pow(10), zero()); // 0^n = 0 for n > 0
    }

    #[test]
    fn inv_operation() {
        let a = from_u64(5);
        let inv_a = a.clone().inv().unwrap();
        assert_eq!(&a * &inv_a, one());

        // Test that zero has no inverse
        let zero = zero();
        assert!(zero.inv().is_none());
    }

    #[test]
    fn checked_neg() {
        // Test with positive number
        let a = from_u64(10);
        let neg_a = a.checked_neg().unwrap();
        assert_eq!(neg_a, from_i64(-10));

        // Test with negative number
        let b = from_i64(-5);
        let neg_b = b.checked_neg().unwrap();
        assert_eq!(neg_b, from_u64(5));

        // Test with zero
        let zero_val = zero();
        let neg_zero = zero_val.checked_neg().unwrap();
        assert_eq!(neg_zero, zero());
    }

    #[test]
    fn checked_add() {
        let a = from_u64(10);
        let b = from_u64(5);

        let c = a.checked_add(&b).unwrap();
        assert_eq!(c, from_u64(15));
    }

    #[test]
    fn checked_sub() {
        let a = from_u64(10);
        let b = from_u64(5);

        let d = a.checked_sub(&b).unwrap();
        assert_eq!(d, from_u64(5));
    }

    #[test]
    fn checked_mul() {
        let a = from_u64(10);
        let b = from_u64(5);

        let e = a.checked_mul(&b).unwrap();
        assert_eq!(e, from_u64(50));
    }

    #[test]
    fn checked_div() {
        let a = from_u64(10);
        let b = from_u64(5);
        let zero = zero();

        // Normal division
        let c = a.checked_div(&b).unwrap();
        assert_eq!(&c * &b, a);

        // Division by zero
        assert!(a.checked_div(&zero).is_none());
    }

    #[test]
    fn checked_rem() {
        let a = from_u64(17);
        let b = from_u64(5);
        let zero = zero();

        // Normal remainder
        let c = a.checked_rem(&b).unwrap();
        assert_eq!(c, from_u64(2));

        // Remainder by zero
        assert!(a.checked_rem(&zero).is_none());
    }

    #[allow(clippy::op_ref)]
    #[test]
    fn ref_and_value_combinations_add_sub_mul() {
        let a = from_u64(42);
        let b = from_u64(123);

        let r1 = &a + &b;
        let a1 = from_u64(42);
        let b1 = from_u64(123);
        let r2 = a1 + b1;
        let a2 = from_u64(42);
        let b2 = from_u64(123);
        let r3 = a2 + &b2;
        let a3 = from_u64(42);
        let b3 = from_u64(123);
        let r4 = &a3 + b3;
        assert_eq!(r1, r2);
        assert_eq!(r1, r3);
        assert_eq!(r1, r4);

        let a = from_u64(88);
        let b = from_u64(59);
        let s1 = &a - &b;
        let a1 = from_u64(88);
        let b1 = from_u64(59);
        let s2 = a1 - b1;
        let a2 = from_u64(88);
        let b2 = from_u64(59);
        let s3 = a2 - &b2;
        let a3 = from_u64(88);
        let b3 = from_u64(59);
        let s4 = &a3 - b3;
        assert_eq!(s1, s2);
        assert_eq!(s1, s3);
        assert_eq!(s1, s4);

        let a = from_u64(9);
        let b = from_u64(14);
        let m1 = &a * &b;
        let a1 = from_u64(9);
        let b1 = from_u64(14);
        let m2 = a1 * b1;
        let a2 = from_u64(9);
        let b2 = from_u64(14);
        let m3 = a2 * &b2;
        let a3 = from_u64(9);
        let b3 = from_u64(14);
        let m4 = &a3 * b3;
        assert_eq!(m1, m2);
        assert_eq!(m1, m3);
        assert_eq!(m1, m4);
    }

    #[test]
    fn assign_ops_with_refs_and_val() {
        let mut a = from_u64(100);
        let b = from_u64(50);
        a += b;
        assert_eq!(a, from_u64(150));

        let mut c = from_u64(100);
        let d = from_u64(50);
        c += &d;
        assert_eq!(c, from_u64(150));

        let mut e = from_u64(100);
        let f = from_u64(30);
        e -= f;
        assert_eq!(e, from_u64(70));

        let mut g = from_u64(100);
        let h = from_u64(30);
        g -= &h;
        assert_eq!(g, from_u64(70));

        let mut i = from_u64(10);
        let j = from_u64(5);
        i *= j;
        assert_eq!(i, from_u64(50));

        let mut k = from_u64(10);
        let l = from_u64(5);
        k *= &l;
        assert_eq!(k, from_u64(50));
    }

    #[test]
    fn aggregate_operations() {
        // Sum
        let values = vec![from_u64(1), from_u64(2), from_u64(3)];
        let sum: F = values.iter().sum();
        assert_eq!(sum, from_u64(6));

        let sum2: F = values.into_iter().sum();
        assert_eq!(sum2, from_u64(6));

        // Product
        let values = vec![from_u64(2), from_u64(3), from_u64(4)];
        let product: F = values.iter().product();
        assert_eq!(product, from_u64(24));

        let product2: F = values.into_iter().product();
        assert_eq!(product2, from_u64(24));

        // Test empty collections panic
        // Note: BoxedMontyField doesn't have a default config, so empty
        // iterators must panic
    }

    #[test]
    fn conversions() {
        let cfg = test_config();

        // Test FromWithConfig for BoxedUint
        let u: BoxedUint = BoxedUint::from(123_u64);
        let f = F::from_with_cfg(u, &cfg);
        assert_eq!(f, from_u64(123));

        // Test inner() and cfg()
        let value = from_u64(456);
        let _inner_ref = value.inner();
        let _cfg_ref = value.cfg();
    }

    #[test]
    fn from_primitive() {
        let cfg = test_config();

        // Test FromWithConfig for various types
        let a = F::from_with_cfg(42_u8, &cfg);
        assert_eq!(a, from_u64(42));

        let b = F::from_with_cfg(12345_u16, &cfg);
        assert_eq!(b, from_u64(12345));

        let c = F::from_with_cfg(1234567890_u32, &cfg);
        assert_eq!(c, from_u64(1234567890));

        let d = F::from_with_cfg(1234567890123456789_u64, &cfg);
        assert_eq!(d, from_u64(1234567890123456789));

        let e = F::from_with_cfg(-42_i8, &cfg);
        assert_eq!(e, from_i64(-42));

        let f = F::from_with_cfg(-12345_i16, &cfg);
        assert_eq!(f, from_i64(-12345));

        let g = F::from_with_cfg(-1234567_i32, &cfg);
        assert_eq!(g, from_i64(-1234567));

        let h = F::from_with_cfg(-1234567890123456789_i64, &cfg);
        assert_eq!(h, from_i64(-1234567890123456789));
    }

    #[test]
    fn clone_works() {
        let a = from_u64(42);
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn equality_and_ordering() {
        let a = from_u64(10);
        let b = from_u64(10);
        let c = from_u64(20);

        // Test equality
        assert_eq!(a, b);
        assert_ne!(a, c);

        // Test ordering
        // Note: ordering is based on Montgomery representation, not value
        // We just test consistency of ordering
        assert_eq!(a.partial_cmp(&b), Some(Ordering::Equal));
        assert_ne!(a.partial_cmp(&c), Some(Ordering::Equal));
        assert!(a <= b);
        assert!(a >= b);

        // Verify transitivity of ordering
        let d = from_u64(30);
        if a < c && c < d {
            assert!(a < d);
        }
    }

    #[test]
    fn hash_trait() {
        use core::hash::{Hash, Hasher};

        // Simple hasher for testing
        struct TestHasher {
            state: u64,
        }

        impl Hasher for TestHasher {
            fn finish(&self) -> u64 {
                self.state
            }

            fn write(&mut self, bytes: &[u8]) {
                for &byte in bytes {
                    self.state = self.state.wrapping_mul(31).wrapping_add(u64::from(byte));
                }
            }
        }

        let a = from_u64(42);
        let b = from_u64(42);

        let mut hasher_a = TestHasher { state: 0 };
        a.hash(&mut hasher_a);
        let hash_a = hasher_a.finish();

        let mut hasher_b = TestHasher { state: 0 };
        b.hash(&mut hasher_b);
        let hash_b = hasher_b.finish();

        // Equal values should have equal hashes
        assert_eq!(hash_a, hash_b);
    }

    // BoxedMontyField-specific tests

    #[test]
    fn prime_field_methods() {
        let a = from_u64(42);
        let cfg = a.cfg();

        // Test that we can get modulus
        let modulus = a.modulus();
        assert!(modulus.bits_precision() > 0);

        // Test modulus_minus_one_div_two
        let m_minus_1_div_2 = a.modulus_minus_one_div_two();
        assert!(m_minus_1_div_2.bits_precision() > 0);

        // Test zero_with_cfg and one_with_cfg
        let z = F::zero_with_cfg(cfg);
        assert!(z.is_zero_with_cfg(cfg));
        let o = F::one_with_cfg(cfg);
        assert!(!o.is_zero_with_cfg(cfg));
    }

    #[test]
    fn make_cfg_works() {
        let modulus = BoxedUint::from_be_hex(
            "00dca94d8a1ecce3b6e8755d8999787d0524d8ca1ea755e7af84fb646fa31f27",
            256,
        )
        .unwrap();
        let cfg = F::make_cfg(&modulus).expect("Should create config");

        // Create a field element using this config
        let a = F::from_with_cfg(123_u64, &cfg);
        assert_eq!(a, from_u64(123));
    }

    #[test]
    fn make_cfg_rejects_even_modulus() {
        let even_modulus = BoxedUint::from(42_u64);
        let result = F::make_cfg(&even_modulus);
        assert!(result.is_err());
    }

    #[test]
    fn int_ring_methods() {
        let even = from_u64(20);
        let odd = from_u64(21);

        assert!(even.is_even());
        assert!(!even.is_odd());
        assert!(odd.is_odd());
        assert!(!odd.is_even());

        let a = from_u64(10);
        assert_eq!(a.checked_abs(), Some(a.clone()));
        assert!(a.is_positive());
        assert!(!a.is_negative());

        let zero = zero();
        assert!(!zero.is_positive());
        assert!(!zero.is_negative());
    }
}
