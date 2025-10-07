use super::*;
use core::{
    cmp::Ordering,
    fmt::{Debug, Display, Formatter, Result as FmtResult},
    hash::{Hash, Hasher},
    iter::{Product, Sum},
    ops::{Add, AddAssign, DivAssign, Mul, MulAssign, Rem, RemAssign, Sub, SubAssign},
};
use crypto_bigint::{
    BoxedUint, NonZero, Odd,
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
    pub fn new(form: BoxedMontyForm) -> Self {
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
        Some(self.cmp(other))
    }
}

impl Ord for BoxedMontyField {
    fn cmp(&self, other: &Self) -> Ordering {
        Ord::cmp(self.0.as_montgomery(), other.0.as_montgomery())
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

//
// Ring and Field
//

impl Ring for BoxedMontyField {}

impl IntRing for BoxedMontyField {}

impl Field for BoxedMontyField {
    type Inner = BoxedUint;

    #[inline(always)]
    fn inner(&self) -> &Self::Inner {
        self.0.as_montgomery()
    }
}

impl PrimeField for BoxedMontyField {
    type Config = BoxedMontyParams;

    fn modulus(&self) -> BoxedUint {
        self.0.params().modulus().clone().get()
    }

    #[allow(clippy::arithmetic_side_effects)] // False alert
    fn modulus_minus_one_div_two(&self) -> BoxedUint {
        let value = self.0.params().modulus().clone().get();
        (value - BoxedUint::one()) / NonZero::new(BoxedUint::from(2_u8)).unwrap()
    }

    fn new_with_modulus(inner: Self::Inner, modulus: &Self::Inner) -> Result<Self, FieldError> {
        let Some(modulus) = Odd::new(modulus.clone()).into_option() else {
            return Err(FieldError::InvalidModulus);
        };
        let params = BoxedMontyParams::new(modulus);
        Ok(Self(BoxedMontyForm::new(inner, params)))
    }

    fn new_with_cfg(inner: Self::Inner, cfg: Self::Config) -> Self {
        Self(BoxedMontyForm::new(inner, cfg))
    }

    fn zero_with_cfg(cfg: &Self::Config) -> Self {
        Self(BoxedMontyForm::zero(cfg.clone()))
    }

    fn is_zero_with_cfg(&self, _cfg: &Self::Config) -> bool {
        self.0.is_zero().into()
    }

    fn one_with_cfg(cfg: &Self::Config) -> Self {
        Self(BoxedMontyForm::zero(cfg.clone()))
    }
}
