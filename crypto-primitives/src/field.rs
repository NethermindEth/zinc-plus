#[cfg(feature = "ark_ff")]
pub mod ark_ff_field;
#[cfg(feature = "ark_ff")]
pub mod ark_ff_fp;
#[cfg(feature = "crypto_bigint")]
pub mod crypto_bigint_boxed_monty;
#[cfg(feature = "crypto_bigint")]
pub mod crypto_bigint_const_monty;

use crate::{ConstIntRing, ring::Ring};
use core::{
    fmt::Debug,
    ops::{Div, DivAssign, Neg},
};
use num_traits::{Inv, Pow, Zero};
use thiserror::Error;

/// Element of a field (F) - a group where addition and multiplication are
/// defined with their respective inverse operations.
pub trait Field:
    Ring
    + Neg<Output=Self>
    + Pow<u32, Output=Self>
    // Arithmetic operations consuming rhs
    + Div<Output=Self>
    + DivAssign
    // Arithmetic operations with rhs reference
    + for<'a> Div<&'a Self, Output=Self>
    + for<'a> DivAssign<&'a Self>
{
    /// Underlying representation of an element
    type Inner: Debug + Eq + Clone;

    fn inner(&self) -> &Self::Inner;
}

/// Element of an integer field modulo prime number (F_p).
/// Prime modulus might be dynamic and can be determined at runtime.
///
/// When performing arithmetic operations, the modulus of both operands must be
/// the same, otherwise operations should panic.
///
/// Constant prime fields are considered a special case of dynamic prime fields.
pub trait PrimeField: Field {
    /// Runtime configuration for the prime field, empty for constant prime
    /// fields. For dynamic prime fields, it could be just modulus or more
    /// complex structure.
    type Config: Debug + Clone + 'static;

    fn cfg(&self) -> &Self::Config;

    fn modulus(&self) -> Self::Inner;

    fn modulus_minus_one_div_two(&self) -> Self::Inner;

    fn make_cfg(modulus: &Self::Inner) -> Result<Self::Config, FieldError>;

    /// Creates a new instance of the prime field element from a representation
    /// known to be valid - should consume exactly the value returned by
    /// `inner()`. Ideally, this should not check the validity of the
    /// element, but it's acceptable to perform a check if it can't be
    /// avoided.
    fn new_unchecked_with_cfg(inner: Self::Inner, cfg: &Self::Config) -> Self;

    fn zero_with_cfg(cfg: &Self::Config) -> Self;

    fn is_zero_with_cfg(&self, cfg: &Self::Config) -> bool;

    fn one_with_cfg(cfg: &Self::Config) -> Self;
}

/// Prime field whose modulus is a constant value known at compile time.
pub trait ConstPrimeField:
    Field + ConstIntRing + Inv<Output = Option<Self>> + From<u64> + From<u128> + From<Self::Inner>
{
    const MODULUS: Self::Inner;
    const MODULUS_MINUS_ONE_DIV_TWO: Self::Inner;

    /// Creates a new instance of the prime field element from a representation
    /// known to be valid - should consume exactly the value returned by
    /// `inner()`. Ideally, this should not check the validity of the
    /// element, but it's acceptable to perform a check if it can't be
    /// avoided.
    fn new_unchecked(inner: Self::Inner) -> Self;
}

impl<T: ConstPrimeField> PrimeField for T {
    /// For constant prime fields, the configuration is empty.
    type Config = ();

    fn cfg(&self) -> &Self::Config {
        &()
    }

    #[inline(always)]
    fn modulus(&self) -> Self::Inner {
        Self::MODULUS
    }

    #[inline(always)]
    fn modulus_minus_one_div_two(&self) -> T::Inner {
        Self::MODULUS_MINUS_ONE_DIV_TWO
    }

    fn make_cfg(modulus: &Self::Inner) -> Result<Self::Config, FieldError> {
        if *modulus == Self::MODULUS {
            Ok(())
        } else {
            Err(FieldError::InvalidModulus)
        }
    }

    #[inline(always)]
    fn new_unchecked_with_cfg(inner: Self::Inner, _cfg: &Self::Config) -> Self {
        ConstPrimeField::new_unchecked(inner)
    }

    #[inline(always)]
    fn zero_with_cfg(_cfg: &Self::Config) -> Self {
        Self::ZERO
    }

    #[inline(always)]
    fn is_zero_with_cfg(&self, _cfg: &Self::Config) -> bool {
        Zero::is_zero(self)
    }

    #[inline(always)]
    fn one_with_cfg(_cfg: &Self::Config) -> Self {
        Self::ONE
    }
}

/// Element of a prime field in its Montgomery representation of - encoded in a
/// way so that modular multiplication can be done without performing an
/// explicit division by pp after each product.
pub trait MontgomeryField: PrimeField {
    // FIXME

    /// INV = -MODULUS^{-1} mod R
    const INV: Self::Inner;
}

/// Analogous to `From` trait, but with a prime field configuration parameter.
pub trait FromWithConfig<T>: PrimeField {
    fn from_with_cfg(value: T, cfg: &Self::Config) -> Self;
}

/// Trivial implementation for types that implement `From<T>`.
impl<F, T> FromWithConfig<T> for F
where
    F: PrimeField + From<T>,
{
    fn from_with_cfg(value: T, _cfg: &Self::Config) -> Self {
        Self::from(value)
    }
}

/// Analogous to `Into` trait, but with a prime field configuration parameter.
/// Preferribly should not be implemented directly.
pub trait IntoWithConfig<F: PrimeField> {
    fn into_with_cfg(self, cfg: &F::Config) -> F;
}

impl<F, T> IntoWithConfig<F> for T
where
    F: PrimeField + FromWithConfig<T>,
{
    fn into_with_cfg(self, cfg: &F::Config) -> F {
        F::from_with_cfg(self, cfg)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum FieldError {
    #[error("Invalid field modulus")]
    InvalidModulus,
}
