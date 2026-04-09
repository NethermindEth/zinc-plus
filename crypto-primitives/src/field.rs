#[cfg(feature = "ark_ff")]
pub mod ark_ff_field;
#[cfg(feature = "ark_ff")]
pub mod ark_ff_fp;
#[cfg(feature = "crypto_bigint")]
pub mod crypto_bigint_boxed_monty;
#[cfg(feature = "crypto_bigint")]
pub mod crypto_bigint_const_monty;
#[cfg(feature = "crypto_bigint")]
pub(crate) mod crypto_bigint_helpers;
#[cfg(feature = "crypto_bigint")]
pub mod crypto_bigint_monty;
pub mod f2;

use crate::{ConstSemiring, ring::Ring};
use core::{
    fmt::Debug,
    ops::{Div, DivAssign, Neg},
};
use num_traits::{Inv, Pow, Zero};
use thiserror::Error;

#[cfg(target_pointer_width = "64")]
pub const WORD_FACTOR: usize = 1;
#[cfg(target_pointer_width = "32")]
pub const WORD_FACTOR: usize = 2;

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
    type Inner: Debug + Eq + Clone + Sync + Send;

    /// Type used to represent the modulus. Usually the same as `Self::Inner`,
    /// but may differ when the modulus doesn't fit in the element representation
    /// (e.g. F2 where elements are `bool` but the modulus is `2: u8`).
    type Modulus: Debug + Eq + Clone + Sync + Send;

    fn inner(&self) -> &Self::Inner;
    fn inner_mut(&mut self) -> &mut Self::Inner;
    fn into_inner(self) -> Self::Inner;
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
    type Config: Debug + Clone + Send + Sync + 'static;

    fn cfg(&self) -> &Self::Config;

    // Note: Not using `&self` to avoid conflicts with `Zero` trait.
    fn is_zero(value: &Self) -> bool;

    fn modulus(&self) -> Self::Modulus;

    fn modulus_minus_one_div_two(&self) -> Self::Inner;

    fn make_cfg(modulus: &Self::Modulus) -> Result<Self::Config, FieldError>;

    /// Creates a new instance of a prime field element from
    /// an arbitrary element of `Self::Inner`. The method
    /// should not assume the `Self::Inner` is coming in a
    /// form internally used by the field type. So it
    /// always should perform a reduction first.
    fn new_with_cfg(inner: Self::Inner, cfg: &Self::Config) -> Self;

    /// Creates a new instance of the prime field element from a representation
    /// known to be valid - should consume exactly the value returned by
    /// `inner()`. Ideally, this should not check the validity of the
    /// element, but it's acceptable to perform a check if it can't be
    /// avoided.
    fn new_unchecked_with_cfg(inner: Self::Inner, cfg: &Self::Config) -> Self;

    fn zero_with_cfg(cfg: &Self::Config) -> Self;

    fn one_with_cfg(cfg: &Self::Config) -> Self;
}

/// Prime field whose modulus is a constant value known at compile time.
pub trait ConstPrimeField:
    Field + ConstSemiring + Inv<Output = Option<Self>> + From<u64> + From<u128> + From<Self::Inner>
{
    const MODULUS: Self::Modulus;
    const MODULUS_MINUS_ONE_DIV_TWO: Self::Inner;

    /// Creates a new instance of a prime field element from
    /// an arbitrary element of `Self::Inner`. The method
    /// should not assume the `Self::Inner` is coming in a
    /// form internally used by the field type. So it
    /// always should perform a reduction first.
    fn new(inner: Self::Inner) -> Self;

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
    fn is_zero(value: &Self) -> bool {
        Zero::is_zero(value)
    }

    #[inline(always)]
    fn modulus(&self) -> Self::Modulus {
        Self::MODULUS
    }

    #[inline(always)]
    fn modulus_minus_one_div_two(&self) -> T::Inner {
        Self::MODULUS_MINUS_ONE_DIV_TWO
    }

    fn make_cfg(modulus: &Self::Modulus) -> Result<Self::Config, FieldError> {
        if *modulus == Self::MODULUS {
            Ok(())
        } else {
            Err(FieldError::InvalidModulus)
        }
    }

    #[inline(always)]
    fn new_with_cfg(inner: Self::Inner, _cfg: &Self::Config) -> Self {
        ConstPrimeField::new(inner)
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

/// The trait combines all `FromWithConfig<u*>` and `FromWithConfig<i*>` into
/// one umbrella trait. Handy when one needs conversion functions for different
/// primitive int types.
pub trait FromPrimitiveWithConfig:
    FromWithConfig<u8>
    + FromWithConfig<u16>
    + FromWithConfig<u32>
    + FromWithConfig<u64>
    + FromWithConfig<u128>
    + FromWithConfig<i8>
    + FromWithConfig<i16>
    + FromWithConfig<i32>
    + FromWithConfig<i64>
    + FromWithConfig<i128>
{
}

/// Blanket implementation.
impl<
    T: FromWithConfig<u8>
        + FromWithConfig<u16>
        + FromWithConfig<u32>
        + FromWithConfig<u64>
        + FromWithConfig<u128>
        + FromWithConfig<i8>
        + FromWithConfig<i16>
        + FromWithConfig<i32>
        + FromWithConfig<i64>
        + FromWithConfig<i128>,
> FromPrimitiveWithConfig for T
{
}

/// Analogous to `Into` trait, but with a prime field configuration parameter.
/// Preferably should not be implemented directly.
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
