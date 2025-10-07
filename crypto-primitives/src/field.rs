#[cfg(feature = "ark_ff")]
pub mod ark_ff_field;
#[cfg(feature = "ark_ff")]
pub mod ark_ff_fp;
#[cfg(feature = "crypto_bigint")]
pub mod crypto_bigint_const_monty;

use crate::{IntRing, ring::Ring};
use core::{
    fmt::Debug,
    ops::{Div, DivAssign, Neg},
};
use num_traits::Inv;

/// Element of a field (F) - a group where addition and multiplication are
/// defined with their respective inverse operations.
pub trait Field:
    Ring
    + Neg<Output=Self>
    // Arithmetic operations consuming rhs
    + Div<Output=Self>
    + DivAssign
    // Arithmetic operations with rhs reference
    + for<'a> Div<&'a Self, Output=Self>
    + for<'a> DivAssign<&'a Self>
    {}

/// Element of an integer field modulo prime number (F_p).
// TODO: FROM<uX>
pub trait PrimeField:
    Field + IntRing + From<u64> + From<u128> + From<Self::Inner> + Inv<Output = Option<Self>>
{
    /// Underlying representation of an element
    type Inner: Debug;

    const MODULUS: Self::Inner;

    fn new_unchecked(inner: Self::Inner) -> Self;

    fn inner(&self) -> &Self::Inner;
}

/// Element of a prime field in its Montgomery representation of - encoded in a
/// way so that modular multiplication can be done without performing an
/// explicit division by pp after each product.
pub trait MontgomeryField: PrimeField {
    // FIXME

    /// INV = -MODULUS^{-1} mod R
    const INV: Self::Inner;
}
