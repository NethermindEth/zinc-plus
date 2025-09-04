use crate::{IntRing, Limb, ring::Ring};
use core::ops::Div;
use num_traits::Inv;

/// Element of a field (F) - a group where addition and multiplication are
/// defined with their respective inverse operations.
pub trait Field:
Ring
    // Arithmetic operations consuming rhs
    + Div<Self, Output=Self>
    // Arithmetic operations with rhs reference
    + for<'a> Div<&'a Self, Output=Self>
    {}

/// Element of an integer field modulo prime number (F_p).
pub trait PrimeField: Field + IntRing + for<'a> From<&'a [Limb]> + Inv {
    /// Underlying representation of an element
    type Inner: IntRing;

    const MODULUS: Self::Inner;

    /// INV = -MODULUS^{-1} mod 2^64
    const INV: Self::Inner;

    fn from_u128(value: u32) -> Self;
}

/// Element of a prime field in its Montgomery representation of - encoded in a
/// way so that modular multiplication can be done without performing an
/// explicit division by pp after each product.
pub trait MontgomeryField: PrimeField {
    // FIXME
}
