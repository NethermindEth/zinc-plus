use crate::ring::Ring;
use crate::{IntRing, Limb};
use std::ops::Div;

/// A field (F) - a group where addition and multiplication are defined with their respective
/// inverse operations.
pub trait Field:
Ring
    // Arithmetic operations consuming rhs
    + Div<Self, Output=Self>
    // Arithmetic operations with rhs reference
    + for<'a> Div<&'a Self, Output=Self>
    {}

/// Integer field modulo prime number.
pub trait PrimeField: Field + IntRing + for<'a> From<&'a [Limb]> {
    type Inner: IntRing;
}
