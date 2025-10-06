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
use pastey::paste;

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

/// Macro to implement infallible checked operations for a field trait.
#[macro_export]
macro_rules! impl_infallible_checked_unary_op {
    // Simple case (works when there are no commas that macro parsing  will treat as top-level separators inside the generics)
    ($name:ident $(< $( $lt:ident $( : $clt:tt $(+ $dlt:tt )* )? ),+ >)?, $trait:ident, $op:ident) => {
        paste! {
            impl $(< $( $lt $( : $clt $(+ $dlt )* )? ),+ >)? $trait for $name $(< $( $lt ),+ >)? {
                fn [<checked_ $op>](&self) -> Option<Self> {
                    Some(self.$op())
                }
            }
        }
    };

    // grouped explicit form for complex generics that contain commas / nested <>
    //
    // Usage:
    // impl_infallible_checked_unary_op!(
    //     (<T: Trait<N>, const N: usize>),
    //     (MyStruct<T, const N>),
    //     CheckedNeg,
    //     neg
    // );
    (($($type_inst:tt)+), ($($impl_gens:tt)+), $trait:ident, $op:ident) => {
        paste! {
            impl $($impl_gens)+ $trait for $($type_inst)+ {
                fn [<checked_ $op>](&self) -> Option<Self> {
                    Some(self.$op())
                }
            }
        }
    };
}

/// Macro to implement infallible checked operations for a field trait.
#[macro_export]
macro_rules! impl_infallible_checked_binary_op {
    // Simple case (works when there are no commas that macro parsing  will treat as top-level separators inside the generics)
    ($name:ident $(< $( $lt:ident $( : $clt:tt $(+ $dlt:tt )* )? ),+ >)?, $trait:ident, $op:ident) => {
        paste! {
            impl $(< $( $lt $( : $clt $(+ $dlt )* )? ),+ >)? $trait for $name $(< $( $lt ),+ >)? {
                fn [<checked_ $op>](&self, rhs: &Self) -> Option<Self> {
                    Some(self.$op(rhs))
                }
            }
        }
    };

    // grouped explicit form for complex generics that contain commas / nested <>
    //
    // Usage:
    // impl_infallible_checked_unary_op!(
    //     (<T: Trait<N>, const N: usize>),
    //     (MyStruct<T, const N>),
    //     CheckedNeg,
    //     neg
    // );
    (($($type_inst:tt)+), ($($impl_gens:tt)+), $trait:ident, $op:ident) => {
        paste! {
            impl $($impl_gens)+ $trait for $($type_inst)+ {
                fn [<checked_ $op>](&self, rhs: &Self) -> Option<Self> {
                    Some(self.$op(rhs))
                }
            }
        }
    };
}
