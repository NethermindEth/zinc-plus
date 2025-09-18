#![no_std]
extern crate alloc;

pub mod field;
pub mod matrix;
pub mod ring;

pub use field::*;
pub use matrix::*;
pub use ring::*;

pub type Limb = u64;

// FIXME
/// Implements CheckedNeg, CheckedAdd, CheckedSub, CheckedMul, CheckedShl, and
/// CheckedShr traits for the given type under the assumption that the type
/// implements the corresponding operations without overflow.
#[macro_export]
macro_rules! trivial_checked_ops {
    (
        $(<
            $(
                $(const )?$lt:tt $( : $clt:tt $(+ $dlt:tt )* )?
            ),+
        >)?
    ) => {
        impl$(< $( $lt $( : $clt $(+ $dlt )* )? ),+ >)? num_traits::CheckedNeg for Qqwe { //$t {
            fn checked_neg(self) -> Option<Self> {
                Some(-self)
            }
        }

        /*impl num_traits::CheckedAdd for $t {
            fn checked_add(self, other: Self) -> Option<Self> {
                Some(self + other)
            }
        }

        impl num_traits::CheckedSub for $t {
            fn checked_sub(self, other: Self) -> Option<Self> {
                Some(self - other)
            }
        }

        impl num_traits::CheckedMul for $t {
            fn checked_mul(self, other: Self) -> Option<Self> {
                Some(self * other)
            }
        }

        impl num_traits::CheckedShl for $t {
            fn checked_shl(self, shift: u32) -> Option<Self> {
                Some(self << shift)
            }
        }

        impl num_traits::CheckedShr for $t {
            fn checked_shr(self, shift: u32) -> Option<Self> {
                Some(self >> shift)
            }
        }*/
    };
}

// trivial_checked_ops!(<A, B, C>);
// trivial_checked_ops!(<Mod: ConstMontyParams, const LIMBS: usize>);
