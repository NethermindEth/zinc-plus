use std::marker::PhantomData;

use crate::from_ref::FromRef;
use crypto_primitives::{boolean::Boolean, crypto_bigint_int::Int};
use num_traits::{CheckedMul, ConstZero};

pub trait MulByScalar<Rhs>: Sized {
    /// Multiplies the current element by a scalar from the right (usually - a
    /// coefficient to obtain a linear combination).
    /// Returns `None` if the multiplication would overflow.
    fn mul_by_scalar<const CHECK: bool>(&self, rhs: Rhs) -> Option<Self>;
}

macro_rules! impl_mul_by_scalar_for_primitives {
    ($($t:ty),*) => {
        $(
            impl MulByScalar<&$t> for $t {
                #[allow(clippy::arithmetic_side_effects)] // By design
                fn mul_by_scalar<const CHECK: bool>(&self, rhs: &$t) -> Option<Self> {
                    if CHECK {
                        self.checked_mul(rhs)
                    } else {
                        Some(self * rhs)
                    }
                }
            }
        )*
    };
}

impl_mul_by_scalar_for_primitives!(i8, i16, i32, i64, i128);

impl<const LIMBS: usize, const LIMBS2: usize> MulByScalar<&Int<LIMBS2>> for Int<LIMBS> {
    #[allow(clippy::arithmetic_side_effects)] // By design
    fn mul_by_scalar<const CHECK: bool>(&self, rhs: &Int<LIMBS2>) -> Option<Self> {
        if LIMBS < LIMBS2 {
            return None; // Cannot multiply if the left operand has fewer limbs than the right
        }
        if CHECK {
            self.checked_mul(&rhs.resize())
        } else {
            Some(*self * rhs.resize())
        }
    }
}

macro_rules! impl_mul_int_by_primitive_scalar {
    ($($t:ty),*) => {
        $(
            impl<const LIMBS: usize> MulByScalar<&$t> for Int<LIMBS> {
                #[allow(clippy::arithmetic_side_effects)] // By design
                fn mul_by_scalar<const CHECK: bool>(&self, rhs: &$t) -> Option<Self> {
                    let rhs = Self::from_ref(rhs);
                    if CHECK {
                        self.checked_mul(&rhs)
                    } else {
                        Some(*self * rhs)
                    }
                }
            }
        )*
    };
}

impl_mul_int_by_primitive_scalar!(i8, i16, i32, i64, i128);

impl<T> MulByScalar<&Boolean> for T
where
    T: Clone + ConstZero + From<Boolean>,
{
    fn mul_by_scalar<const CHECK: bool>(&self, rhs: &Boolean) -> Option<Self> {
        Some(if rhs.into_inner() {
            self.clone()
        } else {
            ConstZero::ZERO
        })
    }
}

impl MulByScalar<&i64> for i128 {
    #[inline(always)]
    #[allow(clippy::arithmetic_side_effects)] // By design
    fn mul_by_scalar<const CHECK: bool>(&self, rhs: &i64) -> Option<Self> {
        let rhs = i128::from(*rhs);
        if CHECK {
            self.checked_mul(&rhs)
        } else {
            Some(self * rhs)
        }
    }
}

pub trait WideningMulByScalar<Lhs, Rhs>: Clone + Default + Send + Sync {
    type Output;

    fn mul_by_scalar_widen(lhs: &Lhs, rhs: &Rhs) -> Self::Output;
}

/// Widening scalar multiplication: widens the left operand into `Output` via
/// [`FromRef`], then multiplies by the right operand in the wider type.
///
/// Use this as the `MT` parameter of [`IprsCode`] for scalar (non-polynomial)
/// evaluation types such as `i64` or `Int<N>`.
#[derive(Clone, Copy, Default)]
pub struct ScalarWideningMulByScalar<Output>(PhantomData<Output>);

impl<Lhs, Rhs, Output> WideningMulByScalar<Lhs, Rhs> for ScalarWideningMulByScalar<Output>
where
    Output: FromRef<Lhs> + for<'a> MulByScalar<&'a Rhs> + Clone + Default + Send + Sync,
{
    type Output = Output;

    #[inline(always)]
    fn mul_by_scalar_widen(lhs: &Lhs, rhs: &Rhs) -> Output {
        Output::from_ref(lhs)
            .mul_by_scalar::<false>(rhs)
            .expect("ScalarWideningMulByScalar: overflow in widening multiplication")
    }
}
