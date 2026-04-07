use crate::from_ref::FromRef;
use crypto_primitives::{boolean::Boolean, crypto_bigint_int::Int};
use num_traits::{CheckedMul, ConstZero};

pub trait MulByScalar<Rhs, Out = Self>: Sized {
    /// Multiplies the current element by a scalar from the right (usually - a
    /// coefficient to obtain a linear combination).
    /// Returns `None` if the multiplication would overflow.
    fn mul_by_scalar<const CHECK: bool>(&self, rhs: Rhs) -> Option<Out>;
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
            impl<const LIMBS: usize, const LIMBS2: usize> MulByScalar<&$t, Int<LIMBS2>> for Int<LIMBS> {
                #[allow(clippy::arithmetic_side_effects)] // By design
                fn mul_by_scalar<const CHECK: bool>(&self, rhs: &$t) -> Option<Int<LIMBS2>> {
                    const {
                        assert!(LIMBS <= LIMBS2, "Cannot multiply if the left operand has more limbs than the output");
                    }
                    let rhs: Int<LIMBS2> = Int::from_ref(rhs);
                    if CHECK {
                        rhs.checked_mul(&self.resize())
                    } else {
                        Some(rhs * self.resize())
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

impl MulByScalar<&i64, i128> for i32 {
    #[inline(always)]
    #[allow(clippy::arithmetic_side_effects)] // Not possible to overflow since we are widening the result to i128
    fn mul_by_scalar<const CHECK: bool>(&self, rhs: &i64) -> Option<i128> {
        Some(i128::from(*self) * i128::from(*rhs))
    }
}

impl MulByScalar<&i64> for i128 {
    #[inline(always)]
    #[allow(clippy::arithmetic_side_effects)] // By design
    fn mul_by_scalar<const CHECK: bool>(&self, rhs: &i64) -> Option<i128> {
        let rhs = i128::from(*rhs);
        if CHECK {
            self.checked_mul(&rhs)
        } else {
            Some(self * rhs)
        }
    }
}

impl MulByScalar<&i64, i128> for i64 {
    #[inline(always)]
    #[allow(clippy::arithmetic_side_effects)] // Not possible to overflow since we are widening the result to i128
    fn mul_by_scalar<const CHECK: bool>(&self, rhs: &i64) -> Option<i128> {
        Some(i128::from(*self) * i128::from(*rhs))
    }
}
