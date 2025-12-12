use crate::from_ref::FromRef;
use crypto_primitives::{boolean::Boolean, crypto_bigint_int::Int};
use num_traits::{CheckedMul, ConstZero};

pub trait MulByScalar<Rhs>: Sized {
    /// Multiplies the current element by a scalar from the right (usually - a
    /// coefficient to obtain a linear combination).
    /// Returns `None` if the multiplication would overflow.
    fn mul_by_scalar(&self, rhs: Rhs) -> Option<Self>;
}

macro_rules! impl_mul_by_scalar_for_primitives {
    ($($t:ty),*) => {
        $(
            impl MulByScalar<&$t> for $t {
                fn mul_by_scalar(&self, rhs: &$t) -> Option<Self> {
                    self.checked_mul(rhs)
                }
            }
        )*
    };
}

impl_mul_by_scalar_for_primitives!(i8, i16, i32, i64, i128);

impl<const LIMBS: usize, const LIMBS2: usize> MulByScalar<&Int<LIMBS2>> for Int<LIMBS> {
    fn mul_by_scalar(&self, rhs: &Int<LIMBS2>) -> Option<Self> {
        if LIMBS < LIMBS2 {
            return None; // Cannot multiply if the left operand has fewer limbs than the right
        }
        self.checked_mul(&rhs.resize())
    }
}

macro_rules! impl_mul_int_by_primitive_scalar {
    ($($t:ty),*) => {
        $(
            impl<const LIMBS: usize> MulByScalar<&$t> for Int<LIMBS> {
                fn mul_by_scalar(&self, rhs: &$t) -> Option<Self> {
                    self.checked_mul(&Self::from_ref(rhs))
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
    fn mul_by_scalar(&self, rhs: &Boolean) -> Option<Self> {
        Some(if rhs.into_inner() {
            self.clone()
        } else {
            ConstZero::ZERO
        })
    }
}

impl MulByScalar<&i64> for i128 {
    #[inline(always)]
    fn mul_by_scalar(&self, rhs: &i64) -> Option<Self> {
        self.checked_mul(&i128::from(*rhs))
    }
}
