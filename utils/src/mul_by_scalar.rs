use crate::from_ref::FromRef;
use crypto_primitives::{boolean::Boolean, crypto_bigint_int::Int};
use num_traits::{CheckedMul, ConstZero};

pub trait MulByScalar<Rhs>: Sized {
    /// Multiplies the current element by a scalar from the right (usually - a
    /// coefficient to obtain a linear combination).
    /// Returns `None` if the multiplication would overflow.
    fn mul_by_scalar<const CHECK: bool>(&self, rhs: Rhs) -> Option<Self>;

    /// Fused multiply-scalar-and-add: `*acc += self * rhs`.
    ///
    /// The default implementation creates a temporary and adds it.
    /// Specialized implementations (e.g. `DensePolynomial`) can fuse the
    /// operation to avoid the intermediate allocation.
    #[inline(always)]
    fn mul_by_scalar_and_add_to<const CHECK: bool>(
        &self,
        rhs: Rhs,
        acc: &mut Self,
    ) where
        Self: for<'a> std::ops::AddAssign<&'a Self>,
    {
        if let Some(term) = self.mul_by_scalar::<CHECK>(rhs) {
            *acc += &term;
        }
    }
}

macro_rules! impl_mul_by_scalar_for_primitives {
    ($($t:ty),*) => {
        $(
            impl MulByScalar<&$t> for $t {
                #[inline(always)]
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

    /// Fused widen-and-add: equivalent to `*acc += mul_by_scalar_widen(lhs, rhs)`
    /// but can be overridden for better performance (e.g., avoiding temporary allocation).
    fn widen_and_add(acc: &mut Self::Output, lhs: &Lhs, rhs: &Rhs)
    where
        Self::Output: for<'a> std::ops::AddAssign<&'a Self::Output>,
    {
        let term = Self::mul_by_scalar_widen(lhs, rhs);
        *acc += &term;
    }
}
