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
            // Make use of an optimized wrapping_mul in the crypto-bigint library.
            Some(widening_wrapping_mul(self, rhs))
        }
    }
}

macro_rules! impl_mul_int_by_primitive_scalar {
    ($(($t:ty, $rhs_limbs:expr)),*) => {
        $(
            impl<const LIMBS: usize, const LIMBS2: usize> MulByScalar<&$t, Int<LIMBS2>> for Int<LIMBS> {
                #[allow(clippy::arithmetic_side_effects)] // By design
                fn mul_by_scalar<const CHECK: bool>(&self, rhs: &$t) -> Option<Int<LIMBS2>> {
                    const {
                        assert!(LIMBS <= LIMBS2, "Cannot multiply if the left operand has more limbs than the output");
                    }
                    if CHECK {
                        let rhs: Int<LIMBS2> = Int::from_ref(rhs);
                        rhs.checked_mul(&self.resize())
                    } else {
                        let rhs_short: Int<{ $rhs_limbs }> = Int::from(*rhs);
                        Some(widening_wrapping_mul(&self.resize::<LIMBS2>(), &rhs_short))
                    }
                }
            }
        )*
    };
}

impl_mul_int_by_primitive_scalar!(
    (i8, crypto_bigint::U64::LIMBS),
    (i16, crypto_bigint::U64::LIMBS),
    (i32, crypto_bigint::U64::LIMBS),
    (i64, crypto_bigint::U64::LIMBS),
    (i128, crypto_bigint::U128::LIMBS)
);

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

/// Helper function, make use of the crypto-bigint inner workings in order to
/// multiply two ints of different number of limbs in `O(LIMBS_1 * LIMBS_2)`
/// rather than in `O(MAX_LIMBS^2)` time.
fn widening_wrapping_mul<const LIMBS: usize, const LIMBS2: usize>(
    lhs: &Int<LIMBS>,
    rhs: &Int<LIMBS2>,
) -> Int<LIMBS> {
    Int::new(lhs.inner().wrapping_mul(rhs.inner()))
}
