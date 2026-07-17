use crate::from_ref::FromRef;
use crypto_primitives::{
    Wrapper,
    boolean::Boolean,
    crypto_bigint_int::Int,
    crypto_bigint_uint::{U64, U128},
};
use num_traits::{CheckedMul, ConstZero};

// TODO: Rework the approach

/// Multiplication of an element by a scalar of a (possibly) different type,
/// performed by the `()` context on self-sufficient types: primitives,
/// [`Int`], and structures lifted over them.
///
/// This is the fixed-world widening-mul toolkit: `Rhs` is embedded into the
/// `Lhs`/`Out` ring as part of the multiplication, and `CHECK` selects
/// overflow-checked semantics. Dynamic (config-carried) field elements never
/// go through this trait — a foreign-typed scalar crosses into the config
/// world via `cfg.project` once, at the boundary, after which all config
/// arithmetic is homogeneous (`SemiringConfig::mul`/`checked_mul`).
pub trait MulByScalar<Lhs, Rhs, Out = Lhs>: Sized {
    /// Multiplies `lhs` by a scalar from the right (usually - a coefficient to
    /// obtain a linear combination).
    /// Returns `None` if the multiplication would overflow.
    fn mul_by_scalar<const CHECK: bool>(&self, lhs: Lhs, rhs: &Rhs) -> Option<Out>;
}

macro_rules! impl_mul_by_scalar_for_primitives {
    ($($t:ty),*) => {
        $(
            impl MulByScalar<$t, $t> for () {
                #[allow(clippy::arithmetic_side_effects)] // By design
                fn mul_by_scalar<const CHECK: bool>(&self, lhs: $t, rhs: &$t) -> Option<$t> {
                    if CHECK {
                        lhs.checked_mul(*rhs)
                    } else {
                        Some(lhs * *rhs)
                    }
                }
            }
        )*
    };
}

impl_mul_by_scalar_for_primitives!(i8, i16, i32, i64, i128);

impl<const LIMBS: usize, const LIMBS2: usize> MulByScalar<Int<LIMBS>, Int<LIMBS2>> for () {
    #[allow(clippy::arithmetic_side_effects)] // By design
    fn mul_by_scalar<const CHECK: bool>(
        &self,
        lhs: Int<LIMBS>,
        rhs: &Int<LIMBS2>,
    ) -> Option<Int<LIMBS>> {
        if LIMBS < LIMBS2 {
            return None; // Cannot multiply if the left operand has fewer limbs than the right
        }
        if CHECK {
            lhs.checked_mul(&rhs.resize())
        } else {
            // Make use of an optimized wrapping_mul in the crypto-bigint library.
            Some(widening_wrapping_mul(lhs, rhs))
        }
    }
}

macro_rules! impl_mul_int_by_primitive_scalar {
    ($(($t:ty, $rhs_limbs:expr)),*) => {
        $(
            impl<const LIMBS: usize, const LIMBS2: usize> MulByScalar<Int<LIMBS>, $t, Int<LIMBS2>> for () {
                #[allow(clippy::arithmetic_side_effects)] // By design
                fn mul_by_scalar<const CHECK: bool>(&self, lhs: Int<LIMBS>, rhs: &$t) -> Option<Int<LIMBS2>> {
                    const {
                        assert!(LIMBS <= LIMBS2, "Cannot multiply if the left operand has more limbs than the output");
                    }
                    if CHECK {
                        let rhs: Int<LIMBS2> = Int::from_ref(rhs);
                        rhs.checked_mul(&lhs.resize())
                    } else {
                        let rhs_short: Int<{ $rhs_limbs }> = Int::from(*rhs);
                        Some(widening_wrapping_mul(lhs.resize::<LIMBS2>(), &rhs_short))
                    }
                }
            }
        )*
    };
}

impl_mul_int_by_primitive_scalar!(
    (i8, U64::LIMBS),
    (i16, U64::LIMBS),
    (i32, U64::LIMBS),
    (i64, U64::LIMBS),
    (i128, U128::LIMBS)
);

/// Multiplication by a [`Boolean`] scalar: selects `lhs` or zero.
impl<T> MulByScalar<T, Boolean> for ()
where
    T: Clone + ConstZero + From<Boolean>,
{
    fn mul_by_scalar<const CHECK: bool>(&self, lhs: T, rhs: &Boolean) -> Option<T> {
        Some(if *rhs.inner() { lhs } else { T::ZERO })
    }
}

impl MulByScalar<i32, i64, i128> for () {
    #[inline(always)]
    #[allow(clippy::arithmetic_side_effects)] // Not possible to overflow since we are widening the result to i128
    fn mul_by_scalar<const CHECK: bool>(&self, lhs: i32, rhs: &i64) -> Option<i128> {
        Some(i128::from(lhs) * i128::from(*rhs))
    }
}

impl MulByScalar<i128, i64, i128> for () {
    #[inline(always)]
    #[allow(clippy::arithmetic_side_effects)] // By design
    fn mul_by_scalar<const CHECK: bool>(&self, lhs: i128, rhs: &i64) -> Option<i128> {
        let rhs = i128::from(*rhs);
        if CHECK {
            lhs.checked_mul(rhs)
        } else {
            Some(lhs * rhs)
        }
    }
}

impl MulByScalar<i64, i64, i128> for () {
    #[inline(always)]
    #[allow(clippy::arithmetic_side_effects)] // Not possible to overflow since we are widening the result to i128
    fn mul_by_scalar<const CHECK: bool>(&self, lhs: i64, rhs: &i64) -> Option<i128> {
        Some(i128::from(lhs) * i128::from(*rhs))
    }
}

/// Helper function, make use of the crypto-bigint inner workings in order to
/// multiply two ints of different number of limbs in `O(LIMBS_1 * LIMBS_2)`
/// rather than in `O(MAX_LIMBS^2)` time.
fn widening_wrapping_mul<const LIMBS: usize, const LIMBS2: usize>(
    lhs: Int<LIMBS>,
    rhs: &Int<LIMBS2>,
) -> Int<LIMBS> {
    Int::new(lhs.into_inner().wrapping_mul(rhs.inner()))
}
