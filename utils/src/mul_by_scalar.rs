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

impl_mul_int_by_primitive_scalar!(i8, i16, i32, i128);

/// Optimized `MulByScalar<&i64>` for `Int<LIMBS>`.
///
/// The unchecked path uses a single-limb schoolbook multiply (LIMBS limb-muls
/// instead of LIMBS² for the generic N×N case).  This is ~LIMBS× faster for
/// the dominant PCS verifier hot path where IPRS encoding-matrix entries
/// (PnttInt = i64) are multiplied by multi-limb CombR integers.
///
/// Two's complement correctness: treating `i64` as `u64` gives the right
/// low LIMBS words, *except* we must subtract `self << 64` when the scalar
/// is negative (because `(-s) as u64 == 2^64 - s`, and the extra `self * 2^64`
/// leaks into words 1..LIMBS).
impl<const LIMBS: usize> MulByScalar<&i64> for Int<LIMBS> {
    #[allow(clippy::arithmetic_side_effects)]
    #[inline]
    fn mul_by_scalar<const CHECK: bool>(&self, rhs: &i64) -> Option<Self> {
        if CHECK {
            let rhs_wide = Self::from_ref(rhs);
            return self.checked_mul(&rhs_wide);
        }

        // Fast path: LIMBS×1 schoolbook multiply
        let s = *rhs as u64; // two's complement reinterpretation
        let words: &[u64; LIMBS] = self.inner().as_words();
        let mut result = [0u64; LIMBS];
        let mut carry: u128 = 0;

        for i in 0..LIMBS {
            let wide = (words[i] as u128) * (s as u128) + carry;
            result[i] = wide as u64;
            carry = wide >> 64;
        }

        // Sign correction: if rhs < 0, the u64 representation is (2^64 + rhs),
        // so we computed self * (2^64 + rhs) = self * rhs + self * 2^64.
        // Subtract the spurious self << 64 from the result (wrapping).
        if *rhs < 0 {
            let mut borrow: u64 = 0;
            for i in 1..LIMBS {
                let (d1, b1) = result[i].overflowing_sub(words[i - 1]);
                let (d2, b2) = d1.overflowing_sub(borrow);
                result[i] = d2;
                borrow = (b1 as u64) + (b2 as u64);
            }
        }

        Some(Self::from_words(result))
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify the optimized single-limb multiply for `Int<N> * i64` matches
    /// the naive full-width multiply against known values.
    #[test]
    fn mul_by_i64_correctness() {
        // Positive × positive
        let a = Int::<4>::from_i64(123_456);
        let s: i64 = 789;
        let result = a.mul_by_scalar::<false>(&s).unwrap();
        let expected = Int::<4>::from_i64(123_456 * 789);
        assert_eq!(result, expected, "positive × positive");

        // Positive × negative
        let s: i64 = -42;
        let result = a.mul_by_scalar::<false>(&s).unwrap();
        let expected = Int::<4>::from_i64(123_456 * -42);
        assert_eq!(result, expected, "positive × negative");

        // Negative × positive
        let a = Int::<4>::from_i64(-99_999);
        let s: i64 = 31337;
        let result = a.mul_by_scalar::<false>(&s).unwrap();
        let expected = Int::<4>::from_i64(-99_999i64 * 31337);
        assert_eq!(result, expected, "negative × positive");

        // Negative × negative
        let a = Int::<4>::from_i64(-500);
        let s: i64 = -300;
        let result = a.mul_by_scalar::<false>(&s).unwrap();
        let expected = Int::<4>::from_i64(150_000);
        assert_eq!(result, expected, "negative × negative");

        // Zero cases
        let zero = Int::<4>::from_i64(0);
        assert_eq!(
            zero.mul_by_scalar::<false>(&42i64).unwrap(),
            zero,
            "zero × positive"
        );
        assert_eq!(
            a.mul_by_scalar::<false>(&0i64).unwrap(),
            zero,
            "nonzero × zero"
        );

        // Large CombR × typical twiddle — matches the PnttInt range [-32768, 32768]
        let large = Int::<6>::from_i64(i64::MAX);
        let twiddle: i64 = -32768;
        let naive = {
            let rhs = Int::<6>::from_ref(&twiddle);
            large * rhs
        };
        let fast = large.mul_by_scalar::<false>(&twiddle).unwrap();
        assert_eq!(fast, naive, "large Int<6> × negative twiddle");

        let twiddle: i64 = 32768;
        let naive = {
            let rhs = Int::<6>::from_ref(&twiddle);
            large * rhs
        };
        let fast = large.mul_by_scalar::<false>(&twiddle).unwrap();
        assert_eq!(fast, naive, "large Int<6> × positive twiddle");
    }

    /// Edge cases: min/max i64 scalars
    #[test]
    fn mul_by_i64_edge_cases() {
        let a = Int::<4>::from_i64(1);
        let result = a.mul_by_scalar::<false>(&i64::MAX).unwrap();
        assert_eq!(result, Int::<4>::from_i64(i64::MAX));

        let result = a.mul_by_scalar::<false>(&i64::MIN).unwrap();
        assert_eq!(result, Int::<4>::from_i64(i64::MIN));

        // i64::MIN × -1 wraps in i64 but should work for Int<4>
        let neg1 = Int::<4>::from_i64(-1);
        let result = neg1.mul_by_scalar::<false>(&i64::MIN).unwrap();
        // -1 * i64::MIN = i64::MAX + 1 = 2^63
        let expected = {
            let rhs = Int::<4>::from_ref(&i64::MIN);
            neg1 * rhs
        };
        assert_eq!(result, expected, "-1 × i64::MIN");
    }

    /// Stress-test: compare optimized path against naive for many random-ish values
    #[test]
    fn mul_by_i64_matches_naive() {
        // Test various limb sizes
        for scalar in [1i64, -1, 32768, -32768, i64::MAX, i64::MIN, 0, 42, -99] {
            // Int<2>
            let a = Int::<2>::from_i64(0x0123_4567_89AB_CDEFi64);
            let fast = a.mul_by_scalar::<false>(&scalar).unwrap();
            let naive = a * Int::<2>::from_ref(&scalar);
            assert_eq!(fast, naive, "Int<2> × {scalar}");

            // Int<6>
            let a = Int::<6>::from_i64(0x0123_4567_89AB_CDEFi64);
            let fast = a.mul_by_scalar::<false>(&scalar).unwrap();
            let naive = a * Int::<6>::from_ref(&scalar);
            assert_eq!(fast, naive, "Int<6> × {scalar}");

            // Int<8>
            let a = Int::<8>::from_i64(0x0123_4567_89AB_CDEFi64);
            let fast = a.mul_by_scalar::<false>(&scalar).unwrap();
            let naive = a * Int::<8>::from_ref(&scalar);
            assert_eq!(fast, naive, "Int<8> × {scalar}");
        }
    }
}
