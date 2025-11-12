use super::{ConstCoeffBitWidth, EvaluatablePolynomial, EvaluationError};
use crate::Polynomial;
use crypto_primitives::crypto_bigint_int::Int;
use zinc_transcript::traits::ConstTranscribable;

macro_rules! impl_zero_degree {
    ($($t:ty),+) => {
        $(
            impl Polynomial<Self> for $t {
                const DEGREE_BOUND: usize = 0;
            }

            impl<T> EvaluatablePolynomial<Self, T, Self> for $t {
                fn evaluate_at_point(&self, point: &[T]) -> Result<Self, EvaluationError> {
                    if !point.is_empty() {
                        return Err(EvaluationError::WrongPointWidth {
                            expected: 0,
                            actual: point.len(),
                        });
                    }
                    Ok(self.clone())
                }
            }

            impl ConstCoeffBitWidth for $t {
                const COEFF_BIT_WIDTH: usize = <$t>::BITS as usize;
            }
        )*
    };
}

impl_zero_degree!(i8, i16, i32, i64, i128);
impl_zero_degree!(u8, u16, u32, u64, u128);

impl<const LIMBS: usize> Polynomial<Self> for Int<LIMBS> {
    const DEGREE_BOUND: usize = 0;
}

impl<T, const LIMBS: usize> EvaluatablePolynomial<Self, T, Self> for Int<LIMBS> {
    fn evaluate_at_point(&self, point: &[T]) -> Result<Self, EvaluationError> {
        if !point.is_empty() {
            return Err(EvaluationError::WrongPointWidth {
                expected: 0,
                actual: point.len(),
            });
        }
        Ok(*self)
    }
}

impl<const LIMBS: usize> ConstCoeffBitWidth for Int<LIMBS> {
    const COEFF_BIT_WIDTH: usize = Self::NUM_BITS;
}
