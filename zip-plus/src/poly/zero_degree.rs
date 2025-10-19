use super::{ConstCoeffBitWidth, EvaluatablePolynomial, EvaluationError, Polynomial};
use crate::traits::ConstTranscribable;
use crypto_primitives::{Semiring, crypto_bigint_int::Int};

macro_rules! impl_zero_degree {
    ($($t:ty),+) => {
        $(
            impl Polynomial<0> for $t {}

            impl<T> EvaluatablePolynomial<T> for $t {
                type Output = Self;

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
        )*
    };
}

impl_zero_degree!(i8, i16, i32, i64, i128);
impl_zero_degree!(u8, u16, u32, u64, u128);

impl<const LIMBS: usize> Polynomial<0> for Int<LIMBS> {}

impl<T, const LIMBS: usize> EvaluatablePolynomial<T> for Int<LIMBS> {
    type Output = Self;

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

impl<R: Semiring + ConstTranscribable> ConstCoeffBitWidth for R {
    const COEFF_BIT_WIDTH: usize = R::NUM_BITS;
}
