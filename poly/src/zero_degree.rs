use super::{ConstCoeffBitWidth, EvaluatablePolynomial, EvaluationError};
use crate::{CoefficientProjectable, Polynomial, univariate::dense::DensePolynomial};
use crypto_primitives::crypto_bigint_int::Int;
use zinc_transcript::traits::ConstTranscribable;

macro_rules! impl_zero_degree {
    ($($t:ty),+) => {
        $(
            impl Polynomial<Self> for $t {
                const DEGREE_BOUND: usize = 0;
            }

            impl EvaluatablePolynomial<Self, Self> for $t {
                type EvaluationPoint = Self;

                fn evaluate_at_point(&self, _point: &Self) -> Result<Self, EvaluationError> {
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

impl<const LIMBS: usize> EvaluatablePolynomial<Self, Self> for Int<LIMBS> {
    type EvaluationPoint = Self;

    fn evaluate_at_point(&self, _point: &Self) -> Result<Self, EvaluationError> {
        Ok(*self)
    }
}

impl<const LIMBS: usize> ConstCoeffBitWidth for Int<LIMBS> {
    const COEFF_BIT_WIDTH: usize = Self::NUM_BITS;
}

impl<const LIMBS: usize> CoefficientProjectable<Int<LIMBS>, 1> for Int<LIMBS> {
    fn project_coefficients<F: crypto_primitives::FromWithConfig<Int<LIMBS>> + 'static>(
        self,
        projecting_element: &F,
    ) -> DensePolynomial<F, 1> {
        DensePolynomial::new_with_zero(
            [F::from_with_cfg(self, projecting_element.cfg())],
            F::zero_with_cfg(projecting_element.cfg()),
        )
    }
}
