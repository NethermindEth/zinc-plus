use super::{ConstCoeffBitWidth, EvaluatablePolynomial, EvaluationError};
use crate::{Polynomial, univariate::dense::DensePolynomial};
use crypto_primitives::{Semiring, crypto_bigint_int::Int};
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

// Treat a `DensePolynomial<R, D>` as a *scalar* (degree-0 polynomial of itself)
// for use as a `Comb` type in `ZipTypes` configurations that skip alpha-projection.
// Disjoint from the `Polynomial<R> for DensePolynomial<R, D>` impl at
// `univariate/dense.rs`, since `R != DensePolynomial<R, D>` for any well-formed
// `R: Semiring`.
impl<R: Semiring, const DEGREE_PLUS_ONE: usize> Polynomial<Self>
    for DensePolynomial<R, DEGREE_PLUS_ONE>
{
    const DEGREE_BOUND: usize = 0;
}

impl<R: Semiring, const DEGREE_PLUS_ONE: usize> EvaluatablePolynomial<Self, Self>
    for DensePolynomial<R, DEGREE_PLUS_ONE>
{
    type EvaluationPoint = Self;

    fn evaluate_at_point(&self, _point: &Self) -> Result<Self, EvaluationError> {
        Ok(self.clone())
    }
}
