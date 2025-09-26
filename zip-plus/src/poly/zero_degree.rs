use super::{EvaluationError, Polynomial};
use crate::pcs::structs::MulByScalar;
use crypto_primitives::crypto_bigint_int::Int;

impl<const LIMBS: usize> Polynomial<Self> for Int<LIMBS> {
    const DEGREE_BOUND: usize = 0;

    fn evaluate<C>(&self, point: &[C]) -> Result<Int<LIMBS>, EvaluationError>
    where
        Self: for<'a> MulByScalar<&'a C>,
    {
        if !point.is_empty() {
            return Err(EvaluationError::WrongPointWidth {
                expected: 0,
                actual: point.len(),
            });
        }
        Ok(*self)
    }
}

macro_rules! impl_zero_degree_for_primitives {
    ($($t:ty),*) => {
        $(
            impl Polynomial<$t> for $t {
                const DEGREE_BOUND: usize = 0;

                fn evaluate<C>(&self, point: &[C]) -> Result<$t, EvaluationError>
                where
                    Self: for<'a> MulByScalar<&'a C>,
                {
                    if !point.is_empty() {
                        return Err(EvaluationError::WrongPointWidth {
                            expected: 0,
                            actual: point.len(),
                        });
                    }
                    Ok(*self)
                }
            }
        )*
    };
}

impl_zero_degree_for_primitives!(i8, i16, i32, i64, i128);
