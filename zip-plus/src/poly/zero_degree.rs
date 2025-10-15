use super::{EvaluationError, Polynomial};
use crate::pcs::structs::MulByScalar;
use crypto_primitives::Semiring;

impl<R: Semiring> Polynomial<Self> for R {
    const DEGREE_BOUND: usize = 0;

    fn map<R2: Semiring>(&self, f: impl Fn(&Self) -> R2) -> impl Polynomial<R2> {
        f(self)
    }

    fn evaluate_at_point<C>(&self, point: &[C]) -> Result<R, EvaluationError>
    where
        Self: for<'a> MulByScalar<&'a C>,
    {
        if !point.is_empty() {
            return Err(EvaluationError::WrongPointWidth {
                expected: 0,
                actual: point.len(),
            });
        }
        Ok(self.clone())
    }
}
