pub mod dense;
pub mod mle;
mod zero_degree;

use crate::pcs::structs::MulByScalar;
use crypto_primitives::Ring;
use thiserror::Error;

pub trait Polynomial<R: Ring> {
    /// The (max) degree of the polynomial - one less than a number of
    /// coefficients.
    const DEGREE_BOUND: usize;

    // Note: we can't reference Self::DEGREE_BOUND here type parameters may not be
    // used in const expressions
    fn evaluate<C>(&self, point: &[C]) -> Result<R, EvaluationError>
    where
        R: for<'a> MulByScalar<&'a C>;
}

#[derive(Clone, Debug, PartialEq, Error)]
pub enum EvaluationError {
    #[error("Wrong number of points provided for evaluation: expected {expected}, got {actual}")]
    WrongPointWidth { expected: usize, actual: usize },
}
