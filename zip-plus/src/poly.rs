pub mod dense;
pub mod mle;
mod zero_degree;

use crate::pcs::structs::MulByScalar;
use crypto_primitives::Ring;
use thiserror::Error;

pub trait Polynomial<R: Ring> {
    /// Returns the degree of the polynomial - a number of coefficients.
    fn degree(&self) -> usize;

    fn evaluate<C>(&self, point: &[C]) -> Result<R, EvaluationError>
    where
        R: MulByScalar<C>;
}

#[derive(Clone, Debug, PartialEq, Error)]
pub enum EvaluationError {
    #[error("Wrong number of points provided for evaluation: expected {expected}, got {actual}")]
    WrongPointWidth { expected: usize, actual: usize },
}
