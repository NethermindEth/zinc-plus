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

    fn map<R2: Ring>(&self, f: impl Fn(&R) -> R2) -> impl Polynomial<R2>;

    /// Evaluates the polynomial at the given point, treating point `[p_0, p_1,
    /// p_2, ...]` as `[x, x^2, x^3, ..., x^DEGREE_BOUND]`, thus it returns
    /// `a_0 + (a_1 * p_0) + (a_2 * p_1) + ... + (a_DEGREE_BOUND *
    /// p_{DEGREE_BOUND - 1})`.
    // Note: we can't reference Self::DEGREE_BOUND here type parameters may not be
    // used in const expressions
    fn evaluate_at_point<C>(&self, point: &[C]) -> Result<R, EvaluationError>
    where
        R: for<'a> MulByScalar<&'a C>;
}

#[derive(Clone, Debug, PartialEq, Error)]
pub enum EvaluationError {
    #[error("Wrong number of points provided for evaluation: expected {expected}, got {actual}")]
    WrongPointWidth { expected: usize, actual: usize },
    #[error("Evaluation failed due to overflow")]
    Overflow,
}
