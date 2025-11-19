pub mod mle;
pub mod univariate;
pub mod utils;
pub mod zero_degree;

use thiserror::Error;

/// Polynomial with coefficients of type `C` and degree bounded by
/// `DEGREE_BOUND`.
pub trait Polynomial<C> {
    const DEGREE_BOUND: usize;
}

pub trait EvaluatablePolynomial<C, S, Out>: Polynomial<C> {
    /// The type of points a polynomial can be evaluated on.
    /// For univariate polynomials this typically is `C`,
    /// for multivariate this is `[C]`.
    type EvaluationPoint: ?Sized;

    /// Evaluates the polynomial at the given point.
    fn evaluate_at_point(&self, point: &Self::EvaluationPoint) -> Result<Out, EvaluationError>;
}

pub trait ConstCoeffBitWidth {
    const COEFF_BIT_WIDTH: usize;
}

#[derive(Clone, Debug, PartialEq, Error)]
pub enum EvaluationError {
    #[error("Wrong number of points provided for evaluation: expected {expected}, got {actual}")]
    WrongPointWidth { expected: usize, actual: usize },
    #[error("Evaluation failed due to overflow")]
    Overflow,
    #[error("Empty polynomials are not allowed to be evaluate")]
    EmptyPolynomial,
}
