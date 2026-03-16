pub mod mle;
pub mod univariate;
pub mod utils;
pub mod zero_degree;

use thiserror::Error;

/// Polynomial with coefficients of type `C` and degree bounded by
/// `DEGREE_BOUND`.
pub trait Polynomial<C>: Clone {
    const DEGREE_BOUND: usize;
}

pub trait EvaluatablePolynomial<C, Out>: Polynomial<C> {
    /// The type of points a polynomial can be evaluated on.
    /// For univariate polynomials this typically is `Out`,
    /// for multivariate this is `[Out]`.
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
    #[error("Unsupported constraint degrees: {degrees:?}")]
    UnsupportedConstraintDegrees { degrees: Vec<usize> },
}
