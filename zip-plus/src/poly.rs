pub mod dense;
pub mod mle;
pub mod zero_degree;

use thiserror::Error;

/// * `DEGREE_BOUND`: The (max) degree of the polynomial - one less than a
///   number of coefficients.
pub trait Polynomial<const DEGREE_BOUND: usize> {}

pub trait EvaluatablePolynomial<S> {
    type Output;

    /// Evaluates the polynomial at the given point, treating point `[p_0, p_1,
    /// p_2, ...]` as `[x, x^2, x^3, ..., x^DEGREE_BOUND]`, thus it returns
    /// `a_0 + (a_1 * p_0) + (a_2 * p_1) + ... + (a_DEGREE_BOUND *
    /// p_{DEGREE_BOUND - 1})`.
    // Note: we can't reference Self::DEGREE_BOUND here, type parameters may not be
    // used in const expressions. As such, no point in declaring Self: Polynomial.
    fn evaluate_at_point(&self, point: &[S]) -> Result<Self::Output, EvaluationError>;
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
}
