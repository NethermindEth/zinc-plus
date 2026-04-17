pub mod rotation;

use crypto_primitives::Semiring;
use std::fmt::Debug;
use thiserror::Error;
use zinc_utils::from_ref::FromRef;

/// A trait for types describing ideals.
pub trait Ideal: FromRef<Self> + Clone + Debug + Send + Sync {}

/// A trait for ideals that implement
/// membership check for an algebraic structure
/// `T`.
pub trait IdealCheck<T> {
    /// Returns true if an element of the type
    /// belongs to this ideal.
    fn contains(&self, value: &T) -> Result<bool, IdealCheckError>;
}

/// A dummy ideal. Convenient when ideal checks
/// have to be ignored.
#[derive(Clone, Copy, Debug)]
pub struct ImpossibleIdeal;

impl Ideal for ImpossibleIdeal {}

impl<R: Semiring> IdealCheck<R> for ImpossibleIdeal {
    #[inline(always)]
    fn contains(&self, _value: &R) -> Result<bool, IdealCheckError> {
        Ok(false)
    }
}

impl<I: Ideal> FromRef<I> for ImpossibleIdeal {
    #[inline(always)]
    fn from_ref(_ideal: &I) -> Self {
        ImpossibleIdeal
    }
}

/// A type alias for [`RotationIdeal`][`rotation::RotationIdeal`] with `W = 1`,
/// i.e. ideals of the form `(X - a)`.
pub type DegreeOneIdeal<F> = rotation::RotationIdeal<F, 1>;

#[derive(Clone, Debug, Error)]
#[error("Ideal check failed: {0}")]
pub struct IdealCheckError(String);
