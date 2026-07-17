pub mod rotation;

use crypto_primitives::SetConfig;
use std::fmt::{Debug, Display, Formatter};
use thiserror::Error;
use zinc_utils::from_ref::FromRef;

/// A trait for types describing ideals.
pub trait Ideal: FromRef<Self> + Clone + Debug + Display + Send + Sync {}

/// A trait for ideals that implement membership check for the algebraic
/// structure configured by `C`.
pub trait IdealCheck<C: SetConfig> {
    /// Returns true if the element belongs to this ideal.
    fn contains(&self, cfg: &C, value: &C::Element) -> Result<bool, IdealCheckError>;
}

/// A dummy ideal. Convenient when ideal checks
/// have to be ignored.
#[derive(Clone, Copy, Debug)]
pub struct ImpossibleIdeal;

impl Ideal for ImpossibleIdeal {}

impl Display for ImpossibleIdeal {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "ImpossibleIdeal")
    }
}

impl<C: SetConfig> IdealCheck<C> for ImpossibleIdeal {
    #[inline(always)]
    fn contains(&self, _cfg: &C, _value: &C::Element) -> Result<bool, IdealCheckError> {
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
