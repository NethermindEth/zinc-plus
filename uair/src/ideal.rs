use std::fmt::Debug;

use crypto_primitives::Semiring;
use zinc_utils::from_ref::FromRef;

/// A trait for types describing ideals.
pub trait Ideal: FromRef<Self> + Clone + Debug + Send + Sync {}

/// A trait for semirings or different structures
/// elements of which can be checked to belong to
/// an ideal of type `I`.
pub trait IdealCheck<T> {
    /// Returns true if an element of the type
    /// belongs to this ideal.
    fn contains(&self, value: &T) -> bool;
}

/// A dummy ideal. Convenient when ideal checks
/// have to be ignored.
#[derive(Clone, Copy, Debug)]
pub struct ImpossibleIdeal;

impl Ideal for ImpossibleIdeal {}

impl<R: Semiring> IdealCheck<R> for ImpossibleIdeal {
    #[inline(always)]
    fn contains(&self, _value: &R) -> bool {
        false
    }
}

impl<I: Ideal> FromRef<I> for ImpossibleIdeal {
    #[inline(always)]
    fn from_ref(_ideal: &I) -> Self {
        ImpossibleIdeal
    }
}
