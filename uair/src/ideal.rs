use std::fmt::Debug;

use crypto_primitives::Semiring;
use num_traits::Zero;
use zinc_utils::from_ref::FromRef;

/// A trait for types describing ideals.
pub trait Ideal: FromRef<Self> + Clone + Debug + Send + Sync {
    /// Get an ideal object defining a zero ideal,
    /// i.e. containing only the additive zero
    /// of a semiring.
    fn zero_ideal() -> Self;
}

/// A trait for semirings or different structures
/// elements of which can be checked to belong to
/// an ideal of type `I`.
pub trait IdealCheck<I: Ideal> {
    /// Returns true if an element of the type
    /// belongs to the ideal `ideal`.
    fn is_contained_in(&self, ideal: &I) -> bool;
}

/// The type of ideals that encode only
/// one ideal: zero ideal containing only the additive zero.
#[derive(Clone, Copy, Debug)]
pub struct ZeroIdeal;

impl Ideal for ZeroIdeal {
    #[inline(always)]
    fn zero_ideal() -> Self {
        ZeroIdeal
    }
}

impl<R: Semiring + Zero> IdealCheck<ZeroIdeal> for R {
    #[inline(always)]
    fn is_contained_in(&self, _ideal: &ZeroIdeal) -> bool {
        self.is_zero()
    }
}

impl FromRef<ZeroIdeal> for ZeroIdeal {
    #[inline(always)]
    fn from_ref(_ideal: &ZeroIdeal) -> Self {
        ZeroIdeal
    }
}

/// A dummy ideal. Convenient when ideal checks
/// have to be ignored.
#[derive(Clone, Copy, Debug)]
pub struct DummyIdeal;

impl Ideal for DummyIdeal {
    #[inline(always)]
    fn zero_ideal() -> Self {
        DummyIdeal
    }
}

impl<R: Semiring> IdealCheck<DummyIdeal> for R {
    #[inline(always)]
    fn is_contained_in(&self, _ideal: &DummyIdeal) -> bool {
        false
    }
}

impl<I: Ideal> FromRef<I> for DummyIdeal {
    #[inline(always)]
    fn from_ref(_ideal: &I) -> Self {
        DummyIdeal
    }
}
