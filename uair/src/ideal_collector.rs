use crypto_primitives::Semiring;
use zinc_utils::from_ref::FromRef;

use crate::{
    ConstraintBuilder, Uair,
    dummy_semiring::DummySemiring,
    ideal::{Ideal, IdealCheck},
};

/// A `ConstraintBuilder` that collects
/// ideals used in a `Uair`.
pub struct IdealCollector<I: Ideal> {
    pub ideals: Vec<CollectedIdeal<I>>,
}

impl<I: Ideal> IdealCollector<I> {
    /// Create a new ideal collector
    /// and hint the number of constraints
    /// a target UAIR might have.
    pub fn new(num_constraints: usize) -> Self {
        Self {
            ideals: Vec::with_capacity(num_constraints),
        }
    }
}

/// Given a `Uair` and a hint of how many constraints
/// it is going to have, creates an `IdealCollector`
/// object and collects ideals from the `Uair`.
pub fn collect_ideals<R: Semiring + 'static, U: Uair<R>>(
    num_constraints: usize,
) -> IdealCollector<U::Ideal> {
    let mut ideal_collector = IdealCollector::new(num_constraints);

    let dummy_up_and_down: Vec<DummySemiring> = vec![DummySemiring; U::num_cols()];

    U::constrain(&mut ideal_collector, &dummy_up_and_down, &dummy_up_and_down);

    ideal_collector
}

impl<I> ConstraintBuilder for IdealCollector<I>
where
    I: Ideal,
{
    type Expr = DummySemiring;
    type Ideal = CollectedIdeal<I>;

    fn assert_in_ideal(&mut self, _expr: Self::Expr, ideal: &Self::Ideal) {
        self.ideals.push(ideal.clone());
    }

    fn assert_zero(&mut self, _expr: Self::Expr) {
        self.ideals.push(CollectedIdeal::zero());
    }
}

/// A type implementing ideal trait
/// that is used to store inner
/// ideal type `I` but ignores all
/// ideal checks for the sake of just
/// collecting the inner ideals.
#[derive(Clone, Copy, Debug)]
pub struct CollectedIdeal<I: Ideal> {
    pub ideal_or_zero: Option<I>,
}

impl<I: Ideal> CollectedIdeal<I> {
    pub fn zero() -> Self {
        CollectedIdeal {
            ideal_or_zero: None,
        }
    }

    pub fn map<I2: Ideal>(&self, f: impl FnOnce(&I) -> I2) -> CollectedIdeal<I2> {
        match &self.ideal_or_zero {
            Some(ideal) => CollectedIdeal {
                ideal_or_zero: Some(f(ideal)),
            },
            None => CollectedIdeal::zero(),
        }
    }
}

impl<I: Ideal> Ideal for CollectedIdeal<I> {}

impl<I: Ideal> FromRef<CollectedIdeal<I>> for CollectedIdeal<I> {
    fn from_ref(value: &CollectedIdeal<I>) -> Self {
        value.clone()
    }
}

impl<I: Ideal> FromRef<I> for CollectedIdeal<I> {
    fn from_ref(value: &I) -> Self {
        Self {
            ideal_or_zero: Some(value.clone()),
        }
    }
}

impl<I: Ideal> IdealCheck<DummySemiring> for CollectedIdeal<I> {
    fn contains(&self, _value: &DummySemiring) -> bool {
        // Do nothing.
        true
    }
}
