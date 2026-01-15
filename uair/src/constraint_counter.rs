use crate::dummy_semiring::DummySemiring;
use crate::{ConstraintBuilder, Uair, ideal::DummyIdeal};
use crypto_primitives::FixedSemiring;

/// Get the number of polynomial constraints in a `Uair`.
pub fn count_constraints<R: FixedSemiring, U: Uair<R>>() -> usize {
    let mut cc = ConstraintCounter::new();

    let dummy_up_and_down = vec![DummySemiring; U::num_cols()];

    U::constrain(&mut cc, &dummy_up_and_down, &dummy_up_and_down);

    cc.0
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ConstraintCounter(usize);

impl ConstraintCounter {
    pub fn new() -> Self {
        Self(0)
    }
}

impl<R: FixedSemiring> ConstraintBuilder<R> for ConstraintCounter {
    type Expr = DummySemiring;
    type Ideal = DummyIdeal<R>;

    #[allow(clippy::arithmetic_side_effects)]
    #[inline(always)]
    fn assert_in_ideal(&mut self, _expr: Self::Expr, _ideal_generator: &Self::Ideal) {
        self.0 += 1;
    }
}
