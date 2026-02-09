use crate::{ConstraintBuilder, Uair, dummy_semiring::DummySemiring, ideal::DummyIdeal};
use crypto_primitives::Semiring;

/// Get the number of polynomial constraints in a `Uair`.
pub fn count_constraints<R, U>() -> usize
where
    R: Semiring + 'static,
    U: Uair<R>,
{
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

impl ConstraintBuilder for ConstraintCounter {
    type Expr = DummySemiring;
    type Ideal = DummyIdeal;

    #[allow(clippy::arithmetic_side_effects)]
    #[inline(always)]
    fn assert_in_ideal(&mut self, _expr: Self::Expr, _ideal_generator: &Self::Ideal) {
        self.0 += 1;
    }

    #[allow(clippy::arithmetic_side_effects)]
    #[inline(always)]
    fn assert_zero(&mut self, _expr: Self::Expr) {
        self.0 += 1;
    }
}
