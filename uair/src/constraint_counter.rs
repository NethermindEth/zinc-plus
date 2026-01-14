mod dummy_semiring;

use crate::{ConstraintBuilder, Uair};
use crypto_primitives::Semiring;
use dummy_semiring::DummySemiring;

/// Get the number of polynomial constraints in a `Uair`.
pub fn count_constraints<R: Semiring, U: Uair<R>>() -> usize {
    let mut cc = ConstraintCounter::default();

    let dummy_up_and_down = vec![DummySemiring; U::num_cols()];

    U::constrain(&mut cc, &dummy_up_and_down, &dummy_up_and_down);

    cc.0
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ConstraintCounter(usize);

impl ConstraintBuilder for ConstraintCounter {
    type Expr = DummySemiring;

    #[allow(clippy::arithmetic_side_effects)]
    #[inline(always)]
    fn assert_in_ideal(&mut self, _expr: Self::Expr, _ideal_generator: &Self::Expr) {
        self.0 += 1;
    }
}
