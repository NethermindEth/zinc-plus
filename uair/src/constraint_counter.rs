use crate::dummy_semiring::DummySemiring;
use crate::ideal::{Ideal, ZeroIdeal};
use crate::{ConstraintBuilder, Uair, ideal::DummyIdeal};
use crypto_primitives::Semiring;

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
    type Ideal = DummyIdeal<Self::Expr, ZeroIdeal<Self::Expr>>;

    #[allow(clippy::arithmetic_side_effects)]
    #[inline(always)]
    fn assert_in_ideal<I>(&mut self, _expr: Self::Expr, _ideal_generator: &I)
    where
        I: Ideal<Self::Expr>,
        Self::Ideal: From<I>,
    {
        self.0 += 1;
    }
}
