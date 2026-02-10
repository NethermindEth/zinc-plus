use crate::{ConstraintBuilder, dummy_semiring::DummySemiring, ideal::ImpossibleIdeal};

#[derive(Clone, Copy, Debug, Default)]
pub struct DoNothingBuilder;

impl ConstraintBuilder for DoNothingBuilder {
    type Expr = DummySemiring;
    type Ideal = ImpossibleIdeal;

    #[inline(always)]
    fn assert_in_ideal(&mut self, _expr: Self::Expr, _ideal: &Self::Ideal) {
        // do nothing
    }

    #[inline(always)]
    fn assert_zero(&mut self, _expr: Self::Expr) {
        // do nothing
    }
}
