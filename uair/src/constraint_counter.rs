use crate::{
    ConstraintBuilder, TraceRow, Uair, dummy_semiring::DummySemiring, ideal::ImpossibleIdeal,
};
use crypto_primitives::Semiring;

/// Get the number of polynomial constraints in a `Uair`.
pub fn count_constraints<U: Uair>() -> usize {
    let mut cc = ConstraintCounter::new();

    let dummy_up_and_down = vec![DummySemiring; U::signature().max_cols()];

    let trace_row = TraceRow {
        binary_poly: &dummy_up_and_down,
        arbitrary_poly: &dummy_up_and_down,
        int: &dummy_up_and_down,
    };

    U::constrain(&mut cc, trace_row, trace_row);

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
    type Ideal = ImpossibleIdeal;

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
