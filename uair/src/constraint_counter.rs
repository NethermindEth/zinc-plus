use crate::{
    ConstraintBuilder, TraceRow, Uair, dummy_semiring::DummySemiring, ideal::ImpossibleIdeal,
};

/// Get the number of polynomial constraints in a `Uair`.
pub fn count_constraints<U: Uair>() -> usize {
    let mut cc = ConstraintCounter::new();

    let sig = U::signature();
    let (up_dummy, down_dummy) = sig.dummy_rows(DummySemiring);
    let up_row = TraceRow::from_slice_with_layout(&up_dummy, sig.total_cols().as_column_layout());
    let down_row = TraceRow::from_slice_with_layout_and_bit_op(
        &down_dummy,
        sig.down_cols().as_column_layout(),
        sig.bit_op_down_count(),
    );

    U::constrain(&mut cc, up_row, down_row);

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
