use std::{cell::RefCell, collections::HashSet};

use crate::{
    TraceRow, Uair, do_nothing_builder::DoNothingBuilder, dummy_semiring::DummySemiring,
    ideal::ImpossibleIdeal,
};

/// Collect all the scalars appearing in a UAIR.
/// Useful to store results of intermediate operations on scalars
/// between protocol stages, e.g. field projections.
pub fn collect_scalars<U: Uair>() -> HashSet<U::Scalar> {
    let scalars = RefCell::new(HashSet::new());

    let sig = U::signature();
    let (up_dummy, down_dummy) = sig.dummy_rows(DummySemiring);
    let up_row = TraceRow::from_slice_with_layout(&up_dummy, sig.total_cols().as_column_layout());
    let down_row =
        TraceRow::from_slice_with_layout(&down_dummy, sig.down_cols().as_column_layout());

    U::constrain_general(
        &mut DoNothingBuilder,
        up_row,
        down_row,
        |x| {
            scalars.borrow_mut().insert(x.clone());
            DummySemiring
        },
        |_x, y| {
            scalars.borrow_mut().insert(y.clone());
            Some(DummySemiring)
        },
        |_| ImpossibleIdeal,
    );

    scalars.into_inner()
}
