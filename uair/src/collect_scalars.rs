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

    let dummy_up_and_down = vec![DummySemiring; U::signature().max_cols()];

    let trace_row = TraceRow {
        binary_poly: &dummy_up_and_down,
        arbitrary_poly: &dummy_up_and_down,
        int: &dummy_up_and_down,
    };

    U::constrain_general(
        &mut DoNothingBuilder,
        trace_row,
        trace_row,
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
