use std::{cell::RefCell, collections::HashSet};

use crypto_primitives::Semiring;

use crate::{
    Uair, do_nothing_builder::DoNothingBuilder, dummy_semiring::DummySemiring,
    ideal::ImpossibleIdeal,
};

/// Collect all the scalars appearing in a UAIR.
/// Useful to store results of intermediate operations on scalars
/// between protocol stages, e.g. field projections.
pub fn collect_scalars<R: Semiring + 'static, U: Uair<R>>() -> HashSet<R> {
    let scalars = RefCell::new(HashSet::new());

    let dummy_up_and_down = vec![DummySemiring; U::num_cols()];

    U::constrain_general(
        &mut DoNothingBuilder,
        &dummy_up_and_down,
        &dummy_up_and_down,
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
