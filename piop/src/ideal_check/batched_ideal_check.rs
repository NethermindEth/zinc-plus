#[cfg(feature = "parallel")]
use rayon::prelude::*;
use thiserror::Error;
use zinc_uair::ideal::{Ideal, IdealCheck, IdealCheckError};
use zinc_utils::cfg_iter;

/// Checks if the collected ideals contain a slice
/// of elements `values`. Returns an error if the
/// lengths mismatch or if any of the `values`
/// does not belong to the corresponding ideal.
pub fn batched_ideal_check<I: Ideal + IdealCheck<R>, R: Clone + Send + Sync>(
    ideals: &[I],
    values: &[R],
) -> Result<(), BatchedIdealCheckError<R, I>> {
    if ideals.len() != values.len() {
        return Err(BatchedIdealCheckError::LengthMismatch {
            num_ideals: ideals.len(),
            provided_values: values.len(),
        });
    }

    cfg_iter!(ideals)
        .zip(cfg_iter!(values))
        .try_for_each(|(ideal, value)| {
            if !ideal.contains(value)? {
                Err(BatchedIdealCheckError::NotInIdeal(
                    value.clone(),
                    ideal.clone(),
                ))
            } else {
                Ok(())
            }
        })
}

#[derive(Clone, Debug, Error)]
pub enum BatchedIdealCheckError<R, I> {
    #[error(
        "length mismatch: the collector has {num_ideals} ideals, provided {provided_values} values to check"
    )]
    LengthMismatch {
        num_ideals: usize,
        provided_values: usize,
    },
    #[error("{0} does not belong to the ideal {1}")]
    NotInIdeal(R, I),
    #[error("Ideal check failed: {}", 0.0)]
    IdealCheckFailed(#[from] IdealCheckError),
}
