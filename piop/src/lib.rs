pub mod combined_poly_resolver;
pub mod ideal_check;
pub mod lookup;
pub mod multipoint_eval;
pub mod multipoint_reducer;
pub mod projections;
pub mod random_field_sumcheck;
pub mod scalar_proj_cache;
pub mod shift_predicate;
pub mod sumcheck;
#[cfg(test)]
pub mod test_utils;

/// Sumcheck combination function: maps MLE evaluations to a single field
/// element.
pub type CombFn<F> = Box<dyn Fn(&[F]) -> F + Send + Sync>;
