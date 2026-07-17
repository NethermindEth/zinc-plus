use crypto_primitives::FieldConfig;

/// Trait for preparing a projection function to a field element from a current
/// type.
pub trait ProjectableToField<F: FieldConfig> {
    /// Prepare a projection function that will project the current type
    /// to a prime field using the given config and sampled value.
    fn prepare_projection(
        cfg: &F,
        sampled_value: &F::Element,
    ) -> impl Fn(&Self) -> F::Element + Send + Sync;
}
