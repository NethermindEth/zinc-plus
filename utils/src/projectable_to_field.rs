use crypto_primitives::PrimeField;

/// Trait for preparing a projection function to a field element from a current
/// type.
pub trait ProjectableToField<F: PrimeField> {
    /// Prepare a projection function that will project the current type
    /// to a prime field using the given sampled value.
    fn prepare_projection(sampled_value: &F) -> impl Fn(&Self) -> F + 'static;
}
