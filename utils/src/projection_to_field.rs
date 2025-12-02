use std::marker::PhantomData;

use crypto_primitives::{FromWithConfig, IntoWithConfig, PrimeField};

/// Trait for preparing a projection function to a field element from a
/// type.
pub trait ProjectionToField<T, F: PrimeField> {
    /// Prepare a projection function that will project the type `T`
    /// to a prime field using the given sampled value.
    fn prepare_projection(sampled_value: &F) -> impl Fn(&T) -> F + 'static;
}

/// If `F` implements `for<'a> FromWithConfig<&'a T>`
/// we can always project this type onto `T` using this projection.
pub struct SimpleProjection<F: PrimeField>(PhantomData<F>);

impl<T, F> ProjectionToField<T, F> for SimpleProjection<F>
where
    F: for<'a> FromWithConfig<&'a T>,
{
    fn prepare_projection(sampled_value: &F) -> impl Fn(&T) -> F + 'static {
        let config = sampled_value.cfg().clone();
        move |value: &T| value.into_with_cfg(&config)
    }
}
