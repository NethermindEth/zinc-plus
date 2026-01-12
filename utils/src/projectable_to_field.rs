use ark_std::cfg_iter_mut;
use crypto_primitives::PrimeField;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Trait for preparing a projection function to a field element from a current
/// type.
pub trait ProjectableToField<F: PrimeField> {
    /// Prepare a projection function that will project the current type
    /// to a prime field using the given sampled value.
    fn prepare_projection(sampled_value: &F) -> impl Fn(&Self) -> F + 'static + Send + Sync;
}

/// An unsafe trait for types that can be projected onto a field's
/// inner type. We expect that this projected element is in a correct
/// Montgomery form representation that corresponds to the config of
/// `sampled_value`.
///
/// # Safety
/// The returned closure expects to receive a non-null pointer to
/// an element of the type and should modify its content to a correct
/// Montgomery representation of an element of `F`.
pub unsafe trait ProjectableToFieldInner<F: PrimeField> {
    fn prepare_projection_to_inner(sampled_value: &F) -> impl Fn(*mut Self) + 'static;
}

/// Apply a projection to each element of the vector
/// to obtain a vector of projected elements.
/// No intermediate vector allocations.
///
/// # Safety
/// The closure `project` is expected to write a correct representation
/// of an object of type `F` into a given pointer.
///
/// # Panics
/// The function asserts if the sizes of the types `T` and `F` are the same.
pub unsafe fn project_vec_in_place<T, F, P>(mut vec: Vec<T>, project: P) -> Vec<F>
where
    T: Send + Sync,
    F: Send + Sync,
    P: Fn(*mut T) + Send + Sync,
{
    assert_eq!(size_of::<T>(), size_of::<F>());

    cfg_iter_mut!(vec).for_each(|x| {
        project(x);
    });

    unsafe { std::mem::transmute(vec) }
}

#[cfg(test)]
mod test {
    use crate::projectable_to_field::project_vec_in_place;

    #[test]
    fn project_vec_in_place_i32_u32() {
        let x: Vec<i32> = vec![-2, -1, 0, 1, 2];

        let y: Vec<u32> = unsafe { project_vec_in_place(x, |_| {}) };

        assert_eq!(y, vec![4294967294, 4294967295, 0, 1, 2]);
    }
}
