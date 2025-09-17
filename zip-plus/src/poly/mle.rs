use std::{
    fmt::Debug,
    ops::{Add, AddAssign, Index, Neg, SubAssign},
};

use num_traits::Zero;
use rand_core::RngCore;

/// This trait describes an interface for the multilinear extension
/// of an array.
/// The latter is a multilinear polynomial represented in terms of its
/// evaluations over the domain {0,1}^`num_vars` (i.e. the Boolean hypercube).
///
/// Index represents a point, which is a vector in {0,1}^`num_vars` in little
/// endian form. For example, `0b1011` represents `P(1,1,0,1)`
pub trait MultilinearExtension<T>:
    Sized
    + Clone
    + Debug
    + PartialEq
    + Eq
    + Add
    + Neg
    + Zero
    + for<'a> AddAssign<&'a Self>
    + for<'a> AddAssign<(T, &'a Self)>
    + for<'a> SubAssign<&'a Self>
    + Index<usize>
{

    /// Reduce the number of variables of `self` by fixing the
    /// `partial_point.len()` variables at `partial_point`.
    fn fix_variables(&mut self, partial_point: &[T]);

    /// Creates a new object with the number of variables of `self` reduced by
    /// fixing the `partial_point.len()` variables at `partial_point`.
    fn fixed_variables(&self, partial_point: &[T]) -> Self;
}

pub trait MultilinearExtensionRand<T>: MultilinearExtension<T> {
    /// Outputs an `l`-variate multilinear extension where value of evaluations
    /// are sampled uniformly at random.
    fn rand<R: RngCore + ?Sized>(num_vars: usize, rng: &mut R) -> Self;
}
