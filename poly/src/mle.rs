pub mod dense;

pub use dense::DenseMultilinearExtension;

use rand::prelude::*;
use std::{
    fmt::Debug,
    ops::{Add, AddAssign, Index, SubAssign},
};
use zinc_utils::mul_by_scalar::MulByScalar;

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
    + for<'a> AddAssign<&'a Self>
    + for<'a> AddAssign<(T, &'a Self)>
    + for<'a> SubAssign<&'a Self>
    + Index<usize>
{
    /// Reduce the number of variables of `self` by fixing the
    /// `partial_point.len()` variables at `partial_point`.
    fn fix_variables<S>(&mut self, partial_point: &[S], zero: T)
    where
        T: for<'a> MulByScalar<&'a S>;

    /// Creates a new object with the number of variables of `self` reduced by
    /// fixing the `partial_point.len()` variables at `partial_point`.
    fn fixed_variables<S>(&self, partial_point: &[S], zero: T) -> Self
    where
        T: for<'a> MulByScalar<&'a S>;
}

pub trait MultilinearExtensionRand<T> {
    /// Outputs an `l`-variate multilinear extension where value of evaluations
    /// are sampled uniformly at random.
    fn rand<R: RngCore + ?Sized>(num_vars: usize, rng: &mut R) -> Self;
}
