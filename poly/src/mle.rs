pub mod dense;

use crypto_primitives::SemiringConfig;
pub use dense::DenseMultilinearExtension;

use rand::prelude::*;
use std::fmt::Debug;

/// This trait describes an interface for the multilinear extension
/// of an array.
/// The latter is a multilinear polynomial represented in terms of its
/// evaluations over the domain {0,1}^`num_vars` (i.e. the Boolean hypercube).
///
/// Index represents a point, which is a vector in {0,1}^`num_vars` in little
/// endian form. For example, `0b1011` represents `P(1,1,0,1)`
pub trait MultilinearExtension<C: SemiringConfig>: Sized + Clone + Debug + PartialEq + Eq {
    /// Reduce the number of variables of `self` by fixing the
    /// `partial_point.len()` variables at `partial_point`.
    fn fix_variables(&mut self, cfg: &C, partial_point: &[C::Element]);

    /// Creates a new object with the number of variables of `self` reduced by
    /// fixing the `partial_point.len()` variables at `partial_point`.
    fn fixed_variables(&self, cfg: &C, partial_point: &[C::Element]) -> Self;
}

pub trait MultilinearExtensionRand<T> {
    /// Outputs an `l`-variate multilinear extension where value of evaluations
    /// are sampled uniformly at random.
    fn rand<R: Rng + ?Sized>(num_vars: usize, rng: &mut R) -> Self;
}
