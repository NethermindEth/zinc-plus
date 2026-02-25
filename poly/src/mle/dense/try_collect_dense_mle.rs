use itertools::Itertools;
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;
use zinc_utils::log2;

use crate::mle::DenseMultilinearExtension;

pub trait TryCollectDenseMle: Iterator {
    fn try_collect_dense_mle(self) -> Option<DenseMultilinearExtension<Self::Item>>;
}

impl<T> TryCollectDenseMle for T
where
    T: Iterator,
{
    fn try_collect_dense_mle(self) -> Option<DenseMultilinearExtension<Self::Item>> {
        let evaluations = self.collect_vec();
        let len = evaluations.len();

        if len.is_power_of_two() {
            Some(DenseMultilinearExtension {
                evaluations,
                num_vars: log2(len) as usize,
            })
        } else {
            None
        }
    }
}

#[cfg(feature = "parallel")]
pub trait ParallelTryCollectDenseMle: ParallelIterator {
    fn try_collect_dense_mle(self) -> Option<DenseMultilinearExtension<Self::Item>>;
}

#[cfg(feature = "parallel")]
impl<T> ParallelTryCollectDenseMle for T
where
    T: ParallelIterator,
{
    fn try_collect_dense_mle(self) -> Option<DenseMultilinearExtension<Self::Item>> {
        let evaluations = self.collect::<Vec<_>>();
        let len = evaluations.len();

        if len.is_power_of_two() {
            Some(DenseMultilinearExtension {
                evaluations,
                num_vars: log2(len) as usize,
            })
        } else {
            None
        }
    }
}
