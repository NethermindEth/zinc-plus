use itertools::Itertools;
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
