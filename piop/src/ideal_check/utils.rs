#[cfg(feature = "parallel")]
use rayon::prelude::*;

use num_traits::Zero;
use zinc_poly::{
    CoefficientProjectable, mle::DenseMultilinearExtension, mle::dense::CollectDenseMleWithZero,
    univariate::dynamic::over_field::DynamicPolynomialF,
};
use zinc_utils::cfg_iter;

use crate::ideal_check::structs::IdealCheckTypes;

/// Apply projection to coefficients of coefficients of the input trace.
pub(crate) fn project_trace_matrix<
    IcTypes: IdealCheckTypes<DEGREE_PLUS_ONE>,
    const DEGREE_PLUS_ONE: usize,
>(
    trace: &[DenseMultilinearExtension<IcTypes::Witness>],
    projecting_element: &IcTypes::F,
) -> Vec<
    DenseMultilinearExtension<DynamicPolynomialF<<IcTypes as IdealCheckTypes<DEGREE_PLUS_ONE>>::F>>,
> {
    cfg_iter!(trace)
        .map(|trace_col| {
            cfg_iter!(trace_col)
                .map(|x| x.project_coefficients(projecting_element).into())
                .collect_dense_mle_with_zero(&DynamicPolynomialF::zero())
        })
        .collect()
}
