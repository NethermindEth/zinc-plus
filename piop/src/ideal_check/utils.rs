use std::mem::MaybeUninit;

use crypto_primitives::{DenseRowMatrix, Matrix};
use zinc_poly::{
    CoefficientProjectable, mle::DenseMultilinearExtension,
    univariate::dynamic::over_field::DynamicPolynomialF,
};
use zinc_utils::cfg_iter;

use crate::ideal_check::structs::IdealCheckTypes;

/// Apply projection to coefficients of coefficients of the input trace.
pub(crate) fn project_trace_matrix<
    IcTypes: IdealCheckTypes<DEGREE_PLUS_ONE>,
    const DEGREE_PLUS_ONE: usize,
>(
    num_rows: usize,
    num_cols: usize,
    trace: &[DenseMultilinearExtension<IcTypes::Witness>],
    projecting_element: &IcTypes::F,
) -> Vec<
    DenseMultilinearExtension<DynamicPolynomialF<<IcTypes as IdealCheckTypes<DEGREE_PLUS_ONE>>::F>>,
> {
    cfg_iter!(trace).map()
}
