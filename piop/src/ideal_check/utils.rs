use crate::ideal_check::IdealCheckField;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::collections::HashMap;
use zinc_poly::{
    CoefficientProjectable,
    mle::DenseMultilinearExtension,
    univariate::{binary::BinaryPoly, dynamic::over_field::DynamicPolynomialF},
};
use zinc_uair::{Uair, collect_scalars::collect_scalars};
use zinc_utils::cfg_iter;

/// Apply projection to coefficients of coefficients of the input trace.
pub(crate) fn project_trace_matrix<F: IdealCheckField, const DEGREE_PLUS_ONE: usize>(
    trace: &[DenseMultilinearExtension<BinaryPoly<DEGREE_PLUS_ONE>>],
    projecting_element: &F,
) -> Vec<DenseMultilinearExtension<DynamicPolynomialF<F>>> {
    cfg_iter!(trace)
        .map(|trace_col| {
            cfg_iter!(trace_col)
                .map(|x| x.project_coefficients(projecting_element).into())
                .collect()
        })
        .collect()
}

pub(crate) fn project_scalars<
    F: IdealCheckField,
    U: Uair<BinaryPoly<DEGREE_PLUS_ONE>>,
    const DEGREE_PLUS_ONE: usize,
>(
    projecting_element: &F,
) -> HashMap<BinaryPoly<DEGREE_PLUS_ONE>, DynamicPolynomialF<F>> {
    let uair_scalars = collect_scalars::<BinaryPoly<DEGREE_PLUS_ONE>, U>();

    // TODO(Ilia): if there's a lot of scalars
    //             we should do this in parallel probably.
    uair_scalars
        .into_iter()
        .map(|scalar| {
            (scalar.clone(), {
                let mut dynamic_poly =
                    DynamicPolynomialF::from(scalar.project_coefficients(projecting_element));

                dynamic_poly.trim();

                dynamic_poly
            })
        })
        .collect()
}
