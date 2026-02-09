use std::mem::MaybeUninit;

use crypto_primitives::{DenseRowMatrix, Matrix};
use zinc_poly::{
    CoefficientProjectable, mle::DenseMultilinearExtension,
    univariate::dynamic::over_field::DynamicPolynomialF,
};

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
) -> DenseRowMatrix<DynamicPolynomialF<<IcTypes as IdealCheckTypes<DEGREE_PLUS_ONE>>::F>> {
    let mut matr = DenseRowMatrix::uninit(num_rows, num_cols);

    matr.cells_mut().enumerate().for_each(|(row_idx, row)| {
        row.for_each(|(col_idx, cell)| {
            *cell = MaybeUninit::new(
                trace[col_idx][row_idx]
                    .project_coefficients(projecting_element)
                    .into(),
            );
        });
    });

    unsafe { matr.init() }
}
