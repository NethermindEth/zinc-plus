use std::mem::MaybeUninit;

use crypto_primitives::{DenseRowMatrix, Field, Matrix, PrimeField};
use itertools::{Itertools, max};
use zinc_poly::{
    CoefficientProjectable,
    mle::{DenseMultilinearExtension, dense::CollectDenseMleWithZero},
    univariate::dynamic::DynamicPolynomial,
};
use zinc_uair::{ConstraintBuilder, Uair, ideal::DummyIdeal};
use zinc_utils::from_ref::FromRef;

use crate::ideal_check::structs::IdealCheckTypes;

pub struct CombinedPolyRowBuilder<F: PrimeField> {
    combined_evaluations: Vec<DynamicPolynomial<F>>,
}

impl<F: PrimeField> ConstraintBuilder for CombinedPolyRowBuilder<F> {
    type Expr = DynamicPolynomial<F>;
    type Ideal = DummyIdeal;

    #[allow(clippy::arithmetic_side_effects)]
    fn assert_in_ideal(&mut self, expr: Self::Expr, _ideal: &Self::Ideal) {
        self.combined_evaluations.push(expr);
    }
}

impl<F: PrimeField> CombinedPolyRowBuilder<F> {
    pub fn new(num_constraints: usize) -> Self {
        Self {
            combined_evaluations: Vec::with_capacity(num_constraints),
        }
    }
}

#[allow(clippy::arithmetic_side_effects)]
pub fn compute_combined_polynomials<
    IcTypes: IdealCheckTypes<DEGREE_PLUS_ONE>,
    U,
    const DEGREE_PLUS_ONE: usize,
>(
    trace: &[DenseMultilinearExtension<IcTypes::Witness>],
    projecting_element: &IcTypes::F,
    num_constraints: usize,
) -> Vec<Vec<DenseMultilinearExtension<<IcTypes::F as Field>::Inner>>>
where
    U: Uair<IcTypes::Witness>,
{
    let num_rows = trace[0].len();
    let num_cols = trace.len();

    let trace_matrix =
        project_trace_matrix::<IcTypes, _>(num_rows, num_cols, trace, projecting_element);

    let field_zero = IcTypes::F::zero_with_cfg(projecting_element.cfg());

    let mut max_degrees_and_combined_poly_rows: Vec<(usize, Vec<DynamicPolynomial<IcTypes::F>>)> =
        trace_matrix
            .as_rows()
            .zip(trace_matrix.as_rows().skip(1))
            .map(|(up, down)| {
                combine_rows_and_get_max_degree::<IcTypes, U, _>(
                    up,
                    down,
                    num_constraints,
                    projecting_element,
                    &field_zero,
                )
            })
            .collect();

    let max_degree = *max(max_degrees_and_combined_poly_rows
        .iter()
        .map(|(max_degree, _)| max_degree))
    .expect("We assume the number of constraints is not zero so this iterator is not empty");

    // For the sake of padding we duplicate
    // the last combined value
    // to have N-sized mle at the end
    // not N-1.
    // This is essentially c^up and c^down
    // thing from the whirlaway.
    max_degrees_and_combined_poly_rows.push(
        max_degrees_and_combined_poly_rows
            .last()
            .expect("We assume the number of constraints is not zero so this iterator is not empty")
            .clone(),
    );

    prepare_coefficient_mles::<IcTypes, _>(
        num_constraints,
        max_degree,
        &max_degrees_and_combined_poly_rows,
        &field_zero,
    )
}

fn project_trace_matrix<IcTypes: IdealCheckTypes<DEGREE_PLUS_ONE>, const DEGREE_PLUS_ONE: usize>(
    num_rows: usize,
    num_cols: usize,
    trace: &[DenseMultilinearExtension<IcTypes::Witness>],
    projecting_element: &IcTypes::F,
) -> DenseRowMatrix<DynamicPolynomial<<IcTypes as IdealCheckTypes<DEGREE_PLUS_ONE>>::F>> {
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

#[allow(clippy::arithmetic_side_effects)]
fn combine_rows_and_get_max_degree<IcTypes, U, const DEGREE_PLUS_ONE: usize>(
    up: &[DynamicPolynomial<IcTypes::F>],
    down: &[DynamicPolynomial<IcTypes::F>],
    num_constraints: usize,
    projecting_element: &IcTypes::F,
    field_zero: &IcTypes::F,
) -> (usize, Vec<DynamicPolynomial<IcTypes::F>>)
where
    IcTypes: IdealCheckTypes<DEGREE_PLUS_ONE>,
    U: Uair<IcTypes::Witness>,
{
    let mut constraint_builder = CombinedPolyRowBuilder::new(num_constraints);

    U::constrain_general(
        &mut constraint_builder,
        up,
        down,
        |x| x.project_coefficients(projecting_element).into(),
        |x, y| Some(DynamicPolynomial::from(y.project_coefficients(projecting_element)) * x),
        DummyIdeal::from_ref,
    );

    let mut combined_evaluations = constraint_builder.combined_evaluations;

    combined_evaluations
        .iter_mut()
        .for_each(|eval| eval.trim_with_zero(field_zero));

    let max_degree = max(combined_evaluations
        .iter()
        .map(|eval| eval.degree_with_zero(field_zero).unwrap_or(0)))
    .expect("We assume the number of constraints is not zero so this iterator is not empty");

    (max_degree, combined_evaluations)
}

fn prepare_coefficient_mles<
    IcTypes: IdealCheckTypes<DEGREE_PLUS_ONE>,
    const DEGREE_PLUS_ONE: usize,
>(
    num_constraints: usize,
    max_degree: usize,
    max_degrees_and_combined_poly_rows: &[(usize, Vec<DynamicPolynomial<IcTypes::F>>)],
    field_zero: &IcTypes::F,
) -> Vec<Vec<DenseMultilinearExtension<<IcTypes::F as Field>::Inner>>> {
    (0..num_constraints)
        .map(|constraint| {
            (0..=max_degree)
                .map(|coeff| {
                    max_degrees_and_combined_poly_rows
                        .iter()
                        .map(|(_, row)| {
                            if coeff >= row[constraint].coeffs.len() {
                                field_zero.inner().clone()
                            } else {
                                row[constraint].coeffs[coeff].inner().clone()
                            }
                        })
                        .collect_dense_mle_with_zero(field_zero.inner())
                })
                .collect_vec()
        })
        .collect_vec()
}
