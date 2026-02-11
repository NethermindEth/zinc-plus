use std::{collections::HashMap, mem::MaybeUninit};

use crate::ideal_check::IdealCheckField;
use crypto_primitives::DenseRowMatrix;
use itertools::Itertools;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use zinc_poly::{
    CoefficientProjectable,
    mle::{DenseMultilinearExtension, dense::CollectDenseMleWithZero},
    univariate::{binary::BinaryPoly, dynamic::over_field::DynamicPolynomialF},
};
use zinc_uair::{ConstraintBuilder, Uair, ideal::ImpossibleIdeal};
use zinc_utils::{cfg_chunks_mut, cfg_into_iter, cfg_iter, from_ref::FromRef};

/// Given a UAIR `U` and a trace `trace` this function
/// obtains the combined polynomials' MLE coefficients.
/// Since each coefficient is also a univariate polynomial
/// we split the resulting MLE into coefficient MLEs.
#[allow(clippy::arithmetic_side_effects)]
pub fn compute_combined_polynomials<F: IdealCheckField, U, const DEGREE_PLUS_ONE: usize>(
    trace: &[DenseMultilinearExtension<BinaryPoly<DEGREE_PLUS_ONE>>],
    projecting_element: &F,
    projected_scalars: &HashMap<BinaryPoly<DEGREE_PLUS_ONE>, DynamicPolynomialF<F>>,
    num_constraints: usize,
) -> Vec<Vec<DenseMultilinearExtension<F::Inner>>>
where
    U: Uair<BinaryPoly<DEGREE_PLUS_ONE>>,
{
    let num_rows = trace[0].len();
    let num_cols = trace.len();

    let trace_matrix = project_trace_matrix(num_rows, num_cols, trace, projecting_element);

    let field_zero = F::zero_with_cfg(projecting_element.cfg());

    let rows: Vec<_> = trace_matrix.as_rows().collect();
    let indices: Vec<usize> = (0..num_rows - 1).collect();
    let mut max_degrees_and_combined_poly_rows: Vec<(usize, Vec<DynamicPolynomialF<F>>)> =
        cfg_iter!(indices)
            .map(|&i| {
                combine_rows_and_get_max_degree::<_, U, _>(
                    rows[i],
                    rows[i + 1],
                    num_constraints,
                    projected_scalars,
                )
            })
            .collect();

    let max_degree = *max_degrees_and_combined_poly_rows
        .iter()
        .map(|(max_degree, _)| max_degree)
        .max()
        .expect("We assume the number of constraints is not zero so this iterator is not empty");

    // For the sake of padding we duplicate
    // the last combined value
    // to have N-sized mle at the end
    // not N-1.
    // This is essentially c^up and c^down
    // thing from the whirlaway.
    // TODO(Ilia): reimplement it using Albert's idea
    //             with selector polynomials.
    max_degrees_and_combined_poly_rows.push(
        max_degrees_and_combined_poly_rows
            .last()
            .expect("We assume the number of constraints is not zero so this iterator is not empty")
            .clone(),
    );

    prepare_coefficient_mles(
        num_constraints,
        max_degree,
        &max_degrees_and_combined_poly_rows,
        &field_zero,
    )
}

/// Apply projection to coefficients of coefficients of the input trace.
fn project_trace_matrix<F: IdealCheckField, const DEGREE_PLUS_ONE: usize>(
    num_rows: usize,
    num_cols: usize,
    trace: &[DenseMultilinearExtension<BinaryPoly<DEGREE_PLUS_ONE>>],
    projecting_element: &F,
) -> DenseRowMatrix<DynamicPolynomialF<F>> {
    let mut matr = DenseRowMatrix::uninit(num_rows, num_cols);

    cfg_chunks_mut!(matr.data, num_cols)
        .enumerate()
        .for_each(|(row_idx, row)| {
            for (col_idx, cell) in row.iter_mut().enumerate() {
                *cell = MaybeUninit::new(
                    trace[col_idx][row_idx]
                        .project_coefficients(projecting_element)
                        .into(),
                );
            }
        });

    unsafe { matr.init() }
}

/// Apply combination polynomial to each row
/// and compute the maximum degree of resulting polynomials
/// to pad the resulting vector of MLEs accordingly.
#[allow(clippy::arithmetic_side_effects)]
fn combine_rows_and_get_max_degree<F, U, const DEGREE_PLUS_ONE: usize>(
    up: &[DynamicPolynomialF<F>],
    down: &[DynamicPolynomialF<F>],
    num_constraints: usize,
    projected_scalars: &HashMap<BinaryPoly<DEGREE_PLUS_ONE>, DynamicPolynomialF<F>>,
) -> (usize, Vec<DynamicPolynomialF<F>>)
where
    F: IdealCheckField,
    U: Uair<BinaryPoly<DEGREE_PLUS_ONE>>,
{
    let mut constraint_builder = CombinedPolyRowBuilder::new(num_constraints);

    let project = |x: &BinaryPoly<DEGREE_PLUS_ONE>| {
        projected_scalars
            .get(x)
            .cloned()
            .expect("all scalars should have been projected at this point")
    };

    U::constrain_general(
        &mut constraint_builder,
        up,
        down,
        &project,
        |x, y| Some(project(y) * x),
        ImpossibleIdeal::from_ref,
    );

    let mut combined_evaluations = constraint_builder.combined_evaluations;

    combined_evaluations.iter_mut().for_each(|eval| eval.trim());

    let max_degree = combined_evaluations
        .iter()
        .map(|eval| eval.degree().unwrap_or(0))
        .max()
        .expect("We assume the number of constraints is not zero so this iterator is not empty");

    (max_degree, combined_evaluations)
}

/// Turn the resulting slice of vectors of dynamic polynomials
/// into a vector of vectors of coefficient MLEs.
fn prepare_coefficient_mles<F: IdealCheckField>(
    num_constraints: usize,
    max_degree: usize,
    max_degrees_and_combined_poly_rows: &[(usize, Vec<DynamicPolynomialF<F>>)],
    field_zero: &F,
) -> Vec<Vec<DenseMultilinearExtension<F::Inner>>> {
    cfg_into_iter!(0..num_constraints)
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
        .collect()
}

pub struct CombinedPolyRowBuilder<F: IdealCheckField> {
    combined_evaluations: Vec<DynamicPolynomialF<F>>,
}

impl<F: IdealCheckField> ConstraintBuilder for CombinedPolyRowBuilder<F> {
    type Expr = DynamicPolynomialF<F>;
    type Ideal = ImpossibleIdeal;

    fn assert_in_ideal(&mut self, expr: Self::Expr, _ideal: &Self::Ideal) {
        self.combined_evaluations.push(expr);
    }

    fn assert_zero(&mut self, expr: Self::Expr) {
        self.combined_evaluations.push(expr);
    }
}

impl<F: IdealCheckField> CombinedPolyRowBuilder<F> {
    pub fn new(num_constraints: usize) -> Self {
        Self {
            combined_evaluations: Vec::with_capacity(num_constraints),
        }
    }
}
