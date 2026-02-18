use crypto_primitives::PrimeField;
use std::collections::HashMap;

use itertools::Itertools;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use zinc_poly::{
    EvaluationError,
    mle::{
        DenseMultilinearExtension, MultilinearExtensionWithConfig,
        dense::CollectDenseMleWithZero,
    },
    univariate::dynamic::over_field::DynamicPolynomialF,
};
use zinc_uair::{ConstraintBuilder, TraceRow, Uair, ideal::ImpossibleIdeal};
use zinc_utils::{cfg_into_iter, cfg_iter, from_ref::FromRef};

/// Given a UAIR `U` and a trace `trace` this function
/// obtains the combined polynomials' MLE coefficients.
/// Since each coefficient is also a univariate polynomial
/// we split the resulting MLE into coefficient MLEs.
#[allow(clippy::arithmetic_side_effects)]
pub fn compute_combined_polynomials<F, U>(
    trace_matrix: &[DenseMultilinearExtension<DynamicPolynomialF<F>>],
    projected_scalars: &HashMap<U::Scalar, DynamicPolynomialF<F>>,
    num_constraints: usize,
    field_cfg: &F::Config,
) -> Vec<Vec<DenseMultilinearExtension<F::Inner>>>
where
    F: PrimeField,
    U: Uair,
{
    let field_zero = F::zero_with_cfg(field_cfg);

    let num_rows = trace_matrix[0].len();

    let mut max_degrees_and_combined_poly_rows: Vec<(usize, Vec<DynamicPolynomialF<F>>)> =
        cfg_into_iter!(0..num_rows - 1)
            .map(|row_idx| {
                let up = trace_matrix
                    .iter()
                    .map(|column| column[row_idx].clone())
                    .collect_vec();

                let down = trace_matrix
                    .iter()
                    .map(|column| column[row_idx + 1].clone())
                    .collect_vec();

                combine_rows_and_get_max_degree::<F, U>(
                    &up,
                    &down,
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
    max_degrees_and_combined_poly_rows
        .push((0, vec![DynamicPolynomialF::new([]); num_constraints]));

    prepare_coefficient_mles(
        num_constraints,
        max_degree,
        &max_degrees_and_combined_poly_rows,
        &field_zero,
    )
}

/// Apply combination polynomial to each row
/// and compute the maximum degree of resulting polynomials
/// to pad the resulting vector of MLEs accordingly.
#[allow(clippy::arithmetic_side_effects)]
fn combine_rows_and_get_max_degree<F, U>(
    up: &[DynamicPolynomialF<F>],
    down: &[DynamicPolynomialF<F>],
    num_constraints: usize,
    projected_scalars: &HashMap<U::Scalar, DynamicPolynomialF<F>>,
) -> (usize, Vec<DynamicPolynomialF<F>>)
where
    F: PrimeField,
    U: Uair,
{
    let mut constraint_builder = CombinedPolyRowBuilder::new(num_constraints);

    let project = |x: &U::Scalar| {
        projected_scalars
            .get(x)
            .cloned()
            .expect("all scalars should have been projected at this point")
    };

    U::constrain_general(
        &mut constraint_builder,
        TraceRow::from_slice_with_signature(up, &U::signature()),
        TraceRow::from_slice_with_signature(down, &U::signature()),
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
fn prepare_coefficient_mles<F: PrimeField>(
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

/// For linear UAIRs (max constraint degree ≤ 1), evaluate the combined
/// polynomials directly at the given evaluation point without constructing
/// intermediate combined polynomial MLEs.
///
/// Instead of computing constraint expressions row-by-row (which involves
/// `DynamicPolynomialF` arithmetic per row), this function:
/// 1. Splits each trace column into per-coefficient MLEs for "up" and "down"
///    views.
/// 2. Evaluates each coefficient MLE at the evaluation point.
/// 3. Reconstructs `DynamicPolynomialF<F>` per column from the evaluated
///    coefficients.
/// 4. Applies the linear constraint expressions once on the evaluated column
///    values.
#[allow(clippy::arithmetic_side_effects)]
pub fn evaluate_combined_polynomials_linear<F, U>(
    trace_matrix: &[DenseMultilinearExtension<DynamicPolynomialF<F>>],
    projected_scalars: &HashMap<U::Scalar, DynamicPolynomialF<F>>,
    evaluation_point: &[F],
    num_constraints: usize,
    field_cfg: &F::Config,
) -> Result<Vec<DynamicPolynomialF<F>>, EvaluationError>
where
    F: PrimeField + zinc_utils::inner_transparent_field::InnerTransparentField,
    U: Uair,
{
    let num_rows = trace_matrix[0].len();
    let inner_zero = F::zero_with_cfg(field_cfg).inner().clone();

    // Find max polynomial degree across all column entries.
    let max_coeff_len = trace_matrix
        .iter()
        .flat_map(|col| col.iter())
        .map(|poly| poly.coeffs.len())
        .max()
        .unwrap_or(0);

    // Evaluate "up" and "down" column MLEs at the evaluation point.
    // up[col][j]   = trace[col][j]     for j = 0..N-2, then 0 for j = N-1
    // down[col][j] = trace[col][j + 1] for j = 0..N-2, then 0 for j = N-1
    //
    // We split each column into per-coefficient MLEs over F::Inner
    // and evaluate them, avoiding row-by-row DynamicPolynomialF arithmetic.
    let column_evals: Vec<(DynamicPolynomialF<F>, DynamicPolynomialF<F>)> = cfg_iter!(
        trace_matrix
    )
    .map(|col| {
        let mut up_coeffs = Vec::with_capacity(max_coeff_len);
        let mut down_coeffs = Vec::with_capacity(max_coeff_len);

        for d in 0..max_coeff_len {
            // up coefficient MLE: col[j].coeffs[d] for j < N-1, zero for j = N-1
            let up_coeff_evals: Vec<_> = (0..num_rows)
                .map(|j| {
                    if j < num_rows - 1 {
                        col[j]
                            .coeffs
                            .get(d)
                            .map_or_else(|| inner_zero.clone(), |c| c.inner().clone())
                    } else {
                        inner_zero.clone()
                    }
                })
                .collect();

            let up_mle = DenseMultilinearExtension::from_evaluations_vec(
                evaluation_point.len(),
                up_coeff_evals,
                inner_zero.clone(),
            );
            up_coeffs.push(up_mle.evaluate_with_config(evaluation_point, field_cfg)?);

            // down coefficient MLE: col[j+1].coeffs[d] for j < N-1, zero for j = N-1
            let down_coeff_evals: Vec<_> = (0..num_rows)
                .map(|j| {
                    if j < num_rows - 1 {
                        col[j + 1]
                            .coeffs
                            .get(d)
                            .map_or_else(|| inner_zero.clone(), |c| c.inner().clone())
                    } else {
                        inner_zero.clone()
                    }
                })
                .collect();

            let down_mle = DenseMultilinearExtension::from_evaluations_vec(
                evaluation_point.len(),
                down_coeff_evals,
                inner_zero.clone(),
            );
            down_coeffs.push(down_mle.evaluate_with_config(evaluation_point, field_cfg)?);
        }

        Ok((
            DynamicPolynomialF::new_trimmed(up_coeffs),
            DynamicPolynomialF::new_trimmed(down_coeffs),
        ))
    })
    .collect::<Result<Vec<_>, _>>()?;

    let (up_evals, down_evals): (Vec<_>, Vec<_>) = column_evals.into_iter().unzip();

    // Apply constraint expressions to the evaluated column values.
    let mut builder = CombinedPolyRowBuilder::new(num_constraints);

    let project = |x: &U::Scalar| {
        projected_scalars
            .get(x)
            .cloned()
            .expect("all scalars should have been projected at this point")
    };

    U::constrain_general(
        &mut builder,
        TraceRow::from_slice_with_signature(&up_evals, &U::signature()),
        TraceRow::from_slice_with_signature(&down_evals, &U::signature()),
        &project,
        |x, y| Some(project(y) * x),
        ImpossibleIdeal::from_ref,
    );

    let mut combined_evaluations = builder.combined_evaluations;
    combined_evaluations.iter_mut().for_each(|eval| eval.trim());

    Ok(combined_evaluations)
}

pub struct CombinedPolyRowBuilder<F: PrimeField> {
    combined_evaluations: Vec<DynamicPolynomialF<F>>,
}

impl<F: PrimeField> ConstraintBuilder for CombinedPolyRowBuilder<F> {
    type Expr = DynamicPolynomialF<F>;
    type Ideal = ImpossibleIdeal;

    fn assert_in_ideal(&mut self, expr: Self::Expr, _ideal: &Self::Ideal) {
        self.combined_evaluations.push(expr);
    }

    fn assert_zero(&mut self, expr: Self::Expr) {
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
