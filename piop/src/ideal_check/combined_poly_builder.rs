#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::projections::{ColumnMajorTrace, RowMajorTrace};
use crypto_primitives::PrimeField;
use itertools::Itertools;
use std::collections::HashMap;
use zinc_poly::{
    EvaluationError,
    mle::{
        DenseMultilinearExtension, MultilinearExtensionWithConfig, dense::CollectDenseMleWithZero,
    },
    univariate::dynamic::over_field::DynamicPolynomialF,
};
use zinc_uair::{
    ConstraintBuilder, TraceRow, Uair,
    degree_counter::{count_constraint_degrees, count_max_degree},
    ideal::ImpossibleIdeal,
};
use zinc_utils::{
    cfg_into_iter, cfg_iter, from_ref::FromRef, inner_transparent_field::InnerTransparentField,
};

/// Given a UAIR `U` and a trace `trace` this function
/// obtains the combined polynomials' MLE coefficients.
/// Since each coefficient is also a univariate polynomial
/// we split the resulting MLE into coefficient MLEs.
///
/// `trace_matrix` is row-indexed: `trace_matrix[row][col]`.
#[allow(clippy::arithmetic_side_effects)]
pub fn compute_combined_polynomials<F, U>(
    trace_matrix: &RowMajorTrace<F>,
    projected_scalars: &HashMap<U::Scalar, DynamicPolynomialF<F>>,
    num_constraints: usize,
    field_cfg: &F::Config,
    skip_constraints: &[bool],
) -> Vec<Vec<DenseMultilinearExtension<F::Inner>>>
where
    F: PrimeField,
    U: Uair,
{
    let field_zero = F::zero_with_cfg(field_cfg);

    let num_rows = trace_matrix.len();

    let mut max_degrees_and_combined_poly_rows: Vec<(usize, Vec<DynamicPolynomialF<F>>)> =
        cfg_into_iter!(0..num_rows - 1)
            .map(|row_idx| {
                let up = &trace_matrix[row_idx];
                let down = &trace_matrix[row_idx + 1];

                combine_rows_and_get_max_degree::<F, U>(
                    up,
                    down,
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
        field_zero.inner(),
        skip_constraints,
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
    field_zero_inner: &F::Inner,
    skip_constraints: &[bool],
) -> Vec<Vec<DenseMultilinearExtension<F::Inner>>> {
    cfg_into_iter!(0..num_constraints)
        .map(|constraint| {
            // Skip building coefficient MLEs for zero-ideal constraints.
            // For an honest prover these MLEs are zero; the combined
            // polynomial resolver handles the zero entries downstream.
            if skip_constraints[constraint] {
                return vec![];
            }
            cfg_into_iter!(0..=max_degree)
                .map(|coeff| {
                    max_degrees_and_combined_poly_rows
                        .iter()
                        .map(|(_, row)| {
                            if coeff >= row[constraint].coeffs.len() {
                                field_zero_inner.clone()
                            } else {
                                row[constraint].coeffs[coeff].inner().clone()
                            }
                        })
                        .collect_dense_mle_with_zero(field_zero_inner)
                })
                .collect()
        })
        .collect()
}

/// For linear UAIRs, evaluate combined polynomials directly
/// by first evaluating trace column MLEs at the evaluation point,
/// then applying UAIR constraints to the evaluated values.
///
/// This avoids building the full combined polynomial MLEs row by row
/// and is more efficient for linear constraints because the evaluation
/// of a linear combination of MLEs equals the linear combination of
/// individual MLE evaluations.
///
/// `trace_matrix` is column-indexed: `trace_matrix[col]` is an MLE.
///
/// Does `2 * num_columns * max_num_coeffs` evaluations of MLEs.
#[allow(clippy::arithmetic_side_effects)]
pub fn evaluate_combined_polynomials<F, U>(
    trace_matrix: &ColumnMajorTrace<F>,
    projected_scalars: &HashMap<U::Scalar, DynamicPolynomialF<F>>,
    num_constraints: usize,
    evaluation_point: &[F],
    field_cfg: &F::Config,
) -> Result<Vec<DynamicPolynomialF<F>>, EvaluationError>
where
    F: InnerTransparentField,
    U: Uair,
{
    let field_zero = F::zero_with_cfg(field_cfg);
    let zero_inner = field_zero.inner().clone();
    let num_rows = trace_matrix.first().map(|c| c.len()).unwrap_or(0);
    let num_vars = evaluation_point.len();

    // Sanity check: this approach only works for linear constraints
    if count_max_degree::<U>() > 1 {
        return Err(EvaluationError::UnsupportedConstraintDegrees {
            degrees: count_constraint_degrees::<U>(),
        });
    }

    // Maximum number of coefficients across all trace entries
    let max_num_coeffs = trace_matrix
        .iter()
        .flat_map(|col| col.evaluations.iter())
        .map(|p| p.coeffs.len())
        .max()
        .unwrap_or(0);

    // Evaluate "up" and "down" versions of each trace column at the evaluation
    // point, these match the semantics of compute_combined_polynomials.
    // With column-indexed format, each column is already an MLE.
    let column_evals: Vec<(DynamicPolynomialF<F>, DynamicPolynomialF<F>)> = cfg_iter!(trace_matrix)
        .map(|col| {
            let mut up_coeffs = Vec::with_capacity(max_num_coeffs);
            let mut down_coeffs = Vec::with_capacity(max_num_coeffs);

            for d in 0..max_num_coeffs {
                macro_rules! make_coeff_evals {
                    ($coeffs_vec:ident, $row_shift:literal) => {
                        let coeff_evals: Vec<F::Inner> = (0..num_rows)
                            .map(|i| {
                                if i < num_rows - 1 {
                                    col.evaluations[i + $row_shift]
                                        .coeffs
                                        .get(d)
                                        .map(|c| c.inner().clone())
                                        .unwrap_or_else(|| zero_inner.clone())
                                } else {
                                    zero_inner.clone()
                                }
                            })
                            .collect();

                        let coeff_mle = DenseMultilinearExtension {
                            evaluations: coeff_evals,
                            num_vars,
                        };
                        $coeffs_vec
                            .push(coeff_mle.evaluate_with_config(evaluation_point, field_cfg)?);
                    };
                }

                make_coeff_evals!(up_coeffs, 0);
                make_coeff_evals!(down_coeffs, 1);
            }

            Ok((
                DynamicPolynomialF::new_trimmed(up_coeffs),
                DynamicPolynomialF::new_trimmed(down_coeffs),
            ))
        })
        .collect::<Result<Vec<_>, EvaluationError>>()?;

    let (up_evals, down_evals): (Vec<_>, Vec<_>) = column_evals.into_iter().unzip();

    // Apply UAIR constraints to the evaluated trace values
    let mut constraint_builder = CombinedPolyRowBuilder::new(num_constraints);

    let project = |x: &U::Scalar| {
        projected_scalars
            .get(x)
            .cloned()
            .expect("all scalars should have been projected at this point")
    };

    U::constrain_general(
        &mut constraint_builder,
        TraceRow::from_slice_with_signature(&up_evals, &U::signature()),
        TraceRow::from_slice_with_signature(&down_evals, &U::signature()),
        &project,
        |x, y| Some(project(y) * x),
        ImpossibleIdeal::from_ref,
    );

    let mut combined_evaluations = constraint_builder.combined_evaluations;
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
