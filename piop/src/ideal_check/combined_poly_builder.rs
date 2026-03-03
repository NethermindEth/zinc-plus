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
    univariate::{binary::BinaryPoly, dynamic::over_field::DynamicPolynomialF},
    utils::build_eq_x_r_inner,
};
use zinc_uair::{ConstraintBuilder, TraceRow, Uair, UairSignature, ideal::ImpossibleIdeal};
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
    let sig = U::signature();
    let down_sig = sig.down_signature();

    let mut max_degrees_and_combined_poly_rows: Vec<(usize, Vec<DynamicPolynomialF<F>>)> =
        cfg_into_iter!(0..num_rows - 1)
            .map(|row_idx| {
                let up = trace_matrix
                    .iter()
                    .map(|column| column[row_idx].clone())
                    .collect_vec();

                let down = if sig.uses_legacy_shifts() {
                    // Legacy: all columns shifted by 1 (look-ahead).
                    trace_matrix
                        .iter()
                        .map(|column| column[row_idx + 1].clone())
                        .collect_vec()
                } else {
                    // Shift-spec mode: construct only the declared shifted
                    // columns.
                    sig.shifts
                        .iter()
                        .map(|spec| {
                            let target = row_idx + spec.shift_amount;
                            if target < num_rows {
                                trace_matrix[spec.source_col][target].clone()
                            } else {
                                DynamicPolynomialF::new([])
                            }
                        })
                        .collect_vec()
                };

                combine_rows_and_get_max_degree::<F, U>(
                    &up,
                    &down,
                    num_constraints,
                    projected_scalars,
                    &sig,
                    &down_sig,
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
    sig: &UairSignature,
    down_sig: &UairSignature,
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
        TraceRow::from_slice_with_signature(up, sig),
        TraceRow::from_slice_with_signature(down, down_sig),
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

// ═══════════════════════════════════════════════════════════════════════
// MLE-first path (linear constraints only)
// ═══════════════════════════════════════════════════════════════════════

/// MLE-first computation of combined polynomial values for linear UAIRs
/// with binary-polynomial trace columns.
///
/// Instead of projecting the full trace to `DynamicPolynomialF<F>` and then
/// evaluating constraints row-by-row, this function:
///
/// 1. Evaluates each column MLE at the random point by processing one
///    polynomial-coefficient position at a time (avoiding full trace
///    projection).
/// 2. Evaluates shifted ("down") column MLEs similarly.
/// 3. Applies the UAIR constraints once on the resulting values.
/// 4. Subtracts the last-row contribution (selector correction).
///
/// This is valid **only** when the UAIR constraints are linear
/// (`max_degree == 1`).  The result is identical to
/// [`compute_combined_polynomials`] followed by coefficient-MLE
/// evaluation.
#[allow(clippy::arithmetic_side_effects)]
pub fn compute_combined_values_mle_first<F, U, const D: usize>(
    binary_poly_trace: &[DenseMultilinearExtension<BinaryPoly<D>>],
    projected_scalars: &HashMap<U::Scalar, DynamicPolynomialF<F>>,
    evaluation_point: &[F],
    num_constraints: usize,
    field_cfg: &F::Config,
) -> Vec<DynamicPolynomialF<F>>
where
    F: PrimeField + zinc_utils::inner_transparent_field::InnerTransparentField,
    F::Inner: Default + Send + Sync + num_traits::Zero,
    U: Uair,
{
    let sig = U::signature();
    let down_sig = sig.down_signature();
    let num_rows = binary_poly_trace[0].len();
    let num_vars = evaluation_point.len();

    // Pre-compute eq(r, ·) table once; used for all MLE evaluations below.
    let eq_table = build_eq_x_r_inner::<F>(evaluation_point, field_cfg)
        .expect("build_eq_x_r_inner should succeed");
    let eq_evals = &eq_table.evaluations;

    // ── Step 1. Evaluate "up" column MLEs at point ──────────────────
    let up_values: Vec<DynamicPolynomialF<F>> = cfg_iter!(binary_poly_trace)
        .map(|col| {
            evaluate_binary_poly_column_mle_with_eq::<F, D>(col, eq_evals, field_cfg)
        })
        .collect();

    // ── Step 2. Evaluate "down" (shifted) column MLEs at point ──────
    let down_values: Vec<DynamicPolynomialF<F>> = if sig.uses_legacy_shifts() {
        cfg_iter!(binary_poly_trace)
            .map(|col| {
                evaluate_shifted_binary_poly_column_mle_with_eq::<F, D>(
                    col, 1, eq_evals, num_vars, field_cfg,
                )
            })
            .collect()
    } else {
        cfg_iter!(sig.shifts)
            .map(|spec| {
                evaluate_shifted_binary_poly_column_mle_with_eq::<F, D>(
                    &binary_poly_trace[spec.source_col],
                    spec.shift_amount,
                    eq_evals,
                    num_vars,
                    field_cfg,
                )
            })
            .collect()
    };

    // ── Step 3. Compute L_{N-1}(r) = eq(r, (1,…,1)) = ∏ r_i ───────
    let one = F::one_with_cfg(field_cfg);
    let zero = F::zero_with_cfg(field_cfg);
    let last_row_lagrange: F =
        evaluation_point.iter().fold(one.clone(), |acc, r_i| acc * r_i);

    // ── Step 4. Project last-row native values for correction ───────
    let last_row_up: Vec<DynamicPolynomialF<F>> = binary_poly_trace
        .iter()
        .map(|col| project_binary_poly_to_dynamic(&col[num_rows - 1], &one, &zero))
        .collect();

    let last_row_down: Vec<DynamicPolynomialF<F>> = if sig.uses_legacy_shifts() {
        // Legacy: down(N-1) would need trace[c][N], out of bounds → zero.
        vec![DynamicPolynomialF::new([]); binary_poly_trace.len()]
    } else {
        sig.shifts
            .iter()
            .map(|spec| {
                let target = (num_rows - 1) + spec.shift_amount;
                if target < num_rows {
                    project_binary_poly_to_dynamic(
                        &binary_poly_trace[spec.source_col][target],
                        &one,
                        &zero,
                    )
                } else {
                    DynamicPolynomialF::new([])
                }
            })
            .collect()
    };

    // ── Step 5. Apply constraints on full MLE-evaluated values ──────
    let mut full_builder = CombinedPolyRowBuilder::new(num_constraints);
    {
        let project = |x: &U::Scalar| {
            projected_scalars
                .get(x)
                .cloned()
                .expect("all scalars should have been projected at this point")
        };
        U::constrain_general(
            &mut full_builder,
            TraceRow::from_slice_with_signature(&up_values, &sig),
            TraceRow::from_slice_with_signature(&down_values, &down_sig),
            &project,
            |x, y| Some(project(y) * x),
            ImpossibleIdeal::from_ref,
        );
    }

    // ── Step 6. Apply constraints on last-row values ────────────────
    let mut last_row_builder = CombinedPolyRowBuilder::new(num_constraints);
    {
        let project = |x: &U::Scalar| {
            projected_scalars
                .get(x)
                .cloned()
                .expect("all scalars should have been projected at this point")
        };
        U::constrain_general(
            &mut last_row_builder,
            TraceRow::from_slice_with_signature(&last_row_up, &sig),
            TraceRow::from_slice_with_signature(&last_row_down, &down_sig),
            &project,
            |x, y| Some(project(y) * x),
            ImpossibleIdeal::from_ref,
        );
    }

    // ── Step 7. Correct: C̄_j(r) = full_j − L_{N-1}(r) · last_j ───
    full_builder
        .combined_evaluations
        .into_iter()
        .zip(last_row_builder.combined_evaluations)
        .map(|(mut full, last)| {
            // Scalar-multiply each coefficient of `last` by L_{N-1}(r).
            let correction = DynamicPolynomialF {
                coeffs: last
                    .coeffs
                    .iter()
                    .map(|c| c.clone() * &last_row_lagrange)
                    .collect(),
            };
            full -= &correction;
            full.trim();
            full
        })
        .collect()
}

/// Evaluate a `DenseMultilinearExtension<BinaryPoly<D>>` at a field
/// point by processing one polynomial-coefficient position at a time.
///
/// For each coefficient position `d` in `0..D`, extracts the boolean
/// MLE (entries are 0 or 1 in `F`), evaluates at `point`, and
/// assembles the results into a `DynamicPolynomialF<F>`.
fn evaluate_binary_poly_column_mle<F, const D: usize>(
    column: &DenseMultilinearExtension<BinaryPoly<D>>,
    point: &[F],
    field_cfg: &F::Config,
) -> DynamicPolynomialF<F>
where
    F: PrimeField + zinc_utils::inner_transparent_field::InnerTransparentField,
    F::Inner: Default + Send + Sync,
{
    let zero_inner = F::zero_with_cfg(field_cfg).inner().clone();
    let one_inner = F::one_with_cfg(field_cfg).inner().clone();
    let n = column.len();
    let num_vars = point.len();

    // Transpose: for each coefficient position d, collect boolean values
    // across all rows into a single MLE over F::Inner.
    let mut coeff_buffers: Vec<Vec<F::Inner>> =
        (0..D).map(|_| Vec::with_capacity(n)).collect();

    for b in 0..n {
        for (d, bit) in column[b].iter().enumerate() {
            coeff_buffers[d].push(if bit.into_inner() {
                one_inner.clone()
            } else {
                zero_inner.clone()
            });
        }
    }

    let coeffs: Vec<F> = coeff_buffers
        .into_iter()
        .map(|evals| {
            let mle = DenseMultilinearExtension::from_evaluations_vec(
                num_vars,
                evals,
                zero_inner.clone(),
            );
            mle.evaluate_with_config(point, field_cfg)
                .expect("MLE evaluation should succeed")
        })
        .collect();

    DynamicPolynomialF::new_trimmed(coeffs)
}

/// Like [`evaluate_binary_poly_column_mle`], but for a shifted column:
/// entry `b` is `column[b + shift_amount]` when in bounds, else zero.
fn evaluate_shifted_binary_poly_column_mle<F, const D: usize>(
    column: &DenseMultilinearExtension<BinaryPoly<D>>,
    shift_amount: usize,
    point: &[F],
    field_cfg: &F::Config,
) -> DynamicPolynomialF<F>
where
    F: PrimeField + zinc_utils::inner_transparent_field::InnerTransparentField,
    F::Inner: Default + Send + Sync,
{
    let zero_inner = F::zero_with_cfg(field_cfg).inner().clone();
    let one_inner = F::one_with_cfg(field_cfg).inner().clone();
    let num_vars = point.len();
    let n = 1usize << num_vars;

    let mut coeff_buffers: Vec<Vec<F::Inner>> =
        (0..D).map(|_| Vec::with_capacity(n)).collect();

    for b in 0..n {
        let target = b + shift_amount;
        if target < column.len() {
            for (d, bit) in column[target].iter().enumerate() {
                coeff_buffers[d].push(if bit.into_inner() {
                    one_inner.clone()
                } else {
                    zero_inner.clone()
                });
            }
        } else {
            for d in 0..D {
                coeff_buffers[d].push(zero_inner.clone());
            }
        }
    }

    let coeffs: Vec<F> = coeff_buffers
        .into_iter()
        .map(|evals| {
            let mle = DenseMultilinearExtension::from_evaluations_vec(
                num_vars,
                evals,
                zero_inner.clone(),
            );
            mle.evaluate_with_config(point, field_cfg)
                .expect("MLE evaluation should succeed")
        })
        .collect();

    DynamicPolynomialF::new_trimmed(coeffs)
}

/// Like [`evaluate_binary_poly_column_mle`], but uses a precomputed
/// `eq(r, ·)` table to evaluate each coefficient MLE via a single
/// conditional-addition pass instead of the O(n)-folding approach.
///
/// For D coefficient positions, the result is:
///   coeff_d = Σ_{j : bit_d(column[j]) = true} eq_table[j]
///
/// This avoids D independent MLE folding passes (each cloning and
/// halving a full buffer) and replaces them with D running sums
/// accumulated in one pass over the column.
fn evaluate_binary_poly_column_mle_with_eq<F, const D: usize>(
    column: &DenseMultilinearExtension<BinaryPoly<D>>,
    eq_table: &[F::Inner],
    field_cfg: &F::Config,
) -> DynamicPolynomialF<F>
where
    F: PrimeField + zinc_utils::inner_transparent_field::InnerTransparentField,
    F::Inner: Default + Send + Sync,
{
    let n = column.len();
    let zero_inner = F::zero_with_cfg(field_cfg).inner().clone();
    let mut coeff_sums: Vec<F::Inner> = vec![zero_inner; D];

    for j in 0..n {
        for (d, bit) in column[j].iter().enumerate() {
            if bit.into_inner() {
                coeff_sums[d] = F::add_inner(&coeff_sums[d], &eq_table[j], field_cfg);
            }
        }
    }

    let coeffs: Vec<F> = coeff_sums
        .into_iter()
        .map(|s| F::new_unchecked_with_cfg(s, field_cfg))
        .collect();

    DynamicPolynomialF::new_trimmed(coeffs)
}

/// Like [`evaluate_shifted_binary_poly_column_mle`], but uses a
/// precomputed `eq(r, ·)` table for efficient evaluation.
///
/// Entry `b` is `column[b + shift_amount]` when in bounds, else zero.
fn evaluate_shifted_binary_poly_column_mle_with_eq<F, const D: usize>(
    column: &DenseMultilinearExtension<BinaryPoly<D>>,
    shift_amount: usize,
    eq_table: &[F::Inner],
    num_vars: usize,
    field_cfg: &F::Config,
) -> DynamicPolynomialF<F>
where
    F: PrimeField + zinc_utils::inner_transparent_field::InnerTransparentField,
    F::Inner: Default + Send + Sync,
{
    let n = 1usize << num_vars;
    let zero_inner = F::zero_with_cfg(field_cfg).inner().clone();
    let mut coeff_sums: Vec<F::Inner> = vec![zero_inner; D];

    for j in 0..n {
        let target = j + shift_amount;
        if target < column.len() {
            for (d, bit) in column[target].iter().enumerate() {
                if bit.into_inner() {
                    coeff_sums[d] = F::add_inner(&coeff_sums[d], &eq_table[j], field_cfg);
                }
            }
        }
    }

    let coeffs: Vec<F> = coeff_sums
        .into_iter()
        .map(|s| F::new_unchecked_with_cfg(s, field_cfg))
        .collect();

    DynamicPolynomialF::new_trimmed(coeffs)
}

/// Project a single `BinaryPoly<D>` into `DynamicPolynomialF<F>`.
fn project_binary_poly_to_dynamic<F: PrimeField, const D: usize>(
    bp: &BinaryPoly<D>,
    one: &F,
    zero: &F,
) -> DynamicPolynomialF<F> {
    DynamicPolynomialF::new_trimmed(
        bp.iter()
            .map(|b| {
                if b.into_inner() {
                    one.clone()
                } else {
                    zero.clone()
                }
            })
            .collect::<Vec<_>>(),
    )
}
