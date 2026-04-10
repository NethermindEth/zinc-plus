#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::{
    projections::{ColumnMajorTrace, RowMajorTrace, ScalarMap},
    scalar_proj_cache::ScalarProjCache,
};
use crypto_primitives::PrimeField;
use num_traits::Zero;
use std::cell::RefCell;
use zinc_poly::{
    EvaluationError,
    mle::{
        DenseMultilinearExtension, MultilinearExtensionWithConfig, dense::CollectDenseMleWithZero,
    },
    univariate::dynamic::over_field::DynamicPolynomialF,
};
use zinc_uair::{
    BitOp, ColumnLayout, ConstraintBuilder, TraceRow, Uair,
    degree_counter::{count_constraint_degrees, count_effective_max_degree},
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
    projected_scalars: &ScalarMap<U::Scalar, DynamicPolynomialF<F>>,
    num_constraints: usize,
    field_cfg: &F::Config,
    skip_constraints: &[bool],
) -> Vec<Vec<DenseMultilinearExtension<F::Inner>>>
where
    F: PrimeField,
    U: Uair,
{
    let field_zero = F::zero_with_cfg(field_cfg);
    let uair_sig = U::signature();
    let down_layout = uair_sig.down_cols().as_column_layout();

    let num_rows = trace_matrix.len();

    let bit_op_count = uair_sig.bit_op_down_count();
    let mut max_degrees_and_combined_poly_rows: Vec<(usize, Vec<DynamicPolynomialF<F>>)> =
        cfg_into_iter!(0..num_rows - 1)
            .map(|row_idx| {
                let up = &trace_matrix[row_idx];

                // Row-shift virtual columns: lifted source value at row+shift.
                let mut down: Vec<DynamicPolynomialF<F>> = uair_sig
                    .shifts()
                    .iter()
                    .map(|spec| {
                        if row_idx + spec.shift_amount() < num_rows {
                            trace_matrix[row_idx + spec.shift_amount()][spec.source_col()].clone()
                        } else {
                            DynamicPolynomialF::zero() // zero padding
                        }
                    })
                    .collect();

                // Bit-op virtual columns: apply op (bit permutation on the
                // 32 coefficients) to the row's source F_q[X] cell. Output
                // appended after the row-shift down evals — the constraint
                // builder reads them via `down.bit_op[k]`.
                for spec in uair_sig.bit_op_specs() {
                    let src_cell = &trace_matrix[row_idx][spec.source_col()];
                    let op_cell = match spec.op() {
                        BitOp::Rot(c) => src_cell.rot_c(c),
                        BitOp::ShiftR(c) => src_cell.shift_r_c(c),
                    };
                    down.push(op_cell);
                }

                combine_rows_and_get_max_degree::<F, U>(
                    up,
                    &down,
                    num_constraints,
                    projected_scalars,
                    down_layout,
                    bit_op_count,
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
    max_degrees_and_combined_poly_rows.push((0, vec![DynamicPolynomialF::zero(); num_constraints]));

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
    projected_scalars: &ScalarMap<U::Scalar, DynamicPolynomialF<F>>,
    down_layout: &ColumnLayout,
    bit_op_count: usize,
) -> (usize, Vec<DynamicPolynomialF<F>>)
where
    F: PrimeField,
    U: Uair,
{
    let mut constraint_builder = CombinedPolyRowBuilder::new(num_constraints);

    // Per-call cache — see `scalar_proj_cache`. This site is called once per
    // trace row (2^num_vars - 1 times per proof) and projects scalars into a
    // `DynamicPolynomialF` (wider than just F), so the HashMap lookup is
    // expensive enough that even modest reuse pays for the cache. Lazy init
    // keeps the overhead at zero for UAIRs that never call project.
    let cache: RefCell<Option<ScalarProjCache<U::Scalar, DynamicPolynomialF<F>>>> =
        RefCell::new(None);
    let project = |x: &U::Scalar| -> DynamicPolynomialF<F> {
        if let Some(v) = cache.borrow().as_ref().and_then(|c| c.get(x)) {
            return v;
        }
        let v = projected_scalars
            .get(x)
            .cloned()
            .expect("all scalars should have been projected at this point");
        cache
            .borrow_mut()
            .get_or_insert_with(ScalarProjCache::new)
            .push(x, v.clone());
        v
    };

    U::constrain_general(
        &mut constraint_builder,
        TraceRow::from_slice_with_layout(up, U::signature().total_cols().as_column_layout()),
        TraceRow::from_slice_with_layout_and_bit_op(down, down_layout, bit_op_count),
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
    zero_as_field_inner: &F::Inner,
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
            (0..=max_degree)
                .map(|coeff| {
                    max_degrees_and_combined_poly_rows
                        .iter()
                        .map(|(_, row)| {
                            if coeff >= row[constraint].coeffs.len() {
                                zero_as_field_inner.clone()
                            } else {
                                row[constraint].coeffs[coeff].inner().clone()
                            }
                        })
                        .collect_dense_mle_with_zero(zero_as_field_inner)
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
/// Does `(num_columns + num_shifted_columns) * max_num_coeffs` evaluations of
/// MLEs.
///
/// Validates that the UAIR is fully linear (any non-zero-ideal constraint
/// has degree ≤ 1). For mixed-degree UAIRs, use
/// [`evaluate_combined_polynomials_unchecked`] and have the caller
/// guarantee that values for non-linear slots are discarded — see
/// `IdealCheckProtocol::prove_hybrid`.
pub fn evaluate_combined_polynomials<F, U>(
    trace_matrix: &ColumnMajorTrace<F>,
    projected_scalars: &ScalarMap<U::Scalar, DynamicPolynomialF<F>>,
    num_constraints: usize,
    evaluation_point: &[F],
    field_cfg: &F::Config,
) -> Result<Vec<DynamicPolynomialF<F>>, EvaluationError>
where
    F: InnerTransparentField,
    U: Uair,
{
    if count_effective_max_degree::<U>() > 1 {
        return Err(EvaluationError::UnsupportedConstraintDegrees {
            degrees: count_constraint_degrees::<U>(),
        });
    }
    evaluate_combined_polynomials_unchecked::<F, U>(
        trace_matrix,
        projected_scalars,
        num_constraints,
        evaluation_point,
        field_cfg,
    )
}

/// Like [`evaluate_combined_polynomials`] but skips the global linearity
/// check. The caller is responsible for guaranteeing that values returned
/// for any non-linear, non-zero-ideal slot are discarded — they would
/// otherwise corrupt the transcript and break soundness.
///
/// Used by the hybrid ideal-check dispatch to compute MLE-first values for
/// the linear subset of a mixed-degree UAIR; the non-linear subset's
/// values are computed via [`compute_combined_polynomials`] and overwrite
/// the corresponding slots before transcript absorption.
#[allow(clippy::arithmetic_side_effects)]
pub fn evaluate_combined_polynomials_unchecked<F, U>(
    trace_matrix: &ColumnMajorTrace<F>,
    projected_scalars: &ScalarMap<U::Scalar, DynamicPolynomialF<F>>,
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

    // Maximum number of coefficients across all trace entries
    let max_num_coeffs = trace_matrix
        .iter()
        .flat_map(|col| col.evaluations.iter())
        .map(|p| p.coeffs.len())
        .max()
        .unwrap_or(0);

    let uair_sig = U::signature();
    let down_layout = uair_sig.down_cols().as_column_layout();

    // Helper: evaluate one column's coefficient-d MLE at `evaluation_point`,
    // reading row `i + shift` (zero-padded beyond trace length).
    let eval_coeff_mle = |col: &DenseMultilinearExtension<DynamicPolynomialF<F>>,
                          d: usize,
                          shift: usize|
     -> Result<F, EvaluationError> {
        let coeff_evals: Vec<F::Inner> = (0..num_rows)
            .map(|i| {
                // Two conditions needed:
                // 1. i < num_rows - 1: zero out the last row for all columns (both up and down)
                //    to match the combined poly builder's explicit zero-padding at row N-1.
                // 2. i + shift < num_rows: prevent OOB access for shifts > 0.
                if i < num_rows - 1 && i + shift < num_rows {
                    col.evaluations[i + shift]
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
        coeff_mle.evaluate_with_config(evaluation_point, field_cfg)
    };

    // Evaluate up (all columns, shift=0).
    let up_evals: Vec<DynamicPolynomialF<F>> = cfg_iter!(trace_matrix)
        .map(|col| {
            let coeffs: Vec<F> = (0..max_num_coeffs)
                .map(|d| eval_coeff_mle(col, d, 0))
                .collect::<Result<_, _>>()?;
            Ok(DynamicPolynomialF::new_trimmed(coeffs))
        })
        .collect::<Result<Vec<_>, EvaluationError>>()?;

    // Evaluate down (only shifted columns, per-spec shift amount).
    let sorted_shifts = uair_sig.shifts();
    let mut down_evals: Vec<DynamicPolynomialF<F>> = cfg_iter!(sorted_shifts)
        .map(|spec| {
            let col = &trace_matrix[spec.source_col()];
            let coeffs: Vec<F> = (0..max_num_coeffs)
                .map(|d| eval_coeff_mle(col, d, spec.shift_amount()))
                .collect::<Result<_, _>>()?;
            Ok(DynamicPolynomialF::new_trimmed(coeffs))
        })
        .collect::<Result<Vec<_>, EvaluationError>>()?;

    // Bit-op virtual columns: apply op to the source's already-evaluated
    // up-eval (`up_evals[source_col]`). MLE[op(col)](point) =
    // op(MLE[col](point)) for any bit permutation `op`.
    let bit_op_count = uair_sig.bit_op_down_count();
    for spec in uair_sig.bit_op_specs() {
        let src = &up_evals[spec.source_col()];
        let mut op_eval = match spec.op() {
            BitOp::Rot(c) => src.rot_c(c),
            BitOp::ShiftR(c) => src.shift_r_c(c),
        };
        op_eval.trim();
        down_evals.push(op_eval);
    }

    // Apply UAIR constraints to the evaluated trace values
    let mut constraint_builder = CombinedPolyRowBuilder::new(num_constraints);

    // See `scalar_proj_cache` module. This site is called once (linear-UAIR
    // fast path), so the win is purely within-call dedup.
    let cache: RefCell<Option<ScalarProjCache<U::Scalar, DynamicPolynomialF<F>>>> =
        RefCell::new(None);
    let project = |x: &U::Scalar| -> DynamicPolynomialF<F> {
        if let Some(v) = cache.borrow().as_ref().and_then(|c| c.get(x)) {
            return v;
        }
        let v = projected_scalars
            .get(x)
            .cloned()
            .expect("all scalars should have been projected at this point");
        cache
            .borrow_mut()
            .get_or_insert_with(ScalarProjCache::new)
            .push(x, v.clone());
        v
    };

    U::constrain_general(
        &mut constraint_builder,
        TraceRow::from_slice_with_layout(&up_evals, uair_sig.total_cols().as_column_layout()),
        TraceRow::from_slice_with_layout_and_bit_op(&down_evals, down_layout, bit_op_count),
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
