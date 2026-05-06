#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::{
    projections::{ColumnMajorTrace, RowMajorTrace, ScalarMap},
    scalar_proj_cache::ScalarProjCache,
};
use crypto_primitives::PrimeField;
use num_traits::{ConstZero, Zero};
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
    compute_combined_polynomials_timed::<F, U>(
        trace_matrix,
        projected_scalars,
        num_constraints,
        field_cfg,
        skip_constraints,
        None,
    )
}

/// Sub-timings within [`compute_combined_polynomials`].
#[derive(Clone, Copy, Default, Debug)]
pub struct ComputeCombinedSubTimings {
    /// Phase 1: per-row `U::constrain_general` walk (parallel) producing
    /// the per-row combined polynomial values. This is the expensive bit
    /// for non-linear / per-row paths.
    pub per_row_constrain: std::time::Duration,
    /// Phase 2: build per-coefficient MLEs from the per-row outputs,
    /// honoring `skip_constraints`.
    pub prepare_coeff_mles: std::time::Duration,
}

/// Like [`compute_combined_polynomials`] but optionally records the
/// time spent in the per-row `constrain_general` walk vs. the
/// coefficient-MLE preparation. Used by the dual-prime Z-branch bench
/// to attribute IC time to the per-row constraint evaluation.
#[allow(clippy::arithmetic_side_effects, clippy::too_many_arguments)]
pub fn compute_combined_polynomials_timed<F, U>(
    trace_matrix: &RowMajorTrace<F>,
    projected_scalars: &ScalarMap<U::Scalar, DynamicPolynomialF<F>>,
    num_constraints: usize,
    field_cfg: &F::Config,
    skip_constraints: &[bool],
    sub_timings: Option<&mut ComputeCombinedSubTimings>,
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
    let _t_phase1 = std::time::Instant::now();
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
    let dt_phase1 = _t_phase1.elapsed();

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

    let _t_phase2 = std::time::Instant::now();
    let result = prepare_coefficient_mles(
        num_constraints,
        max_degree,
        &max_degrees_and_combined_poly_rows,
        field_zero.inner(),
        skip_constraints,
    );
    let dt_phase2 = _t_phase2.elapsed();

    if let Some(t) = sub_timings {
        t.per_row_constrain = dt_phase1;
        t.prepare_coeff_mles = dt_phase2;
    }

    result
}

/// Z-only variant of [`compute_combined_polynomials`]. Uses
/// [`CombinedPolyRowBuilderZOnly`] so each row's `U::constrain_general`
/// call only walks the Z-tagged sub-graph (assuming the UAIR gates its
/// F_p / zero-ideal sections via [`ConstraintBuilder::is_active_for`]).
///
/// The output shape matches [`compute_combined_polynomials`] with the
/// implicit skip set `{i : tags[i] != Z || ideals[i].is_zero_ideal()}`:
/// per-coefficient MLEs are produced only for Z-tagged non-zero-ideal
/// slots, and `ZERO` polynomials elsewhere.
#[allow(clippy::arithmetic_side_effects, clippy::too_many_arguments)]
pub fn compute_combined_polynomials_z_only<F, U>(
    trace_matrix: &RowMajorTrace<F>,
    projected_scalars: &ScalarMap<U::Scalar, DynamicPolynomialF<F>>,
    num_constraints: usize,
    field_cfg: &F::Config,
    z_indices: &[usize],
    sub_timings: Option<&mut ComputeCombinedSubTimings>,
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

    // Skip mask covers everything except Z-tagged non-zero-ideal slots.
    let skip: Vec<bool> = {
        let mut s = vec![true; num_constraints];
        for &i in z_indices {
            s[i] = false;
        }
        s
    };

    let _t_phase1 = std::time::Instant::now();
    let mut max_degrees_and_combined_poly_rows: Vec<(usize, Vec<DynamicPolynomialF<F>>)> =
        cfg_into_iter!(0..num_rows - 1)
            .map(|row_idx| {
                let up = &trace_matrix[row_idx];

                let mut down: Vec<DynamicPolynomialF<F>> = uair_sig
                    .shifts()
                    .iter()
                    .map(|spec| {
                        if row_idx + spec.shift_amount() < num_rows {
                            trace_matrix[row_idx + spec.shift_amount()][spec.source_col()].clone()
                        } else {
                            DynamicPolynomialF::zero()
                        }
                    })
                    .collect();

                for spec in uair_sig.bit_op_specs() {
                    let src_cell = &trace_matrix[row_idx][spec.source_col()];
                    let op_cell = match spec.op() {
                        BitOp::Rot(c) => src_cell.rot_c(c),
                        BitOp::ShiftR(c) => src_cell.shift_r_c(c),
                    };
                    down.push(op_cell);
                }

                let mut builder = CombinedPolyRowBuilderZOnly::new(num_constraints, z_indices);

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
                    &mut builder,
                    TraceRow::from_slice_with_layout(up, U::signature().total_cols().as_column_layout()),
                    TraceRow::from_slice_with_layout_and_bit_op(&down, down_layout, bit_op_count),
                    &project,
                    |x, y| Some(project(y) * x),
                    ImpossibleIdeal::from_ref,
                );

                let mut combined_evaluations = builder.combined_evaluations;
                combined_evaluations.iter_mut().for_each(|eval| eval.trim());

                let max_degree = z_indices
                    .iter()
                    .map(|&i| combined_evaluations[i].degree().unwrap_or(0))
                    .max()
                    .unwrap_or(0);

                (max_degree, combined_evaluations)
            })
            .collect();
    let dt_phase1 = _t_phase1.elapsed();

    let max_degree = *max_degrees_and_combined_poly_rows
        .iter()
        .map(|(max_degree, _)| max_degree)
        .max()
        .expect("Z-branch IC must have at least one row");

    max_degrees_and_combined_poly_rows.push((0, vec![DynamicPolynomialF::zero(); num_constraints]));

    let _t_phase2 = std::time::Instant::now();
    let result = prepare_coefficient_mles(
        num_constraints,
        max_degree,
        &max_degrees_and_combined_poly_rows,
        field_zero.inner(),
        &skip,
    );
    let dt_phase2 = _t_phase2.elapsed();

    if let Some(t) = sub_timings {
        t.per_row_constrain = dt_phase1;
        t.prepare_coeff_mles = dt_phase2;
    }

    result
}

/// F_p-only mirror of [`compute_combined_polynomials_z_only`]. Uses
/// [`CombinedPolyRowBuilderFpOnly`] so each row's
/// `U::constrain_general` call only walks the F_p-tagged sub-graph
/// (assuming the UAIR gates its Z / zero-ideal sections via
/// [`ConstraintBuilder::is_active_for`]).
#[allow(clippy::arithmetic_side_effects, clippy::too_many_arguments)]
pub fn compute_combined_polynomials_fp_only<F, U>(
    trace_matrix: &RowMajorTrace<F>,
    projected_scalars: &ScalarMap<U::Scalar, DynamicPolynomialF<F>>,
    num_constraints: usize,
    field_cfg: &F::Config,
    fp_indices: &[usize],
    sub_timings: Option<&mut ComputeCombinedSubTimings>,
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

    let skip: Vec<bool> = {
        let mut s = vec![true; num_constraints];
        for &i in fp_indices {
            s[i] = false;
        }
        s
    };

    let _t_phase1 = std::time::Instant::now();
    let mut max_degrees_and_combined_poly_rows: Vec<(usize, Vec<DynamicPolynomialF<F>>)> =
        cfg_into_iter!(0..num_rows - 1)
            .map(|row_idx| {
                let up = &trace_matrix[row_idx];

                let mut down: Vec<DynamicPolynomialF<F>> = uair_sig
                    .shifts()
                    .iter()
                    .map(|spec| {
                        if row_idx + spec.shift_amount() < num_rows {
                            trace_matrix[row_idx + spec.shift_amount()][spec.source_col()].clone()
                        } else {
                            DynamicPolynomialF::zero()
                        }
                    })
                    .collect();

                for spec in uair_sig.bit_op_specs() {
                    let src_cell = &trace_matrix[row_idx][spec.source_col()];
                    let op_cell = match spec.op() {
                        BitOp::Rot(c) => src_cell.rot_c(c),
                        BitOp::ShiftR(c) => src_cell.shift_r_c(c),
                    };
                    down.push(op_cell);
                }

                let mut builder =
                    CombinedPolyRowBuilderFpOnly::new(num_constraints, fp_indices);

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
                    &mut builder,
                    TraceRow::from_slice_with_layout(
                        up,
                        U::signature().total_cols().as_column_layout(),
                    ),
                    TraceRow::from_slice_with_layout_and_bit_op(&down, down_layout, bit_op_count),
                    &project,
                    |x, y| Some(project(y) * x),
                    ImpossibleIdeal::from_ref,
                );

                let mut combined_evaluations = builder.combined_evaluations;
                combined_evaluations.iter_mut().for_each(|eval| eval.trim());

                let max_degree = fp_indices
                    .iter()
                    .map(|&i| combined_evaluations[i].degree().unwrap_or(0))
                    .max()
                    .unwrap_or(0);

                (max_degree, combined_evaluations)
            })
            .collect();
    let dt_phase1 = _t_phase1.elapsed();

    let max_degree = *max_degrees_and_combined_poly_rows
        .iter()
        .map(|(max_degree, _)| max_degree)
        .max()
        .expect("F_p-branch IC must have at least one row");

    max_degrees_and_combined_poly_rows.push((0, vec![DynamicPolynomialF::zero(); num_constraints]));

    let _t_phase2 = std::time::Instant::now();
    let result = prepare_coefficient_mles(
        num_constraints,
        max_degree,
        &max_degrees_and_combined_poly_rows,
        field_zero.inner(),
        &skip,
    );
    let dt_phase2 = _t_phase2.elapsed();

    if let Some(t) = sub_timings {
        t.per_row_constrain = dt_phase1;
        t.prepare_coeff_mles = dt_phase2;
    }

    result
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

/// Z-only MLE-first evaluator. Mirrors
/// [`evaluate_combined_polynomials_fp_only`] but uses the Z-only row
/// builder. Provided for symmetry; in practice the dual-prime
/// Z-branch goes through `compute_combined_polynomials_z_only`
/// (combined-poly lane) because Z-tagged constraints in real UAIRs
/// are typically non-linear.
#[allow(clippy::arithmetic_side_effects, clippy::too_many_arguments)]
pub fn evaluate_combined_polynomials_z_only<F, U>(
    trace_matrix: &ColumnMajorTrace<F>,
    projected_scalars: &ScalarMap<U::Scalar, DynamicPolynomialF<F>>,
    num_constraints: usize,
    evaluation_point: &[F],
    field_cfg: &F::Config,
    z_indices: &[usize],
) -> Result<Vec<DynamicPolynomialF<F>>, EvaluationError>
where
    F: InnerTransparentField,
    U: Uair,
{
    evaluate_combined_polynomials_with_builder::<F, U>(
        trace_matrix,
        projected_scalars,
        num_constraints,
        evaluation_point,
        field_cfg,
        TagBuilderKind::ZOnly(z_indices),
    )
}

/// F_p-only MLE-first evaluator. Mirrors
/// [`evaluate_combined_polynomials_unchecked`] but uses
/// [`CombinedPolyRowBuilderFpOnly`] so the single
/// `U::constrain_general` call short-circuits on the UAIR's
/// `is_active_for(Z)` / `is_active_for_zero_ideal()` gates and only
/// runs the F_p-tagged sub-graph. Off-tag and zero-ideal slots come
/// back as `ZERO`.
///
/// `fp_indices` lists the global constraint-vector positions of
/// F_p-tagged non-zero-ideal slots, in declaration order.
#[allow(clippy::arithmetic_side_effects, clippy::too_many_arguments)]
pub fn evaluate_combined_polynomials_fp_only<F, U>(
    trace_matrix: &ColumnMajorTrace<F>,
    projected_scalars: &ScalarMap<U::Scalar, DynamicPolynomialF<F>>,
    num_constraints: usize,
    evaluation_point: &[F],
    field_cfg: &F::Config,
    fp_indices: &[usize],
) -> Result<Vec<DynamicPolynomialF<F>>, EvaluationError>
where
    F: InnerTransparentField,
    U: Uair,
{
    evaluate_combined_polynomials_with_builder::<F, U>(
        trace_matrix,
        projected_scalars,
        num_constraints,
        evaluation_point,
        field_cfg,
        TagBuilderKind::FpOnly(fp_indices),
    )
}

enum TagBuilderKind<'a> {
    ZOnly(&'a [usize]),
    FpOnly(&'a [usize]),
}

#[allow(clippy::arithmetic_side_effects, clippy::too_many_arguments)]
fn evaluate_combined_polynomials_with_builder<F, U>(
    trace_matrix: &ColumnMajorTrace<F>,
    projected_scalars: &ScalarMap<U::Scalar, DynamicPolynomialF<F>>,
    num_constraints: usize,
    evaluation_point: &[F],
    field_cfg: &F::Config,
    builder_kind: TagBuilderKind<'_>,
) -> Result<Vec<DynamicPolynomialF<F>>, EvaluationError>
where
    F: InnerTransparentField,
    U: Uair,
{
    let field_zero = F::zero_with_cfg(field_cfg);
    let zero_inner = field_zero.inner().clone();
    let num_rows = trace_matrix.first().map(|c| c.len()).unwrap_or(0);
    let num_vars = evaluation_point.len();

    let max_num_coeffs = trace_matrix
        .iter()
        .flat_map(|col| col.evaluations.iter())
        .map(|p| p.coeffs.len())
        .max()
        .unwrap_or(0);

    let uair_sig = U::signature();
    let down_layout = uair_sig.down_cols().as_column_layout();

    let eval_coeff_mle = |col: &DenseMultilinearExtension<DynamicPolynomialF<F>>,
                          d: usize,
                          shift: usize|
     -> Result<F, EvaluationError> {
        let coeff_evals: Vec<F::Inner> = (0..num_rows)
            .map(|i| {
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

    let up_evals: Vec<DynamicPolynomialF<F>> = cfg_iter!(trace_matrix)
        .map(|col| {
            let coeffs: Vec<F> = (0..max_num_coeffs)
                .map(|d| eval_coeff_mle(col, d, 0))
                .collect::<Result<_, _>>()?;
            Ok(DynamicPolynomialF::new_trimmed(coeffs))
        })
        .collect::<Result<Vec<_>, EvaluationError>>()?;

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

    let combined_evaluations = match builder_kind {
        TagBuilderKind::FpOnly(fp_indices) => {
            let mut builder = CombinedPolyRowBuilderFpOnly::new(num_constraints, fp_indices);
            U::constrain_general(
                &mut builder,
                TraceRow::from_slice_with_layout(
                    &up_evals,
                    uair_sig.total_cols().as_column_layout(),
                ),
                TraceRow::from_slice_with_layout_and_bit_op(&down_evals, down_layout, bit_op_count),
                &project,
                |x, y| Some(project(y) * x),
                ImpossibleIdeal::from_ref,
            );
            builder.combined_evaluations
        }
        TagBuilderKind::ZOnly(z_indices) => {
            let mut builder = CombinedPolyRowBuilderZOnly::new(num_constraints, z_indices);
            U::constrain_general(
                &mut builder,
                TraceRow::from_slice_with_layout(
                    &up_evals,
                    uair_sig.total_cols().as_column_layout(),
                ),
                TraceRow::from_slice_with_layout_and_bit_op(&down_evals, down_layout, bit_op_count),
                &project,
                |x, y| Some(project(y) * x),
                ImpossibleIdeal::from_ref,
            );
            builder.combined_evaluations
        }
    };

    let mut combined_evaluations = combined_evaluations;
    combined_evaluations.iter_mut().for_each(|eval| eval.trim());

    Ok(combined_evaluations)
}

pub struct CombinedPolyRowBuilder<F: PrimeField> {
    combined_evaluations: Vec<DynamicPolynomialF<F>>,
}

/// Z-only row builder: pre-fills `combined_evaluations` with `ZERO` for
/// every constraint and only writes at slots whose tag is
/// [`ConstraintRing::Z`]. Used by the dual-prime Z-branch IC together
/// with a UAIR that gates its F_p / zero-ideal sub-graphs via
/// [`ConstraintBuilder::is_active_for`] / [`ConstraintBuilder::is_active_for_zero_ideal`]
/// — those gates short-circuit before the expensive polynomial
/// arithmetic, leaving only the Z-tagged work per row.
///
/// `z_indices` lists the global constraint-vector positions of
/// Z-tagged non-zero-ideal slots, in declaration order. The k-th call
/// to `assert_in_ideal_typed(_, _, Z)` writes to
/// `combined_evaluations[z_indices[k]]`. Off-tag and zero-ideal
/// assertions are no-ops (the gate prevents them in correctly-written
/// UAIRs; if they fire anyway the behaviour is still safe — the
/// pre-filled `ZERO` survives, matching the IC's tag-skip mask).
pub struct CombinedPolyRowBuilderZOnly<'a, F: PrimeField> {
    combined_evaluations: Vec<DynamicPolynomialF<F>>,
    z_indices: &'a [usize],
    next_z_local: usize,
}

impl<'a, F: PrimeField> CombinedPolyRowBuilderZOnly<'a, F> {
    pub fn new(num_constraints: usize, z_indices: &'a [usize]) -> Self {
        Self {
            combined_evaluations: vec![DynamicPolynomialF::zero(); num_constraints],
            z_indices,
            next_z_local: 0,
        }
    }
}

impl<F: PrimeField> ConstraintBuilder for CombinedPolyRowBuilderZOnly<'_, F> {
    type Expr = DynamicPolynomialF<F>;
    type Ideal = ImpossibleIdeal;

    fn assert_in_ideal(&mut self, _expr: Self::Expr, _ideal: &Self::Ideal) {
        // Untyped asserts are tag-Z-by-default in IdealCollector but the
        // user's gate should prevent them firing here. Treat as no-op.
    }

    fn assert_in_ideal_typed(
        &mut self,
        expr: Self::Expr,
        _ideal: &Self::Ideal,
        ring: zinc_uair::ConstraintRing,
    ) {
        if ring == zinc_uair::ConstraintRing::Z && self.next_z_local < self.z_indices.len() {
            let idx = self.z_indices[self.next_z_local];
            self.combined_evaluations[idx] = expr;
            #[allow(clippy::arithmetic_side_effects)]
            {
                self.next_z_local += 1;
            }
        }
    }

    fn assert_zero(&mut self, _expr: Self::Expr) {
        // Zero-ideal slots are not produced by the Z-branch IC.
    }

    #[inline(always)]
    fn is_active_for(&self, ring: zinc_uair::ConstraintRing) -> bool {
        ring == zinc_uair::ConstraintRing::Z
    }

    #[inline(always)]
    fn is_active_for_zero_ideal(&self) -> bool {
        false
    }
}

/// F_p-only row builder, mirror of [`CombinedPolyRowBuilderZOnly`].
/// Pre-fills `combined_evaluations` with `ZERO` for every constraint
/// and only writes at slots whose tag is
/// [`zinc_uair::ConstraintRing::Fp`]. Used by the dual-prime F_p
/// branch IC together with a UAIR that gates its Z / zero-ideal
/// sub-graphs via [`ConstraintBuilder::is_active_for`] /
/// [`ConstraintBuilder::is_active_for_zero_ideal`] — the gates
/// short-circuit before the expensive polynomial arithmetic, leaving
/// only the F_p-tagged work per row.
///
/// `fp_indices` lists the global constraint-vector positions of
/// F_p-tagged non-zero-ideal slots, in declaration order. The k-th
/// call to `assert_in_ideal_typed(_, _, Fp)` writes to
/// `combined_evaluations[fp_indices[k]]`.
pub struct CombinedPolyRowBuilderFpOnly<'a, F: PrimeField> {
    combined_evaluations: Vec<DynamicPolynomialF<F>>,
    fp_indices: &'a [usize],
    next_fp_local: usize,
}

impl<'a, F: PrimeField> CombinedPolyRowBuilderFpOnly<'a, F> {
    pub fn new(num_constraints: usize, fp_indices: &'a [usize]) -> Self {
        Self {
            combined_evaluations: vec![DynamicPolynomialF::zero(); num_constraints],
            fp_indices,
            next_fp_local: 0,
        }
    }
}

impl<F: PrimeField> ConstraintBuilder for CombinedPolyRowBuilderFpOnly<'_, F> {
    type Expr = DynamicPolynomialF<F>;
    type Ideal = ImpossibleIdeal;

    fn assert_in_ideal(&mut self, _expr: Self::Expr, _ideal: &Self::Ideal) {
        // Untyped asserts default to Z in IdealCollector — F_p-only
        // never wants them. Treat as no-op (gate should prevent them).
    }

    fn assert_in_ideal_typed(
        &mut self,
        expr: Self::Expr,
        _ideal: &Self::Ideal,
        ring: zinc_uair::ConstraintRing,
    ) {
        if ring == zinc_uair::ConstraintRing::Fp && self.next_fp_local < self.fp_indices.len() {
            let idx = self.fp_indices[self.next_fp_local];
            self.combined_evaluations[idx] = expr;
            #[allow(clippy::arithmetic_side_effects)]
            {
                self.next_fp_local += 1;
            }
        }
    }

    fn assert_zero(&mut self, _expr: Self::Expr) {
        // Zero-ideal slots are not produced by the F_p-branch IC.
    }

    #[inline(always)]
    fn is_active_for(&self, ring: zinc_uair::ConstraintRing) -> bool {
        ring == zinc_uair::ConstraintRing::Fp
    }

    #[inline(always)]
    fn is_active_for_zero_ideal(&self) -> bool {
        false
    }
}

impl<F: PrimeField> ConstraintBuilder for CombinedPolyRowBuilder<F> {
    type Expr = DynamicPolynomialF<F>;
    type Ideal = ImpossibleIdeal;

    fn assert_in_ideal(&mut self, expr: Self::Expr, _ideal: &Self::Ideal) {
        self.combined_evaluations.push(expr);
    }

    /// Preserve the actual `F[X]` polynomial expression. The earlier
    /// optimization that substituted `DynamicPolynomialF::ZERO` here
    /// relied on the per-row F[X] expression being identically the
    /// zero polynomial — which is true when `F::from_with_cfg(cell)`
    /// produces a zero F coefficient for every constraint-vanishing
    /// witness. That is a stronger property than the constraint
    /// vanishing in F: the F[X] polynomial can be non-trivial yet
    /// evaluate to zero at the projecting element. Substituting `ZERO`
    /// drops information that the verifier needs in the sumcheck
    /// consistency check (`claimed_sum == expected_sum`). Preserving
    /// the expression matches what `assert_in_ideal` already does and
    /// keeps the proof self-consistent for any cell representation.
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
