#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::projections::{ColumnMajorTrace, ProjectedScalars, RowMajorTrace};
use crypto_primitives::{PrimeField, Semiring};
use num_traits::Zero;
use zinc_poly::{
    EvaluationError,
    mle::{
        DenseMultilinearExtension, MultilinearExtensionWithConfig, dense::CollectDenseMleWithZero,
    },
    univariate::dynamic::over_field::DynamicPolynomialF,
    utils::{ArithErrors as PolyArithErrors, build_eq_x_r_vec},
};
use zinc_uair::{
    BitOp, BitOpSpec, ColumnLayout, ConstraintBuilder, ShiftSpec, TraceRow, Uair, UairSignature,
    ideal::ImpossibleIdeal,
};
use zinc_utils::{
    add, cfg_into_iter, cfg_iter, from_ref::FromRef, inner_transparent_field::InnerTransparentField,
};

/// Evaluate combined polynomial MLEs at `evaluation_point` for a selected
/// subset of constraints ($Q[X]$ or $F_{q_i}[X]$).
///
/// Runs `constrain_general` row-by-row to produce all constraint values (since
/// it is monolithic), then selects only the entries at `constraint_indices`,
/// builds their coefficient MLEs, and evaluates them via the eq table.
///
/// For $F_{q_i}[X]$ constraints, `trace_matrix` and `projected_scalars` must
/// already be projected mod $q_i$.
///
/// `family_idx` selects the constraint family: `0` for $Q[X]$, `i >= 1`
/// for $F_{q_{i-1}}[X]$.
///
/// Returns one `DynamicPolynomialF<F>` per requested constraint, in the
/// same order as `constraint_indices`.
///
/// `trace_matrix` is row-indexed: `trace_matrix[row][col]`.
#[allow(clippy::arithmetic_side_effects)]
pub fn evaluate_for_constraints<F, U, const DEGREE_PLUS_ONE: usize>(
    trace_matrix: &RowMajorTrace<F>,
    projected_scalars: &ProjectedScalars<U::Scalar, DynamicPolynomialF<F>>,
    family_idx: usize,
    num_constraints: usize,
    field_cfg: &F::Config,
    constraint_indices: &[usize],
    evaluation_point: &[F],
) -> Result<Vec<DynamicPolynomialF<F>>, PolyArithErrors>
where
    F: InnerTransparentField,
    U: Uair,
{
    let field_zero = F::zero_with_cfg(field_cfg);
    let uair_sig = U::signature();
    let down_layout = uair_sig.down_cols().as_column_layout();

    let num_rows = trace_matrix.len();

    // Evaluate all constraints at every row of the trace.
    //
    // `constrain_general` is monolithic - it produces all `num_constraints`
    // values per call.  We keep the full vectors and select only the
    // requested `constraint_indices` further down.
    //
    // `all_rows[row_idx][constraint_idx]` is a `DynamicPolynomialF<F>`:
    // the combined polynomial value of constraint `constraint_idx` at
    // trace row `row_idx`.
    let bit_op_down_offset = bit_op_down_offset(&uair_sig);
    let mut all_rows: Vec<Vec<DynamicPolynomialF<F>>> = cfg_into_iter!(0..num_rows - 1)
        .map(|row_idx| {
            let up = &trace_matrix[row_idx];

            // Build the down row in the canonical order:
            //   [shifted_binary, bit_op_binary, shifted_arbitrary, shifted_int]
            // (cf. `UairSignature::with_bit_op_specs`). Splicing bit-op
            // virtuals into the binary_poly slice keeps `down` consistent
            // with `down_layout`; appending at the tail would misalign
            // constraints on mixed-type shift UAIRs.
            let mut down: Vec<DynamicPolynomialF<F>> =
                Vec::with_capacity(uair_sig.shifts().len() + uair_sig.bit_op_specs().len());

            let shifts = uair_sig.shifts();
            for spec in &shifts[..bit_op_down_offset] {
                push_shifted_down_entry(&mut down, trace_matrix, row_idx, num_rows, spec);
            }

            for spec in uair_sig.bit_op_specs() {
                let source = &trace_matrix[row_idx][spec.source_col()];
                down.push(apply_bit_op_to_poly::<F, DEGREE_PLUS_ONE>(
                    source, spec, field_cfg,
                ));
            }

            for spec in &shifts[bit_op_down_offset..] {
                push_shifted_down_entry(&mut down, trace_matrix, row_idx, num_rows, spec);
            }

            evaluate_constraints_for_row::<F, U>(
                up,
                &down,
                family_idx,
                num_constraints,
                projected_scalars,
                down_layout,
            )
        })
        .collect();

    // Zero-pad to 2^num_vars evaluations for the MLE.
    // TODO(Ilia): reimplement using Albert's idea with selector polynomials.
    all_rows.push(vec![DynamicPolynomialF::zero(); num_constraints]);

    // Determine the maximum polynomial degree across the selected
    // constraints and all rows.  This controls how many coefficient MLEs
    // we build per constraint (one MLE per coefficient 0..=max_degree).
    let max_degree = all_rows
        .iter()
        .flat_map(|row| {
            constraint_indices
                .iter()
                .map(|&ci| row[ci].degree().unwrap_or(0))
        })
        .max()
        .unwrap_or(0);

    let zero_inner = field_zero.inner();

    // For each selected constraint, build its coefficient MLEs.
    //
    // Each combined polynomial value at row `b` is a univariate in X:
    //   c(b, X) = c_0(b) + c_1(b) X + ... + c_d(b) X^d.
    //
    // For a fixed coefficient index `coeff`, the MLE is the multilinear
    // extension of the table  b -> c_coeff(b)  over the Boolean hypercube.
    let coeff_mles: Vec<Vec<DenseMultilinearExtension<F::Inner>>> =
        cfg_into_iter!(0..constraint_indices.len())
            .map(|sel_idx| {
                let ci = constraint_indices[sel_idx];
                (0..=max_degree)
                    .map(|coeff| {
                        all_rows
                            .iter()
                            .map(|row| {
                                row[ci]
                                    .coeffs
                                    .get(coeff)
                                    .map(|c| c.inner().clone())
                                    .unwrap_or_else(|| zero_inner.clone())
                            })
                            .collect_dense_mle_with_zero(zero_inner)
                    })
                    .collect()
            })
            .collect();

    // Step 4: Evaluate each coefficient MLE at the challenge point `r`.
    //
    // Uses the precomputed eq table eq(r, ·) so that evaluating one MLE
    // is a single inner product.  Reassembling the per-coefficient results
    // gives the univariate  sum_b eq(r, b) * c(b, X)  for each selected
    // constraint.
    let eq_table = build_eq_x_r_vec(evaluation_point, field_cfg)?;

    let values: Vec<DynamicPolynomialF<F>> = cfg_into_iter!(coeff_mles)
        .map(|mles| {
            let coeffs: Vec<F> = mles
                .into_iter()
                .map(|mle| {
                    zinc_poly::utils::mle_eval_with_eq_table(&mle.evaluations, &eq_table, field_cfg)
                })
                .collect();
            DynamicPolynomialF::new_trimmed(coeffs)
        })
        .collect();

    Ok(values)
}

/// Evaluate all UAIR constraints of the given family for a single row and
/// return trimmed results.
#[allow(clippy::arithmetic_side_effects)]
fn evaluate_constraints_for_row<F, U>(
    up: &[DynamicPolynomialF<F>],
    down: &[DynamicPolynomialF<F>],
    family_idx: usize,
    num_constraints: usize,
    projected_scalars: &ProjectedScalars<U::Scalar, DynamicPolynomialF<F>>,
    down_layout: &ColumnLayout,
) -> Vec<DynamicPolynomialF<F>>
where
    F: PrimeField,
    U: Uair,
{
    let mut constraint_builder = CombinedPolyRowBuilder::new(family_idx, num_constraints);

    let project = |x: &U::Scalar| {
        projected_scalars
            .get(x)
            .expect("all scalars should have been projected at this point")
    };

    U::constrain_general(
        &mut constraint_builder,
        TraceRow::from_slice_with_layout(up, U::signature().total_cols().as_column_layout()),
        TraceRow::from_slice_with_layout(down, down_layout),
        &project,
        |x, y| Some(project(y) * x),
        ImpossibleIdeal::from_ref,
        ImpossibleIdeal::from_ref,
    );

    let mut combined_evaluations = constraint_builder.combined_evaluations;
    combined_evaluations.iter_mut().for_each(|eval| eval.trim());
    combined_evaluations
}

/// Evaluate combined polynomials by first evaluating trace column MLEs at the
/// evaluation point, then applying UAIR constraints to the evaluated values.
///
/// This avoids building per-row combined polynomials and is faster than
/// `evaluate_for_constraints`. The technique relies on the evaluation of a
/// linear combination of MLEs equalling the linear combination of individual
/// MLE evaluations, so the entries of the returned vector are only correct
/// for constraints that are linear in the column values; non-linear entries
/// are garbage and must be ignored by the caller. The prover-side caller
/// [`crate::ideal_check`] partitions constraints by degree and reads only
/// the linear indices from this output, routing non-linear constraints
/// through [`evaluate_for_constraints`] instead.
///
/// `trace_matrix` is column-indexed: `trace_matrix[col]` is an MLE.
///
/// For $F_{q_i}[X]$ constraints, `trace_matrix` and `projected_scalars` must
/// already be projected mod $q_i$.
///
/// `family_idx` selects the constraint family: `0` for $Q[X]$, `i >= 1`
/// for $F_{q_{i-1}}[X]$.
///
/// Does `(num_columns + num_shifted_columns) * max_num_coeffs` evaluations of
/// MLEs.
#[allow(clippy::arithmetic_side_effects)]
pub fn evaluate_combined_polynomials<F, U, const DEGREE_PLUS_ONE: usize>(
    trace_matrix: &ColumnMajorTrace<F>,
    projected_scalars: &ProjectedScalars<U::Scalar, DynamicPolynomialF<F>>,
    family_idx: usize,
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
    let shift_down_evals: Vec<DynamicPolynomialF<F>> = cfg_iter!(sorted_shifts)
        .map(|spec| {
            let col = &trace_matrix[spec.source_col()];
            let coeffs: Vec<F> = (0..max_num_coeffs)
                .map(|d| eval_coeff_mle(col, d, spec.shift_amount()))
                .collect::<Result<_, _>>()?;
            Ok(DynamicPolynomialF::new_trimmed(coeffs))
        })
        .collect::<Result<Vec<_>, EvaluationError>>()?;

    // Evaluate bit-op virtuals from the already-computed up evaluations:
    // bit-ops act coefficient-wise, so MLE-eval-of-op at the same point is
    // op-of-MLE-eval on the source coefficient vector.
    let bit_op_down_offset = bit_op_down_offset(&uair_sig);
    let bit_op_down_evals: Vec<DynamicPolynomialF<F>> = uair_sig
        .bit_op_specs()
        .iter()
        .map(|spec| {
            apply_bit_op_to_poly::<F, DEGREE_PLUS_ONE>(
                &up_evals[spec.source_col()],
                spec,
                field_cfg,
            )
        })
        .collect();

    // Splice into the canonical down ordering; see UairSignature docs.
    let mut down_evals: Vec<DynamicPolynomialF<F>> =
        Vec::with_capacity(shift_down_evals.len() + bit_op_down_evals.len());
    down_evals.extend_from_slice(&shift_down_evals[..bit_op_down_offset]);
    down_evals.extend(bit_op_down_evals);
    down_evals.extend_from_slice(&shift_down_evals[bit_op_down_offset..]);

    // Apply UAIR constraints to the evaluated trace values
    let mut constraint_builder = CombinedPolyRowBuilder::new(family_idx, num_constraints);

    let project = |x: &U::Scalar| {
        projected_scalars
            .get(x)
            .expect("all scalars should have been projected at this point")
    };

    U::constrain_general(
        &mut constraint_builder,
        TraceRow::from_slice_with_layout(&up_evals, uair_sig.total_cols().as_column_layout()),
        TraceRow::from_slice_with_layout(&down_evals, down_layout),
        &project,
        |x, y| Some(project(y) * x),
        ImpossibleIdeal::from_ref,
        ImpossibleIdeal::from_ref,
    );

    let mut combined_evaluations = constraint_builder.combined_evaluations;
    combined_evaluations.iter_mut().for_each(|eval| eval.trim());

    Ok(combined_evaluations)
}

fn bit_op_down_offset<P: Semiring>(uair_sig: &UairSignature<P>) -> usize {
    let binary_poly_end = uair_sig.total_cols().num_binary_poly_cols();
    uair_sig
        .shifts()
        .iter()
        .take_while(|spec| spec.source_col() < binary_poly_end)
        .count()
}

fn push_shifted_down_entry<F: PrimeField>(
    down: &mut Vec<DynamicPolynomialF<F>>,
    trace_matrix: &RowMajorTrace<F>,
    row_idx: usize,
    num_rows: usize,
    spec: &ShiftSpec,
) {
    let shifted_row_idx = add!(row_idx, spec.shift_amount());
    if shifted_row_idx < num_rows {
        down.push(trace_matrix[shifted_row_idx][spec.source_col()].clone());
    } else {
        down.push(DynamicPolynomialF::zero());
    }
}

fn apply_bit_op_to_poly<F: PrimeField, const DEGREE_PLUS_ONE: usize>(
    source: &DynamicPolynomialF<F>,
    spec: &BitOpSpec,
    field_cfg: &F::Config,
) -> DynamicPolynomialF<F> {
    match spec.op() {
        BitOp::Rot(c) => source.rotate_right::<DEGREE_PLUS_ONE>(c, field_cfg),
        BitOp::ShR(c) => source.shr::<DEGREE_PLUS_ONE>(c, field_cfg),
    }
}

/// Row-builder that accumulates only the specific family of constraints,
/// selected by `family_idx`:
/// - `family_idx == 0` -> $Q[X]$ constraints
/// - `family_idx == i (i >= 1)` -> $F_{q_{i-1}}[X]$ constraints (i.e.
///   UAIR-level `prime_idx = i - 1`).
pub struct CombinedPolyRowBuilder<F: PrimeField> {
    family_idx: usize,
    combined_evaluations: Vec<DynamicPolynomialF<F>>,
}

impl<F: PrimeField> ConstraintBuilder for CombinedPolyRowBuilder<F> {
    type Expr = DynamicPolynomialF<F>;
    type Ideal = ImpossibleIdeal;
    type FqIdeal = ImpossibleIdeal;

    fn assert_in_ideal(&mut self, expr: Self::Expr, _ideal: &Self::Ideal) {
        // Q[X] family -> index 0.
        if self.family_idx == 0 {
            self.combined_evaluations.push(expr);
        }
    }

    fn assert_zero(&mut self, expr: Self::Expr) {
        // Q[X] family -> index 0.
        if self.family_idx == 0 {
            self.combined_evaluations.push(expr);
        }
    }

    #[allow(clippy::arithmetic_side_effects)]
    fn assert_in_fq_ideal(&mut self, prime_idx: usize, expr: Self::Expr, _ideal: &Self::FqIdeal) {
        // F_{q_{prime_idx}}[X] family -> index (prime_idx + 1).
        if self.family_idx == prime_idx + 1 {
            self.combined_evaluations.push(expr);
        }
        // Otherwise it belongs to a different family; silently skip.
    }
}

impl<F: PrimeField> CombinedPolyRowBuilder<F> {
    pub fn new(family_idx: usize, num_constraints: usize) -> Self {
        Self {
            family_idx,
            combined_evaluations: Vec::with_capacity(num_constraints),
        }
    }
}
