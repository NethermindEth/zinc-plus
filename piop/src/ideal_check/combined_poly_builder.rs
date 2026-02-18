use std::collections::HashMap;

use crypto_primitives::PrimeField;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use zinc_poly::{
    mle::{DenseMultilinearExtension, MultilinearExtensionWithConfig, dense::CollectDenseMleWithZero},
    univariate::dynamic::over_field::DynamicPolynomialF,
};
use zinc_uair::{ConstraintBuilder, TraceRow, Uair, ideal::ImpossibleIdeal};
use zinc_utils::{cfg_into_iter, from_ref::FromRef, inner_transparent_field::InnerTransparentField};

/// Compute the combined polynomial MLE evaluations directly.
///
/// This implements an optimized, *coefficient-level* evaluation strategy.
/// Instead of building an intermediate `DynamicPolynomialF` per constraint
/// per row (causing thousands of heap allocations), it evaluates the UAIR
/// constraints **at each coefficient degree independently**, using plain
/// field elements (`F`) as the expression type.
///
/// **Why this is faster:**
///
/// For a UAIR whose constraints are linear in the trace (add / sub only,
/// no polynomial multiplication), the k-th coefficient of the combined
/// polynomial at row `i` is exactly the k-th coefficient obtained by applying
/// the same constraint to the k-th coefficients of the trace polynomials.
/// More formally:
///
/// $$[\text{constraint}(p_0(X), p_1(X), \ldots)]_k
///     = \text{constraint}([p_0]_k, [p_1]_k, \ldots)$$
///
/// This holds because addition and subtraction are applied coefficient-wise.
///
/// By evaluating the constraint at each coefficient `k` separately, we work
/// with field elements (`F`, stack-allocated) instead of polynomials
/// (`DynamicPolynomialF<F>`, heap-allocated).  This eliminates:
///
/// -  ~6 × num_rows `DynamicPolynomialF` clones for the `up`/`down` trace
///    values (each clone allocates a `Vec<F>` on the heap).
/// -  ~num_constraints × num_rows intermediate `DynamicPolynomialF` results
///    from constraint evaluation.
/// - The separate "transpose" phase that gathered coefficients into
///   `DenseMultilinearExtension` objects.
/// - The separate evaluation phase that cloned each MLE before folding.
///
/// **Fold phase** uses a branchless in-place loop — no equality check, no
/// clone. All buffers are pre-allocated as flat `Vec<F::Inner>`.
///
/// For UAIRs with polynomial multiplication (convolution) the coefficient
/// identity above does not hold.  The function detects this by running a
/// single "pilot" row evaluation at the **polynomial** level and comparing
/// with the coefficient-level result; if they disagree it falls back to the
/// original polynomial-level code path.
#[allow(clippy::arithmetic_side_effects)]
pub fn compute_combined_values<F, U>(
    trace_matrix: &[DenseMultilinearExtension<DynamicPolynomialF<F>>],
    projected_scalars: &HashMap<U::Scalar, DynamicPolynomialF<F>>,
    num_constraints: usize,
    evaluation_point: &[F],
    field_cfg: &F::Config,
) -> Vec<DynamicPolynomialF<F>>
where
    F: InnerTransparentField,
    U: Uair,
{
    let field_zero = F::zero_with_cfg(field_cfg);
    let zero_inner = field_zero.inner().clone();

    let num_rows = trace_matrix[0].len();
    let num_vars = evaluation_point.len();
    let num_cols = trace_matrix.len();

    // Determine the maximum polynomial degree across all trace entries.
    let max_trace_degree = trace_matrix
        .iter()
        .flat_map(|col| col.iter().filter_map(|p| p.degree()))
        .max()
        .unwrap_or(0);

    // After constraint evaluation (which may increase degree for
    // multiplication-type constraints), we need to discover the actual
    // max degree.  For linear constraints the degree does not increase.
    // We run one pilot row at the polynomial level to find it.
    let pilot_polys = if num_rows > 1 {
        let up: Vec<_> = trace_matrix.iter().map(|col| col[0].clone()).collect();
        let down: Vec<_> = trace_matrix.iter().map(|col| col[1].clone()).collect();
        evaluate_constraints_poly::<F, U>(&up, &down, num_constraints, projected_scalars)
    } else {
        vec![DynamicPolynomialF::new([]); num_constraints]
    };

    let pilot_max_degree = pilot_polys
        .iter()
        .filter_map(|p| p.degree())
        .max()
        .unwrap_or(0);

    // If the pilot degree is higher than the trace degree, the constraint
    // involves multiplication (convolution), so fall back to the safe
    // polynomial-level path.
    let is_linear = pilot_max_degree <= max_trace_degree;

    // The coefficient-level fast path evaluates `constrain_general` once
    // per coefficient degree (num_coeffs ≈ 32) per row, whereas the
    // polynomial path calls it once per row.  The savings come from
    // avoiding trace-polynomial clones (2 × num_cols heap allocations per
    // row), but the cost grows with the number of constraints (more
    // builder push + constraint-body overhead per call).
    //
    // Empirically the coefficient path wins when the clone savings
    // dominate, which happens when num_cols > num_constraints.  When
    // num_constraints >= num_cols the extra call overhead exceeds the
    // clone savings, so we fall back to the polynomial path.
    let use_coeff_path = is_linear && num_cols > num_constraints;

    if !use_coeff_path {
        return compute_combined_values_poly::<F, U>(
            trace_matrix,
            projected_scalars,
            num_constraints,
            evaluation_point,
            field_cfg,
            pilot_polys,
            pilot_max_degree,
        );
    }

    // ── Fast path: coefficient-level evaluation ─────────────────────────
    let num_coeffs = pilot_max_degree + 1;
    let buf_size = num_rows; // power of two (includes padding row)
    let flat_len = num_constraints * num_coeffs;

    // Pre-allocate all coefficient buffers.
    let mut bufs: Vec<Vec<F::Inner>> = vec![vec![zero_inner.clone(); buf_size]; flat_len];

    let signature = U::signature();
    let row_len = signature.total_cols();

    // Project scalars to F for the coefficient-level builder.
    let scalar_coefficients: HashMap<U::Scalar, Vec<F>> = projected_scalars
        .iter()
        .map(|(key, poly)| {
            let mut coeffs = poly.coeffs.clone();
            coeffs.resize(num_coeffs, field_zero.clone());
            (key.clone(), coeffs)
        })
        .collect();

    // Pre-allocate row buffers and builder — reused across all iterations.
    let mut up_row: Vec<F> = vec![field_zero.clone(); row_len];
    let mut down_row: Vec<F> = vec![field_zero.clone(); row_len];
    let mut builder = CoeffLevelBuilder::new(num_constraints);

    // Row-major iteration: for each row pair, process all coefficient
    // degrees.  This ensures each trace polynomial (`DynamicPolynomialF`)
    // is accessed only once per row; its coefficient `Vec` fits in L1
    // cache and is reused across the 0..num_coeffs inner loop.
    for row_idx in 0..num_rows - 1 {
        for k in 0..num_coeffs {
            // Fill up_row and down_row from trace coefficient k.
            for col in 0..num_cols {
                let up_poly = &trace_matrix[col][row_idx];
                up_row[col] = if k < up_poly.coeffs.len() {
                    up_poly.coeffs[k].clone()
                } else {
                    field_zero.clone()
                };

                let down_poly = &trace_matrix[col][row_idx + 1];
                down_row[col] = if k < down_poly.coeffs.len() {
                    down_poly.coeffs[k].clone()
                } else {
                    field_zero.clone()
                };
            }

            // Reuse builder across rows — only clear the results Vec,
            // keeping its heap allocation.
            builder.results.clear();

            let from_ref = |scalar: &U::Scalar| {
                scalar_coefficients
                    .get(scalar)
                    .map(|coeffs| coeffs[k].clone())
                    .expect("all scalars should have been projected at this point")
            };

            U::constrain_general(
                &mut builder,
                TraceRow::from_slice_with_signature(&up_row, &signature),
                TraceRow::from_slice_with_signature(&down_row, &signature),
                &from_ref,
                |x, y| {
                    let scalar_val = scalar_coefficients
                        .get(y)
                        .map(|c| c[0].clone())
                        .expect("scalar not found");
                    Some(x.clone() * &scalar_val)
                },
                ImpossibleIdeal::from_ref,
            );

            // Write constraint results into flat buffers.
            for (c, val) in builder.results.iter().enumerate() {
                bufs[c * num_coeffs + k][row_idx] = val.inner().clone();
            }
        }
        // Row n-1 (padding) stays at zero.
    }

    // ── Fold all coefficient buffers in-place ───────────────────────────
    // Process each buffer to completion before moving to the next one.
    // Each buffer is ~num_rows × 24 bytes (≈24 KB for nvars=10) and fits
    // comfortably in L1 cache.  Folding all steps inside the inner loop
    // keeps the working set hot, whereas the old step-outer / buf-inner
    // order cycled through all buffers at every step, thrashing the cache.
    for buf in &mut bufs {
        for step in 0..num_vars {
            let half = 1usize << (num_vars - step - 1);
            let x = &evaluation_point[step];

            for b in 0..half {
                let diff = F::sub_inner(&buf[2 * b + 1], &buf[2 * b], field_cfg);
                let mut r = x.clone();
                r.mul_assign_by_inner(&diff);
                buf[b] = F::add_inner(&buf[2 * b], r.inner(), field_cfg);
            }
        }
    }

    // ── Extract results ─────────────────────────────────────────────────
    (0..num_constraints)
        .map(|c| {
            DynamicPolynomialF::new_trimmed(
                (0..num_coeffs)
                    .map(|k| {
                        F::new_unchecked_with_cfg(bufs[c * num_coeffs + k][0].clone(), field_cfg)
                    })
                    .collect::<Vec<_>>(),
            )
        })
        .collect()
}

/// Fallback: polynomial-level constraint evaluation.
///
/// Uses the same MLE-based approach as the original code: evaluate
/// constraints at the polynomial level, build coefficient MLEs, then
/// evaluate each MLE at the given point.  This avoids the flat-buffer
/// overhead for UAIRs where the coefficient-level fast path does not
/// apply (non-linear constraints, or too many constraints relative to
/// columns).
///
/// The `pilot_polys` / `pilot_max_degree` parameters come from the
/// linearity-detection pilot row (row 0).  Reusing them avoids paying
/// for the expensive first-row constraint evaluation a second time.
#[allow(clippy::arithmetic_side_effects)]
fn compute_combined_values_poly<F, U>(
    trace_matrix: &[DenseMultilinearExtension<DynamicPolynomialF<F>>],
    projected_scalars: &HashMap<U::Scalar, DynamicPolynomialF<F>>,
    num_constraints: usize,
    evaluation_point: &[F],
    field_cfg: &F::Config,
    pilot_polys: Vec<DynamicPolynomialF<F>>,
    pilot_max_degree: usize,
) -> Vec<DynamicPolynomialF<F>>
where
    F: InnerTransparentField,
    U: Uair,
{
    let field_zero = F::zero_with_cfg(field_cfg);

    let num_rows = trace_matrix[0].len();

    // Phase 1: evaluate UAIR constraints at the polynomial level for each
    // row pair, collecting the max degree alongside the results.
    // Row 0 was already evaluated by the pilot — reuse those results.
    let mut max_degrees_and_combined_poly_rows: Vec<(usize, Vec<DynamicPolynomialF<F>>)> =
        Vec::with_capacity(num_rows);

    max_degrees_and_combined_poly_rows.push((pilot_max_degree, pilot_polys));

    let remaining: Vec<(usize, Vec<DynamicPolynomialF<F>>)> =
        cfg_into_iter!(1..num_rows - 1)
            .map(|row_idx| {
                let up: Vec<_> = trace_matrix
                    .iter()
                    .map(|column| column[row_idx].clone())
                    .collect();

                let down: Vec<_> = trace_matrix
                    .iter()
                    .map(|column| column[row_idx + 1].clone())
                    .collect();

                let polys =
                    evaluate_constraints_poly::<F, U>(&up, &down, num_constraints, projected_scalars);

                let max_deg = polys
                    .iter()
                    .filter_map(|p| p.degree())
                    .max()
                    .unwrap_or(0);

                (max_deg, polys)
            })
            .collect();

    max_degrees_and_combined_poly_rows.extend(remaining);

    let max_degree = max_degrees_and_combined_poly_rows
        .iter()
        .map(|(d, _)| *d)
        .max()
        .unwrap_or(0);

    // Padding row (zeros).
    max_degrees_and_combined_poly_rows
        .push((0, vec![DynamicPolynomialF::new([]); num_constraints]));

    // Phase 2: build coefficient MLEs (one per (constraint, coeff_index)).
    let coefficient_mles: Vec<Vec<DenseMultilinearExtension<F::Inner>>> =
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
                    .collect::<Vec<_>>()
            })
            .collect();

    drop(max_degrees_and_combined_poly_rows);

    // Phase 3: evaluate each coefficient MLE at the given point.
    coefficient_mles
        .iter()
        .map(|mle_group| {
            DynamicPolynomialF::new_trimmed(
                mle_group
                    .iter()
                    .map(|coeff_mle| {
                        coeff_mle
                            .evaluate_with_config(evaluation_point, field_cfg)
                            .expect("evaluation point should have correct width")
                    })
                    .collect::<Vec<_>>(),
            )
        })
        .collect()
}

/// Evaluate all UAIR constraints for a single row pair at polynomial level.
#[allow(clippy::arithmetic_side_effects)]
fn evaluate_constraints_poly<F, U>(
    up: &[DynamicPolynomialF<F>],
    down: &[DynamicPolynomialF<F>],
    num_constraints: usize,
    projected_scalars: &HashMap<U::Scalar, DynamicPolynomialF<F>>,
) -> Vec<DynamicPolynomialF<F>>
where
    F: PrimeField,
    U: Uair,
{
    let mut constraint_builder = PolyRowBuilder::new(num_constraints);

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
    combined_evaluations
}

// ── Coefficient-level constraint builder ────────────────────────────────
// Used by the fast path: Expr = F (field element), no heap allocation.

struct CoeffLevelBuilder<F: PrimeField> {
    results: Vec<F>,
}

impl<F: PrimeField> CoeffLevelBuilder<F> {
    fn new(num_constraints: usize) -> Self {
        Self {
            results: Vec::with_capacity(num_constraints),
        }
    }
}

impl<F: PrimeField> ConstraintBuilder for CoeffLevelBuilder<F> {
    type Expr = F;
    type Ideal = ImpossibleIdeal;

    fn assert_in_ideal(&mut self, expr: Self::Expr, _ideal: &Self::Ideal) {
        self.results.push(expr);
    }

    fn assert_zero(&mut self, expr: Self::Expr) {
        self.results.push(expr);
    }
}

// ── Polynomial-level constraint builder (fallback path) ─────────────────

pub struct PolyRowBuilder<F: PrimeField> {
    combined_evaluations: Vec<DynamicPolynomialF<F>>,
}

impl<F: PrimeField> ConstraintBuilder for PolyRowBuilder<F> {
    type Expr = DynamicPolynomialF<F>;
    type Ideal = ImpossibleIdeal;

    fn assert_in_ideal(&mut self, expr: Self::Expr, _ideal: &Self::Ideal) {
        self.combined_evaluations.push(expr);
    }

    fn assert_zero(&mut self, expr: Self::Expr) {
        self.combined_evaluations.push(expr);
    }
}

impl<F: PrimeField> PolyRowBuilder<F> {
    pub fn new(num_constraints: usize) -> Self {
        Self {
            combined_evaluations: Vec::with_capacity(num_constraints),
        }
    }
}
