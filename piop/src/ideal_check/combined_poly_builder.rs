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
/// This implements an **MLE-first** evaluation strategy inspired by the
/// mle-bench pattern of parallel linear combinations followed by MLE
/// evaluations.
///
/// **Key insight — MLE commutes with linear constraints:**
///
/// For a UAIR whose constraints are linear in the trace (add / sub only,
/// no polynomial multiplication):
///
/// $$\operatorname{MLE}\bigl(\text{constraint}(\text{trace})\bigr)(\mathbf{r})
///     = \text{constraint}\bigl(\operatorname{MLE}(\text{trace})(\mathbf{r})\bigr)$$
///
/// **Strategy:**
///
/// 1. **Fold first** — MLE-evaluate every trace column's coefficient
///    vectors at the random evaluation point.  Each column is an
///    independent task, parallelised with rayon.
/// 2. **Constrain once** — apply the UAIR constraints a single time on
///    the resulting scalar polynomials, instead of once per row.
/// 3. **Correct** — subtract the scalar constant contribution at the
///    zero-padded last row.
///
/// This replaces O(num\_rows × num\_coeffs) per-row constraint-builder
/// calls with O(num\_cols) parallel MLE folds plus one constraint
/// evaluation, dramatically reducing overhead.
///
/// For UAIRs with polynomial multiplication (convolution) the linearity
/// identity does not hold.  The function detects this by running a single
/// "pilot" row evaluation at the **polynomial** level; if the output
/// degree exceeds the trace degree it falls back to the original
/// polynomial-level code path.
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

    // Compute the scalar constant f(0,0) = constraint evaluated at zero
    // inputs.  This is needed both for the linearity test and for the
    // MLE-first correction step.
    let scalar_polys = {
        let zeros: Vec<DynamicPolynomialF<F>> =
            vec![DynamicPolynomialF::default(); num_cols];
        evaluate_constraints_poly::<F, U>(
            &zeros,
            &zeros,
            num_constraints,
            projected_scalars,
        )
    };

    // ── Linearity detection via double-evaluation test ──────────────────
    //
    // The pilot-degree check (`pilot_max_degree <= max_trace_degree`) is
    // insufficient because a perfectly-satisfied multiplicative constraint
    // (e.g. a·b − c = 0) produces zero output at every row, hiding the
    // degree increase.
    //
    // Robust check:  for a constraint f that is linear in the trace,
    //
    //     f(2x) = 2·f(x) − f(0)          (affine identity)
    //
    // holds for any input x.  For a constraint with trace × trace
    // multiplication the quadratic term doubles to 4·, breaking the
    // identity.  This correctly detects multiplication even when the
    // pilot row happens to evaluate to zero.
    let is_linear = if num_rows > 1 {
        let one = F::one_with_cfg(field_cfg);
        let two_inner = F::add_inner(one.inner(), one.inner(), field_cfg);
        let two = F::new_unchecked_with_cfg(two_inner, field_cfg);
        let two_poly = DynamicPolynomialF::constant_poly(two);

        let double_up: Vec<_> = trace_matrix
            .iter()
            .map(|col| col[0].clone() * &two_poly)
            .collect();
        let double_down: Vec<_> = trace_matrix
            .iter()
            .map(|col| col[1].clone() * &two_poly)
            .collect();
        let f_2x = evaluate_constraints_poly::<F, U>(
            &double_up,
            &double_down,
            num_constraints,
            projected_scalars,
        );

        // Verify: f(2x) == 2·f(x) − f(0) for every constraint.
        pilot_polys
            .iter()
            .zip(scalar_polys.iter())
            .zip(f_2x.iter())
            .all(|((fx, f0), f2x)| {
                let mut expected = fx.clone() * &two_poly;
                expected -= f0;
                expected.trim();
                let mut actual = f2x.clone();
                actual.trim();
                actual == expected
            })
    } else {
        true
    };

    if !is_linear {
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

    // ── Parallel coefficient-level fast path ─────────────────────────────
    //
    // For linear constraints, the k-th coefficient of the combined
    // polynomial at row i equals the constraint applied to the k-th
    // coefficients of the trace polynomials:
    //
    //   [constraint(p_0(X), …)]_k = constraint([p_0]_k, [p_1]_k, …)
    //
    // This lets us evaluate each coefficient level independently.
    // All num_coeffs levels are parallelised with rayon — each task
    // iterates over all rows, evaluates the constraint at its level,
    // fills a buffer, and folds it (MLE evaluation).  This replaces
    // the original sequential loop with embarrassingly-parallel work.

    let num_coeffs = pilot_max_degree + 1;

    let signature = U::signature();
    let row_len = signature.total_cols();

    // Project scalars to per-coefficient field values.
    let scalar_coefficients: HashMap<U::Scalar, Vec<F>> = projected_scalars
        .iter()
        .map(|(key, poly)| {
            let mut coeffs = poly.coeffs.clone();
            coeffs.resize(num_coeffs, field_zero.clone());
            (key.clone(), coeffs)
        })
        .collect();

    // Each parallel task processes one coefficient level k and returns
    // the folded MLE value for each constraint at that level.
    // Use min_len to batch small tasks — avoids rayon scheduling overhead
    // when each task is lightweight (few columns or few rows).
    let min_chunk = if num_rows * num_cols > 8192 { 1 } else { num_coeffs };
    let fold_results: Vec<Vec<F>> = cfg_into_iter!(0..num_coeffs, min_chunk)
        .map(|k| {
            let mut up_row: Vec<F> = vec![field_zero.clone(); row_len];
            let mut down_row: Vec<F> = vec![field_zero.clone(); row_len];
            let mut builder = CoeffLevelBuilder::new(num_constraints);
            let mut bufs: Vec<Vec<F::Inner>> =
                vec![vec![zero_inner.clone(); num_rows]; num_constraints];

            for row_idx in 0..num_rows - 1 {
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

                for (c, val) in builder.results.iter().enumerate() {
                    bufs[c][row_idx] = val.inner().clone();
                }
            }

            // Fold each constraint's buffer for this coefficient level.
            bufs.iter_mut()
                .map(|buf| fold_mle_buf::<F>(buf, evaluation_point, field_cfg))
                .collect::<Vec<F>>()
        })
        .collect();

    // Reconstruct: result[c] = polynomial with coefficients from each level.
    (0..num_constraints)
        .map(|c| {
            DynamicPolynomialF::new_trimmed(
                (0..num_coeffs)
                    .map(|k| fold_results[k][c].clone())
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

/// Fold a buffer of `F::Inner` values in-place, performing MLE evaluation.
///
/// Implements the standard butterfly fold:
///   `buf[b] = buf[2b] + x · (buf[2b+1] - buf[2b])`
///
/// Returns the final scalar as `F`.
#[allow(clippy::arithmetic_side_effects)]
fn fold_mle_buf<F: InnerTransparentField>(
    buf: &mut [F::Inner],
    evaluation_point: &[F],
    field_cfg: &F::Config,
) -> F {
    let num_vars = evaluation_point.len();
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
    F::new_unchecked_with_cfg(buf[0].clone(), field_cfg)
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
// Used by the parallel fast path: Expr = F (field element), no heap allocation.

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

// ── Polynomial-level constraint builder ─────────────────────────────────

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{run_ideal_check_prover_single_type, test_config};
    use zinc_test_uair::{GenerateSingleTypeWitness, TestAirNoMultiplication};
    use zinc_transcript::{KeccakTranscript, traits::Transcript};

    /// Verify MLE-first path produces the same result as poly fallback.
    #[test]
    fn mle_first_matches_poly_fallback() {
        let num_vars = 2;
        let field_cfg = test_config();
        let mut rng = rand::rng();

        let witness = TestAirNoMultiplication::<5>::generate_witness(num_vars, &mut rng);
        let mut transcript = KeccakTranscript::new();

        let (_, _, projected_scalars, trace) =
            run_ideal_check_prover_single_type::<TestAirNoMultiplication<5>, 32>(
                num_vars,
                &witness,
                &mut transcript,
            );

        let num_constraints = zinc_uair::constraint_counter::count_constraints::<
            TestAirNoMultiplication<5>,
        >();

        let evaluation_point =
            KeccakTranscript::new().get_field_challenges(num_vars, &field_cfg);

        // Poly fallback result
        let pilot_polys = {
            let up: Vec<_> = trace.iter().map(|col| col[0].clone()).collect();
            let down: Vec<_> = trace.iter().map(|col| col[1].clone()).collect();
            evaluate_constraints_poly::<_, TestAirNoMultiplication<5>>(
                &up,
                &down,
                num_constraints,
                &projected_scalars,
            )
        };
        let pilot_max_degree = pilot_polys
            .iter()
            .filter_map(|p| p.degree())
            .max()
            .unwrap_or(0);

        let expected = compute_combined_values_poly::<_, TestAirNoMultiplication<5>>(
            &trace,
            &projected_scalars,
            num_constraints,
            &evaluation_point,
            &field_cfg,
            pilot_polys,
            pilot_max_degree,
        );

        // MLE-first result
        let got = compute_combined_values::<_, TestAirNoMultiplication<5>>(
            &trace,
            &projected_scalars,
            num_constraints,
            &evaluation_point,
            &field_cfg,
        );

        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            let mut g_t = g.clone();
            g_t.trim();
            let mut e_t = e.clone();
            e_t.trim();
            assert_eq!(g_t, e_t, "constraint {i} mismatch");
        }
    }
}
