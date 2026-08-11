//! Ideal-check subprotocol.
mod batched_ideal_check;
mod combined_poly_builder;
mod structs;

pub use batched_ideal_check::BatchedIdealCheckError;
pub use structs::*;

use crate::projections::{
    ColumnMajorTrace, ProjectedScalars, RowMajorTrace, column_major_to_row_major,
};
use batched_ideal_check::*;
use crypto_primitives::{BaseFieldConfig, ProjectPrimitiveIntegersWithConfig};
use std::marker::PhantomData;
use thiserror::Error;
use zinc_poly::{
    EvaluationError,
    univariate::dynamic::{DynamicPolynomial, DynamicPolynomialConfig, HasDynamicPolynomialConfig},
    utils::ArithErrors as PolyArithErrors,
};
use zinc_transcript::traits::{ConstTranscribable, Transcript};
use zinc_uair::{
    Uair,
    degree_counter::count_constraint_degrees,
    ideal::{Ideal, IdealCheck},
    ideal_collector::{IdealOrZero, collect_ideals},
};
use zinc_utils::sub;

/// Ideal-check subprotocol.
///
/// The evaluation point $r \in F^\mu$ at which the
/// combined polynomial MLEs are pinned down is supplied **by the caller**
/// rather than squeezed from the transcript inside this subprotocol:
///
/// The protocol layer samples a single shared integer vector
/// $r \in [0, q^*)^\mu$ once and projects it into each family's field,
/// so all $n + 1$ families re-use the same underlying integers (just typed in
/// their respective fields).
#[derive(Default, Clone, Copy)]
pub struct IdealCheckProtocol<U: Uair>(PhantomData<U>);

impl<U: Uair> IdealCheckProtocol<U> {
    /// Prover using MLE-first evaluation (column-indexed trace).
    ///
    /// Routes each constraint through the most efficient evaluation path:
    /// - Linear constraints with non-zero ideals are batched through
    ///   [`evaluate_combined_polynomials`], which evaluates trace column MLEs
    ///   at the challenge point and then applies the constraints to the
    ///   evaluated values.
    /// - Non-linear constraints with non-zero ideals fall back to the row-major
    ///   [`evaluate_for_constraints`] path; the trace is transposed on demand
    ///   via [`column_major_to_row_major`].
    /// - Constraints with zero ideals are short-circuited to zero (their
    ///   combined polynomial value is zero by construction for an honest
    ///   prover).
    ///
    /// For $F_{q_i}[X]$ constraints, `trace_matrix` and `projected_scalars`
    /// must already be projected mod $q_i$.
    ///
    /// # Parameters
    /// - `transcript`: the Fiat-Shamir transcript.
    /// - `trace_matrix`: input trace for the UAIR `U` projected to
    ///   `DynamicPolynomialF<F>`, column-indexed: `trace_matrix[col][row]`.
    /// - `projected_scalars`: UAIR scalars projected to
    ///   `DynamicPolynomialF<F>`.
    /// - `family_idx`: which constraint family to prove. `0` -> $Q[X]$; `i >=
    ///   1` -> $F_{q_{i-1}}[X]$ (i.e. UAIR-level `prime_idx = i - 1`).
    /// - `num_constraints`: number of constraints this UAIR encodes.
    /// - `evaluation_point`: pre-sampled MLE evaluation point, shared for all
    ///   families of constraints ($Q[X]$ and $F_q[X]$).
    /// - `field_cfg`: random field configuration sampled on the previous steps
    ///   of the overall protocol.
    #[allow(clippy::type_complexity, clippy::arithmetic_side_effects)]
    pub fn prove_mle_first<C, const DEGREE_PLUS_ONE: usize>(
        transcript: &mut impl Transcript,
        trace_matrix: &ColumnMajorTrace<C::Element>,
        projected_scalars: &ProjectedScalars<U::Scalar, DynamicPolynomial<C::Element>>,
        family_idx: usize,
        num_constraints: usize,
        evaluation_point: &[C::Element],
        field_cfg: &C,
    ) -> Result<Proof<C::Element>, IdealCheckError<C::Element>>
    where
        C: BaseFieldConfig + ProjectPrimitiveIntegersWithConfig,
        C::Integer: ConstTranscribable,
    {
        // Classify constraints to drive dispatch below:
        // * Linear non-zero-ideal goes through the column-major MLE-first path
        // * Linear zero-ideal constraints need no tracking — their value is zero by
        //   construction.
        // * Non-linear non-zero-ideal goes through the row-major fallback
        // * Non-linear zero-ideal entries are zeroed afterwards.
        let ideal_collector = collect_ideals::<U>(num_constraints);
        let degrees = count_constraint_degrees::<U>();

        let mut has_linear_nonzero: bool = false;
        let mut nonlinear_zero: Vec<usize> = Vec::new();
        let mut nonlinear_nonzero: Vec<usize> = Vec::new();

        macro_rules! categorize_ideals {
            ($ideals:expr, $degrees:expr) => {
                for (idx, ideal) in $ideals.iter().enumerate() {
                    if $degrees[idx] <= 1 {
                        has_linear_nonzero |= !ideal.is_zero_ideal();
                    } else if ideal.is_zero_ideal() {
                        nonlinear_zero.push(idx);
                    } else {
                        nonlinear_nonzero.push(idx);
                    }
                }
            };
        }

        if family_idx == 0 {
            categorize_ideals!(ideal_collector.ideals, degrees.q_degrees);
        } else {
            let prime_idx = family_idx - 1;
            categorize_ideals!(
                ideal_collector.fq_ideals[prime_idx],
                degrees.fq_degrees[prime_idx]
            );
        }

        // When any linear non-zero-ideal constraint exists, run
        // `evaluate_combined_polynomials` once: all linear entries (zero or not) come
        // out correct.
        // Non-linear entries are garbage and get replaced below.
        let mut combined_mle_values: Vec<DynamicPolynomial<C::Element>> = if has_linear_nonzero {
            let mut res =
                combined_poly_builder::evaluate_combined_polynomials::<_, U, DEGREE_PLUS_ONE>(
                    trace_matrix,
                    projected_scalars,
                    family_idx,
                    num_constraints,
                    evaluation_point,
                    field_cfg,
                )?;

            // Scrub garbage left by `evaluate_combined_polynomials` at non-linear
            // zero-ideal indices.
            for &i in &nonlinear_zero {
                res[i] = DynamicPolynomial::ZERO;
            }

            res
        } else {
            vec![DynamicPolynomial::ZERO; num_constraints]
        };

        if !nonlinear_nonzero.is_empty() {
            // `evaluate_for_constraints` works with row-major traces, so we have to
            // transpose here.
            // TODO(alex): Can/should we avoid the transposition?
            let row_major = column_major_to_row_major(trace_matrix);
            let values = combined_poly_builder::evaluate_for_constraints::<_, U, DEGREE_PLUS_ONE>(
                &row_major,
                projected_scalars,
                family_idx,
                num_constraints,
                field_cfg,
                &nonlinear_nonzero,
                evaluation_point,
            )?;
            for (&i, v) in nonlinear_nonzero.iter().zip(values) {
                combined_mle_values[i] = v;
            }
        }

        let mut transcription_buf: Vec<u8> = vec![0; <C::Integer as ConstTranscribable>::NUM_BYTES];
        combined_mle_values.iter().for_each(|cv| {
            transcript.absorb_field_element_slice(field_cfg, &cv.coeffs, &mut transcription_buf);
        });

        Ok(Proof {
            combined_mle_values,
        })
    }

    /// Prover for any UAIR using combined polynomial construction.
    ///
    /// Uses row-indexed (transposed) trace for efficient row-by-row
    /// combined polynomial construction.
    ///
    /// For $F_{q_i}[X]$ constraints, `trace_matrix` and `projected_scalars`
    /// must already be projected mod $q_i$.
    ///
    /// # Parameters
    /// - `transcript`: the Fiat-Shamir transcript.
    /// - `trace_matrix`: input trace for the UAIR `U` projected to
    ///   `DynamicPolynomialF<F>`, row-indexed: `trace_matrix[row][col]`.
    /// - `projected_scalars`: UAIR scalars projected to
    ///   `DynamicPolynomialF<F>`.
    /// - `family_idx`: which constraint family to prove. `0` -> $Q[X]$; `i >=
    ///   1` -> $F_{q_{i-1}}[X]$.
    /// - `num_constraints`: number of constraints this UAIR encodes.
    /// - `evaluation_point`: pre-sampled MLE evaluation point, shared for all
    ///   families of constraints ($Q[X]$ and $F_q[X]$).
    /// - `field_cfg`: random field configuration sampled on the previous steps
    ///   of the overall protocol.
    #[allow(clippy::type_complexity)]
    pub fn prove_combined<C, const DEGREE_PLUS_ONE: usize>(
        transcript: &mut impl Transcript,
        trace_matrix: &RowMajorTrace<C::Element>,
        projected_scalars: &ProjectedScalars<U::Scalar, DynamicPolynomial<C::Element>>,
        family_idx: usize,
        num_constraints: usize,
        evaluation_point: &[C::Element],
        field_cfg: &C,
    ) -> Result<Proof<C::Element>, IdealCheckError<C::Element>>
    where
        C: BaseFieldConfig + ProjectPrimitiveIntegersWithConfig,
        C::Integer: ConstTranscribable,
    {
        // Collect ideals to identify assert_zero constraints whose
        // combined polynomial is zero by construction (for honest provers).
        macro_rules! get_non_zero_indices {
            ($ideals:expr) => {
                $ideals
                    .iter()
                    .enumerate()
                    .filter(|(_, i)| !i.is_zero_ideal())
                    .map(|(idx, _)| idx)
                    .collect::<Vec<usize>>()
            };
        }
        let ideal_collector = collect_ideals::<U>(num_constraints);
        let non_zero_indices: Vec<usize> = if family_idx == 0 {
            get_non_zero_indices!(ideal_collector.ideals)
        } else {
            let prime_idx = sub!(family_idx, 1);
            get_non_zero_indices!(ideal_collector.fq_ideals.get(prime_idx).unwrap_or(&vec![]))
        };

        let mut combined_mle_values = vec![DynamicPolynomial::ZERO; num_constraints];
        if !non_zero_indices.is_empty() {
            let computed = combined_poly_builder::evaluate_for_constraints::<_, U, DEGREE_PLUS_ONE>(
                trace_matrix,
                projected_scalars,
                family_idx,
                num_constraints,
                field_cfg,
                &non_zero_indices,
                evaluation_point,
            )?;

            for (&ci, val) in non_zero_indices.iter().zip(computed) {
                combined_mle_values[ci] = val;
            }
        };

        let mut transcription_buf: Vec<u8> = vec![0; <C::Integer as ConstTranscribable>::NUM_BYTES];

        combined_mle_values.iter().for_each(|v| {
            transcript.absorb_field_element_slice(field_cfg, &v.coeffs, &mut transcription_buf);
        });

        Ok(Proof {
            combined_mle_values,
        })
    }

    /// The verifier part of the ideal-check subprotocol.
    ///
    /// Mirrors the prover: receives the prover's `combined_mle_values`,
    /// absorbs them into the transcript, then checks each non-trivial
    /// constraint's value against the corresponding lifted ideal.
    ///
    /// # Parameters
    /// - `transcript`: the Fiat-Shamir transcript.
    /// - `proof`: a purported proof produced by the prover.
    /// - `family_idx`: which constraint family to verify. `0` -> $Q[X]$; `i >=
    ///   1` -> $F_{q_{i-1}}[X]$.
    /// - `num_constraints`: the number of constraints the UAIR `U` encodes.
    /// - `evaluation_point`: pre-sampled MLE evaluation point matching the one
    ///   used by the prover. The caller is responsible for ensuring transcript
    ///   ordering matches the prover.
    /// - `ideal_over_f_from_ref`: since the UAIR `U` is not aware of the field
    ///   the ideal check is operating on it defines ideals over the ring
    ///   `IcTypes::Witness`. `ideal_over_f_from_ref` allows to convert the
    ///   ideals over `IcTypes::Witness` into ideals over the field
    ///   `IcTypes::F`. Think of this as a projection for ideals.
    /// - `field_cfg`: random field configuration sampled on the previous steps
    ///   of the overall protocol.
    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    pub fn verify_as_subprotocol<'cfg, C, IdealOverF, IdealOverFFromRef, IdealOverFFromFqRef>(
        transcript: &mut impl Transcript,
        proof: Proof<C::Element>,
        family_idx: usize,
        num_constraints: usize,
        evaluation_point: &[C::Element],
        ideal_over_f_from_ref: IdealOverFFromRef,
        ideal_over_f_from_fq_ref: IdealOverFFromFqRef,
        field_cfg: &'cfg C,
    ) -> Result<VerifierSubclaim<C::Element>, IdealCheckError<C::Element>>
    where
        C: BaseFieldConfig,
        C::Integer: ConstTranscribable,
        IdealOverF: Ideal + IdealCheck<DynamicPolynomialConfig<'cfg, C>>,
        IdealOverFFromRef: Fn(&IdealOrZero<U::Ideal>) -> IdealOverF,
        IdealOverFFromFqRef: Fn(&IdealOrZero<U::FqIdeal>) -> IdealOverF,
    {
        let mut transcription_buf: Vec<u8> = vec![0; <C::Integer as ConstTranscribable>::NUM_BYTES];

        let combined_mle_values = proof.combined_mle_values;

        for mle_value in &combined_mle_values {
            transcript.absorb_field_element_slice(
                field_cfg,
                &mle_value.coeffs,
                &mut transcription_buf,
            );
        }

        let ideal_collector = collect_ideals::<U>(num_constraints);

        macro_rules! collect_non_trivial {
            ($ideals:expr, $ideal_from_ref:ident) => {
                $ideals
                    .iter()
                    .zip(combined_mle_values.iter())
                    .filter(|(ideal, _)| !ideal.is_zero_ideal())
                    .map(|(ideal, value)| ($ideal_from_ref(ideal), value.clone()))
                    .unzip()
            };
        }

        // Only check non-trivial ideals. For assert_zero constraints
        // the ideal is the zero ideal and the combined polynomial
        // value is zero by construction; the sumcheck that follows
        // verifies consistency of the claimed evaluations with the
        // actual trace.
        let (non_trivial_ideals, non_trivial_values): (Vec<_>, Vec<_>) = if family_idx == 0 {
            collect_non_trivial!(ideal_collector.ideals, ideal_over_f_from_ref)
        } else {
            let prime_idx = sub!(family_idx, 1);
            collect_non_trivial!(
                ideal_collector.fq_ideals.get(prime_idx).unwrap_or(&vec![]),
                ideal_over_f_from_fq_ref
            )
        };

        batched_ideal_check(
            &field_cfg.dyn_poly_cfg(),
            &non_trivial_ideals,
            &non_trivial_values,
        )?;

        Ok(VerifierSubclaim {
            evaluation_point: evaluation_point.to_vec(),
            values: combined_mle_values,
        })
    }
}

#[derive(Clone, Debug, Error)]
pub enum IdealCheckError<F> {
    #[error("ideal check prover failed to evaluate an mle: {0}")]
    MleEvaluationError(#[from] EvaluationError),
    #[error("mle evaluation ideal check failure: {0}")]
    IdealCollectorError(#[from] BatchedIdealCheckError<DynamicPolynomial<F>>),
    #[error("`eq` polynomial construction failure: {0}")]
    EqPolyConstructionError(#[from] PolyArithErrors),
}

#[allow(clippy::arithmetic_side_effects, clippy::redundant_clone)]
#[cfg(test)]
mod tests {
    use crate::test_utils::{
        LIMBS, run_ideal_check_prover_combined, run_ideal_check_prover_linear, test_config,
    };
    use crypto_primitives::{
        crypto_bigint_int::Int, crypto_bigint_monty::MontyField, crypto_bigint_uint::Uint,
    };
    use rand::rng;
    use zinc_poly::univariate::{dense::DensePolynomial, dynamic::DynamicPolynomialConfig};
    use zinc_test_uair::{
        GenerateRandomTrace, TestUairBitOpsMixedSplice, TestUairNoMultiplication,
        TestUairSimpleMultiplication,
    };
    use zinc_transcript::Blake3Transcript;
    use zinc_uair::{
        constraint_counter::count_constraints,
        ideal::{DegreeOneIdeal, Ideal, IdealCheck},
    };

    use super::*;

    // TODO(Ilia): These tests are absolute joke.
    //             Once we have time we need to create a comprehensive test suite
    //             akin to the one we have for the PCS or the sumcheck.

    fn do_test<
        U,
        IdealOverF,
        IdealOverFFromRef,
        IdealOverFFromFqRef,
        const DEGREE_PLUS_ONE: usize,
    >(
        num_vars: usize,
        prime_idx: Option<usize>,
        ideal_over_f_from_ref: IdealOverFFromRef,
        ideal_over_f_from_fq_ref: IdealOverFFromFqRef,
    ) where
        U: Uair<Scalar = DensePolynomial<Int<5>, DEGREE_PLUS_ONE>>
            + GenerateRandomTrace<DEGREE_PLUS_ONE, PolyCoeff = Int<5>, Int = Int<5>>,
        IdealOverF: Ideal + for<'a> IdealCheck<DynamicPolynomialConfig<'a, MontyField<LIMBS>>>,
        IdealOverFFromRef: Fn(&IdealOrZero<U::Ideal>) -> IdealOverF + Copy,
        IdealOverFFromFqRef: Fn(&IdealOrZero<U::FqIdeal>) -> IdealOverF + Copy,
    {
        let mut rng = rng();
        let field_cfg = test_config();

        let num_constraints = count_constraints::<U>();
        let num_constraints = prime_idx
            .map(|i| num_constraints.for_prime(i))
            .unwrap_or(num_constraints.q);
        let family_idx = prime_idx.map_or(0, |i| i + 1);

        // Combined approach
        {
            let transcript = Blake3Transcript::new();
            let (proof, evaluation_point, ..) = run_ideal_check_prover_combined::<U, DEGREE_PLUS_ONE>(
                num_vars,
                &U::generate_random_trace(num_vars, &mut rng),
                prime_idx,
                &mut transcript.clone(),
            );

            // With the unified-challenge shape, the evaluation point is
            // sampled by the caller; mirror the prover here.
            let verifier_result = IdealCheckProtocol::<U>::verify_as_subprotocol(
                &mut transcript.clone(),
                proof,
                family_idx,
                num_constraints,
                &evaluation_point,
                ideal_over_f_from_ref,
                ideal_over_f_from_fq_ref,
                &field_cfg,
            )
            .expect("Verification failed");

            assert_eq!(evaluation_point, verifier_result.evaluation_point);
        }

        // MLE-first
        {
            let transcript = Blake3Transcript::new();
            let (proof, evaluation_point, ..) = run_ideal_check_prover_linear::<U, DEGREE_PLUS_ONE>(
                num_vars,
                &U::generate_random_trace(num_vars, &mut rng),
                prime_idx,
                &mut transcript.clone(),
            );

            let verifier_result = IdealCheckProtocol::<U>::verify_as_subprotocol(
                &mut transcript.clone(),
                proof,
                family_idx,
                num_constraints,
                &evaluation_point,
                ideal_over_f_from_ref,
                ideal_over_f_from_fq_ref,
                &field_cfg,
            )
            .expect("Verification failed");

            assert_eq!(evaluation_point, verifier_result.evaluation_point);
        }
    }

    #[test]
    fn test_successful_verification() {
        let field_cfg = test_config();

        let num_vars = 2;

        // Linear UAIR with non-zero ideals
        do_test::<TestUairNoMultiplication<Int<5>, Uint<LIMBS>>, _, _, _, 32>(
            num_vars,
            None,
            |ideal_over_ring| ideal_over_ring.map(|i| DegreeOneIdeal::project(&field_cfg, i)),
            |_| unreachable!("F_q[X] should not be used for this UAIR"),
        );

        // Non-linear UAIR with all-zero ideals
        do_test::<TestUairSimpleMultiplication<Int<5>, Uint<LIMBS>>, _, _, _, 32>(
            num_vars,
            None,
            |_ideal_over_ring| IdealOrZero::<DegreeOneIdeal<_>>::zero(),
            |_| unreachable!("F_q[X] should not be used for this UAIR"),
        );

        // Linear UAIR with bit-op virtuals and mixed down-row splicing.
        do_test::<TestUairBitOpsMixedSplice<Int<5>, Uint<LIMBS>>, _, _, _, 32>(
            num_vars,
            None,
            |ideal_over_ring| ideal_over_ring.map(|i| DegreeOneIdeal::project(&field_cfg, i)),
            |_| unreachable!("F_q[X] should not be used for this UAIR"),
        );
    }
}
