//! Combined polynomial resolver subprotocol.

mod folder;
mod structs;

pub use structs::*;

use crate::{
    CombFn,
    combined_poly_resolver::{
        folder::ConstraintFolder,
        structs::{Proof as CprProof, ProverState as CprProverState},
    },
    ideal_check,
    sumcheck::{
        SumCheckError, multi_degree::MultiDegreeSumcheckGroup,
        prover::ProverState as SumcheckProverState,
    },
};
use crypto_primitives::{FromPrimitiveWithConfig, PrimeField};
use itertools::Itertools;
use num_traits::Zero;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::{collections::HashMap, marker::PhantomData, slice};
use thiserror::Error;
use zinc_poly::{
    EvaluationError,
    mle::{DenseMultilinearExtension, MultilinearExtensionWithConfig},
    univariate::{
        binary::BinaryPoly,
        dynamic::over_field::{DynamicPolyFInnerProduct, DynamicPolynomialF},
    },
    utils::{ArithErrors, build_eq_x_r_inner, eq_eval},
};
use zinc_transcript::traits::{ConstTranscribable, Transcript};
use zinc_uair::{BitOp, TraceRow, Uair, ideal::ImpossibleIdeal};
use zinc_utils::{
    UNCHECKED, add, cfg_iter, from_ref::FromRef, inner_product::InnerProduct,
    inner_transparent_field::InnerTransparentField, powers,
};

/// Materialize the bit-op virtual MLEs given by `bit_op_specs`.
///
/// For each `BitOpSpec(src, op)`, walks the source binary column
/// cell-wise, applies `op` as a bit-position permutation on the 32
/// coefficients of each cell, and ψ_α-projects the result. The
/// output is a length-`n` MLE in `F::Inner` that downstream
/// subprotocols can consume as an extra MLE source.
///
/// Used by both the CPR sumcheck (Step 4) and the multipoint-eval
/// sumcheck (Step 5): bit-op virtual columns become additional
/// sources for mp_eval's `up` reduction at `r*`.
///
/// # Panics
///
/// - If a `BitOpSpec.source_col()` references a non-binary-poly
///   column (i.e. `>= num_total_bin`).
/// - If `D != 32` (bit ops are defined on 32-coefficient cells).
#[allow(clippy::arithmetic_side_effects)]
pub fn build_bit_op_mles<F, const D: usize>(
    trace_bin_poly: &[DenseMultilinearExtension<BinaryPoly<D>>],
    bit_op_specs: &[zinc_uair::BitOpSpec],
    num_total_bin: usize,
    projecting_element_f: &F,
    num_vars: usize,
    field_cfg: &F::Config,
) -> Vec<DenseMultilinearExtension<F::Inner>>
where
    F: InnerTransparentField,
    F::Inner: Zero + Default + Clone,
{
    let zero = F::zero_with_cfg(field_cfg);
    let one = F::one_with_cfg(field_cfg);
    let zero_inner = zero.clone().into_inner();
    let n = 1usize << num_vars;

    // Pre-build per-op power arrays of length 32: position-`i` entry
    // is α^{p(i)} where p is the bit-op's permutation. Coefficient
    // positions past 31 (only relevant if D > 32, which we don't
    // support for bit ops) get a zero entry.
    let alpha_powers: Vec<F> = powers(projecting_element_f.clone(), one, 32);

    bit_op_specs
        .iter()
        .map(|spec| {
            assert!(
                spec.source_col() < num_total_bin,
                "BitOpSpec source_col {} must reference a binary_poly column \
                 (num binary cols = {num_total_bin})",
                spec.source_col(),
            );
            assert!(
                D == 32,
                "BitOpSpec virtual columns require D == 32, got D = {D}",
            );
            // Build the per-bit weight table `w[i] = α^{p(i)}` where
            // `p` is the bit-op's coefficient permutation, returning
            // `None` for positions dropped by ShiftR.
            let mut weights: Vec<Option<F>> = vec![None; D];
            match spec.op() {
                BitOp::Rot(c) => {
                    for i in 0..32usize.min(D) {
                        let dst = (i + c as usize) % 32;
                        weights[i] = Some(alpha_powers[dst].clone());
                    }
                }
                BitOp::ShiftR(c) => {
                    let c = c as usize;
                    for i in c..32usize.min(D) {
                        let dst = i - c;
                        weights[i] = Some(alpha_powers[dst].clone());
                    }
                }
            }
            let col = &trace_bin_poly[spec.source_col()];
            let evals: Vec<F::Inner> = col
                .iter()
                .map(|cell| {
                    let mut acc = zero.clone();
                    for (i, coeff) in cell.iter().enumerate().take(D) {
                        if !coeff.into_inner() {
                            continue;
                        }
                        if let Some(w) = &weights[i] {
                            acc += w.clone();
                        }
                    }
                    acc.into_inner()
                })
                .collect();
            let mut evals = evals;
            evals.resize(n, zero_inner.clone());
            DenseMultilinearExtension {
                evaluations: evals,
                num_vars,
            }
        })
        .collect()
}

pub struct CombinedPolyResolver<F: InnerTransparentField>(PhantomData<F>);

impl<F: InnerTransparentField + FromPrimitiveWithConfig + Send + Sync> CombinedPolyResolver<F> {
    /// Build the CPR sumcheck group for use in the multi-degree sumcheck.
    ///
    /// Pre-sumcheck half of the CPR prover. Samples the folding challenge `α`,
    /// builds the MLE vector and combination function with the constraint
    /// polynomial identity:
    ///
    /// $$
    /// \sum_{b \in H} (f_0(b, x_0[b],...,x_n[b], x_0ˆdown[b],...,x_nˆdown[b])
    ///                 + \alpha f_1(...) + ... + \alpha^k f_k(...)) = v_0 +
    ///                   \alpha * v_1 + ... + \alphaˆk * v_k,
    /// $$
    /// where $f_i(b, x_0[b],...,x_n[b], x_0ˆdown[b],...,x_nˆdown[b])
    ///         = eq(r, b) * (1 - eq(r, 1,...1))
    ///             * g_i(x_0[b],...,x_n[b], x_0ˆdown[b],...,x_nˆdown[b])$
    /// and `g_i` is a constraint polynomial given by the UAIR `U`.
    /// `v_0,...,v_k` are the claimed evaluations of the combined polynomials.
    ///
    /// # Parameters
    /// - `transcript`: FS-transcript.
    /// - `trace_matrix`: The trace that have been projected to F.
    /// - `evaluation_point`: The evaluation point for the claims.
    /// - `projected_scalars`: The UAIR scalars projected to `F`.
    /// - `num_constraints`: The number of constraint polynomials in the UAIR
    ///   `U`.
    /// - `num_vars`: The number of variables of the trace MLEs.
    /// - `max_degree`: The degree of the UAIR `U`.
    /// - `field_cfg`: The random field config.
    #[allow(clippy::arithmetic_side_effects, clippy::too_many_arguments)]
    pub fn prepare_sumcheck_group<U, const D: usize>(
        transcript: &mut impl Transcript,
        trace_matrix: Vec<DenseMultilinearExtension<F::Inner>>,
        evaluation_point: &[F],
        projected_scalars: &HashMap<U::Scalar, F>,
        num_constraints: usize,
        num_vars: usize,
        max_degree: usize,
        field_cfg: &F::Config,
        trace_bin_poly: &[DenseMultilinearExtension<BinaryPoly<D>>],
        projecting_element_f: &F,
    ) -> Result<(MultiDegreeSumcheckGroup<F>, CprProverAncillary), CombinedPolyResolverError<F>>
    where
        F::Inner: ConstTranscribable + Send + Sync + Zero + Default,
        F::Modulus: ConstTranscribable,
        F: 'static,
        U::Scalar: 'static,
        U: Uair,
    {
        debug_assert_ne!(
            num_vars, 1,
            "The protocol is not needed when the number of variables is 1 :)"
        );

        let zero = F::zero_with_cfg(field_cfg);
        let one = F::one_with_cfg(field_cfg);

        // Shifted trace: for each ShiftSpec, take the source column,
        // drop the first `shift_amount` rows, and zero-pad to the full
        // domain size so the MLE keeps the correct `num_vars`.
        // TODO consider working with pointers since down cols are virtual cols until
        // folded in sumcheck - virtual MLE trait needed in sumcheck.
        let uair_sig = U::signature();
        let zero_inner = zero.clone().into_inner();
        let n = 1usize << num_vars;
        let down: Vec<DenseMultilinearExtension<F::Inner>> = cfg_iter!(uair_sig.shifts())
            .map(|spec| {
                let mut evals = trace_matrix[spec.source_col()][spec.shift_amount()..].to_vec();
                evals.resize(n, zero_inner.clone());
                DenseMultilinearExtension {
                    evaluations: evals,
                    num_vars,
                }
            })
            .collect();

        // Bit-op virtual columns: materialize via the shared helper so
        // mp_eval can build the same MLEs (mp_eval consumes them as
        // additional sources at r*).
        let bit_op_specs = uair_sig.bit_op_specs();
        let bit_op_down = build_bit_op_mles::<F, D>(
            trace_bin_poly,
            bit_op_specs,
            uair_sig.total_cols().num_binary_poly_cols(),
            projecting_element_f,
            num_vars,
            field_cfg,
        );

        let eq_r = build_eq_x_r_inner(evaluation_point, field_cfg)?;
        // To get the constraints on the last row ignored
        // we multiply each constraint polynomial
        // by the selector (1 - eq(1,...,1, x))
        let last_row_selector = DenseMultilinearExtension {
            num_vars,
            evaluations: {
                let mut evals = vec![zero.inner().clone(); 1 << num_vars];
                evals[(1 << num_vars) - 1] = one.inner().clone();
                evals
            },
        };

        // The challenge '\alpha' to batch multiple evaluation claims
        let folding_challenge: F = transcript.get_field_challenge(field_cfg);

        let folding_challenge_powers: Vec<F> =
            powers(folding_challenge, one.clone(), num_constraints);

        let num_cols = trace_matrix.len();
        let num_down_cols = down.len();
        let num_bit_op_cols = bit_op_down.len();
        let mles: Vec<DenseMultilinearExtension<F::Inner>> = {
            let mut mles = Vec::with_capacity(2 + num_cols + num_down_cols + num_bit_op_cols);

            mles.push(last_row_selector);
            mles.push(eq_r);

            mles.extend(trace_matrix);
            mles.extend(down);
            mles.extend(bit_op_down);

            mles
        };

        let projected_scalars = projected_scalars.clone();
        let comb_fn: CombFn<F> = Box::new(move |mle_values: &[F]| {
            let uair_sig = U::signature();
            let up_layout = uair_sig.total_cols().as_column_layout();
            let down_layout = uair_sig.down_cols().as_column_layout();
            let bit_op_count = uair_sig.bit_op_down_count();

            let selector = &mle_values[0];
            let eq_r = &mle_values[1];

            let mut folder = ConstraintFolder::new(&folding_challenge_powers, &zero);

            let project = |scalar: &U::Scalar| {
                projected_scalars
                    .get(scalar)
                    .cloned()
                    .expect("all scalars should have been projected at this point")
            };

            U::constrain_general(
                &mut folder,
                TraceRow::from_slice_with_layout(&mle_values[2..num_cols + 2], up_layout),
                TraceRow::from_slice_with_layout_and_bit_op(
                    &mle_values[num_cols + 2..],
                    down_layout,
                    bit_op_count,
                ),
                project,
                |x, y| Some(project(y) * x),
                ImpossibleIdeal::from_ref,
            );

            folder.folded_constraints * (one.clone() - selector) * eq_r
        });

        Ok((
            MultiDegreeSumcheckGroup::new(max_degree + 2, mles, comb_fn),
            CprProverAncillary {
                num_cols,
                num_down_cols,
                num_bit_op_cols,
                num_vars,
            },
        ))
    }

    /// Finalize the CPR proof after the multi-degree sumcheck completes.
    ///
    /// # Parameters
    /// - `transcript`: FS-transcript (absorbs `up_evals` and `down_evals`).
    /// - `sumcheck_prover_state`: The CPR group's `ProverState` from
    ///   `MultiDegreeSumcheck::prove_as_subprotocol` (states\[0\]).
    /// - `ancillary`: Produced by [`prepare_sumcheck_group`]; carries column
    ///   counts and `num_vars` needed to split the flat eval vector.
    /// - `field_cfg`: Field configuration.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn finalize_prover(
        transcript: &mut impl Transcript,
        sumcheck_prover_state: SumcheckProverState<F>,
        ancillary: CprProverAncillary,
        field_cfg: &F::Config,
    ) -> Result<(CprProof<F>, CprProverState<F>), CombinedPolyResolverError<F>>
    where
        F::Inner: ConstTranscribable + Zero,
        F::Modulus: ConstTranscribable,
    {
        // Sumcheck prover stops evaluating MLEs
        // at the second to last challenge
        // leaving all MLEs in num_vars=1
        // state. We need to evaluate them up
        // and send to the verifier.
        debug_assert!(
            sumcheck_prover_state
                .mles
                .iter()
                .all(|mle| mle.num_vars == 1)
        );

        let last_sumcheck_challenge = sumcheck_prover_state
            .randomness
            .last()
            .expect("sumcheck could not have had 0 rounds");

        let mut mles = sumcheck_prover_state.mles;
        let evals: Vec<F> = mles
            .drain(2..)
            .map(|mle| {
                mle.evaluate_with_config(slice::from_ref(last_sumcheck_challenge), field_cfg)
            })
            .try_collect()?;

        debug_assert_eq!(
            evals.len(),
            ancillary.num_cols + ancillary.num_down_cols + ancillary.num_bit_op_cols
        );
        let mut transcription_buf: Vec<u8> = vec![0; F::Inner::NUM_BYTES];
        transcript.absorb_random_field_slice(&evals, &mut transcription_buf);
        let down_end = ancillary.num_cols + ancillary.num_down_cols;
        let up_evals = evals[0..ancillary.num_cols].to_vec();
        let down_evals = evals[ancillary.num_cols..down_end].to_vec();
        let bit_op_down_evals = evals[down_end..].to_vec();
        Ok((
            CprProof {
                up_evals,
                down_evals,
                // Filled by `booleanity::finalize_booleanity_prover` when
                // a separate booleanity sumcheck group is run alongside
                // the CPR group; left empty here.
                bit_slice_evals: Vec::new(),
                bit_op_down_evals,
                // Filled by the protocol-level prover after CPR finalize.
                shifted_bit_slice_evals: Vec::new(),
            },
            CprProverState {
                evaluation_point: sumcheck_prover_state.randomness,
            },
        ))
    }

    /// Pre-sumcheck half of the CPR verifier.
    ///
    /// Must run before [`MultiDegreeSumcheck::verify_as_subprotocol`] to
    /// maintain transcript ordering (samples folding challenge α here).
    ///
    /// # Parameters
    /// - `transcript`: FS-transcript.
    /// - `proof`: The CPR proof (`up_evals`, `down_evals`).
    /// - `claimed_sum`: The claimed sum from
    ///   `combined_sumcheck.claimed_sums()[0]`.
    /// - `ic_check_subclaim`: Subclaim from the ideal check; provides the
    ///   evaluation point and claimed values used to verify the sumcheck sum.
    /// - `num_constraints`: Number of constraint polynomials in `U`.
    /// - `num_vars`: Number of variables of the trace MLEs.
    /// - `projecting_element`: The random challenge used to project `F[X] → F`.
    /// - `field_cfg`: Field configuration.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_verifier<U>(
        transcript: &mut impl Transcript,
        proof: &CprProof<F>,
        claimed_sum: F,
        ic_check_subclaim: &ideal_check::VerifierSubclaim<F>,
        num_constraints: usize,
        num_bit_slices: usize,
        num_shifted_bit_slices: usize,
        num_vars: usize,
        projecting_element: &F,
        field_cfg: &F::Config,
    ) -> Result<CprVerifierAncillary<F>, CombinedPolyResolverError<F>>
    where
        F::Inner: ConstTranscribable,
        F::Modulus: ConstTranscribable,
        U: Uair,
    {
        let uair_sig = U::signature();
        proof.validate_evaluation_sizes(
            uair_sig.total_cols().cols(),
            uair_sig.down_cols().cols(),
            num_bit_slices,
            uair_sig.bit_op_specs().len(),
            num_shifted_bit_slices,
        )?;

        let zero = F::zero_with_cfg(field_cfg);
        let one = F::one_with_cfg(field_cfg);

        // Precompute powers of the projecting element for batch evaluation.
        let projection_powers: Vec<F> = {
            let max_coeffs_len = ic_check_subclaim
                .values
                .iter()
                .map(|poly| poly.degree().map_or(0, |d| add!(d, 1)))
                .max()
                .unwrap_or(0)
                .max(1);
            powers(projecting_element.clone(), one.clone(), max_coeffs_len)
        };

        let folding_challenge: F = transcript.get_field_challenge(field_cfg);

        let folding_challenge_powers: Vec<F> =
            powers(folding_challenge, one.clone(), num_constraints);

        // TODO(Alex): investigate if parallelising this is beneficial.
        // Compute v_0 + \alpha * v_1 + ... + \alpha ^ k * v_k.
        let expected_sum = ic_check_subclaim
            .values
            .iter()
            .zip(&folding_challenge_powers)
            .map(|(claimed_value, random_coeff)| {
                let deg = claimed_value.degree().map_or(0, |d| add!(d, 1));
                DynamicPolyFInnerProduct::inner_product::<UNCHECKED>(
                    &claimed_value.coeffs[..deg],
                    &projection_powers[..deg],
                    zero.clone(),
                )
                .expect("inner product cannot fail here")
                    * random_coeff
            })
            .fold(zero.clone(), |acc, term| acc + term);

        if claimed_sum != expected_sum {
            return Err(CombinedPolyResolverError::WrongSumcheckSum {
                got: claimed_sum,
                expected: expected_sum,
            });
        }

        Ok(CprVerifierAncillary {
            folding_challenge_powers,
            ic_evaluation_point: ic_check_subclaim.evaluation_point.clone(),
            num_vars,
        })
    }

    /// Post-sumcheck half of the CPR verifier.
    ///
    /// Runs after [`MultiDegreeSumcheck::verify_as_subprotocol`] produces the
    /// shared evaluation point.
    ///
    /// # Parameters
    /// - `transcript`: FS-transcript (absorbs `up_evals` and `down_evals`).
    /// - `proof`: The CPR proof (consumed to produce the subclaim).
    /// - `shared_point`: The shared evaluation point `r*` from the multi-degree
    ///   sumcheck.
    /// - `expected_evaluation`: `md_subclaims.expected_evaluations()[0]` — the
    ///   expected value of the CPR combination function at `r*`.
    /// - `ancillary`: Produced by [`prepare_verifier`]; carries folding
    ///   challenge powers, ideal-check evaluation point, and `num_vars`.
    /// - `projected_scalars`: UAIR scalars projected to `F`.
    /// - `field_cfg`: Field configuration.
    #[allow(clippy::too_many_arguments)]
    pub fn finalize_verifier<U>(
        transcript: &mut impl Transcript,
        proof: CprProof<F>,
        shared_point: Vec<F>,
        expected_evaluation: F,
        ancillary: CprVerifierAncillary<F>,
        projected_scalars: &HashMap<U::Scalar, F>,
        field_cfg: &F::Config,
    ) -> Result<VerifierSubclaim<F>, CombinedPolyResolverError<F>>
    where
        F::Inner: ConstTranscribable,
        F::Modulus: ConstTranscribable,
        U: Uair,
    {
        let uair_sig = U::signature();
        let down_layout = uair_sig.down_cols().as_column_layout();
        let bit_op_count = uair_sig.bit_op_down_count();
        let zero = F::zero_with_cfg(field_cfg);
        let one = F::one_with_cfg(field_cfg);

        let eq_r_value = eq_eval(&shared_point, &ancillary.ic_evaluation_point, one.clone())?;
        let selector_value = eq_eval(
            &shared_point,
            &vec![one.clone(); ancillary.num_vars],
            one.clone(),
        )?;

        let mut folder = ConstraintFolder::new(&ancillary.folding_challenge_powers, &zero);

        let project = |scalar: &U::Scalar| {
            projected_scalars
                .get(scalar)
                .cloned()
                .expect("all scalars should have been projected at this point")
        };

        // Build the down trace row including the bit-op virtual column
        // half: `[down_evals... | bit_op_down_evals...]`.
        let mut down_combined: Vec<F> =
            Vec::with_capacity(proof.down_evals.len() + proof.bit_op_down_evals.len());
        down_combined.extend(proof.down_evals.iter().cloned());
        down_combined.extend(proof.bit_op_down_evals.iter().cloned());

        U::constrain_general(
            &mut folder,
            TraceRow::from_slice_with_layout(
                &proof.up_evals,
                uair_sig.total_cols().as_column_layout(),
            ),
            TraceRow::from_slice_with_layout_and_bit_op(
                &down_combined,
                down_layout,
                bit_op_count,
            ),
            project,
            |x, y| Some(project(y) * x),
            ImpossibleIdeal::from_ref,
        );

        let expected_claim_value = eq_r_value * (one - selector_value) * folder.folded_constraints;

        if expected_claim_value != expected_evaluation {
            return Err(CombinedPolyResolverError::ClaimValueDoesNotMatch {
                got: expected_evaluation,
                expected: expected_claim_value,
            });
        }

        // Mirror prover-side single-slice absorption order
        // `up || down || bit_op`.
        let mut transcription_buf: Vec<u8> = vec![0; F::Inner::NUM_BYTES];
        let mut evals_concat: Vec<F> = Vec::with_capacity(
            proof.up_evals.len() + proof.down_evals.len() + proof.bit_op_down_evals.len(),
        );
        evals_concat.extend(proof.up_evals.iter().cloned());
        evals_concat.extend(proof.down_evals.iter().cloned());
        evals_concat.extend(proof.bit_op_down_evals.iter().cloned());
        transcript.absorb_random_field_slice(&evals_concat, &mut transcription_buf);

        Ok(VerifierSubclaim {
            up_evals: proof.up_evals,
            down_evals: proof.down_evals,
            bit_slice_evals: proof.bit_slice_evals,
            bit_op_down_evals: proof.bit_op_down_evals,
            shifted_bit_slice_evals: proof.shifted_bit_slice_evals,
            evaluation_point: shared_point,
        })
    }
}

#[derive(Debug, Error)]
pub enum CombinedPolyResolverError<F: PrimeField> {
    #[error("failed to build eq_r: {0}")]
    EqrError(ArithErrors),
    #[error("error evaluating MLE: {0}")]
    MleEvaluationError(EvaluationError),
    #[error("error projecting polynomial {0} by point {1}: {2}")]
    ProjectionError(DynamicPolynomialF<F>, F, EvaluationError),
    #[error("wrong trace columns evaluations number: got {got}, expected {expected}")]
    WrongUpEvalsNumber { got: usize, expected: usize },
    #[error("wrong shifted trace columns evaluations number: got {got}, expected {expected}")]
    WrongDownEvalsNumber { got: usize, expected: usize },
    #[error("wrong bit-slice evaluations number: got {got}, expected {expected}")]
    WrongBitSliceEvalsNumber { got: usize, expected: usize },
    #[error("wrong bit-op down evaluations number: got {got}, expected {expected}")]
    WrongBitOpDownEvalsNumber { got: usize, expected: usize },
    #[error("wrong shifted bit-slice evaluations number: got {got}, expected {expected}")]
    WrongShiftedBitSliceEvalsNumber { got: usize, expected: usize },
    #[error("sumcheck verification failed: {0}")]
    SumcheckError(SumCheckError<F>),
    #[error("wrong sumcheck claimed sum: received {got}, expected {expected}")]
    WrongSumcheckSum { got: F, expected: F },
    #[error("resulting claim value does not match: received {got}, expected {expected}")]
    ClaimValueDoesNotMatch { got: F, expected: F },
}

impl<F: PrimeField> From<EvaluationError> for CombinedPolyResolverError<F> {
    fn from(eval_error: EvaluationError) -> Self {
        Self::MleEvaluationError(eval_error)
    }
}

impl<F: PrimeField> From<ArithErrors> for CombinedPolyResolverError<F> {
    fn from(arith_error: ArithErrors) -> Self {
        Self::EqrError(arith_error)
    }
}

impl<F: PrimeField> From<SumCheckError<F>> for CombinedPolyResolverError<F> {
    fn from(sumcheck_error: SumCheckError<F>) -> Self {
        Self::SumcheckError(sumcheck_error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ideal_check::IdealCheckProtocol,
        projections::{ProjectedTrace, evaluate_trace_to_column_mles, project_scalars_to_field},
        sumcheck::multi_degree::MultiDegreeSumcheck,
        test_utils::{LIMBS, run_ideal_check_prover_combined, test_config},
    };
    use crypto_primitives::{crypto_bigint_int::Int, crypto_bigint_monty::MontyField};
    use rand::rng;
    use zinc_poly::univariate::dense::DensePolynomial;
    use zinc_test_uair::{
        GenerateRandomTrace, TestUairNoMultiplication, TestUairSimpleMultiplication,
    };
    use zinc_transcript::Blake3Transcript;
    use zinc_uair::{
        constraint_counter::count_constraints,
        degree_counter::count_max_degree,
        ideal::{DegreeOneIdeal, Ideal, IdealCheck},
        ideal_collector::IdealOrZero,
    };

    // TODO(Ilia): These tests are absolute joke.
    //             Once we have time we need to create a comprehensive test suite
    //             akin to the one we have for the PCS or the sumcheck.

    fn test_successful_verification_generic<
        U,
        IdealOverF,
        IdealOverFFromRef,
        const DEGREE_PLUS_ONE: usize,
    >(
        num_vars: usize,
        ideal_over_f_from_ref: IdealOverFFromRef,
    ) where
        U: Uair<Scalar = DensePolynomial<Int<5>, DEGREE_PLUS_ONE>>
            + GenerateRandomTrace<DEGREE_PLUS_ONE, PolyCoeff = Int<5>, Int = Int<5>>
            + IdealCheckProtocol,
        IdealOverF: Ideal + IdealCheck<DynamicPolynomialF<MontyField<LIMBS>>>,
        IdealOverFFromRef: Fn(&IdealOrZero<U::Ideal>) -> IdealOverF,
    {
        let mut rng = rng();

        let mut prover_transcript = Blake3Transcript::new();
        let mut verifier_transcript = prover_transcript.clone();

        let trace = U::generate_random_trace(num_vars, &mut rng);

        let (ic_proof, ic_prover_state, projected_scalars, projected_trace) =
            run_ideal_check_prover_combined::<U, DEGREE_PLUS_ONE>(
                num_vars,
                &trace,
                &mut prover_transcript,
            );

        let num_constraints = count_constraints::<U>();

        let ic_check_subclaim = U::verify_as_subprotocol(
            &mut verifier_transcript,
            ic_proof,
            num_constraints,
            num_vars,
            ideal_over_f_from_ref,
            &test_config(),
        )
        .expect("Verification failed");

        let max_degree = count_max_degree::<U>();

        let projecting_element: MontyField<4> =
            prover_transcript.get_field_challenge(&test_config());

        let projected_scalars =
            project_scalars_to_field(projected_scalars, &projecting_element).unwrap();

        // Prover: prepare → MultiDegreeSumcheck → finalize
        let (cpr_group, cpr_ancillary) = CombinedPolyResolver::prepare_sumcheck_group::<U, DEGREE_PLUS_ONE>(
            &mut prover_transcript,
            evaluate_trace_to_column_mles(
                &ProjectedTrace::RowMajor(projected_trace),
                &projecting_element,
            ),
            &ic_prover_state.evaluation_point,
            &projected_scalars,
            num_constraints,
            num_vars,
            max_degree,
            &test_config(),
            &trace.binary_poly,
            &projecting_element,
        )
        .expect("CPR prepare failed");

        let (md_proof, states) = MultiDegreeSumcheck::prove_as_subprotocol(
            &mut prover_transcript,
            vec![cpr_group],
            num_vars,
            &test_config(),
        );

        let (proof, _) = CombinedPolyResolver::finalize_prover(
            &mut prover_transcript,
            states.into_iter().next().unwrap(),
            cpr_ancillary,
            &test_config(),
        )
        .expect("CPR finalize failed");

        let projecting_element: MontyField<LIMBS> =
            verifier_transcript.get_field_challenge(&test_config());

        // Verifier: prepare → MultiDegreeSumcheck → finalize
        let cpr_verifier_ancillary = CombinedPolyResolver::prepare_verifier::<U>(
            &mut verifier_transcript,
            &proof,
            md_proof.claimed_sums()[0].clone(),
            &ic_check_subclaim,
            num_constraints,
            0, // num_bit_slices: this test has no binary_poly columns
            0, // num_shifted_bit_slices
            num_vars,
            &projecting_element,
            &test_config(),
        )
        .expect("CPR prepare_verifier failed");

        let md_subclaims = MultiDegreeSumcheck::verify_as_subprotocol(
            &mut verifier_transcript,
            num_vars,
            &md_proof,
            &test_config(),
        )
        .expect("MultiDegreeSumcheck verify failed");

        assert!(
            CombinedPolyResolver::finalize_verifier::<U>(
                &mut verifier_transcript,
                proof,
                md_subclaims.point().to_vec(),
                md_subclaims.expected_evaluations()[0].clone(),
                cpr_verifier_ancillary,
                &projected_scalars,
                &test_config(),
            )
            .is_ok()
        );
    }

    #[test]
    fn test_successful_verification() {
        let field_cfg = test_config();

        let num_vars = 2;

        test_successful_verification_generic::<TestUairNoMultiplication<Int<5>>, _, _, 32>(
            num_vars,
            |ideal_over_ring| ideal_over_ring.map(|i| DegreeOneIdeal::from_with_cfg(i, &field_cfg)),
        );
        test_successful_verification_generic::<TestUairSimpleMultiplication<Int<5>>, _, _, 32>(
            num_vars,
            |_ideal_over_ring| IdealOrZero::<DegreeOneIdeal<_>>::zero(),
        );
    }
}
