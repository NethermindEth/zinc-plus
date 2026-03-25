//! Multi-point evaluation subprotocol.
//!
//! Reduces two sets of MLE evaluation claims at a shared point r' - the
//! "up" evaluations `v_j(r')` and the "down" (shifted) evaluations
//! `v_j^{down}(r')` - to a single set of standard MLE evaluation claims
//! `v_j(r_0)` at a new random point `r_0` via one sumcheck.
//!
//! The trace column MLEs are precombined into a single MLE
//! `precombined(b) = \sum_j \gamma_j * v_j(b)` before entering the sumcheck, so
//! the prover works with only 3 MLEs (`eq`, `next`, `precombined`) regardless
//! of the number of columns. The sumcheck proves:
//! ```text
//! \sum_b [eq(b, r') * \sum_j \gamma_j * v_j(b)
//!         + \sum_k \alpha_k * next_{c_k}(r', b) * v_{src_k}(b)]
//!   = \sum_j \gamma_j * up_eval_j + \sum_k \alpha_k * down_eval_k
//! ```
//!
//! where `\alpha_k` batch the per-shift evaluation kernels and `\gamma_j`
//! batch across columns. After the sumcheck reduces to point `r_0`, the
//! verifier calls [`MultipointEval::verify_subclaim`] with the `open_evals`
//! (the F_q-valued MLE evaluations at `r_0`, typically derived from
//! polynomial-valued `lifted_evals` via `\psi_a`) to check the final
//! consistency equation.
//!
//! This corresponds to the T=2 case of Pi_{BMLE} in the paper. Following
//! the paper, the prover sends only the polynomial-valued lifted evaluations
//! (alpha'_j in F_q[X]); the scalar open_evals are derived by the verifier
//! via \psi_a rather than being sent as a separate proof element.

use crate::{
    shift_predicate::eval_shift_predicate,
    sumcheck::{MLSumcheck, SumCheckError, SumcheckProof, verifier::Subclaim as SumcheckSubclaim},
};
use crypto_primitives::{FromPrimitiveWithConfig, PrimeField};
use num_traits::Zero;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::marker::PhantomData;
use thiserror::Error;
use zinc_poly::{
    mle::DenseMultilinearExtension,
    utils::{ArithErrors, build_eq_x_r_inner, build_next_c_r_mle},
};
use zinc_transcript::traits::{ConstTranscribable, Transcript};
use zinc_uair::ShiftSpec;
use zinc_utils::{cfg_into_iter, inner_transparent_field::InnerTransparentField};

//
// Data structures
//

/// Proof for the multi-point evaluation protocol.
///
/// Contains only the sumcheck proof. The MLE evaluations at `r_0` are
/// provided externally via `lifted_evals` (in `F_q[X]`), from which the
/// verifier derives the scalar `open_evals` via `\psi_a`.
#[derive(Clone, Debug)]
pub struct Proof<F: PrimeField> {
    /// The inner sumcheck proof.
    pub sumcheck_proof: SumcheckProof<F>,
}

/// Prover state after the multi-point evaluation protocol.
pub struct ProverState<F: PrimeField> {
    /// The combined evaluation point `r_0` produced by the sumcheck.
    pub eval_point: Vec<F>,
}

/// Verifier subclaim after the multi-point evaluation sumcheck.
///
/// Carries the inner sumcheck [`SumcheckSubclaim`] plus the intermediate values
/// needed to finalize the check via [`MultipointEval::verify_subclaim`]
/// once the caller has assembled the `open_evals` (e.g. after computing
/// public lifted evaluations from public data at `r_0`).
#[derive(Clone, Debug)]
pub struct Subclaim<F: PrimeField> {
    /// Inner sumcheck subclaim. Its `point` field is `r_0`; its
    /// `expected_evaluation` is the value that `verify_subclaim` checks
    /// against the batched open_evals.
    pub sumcheck_subclaim: SumcheckSubclaim<F>,
    /// Column batching coefficients \gamma_j sampled during the protocol.
    pub gammas: Vec<F>,
    /// Per-shift batching coefficients \alpha_k sampled during the protocol.
    pub alphas: Vec<F>,
    /// `eq(r_0, r')` — the equality selector at the sumcheck output point.
    pub eq_at_r0: F,
    /// Per-shift selector values at r_0:
    /// `shifts_at_r0[k] = next_{c_k}(r', r_0)`.
    pub shifts_at_r0: Vec<F>,
}

//
// Protocol
//

pub struct MultipointEval<F>(PhantomData<F>);

impl<F> MultipointEval<F>
where
    F: InnerTransparentField + FromPrimitiveWithConfig + Send + Sync + 'static,
    F::Inner: ConstTranscribable + Zero + Default + Send + Sync,
    F::Modulus: ConstTranscribable,
{
    /// Multi-point evaluation protocol prover.
    ///
    /// Runs the combined sumcheck over
    /// `eq(b, r') * \sum_j(\gamma_j * v_j(b)) + \sum_k \alpha_k *
    /// next_{c_k}(r', b) * v_{src_k}(b)`. Returns only the sumcheck proof
    /// and the challenge point `r_0`; the caller is responsible for
    /// computing and sending `lifted_evals` at `r_0`.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn prove_as_subprotocol(
        transcript: &mut impl Transcript,
        trace_mles: &[DenseMultilinearExtension<F::Inner>],
        eval_point: &[F],
        up_evals: &[F],
        down_evals: &[F],
        shifts: &[ShiftSpec],
        field_cfg: &F::Config,
    ) -> Result<(Proof<F>, ProverState<F>), MultipointEvalError<F>> {
        let num_cols = trace_mles.len();
        let num_down_cols = shifts.len();
        let num_vars = eval_point.len();
        let zero = F::zero_with_cfg(field_cfg);
        let zero_inner = zero.inner();

        // Step 1: Sample multi-point batching coefficient \alpha and column
        // batching coefficients \gamma_1,...,\gamma_J.
        let alphas: Vec<F> = transcript.get_field_challenges(num_down_cols, field_cfg);
        let gammas: Vec<F> = transcript.get_field_challenges(num_cols, field_cfg);

        // Step 2: Build the two selector MLEs:
        //   eq_r(b)   = eq(b, r')
        //   next_r(b) = next_tilde(r', b)
        let eq_r = build_eq_x_r_inner(eval_point, field_cfg)?;
        let (next_mles, down_cols): (Vec<_>, Vec<_>) = shifts
            .iter()
            .map(|spec| {
                let next = build_next_c_r_mle(eval_point, spec.shift_amount(), field_cfg)?;
                let col = trace_mles[spec.source_col()].clone();
                Ok((next, col))
            })
            .collect::<Result<Vec<_>, ArithErrors>>()?
            .into_iter()
            .unzip();

        // Precombine up cols with gammas, precombined[b] = Σ_j γ_j trace[j][b]
        let precombined = {
            let evaluations: Vec<_> = cfg_into_iter!(0..1 << num_vars)
                .map(|b| {
                    gammas
                        .iter()
                        .enumerate()
                        .fold(zero.clone(), |acc, (i, gamma)| {
                            let eval_f = F::new_unchecked_with_cfg(
                                trace_mles[i].evaluations[b].clone(),
                                field_cfg,
                            );
                            acc + gamma.clone() * eval_f
                        })
                        .into_inner()
                })
                .collect();
            DenseMultilinearExtension::from_evaluations_vec(
                num_vars,
                evaluations,
                zero_inner.clone(),
            )
        };

        // Step 3: Pack MLEs: [eq_r, next_mles[..], precombined, down_cols[..]]
        let mut mles = Vec::with_capacity(2 + 2 * num_down_cols);
        mles.push(eq_r);
        mles.extend(next_mles);
        mles.push(precombined);
        mles.extend(down_cols);

        // Step 4: Run sumcheck with degree=2.

        // comb_fn([eq_r, next_mles[..], precombined, down_cols[..]]) =
        //     eq_r * precombined + \alphas[i] * next_mle[i] * down_cols[i]
        let (sumcheck_proof, sumcheck_prover_state) = MLSumcheck::prove_as_subprotocol(
            transcript,
            mles,
            num_vars,
            2,
            |mle_values: &[F]| {
                let eq_val = &mle_values[0];
                let precombined = &mle_values[num_down_cols + 1];
                alphas
                    .iter()
                    .enumerate()
                    .fold(eq_val.clone() * precombined, |acc, (i, alpha)| {
                        let next = &mle_values[1 + i];
                        let down_col = &mle_values[num_down_cols + 2 + i];
                        acc + alpha.clone() * next * down_col
                    })
            },
            field_cfg,
        );

        // Sanity check
        debug_assert_eq!(
            sumcheck_proof.claimed_sum,
            compute_expected_sum(up_evals, down_evals, &gammas, &alphas, zero)
        );

        Ok((
            Proof { sumcheck_proof },
            ProverState {
                eval_point: sumcheck_prover_state.randomness,
            },
        ))
    }

    /// Multi-point evaluation protocol verifier (sumcheck phase).
    ///
    /// Runs the sumcheck verification and computes the intermediate values
    /// needed for the open-eval consistency check. Returns a [`Subclaim`]
    /// carrying `r_0`, `gammas`, `alphas`, `eq_at_r0`, and
    /// `shifts_at_r0`. The caller finalizes via
    /// [`verify_subclaim`](Self::verify_subclaim) once `open_evals` are
    /// available.
    #[allow(clippy::arithmetic_side_effects, clippy::too_many_arguments)]
    pub fn verify_as_subprotocol(
        transcript: &mut impl Transcript,
        proof: Proof<F>,
        eval_point: &[F],
        up_evals: &[F],
        down_evals: &[F],
        shifts: &[ShiftSpec],
        num_vars: usize,
        field_cfg: &F::Config,
    ) -> Result<Subclaim<F>, MultipointEvalError<F>> {
        let num_cols = up_evals.len();
        let num_down_cols = shifts.len();
        let zero = F::zero_with_cfg(field_cfg);
        let one = F::one_with_cfg(field_cfg);

        // Step 1: Sample \alpha_k and \gamma_j (must match prover).
        let alphas: Vec<F> = transcript.get_field_challenges(num_down_cols, field_cfg);
        let gammas: Vec<F> = transcript.get_field_challenges(num_cols, field_cfg);

        // Step 2: Compute expected sum
        let expected_sum: F =
            compute_expected_sum(up_evals, down_evals, &gammas, &alphas, zero.clone());

        if proof.sumcheck_proof.claimed_sum != expected_sum {
            return Err(MultipointEvalError::WrongSumcheckSum {
                got: proof.sumcheck_proof.claimed_sum.clone(),
                expected: expected_sum,
            });
        }

        // Step 3: Verify the sumcheck.
        let sumcheck_subclaim = MLSumcheck::verify_as_subprotocol(
            transcript,
            num_vars,
            2,
            &proof.sumcheck_proof,
            field_cfg,
        )?;

        let r_0 = &sumcheck_subclaim.point;

        // Step 4: Recompute the selectors at r_0.
        let eq_at_r0 = zinc_poly::utils::eq_eval(r_0, eval_point, one)?;
        let shifts_at_r0: Vec<F> = shifts
            .iter()
            .map(|spec| eval_shift_predicate(eval_point, r_0, spec.shift_amount(), field_cfg))
            .collect();

        Ok(Subclaim {
            sumcheck_subclaim,
            gammas,
            alphas,
            eq_at_r0,
            shifts_at_r0,
        })
    }

    /// Finalize the multi-point evaluation check given `open_evals`.
    ///
    /// Verifies that
    /// `eq_at_r0 * \sum_j(gamma_j * open_eval_j) + \sum_k(alpha_k *
    /// shift_at_r0_k * open_eval[source_col_k])` equals the sumcheck's
    /// expected evaluation. This is a pure arithmetic check with no
    /// transcript interaction.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn verify_subclaim(
        subclaim: &Subclaim<F>,
        open_evals: &[F],
        shifts: &[ShiftSpec],
        field_cfg: &F::Config,
    ) -> Result<(), MultipointEvalError<F>> {
        let num_cols = subclaim.gammas.len();

        if open_evals.len() != num_cols {
            return Err(MultipointEvalError::WrongOpenEvalsNumber {
                got: open_evals.len(),
                expected: num_cols,
            });
        }

        let zero = F::zero_with_cfg(field_cfg);

        let batched_up: F = subclaim
            .gammas
            .iter()
            .zip(open_evals.iter())
            .fold(zero.clone(), |acc, (gamma, eval)| {
                acc + gamma.clone() * eval
            });

        // open_evals[j] = trace_col_j(r_0) for all committed (up) columns.
        // Shifted columns reuse the same opening: the shift is captured by
        // the shift_at_r0 selector, so we index by source_col into open_evals.
        let batched_down: F = subclaim
            .alphas
            .iter()
            .enumerate()
            .zip(subclaim.shifts_at_r0.iter())
            .fold(zero, |acc, ((k, alpha), shift_at_r0)| {
                let src_col = shifts[k].source_col();
                acc + alpha.clone() * shift_at_r0 * &open_evals[src_col]
            });

        let expected_evaluation = subclaim.eq_at_r0.clone() * &batched_up + batched_down;

        if expected_evaluation != subclaim.sumcheck_subclaim.expected_evaluation {
            return Err(MultipointEvalError::ClaimMismatch {
                got: subclaim.sumcheck_subclaim.expected_evaluation.clone(),
                expected: expected_evaluation,
            });
        }

        Ok(())
    }
}

/// `expected_sum = \sum_j \gamma_j * up_eval_j + \sum_k \alpha_k *
/// down_eval_k`
fn compute_expected_sum<F: PrimeField>(
    up_evals: &[F],
    down_evals: &[F],
    gammas: &[F],
    alphas: &[F],
    zero: F,
) -> F {
    let up_sum = gammas
        .iter()
        .zip(up_evals.iter())
        .fold(zero, |acc, (gamma, up)| acc + gamma.clone() * up);

    alphas
        .iter()
        .zip(down_evals.iter())
        .fold(up_sum, |acc, (alpha, down)| acc + alpha.clone() * down)
}

//
// Error type
//

#[derive(Debug, Error)]
pub enum MultipointEvalError<F: PrimeField> {
    #[error("wrong number of open evaluations: got {got}, expected {expected}")]
    WrongOpenEvalsNumber { got: usize, expected: usize },
    #[error("wrong sumcheck claimed sum: got {got}, expected {expected}")]
    WrongSumcheckSum { got: F, expected: F },
    #[error("multi-point eval claim mismatch: got {got}, expected {expected}")]
    ClaimMismatch { got: F, expected: F },
    #[error("sumcheck error: {0}")]
    SumcheckError(#[from] SumCheckError<F>),
    #[error("arithmetic error: {0}")]
    ArithError(#[from] ArithErrors),
}

#[cfg(test)]
#[allow(
    clippy::arithmetic_side_effects,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]
mod tests {
    use super::*;
    use crypto_bigint::{U128, const_monty_params};
    use crypto_primitives::crypto_bigint_const_monty::ConstMontyField;
    use num_traits::{ConstOne, ConstZero};
    use zinc_poly::mle::{DenseMultilinearExtension, MultilinearExtensionWithConfig};
    use zinc_transcript::KeccakTranscript;

    const N: usize = 2;
    const_monty_params!(Params, U128, "00000000b933426489189cb5b47d567f");
    type F = ConstMontyField<Params, N>;

    /// Data known to both prover and verifier from earlier protocol steps.
    #[derive(Clone)]
    struct SharedSubprotocolInput {
        eval_point: Vec<F>,
        up_evals: Vec<F>,
        down_evals: Vec<F>,
        shifts: Vec<ShiftSpec>,
        num_vars: usize,
    }

    /// What the prover sends to the verifier.
    #[derive(Clone)]
    struct ProverMessage {
        proof: Proof<F>,
        open_evals: Vec<F>,
    }

    fn make_transcript() -> KeccakTranscript {
        let mut t = KeccakTranscript::default();
        t.absorb_slice(b"Lorem ipsum");
        t
    }

    fn build_trace(
        num_vars: usize,
        num_cols: usize,
        shifts: &[ShiftSpec],
    ) -> (
        Vec<DenseMultilinearExtension<<F as crypto_primitives::Field>::Inner>>,
        SharedSubprotocolInput,
    ) {
        let n = 1usize << num_vars;
        let zero_inner = F::ZERO.into_inner();

        let trace_mles: Vec<DenseMultilinearExtension<_>> = (0..num_cols)
            .map(|col| {
                let evals: Vec<_> = (0..n)
                    .map(|i| F::from((col * n + i + 1) as u32).into_inner())
                    .collect();
                DenseMultilinearExtension::from_evaluations_vec(num_vars, evals, zero_inner)
            })
            .collect();

        let eval_point: Vec<F> = (0..num_vars).map(|i| F::from((i + 7) as u32)).collect();

        let up_evals: Vec<F> = trace_mles
            .iter()
            .map(|mle| mle.clone().evaluate_with_config(&eval_point, &()).unwrap())
            .collect();

        let down_evals: Vec<F> = shifts
            .iter()
            .map(|spec| {
                let mle = &trace_mles[spec.source_col()];
                let c = spec.shift_amount();
                let mut shifted = mle.evaluations[c..].to_vec();
                shifted.extend(vec![zero_inner; c]);
                let shifted_mle =
                    DenseMultilinearExtension::from_evaluations_vec(num_vars, shifted, zero_inner);
                shifted_mle.evaluate_with_config(&eval_point, &()).unwrap()
            })
            .collect();

        let public = SharedSubprotocolInput {
            eval_point,
            up_evals,
            down_evals,
            shifts: shifts.to_vec(),
            num_vars,
        };
        (trace_mles, public)
    }

    /// Prover: has access to the trace, produces a proof and open_evals.
    fn run_prover(
        trace_mles: &[DenseMultilinearExtension<<F as crypto_primitives::Field>::Inner>],
        public: &SharedSubprotocolInput,
    ) -> ProverMessage {
        let mut transcript = make_transcript();
        let (proof, prover_state) = MultipointEval::<F>::prove_as_subprotocol(
            &mut transcript,
            trace_mles,
            &public.eval_point,
            &public.up_evals,
            &public.down_evals,
            &public.shifts,
            &(),
        )
        .expect("prover should succeed");

        let r_0 = &prover_state.eval_point;
        let open_evals: Vec<F> = trace_mles
            .iter()
            .map(|mle| mle.clone().evaluate_with_config(r_0, &()).unwrap())
            .collect();

        ProverMessage { proof, open_evals }
    }

    /// Verifier: only receives the proof + open_evals + public data.
    fn run_verifier(
        public: &SharedSubprotocolInput,
        msg: &ProverMessage,
    ) -> Result<Subclaim<F>, MultipointEvalError<F>> {
        let subclaim = MultipointEval::<F>::verify_as_subprotocol(
            &mut make_transcript(),
            msg.proof.clone(),
            &public.eval_point,
            &public.up_evals,
            &public.down_evals,
            &public.shifts,
            public.num_vars,
            &(),
        )?;

        MultipointEval::<F>::verify_subclaim(&subclaim, &msg.open_evals, &public.shifts, &())?;

        Ok(subclaim)
    }

    /// Convenience: build trace, prove, return (public, message).
    fn honest_interaction(
        num_vars: usize,
        num_cols: usize,
        shifts: &[ShiftSpec],
    ) -> (SharedSubprotocolInput, ProverMessage) {
        let (trace, public) = build_trace(num_vars, num_cols, shifts);
        let msg = run_prover(&trace, &public);
        (public, msg)
    }

    /// Helper: all-columns shift-by-1
    fn all_shift_by_1(num_cols: usize) -> Vec<ShiftSpec> {
        (0..num_cols).map(|i| ShiftSpec::new(i, 1)).collect()
    }

    // --- Happy-path ---

    #[test]
    fn honest_prove_verify_single_column() {
        let shifts = all_shift_by_1(1);
        let (public, msg) = honest_interaction(4, 1, &shifts);
        run_verifier(&public, &msg).unwrap();
    }

    #[test]
    fn honest_prove_verify_many_columns() {
        let shifts = all_shift_by_1(10);
        let (public, msg) = honest_interaction(3, 10, &shifts);
        run_verifier(&public, &msg).unwrap();
    }

    #[test]
    fn honest_prove_verify_no_shifts() {
        let (public, msg) = honest_interaction(3, 3, &[]);
        run_verifier(&public, &msg).unwrap();
    }

    #[test]
    fn honest_prove_verify_mixed_shifts() {
        let shifts = vec![ShiftSpec::new(0, 1), ShiftSpec::new(1, 3)];
        let (public, msg) = honest_interaction(4, 3, &shifts);
        run_verifier(&public, &msg).unwrap();
    }

    #[test]
    fn honest_prove_verify_shift_by_3() {
        let shifts = vec![
            ShiftSpec::new(0, 3),
            ShiftSpec::new(1, 3),
            ShiftSpec::new(2, 3),
        ];
        let (public, msg) = honest_interaction(4, 3, &shifts);
        run_verifier(&public, &msg).unwrap();
    }

    #[test]
    fn honest_prove_verify_same_col_different_shifts() {
        // Column 0 shifted by 2 and by 5
        let shifts = vec![ShiftSpec::new(0, 2), ShiftSpec::new(0, 5)];
        let (public, msg) = honest_interaction(4, 3, &shifts);
        run_verifier(&public, &msg).unwrap();
    }

    // --- Failure: corrupted down_evals with mixed shifts ---

    #[test]
    fn bad_down_eval_rejected_mixed_shifts() {
        let shifts = vec![ShiftSpec::new(0, 1), ShiftSpec::new(1, 3)];
        let (mut public, msg) = honest_interaction(4, 3, &shifts);
        public.down_evals[0] += F::ONE;
        let err = run_verifier(&public, &msg).unwrap_err();
        assert!(
            matches!(err, MultipointEvalError::WrongSumcheckSum { .. }),
            "expected WrongSumcheckSum, got {err:?}",
        );
    }

    // --- Failure: wrong number of open_evals ---

    #[test]
    fn wrong_open_evals_count() {
        let shifts = all_shift_by_1(3);
        let (public, msg) = honest_interaction(3, 3, &shifts);

        let mut msg_short = msg.clone();
        msg_short.open_evals.pop();

        let mut msg_long = msg;
        msg_long.open_evals.push(F::from(42_u32));

        for bad_msg in [&msg_short, &msg_long] {
            let err = run_verifier(&public, bad_msg).unwrap_err();
            assert!(
                matches!(err, MultipointEvalError::WrongOpenEvalsNumber {
                    got,
                    expected: 3,
                } if got == bad_msg.open_evals.len()),
                "expected WrongOpenEvalsNumber, got {err:?}",
            );
        }
    }

    // --- Failure: wrong claimed sum ---

    #[test]
    fn wrong_claimed_sum_via_corrupted_up_evals() {
        let shifts = all_shift_by_1(3);
        let (mut public, msg) = honest_interaction(3, 3, &shifts);
        public.up_evals[0] += F::ONE;
        let err = run_verifier(&public, &msg).unwrap_err();
        assert!(
            matches!(err, MultipointEvalError::WrongSumcheckSum { .. }),
            "expected WrongSumcheckSum, got {err:?}",
        );
    }

    #[test]
    fn wrong_claimed_sum_via_corrupted_down_evals() {
        let shifts = all_shift_by_1(3);
        let (mut public, msg) = honest_interaction(3, 3, &shifts);
        public.down_evals[1] += F::ONE;
        let err = run_verifier(&public, &msg).unwrap_err();
        assert!(
            matches!(err, MultipointEvalError::WrongSumcheckSum { .. }),
            "expected WrongSumcheckSum, got {err:?}",
        );
    }

    // --- Failure: wrong open_evals values ---

    #[test]
    fn wrong_open_eval_value() {
        let shifts = all_shift_by_1(3);
        let (public, mut msg) = honest_interaction(3, 3, &shifts);
        msg.open_evals[0] += F::ONE;
        let err = run_verifier(&public, &msg).unwrap_err();
        assert!(
            matches!(err, MultipointEvalError::ClaimMismatch { .. }),
            "expected ClaimMismatch, got {err:?}",
        );
    }

    #[test]
    fn all_open_evals_zeroed() {
        let shifts = all_shift_by_1(3);
        let (public, mut msg) = honest_interaction(3, 3, &shifts);
        for e in &mut msg.open_evals {
            *e = F::ZERO;
        }
        let err = run_verifier(&public, &msg).unwrap_err();
        assert!(
            matches!(err, MultipointEvalError::ClaimMismatch { .. }),
            "expected ClaimMismatch, got {err:?}",
        );
    }

    // --- Failure: mixed shifts ---

    fn mixed_shifts() -> Vec<ShiftSpec> {
        vec![ShiftSpec::new(0, 1), ShiftSpec::new(1, 3)]
    }

    #[test]
    fn mixed_shifts_corrupted_up_eval() {
        let (mut public, msg) = honest_interaction(4, 3, &mixed_shifts());
        public.up_evals[2] += F::ONE; // corrupt unshifted column
        let err = run_verifier(&public, &msg).unwrap_err();
        assert!(
            matches!(err, MultipointEvalError::WrongSumcheckSum { .. }),
            "expected WrongSumcheckSum, got {err:?}",
        );
    }

    #[test]
    fn mixed_shifts_wrong_open_eval() {
        let (public, mut msg) = honest_interaction(4, 3, &mixed_shifts());
        msg.open_evals[1] += F::ONE; // corrupt a shifted column's opening
        let err = run_verifier(&public, &msg).unwrap_err();
        assert!(
            matches!(err, MultipointEvalError::ClaimMismatch { .. }),
            "expected ClaimMismatch, got {err:?}",
        );
    }

    #[test]
    fn mixed_shifts_tampered_sumcheck() {
        let (public, mut msg) = honest_interaction(4, 3, &mixed_shifts());
        msg.proof.sumcheck_proof.messages[0].0.tail_evaluations[0] += F::ONE;
        let err = run_verifier(&public, &msg).unwrap_err();
        assert!(
            matches!(
                err,
                MultipointEvalError::SumcheckError(_) | MultipointEvalError::ClaimMismatch { .. }
            ),
            "expected sumcheck or consistency error, got {err:?}",
        );
    }

    // --- Failure: tampered sumcheck round messages ---

    #[test]
    fn tampered_sumcheck_round_message() {
        let shifts = all_shift_by_1(3);
        let (public, mut msg) = honest_interaction(3, 3, &shifts);
        msg.proof.sumcheck_proof.messages[0].0.tail_evaluations[0] += F::ONE;
        let err = run_verifier(&public, &msg).unwrap_err();
        assert!(
            matches!(
                err,
                MultipointEvalError::SumcheckError(_) | MultipointEvalError::ClaimMismatch { .. }
            ),
            "expected sumcheck or consistency error, got {err:?}",
        );
    }
}
