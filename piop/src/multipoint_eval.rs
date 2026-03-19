//! Multi-point evaluation subprotocol.
//!
//! Reduces two sets of MLE evaluation claims at a shared point r' - the
//! "up" evaluations `v_j(r')` and the "down" (shifted) evaluations
//! `v_j^{down}(r')` - to a single set of standard MLE evaluation claims
//! `v_j(r_0)` at a new random point `r_0` via one sumcheck.
//!
//! The sumcheck proves:
//! ```text
//! \sum_b [eq(b, r') + \alpha * next(r', b)] * [\sum_j \gamma_j * v_j(b)]
//!   = \sum_j \gamma_j * (up_eval_j + \alpha * down_eval_j)
//! ```
//!
//! where `\alpha` batches the two evaluation kernels and `\gamma_j` batch
//! across columns. After the sumcheck reduces to point `r_0`, the caller
//! provides `open_evals` (the F_q-valued MLE evaluations at `r_0`, typically
//! derived from polynomial-valued `lifted_evals` via `\psi_a`) and the
//! verifier checks the sumcheck consistency.
//!
//! This corresponds to the T=2 case of Pi_{BMLE} in the paper. Following
//! the paper, the prover sends only the polynomial-valued lifted evaluations
//! (alpha'_j in F_q[X]); the scalar open_evals are derived by the verifier
//! via \psi_a rather than being sent as a separate proof element.

use std::marker::PhantomData;

use crate::sumcheck::{MLSumcheck, SumCheckError, SumcheckProof};
use crypto_primitives::{FromPrimitiveWithConfig, PrimeField};
use itertools::Itertools;
use num_traits::Zero;
use thiserror::Error;
use zinc_poly::{mle::DenseMultilinearExtension, utils::ArithErrors};
use zinc_transcript::traits::{ConstTranscribable, Transcript};
use zinc_utils::inner_transparent_field::InnerTransparentField;

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

/// Verifier subclaim after the multi-point evaluation protocol.
#[derive(Clone, Debug)]
pub struct Subclaim<F: PrimeField> {
    /// The combined evaluation point r_0.
    pub eval_point: Vec<F>,
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
    /// `[eq(b, r') + \alpha * next(r', b)] * \sum_j(\gamma_j * v_j(b))`.
    /// Returns only the sumcheck proof and
    /// the challenge point `r_0`; the caller is responsible for computing
    /// and sending `lifted_evals` at `r_0`.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn prove_as_subprotocol(
        transcript: &mut impl Transcript,
        trace_mles: &[DenseMultilinearExtension<F::Inner>],
        eval_point: &[F],
        up_evals: &[F],
        down_evals: &[F],
        field_cfg: &F::Config,
    ) -> Result<(Proof<F>, ProverState<F>), MultipointEvalError<F>> {
        let num_cols = trace_mles.len();
        let num_vars = eval_point.len();
        let zero = F::zero_with_cfg(field_cfg);
        let zero_inner = zero.inner();

        // Step 1: Sample multi-point batching coefficient \alpha and column
        // batching coefficients \gamma_1,...,\gamma_J.
        let alpha: F = transcript.get_field_challenge(field_cfg);
        let gammas: Vec<F> = transcript.get_field_challenges(num_cols, field_cfg);

        // Step 2: Build the two selector MLEs:
        //   eq_r(b)   = eq(b, r')
        //   next_r(b) = next_tilde(r', b)
        let eq_r = zinc_poly::utils::build_eq_x_r_inner(eval_point, field_cfg)?;
        let next_r = zinc_poly::utils::build_next_r_mle(eval_point, field_cfg)?;

        // Precombine up cols with gammas, precombined[b] = Σ_j γ_j trace[j][b]
        let precombined = {
            let evaluations = (0..1 << num_vars).map(|b| gammas.iter().enumerate().fold(zero.clone(), |acc, (i, gamma)| {
                let eval_f = F::new_unchecked_with_cfg(trace_mles[i].evaluations[b].clone(), field_cfg);
                acc + gamma.clone() * eval_f
            }).into_inner()).collect_vec();
            DenseMultilinearExtension::from_evaluations_vec(num_vars, evaluations, zero_inner.clone())
        };
        
        // Step 3: Pack MLEs: [eq_r, next_r, precombined]
        let mles = vec![eq_r, next_r, precombined];

        // Step 4: Run sumcheck with degree=2.

        // comb_fn([eq_r, next_r, precombined]) =
        //     (eq_r + \alpha * next_r) * precombined
        let (sumcheck_proof, sumcheck_prover_state) = MLSumcheck::prove_as_subprotocol(
            transcript,
            mles,
            num_vars,
            2,
            |mle_values: &[F]| {
                let eq_val = &mle_values[0];
                let next_val = &mle_values[1];
                let selector = eq_val.clone() + alpha.clone() * next_val;
                selector * &mle_values[2]
            },
            field_cfg,
        );

        // Sanity check
        debug_assert_eq!(
            sumcheck_proof.claimed_sum,
            compute_expected_sum(up_evals, down_evals, &gammas, &alpha, zero)
        );

        Ok((
            Proof { sumcheck_proof },
            ProverState {
                eval_point: sumcheck_prover_state.randomness,
            },
        ))
    }

    /// Multi-point evaluation protocol verifier.
    ///
    /// Verifies the sumcheck and checks the final-round consistency using
    /// the caller-provided `open_evals` (typically derived from
    /// `lifted_evals` via `\psi_a`). Returns the evaluation point `r_0`.
    #[allow(clippy::arithmetic_side_effects, clippy::too_many_arguments)]
    pub fn verify_as_subprotocol(
        transcript: &mut impl Transcript,
        proof: Proof<F>,
        eval_point: &[F],
        up_evals: &[F],
        down_evals: &[F],
        open_evals: &[F],
        num_vars: usize,
        field_cfg: &F::Config,
    ) -> Result<Subclaim<F>, MultipointEvalError<F>> {
        let num_cols = up_evals.len();
        let zero = F::zero_with_cfg(field_cfg);
        let one = F::one_with_cfg(field_cfg);

        if open_evals.len() != num_cols {
            return Err(MultipointEvalError::WrongOpenEvalsNumber {
                got: open_evals.len(),
                expected: num_cols,
            });
        }

        // Step 1: Sample \alpha and \gamma_j (must match prover).
        let alpha: F = transcript.get_field_challenge(field_cfg);
        let gammas: Vec<F> = transcript.get_field_challenges(num_cols, field_cfg);

        // Step 2: Compute expected sum
        let expected_sum: F =
            compute_expected_sum(up_evals, down_evals, &gammas, &alpha, zero.clone());

        if proof.sumcheck_proof.claimed_sum != expected_sum {
            return Err(MultipointEvalError::WrongSumcheckSum {
                got: proof.sumcheck_proof.claimed_sum.clone(),
                expected: expected_sum,
            });
        }

        // Step 3: Verify the sumcheck.
        let subclaim = MLSumcheck::verify_as_subprotocol(
            transcript,
            num_vars,
            2,
            &proof.sumcheck_proof,
            field_cfg,
        )?;

        let r_0 = &subclaim.point;

        // Step 4: Recompute the combined selector at r_0.
        //   selector(r_0) = eq(r_0, r') + \alpha * next(r', r_0)
        let eq_at_r0 = zinc_poly::utils::eq_eval(r_0, eval_point, one.clone())?;

        let mut r_prime_r0 = Vec::with_capacity(2 * num_vars);
        r_prime_r0.extend_from_slice(eval_point);
        r_prime_r0.extend_from_slice(r_0);
        let next_at_r0 = zinc_poly::utils::next_mle_eval(&r_prime_r0, zero.clone(), one);

        let selector_at_r0 = eq_at_r0 + alpha * &next_at_r0;

        // Check consistency with provided open_evals.
        let batched_eval: F = gammas
            .iter()
            .zip(open_evals.iter())
            .fold(zero, |acc, (g, v)| acc + g.clone() * v);

        let expected_evaluation = selector_at_r0 * &batched_eval;

        if expected_evaluation != subclaim.expected_evaluation {
            return Err(MultipointEvalError::ClaimMismatch {
                got: subclaim.expected_evaluation,
                expected: expected_evaluation,
            });
        }

        Ok(Subclaim {
            eval_point: subclaim.point,
        })
    }
}

/// `expected_sum = \sum_j \gamma_j * (up_eval_j + \alpha * down_eval_j)`
fn compute_expected_sum<F: PrimeField>(
    up_evals: &[F],
    down_evals: &[F],
    gammas: &[F],
    alpha: &F,
    zero: F,
) -> F {
    gammas
        .iter()
        .zip(up_evals.iter().zip(down_evals.iter()))
        .fold(zero, |acc, (gamma, (up, down))| {
            acc + gamma.clone() * &(up.clone() + alpha.clone() * down)
        })
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
    struct PublicInput {
        eval_point: Vec<F>,
        up_evals: Vec<F>,
        down_evals: Vec<F>,
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
    ) -> (
        Vec<DenseMultilinearExtension<<F as crypto_primitives::Field>::Inner>>,
        PublicInput,
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

        let down_evals: Vec<F> = trace_mles
            .iter()
            .map(|mle| {
                let mut shifted = mle.evaluations[1..].to_vec();
                shifted.push(zero_inner);
                let shifted_mle =
                    DenseMultilinearExtension::from_evaluations_vec(num_vars, shifted, zero_inner);
                shifted_mle.evaluate_with_config(&eval_point, &()).unwrap()
            })
            .collect();

        let public = PublicInput {
            eval_point,
            up_evals,
            down_evals,
            num_vars,
        };
        (trace_mles, public)
    }

    /// Prover: has access to the trace, produces a proof and open_evals.
    fn run_prover(
        trace_mles: &[DenseMultilinearExtension<<F as crypto_primitives::Field>::Inner>],
        public: &PublicInput,
    ) -> ProverMessage {
        let mut transcript = make_transcript();
        let (proof, prover_state) = MultipointEval::<F>::prove_as_subprotocol(
            &mut transcript,
            trace_mles,
            &public.eval_point,
            &public.up_evals,
            &public.down_evals,
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
        public: &PublicInput,
        msg: &ProverMessage,
    ) -> Result<Subclaim<F>, MultipointEvalError<F>> {
        MultipointEval::<F>::verify_as_subprotocol(
            &mut make_transcript(),
            msg.proof.clone(),
            &public.eval_point,
            &public.up_evals,
            &public.down_evals,
            &msg.open_evals,
            public.num_vars,
            &(),
        )
    }

    /// Convenience: build trace, prove, return (public, message).
    fn honest_interaction(num_vars: usize, num_cols: usize) -> (PublicInput, ProverMessage) {
        let (trace, public) = build_trace(num_vars, num_cols);
        let msg = run_prover(&trace, &public);
        (public, msg)
    }

    // --- Happy-path ---

    #[test]
    fn honest_prove_verify_multi_column() {
        let (public, msg) = honest_interaction(3, 3);
        let subclaim = run_verifier(&public, &msg).unwrap();
        assert_eq!(subclaim.eval_point.len(), public.num_vars);
    }

    #[test]
    fn honest_prove_verify_single_column() {
        let (public, msg) = honest_interaction(4, 1);
        run_verifier(&public, &msg).unwrap();
    }

    #[test]
    fn honest_prove_verify_many_columns() {
        let (public, msg) = honest_interaction(3, 10);
        run_verifier(&public, &msg).unwrap();
    }

    // --- Failure: wrong number of open_evals ---

    #[test]
    fn wrong_open_evals_count() {
        let (public, msg) = honest_interaction(3, 3);

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
        let (mut public, msg) = honest_interaction(3, 3);
        public.up_evals[0] += F::ONE;
        let err = run_verifier(&public, &msg).unwrap_err();
        assert!(
            matches!(err, MultipointEvalError::WrongSumcheckSum { .. }),
            "expected WrongSumcheckSum, got {err:?}",
        );
    }

    #[test]
    fn wrong_claimed_sum_via_corrupted_down_evals() {
        let (mut public, msg) = honest_interaction(3, 3);
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
        let (public, mut msg) = honest_interaction(3, 3);
        msg.open_evals[0] += F::ONE;
        let err = run_verifier(&public, &msg).unwrap_err();
        assert!(
            matches!(err, MultipointEvalError::ClaimMismatch { .. }),
            "expected ClaimMismatch, got {err:?}",
        );
    }

    #[test]
    fn all_open_evals_zeroed() {
        let (public, mut msg) = honest_interaction(3, 3);
        for e in &mut msg.open_evals {
            *e = F::ZERO;
        }
        let err = run_verifier(&public, &msg).unwrap_err();
        assert!(
            matches!(err, MultipointEvalError::ClaimMismatch { .. }),
            "expected ClaimMismatch, got {err:?}",
        );
    }

    // --- Failure: tampered sumcheck round messages ---

    #[test]
    fn tampered_sumcheck_round_message() {
        let (public, mut msg) = honest_interaction(3, 3);
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
