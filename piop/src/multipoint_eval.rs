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
//! across columns. After the sumcheck reduces to point `r_0`, the verifier
//! needs only `v_j(r_0)` from the PCS - a single invocation.
//!
//! This corresponds to the T=2 case of Pi_{BMLE} in the paper.

use std::{marker::PhantomData, slice};

use crate::sumcheck::{MLSumcheck, SumCheckError, SumcheckProof};
use crypto_primitives::{FromPrimitiveWithConfig, PrimeField};
use num_traits::Zero;
use thiserror::Error;
use zinc_poly::{
    EvaluationError,
    mle::{DenseMultilinearExtension, MultilinearExtensionWithConfig},
    utils::ArithErrors,
};
use zinc_transcript::traits::{ConstTranscribable, Transcript};
use zinc_utils::inner_transparent_field::InnerTransparentField;

//
// Data structures
//

/// Proof for the multi-point evaluation protocol.
#[derive(Clone, Debug)]
pub struct Proof<F: PrimeField> {
    /// The inner sumcheck proof.
    pub sumcheck_proof: SumcheckProof<F>,
    /// Evaluations of each trace column MLE at the combined point `r_0`.
    pub open_evals: Vec<F>,
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
    /// The claimed trace MLE evaluations at r_0.
    pub open_evals: Vec<F>,
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
    /// Reduces J "up" evaluation claims and J "down" (shifted) evaluation
    /// claims at the shared point `eval_point` (r') to J standard MLE
    /// evaluation claims (`open_evals`) at a new random point r_0.
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

        // Step 1: Sample multi-point batching coefficient \alpha and column
        // batching coefficients \gamma_1,...,\gamma_J.
        let alpha: F = transcript.get_field_challenge(field_cfg);
        let gammas: Vec<F> = transcript.get_field_challenges(num_cols, field_cfg);

        // Step 2: Build the two selector MLEs:
        //   eq_r(b)   = eq(b, r')
        //   next_r(b) = next_tilde(r', b)
        let eq_r = zinc_poly::utils::build_eq_x_r_inner(eval_point, field_cfg)?;
        let next_r = zinc_poly::utils::build_next_r_mle(eval_point, field_cfg)?;

        // Step 3: Pack MLEs: [eq_r, next_r, v_1, ..., v_J]
        let mut mles: Vec<DenseMultilinearExtension<F::Inner>> = Vec::with_capacity(2 + num_cols);
        mles.push(eq_r);
        mles.push(next_r);
        for col in trace_mles {
            mles.push(col.clone());
        }

        // Step 4: Run sumcheck with degree=2.

        // comb_fn([eq_r, next_r, v_1, ..., v_J]) =
        //     (eq_r + \alpha * next_r) * \sum_j(\gamma_j * v_j)
        let (sumcheck_proof, sumcheck_prover_state) = MLSumcheck::prove_as_subprotocol(
            transcript,
            mles,
            num_vars,
            2,
            |mle_values: &[F]| {
                let eq_val = &mle_values[0];
                let next_val = &mle_values[1];
                let selector = eq_val.clone() + alpha.clone() * next_val;
                let batched = gammas
                    .iter()
                    .zip(mle_values[2..].iter())
                    .fold(zero.clone(), |acc, (g, v)| acc + g.clone() * v);
                selector * &batched
            },
            field_cfg,
        );

        // Sanity check
        debug_assert_eq!(
            sumcheck_proof.claimed_sum,
            compute_expected_sum(up_evals, down_evals, &gammas, &alpha, zero.clone())
        );

        // Step 5: Extract open_evals at the sumcheck challenge point r_0.
        debug_assert!(
            sumcheck_prover_state
                .mles
                .iter()
                .all(|mle| mle.num_vars == 1)
        );

        let last_challenge = sumcheck_prover_state
            .randomness
            .last()
            .expect("sumcheck must have at least one round");

        // Skip eq_r (index 0) and next_r (index 1), evaluate trace column MLEs.
        let open_evals: Vec<F> = sumcheck_prover_state.mles[2..]
            .iter()
            .map(|mle| mle.evaluate_with_config(slice::from_ref(last_challenge), field_cfg))
            .collect::<Result<Vec<_>, _>>()?;

        // Step 7: Absorb open_evals into transcript.
        let mut buf: Vec<u8> = vec![0; F::Inner::NUM_BYTES];
        transcript.absorb_random_field_slice(&open_evals, &mut buf);

        Ok((
            Proof {
                sumcheck_proof,
                open_evals,
            },
            ProverState {
                eval_point: sumcheck_prover_state.randomness,
            },
        ))
    }

    /// Multi-point evaluation protocol verifier.
    ///
    /// Verifies the sumcheck and returns evaluation claims at the combined
    /// point r_0, to be checked by the PCS.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn verify_as_subprotocol(
        transcript: &mut impl Transcript,
        proof: Proof<F>,
        eval_point: &[F],
        up_evals: &[F],
        down_evals: &[F],
        num_vars: usize,
        field_cfg: &F::Config,
    ) -> Result<Subclaim<F>, MultipointEvalError<F>> {
        let num_cols = up_evals.len();
        let zero = F::zero_with_cfg(field_cfg);
        let one = F::one_with_cfg(field_cfg);

        if proof.open_evals.len() != num_cols {
            return Err(MultipointEvalError::WrongOpenEvalsNumber {
                got: proof.open_evals.len(),
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

        // Step 5: Check consistency.
        let batched_eval: F = gammas
            .iter()
            .zip(proof.open_evals.iter())
            .fold(zero, |acc, (g, v)| acc + g.clone() * v);

        let expected_evaluation = selector_at_r0 * &batched_eval;

        if expected_evaluation != subclaim.expected_evaluation {
            return Err(MultipointEvalError::ClaimMismatch {
                got: subclaim.expected_evaluation,
                expected: expected_evaluation,
            });
        }

        // Step 6: Absorb open_evals into transcript.
        let mut buf: Vec<u8> = vec![0; F::Inner::NUM_BYTES];
        transcript.absorb_random_field_slice(&proof.open_evals, &mut buf);

        Ok(Subclaim {
            eval_point: subclaim.point,
            open_evals: proof.open_evals,
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
    #[error("MLE evaluation error: {0}")]
    MleEvaluation(#[from] EvaluationError),
}
