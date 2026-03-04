//! Batched Shift Protocol.
//!
//! Reduces shifted-MLE evaluation claims ("down" evaluations) to standard
//! MLE evaluation claims at a new random point `ρ` via a sumcheck.
//!
//! Given J shifted-MLE claims `y_j = \tilde{v}^{down}_j(r)` for trace columns `\tilde{v}_j`
//! and the shared evaluation point `r`, the protocol proves:
//!
//! ```text
//! \sum{b \in {0,1}^μ} shiftdown_r(b) * (\sum{j} \gamma_j * \tilde{v}_j(b)) = \sum{j} \gamma_j * y_j
//! ```
//!
//! where `shiftdown_r(b) = \tilde{next}(r, b) + eq(r, 1) * eq(b, 1)`.
//!
//! The sumcheck reduces verification to openings of each `\tilde{v}_j` at a
//! single random point `ρ`, which is then verified by the PCS.

use std::slice;

use crypto_primitives::{FromPrimitiveWithConfig, PrimeField};
use num_traits::Zero;
use thiserror::Error;
use zinc_poly::{
    EvaluationError,
    mle::{DenseMultilinearExtension, MultilinearExtensionWithConfig},
    utils::{ArithErrors, build_eq_x_r_inner, next_mle_eval},
};
use zinc_transcript::traits::{ConstTranscribable, Transcript};
use zinc_utils::inner_transparent_field::InnerTransparentField;

use crate::sumcheck::{MLSumcheck, SumCheckError, SumcheckProof};

//
// Data structures
//

/// Proof for the batched shift protocol.
#[derive(Clone, Debug)]
pub struct Proof<F: PrimeField> {
    /// The inner sumcheck proof.
    pub sumcheck_proof: SumcheckProof<F>,
    /// Evaluations of each trace column MLE at the new point `ρ`.
    pub shift_evals: Vec<F>,
}

/// Prover state after the batched shift protocol.
pub struct ProverState<F: PrimeField> {
    /// The new evaluation point `ρ` produced by the shift sumcheck.
    pub shift_point: Vec<F>,
}

/// Verifier subclaim after the batched shift protocol.
#[derive(Clone, Debug)]
pub struct Subclaim<F: PrimeField> {
    /// The new evaluation point ρ.
    pub shift_point: Vec<F>,
    /// The claimed trace MLE evaluations at ρ.
    pub shift_evals: Vec<F>,
}

//
// Prover
//

/// Build the shift selector MLE `\tilde{next}(r, ?)` with the first `num_vars` variables fixed
/// to `r`.
///
/// For each `b \in {0,1}^{num_vars}`:
///   `shift_r(b) = \tilde{next}̃(r, b)`
///
/// Uses the identity `next̃(r, b) = eq(r, b−1)` for `b ≥ 1` and `0` for `b = 0`,
/// delegating to [`build_eq_x_r_inner`] for an O(2^μ) construction (vs the
/// naive O(μ² · 2^μ) pointwise evaluation).
///
/// This matches the zero-padded shift convention used by
/// [`CombinedPolyResolver`]: `shift(a) = f(a+1)` for `a < N-1`, `shift(N-1) =
/// 0`.
///
/// Note: the paper uses a clamping variant `shiftdown_r(b) = \tilde{next}(r, b) +
/// eq(r,1) * eq(b,1)` which repeats the last row instead of zero-padding. We use
/// the simpler zero-padded version. This is sound because `CombinedPolyResolver`
/// multiplies every constraint by `(1 - eq((1,...,1), b))`, which is zero at
/// `b = N-1`, so the value of `down[N-1]` never contributes to the proved sum.
fn build_shift_r_mle<F>(
    r: &[F],
    field_cfg: &F::Config,
) -> Result<DenseMultilinearExtension<F::Inner>, BatchedShiftError<F>>
where
    F: InnerTransparentField + FromPrimitiveWithConfig,
    F::Inner: Zero + Default,
{
    let num_vars = r.len();
    let n = 1 << num_vars;
    let zero_inner = F::zero_with_cfg(field_cfg).into_inner();

    let eq_r = build_eq_x_r_inner(r, field_cfg)?;

    // \tilde{next}(r, 0) = 0; \tilde{next}(r, b) = eq(r, b-1) for b >= 1.
    let mut evaluations = Vec::with_capacity(n);
    evaluations.push(zero_inner);
    evaluations.extend_from_slice(&eq_r.evaluations[..n - 1]);

    Ok(DenseMultilinearExtension {
        num_vars,
        evaluations,
    })
}

/// Batched shift protocol prover.
///
/// Reduces J shifted-MLE evaluation claims (`down_evals`) to J standard
/// MLE evaluation claims (`shift_evals`) at a new random point ρ.
#[allow(clippy::arithmetic_side_effects)]
pub fn prove_as_subprotocol<F>(
    transcript: &mut impl Transcript,
    trace_mles: &[DenseMultilinearExtension<F::Inner>],
    evaluation_point: &[F],
    _down_evals: &[F],
    field_cfg: &F::Config,
) -> Result<(Proof<F>, ProverState<F>), BatchedShiftError<F>>
where
    F: InnerTransparentField + FromPrimitiveWithConfig + Send + Sync + 'static,
    F::Inner: ConstTranscribable + Send + Sync + Zero + Default,
    F::Modulus: ConstTranscribable,
{
    let num_cols = trace_mles.len();
    let num_vars = evaluation_point.len();
    let zero = F::zero_with_cfg(field_cfg);

    // Step 1: Sample batching coefficients γ_1,...,γ_J
    let gammas: Vec<F> = transcript.get_field_challenges(num_cols, field_cfg);

    // Step 2: Build shift_r MLE (next̃(r, ·))
    let shift_r = build_shift_r_mle(evaluation_point, field_cfg)?;

    // Step 3: Pack MLEs: [shift_r, v_1, ..., v_J]
    let mut mles: Vec<DenseMultilinearExtension<F::Inner>> = Vec::with_capacity(1 + num_cols);
    mles.push(shift_r);
    for col in trace_mles {
        mles.push(col.clone());
    }

    // Step 4: Run sumcheck with degree=2
    // comb_fn([sd, v_1, ..., v_J]) = sd · Σ_j(γ_j · v_j)
    let (sumcheck_proof, sumcheck_prover_state) = MLSumcheck::prove_as_subprotocol(
        transcript,
        mles,
        num_vars,
        2, // degree: product of two multilinear polynomials
        |mle_values: &[F]| {
            let sd = &mle_values[0];
            let batched = gammas
                .iter()
                .zip(mle_values[1..].iter())
                .fold(zero.clone(), |acc, (g, v)| acc + g.clone() * v);
            sd.clone() * &batched
        },
        field_cfg,
    );

    // Step 5: Extract shift_evals at the sumcheck challenge point ρ.
    // The sumcheck prover leaves all MLEs in num_vars=1 state.
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

    // Evaluate each trace column MLE (skip shiftdown_r at index 0)
    let shift_evals: Vec<F> = sumcheck_prover_state.mles[1..]
        .iter()
        .map(|mle| mle.evaluate_with_config(slice::from_ref(last_challenge), field_cfg))
        .collect::<Result<Vec<_>, _>>()?;

    // Step 6: Absorb shift_evals into transcript
    let mut buf: Vec<u8> = vec![0; F::Inner::NUM_BYTES];
    transcript.absorb_random_field_slice(&shift_evals, &mut buf);

    Ok((
        Proof {
            sumcheck_proof,
            shift_evals,
        },
        ProverState {
            shift_point: sumcheck_prover_state.randomness,
        },
    ))
}

//
// Verifier
//

/// Batched shift protocol verifier.
///
/// Verifies the shift sumcheck and returns evaluation claims at the new
/// point ρ, to be checked by the PCS.
#[allow(clippy::arithmetic_side_effects)]
pub fn verify_as_subprotocol<F>(
    transcript: &mut impl Transcript,
    proof: Proof<F>,
    evaluation_point: &[F],
    down_evals: &[F],
    num_vars: usize,
    field_cfg: &F::Config,
) -> Result<Subclaim<F>, BatchedShiftError<F>>
where
    F: InnerTransparentField + FromPrimitiveWithConfig + Send + Sync,
    F::Inner: ConstTranscribable + Zero + Default,
    F::Modulus: ConstTranscribable,
{
    let num_cols = down_evals.len();
    let zero = F::zero_with_cfg(field_cfg);
    let one = F::one_with_cfg(field_cfg);

    // Validate proof dimensions
    if proof.shift_evals.len() != num_cols {
        return Err(BatchedShiftError::WrongShiftEvalsNumber {
            got: proof.shift_evals.len(),
            expected: num_cols,
        });
    }

    // Step 1: Sample batching coefficients (must match prover)
    let gammas: Vec<F> = transcript.get_field_challenges(num_cols, field_cfg);

    // Step 2: Compute expected sum = Σ_j γ_j · y_j
    let expected_sum: F = gammas
        .iter()
        .zip(down_evals.iter())
        .fold(zero.clone(), |acc, (g, y)| acc + g.clone() * y);

    // Step 3: Check claimed sum
    if proof.sumcheck_proof.claimed_sum != expected_sum {
        return Err(BatchedShiftError::WrongSumcheckSum {
            got: proof.sumcheck_proof.claimed_sum.clone(),
            expected: expected_sum,
        });
    }

    // Step 4: Verify the sumcheck
    let subclaim = MLSumcheck::verify_as_subprotocol(
        transcript,
        num_vars,
        2,
        &proof.sumcheck_proof,
        field_cfg,
    )?;

    let rho = &subclaim.point;

    // Step 5: Recompute the combined polynomial at ρ.
    // shift_r(ρ) = next̃(r, ρ)  (zero-padded convention, no clamping term)
    let mut r_rho = Vec::with_capacity(2 * num_vars);
    r_rho.extend_from_slice(evaluation_point);
    r_rho.extend_from_slice(rho);

    let shift_at_rho = next_mle_eval(&r_rho, zero.clone(), one);

    let batched_eval: F = gammas
        .iter()
        .zip(proof.shift_evals.iter())
        .fold(zero, |acc, (g, v)| acc + g.clone() * v);

    let expected_evaluation = shift_at_rho * &batched_eval;

    if expected_evaluation != subclaim.expected_evaluation {
        return Err(BatchedShiftError::ClaimMismatch {
            got: subclaim.expected_evaluation,
            expected: expected_evaluation,
        });
    }

    // Step 6: Absorb shift_evals into transcript
    let mut buf: Vec<u8> = vec![0; F::Inner::NUM_BYTES];
    transcript.absorb_random_field_slice(&proof.shift_evals, &mut buf);

    Ok(Subclaim {
        shift_point: subclaim.point,
        shift_evals: proof.shift_evals,
    })
}

//
// Error type
//

#[derive(Debug, Error)]
pub enum BatchedShiftError<F: PrimeField> {
    #[error("wrong number of shift evaluations: got {got}, expected {expected}")]
    WrongShiftEvalsNumber { got: usize, expected: usize },
    #[error("wrong sumcheck claimed sum: got {got}, expected {expected}")]
    WrongSumcheckSum { got: F, expected: F },
    #[error("shift claim mismatch: got {got}, expected {expected}")]
    ClaimMismatch { got: F, expected: F },
    #[error("sumcheck error: {0}")]
    SumcheckError(#[from] SumCheckError<F>),
    #[error("arithmetic error: {0}")]
    ArithError(#[from] ArithErrors),
    #[error("MLE evaluation error: {0}")]
    MleEvaluation(#[from] EvaluationError),
}
