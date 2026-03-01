//! Batched shift sumcheck verifier.
//!
//! Verifies the proof produced by [`shift_sumcheck_prove`] and returns
//! per-claim evaluation claims about the **unshifted** source columns.

use crypto_primitives::{FromPrimitiveWithConfig, PrimeField};
use zinc_transcript::traits::{ConstTranscribable, Transcript};
use zinc_utils::inner_transparent_field::InnerTransparentField;

use super::predicate::eval_left_shift_predicate;
use super::structs::*;

/// Result of verifying the shift sumcheck.
#[derive(Clone, Debug)]
pub struct ShiftSumcheckVerifierOutput<F: PrimeField> {
    /// The random challenge point `s ∈ F^m` reconstructed from the proof.
    pub challenge_point: Vec<F>,
    /// Per-claim: the source column index whose opening at `s` is needed.
    pub source_cols: Vec<usize>,
    /// Per-claim: the claimed evaluation `v_i(s)` (provided by the prover,
    /// must be checked against a PCS opening).
    pub v_finals: Vec<F>,
}

/// Verify a batched shift sumcheck proof.
///
/// Returns [`ShiftSumcheckVerifierOutput`] on success, or an error string.
///
/// The verifier:
/// 1. Draws the same batching coefficients α_i from the transcript.
/// 2. Replays the sumcheck: for each round, absorbs the round poly,
///    checks `p(0) + p(1) == current_claim`, draws the challenge.
/// 3. Computes `S_{c_i}(s, r_i)` for each claim via [`eval_shift_predicate`].
/// 4. Absorbs the prover-supplied `v_finals`.
/// 5. Checks the final claim: `final == Σ_i α_i · S_{c_i}(s, r_i) · v_i(s)`.
///
/// The caller must additionally verify that each `v_finals[i]` matches
/// a PCS opening of `source_cols[i]` at the challenge point `s`.
#[allow(clippy::arithmetic_side_effects)]
pub fn shift_sumcheck_verify<F>(
    transcript: &mut impl Transcript,
    proof: &ShiftSumcheckProof<F>,
    claims: &[ShiftClaim<F>],
    v_finals: &[F],
    num_vars: usize,
    field_cfg: &F::Config,
) -> Result<ShiftSumcheckVerifierOutput<F>, String>
where
    F: InnerTransparentField + FromPrimitiveWithConfig + Send + Sync,
    F::Inner: ConstTranscribable + Send + Sync + num_traits::Zero,
{
    let k = claims.len();
    if proof.rounds.len() != num_vars {
        return Err(format!(
            "expected {} round polynomials, got {}",
            num_vars,
            proof.rounds.len()
        ));
    }
    if v_finals.len() != k {
        return Err(format!(
            "expected {} v_finals, got {}",
            k,
            v_finals.len()
        ));
    }

    // Draw the same batching coefficients.
    let alphas: Vec<F> = (0..k)
        .map(|_| transcript.get_field_challenge(field_cfg))
        .collect();

    // Combined claim.
    let combined_claim: F = alphas
        .iter()
        .zip(claims.iter())
        .map(|(a, c)| a.clone() * &c.claimed_eval)
        .fold(F::zero_with_cfg(field_cfg), |acc, x| acc + &x);

    // Replay the sumcheck rounds.
    let mut current_claim = combined_claim;
    let mut challenges = Vec::with_capacity(num_vars);
    let mut buf = vec![0u8; F::Inner::NUM_BYTES];

    for round_poly in &proof.rounds {
        // Check round polynomial consistency: p(0) + p(1) == current_claim.
        let sum = round_poly.evals[0].clone() + &round_poly.evals[1];
        if sum != current_claim {
            return Err("round polynomial check failed: p(0) + p(1) != claim".into());
        }

        // Absorb round polynomial.
        for eval in &round_poly.evals {
            transcript.absorb_random_field(eval, &mut buf);
        }

        // Draw challenge.
        let s: F = transcript.get_field_challenge(field_cfg);
        current_claim = round_poly.evaluate(&s);
        challenges.push(s);
    }

    // Absorb the prover-supplied v_finals.
    for v in v_finals {
        transcript.absorb_random_field(v, &mut buf);
    }

    // Compute expected final claim:
    //   final == Σ_i α_i · L_{c_i}(s, r_i) · v_i(s)
    // where L_c is the left-shift predicate.
    //
    // The eval_point r comes from CPR in little-endian convention (matching
    // build_eq_x_r_inner).  The sumcheck challenges are in big-endian order
    // (round 0 binds the MSB of the table index).  eval_left_shift_predicate
    // uses big-endian eval_delta / eval_next, so we must reverse r into BE.
    let mut expected_final = F::zero_with_cfg(field_cfg);
    for i in 0..k {
        let eval_point_rev: Vec<F> = claims[i].eval_point.iter().rev().cloned().collect();
        let h_val = eval_left_shift_predicate(&challenges, &eval_point_rev, claims[i].shift_amount);
        expected_final = expected_final + &(alphas[i].clone() * &h_val * &v_finals[i]);
    }

    if current_claim != expected_final {
        return Err("final claim mismatch: sumcheck reduced value != Σ αᵢ·Sᵢ·vᵢ".into());
    }

    Ok(ShiftSumcheckVerifierOutput {
        challenge_point: challenges,
        source_cols: claims.iter().map(|c| c.source_col).collect(),
        v_finals: v_finals.to_vec(),
    })
}
