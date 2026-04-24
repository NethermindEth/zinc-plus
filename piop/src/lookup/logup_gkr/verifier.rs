//! Logup-GKR verifier.

use crypto_primitives::{FromPrimitiveWithConfig, PrimeField};
use num_traits::Zero;
use std::marker::PhantomData;
use zinc_poly::utils::eq_eval;
use zinc_transcript::traits::{ConstTranscribable, Transcript};
use zinc_utils::inner_transparent_field::InnerTransparentField;

use crate::sumcheck::MLSumcheck;

use super::{
    error::LogupGkrError,
    proof::{LogupGkrProof, LogupGkrRoundProof},
    prover::LogupGkrSubclaim,
};

/// Logup-GKR verifier entry point.
pub struct LogupGkrVerifier<F>(PhantomData<F>);

impl<F> LogupGkrVerifier<F>
where
    F: PrimeField + InnerTransparentField + FromPrimitiveWithConfig,
    F::Inner: ConstTranscribable + Zero,
    F::Modulus: ConstTranscribable,
{
    /// Verify a logup-GKR proof.
    ///
    /// On success returns a [`LogupGkrSubclaim`] the caller must check
    /// against the *actual* leaf MLEs (i.e. the verifier's independent
    /// evaluations of `N_leaves` and `D_leaves` at the returned
    /// `point`). The protocol does NOT bind the leaves — that's the
    /// caller's responsibility.
    ///
    /// Also returns the claimed cumulative sum `root_n / root_d` via
    /// `(root_numerator, root_denominator)` so the caller can check it
    /// matches their expected target (e.g. zero for the standard
    /// lookup identity).
    #[allow(clippy::type_complexity)]
    pub fn verify(
        transcript: &mut impl Transcript,
        num_leaf_vars: usize,
        proof: &LogupGkrProof<F>,
        cfg: &F::Config,
    ) -> Result<(LogupGkrSubclaim<F>, F, F), LogupGkrError<F>> {
        assert!(
            num_leaf_vars >= 1,
            "logup-GKR requires at least 1 leaf variable"
        );

        if proof.round_proofs.len() != num_leaf_vars {
            return Err(LogupGkrError::InvalidShape {
                expected: num_leaf_vars,
                got: proof.round_proofs.len(),
            });
        }

        let mut buf = vec![0u8; F::Inner::NUM_BYTES];
        let nvars_f = F::from_with_cfg(num_leaf_vars as u64, cfg);
        transcript.absorb_random_field(&nvars_f, &mut buf);

        // Absorb the root values in the same order the prover sent them.
        transcript.absorb_random_field(&proof.root_numerator, &mut buf);
        transcript.absorb_random_field(&proof.root_denominator, &mut buf);

        let mut current_point: Vec<F> = Vec::new();
        let mut current_n = proof.root_numerator.clone();
        let mut current_d = proof.root_denominator.clone();

        for (layer_idx, round_proof) in proof.round_proofs.iter().enumerate() {
            // layer_idx = 0 -> nvp = 0 (first descent from scalar root).
            let nvp = layer_idx;

            let (next_point, next_n, next_d) = verify_one_layer::<F>(
                transcript,
                nvp,
                &current_point,
                current_n.clone(),
                current_d.clone(),
                round_proof,
                layer_idx,
                cfg,
                &mut buf,
            )?;

            current_point = next_point;
            current_n = next_n;
            current_d = next_d;
        }

        Ok((
            LogupGkrSubclaim {
                point: current_point,
                numerator_eval: current_n,
                denominator_eval: current_d,
            },
            proof.root_numerator.clone(),
            proof.root_denominator.clone(),
        ))
    }
}

#[allow(clippy::arithmetic_side_effects, clippy::too_many_arguments)]
fn verify_one_layer<F>(
    transcript: &mut impl Transcript,
    nvp: usize,
    current_point: &[F],
    current_n: F,
    current_d: F,
    round_proof: &LogupGkrRoundProof<F>,
    layer_idx: usize,
    cfg: &F::Config,
    buf: &mut Vec<u8>,
) -> Result<(Vec<F>, F, F), LogupGkrError<F>>
where
    F: PrimeField + InnerTransparentField + FromPrimitiveWithConfig,
    F::Inner: ConstTranscribable + Zero,
    F::Modulus: ConstTranscribable,
{
    let lambda: F = transcript.get_field_challenge(cfg);
    transcript.absorb_random_field(&lambda, buf);

    let one = F::from_with_cfg(1u64, cfg);

    let sumcheck_point: Vec<F> = if nvp == 0 {
        if round_proof.sumcheck_proof.is_some() {
            return Err(LogupGkrError::InvalidShape {
                expected: 0,
                got: 1,
            });
        }
        Vec::new()
    } else {
        let sc_proof = round_proof
            .sumcheck_proof
            .as_ref()
            .ok_or(LogupGkrError::InvalidShape {
                expected: 1,
                got: 0,
            })?;

        // Run the inner sumcheck verifier.
        let subclaim = MLSumcheck::<F>::verify_as_subprotocol(transcript, nvp, 3, sc_proof, cfg)
            .map_err(|e| LogupGkrError::Sumcheck {
                layer: layer_idx,
                source: e,
            })?;

        // Cross-check: the sumcheck's claimed sum must be lambda * current_n + current_d.
        let expected_claimed_sum = lambda.clone() * &current_n + current_d.clone();
        if sc_proof.claimed_sum != expected_claimed_sum {
            return Err(LogupGkrError::FinalEvalMismatch(layer_idx));
        }

        // Reconstruct the sumcheck's expected final evaluation from the tail values.
        let eq_eval_val = eq_eval(&subclaim.point, current_point, one.clone())
            .expect("eq_eval: matching dimensions");
        let num_part = round_proof.numerator_0.clone() * &round_proof.denominator_1
            + round_proof.numerator_1.clone() * &round_proof.denominator_0;
        let den_part = round_proof.denominator_0.clone() * &round_proof.denominator_1;
        let expected_final = eq_eval_val * (lambda.clone() * &num_part + den_part);
        if subclaim.expected_evaluation != expected_final {
            return Err(LogupGkrError::FinalEvalMismatch(layer_idx));
        }

        subclaim.point
    };

    if nvp == 0 {
        // Direct fold check: lambda * current_n + current_d
        //   == lambda * (n0*d1 + n1*d0) + d0*d1.
        let num_part = round_proof.numerator_0.clone() * &round_proof.denominator_1
            + round_proof.numerator_1.clone() * &round_proof.denominator_0;
        let den_part = round_proof.denominator_0.clone() * &round_proof.denominator_1;
        let expected_root_combo = lambda.clone() * &num_part + den_part;
        let actual_root_combo = lambda.clone() * &current_n + current_d.clone();
        if expected_root_combo != actual_root_combo {
            return Err(LogupGkrError::FinalEvalMismatch(layer_idx));
        }
    }

    // Absorb tail values (matching prover order).
    transcript.absorb_random_field(&round_proof.numerator_0, buf);
    transcript.absorb_random_field(&round_proof.numerator_1, buf);
    transcript.absorb_random_field(&round_proof.denominator_0, buf);
    transcript.absorb_random_field(&round_proof.denominator_1, buf);

    // Sample beta.
    let beta: F = transcript.get_field_challenge(cfg);
    transcript.absorb_random_field(&beta, buf);

    let one_minus_beta = one - &beta;
    let next_n =
        one_minus_beta.clone() * &round_proof.numerator_0 + beta.clone() * &round_proof.numerator_1;
    let next_d = one_minus_beta * &round_proof.denominator_0
        + beta.clone() * &round_proof.denominator_1;

    let mut next_point = sumcheck_point;
    next_point.push(beta);

    Ok((next_point, next_n, next_d))
}
