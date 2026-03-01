//! Batched shift sumcheck prover.
//!
//! Proves multiple shift-evaluation claims in a single sumcheck.

use crypto_primitives::FromPrimitiveWithConfig;
use num_traits::Zero;
use zinc_poly::mle::DenseMultilinearExtension;
use zinc_transcript::traits::{ConstTranscribable, Transcript};
use zinc_utils::inner_transparent_field::InnerTransparentField;

use super::predicate::build_left_shift_table;
use super::structs::*;

/// Run the batched shift sumcheck prover for **left-shift** (look-ahead)
/// claims.
///
/// Given `k` shift-evaluation claims, proves:
///
///   sum_{b in {0,1}^m} [ sum_i alpha_i · h_i[b] · v_i[b] ] = combined_claim
///
/// where h_i is the left-shift table for claim i and v_i is the source
/// column.  Batching coefficients alpha_i are drawn from the
/// Fiat-Shamir transcript.
///
/// Returns the proof and per-claim final evaluations needed for the
/// verifier to check the final claim and defer column openings to the PCS.
#[allow(clippy::arithmetic_side_effects)]
pub fn shift_sumcheck_prove<F>(
    transcript: &mut impl Transcript,
    claims: &[ShiftClaim<F>],
    trace_columns: &[DenseMultilinearExtension<F::Inner>],
    num_vars: usize,
    field_cfg: &F::Config,
) -> ShiftSumcheckProverOutput<F>
where
    F: InnerTransparentField + FromPrimitiveWithConfig + Send + Sync,
    F::Inner: ConstTranscribable + Send + Sync + Zero,
{
    assert!(!claims.is_empty(), "need at least one shift claim");
    let k = claims.len();
    let one = F::one_with_cfg(field_cfg);

    // Draw batching coefficients from the transcript.
    let alphas: Vec<F> = (0..k)
        .map(|_| transcript.get_field_challenge(field_cfg))
        .collect();

    // Build left-shift tables for each claim.
    let shift_tables: Vec<DenseMultilinearExtension<F::Inner>> = claims
        .iter()
        .map(|claim| build_left_shift_table(&claim.eval_point, claim.shift_amount, field_cfg))
        .collect();

    // Combined claim = sum_i alpha_i * claimed_eval_i.
    let combined_claim: F = alphas
        .iter()
        .zip(claims.iter())
        .map(|(a, c)| a.clone() * &c.claimed_eval)
        .fold(F::zero_with_cfg(field_cfg), |acc, x| acc + &x);

    // Working copies of shift tables and witness columns, lifted to F.
    let mut h_tables: Vec<Vec<F>> = shift_tables
        .iter()
        .map(|t| {
            t.evaluations
                .iter()
                .map(|e| F::new_unchecked_with_cfg(e.clone(), field_cfg))
                .collect()
        })
        .collect();

    let mut v_tables: Vec<Vec<F>> = claims
        .iter()
        .map(|claim| {
            trace_columns[claim.source_col]
                .evaluations
                .iter()
                .map(|e| F::new_unchecked_with_cfg(e.clone(), field_cfg))
                .collect()
        })
        .collect();

    let mut rounds = Vec::with_capacity(num_vars);
    let mut challenges = Vec::with_capacity(num_vars);
    let mut current_claim = combined_claim;

    for _round in 0..num_vars {
        let half = h_tables[0].len() / 2;

        // Compute round polynomial at X = 0, 1, 2.
        let mut total_e0 = F::zero_with_cfg(field_cfg);
        let mut total_e1 = F::zero_with_cfg(field_cfg);
        let mut total_e2 = F::zero_with_cfg(field_cfg);

        for i in 0..k {
            let (h_lo, h_hi) = h_tables[i].split_at(half);
            let (v_lo, v_hi) = v_tables[i].split_at(half);

            let mut e0 = F::zero_with_cfg(field_cfg);
            let mut e1 = F::zero_with_cfg(field_cfg);
            let mut e2 = F::zero_with_cfg(field_cfg);

            for j in 0..half {
                e0 = e0 + &(h_lo[j].clone() * &v_lo[j]);
                e1 = e1 + &(h_hi[j].clone() * &v_hi[j]);
                let h2 = h_hi[j].clone() + &h_hi[j] - &h_lo[j];
                let v2 = v_hi[j].clone() + &v_hi[j] - &v_lo[j];
                e2 = e2 + &(h2 * &v2);
            }

            total_e0 = total_e0 + &(alphas[i].clone() * &e0);
            total_e1 = total_e1 + &(alphas[i].clone() * &e1);
            total_e2 = total_e2 + &(alphas[i].clone() * &e2);
        }

        let rp = ShiftRoundPoly {
            evals: [total_e0, total_e1, total_e2],
        };

        // Absorb round polynomial into transcript.
        let mut buf = vec![0u8; F::Inner::NUM_BYTES];
        for eval in &rp.evals {
            transcript.absorb_random_field(eval, &mut buf);
        }

        // Get verifier challenge.
        let s: F = transcript.get_field_challenge(field_cfg);
        current_claim = rp.evaluate(&s);
        challenges.push(s.clone());

        // Fold tables: new[j] = (1 - s) * old[j] + s * old[half + j].
        let one_minus_s = one.clone() - &s;
        for i in 0..k {
            let half_len = h_tables[i].len() / 2;
            let mut new_h = Vec::with_capacity(half_len);
            let mut new_v = Vec::with_capacity(half_len);
            for j in 0..half_len {
                new_h.push(
                    h_tables[i][j].clone() * &one_minus_s
                        + &(h_tables[i][half_len + j].clone() * &s),
                );
                new_v.push(
                    v_tables[i][j].clone() * &one_minus_s
                        + &(v_tables[i][half_len + j].clone() * &s),
                );
            }
            h_tables[i] = new_h;
            v_tables[i] = new_v;
        }

        rounds.push(rp);
    }

    let h_finals: Vec<F> = h_tables.iter().map(|t| t[0].clone()).collect();
    let v_finals: Vec<F> = v_tables.iter().map(|t| t[0].clone()).collect();

    // Absorb the per-claim v_finals into transcript.
    let mut buf = vec![0u8; F::Inner::NUM_BYTES];
    for v in &v_finals {
        transcript.absorb_random_field(v, &mut buf);
    }

    ShiftSumcheckProverOutput {
        proof: ShiftSumcheckProof { rounds },
        challenge_point: challenges,
        final_claim: current_claim,
        h_finals,
        v_finals,
    }
}