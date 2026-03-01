//! Batched shift sumcheck prover.
//!
//! Proves multiple shift-evaluation claims in a single sumcheck.
//!
//! # Optimisations
//!
//! * **Table grouping** – claims that share the same `(eval_point,
//!   shift_amount)` reference the same predicate table.  The prover
//!   builds one table per *distinct* pair and pre-combines the witness
//!   columns within each group:  `w_g[j] = Σ_{i∈g} α_i · v_i[j]`.
//!   In the typical unified-eval-sumcheck scenario this reduces 29
//!   per-claim tables to 2 group tables.
//!
//! * **Parallel inner loops** – table construction, pre-combination,
//!   round-polynomial evaluation, table folding, and final MLE
//!   evaluations all use `cfg_into_iter!` so that rayon kicks in when
//!   the `parallel` feature is active.

use crypto_primitives::FromPrimitiveWithConfig;
use num_traits::Zero;
use zinc_poly::mle::DenseMultilinearExtension;
use zinc_poly::utils::build_eq_x_r_inner;
use zinc_transcript::traits::{ConstTranscribable, Transcript};
use zinc_utils::inner_transparent_field::InnerTransparentField;

#[cfg(feature = "parallel")]
use rayon::iter::*;
use zinc_utils::cfg_into_iter;

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
/// Claims that share the same `(eval_point, shift_amount)` are grouped
/// so that only one predicate table is built and folded per group.
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
    let n = 1usize << num_vars;
    let one = F::one_with_cfg(field_cfg);

    // ── Draw batching coefficients ───────────────────────────────────
    let alphas: Vec<F> = (0..k)
        .map(|_| transcript.get_field_challenge(field_cfg))
        .collect();

    // Combined claim = Σ_i α_i · claimed_eval_i.
    let combined_claim: F = alphas
        .iter()
        .zip(claims.iter())
        .map(|(a, c)| a.clone() * &c.claimed_eval)
        .fold(F::zero_with_cfg(field_cfg), |acc, x| acc + &x);

    // ── Group claims by (eval_point, shift_amount) ───────────────────
    // Each group shares a single predicate table h.
    //
    // `group_rep[g]`         – index of the representative claim for group g
    // `group_members[g]`     – list of claim indices in group g
    // `claim_to_group[i]`    – group index for claim i
    let mut group_rep: Vec<usize> = Vec::new();
    let mut group_members: Vec<Vec<usize>> = Vec::new();
    let mut claim_to_group: Vec<usize> = Vec::with_capacity(k);

    for i in 0..k {
        let found = group_rep.iter().position(|&rep| {
            claims[rep].shift_amount == claims[i].shift_amount
                && claims[rep].eval_point == claims[i].eval_point
        });
        match found {
            Some(gi) => {
                group_members[gi].push(i);
                claim_to_group.push(gi);
            }
            None => {
                claim_to_group.push(group_rep.len());
                group_rep.push(i);
                group_members.push(vec![i]);
            }
        }
    }
    let num_groups = group_rep.len();

    // ── Build one predicate table per group (parallelised) ───────────
    let mut h_tables: Vec<Vec<F>> = Vec::with_capacity(num_groups);
    for &rep in &group_rep {
        let raw = build_left_shift_table(
            &claims[rep].eval_point,
            claims[rep].shift_amount,
            field_cfg,
        );
        let lifted: Vec<F> = cfg_into_iter!(raw.evaluations)
            .map(|e| F::new_unchecked_with_cfg(e, field_cfg))
            .collect();
        h_tables.push(lifted);
    }

    // ── Pre-combine witness columns per group ────────────────────────
    // w_g[j] = Σ_{i ∈ group_g} α_i · v_i[j]
    let mut w_tables: Vec<Vec<F>> = Vec::with_capacity(num_groups);
    for members in &group_members {
        // Collect (alpha, column_slice) pairs for this group.
        let pairs: Vec<(&F, &[F::Inner])> = members
            .iter()
            .map(|&i| (&alphas[i], trace_columns[claims[i].source_col].evaluations.as_slice()))
            .collect();

        let combined: Vec<F> = cfg_into_iter!(0..n)
            .map(|j| {
                let mut acc = F::zero_with_cfg(field_cfg);
                for &(alpha, col) in &pairs {
                    let v = F::new_unchecked_with_cfg(col[j].clone(), field_cfg);
                    acc = acc + &(alpha.clone() * &v);
                }
                acc
            })
            .collect();
        w_tables.push(combined);
    }

    // ── Sumcheck rounds ──────────────────────────────────────────────
    let mut rounds = Vec::with_capacity(num_vars);
    let mut challenges = Vec::with_capacity(num_vars);
    let mut current_claim = combined_claim;

    for _round in 0..num_vars {
        let half = h_tables[0].len() / 2;

        // Accumulate round-polynomial evaluations across groups.
        let mut total_e0 = F::zero_with_cfg(field_cfg);
        let mut total_e1 = F::zero_with_cfg(field_cfg);
        let mut total_e2 = F::zero_with_cfg(field_cfg);

        for g in 0..num_groups {
            let (h_lo, h_hi) = h_tables[g].split_at(half);
            let (w_lo, w_hi) = w_tables[g].split_at(half);

            // Parallel fold over j ∈ [0, half).
            let zero = F::zero_with_cfg(field_cfg);
            #[cfg(not(feature = "parallel"))]
            let init = (zero.clone(), zero.clone(), zero.clone());
            #[cfg(feature = "parallel")]
            let init = || (zero.clone(), zero.clone(), zero.clone());

            let folded = cfg_into_iter!(0..half).fold(init, |(mut e0, mut e1, mut e2), j| {
                e0 = e0 + &(h_lo[j].clone() * &w_lo[j]);
                e1 = e1 + &(h_hi[j].clone() * &w_hi[j]);
                let h2 = h_hi[j].clone() + &h_hi[j] - &h_lo[j];
                let w2 = w_hi[j].clone() + &w_hi[j] - &w_lo[j];
                e2 = e2 + &(h2 * &w2);
                (e0, e1, e2)
            });

            #[cfg(feature = "parallel")]
            let (e0, e1, e2) = folded.reduce(
                || (zero.clone(), zero.clone(), zero.clone()),
                |(a0, a1, a2), (b0, b1, b2)| (a0 + &b0, a1 + &b1, a2 + &b2),
            );
            #[cfg(not(feature = "parallel"))]
            let (e0, e1, e2) = folded;

            total_e0 = total_e0 + &e0;
            total_e1 = total_e1 + &e1;
            total_e2 = total_e2 + &e2;
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

        // Fold group tables: new[j] = (1−s)·old[j] + s·old[j+half].
        let one_minus_s = one.clone() - &s;
        for g in 0..num_groups {
            let half_len = h_tables[g].len() / 2;
            let (h_src, w_src) = (&h_tables[g], &w_tables[g]);

            let new_hw: Vec<(F, F)> = cfg_into_iter!(0..half_len)
                .map(|j| {
                    let h = h_src[j].clone() * &one_minus_s
                        + &(h_src[half_len + j].clone() * &s);
                    let w = w_src[j].clone() * &one_minus_s
                        + &(w_src[half_len + j].clone() * &s);
                    (h, w)
                })
                .collect();

            let (new_h, new_w): (Vec<F>, Vec<F>) = new_hw.into_iter().unzip();
            h_tables[g] = new_h;
            w_tables[g] = new_w;
        }

        rounds.push(rp);
    }

    // ── Per-claim h_finals ───────────────────────────────────────────
    // Claims in the same group share the same final h value.
    let h_finals: Vec<F> = claim_to_group
        .iter()
        .map(|&gi| h_tables[gi][0].clone())
        .collect();

    // ── Per-claim v_finals via shared eq table ───────────────────────
    // MLE[v_i](s) = Σ_j eq(s, j) · v_i[j] where s is the challenge
    // point.  Build eq(s, ·) once (O(n)) and re-use for all claims.
    //
    // The sumcheck folds MSB-first, so challenges[0] bound the top bit.
    // `build_eq_x_r_inner` uses LE ordering (r[0] = bit 0), so we
    // pass the challenges reversed.
    let challenges_le: Vec<F> = challenges.iter().rev().cloned().collect();
    let eq_at_s = build_eq_x_r_inner(&challenges_le, field_cfg)
        .expect("build_eq_x_r_inner should succeed");

    let v_finals: Vec<F> = cfg_into_iter!(0..k)
        .map(|i| {
            let col = &trace_columns[claims[i].source_col].evaluations;
            col.iter()
                .zip(eq_at_s.evaluations.iter())
                .map(|(v, eq)| {
                    let v_f = F::new_unchecked_with_cfg(v.clone(), field_cfg);
                    let eq_f = F::new_unchecked_with_cfg(eq.clone(), field_cfg);
                    v_f * &eq_f
                })
                .fold(F::zero_with_cfg(field_cfg), |acc, x| acc + &x)
        })
        .collect();

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