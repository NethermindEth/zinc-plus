//! Hybrid GKR + send-in-clear LogUp protocol.
//!
//! Explores the tradeoff between running full GKR (small proof, more
//! verifier work) and sending intermediate fraction values in the clear
//! (larger proof, less verifier work).
//!
//! ## Protocol overview
//!
//! Given a GKR fraction tree of depth `d`, the hybrid protocol runs only
//! the first `c` layers of GKR (from root), where `c` is the "cutoff depth".
//! Instead of continuing the GKR for layers `c..d-1`, the prover sends
//! the intermediate fraction values at layer `c` in the clear.
//!
//! ### Soundness note
//!
//! The intermediate values are absorbed into the transcript **before** the
//! GKR challenges are derived, ensuring the prover commits to them before
//! seeing the evaluation point `r_c`. The GKR layers 0..c-1 verify
//! consistency between the root and the intermediate values.
//!
//! For full soundness, an additional verification step is needed to
//! check that the intermediate values are consistent with the committed
//! leaf data (e.g., via remaining GKR layers, a product argument, or
//! classic inverse-witness sumcheck). This module provides the "top half"
//! of the hybrid and measures the associated costs; the "bottom half"
//! verification is application-specific and measured separately.
//!
//! ## Cost tradeoff
//!
//! At cutoff depth `c` from the root (tree depth `d`, `L` batched trees):
//!
//! - **GKR layers saved**: layers c..d-1, each with k-variable sumchecks
//!   - Sumcheck rounds saved: `Σ_{k=c}^{d-1} k = (d-c)(d+c-1)/2`
//!   - Evaluation messages saved: `4L × (d - c)` field elements
//!
//! - **Intermediate values sent**: `2L × 2^c` field elements (p_c + q_c)
//!
//! - **Verifier work added**: MLE evaluation of sent vectors = O(L × 2^c)
//!
//! ## Usage
//!
//! ```ignore
//! let result = HybridGkrBatchedDecompLogupProtocol::prove_as_subprotocol(
//!     &mut transcript, &instance, cutoff, &field_cfg,
//! );
//! ```

use crypto_primitives::{FromPrimitiveWithConfig, PrimeField};
use num_traits::Zero;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::marker::PhantomData;
use zinc_poly::utils::build_eq_x_r_vec;
use zinc_transcript::traits::{ConstTranscribable, Transcript};
use zinc_utils::{cfg_iter, cfg_into_iter, inner_transparent_field::InnerTransparentField};

use super::gkr_logup::{
    build_fraction_tree, FractionLayer,
    gkr_fraction_prove, gkr_fraction_verify,
};
use super::structs::{
    BatchedDecompLookupInstance, LookupError,
};
use super::tables::{build_table_index, compute_multiplicities_with_index};

use crate::sumcheck::MLSumcheck;
use zinc_poly::mle::DenseMultilinearExtension;
use zinc_poly::utils::build_eq_x_r_inner;

// ---------------------------------------------------------------------------
// Proof structures
// ---------------------------------------------------------------------------

/// Proof for lattice layer k of the hybrid batched GKR.
#[derive(Clone, Debug)]
pub struct HybridBatchedGkrLayerProof<F: PrimeField> {
    /// Shared sumcheck proof for this layer (`None` for k = 0).
    pub sumcheck_proof: Option<crate::sumcheck::SumcheckProof<F>>,
    /// Per-tree left-child numerator evaluations.
    pub p_lefts: Vec<F>,
    /// Per-tree right-child numerator evaluations.
    pub p_rights: Vec<F>,
    /// Per-tree left-child denominator evaluations.
    pub q_lefts: Vec<F>,
    /// Per-tree right-child denominator evaluations.
    pub q_rights: Vec<F>,
}

/// Proof for the batched hybrid GKR witness trees.
#[derive(Clone, Debug)]
pub struct HybridBatchedGkrFractionProof<F: PrimeField> {
    /// Per-tree root numerators.
    pub roots_p: Vec<F>,
    /// Per-tree root denominators.
    pub roots_q: Vec<F>,
    /// Per-layer proofs for layers 0..cutoff.
    pub layer_proofs: Vec<HybridBatchedGkrLayerProof<F>>,
    /// Cutoff depth from root.
    pub cutoff: usize,
    /// Per-tree intermediate numerators at the cutoff layer.
    /// `sent_p[ℓ][j]` for j ∈ [2^cutoff].
    pub sent_p: Vec<Vec<F>>,
    /// Per-tree intermediate denominators at the cutoff layer.
    /// `sent_q[ℓ][j]` for j ∈ [2^cutoff].
    pub sent_q: Vec<Vec<F>>,
}

/// Complete proof for the hybrid GKR batched decomposition + LogUp.
#[derive(Clone, Debug)]
pub struct HybridGkrBatchedDecompLogupProof<F: PrimeField> {
    /// Per-lookup aggregated multiplicity vectors.
    pub aggregated_multiplicities: Vec<Vec<F>>,
    /// Hybrid batched GKR proof for L witness trees.
    pub witness_gkr: HybridBatchedGkrFractionProof<F>,
    /// Full GKR proof for the table tree (typically small depth).
    pub table_gkr: super::structs::GkrFractionProof<F>,
}

/// Prover state after hybrid GKR batched decomposition + LogUp.
#[derive(Clone, Debug)]
pub struct HybridGkrProverState<F: PrimeField> {
    /// Evaluation point at the witness cutoff layer: `r ∈ F^{cutoff}`.
    pub witness_eval_point: Vec<F>,
    /// Evaluation point at the table-tree leaf level.
    pub table_eval_point: Vec<F>,
    /// Number of witness variables (full tree depth).
    pub witness_num_vars: usize,
    /// Number of table variables.
    pub table_num_vars: usize,
    /// Batching challenge α.
    pub alpha: F,
    /// Challenge β.
    pub beta: F,
    /// Number of lookups L.
    pub num_lookups: usize,
    /// Number of chunks K.
    pub num_chunks: usize,
    /// Witness length W.
    pub witness_len: usize,
    /// Cutoff depth c.
    pub cutoff: usize,
}

/// Verifier sub-claim after hybrid GKR verification.
#[derive(Clone, Debug)]
pub struct HybridGkrVerifierSubClaim<F: PrimeField> {
    /// Evaluation point at the witness cutoff layer: `r ∈ F^{cutoff}`.
    pub witness_eval_point: Vec<F>,
    /// Per-tree expected p MLE evaluations at r_c.
    pub expected_witness_p_evals: Vec<F>,
    /// Per-tree expected q MLE evaluations at r_c.
    pub expected_witness_q_evals: Vec<F>,
    /// Table evaluation point.
    pub table_eval_point: Vec<F>,
    /// The sent intermediate p values (for leaf verification).
    pub sent_p: Vec<Vec<F>>,
    /// The sent intermediate q values (for leaf verification).
    pub sent_q: Vec<Vec<F>>,
}

// ---------------------------------------------------------------------------
// Pipeline-level proof container
// ---------------------------------------------------------------------------

/// Complete hybrid GKR-based lookup proof for the pipeline.
#[derive(Clone, Debug)]
pub struct HybridGkrPipelineLookupProof<F: PrimeField> {
    /// Per-group hybrid GKR proofs.
    pub group_proofs: Vec<HybridGkrBatchedDecompLogupProof<F>>,
    /// Per-group metadata.
    pub group_meta: Vec<super::pipeline::LookupGroupMeta>,
}

/// Prover state from hybrid GKR pipeline.
#[derive(Clone, Debug)]
pub struct HybridGkrPipelineLookupProverState<F: PrimeField> {
    /// Per-group prover states.
    pub group_states: Vec<HybridGkrProverState<F>>,
}

// ---------------------------------------------------------------------------
// Hybrid batched GKR fraction prove/verify
// ---------------------------------------------------------------------------

/// Result of the hybrid batched GKR fractional sumcheck prover.
struct HybridBatchedGkrFractionProveResult<F: PrimeField> {
    proof: HybridBatchedGkrFractionProof<F>,
    eval_point: Vec<F>,
}

/// Run the hybrid batched GKR prover for L fraction trees.
///
/// Processes only the first `cutoff` GKR layers (from root), then sends
/// the intermediate fraction values at the cutoff layer.
///
/// The intermediate values are absorbed into the transcript BEFORE the
/// root values, ensuring the prover commits before seeing challenges.
#[allow(clippy::arithmetic_side_effects, clippy::too_many_arguments)]
fn hybrid_batched_gkr_fraction_prove<F>(
    transcript: &mut impl Transcript,
    all_layers: &[Vec<FractionLayer<F>>], // L trees, each [leaves, ..., root]
    cutoff: usize,
    field_cfg: &F::Config,
) -> HybridBatchedGkrFractionProveResult<F>
where
    F: InnerTransparentField + FromPrimitiveWithConfig + Send + Sync,
    F::Inner: ConstTranscribable + Zero + Default + Send + Sync,
    F::Config: Sync,
{
    let num_trees = all_layers.len(); // L
    let d = all_layers[0].len() - 1; // number of GKR levels
    let c = cutoff.min(d); // actual cutoff (can't exceed tree depth)

    // Collect per-tree roots.
    let roots_p: Vec<F> = all_layers
        .iter()
        .map(|layers| layers.last().expect("tree non-empty").p[0].clone())
        .collect();
    let roots_q: Vec<F> = all_layers
        .iter()
        .map(|layers| layers.last().expect("tree non-empty").q[0].clone())
        .collect();

    // Extract intermediate values at the cutoff layer.
    // In the layers array: layers[0] = leaves (layer d), layers[d] = root (layer 0).
    // GKR layer c (from root) = layers array index d - c.
    let cutoff_layer_idx = d - c;
    let cutoff_layer_size = 1usize << c;

    let sent_p: Vec<Vec<F>> = all_layers
        .iter()
        .map(|layers| {
            let layer = &layers[cutoff_layer_idx];
            layer.p[..cutoff_layer_size].to_vec()
        })
        .collect();
    let sent_q: Vec<Vec<F>> = all_layers
        .iter()
        .map(|layers| {
            let layer = &layers[cutoff_layer_idx];
            layer.q[..cutoff_layer_size].to_vec()
        })
        .collect();

    // ---- Transcript ordering for soundness ----
    // 1. Absorb intermediate values FIRST (commitment before challenges)
    let mut buf = vec![0u8; F::Inner::NUM_BYTES];
    for ell in 0..num_trees {
        transcript.absorb_random_field_slice(&sent_p[ell], &mut buf);
        transcript.absorb_random_field_slice(&sent_q[ell], &mut buf);
    }

    // 2. Absorb root values.
    for ell in 0..num_trees {
        transcript.absorb_random_field(&roots_p[ell], &mut buf);
        transcript.absorb_random_field(&roots_q[ell], &mut buf);
    }

    if c == 0 {
        // No GKR layers to run: just return the roots + intermediate values.
        return HybridBatchedGkrFractionProveResult {
            proof: HybridBatchedGkrFractionProof {
                roots_p,
                roots_q,
                layer_proofs: vec![],
                cutoff: c,
                sent_p,
                sent_q,
            },
            eval_point: vec![],
        };
    }

    // 3. Run GKR layers 0..c-1 (from root).
    let mut layer_proofs = Vec::with_capacity(c);
    let inner_zero = F::zero_with_cfg(field_cfg).inner().clone();

    let mut v_ps: Vec<F> = roots_p.clone();
    let mut v_qs: Vec<F> = roots_q.clone();
    let mut r_k: Vec<F> = Vec::new();

    for round in 0..c {
        let k = round;
        let half = 1usize << k;

        // Child layer for this GKR round.
        let child_layer_idx = d - (round + 1);
        let per_tree: Vec<(&[F], &[F], &[F], &[F])> = all_layers
            .iter()
            .map(|layers| {
                let cl = &layers[child_layer_idx];
                (
                    &cl.p[..half],
                    &cl.p[half..],
                    &cl.q[..half],
                    &cl.q[half..],
                )
            })
            .collect();

        let alpha: F = transcript.get_field_challenge(field_cfg);
        let delta: F = transcript.get_field_challenge(field_cfg);

        let one = F::one_with_cfg(field_cfg);
        let mut delta_powers = Vec::with_capacity(num_trees);
        let mut dp = one.clone();
        for _ in 0..num_trees {
            delta_powers.push(dp.clone());
            dp *= &delta;
        }

        if k == 0 {
            // Round 0: direct algebraic check per tree.
            let mut p_lefts = Vec::with_capacity(num_trees);
            let mut p_rights = Vec::with_capacity(num_trees);
            let mut q_lefts = Vec::with_capacity(num_trees);
            let mut q_rights = Vec::with_capacity(num_trees);

            for ell in 0..num_trees {
                let (pl_vals, pr_vals, ql_vals, qr_vals) = per_tree[ell];
                let pl = pl_vals[0].clone();
                let pr = pr_vals[0].clone();
                let ql = ql_vals[0].clone();
                let qr = qr_vals[0].clone();

                transcript.absorb_random_field(&pl, &mut buf);
                transcript.absorb_random_field(&pr, &mut buf);
                transcript.absorb_random_field(&ql, &mut buf);
                transcript.absorb_random_field(&qr, &mut buf);

                p_lefts.push(pl);
                p_rights.push(pr);
                q_lefts.push(ql);
                q_rights.push(qr);
            }

            layer_proofs.push(HybridBatchedGkrLayerProof {
                sumcheck_proof: None,
                p_lefts: p_lefts.clone(),
                p_rights: p_rights.clone(),
                q_lefts: q_lefts.clone(),
                q_rights: q_rights.clone(),
            });

            let lambda: F = transcript.get_field_challenge(field_cfg);
            let one_minus_lambda = one.clone() - &lambda;
            for ell in 0..num_trees {
                v_ps[ell] = one_minus_lambda.clone() * &p_lefts[ell]
                    + &(lambda.clone() * &p_rights[ell]);
                v_qs[ell] = one_minus_lambda.clone() * &q_lefts[ell]
                    + &(lambda.clone() * &q_rights[ell]);
            }
            r_k = vec![lambda];
        } else {
            // Round k ≥ 1: batched sumcheck over k variables.
            let eq_r = build_eq_x_r_inner(&r_k, field_cfg)
                .expect("eq polynomial construction should succeed");

            let mk_mle = |data: &[F]| -> DenseMultilinearExtension<F::Inner> {
                DenseMultilinearExtension::from_evaluations_vec(
                    k,
                    data.iter().map(|x| x.inner().clone()).collect(),
                    inner_zero.clone(),
                )
            };

            let mut mles = Vec::with_capacity(1 + 4 * num_trees);
            mles.push(eq_r);
            for ell in 0..num_trees {
                let (pl_vals, pr_vals, ql_vals, qr_vals) = per_tree[ell];
                mles.push(mk_mle(pl_vals));
                mles.push(mk_mle(ql_vals));
                mles.push(mk_mle(pr_vals));
                mles.push(mk_mle(qr_vals));
            }

            let alpha_clone = alpha.clone();
            let delta_powers_clone = delta_powers.clone();
            let nt = num_trees;
            let comb_fn = move |vals: &[F]| -> F {
                let eq_val = &vals[0];
                let mut acc = vals[0].clone() - eq_val; // zero
                for ell in 0..nt {
                    let base = 1 + 4 * ell;
                    let pl_val = &vals[base];
                    let ql_val = &vals[base + 1];
                    let pr_val = &vals[base + 2];
                    let qr_val = &vals[base + 3];
                    let pl_plus_alpha_ql =
                        pl_val.clone() + &(alpha_clone.clone() * ql_val);
                    let inner = pl_plus_alpha_ql * qr_val + &(pr_val.clone() * ql_val);
                    acc += &(delta_powers_clone[ell].clone() * &inner);
                }
                eq_val.clone() * &acc
            };

            let (sumcheck_proof, sumcheck_prover_state) =
                MLSumcheck::prove_as_subprotocol(
                    transcript,
                    mles,
                    k,
                    3,
                    comb_fn,
                    field_cfg,
                );

            let s = &sumcheck_prover_state.randomness;
            let last_r = s.last().expect("sumcheck should have ≥1 challenge");
            let one_minus_last = F::one_with_cfg(field_cfg) - last_r;

            let interp_mle =
                |mle: &DenseMultilinearExtension<F::Inner>| -> F {
                    debug_assert_eq!(mle.num_vars, 1);
                    let v0 = F::new_unchecked_with_cfg(mle[0].clone(), field_cfg);
                    let v1 = F::new_unchecked_with_cfg(mle[1].clone(), field_cfg);
                    one_minus_last.clone() * &v0 + &(last_r.clone() * &v1)
                };

            let mut p_lefts = Vec::with_capacity(num_trees);
            let mut p_rights = Vec::with_capacity(num_trees);
            let mut q_lefts = Vec::with_capacity(num_trees);
            let mut q_rights = Vec::with_capacity(num_trees);

            for ell in 0..num_trees {
                let base = 1 + 4 * ell;
                let pl = interp_mle(&sumcheck_prover_state.mles[base]);
                let ql = interp_mle(&sumcheck_prover_state.mles[base + 1]);
                let pr = interp_mle(&sumcheck_prover_state.mles[base + 2]);
                let qr = interp_mle(&sumcheck_prover_state.mles[base + 3]);

                transcript.absorb_random_field(&pl, &mut buf);
                transcript.absorb_random_field(&pr, &mut buf);
                transcript.absorb_random_field(&ql, &mut buf);
                transcript.absorb_random_field(&qr, &mut buf);

                p_lefts.push(pl);
                p_rights.push(pr);
                q_lefts.push(ql);
                q_rights.push(qr);
            }

            layer_proofs.push(HybridBatchedGkrLayerProof {
                sumcheck_proof: Some(sumcheck_proof),
                p_lefts: p_lefts.clone(),
                p_rights: p_rights.clone(),
                q_lefts: q_lefts.clone(),
                q_rights: q_rights.clone(),
            });

            let lambda: F = transcript.get_field_challenge(field_cfg);
            let one_minus_lambda = one.clone() - &lambda;
            for ell in 0..num_trees {
                v_ps[ell] = one_minus_lambda.clone() * &p_lefts[ell]
                    + &(lambda.clone() * &p_rights[ell]);
                v_qs[ell] = one_minus_lambda.clone() * &q_lefts[ell]
                    + &(lambda.clone() * &q_rights[ell]);
            }
            r_k = s.clone();
            r_k.push(lambda);
        }
    }

    HybridBatchedGkrFractionProveResult {
        proof: HybridBatchedGkrFractionProof {
            roots_p,
            roots_q,
            layer_proofs,
            cutoff: c,
            sent_p,
            sent_q,
        },
        eval_point: r_k,
    }
}

/// Result of the hybrid batched GKR verifier.
struct HybridBatchedGkrFractionVerifyResult<F> {
    point: Vec<F>,
    expected_ps: Vec<F>,
    expected_qs: Vec<F>,
}

/// Run the hybrid batched GKR verifier for L fraction trees.
///
/// Verifies the first `cutoff` GKR layers, then checks the MLE of the
/// sent intermediate values at the derived point.
#[allow(clippy::arithmetic_side_effects, clippy::too_many_arguments)]
fn hybrid_batched_gkr_fraction_verify<F>(
    transcript: &mut impl Transcript,
    proof: &HybridBatchedGkrFractionProof<F>,
    num_vars: usize,
    field_cfg: &F::Config,
) -> Result<HybridBatchedGkrFractionVerifyResult<F>, LookupError<F>>
where
    F: InnerTransparentField + FromPrimitiveWithConfig + Send + Sync,
    F::Inner: ConstTranscribable + Zero,
{
    let d = num_vars;
    let c = proof.cutoff.min(d);
    let num_trees = proof.roots_p.len();
    let one = F::one_with_cfg(field_cfg);

    let mut buf = vec![0u8; F::Inner::NUM_BYTES];

    // 1. Absorb intermediate values (must match prover ordering).
    for ell in 0..num_trees {
        transcript.absorb_random_field_slice(&proof.sent_p[ell], &mut buf);
        transcript.absorb_random_field_slice(&proof.sent_q[ell], &mut buf);
    }

    // 2. Absorb roots.
    for ell in 0..num_trees {
        transcript.absorb_random_field(&proof.roots_p[ell], &mut buf);
        transcript.absorb_random_field(&proof.roots_q[ell], &mut buf);
    }

    if c == 0 {
        // No GKR layers: directly check roots against sent values (which
        // should be the same as roots for a depth-0 cutoff).
        return Ok(HybridBatchedGkrFractionVerifyResult {
            point: vec![],
            expected_ps: proof.roots_p.clone(),
            expected_qs: proof.roots_q.clone(),
        });
    }

    if proof.layer_proofs.len() != c {
        return Err(LookupError::GkrLeafMismatch);
    }

    // 3. Run GKR verifier for c layers.
    let mut v_ps: Vec<F> = proof.roots_p.clone();
    let mut v_qs: Vec<F> = proof.roots_q.clone();
    let mut r_k: Vec<F> = Vec::new();

    for round in 0..c {
        let k = round;
        let layer_proof = &proof.layer_proofs[round];

        let alpha: F = transcript.get_field_challenge(field_cfg);
        let delta: F = transcript.get_field_challenge(field_cfg);

        let mut delta_powers = Vec::with_capacity(num_trees);
        let mut dp = one.clone();
        for _ in 0..num_trees {
            delta_powers.push(dp.clone());
            dp *= &delta;
        }

        if k == 0 {
            for ell in 0..num_trees {
                let pl = &layer_proof.p_lefts[ell];
                let pr = &layer_proof.p_rights[ell];
                let ql = &layer_proof.q_lefts[ell];
                let qr = &layer_proof.q_rights[ell];

                transcript.absorb_random_field(pl, &mut buf);
                transcript.absorb_random_field(pr, &mut buf);
                transcript.absorb_random_field(ql, &mut buf);
                transcript.absorb_random_field(qr, &mut buf);

                let lhs = v_ps[ell].clone() + &(alpha.clone() * &v_qs[ell]);
                let cross = pl.clone() * qr + &(pr.clone() * ql);
                let prod = ql.clone() * qr;
                let rhs = cross + &(alpha.clone() * &prod);

                if lhs != rhs {
                    return Err(LookupError::GkrLayer0Mismatch {
                        expected: lhs,
                        got: rhs,
                    });
                }
            }

            let lambda: F = transcript.get_field_challenge(field_cfg);
            let one_minus_lambda = one.clone() - &lambda;
            for ell in 0..num_trees {
                v_ps[ell] = one_minus_lambda.clone() * &layer_proof.p_lefts[ell]
                    + &(lambda.clone() * &layer_proof.p_rights[ell]);
                v_qs[ell] = one_minus_lambda.clone() * &layer_proof.q_lefts[ell]
                    + &(lambda.clone() * &layer_proof.q_rights[ell]);
            }
            r_k = vec![lambda];
        } else {
            let sumcheck_proof = layer_proof
                .sumcheck_proof
                .as_ref()
                .ok_or(LookupError::GkrLeafMismatch)?;

            // Batched claimed sum = Σ_ℓ δ^ℓ · (v_p[ℓ] + α · v_q[ℓ])
            let mut claimed_sum = F::zero_with_cfg(field_cfg);
            for ell in 0..num_trees {
                let term = v_ps[ell].clone() + &(alpha.clone() * &v_qs[ell]);
                claimed_sum += &(delta_powers[ell].clone() * &term);
            }
            if sumcheck_proof.claimed_sum != claimed_sum {
                return Err(LookupError::FinalEvaluationMismatch {
                    expected: claimed_sum,
                    got: sumcheck_proof.claimed_sum.clone(),
                });
            }

            let subclaim = MLSumcheck::verify_as_subprotocol(
                transcript,
                k,
                3,
                sumcheck_proof,
                field_cfg,
            )?;

            let s = &subclaim.point;
            let eq_val = zinc_poly::utils::eq_eval(s, &r_k, one.clone())?;
            let mut combined_inner = F::zero_with_cfg(field_cfg);

            for ell in 0..num_trees {
                let pl = &layer_proof.p_lefts[ell];
                let pr = &layer_proof.p_rights[ell];
                let ql = &layer_proof.q_lefts[ell];
                let qr = &layer_proof.q_rights[ell];

                transcript.absorb_random_field(pl, &mut buf);
                transcript.absorb_random_field(pr, &mut buf);
                transcript.absorb_random_field(ql, &mut buf);
                transcript.absorb_random_field(qr, &mut buf);

                let pl_plus_alpha_ql = pl.clone() + &(alpha.clone() * ql);
                let inner = pl_plus_alpha_ql * qr + &(pr.clone() * ql);
                combined_inner += &(delta_powers[ell].clone() * &inner);
            }

            let expected = eq_val * &combined_inner;
            if expected != subclaim.expected_evaluation {
                return Err(LookupError::FinalEvaluationMismatch {
                    expected: subclaim.expected_evaluation,
                    got: expected,
                });
            }

            let lambda: F = transcript.get_field_challenge(field_cfg);
            let one_minus_lambda = one.clone() - &lambda;
            for ell in 0..num_trees {
                v_ps[ell] = one_minus_lambda.clone() * &layer_proof.p_lefts[ell]
                    + &(lambda.clone() * &layer_proof.p_rights[ell]);
                v_qs[ell] = one_minus_lambda.clone() * &layer_proof.q_lefts[ell]
                    + &(lambda.clone() * &layer_proof.q_rights[ell]);
            }
            r_k = s.clone();
            r_k.push(lambda);
        }
    }

    // 4. Check MLE of sent intermediate values at r_c.
    // The sent p_c, q_c vectors should satisfy:
    //   p̃_c(r_c) = v_p[ℓ]  and  q̃_c(r_c) = v_q[ℓ]  for each tree ℓ.
    let eq_at_rc = build_eq_x_r_vec(&r_k, field_cfg)?;
    let cutoff_size = 1usize << c;

    for ell in 0..num_trees {
        // Evaluate MLE of sent_p[ell] at r_k.
        let mut p_eval = F::zero_with_cfg(field_cfg);
        let mut q_eval = F::zero_with_cfg(field_cfg);
        for j in 0..cutoff_size {
            p_eval += &(eq_at_rc[j].clone() * &proof.sent_p[ell][j]);
            q_eval += &(eq_at_rc[j].clone() * &proof.sent_q[ell][j]);
        }

        if p_eval != v_ps[ell] {
            return Err(LookupError::GkrLeafMismatch);
        }
        if q_eval != v_qs[ell] {
            return Err(LookupError::GkrLeafMismatch);
        }
    }

    Ok(HybridBatchedGkrFractionVerifyResult {
        point: r_k,
        expected_ps: v_ps,
        expected_qs: v_qs,
    })
}

// ---------------------------------------------------------------------------
// Full hybrid protocol
// ---------------------------------------------------------------------------

/// The hybrid GKR Batched Decomposition + LogUp protocol.
pub struct HybridGkrBatchedDecompLogupProtocol<F: InnerTransparentField>(PhantomData<F>);

impl<F: InnerTransparentField + FromPrimitiveWithConfig + Send + Sync>
    HybridGkrBatchedDecompLogupProtocol<F>
{
    /// Prover for the hybrid GKR batched decomposition + LogUp protocol.
    ///
    /// Runs `cutoff` layers of GKR for the witness trees, then sends the
    /// intermediate fraction values. The table GKR runs fully (small depth).
    #[allow(clippy::arithmetic_side_effects, clippy::too_many_arguments)]
    pub fn prove_as_subprotocol(
        transcript: &mut impl Transcript,
        instance: &BatchedDecompLookupInstance<F>,
        cutoff: usize,
        field_cfg: &F::Config,
    ) -> Result<
        (HybridGkrBatchedDecompLogupProof<F>, HybridGkrProverState<F>),
        LookupError<F>,
    >
    where
        F::Inner: ConstTranscribable + Zero + Default + Send + Sync,
        F::Config: Sync,
    {
        let witnesses = &instance.witnesses;
        let subtable = &instance.subtable;
        let all_chunks = &instance.chunks;

        let num_lookups = witnesses.len();
        let num_chunks = instance.shifts.len();
        let witness_len = witnesses[0].len();
        let table_len = subtable.len();

        let zero = F::zero_with_cfg(field_cfg);
        let one = F::one_with_cfg(field_cfg);
        let mut buf = vec![0u8; F::Inner::NUM_BYTES];

        // ---- Step 1: Compute aggregated multiplicities ----
        let table_index = build_table_index(subtable);
        let all_chunk_multiplicities: Vec<Vec<Vec<F>>> = cfg_iter!(all_chunks)
            .map(|lookup_chunks| {
                lookup_chunks
                    .iter()
                    .map(|chunk| {
                        compute_multiplicities_with_index(
                            chunk, &table_index, table_len, field_cfg,
                        )
                        .ok_or(LookupError::WitnessNotInTable)
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<_, _>>()?;

        let all_aggregated_multiplicities: Vec<Vec<F>> = all_chunk_multiplicities
            .iter()
            .map(|lookup_mults| {
                let mut agg = vec![zero.clone(); table_len];
                for m in lookup_mults {
                    for (a, mk) in agg.iter_mut().zip(m.iter()) {
                        *a += mk;
                    }
                }
                agg
            })
            .collect();

        // ---- Step 2: Absorb aggregated multiplicities ----
        for agg in &all_aggregated_multiplicities {
            transcript.absorb_random_field_slice(agg, &mut buf);
        }

        // ---- Step 3: Challenges β and α ----
        let beta: F = transcript.get_field_challenge(field_cfg);
        let alpha: F = transcript.get_field_challenge(field_cfg);

        let mut alpha_powers = Vec::with_capacity(num_lookups);
        let mut ap = one.clone();
        for _ in 0..num_lookups {
            alpha_powers.push(ap.clone());
            ap *= &alpha;
        }

        // ---- Step 4: Build witness fraction trees ----
        let per_lookup_leaves = num_chunks * witness_len;
        let w_num_vars = zinc_utils::log2(per_lookup_leaves.next_power_of_two()) as usize;
        let w_size = 1usize << w_num_vars;

        let witness_trees: Vec<_> = (0..num_lookups)
            .map(|ell| {
                let mut leaf_p = Vec::with_capacity(w_size);
                let mut leaf_q = Vec::with_capacity(w_size);
                for k in 0..num_chunks {
                    for i in 0..witness_len {
                        leaf_p.push(one.clone());
                        leaf_q.push(beta.clone() - &all_chunks[ell][k][i]);
                    }
                }
                leaf_p.resize(w_size, zero.clone());
                leaf_q.resize(w_size, one.clone());
                build_fraction_tree(leaf_p, leaf_q)
            })
            .collect();

        // ---- Step 5: Build table fraction tree ----
        let t_num_vars = zinc_utils::log2(table_len.next_power_of_two()) as usize;
        let t_size = 1usize << t_num_vars;

        let (mut t_leaf_p, mut t_leaf_q): (Vec<F>, Vec<F>) =
            cfg_into_iter!(0..table_len, 256)
                .map(|j| {
                    let mut combined_mult = zero.clone();
                    for ell in 0..num_lookups {
                        combined_mult += &(alpha_powers[ell].clone()
                            * &all_aggregated_multiplicities[ell][j]);
                    }
                    (combined_mult, beta.clone() - &subtable[j])
                })
                .unzip();
        t_leaf_p.resize(t_size, zero.clone());
        t_leaf_q.resize(t_size, one.clone());

        let table_tree = build_fraction_tree(t_leaf_p, t_leaf_q);

        // ---- Step 6: Hybrid GKR for witness trees ----
        let witness_result =
            hybrid_batched_gkr_fraction_prove(transcript, &witness_trees, cutoff, field_cfg);
        let witness_eval_point = witness_result.eval_point;

        // ---- Step 7: Full GKR for table tree (small) ----
        let (table_gkr, table_eval_point) =
            gkr_fraction_prove(transcript, &table_tree, field_cfg);

        // ---- Step 8: Cross-check (debug) ----
        debug_assert!({
            let witness_gkr = &witness_result.proof;
            let q_w_product: F = witness_gkr.roots_q.iter().cloned()
                .fold(one.clone(), |acc, q| acc * &q);
            let mut lhs = zero.clone();
            for ell in 0..num_lookups {
                let mut others_q = one.clone();
                for j in 0..num_lookups {
                    if j != ell {
                        others_q *= &witness_gkr.roots_q[j];
                    }
                }
                lhs += &(alpha_powers[ell].clone() * &witness_gkr.roots_p[ell] * &others_q);
            }
            lhs *= &table_gkr.root_q;
            let rhs = table_gkr.root_p.clone() * &q_w_product;
            lhs == rhs
        });

        Ok((
            HybridGkrBatchedDecompLogupProof {
                aggregated_multiplicities: all_aggregated_multiplicities,
                witness_gkr: witness_result.proof,
                table_gkr,
            },
            HybridGkrProverState {
                witness_eval_point,
                table_eval_point,
                witness_num_vars: w_num_vars,
                table_num_vars: t_num_vars,
                alpha: alpha.clone(),
                beta: beta.clone(),
                num_lookups,
                num_chunks,
                witness_len,
                cutoff,
            },
        ))
    }

    /// Verifier for the hybrid GKR batched decomposition + LogUp protocol.
    #[allow(clippy::arithmetic_side_effects, clippy::too_many_arguments)]
    pub fn verify_as_subprotocol(
        transcript: &mut impl Transcript,
        proof: &HybridGkrBatchedDecompLogupProof<F>,
        subtable: &[F],
        shifts: &[F],
        num_lookups: usize,
        witness_len: usize,
        field_cfg: &F::Config,
    ) -> Result<HybridGkrVerifierSubClaim<F>, LookupError<F>>
    where
        F::Inner: ConstTranscribable + Zero + Default + Send + Sync,
        F::Config: Sync,
    {
        let zero = F::zero_with_cfg(field_cfg);
        let one = F::one_with_cfg(field_cfg);
        let mut buf = vec![0u8; F::Inner::NUM_BYTES];

        let num_chunks = shifts.len();
        let table_len = subtable.len();
        let all_agg_mults = &proof.aggregated_multiplicities;

        // ---- Step 1: Absorb aggregated multiplicities ----
        for agg in all_agg_mults {
            transcript.absorb_random_field_slice(agg, &mut buf);
        }

        // ---- Step 2: Challenges β and α ----
        let beta: F = transcript.get_field_challenge(field_cfg);
        let alpha: F = transcript.get_field_challenge(field_cfg);

        let mut alpha_powers = Vec::with_capacity(num_lookups);
        let mut ap = one.clone();
        for _ in 0..num_lookups {
            alpha_powers.push(ap.clone());
            ap *= &alpha;
        }

        // ---- Step 3: Compute tree dimensions ----
        let per_lookup_leaves = num_chunks * witness_len;
        let w_num_vars = zinc_utils::log2(per_lookup_leaves.next_power_of_two()) as usize;
        let t_num_vars = zinc_utils::log2(table_len.next_power_of_two()) as usize;

        // ---- Step 4: Verify hybrid witness GKR ----
        let witness_result = hybrid_batched_gkr_fraction_verify(
            transcript,
            &proof.witness_gkr,
            w_num_vars,
            field_cfg,
        )?;

        // ---- Step 5: Verify full table GKR ----
        let table_result = gkr_fraction_verify(
            transcript,
            &proof.table_gkr,
            t_num_vars,
            field_cfg,
        )?;

        // ---- Step 6: Cross-check roots ----
        {
            let q_w_product: F = proof.witness_gkr.roots_q.iter().cloned()
                .fold(one.clone(), |acc, q| acc * &q);
            let mut lhs = zero.clone();
            for ell in 0..num_lookups {
                let mut others_q = one.clone();
                for j in 0..num_lookups {
                    if j != ell {
                        others_q *= &proof.witness_gkr.roots_q[j];
                    }
                }
                lhs += &(alpha_powers[ell].clone() * &proof.witness_gkr.roots_p[ell] * &others_q);
            }
            lhs *= &proof.table_gkr.root_q;
            let rhs = proof.table_gkr.root_p.clone() * &q_w_product;
            if lhs != rhs {
                return Err(LookupError::GkrRootMismatch);
            }
        }

        // ---- Step 7: Verify table-side leaf claims ----
        if table_result.point.is_empty() {
            let expected_p = {
                let mut val = zero.clone();
                for ell in 0..num_lookups {
                    val += &(alpha_powers[ell].clone() * &all_agg_mults[ell][0]);
                }
                val
            };
            let expected_q = beta.clone() - &subtable[0];
            if expected_p != table_result.expected_p || expected_q != table_result.expected_q {
                return Err(LookupError::GkrLeafMismatch);
            }
        } else {
            let eq_at_t = build_eq_x_r_vec(&table_result.point, field_cfg)?;
            let mut p_eval = zero.clone();
            for j in 0..table_len {
                let mut combined_mult = zero.clone();
                for ell in 0..num_lookups {
                    combined_mult += &(alpha_powers[ell].clone() * &all_agg_mults[ell][j]);
                }
                p_eval += &(combined_mult * &eq_at_t[j]);
            }
            let mut q_eval = zero.clone();
            for j in 0..table_len {
                q_eval += &((beta.clone() - &subtable[j]) * &eq_at_t[j]);
            }
            for j in table_len..eq_at_t.len() {
                q_eval += &eq_at_t[j];
            }
            if p_eval != table_result.expected_p {
                return Err(LookupError::GkrLeafMismatch);
            }
            if q_eval != table_result.expected_q {
                return Err(LookupError::GkrLeafMismatch);
            }
        }

        // ---- Step 8: Verify multiplicity sums ----
        let expected_sum = F::from_with_cfg((num_chunks * witness_len) as u64, field_cfg);
        for ell in 0..num_lookups {
            let m_sum: F = all_agg_mults[ell].iter().cloned().fold(zero.clone(), |a, b| a + &b);
            if m_sum != expected_sum {
                return Err(LookupError::MultiplicitySumMismatch {
                    expected: (num_chunks * witness_len) as u64,
                    got: 0,
                });
            }
        }

        Ok(HybridGkrVerifierSubClaim {
            witness_eval_point: witness_result.point,
            expected_witness_p_evals: witness_result.expected_ps,
            expected_witness_q_evals: witness_result.expected_qs,
            table_eval_point: table_result.point,
            sent_p: proof.witness_gkr.sent_p.clone(),
            sent_q: proof.witness_gkr.sent_q.clone(),
        })
    }
}

// ---------------------------------------------------------------------------
// Cost analysis helpers
// ---------------------------------------------------------------------------

/// Compute the proof size and verifier cost metrics for the hybrid
/// approach at various cutoff depths.
///
/// Returns a vector of `(cutoff, HybridCostMetrics)` pairs for each
/// cutoff from 0 to `max_cutoff` (inclusive).
pub fn analyze_hybrid_costs(
    num_lookups: usize,    // L
    num_chunks: usize,     // K
    witness_len: usize,    // W
    table_len: usize,      // T
    fe_bytes: usize,       // field element byte size
) -> Vec<HybridCostMetrics> {
    let per_lookup_leaves = num_chunks * witness_len;
    let d_w = zinc_utils::log2(per_lookup_leaves.next_power_of_two()) as usize;
    let d_t = zinc_utils::log2(table_len.next_power_of_two()) as usize;

    let l = num_lookups;

    // Full GKR baseline.
    let _full_gkr_witness = full_gkr_cost(d_w, l, fe_bytes);
    let full_gkr_table = full_gkr_cost_single(d_t, fe_bytes);

    // Classic baseline.
    let classic_witness_fe = l * num_chunks * witness_len  // inverse witnesses
        + l * table_len                                     // aggregated multiplicities
        + table_len;                                        // inverse table
    let classic_witness_bytes = classic_witness_fe * fe_bytes;

    let mut results = Vec::with_capacity(d_w + 1);

    for c in 0..=d_w {
        // ── Top half: GKR layers 0..c-1 ─────────────────────────────
        let top_layers_fe = gkr_partial_cost_fe(c, l);
        let top_sc_msgs_fe = gkr_partial_sumcheck_msgs_fe(c);

        // Sent intermediate values: 2L × 2^c.
        let sent_fe = 2 * l * (1usize << c);

        // ── Bottom half: fresh GKR of depth d-c ─────────────────────
        // This is the KEY optimization: instead of continuing layers c..d-1
        // (with escalating sumcheck sizes k=c,c+1,...,d-1), we run a
        // FRESH GKR of depth d-c, where sumchecks have 1,2,...,d-c-1 vars.
        let bottom_depth = d_w.saturating_sub(c);
        let bottom_layers_fe = gkr_partial_cost_fe(bottom_depth, l);
        let bottom_sc_msgs_fe = gkr_partial_sumcheck_msgs_fe(bottom_depth);
        let bottom_roots_fe = 2 * l; // roots of the bottom GKR

        // ── Full GKR cost (for comparison) ──────────────────────────
        let full_gkr_layers_fe = gkr_partial_cost_fe(d_w, l);
        let full_gkr_sc_msgs_fe = gkr_partial_sumcheck_msgs_fe(d_w);
        let full_gkr_witness_fe = 2 * l + full_gkr_layers_fe + full_gkr_sc_msgs_fe;

        // ── Continuing-GKR cost (layers c..d-1, NOT fresh) ──────────
        let continuing_gkr_sc_msgs_fe = gkr_remaining_sumcheck_msgs_fe(c, d_w);
        let continuing_gkr_layers_fe = gkr_remaining_cost_fe(c, d_w, l);

        // ── Hybrid-only (top + intermediate, no bottom verification) ─
        let hybrid_top_only_fe = 2 * l + top_layers_fe + top_sc_msgs_fe + sent_fe;

        // ── Hybrid + fresh bottom GKR ───────────────────────────────
        let hybrid_full_fe = hybrid_top_only_fe
            + bottom_roots_fe + bottom_layers_fe + bottom_sc_msgs_fe;

        // ── Sumcheck rounds ─────────────────────────────────────────
        let top_sc_rounds: usize = (1..c).sum();
        let bottom_sc_rounds: usize = (1..bottom_depth).sum();
        let full_sc_rounds: usize = (1..d_w).sum();
        let continuing_sc_rounds: usize = (c..d_w).sum();
        let hybrid_total_sc_rounds = top_sc_rounds + bottom_sc_rounds;
        let sc_rounds_saved = full_sc_rounds.saturating_sub(hybrid_total_sc_rounds);

        // MLE evaluation cost at cutoff.
        let mle_eval_ops = l * (1usize << c);

        results.push(HybridCostMetrics {
            cutoff: c,
            tree_depth: d_w,
            num_lookups: l,
            // Proof sizes (bytes).
            hybrid_top_only_proof_bytes: hybrid_top_only_fe * fe_bytes,
            hybrid_full_proof_bytes: hybrid_full_fe * fe_bytes,
            full_gkr_witness_proof_bytes: full_gkr_witness_fe * fe_bytes,
            classic_witness_proof_bytes: classic_witness_bytes,
            sent_intermediate_bytes: sent_fe * fe_bytes,
            sent_intermediate_fe: sent_fe,
            // Verifier sumcheck rounds.
            top_sc_rounds,
            bottom_sc_rounds,
            hybrid_total_sc_rounds,
            full_sc_rounds,
            continuing_sc_rounds,
            sc_rounds_saved,
            // Per-layer evaluation checks.
            hybrid_per_layer_evals: (c + bottom_depth) * 4 * l,
            full_per_layer_evals: d_w * 4 * l,
            mle_eval_ops,
            // Table metrics (unchanged).
            table_depth: d_t,
            table_gkr_proof_bytes: full_gkr_table * fe_bytes,
        });
    }

    results
}

/// Cost metrics for one cutoff depth.
#[derive(Clone, Debug)]
pub struct HybridCostMetrics {
    /// Cutoff depth from root.
    pub cutoff: usize,
    /// Full tree depth d_w.
    pub tree_depth: usize,
    /// Number of batched trees L.
    pub num_lookups: usize,
    /// Hybrid top-only proof size (no bottom verification).
    pub hybrid_top_only_proof_bytes: usize,
    /// Hybrid full proof size (top + intermediate + bottom GKR).
    pub hybrid_full_proof_bytes: usize,
    /// Full GKR witness proof size in bytes.
    pub full_gkr_witness_proof_bytes: usize,
    /// Classic witness proof size in bytes.
    pub classic_witness_proof_bytes: usize,
    /// Bytes for the intermediate values alone.
    pub sent_intermediate_bytes: usize,
    /// Field elements for intermediate values.
    pub sent_intermediate_fe: usize,
    /// Sumcheck rounds in top half.
    pub top_sc_rounds: usize,
    /// Sumcheck rounds in bottom half (fresh GKR).
    pub bottom_sc_rounds: usize,
    /// Total hybrid sumcheck rounds (top + bottom).
    pub hybrid_total_sc_rounds: usize,
    /// Sumcheck rounds in full GKR verifier.
    pub full_sc_rounds: usize,
    /// Sumcheck rounds if continuing GKR from cutoff (not fresh).
    pub continuing_sc_rounds: usize,
    /// Sumcheck rounds saved by hybrid (full − hybrid total).
    pub sc_rounds_saved: usize,
    /// Per-layer evaluation checks in hybrid.
    pub hybrid_per_layer_evals: usize,
    /// Per-layer evaluation checks in full GKR.
    pub full_per_layer_evals: usize,
    /// MLE evaluation operations at cutoff.
    pub mle_eval_ops: usize,
    /// Table tree depth.
    pub table_depth: usize,
    /// Table GKR proof size in bytes.
    pub table_gkr_proof_bytes: usize,
}

/// Total field elements in a full batched GKR fraction proof (L trees, depth d).
fn full_gkr_cost(d: usize, l: usize, _fe_bytes: usize) -> usize {
    let roots = 2 * l;
    let layer_evals: usize = d * 4 * l;
    // Sumcheck messages: layer k has k rounds, each sending (degree-1) = 2 tail evaluations.
    // Actually degree 3 → 3 evaluations per round (but one is derivable, so 2 tail).
    let sc_msgs: usize = (1..d).map(|k| k * 2).sum();
    let sc_claimed_sums = d.saturating_sub(1); // one claimed sum per sumcheck
    roots + layer_evals + sc_msgs + sc_claimed_sums
}

/// Total field elements in a single GKR fraction proof (1 tree, depth d).
fn full_gkr_cost_single(d: usize, _fe_bytes: usize) -> usize {
    let roots = 2;
    let layer_evals: usize = d * 4;
    let sc_msgs: usize = (1..d).map(|k| k * 2).sum();
    let sc_claimed_sums = d.saturating_sub(1);
    roots + layer_evals + sc_msgs + sc_claimed_sums
}

/// Field elements for GKR layer evaluations (layers 0..c).
fn gkr_partial_cost_fe(c: usize, l: usize) -> usize {
    c * 4 * l
}

/// Field elements for GKR sumcheck messages (layers 1..c).
fn gkr_partial_sumcheck_msgs_fe(c: usize) -> usize {
    let msgs: usize = (1..c).map(|k| k * 2).sum();
    let claimed_sums = c.saturating_sub(1);
    msgs + claimed_sums
}

/// Field elements for remaining GKR layer evaluations (layers c..d).
fn gkr_remaining_cost_fe(c: usize, d: usize, l: usize) -> usize {
    if c >= d { return 0; }
    (d - c) * 4 * l
}

/// Field elements for remaining GKR sumcheck messages (layers c..d).
fn gkr_remaining_sumcheck_msgs_fe(c: usize, d: usize) -> usize {
    if c >= d { return 0; }
    let msgs: usize = (c..d).map(|k| k * 2).sum();
    let claimed_sums = d - c;
    msgs + claimed_sums
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::structs::BatchedDecompLookupInstance;
    use super::super::tables::{bitpoly_shift, generate_bitpoly_table};
    use crypto_bigint::{U128, const_monty_params};
    use crypto_primitives::crypto_bigint_const_monty::ConstMontyField;
    use zinc_transcript::KeccakTranscript;

    const N: usize = 2;
    const_monty_params!(TestParams, U128, "00000000b933426489189cb5b47d567f");
    type F = ConstMontyField<TestParams, N>;

    fn make_lookup(
        indices: &[usize],
        subtable: &[F],
        chunk_width: usize,
        num_chunks: usize,
        shifts: &[F],
    ) -> (Vec<F>, Vec<Vec<F>>) {
        let mask = (1usize << chunk_width) - 1;
        let chunks: Vec<Vec<F>> = (0..num_chunks)
            .map(|k| {
                indices
                    .iter()
                    .map(|&idx| subtable[(idx >> (k * chunk_width)) & mask].clone())
                    .collect()
            })
            .collect();
        let witness: Vec<F> = (0..indices.len())
            .map(|i| {
                let mut val = F::from(0u32);
                for k in 0..num_chunks {
                    val += &(shifts[k].clone() * &chunks[k][i]);
                }
                val
            })
            .collect();
        (witness, chunks)
    }

    #[test]
    fn hybrid_gkr_batched_3_lookups_cutoff_1() {
        let a = F::from(3u32);
        let chunk_width = 2;
        let num_chunks = 2;
        let subtable = generate_bitpoly_table(chunk_width, &a, &());
        let shifts: Vec<F> = (0..num_chunks)
            .map(|k| bitpoly_shift(k * chunk_width, &a))
            .collect();

        let (w0, c0) = make_lookup(&[0, 3, 5, 15], &subtable, chunk_width, num_chunks, &shifts);
        let (w1, c1) = make_lookup(&[1, 2, 7, 10], &subtable, chunk_width, num_chunks, &shifts);
        let (w2, c2) = make_lookup(&[15, 15, 0, 8], &subtable, chunk_width, num_chunks, &shifts);

        let instance = BatchedDecompLookupInstance {
            witnesses: vec![w0, w1, w2],
            subtable: subtable.clone(),
            shifts: shifts.clone(),
            chunks: vec![c0, c1, c2],
        };

        // Test cutoffs from 1 to full depth.
        let per_lookup_leaves = num_chunks * 4; // K * W
        let depth = zinc_utils::log2(per_lookup_leaves.next_power_of_two()) as usize;

        for cutoff in 1..=depth {
            let mut pt = KeccakTranscript::new();
            let (proof, _state) =
                HybridGkrBatchedDecompLogupProtocol::<F>::prove_as_subprotocol(
                    &mut pt, &instance, cutoff, &(),
                )
                .expect("hybrid prover should succeed");

            let mut vt = KeccakTranscript::new();
            let _sub = HybridGkrBatchedDecompLogupProtocol::<F>::verify_as_subprotocol(
                &mut vt,
                &proof,
                &subtable,
                &shifts,
                3,
                4,
                &(),
            )
            .expect(&format!("hybrid verifier should accept at cutoff={cutoff}"));
        }
    }

    #[test]
    fn hybrid_gkr_batched_5_lookups_4_chunks() {
        let a = F::from(3u32);
        let chunk_width = 2;
        let num_chunks = 4;
        let subtable = generate_bitpoly_table(chunk_width, &a, &());
        let shifts: Vec<F> = (0..num_chunks)
            .map(|k| bitpoly_shift(k * chunk_width, &a))
            .collect();

        let index_sets: Vec<Vec<usize>> = vec![
            vec![0, 42, 127, 200, 255, 170, 85, 13],
            vec![1, 2, 3, 4, 5, 6, 7, 8],
            vec![100, 200, 50, 150, 250, 10, 90, 180],
            vec![255, 0, 128, 64, 32, 16, 8, 4],
            vec![17, 34, 68, 136, 119, 238, 221, 187],
        ];

        let mut witnesses = Vec::new();
        let mut all_chunks = Vec::new();
        for indices in &index_sets {
            let (w, c) = make_lookup(indices, &subtable, chunk_width, num_chunks, &shifts);
            witnesses.push(w);
            all_chunks.push(c);
        }

        let instance = BatchedDecompLookupInstance {
            witnesses,
            subtable: subtable.clone(),
            shifts: shifts.clone(),
            chunks: all_chunks,
        };

        let per_lookup_leaves = num_chunks * 8;
        let depth = zinc_utils::log2(per_lookup_leaves.next_power_of_two()) as usize;

        for cutoff in 1..=depth {
            let mut pt = KeccakTranscript::new();
            let (proof, _state) =
                HybridGkrBatchedDecompLogupProtocol::<F>::prove_as_subprotocol(
                    &mut pt, &instance, cutoff, &(),
                )
                .expect("hybrid prover should succeed");

            let mut vt = KeccakTranscript::new();
            let _sub = HybridGkrBatchedDecompLogupProtocol::<F>::verify_as_subprotocol(
                &mut vt,
                &proof,
                &subtable,
                &shifts,
                5,
                8,
                &(),
            )
            .expect(&format!("hybrid verifier should accept at cutoff={cutoff}"));
        }
    }

    #[test]
    fn cost_analysis_sha256_like() {
        // SHA-256 8x parameters: L=10, K=8, W=512, T=16
        let metrics = analyze_hybrid_costs(10, 8, 512, 16, 16);

        eprintln!("\n=== Hybrid GKR Cost Analysis (SHA-256 8x: L=10, K=8, W=512, T=16, d=12) ===");
        eprintln!("  Full GKR witness:  {} B", metrics[0].full_gkr_witness_proof_bytes);
        eprintln!("  Classic witness:   {} B", metrics[0].classic_witness_proof_bytes);
        eprintln!("  Full GKR SC rnds:  {}", metrics[0].full_sc_rounds);
        eprintln!();
        eprintln!("  KEY INSIGHT: By splitting at cutoff c, the bottom half runs as a");
        eprintln!("  fresh GKR of depth d-c (sumcheck rounds k=1..d-c-1) instead of");
        eprintln!("  continuing from layer c (rounds k=c..d-1). This saves c·(d-c) rounds.");
        eprintln!();
        eprintln!("  {:>3} | {:>8} | {:>8} | {:>5} {:>5} {:>5} | {:>5} | {:>6} | {:>8}",
            "c", "TopOnly", "Top+Bot", "TopSC", "BotSC", "Total", "Saved", "MLE", "Sent FE");
        eprintln!("  {:-<3}-+-{:-<8}-+-{:-<8}-+-{:-<5}-{:-<5}-{:-<5}-+-{:-<5}-+-{:-<6}-+-{:-<8}",
            "", "", "", "", "", "", "", "", "");
        for m in &metrics {
            eprintln!("  {:>3} | {:>6} B | {:>6} B | {:>5} {:>5} {:>5} | {:>5} | {:>6} | {:>8}",
                m.cutoff,
                m.hybrid_top_only_proof_bytes,
                m.hybrid_full_proof_bytes,
                m.top_sc_rounds,
                m.bottom_sc_rounds,
                m.hybrid_total_sc_rounds,
                m.sc_rounds_saved,
                m.mle_eval_ops,
                m.sent_intermediate_fe,
            );
        }
        eprintln!();
        eprintln!("  Observations:");
        eprintln!("  • Optimal cutoff at c=d/2=6: saves {} rounds ({:.0}% reduction)",
            metrics[6].sc_rounds_saved,
            metrics[6].sc_rounds_saved as f64 / metrics[6].full_sc_rounds as f64 * 100.0);
        eprintln!("  • Full hybrid (c=6) proof: {} B vs full GKR {} B ({:.1}x)",
            metrics[6].hybrid_full_proof_bytes,
            metrics[6].full_gkr_witness_proof_bytes,
            metrics[6].hybrid_full_proof_bytes as f64 / metrics[6].full_gkr_witness_proof_bytes as f64);
        eprintln!("  • Still {:.0}x smaller than classic ({} B)",
            metrics[6].classic_witness_proof_bytes as f64 / metrics[6].hybrid_full_proof_bytes as f64,
            metrics[6].classic_witness_proof_bytes);
    }
}
