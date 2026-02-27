//! GKR-based LogUp protocol implementation.
//!
//! Implements the LogUp PIOP using the GKR (Goldwasser-Kalai-Rothblum)
//! protocol for fractional sumchecks, as described in:
//!
//!   Papini & Haböck, "Improving logarithmic derivative lookups using GKR"
//!   <https://eprint.iacr.org/2023/1284>
//!
//! ## Key advantage over the standard LogUp
//!
//! The prover only needs to send the **multiplicity vector** `m` — no
//! inverse vectors `u`, `v` are required. This saves transmitting
//! `O(W + N)` field elements at the cost of `O(log²(max(W,N)))` extra
//! field elements in the GKR layer proofs.
//!
//! ## Protocol overview
//!
//! Given witness `w ∈ F_q^W` and table `T ∈ F_q^N`, the GKR LogUp
//! protocol proves `w_i ∈ {T_j}` for all `i` by verifying:
//!
//! ```text
//! Σ_i 1/(β − w_i) = Σ_j m_j/(β − T_j)
//! ```
//!
//! Each side is proven via a **GKR fractional sumcheck**: a binary tree
//! of fraction additions is verified layer-by-layer from root to leaves
//! using sumchecks. At the leaves, the verifier checks evaluations against
//! the known input polynomials.

use crypto_primitives::FromPrimitiveWithConfig;
use num_traits::Zero;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::marker::PhantomData;
use zinc_poly::{
    mle::DenseMultilinearExtension,
    utils::{build_eq_x_r_inner, build_eq_x_r_vec},
};
use zinc_transcript::traits::{ConstTranscribable, Transcript};
use zinc_utils::{cfg_into_iter, cfg_iter, inner_transparent_field::InnerTransparentField};

use crate::sumcheck::MLSumcheck;

use super::{
    structs::{
        GkrFractionProof, GkrLayerProof, GkrLogupProof, GkrLogupProverState,
        GkrLogupVerifierSubClaim, LookupError,
    },
    tables::compute_multiplicities,
};

// ---------------------------------------------------------------------------
// Fraction tree helpers
// ---------------------------------------------------------------------------

/// A single layer of the fraction tree, storing numerators and denominators.
#[derive(Clone, Debug)]
pub(super) struct FractionLayer<F> {
    pub(super) p: Vec<F>, // numerators
    pub(super) q: Vec<F>, // denominators
}

/// Build the full fraction tree bottom-up.
///
/// The tree has `d + 1` layers (layer `d` = leaves, layer 0 = root).
/// Returns layers `[layer_d, layer_{d-1}, ..., layer_0]`, i.e. indexed
/// from leaves to root.
///
/// Layer indexing: GKR layer `k` has `2^k` entries.
/// - Layer 0 (root): 1 entry
/// - Layer `d` (leaves): `2^d` entries
///
/// Transition from layer `k+1` to layer `k`:
/// ```text
/// p_k[i] = p_{k+1}[i] · q_{k+1}[i + 2^k] + p_{k+1}[i + 2^k] · q_{k+1}[i]
/// q_k[i] = q_{k+1}[i] · q_{k+1}[i + 2^k]
/// ```
#[allow(clippy::arithmetic_side_effects)]
pub(super) fn build_fraction_tree<F: InnerTransparentField + Send + Sync>(
    leaf_p: Vec<F>,
    leaf_q: Vec<F>,
) -> Vec<FractionLayer<F>>
where
    F::Config: Sync,
{
    let d = zinc_utils::log2(leaf_p.len()) as usize;
    debug_assert_eq!(leaf_p.len(), 1 << d);
    debug_assert_eq!(leaf_q.len(), 1 << d);

    // layers[0] = leaves (layer d), layers[d] = root (layer 0)
    let mut layers = Vec::with_capacity(d + 1);
    layers.push(FractionLayer {
        p: leaf_p,
        q: leaf_q,
    });

    for level in (0..d).rev() {
        // Going from GKR layer (level+1) to layer (level).
        // Layer (level+1) has 2^{level+1} entries. Layer (level) has 2^level entries.
        let child = layers.last().expect("tree is non-empty during construction");
        let half = 1usize << level;

        // Parallelize over the half pairs: entries are independent within each layer.
        // Use a threshold to avoid rayon overhead for very small layers (half < 64).
        let (parent_p, parent_q): (Vec<F>, Vec<F>) = if half >= 64 {
            cfg_into_iter!(0..half)
                .map(|i| {
                    let pl = &child.p[i];
                    let ql = &child.q[i];
                    let pr = &child.p[i + half];
                    let qr = &child.q[i + half];
                    // p_parent = p_left * q_right + p_right * q_left
                    // q_parent = q_left * q_right
                    (pl.clone() * qr + &(pr.clone() * ql), ql.clone() * qr)
                })
                .unzip()
        } else {
            (0..half)
                .map(|i| {
                    let pl = &child.p[i];
                    let ql = &child.q[i];
                    let pr = &child.p[i + half];
                    let qr = &child.q[i + half];
                    (pl.clone() * qr + &(pr.clone() * ql), ql.clone() * qr)
                })
                .unzip()
        };

        layers.push(FractionLayer {
            p: parent_p,
            q: parent_q,
        });
    }

    // layers is now [leaves, ..., root]
    layers
}

/// Evaluate a k-variable MLE (given as evaluations over {0,1}^k in
/// little-endian order) at a point in F^k.
///
/// Variables are fixed starting from x_0 (the least significant bit),
/// which corresponds to interleaving consecutive pairs.
#[allow(clippy::arithmetic_side_effects)]
pub(super) fn evaluate_mle_at<F: InnerTransparentField>(
    evals: &[F],
    point: &[F],
    field_cfg: &F::Config,
) -> F {
    let n = point.len();
    debug_assert_eq!(evals.len(), 1 << n);
    if n == 0 {
        return evals[0].clone();
    }

    // Iteratively reduce: fix variables one at a time in little-endian
    // order (x_0 first). At each step, fold consecutive pairs:
    //   next[j] = (1 - r_i) * current[2j] + r_i * current[2j + 1]
    let mut current = evals.to_vec();
    for r_i in point {
        let half = current.len() / 2;
        let one_minus_r = F::one_with_cfg(field_cfg) - r_i;
        let mut next = Vec::with_capacity(half);
        for j in 0..half {
            next.push(
                one_minus_r.clone() * &current[2 * j]
                    + &(r_i.clone() * &current[2 * j + 1]),
            );
        }
        current = next;
    }
    current[0].clone()
}

// ---------------------------------------------------------------------------
// GKR fractional sumcheck: prover + verifier
// ---------------------------------------------------------------------------

/// Run the GKR prover for a single fractional sumcheck.
///
/// Proves `Σ_{x ∈ {0,1}^d} p(x)/q(x) = root_p/root_q` and returns
/// a `GkrFractionProof`.
#[allow(clippy::arithmetic_side_effects, clippy::too_many_arguments)]
pub(super) fn gkr_fraction_prove<F>(
    transcript: &mut impl Transcript,
    layers: &[FractionLayer<F>], // [leaves, ..., root]
    field_cfg: &F::Config,
) -> GkrFractionProof<F>
where
    F: InnerTransparentField + FromPrimitiveWithConfig + Send + Sync,
    F::Inner: ConstTranscribable + Zero + Default + Send + Sync,
{
    let d = layers.len() - 1; // number of GKR levels (= log2(num_leaves))
    let root = layers.last().expect("tree is non-empty");
    let root_p = root.p[0].clone();
    let root_q = root.q[0].clone();

    // Absorb root values into transcript.
    let mut buf = vec![0u8; F::Inner::NUM_BYTES];
    transcript.absorb_random_field(&root_p, &mut buf);
    transcript.absorb_random_field(&root_q, &mut buf);

    if d == 0 {
        // Trivial case: single leaf = root.
        return GkrFractionProof {
            root_p,
            root_q,
            layer_proofs: vec![],
        };
    }

    let mut layer_proofs = Vec::with_capacity(d);
    let inner_zero = F::zero_with_cfg(field_cfg).inner().clone();

    // Current claim: (v_p, v_q) at point r_k for GKR layer k.
    let mut v_p = root_p.clone();
    let mut v_q = root_q.clone();
    let mut r_k: Vec<F> = Vec::new(); // starts empty (0 variables for root)

    for round in 0..d {
        // Going from GKR layer (d - 1 - round) to the layer below.
        // The child layer in our layers array: child_idx is such that
        // layers[child_idx] corresponds to GKR layer (d - round).
        // layers[0] = leaves (GKR layer d), layers[d] = root (GKR layer 0)
        // GKR layer k corresponds to layers[d - k].
        // Current GKR layer is k_current = round (starting from 0).
        // Next GKR layer is k_next = round + 1.
        // layers index for k_next: d - (round + 1)
        let child_layer_idx = d - (round + 1);
        let child_layer = &layers[child_layer_idx];

        let k = round; // current GKR layer has 2^k entries, k variables
        let half = 1usize << k; // number of entries in the current layer

        // Split child layer (2^{k+1} entries) into left/right halves.
        let p_left_vals = &child_layer.p[..half];
        let p_right_vals = &child_layer.p[half..];
        let q_left_vals = &child_layer.q[..half];
        let q_right_vals = &child_layer.q[half..];

        // Get batching challenge α.
        let alpha: F = transcript.get_field_challenge(field_cfg);

        if k == 0 {
            // Round 0: zero sumcheck variables — direct algebraic check.
            // p_left, p_right, q_left, q_right are single values.
            let pl = p_left_vals[0].clone();
            let pr = p_right_vals[0].clone();
            let ql = q_left_vals[0].clone();
            let qr = q_right_vals[0].clone();

            // Absorb the 4 evaluations.
            transcript.absorb_random_field(&pl, &mut buf);
            transcript.absorb_random_field(&pr, &mut buf);
            transcript.absorb_random_field(&ql, &mut buf);
            transcript.absorb_random_field(&qr, &mut buf);

            // Verifier will check:
            //   v_p + α·v_q == (pl·qr + pr·ql) + α·(ql·qr)
            debug_assert!({
                let lhs = v_p.clone() + &(alpha.clone() * &v_q);
                let cross = pl.clone() * &qr + &(pr.clone() * &ql);
                let prod = ql.clone() * &qr;
                let rhs = cross + &(alpha.clone() * &prod);
                lhs == rhs
            });

            layer_proofs.push(GkrLayerProof {
                sumcheck_proof: None,
                p_left: pl.clone(),
                p_right: pr.clone(),
                q_left: ql.clone(),
                q_right: qr.clone(),
            });

            // Sample λ and compute new claim at layer 1.
            let lambda: F = transcript.get_field_challenge(field_cfg);
            let one = F::one_with_cfg(field_cfg);
            let one_minus_lambda = one - &lambda;
            v_p = one_minus_lambda.clone() * &pl + &(lambda.clone() * &pr);
            v_q = one_minus_lambda * &ql + &(lambda.clone() * &qr);
            r_k = vec![lambda];
        } else {
            // Round k ≥ 1: sumcheck over k variables.

            // Build eq(·, r_k) MLE.
            let eq_r = build_eq_x_r_inner(&r_k, field_cfg)
                .expect("eq polynomial construction should succeed");

            // Build MLEs for the child-layer halves.
            let mk_mle = |data: &[F]| -> DenseMultilinearExtension<F::Inner> {
                DenseMultilinearExtension::from_evaluations_vec(
                    k,
                    data.iter().map(|x| x.inner().clone()).collect(),
                    inner_zero.clone(),
                )
            };

            let pl_mle = mk_mle(p_left_vals);
            let pr_mle = mk_mle(p_right_vals);
            let ql_mle = mk_mle(q_left_vals);
            let qr_mle = mk_mle(q_right_vals);

            let mles = vec![eq_r, pl_mle, ql_mle, pr_mle, qr_mle];

            // Combination function:
            //   eq(x, rk) · [pL·qR + pR·qL + α·qL·qR]
            let alpha_clone = alpha.clone();
            let comb_fn = move |vals: &[F]| -> F {
                let eq_val = &vals[0];
                let pl_val = &vals[1];
                let ql_val = &vals[2];
                let pr_val = &vals[3];
                let qr_val = &vals[4];

                let cross = pl_val.clone() * qr_val + &(pr_val.clone() * ql_val);
                let prod = ql_val.clone() * qr_val;
                let inner = cross + &(alpha_clone.clone() * &prod);
                eq_val.clone() * &inner
            };

            // Claimed sum = v_p + α · v_q
            // degree = 3 (eq is deg-1, cross/prod are deg-2)
            let (sumcheck_proof, sumcheck_prover_state) = MLSumcheck::prove_as_subprotocol(
                transcript,
                mles,
                k,
                3,
                comb_fn,
                field_cfg,
            );

            // Evaluate the child-layer halves at the subclaim point.
            let s = &sumcheck_prover_state.randomness;
            // Four independent MLE evaluations — run in parallel only when
            // each MLE is large enough to amortise rayon::join dispatch overhead.
            #[cfg(feature = "parallel")]
            let (pl_at_s, pr_at_s, ql_at_s, qr_at_s) = if half >= 128 {
                let ((pl, pr), (ql, qr)) = rayon::join(
                    || rayon::join(
                        || evaluate_mle_at(p_left_vals, s, field_cfg),
                        || evaluate_mle_at(p_right_vals, s, field_cfg),
                    ),
                    || rayon::join(
                        || evaluate_mle_at(q_left_vals, s, field_cfg),
                        || evaluate_mle_at(q_right_vals, s, field_cfg),
                    ),
                );
                (pl, pr, ql, qr)
            } else {
                (
                    evaluate_mle_at(p_left_vals, s, field_cfg),
                    evaluate_mle_at(p_right_vals, s, field_cfg),
                    evaluate_mle_at(q_left_vals, s, field_cfg),
                    evaluate_mle_at(q_right_vals, s, field_cfg),
                )
            };
            #[cfg(not(feature = "parallel"))]
            let (pl_at_s, pr_at_s, ql_at_s, qr_at_s) = (
                evaluate_mle_at(p_left_vals, s, field_cfg),
                evaluate_mle_at(p_right_vals, s, field_cfg),
                evaluate_mle_at(q_left_vals, s, field_cfg),
                evaluate_mle_at(q_right_vals, s, field_cfg),
            );

            // Absorb the 4 evaluations.
            transcript.absorb_random_field(&pl_at_s, &mut buf);
            transcript.absorb_random_field(&pr_at_s, &mut buf);
            transcript.absorb_random_field(&ql_at_s, &mut buf);
            transcript.absorb_random_field(&qr_at_s, &mut buf);

            layer_proofs.push(GkrLayerProof {
                sumcheck_proof: Some(sumcheck_proof),
                p_left: pl_at_s.clone(),
                p_right: pr_at_s.clone(),
                q_left: ql_at_s.clone(),
                q_right: qr_at_s.clone(),
            });

            // Sample λ and compute new claim.
            let lambda: F = transcript.get_field_challenge(field_cfg);
            let one = F::one_with_cfg(field_cfg);
            let one_minus_lambda = one - &lambda;
            v_p = one_minus_lambda.clone() * &pl_at_s + &(lambda.clone() * &pr_at_s);
            v_q = one_minus_lambda * &ql_at_s + &(lambda.clone() * &qr_at_s);
            // r_{k+1} = (s, λ) — append λ to the subclaim point.
            r_k = s.clone();
            r_k.push(lambda);
        }
    }

    GkrFractionProof {
        root_p,
        root_q,
        layer_proofs,
    }
}

/// Result of verifying a GKR fractional sumcheck.
///
/// Contains the final evaluation point at the leaf layer and the
/// expected evaluations of the numerator and denominator MLEs.
pub(super) struct GkrFractionVerifyResult<F> {
    /// The evaluation point at the leaf layer: `r ∈ F^d`.
    pub(super) point: Vec<F>,
    /// Expected evaluation of the numerator MLE: `p̃(r)`.
    pub(super) expected_p: F,
    /// Expected evaluation of the denominator MLE: `q̃(r)`.
    pub(super) expected_q: F,
}

/// Run the GKR verifier for a single fractional sumcheck.
///
/// Returns the leaf-layer evaluation point and expected (p, q) values.
#[allow(clippy::arithmetic_side_effects, clippy::too_many_arguments)]
pub(super) fn gkr_fraction_verify<F>(
    transcript: &mut impl Transcript,
    proof: &GkrFractionProof<F>,
    num_vars: usize, // d = log2(num_leaves)
    field_cfg: &F::Config,
) -> Result<GkrFractionVerifyResult<F>, LookupError<F>>
where
    F: InnerTransparentField + FromPrimitiveWithConfig + Send + Sync,
    F::Inner: ConstTranscribable + Zero,
{
    let d = num_vars;
    let one = F::one_with_cfg(field_cfg);

    // Absorb root values.
    let mut buf = vec![0u8; F::Inner::NUM_BYTES];
    transcript.absorb_random_field(&proof.root_p, &mut buf);
    transcript.absorb_random_field(&proof.root_q, &mut buf);

    if d == 0 {
        return Ok(GkrFractionVerifyResult {
            point: vec![],
            expected_p: proof.root_p.clone(),
            expected_q: proof.root_q.clone(),
        });
    }

    if proof.layer_proofs.len() != d {
        return Err(LookupError::GkrLeafMismatch);
    }

    let mut v_p = proof.root_p.clone();
    let mut v_q = proof.root_q.clone();
    let mut r_k: Vec<F> = Vec::new();

    for round in 0..d {
        let k = round;
        let layer_proof = &proof.layer_proofs[round];

        // Get batching challenge α.
        let alpha: F = transcript.get_field_challenge(field_cfg);

        if k == 0 {
            // Round 0: direct check (0 sumcheck variables).
            let pl = &layer_proof.p_left;
            let pr = &layer_proof.p_right;
            let ql = &layer_proof.q_left;
            let qr = &layer_proof.q_right;

            // Absorb the 4 evaluations.
            transcript.absorb_random_field(pl, &mut buf);
            transcript.absorb_random_field(pr, &mut buf);
            transcript.absorb_random_field(ql, &mut buf);
            transcript.absorb_random_field(qr, &mut buf);

            // Check: v_p + α·v_q == (pl·qr + pr·ql) + α·(ql·qr)
            let lhs = v_p.clone() + &(alpha.clone() * &v_q);
            let cross = pl.clone() * qr + &(pr.clone() * ql);
            let prod = ql.clone() * qr;
            let rhs = cross + &(alpha * &prod);

            if lhs != rhs {
                return Err(LookupError::GkrLayer0Mismatch {
                    expected: lhs,
                    got: rhs,
                });
            }

            // Sample λ, compute new claim.
            let lambda: F = transcript.get_field_challenge(field_cfg);
            let one_minus_lambda = one.clone() - &lambda;
            v_p = one_minus_lambda.clone() * pl + &(lambda.clone() * pr);
            v_q = one_minus_lambda * ql + &(lambda.clone() * qr);
            r_k = vec![lambda];
        } else {
            // Round k ≥ 1: verify sumcheck.
            let sumcheck_proof = layer_proof
                .sumcheck_proof
                .as_ref()
                .ok_or(LookupError::GkrLeafMismatch)?;

            // Claimed sum = v_p + α · v_q
            // The sumcheck proof contains claimed_sum which should match.
            let claimed_sum = v_p.clone() + &(alpha.clone() * &v_q);
            if sumcheck_proof.claimed_sum != claimed_sum {
                return Err(LookupError::FinalEvaluationMismatch {
                    expected: claimed_sum,
                    got: sumcheck_proof.claimed_sum.clone(),
                });
            }

            let subclaim = MLSumcheck::verify_as_subprotocol(
                transcript,
                k,
                3, // degree
                sumcheck_proof,
                field_cfg,
            )?;

            // Verify the final evaluation at the subclaim point.
            let s = &subclaim.point;
            let pl = &layer_proof.p_left;
            let pr = &layer_proof.p_right;
            let ql = &layer_proof.q_left;
            let qr = &layer_proof.q_right;

            // Absorb the 4 evaluations.
            transcript.absorb_random_field(pl, &mut buf);
            transcript.absorb_random_field(pr, &mut buf);
            transcript.absorb_random_field(ql, &mut buf);
            transcript.absorb_random_field(qr, &mut buf);

            // Recompute the combination function at s.
            let eq_val = zinc_poly::utils::eq_eval(s, &r_k, one.clone())?;

            let cross = pl.clone() * qr + &(pr.clone() * ql);
            let prod = ql.clone() * qr;
            let inner = cross + &(alpha * &prod);
            let expected = eq_val * &inner;

            if expected != subclaim.expected_evaluation {
                return Err(LookupError::FinalEvaluationMismatch {
                    expected: subclaim.expected_evaluation,
                    got: expected,
                });
            }

            // Sample λ, compute new claim.
            let lambda: F = transcript.get_field_challenge(field_cfg);
            let one_minus_lambda = one.clone() - &lambda;
            v_p = one_minus_lambda.clone() * pl + &(lambda.clone() * pr);
            v_q = one_minus_lambda * ql + &(lambda.clone() * qr);
            r_k = s.clone();
            r_k.push(lambda);
        }
    }

    Ok(GkrFractionVerifyResult {
        point: r_k,
        expected_p: v_p,
        expected_q: v_q,
    })
}

// ---------------------------------------------------------------------------
// GKR LogUp protocol
// ---------------------------------------------------------------------------

/// The GKR LogUp protocol.
///
/// Proves the log-derivative lookup identity using GKR fractional
/// sumchecks instead of committing to inverse vectors.
pub struct GkrLogupProtocol<F: InnerTransparentField>(PhantomData<F>);

impl<F: InnerTransparentField + FromPrimitiveWithConfig + Send + Sync> GkrLogupProtocol<F> {
    /// Prover for the GKR LogUp protocol.
    ///
    /// # Arguments
    ///
    /// - `transcript`: Fiat-Shamir transcript.
    /// - `witness`: The witness vector as field elements (projected trace
    ///   column). Will be padded to the next power of two.
    /// - `table`: The lookup table entries. Must be a power of two.
    /// - `field_cfg`: Field configuration.
    ///
    /// # Returns
    ///
    /// `(GkrLogupProof, GkrLogupProverState)` on success, or a `LookupError`.
    #[allow(clippy::arithmetic_side_effects, clippy::too_many_arguments)]
    pub fn prove_as_subprotocol(
        transcript: &mut impl Transcript,
        witness: &[F],
        table: &[F],
        field_cfg: &F::Config,
    ) -> Result<(GkrLogupProof<F>, GkrLogupProverState<F>), LookupError<F>>
    where
        F::Inner: ConstTranscribable + Zero + Default + Send + Sync,
    {
        let witness_len = witness.len();
        let zero = F::zero_with_cfg(field_cfg);
        let one = F::one_with_cfg(field_cfg);

        // ---- Step 1: Compute multiplicities ----
        let multiplicities = compute_multiplicities(witness, table, field_cfg)
            .ok_or(LookupError::WitnessNotInTable)?;

        // ---- Step 2: Absorb multiplicities into transcript ----
        let mut buf = vec![0u8; F::Inner::NUM_BYTES];
        transcript.absorb_random_field_slice(&multiplicities, &mut buf);

        // ---- Step 3: Get challenge β ----
        let beta: F = transcript.get_field_challenge(field_cfg);

        // ---- Step 4: Build fraction trees ----
        let w_num_vars = zinc_utils::log2(witness_len.next_power_of_two()) as usize;
        let t_num_vars = zinc_utils::log2(table.len().next_power_of_two()) as usize;

        // Witness tree: leaves are (1, β − w_i), padded to 2^w_num_vars.
        let w_size = 1usize << w_num_vars;
        let mut w_leaf_p = vec![one.clone(); w_size];
        let mut w_leaf_q: Vec<F> = cfg_iter!(witness)
            .map(|w_i| beta.clone() - w_i)
            .collect();
        // Pad with (0, 1) — zero fraction, doesn't affect the sum.
        w_leaf_p.resize(w_size, zero.clone());
        w_leaf_q.resize(w_size, one.clone());

        let witness_tree = build_fraction_tree(w_leaf_p, w_leaf_q);

        // Table tree: leaves are (m_j, β − T_j), padded to 2^t_num_vars.
        let t_size = 1usize << t_num_vars;
        let mut t_leaf_p = multiplicities.clone();
        let mut t_leaf_q: Vec<F> = cfg_iter!(table)
            .map(|t_j| beta.clone() - t_j)
            .collect();
        // Pad with (0, 1).
        t_leaf_p.resize(t_size, zero.clone());
        t_leaf_q.resize(t_size, one.clone());

        let table_tree = build_fraction_tree(t_leaf_p, t_leaf_q);

        // ---- Step 5: GKR protocol for witness tree ----
        let witness_gkr = gkr_fraction_prove(transcript, &witness_tree, field_cfg);

        // ---- Step 6: GKR protocol for table tree ----
        let table_gkr = gkr_fraction_prove(transcript, &table_tree, field_cfg);

        // ---- Step 7: Verify root cross-check (prover-side debug) ----
        debug_assert!({
            let lhs = witness_gkr.root_p.clone() * &table_gkr.root_q;
            let rhs = table_gkr.root_p.clone() * &witness_gkr.root_q;
            lhs == rhs
        });

        // ---- Step 8: Verify multiplicity sum = W (prover-side debug) ----
        debug_assert!({
            let sum: F = multiplicities.iter().cloned().fold(zero, |a, b| a + &b);
            sum == F::from_with_cfg(witness_len as u64, field_cfg)
        });

        // Recover evaluation points from the GKR proofs.
        let w_eval_point = recover_gkr_eval_point(&witness_gkr, w_num_vars, field_cfg);
        let t_eval_point = recover_gkr_eval_point(&table_gkr, t_num_vars, field_cfg);

        Ok((
            GkrLogupProof {
                multiplicities,
                witness_gkr,
                table_gkr,
            },
            GkrLogupProverState {
                witness_eval_point: w_eval_point,
                table_eval_point: t_eval_point,
            },
        ))
    }

    /// Verifier for the GKR LogUp protocol.
    ///
    /// # Arguments
    ///
    /// - `transcript`: Fiat-Shamir transcript (must be in the same state
    ///   as the prover's at the start of the protocol).
    /// - `proof`: The `GkrLogupProof` received from the prover.
    /// - `table`: The lookup table entries.
    /// - `witness_len`: The length of the witness vector.
    /// - `field_cfg`: Field configuration.
    ///
    /// # Returns
    ///
    /// `GkrLogupVerifierSubClaim` on success, or a `LookupError`.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn verify_as_subprotocol(
        transcript: &mut impl Transcript,
        proof: &GkrLogupProof<F>,
        table: &[F],
        witness_len: usize,
        field_cfg: &F::Config,
    ) -> Result<GkrLogupVerifierSubClaim<F>, LookupError<F>>
    where
        F::Inner: ConstTranscribable + Zero,
    {
        let zero = F::zero_with_cfg(field_cfg);
        let one = F::one_with_cfg(field_cfg);

        // ---- Step 1: Absorb multiplicities ----
        let mut buf = vec![0u8; F::Inner::NUM_BYTES];
        transcript.absorb_random_field_slice(&proof.multiplicities, &mut buf);

        // ---- Step 2: Get challenge β ----
        let beta: F = transcript.get_field_challenge(field_cfg);

        let w_num_vars = zinc_utils::log2(witness_len.next_power_of_two()) as usize;
        let t_num_vars = zinc_utils::log2(table.len().next_power_of_two()) as usize;

        // ---- Step 3: Verify GKR for witness tree ----
        let witness_result = gkr_fraction_verify(
            transcript,
            &proof.witness_gkr,
            w_num_vars,
            field_cfg,
        )?;

        // ---- Step 4: Verify GKR for table tree ----
        let table_result = gkr_fraction_verify(
            transcript,
            &proof.table_gkr,
            t_num_vars,
            field_cfg,
        )?;

        // ---- Step 5: Cross-check roots ----
        // P_w · Q_t == P_t · Q_w   ⟺   P_w/Q_w == P_t/Q_t
        let lhs = proof.witness_gkr.root_p.clone() * &proof.table_gkr.root_q;
        let rhs = proof.table_gkr.root_p.clone() * &proof.witness_gkr.root_q;
        if lhs != rhs {
            return Err(LookupError::GkrRootMismatch);
        }

        // ---- Step 6: Verify leaf-level claims ----

        // Witness leaves: p(x) = 1, q(x) = β − w(x).
        // Claim: v_p should equal 1 (MLE of constant 1).
        if witness_result.expected_p != one {
            return Err(LookupError::GkrLeafMismatch);
        }
        // v_q should equal β − w̃(r_w).
        // So w̃(r_w) = β − v_q.
        let w_eval = beta.clone() - &witness_result.expected_q;

        // Table leaves: p(x) = m(x), q(x) = β − T(x).
        // v_p should equal m̃(r_t), v_q should equal β − T̃(r_t).
        // Verifier computes these from the known data.

        // Evaluate m̃ and T̃ at r_t.
        let (m_eval, t_eval) = if table_result.point.is_empty() {
            // 0-variable case: single leaf, MLEs evaluate to their sole entry.
            (proof.multiplicities[0].clone(), table[0].clone())
        } else {
            let eq_at_t = build_eq_x_r_vec(&table_result.point, field_cfg)?;
            let m_e: F = proof
                .multiplicities
                .iter()
                .zip(eq_at_t.iter())
                .fold(zero.clone(), |acc, (m_j, eq_j)| acc + &(m_j.clone() * eq_j));
            let t_e: F = table
                .iter()
                .zip(eq_at_t.iter())
                .fold(zero.clone(), |acc, (t_j, eq_j)| acc + &(t_j.clone() * eq_j));
            (m_e, t_e)
        };

        if m_eval != table_result.expected_p {
            return Err(LookupError::GkrLeafMismatch);
        }

        let expected_q = beta.clone() - &t_eval;
        if expected_q != table_result.expected_q {
            return Err(LookupError::GkrLeafMismatch);
        }

        // ---- Step 7: Check multiplicity sum = witness_len ----
        let m_sum: F = proof
            .multiplicities
            .iter()
            .cloned()
            .fold(zero, |a, b| a + &b);
        let expected_sum = F::from_with_cfg(witness_len as u64, field_cfg);
        if m_sum != expected_sum {
            return Err(LookupError::MultiplicitySumMismatch {
                expected: witness_len as u64,
                got: 0,
            });
        }

        Ok(GkrLogupVerifierSubClaim {
            witness_eval_point: witness_result.point,
            witness_expected_eval: w_eval,
            table_eval_point: table_result.point,
            mult_expected_eval: m_eval,
            table_expected_eval: t_eval,
        })
    }
}

/// Recover the final GKR evaluation point from the proof structure.
///
/// The point is built up round-by-round: for round 0, it's (λ_0);
/// for round k ≥ 1, it's (sumcheck_point, λ_k). We can't reconstruct
/// the exact sumcheck randomness without re-running the transcript, so
/// the prover records this externally. This helper is a fallback that
/// returns an empty vector when the point can't be reconstructed
/// (the actual point is recovered by the verifier via the transcript).
fn recover_gkr_eval_point<F: InnerTransparentField + FromPrimitiveWithConfig>(
    _proof: &GkrFractionProof<F>,
    _num_vars: usize,
    _field_cfg: &F::Config,
) -> Vec<F> {
    // The prover's evaluation point is obtained by running the verifier
    // side of the GKR protocol. Since the prover already ran the sumchecks,
    // the point is implicit in the transcript. We return an empty vec here;
    // the caller can re-derive it from the transcript if needed.
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crypto_bigint::{U128, const_monty_params};
    use crypto_primitives::crypto_bigint_const_monty::ConstMontyField;
    use zinc_transcript::KeccakTranscript;

    const N: usize = 2;
    const_monty_params!(TestParams, U128, "00000000b933426489189cb5b47d567f");
    type F = ConstMontyField<TestParams, N>;

    // --- Fraction tree unit tests ---

    #[test]
    fn fraction_tree_single_leaf() {
        let p = vec![F::from(3u32)];
        let q = vec![F::from(7u32)];
        let tree = build_fraction_tree(p.clone(), q.clone());
        assert_eq!(tree.len(), 1); // just the leaf = root
        assert_eq!(tree[0].p[0], F::from(3u32));
        assert_eq!(tree[0].q[0], F::from(7u32));
    }

    #[test]
    fn fraction_tree_two_leaves() {
        // Fractions: 3/7 + 5/11 = (3·11 + 5·7)/(7·11) = (33 + 35)/77 = 68/77
        let p = vec![F::from(3u32), F::from(5u32)];
        let q = vec![F::from(7u32), F::from(11u32)];
        let tree = build_fraction_tree(p, q);
        assert_eq!(tree.len(), 2); // [leaves, root]
        let root = tree.last().unwrap();
        assert_eq!(root.p[0], F::from(68u32));
        assert_eq!(root.q[0], F::from(77u32));
    }

    #[test]
    fn fraction_tree_four_leaves() {
        // 1/2 + 1/3 + 1/5 + 1/7
        // Level 1 (after pairing):
        //   left:  1/2 + 1/3 = (3+2)/6 = 5/6
        //   right: 1/5 + 1/7 = (7+5)/35 = 12/35
        // Root: 5/6 + 12/35 = (5·35 + 12·6)/(6·35) = (175+72)/210 = 247/210
        let p = vec![F::from(1u32); 4];
        let q = vec![
            F::from(2u32),
            F::from(3u32),
            F::from(5u32),
            F::from(7u32),
        ];
        let tree = build_fraction_tree(p, q);
        assert_eq!(tree.len(), 3); // [leaves, level1, root]
        let root = tree.last().unwrap();
        assert_eq!(root.p[0], F::from(247u32));
        assert_eq!(root.q[0], F::from(210u32));
    }

    #[test]
    fn evaluate_mle_at_basic() {
        // MLE with 2 variables: f(0,0)=1, f(1,0)=2, f(0,1)=3, f(1,1)=4
        let evals = vec![F::from(1u32), F::from(2u32), F::from(3u32), F::from(4u32)];
        // Evaluate at (0, 0) should give 1
        let val = evaluate_mle_at(&evals, &[F::from(0u32), F::from(0u32)], &());
        assert_eq!(val, F::from(1u32));
        // Evaluate at (1, 0) should give 2
        let val = evaluate_mle_at(&evals, &[F::from(1u32), F::from(0u32)], &());
        assert_eq!(val, F::from(2u32));
        // Evaluate at (1, 1) should give 4
        let val = evaluate_mle_at(&evals, &[F::from(1u32), F::from(1u32)], &());
        assert_eq!(val, F::from(4u32));
    }

    // --- GKR LogUp protocol tests ---

    #[test]
    fn gkr_logup_prove_verify_small() {
        // Table: {0, 1, 2, 3}
        let table: Vec<F> = (0..4u32).map(F::from).collect();

        // Witness: [0, 1, 1, 3] — all entries are in the table.
        let witness: Vec<F> = vec![0u32, 1, 1, 3].into_iter().map(F::from).collect();

        let mut prover_transcript = KeccakTranscript::new();
        let (proof, _prover_state) = GkrLogupProtocol::<F>::prove_as_subprotocol(
            &mut prover_transcript,
            &witness,
            &table,
            &(),
        )
        .expect("prover should succeed");

        let mut verifier_transcript = KeccakTranscript::new();
        let _subclaim = GkrLogupProtocol::<F>::verify_as_subprotocol(
            &mut verifier_transcript,
            &proof,
            &table,
            witness.len(),
            &(),
        )
        .expect("verifier should accept");
    }

    #[test]
    fn gkr_logup_prove_verify_repeated_entries() {
        // Table: {0, 1, 2, 3, 4, 5, 6, 7}
        let table: Vec<F> = (0..8u32).map(F::from).collect();

        // Witness: 8 lookups, some repeated.
        let witness: Vec<F> = vec![0u32, 0, 1, 1, 2, 3, 7, 5]
            .into_iter()
            .map(F::from)
            .collect();

        let mut prover_transcript = KeccakTranscript::new();
        let (proof, _) = GkrLogupProtocol::<F>::prove_as_subprotocol(
            &mut prover_transcript,
            &witness,
            &table,
            &(),
        )
        .expect("prover should succeed");

        let mut verifier_transcript = KeccakTranscript::new();
        let _subclaim = GkrLogupProtocol::<F>::verify_as_subprotocol(
            &mut verifier_transcript,
            &proof,
            &table,
            witness.len(),
            &(),
        )
        .expect("verifier should accept");
    }

    #[test]
    fn gkr_logup_prove_verify_larger() {
        // Table: {0, 1, ..., 15}
        let table: Vec<F> = (0..16u32).map(F::from).collect();

        // Witness: 16 entries, all valid.
        let witness: Vec<F> = vec![0u32, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
            .into_iter()
            .map(F::from)
            .collect();

        let mut prover_transcript = KeccakTranscript::new();
        let (proof, _) = GkrLogupProtocol::<F>::prove_as_subprotocol(
            &mut prover_transcript,
            &witness,
            &table,
            &(),
        )
        .expect("prover should succeed");

        let mut verifier_transcript = KeccakTranscript::new();
        let _subclaim = GkrLogupProtocol::<F>::verify_as_subprotocol(
            &mut verifier_transcript,
            &proof,
            &table,
            witness.len(),
            &(),
        )
        .expect("verifier should accept");
    }

    #[test]
    fn gkr_logup_reject_invalid_witness() {
        let table: Vec<F> = (0..4u32).map(F::from).collect();

        // Witness contains 5, which is NOT in the table.
        let witness: Vec<F> = vec![0u32, 5].into_iter().map(F::from).collect();

        let mut prover_transcript = KeccakTranscript::new();
        let result = GkrLogupProtocol::<F>::prove_as_subprotocol(
            &mut prover_transcript,
            &witness,
            &table,
            &(),
        );

        assert!(result.is_err());
    }

    #[test]
    fn gkr_logup_different_witness_table_sizes() {
        // Table: {0, 1, 2, 3, 4, 5, 6, 7} (8 entries, 3 vars)
        let table: Vec<F> = (0..8u32).map(F::from).collect();

        // Witness: [0, 1] (2 entries, 1 var)
        let witness: Vec<F> = vec![0u32, 1].into_iter().map(F::from).collect();

        let mut prover_transcript = KeccakTranscript::new();
        let (proof, _) = GkrLogupProtocol::<F>::prove_as_subprotocol(
            &mut prover_transcript,
            &witness,
            &table,
            &(),
        )
        .expect("prover should succeed");

        let mut verifier_transcript = KeccakTranscript::new();
        let _subclaim = GkrLogupProtocol::<F>::verify_as_subprotocol(
            &mut verifier_transcript,
            &proof,
            &table,
            witness.len(),
            &(),
        )
        .expect("verifier should accept");
    }

    #[test]
    fn gkr_logup_single_entry() {
        // Table: {42}
        let table: Vec<F> = vec![F::from(42u32)];

        // Witness: [42]
        let witness: Vec<F> = vec![F::from(42u32)];

        let mut prover_transcript = KeccakTranscript::new();
        let (proof, _) = GkrLogupProtocol::<F>::prove_as_subprotocol(
            &mut prover_transcript,
            &witness,
            &table,
            &(),
        )
        .expect("prover should succeed");

        let mut verifier_transcript = KeccakTranscript::new();
        let _subclaim = GkrLogupProtocol::<F>::verify_as_subprotocol(
            &mut verifier_transcript,
            &proof,
            &table,
            witness.len(),
            &(),
        )
        .expect("verifier should accept");
    }
}
