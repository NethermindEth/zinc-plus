//! GKR-based LogUp fractional sumcheck.
//!
//! Implements the GKR (Goldwasser-Kalai-Rothblum) protocol for proving
//! fractional sum identities, as described in:
//!
//!   Papini & Haböck, "Improving logarithmic derivative lookups using GKR"
//!   <https://eprint.iacr.org/2023/1284>
//!
//! ## Key advantage over the standard LogUp
//!
//! The prover only needs to send the **multiplicity vector** `m` — no
//! inverse vectors `u`, `v` are required.  This saves transmitting
//! `O(W + N)` field elements at the cost of `O(log²(max(W,N)))` extra
//! field elements in the GKR layer proofs.
//!
//! ## Protocol overview
//!
//! Given leaf fractions `p_i / q_i` for `i = 0..2^d`, the GKR fractional
//! sumcheck proves `Σ_i p_i/q_i = root_p/root_q` using a binary tree
//! of fraction additions verified layer-by-layer from root to leaves
//! using sumchecks.  At the leaves, the verifier checks evaluations against
//! the known input polynomials.

use crypto_primitives::FromPrimitiveWithConfig;
use num_traits::Zero;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use zinc_poly::{
    mle::DenseMultilinearExtension,
    utils::build_eq_x_r_inner,
};
use zinc_transcript::traits::{ConstTranscribable, Transcript};
use zinc_utils::{cfg_into_iter, inner_transparent_field::InnerTransparentField};

use crate::sumcheck::MLSumcheck;

use super::structs::{
    GkrFractionProof, GkrLayerProof, LookupError,
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
        let child = layers.last().expect("tree is non-empty during construction");
        let half = 1usize << level;

        let (parent_p, parent_q): (Vec<F>, Vec<F>) = if half >= 64 {
            cfg_into_iter!(0..half)
                .map(|i| {
                    let pl = &child.p[i];
                    let ql = &child.q[i];
                    let pr = &child.p[i + half];
                    let qr = &child.q[i + half];
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
    let d = layers.len() - 1; // number of GKR levels
    let root = layers.last().expect("tree is non-empty");
    let root_p = root.p[0].clone();
    let root_q = root.q[0].clone();

    // Absorb root values into transcript.
    let mut buf = vec![0u8; F::Inner::NUM_BYTES];
    transcript.absorb_random_field(&root_p, &mut buf);
    transcript.absorb_random_field(&root_q, &mut buf);

    if d == 0 {
        return GkrFractionProof {
            root_p,
            root_q,
            layer_proofs: vec![],
        };
    }

    let mut layer_proofs = Vec::with_capacity(d);
    let inner_zero = F::zero_with_cfg(field_cfg).inner().clone();

    let mut v_p = root_p.clone();
    let mut v_q = root_q.clone();
    let mut r_k: Vec<F> = Vec::new();

    for round in 0..d {
        // Child layer in our array: layers[d - (round + 1)]
        let child_layer_idx = d - (round + 1);
        let child_layer = &layers[child_layer_idx];

        let k = round;
        let half = 1usize << k;

        let p_left_vals = &child_layer.p[..half];
        let p_right_vals = &child_layer.p[half..];
        let q_left_vals = &child_layer.q[..half];
        let q_right_vals = &child_layer.q[half..];

        // Get batching challenge α_layer.
        let alpha: F = transcript.get_field_challenge(field_cfg);

        if k == 0 {
            // Round 0: zero sumcheck variables — direct algebraic check.
            let pl = p_left_vals[0].clone();
            let pr = p_right_vals[0].clone();
            let ql = q_left_vals[0].clone();
            let qr = q_right_vals[0].clone();

            transcript.absorb_random_field(&pl, &mut buf);
            transcript.absorb_random_field(&pr, &mut buf);
            transcript.absorb_random_field(&ql, &mut buf);
            transcript.absorb_random_field(&qr, &mut buf);

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

            let lambda: F = transcript.get_field_challenge(field_cfg);
            let one = F::one_with_cfg(field_cfg);
            let one_minus_lambda = one - &lambda;
            v_p = one_minus_lambda.clone() * &pl + &(lambda.clone() * &pr);
            v_q = one_minus_lambda * &ql + &(lambda.clone() * &qr);
            r_k = vec![lambda];
        } else {
            // Round k ≥ 1: sumcheck over k variables.
            let eq_r = build_eq_x_r_inner(&r_k, field_cfg)
                .expect("eq polynomial construction should succeed");

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

            let (sumcheck_proof, sumcheck_prover_state) = MLSumcheck::prove_as_subprotocol(
                transcript,
                mles,
                k,
                3,
                comb_fn,
                field_cfg,
            );

            let s = &sumcheck_prover_state.randomness;

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

            let lambda: F = transcript.get_field_challenge(field_cfg);
            let one = F::one_with_cfg(field_cfg);
            let one_minus_lambda = one - &lambda;
            v_p = one_minus_lambda.clone() * &pl_at_s + &(lambda.clone() * &pr_at_s);
            v_q = one_minus_lambda * &ql_at_s + &(lambda.clone() * &qr_at_s);
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

        let alpha: F = transcript.get_field_challenge(field_cfg);

        if k == 0 {
            let pl = &layer_proof.p_left;
            let pr = &layer_proof.p_right;
            let ql = &layer_proof.q_left;
            let qr = &layer_proof.q_right;

            transcript.absorb_random_field(pl, &mut buf);
            transcript.absorb_random_field(pr, &mut buf);
            transcript.absorb_random_field(ql, &mut buf);
            transcript.absorb_random_field(qr, &mut buf);

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

            let lambda: F = transcript.get_field_challenge(field_cfg);
            let one_minus_lambda = one.clone() - &lambda;
            v_p = one_minus_lambda.clone() * pl + &(lambda.clone() * pr);
            v_q = one_minus_lambda * ql + &(lambda.clone() * qr);
            r_k = vec![lambda];
        } else {
            let sumcheck_proof = layer_proof
                .sumcheck_proof
                .as_ref()
                .ok_or(LookupError::GkrLeafMismatch)?;

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
                3,
                sumcheck_proof,
                field_cfg,
            )?;

            let s = &subclaim.point;
            let pl = &layer_proof.p_left;
            let pr = &layer_proof.p_right;
            let ql = &layer_proof.q_left;
            let qr = &layer_proof.q_right;

            transcript.absorb_random_field(pl, &mut buf);
            transcript.absorb_random_field(pr, &mut buf);
            transcript.absorb_random_field(ql, &mut buf);
            transcript.absorb_random_field(qr, &mut buf);

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

#[cfg(test)]
mod tests {
    use super::*;
    use crypto_bigint::{U128, const_monty_params};
    use crypto_primitives::crypto_bigint_const_monty::ConstMontyField;
    use zinc_transcript::KeccakTranscript;

    const N: usize = 2;
    const_monty_params!(TestParams, U128, "00000000b933426489189cb5b47d567f");
    type F = ConstMontyField<TestParams, N>;

    #[test]
    fn fraction_tree_single_leaf() {
        let p = vec![F::from(3u32)];
        let q = vec![F::from(7u32)];
        let tree = build_fraction_tree(p, q);
        assert_eq!(tree.len(), 1);
        assert_eq!(tree[0].p[0], F::from(3u32));
        assert_eq!(tree[0].q[0], F::from(7u32));
    }

    #[test]
    fn fraction_tree_two_leaves() {
        let p = vec![F::from(3u32), F::from(5u32)];
        let q = vec![F::from(7u32), F::from(11u32)];
        let tree = build_fraction_tree(p, q);
        assert_eq!(tree.len(), 2);
        let root = tree.last().unwrap();
        assert_eq!(root.p[0], F::from(68u32));
        assert_eq!(root.q[0], F::from(77u32));
    }

    #[test]
    fn fraction_tree_four_leaves() {
        let p = vec![F::from(1u32); 4];
        let q = vec![
            F::from(2u32),
            F::from(3u32),
            F::from(5u32),
            F::from(7u32),
        ];
        let tree = build_fraction_tree(p, q);
        assert_eq!(tree.len(), 3);
        let root = tree.last().unwrap();
        assert_eq!(root.p[0], F::from(247u32));
        assert_eq!(root.q[0], F::from(210u32));
    }

    #[test]
    fn gkr_fraction_prove_verify_two_leaves() {
        let p = vec![F::from(3u32), F::from(5u32)];
        let q = vec![F::from(7u32), F::from(11u32)];
        let tree = build_fraction_tree(p, q);

        let mut pt = KeccakTranscript::new();
        let proof = gkr_fraction_prove(&mut pt, &tree, &());

        let mut vt = KeccakTranscript::new();
        let result = gkr_fraction_verify(&mut vt, &proof, 1, &())
            .expect("verifier should accept");

        assert_eq!(result.point.len(), 1);
    }

    #[test]
    fn gkr_fraction_prove_verify_eight_leaves() {
        let p: Vec<F> = (1..=8u32).map(F::from).collect();
        let q: Vec<F> = (10..=17u32).map(F::from).collect();
        let tree = build_fraction_tree(p, q);

        let mut pt = KeccakTranscript::new();
        let proof = gkr_fraction_prove(&mut pt, &tree, &());

        let mut vt = KeccakTranscript::new();
        let result = gkr_fraction_verify(&mut vt, &proof, 3, &())
            .expect("verifier should accept");

        assert_eq!(result.point.len(), 3);
    }
}
