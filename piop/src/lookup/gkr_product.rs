//! GKR grand-product argument over a multilinear field.
//!
//! Proves `∏_{x ∈ {0,1}^d} v(x) = root` for a leaf vector `v` of length
//! `2^d`, via a layer-by-layer product tree verified with sumchecks
//! (Thaler's grand product — equivalently the *denominator* component of
//! the Papini–Häböck GKR fraction tree, specialised to a pure product, so
//! there is no numerator and no `α` batching).
//!
//! This is the reusable engine for the char-2-sound multiplicative lookup
//! `∏_i (δ − a_i) = ∏_t (δ − fp_t)^{m_t}`: the witness side and the
//! binary-multiplicity table side are each one product tree. The argument
//! is field-agnostic and char-2-safe — products never cancel (`(δ−v)² ≠ 0`),
//! unlike the additive LogUp sum.
//!
//! ## Layer layout
//!
//! Layer `k` has `2^k` entries; layer `d` = leaves, layer `0` = root.
//! ```text
//!   v_k[i] = v_{k+1}[i] · v_{k+1}[i + 2^k]
//! ```
//! For `k ≥ 1` the parent claim `ṽ_k(r) = Σ_z eq(r,z)·left(z)·right(z)`
//! (`left = v_{k+1}[..2^k]`, `right = v_{k+1}[2^k..]`) is a degree-3
//! sumcheck; layer `0` is a direct product check `root = left · right`.
//!
//! ## Output / binding
//!
//! [`prove_product_tree`] returns `(proof, point r ∈ F^d, leaf_eval)` where
//! `leaf_eval = ṽ_d(r)` is the claimed evaluation of the leaf MLE at `r`.
//! [`verify_product_tree`] re-derives `(r, leaf_eval)` from the proof. The
//! caller **binds** `leaf_eval` to the actual leaves at `r` (for witness
//! leaves: a `ψ_z` read-off of the committed words; for table leaves: the
//! public / structured fingerprints and the committed multiplicity bits).
//! Leaf count must be a power of two; pad with the multiplicative identity
//! `1` (a no-op factor) otherwise.

use crypto_primitives::{FromPrimitiveWithConfig, PrimeField};
use num_traits::Zero;
use zinc_poly::{
    mle::DenseMultilinearExtension,
    utils::{build_eq_x_r_inner, eq_eval},
};
use zinc_transcript::traits::{ConstTranscribable, Transcript};
use zinc_utils::{cfg_into_iter, inner_transparent_field::InnerTransparentField};

use crate::sumcheck::{MLSumcheck, SumcheckProof};

/// Proof that `∏ leaves = root` for a leaf vector of length `2^d`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProductTreeProof<F: PrimeField> {
    /// The claimed product of all leaves.
    pub root: F,
    /// Per-layer proofs, one per GKR level `k = 0..d`.
    pub layers: Vec<ProductLayerProof<F>>,
}

/// Proof for a single product-tree layer: the two child evaluations at the
/// layer's subclaim point, plus the sumcheck (`None` for layer 0, which has
/// zero variables and is a direct product check).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProductLayerProof<F: PrimeField> {
    /// Sumcheck proof for this layer (`None` for layer `k = 0`).
    pub sumcheck_proof: Option<SumcheckProof<F>>,
    /// Left-child MLE evaluation at the subclaim point.
    pub left: F,
    /// Right-child MLE evaluation at the subclaim point.
    pub right: F,
}

/// Failure modes of [`verify_product_tree`].
#[derive(Debug, thiserror::Error)]
pub enum ProductTreeError {
    #[error("product-tree layer count != num_vars")]
    LayerCountMismatch,
    #[error("product-tree layer-0 product check failed (root != left·right)")]
    Layer0Mismatch,
    #[error("product-tree layer missing its sumcheck proof")]
    MissingSumcheck,
    #[error("product-tree layer claimed-sum != parent claim")]
    ClaimedSumMismatch,
    #[error("product-tree layer final-eval != eq·left·right")]
    FinalEvalMismatch,
    #[error("product-tree inner sumcheck/eq error")]
    Sumcheck,
}

/// Build the product tree bottom-up: returns `[leaves, …, root]` where
/// entry `k` is GKR layer `d − k` (so `[0]` = leaves of `2^d`, last = root).
#[allow(clippy::arithmetic_side_effects)]
fn build_product_tree<F>(leaves: Vec<F>) -> Vec<Vec<F>>
where
    F: InnerTransparentField + Send + Sync,
{
    let d = zinc_utils::log2(leaves.len()) as usize;
    debug_assert_eq!(leaves.len(), 1usize << d, "leaf count must be a power of two");
    let mut layers: Vec<Vec<F>> = Vec::with_capacity(d + 1);
    layers.push(leaves);
    for level in (0..d).rev() {
        let half = 1usize << level;
        let parent: Vec<F> = {
            let child = layers.last().expect("tree non-empty during construction");
            cfg_into_iter!(0..half)
                .map(|i| child[i].clone() * &child[i + half])
                .collect()
        };
        layers.push(parent);
    }
    layers
}

/// GKR grand-product prover.
///
/// Returns the proof, the leaf-layer evaluation point `r ∈ F^d`, and the
/// claimed leaf-MLE evaluation `ṽ_d(r)` that the caller must bind to the
/// actual leaves.
#[allow(clippy::arithmetic_side_effects)]
pub fn prove_product_tree<F>(
    transcript: &mut impl Transcript,
    leaves: Vec<F>,
    field_cfg: &F::Config,
) -> (ProductTreeProof<F>, Vec<F>, F)
where
    F: InnerTransparentField + FromPrimitiveWithConfig + Send + Sync,
    F::Inner: ConstTranscribable + Zero + Default + Send + Sync,
    F::Modulus: ConstTranscribable,
    F::Config: Sync,
{
    let layers = build_product_tree(leaves);
    let d = layers.len() - 1;
    let root = layers[d][0].clone();

    let mut buf = vec![0u8; F::Inner::NUM_BYTES];
    transcript.absorb_random_field(&root, &mut buf);

    if d == 0 {
        return (ProductTreeProof { root: root.clone(), layers: vec![] }, vec![], root);
    }

    let inner_zero = F::zero_with_cfg(field_cfg).inner().clone();
    let mut layer_proofs = Vec::with_capacity(d);
    let mut v = root.clone(); // current parent claim ṽ_k(r_k)
    let mut r_k: Vec<F> = Vec::new();

    for round in 0..d {
        let k = round;
        let half = 1usize << k;
        let child = &layers[d - (round + 1)]; // GKR layer k+1 (2^{k+1} entries)
        let left = &child[..half];
        let right = &child[half..];

        if k == 0 {
            let l = left[0].clone();
            let r = right[0].clone();
            transcript.absorb_random_field(&l, &mut buf);
            transcript.absorb_random_field(&r, &mut buf);
            debug_assert_eq!(v, l.clone() * &r, "root must equal left·right at layer 0");
            layer_proofs.push(ProductLayerProof { sumcheck_proof: None, left: l.clone(), right: r.clone() });

            let lambda: F = transcript.get_field_challenge(field_cfg);
            let one = F::one_with_cfg(field_cfg);
            v = (one - &lambda) * &l + &(lambda.clone() * &r);
            r_k = vec![lambda];
        } else {
            let eq_r = build_eq_x_r_inner(&r_k, field_cfg).expect("eq(x,r_k) construction");
            let mk = |data: &[F]| -> DenseMultilinearExtension<F::Inner> {
                DenseMultilinearExtension::from_evaluations_vec(
                    k,
                    data.iter().map(|x| x.inner().clone()).collect(),
                    inner_zero.clone(),
                )
            };
            // mles = [eq, left, right]; comb_fn = eq · left · right (degree 3).
            let mles = vec![eq_r, mk(left), mk(right)];
            let comb_fn = move |vals: &[F]| -> F { vals[0].clone() * &(vals[1].clone() * &vals[2]) };

            let (sumcheck_proof, st) =
                MLSumcheck::prove_as_subprotocol(transcript, mles, k, 3, comb_fn, field_cfg);

            // Recover left(s), right(s): each folded MLE has 1 var left;
            // interpolate with the final challenge (mirrors the sumcheck fold).
            let s = &st.randomness;
            let last = s.last().expect("sumcheck has ≥ 1 challenge");
            let one_minus_last = F::one_with_cfg(field_cfg) - last;
            let interp = |m: &DenseMultilinearExtension<F::Inner>| -> F {
                let v0 = F::new_unchecked_with_cfg(m[0].clone(), field_cfg);
                let v1 = F::new_unchecked_with_cfg(m[1].clone(), field_cfg);
                one_minus_last.clone() * &v0 + &(last.clone() * &v1)
            };
            let l_at = interp(&st.mles[1]);
            let r_at = interp(&st.mles[2]);

            transcript.absorb_random_field(&l_at, &mut buf);
            transcript.absorb_random_field(&r_at, &mut buf);
            layer_proofs.push(ProductLayerProof {
                sumcheck_proof: Some(sumcheck_proof),
                left: l_at.clone(),
                right: r_at.clone(),
            });

            let lambda: F = transcript.get_field_challenge(field_cfg);
            let one = F::one_with_cfg(field_cfg);
            v = (one - &lambda) * &l_at + &(lambda.clone() * &r_at);
            r_k = s.clone();
            r_k.push(lambda);
        }
    }

    (ProductTreeProof { root: root.clone(), layers: layer_proofs }, r_k.clone(), v)
}

/// GKR grand-product verifier.
///
/// Re-derives the leaf-layer point `r ∈ F^d` and the claimed leaf-MLE
/// evaluation `ṽ_d(r)`. The caller binds the latter to the actual leaves.
#[allow(clippy::arithmetic_side_effects)]
pub fn verify_product_tree<F>(
    transcript: &mut impl Transcript,
    proof: &ProductTreeProof<F>,
    num_vars: usize,
    field_cfg: &F::Config,
) -> Result<(Vec<F>, F), ProductTreeError>
where
    F: InnerTransparentField + FromPrimitiveWithConfig + Send + Sync,
    F::Inner: ConstTranscribable + Zero,
    F::Modulus: ConstTranscribable,
{
    let d = num_vars;
    let one = F::one_with_cfg(field_cfg);

    let mut buf = vec![0u8; F::Inner::NUM_BYTES];
    transcript.absorb_random_field(&proof.root, &mut buf);

    if d == 0 {
        return Ok((vec![], proof.root.clone()));
    }
    if proof.layers.len() != d {
        return Err(ProductTreeError::LayerCountMismatch);
    }

    let mut v = proof.root.clone();
    let mut r_k: Vec<F> = Vec::new();

    for round in 0..d {
        let k = round;
        let lp = &proof.layers[round];

        if k == 0 {
            transcript.absorb_random_field(&lp.left, &mut buf);
            transcript.absorb_random_field(&lp.right, &mut buf);
            if v != lp.left.clone() * &lp.right {
                return Err(ProductTreeError::Layer0Mismatch);
            }
            let lambda: F = transcript.get_field_challenge(field_cfg);
            v = (one.clone() - &lambda) * &lp.left + &(lambda.clone() * &lp.right);
            r_k = vec![lambda];
        } else {
            let sc = lp.sumcheck_proof.as_ref().ok_or(ProductTreeError::MissingSumcheck)?;
            if sc.claimed_sum != v {
                return Err(ProductTreeError::ClaimedSumMismatch);
            }
            let subclaim = MLSumcheck::verify_as_subprotocol(transcript, k, 3, sc, field_cfg)
                .map_err(|_| ProductTreeError::Sumcheck)?;
            let s = &subclaim.point;

            transcript.absorb_random_field(&lp.left, &mut buf);
            transcript.absorb_random_field(&lp.right, &mut buf);

            let eq_val = eq_eval(s, &r_k, one.clone()).map_err(|_| ProductTreeError::Sumcheck)?;
            let expected = eq_val * &(lp.left.clone() * &lp.right);
            if expected != subclaim.expected_evaluation {
                return Err(ProductTreeError::FinalEvalMismatch);
            }

            let lambda: F = transcript.get_field_challenge(field_cfg);
            v = (one.clone() - &lambda) * &lp.left + &(lambda.clone() * &lp.right);
            r_k = s.clone();
            r_k.push(lambda);
        }
    }

    Ok((r_k, v))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crypto_bigint::{U128, const_monty_params};
    use crypto_primitives::crypto_bigint_const_monty::ConstMontyField;
    use zinc_transcript::Blake3Transcript;

    const N: usize = 2;
    const_monty_params!(TestParams, U128, "00000000b933426489189cb5b47d567f");
    type F = ConstMontyField<TestParams, N>;

    fn product(leaves: &[F]) -> F {
        leaves.iter().fold(F::from(1u32), |a, b| a * b)
    }

    #[test]
    fn product_tree_roundtrip_eight() {
        let leaves: Vec<F> = (1..=8u32).map(F::from).collect();
        let expected = product(&leaves);

        let mut pt = Blake3Transcript::new();
        let (proof, point, leaf_eval) = prove_product_tree(&mut pt, leaves, &());
        assert_eq!(proof.root, expected, "root must be the product of all leaves");
        assert_eq!(point.len(), 3);

        let mut vt = Blake3Transcript::new();
        let (vpoint, veval) = verify_product_tree(&mut vt, &proof, 3, &()).expect("verifier accepts");
        assert_eq!(vpoint, point, "verifier point must match prover");
        assert_eq!(veval, leaf_eval, "verifier leaf-eval must match prover");
    }

    /// The engine over GF(2^128) — the actual field the F_2 lookup adder
    /// uses. Confirms the generic bounds instantiate at the char-2 field
    /// (the product tree itself is char-2-agnostic; soundness vs. additive
    /// cancellation is a property of the lookup multiset, not this engine).
    #[test]
    fn product_tree_roundtrip_gf128() {
        use zinc_poly::univariate::binary_gf128::BinaryFieldGF128 as Gf;
        let leaves: Vec<Gf> = (1u64..=8).map(|v| Gf::from_words([v, 0])).collect();
        let expected = leaves.iter().fold(Gf::from_words([1, 0]), |a, b| a * b);

        let mut pt = Blake3Transcript::new();
        let (proof, point, leaf_eval) = prove_product_tree(&mut pt, leaves, &());
        assert_eq!(proof.root, expected, "GF(2^128) root must be the product of all leaves");

        let mut vt = Blake3Transcript::new();
        let (vpoint, veval) = verify_product_tree(&mut vt, &proof, 3, &()).expect("verifier accepts");
        assert_eq!(vpoint, point);
        assert_eq!(veval, leaf_eval);
    }

    #[test]
    fn product_tree_single_leaf() {
        let leaves = vec![F::from(42u32)];
        let mut pt = Blake3Transcript::new();
        let (proof, point, leaf_eval) = prove_product_tree(&mut pt, leaves, &());
        assert_eq!(proof.root, F::from(42u32));
        assert!(point.is_empty());
        assert_eq!(leaf_eval, F::from(42u32));

        let mut vt = Blake3Transcript::new();
        let (vpoint, veval) = verify_product_tree(&mut vt, &proof, 0, &()).expect("verifier accepts");
        assert!(vpoint.is_empty());
        assert_eq!(veval, F::from(42u32));
    }

    #[test]
    fn product_tree_tampered_root_rejected() {
        let leaves: Vec<F> = (1..=4u32).map(F::from).collect();
        let mut pt = Blake3Transcript::new();
        let (mut proof, _p, _e) = prove_product_tree(&mut pt, leaves, &());
        proof.root = proof.root.clone() + F::from(1u32); // corrupt the claimed product

        let mut vt = Blake3Transcript::new();
        assert!(verify_product_tree(&mut vt, &proof, 2, &()).is_err());
    }
}
