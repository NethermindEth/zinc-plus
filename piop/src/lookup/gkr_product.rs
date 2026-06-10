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
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use zinc_poly::utils::eq_eval;
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

/// Proof for a **forest** of product trees with their layer sumchecks
/// BATCHED: same-index layers of all active trees run as ONE multi-group
/// eq-factored sumcheck, the per-tree claims combined under a fresh
/// per-layer challenge ρ (group `t` enters with scale `ρ^t`) and reduced
/// at a SHARED point with one line-challenge λ. Trees must come in
/// non-increasing depth order (the active set is always a prefix). A
/// single-tree forest is transcript-identical to the pre-forest
/// [`prove_product_tree`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProductForestProof<F: PrimeField> {
    /// Per-tree claimed products, tree-major order.
    pub roots: Vec<F>,
    /// Per merged layer `k = 0..max_depth−1`.
    pub layers: Vec<ForestLayerProof<F>>,
}

/// One merged forest layer: the shared sumcheck (`None` for layer 0, the
/// direct per-tree `root = l·r` check) and the per-ACTIVE-tree
/// `(left, right)` child evaluations at the layer's shared subclaim point.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ForestLayerProof<F: PrimeField> {
    pub sumcheck_proof: Option<SumcheckProof<F>>,
    pub evals: Vec<(F, F)>,
}

/// Batched-forest GKR prover: proves `∏ leaves_t = root_t` for every tree
/// at once, with same-index layers merged (see [`ProductForestProof`]).
/// Returns the proof and, per tree, the leaf-layer point and claimed
/// leaf-MLE evaluation `(r_t ∈ F^{d_t}, ṽ_{d_t}(r_t))` the caller binds.
#[allow(clippy::arithmetic_side_effects)]
pub fn prove_product_forest<F>(
    transcript: &mut impl Transcript,
    leaves_per_tree: Vec<Vec<F>>,
    field_cfg: &F::Config,
) -> (ProductForestProof<F>, Vec<(Vec<F>, F)>)
where
    F: InnerTransparentField + FromPrimitiveWithConfig + Send + Sync,
    F::Inner: ConstTranscribable + Zero + Default + Send + Sync,
    F::Modulus: ConstTranscribable,
    F::Config: Sync,
{
    let trees: Vec<Vec<Vec<F>>> = {
        let _g = zinc_utils::prof::scope("gkr:build");
        leaves_per_tree.into_iter().map(build_product_tree).collect()
    };
    let depths: Vec<usize> = trees.iter().map(|t| t.len() - 1).collect();
    debug_assert!(!depths.is_empty() && depths.iter().all(|&d| d >= 1));
    debug_assert!(depths.windows(2).all(|w| w[0] >= w[1]), "non-increasing depths");
    let max_d = depths[0];
    let num_trees = trees.len();
    let one = F::one_with_cfg(field_cfg);

    let roots: Vec<F> = trees.iter().map(|t| t.last().expect("built")[0].clone()).collect();
    let mut buf = vec![0u8; F::Inner::NUM_BYTES];
    for root in &roots {
        transcript.absorb_random_field(root, &mut buf);
    }

    // Per-tree state: the running claim v_t at the running point r_t.
    let mut v: Vec<F> = roots.clone();
    let mut r: Vec<Vec<F>> = vec![Vec::new(); num_trees];
    let mut layer_proofs = Vec::with_capacity(max_d);

    for k in 0..max_d {
        // Active trees at this layer (a prefix, by the depth ordering).
        let active = depths.iter().filter(|&&d| d > k).count();
        let child = |t: usize| -> &Vec<F> { &trees[t][depths[t] - (k + 1)] };

        if k == 0 {
            let mut evals = Vec::with_capacity(active);
            for t in 0..active {
                let c = child(t);
                let (l, rgt) = (c[0].clone(), c[1].clone());
                transcript.absorb_random_field(&l, &mut buf);
                transcript.absorb_random_field(&rgt, &mut buf);
                debug_assert_eq!(v[t], l.clone() * &rgt, "root must equal left·right");
                evals.push((l, rgt));
            }
            let lambda: F = transcript.get_field_challenge(field_cfg);
            for (t, (l, rgt)) in evals.iter().enumerate() {
                v[t] = (one.clone() - &lambda) * l + &(lambda.clone() * rgt);
                r[t] = vec![lambda.clone()];
            }
            layer_proofs.push(ForestLayerProof { sumcheck_proof: None, evals });
        } else {
            let _g = zinc_utils::prof::scope("gkr:round");
            // Fresh per-layer batching challenge (only when ≥ 2 trees are
            // active); group t enters the merged sumcheck with scale ρ^t,
            // so the claimed sum is Σ_t ρ^t·v_t and the final evaluation
            // pins each tree's contribution by Schwartz–Zippel.
            let rho: Option<F> =
                (active > 1).then(|| transcript.get_field_challenge(field_cfg));
            let mut scale = one.clone();
            let mut groups = Vec::with_capacity(active);
            for t in 0..active {
                let c = child(t);
                let half = 1usize << k;
                groups.push(crate::sumcheck::eq_factored::EqInnerGroup {
                    q: r[t].clone(),
                    pairs: vec![(c[..half].to_vec(), c[half..].to_vec())],
                    scale: scale.clone(),
                });
                if let Some(rho) = &rho {
                    scale = scale * rho;
                }
            }
            let (sumcheck_proof, s, final_evals) =
                crate::sumcheck::eq_factored::prove_eq_inner_sumcheck(
                    transcript, groups, field_cfg,
                );

            let mut evals = Vec::with_capacity(active);
            for fe in &final_evals {
                let (l_at, r_at) = fe[0].clone();
                transcript.absorb_random_field(&l_at, &mut buf);
                transcript.absorb_random_field(&r_at, &mut buf);
                evals.push((l_at, r_at));
            }
            let lambda: F = transcript.get_field_challenge(field_cfg);
            for (t, (l_at, r_at)) in evals.iter().enumerate() {
                v[t] = (one.clone() - &lambda) * l_at + &(lambda.clone() * r_at);
                r[t] = s.clone();
                r[t].push(lambda.clone());
            }
            layer_proofs.push(ForestLayerProof { sumcheck_proof: Some(sumcheck_proof), evals });
        }
    }

    let claims: Vec<(Vec<F>, F)> = r.into_iter().zip(v).collect();
    (ProductForestProof { roots, layers: layer_proofs }, claims)
}

/// GKR grand-product prover (single tree).
///
/// Returns the proof, the leaf-layer evaluation point `r ∈ F^d`, and the
/// claimed leaf-MLE evaluation `ṽ_d(r)` that the caller must bind to the
/// actual leaves. Thin wrapper over the single-tree
/// [`prove_product_forest`] (transcript-identical to the pre-forest form).
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
    if leaves.len() == 1 {
        let root = leaves[0].clone();
        let mut buf = vec![0u8; F::Inner::NUM_BYTES];
        transcript.absorb_random_field(&root, &mut buf);
        return (ProductTreeProof { root: root.clone(), layers: vec![] }, vec![], root);
    }
    let (forest, mut claims) = prove_product_forest(transcript, vec![leaves], field_cfg);
    let (point, eval) = claims.pop().expect("one tree");
    let layers = forest
        .layers
        .into_iter()
        .map(|fl| {
            let (left, right) = fl.evals.into_iter().next().expect("one tree");
            ProductLayerProof { sumcheck_proof: fl.sumcheck_proof, left, right }
        })
        .collect();
    (
        ProductTreeProof { root: forest.roots.into_iter().next().expect("one tree"), layers },
        point,
        eval,
    )
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
    if num_vars == 0 {
        let mut buf = vec![0u8; F::Inner::NUM_BYTES];
        transcript.absorb_random_field(&proof.root, &mut buf);
        return Ok((vec![], proof.root.clone()));
    }
    let forest = ProductForestProof {
        roots: vec![proof.root.clone()],
        layers: proof
            .layers
            .iter()
            .map(|lp| ForestLayerProof {
                sumcheck_proof: lp.sumcheck_proof.clone(),
                evals: vec![(lp.left.clone(), lp.right.clone())],
            })
            .collect(),
    };
    let mut claims = verify_product_forest(transcript, &forest, &[num_vars], field_cfg)?;
    Ok(claims.pop().expect("one tree"))
}

/// Batched-forest GKR verifier (see [`ProductForestProof`]): re-derives,
/// per tree, the leaf-layer point and claimed leaf-MLE evaluation. The
/// caller supplies the per-tree depths (non-increasing) and binds the
/// returned claims to the actual leaves.
#[allow(clippy::arithmetic_side_effects)]
pub fn verify_product_forest<F>(
    transcript: &mut impl Transcript,
    proof: &ProductForestProof<F>,
    depths: &[usize],
    field_cfg: &F::Config,
) -> Result<Vec<(Vec<F>, F)>, ProductTreeError>
where
    F: InnerTransparentField + FromPrimitiveWithConfig + Send + Sync,
    F::Inner: ConstTranscribable + Zero,
    F::Modulus: ConstTranscribable,
{
    let num_trees = depths.len();
    if num_trees == 0
        || depths.iter().any(|&d| d < 1)
        || depths.windows(2).any(|w| w[0] < w[1])
        || proof.roots.len() != num_trees
    {
        return Err(ProductTreeError::LayerCountMismatch);
    }
    let max_d = depths[0];
    if proof.layers.len() != max_d {
        return Err(ProductTreeError::LayerCountMismatch);
    }
    let one = F::one_with_cfg(field_cfg);

    let mut buf = vec![0u8; F::Inner::NUM_BYTES];
    for root in &proof.roots {
        transcript.absorb_random_field(root, &mut buf);
    }

    let mut v: Vec<F> = proof.roots.clone();
    let mut r: Vec<Vec<F>> = vec![Vec::new(); num_trees];

    for k in 0..max_d {
        let active = depths.iter().filter(|&&d| d > k).count();
        let lp = &proof.layers[k];
        if lp.evals.len() != active {
            return Err(ProductTreeError::LayerCountMismatch);
        }

        if k == 0 {
            for (t, (l, rgt)) in lp.evals.iter().enumerate() {
                transcript.absorb_random_field(l, &mut buf);
                transcript.absorb_random_field(rgt, &mut buf);
                if v[t] != l.clone() * rgt {
                    return Err(ProductTreeError::Layer0Mismatch);
                }
            }
            let lambda: F = transcript.get_field_challenge(field_cfg);
            for (t, (l, rgt)) in lp.evals.iter().enumerate() {
                v[t] = (one.clone() - &lambda) * l + &(lambda.clone() * rgt);
                r[t] = vec![lambda.clone()];
            }
        } else {
            let sc = lp.sumcheck_proof.as_ref().ok_or(ProductTreeError::MissingSumcheck)?;
            // ρ batches the active trees' claims; the claimed sum must be
            // Σ_t ρ^t·v_t, and the final evaluation pins each tree's
            // eq_t(s)·l_t·r_t contribution under the same ρ-powers.
            let rho: Option<F> =
                (active > 1).then(|| transcript.get_field_challenge(field_cfg));
            let mut claimed = F::zero_with_cfg(field_cfg);
            let mut scale = one.clone();
            for vt in v.iter().take(active) {
                claimed = claimed + &(scale.clone() * vt);
                if let Some(rho) = &rho {
                    scale = scale * rho;
                }
            }
            if sc.claimed_sum != claimed {
                return Err(ProductTreeError::ClaimedSumMismatch);
            }
            let subclaim = MLSumcheck::verify_as_subprotocol(transcript, k, 3, sc, field_cfg)
                .map_err(|_| ProductTreeError::Sumcheck)?;
            let s = &subclaim.point;

            for (l, rgt) in &lp.evals {
                transcript.absorb_random_field(l, &mut buf);
                transcript.absorb_random_field(rgt, &mut buf);
            }
            let mut expected = F::zero_with_cfg(field_cfg);
            let mut scale = one.clone();
            for (t, (l, rgt)) in lp.evals.iter().enumerate() {
                let eq_val =
                    eq_eval(s, &r[t], one.clone()).map_err(|_| ProductTreeError::Sumcheck)?;
                expected = expected + &(scale.clone() * &(eq_val * &(l.clone() * rgt)));
                if let Some(rho) = &rho {
                    scale = scale * rho;
                }
            }
            if expected != subclaim.expected_evaluation {
                return Err(ProductTreeError::FinalEvalMismatch);
            }

            let lambda: F = transcript.get_field_challenge(field_cfg);
            for (t, (l, rgt)) in lp.evals.iter().enumerate() {
                v[t] = (one.clone() - &lambda) * l + &(lambda.clone() * rgt);
                r[t] = s.clone();
                r[t].push(lambda.clone());
            }
        }
    }

    Ok(r.into_iter().zip(v).collect())
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
