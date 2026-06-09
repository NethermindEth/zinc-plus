//! Char-2-sound **multiplicative grand-product lookup** on top of the GKR
//! product-tree engine ([`super::gkr_product`]).
//!
//! Proves multiset inclusion `{a_i} ⊆ T` (public table `T`) over any field —
//! sound in characteristic 2 because products never cancel (`(δ−v)² ≠ 0`),
//! unlike the additive LogUp sum (`1/(δ−v)+1/(δ−v)=0`). The identity is
//! ```text
//!   ∏_i (δ − a_i)  =  ∏_t (δ − fp_t)^{m_t}
//! ```
//! with the multiplicity-weighted table side realised by **binary
//! multiplicity** (`m_t = Σ_b m_{t,b} 2^b`):
//! ```text
//!   ∏_t (δ − fp_t)^{m_t}  =  ∏_t ∏_b [ m_{t,b} ? (δ − fp_t)^{2^b} : 1 ].
//! ```
//! Each side is one product tree; **equality of the two roots is the lookup
//! check**. `(δ − fp_t)^{2^b}` is formed by repeated squaring.
//!
//! Roles of the inputs:
//! - `a_i` (`witness_fps`) — witness fingerprints, built by the caller from
//!   limb tuples via a compression challenge `γ`; opaque field elements here,
//!   bound externally by a `ψ_z` read-off of the committed words.
//! - `fp_t` (`table_fps`) — the public table fingerprints.
//! - `m_{t,b}` (`mult_bits`) — committed multiplicity bits (LSB-first).
//! - `δ` (`delta`) — grand-product challenge, sampled by the caller from the
//!   transcript *after* the witness and multiplicities are fixed.
//!
//! [`prove_lookup`] / [`verify_lookup`] return a [`LookupBinding`]: the two
//! leaf-layer points and claimed leaf-MLE evaluations the caller must bind
//! (witness side → `ψ_z` read-off; table side → the public `fp_t` + committed
//! `m_{t,b}`, the "structured-table" obligation).

use crypto_primitives::{FromPrimitiveWithConfig, PrimeField};
use num_traits::Zero;
use zinc_transcript::traits::{ConstTranscribable, Transcript};
use zinc_utils::inner_transparent_field::InnerTransparentField;

use super::gkr_product::{
    ProductTreeError, ProductTreeProof, prove_product_tree, verify_product_tree,
};

/// Proof of `{a_i} ⊆ T`: the two product-tree proofs whose roots must match.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GrandProductLookupProof<F: PrimeField> {
    /// Product tree for `∏_i (δ − a_i)`.
    pub witness: ProductTreeProof<F>,
    /// Product tree for `∏_t ∏_b [ m_{t,b} ? (δ − fp_t)^{2^b} : 1 ]`.
    pub table: ProductTreeProof<F>,
}

/// Leaf-binding obligations returned by prove/verify. The caller checks
/// `eval_w` against the witness fingerprint MLE at `r_w` (a `ψ_z` read-off)
/// and `eval_t` against the table-leaf MLE at `r_t` (public `fp_t` +
/// committed `m_{t,b}`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LookupBinding<F: PrimeField> {
    pub r_w: Vec<F>,
    pub eval_w: F,
    pub r_t: Vec<F>,
    pub eval_t: F,
}

/// Failure modes of [`verify_lookup`].
#[derive(Debug, thiserror::Error)]
pub enum LookupError {
    #[error("product tree error")]
    ProductTree(#[from] ProductTreeError),
    #[error("lookup root mismatch: ∏(δ−a_i) != ∏_t(δ−fp_t)^(m_t) — witness not contained in table")]
    RootMismatch,
}

/// Witness leaves `δ − a_i`.
pub fn witness_leaves<F: InnerTransparentField>(witness_fps: &[F], delta: &F) -> Vec<F> {
    witness_fps.iter().map(|a| delta.clone() - a).collect()
}

/// Table leaves `[ m_{t,b} ? (δ − fp_t)^{2^b} : 1 ]` flattened over `(t, b)`.
/// `mult_bits[t]` is the binary multiplicity of table row `t`, LSB-first;
/// all rows must share the same bit-width `K`.
#[allow(clippy::arithmetic_side_effects)]
pub fn table_leaves<F: InnerTransparentField>(
    table_fps: &[F],
    mult_bits: &[Vec<bool>],
    delta: &F,
    field_cfg: &F::Config,
) -> Vec<F> {
    let one = F::one_with_cfg(field_cfg);
    let mut leaves = Vec::with_capacity(table_fps.len().saturating_mul(mult_bits.first().map_or(0, Vec::len)));
    for (fp, bits) in table_fps.iter().zip(mult_bits.iter()) {
        // pow = (δ − fp)^{2^b}, squared after each bit.
        let mut pow = delta.clone() - fp;
        for &bit in bits.iter() {
            leaves.push(if bit { pow.clone() } else { one.clone() });
            pow = pow.clone() * &pow;
        }
    }
    leaves
}

/// Pad `leaves` up to the next power of two with the multiplicative identity
/// `1` (a no-op factor that leaves the product unchanged).
fn pad_to_pow2<F: Clone>(mut leaves: Vec<F>, one: &F) -> Vec<F> {
    let target = leaves.len().max(1).next_power_of_two();
    leaves.resize(target, one.clone());
    leaves
}

/// Prove `{a_i} ⊆ T` from the prepared witness and table leaves.
pub fn prove_lookup<F>(
    transcript: &mut impl Transcript,
    witness_leaves: Vec<F>,
    table_leaves: Vec<F>,
    field_cfg: &F::Config,
) -> (GrandProductLookupProof<F>, LookupBinding<F>)
where
    F: InnerTransparentField + FromPrimitiveWithConfig + Send + Sync,
    F::Inner: ConstTranscribable + Zero + Default + Send + Sync,
    F::Modulus: ConstTranscribable,
    F::Config: Sync,
{
    let one = F::one_with_cfg(field_cfg);
    let witness_leaves = pad_to_pow2(witness_leaves, &one);
    let table_leaves = pad_to_pow2(table_leaves, &one);

    let (witness, r_w, eval_w) = prove_product_tree(transcript, witness_leaves, field_cfg);
    let (table, r_t, eval_t) = prove_product_tree(transcript, table_leaves, field_cfg);

    (
        GrandProductLookupProof { witness, table },
        LookupBinding { r_w, eval_w, r_t, eval_t },
    )
}

/// Verify `{a_i} ⊆ T`: both product trees, then the root-equality lookup
/// check. The leaf-layer sizes are read from the proofs.
pub fn verify_lookup<F>(
    transcript: &mut impl Transcript,
    proof: &GrandProductLookupProof<F>,
    field_cfg: &F::Config,
) -> Result<LookupBinding<F>, LookupError>
where
    F: InnerTransparentField + FromPrimitiveWithConfig + Send + Sync,
    F::Inner: ConstTranscribable + Zero,
    F::Modulus: ConstTranscribable,
{
    let dw = proof.witness.layers.len();
    let dt = proof.table.layers.len();
    let (r_w, eval_w) = verify_product_tree(transcript, &proof.witness, dw, field_cfg)?;
    let (r_t, eval_t) = verify_product_tree(transcript, &proof.table, dt, field_cfg)?;

    if proof.witness.root != proof.table.root {
        return Err(LookupError::RootMismatch);
    }
    Ok(LookupBinding { r_w, eval_w, r_t, eval_t })
}

#[cfg(test)]
mod tests {
    use super::*;
    use zinc_poly::univariate::binary_gf128::BinaryFieldGF128 as Gf;
    use zinc_transcript::Blake3Transcript;

    fn gf(v: u64) -> Gf {
        Gf::from_words([v, 0])
    }

    /// Witness `{10, 20, 20, 40}` drawn from table `{10,20,30,40}` with
    /// multiplicities `{1,2,0,1}`. The two products are equal, so verify
    /// accepts (over GF(2^128) — the real field).
    #[test]
    fn lookup_accepts_contained_multiset() {
        let delta = gf(7);
        let table_fps = vec![gf(10), gf(20), gf(30), gf(40)];
        // LSB-first 2-bit multiplicities: 1=[1,0], 2=[0,1], 0=[0,0], 1=[1,0].
        let mult_bits = vec![
            vec![true, false],
            vec![false, true],
            vec![false, false],
            vec![true, false],
        ];
        let witness_fps = vec![gf(10), gf(20), gf(20), gf(40)];

        let w = witness_leaves(&witness_fps, &delta);
        let t = table_leaves(&table_fps, &mult_bits, &delta, &());

        let mut pt = Blake3Transcript::new();
        let (proof, _bind) = prove_lookup(&mut pt, w, t, &());
        assert_eq!(proof.witness.root, proof.table.root, "products must match for a contained multiset");

        let mut vt = Blake3Transcript::new();
        verify_lookup(&mut vt, &proof, &()).expect("verifier accepts a contained multiset");
    }

    /// Char-2 soundness: an off-table value used an **even** number of times
    /// must still be rejected. Additive LogUp would accept (the two
    /// `1/(δ−v)` terms cancel in char 2); the multiplicative product has
    /// `(δ−v)² ≠ 1`, so the roots differ and verify rejects.
    #[test]
    fn lookup_rejects_off_table_even_multiplicity() {
        let delta = gf(7);
        let table_fps = vec![gf(10), gf(20)];
        let mult_bits = vec![vec![false, false], vec![false, false]]; // honest counts of in-table = 0
        let witness_fps = vec![gf(99), gf(99)]; // off-table value, used twice

        let w = witness_leaves(&witness_fps, &delta);
        let t = table_leaves(&table_fps, &mult_bits, &delta, &());

        let mut pt = Blake3Transcript::new();
        let (proof, _bind) = prove_lookup(&mut pt, w, t, &());
        assert_ne!(proof.witness.root, proof.table.root, "off-table even multiplicity must NOT cancel");

        let mut vt = Blake3Transcript::new();
        let err = verify_lookup(&mut vt, &proof, &()).expect_err("verifier rejects off-table witness");
        assert!(matches!(err, LookupError::RootMismatch));
    }

    /// A cheating multiplicity (claiming a table row is used more often than
    /// it is) also breaks the product equality.
    #[test]
    fn lookup_rejects_wrong_multiplicity() {
        let delta = gf(7);
        let table_fps = vec![gf(10), gf(20)];
        let witness_fps = vec![gf(10), gf(20)]; // each used once
        let mult_bits = vec![vec![false, true], vec![true, false]]; // lies: claims m_10=2, m_20=1

        let w = witness_leaves(&witness_fps, &delta);
        let t = table_leaves(&table_fps, &mult_bits, &delta, &());

        let mut pt = Blake3Transcript::new();
        let (proof, _bind) = prove_lookup(&mut pt, w, t, &());
        assert_ne!(proof.witness.root, proof.table.root);

        let mut vt = Blake3Transcript::new();
        assert!(verify_lookup(&mut vt, &proof, &()).is_err());
    }
}
