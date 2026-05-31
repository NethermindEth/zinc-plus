//! Coefficient-wise Hadamard-product check for binary_poly columns.
//!
//! Proves `W = U ⊙ V` (bitwise AND / coefficient-wise product) for
//! triples of binary_poly witness columns. Each column is written as
//! its bit-slice MLEs `C_b` over `F`, and we run a zerocheck on
//! `Σ_k Σ_b (γ')^k · σ^b · (U_{k,b}·V_{k,b} − W_{k,b}) · eq(r, b')`
//! as a degree-3 group inside the protocol's multi-degree sumcheck.
//!
//! This is the cross-product sibling of [`super::booleanity`]: it reuses
//! the same bit-slice machinery and the same send-and-recombine
//! discharge ([`super::booleanity::verify_bit_decomposition_consistency`]
//! ties the per-slice evals to a single column opening), with the
//! self-product `v·(v−1)` replaced by the two-column product `U·V − W`.
//!
//! Soundness notes (see `protocol/src/f2_hadamard_plan.md`):
//! - `γ'` batches the relations, `σ` batches the 32 coefficient slices;
//!   both must be drawn after the columns are committed.
//! - The per-slice evals are pinned to the committed columns by the
//!   recombination check `Σ_b a^b·v_b(r*) = parent_eval`, whose element
//!   `a` must be fresh *after* the bit-slice evals are absorbed (Wiring R
//!   reuses the main projection α for this by running this zerocheck
//!   before α is sampled).

use crypto_primitives::PrimeField;
use num_traits::Zero;
use std::slice;
use thiserror::Error;
use zinc_poly::{
    EvaluationError,
    mle::{DenseMultilinearExtension, MultilinearExtensionWithConfig},
    utils::{ArithErrors, build_eq_x_r_inner, eq_eval},
};
use zinc_transcript::traits::{ConstTranscribable, Transcript};
use zinc_utils::{inner_transparent_field::InnerTransparentField, powers};

use crate::{
    CombFn,
    sumcheck::{multi_degree::MultiDegreeSumcheckGroup, prover::ProverState as SumcheckProverState},
};

/// A single Hadamard relation `W = U ⊙ V`, identified by the indices of
/// the U, V, W base columns within the bit-slice MLE set. Each column
/// owns `D` consecutive slices in column-major order, so column `c`'s
/// bit `b` lives at flat index `c*D + b`.
#[derive(Clone, Copy, Debug)]
pub struct HadamardTriple {
    pub u_col: usize,
    pub v_col: usize,
    pub w_col: usize,
}

/// Ancillary data produced by [`prepare_hadamard_group`] and consumed by
/// [`finalize_hadamard_prover`].
pub struct HadamardProverAncillary {
    /// Number of bit-slice MLEs in the group (excludes the leading eq_r).
    pub num_bit_slices: usize,
}

/// Ancillary data produced by [`prepare_hadamard_verifier`] and consumed
/// by [`finalize_hadamard_verifier`].
pub struct HadamardVerifierAncillary<F: PrimeField> {
    /// Powers of the relation-batching challenge `[1, γ', …, γ'^{K-1}]`.
    pub gamma_powers: Vec<F>,
    /// Powers of the slice-batching challenge `[1, σ, …, σ^{D-1}]`.
    pub sigma_powers: Vec<F>,
    /// The Hadamard relations (re-derived layout for the closing check).
    pub relations: Vec<HadamardTriple>,
    /// Evaluation point used to build `eq_r` (mirrors the prover).
    pub ic_evaluation_point: Vec<F>,
}

/// Build the degree-3 Hadamard zerocheck group. Samples `γ'` (relation
/// batch) and `σ` (slice batch) from the transcript, in that order, then
/// returns the group plus ancillary data. The group's `poly` is
/// `[eq_r, bit_slice_mles…]` (column-major, `D` slices per column).
///
/// Returns `None` when there is nothing to check.
#[allow(clippy::arithmetic_side_effects)]
pub fn prepare_hadamard_group<F, const D: usize>(
    transcript: &mut impl Transcript,
    bit_slice_mles: Vec<DenseMultilinearExtension<F::Inner>>,
    relations: &[HadamardTriple],
    ic_evaluation_point: &[F],
    field_cfg: &F::Config,
) -> Result<Option<(MultiDegreeSumcheckGroup<F>, HadamardProverAncillary)>, HadamardError<F>>
where
    F: InnerTransparentField + Send + Sync + 'static,
    F::Inner: ConstTranscribable + Send + Sync + Zero + Default + Clone,
    F::Modulus: ConstTranscribable,
{
    if bit_slice_mles.is_empty() || relations.is_empty() {
        return Ok(None);
    }
    debug_assert_eq!(bit_slice_mles.len() % D, 0);
    let num_bit_slices = bit_slice_mles.len();

    let zero = F::zero_with_cfg(field_cfg);
    let one = F::one_with_cfg(field_cfg);

    let gamma_prime: F = transcript.get_field_challenge(field_cfg);
    let sigma: F = transcript.get_field_challenge(field_cfg);
    let gamma_powers: Vec<F> = powers(gamma_prime, one.clone(), relations.len());
    let sigma_powers: Vec<F> = powers(sigma, one, D);

    let eq_r_mle = build_eq_x_r_inner(ic_evaluation_point, field_cfg)?;

    let mut poly: Vec<DenseMultilinearExtension<F::Inner>> =
        Vec::with_capacity(1usize.saturating_add(bit_slice_mles.len()));
    poly.push(eq_r_mle);
    poly.extend(bit_slice_mles);

    let relations_vec = relations.to_vec();
    let comb_fn: CombFn<F> = Box::new(move |mle_values: &[F]| {
        let eq_r = mle_values[0].clone();
        let slices = &mle_values[1..];
        let mut acc = zero.clone();
        for (k, tri) in relations_vec.iter().enumerate() {
            let gpow = gamma_powers[k].clone();
            for b in 0..D {
                let u = slices[tri.u_col * D + b].clone();
                let v = slices[tri.v_col * D + b].clone();
                let w = slices[tri.w_col * D + b].clone();
                // U·V − W ; in characteristic 2 this is U·V + W.
                acc = acc + gpow.clone() * sigma_powers[b].clone() * (u * v - w);
            }
        }
        acc * eq_r
    });

    Ok(Some((
        MultiDegreeSumcheckGroup::new(3, poly, comb_fn),
        HadamardProverAncillary { num_bit_slices },
    )))
}

/// Extract the bit-slice evals at the shared sumcheck point from the
/// Hadamard group's prover state and absorb them. The leading `eq_r` MLE
/// is dropped (the verifier recomputes it). Mirrors
/// [`super::booleanity::finalize_booleanity_prover`].
#[allow(clippy::arithmetic_side_effects)]
pub fn finalize_hadamard_prover<F>(
    transcript: &mut impl Transcript,
    sumcheck_prover_state: SumcheckProverState<F>,
    ancillary: HadamardProverAncillary,
    field_cfg: &F::Config,
) -> Result<Vec<F>, HadamardError<F>>
where
    F: InnerTransparentField,
    F::Inner: ConstTranscribable + Zero,
    F::Modulus: ConstTranscribable,
{
    let last_challenge = sumcheck_prover_state
        .randomness
        .last()
        .expect("sumcheck must have at least one round")
        .clone();

    let mut mles = sumcheck_prover_state.mles;
    let _eq_r_mle = mles.remove(0);
    let bit_slice_evals: Vec<F> = mles
        .into_iter()
        .map(|m| m.evaluate_with_config(slice::from_ref(&last_challenge), field_cfg))
        .collect::<Result<Vec<_>, _>>()?;

    debug_assert_eq!(bit_slice_evals.len(), ancillary.num_bit_slices);

    let mut buf: Vec<u8> = vec![0; F::Inner::NUM_BYTES];
    transcript.absorb_random_field_slice(&bit_slice_evals, &mut buf);

    Ok(bit_slice_evals)
}

/// Pre-sumcheck verifier half: validates the zerocheck's claimed sum is
/// zero and samples `γ'`, `σ` (matching the prover order).
pub fn prepare_hadamard_verifier<F, const D: usize>(
    transcript: &mut impl Transcript,
    claimed_sum: F,
    relations: &[HadamardTriple],
    ic_evaluation_point: &[F],
    field_cfg: &F::Config,
) -> Result<HadamardVerifierAncillary<F>, HadamardError<F>>
where
    F: InnerTransparentField,
    F::Inner: ConstTranscribable,
    F::Modulus: ConstTranscribable,
{
    let zero = F::zero_with_cfg(field_cfg);
    let one = F::one_with_cfg(field_cfg);

    if claimed_sum != zero {
        return Err(HadamardError::NonZeroClaimedSum { got: claimed_sum });
    }

    let gamma_prime: F = transcript.get_field_challenge(field_cfg);
    let sigma: F = transcript.get_field_challenge(field_cfg);
    let gamma_powers = powers(gamma_prime, one.clone(), relations.len());
    let sigma_powers = powers(sigma, one, D);

    Ok(HadamardVerifierAncillary {
        gamma_powers,
        sigma_powers,
        relations: relations.to_vec(),
        ic_evaluation_point: ic_evaluation_point.to_vec(),
    })
}

/// Post-sumcheck verifier half: recomputes
/// `eq_r(r*) · Σ_k Σ_b γ'^k σ^b (U_{k,b}·V_{k,b} − W_{k,b})` from the sent
/// bit-slice evals and checks it equals the sumcheck's expected
/// evaluation, then absorbs the evals. The caller must additionally run
/// [`super::booleanity::verify_bit_decomposition_consistency`] to tie the
/// evals to the committed columns.
#[allow(clippy::arithmetic_side_effects)]
pub fn finalize_hadamard_verifier<F, const D: usize>(
    transcript: &mut impl Transcript,
    bit_slice_evals: &[F],
    shared_point: &[F],
    expected_evaluation: F,
    ancillary: HadamardVerifierAncillary<F>,
    field_cfg: &F::Config,
) -> Result<(), HadamardError<F>>
where
    F: InnerTransparentField,
    F::Inner: ConstTranscribable,
    F::Modulus: ConstTranscribable,
{
    let zero = F::zero_with_cfg(field_cfg);
    let one = F::one_with_cfg(field_cfg);

    let eq_r_value = eq_eval(shared_point, &ancillary.ic_evaluation_point, one)?;

    let mut acc = zero;
    for (k, tri) in ancillary.relations.iter().enumerate() {
        let gpow = ancillary.gamma_powers[k].clone();
        for b in 0..D {
            let u = bit_slice_evals[tri.u_col * D + b].clone();
            let v = bit_slice_evals[tri.v_col * D + b].clone();
            let w = bit_slice_evals[tri.w_col * D + b].clone();
            acc = acc + gpow.clone() * ancillary.sigma_powers[b].clone() * (u * v - w);
        }
    }
    let recomputed = acc * eq_r_value;

    if recomputed != expected_evaluation {
        return Err(HadamardError::SumcheckClaimMismatch {
            got: expected_evaluation,
            expected: recomputed,
        });
    }

    let mut buf: Vec<u8> = vec![0; F::Inner::NUM_BYTES];
    transcript.absorb_random_field_slice(bit_slice_evals, &mut buf);

    Ok(())
}

#[derive(Debug, Error)]
pub enum HadamardError<F: PrimeField> {
    #[error("hadamard zerocheck claimed sum non-zero: {got:?}")]
    NonZeroClaimedSum { got: F },
    #[error("wrong bit-slice evaluation count: got {got}, expected {expected}")]
    WrongBitSliceEvalCount { got: usize, expected: usize },
    #[error("hadamard sumcheck claim mismatch: got {got:?}, expected {expected:?}")]
    SumcheckClaimMismatch { got: F, expected: F },
    #[error("eq_r evaluation failed: {0}")]
    EqEvalError(#[from] ArithErrors),
    #[error("MLE evaluation failed: {0}")]
    MleEvaluationError(#[from] EvaluationError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sumcheck::multi_degree::MultiDegreeSumcheck;
    use crypto_primitives::Field;
    use zinc_poly::univariate::binary_gf128::BinaryFieldGF128;
    use zinc_transcript::Blake3Transcript;

    type Gf = BinaryFieldGF128;

    /// Build column-major bit-slice MLEs (`D` per column) from u32 columns.
    fn build_slices<const D: usize>(
        cols: &[Vec<u32>],
        num_vars: usize,
    ) -> Vec<DenseMultilinearExtension<<Gf as Field>::Inner>> {
        let cfg = &();
        let one_i = Gf::one_with_cfg(cfg).into_inner();
        let zero_i = Gf::zero_with_cfg(cfg).into_inner();
        let n = 1usize << num_vars;
        let mut out = Vec::new();
        for col in cols {
            for b in 0..D {
                let evals: Vec<_> = (0..n)
                    .map(|row| {
                        if (col[row] >> b) & 1 == 1 {
                            one_i.clone()
                        } else {
                            zero_i.clone()
                        }
                    })
                    .collect();
                out.push(DenseMultilinearExtension::from_evaluations_vec(
                    num_vars,
                    evals,
                    zero_i.clone(),
                ));
            }
        }
        out
    }

    fn ic_point() -> Vec<Gf> {
        vec![
            Gf::from_words([3, 0]),
            Gf::from_words([5, 0]),
            Gf::from_words([7, 0]),
        ]
    }

    #[test]
    fn hadamard_accepts_valid_and_detects_corruption() {
        let cfg = &();
        const D: usize = 4;
        let num_vars = 3;
        let relations = [HadamardTriple {
            u_col: 0,
            v_col: 1,
            w_col: 2,
        }];
        let ic = ic_point();

        let u: Vec<u32> = vec![0b1011, 0b0110, 0b1111, 0b0001, 0b1010, 0b0101, 0b1100, 0b0011];
        let v: Vec<u32> = vec![0b1101, 0b1010, 0b0111, 0b1001, 0b0110, 0b1111, 0b1000, 0b0101];
        let w: Vec<u32> = u.iter().zip(&v).map(|(a, b)| a & b).collect();

        // ---- honest prover ----
        let slices = build_slices::<D>(&[u.clone(), v.clone(), w.clone()], num_vars);
        let mut pt = Blake3Transcript::new();
        let (group, panc) =
            prepare_hadamard_group::<Gf, D>(&mut pt, slices, &relations, &ic, cfg)
                .unwrap()
                .unwrap();
        let (proof, mut states) =
            MultiDegreeSumcheck::<Gf>::prove_as_subprotocol(&mut pt, vec![group], num_vars, cfg);
        // Honest AND ⇒ zerocheck sum is zero.
        assert_eq!(proof.claimed_sums()[0], Gf::zero());
        let bse = finalize_hadamard_prover::<Gf>(&mut pt, states.remove(0), panc, cfg).unwrap();

        // ---- verifier ----
        let mut vt = Blake3Transcript::new();
        let vanc = prepare_hadamard_verifier::<Gf, D>(
            &mut vt,
            proof.claimed_sums()[0],
            &relations,
            &ic,
            cfg,
        )
        .unwrap();
        let subclaims =
            MultiDegreeSumcheck::<Gf>::verify_as_subprotocol(&mut vt, num_vars, &proof, cfg)
                .expect("sumcheck verify");
        finalize_hadamard_verifier::<Gf, D>(
            &mut vt,
            &bse,
            subclaims.point(),
            subclaims.expected_evaluations()[0],
            vanc,
            cfg,
        )
        .expect("hadamard finalize");

        // ---- corrupt W: the zerocheck must see a non-zero claimed sum ----
        let mut w_bad = w.clone();
        w_bad[0] ^= 1; // flip a bit so W ≠ U⊙V
        let slices2 = build_slices::<D>(&[u, v, w_bad], num_vars);
        let mut pt2 = Blake3Transcript::new();
        let (group2, _) =
            prepare_hadamard_group::<Gf, D>(&mut pt2, slices2, &relations, &ic, cfg)
                .unwrap()
                .unwrap();
        let (proof2, _) =
            MultiDegreeSumcheck::<Gf>::prove_as_subprotocol(&mut pt2, vec![group2], num_vars, cfg);
        assert_ne!(proof2.claimed_sums()[0], Gf::zero());

        // And the verifier's claimed-sum gate rejects it.
        let mut vt2 = Blake3Transcript::new();
        let rejected = prepare_hadamard_verifier::<Gf, D>(
            &mut vt2,
            proof2.claimed_sums()[0],
            &relations,
            &ic,
            cfg,
        );
        assert!(matches!(rejected, Err(HadamardError::NonZeroClaimedSum { .. })));
    }
}
