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

use crypto_primitives::{FromPrimitiveWithConfig, PrimeField};
use num_traits::Zero;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::slice;
use thiserror::Error;
use zinc_poly::{
    EvaluationError,
    mle::{DenseMultilinearExtension, MultilinearExtensionWithConfig},
    univariate::binary::BinaryPoly,
    utils::{ArithErrors, build_eq_x_r_inner, eq_eval},
};
use zinc_transcript::traits::{ConstTranscribable, Transcript};
use zinc_utils::{cfg_into_iter, cfg_iter, inner_transparent_field::InnerTransparentField, powers};

use crate::{
    CombFn,
    sumcheck::{
        multi_degree::{MultiDegreeSumcheckGroup, Round1FastPath, Round1Output},
        prover::ProverState as SumcheckProverState,
    },
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

/// Round-1 fast path for the degree-3 Hadamard zerocheck — the
/// `GF(2^128)` sibling of [`super::booleanity::BooleanityRound1FastPath`].
///
/// Reads the **packed** operand columns (`U_k, V_k, W_k` as
/// `BinaryPoly<D>`, 3 per relation in operand order) directly, so round 1
/// never materialises the 1536 full-width `F::Inner` bit-slice MLEs (the
/// 24 GB-at-nvars=20 blowup). `fold_with_r1` emits the half-size folded
/// slices for rounds 2..n in the `comb_fn`'s expected order
/// (`[eq_r, slice_0, …]`, `slice = operand·D + bit`).
///
/// Boundary points are `F::from_with_cfg(k)` (the field element `X` for
/// `k = 2`, `X+1` for `k = 3`) — **not** `one + one`: over a binary field
/// `2 = 0`, `3 = 1` collapse (see `sumcheck::prover` and the degree-2
/// `F2EqColRound1FastPath`). The honest round poly has `M(0) = M(1) = 0`
/// (on the hypercube `U_b·V_b = W_b`), so only `M(2), M(3)` are computed.
pub struct HadamardRound1FastPath<F: PrimeField, const D: usize> {
    /// Packed operand columns, 3 per relation in order `U_0, V_0, W_0, U_1, …`.
    operand_cols: Vec<DenseMultilinearExtension<BinaryPoly<D>>>,
    /// `[1, γ', …, γ'^{K-1}]` (relation batch).
    gamma_powers: Vec<F>,
    /// `[1, σ, …, σ^{D-1}]` (slice batch).
    sigma_powers: Vec<F>,
    /// `eq(b', ic_evaluation_point[1..])` for `b' ∈ {0,1}^{num_vars-1}`.
    eq_other_table: Vec<F::Inner>,
    /// `ic_evaluation_point[0]`.
    ic_ep_0: F,
    num_vars: usize,
    /// `K` (= `operand_cols.len() / 3`).
    num_relations: usize,
}

impl<F, const D: usize> Round1FastPath<F> for HadamardRound1FastPath<F, D>
where
    F: InnerTransparentField + FromPrimitiveWithConfig + Send + Sync + 'static,
    F::Inner: Send + Sync + Zero + Default + Clone,
{
    #[allow(clippy::arithmetic_side_effects)]
    fn round_message(&self, _round: usize, _prior: &[F], config: &F::Config) -> Round1Output<F> {
        let zero = F::zero_with_cfg(config);
        let one = F::one_with_cfg(config);
        let half = 1usize << (self.num_vars - 1);
        debug_assert_eq!(self.operand_cols.len(), 3 * self.num_relations);
        debug_assert_eq!(self.eq_other_table.len(), half);

        // T(t) = Σ_{b'} E_other(b') · Σ_k γ'^k Σ_b σ^b (U_b(t)·V_b(t) − W_b(t)),
        // where each operand bit folds over variable 0 as
        // X_b(t) = (1−t)·X_b(0,b') + t·X_b(1,b') ∈ {0, t, 1−t, 1} for the
        // four `{0,1}²` row pairs.
        let eval_t = |t: &F| -> F {
            let one_minus_t = one.clone() - t.clone();
            let per_bprime: Vec<F> = cfg_into_iter!(0..half)
                .map(|b_prime| {
                    let lift = |x0: bool, x1: bool| -> F {
                        match (x0, x1) {
                            (false, false) => zero.clone(),
                            (true, true) => one.clone(),
                            (false, true) => t.clone(),
                            (true, false) => one_minus_t.clone(),
                        }
                    };
                    let mut g = zero.clone();
                    for k in 0..self.num_relations {
                        let u_col = &self.operand_cols[3 * k];
                        let v_col = &self.operand_cols[3 * k + 1];
                        let w_col = &self.operand_cols[3 * k + 2];
                        let u0 = &u_col.evaluations[2 * b_prime];
                        let u1 = &u_col.evaluations[2 * b_prime + 1];
                        let v0 = &v_col.evaluations[2 * b_prime];
                        let v1 = &v_col.evaluations[2 * b_prime + 1];
                        let w0 = &w_col.evaluations[2 * b_prime];
                        let w1 = &w_col.evaluations[2 * b_prime + 1];
                        let mut bit_acc = zero.clone();
                        for (b, (((u0b, u1b), (v0b, v1b)), (w0b, w1b))) in u0
                            .iter()
                            .zip(u1.iter())
                            .zip(v0.iter().zip(v1.iter()))
                            .zip(w0.iter().zip(w1.iter()))
                            .enumerate()
                        {
                            let ut = lift(u0b.into_inner(), u1b.into_inner());
                            let vt = lift(v0b.into_inner(), v1b.into_inner());
                            let wt = lift(w0b.into_inner(), w1b.into_inner());
                            bit_acc = bit_acc + self.sigma_powers[b].clone() * (ut * vt - wt);
                        }
                        g = g + self.gamma_powers[k].clone() * bit_acc;
                    }
                    let e_other =
                        F::new_unchecked_with_cfg(self.eq_other_table[b_prime].clone(), config);
                    g * e_other
                })
                .collect();
            per_bprime.into_iter().fold(zero.clone(), |a, b| a + b)
        };

        // eq(t, r0) = (1−t)(1−r0) + t·r0.
        let eq_at = |t: &F| -> F {
            (one.clone() - t.clone()) * (one.clone() - self.ic_ep_0.clone())
                + t.clone() * self.ic_ep_0.clone()
        };

        // Evaluate M(t) = eq(t,r0)·T(t) at the four boundary points
        // {0, 1, X, X+1}. We do NOT assume M(0)=M(1)=0: unlike booleanity
        // (`v·(v−1)` is structurally 0 for any bit), the Hadamard term
        // `U_b·V_b − W_b` is non-zero on a *corrupt* row (W_b ≠ U_b·V_b),
        // so the round-1 message must match the generic path for the
        // verifier to reject it (cf. `corrupt_w_is_rejected`,
        // `adder_rejects_wrong_sum`). For honest traces M(0)=M(1)=0.
        let two = F::from_with_cfg(2u64, config);
        let three = F::from_with_cfg(3u64, config);
        let m0 = eq_at(&zero) * eval_t(&zero);
        let m1 = eq_at(&one) * eval_t(&one);
        let m2 = eq_at(&two) * eval_t(&two);
        let m3 = eq_at(&three) * eval_t(&three);

        Round1Output {
            asserted_sum: m0 + m1.clone(),
            tail_evaluations: vec![m1, m2, m3],
        }
    }

    #[allow(clippy::arithmetic_side_effects)]
    fn fold(
        self: Box<Self>,
        challenges: &[F],
        config: &F::Config,
    ) -> Vec<DenseMultilinearExtension<F::Inner>> {
        let r_1 = &challenges[0];
        let one = F::one_with_cfg(config);
        let one_minus_r1 = one.clone() - r_1.clone();
        let half = 1usize << (self.num_vars - 1);

        // eq_r folded over variable 0: ((1−r1)(1−r0) + r1·r0) · E_other(b').
        let eq_scalar = one_minus_r1.clone() * (one.clone() - self.ic_ep_0.clone())
            + r_1.clone() * self.ic_ep_0.clone();
        let eq_folded_evals: Vec<F::Inner> = cfg_iter!(self.eq_other_table)
            .map(|e| {
                let lifted = F::new_unchecked_with_cfg(e.clone(), config);
                (eq_scalar.clone() * lifted).into_inner()
            })
            .collect();

        let zero_inner = F::zero_with_cfg(config).into_inner();
        let one_inner = one.inner().clone();
        let r1_inner = r_1.inner().clone();
        let one_minus_r1_inner = one_minus_r1.inner().clone();

        let mut mles: Vec<DenseMultilinearExtension<F::Inner>> =
            Vec::with_capacity(1 + self.operand_cols.len() * D);
        mles.push(DenseMultilinearExtension {
            num_vars: self.num_vars - 1,
            evaluations: eq_folded_evals,
        });

        // Per operand column, fold each of its D bits over variable 0:
        // (A,B) ∈ {0,1}² → {0, r1, 1−r1, 1}. Emit D slices in bit order so
        // the layout matches the generic `build_all_operand_slices`.
        for col in &self.operand_cols {
            let mut per_bit: Vec<Vec<F::Inner>> =
                (0..D).map(|_| Vec::with_capacity(half)).collect();
            for b_prime in 0..half {
                let row_a = &col.evaluations[2 * b_prime];
                let row_b = &col.evaluations[2 * b_prime + 1];
                for (bit_idx, (a_bit, b_bit)) in row_a.iter().zip(row_b.iter()).enumerate() {
                    let v = match (a_bit.into_inner(), b_bit.into_inner()) {
                        (false, false) => zero_inner.clone(),
                        (true, true) => one_inner.clone(),
                        (false, true) => r1_inner.clone(),
                        (true, false) => one_minus_r1_inner.clone(),
                    };
                    per_bit[bit_idx].push(v);
                }
            }
            for evals in per_bit {
                mles.push(DenseMultilinearExtension {
                    num_vars: self.num_vars - 1,
                    evaluations: evals,
                });
            }
        }
        mles
    }
}

/// Like [`prepare_hadamard_group`], but supplies a round-1
/// [`HadamardRound1FastPath`] instead of materialising the bit-slice MLEs:
/// takes the **packed** operand columns (`U_k, V_k, W_k` as `BinaryPoly<D>`,
/// 3 per relation in operand order). Samples `γ'`, `σ` in the same order as
/// [`prepare_hadamard_group`], so the two are transcript-interchangeable.
/// The `comb_fn` (rounds 2..n) is identical; only round 1 differs.
#[allow(clippy::arithmetic_side_effects)]
pub fn prepare_hadamard_group_with_fast<F, const D: usize>(
    transcript: &mut impl Transcript,
    operand_cols: Vec<DenseMultilinearExtension<BinaryPoly<D>>>,
    relations: &[HadamardTriple],
    ic_evaluation_point: &[F],
    field_cfg: &F::Config,
) -> Result<Option<(MultiDegreeSumcheckGroup<F>, HadamardProverAncillary)>, HadamardError<F>>
where
    F: InnerTransparentField + FromPrimitiveWithConfig + Send + Sync + 'static,
    F::Inner: ConstTranscribable + Send + Sync + Zero + Default + Clone,
    F::Modulus: ConstTranscribable,
{
    if operand_cols.is_empty() || relations.is_empty() {
        return Ok(None);
    }
    debug_assert_eq!(operand_cols.len(), 3 * relations.len());
    debug_assert!(
        !ic_evaluation_point.is_empty(),
        "ic_evaluation_point must be non-empty (num_vars >= 1)"
    );
    let num_bit_slices = operand_cols.len() * D;
    let num_vars = ic_evaluation_point.len();

    let zero = F::zero_with_cfg(field_cfg);
    let one = F::one_with_cfg(field_cfg);

    let gamma_prime: F = transcript.get_field_challenge(field_cfg);
    let sigma: F = transcript.get_field_challenge(field_cfg);
    let gamma_powers: Vec<F> = powers(gamma_prime, one.clone(), relations.len());
    let sigma_powers: Vec<F> = powers(sigma, one.clone(), D);

    let eq_other_table: Vec<F::Inner> = if num_vars >= 2 {
        build_eq_x_r_inner(&ic_evaluation_point[1..], field_cfg)?.evaluations
    } else {
        vec![one.inner().clone()]
    };

    let fast_path: Box<dyn Round1FastPath<F>> = Box::new(HadamardRound1FastPath::<F, D> {
        operand_cols,
        gamma_powers: gamma_powers.clone(),
        sigma_powers: sigma_powers.clone(),
        eq_other_table,
        ic_ep_0: ic_evaluation_point[0].clone(),
        num_vars,
        num_relations: relations.len(),
    });

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
                acc = acc + gpow.clone() * sigma_powers[b].clone() * (u * v - w);
            }
        }
        acc * eq_r
    });

    Ok(Some((
        MultiDegreeSumcheckGroup::with_round_1_fast(3, Vec::new(), comb_fn, fast_path),
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

    /// Build packed `BinaryPoly<D>` operand columns from u32 columns.
    /// Uses `BinaryPoly::new([Boolean; D])` (feature-agnostic — `From<u32>`
    /// only exists on the `simd` `BinaryU64Poly` alias).
    fn build_cols<const D: usize>(
        cols: &[Vec<u32>],
        num_vars: usize,
    ) -> Vec<DenseMultilinearExtension<BinaryPoly<D>>> {
        use crypto_primitives::semiring::boolean::Boolean;
        cols.iter()
            .map(|col| {
                DenseMultilinearExtension::from_evaluations_vec(
                    num_vars,
                    col.iter()
                        .map(|&x| {
                            let coeffs: [Boolean; D] =
                                std::array::from_fn(|i| Boolean::new((x >> i) & 1 != 0));
                            BinaryPoly::<D>::new(coeffs)
                        })
                        .collect(),
                    BinaryPoly::<D>::default(),
                )
            })
            .collect()
    }

    /// The `Round1FastPath` contract: `prepare_hadamard_group_with_fast`
    /// must emit a `MultiDegreeSumcheckProof` **identical** to the generic
    /// `prepare_hadamard_group` on the same data (same transcript ⇒ same
    /// γ'/σ, same round messages ⇒ same challenges ⇒ same proof). Covers
    /// both round 1 (the fast path) and rounds 2..n (the folded slices).
    #[test]
    fn fast_path_matches_generic() {
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

        // Generic (materialised slices) path.
        let slices = build_slices::<D>(&[u.clone(), v.clone(), w.clone()], num_vars);
        let mut pt_g = Blake3Transcript::new();
        let (group_g, _) = prepare_hadamard_group::<Gf, D>(&mut pt_g, slices, &relations, &ic, cfg)
            .unwrap()
            .unwrap();
        let (proof_g, _) =
            MultiDegreeSumcheck::<Gf>::prove_as_subprotocol(&mut pt_g, vec![group_g], num_vars, cfg);

        // Fast (packed operand columns, round-1 fast path) path.
        let cols = build_cols::<D>(&[u, v, w], num_vars);
        let mut pt_f = Blake3Transcript::new();
        let (group_f, _) =
            prepare_hadamard_group_with_fast::<Gf, D>(&mut pt_f, cols, &relations, &ic, cfg)
                .unwrap()
                .unwrap();
        let (proof_f, _) =
            MultiDegreeSumcheck::<Gf>::prove_as_subprotocol(&mut pt_f, vec![group_f], num_vars, cfg);

        assert_eq!(
            proof_g, proof_f,
            "fast-path proof must be identical to the generic-path proof"
        );
    }

    /// Validates the `v=2` small-value **prefix-polynomial** math (the core
    /// of the multi-round-skip prover) in isolation, before any framework
    /// wiring: compute `q(X_1,X_2) = Σ_{x''} comb(X_1,X_2,x'')` over the
    /// `{0, 1, X, X+1}²` grid, derive the round-1 and round-2 messages from
    /// it, and assert they equal a faithful generic 2-round run's
    /// `round_tails`. A bug here localises to the prefix math, not the
    /// sumcheck framework.
    #[test]
    fn prefix_v2_matches_generic() {
        use crypto_primitives::FromWithConfig;
        let cfg = &();
        const D: usize = 4;
        let num_vars = 4;
        let relations = [HadamardTriple {
            u_col: 0,
            v_col: 1,
            w_col: 2,
        }];
        let ic = vec![
            Gf::from_words([3, 0]),
            Gf::from_words([5, 0]),
            Gf::from_words([7, 0]),
            Gf::from_words([11, 0]),
        ];
        let n = 1usize << num_vars;
        let u: Vec<u32> = (0..n).map(|i| (0xACE1u32.wrapping_mul(i as u32 + 1)) & 0xF).collect();
        let v: Vec<u32> = (0..n).map(|i| (0x5A5Bu32.wrapping_mul(i as u32 + 3)) & 0xF).collect();
        let w: Vec<u32> = u.iter().zip(&v).map(|(a, b)| a & b).collect();
        let slices = build_slices::<D>(&[u, v, w], num_vars);

        // Faithful generic run: capture round-1/2 message tails + r_1.
        let mut pt = Blake3Transcript::new();
        let (group, _) =
            prepare_hadamard_group::<Gf, D>(&mut pt, slices.clone(), &relations, &ic, cfg)
                .unwrap()
                .unwrap();
        let (proof, states) =
            MultiDegreeSumcheck::<Gf>::prove_as_subprotocol(&mut pt, vec![group], num_vars, cfg);
        let tails = proof.round_tails(0);
        let r_1 = states[0].randomness[0];

        // Re-derive σ (prepare samples γ' then σ from a fresh transcript).
        let mut st = Blake3Transcript::new();
        let _gamma: Gf = st.get_field_challenge(cfg);
        let sigma: Gf = st.get_field_challenge(cfg);
        let sigma_powers = powers(sigma, Gf::one(), D);

        // Grid points {0, 1, X, X+1} — the GF(2^128) sumcheck boundary.
        let pts = [
            Gf::zero(),
            Gf::one(),
            Gf::from_with_cfg(2u64, cfg),
            Gf::from_with_cfg(3u64, cfg),
        ];
        let eq1 = |x: &Gf, e: &Gf| -> Gf {
            (Gf::one() - *x) * (Gf::one() - *e) + *x * *e
        };
        // Bilinear interp of slice `s` over (X_1,X_2) at (pts[i],pts[j]),
        // high vars fixed to x'' (rows x''·4 .. x''·4+3 = the 4 corners).
        let slice_at = |s: usize, i: usize, j: usize, xpp: usize| -> Gf {
            let base = 4 * xpp;
            let lift = |idx: usize| Gf::new_unchecked_with_cfg(slices[s].evaluations[idx].clone(), cfg);
            let (sx, tx) = (pts[i], pts[j]);
            let oms = Gf::one() - sx;
            let omt = Gf::one() - tx;
            oms * omt * lift(base) + sx * omt * lift(base + 1) + oms * tx * lift(base + 2) + sx * tx * lift(base + 3)
        };

        // q[i][j] = Σ_{x''} eq_r(pts[i],pts[j],x'') · Σ_b σ^b (U_b·V_b − W_b)
        let mut q = [[Gf::zero(); 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                let mut acc = Gf::zero();
                for xpp in 0..4 {
                    let b2 = if xpp & 1 == 1 { Gf::one() } else { Gf::zero() };
                    let b3 = if (xpp >> 1) & 1 == 1 { Gf::one() } else { Gf::zero() };
                    let eq_r = eq1(&pts[i], &ic[0]) * eq1(&pts[j], &ic[1]) * eq1(&b2, &ic[2]) * eq1(&b3, &ic[3]);
                    let mut inner = Gf::zero();
                    for b in 0..D {
                        let ub = slice_at(b, i, j, xpp);
                        let vb = slice_at(D + b, i, j, xpp);
                        let wb = slice_at(2 * D + b, i, j, xpp);
                        inner = inner + sigma_powers[b] * (ub * vb - wb);
                    }
                    acc = acc + eq_r * inner;
                }
                q[i][j] = acc;
            }
        }

        // s_1(p) = Σ_{X_2∈{0,1}} q(p, X_2), tail at p ∈ {1, X, X+1}.
        let s1_tail: Vec<Gf> = (1..4).map(|i| q[i][0] + q[i][1]).collect();
        assert_eq!(s1_tail.as_slice(), tails[0], "round-1 message mismatch");

        // s_2(qq) = q(r_1, qq): Lagrange-interp q's X_1-column to r_1.
        let lagrange = |col: &[Gf; 4], r: &Gf| -> Gf {
            let mut acc = Gf::zero();
            for a in 0..4 {
                let mut num = Gf::one();
                let mut den = Gf::one();
                for k in 0..4 {
                    if a != k {
                        num = num * (*r - pts[k]);
                        den = den * (pts[a] - pts[k]);
                    }
                }
                acc = acc + col[a] * num * den.inverse();
            }
            acc
        };
        let s2_tail: Vec<Gf> = (1..4)
            .map(|j| lagrange(&[q[0][j], q[1][j], q[2][j], q[3][j]], &r_1))
            .collect();
        assert_eq!(s2_tail.as_slice(), tails[1], "round-2 message mismatch");
    }
}
