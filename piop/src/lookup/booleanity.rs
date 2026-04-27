//! Algebraic booleanity check for binary_poly columns.
//!
//! Proves that each entry of a `binary_poly` witness column `v` is a
//! binary polynomial of degree `< D` by writing `v = Σ_{i=0}^{D-1} X^i ·
//! v_i` for bit-slice MLEs `v_i` over `F`, and running a zerocheck on
//! `Σ_k α^k · v_k(b) · (v_k(b) - 1) · eq(r, b)` using a dedicated
//! degree-3 group inside the protocol's multi-degree sumcheck.
//!
//! Running this as a *separate* group (not folded into the CPR group)
//! avoids paying the CPR's higher per-variable degree (`max_degree + 2`)
//! on the booleanity term — the booleanity-only group is degree 3, so
//! its `comb_fn` is invoked at 4 evaluation points per round instead of
//! `max_degree + 3`. For SHA-style UAIRs (max_degree ≥ 6) with hundreds
//! of bit-slice MLEs, this is a 2–2.5× saving on step 4 alone.

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
use zinc_utils::{
    cfg_into_iter, cfg_iter, inner_transparent_field::InnerTransparentField, powers,
};

use crate::{
    CombFn,
    sumcheck::{
        multi_degree::{MultiDegreeSumcheckGroup, Round1FastPath, Round1Output},
        prover::ProverState as SumcheckProverState,
    },
};

/// Build bit-slice MLEs over `F::Inner` for every binary_poly column.
///
/// Output ordering is **flat, column-major-then-bit-major**: index
/// `col_idx * D + bit_idx` is `MLE<F::Inner>` whose evaluations are the
/// `bit_idx`-th bit of each row's `BinaryPoly<D>` cast to 0/1 in `F::Inner`.
///
/// Length: `trace_binary_poly.len() * D`.
#[allow(clippy::arithmetic_side_effects)]
pub fn compute_bit_slices_flat<F, const D: usize>(
    trace_binary_poly: &[DenseMultilinearExtension<BinaryPoly<D>>],
    field_cfg: &F::Config,
) -> Vec<DenseMultilinearExtension<F::Inner>>
where
    F: InnerTransparentField,
    F::Inner: Clone + Send + Sync,
{
    let zero_inner = F::zero_with_cfg(field_cfg).into_inner();
    let one_inner = F::one_with_cfg(field_cfg).into_inner();

    cfg_iter!(trace_binary_poly)
        .flat_map(|col| {
            let num_vars = col.num_vars;
            // Per-column transpose: bit_evals[bit_idx][row_idx] = bit
            let mut bit_evals: Vec<Vec<F::Inner>> = (0..D)
                .map(|_| Vec::with_capacity(col.evaluations.len()))
                .collect();
            for bp in &col.evaluations {
                for (bit_idx, coeff) in bp.iter().enumerate() {
                    bit_evals[bit_idx].push(if coeff.into_inner() {
                        one_inner.clone()
                    } else {
                        zero_inner.clone()
                    });
                }
            }
            bit_evals
                .into_iter()
                .map(move |evaluations| DenseMultilinearExtension {
                    evaluations,
                    num_vars,
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Round-1 fast path
// ---------------------------------------------------------------------------

/// Round-1 fast path for the booleanity zerocheck.
///
/// Pre-fold, every bit-slice value `v_i^{(k)}(b)` lies in `{0, 1} ⊂ F`.
/// In all four `(A, B) ∈ {0, 1}²` cases, the linear interpolation
/// `v(X, b') = (1 − X) A + X B` satisfies
///
/// ```text
/// v(X, b') · (v(X, b') − 1) = (A ⊕ B) · X(X − 1)
/// ```
///
/// (Cases (0, 0) and (1, 1): the polynomial is identically zero. Cases
/// (0, 1) and (1, 0): both expand to `X(X − 1)`.) The eq factor splits
/// as `eq_r*(X, b') = e_0(X, ic_ep_0) · E_other(b')`, so the round-1
/// polynomial collapses to
///
/// ```text
/// p_1(X) = e_0(X, ic_ep_0) · X(X − 1) · T_1
/// T_1 = Σ_{b'} S(b') · E_other(b')
/// S(b') = Σ_{(k, i)} α^{k·D + i} · (A_{k,i,b'} ⊕ B_{k,i,b'})
/// ```
///
/// The prover sends `[p_1(1), p_1(2), p_1(3)]`. `p_1(1) = 0` because
/// `X(X − 1)` vanishes at X = 1. `p_1(0) = 0` (also vanishes), so
/// the asserted sum `p_1(0) + p_1(1) = 0`. Computing `T_1` is one pass
/// over `2^(num_vars − 1)` row pairs, doing `K · D` bit-XOR-tests per
/// row pair — no F-multiplications inside the inner loop.
pub struct BooleanityRound1FastPath<F: PrimeField, const D: usize> {
    /// Owned clones of the binary_poly trace columns. Read directly to
    /// avoid materializing F-valued bit-slice MLEs at full size.
    binary_cols: Vec<DenseMultilinearExtension<BinaryPoly<D>>>,
    /// `[1, α, α², ..., α^{K·D − 1}]`.
    alpha_powers: Vec<F>,
    /// `eq(b', ic_evaluation_point[1..])` for `b' ∈ {0, 1}^(num_vars − 1)`.
    eq_other_table: Vec<F::Inner>,
    /// `ic_evaluation_point[0]`.
    ic_ep_0: F,
    /// Number of variables of the parent sumcheck.
    num_vars: usize,
}

impl<F, const D: usize> Round1FastPath<F> for BooleanityRound1FastPath<F, D>
where
    F: InnerTransparentField + Send + Sync + 'static,
    F::Inner: Send + Sync + Zero + Default,
{
    fn round_1_message(&self, config: &F::Config) -> Round1Output<F> {
        debug_assert_eq!(
            self.alpha_powers.len(),
            self.binary_cols.len() * D,
            "alpha_powers length must match K·D"
        );
        debug_assert_eq!(
            self.eq_other_table.len(),
            1usize << self.num_vars.saturating_sub(1),
            "eq_other_table size must be 2^(num_vars - 1)"
        );

        let zero = F::zero_with_cfg(config);
        let one = F::one_with_cfg(config);
        let half = 1usize << (self.num_vars - 1);

        // T_1 = Σ_{b'} S(b') · E_other(b'). Compute S(b') row pair by row
        // pair: for each binary_poly column, a single XOR of the two adjacent
        // BinaryPoly rows yields all D bit differences in one shot; iterate
        // the bits and conditionally fold α-powers in.
        let t1: F = cfg_into_iter!(0..half)
            .map(|b_prime| {
                let mut s_b: F = zero.clone();
                for (k, col) in self.binary_cols.iter().enumerate() {
                    let row_a = &col.evaluations[2 * b_prime];
                    let row_b = &col.evaluations[2 * b_prime + 1];
                    for (i, (a_bit, b_bit)) in row_a.iter().zip(row_b.iter()).enumerate() {
                        if a_bit.into_inner() != b_bit.into_inner() {
                            s_b = s_b + self.alpha_powers[k * D + i].clone();
                        }
                    }
                }
                let e_other =
                    F::new_unchecked_with_cfg(self.eq_other_table[b_prime].clone(), config);
                s_b * e_other
            })
            .reduce_with_or_default(zero.clone());

        // p_1(2) = 2 · (3·ic_ep_0 − 1) · T_1
        // p_1(3) = 6 · (5·ic_ep_0 − 2) · T_1
        let two = one.clone() + one.clone();
        let three = two.clone() + one.clone();
        let five = three.clone() + two.clone();
        let six = three.clone() + three.clone();

        let coeff_2 = three * self.ic_ep_0.clone() - one.clone();
        let coeff_3 = five * self.ic_ep_0.clone() - two.clone();

        let p1_at_1 = zero.clone();
        let p1_at_2 = (one.clone() + one.clone()) * coeff_2 * t1.clone();
        let p1_at_3 = six * coeff_3 * t1;

        Round1Output {
            asserted_sum: zero,
            tail_evaluations: vec![p1_at_1, p1_at_2, p1_at_3],
        }
    }

    fn fold_with_r1(
        self: Box<Self>,
        r_1: &F,
        config: &F::Config,
    ) -> Vec<DenseMultilinearExtension<F::Inner>> {
        let one = F::one_with_cfg(config);
        let one_minus_r1 = one.clone() - r_1.clone();
        let r1 = r_1.clone();
        let half = 1usize << (self.num_vars - 1);

        // eq_r_folded(b') = ((1 − r_1)(1 − ic_ep_0) + r_1 · ic_ep_0) · E_other(b').
        let eq_scalar = one_minus_r1.clone() * (one.clone() - self.ic_ep_0.clone())
            + r1.clone() * self.ic_ep_0.clone();
        let eq_folded_evals: Vec<F::Inner> = cfg_iter!(self.eq_other_table)
            .map(|e| {
                let lifted = F::new_unchecked_with_cfg(e.clone(), config);
                (eq_scalar.clone() * lifted).into_inner()
            })
            .collect();
        let eq_folded = DenseMultilinearExtension {
            num_vars: self.num_vars - 1,
            evaluations: eq_folded_evals,
        };

        // For each (k, i) bit-slice: fold over variable 0 with r_1.
        // (A, B) ∈ {0, 1}² → folded ∈ {0, r_1, 1 − r_1, 1} via lookup.
        let zero_inner = F::zero_with_cfg(config).into_inner();
        let one_inner = one.inner().clone();
        let r1_inner = r1.inner().clone();
        let one_minus_r1_inner = one_minus_r1.inner().clone();

        let mut mles: Vec<DenseMultilinearExtension<F::Inner>> =
            Vec::with_capacity(1 + self.binary_cols.len() * D);
        mles.push(eq_folded);

        // BinaryPoly<D> does not impl Index<usize> on every backend
        // (the simd-feature `BinaryU64Poly` exposes only `iter()`).
        // Iterate via `iter().zip(...)` per row pair, distributing each
        // bit's folded value into a per-bit accumulator.
        for col in &self.binary_cols {
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

/// Helper: reduce_with on a parallel iterator, falling back to a default
/// when the iterator is empty. Provided here to keep the parallel and
/// sequential paths uniform.
trait ReduceWithOrDefault<T: Clone> {
    fn reduce_with_or_default(self, default: T) -> T;
}

#[cfg(feature = "parallel")]
impl<I, T> ReduceWithOrDefault<T> for I
where
    I: ParallelIterator<Item = T>,
    T: Clone + Send + Sync + std::ops::Add<Output = T>,
{
    fn reduce_with_or_default(self, default: T) -> T {
        self.reduce(|| default.clone(), |a, b| a + b)
    }
}

#[cfg(not(feature = "parallel"))]
impl<I, T> ReduceWithOrDefault<T> for I
where
    I: Iterator<Item = T>,
    T: Clone + std::ops::Add<Output = T>,
{
    fn reduce_with_or_default(self, default: T) -> T {
        self.fold(default, |a, b| a + b)
    }
}

// ---------------------------------------------------------------------------
// Sumcheck group prep / finalize (separate degree-3 group)
// ---------------------------------------------------------------------------

/// Ancillary data produced by [`prepare_booleanity_group`] and consumed
/// by [`finalize_booleanity_prover`]. Carries the bit-slice count needed
/// to extract the right slice of evals after sumcheck completes.
pub struct BooleanityProverAncillary {
    /// Number of bit-slice MLEs in the group (excludes the leading eq_r MLE).
    pub num_bit_slices: usize,
}

/// Ancillary data produced by [`prepare_booleanity_verifier`] and
/// consumed by [`finalize_booleanity_verifier`].
pub struct BooleanityVerifierAncillary<F: PrimeField> {
    /// Powers of the booleanity folding challenge: `[1, α, α², ..., α^{B-1}]`.
    pub folding_challenge_powers: Vec<F>,
    /// Evaluation point used to build `eq_r` (mirrors what the prover used).
    pub ic_evaluation_point: Vec<F>,
}

/// Build a degree-3 multi-degree sumcheck group for the booleanity
/// zerocheck. MLE layout (post round-1 fold): `[eq_r, v_0, v_1, ..., v_{B-1}]`
/// where `B = K · D`.
///
/// Round 1 is supplied via [`BooleanityRound1FastPath`], which reads the
/// `binary_cols` directly and never materializes F-valued bit-slice MLEs
/// at full size. Rounds 2..n use the standard sumcheck path on the
/// half-size folded MLEs the fast path emits.
///
/// Returns `None` when `binary_cols` is empty (no binary_poly columns →
/// no booleanity check needed; caller should skip pushing this group).
#[allow(clippy::arithmetic_side_effects)]
pub fn prepare_booleanity_group<F, const D: usize>(
    transcript: &mut impl Transcript,
    binary_cols: &[DenseMultilinearExtension<BinaryPoly<D>>],
    ic_evaluation_point: &[F],
    field_cfg: &F::Config,
) -> Result<Option<(MultiDegreeSumcheckGroup<F>, BooleanityProverAncillary)>, BooleanityError<F>>
where
    F: InnerTransparentField + FromPrimitiveWithConfig + Send + Sync + 'static,
    F::Inner: ConstTranscribable + Send + Sync + Zero + Default + Clone,
    F::Modulus: ConstTranscribable,
{
    if binary_cols.is_empty() {
        return Ok(None);
    }

    let num_bit_slices = binary_cols.len() * D;
    let zero = F::zero_with_cfg(field_cfg);
    let one = F::one_with_cfg(field_cfg);

    let folding_challenge: F = transcript.get_field_challenge(field_cfg);
    let folding_challenge_powers: Vec<F> =
        powers(folding_challenge, one.clone(), num_bit_slices);

    // Pre-build E_other = eq(b', ic_evaluation_point[1..]) for the
    // round-1 fast path. The full-size eq_r is only needed for rounds
    // 2..n in folded form, which fold_with_r1 produces directly.
    assert!(
        !ic_evaluation_point.is_empty(),
        "ic_evaluation_point must be non-empty for booleanity (num_vars >= 1)"
    );
    let num_vars = ic_evaluation_point.len();
    let eq_other_table: Vec<F::Inner> = if num_vars >= 2 {
        build_eq_x_r_inner(&ic_evaluation_point[1..], field_cfg)?.evaluations
    } else {
        // num_vars == 1: the "other" subspace is empty, E_other is a single 1.
        vec![one.inner().clone()]
    };

    let fast_path: Box<dyn Round1FastPath<F>> = Box::new(BooleanityRound1FastPath::<F, D> {
        binary_cols: binary_cols.to_vec(),
        alpha_powers: folding_challenge_powers.clone(),
        eq_other_table,
        ic_ep_0: ic_evaluation_point[0].clone(),
        num_vars,
    });

    let comb_fn: CombFn<F> = Box::new(move |mle_values: &[F]| {
        let eq_r = &mle_values[0];
        let bits = &mle_values[1..];
        debug_assert_eq!(bits.len(), folding_challenge_powers.len());

        // Σ_k α^k · v_k · (v_k - 1) computed as Σ_k α^k · (v_k² - v_k) to
        // avoid a per-iteration `(v - one)` clone.
        let mut acc = zero.clone();
        for (v, coeff) in bits.iter().zip(folding_challenge_powers.iter()) {
            let v_sq = v.clone() * v.clone();
            acc = acc + coeff.clone() * (v_sq - v.clone());
        }
        acc * eq_r.clone()
    });

    // Empty `poly` — the fast path supplies post-round-1 MLEs.
    Ok(Some((
        MultiDegreeSumcheckGroup::with_round_1_fast(3, Vec::new(), comb_fn, fast_path),
        BooleanityProverAncillary { num_bit_slices },
    )))
}

/// Extract `bit_slice_evals` from the booleanity group's prover state
/// after the multi-degree sumcheck completes. The leading `eq_r` MLE
/// eval is dropped (verifier recomputes it).
#[allow(clippy::arithmetic_side_effects)]
pub fn finalize_booleanity_prover<F>(
    transcript: &mut impl Transcript,
    sumcheck_prover_state: SumcheckProverState<F>,
    ancillary: BooleanityProverAncillary,
    field_cfg: &F::Config,
) -> Result<Vec<F>, BooleanityError<F>>
where
    F: InnerTransparentField,
    F::Inner: ConstTranscribable + Zero,
    F::Modulus: ConstTranscribable,
{
    debug_assert!(
        sumcheck_prover_state
            .mles
            .iter()
            .all(|mle| mle.num_vars == 1)
    );

    let last_challenge = sumcheck_prover_state
        .randomness
        .last()
        .expect("sumcheck must have at least one round")
        .clone();

    let mut mles = sumcheck_prover_state.mles;
    // mles[0] is eq_r — drop it; the rest are the bit-slices in order.
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

/// Pre-sumcheck verifier half. Samples α (matching prover order),
/// validates that the booleanity group's claimed sum is zero (this is a
/// pure zerocheck), and stashes per-bit α-powers for the post-sumcheck
/// finalize.
///
/// Returns `None` when there are no binary_poly columns (mirrors the
/// prover's early-out).
#[allow(clippy::arithmetic_side_effects)]
pub fn prepare_booleanity_verifier<F>(
    transcript: &mut impl Transcript,
    claimed_sum: F,
    num_bit_slices: usize,
    ic_evaluation_point: &[F],
    field_cfg: &F::Config,
) -> Result<Option<BooleanityVerifierAncillary<F>>, BooleanityError<F>>
where
    F: InnerTransparentField + FromPrimitiveWithConfig,
    F::Inner: ConstTranscribable,
    F::Modulus: ConstTranscribable,
{
    if num_bit_slices == 0 {
        return Ok(None);
    }

    let zero = F::zero_with_cfg(field_cfg);
    let one = F::one_with_cfg(field_cfg);

    if claimed_sum != zero {
        return Err(BooleanityError::NonZeroClaimedSum { got: claimed_sum });
    }

    let folding_challenge: F = transcript.get_field_challenge(field_cfg);
    let folding_challenge_powers: Vec<F> = powers(folding_challenge, one, num_bit_slices);

    Ok(Some(BooleanityVerifierAncillary {
        folding_challenge_powers,
        ic_evaluation_point: ic_evaluation_point.to_vec(),
    }))
}

/// Post-sumcheck verifier half. Validates that
/// `expected_evaluation == Σ_k α^k · v_k · (v_k - 1) · eq_r(ic_eval_point, r*)`
/// where `r*` is the multi-degree sumcheck's shared point. Absorbs
/// `bit_slice_evals` into the transcript.
#[allow(clippy::arithmetic_side_effects)]
pub fn finalize_booleanity_verifier<F>(
    transcript: &mut impl Transcript,
    bit_slice_evals: &[F],
    shared_point: &[F],
    expected_evaluation: F,
    ancillary: BooleanityVerifierAncillary<F>,
    field_cfg: &F::Config,
) -> Result<(), BooleanityError<F>>
where
    F: InnerTransparentField,
    F::Inner: ConstTranscribable,
    F::Modulus: ConstTranscribable,
{
    if bit_slice_evals.len() != ancillary.folding_challenge_powers.len() {
        return Err(BooleanityError::WrongBitSliceEvalCount {
            got: bit_slice_evals.len(),
            expected: ancillary.folding_challenge_powers.len(),
        });
    }

    let zero = F::zero_with_cfg(field_cfg);
    let one = F::one_with_cfg(field_cfg);

    let eq_r_value = eq_eval(shared_point, &ancillary.ic_evaluation_point, one.clone())?;

    let bool_folded = bit_slice_evals
        .iter()
        .zip(ancillary.folding_challenge_powers.iter())
        .fold(zero, |acc, (v, coeff)| {
            let v_sq = v.clone() * v.clone();
            acc + coeff.clone() * (v_sq - v.clone())
        });

    let recomputed = bool_folded * eq_r_value;

    if recomputed != expected_evaluation {
        return Err(BooleanityError::SumcheckClaimMismatch {
            got: expected_evaluation,
            expected: recomputed,
        });
    }

    let mut buf: Vec<u8> = vec![0; F::Inner::NUM_BYTES];
    transcript.absorb_random_field_slice(bit_slice_evals, &mut buf);

    Ok(())
}

// ---------------------------------------------------------------------------
// Lifting consistency check (between bit-slice evals and parent column)
// ---------------------------------------------------------------------------

/// Verifier check: each binary_poly column's projected MLE evaluation at
/// `r*` (`up_evals[col_idx]`) must equal `Σ_i a^i · bit_slice_evals[i]`,
/// where `a` is the field-projection element used to send `F[X] → F`.
///
/// In projected `F`-land, `ψ_a(MLE[v](r*)) = Σ_i a^i · MLE[v_i](r*)`. With
/// overwhelming probability over the random `a`, the equation pins down
/// each bit-slice eval against the true bit-decomposition of the
/// committed parent column.
#[allow(clippy::arithmetic_side_effects)]
pub fn verify_bit_decomposition_consistency<F: PrimeField>(
    parent_evals_per_col: &[F],
    bit_slice_evals: &[F],
    projecting_element: &F,
    bits_per_col: usize,
) -> Result<(), BooleanityError<F>> {
    if bit_slice_evals.len() != parent_evals_per_col.len() * bits_per_col {
        return Err(BooleanityError::WrongBitSliceEvalCount {
            got: bit_slice_evals.len(),
            expected: parent_evals_per_col.len() * bits_per_col,
        });
    }

    if bits_per_col == 0 {
        return Ok(());
    }

    let zero = F::zero_with_cfg(projecting_element.cfg());
    let one = F::one_with_cfg(projecting_element.cfg());

    // Powers [1, a, a^2, ..., a^{bits_per_col - 1}].
    let mut a_powers: Vec<F> = Vec::with_capacity(bits_per_col);
    let mut acc = one;
    for _ in 0..bits_per_col {
        a_powers.push(acc.clone());
        acc *= projecting_element;
    }

    for (col_idx, parent_eval) in parent_evals_per_col.iter().enumerate() {
        let base = col_idx * bits_per_col;
        let recombined =
            bit_slice_evals[base..base + bits_per_col]
                .iter()
                .zip(&a_powers)
                .fold(zero.clone(), |acc, (bit_eval, a_pow)| {
                    acc + bit_eval.clone() * a_pow
                });

        if &recombined != parent_eval {
            return Err(BooleanityError::ConsistencyMismatch {
                col_idx,
                got: recombined,
                expected: parent_eval.clone(),
            });
        }
    }

    Ok(())
}

#[derive(Debug, Error)]
pub enum BooleanityError<F: PrimeField> {
    #[error(
        "wrong bit-slice evaluation count: got {got}, expected {expected}"
    )]
    WrongBitSliceEvalCount { got: usize, expected: usize },
    #[error(
        "bit-decomposition consistency mismatch on binary_poly column {col_idx}: got Σ a^i·bᵢ = {got:?}, expected parent eval {expected:?}"
    )]
    ConsistencyMismatch { col_idx: usize, got: F, expected: F },
    #[error("booleanity zerocheck claimed sum non-zero: {got:?}")]
    NonZeroClaimedSum { got: F },
    #[error("booleanity sumcheck claim mismatch: got {got:?}, expected {expected:?}")]
    SumcheckClaimMismatch { got: F, expected: F },
    #[error("eq_r evaluation failed: {0}")]
    EqEvalError(#[from] ArithErrors),
    #[error("MLE evaluation failed: {0}")]
    MleEvaluationError(#[from] EvaluationError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crypto_primitives::{
        FromWithConfig, boolean::Boolean, crypto_bigint_monty::MontyField,
    };

    type F = MontyField<4>;

    fn test_cfg() -> <F as crypto_primitives::PrimeField>::Config {
        crate::test_utils::test_config()
    }

    fn col_from_u8s(patterns: &[u8]) -> DenseMultilinearExtension<BinaryPoly<8>> {
        use std::array;
        let evaluations: Vec<BinaryPoly<8>> = patterns
            .iter()
            .map(|&p| {
                let coeffs: [Boolean; 8] =
                    array::from_fn(|i| Boolean::new((p >> i) & 1 != 0));
                BinaryPoly::<8>::new(coeffs)
            })
            .collect();
        let num_vars = evaluations.len().next_power_of_two().trailing_zeros() as usize;
        DenseMultilinearExtension {
            num_vars,
            evaluations,
        }
    }

    #[test]
    fn bit_slices_round_trip_recovers_original_bits() {
        let cfg = test_cfg();
        let col = col_from_u8s(&[0b00000000, 0b11111111, 0b10101010, 0b01010101]);
        let bit_slices = compute_bit_slices_flat::<F, 8>(std::slice::from_ref(&col), &cfg);

        assert_eq!(bit_slices.len(), 8);
        let one = F::one_with_cfg(&cfg).into_inner();
        let zero = F::zero_with_cfg(&cfg).into_inner();
        for (row, p) in [0b00000000u8, 0b11111111, 0b10101010, 0b01010101]
            .iter()
            .enumerate()
        {
            for bit in 0..8 {
                let want = if (p >> bit) & 1 != 0 {
                    one.clone()
                } else {
                    zero.clone()
                };
                assert_eq!(bit_slices[bit].evaluations[row], want, "row {row} bit {bit}");
            }
        }
    }

    #[test]
    fn consistency_check_accepts_honest_decomposition() {
        let cfg = test_cfg();
        let one = F::one_with_cfg(&cfg);
        let zero = F::zero_with_cfg(&cfg);

        let bits: [u32; 8] = [1, 0, 1, 1, 0, 0, 0, 1];
        let bit_evals: Vec<F> = bits
            .iter()
            .map(|&b| if b == 1 { one.clone() } else { zero.clone() })
            .collect();
        let a = one.clone() + one.clone() + one.clone();

        let mut parent_eval = zero.clone();
        let mut a_pow = one.clone();
        for be in &bit_evals {
            parent_eval = parent_eval + be.clone() * a_pow.clone();
            a_pow = a_pow * a.clone();
        }

        verify_bit_decomposition_consistency(
            std::slice::from_ref(&parent_eval),
            &bit_evals,
            &a,
            8,
        )
        .expect("honest decomposition should satisfy consistency check");
    }

    #[test]
    fn consistency_check_rejects_tampered_bit() {
        let cfg = test_cfg();
        let one = F::one_with_cfg(&cfg);
        let zero = F::zero_with_cfg(&cfg);

        let bits: [u32; 4] = [1, 1, 1, 1];
        let bit_evals: Vec<F> = bits
            .iter()
            .map(|&b| if b == 1 { one.clone() } else { zero.clone() })
            .collect();
        let a = one.clone() + one.clone();

        let mut parent_eval = zero.clone();
        let mut a_pow = one.clone();
        for be in &bit_evals {
            parent_eval = parent_eval + be.clone() * a_pow.clone();
            a_pow = a_pow * a.clone();
        }

        let mut tampered = bit_evals.clone();
        tampered[0] = tampered[0].clone() + one;

        let res = verify_bit_decomposition_consistency(
            std::slice::from_ref(&parent_eval),
            &tampered,
            &a,
            4,
        );
        assert!(matches!(res, Err(BooleanityError::ConsistencyMismatch { .. })));
    }

    #[test]
    fn consistency_check_no_op_when_no_binary_poly_columns() {
        let cfg = test_cfg();
        let one = F::one_with_cfg(&cfg);
        let parent_evals: Vec<F> = vec![];
        let bit_evals: Vec<F> = vec![];
        verify_bit_decomposition_consistency(&parent_evals, &bit_evals, &one, 8).unwrap();
    }

    /// Cross-validate the round-1 fast path against a faithful standard
    /// run of `ProverState::prove_round`. Both must produce the same
    /// tail evaluations and the same asserted sum (zero, since this is
    /// a zerocheck).
    #[test]
    fn fast_path_round_1_matches_standard_prove_round() {
        use crate::CombFn;
        use crate::sumcheck::prover::ProverState as SumcheckProverState;

        let cfg = test_cfg();
        let one = F::one_with_cfg(&cfg);
        let zero = F::zero_with_cfg(&cfg);

        // 3-variable case (8 rows). Two binary_poly cols of width 8 each.
        // Mix of fully-zero, fully-one, and varying patterns to exercise all
        // four (A, B) cases for the XOR fold structure.
        let binary_cols = vec![
            col_from_u8s(&[0b00000000, 0b00010001, 0b00100010, 0b00110011, 0b01000100, 0b01010101, 0b01100110, 0b01110111]),
            col_from_u8s(&[0b11110000, 0b11100001, 0b11010010, 0b11000011, 0b10110100, 0b10100101, 0b10010110, 0b10000111]),
        ];
        let num_vars = 3;
        const D: usize = 8;
        let k = binary_cols.len();
        let num_bit_slices = k * D;

        let alpha = F::from_with_cfg(7u64, &cfg);
        let alpha_powers: Vec<F> = powers(alpha, one.clone(), num_bit_slices);

        let ic_ep: Vec<F> = vec![
            F::from_with_cfg(3u64, &cfg),
            F::from_with_cfg(5u64, &cfg),
            F::from_with_cfg(11u64, &cfg),
        ];

        // ---- Fast path ----
        let eq_other_table = build_eq_x_r_inner(&ic_ep[1..], &cfg).unwrap().evaluations;
        let fast_path = BooleanityRound1FastPath::<F, D> {
            binary_cols: binary_cols.clone(),
            alpha_powers: alpha_powers.clone(),
            eq_other_table,
            ic_ep_0: ic_ep[0].clone(),
            num_vars,
        };
        let fast_out = fast_path.round_1_message(&cfg);

        // ---- Standard path ----
        let bit_slices = compute_bit_slices_flat::<F, D>(&binary_cols, &cfg);
        let eq_r = build_eq_x_r_inner(&ic_ep, &cfg).unwrap();
        let mut mles: Vec<DenseMultilinearExtension<<F as crypto_primitives::Field>::Inner>> =
            Vec::with_capacity(1 + bit_slices.len());
        mles.push(eq_r);
        mles.extend(bit_slices);

        let alpha_powers_for_comb = alpha_powers.clone();
        let zero_for_comb = zero.clone();
        let comb_fn: CombFn<F> = Box::new(move |mle_values: &[F]| {
            let eq_r = &mle_values[0];
            let bits = &mle_values[1..];
            let mut acc = zero_for_comb.clone();
            for (v, coeff) in bits.iter().zip(alpha_powers_for_comb.iter()) {
                let v_sq = v.clone() * v.clone();
                acc = acc + coeff.clone() * (v_sq - v.clone());
            }
            acc * eq_r.clone()
        });

        let mut state = SumcheckProverState::new(mles, num_vars, 3);
        let std_msg = state.prove_round(&None, comb_fn, &cfg);
        let std_asserted = state
            .asserted_sum
            .clone()
            .expect("asserted_sum recorded after first round");

        assert_eq!(
            fast_out.tail_evaluations.len(),
            std_msg.0.tail_evaluations.len(),
            "tail length mismatch (fast vs standard)"
        );
        assert_eq!(
            fast_out.tail_evaluations, std_msg.0.tail_evaluations,
            "fast-path round-1 tail must match standard path"
        );
        assert_eq!(
            fast_out.asserted_sum, std_asserted,
            "fast-path asserted_sum must match standard path (zerocheck → 0)"
        );
        assert_eq!(
            fast_out.asserted_sum, zero,
            "booleanity zerocheck asserted_sum must be 0"
        );
    }

    /// Cross-validate `fold_with_r1` against folding bit-slice MLEs via
    /// the standard `fix_variables` path. Both must produce the same
    /// post-round-1 MLE values for every (k, i) bit-slice.
    #[test]
    fn fast_path_fold_with_r1_matches_standard_fix_variables() {
        use std::slice;
        use zinc_poly::mle::MultilinearExtensionWithConfig;

        let cfg = test_cfg();
        let one = F::one_with_cfg(&cfg);

        let binary_cols = vec![col_from_u8s(&[
            0b00000000, 0b11111111, 0b01010101, 0b10101010, 0b11001100, 0b00110011, 0b11110000,
            0b00001111,
        ])];
        let num_vars = 3;
        const D: usize = 8;
        let alpha = F::from_with_cfg(13u64, &cfg);
        let alpha_powers: Vec<F> = powers(alpha, one.clone(), D);
        let ic_ep: Vec<F> = vec![
            F::from_with_cfg(2u64, &cfg),
            F::from_with_cfg(4u64, &cfg),
            F::from_with_cfg(8u64, &cfg),
        ];
        let r_1 = F::from_with_cfg(17u64, &cfg);

        let eq_other_table = build_eq_x_r_inner(&ic_ep[1..], &cfg).unwrap().evaluations;
        let fast_path = Box::new(BooleanityRound1FastPath::<F, D> {
            binary_cols: binary_cols.clone(),
            alpha_powers,
            eq_other_table,
            ic_ep_0: ic_ep[0].clone(),
            num_vars,
        });

        let fast_mles = fast_path.fold_with_r1(&r_1, &cfg);
        // [eq_r_folded, bit_0, bit_1, ..., bit_{D-1}].
        assert_eq!(fast_mles.len(), 1 + D);

        // Standard-path comparison: build full-size bit-slice MLEs and
        // fold each with r_1. Build full-size eq_r and fold the same way.
        let bit_slices_full = compute_bit_slices_flat::<F, D>(&binary_cols, &cfg);
        let eq_r_full = build_eq_x_r_inner(&ic_ep, &cfg).unwrap();

        let mut std_eq_r = eq_r_full;
        std_eq_r.fix_variables_with_config(slice::from_ref(&r_1), &cfg);
        assert_eq!(
            fast_mles[0].num_vars, num_vars - 1,
            "fast-path eq_r_folded must have num_vars - 1 variables"
        );
        assert_eq!(
            fast_mles[0].evaluations, std_eq_r.evaluations,
            "fast-path eq_r_folded must match standard fix_variables"
        );

        for (idx, mut bit_mle) in bit_slices_full.into_iter().enumerate() {
            bit_mle.fix_variables_with_config(slice::from_ref(&r_1), &cfg);
            assert_eq!(
                fast_mles[1 + idx].evaluations, bit_mle.evaluations,
                "fast-path bit-slice {idx} folded value must match standard fix_variables"
            );
        }
    }

    #[test]
    fn consistency_check_rejects_wrong_eval_count() {
        let cfg = test_cfg();
        let one = F::one_with_cfg(&cfg);
        let parent_evals = vec![one.clone()];
        let bit_evals: Vec<F> = vec![one.clone(), one.clone()];
        let res = verify_bit_decomposition_consistency(&parent_evals, &bit_evals, &one, 8);
        assert!(matches!(res, Err(BooleanityError::WrongBitSliceEvalCount { .. })));
    }
}
