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
use zinc_utils::{cfg_iter, inner_transparent_field::InnerTransparentField, powers};

use crate::{
    CombFn,
    sumcheck::{multi_degree::MultiDegreeSumcheckGroup, prover::ProverState as SumcheckProverState},
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
/// zerocheck. MLE layout: `[eq_r, v_0, v_1, ..., v_{B-1}]`.
///
/// Returns `None` when `bit_slices` is empty (no binary_poly columns →
/// no booleanity check needed; caller should skip pushing this group).
#[allow(clippy::arithmetic_side_effects)]
pub fn prepare_booleanity_group<F>(
    transcript: &mut impl Transcript,
    bit_slices: Vec<DenseMultilinearExtension<F::Inner>>,
    ic_evaluation_point: &[F],
    field_cfg: &F::Config,
) -> Result<Option<(MultiDegreeSumcheckGroup<F>, BooleanityProverAncillary)>, BooleanityError<F>>
where
    F: InnerTransparentField + FromPrimitiveWithConfig + Send + Sync + 'static,
    F::Inner: ConstTranscribable + Send + Sync + Zero + Default,
    F::Modulus: ConstTranscribable,
{
    let num_bit_slices = bit_slices.len();
    if num_bit_slices == 0 {
        return Ok(None);
    }

    let zero = F::zero_with_cfg(field_cfg);
    let one = F::one_with_cfg(field_cfg);

    let eq_r = build_eq_x_r_inner(ic_evaluation_point, field_cfg)?;

    let folding_challenge: F = transcript.get_field_challenge(field_cfg);
    let folding_challenge_powers: Vec<F> =
        powers(folding_challenge, one.clone(), num_bit_slices);

    let mut mles: Vec<DenseMultilinearExtension<F::Inner>> =
        Vec::with_capacity(1 + num_bit_slices);
    mles.push(eq_r);
    mles.extend(bit_slices);

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

    Ok(Some((
        MultiDegreeSumcheckGroup::new(3, mles, comb_fn),
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
    use crypto_primitives::{boolean::Boolean, crypto_bigint_monty::MontyField};

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
