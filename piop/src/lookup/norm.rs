//! Per-signature squared-norm zerocheck (batched Falcon).
//!
//! Adapts the booleanity machinery ([`super::booleanity`]) to prove a
//! short-vector norm bound for each signature *independently*. With one
//! signature per hypercube row, the bound
//!
//! ```text
//!   Σ_i s_1[i]² + Σ_i s_2[i]² + slack = ⌊β²⌋ ,   slack ≥ 0
//! ```
//!
//! is a zerocheck over the signature rows:
//!
//! ```text
//!   Σ_j eq(r, j) · ( Σ_slices slice(j)²  +  slack(j)  −  ⌊β²⌋ )  =  0 .
//! ```
//!
//! `slice` ranges over the coefficient-slices of the `s_1`/`s_2` limb cells
//! (extracted by [`compute_coeff_slices_flat`], the integer-coefficient
//! analogue of booleanity's [`super::booleanity::compute_bit_slices_flat`]).
//! Like booleanity this is a degree-3 group (`slice² · eq_r`) plugged into the
//! shared multi-degree sumcheck, with **claimed sum 0**: for an honest witness
//! every per-row bracket is 0, and `Σ_j eq(r,j) = 1` makes the constant `−⌊β²⌋`
//! cancel. The `{0,1}` round-1 fast path does *not* apply (the coefficients are
//! not bits), so this uses the plain group constructor.
//!
//! `slack ≥ 0` and the centered coefficient ranges are separate range checks
//! (not this group).

use crypto_primitives::{FromPrimitiveWithConfig, FromWithConfig, PrimeField};
use num_traits::Zero;
use std::slice;
use thiserror::Error;
use zinc_poly::{
    EvaluationError,
    mle::{DenseMultilinearExtension, MultilinearExtensionWithConfig},
    univariate::dense::DensePolynomial,
    utils::{ArithErrors, build_eq_x_r_inner, eq_eval},
};
use zinc_transcript::traits::{ConstTranscribable, Transcript};
use zinc_utils::{cfg_iter, inner_transparent_field::InnerTransparentField};

use crate::{
    CombFn,
    sumcheck::{multi_degree::MultiDegreeSumcheckGroup, prover::ProverState as SumcheckProverState},
};

/// Build `F::Inner`-valued coefficient-slice MLEs for every `arbitrary_poly`
/// column. The integer-coefficient analogue of
/// [`super::booleanity::compute_bit_slices_flat`].
///
/// Output ordering is flat, **column-major-then-coefficient-major**: index
/// `col_idx * D + p` is the MLE whose evaluations are coefficient `p` of each
/// row's `DensePolynomial<PolyCoeff, D>` cell, projected to `F::Inner` via the
/// field's `from_with_cfg` (the `φ_q` map). Length: `cols.len() * D`.
#[allow(clippy::arithmetic_side_effects)]
pub fn compute_coeff_slices_flat<F, PolyCoeff, const D: usize>(
    trace_arb_poly: &[DenseMultilinearExtension<DensePolynomial<PolyCoeff, D>>],
    field_cfg: &F::Config,
) -> Vec<DenseMultilinearExtension<F::Inner>>
where
    F: InnerTransparentField + for<'a> FromWithConfig<&'a PolyCoeff>,
    F::Inner: Clone + Send + Sync,
    F::Config: Sync,
    PolyCoeff: Clone + Send + Sync,
{
    cfg_iter!(trace_arb_poly)
        .flat_map(|col| {
            let num_vars = col.num_vars;
            // Per-column transpose: coeff_evals[p][row] = coefficient p.
            let mut coeff_evals: Vec<Vec<F::Inner>> = (0..D)
                .map(|_| Vec::with_capacity(col.evaluations.len()))
                .collect();
            for cell in &col.evaluations {
                for (p, coeff) in cell.coeffs.iter().enumerate() {
                    coeff_evals[p].push(F::from_with_cfg(coeff, field_cfg).into_inner());
                }
            }
            coeff_evals
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
// Group prep / finalize
// ---------------------------------------------------------------------------

/// Ancillary data from [`prepare_norm_group`] for [`finalize_norm_prover`].
pub struct NormProverAncillary {
    /// Number of value MLEs (coefficient-slices + the trailing `slack`),
    /// i.e. every group MLE except the leading `eq_r`.
    pub num_value_mles: usize,
}

/// Ancillary data from [`prepare_norm_verifier`] for [`finalize_norm_verifier`].
pub struct NormVerifierAncillary<F: PrimeField> {
    /// The public acceptance bound `⌊β²⌋`.
    pub beta_sq: F,
    /// The zerocheck anchor `r` used to build `eq_r` (mirrors the prover).
    pub ic_evaluation_point: Vec<F>,
    /// Number of coefficient-slices (the value MLEs minus the `slack` tail).
    pub num_slices: usize,
}

/// Build the degree-3 squared-norm zerocheck group. MLE layout:
/// `[eq_r, slice_0, …, slice_{S−1}, slack]`. The combiner is
/// `eq_r · ( Σ slice² + slack − ⌊β²⌋ )`; claimed sum is 0.
#[allow(clippy::arithmetic_side_effects)]
pub fn prepare_norm_group<F>(
    slice_mles: &[DenseMultilinearExtension<F::Inner>],
    slack_mle: &DenseMultilinearExtension<F::Inner>,
    beta_sq: F,
    ic_evaluation_point: &[F],
    field_cfg: &F::Config,
) -> Result<(MultiDegreeSumcheckGroup<F>, NormProverAncillary), NormError<F>>
where
    F: InnerTransparentField + FromPrimitiveWithConfig + Send + Sync + 'static,
    F::Inner: ConstTranscribable + Send + Sync + Zero + Default + Clone,
    F::Modulus: ConstTranscribable,
{
    assert!(
        !ic_evaluation_point.is_empty(),
        "norm zerocheck needs a non-empty anchor (num_vars >= 1)"
    );
    let num_slices = slice_mles.len();

    let eq_r = build_eq_x_r_inner(ic_evaluation_point, field_cfg)?;

    let mut poly: Vec<DenseMultilinearExtension<F::Inner>> =
        Vec::with_capacity(num_slices + 2);
    poly.push(eq_r);
    poly.extend(slice_mles.iter().cloned());
    poly.push(slack_mle.clone());

    let beta = beta_sq;
    let comb_fn: CombFn<F> = Box::new(move |v: &[F]| {
        let eq_r = &v[0];
        let n = v.len();
        // v[1..n-1] are the coefficient-slices; v[n-1] is slack.
        let slack = &v[n - 1];
        let mut acc = slack.clone() - beta.clone();
        for s in &v[1..n - 1] {
            acc = acc + s.clone() * s.clone();
        }
        acc * eq_r.clone()
    });

    Ok((
        MultiDegreeSumcheckGroup::new(3, poly, comb_fn),
        NormProverAncillary {
            num_value_mles: num_slices + 1,
        },
    ))
}

/// Extract the value evals (`[slice_0, …, slice_{S−1}, slack]` at `r*`) from
/// the group's prover state; absorbs them and returns them. The leading
/// `eq_r` MLE eval is dropped (verifier recomputes it).
#[allow(clippy::arithmetic_side_effects)]
pub fn finalize_norm_prover<F>(
    transcript: &mut impl Transcript,
    sumcheck_prover_state: SumcheckProverState<F>,
    ancillary: NormProverAncillary,
    field_cfg: &F::Config,
) -> Result<Vec<F>, NormError<F>>
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
    let _eq_r_mle = mles.remove(0); // eq_r — verifier recomputes it.
    let value_evals: Vec<F> = mles
        .into_iter()
        .map(|m| m.evaluate_with_config(slice::from_ref(&last_challenge), field_cfg))
        .collect::<Result<Vec<_>, _>>()?;

    debug_assert_eq!(value_evals.len(), ancillary.num_value_mles);

    let mut buf: Vec<u8> = vec![0; F::Inner::NUM_BYTES];
    transcript.absorb_random_field_slice(&value_evals, &mut buf);

    Ok(value_evals)
}

/// Pre-sumcheck verifier half: the squared-norm group is a zerocheck, so its
/// claimed sum must be 0.
pub fn prepare_norm_verifier<F>(
    claimed_sum: F,
    beta_sq: F,
    num_slices: usize,
    ic_evaluation_point: &[F],
    field_cfg: &F::Config,
) -> Result<NormVerifierAncillary<F>, NormError<F>>
where
    F: InnerTransparentField,
{
    let zero = F::zero_with_cfg(field_cfg);
    if claimed_sum != zero {
        return Err(NormError::NonZeroClaimedSum { got: claimed_sum });
    }
    Ok(NormVerifierAncillary {
        beta_sq,
        ic_evaluation_point: ic_evaluation_point.to_vec(),
        num_slices,
    })
}

/// Post-sumcheck verifier half: check
/// `expected_evaluation == eq_r(r*) · ( Σ slice² + slack − ⌊β²⌋ )`.
///
/// `closing_overrides_tail` substitutes trailing value positions for the
/// closing computation only (used during protocol wiring to bind a value to
/// its committed column's CPR `up_eval`); pass an empty slice to disable.
/// Absorption is always over the original `value_evals`.
#[allow(clippy::arithmetic_side_effects)]
pub fn finalize_norm_verifier<F>(
    transcript: &mut impl Transcript,
    value_evals: &[F],
    closing_overrides_tail: &[F],
    shared_point: &[F],
    expected_evaluation: F,
    ancillary: NormVerifierAncillary<F>,
    field_cfg: &F::Config,
) -> Result<(), NormError<F>>
where
    F: InnerTransparentField,
    F::Inner: ConstTranscribable,
    F::Modulus: ConstTranscribable,
{
    if value_evals.len() != ancillary.num_slices + 1 {
        return Err(NormError::WrongValueEvalCount {
            got: value_evals.len(),
            expected: ancillary.num_slices + 1,
        });
    }
    if closing_overrides_tail.len() > value_evals.len() {
        return Err(NormError::WrongValueEvalCount {
            got: closing_overrides_tail.len(),
            expected: value_evals.len(),
        });
    }

    let one = F::one_with_cfg(field_cfg);
    let eq_r_value = eq_eval(shared_point, &ancillary.ic_evaluation_point, one)?;

    // Apply the closing overrides over the trailing positions.
    let n_proof = value_evals.len() - closing_overrides_tail.len();
    let effective: Vec<F> = value_evals[..n_proof]
        .iter()
        .cloned()
        .chain(closing_overrides_tail.iter().cloned())
        .collect();

    let slack = &effective[ancillary.num_slices];
    let mut bracket = slack.clone() - ancillary.beta_sq.clone();
    for s in &effective[..ancillary.num_slices] {
        bracket = bracket + s.clone() * s.clone();
    }
    let recomputed = bracket * eq_r_value;

    if recomputed != expected_evaluation {
        return Err(NormError::SumcheckClaimMismatch {
            got: expected_evaluation,
            expected: recomputed,
        });
    }

    let mut buf: Vec<u8> = vec![0; F::Inner::NUM_BYTES];
    transcript.absorb_random_field_slice(value_evals, &mut buf);

    Ok(())
}

#[derive(Debug, Error)]
pub enum NormError<F: PrimeField> {
    #[error("wrong value-eval count: got {got}, expected {expected}")]
    WrongValueEvalCount { got: usize, expected: usize },
    #[error("squared-norm zerocheck claimed sum non-zero: {got:?}")]
    NonZeroClaimedSum { got: F },
    #[error("squared-norm sumcheck claim mismatch: got {got:?}, expected {expected:?}")]
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
    use crypto_primitives::crypto_bigint_monty::MontyField;
    use zinc_transcript::Blake3Transcript;

    type F = MontyField<4>;

    fn test_cfg() -> <F as PrimeField>::Config {
        crate::test_utils::test_config()
    }

    /// `2` coefficient-slices + `slack` over `4` signatures (rows). Honest
    /// `slack(j) = ⌊β²⌋ − (s0(j)² + s1(j)²)`, so every per-row bracket is 0,
    /// the zerocheck verifies, and the closing matches. Tampering a slice
    /// eval breaks the closing.
    #[test]
    fn norm_group_accepts_honest_rejects_tampered() {
        let cfg = test_cfg();
        let num_vars = 2usize;
        let n = 1usize << num_vars;
        let one = F::one_with_cfg(&cfg);

        let f = |x: u64| F::from_with_cfg(x, &cfg);
        let s0: Vec<F> = (0..n as u64).map(|i| f(i + 1)).collect();
        let s1: Vec<F> = (0..n as u64).map(|i| f(i + 2)).collect();
        let beta = f(1000);
        let slack: Vec<F> = (0..n)
            .map(|i| beta.clone() - (s0[i].clone() * s0[i].clone() + s1[i].clone() * s1[i].clone()))
            .collect();

        let to_mle = |vals: &[F]| DenseMultilinearExtension {
            num_vars,
            evaluations: vals.iter().map(|x| x.inner().clone()).collect(),
        };
        let slice_mles = vec![to_mle(&s0), to_mle(&s1)];
        let slack_mle = to_mle(&slack);
        let ic_ep: Vec<F> = vec![f(3), f(5)];

        // ---- prover ----
        let (group, anc) =
            prepare_norm_group(&slice_mles, &slack_mle, beta.clone(), &ic_ep, &cfg).unwrap();
        let mut pt = Blake3Transcript::new();
        let (proof, mut states) =
            MultiDegreeSumcheck::<F>::prove_as_subprotocol(&mut pt, vec![group], num_vars, &cfg);
        let value_evals =
            finalize_norm_prover(&mut pt, states.remove(0), anc, &cfg).unwrap();

        // zerocheck ⇒ claimed sum is 0.
        assert_eq!(proof.claimed_sums()[0], F::zero_with_cfg(&cfg));

        // ---- verifier (honest) ----
        let mut vt = Blake3Transcript::new();
        let subclaims =
            MultiDegreeSumcheck::<F>::verify_as_subprotocol(&mut vt, num_vars, &proof, &cfg)
                .unwrap();
        let vanc =
            prepare_norm_verifier(proof.claimed_sums()[0].clone(), beta.clone(), 2, &ic_ep, &cfg)
                .unwrap();
        finalize_norm_verifier(
            &mut vt,
            &value_evals,
            &[],
            subclaims.point(),
            subclaims.expected_evaluations()[0].clone(),
            vanc,
            &cfg,
        )
        .expect("honest squared-norm witness must verify");

        // ---- verifier (tampered slice eval) ----
        let mut bad = value_evals.clone();
        bad[0] = bad[0].clone() + one;
        let mut vt2 = Blake3Transcript::new();
        let subclaims2 =
            MultiDegreeSumcheck::<F>::verify_as_subprotocol(&mut vt2, num_vars, &proof, &cfg)
                .unwrap();
        let vanc2 =
            prepare_norm_verifier(proof.claimed_sums()[0].clone(), beta.clone(), 2, &ic_ep, &cfg)
                .unwrap();
        let res = finalize_norm_verifier(
            &mut vt2,
            &bad,
            &[],
            subclaims2.point(),
            subclaims2.expected_evaluations()[0].clone(),
            vanc2,
            &cfg,
        );
        assert!(matches!(res, Err(NormError::SumcheckClaimMismatch { .. })));
    }

    /// A dishonest witness whose per-signature norm exceeds `⌊β²⌋` (so the
    /// honest non-negative `slack` would be negative) cannot make the
    /// zerocheck claimed sum 0 *and* satisfy the closing: with a forced
    /// `slack = 0` the per-row brackets are non-zero, so the prover's own
    /// claimed sum is non-zero and `prepare_norm_verifier` rejects.
    #[test]
    fn norm_group_rejects_over_bound_with_zero_slack() {
        let cfg = test_cfg();
        let num_vars = 2usize;
        let n = 1usize << num_vars;
        let f = |x: u64| F::from_with_cfg(x, &cfg);

        // Large coefficients, tiny bound ⇒ Σ slice² ≫ β².
        let s0: Vec<F> = (0..n as u64).map(|i| f(100 + i)).collect();
        let s1: Vec<F> = (0..n as u64).map(|i| f(200 + i)).collect();
        let beta = f(5);
        let slack: Vec<F> = (0..n).map(|_| F::zero_with_cfg(&cfg)).collect();

        let to_mle = |vals: &[F]| DenseMultilinearExtension {
            num_vars,
            evaluations: vals.iter().map(|x| x.inner().clone()).collect(),
        };
        let slice_mles = vec![to_mle(&s0), to_mle(&s1)];
        let slack_mle = to_mle(&slack);
        let ic_ep: Vec<F> = vec![f(3), f(5)];

        let (group, _anc) =
            prepare_norm_group(&slice_mles, &slack_mle, beta.clone(), &ic_ep, &cfg).unwrap();
        let mut pt = Blake3Transcript::new();
        let (proof, _states) =
            MultiDegreeSumcheck::<F>::prove_as_subprotocol(&mut pt, vec![group], num_vars, &cfg);

        // The prover's own claimed sum is non-zero ⇒ the verifier's zerocheck
        // gate rejects before even checking the closing.
        let res = prepare_norm_verifier(proof.claimed_sums()[0].clone(), beta, 2, &ic_ep, &cfg);
        assert!(matches!(res, Err(NormError::NonZeroClaimedSum { .. })));
    }
}
