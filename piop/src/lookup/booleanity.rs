//! Booleanity (binary-polynomial lookup) argument.
//!
//! Proves that every coefficient of every witness binary-polynomial column is
//! a bit $\in \\{0,1\\}$. The argument is structured as a single
//! [`MultiDegreeSumcheckGroup`] of degree 3, batched alongside the existing
//! CPR group with shared randomness, and discharges its bit-slice claims for
//! free against the existing `lifted_evals` payload.
//!
//! # Relation
//!
//! For each witness binary-poly column $u_j \in (F_q^{<D}[X])^n$ with
//! $n = 2^\mu$, decompose row-wise as
//!
//! $$
//! {u_j}[b](X) = \sum_{i=0}^{D-1} v_{j,i,b} * X^i
//! $$
//!
//! The booleanity claim is:
//!
//! $$
//!   \forall j, i, b:  v_{j,i,b} \in \\{0,1\\}
//! $$
//!
//! Equivalently, the MLE statement
//! $\widetilde{v_{j,i}}(b)*(\widetilde{v_{j,i}}(b)-1) = 0$ for all
//! $b \in {0,1}^\mu$. The protocol reduces this to a single batched sumcheck
//!
//! $$
//! \sum_{b in {0,1}^\mu} eq(r, b) *
//!   \sum_{j=0}^{N-1} \sum_{i=0}^{D-1} \delta^j * \gamma^i *
//!     \widetilde{v_{j,i}}(b) * (\widetilde{v_{j,i}}(b) - 1)  =  0
//! $$
//!
//! with batching challenges $\gamma$ (over the `D` bit-slices) and $\delta$
//! (over the `N` witness binary-poly columns), and zerocheck point $r$.
//! After the sumcheck reaches $r*$, the prover sends `bit_slice_evals` =
//! $(\widetilde{v_{j,i}}(r*))$ to the verifier; these are then folded by
//! `MultipointEval` into the standard evaluation point $r_0$ together with
//! the CPR up/down evals. At $r_0$ the bit-slice MLE evaluations are exactly
//! the coefficients of the polynomial-valued `lifted_evals` (by the
//! MLE-commutes-with-coefficient-extraction identity), so the verifier
//! discharges them for free.

use crate::{
    CombFn,
    sumcheck::{
        SumCheckError, multi_degree::MultiDegreeSumcheckGroup,
        prover::ProverState as SumcheckProverState,
    },
};
use crypto_primitives::PrimeField;
use num_traits::Zero;
use std::{marker::PhantomData, slice};
use thiserror::Error;
use zinc_poly::{
    EvaluationError,
    mle::{DenseMultilinearExtension, MultilinearExtensionWithConfig},
    univariate::binary::BinaryPoly,
    utils::{ArithErrors, build_eq_x_r_inner, eq_eval},
};
use zinc_transcript::{
    delegate_transcribable,
    traits::{ConstTranscribable, Transcript},
};
use zinc_utils::{add, inner_transparent_field::InnerTransparentField, mul, powers};

//
// Structs
//

/// Booleanity sumcheck group constructor / verifier.
pub struct BooleanityChecker<F: InnerTransparentField>(PhantomData<F>);

/// Proof produced by the booleanity prover. Carries only the flat list of
/// bit-slice MLE evaluations at the multi-degree sumcheck output point
/// $r*$. The remaining open-eval consistency check at $r_0$ is discharged
/// by the protocol-layer caller against `lifted_evals.coeffs`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BooleanityProof<F: PrimeField> {
    /// Flat list of `\widetilde{v_{j,i}}(r*)`, ordered `(j-major, i-minor)`.
    /// Length = `num_wit_bin_cols * D`.
    pub bit_slice_evals: Vec<F>,
}

delegate_transcribable!(BooleanityProof<F> { bit_slice_evals: Vec<F> }
    where F: PrimeField, F::Inner: ConstTranscribable, F::Modulus: ConstTranscribable);

/// Ancillary data produced by [`BooleanityChecker::prepare_sumcheck_group`]
/// and consumed by [`BooleanityChecker::finalize_prover`].
pub struct BoolProverAncillary {
    /// Number of witness binary-poly columns (`N`).
    pub num_wit_bin_cols: usize,
    /// Bit-width of each binary-poly coefficient cell (`D`).
    pub bit_width: usize,
    /// Number of variables of the trace MLEs.
    pub num_vars: usize,
}

/// Ancillary data produced by [`BooleanityChecker::prepare_verifier`] and
/// consumed by [`BooleanityChecker::finalize_verifier`].
pub struct BoolVerifierAncillary<F: PrimeField> {
    /// Powers of the bit-slice batching challenge:
    /// `[1, gamma, ..., gamma^{D-1}]`.
    pub gamma_powers: Vec<F>,
    /// Powers of the column batching challenge:
    /// `[1, delta, ..., delta^{N-1}]`.
    pub delta_powers: Vec<F>,
    /// The zerocheck point `r` sampled before the multi-degree sumcheck.
    pub zerocheck_point: Vec<F>,
    /// Number of witness binary-poly columns (`N`).
    pub num_wit_bin_cols: usize,
    /// Bit-width of each binary-poly coefficient cell (`D`).
    pub bit_width: usize,
    /// Number of variables (for sanity-checking the shared point length).
    pub num_vars: usize,
}

/// Subclaim emitted by [`BooleanityChecker::finalize_verifier`].
///
/// Carries the (now-validated against the sumcheck residue)
/// `bit_slice_evals` at the shared multi-degree sumcheck point `r*`. The
/// protocol-layer caller threads these as additional `up_evals` into
/// [`crate::multipoint_eval::MultipointEval`], whose final consistency
/// check at `r_0` is discharged against the coefficients of the existing
/// `lifted_evals` polynomials.
#[derive(Clone, Debug)]
pub struct BoolVerifierSubclaim<F: PrimeField> {
    /// Bit-slice MLE evaluations at `r*`, in `(j-major, i-minor)` order.
    pub bit_slice_evals: Vec<F>,
    /// Shared evaluation point `r*` from the multi-degree sumcheck.
    pub evaluation_point: Vec<F>,
}

//
// Protocol
//

impl<F> BooleanityChecker<F>
where
    F: InnerTransparentField + Send + Sync + 'static,
    F::Inner: ConstTranscribable + Clone + Send + Sync + Zero + Default,
    F::Modulus: ConstTranscribable,
{
    /// Build the booleanity sumcheck group, to be appended to the
    /// multi-degree sumcheck.
    pub fn prepare_sumcheck_group<const D: usize>(
        transcript: &mut impl Transcript,
        trace_bin_poly: &[DenseMultilinearExtension<BinaryPoly<D>>],
        num_vars: usize,
        field_cfg: &F::Config,
    ) -> Result<(MultiDegreeSumcheckGroup<F>, BoolProverAncillary), BooleanityError<F>> {
        let n = trace_bin_poly.len();
        let one = F::one_with_cfg(field_cfg);
        let zero = F::zero_with_cfg(field_cfg);

        // Order of challenge squeezing must match between prover and verifier.

        // 1. Zerocheck point r.
        let r: Vec<F> = transcript.get_field_challenges(num_vars, field_cfg);

        // 2. Slice batching challenge gamma and its powers.
        let gamma: F = transcript.get_field_challenge(field_cfg);
        let gamma_powers: Vec<F> = powers(gamma, one.clone(), D);

        // 3. Column batching challenge delta and its powers.
        let delta: F = transcript.get_field_challenge(field_cfg);
        let delta_powers: Vec<F> = powers(delta, one.clone(), n);

        // 4. Build eq(r, *) MLE.
        let eq_r = build_eq_x_r_inner(&r, field_cfg)?;

        // 5. Build N*D bit-slice MLEs in canonical (j-major, i-minor) order.
        let mut mles = Vec::with_capacity(add!(n.saturating_mul(D), 1));
        mles.push(eq_r);
        mles.extend(build_witness_bit_slice_mles::<F, D>(
            trace_bin_poly,
            field_cfg,
        ));

        // 6. Build comb_fn (degree 3 in the variables):
        //   eq_r(b) * sum_{j,i} delta^j * gamma^i * v_{j,i}(b) * (v_{j,i}(b) - 1)
        let comb_fn: CombFn<F> = Box::new(move |mle_values: &[F]| {
            let eq_r_val = &mle_values[0];
            let sum =
                batched_booleanity_sum(&mle_values[1..], &delta_powers, &gamma_powers, &zero, &one);
            sum * eq_r_val
        });

        Ok((
            MultiDegreeSumcheckGroup::new(3, mles, comb_fn),
            BoolProverAncillary {
                num_wit_bin_cols: n,
                bit_width: D,
                num_vars,
            },
        ))
    }

    /// Finalize the booleanity proof after the multi-degree sumcheck
    /// completes.
    ///
    /// Mirrors the structure of `CombinedPolyResolver::finalize_prover`:
    /// evaluates each bit-slice MLE at the final sumcheck challenge,
    /// emits the flat `bit_slice_evals` vector, and absorbs it into the
    /// transcript.
    pub fn finalize_prover(
        transcript: &mut impl Transcript,
        sumcheck_prover_state: SumcheckProverState<F>,
        ancillary: BoolProverAncillary,
        field_cfg: &F::Config,
    ) -> Result<BooleanityProof<F>, BooleanityError<F>> {
        debug_assert!(
            sumcheck_prover_state
                .mles
                .iter()
                .all(|mle| mle.num_vars == 1),
            "sumcheck should reduce MLEs to num_vars == 1"
        );

        let last_sumcheck_challenge = sumcheck_prover_state
            .randomness
            .last()
            .expect("sumcheck cannot have had 0 rounds")
            .clone();

        let expected_len = ancillary
            .num_wit_bin_cols
            .saturating_mul(ancillary.bit_width);

        let mut mles = sumcheck_prover_state.mles;
        // Skip MLE 0 = eq_r (verifier recomputes it).
        let bit_slice_evals: Vec<F> = mles
            .drain(1..)
            .map(|mle| {
                mle.evaluate_with_config(slice::from_ref(&last_sumcheck_challenge), field_cfg)
            })
            .collect::<Result<Vec<_>, _>>()?;

        debug_assert_eq!(bit_slice_evals.len(), expected_len);

        let mut transcription_buf: Vec<u8> = vec![0; F::Inner::NUM_BYTES];
        transcript.absorb_random_field_slice(&bit_slice_evals, &mut transcription_buf);

        Ok(BooleanityProof { bit_slice_evals })
    }

    /// Pre-sumcheck half of the booleanity verifier.
    ///
    /// Must run after the CPR `prepare_verifier` and before
    /// `MultiDegreeSumcheck::verify_as_subprotocol` to maintain transcript
    /// ordering.
    pub fn prepare_verifier(
        transcript: &mut impl Transcript,
        claimed_sum: &F,
        num_wit_bin_cols: usize,
        bit_width: usize,
        num_vars: usize,
        field_cfg: &F::Config,
    ) -> Result<BoolVerifierAncillary<F>, BooleanityError<F>> {
        if num_wit_bin_cols == 0 {
            return Err(BooleanityError::NoBinaryPolyColumns);
        }
        if !F::is_zero(claimed_sum) {
            return Err(BooleanityError::NonZeroClaimedSum {
                got: claimed_sum.clone(),
            });
        }

        let one = F::one_with_cfg(field_cfg);

        // Re-squeeze in the same order as the prover.
        let zerocheck_point: Vec<F> = transcript.get_field_challenges(num_vars, field_cfg);
        let gamma: F = transcript.get_field_challenge(field_cfg);
        let gamma_powers: Vec<F> = powers(gamma, one.clone(), bit_width);
        let delta: F = transcript.get_field_challenge(field_cfg);
        let delta_powers: Vec<F> = powers(delta, one, num_wit_bin_cols);

        Ok(BoolVerifierAncillary {
            gamma_powers,
            delta_powers,
            zerocheck_point,
            num_wit_bin_cols,
            bit_width,
            num_vars,
        })
    }

    /// Post-sumcheck half of the booleanity verifier.
    ///
    /// Validates the length of `bit_slice_evals`, recomputes the expected
    /// combination-function evaluation at the shared sumcheck point `r*`
    /// using the received `bit_slice_evals`, and compares it against the
    /// sumcheck's `expected_evaluation`. On success, absorbs
    /// `bit_slice_evals` into the transcript (mirroring the prover's
    /// final absorption in `finalize_prover`).
    pub fn finalize_verifier(
        transcript: &mut impl Transcript,
        proof: BooleanityProof<F>,
        shared_point: Vec<F>,
        expected_evaluation: &F,
        ancillary: BoolVerifierAncillary<F>,
        field_cfg: &F::Config,
    ) -> Result<BoolVerifierSubclaim<F>, BooleanityError<F>> {
        let expected_len = ancillary
            .num_wit_bin_cols
            .saturating_mul(ancillary.bit_width);
        if proof.bit_slice_evals.len() != expected_len {
            return Err(BooleanityError::WrongBitSliceEvalsNumber {
                expected: expected_len,
                got: proof.bit_slice_evals.len(),
            });
        }

        let one = F::one_with_cfg(field_cfg);
        let zero = F::zero_with_cfg(field_cfg);

        // eq(r*, r) — selector value at the shared sumcheck point.
        let eq_r_val = eq_eval(&shared_point, &ancillary.zerocheck_point, one.clone())?;

        // Recompute the comb_fn body at r* using the received bit-slice evals.
        let sum = batched_booleanity_sum(
            &proof.bit_slice_evals,
            &ancillary.delta_powers,
            &ancillary.gamma_powers,
            &zero,
            &one,
        );
        let expected_claim_value = sum * eq_r_val;

        if expected_claim_value != *expected_evaluation {
            return Err(BooleanityError::ClaimValueDoesNotMatch {
                expected: expected_claim_value,
                got: expected_evaluation.clone(),
            });
        }

        let mut transcription_buf: Vec<u8> = vec![0; F::Inner::NUM_BYTES];
        transcript.absorb_random_field_slice(&proof.bit_slice_evals, &mut transcription_buf);

        Ok(BoolVerifierSubclaim {
            bit_slice_evals: proof.bit_slice_evals,
            evaluation_point: shared_point,
        })
    }
}

/// Errors from the booleanity subprotocol.
#[derive(Debug, Error)]
pub enum BooleanityError<F: PrimeField> {
    #[error("no binary polynomial columns provided to booleanity checker")]
    NoBinaryPolyColumns,
    #[error("sumcheck error: {0}")]
    SumcheckError(#[from] SumCheckError<F>),
    #[error("expected booleanity claimed sum is non-zero: got {got}")]
    NonZeroClaimedSum { got: F },
    #[error("wrong number of bit-slice evaluations: expected {expected}, got {got}")]
    WrongBitSliceEvalsNumber { expected: usize, got: usize },
    #[error("booleanity claim value does not match: expected {expected}, got {got}")]
    ClaimValueDoesNotMatch { expected: F, got: F },
    #[error("error evaluating MLE: {0}")]
    MleEvaluationError(#[from] EvaluationError),
    #[error("arithmetic error: {0}")]
    Arith(#[from] ArithErrors),
}

//
// Helpers
//

/// Compute the booleanity residue
///
/// $$
/// sum_{j=0}^{N-1} sum_{i=0}^{D-1}
///   \delta^j * \gamma^i * v_{j,i} * (v_{j,i} - 1)
/// $$
///
/// over a flat `(j-major, i-minor)` slice of bit-slice values
/// (`N = delta_powers.len()`, `D = gamma_powers.len()`,
/// `bit_slice_values.len() == N * D`).
///
/// This is the body of the booleanity sumcheck's combination function
/// (without the leading `eq_r` factor).
fn batched_booleanity_sum<F: PrimeField>(
    bit_slice_values: &[F],
    delta_powers: &[F],
    gamma_powers: &[F],
    zero: &F,
    one: &F,
) -> F {
    let n = delta_powers.len();
    let d = gamma_powers.len();
    debug_assert_eq!(bit_slice_values.len(), mul!(n, d));

    let mut sum = zero.clone();
    for (j, delta_j) in delta_powers.iter().enumerate().take(n) {
        let jd = mul!(j, d);
        for i in 0..d {
            let v = &bit_slice_values[add!(i, jd)];
            let gamma_i = &gamma_powers[i];
            let booleanity = (v.clone() - one) * v;
            sum += booleanity * delta_j * gamma_i;
        }
    }
    sum
}

/// Build per-bit-slice MLEs for a set of binary-poly columns.
///
/// Returns `N * D` MLEs in `(j-major, i-minor)` order:
/// $[v_{0,0}, v_{0,1}, ..., v_{0,D-1}, v_{1,0}, ...]$. The j-th column,
/// i-th bit MLE evaluates at hypercube point `b` to the i-th bit of the
/// row entry `trace_bin_poly[j][b]`.
///
/// This helper is the single source of truth for MLE ordering. Both the
/// booleanity sumcheck group and the `MultipointEval` extension call it
/// to guarantee identical ordering between prover and verifier.
pub fn build_witness_bit_slice_mles<F, const D: usize>(
    trace_bin_poly: &[DenseMultilinearExtension<BinaryPoly<D>>],
    field_cfg: &F::Config,
) -> Vec<DenseMultilinearExtension<F::Inner>>
where
    F: PrimeField,
    F::Inner: Clone + Send + Sync,
{
    let zero_inner = F::zero_with_cfg(field_cfg).into_inner();
    let one_inner = F::one_with_cfg(field_cfg).into_inner();

    let mut out = Vec::with_capacity(trace_bin_poly.len().saturating_mul(D));
    for col in trace_bin_poly {
        let num_vars = col.num_vars;
        let n_rows = col.evaluations.len();
        for i in 0..D {
            let mut evals: Vec<F::Inner> = Vec::with_capacity(n_rows);
            for entry in col.iter() {
                let bit = entry
                    .iter()
                    .nth(i)
                    .expect("BinaryPoly<D> has D coefficients")
                    .into_inner();
                evals.push(if bit {
                    one_inner.clone()
                } else {
                    zero_inner.clone()
                });
            }
            out.push(DenseMultilinearExtension {
                num_vars,
                evaluations: evals,
            });
        }
    }
    out
}

//
// Tests
//

#[cfg(test)]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
mod tests {
    use super::*;
    use crate::sumcheck::multi_degree::MultiDegreeSumcheck;
    use crypto_bigint::{U128, const_monty_params};
    use crypto_primitives::crypto_bigint_const_monty::ConstMontyField;
    use zinc_transcript::Blake3Transcript;

    const_monty_params!(TestParams, U128, "00000000b933426489189cb5b47d567f");
    type F = ConstMontyField<TestParams, { U128::LIMBS }>;

    const D: usize = 4;

    /// Build a `BinaryPoly<D>` whose i-th coefficient is `bits[i]` (LSB-first).
    fn binp_from_bits(bits: [bool; D]) -> BinaryPoly<D> {
        let value: u64 = bits
            .iter()
            .enumerate()
            .filter_map(|(i, &b)| b.then_some(1_u64 << i))
            .fold(0_u64, |acc, mask| acc | mask);
        BinaryPoly::from(value)
    }

    /// Build a single binary-poly trace column from a vec of row-bit patterns.
    fn build_col(rows: Vec<[bool; D]>) -> DenseMultilinearExtension<BinaryPoly<D>> {
        let num_vars = (rows.len() as f64).log2().round() as usize;
        debug_assert_eq!(rows.len(), 1usize << num_vars);
        let zero = BinaryPoly::zero();
        let mut evals: Vec<BinaryPoly<D>> = rows.into_iter().map(binp_from_bits).collect();
        // Sanity: replace any padding (none expected here) with zero.
        if evals.is_empty() {
            evals.push(zero);
        }
        DenseMultilinearExtension {
            num_vars,
            evaluations: evals,
        }
    }

    /// Helper: 4 rows of all-zero bits.
    fn zero_col_4rows() -> DenseMultilinearExtension<BinaryPoly<D>> {
        build_col(vec![[false; D]; 4])
    }

    fn make_transcript() -> Blake3Transcript {
        let mut t = Blake3Transcript::default();
        t.absorb_slice(b"booleanity test");
        t
    }

    /// Run prepare → MultiDegreeSumcheck::prove → finalize on the prover,
    /// then prepare → verify → finalize on the verifier, asserting the
    /// outcome via `expect_ok`.
    fn run_roundtrip(
        cols: &[DenseMultilinearExtension<BinaryPoly<D>>],
        num_vars: usize,
        tamper_proof: impl FnOnce(&mut BooleanityProof<F>),
        expect_err_is: impl FnOnce(&BooleanityError<F>) -> bool,
        expect_ok: bool,
    ) {
        let cfg = &();

        // Prover side.
        let mut pt = make_transcript();
        let (group, anc) =
            BooleanityChecker::<F>::prepare_sumcheck_group::<D>(&mut pt, cols, num_vars, cfg)
                .expect("prepare_sumcheck_group failed");
        let (md_proof, states) =
            MultiDegreeSumcheck::<F>::prove_as_subprotocol(&mut pt, vec![group], num_vars, cfg);
        let state = states.into_iter().next().unwrap();
        let mut proof = BooleanityChecker::<F>::finalize_prover(&mut pt, state, anc, cfg)
            .expect("finalize prover");
        tamper_proof(&mut proof);

        // Verifier side.
        let mut vt = make_transcript();
        let anc_v = BooleanityChecker::<F>::prepare_verifier(
            &mut vt,
            &md_proof.claimed_sums()[0],
            cols.len(),
            D,
            num_vars,
            cfg,
        )
        .expect("prepare_verifier (claimed sum should be zero for honest input)");
        let md_subclaims =
            MultiDegreeSumcheck::<F>::verify_as_subprotocol(&mut vt, num_vars, &md_proof, cfg)
                .expect("md verify");

        let res = BooleanityChecker::<F>::finalize_verifier(
            &mut vt,
            proof,
            md_subclaims.point().to_vec(),
            &md_subclaims.expected_evaluations()[0],
            anc_v,
            cfg,
        );

        if expect_ok {
            res.expect("finalize_verifier should succeed");
        } else {
            let err = res.expect_err("finalize_verifier should fail");
            assert!(
                expect_err_is(&err),
                "unexpected booleanity error variant: {err:?}",
            );
        }
    }

    #[test]
    fn happy_path_all_bits() {
        // Two columns, 4 rows each, all valid bit patterns.
        let c0 = build_col(vec![
            [true, false, true, false],
            [false, true, false, true],
            [true, true, false, false],
            [false, false, true, true],
        ]);
        let c1 = build_col(vec![
            [false, false, false, false],
            [true, true, true, true],
            [false, true, true, false],
            [true, false, false, true],
        ]);
        run_roundtrip(&[c0, c1], 2, |_| {}, |_| false, true);
    }

    #[test]
    fn empty_column_set_is_supported_by_helper() {
        // The helper itself returns an empty Vec for N = 0. We don't run a
        // sumcheck (MultiDegreeSumcheck requires at least one group) but we
        // do exercise the helper for the empty case.
        let cols: Vec<DenseMultilinearExtension<BinaryPoly<D>>> = vec![];
        let mles = build_witness_bit_slice_mles::<F, D>(&cols, &());
        assert!(mles.is_empty());
    }

    #[test]
    fn tampered_bit_slice_evals_rejected() {
        let c0 = zero_col_4rows();
        // Tamper one entry of bit_slice_evals to a non-{0,1} value. The
        // booleanity comb_fn is v*(v-1) which vanishes on {0,1}, so we must
        // pick a non-bit value to actually corrupt the recomputed residue.
        run_roundtrip(
            &[c0],
            2,
            |proof| {
                proof.bit_slice_evals[0] += F::from(7u32);
            },
            |err| matches!(err, BooleanityError::ClaimValueDoesNotMatch { .. }),
            false,
        );
    }

    #[test]
    fn non_bit_witness_rejected() {
        // Build cols honestly, then directly tamper one bit-slice eval in
        // the proof to simulate a malicious prover whose actual witness
        // had a non-bit entry. Note: we cannot inject a non-bit entry
        // through `BinaryPoly<D>` (it stores `Boolean`), so this is the
        // verifier-visible failure mode of a non-bit witness:
        // `bit_slice_evals` whose recomputed booleanity residue is
        // non-zero at the sumcheck point.
        let c0 = build_col(vec![
            [true, true, false, false],
            [false, true, true, false],
            [false, false, false, true],
            [true, false, true, true],
        ]);
        run_roundtrip(
            &[c0],
            2,
            |proof| {
                // Two of v=0/1 nudges that produce a non-{0,1} residue.
                proof.bit_slice_evals[1] += F::from(3u32);
            },
            |err| matches!(err, BooleanityError::ClaimValueDoesNotMatch { .. }),
            false,
        );
    }

    #[test]
    fn wrong_bit_slice_evals_length_rejected() {
        let c0 = zero_col_4rows();
        run_roundtrip(
            &[c0],
            2,
            |proof| {
                proof.bit_slice_evals.pop();
            },
            |err| matches!(err, BooleanityError::WrongBitSliceEvalsNumber { .. }),
            false,
        );
    }
}
