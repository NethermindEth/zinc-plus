//! Booleanity (binary-polynomial lookup) argument.
//!
//! Proves that every coefficient of every witness binary-polynomial column is
//! a bit $\in \\{0,1\\}$. The argument is structured as a single
//! [`MultiDegreeSumcheckGroup`] of degree 3, batched alongside the existing
//! CPR group with shared randomness, and verifies its bit-slice claims at
//! the multi-degree sumcheck output point $r^\star$ against the projected
//! parent column evaluations (CPR `up_evals`) using the $\psi_a$ identity.
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
//!   \sum_{k=0}^{N*D-1} \alpha^k *
//!     \widetilde{v_k}(b) * (\widetilde{v_k}(b) - 1)  =  0
//! $$
//!
//! with a single batching challenge $\alpha$ over the flat
//! $(j\text{-major}, i\text{-minor})$ index $k = j \cdot D + i$, and
//! zerocheck point $r$. After the sumcheck reaches $r^\star$, the prover
//! sends `bit_slice_evals` $= (\widetilde{v_{j,i}}(r^\star))$ to the
//! verifier.
//!
//! # Bit-decomposition consistency check at $r^\star$
//!
//! The bit-slice claims at $r^\star$ are tied back to the committed parent
//! columns via the $\psi_a$ identity at the MLE level: for every binary-poly
//! column $u_j$,
//!
//! $$
//!   \psi_a\bigl(\widetilde{u_j}(r^\star)\bigr)
//!     = \sum_{i=0}^{D-1} a^i \cdot \widetilde{v_{j,i}}(r^\star).
//! $$
//!
//! The left-hand side is already available to the verifier as
//! `cpr_subclaim.up_evals[num_pub_bin + j]` (the projected MLE evaluation of
//! the $j$-th witness binary-poly column at $r^\star$, coming out of the
//! CPR group). The right-hand side is rebuilt from the prover-supplied
//! `bit_slice_evals`. With overwhelming probability over the random $a$
//! used by $\psi_a$, this linear pin-down forces the `bit_slice_evals` to
//! be the true bit decomposition of $u_j$ at $r^\star$.
//!
//! Because the consistency check closes at $r^\star$, the bit-slice MLEs
//! are **not** routed through `MultipointEval`: they participate only in
//! the booleanity sumcheck group and then vanish from the rest of the
//! protocol.

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
use zinc_utils::{add, inner_transparent_field::InnerTransparentField, powers};

//
// Structs
//

/// Booleanity sumcheck group constructor / verifier.
pub struct BooleanityChecker<F: InnerTransparentField>(PhantomData<F>);

/// Proof produced by the booleanity prover. Carries only the flat list of
/// bit-slice MLE evaluations at the multi-degree sumcheck output point
/// $r*$. The bit-decomposition consistency at $r^\star$ is verified by
/// the protocol-layer caller against the CPR `up_evals` of the witness
/// binary-poly columns.
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
    /// Powers of the single batching challenge over the flat
    /// `(j-major, i-minor)` index: `[1, alpha, ..., alpha^{N*D - 1}]`.
    pub alpha_powers: Vec<F>,
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
/// `bit_slice_evals` at the shared multi-degree sumcheck point `r*``.
/// The protocol-layer caller passes these into
/// [`BooleanityChecker::verify_bit_decomposition_consistency`] together
/// with the CPR `up_evals` for the witness binary-poly columns to close
/// the booleanity argument entirely inside step 4.
#[derive(Clone, Debug)]
pub struct BoolVerifierSubclaim<F: PrimeField> {
    /// Bit-slice MLE evaluations at `r*`, in `(j-major, i-minor)` order.
    pub bit_slice_evals: Vec<F>,
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

        // 2. Single batching challenge alpha over the flat (j-major, i-minor) index.
        //    Powers vector has length N*D.
        let alpha: F = transcript.get_field_challenge(field_cfg);
        let alpha_powers: Vec<F> = powers(alpha, one.clone(), n.saturating_mul(D));

        // 3. Build eq(r, *) MLE.
        let eq_r = build_eq_x_r_inner(&r, field_cfg)?;

        // 4. Build N*D bit-slice MLEs in canonical (j-major, i-minor) order.
        let mut mles = Vec::with_capacity(add!(n.saturating_mul(D), 1));
        mles.push(eq_r);
        mles.extend(build_witness_bit_slice_mles::<F, D>(
            trace_bin_poly,
            field_cfg,
        ));

        // 5. Build comb_fn (degree 3 in the variables):
        //   eq_r(b) * sum_k alpha^k * v_k(b) * (v_k(b) - 1)
        let comb_fn: CombFn<F> = Box::new(move |mle_values: &[F]| {
            let eq_r_val = &mle_values[0];
            let sum = batched_booleanity_sum(&mle_values[1..], &alpha_powers, &zero, &one);
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
        let alpha: F = transcript.get_field_challenge(field_cfg);
        let alpha_powers: Vec<F> = powers(alpha, one, num_wit_bin_cols.saturating_mul(bit_width));

        Ok(BoolVerifierAncillary {
            alpha_powers,
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
    /// sumcheck's `expected_eval`. On success, absorbs `bit_slice_evals` into
    /// the transcript (mirroring the prover's final absorption in
    /// `finalize_prover`).
    pub fn finalize_verifier(
        transcript: &mut impl Transcript,
        proof: BooleanityProof<F>,
        shared_point: Vec<F>,
        expected_eval: &F,
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
        let sum =
            batched_booleanity_sum(&proof.bit_slice_evals, &ancillary.alpha_powers, &zero, &one);
        let expected_claim_value = sum * eq_r_val;

        if expected_claim_value != *expected_eval {
            return Err(BooleanityError::ClaimValueDoesNotMatch {
                expected: expected_claim_value,
                got: expected_eval.clone(),
            });
        }

        let mut transcription_buf: Vec<u8> = vec![0; F::Inner::NUM_BYTES];
        transcript.absorb_random_field_slice(&proof.bit_slice_evals, &mut transcription_buf);

        Ok(BoolVerifierSubclaim {
            bit_slice_evals: proof.bit_slice_evals,
        })
    }

    /// Verify the bit-slice claims at $r^\star$ against the projected
    /// parent column evaluations (CPR `up_evals`) using the $\psi_a$
    /// identity.
    ///
    /// For each witness binary-poly column $j$, checks
    ///
    /// $$
    ///   \texttt{parent\_evals}[j] \;=\; \sum_{i=0}^{D-1} a^i \cdot
    ///     \texttt{bit\_slice\_evals}[j \cdot D + i]
    /// $$
    ///
    /// where $a$ is the random projection element used by $\psi_a$ to send
    /// $F_q[X] \to F_q$. The left-hand side is the MLE evaluation at $r^\star$
    /// of the projected parent column (i.e. `cpr_subclaim.up_evals[..]`
    /// restricted to the witness binary-poly slice). The right-hand side is
    /// the linear recombination of the prover-supplied bit-slice evals.
    ///
    /// With overwhelming probability over the random $a$, agreement forces
    /// `bit_slice_evals` to be the true bit decomposition of the committed
    /// parent column at $r^\star$.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn verify_bit_decomposition_consistency(
        bit_slice_evals: &[F],
        parent_evals: &[F],
        projecting_element: &F,
        bits_per_col: usize,
        field_cfg: &F::Config,
    ) -> Result<(), BooleanityError<F>> {
        let expected_len = parent_evals.len().saturating_mul(bits_per_col);
        if bit_slice_evals.len() != expected_len {
            return Err(BooleanityError::WrongBitSliceEvalsNumber {
                expected: expected_len,
                got: bit_slice_evals.len(),
            });
        }

        if bits_per_col == 0 || parent_evals.is_empty() {
            return Ok(());
        }

        let zero = F::zero_with_cfg(field_cfg);
        let one = F::one_with_cfg(field_cfg);

        let a_powers: Vec<F> = powers(projecting_element.clone(), one, bits_per_col);

        for (col_idx, parent_eval) in parent_evals.iter().enumerate() {
            let base = col_idx.saturating_mul(bits_per_col);
            let mut recombined = zero.clone();
            for (a_pow, bit_eval) in a_powers
                .iter()
                .zip(&bit_slice_evals[base..base + bits_per_col])
            {
                recombined += a_pow.clone() * bit_eval;
            }

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
    #[error(
        "bit-decomposition consistency mismatch on witness binary-poly column {col_idx}: \
         recombined Sum_i a^i * bit_slice = {got}, expected parent eval {expected}"
    )]
    ConsistencyMismatch { col_idx: usize, got: F, expected: F },
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
/// sum_{k=0}^{N*D - 1} \alpha^k * v_k * (v_k - 1)
/// $$
///
/// over a flat $(j\text{-major}, i\text{-minor})$ slice of bit-slice
/// values with $k = j \cdot D + i$. `alpha_powers.len() ==
/// bit_slice_values.len() == N * D`.
///
/// This is the body of the booleanity sumcheck's combination function
/// (without the leading `eq_r` factor).
fn batched_booleanity_sum<F: PrimeField>(
    bit_slice_values: &[F],
    alpha_powers: &[F],
    zero: &F,
    one: &F,
) -> F {
    debug_assert_eq!(bit_slice_values.len(), alpha_powers.len());

    let mut sum = zero.clone();
    for (v, alpha_k) in bit_slice_values.iter().zip(alpha_powers.iter()) {
        let booleanity = (v.clone() - one) * v;
        sum += booleanity * alpha_k;
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

    /// Honest `bit_slice_evals` recombine to the parent eval via $\psi_a$.
    #[test]
    fn consistency_check_happy_path() {
        let cfg = &();
        let one = F::one_with_cfg(cfg);
        let two = one + one;
        let three = two + one;
        let a = three; // arbitrary nonzero projection element

        // Two columns, D bit-slices each, populated with arbitrary field
        // values. The "true" parent eval is the linear recombination via
        // a^i, so we recompute it directly.
        let n = 2usize;
        let bit_slice_evals: Vec<F> = (0..n * D)
            .map(|k| F::from(((k as u32) * 7 + 11) % 13))
            .collect();

        let mut parent_evals: Vec<F> = Vec::with_capacity(n);
        let zero = F::zero_with_cfg(cfg);
        for j in 0..n {
            let mut acc = zero;
            let mut a_pow = one;
            for i in 0..D {
                acc += a_pow * bit_slice_evals[j * D + i];
                a_pow *= &a;
            }
            parent_evals.push(acc);
        }

        BooleanityChecker::<F>::verify_bit_decomposition_consistency(
            &bit_slice_evals,
            &parent_evals,
            &a,
            D,
            cfg,
        )
        .expect("honest recombination should match");
    }

    /// Tampered `bit_slice_evals[k]` triggers `ConsistencyMismatch` with
    /// `col_idx = k / D`.
    #[test]
    fn consistency_check_tampered_bit_slice_rejected() {
        let cfg = &();
        let one = F::one_with_cfg(cfg);
        let two = one + one;
        let a = two + one; // 3

        let n = 2usize;
        let mut bit_slice_evals: Vec<F> = (0..n * D)
            .map(|k| F::from(((k as u32) * 7 + 11) % 13))
            .collect();

        let zero = F::zero_with_cfg(cfg);
        let mut parent_evals: Vec<F> = Vec::with_capacity(n);
        for j in 0..n {
            let mut acc = zero;
            let mut a_pow = one;
            for i in 0..D {
                acc += a_pow * bit_slice_evals[j * D + i];
                a_pow *= a;
            }
            parent_evals.push(acc);
        }

        let tamper_idx = D + 1; // column j = 1, bit i = 1
        bit_slice_evals[tamper_idx] += &two;

        let err = BooleanityChecker::<F>::verify_bit_decomposition_consistency(
            &bit_slice_evals,
            &parent_evals,
            &a,
            D,
            cfg,
        )
        .expect_err("tamper should be rejected");

        assert!(
            matches!(err, BooleanityError::ConsistencyMismatch { col_idx, .. } if col_idx == 1),
            "expected ConsistencyMismatch on column 1, got {err:?}"
        );
    }

    /// Length mismatch is reported as `WrongBitSliceEvalsNumber`.
    #[test]
    fn consistency_check_wrong_length_rejected() {
        let cfg = &();
        let one = F::one_with_cfg(cfg);
        let a = one + one;

        let parent_evals = vec![F::zero_with_cfg(cfg); 2];
        let bit_slice_evals = vec![F::zero_with_cfg(cfg); 2 * D - 1];

        let err = BooleanityChecker::<F>::verify_bit_decomposition_consistency(
            &bit_slice_evals,
            &parent_evals,
            &a,
            D,
            cfg,
        )
        .expect_err("length mismatch should be rejected");

        assert!(
            matches!(err, BooleanityError::WrongBitSliceEvalsNumber { .. }),
            "expected WrongBitSliceEvalsNumber, got {err:?}"
        );
    }
}
