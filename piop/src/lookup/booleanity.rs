//! Booleanity (binary-polynomial lookup) argument.
//!
//! Proves that every coefficient of every witness binary-polynomial column is
//! a bit $\in \\{0,1\\}$. The argument is structured as a single
//! [`MultiDegreeSumcheckGroup`] of degree 3, batched alongside the existing
//! CPR group with shared randomness, and emits bit-slice claims at the
//! multi-degree sumcheck output point $r^\star$ for the protocol layer's
//! $\alpha'$ bridge into multipoint-eval (see "Soundness bridge" below).
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
//! # Soundness bridge to multipoint-eval ($\alpha'$)
//!
//! Booleanity does **not** itself close the bit-decomposition consistency
//! at $r^\star$. The protocol-layer caller squeezes a fresh challenge
//! $\alpha'$ from the transcript **after** `bit_slice_evals` are absorbed
//! (the absorption is performed by [`BooleanityChecker::finalize_prover`] /
//! `finalize_verifier`), then, for each witness binary-poly column $u_j$,
//! *appends* one extra column to the multipoint-eval input list:
//! - an extra MLE $\widetilde{\psi_{\alpha'}(u_j)}$ (the prover-side
//!   $\alpha'$-projection of $u_j$), and
//! - an extra up-eval scalar $c'_j \;:=\; \sum_{i=0}^{D-1}
//!   b_{j,i}\,\alpha'^{\,i}$ (where $b_{j,i} = \widetilde{v_{j,i}}(r^\star)$
//!   from `bit_slice_evals`).
//!
//! No `ShiftSpec` references the appended slot, so down-evals / shifts
//! pass through unchanged: row-shifted projections of witness binary-poly
//! columns inherit booleanity from the un-shifted, $\psi_a$-projected
//! slot they already reference in the multipoint-eval sumcheck.
//!
//! Multipoint-eval / PCS is $\psi$-oblivious, so the appended slot
//! combined with the lifted-evals step evaluating the corresponding
//! $\bar u_j$ at $\alpha'$ enforces
//! $\widetilde{\psi_{\alpha'}(u_j)}(r^\star) = c'_j$ via the PCS chain.
//! By Schwartz–Zippel on the indeterminate $X$, if `bit_slice_evals` are
//! not the true bit-decomposition then equality fails with probability
//! $\le (D-1)/|F|$, so the verifier rejects.

use crate::{
    CombFn,
    sumcheck::{
        SumCheckError,
        multi_degree::{MultiDegreeSumcheckGroup, Round1FastPath, Round1Output},
        prover::ProverState as SumcheckProverState,
    },
};
use crypto_primitives::PrimeField;
use itertools::Itertools;
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

/// Proof produced by the booleanity prover.
///
/// Carries `bit_slice_evals`: per-column bit-slice MLE evaluations at the
/// multi-degree sumcheck output point $r^\star$, flat in
/// `(j-major, i-minor)` order with length `num_wit_bin_cols * D`. It is
/// absorbed into the transcript by `finalize_prover` / `finalize_verifier`,
/// after which the protocol layer squeezes the $\alpha'$ challenge
/// used to bind the bit-decomposition into the multipoint-eval boundary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BooleanityProof<F: PrimeField> {
    /// Flat list of `\widetilde{v_{j,i}}(r*)`, ordered `(j-major, i-minor)`.
    /// Length = `num_wit_bin_cols * D`.
    pub bit_slice_evals: Vec<F>,
}

delegate_transcribable!(BooleanityProof<F> { bit_slice_evals: Vec<F> }
    where F: PrimeField, F::Integer: ConstTranscribable);

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
/// Carries the (sumcheck-residue-validated) bit-slice evaluations at the
/// shared multi-degree sumcheck point $r^\star$. The protocol-layer
/// caller squeezes $\alpha'$ from the transcript (after this struct
/// is produced) and uses it together with these values to *append* one
/// extra multipoint-eval column (and up-eval $c'_j$) per witness
/// binary-poly column — see the module-level docs.
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
    F::Integer: ConstTranscribable + Clone + Send + Sync + Zero + Default,
{
    /// Build the booleanity sumcheck group, to be appended to the
    /// multi-degree sumcheck.
    ///
    /// Installs a [`BooleanityRound1FastPath`] hook on the returned group:
    /// the round-1 polynomial and the post-round-1 MLE fold are computed
    /// in closed form directly from the `BinaryPoly`-typed trace columns
    /// (no full-size `F`-valued bit-slice MLEs are ever materialized).
    /// Verifier doesn't see a difference.
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

        // 3. comb_fn (degree 3 in the variables), used by `MultiDegreeSumcheck` in
        //    rounds 2..num_vars:
        //
        //       eq_r(b) * sum_k alpha^k * v_k(b) * (v_k(b) - 1)
        //
        //    The fast path emits the round-1 message and the post-round-1
        //    folded MLEs directly, so this closure is *not* invoked in round 1.
        let comb_fn: CombFn<F> = {
            let alpha_powers = alpha_powers.clone();
            let one = one.clone();
            let zero = zero.clone();
            Box::new(move |mle_values: &[F]| {
                let eq_r_val = &mle_values[0];
                let sum = batched_booleanity_sum(&mle_values[1..], &alpha_powers, &zero, &one);
                sum * eq_r_val
            })
        };

        // 4. Precompute `E_other(b')` = eq(b', r[1..]) for the fast path.
        //    `build_eq_x_r_inner` rejects empty inputs, so handle num_vars == 1
        //    explicitly by emitting the empty-product table `[1]`.
        let eq_other_table: Vec<F::Inner> = if num_vars <= 1 {
            vec![one.clone().into_inner()]
        } else {
            build_eq_x_r_inner(&r[1..], field_cfg)?.evaluations
        };

        let r_first_coord = r
            .into_iter()
            .next()
            .expect("num_vars >= 1 guarantees a first coordinate");

        let fast_path = BooleanityRound1FastPath::<F, D> {
            binary_cols: trace_bin_poly.to_vec(),
            alpha_powers,
            eq_other_table,
            r_first_coord,
            num_vars,
        };

        Ok((
            MultiDegreeSumcheckGroup::new_with_fast_path(
                3,
                Vec::new(),
                comb_fn,
                Box::new(fast_path),
            ),
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

        let active_len = ancillary
            .num_wit_bin_cols
            .saturating_mul(ancillary.bit_width);

        let mut mles = sumcheck_prover_state.mles;
        debug_assert_eq!(mles.len(), add!(1, active_len));

        // Skip MLE 0 = eq_r (verifier recomputes it).
        let bit_slice_evals: Vec<F> = mles
            .drain(1..)
            .map(|mle| {
                mle.evaluate_with_config(slice::from_ref(&last_sumcheck_challenge), field_cfg)
            })
            .collect::<Result<Vec<_>, _>>()?;

        debug_assert_eq!(bit_slice_evals.len(), active_len);

        let mut transcription_buf: Vec<u8> = vec![0; F::Integer::NUM_BYTES];
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
    /// Validates the length of `bit_slice_evals`, recomputes the
    /// booleanity residue at the shared sumcheck point $r^\star$, and
    /// compares it against the sumcheck's `expected_eval`. On success,
    /// absorbs `bit_slice_evals` into the transcript.
    ///
    /// The bit-decomposition consistency at $r^\star$ is **not** checked
    /// here; the protocol layer squeezes $\alpha'$ and appends an extra
    /// multipoint-eval column (and up-eval $c'_j$) per witness
    /// binary-poly column derived from `bit_slice_evals` — see the
    /// module-level docs.
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

        let mut transcription_buf: Vec<u8> = vec![0; F::Integer::NUM_BYTES];
        transcript.absorb_random_field_slice(&proof.bit_slice_evals, &mut transcription_buf);

        Ok(BoolVerifierSubclaim {
            bit_slice_evals: proof.bit_slice_evals,
        })
    }
}

//
// Round-1 fast path
//

/// Closed-form round-1 message and round-1 fold for the booleanity sumcheck
/// group. Bit-identical to running [`SumcheckProverState::prove_round`].
///
/// Algebraic identities exploited:
///
/// 1. Bit-slice $v(v-1)$ collapse: for $v(X, b') = (1-X) \cdot A + X \cdot B$
///    with $A, B \in \\{0,1\\}$, $$ v(X, b') (v(X, b') - 1) = (A \oplus B)
///    \cdot X(X-1). $$ Only the bit `A XOR B` depends on data; the factor
///    $X(X-1)$ is universal.
/// 2. $eq_r$ factorization on the first variable: $$\widetilde{eq_r}(X, b') =
///    e_0(X) \cdot E_{\text{other}}(b'),$$ with $e_0(X) = (1-X)(1-r_0) + X r_0$
///    and $E_{\text{other}}(b') = \widetilde{eq_{r_{[1..]}}}(b')$.
///
/// The round-1 polynomial collapses to
/// $$
/// p_1(X) = e_0(X) \cdot X(X-1) \cdot T_1, \quad T_1 = \sum_{b'} S(b')
///   E_{\text{other}}(b'), \quad S(b') = \sum_k \alpha^k (A_k(b') \oplus
///   B_k(b')),
/// $$
/// The round-1 fold consumes the verifier challenge $r_1$ and writes each
/// post-fold bit-slice entry via a 4-way table lookup keyed by $(A, B) \in
/// \{0,1\}^2$.
struct BooleanityRound1FastPath<F: PrimeField, const D: usize> {
    /// Per-column binary trace (cloned to avoid lifetime issues).
    binary_cols: Vec<DenseMultilinearExtension<BinaryPoly<D>>>,
    /// Powers of the batching challenge over the flat `(j-major, i-minor)`
    /// index: `[1, alpha, ..., alpha^{N*D - 1}]`.
    alpha_powers: Vec<F>,
    /// Evaluations of $E_{\text{other}}(b') = \widetilde{eq_{r_{[1..]}}}(b')$
    /// on `{0,1}^{num_vars - 1}`. When `num_vars == 1` this is the single
    /// entry `[1]` (the empty product).
    eq_other_table: Vec<F::Inner>,
    /// The first coordinate of the zerocheck point, $r_0$.
    r_first_coord: F,
    /// Number of variables of the sumcheck.
    num_vars: usize,
}

impl<F, const D: usize> Round1FastPath<F> for BooleanityRound1FastPath<F, D>
where
    F: InnerTransparentField + Send + Sync + 'static,
    F::Integer: Zero + Clone + Send + Sync,
{
    fn round_1_message(&self, config: &F::Config) -> Round1Output<F> {
        let zero = F::zero_with_cfg(config);
        let one = F::one_with_cfg(config);
        let half_n: usize = 1_usize << self.num_vars.saturating_sub(1);

        // T_1 = sum_{b'} S(b') * E_other(b').
        let mut t1 = zero.clone();
        for b_prime in 0..half_n {
            let mut s_b = zero.clone();
            for (j, col) in self.binary_cols.iter().enumerate() {
                let two_b = mul!(2, b_prime);
                let row_a = &col.evaluations[two_b];
                let row_b = &col.evaluations[add!(two_b, 1)];
                for (i, (a_bit, b_bit)) in row_a.iter().zip(row_b.iter()).enumerate() {
                    if a_bit != b_bit {
                        let k = add!(i, mul!(j, D));
                        s_b += &self.alpha_powers[k];
                    }
                }
            }
            let eq_other = F::new_unchecked_with_cfg(self.eq_other_table[b_prime].clone(), config);
            t1 += s_b * &eq_other;
        }

        // Closed-form tail evaluations (with X(X-1) = 0 at X=1):
        //   p_1(1) = 0
        //   p_1(2) = e_0(2) * 2*1*T_1 = 2 * (3*r_0 - 1) * T_1
        //   p_1(3) = e_0(3) * 3*2*T_1 = 6 * (5*r_0 - 2) * T_1
        let two = one.clone() + &one;
        let three = two.clone() + &one;
        let five = three.clone() + &two;
        let six = three.clone() * &two;

        let three_r0_minus_one = three * &self.r_first_coord - &one;
        let five_r0_minus_two = five * &self.r_first_coord - &two;

        let p1_at_1 = zero.clone();
        let p1_at_2 = two * three_r0_minus_one * &t1;
        let p1_at_3 = six * five_r0_minus_two * &t1;

        Round1Output {
            // Booleanity is a zerocheck; the asserted sum p_1(0)+p_1(1) is 0.
            asserted_sum: zero,
            tail_evaluations: vec![p1_at_1, p1_at_2, p1_at_3],
        }
    }

    fn fold_with_challenge(
        self: Box<Self>,
        r_1: &F,
        config: &F::Config,
    ) -> Vec<DenseMultilinearExtension<F::Inner>> {
        // Edge case: when the sumcheck has only one round, the standard prover never
        // performs the folding.
        //
        // The post-round-1 state expected by `finalize_prover` is the un-folded
        // 1-variable MLE bundle (it then evaluates each MLE at `r_1`). Return that
        // directly without applying the fold.
        if self.num_vars <= 1 {
            let r = [self.r_first_coord.clone()];
            let eq_r = build_eq_x_r_inner(&r, config)
                .expect("num_vars == 1 implies a single-coordinate r");
            let mut out: Vec<DenseMultilinearExtension<F::Inner>> =
                Vec::with_capacity(self.binary_cols.len().saturating_mul(D).saturating_add(1));
            out.push(eq_r);
            out.extend(build_witness_bit_slice_mles::<F, D>(
                &self.binary_cols,
                config,
            ));
            return out;
        }

        let n_cols = self.binary_cols.len();
        let total_bit_slices = n_cols.saturating_mul(D);
        let new_num_vars = self.num_vars.saturating_sub(1);
        let half_n: usize = 1_usize << new_num_vars;

        let one = F::one_with_cfg(config);

        // 4-way table for bit-slice fold: index = (A as usize) << 1 | (B as usize).
        //   (0,0) -> 0
        //   (0,1) -> r_1
        //   (1,0) -> 1 - r_1
        //   (1,1) -> 1
        let zero_inner = F::zero_with_cfg(config).into_inner();
        let one_inner = one.inner().clone();
        let r1_inner = r_1.inner().clone();
        let one_minus_r1_inner = (one.clone() - r_1).into_inner();
        let bit_fold_table: [F::Inner; 4] = [zero_inner, r1_inner, one_minus_r1_inner, one_inner];

        // eq_r fold: e_0(r_1) * E_other(b').
        // e_0(r_1) = (1 - r_1)(1 - r_0) + r_1 * r_0.
        let one_minus_r0 = one.clone() - &self.r_first_coord;
        let one_minus_r1 = one - r_1;
        let e_0_r1: F = (one_minus_r1 * &one_minus_r0) + (r_1.clone() * &self.r_first_coord);

        let folded_eq_r_evals = (0..half_n)
            .map(|b_prime| {
                let e_other =
                    F::new_unchecked_with_cfg(self.eq_other_table[b_prime].clone(), config);
                (e_0_r1.clone() * &e_other).into_inner()
            })
            .collect_vec();

        // Length = N*D + 1
        let mut out: Vec<DenseMultilinearExtension<F::Inner>> =
            Vec::with_capacity(total_bit_slices.saturating_add(1));
        out.push(DenseMultilinearExtension {
            num_vars: new_num_vars,
            evaluations: folded_eq_r_evals,
        });

        // Bit-slice fold via 4-way table lookup; emit MLEs in
        // (j-major, i-minor) order.
        for col in &self.binary_cols {
            let mut col_evals: Vec<Vec<F::Inner>> =
                (0..D).map(|_| Vec::with_capacity(half_n)).collect();
            for b_prime in 0..half_n {
                let two_b = mul!(2, b_prime);
                let row_a = &col.evaluations[two_b];
                let row_b = &col.evaluations[add!(two_b, 1)];
                for (i, (a_bit, b_bit)) in row_a.iter().zip(row_b.iter()).enumerate() {
                    let idx =
                        (usize::from(a_bit.into_inner()) << 1) | usize::from(b_bit.into_inner());
                    col_evals[i].push(bit_fold_table[idx].clone());
                }
            }
            for evals in col_evals {
                out.push(DenseMultilinearExtension {
                    num_vars: new_num_vars,
                    evaluations: evals,
                });
            }
        }

        out
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
/// Used by the `num_vars == 1` fallback of
/// [`BooleanityRound1FastPath::fold_with_challenge`] (where the standard
/// prover never performs the round-1 fold, so the fast path must emit
/// the un-folded MLE bundle directly) and by booleanity unit tests.
fn build_witness_bit_slice_mles<F, const D: usize>(
    trace_bin_poly: &[DenseMultilinearExtension<BinaryPoly<D>>],
    field_cfg: &F::Config,
) -> Vec<DenseMultilinearExtension<F::Inner>>
where
    F: PrimeField,
    F::Integer: Clone + Send + Sync,
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
    clippy::arithmetic_side_effects,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::clone_on_copy
)]
mod tests {
    use super::*;
    use crate::sumcheck::multi_degree::MultiDegreeSumcheck;
    use crypto_bigint::{U128, const_monty_params};
    use crypto_primitives::crypto_bigint_const_monty::ConstMontyField;
    use num_traits::One;
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
        debug_assert_eq!(rows.len(), 1_usize << num_vars);
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
        let mut sumcheck_outputs = MultiDegreeSumcheck::<F>::prove_as_subprotocol(
            &mut pt,
            vec![(vec![group], cfg)],
            num_vars,
            cfg,
        );
        let (md_proof, states) = sumcheck_outputs.pop().expect("single family");
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
        let md_subclaims = MultiDegreeSumcheck::<F>::verify_as_subprotocol(
            &mut vt,
            num_vars,
            &[(&md_proof, cfg)],
            cfg,
        )
        .expect("md verify")
        .pop()
        .expect("single family");

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

    /// Exercises the fast-path edge case where `eq_other_table = [1]`
    /// (empty product) and `fold_with_challenge` produces 0-variable
    /// MLEs, after which the rounds-2..num_vars loop runs zero times.
    #[test]
    fn happy_path_num_vars_one() {
        // Two columns, two rows each. Mixed bit patterns so the bit-slice
        // MLEs are not all-zero (exercises every branch of the 4-way
        // bit-fold table in `fold_with_challenge`).
        let c0 = build_col(vec![[true, false, true, false], [false, true, true, true]]);
        let c1 = build_col(vec![[false, false, true, true], [true, true, false, false]]);
        run_roundtrip(&[c0, c1], 1, |_| {}, |_| false, true);
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

    //
    // Round-1 fast path cross-validation.
    //
    // These tests pin down the fast path's bit-identical equivalence with
    // the standard prover (`prove_round` + `fix_variables_with_config`) on
    // the same booleanity inputs. If either of these fails the fast path
    // has diverged from the standard path and the verifier will reject.
    //

    /// Inputs shared between the two cross-validation tests.
    struct FastPathHarness {
        cols: Vec<DenseMultilinearExtension<BinaryPoly<D>>>,
        num_vars: usize,
        r: Vec<F>,
        alpha_powers: Vec<F>,
        zero: F,
        one: F,
    }

    fn cross_validation_inputs() -> FastPathHarness {
        // Three columns, eight rows, mixed bit patterns. nvars = 3 exercises
        // both round-1 (closed form) and rounds 2..num_vars (standard path).
        let c0 = build_col(vec![
            [true, false, true, false],
            [false, true, false, true],
            [true, true, false, false],
            [false, false, true, true],
            [true, true, true, false],
            [false, true, true, false],
            [true, false, false, true],
            [false, false, false, false],
        ]);
        let c1 = build_col(vec![
            [false, false, false, false],
            [true, true, true, true],
            [false, true, true, false],
            [true, false, false, true],
            [false, false, true, false],
            [true, true, false, true],
            [false, true, false, true],
            [true, false, true, false],
        ]);
        let c2 = build_col(vec![
            [true; D],
            [false; D],
            [true, false, false, false],
            [false, false, false, true],
            [true, true, false, false],
            [false, false, true, true],
            [true, false, true, true],
            [false, true, false, false],
        ]);
        let cols = vec![c0, c1, c2];

        let num_vars = 3usize;
        let one = F::one();
        let zero = F::zero();

        // Arbitrary (but deterministic) zerocheck point and batching challenge.
        let r: Vec<F> = (0..num_vars)
            .map(|i| F::from((i as u32) * 13 + 41))
            .collect();
        let alpha = F::from(23u32);
        let alpha_powers = powers(alpha, one.clone(), cols.len() * D);

        FastPathHarness {
            cols,
            num_vars,
            r,
            alpha_powers,
            zero,
            one,
        }
    }

    fn standard_path_mles(
        cols: &[DenseMultilinearExtension<BinaryPoly<D>>],
        r: &[F],
        cfg: &(),
    ) -> Vec<DenseMultilinearExtension<<F as crypto_primitives::Field>::Inner>> {
        let mut mles = Vec::with_capacity(1_usize.saturating_add(cols.len().saturating_mul(D)));
        mles.push(build_eq_x_r_inner(r, cfg).expect("eq_r build"));
        mles.extend(build_witness_bit_slice_mles::<F, D>(cols, cfg));
        mles
    }

    fn make_comb_fn(alpha_powers: Vec<F>, zero: F, one: F) -> CombFn<F> {
        Box::new(move |mle_values: &[F]| {
            let eq_r_val = &mle_values[0];
            let sum = batched_booleanity_sum(&mle_values[1..], &alpha_powers, &zero, &one);
            sum * eq_r_val
        })
    }

    fn make_fast_path(h: &FastPathHarness, cfg: &()) -> BooleanityRound1FastPath<F, D> {
        let eq_other_table: Vec<<F as crypto_primitives::Field>::Inner> = if h.num_vars <= 1 {
            vec![h.one.clone().into_inner()]
        } else {
            build_eq_x_r_inner(&h.r[1..], cfg)
                .expect("eq_other build")
                .evaluations
        };
        BooleanityRound1FastPath::<F, D> {
            binary_cols: h.cols.clone(),
            alpha_powers: h.alpha_powers.clone(),
            eq_other_table,
            r_first_coord: h.r[0].clone(),
            num_vars: h.num_vars,
        }
    }

    /// Round-1 polynomial tail and asserted sum produced by the fast path
    /// match what `SumcheckProverState::prove_round` would emit on the same
    /// inputs.
    #[test]
    fn fast_path_round_1_matches_standard_prove_round() {
        let cfg = &();
        let h = cross_validation_inputs();
        let fast_path = make_fast_path(&h, cfg);

        let fp_out = fast_path.round_1_message(cfg);

        // Standard path: build full-size MLEs, then run a single prove_round
        // (with no verifier message — this is the first round).
        let mles = standard_path_mles(&h.cols, &h.r, cfg);
        let comb_fn = make_comb_fn(h.alpha_powers.clone(), h.zero.clone(), h.one.clone());
        let mut state = SumcheckProverState::<F>::new(mles, h.num_vars, 3);
        let std_msg = state.prove_round(&None, &comb_fn, cfg);

        assert_eq!(
            fp_out.tail_evaluations, std_msg.0.tail_evaluations,
            "round-1 tail evaluations differ between fast and standard paths"
        );
        assert_eq!(
            Some(fp_out.asserted_sum.clone()),
            state.asserted_sum,
            "round-1 asserted sums differ between fast and standard paths"
        );
        assert_eq!(
            fp_out.asserted_sum, h.zero,
            "booleanity is a zerocheck — asserted sum must be 0"
        );
    }

    /// Post-round-1 MLE values from `fold_with_challenge(&r_1)` are
    /// bit-identical to what `fix_variables_with_config(&[r_1], cfg)`
    /// would produce on the standard-path full-size MLEs.
    #[test]
    fn fast_path_fold_with_r1_matches_standard_fix_variables() {
        let cfg = &();
        let h = cross_validation_inputs();
        let r_1 = F::from(97u32);

        // Fast path.
        let fast_path = Box::new(make_fast_path(&h, cfg));
        let fp_mles = fast_path.fold_with_challenge(&r_1, cfg);

        // Standard path: build full-size MLEs, then fold the first variable.
        let mut std_mles = standard_path_mles(&h.cols, &h.r, cfg);
        for mle in &mut std_mles {
            mle.fix_variables_with_config(std::slice::from_ref(&r_1), cfg);
        }

        assert_eq!(
            fp_mles.len(),
            std_mles.len(),
            "fast and standard paths must produce the same number of MLEs"
        );
        for (k, (fp_mle, std_mle)) in fp_mles.iter().zip(std_mles.iter()).enumerate() {
            assert_eq!(
                fp_mle.num_vars, std_mle.num_vars,
                "num_vars mismatch on folded MLE #{k}"
            );
            assert_eq!(
                fp_mle.evaluations, std_mle.evaluations,
                "evaluations mismatch on folded MLE #{k}"
            );
        }
    }
}
