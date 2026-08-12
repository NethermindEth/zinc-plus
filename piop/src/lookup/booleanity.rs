//! Booleanity (binary-polynomial lookup) argument.
//!
//! Proves that every coefficient of every witness binary-polynomial column is
//! a bit $\in \\{0,1\\}$. The committed-witness path is structured as a
//! [`MultiDegreeSumcheckGroup`] of degree 3, batched alongside the existing CPR
//! group with shared randomness, and emits bit-slice claims at the
//! multi-degree sumcheck output point $r^\star$ for the protocol layer's
//! $\alpha'$ bridge into multipoint-eval (see "Soundness bridge" below).
//! Affine virtual booleanity targets use a generic degree-3 group without the
//! `BinaryPoly` round-1 fast path.
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
//!
//! For an unshifted affine virtual, the protocol collapses its bit-slice
//! claims at the same $\alpha'$ and compares the result directly with the
//! declared affine combination of source-column collapses at $r^\star$.
//! Public source collapses are recomputed from the public trace; witness source
//! collapses are the committed-column values bound through the MP/PCS chain
//! above. Shifted affine terms require a separate row-shift opening and are
//! rejected when affine specs are attached to the UAIR signature until that
//! binding is implemented.

use crate::{
    CombFn,
    sumcheck::{
        SumCheckError,
        multi_degree::{MultiDegreeSumcheckGroup, Round1FastPath, Round1Output},
        prover::ProverState as SumcheckProverState,
    },
};
use crypto_primitives::{
    BaseFieldConfig, ProjectPrimitiveIntegersWithConfig, SemiringConfig, SetConfig, Wrapper,
};
use itertools::Itertools;
use std::{marker::PhantomData, slice};
use thiserror::Error;
use zinc_poly::{
    EvaluationError,
    mle::DenseMultilinearExtension,
    univariate::binary::BinaryPoly,
    utils::{ArithErrors, build_eq_x_r, eq_eval},
};
use zinc_transcript::{
    delegate_transcribable,
    traits::{ConstTranscribable, Transcript},
};
use zinc_uair::AffineVirtualSpec;
use zinc_utils::{add, mul, powers};

//
// Structs
//

/// Booleanity sumcheck group constructor / verifier.
pub struct BooleanityChecker<C: SemiringConfig>(PhantomData<C>);

type BooleanityZerocheckSetup<F> = (Vec<F>, Vec<F>, CombFn<F>);

/// Proof produced by the booleanity prover.
///
/// Carries `bit_slice_evals`: per-booleanity-target bit-slice MLE evaluations
/// at the multi-degree sumcheck output point $r^\star$, flat in
/// `(j-major, i-minor)` order. It is absorbed into the transcript by
/// `finalize_prover` / `finalize_verifier`, after which the protocol layer
/// squeezes the $\alpha'$ challenge used to bind the bit-decomposition into
/// the multipoint-eval boundary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BooleanityProof<F> {
    /// Flat list of `\widetilde{v_{j,i}}(r*)`, ordered `(j-major, i-minor)`.
    /// Length is `num_cols * D` for the corresponding Booleanity group.
    pub bit_slice_evals: Vec<F>,
}

impl<F> BooleanityProof<F> {
    /// Maps every field element through `f`, preserving structure — used to
    /// lift elements into wire integers and to project wire integers back
    /// into elements at the (de)serialization boundary.
    pub fn try_map<T, E>(
        &self,
        f: impl FnMut(&F) -> Result<T, E> + Copy,
    ) -> Result<BooleanityProof<T>, E> {
        Ok(BooleanityProof {
            bit_slice_evals: self.bit_slice_evals.iter().map(f).try_collect()?,
        })
    }
}

delegate_transcribable!(BooleanityProof<F> { bit_slice_evals: Vec<F> }
    where F: ConstTranscribable);

/// Ancillary data produced by [`BooleanityChecker::prepare_sumcheck_group`]
/// and consumed by [`BooleanityChecker::finalize_prover`].
pub struct BoolProverAncillary {
    /// Number of binary-poly booleanity target columns (`N`).
    ///
    /// This is the witness binary-poly count for the committed-column group
    /// and the affine virtual count for an affine virtual group.
    pub num_wit_bin_cols: usize,
    /// Bit-width of each binary-poly coefficient cell (`D`).
    pub bit_width: usize,
    /// Number of variables of the trace MLEs.
    pub num_vars: usize,
}

/// Ancillary data produced by [`BooleanityChecker::prepare_verifier`] and
/// consumed by [`BooleanityChecker::finalize_verifier`].
pub struct BoolVerifierAncillary<F> {
    /// Powers of the single batching challenge over the flat
    /// `(j-major, i-minor)` index: `[1, alpha, ..., alpha^{N*D - 1}]`.
    pub alpha_powers: Vec<F>,
    /// The zerocheck point `r` sampled before the multi-degree sumcheck.
    pub zerocheck_point: Vec<F>,
    /// Number of binary-poly booleanity target columns (`N`).
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
pub struct BoolVerifierSubclaim<F> {
    /// Bit-slice MLE evaluations at `r*`, in `(j-major, i-minor)` order.
    pub bit_slice_evals: Vec<F>,
}

//
// Protocol
//

impl<C> BooleanityChecker<C>
where
    C: BaseFieldConfig + ProjectPrimitiveIntegersWithConfig + 'static,
    C::Integer: ConstTranscribable,
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
        field_cfg: &C,
    ) -> Result<(MultiDegreeSumcheckGroup<C>, BoolProverAncillary), BooleanityError<C::Element>>
    {
        let n = trace_bin_poly.len();
        let (r, alpha_powers, comb_fn) =
            booleanity_zerocheck_setup(transcript, n, D, num_vars, field_cfg);

        // 4. Precompute `E_other(b')` = eq(b', r[1..]) for the fast path.
        //    `build_eq_x_r` rejects empty inputs, so handle num_vars == 1 explicitly by
        //    emitting the empty-product table `[1]`.
        let one = field_cfg.one();
        let eq_other_table: Vec<C::Element> = if num_vars <= 1 {
            vec![one]
        } else {
            build_eq_x_r(field_cfg, &r[1..])?.evaluations
        };

        let r_first_coord = r
            .into_iter()
            .next()
            .expect("num_vars >= 1 guarantees a first coordinate");

        let fast_path = BooleanityRound1FastPath::<C, D> {
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

    /// Build a generic booleanity sumcheck group for affine virtual targets.
    ///
    /// Unlike [`Self::prepare_sumcheck_group`], this path does not install the
    /// `BinaryPoly` round-1 fast path: affine residual cells may be non-binary
    /// field values before the booleanity relation is enforced. The trace
    /// slice must contain all binary-polynomial columns in public-then-witness
    /// order so that affine source indices match the UAIR layout.
    pub fn prepare_affine_virtual_sumcheck_group<const D: usize>(
        transcript: &mut impl Transcript,
        all_trace_bin_poly: &[DenseMultilinearExtension<BinaryPoly<D>>],
        affine_virtual_specs: &[AffineVirtualSpec],
        num_vars: usize,
        field_cfg: &C,
    ) -> Result<(MultiDegreeSumcheckGroup<C>, BoolProverAncillary), BooleanityError<C::Element>>
    {
        if affine_virtual_specs.is_empty() {
            return Err(BooleanityError::NoAffineVirtualSpecs);
        }
        if all_trace_bin_poly.is_empty() {
            return Err(BooleanityError::NoBinaryPolyColumns);
        }

        let bit_slice_mles = build_affine_virtual_bit_slice_mles::<C, D>(
            all_trace_bin_poly,
            affine_virtual_specs,
            num_vars,
            field_cfg,
        );
        let num_cols = affine_virtual_specs.len();
        let active_len = num_cols.saturating_mul(D);
        let (r, _, comb_fn) =
            booleanity_zerocheck_setup(transcript, num_cols, D, num_vars, field_cfg);

        debug_assert_eq!(bit_slice_mles.len(), active_len);

        let mut mles = Vec::with_capacity(add!(1, bit_slice_mles.len()));
        mles.push(build_eq_x_r(field_cfg, &r)?);
        mles.extend(bit_slice_mles);

        Ok((
            MultiDegreeSumcheckGroup::new(3, mles, comb_fn),
            BoolProverAncillary {
                num_wit_bin_cols: num_cols,
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
        sumcheck_prover_state: SumcheckProverState<C>,
        ancillary: BoolProverAncillary,
        field_cfg: &C,
    ) -> Result<BooleanityProof<C::Element>, BooleanityError<C::Element>> {
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
        let bit_slice_evals: Vec<C::Element> = mles
            .drain(1..)
            .map(|mle| mle.evaluate(field_cfg, slice::from_ref(&last_sumcheck_challenge)))
            .collect::<Result<Vec<_>, _>>()?;

        debug_assert_eq!(bit_slice_evals.len(), active_len);

        let mut transcription_buf: Vec<u8> = vec![0; <C::Integer as ConstTranscribable>::NUM_BYTES];
        transcript.absorb_field_element_slice(field_cfg, &bit_slice_evals, &mut transcription_buf);

        Ok(BooleanityProof { bit_slice_evals })
    }

    /// Pre-sumcheck half of the booleanity verifier.
    ///
    /// Must run after the CPR `prepare_verifier` and before
    /// `MultiDegreeSumcheck::verify_as_subprotocol` to maintain transcript
    /// ordering.
    pub fn prepare_verifier(
        transcript: &mut impl Transcript,
        claimed_sum: &C::Element,
        num_wit_bin_cols: usize,
        bit_width: usize,
        num_vars: usize,
        field_cfg: &C,
    ) -> Result<BoolVerifierAncillary<C::Element>, BooleanityError<C::Element>> {
        if num_wit_bin_cols == 0 {
            return Err(BooleanityError::NoBinaryPolyColumns);
        }
        if !field_cfg.is_zero(claimed_sum) {
            return Err(BooleanityError::NonZeroClaimedSum {
                got: claimed_sum.clone(),
            });
        }

        // Re-squeeze in the same order as the prover.
        let zerocheck_point: Vec<C::Element> = transcript.get_field_challenges(num_vars, field_cfg);
        let alpha: C::Element = transcript.get_field_challenge(field_cfg);
        let alpha_powers: Vec<C::Element> = powers(
            field_cfg,
            &alpha,
            num_wit_bin_cols.saturating_mul(bit_width),
        );

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
        proof: BooleanityProof<C::Element>,
        shared_point: Vec<C::Element>,
        expected_eval: &C::Element,
        ancillary: BoolVerifierAncillary<C::Element>,
        field_cfg: &C,
    ) -> Result<BoolVerifierSubclaim<C::Element>, BooleanityError<C::Element>> {
        let expected_len = ancillary
            .num_wit_bin_cols
            .saturating_mul(ancillary.bit_width);
        if proof.bit_slice_evals.len() != expected_len {
            return Err(BooleanityError::WrongBitSliceEvalsNumber {
                expected: expected_len,
                got: proof.bit_slice_evals.len(),
            });
        }

        // eq(r*, r) — selector value at the shared sumcheck point.
        let eq_r_val = eq_eval(field_cfg, &shared_point, &ancillary.zerocheck_point)?;

        // Recompute the comb_fn body at r* using the received bit-slice evals.
        let sum =
            batched_booleanity_sum(field_cfg, &proof.bit_slice_evals, &ancillary.alpha_powers);
        let expected_claim_value = field_cfg.mul(&sum, &eq_r_val);

        if expected_claim_value != *expected_eval {
            return Err(BooleanityError::ClaimValueDoesNotMatch {
                expected: expected_claim_value,
                got: expected_eval.clone(),
            });
        }

        let mut transcription_buf: Vec<u8> = vec![0; <C::Integer as ConstTranscribable>::NUM_BYTES];
        transcript.absorb_field_element_slice(
            field_cfg,
            &proof.bit_slice_evals,
            &mut transcription_buf,
        );

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
struct BooleanityRound1FastPath<C: SetConfig, const D: usize> {
    /// Per-column binary trace (cloned to avoid lifetime issues).
    binary_cols: Vec<DenseMultilinearExtension<BinaryPoly<D>>>,
    /// Powers of the batching challenge over the flat `(j-major, i-minor)`
    /// index: `[1, alpha, ..., alpha^{N*D - 1}]`.
    alpha_powers: Vec<C::Element>,
    /// Evaluations of $E_{\text{other}}(b') = \widetilde{eq_{r_{[1..]}}}(b')$
    /// on `{0,1}^{num_vars - 1}`. When `num_vars == 1` this is the single
    /// entry `[1]` (the empty product).
    eq_other_table: Vec<C::Element>,
    /// The first coordinate of the zerocheck point, $r_0$.
    r_first_coord: C::Element,
    /// Number of variables of the sumcheck.
    num_vars: usize,
}

impl<C, const D: usize> Round1FastPath<C> for BooleanityRound1FastPath<C, D>
where
    C: SemiringConfig,
{
    fn round_1_message(&self, config: &C) -> Round1Output<C::Element> {
        let zero = config.zero();
        let one = config.one();
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
                        config.add_assign(&mut s_b, &self.alpha_powers[k]);
                    }
                }
            }
            let term = config.mul(&s_b, &self.eq_other_table[b_prime]);
            config.add_assign(&mut t1, &term);
        }

        // Closed-form tail evaluations (with X(X-1) = 0 at X=1):
        //   p_1(1) = 0
        //   p_1(2) = e_0(2) * 2*1*T_1 = 2 * (3*r_0 - 1) * T_1
        //   p_1(3) = e_0(3) * 3*2*T_1 = 6 * (5*r_0 - 2) * T_1
        let two = config.add(&one, &one);
        let three = config.add(&two, &one);
        let five = config.add(&three, &two);
        let six = config.mul(&three, &two);

        let three_r0_minus_one = config.sub(&config.mul(&three, &self.r_first_coord), &one);
        let five_r0_minus_two = config.sub(&config.mul(&five, &self.r_first_coord), &two);

        let p1_at_1 = zero.clone();
        let p1_at_2 = config.mul(&config.mul(&two, &three_r0_minus_one), &t1);
        let p1_at_3 = config.mul(&config.mul(&six, &five_r0_minus_two), &t1);

        Round1Output {
            // Booleanity is a zerocheck; the asserted sum p_1(0)+p_1(1) is 0.
            asserted_sum: zero,
            tail_evaluations: vec![p1_at_1, p1_at_2, p1_at_3],
        }
    }

    fn fold_with_challenge(
        self: Box<Self>,
        r_1: &C::Element,
        config: &C,
    ) -> Vec<DenseMultilinearExtension<C::Element>> {
        // Edge case: when the sumcheck has only one round, the standard prover never
        // performs the folding.
        //
        // The post-round-1 state expected by `finalize_prover` is the un-folded
        // 1-variable MLE bundle (it then evaluates each MLE at `r_1`). Return that
        // directly without applying the fold.
        if self.num_vars <= 1 {
            let r = [self.r_first_coord.clone()];
            let eq_r =
                build_eq_x_r(config, &r).expect("num_vars == 1 implies a single-coordinate r");
            let mut out: Vec<DenseMultilinearExtension<C::Element>> =
                Vec::with_capacity(self.binary_cols.len().saturating_mul(D).saturating_add(1));
            out.push(eq_r);
            out.extend(build_witness_bit_slice_mles::<C, D>(
                &self.binary_cols,
                config,
            ));
            return out;
        }

        let n_cols = self.binary_cols.len();
        let total_bit_slices = n_cols.saturating_mul(D);
        let new_num_vars = self.num_vars.saturating_sub(1);
        let half_n: usize = 1_usize << new_num_vars;

        let one = config.one();

        // 4-way table for bit-slice fold: index = (A as usize) << 1 | (B as usize).
        //   (0,0) -> 0
        //   (0,1) -> r_1
        //   (1,0) -> 1 - r_1
        //   (1,1) -> 1
        let bit_fold_table: [C::Element; 4] = [
            config.zero(),
            r_1.clone(),
            config.sub(&one, r_1),
            one.clone(),
        ];

        // eq_r fold: e_0(r_1) * E_other(b').
        // e_0(r_1) = (1 - r_1)(1 - r_0) + r_1 * r_0.
        let one_minus_r0 = config.sub(&one, &self.r_first_coord);
        let one_minus_r1 = config.sub(&one, r_1);
        let e_0_r1: C::Element = config.add(
            &config.mul(&one_minus_r1, &one_minus_r0),
            &config.mul(r_1, &self.r_first_coord),
        );

        let folded_eq_r_evals = (0..half_n)
            .map(|b_prime| config.mul(&e_0_r1, &self.eq_other_table[b_prime]))
            .collect_vec();

        // Length = N*D + 1
        let mut out: Vec<DenseMultilinearExtension<C::Element>> =
            Vec::with_capacity(total_bit_slices.saturating_add(1));
        out.push(DenseMultilinearExtension {
            num_vars: new_num_vars,
            evaluations: folded_eq_r_evals,
        });

        // Bit-slice fold via 4-way table lookup; emit MLEs in
        // (j-major, i-minor) order.
        for col in &self.binary_cols {
            let mut col_evals: Vec<Vec<C::Element>> =
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
pub enum BooleanityError<F: std::fmt::Debug> {
    #[error("no binary polynomial columns provided to booleanity checker")]
    NoBinaryPolyColumns,
    #[error("no affine virtual specs provided to booleanity checker")]
    NoAffineVirtualSpecs,
    #[error("sumcheck error: {0}")]
    SumcheckError(#[from] SumCheckError<F>),
    #[error("expected booleanity claimed sum is non-zero: got {got:?}")]
    NonZeroClaimedSum { got: F },
    #[error("wrong number of bit-slice evaluations: expected {expected}, got {got}")]
    WrongBitSliceEvalsNumber { expected: usize, got: usize },
    #[error("booleanity claim value does not match: expected {expected:?}, got {got:?}")]
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
fn batched_booleanity_sum<C: SemiringConfig>(
    cfg: &C,
    bit_slice_values: &[C::Element],
    alpha_powers: &[C::Element],
) -> C::Element {
    debug_assert_eq!(bit_slice_values.len(), alpha_powers.len());

    let one = cfg.one();
    let mut sum = cfg.zero();
    for (v, alpha_k) in bit_slice_values.iter().zip(alpha_powers.iter()) {
        let booleanity = cfg.mul(&cfg.sub(v, &one), v);
        cfg.add_assign(&mut sum, &cfg.mul(&booleanity, alpha_k));
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
fn build_witness_bit_slice_mles<C, const D: usize>(
    trace_bin_poly: &[DenseMultilinearExtension<BinaryPoly<D>>],
    field_cfg: &C,
) -> Vec<DenseMultilinearExtension<C::Element>>
where
    C: SemiringConfig,
{
    let zero = field_cfg.zero();
    let one = field_cfg.one();

    let mut out = Vec::with_capacity(trace_bin_poly.len().saturating_mul(D));
    for col in trace_bin_poly {
        let num_vars = col.num_vars;
        let n_rows = col.evaluations.len();
        for i in 0..D {
            let mut evals: Vec<C::Element> = Vec::with_capacity(n_rows);
            for entry in col.iter() {
                let bit = entry
                    .iter()
                    .nth(i)
                    .expect("BinaryPoly<D> has D coefficients")
                    .into_inner();
                evals.push(if bit { one.clone() } else { zero.clone() });
            }
            out.push(DenseMultilinearExtension {
                num_vars,
                evaluations: evals,
            });
        }
    }
    out
}

fn booleanity_zerocheck_setup<C>(
    transcript: &mut impl Transcript,
    num_cols: usize,
    bit_width: usize,
    num_vars: usize,
    field_cfg: &C,
) -> BooleanityZerocheckSetup<C::Element>
where
    C: BaseFieldConfig + ProjectPrimitiveIntegersWithConfig + 'static,
    C::Integer: ConstTranscribable,
{
    // Order of challenge squeezing must match between prover and verifier.

    // 1. Zerocheck point r.
    let r: Vec<C::Element> = transcript.get_field_challenges(num_vars, field_cfg);

    // 2. Single batching challenge alpha over the flat (j-major, i-minor) index.
    //    Powers vector has length N*D.
    let alpha: C::Element = transcript.get_field_challenge(field_cfg);
    let alpha_powers: Vec<C::Element> =
        powers(field_cfg, &alpha, num_cols.saturating_mul(bit_width));

    // 3. comb_fn (degree 3 in the variables):
    //
    //       eq_r(b) * sum_k alpha^k * v_k(b) * (v_k(b) - 1)
    //
    //    The committed-column group installs a fast path that emits the
    //    round-1 message and folded MLEs directly, so this closure is not
    //    invoked in its first round. Generic affine virtual groups use it in
    //    every round.
    let comb_fn: CombFn<C::Element> = {
        let alpha_powers = alpha_powers.clone();
        let field_cfg = field_cfg.clone();
        Box::new(move |mle_values: &[C::Element]| {
            let eq_r_val = &mle_values[0];
            let sum = batched_booleanity_sum(&field_cfg, &mle_values[1..], &alpha_powers);
            field_cfg.mul(&sum, eq_r_val)
        })
    };

    (r, alpha_powers, comb_fn)
}

/// Build per-bit-slice MLEs for affine virtual booleanity targets.
///
/// Returns `N * D` MLEs in `(spec-major, i-minor)` order:
/// `$[v_{0,0}, v_{0,1}, ..., v_{0,D-1}, v_{1,0}, ...]$`. The `j`-th
/// spec, `i`-th bit MLE evaluates at row `b` to the `i`-th coefficient of
///
/// ```text
/// ones_coefficient * 1_D + sum_t coefficient_t * source_t[b + row_shift_t]
/// ```
///
/// with out-of-range shifted rows contributing zero, matching `ShiftSpec`.
/// `all_trace_bin_poly` must contain every binary-polynomial trace column in
/// the UAIR's flat total-column order: public columns first, then witness
/// columns.
pub fn build_affine_virtual_bit_slice_mles<C, const D: usize>(
    all_trace_bin_poly: &[DenseMultilinearExtension<BinaryPoly<D>>],
    affine_virtual_specs: &[AffineVirtualSpec],
    num_vars: usize,
    field_cfg: &C,
) -> Vec<DenseMultilinearExtension<C::Element>>
where
    C: SemiringConfig + ProjectPrimitiveIntegersWithConfig,
{
    if affine_virtual_specs.is_empty() {
        return Vec::new();
    }

    let n_rows = 1usize << num_vars;
    for col in all_trace_bin_poly {
        assert_eq!(
            col.num_vars, num_vars,
            "all affine virtual source columns must have the same num_vars",
        );
        assert_eq!(
            col.evaluations.len(),
            n_rows,
            "source column length must match num_vars",
        );
    }

    let mut out = Vec::with_capacity(affine_virtual_specs.len().saturating_mul(D));
    for spec in affine_virtual_specs {
        let ones = field_cfg.project(&spec.ones_coefficient());
        let terms: Vec<_> = spec
            .terms()
            .iter()
            .map(|term| {
                assert!(
                    term.source_col() < all_trace_bin_poly.len(),
                    "AffineVirtualTerm source_col {} out of range (binary columns = {})",
                    term.source_col(),
                    all_trace_bin_poly.len(),
                );
                (
                    term.source_col(),
                    field_cfg.project(&term.coefficient()),
                    term.row_shift(),
                )
            })
            .collect();

        for bit_idx in 0..D {
            let mut evals: Vec<C::Element> = Vec::with_capacity(n_rows);
            for row_idx in 0..n_rows {
                let mut residual = ones.clone();
                for (source_col, coeff, row_shift) in &terms {
                    let shifted_row = add!(row_idx, *row_shift);
                    if shifted_row >= n_rows {
                        continue;
                    }
                    let bit = all_trace_bin_poly[*source_col].evaluations[shifted_row]
                        .iter()
                        .nth(bit_idx)
                        .expect("BinaryPoly<D> has D coefficients")
                        .into_inner();
                    if bit {
                        field_cfg.add_assign(&mut residual, coeff);
                    }
                }
                evals.push(residual);
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
    use crypto_primitives::{
        FixedConfig, ProjectElementWithConfig, crypto_bigint_const_monty::ConstMontyField,
    };
    use num_traits::{One, Zero};
    use zinc_poly::mle::MultilinearExtension;
    use zinc_transcript::Blake3Transcript;
    use zinc_uair::{AffineVirtualSpec, AffineVirtualTerm};
    use zinc_utils::powers;

    const_monty_params!(TestParams, U128, "00000000b933426489189cb5b47d567f");
    type F = ConstMontyField<TestParams, { U128::LIMBS }>;
    type FC = FixedConfig<F>;

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
        t.absorb_bytes(b"booleanity test");
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
        let cfg = &FC::default();

        // Prover side.
        let mut pt = make_transcript();
        let (group, anc) =
            BooleanityChecker::<FC>::prepare_sumcheck_group::<D>(&mut pt, cols, num_vars, cfg)
                .expect("prepare_sumcheck_group failed");
        let mut sumcheck_outputs = MultiDegreeSumcheck::<FC>::prove_as_subprotocol(
            &mut pt,
            vec![(vec![group], cfg)],
            num_vars,
            cfg,
        );
        let (md_proof, states) = sumcheck_outputs.pop().expect("single family");
        let state = states.into_iter().next().unwrap();
        let mut proof = BooleanityChecker::<FC>::finalize_prover(&mut pt, state, anc, cfg)
            .expect("finalize prover");
        tamper_proof(&mut proof);

        // Verifier side.
        let mut vt = make_transcript();
        let anc_v = BooleanityChecker::<FC>::prepare_verifier(
            &mut vt,
            &md_proof.claimed_sums()[0],
            cols.len(),
            D,
            num_vars,
            cfg,
        )
        .expect("prepare_verifier (claimed sum should be zero for honest input)");
        let md_subclaims = MultiDegreeSumcheck::<FC>::verify_as_subprotocol(
            &mut vt,
            num_vars,
            &[(&md_proof, cfg)],
            cfg,
        )
        .expect("md verify")
        .pop()
        .expect("single family");

        let res = BooleanityChecker::<FC>::finalize_verifier(
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
        let mles = build_witness_bit_slice_mles::<FC, D>(&cols, &FC::default());
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

    #[test]
    fn affine_virtual_bit_slice_mles_materialize_residual_coefficients() {
        let cfg = &FC::default();
        let public = build_col(vec![[true; D], [true; D], [true; D], [true; D]]);
        let a = build_col(vec![
            [true, false, true, false],
            [false, true, false, true],
            [true, true, false, false],
            [false, false, true, true],
        ]);
        let b = build_col(vec![
            [false, true, true, false],
            [true, false, false, false],
            [false, true, true, true],
            [true, true, false, false],
        ]);
        let m = build_col(vec![
            [false, false, true, false],
            [false, false, false, false],
            [false, true, false, false],
            [false, false, false, false],
        ]);
        let specs = vec![AffineVirtualSpec::new(vec![
            AffineVirtualTerm::new(1, 1),
            AffineVirtualTerm::new(2, 1),
            AffineVirtualTerm::new(3, -2),
        ])];

        // Index 0 is the public binary-polynomial prefix. Affine source
        // indices use the full public-then-witness trace layout.
        let mles = build_affine_virtual_bit_slice_mles::<FC, D>(&[public, a, b, m], &specs, 2, cfg);

        assert_eq!(mles.len(), D);
        let expected_by_bit: [[i32; 4]; D] =
            [[1, 1, 1, 1], [1, 1, 0, 1], [0, 0, 1, 1], [0, 1, 1, 1]];
        for (bit_idx, expected_rows) in expected_by_bit.iter().enumerate() {
            let expected: Vec<_> = expected_rows
                .iter()
                .map(|value| cfg.project(value))
                .collect();
            assert_eq!(
                mles[bit_idx].evaluations, expected,
                "unexpected affine virtual residual at bit {bit_idx}"
            );
        }
    }

    #[test]
    fn affine_virtual_bit_slice_mles_apply_forward_shift_and_ones_offset() {
        let cfg = &FC::default();
        let a = build_col(vec![
            [true, false, false, false],
            [false, true, false, false],
            [true, true, false, false],
            [false, false, false, false],
        ]);
        let specs = vec![AffineVirtualSpec::with_ones_coefficient(
            vec![AffineVirtualTerm::new_shifted(0, -1, 1)],
            1,
        )];

        let mles = build_affine_virtual_bit_slice_mles::<FC, D>(&[a], &specs, 2, cfg);

        assert_eq!(mles.len(), D);
        let expected_by_bit: [[i32; 4]; D] =
            [[1, 0, 1, 1], [0, 0, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]];
        for (bit_idx, expected_rows) in expected_by_bit.iter().enumerate() {
            let expected: Vec<_> = expected_rows
                .iter()
                .map(|value| cfg.project(value))
                .collect();
            assert_eq!(
                mles[bit_idx].evaluations, expected,
                "unexpected shifted affine virtual residual at bit {bit_idx}"
            );
        }
    }

    fn run_affine_virtual_roundtrip(
        all_trace_bin_poly: &[DenseMultilinearExtension<BinaryPoly<D>>],
        specs: &[AffineVirtualSpec],
        num_vars: usize,
        tamper_proof: impl FnOnce(&mut BooleanityProof<F>),
    ) -> Result<(), BooleanityError<F>> {
        let cfg = &FC::default();

        let mut pt = make_transcript();
        let (group, anc) = BooleanityChecker::<FC>::prepare_affine_virtual_sumcheck_group::<D>(
            &mut pt,
            all_trace_bin_poly,
            specs,
            num_vars,
            cfg,
        )
        .expect("prepare_affine_virtual_sumcheck_group failed");
        let mut sumcheck_outputs = MultiDegreeSumcheck::<FC>::prove_as_subprotocol(
            &mut pt,
            vec![(vec![group], cfg)],
            num_vars,
            cfg,
        );
        let (md_proof, states) = sumcheck_outputs.pop().expect("single family");
        let state = states.into_iter().next().unwrap();
        let mut proof = BooleanityChecker::<FC>::finalize_prover(&mut pt, state, anc, cfg)
            .expect("finalize prover");
        tamper_proof(&mut proof);

        let mut vt = make_transcript();
        let verifier_anc = BooleanityChecker::<FC>::prepare_verifier(
            &mut vt,
            &md_proof.claimed_sums()[0],
            specs.len(),
            D,
            num_vars,
            cfg,
        )?;
        let md_subclaims = MultiDegreeSumcheck::<FC>::verify_as_subprotocol(
            &mut vt,
            num_vars,
            &[(&md_proof, cfg)],
            cfg,
        )
        .expect("md verify")
        .pop()
        .expect("single family");

        BooleanityChecker::<FC>::finalize_verifier(
            &mut vt,
            proof,
            md_subclaims.point().to_vec(),
            &md_subclaims.expected_evaluations()[0],
            verifier_anc,
            cfg,
        )
        .map(|_| ())
    }

    #[test]
    fn affine_virtual_group_accepts_boolean_residual_and_rejects_non_boolean_residual() {
        let public = build_col(vec![[true; D], [true; D], [true; D], [true; D]]);
        let a = build_col(vec![
            [true, false, true, false],
            [false, true, false, true],
            [true, true, false, false],
            [false, false, true, true],
        ]);
        let b = build_col(vec![
            [false, true, true, false],
            [true, false, false, false],
            [false, true, true, true],
            [true, true, false, false],
        ]);
        let m = build_col(vec![
            [false, false, true, false],
            [false, false, false, false],
            [false, true, false, false],
            [false, false, false, false],
        ]);
        let specs = vec![AffineVirtualSpec::new(vec![
            AffineVirtualTerm::new(1, 1),
            AffineVirtualTerm::new(2, 1),
            AffineVirtualTerm::new(3, -2),
        ])];

        run_affine_virtual_roundtrip(
            &[public.clone(), a.clone(), b.clone(), m],
            &specs,
            2,
            |_| {},
        )
        .expect("boolean affine residual should verify");

        let bad_m = build_col(vec![
            [true, false, true, false],
            [false, false, false, false],
            [false, true, false, false],
            [false, false, false, false],
        ]);
        let err = run_affine_virtual_roundtrip(&[public, a, b, bad_m], &specs, 2, |_| {})
            .expect_err("non-boolean affine residual should be rejected");
        assert!(matches!(err, BooleanityError::NonZeroClaimedSum { .. }));
    }

    #[test]
    fn affine_virtual_group_rejects_tampered_bit_slice_eval() {
        let zero = zero_col_4rows();
        let specs = vec![AffineVirtualSpec::new(vec![AffineVirtualTerm::new(0, 1)])];

        let err = run_affine_virtual_roundtrip(&[zero], &specs, 2, |proof| {
            proof.bit_slice_evals[0] += F::from(7u32);
        })
        .expect_err("tampered affine bit-slice evaluation should be rejected");
        assert!(matches!(
            err,
            BooleanityError::ClaimValueDoesNotMatch { .. }
        ));
    }

    #[test]
    fn affine_virtual_prepare_rejects_missing_inputs_with_precise_errors() {
        let cfg = &FC::default();
        let mut transcript = make_transcript();
        let col = zero_col_4rows();
        let no_specs = BooleanityChecker::<FC>::prepare_affine_virtual_sumcheck_group::<D>(
            &mut transcript,
            &[col],
            &[],
            2,
            cfg,
        );
        assert!(matches!(
            no_specs,
            Err(BooleanityError::NoAffineVirtualSpecs)
        ));

        let mut transcript = make_transcript();
        let specs = vec![AffineVirtualSpec::new(vec![AffineVirtualTerm::new(0, 1)])];
        let no_columns = BooleanityChecker::<FC>::prepare_affine_virtual_sumcheck_group::<D>(
            &mut transcript,
            &[],
            &specs,
            2,
            cfg,
        );
        assert!(matches!(
            no_columns,
            Err(BooleanityError::NoBinaryPolyColumns)
        ));
    }

    //
    // Round-1 fast path cross-validation.
    //
    // These tests pin down the fast path's bit-identical equivalence with
    // the standard prover (`prove_round` + `fix_variables`) on
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
        let alpha_powers = powers(&FC::default(), &alpha, cols.len() * D);

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
        cfg: &FC,
    ) -> Vec<DenseMultilinearExtension<F>> {
        let mut mles = Vec::with_capacity(1_usize.saturating_add(cols.len().saturating_mul(D)));
        mles.push(build_eq_x_r(cfg, r).expect("eq_r build"));
        mles.extend(build_witness_bit_slice_mles::<FC, D>(cols, cfg));
        mles
    }

    fn make_comb_fn(alpha_powers: Vec<F>) -> CombFn<F> {
        Box::new(move |mle_values: &[F]| {
            let cfg = FC::default();
            let eq_r_val = &mle_values[0];
            let sum = batched_booleanity_sum(&cfg, &mle_values[1..], &alpha_powers);
            sum * eq_r_val
        })
    }

    fn make_fast_path(h: &FastPathHarness, cfg: &FC) -> BooleanityRound1FastPath<FC, D> {
        let eq_other_table: Vec<F> = if h.num_vars <= 1 {
            vec![h.one.clone()]
        } else {
            build_eq_x_r(cfg, &h.r[1..])
                .expect("eq_other build")
                .evaluations
        };
        BooleanityRound1FastPath::<FC, D> {
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
        let cfg = &FC::default();
        let h = cross_validation_inputs();
        let fast_path = make_fast_path(&h, cfg);

        let fp_out = fast_path.round_1_message(cfg);

        // Standard path: build full-size MLEs, then run a single prove_round
        // (with no verifier message — this is the first round).
        let mles = standard_path_mles(&h.cols, &h.r, cfg);
        let comb_fn = make_comb_fn(h.alpha_powers.clone());
        let mut state = SumcheckProverState::<FC>::new(mles, h.num_vars, 3);
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
    /// bit-identical to what `fix_variables(cfg, &[r_1])`
    /// would produce on the standard-path full-size MLEs.
    #[test]
    fn fast_path_fold_with_r1_matches_standard_fix_variables() {
        let cfg = &FC::default();
        let h = cross_validation_inputs();
        let r_1 = F::from(97u32);

        // Fast path.
        let fast_path = Box::new(make_fast_path(&h, cfg));
        let fp_mles = fast_path.fold_with_challenge(&r_1, cfg);

        // Standard path: build full-size MLEs, then fold the first variable.
        let mut std_mles = standard_path_mles(&h.cols, &h.r, cfg);
        for mle in &mut std_mles {
            mle.fix_variables(cfg, std::slice::from_ref(&r_1));
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
