//! Multi-point evaluation subprotocol.
//!
//! Reduces two sets of MLE evaluation claims at a shared point r' - the
//! "up" evaluations `v_j(r')` and the "down" (shifted) evaluations
//! `v_j^{down}(r')` - to a single set of standard MLE evaluation claims
//! `v_j(r_0)` at a new random point `r_0` via one sumcheck.
//!
//! The trace column MLEs are precombined into a single MLE
//! `precombined(b) = \sum_j \gamma_j * v_j(b)` before entering the sumcheck, so
//! the prover works with only 3 MLEs (`eq`, `next`, `precombined`) regardless
//! of the number of columns. The sumcheck proves:
//! ```text
//! \sum_b [eq(b, r') * \sum_j \gamma_j * v_j(b)
//!         + \sum_k \alpha_k * next_{c_k}(r', b) * v_{src_k}(b)]
//!   = \sum_j \gamma_j * up_eval_j + \sum_k \alpha_k * down_eval_k
//! ```
//!
//! where `\alpha_k` batch the per-shift evaluation kernels and `\gamma_j`
//! batch across columns. After the sumcheck reduces to point `r_0`, the
//! verifier calls [`MultipointEval::verify_subclaim`] with the `open_evals`
//! (the F_q-valued MLE evaluations at `r_0`, typically derived from
//! polynomial-valued `lifted_evals` via `\psi_a`) to check the final
//! consistency equation.
//!
//! This corresponds to the T=2 case of Pi_{BMLE} in the paper. Following
//! the paper, the prover sends only the polynomial-valued lifted evaluations
//! (alpha'_j in F_q[X]); the scalar open_evals are derived by the verifier
//! via \psi_a rather than being sent as a separate proof element.

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::{
    CombFn,
    shift_predicate::eval_shift_predicate,
    sumcheck::{
        SumCheckError,
        multi_degree::{
            MultiDegreeSubClaims, MultiDegreeSumcheck, MultiDegreeSumcheckGroup,
            MultiDegreeSumcheckProof,
        },
    },
};
use crypto_primitives::{FromPrimitiveWithConfig, PrimeField};
use num_traits::Zero;
use std::marker::PhantomData;
use thiserror::Error;
use zinc_poly::{
    mle::DenseMultilinearExtension,
    utils::{ArithErrors, build_eq_x_r_inner, build_next_c_r_mle},
};
use zinc_transcript::{
    delegate_transcribable,
    traits::{ConstTranscribable, Transcript},
};
use zinc_uair::ShiftSpec;
use zinc_utils::{cfg_into_iter, inner_transparent_field::InnerTransparentField};

//
// Data structures
//

/// Proof for the multi-point evaluation protocol.
///
/// Wraps the inner [`MultiDegreeSumcheckProof`] driven by
/// [`MultiDegreeSumcheck`]: each family contributes a single degree-2
/// group, all families sharing one $r_0$ per round. The MLE evaluations
/// at $r_0$ are provided externally via `lifted_evals` (in $F_q[X]$),
/// from which the verifier derives the scalar `open_evals` via $\psi_a$.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Proof<F: PrimeField> {
    /// The inner multi-degree sumcheck proof. Single-family shape: one
    /// degree-2 group.
    pub sumcheck_proof: MultiDegreeSumcheckProof<F>,
}

delegate_transcribable!(Proof<F> { sumcheck_proof: MultiDegreeSumcheckProof<F> }
    where F: PrimeField, F::Integer: ConstTranscribable);

/// Per-family inputs to the lockstep multi-point evaluation protocol,
/// those that aren't shared between families.
///
/// All families share the same `shifts` (UAIR-static), the same column
/// counts, and the same `num_vars`. They differ only in (a) the field
/// config they operate over and (b) the per-family field-projected
/// `trace_mles`, `eval_point`, `up_evals`, and `down_evals`.
pub struct MultipointEvalFamilyInputs<'a, F: PrimeField> {
    /// Field configuration for this family.
    pub field_cfg: &'a F::Config,
    /// Trace MLEs for this family (projected into this family's field).
    pub trace_mles: &'a [DenseMultilinearExtension<F::Inner>],
    /// Evaluation point $r^\star$ for this family.
    pub eval_point: &'a [F],
    /// `up_eval_j = v_j(r*)` for every column $j$, in this family's
    /// field.
    pub up_evals: &'a [F],
    /// `down_eval_k = v_{src_k}^{<<c_k}(r*)` for every shift $k$, in
    /// this family's field.
    pub down_evals: &'a [F],
}

/// Prover state after the multi-point evaluation protocol for one constraint
/// family.
#[derive(Clone, Debug)]
pub struct ProverState<F: PrimeField> {
    /// The combined evaluation point `r_0` produced by the sumcheck
    /// (lifted into this family's field — the underlying integer is shared
    /// across all families).
    pub eval_point: Vec<F>,
}

/// Verifier subclaim after the multi-point evaluation sumcheck for one
/// constraint family.
///
/// Carries the shared evaluation point $r_0$ (lifted into this family's
/// field), the expected MLE-combination evaluation handed back by the
/// inner multi-degree sumcheck, plus the intermediate values needed to
/// finalize the check via [`MultipointEval::verify_subclaim`] once the
/// caller has assembled the `open_evals`.
#[derive(Clone, Debug)]
pub struct Subclaim<F: PrimeField> {
    /// Shared sumcheck output point $r_0$ (in this family's field).
    pub r0: Vec<F>,
    /// Expected evaluation of the combined polynomial at $r_0$ handed back
    /// by the inner multi-degree sumcheck — `verify_subclaim` checks this
    /// against the batched `open_evals`.
    pub expected_evaluation: F,
    /// Column batching coefficients $\gamma_j$ sampled during the protocol
    /// (lifted into this family's field).
    pub gammas: Vec<F>,
    /// Per-shift batching coefficients $\alpha_k$ sampled during the
    /// protocol (lifted into this family's field).
    pub alphas: Vec<F>,
    /// `eq(r_0, r^\star)` — the equality selector at the sumcheck output
    /// point.
    pub eq_at_r0: F,
    /// Per-shift selector values at $r_0$:
    /// `shifts_at_r0[k] = next_{c_k}(r^\star, r_0)`.
    pub shifts_at_r0: Vec<F>,
}

//
// Protocol
//

pub struct MultipointEval<F>(PhantomData<F>);

impl<F> MultipointEval<F>
where
    F: InnerTransparentField + FromPrimitiveWithConfig + Send + Sync + 'static,
    F::Integer: ConstTranscribable + Zero + Default + Send + Sync,
{
    /// Multi-point evaluation protocol prover (lockstep over families).
    ///
    /// Drives one or more **families** of multi-point evaluation in lockstep,
    /// each family operating over its own field config. All families share
    /// the same UAIR-static [`ShiftSpec`]s, the same number of columns, and
    /// the same `num_vars`; they differ only in their per-family projected
    /// MLEs/scalars.
    ///
    /// All protocol-level challenges $(\alpha_k, \gamma_j)$ are sampled once
    /// as integers in $[0, q^*)$ via `q_star_cfg` and lifted into each
    /// family's field via [`F::from_with_cfg`]. The inner sumcheck is
    /// driven by [`MultiDegreeSumcheck`] (one degree-2 group per family),
    /// so each round's challenge is likewise shared across families and
    /// lifted per-family.
    ///
    /// The single-family case (`families.len() == 1`) is the natural
    /// degenerate form; pass `q_star_cfg = families[0].field_cfg`.
    ///
    /// Per family, proves
    ///
    /// $$\sum_b \Bigl[\, \mathrm{eq}(b, r^\star) \sum_j \gamma_j v_j(b) +
    /// \sum_k \alpha_k \cdot \mathrm{next}_{c_k}(r^\star, b) \cdot
    /// v_{\mathrm{src}_k}(b) \Bigr] = \sum_j \gamma_j \cdot
    /// \mathrm{up\\_eval}_j + \sum_k \alpha_k \cdot \mathrm{down\\_eval}_k.$$
    ///
    /// Returns one `(Proof, ProverState)` per family in family order. The
    /// caller is responsible for computing and sending `lifted_evals` at
    /// the shared $r_0$ for each family.
    ///
    /// # Panics
    ///
    /// * If `families` is empty.
    /// * If families disagree on `num_vars` or column counts.
    #[allow(
        clippy::arithmetic_side_effects,
        clippy::too_many_lines,
        clippy::type_complexity
    )]
    pub fn prove_as_subprotocol(
        transcript: &mut impl Transcript,
        families: Vec<MultipointEvalFamilyInputs<'_, F>>,
        shifts: &[ShiftSpec],
        q_star_cfg: &F::Config,
    ) -> Result<Vec<(Proof<F>, ProverState<F>)>, MultipointEvalError<F>> {
        assert!(!families.is_empty(), "need at least one family");

        let num_families = families.len();
        let num_cols = families[0].trace_mles.len();
        let num_vars = families[0].eval_point.len();
        let num_down_cols = shifts.len();

        for b in &families {
            assert_eq!(
                b.trace_mles.len(),
                num_cols,
                "all families must have the same number of columns",
            );
            assert_eq!(
                b.eval_point.len(),
                num_vars,
                "all families must have the same num_vars",
            );
            assert_eq!(
                b.up_evals.len(),
                num_cols,
                "up_evals length must match trace_mles length",
            );
            assert_eq!(
                b.down_evals.len(),
                num_down_cols,
                "down_evals length must match shifts length",
            );
        }

        // Step 1: Sample shared batching coefficients $\alpha_k$ and
        // $\gamma_j$ as integers in $[0, q^*)$, then lift into each
        // family's field.
        let shared_alpha_ints: Vec<F::Integer> = (0..num_down_cols)
            .map(|_| {
                transcript
                    .get_field_challenge::<F>(q_star_cfg)
                    .lift_to_integer()
            })
            .collect();
        let shared_gamma_ints: Vec<F::Integer> = (0..num_cols)
            .map(|_| {
                transcript
                    .get_field_challenge::<F>(q_star_cfg)
                    .lift_to_integer()
            })
            .collect();

        let per_family_alphas: Vec<Vec<F>> = families
            .iter()
            .map(|b| {
                shared_alpha_ints
                    .iter()
                    .map(|v| F::from_with_cfg(v.clone(), b.field_cfg))
                    .collect()
            })
            .collect();
        let per_family_gammas: Vec<Vec<F>> = families
            .iter()
            .map(|b| {
                shared_gamma_ints
                    .iter()
                    .map(|v| F::from_with_cfg(v.clone(), b.field_cfg))
                    .collect()
            })
            .collect();

        // Step 2: Build per-family sumcheck groups.
        //
        // Each family contributes one degree-2 group with MLE layout
        // `[eq_r, next_mles[..], precombined, down_cols[..]]` and
        // comb_fn `eq * precombined + \sum_k \alpha_k * next_k * down_k`.
        let mut family_groups: Vec<(Vec<MultiDegreeSumcheckGroup<F>>, &F::Config)> =
            Vec::with_capacity(num_families);

        // Sanity-check claimed sums in debug builds.
        let mut debug_expected_sums: Vec<F> = Vec::with_capacity(num_families);

        for (b_idx, family) in families.iter().enumerate() {
            let cfg = family.field_cfg;
            let alphas_b = per_family_alphas[b_idx].clone();
            let gammas_b = &per_family_gammas[b_idx];
            let zero = F::zero_with_cfg(cfg);
            let zero_inner = zero.inner();

            // Build the two selector MLEs:
            //   eq_r(b)   = eq(b, r')
            //   next_c_r_mle(b) = next_c_mle(r', b)
            let eq_r = build_eq_x_r_inner(family.eval_point, cfg)?;
            let (next_mles, down_cols): (Vec<_>, Vec<_>) = shifts
                .iter()
                .map(|spec| {
                    let next = build_next_c_r_mle(family.eval_point, spec.shift_amount(), cfg)?;
                    let col = family.trace_mles[spec.source_col()].clone();
                    Ok((next, col))
                })
                .collect::<Result<Vec<_>, ArithErrors>>()?
                .into_iter()
                .unzip();

            // Precombine up cols with gammas: precombined[i] = \sum_j gamma_j trace[j][i].
            let precombined = {
                let evaluations: Vec<_> = cfg_into_iter!(0..1usize << num_vars)
                    .map(|i| {
                        gammas_b
                            .iter()
                            .enumerate()
                            .fold(zero.clone(), |acc, (j, gamma)| {
                                let eval_f = F::new_unchecked_with_cfg(
                                    family.trace_mles[j].evaluations[i].clone(),
                                    cfg,
                                );
                                acc + eval_f * gamma
                            })
                            .into_inner()
                    })
                    .collect();
                DenseMultilinearExtension::from_evaluations_vec(
                    num_vars,
                    evaluations,
                    zero_inner.clone(),
                )
            };

            // Pack MLEs: [eq_r, next_mles[..], precombined, down_cols[..]]
            let mut mles = Vec::with_capacity(2 + 2 * num_down_cols);
            mles.push(eq_r);
            mles.extend(next_mles);
            mles.push(precombined);
            mles.extend(down_cols);

            let comb_fn: CombFn<F> = {
                let alphas_b = alphas_b.clone();
                let num_down_cols_local = num_down_cols;
                Box::new(move |mle_values: &[F]| {
                    let eq_val = &mle_values[0];
                    let precombined_val = &mle_values[num_down_cols_local + 1];
                    alphas_b.iter().enumerate().fold(
                        eq_val.clone() * precombined_val,
                        |acc, (k, alpha)| {
                            let next = &mle_values[1 + k];
                            let down_col = &mle_values[num_down_cols_local + 2 + k];
                            acc + alpha.clone() * next * down_col
                        },
                    )
                })
            };

            let group = MultiDegreeSumcheckGroup::new(2, mles, comb_fn);
            family_groups.push((vec![group], cfg));

            if cfg!(debug_assertions) {
                debug_expected_sums.push(compute_expected_sum(
                    family.up_evals,
                    family.down_evals,
                    gammas_b,
                    &alphas_b,
                    zero,
                ));
            }
        }

        // Step 3: Run the lockstep multi-degree sumcheck (degree 2, one
        // group per family).
        let sumcheck_outputs = MultiDegreeSumcheck::prove_as_subprotocol(
            transcript,
            family_groups,
            num_vars,
            q_star_cfg,
        );

        // Step 4: Repackage into per-family (Proof, ProverState).
        let mut result = Vec::with_capacity(num_families);
        for (b_idx, (proof, mut prover_states)) in sumcheck_outputs.into_iter().enumerate() {
            debug_assert_eq!(
                prover_states.len(),
                1,
                "each family contributed exactly one degree group",
            );
            debug_assert_eq!(
                proof.claimed_sums()[0],
                debug_expected_sums[b_idx],
                "claimed sum mismatch on family {b_idx}",
            );
            let state = prover_states.pop().expect("single group per family");
            result.push((
                Proof {
                    sumcheck_proof: proof,
                },
                ProverState {
                    eval_point: state.randomness,
                },
            ));
        }

        Ok(result)
    }

    /// Multi-point evaluation protocol verifier (sumcheck phase, lockstep).
    ///
    /// Mirror of [`prove_as_subprotocol`]: drives one or more families in
    /// lockstep, sharing batching coefficients and per-round challenges
    /// in $[0, q^*)$ via `q_star_cfg`. Returns one [`Subclaim`] per family
    /// carrying $r_0$, $\gamma_j$, $\alpha_k$, `eq_at_r0`, `shifts_at_r0`,
    /// and the inner sumcheck's `expected_evaluation`. The caller
    /// finalizes via [`verify_subclaim`](Self::verify_subclaim) once
    /// `open_evals` are available.
    ///
    /// # Panics
    ///
    /// * If `families` is empty.
    /// * If families disagree on `num_vars` or column counts.
    #[allow(clippy::arithmetic_side_effects, clippy::too_many_arguments)]
    pub fn verify_as_subprotocol(
        transcript: &mut impl Transcript,
        proofs: Vec<Proof<F>>,
        families: Vec<MultipointEvalFamilyInputs<'_, F>>,
        shifts: &[ShiftSpec],
        num_vars: usize,
        q_star_cfg: &F::Config,
    ) -> Result<Vec<Subclaim<F>>, MultipointEvalError<F>> {
        assert!(!families.is_empty(), "need at least one family");
        assert_eq!(
            proofs.len(),
            families.len(),
            "proofs and families must have the same length",
        );

        let num_cols = families[0].up_evals.len();
        let num_down_cols = shifts.len();

        // Sanity check
        for b in &families {
            assert_eq!(
                b.up_evals.len(),
                num_cols,
                "all families must have the same number of up_evals",
            );
            assert_eq!(
                b.down_evals.len(),
                num_down_cols,
                "down_evals length must match shifts length",
            );
            assert_eq!(
                b.eval_point.len(),
                num_vars,
                "all families must have the same num_vars",
            );
        }

        // Step 1: Sample shared $\alpha_k$ and $\gamma_j$ in $[0, q^*)$
        // (must match prover transcript order: alphas first, then gammas).
        let shared_alpha_ints: Vec<F::Integer> = (0..num_down_cols)
            .map(|_| {
                transcript
                    .get_field_challenge::<F>(q_star_cfg)
                    .lift_to_integer()
            })
            .collect();
        let shared_gamma_ints: Vec<F::Integer> = (0..num_cols)
            .map(|_| {
                transcript
                    .get_field_challenge::<F>(q_star_cfg)
                    .lift_to_integer()
            })
            .collect();

        let per_family_alphas: Vec<Vec<F>> = families
            .iter()
            .map(|b| {
                shared_alpha_ints
                    .iter()
                    .map(|v| F::from_with_cfg(v.clone(), b.field_cfg))
                    .collect()
            })
            .collect();
        let per_family_gammas: Vec<Vec<F>> = families
            .iter()
            .map(|b| {
                shared_gamma_ints
                    .iter()
                    .map(|v| F::from_with_cfg(v.clone(), b.field_cfg))
                    .collect()
            })
            .collect();

        // Step 2: Per-family claimed-sum check (must equal the integer-
        // shared expected sum derived from each family's up/down evals).
        for (b_idx, (proof, family)) in proofs.iter().zip(families.iter()).enumerate() {
            let zero = F::zero_with_cfg(family.field_cfg);
            let expected = compute_expected_sum(
                family.up_evals,
                family.down_evals,
                &per_family_gammas[b_idx],
                &per_family_alphas[b_idx],
                zero,
            );
            let claimed = &proof.sumcheck_proof.claimed_sums()[0];
            if claimed != &expected {
                return Err(MultipointEvalError::WrongSumcheckSum {
                    got: claimed.clone(),
                    expected,
                });
            }
        }

        // Step 3: Run the lockstep multi-degree sumcheck verifier.
        let proof_refs: Vec<(&MultiDegreeSumcheckProof<F>, &F::Config)> = proofs
            .iter()
            .zip(families.iter())
            .map(|(p, b)| (&p.sumcheck_proof, b.field_cfg))
            .collect();
        let sub_claims: Vec<MultiDegreeSubClaims<F>> = MultiDegreeSumcheck::verify_as_subprotocol(
            transcript,
            num_vars,
            &proof_refs,
            q_star_cfg,
        )?;

        // Step 4: Per-family finalize: recompute selectors at $r_0$.
        families
            .iter()
            .zip(sub_claims)
            .enumerate()
            .map(|(b_idx, (family, sub))| {
                let cfg = family.field_cfg;
                let one = F::one_with_cfg(cfg);
                let r0: Vec<F> = sub.point().to_vec();
                let expected_evaluation = sub.expected_evaluations()[0].clone();

                let eq_at_r0 = zinc_poly::utils::eq_eval(&r0, family.eval_point, one)?;
                let shifts_at_r0: Vec<F> = shifts
                    .iter()
                    .map(|spec| {
                        eval_shift_predicate(family.eval_point, &r0, spec.shift_amount(), cfg)
                    })
                    .collect();

                Ok(Subclaim {
                    r0,
                    expected_evaluation,
                    gammas: per_family_gammas[b_idx].clone(),
                    alphas: per_family_alphas[b_idx].clone(),
                    eq_at_r0,
                    shifts_at_r0,
                })
            })
            .collect()
    }

    /// Finalize the multi-point evaluation check given `open_evals` for one
    /// family.
    ///
    /// Verifies that
    /// `eq_at_r0 * \sum_j(gamma_j * open_eval_j) + \sum_k(alpha_k *
    /// shift_at_r0_k * open_eval[source_col_k])` equals the sumcheck's
    /// expected evaluation. This is a pure arithmetic check with no
    /// transcript interaction — call it once per family with that family's
    /// `open_evals`.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn verify_subclaim(
        subclaim: &Subclaim<F>,
        open_evals: &[F],
        shifts: &[ShiftSpec],
        field_cfg: &F::Config,
    ) -> Result<(), MultipointEvalError<F>> {
        let num_cols = subclaim.gammas.len();

        if open_evals.len() != num_cols {
            return Err(MultipointEvalError::WrongOpenEvalsNumber {
                got: open_evals.len(),
                expected: num_cols,
            });
        }

        let zero = F::zero_with_cfg(field_cfg);

        let batched_up: F = subclaim
            .gammas
            .iter()
            .zip(open_evals.iter())
            .fold(zero.clone(), |acc, (gamma, eval)| {
                acc + gamma.clone() * eval
            });

        // open_evals[j] = trace_col_j(r_0) for all committed (up) columns.
        // Shifted columns reuse the same opening: the shift is captured by
        // the shift_at_r0 selector, so we index by source_col into open_evals.
        let batched_down: F = subclaim
            .alphas
            .iter()
            .enumerate()
            .zip(subclaim.shifts_at_r0.iter())
            .fold(zero, |acc, ((k, alpha), shift_at_r0)| {
                let src_col = shifts[k].source_col();
                acc + alpha.clone() * shift_at_r0 * &open_evals[src_col]
            });

        let expected_evaluation = subclaim.eq_at_r0.clone() * &batched_up + batched_down;

        if expected_evaluation != subclaim.expected_evaluation {
            return Err(MultipointEvalError::ClaimMismatch {
                got: subclaim.expected_evaluation.clone(),
                expected: expected_evaluation,
            });
        }

        Ok(())
    }
}

/// `expected_sum = \sum_j \gamma_j * up_eval_j + \sum_k \alpha_k *
/// down_eval_k`
fn compute_expected_sum<F: PrimeField>(
    up_evals: &[F],
    down_evals: &[F],
    gammas: &[F],
    alphas: &[F],
    zero: F,
) -> F {
    let up_sum = gammas
        .iter()
        .zip(up_evals.iter())
        .fold(zero, |acc, (gamma, up)| acc + gamma.clone() * up);

    alphas
        .iter()
        .zip(down_evals.iter())
        .fold(up_sum, |acc, (alpha, down)| acc + alpha.clone() * down)
}

//
// Error type
//

#[derive(Debug, Error)]
pub enum MultipointEvalError<F: PrimeField> {
    #[error("wrong number of open evaluations: got {got}, expected {expected}")]
    WrongOpenEvalsNumber { got: usize, expected: usize },
    #[error("wrong sumcheck claimed sum: got {got}, expected {expected}")]
    WrongSumcheckSum { got: F, expected: F },
    #[error("multi-point eval claim mismatch: got {got}, expected {expected}")]
    ClaimMismatch { got: F, expected: F },
    #[error("sumcheck error: {0}")]
    SumcheckError(#[from] SumCheckError<F>),
    #[error("arithmetic error: {0}")]
    ArithError(#[from] ArithErrors),
}

#[cfg(test)]
#[allow(
    clippy::arithmetic_side_effects,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]
mod tests {
    use super::*;
    use crypto_bigint::{U128, const_monty_params};
    use crypto_primitives::crypto_bigint_const_monty::ConstMontyField;
    use num_traits::{ConstOne, ConstZero};
    use zinc_poly::mle::{DenseMultilinearExtension, MultilinearExtensionWithConfig};
    use zinc_transcript::Blake3Transcript;

    const_monty_params!(Params, U128, "00000000b933426489189cb5b47d567f");
    type F = ConstMontyField<Params, { U128::LIMBS }>;

    /// Data known to both prover and verifier from earlier protocol steps.
    #[derive(Clone)]
    struct SharedSubprotocolInput {
        eval_point: Vec<F>,
        up_evals: Vec<F>,
        down_evals: Vec<F>,
        shifts: Vec<ShiftSpec>,
        num_vars: usize,
    }

    /// What the prover sends to the verifier.
    #[derive(Clone)]
    struct ProverMessage {
        proof: Proof<F>,
        open_evals: Vec<F>,
    }

    fn make_transcript() -> Blake3Transcript {
        let mut t = Blake3Transcript::default();
        t.absorb_slice(b"Lorem ipsum");
        t
    }

    fn build_trace(
        num_vars: usize,
        num_cols: usize,
        shifts: &[ShiftSpec],
    ) -> (
        Vec<DenseMultilinearExtension<<F as crypto_primitives::Field>::Inner>>,
        SharedSubprotocolInput,
    ) {
        let n = 1usize << num_vars;
        let zero_inner = F::ZERO.into_inner();

        let trace_mles: Vec<DenseMultilinearExtension<_>> = (0..num_cols)
            .map(|col| {
                let evals: Vec<_> = (0..n)
                    .map(|i| F::from((col * n + i + 1) as u32).into_inner())
                    .collect();
                DenseMultilinearExtension::from_evaluations_vec(num_vars, evals, zero_inner)
            })
            .collect();

        let eval_point: Vec<F> = (0..num_vars).map(|i| F::from((i + 7) as u32)).collect();

        let up_evals: Vec<F> = trace_mles
            .iter()
            .map(|mle| mle.clone().evaluate_with_config(&eval_point, &()).unwrap())
            .collect();

        let down_evals: Vec<F> = shifts
            .iter()
            .map(|spec| {
                let mle = &trace_mles[spec.source_col()];
                let c = spec.shift_amount();
                let mut shifted = mle.evaluations[c..].to_vec();
                shifted.extend(vec![zero_inner; c]);
                let shifted_mle =
                    DenseMultilinearExtension::from_evaluations_vec(num_vars, shifted, zero_inner);
                shifted_mle.evaluate_with_config(&eval_point, &()).unwrap()
            })
            .collect();

        let public = SharedSubprotocolInput {
            eval_point,
            up_evals,
            down_evals,
            shifts: shifts.to_vec(),
            num_vars,
        };
        (trace_mles, public)
    }

    /// Prover: has access to the trace, produces a proof and open_evals.
    fn run_prover(
        trace_mles: &[DenseMultilinearExtension<<F as crypto_primitives::Field>::Inner>],
        public: &SharedSubprotocolInput,
    ) -> ProverMessage {
        let mut transcript = make_transcript();
        let mut outputs = MultipointEval::<F>::prove_as_subprotocol(
            &mut transcript,
            vec![MultipointEvalFamilyInputs {
                field_cfg: &(),
                trace_mles,
                eval_point: &public.eval_point,
                up_evals: &public.up_evals,
                down_evals: &public.down_evals,
            }],
            &public.shifts,
            &(),
        )
        .expect("prover should succeed");
        assert_eq!(outputs.len(), 1, "single-family shape");
        let (proof, prover_state) = outputs.pop().expect("single family");

        let r_0 = &prover_state.eval_point;
        let open_evals: Vec<F> = trace_mles
            .iter()
            .map(|mle| mle.clone().evaluate_with_config(r_0, &()).unwrap())
            .collect();

        ProverMessage { proof, open_evals }
    }

    /// Verifier: only receives the proof + open_evals + public data.
    fn run_verifier(
        public: &SharedSubprotocolInput,
        msg: &ProverMessage,
    ) -> Result<Subclaim<F>, MultipointEvalError<F>> {
        let mut subclaims = MultipointEval::<F>::verify_as_subprotocol(
            &mut make_transcript(),
            vec![msg.proof.clone()],
            vec![MultipointEvalFamilyInputs {
                field_cfg: &(),
                trace_mles: &[],
                eval_point: &public.eval_point,
                up_evals: &public.up_evals,
                down_evals: &public.down_evals,
            }],
            &public.shifts,
            public.num_vars,
            &(),
        )?;
        assert_eq!(subclaims.len(), 1, "single-family shape");
        let subclaim = subclaims.pop().expect("single family");

        MultipointEval::<F>::verify_subclaim(&subclaim, &msg.open_evals, &public.shifts, &())?;

        Ok(subclaim)
    }

    /// Convenience: build trace, prove, return (public, message).
    fn honest_interaction(
        num_vars: usize,
        num_cols: usize,
        shifts: &[ShiftSpec],
    ) -> (SharedSubprotocolInput, ProverMessage) {
        let (trace, public) = build_trace(num_vars, num_cols, shifts);
        let msg = run_prover(&trace, &public);
        (public, msg)
    }

    /// Helper: all-columns shift-by-1
    fn all_shift_by_1(num_cols: usize) -> Vec<ShiftSpec> {
        (0..num_cols).map(|i| ShiftSpec::new(i, 1)).collect()
    }

    // --- Happy-path ---

    #[test]
    fn honest_prove_verify_single_column() {
        let shifts = all_shift_by_1(1);
        let (public, msg) = honest_interaction(4, 1, &shifts);
        run_verifier(&public, &msg).unwrap();
    }

    #[test]
    fn honest_prove_verify_many_columns() {
        let shifts = all_shift_by_1(10);
        let (public, msg) = honest_interaction(3, 10, &shifts);
        run_verifier(&public, &msg).unwrap();
    }

    #[test]
    fn honest_prove_verify_no_shifts() {
        let (public, msg) = honest_interaction(3, 3, &[]);
        run_verifier(&public, &msg).unwrap();
    }

    #[test]
    fn honest_prove_verify_mixed_shifts() {
        let shifts = vec![ShiftSpec::new(0, 1), ShiftSpec::new(1, 3)];
        let (public, msg) = honest_interaction(4, 3, &shifts);
        run_verifier(&public, &msg).unwrap();
    }

    #[test]
    fn honest_prove_verify_shift_by_3() {
        let shifts = vec![
            ShiftSpec::new(0, 3),
            ShiftSpec::new(1, 3),
            ShiftSpec::new(2, 3),
        ];
        let (public, msg) = honest_interaction(4, 3, &shifts);
        run_verifier(&public, &msg).unwrap();
    }

    #[test]
    fn honest_prove_verify_same_col_different_shifts() {
        // Column 0 shifted by 2 and by 5
        let shifts = vec![ShiftSpec::new(0, 2), ShiftSpec::new(0, 5)];
        let (public, msg) = honest_interaction(4, 3, &shifts);
        run_verifier(&public, &msg).unwrap();
    }

    // --- Failure: corrupted down_evals with mixed shifts ---

    #[test]
    fn bad_down_eval_rejected_mixed_shifts() {
        let shifts = vec![ShiftSpec::new(0, 1), ShiftSpec::new(1, 3)];
        let (mut public, msg) = honest_interaction(4, 3, &shifts);
        public.down_evals[0] += F::ONE;
        let err = run_verifier(&public, &msg).unwrap_err();
        assert!(
            matches!(err, MultipointEvalError::WrongSumcheckSum { .. }),
            "expected WrongSumcheckSum, got {err:?}",
        );
    }

    // --- Failure: wrong number of open_evals ---

    #[test]
    fn wrong_open_evals_count() {
        let shifts = all_shift_by_1(3);
        let (public, msg) = honest_interaction(3, 3, &shifts);

        let mut msg_short = msg.clone();
        msg_short.open_evals.pop();

        let mut msg_long = msg;
        msg_long.open_evals.push(F::from(42_u32));

        for bad_msg in [&msg_short, &msg_long] {
            let err = run_verifier(&public, bad_msg).unwrap_err();
            assert!(
                matches!(err, MultipointEvalError::WrongOpenEvalsNumber {
                    got,
                    expected: 3,
                } if got == bad_msg.open_evals.len()),
                "expected WrongOpenEvalsNumber, got {err:?}",
            );
        }
    }

    // --- Failure: wrong claimed sum ---

    #[test]
    fn wrong_claimed_sum_via_corrupted_up_evals() {
        let shifts = all_shift_by_1(3);
        let (mut public, msg) = honest_interaction(3, 3, &shifts);
        public.up_evals[0] += F::ONE;
        let err = run_verifier(&public, &msg).unwrap_err();
        assert!(
            matches!(err, MultipointEvalError::WrongSumcheckSum { .. }),
            "expected WrongSumcheckSum, got {err:?}",
        );
    }

    #[test]
    fn wrong_claimed_sum_via_corrupted_down_evals() {
        let shifts = all_shift_by_1(3);
        let (mut public, msg) = honest_interaction(3, 3, &shifts);
        public.down_evals[1] += F::ONE;
        let err = run_verifier(&public, &msg).unwrap_err();
        assert!(
            matches!(err, MultipointEvalError::WrongSumcheckSum { .. }),
            "expected WrongSumcheckSum, got {err:?}",
        );
    }

    // --- Failure: wrong open_evals values ---

    #[test]
    fn wrong_open_eval_value() {
        let shifts = all_shift_by_1(3);
        let (public, mut msg) = honest_interaction(3, 3, &shifts);
        msg.open_evals[0] += F::ONE;
        let err = run_verifier(&public, &msg).unwrap_err();
        assert!(
            matches!(err, MultipointEvalError::ClaimMismatch { .. }),
            "expected ClaimMismatch, got {err:?}",
        );
    }

    #[test]
    fn all_open_evals_zeroed() {
        let shifts = all_shift_by_1(3);
        let (public, mut msg) = honest_interaction(3, 3, &shifts);
        for e in &mut msg.open_evals {
            *e = F::ZERO;
        }
        let err = run_verifier(&public, &msg).unwrap_err();
        assert!(
            matches!(err, MultipointEvalError::ClaimMismatch { .. }),
            "expected ClaimMismatch, got {err:?}",
        );
    }

    // --- Failure: mixed shifts ---

    fn mixed_shifts() -> Vec<ShiftSpec> {
        vec![ShiftSpec::new(0, 1), ShiftSpec::new(1, 3)]
    }

    #[test]
    fn mixed_shifts_corrupted_up_eval() {
        let (mut public, msg) = honest_interaction(4, 3, &mixed_shifts());
        public.up_evals[2] += F::ONE; // corrupt unshifted column
        let err = run_verifier(&public, &msg).unwrap_err();
        assert!(
            matches!(err, MultipointEvalError::WrongSumcheckSum { .. }),
            "expected WrongSumcheckSum, got {err:?}",
        );
    }

    #[test]
    fn mixed_shifts_wrong_open_eval() {
        let (public, mut msg) = honest_interaction(4, 3, &mixed_shifts());
        msg.open_evals[1] += F::ONE; // corrupt a shifted column's opening
        let err = run_verifier(&public, &msg).unwrap_err();
        assert!(
            matches!(err, MultipointEvalError::ClaimMismatch { .. }),
            "expected ClaimMismatch, got {err:?}",
        );
    }

    #[test]
    fn mixed_shifts_tampered_sumcheck() {
        let (public, mut msg) = honest_interaction(4, 3, &mixed_shifts());
        msg.proof.sumcheck_proof.group_messages_mut()[0][0]
            .0
            .tail_evaluations[0] += F::ONE;
        let err = run_verifier(&public, &msg).unwrap_err();
        assert!(
            matches!(
                err,
                MultipointEvalError::SumcheckError(_) | MultipointEvalError::ClaimMismatch { .. }
            ),
            "expected sumcheck or consistency error, got {err:?}",
        );
    }

    // --- Failure: tampered sumcheck round messages ---

    #[test]
    fn tampered_sumcheck_round_message() {
        let shifts = all_shift_by_1(3);
        let (public, mut msg) = honest_interaction(3, 3, &shifts);
        msg.proof.sumcheck_proof.group_messages_mut()[0][0]
            .0
            .tail_evaluations[0] += F::ONE;
        let err = run_verifier(&public, &msg).unwrap_err();
        assert!(
            matches!(
                err,
                MultipointEvalError::SumcheckError(_) | MultipointEvalError::ClaimMismatch { .. }
            ),
            "expected sumcheck or consistency error, got {err:?}",
        );
    }
}
