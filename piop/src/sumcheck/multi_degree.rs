//! Multi-degree sumcheck: runs multiple degree groups in lockstep with
//! shared verifier randomness, producing a common evaluation point.
//!
//! # Protocol
//!
//! Given G degree groups each with (degree_g, mles_g, comb_fn_g):
//!
//! 1. Absorb metadata: num_vars, num_groups, per-group degrees
//! 2. For each round `i = 1..num_vars`:
//!    - Each group computes its round polynomial `P_g` (parallelizable)
//!    - Absorb all round messages in deterministic order
//!    - Sample ONE shared challenge `r_i`
//!    - All groups fix variable `i` at `r_i`
//! 3. Each group produces a subclaim at the shared point r = (r_1, ..., r_n)

use std::marker::PhantomData;
use crypto_primitives::{FromPrimitiveWithConfig, PrimeField};
use num_traits::Zero;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use zinc_poly::mle::DenseMultilinearExtension;
use zinc_transcript::traits::{ConstTranscribable, Transcript};
use zinc_utils::{cfg_iter, cfg_iter_mut, inner_transparent_field::InnerTransparentField};

use crate::CombFn;

use super::{
    SumCheckError,
    prover::{ProverMsg, ProverState},
    verifier::VerifierState,
};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A single degree group for the multi-degree sumcheck: (degree, mles,
/// comb_fn).
pub struct MultiDegreeSumcheckGroup<F: PrimeField> {
    degree: usize,
    poly: Vec<DenseMultilinearExtension<F::Inner>>,
    comb_fn: CombFn<F>,
}

impl<F: PrimeField> MultiDegreeSumcheckGroup<F> {
    pub fn new(
        degree: usize,
        poly: Vec<DenseMultilinearExtension<F::Inner>>,
        comb_fn: CombFn<F>,
    ) -> Self {
        Self {
            degree,
            poly,
            comb_fn,
        }
    }
}

/// Proof for a multi-degree sumcheck.
///
/// `group_messages[g][round]` = prover message for group g in that round.
/// All groups share verifier challenges, common evaluation point.
#[derive(Clone, Debug)]
pub struct MultiDegreeSumcheckProof<F> {
    /// List of prover messages, one for each round per group.
    group_messages: Vec<Vec<ProverMsg<F>>>,
    // The claimed sum for the first round polynomial per group.
    claimed_sums: Vec<F>,
    // Max degrees per group.
    degrees: Vec<usize>,
}

impl<F> MultiDegreeSumcheckProof<F> {
    /// Needed by the verifier to check against expected
    /// sums before running the sumcheck.
    pub fn claimed_sums(&self) -> &[F] {
        &self.claimed_sums
    }
}

/// Sub-claims: shared evaluation point + per-group expected evaluation.
#[derive(Debug)]
pub struct MultiDegreeSubClaims<F> {
    point: Vec<F>,
    expected_evaluations: Vec<F>,
}

impl<F> MultiDegreeSubClaims<F> {
    pub fn point(&self) -> &[F] {
        &self.point
    }

    pub fn expected_evaluations(&self) -> &[F] {
        &self.expected_evaluations
    }
}

// ---------------------------------------------------------------------------
// MultiDegreeSumcheck
// ---------------------------------------------------------------------------

pub struct MultiDegreeSumcheck<F>(PhantomData<F>);

impl<F: FromPrimitiveWithConfig> MultiDegreeSumcheck<F> {
    /// Multi-degree sumcheck prover.
    ///
    /// Runs the prover side of the sumcheck protocol for G degree groups
    /// sharing one verifier challenge per round. Proves the claim:
    ///
    /// $$
    /// \sum_{x \in \{0, 1\}^{\text{num\\_vars}}} G_g(x) =
    /// \text{claimed\\_sum}_g \quad \forall g
    /// $$
    ///
    /// where $G_g(x) = \text{comb\\_fn}_g(\text{mles}_g(x))$ is the combination
    /// function for group $g$ applied to its MLEs.
    ///
    /// It is designed to be used as a subprotocol within a larger system.
    /// since it takes the FS transcript (`transcript` argument) as input
    /// and returns the **internal ProverState** alongside the sumcheck proof.
    ///
    /// Claimed sums are derived by the prover during the first round.
    ///
    /// # Arguments
    ///
    /// * `transcript`: Fiat-Shamir transcript.
    /// * `groups`: One [`MultiDegreeSumcheckGroup`] per degree bucket, each
    ///   carrying its MLEs and combination function.
    /// * `num_vars`: Number of variables (must be consistent across all
    ///   groups).
    /// * `config`: Field configuration.
    ///
    /// # Returns
    ///
    /// A tuple containing:
    ///
    /// 1. [`MultiDegreeSumcheckProof<F>`]: The proof (group messages, claimed
    ///    sums, degrees).
    /// 2. `Vec<ProverState<F>>`: Per-group prover states — needed by the caller
    ///    to evaluate MLEs at the shared point after the sumcheck.
    ///
    /// # Panics
    ///
    /// * Panics if `num_vars == 0` or `groups` is empty.
    #[allow(clippy::type_complexity)]
    pub fn prove_as_subprotocol(
        transcript: &mut impl Transcript,
        groups: Vec<MultiDegreeSumcheckGroup<F>>,
        num_vars: usize,
        config: &F::Config,
    ) -> (MultiDegreeSumcheckProof<F>, Vec<ProverState<F>>)
    where
        F: InnerTransparentField + Send + Sync,
        F::Inner: ConstTranscribable + Zero,
        F::Modulus: ConstTranscribable,
    {
        assert!(
            num_vars > 0,
            "Attempts to prove a constant: num_vars must be > 0"
        );
        assert!(!groups.is_empty(), "need at least one degree group");

        let num_groups = groups.len();
        let mut buf = vec![0; F::Inner::NUM_BYTES];
        let nvars_field = F::from_with_cfg(num_vars as u64, config);
        let ngroups_field = F::from_with_cfg(num_groups as u64, config);
        transcript.absorb_random_field(&nvars_field, &mut buf);
        transcript.absorb_random_field(&ngroups_field, &mut buf);

        let mut verifier_msg = None;
        let mut group_messages: Vec<Vec<ProverMsg<F>>> = (0..num_groups)
            .map(|_| Vec::with_capacity(num_vars))
            .collect();
        let mut claimed_sums = Vec::with_capacity(num_groups);

        let (mut prover_states, comb_fns): (Vec<_>, Vec<_>) = groups
            .into_iter()
            .map(|group| {
                let degree_field = F::from_with_cfg(group.degree as u64, config);
                transcript.absorb_random_field(&degree_field, &mut buf);

                (
                    ProverState::new(group.poly, num_vars, group.degree),
                    group.comb_fn,
                )
            })
            .unzip();

        for _ in 0..num_vars {
            // Parallel: each group computes its round polynomial independently
            let round_msgs: Vec<ProverMsg<F>> = cfg_iter_mut!(prover_states)
                .zip(cfg_iter!(comb_fns))
                .map(|(state, comb_fn)| state.prove_round(&verifier_msg, comb_fn, config))
                .collect();

            // Sequential: absorb in deterministic order, sample one shared challenge
            for msg in &round_msgs {
                transcript.absorb_random_field_slice(&msg.0.tail_evaluations, &mut buf);
            }

            for (j, msg) in round_msgs.into_iter().enumerate() {
                group_messages[j].push(msg);
            }

            let next_verifier_msg = transcript.get_field_challenge(config);
            transcript.absorb_random_field(&next_verifier_msg, &mut buf);

            verifier_msg = Some(next_verifier_msg);
        }

        prover_states.iter_mut().for_each(|state| {
            let sum = state
                .asserted_sum
                .clone()
                .expect("asserted sum should be recorded after the first prover round");
            claimed_sums.push(sum);

            if let Some(ref vmsg) = verifier_msg {
                state.randomness.push(vmsg.clone());
            }
        });

        let degrees = prover_states.iter().map(|s| s.max_degree).collect();

        (
            MultiDegreeSumcheckProof {
                group_messages,
                claimed_sums,
                degrees,
            },
            prover_states,
        )
    }

    /// Multi-degree sumcheck verifier.
    ///
    /// Runs the verifier side of the sumcheck protocol for G degree groups
    /// sharing one verifier challenge per round. Verifies the claim:
    ///
    /// $$
    /// \sum_{x \in \{0, 1\}^{\text{num\\_vars}}} G_g(x) =
    /// \text{claimed\\_sum}_g \quad \forall g
    /// $$
    ///
    /// where $G_g(x) = \text{comb\\_fn}_g(\text{mles}_g(x))$.
    ///
    /// It is designed to be used as a subprotocol within a larger system.
    /// If successful, it returns **Subclaim** for each group, a final equation
    /// that the outer protocol must satisfy for the overall sumcheck proof
    /// to be valid.
    ///
    /// Mirrors the prover transcript exactly: absorbs metadata, then per-round
    /// absorbs all group messages, samples one shared challenge, and calls
    /// [`VerifierState::check_and_generate_subclaim`] per group. Per-group
    /// degrees are read from the proof — no external degree parameter needed.
    ///
    /// # Arguments
    ///
    /// * `transcript`: Fiat-Shamir transcript (must match prover state at the
    ///   start of the sumcheck).
    /// * `num_vars`: Number of variables (sumcheck rounds).
    /// * `proof`: The [`MultiDegreeSumcheckProof`] produced by the prover.
    /// * `config`: Field configuration.
    ///
    /// # Returns
    ///
    /// * `Ok(MultiDegreeSubClaims<F>)`: Shared evaluation point `r*` and
    ///   per-group expected evaluations. The caller must verify each group's
    ///   MLE combination at `r*` equals its expected evaluation.
    /// * `Err(SumCheckError<F>)`: If any round check fails.
    ///
    /// # Panics
    ///
    /// * Panics if `num_vars == 0` or the proof has no groups.
    pub fn verify_as_subprotocol(
        transcript: &mut impl Transcript,
        num_vars: usize,
        proof: &MultiDegreeSumcheckProof<F>,
        config: &F::Config,
    ) -> Result<MultiDegreeSubClaims<F>, SumCheckError<F>>
    where
        F: InnerTransparentField,
        F::Inner: ConstTranscribable,
        F::Modulus: ConstTranscribable,
    {
        assert!(
            num_vars > 0,
            "Attempts to prove a constant: num_vars must be > 0"
        );
        let num_groups = proof.degrees.len();
        assert!(num_groups != 0, "need at least one degree group");

        let mut buf = vec![0; F::Inner::NUM_BYTES];
        let nvars_field = F::from_with_cfg(num_vars as u64, config);
        let ngroups_field = F::from_with_cfg(num_groups as u64, config);
        transcript.absorb_random_field(&nvars_field, &mut buf);
        transcript.absorb_random_field(&ngroups_field, &mut buf);

        let mut verifier_states: Vec<VerifierState<F>> = (0..num_groups)
            .map(|j| {
                let degree = proof.degrees[j];
                let degree_field = F::from_with_cfg(proof.degrees[j] as u64, config);
                transcript.absorb_random_field(&degree_field, &mut buf);

                VerifierState::new(num_vars, degree, config)
            })
            .collect();

        for msgs in &proof.group_messages {
            if msgs.len() != num_vars {
                return Err(SumCheckError::InvalidProofLength {
                    expected: num_vars,
                    got: msgs.len(),
                });
            }
        }

        assert_eq!(
            verifier_states.len(),
            proof.group_messages.len(),
            "number of verifier states ({}) must match number of proof groups ({})",
            verifier_states.len(),
            proof.group_messages.len(),
        );

        for i in 0..num_vars {
            proof.group_messages.iter().for_each(|msg| {
                transcript.absorb_random_field_slice(&msg[i].0.tail_evaluations, &mut buf)
            });

            let shared_challenge: F = transcript.get_field_challenge(config);
            transcript.absorb_random_field(&shared_challenge, &mut buf);

            verifier_states
                .iter_mut()
                .zip(proof.group_messages.iter())
                .for_each(|(state, msg)| state.receive_round(&msg[i], shared_challenge.clone()));
        }

        let mut shared_point: Option<Vec<F>> = None;
        let mut expected_evaluations = Vec::with_capacity(num_groups);
        // TODO: parallelize when multiple lookup groups exist
        for (j, state) in verifier_states.into_iter().enumerate() {
            let subclaim = state.check_and_generate_subclaim(proof.claimed_sums[j].clone())?;
            if let Some(ref p) = shared_point {
                debug_assert_eq!(p, &subclaim.point);
            } else {
                shared_point = Some(subclaim.point)
            }

            expected_evaluations.push(subclaim.expected_evaluation);
        }

        Ok(MultiDegreeSubClaims {
            point: shared_point.expect("at least one group"),
            expected_evaluations,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crypto_bigint::{U128, const_monty_params};
    use crypto_primitives::crypto_bigint_const_monty::ConstMontyField;
    use zinc_poly::{mle::MultilinearExtensionWithConfig, utils::build_eq_x_r_inner};
    use zinc_transcript::KeccakTranscript;

    const N: usize = 2;
    const_monty_params!(TestParams, U128, "00000000b933426489189cb5b47d567f");
    type F = ConstMontyField<TestParams, N>;

    /// Two degree groups sharing the same evaluation point.
    ///
    /// - Group 0 (degree 2): `eq(y, r) · (a(y) + b(y))`
    /// - Group 1 (degree 3): `eq(y, r) · a(y) · b(y)`
    #[test]
    fn multi_degree_two_groups() {
        let num_vars = 3;
        let cfg = &();

        let a_vals: Vec<F> = (0u32..8).map(|i| F::from(i + 1)).collect();
        let b_vals: Vec<F> = (0u32..8).map(|i| F::from(i + 10)).collect();
        let inner_zero = *F::from(0u32).inner();

        let a_mle = DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            a_vals.iter().map(|x| *x.inner()).collect(),
            inner_zero,
        );
        let b_mle = DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            b_vals.iter().map(|x| *x.inner()).collect(),
            inner_zero,
        );

        let r: Vec<F> = vec![F::from(5u32), F::from(7u32), F::from(11u32)];
        let eq_r = build_eq_x_r_inner(&r, cfg).unwrap();

        // Group 0 (degree 2): eq · (a + b)
        let g0 = MultiDegreeSumcheckGroup::new(
            2,
            vec![eq_r.clone(), a_mle.clone(), b_mle.clone()],
            Box::new(|v: &[F]| v[0] * (v[1] + v[2])),
        );

        // Group 1 (degree 3): eq · a · b
        let g1 = MultiDegreeSumcheckGroup::new(
            3,
            vec![eq_r.clone(), a_mle.clone(), b_mle.clone()],
            Box::new(|v: &[F]| v[0] * v[1] * v[2]),
        );

        // Prove
        let mut pt = KeccakTranscript::new();
        let (proof, _states) =
            MultiDegreeSumcheck::<F>::prove_as_subprotocol(&mut pt, vec![g0, g1], num_vars, cfg);

        // Verify
        let mut vt = KeccakTranscript::new();
        let subclaims =
            MultiDegreeSumcheck::<F>::verify_as_subprotocol(&mut vt, num_vars, &proof, cfg)
                .expect("verification should succeed");

        assert_eq!(subclaims.expected_evaluations.len(), 2);

        // Check final evaluations manually
        let point = &subclaims.point;
        let eq_eval = zinc_poly::utils::eq_eval(point, &r, F::from(1u32)).unwrap();
        let a_eval = a_mle.evaluate_with_config(point, cfg).unwrap();
        let b_eval = b_mle.evaluate_with_config(point, cfg).unwrap();

        assert_eq!(
            subclaims.expected_evaluations[0],
            eq_eval * (a_eval + b_eval)
        );
        assert_eq!(subclaims.expected_evaluations[1], eq_eval * a_eval * b_eval);
    }

    /// Multi-degree sumcheck with a single group produces a valid subclaim.
    #[test]
    fn multi_degree_single_group() {
        let num_vars = 2;
        let cfg = &();

        let vals: Vec<F> = (0u32..4).map(|i| F::from(i + 1)).collect();
        let inner_zero = *F::from(0u32).inner();
        let mle = DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            vals.iter().map(|x| *x.inner()).collect(),
            inner_zero,
        );

        let r: Vec<F> = vec![F::from(3u32), F::from(7u32)];
        let eq_r = build_eq_x_r_inner(&r, cfg).unwrap();

        let g = MultiDegreeSumcheckGroup::new(
            2,
            vec![eq_r.clone(), mle.clone()],
            Box::new(|v: &[F]| v[0] * v[1]),
        );

        let mut pt = KeccakTranscript::new();
        let (proof, _) =
            MultiDegreeSumcheck::<F>::prove_as_subprotocol(&mut pt, vec![g], num_vars, cfg);

        let mut vt = KeccakTranscript::new();
        let subclaims =
            MultiDegreeSumcheck::<F>::verify_as_subprotocol(&mut vt, num_vars, &proof, cfg)
                .expect("verification should succeed");

        let point = &subclaims.point;
        let eq_eval = zinc_poly::utils::eq_eval(point, &r, F::from(1u32)).unwrap();
        let a_eval = mle.clone().evaluate_with_config(point, cfg).unwrap();

        assert_eq!(subclaims.expected_evaluations[0], eq_eval * a_eval);
    }
}
