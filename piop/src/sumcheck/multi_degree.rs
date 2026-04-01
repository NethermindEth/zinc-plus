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
use zinc_poly::mle::DenseMultilinearExtension;
use zinc_transcript::traits::{ConstTranscribable, Transcript};
use zinc_utils::inner_transparent_field::InnerTransparentField;

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
    group_messages: Vec<Vec<ProverMsg<F>>>,
    claimed_sums: Vec<F>,
    degrees: Vec<usize>,
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
    /// # Arguments
    /// - `groups`: Vec of (degree, mles, comb_fn) per degree bucket
    /// - `num_vars`: number of variables (same for all groups)
    ///
    /// # Algorithm
    /// 1. Absorb metadata: num_vars (as field), num_groups (as field), each
    ///    degree (as field)
    /// 2. Create ProverState per group via ProverState::new(mles, num_vars,
    ///    degree)
    /// 3. Main loop (num_vars rounds):
    ///    - Each group: state.prove_round(&verifier_msg, &comb_fn, config) →
    ///      ProverMsg
    ///    - Absorb each group's msg.0.tail_evaluations into transcript
    ///    - Sample one shared challenge: transcript.get_field_challenge(config)
    ///    - Absorb the challenge
    ///    - Set verifier_msg = Some(challenge)
    /// 4. Collect claimed_sums from each state.asserted_sum
    /// 5. Push final challenge into each state.randomness
    ///
    /// # Returns
    /// (MultiDegreeSumcheckProof, Vec<ProverState>) — states needed for MLE
    /// evals after
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
            for j in 0..num_groups {
                let prover_msg = prover_states[j].prove_round(&verifier_msg, &comb_fns[j], config);
                transcript.absorb_random_field_slice(&prover_msg.0.tail_evaluations, &mut buf);
                group_messages[j].push(prover_msg);
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
    /// # Algorithm
    /// 1. Absorb metadata (must mirror prover): num_vars, num_groups, per-group
    ///    degrees
    /// 2. Create VerifierState per group via VerifierState::new(num_vars,
    ///    degree, config)
    /// 3. Validate proof.group_messages[g].len() == num_vars for all g
    /// 4. Main loop (num_vars rounds):
    ///    - Absorb each group's round message tail_evaluations
    ///    - Sample one shared challenge: transcript.get_field_challenge(config)
    ///    - Absorb the challenge
    ///    - Each group: verifier_state.receive_round(&msg, challenge)
    /// 5. Generate subclaims: each
    ///    state.check_and_generate_subclaim(claimed_sum)
    /// 6. Assert all subclaim points are equal (shared challenge → same point)
    ///
    /// # Returns
    /// MultiDegreeSubClaims with shared point + per-group expected_evaluations
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
        let inner_zero = F::from(0u32).inner().clone();

        let a_mle = DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            a_vals.iter().map(|x| x.inner().clone()).collect(),
            inner_zero.clone(),
        );
        let b_mle = DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            b_vals.iter().map(|x| x.inner().clone()).collect(),
            inner_zero.clone(),
        );

        let r: Vec<F> = vec![F::from(5u32), F::from(7u32), F::from(11u32)];
        let eq_r = build_eq_x_r_inner(&r, cfg).unwrap();

        // Group 0 (degree 2): eq · (a + b)
        let g0 = MultiDegreeSumcheckGroup::new(
            2,
            vec![eq_r.clone(), a_mle.clone(), b_mle.clone()],
            Box::new(|v: &[F]| v[0].clone() * &(v[1].clone() + &v[2])),
        );

        // Group 1 (degree 3): eq · a · b
        let g1 = MultiDegreeSumcheckGroup::new(
            3,
            vec![eq_r.clone(), a_mle.clone(), b_mle.clone()],
            Box::new(|v: &[F]| v[0].clone() * &v[1] * &v[2]),
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
            eq_eval.clone() * &(a_eval.clone() + &b_eval)
        );
        assert_eq!(
            subclaims.expected_evaluations[1],
            eq_eval * &a_eval * &b_eval
        );
    }

    /// Multi-degree sumcheck with a single group produces a valid subclaim.
    #[test]
    fn multi_degree_single_group() {
        let num_vars = 2;
        let cfg = &();

        let vals: Vec<F> = (0u32..4).map(|i| F::from(i + 1)).collect();
        let inner_zero = F::from(0u32).inner().clone();
        let mle = DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            vals.iter().map(|x| x.inner().clone()).collect(),
            inner_zero,
        );

        let r: Vec<F> = vec![F::from(3u32), F::from(7u32)];
        let eq_r = build_eq_x_r_inner(&r, cfg).unwrap();

        let g = MultiDegreeSumcheckGroup::new(
            2,
            vec![eq_r.clone(), mle.clone()],
            Box::new(|v: &[F]| v[0].clone() * &v[1]),
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

        assert_eq!(subclaims.expected_evaluations[0], eq_eval * &a_eval);
    }
}
