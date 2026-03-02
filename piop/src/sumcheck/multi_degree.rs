//! Multi-degree sumcheck: runs multiple degree groups in lockstep with
//! shared verifier randomness, producing a common evaluation point.
//!
//! # Motivation
//!
//! When batching identities of different degrees, a single sumcheck at the
//! maximum degree wastes prover work on low-degree identities (evaluating
//! combination functions at unnecessary extra points). By grouping
//! identities by degree and running them as parallel sumchecks that share
//! verifier challenges, each group sends only `degree` field elements per
//! round, and the prover evaluates each group's combination function at
//! only `degree + 1` points instead of `max_degree + 1`.
//!
//! # Protocol
//!
//! Given G degree groups each with its own degree $d_g$, MLEs, and
//! combination function, the protocol proceeds as follows:
//!
//! 1. Absorb metadata: `num_vars`, `num_groups`, per-group degrees.
//! 2. For each round $i = 1, \ldots, \text{num\_vars}$:
//!    a. Each group $g$ computes its round polynomial $P_g$ and sends
//!       evaluations $P_g(1), \ldots, P_g(d_g)$.
//!    b. All round polynomials are absorbed into the transcript.
//!    c. **One** challenge $r_i$ is sampled from the transcript.
//!    d. All groups fix variable $i$ at $r_i$.
//! 3. Each group produces a subclaim at the **shared** point
//!    $\mathbf{r} = (r_1, \ldots, r_n)$.
//!
//! The verifier checks each group independently and applies any
//! cross-group constraints (e.g. equal claimed sums) imposed by the
//! outer protocol.

use std::marker::PhantomData;

use crypto_primitives::FromPrimitiveWithConfig;
use num_traits::Zero;
use zinc_poly::mle::DenseMultilinearExtension;
use zinc_transcript::traits::{ConstTranscribable, Transcript};
use zinc_utils::inner_transparent_field::InnerTransparentField;

use super::{
    SumCheckError,
    prover::{ProverMsg, ProverState},
    verifier::VerifierState,
};

// ---------------------------------------------------------------------------
// Proof & subclaim types
// ---------------------------------------------------------------------------

/// Proof for a multi-degree sumcheck.
///
/// Contains per-group round messages and claimed sums. All groups share
/// the same verifier challenges, producing a common evaluation point.
#[derive(Clone, Debug)]
pub struct MultiDegreeSumcheckProof<F> {
    /// Per-group round messages.
    ///
    /// `group_messages[g][round]` is the prover message for group `g` in
    /// the given round.
    pub group_messages: Vec<Vec<ProverMsg<F>>>,
    /// Claimed sum for each degree group (derived by the prover during
    /// round 1).
    pub claimed_sums: Vec<F>,
    /// Degree of each group (needed for verification).
    pub degrees: Vec<usize>,
}

/// Sub-claims produced by the multi-degree sumcheck verifier.
///
/// All groups share the same evaluation point since they use common
/// verifier challenges.
#[derive(Debug)]
pub struct MultiDegreeSubClaims<F> {
    /// The shared evaluation point (verifier randomness).
    pub point: Vec<F>,
    /// Expected evaluation for each degree group at the shared point.
    pub expected_evaluations: Vec<F>,
}

// ---------------------------------------------------------------------------
// Prover & verifier
// ---------------------------------------------------------------------------

/// Stateless entry-point for multi-degree sumcheck prove / verify.
pub struct MultiDegreeSumcheck<F>(PhantomData<F>);

impl<F: FromPrimitiveWithConfig> MultiDegreeSumcheck<F> {
    /// Multi-degree sumcheck prover.
    ///
    /// Each element of `groups` is a triple
    /// `(degree, mles, comb_fn)` describing one degree bucket.
    /// All buckets must be defined over the same number of variables
    /// `num_vars`.
    ///
    /// Returns the proof together with the per-group prover states
    /// (so the outer protocol can read final MLE evaluations).
    #[allow(clippy::type_complexity)]
    pub fn prove_as_subprotocol(
        transcript: &mut impl Transcript,
        groups: Vec<(
            usize,
            Vec<DenseMultilinearExtension<F::Inner>>,
            Box<dyn Fn(&[F]) -> F + Send + Sync>,
        )>,
        num_vars: usize,
        config: &F::Config,
    ) -> (MultiDegreeSumcheckProof<F>, Vec<ProverState<F>>)
    where
        F: InnerTransparentField + Send + Sync,
        F::Inner: ConstTranscribable + Zero,
    {
        assert!(!groups.is_empty(), "need at least one degree group");
        assert!(num_vars > 0, "num_vars must be > 0");

        let num_groups = groups.len();
        let mut buf = vec![0u8; F::Inner::NUM_BYTES];

        // ---- Absorb metadata ----
        let nvars_field = F::from_with_cfg(num_vars as u64, config);
        transcript.absorb_random_field(&nvars_field, &mut buf);
        let ngroups_field = F::from_with_cfg(num_groups as u64, config);
        transcript.absorb_random_field(&ngroups_field, &mut buf);

        let mut degrees = Vec::with_capacity(num_groups);
        let mut prover_states = Vec::with_capacity(num_groups);
        let mut comb_fns: Vec<Box<dyn Fn(&[F]) -> F + Send + Sync>> =
            Vec::with_capacity(num_groups);

        for (degree, mles, comb_fn) in groups {
            let deg_field = F::from_with_cfg(degree as u64, config);
            transcript.absorb_random_field(&deg_field, &mut buf);
            degrees.push(degree);
            prover_states.push(ProverState::new(mles, num_vars, degree));
            comb_fns.push(comb_fn);
        }

        // ---- Main rounds ----
        let mut verifier_msg: Option<F> = None;
        let mut group_messages: Vec<Vec<ProverMsg<F>>> =
            vec![Vec::with_capacity(num_vars); num_groups];

        for _ in 0..num_vars {
            // Compute all groups' round polynomials in parallel (each
            // group's prove_round mutates only its own ProverState).
            // Absorb results sequentially to keep transcript deterministic.
            #[cfg(feature = "parallel")]
            let round_msgs: Vec<ProverMsg<F>> = {
                use rayon::prelude::*;
                prover_states
                    .par_iter_mut()
                    .zip(comb_fns.par_iter())
                    .map(|(state, cfn)| state.prove_round(&verifier_msg, &**cfn, config))
                    .collect()
            };
            #[cfg(not(feature = "parallel"))]
            let round_msgs: Vec<ProverMsg<F>> = prover_states
                .iter_mut()
                .zip(comb_fns.iter())
                .map(|(state, cfn)| state.prove_round(&verifier_msg, &**cfn, config))
                .collect();

            // Absorb round messages in deterministic order.
            for (g, msg) in round_msgs.into_iter().enumerate() {
                transcript.absorb_random_field_slice(
                    &msg.0.tail_evaluations,
                    &mut buf,
                );
                group_messages[g].push(msg);
            }

            // Single shared challenge.
            let challenge: F = transcript.get_field_challenge(config);
            transcript.absorb_random_field(&challenge, &mut buf);
            verifier_msg = Some(challenge);
        }

        // ---- Collect claimed sums & push last challenge ----
        let claimed_sums: Vec<F> = prover_states
            .iter()
            .map(|s| {
                s.asserted_sum
                    .clone()
                    .expect("asserted sum should be recorded after the first round")
            })
            .collect();

        if let Some(vmsg) = verifier_msg {
            for state in &mut prover_states {
                state.randomness.push(vmsg.clone());
            }
        }

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
    /// Verifies all degree groups using shared challenges. Returns
    /// per-group subclaims at a common evaluation point.
    pub fn verify_as_subprotocol(
        transcript: &mut impl Transcript,
        num_vars: usize,
        proof: &MultiDegreeSumcheckProof<F>,
        config: &F::Config,
    ) -> Result<MultiDegreeSubClaims<F>, SumCheckError<F>>
    where
        F: InnerTransparentField,
        F::Inner: ConstTranscribable,
    {
        let num_groups = proof.group_messages.len();
        assert!(num_groups > 0, "need at least one degree group");
        assert!(num_vars > 0, "num_vars must be > 0");

        let mut buf = vec![0u8; F::Inner::NUM_BYTES];

        // ---- Absorb metadata (must mirror the prover) ----
        let nvars_field = F::from_with_cfg(num_vars as u64, config);
        transcript.absorb_random_field(&nvars_field, &mut buf);
        let ngroups_field = F::from_with_cfg(num_groups as u64, config);
        transcript.absorb_random_field(&ngroups_field, &mut buf);

        let mut verifier_states: Vec<VerifierState<F>> =
            Vec::with_capacity(num_groups);
        for g in 0..num_groups {
            let deg_field = F::from_with_cfg(proof.degrees[g] as u64, config);
            transcript.absorb_random_field(&deg_field, &mut buf);
            if proof.group_messages[g].len() != num_vars {
                return Err(SumCheckError::InvalidProofLength {
                    expected: num_vars,
                    got: proof.group_messages[g].len(),
                });
            }
            verifier_states.push(VerifierState::new(
                num_vars,
                proof.degrees[g],
                config,
            ));
        }

        // ---- Main rounds ----
        for round in 0..num_vars {
            // Absorb all groups' round messages.
            for g in 0..num_groups {
                let msg = &proof.group_messages[g][round];
                transcript.absorb_random_field_slice(
                    &msg.0.tail_evaluations,
                    &mut buf,
                );
            }

            // Single shared challenge.
            let challenge: F = transcript.get_field_challenge(config);
            transcript.absorb_random_field(&challenge, &mut buf);

            // Store in each group's verifier state.
            for g in 0..num_groups {
                verifier_states[g].receive_round(
                    &proof.group_messages[g][round],
                    challenge.clone(),
                );
            }
        }

        // ---- Generate per-group subclaims ----
        let mut point: Option<Vec<F>> = None;
        let mut expected_evaluations = Vec::with_capacity(num_groups);

        for (g, vs) in verifier_states.into_iter().enumerate() {
            let subclaim =
                vs.check_and_generate_subclaim(proof.claimed_sums[g].clone())?;
            if let Some(ref p) = point {
                debug_assert_eq!(
                    p, &subclaim.point,
                    "all groups must share the same evaluation point"
                );
            } else {
                point = Some(subclaim.point);
            }
            expected_evaluations.push(subclaim.expected_evaluation);
        }

        Ok(MultiDegreeSubClaims {
            point: point.expect("at least one group"),
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
    use zinc_poly::{
        mle::MultilinearExtensionWithConfig,
        utils::build_eq_x_r_inner,
    };
    use zinc_transcript::KeccakTranscript;

    const N: usize = 2;
    const_monty_params!(TestParams, U128, "00000000b933426489189cb5b47d567f");
    type F = ConstMontyField<TestParams, N>;

    /// Two degree groups sharing the same evaluation point.
    ///
    /// - Group 0 (degree 2): `eq(y,r) · (a(y) + b(y))`
    /// - Group 1 (degree 3): `eq(y,r) · a(y) · b(y)`
    #[test]
    fn multi_degree_two_groups() {
        let num_vars = 3;
        let cfg = &();

        let a_vals: Vec<F> =
            (0u32..8).map(|i| F::from(i + 1)).collect();
        let b_vals: Vec<F> =
            (0u32..8).map(|i| F::from(i + 10)).collect();

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

        // Degree-2 group: eq · (a + b)
        let degree_2_mles =
            vec![eq_r.clone(), a_mle.clone(), b_mle.clone()];
        let degree_2_fn: Box<dyn Fn(&[F]) -> F + Send + Sync> =
            Box::new(|vals: &[F]| {
                vals[0].clone() * &(vals[1].clone() + &vals[2])
            });

        // Degree-3 group: eq · a · b
        let degree_3_mles =
            vec![eq_r.clone(), a_mle.clone(), b_mle.clone()];
        let degree_3_fn: Box<dyn Fn(&[F]) -> F + Send + Sync> =
            Box::new(|vals: &[F]| {
                vals[0].clone() * &vals[1] * &vals[2]
            });

        let groups = vec![
            (2usize, degree_2_mles, degree_2_fn),
            (3usize, degree_3_mles, degree_3_fn),
        ];

        // ---- Prove ----
        let mut pt = KeccakTranscript::new();
        let (proof, _states) =
            MultiDegreeSumcheck::<F>::prove_as_subprotocol(
                &mut pt, groups, num_vars, cfg,
            );

        // ---- Verify ----
        let mut vt = KeccakTranscript::new();
        let subclaims =
            MultiDegreeSumcheck::<F>::verify_as_subprotocol(
                &mut vt, num_vars, &proof, cfg,
            )
            .expect("verification should succeed");

        assert_eq!(subclaims.expected_evaluations.len(), 2);

        // ---- Check final evaluations manually ----
        let point = &subclaims.point;
        let eq_eval = zinc_poly::utils::eq_eval(
            point,
            &r,
            F::from(1u32),
        )
        .unwrap();
        let a_eval =
            a_mle.evaluate_with_config(point, cfg).unwrap();
        let b_eval =
            b_mle.evaluate_with_config(point, cfg).unwrap();

        let expected_deg2 =
            eq_eval.clone() * &(a_eval.clone() + &b_eval);
        let expected_deg3 = eq_eval * &a_eval * &b_eval;

        assert_eq!(subclaims.expected_evaluations[0], expected_deg2);
        assert_eq!(subclaims.expected_evaluations[1], expected_deg3);
    }

    /// Single degree group degenerates to the standard sumcheck.
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

        // Degree 2: eq · a
        let mles = vec![eq_r.clone(), mle.clone()];
        let comb_fn: Box<dyn Fn(&[F]) -> F + Send + Sync> =
            Box::new(|v: &[F]| v[0].clone() * &v[1]);

        let groups = vec![(2usize, mles, comb_fn)];

        let mut pt = KeccakTranscript::new();
        let (proof, _) =
            MultiDegreeSumcheck::<F>::prove_as_subprotocol(
                &mut pt, groups, num_vars, cfg,
            );

        let mut vt = KeccakTranscript::new();
        let subclaims =
            MultiDegreeSumcheck::<F>::verify_as_subprotocol(
                &mut vt, num_vars, &proof, cfg,
            )
            .expect("verification should succeed");

        let point = &subclaims.point;
        let eq_eval =
            zinc_poly::utils::eq_eval(point, &r, F::from(1u32))
                .unwrap();
        let a_eval = mle.evaluate_with_config(point, cfg).unwrap();
        let expected = eq_eval * &a_eval;

        assert_eq!(subclaims.expected_evaluations[0], expected);
    }
}
