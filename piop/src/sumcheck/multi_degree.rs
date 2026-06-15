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

use crypto_primitives::{FromPrimitiveWithConfig, PrimeField};
use num_traits::Zero;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::marker::PhantomData;
use zinc_poly::mle::DenseMultilinearExtension;
use zinc_transcript::traits::{ConstTranscribable, GenTranscribable, Transcribable, Transcript};
use zinc_utils::{
    add, cfg_into_iter, cfg_iter, cfg_iter_mut, inner_transparent_field::InnerTransparentField, mul,
};

use crate::CombFn;

use super::{
    SumCheckError,
    prover::{
        NatEvaluatedPolyWithoutConstant, ProverMsg as SumcheckProverMsg,
        ProverState as SumcheckProverState,
    },
    verifier::VerifierState,
};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Output of a [`Round1FastPath::round_1_message`] call.
/// Carries the asserted sum and the round-1 polynomial tail (evaluations at `1,
/// 2, ..., degree`, omitting the constant term) that the framework wraps into a
/// regular [`SumcheckProverMsg`] and absorbs into the transcript.
pub struct Round1Output<F> {
    pub asserted_sum: F,
    pub tail_evaluations: Vec<F>,
}

/// Opt-in hook on a [`MultiDegreeSumcheckGroup`] that bypasses
/// [`SumcheckProverState::prove_round`] in the first round and replaces
/// the full-size MLE fold by a closed-form half-size construction. Groups
/// that don't supply it are run as usual.
///
/// Implementors must produce a round-1 message bit-identical to what the
/// standard prover would emit and post-fold MLEs bit-identical to what
/// `fix_variables_with_config(&[r_1], cfg)` would produce on the standard
/// path.
pub trait Round1FastPath<F: PrimeField>: Send + Sync {
    /// Closed-form computation of the round-1 polynomial tail plus the
    /// asserted sum `p_1(0) + p_1(1)`.
    fn round_1_message(&self, config: &F::Config) -> Round1Output<F>;

    /// Closed-form fold of the group MLEs by the verifier's first
    /// challenge `r_1`. Returns the half-size MLEs in the same order
    /// `prepare_sumcheck_group` would have produced for the standard
    /// path.
    fn fold_with_challenge(
        self: Box<Self>,
        challenge: &F,
        config: &F::Config,
    ) -> Vec<DenseMultilinearExtension<F::Inner>>;
}

/// A single degree group for the multi-degree sumcheck: (degree, mles,
/// comb_fn).
pub struct MultiDegreeSumcheckGroup<F: PrimeField> {
    degree: usize,
    poly: Vec<DenseMultilinearExtension<F::Inner>>,
    comb_fn: CombFn<F>,
    round_1_fast_path: Option<Box<dyn Round1FastPath<F>>>,
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
            round_1_fast_path: None,
        }
    }

    pub fn new_with_fast_path(
        degree: usize,
        poly: Vec<DenseMultilinearExtension<F::Inner>>,
        comb_fn: CombFn<F>,
        round_1_fast: Box<dyn Round1FastPath<F>>,
    ) -> Self {
        Self {
            degree,
            poly,
            comb_fn,
            round_1_fast_path: Some(round_1_fast),
        }
    }
}

/// Proof for a multi-degree sumcheck.
///
/// `group_messages[g][round]` = prover message for group g in that round.
/// All groups share verifier challenges, common evaluation point.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MultiDegreeSumcheckProof<F> {
    /// List of prover messages, one for each round per group.
    group_messages: Vec<Vec<SumcheckProverMsg<F>>>,
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

impl<F: PrimeField> GenTranscribable for MultiDegreeSumcheckProof<F>
where
    F::Integer: ConstTranscribable,
{
    fn read_transcription_bytes_exact(bytes: &[u8]) -> Self {
        let mod_size = F::Integer::NUM_BYTES;
        let cfg = zinc_transcript::read_field_cfg::<F>(&bytes[..mod_size]);
        let bytes = &bytes[mod_size..];

        let (num_groups, bytes) = u32::read_transcription_bytes_subset(bytes);
        let num_groups = usize::try_from(num_groups).expect("group count must fit into usize");

        let (num_vars, mut bytes) = u32::read_transcription_bytes_subset(bytes);
        let num_vars = usize::try_from(num_vars).expect("num_vars must fit into usize");

        let mut degrees = Vec::with_capacity(num_groups);
        for _ in 0..num_groups {
            let (deg, rest) = u32::read_transcription_bytes_subset(bytes);
            degrees.push(usize::try_from(deg).expect("degree must fit into usize"));
            bytes = rest;
        }

        let mut group_messages = Vec::with_capacity(num_groups);
        for &deg in &degrees {
            let msg_bytes = mul!(deg, F::Integer::NUM_BYTES);
            let mut msgs = Vec::with_capacity(num_vars);
            for _ in 0..num_vars {
                let tail_evaluations =
                    zinc_transcript::read_field_vec_with_cfg(&bytes[..msg_bytes], &cfg);
                msgs.push(SumcheckProverMsg(NatEvaluatedPolyWithoutConstant {
                    tail_evaluations,
                }));
                bytes = &bytes[msg_bytes..];
            }
            group_messages.push(msgs);
        }

        let mut claimed_sums = Vec::with_capacity(num_groups);
        for _ in 0..num_groups {
            let cs = F::Integer::read_transcription_bytes_exact(&bytes[..F::Integer::NUM_BYTES]);
            let cs = F::from_with_cfg(cs, &cfg);
            claimed_sums.push(cs);
            bytes = &bytes[F::Integer::NUM_BYTES..];
        }

        Self {
            group_messages,
            claimed_sums,
            degrees,
        }
    }

    fn write_transcription_bytes_exact(&self, mut buf: &mut [u8]) {
        buf = zinc_transcript::append_field_cfg::<F>(buf, &F::modulus(self.claimed_sums[0].cfg()));

        let num_groups =
            u32::try_from(self.group_messages.len()).expect("num groups must fit into u32");
        num_groups.write_transcription_bytes_exact(&mut buf[..u32::NUM_BYTES]);
        buf = &mut buf[u32::NUM_BYTES..];

        // All groups share the same number of rounds (num_vars).
        let num_vars =
            u32::try_from(self.group_messages[0].len()).expect("num_vars must fit into u32");
        num_vars.write_transcription_bytes_exact(&mut buf[..u32::NUM_BYTES]);
        buf = &mut buf[u32::NUM_BYTES..];

        for &deg in &self.degrees {
            let deg = u32::try_from(deg).expect("degree must fit into u32");
            deg.write_transcription_bytes_exact(&mut buf[..u32::NUM_BYTES]);
            buf = &mut buf[u32::NUM_BYTES..];
        }

        for group in &self.group_messages {
            for msg in group {
                buf = zinc_transcript::append_field_vec_lifted(buf, &msg.0.tail_evaluations);
            }
        }

        for cs in &self.claimed_sums {
            cs.lift_to_integer()
                .write_transcription_bytes_exact(&mut buf[..F::Integer::NUM_BYTES]);
            buf = &mut buf[F::Integer::NUM_BYTES..];
        }
    }
}

impl<F: PrimeField> Transcribable for MultiDegreeSumcheckProof<F>
where
    F::Integer: ConstTranscribable,
{
    fn get_num_bytes(&self) -> usize {
        let num_groups = self.group_messages.len();
        let num_vars = self.group_messages[0].len();
        // total_evals = Σ_g (degree_g × num_vars)
        let total_evals: usize = self.degrees.iter().map(|&d| mul!(d, num_vars)).sum();

        // [field_cfg][num_groups][num_vars][deg₀..degₙ][evals...][claimed_sums]
        let header = add!(F::Integer::NUM_BYTES, add!(u32::NUM_BYTES, u32::NUM_BYTES));
        let degrees = mul!(num_groups, u32::NUM_BYTES);
        let eval_data = mul!(total_evals, F::Integer::NUM_BYTES);
        let claimed = mul!(num_groups, F::Integer::NUM_BYTES);

        add!(header, add!(degrees, add!(eval_data, claimed)))
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
    /// Drives one or more **branches** of multi-degree sumchecks in lockstep,
    /// sharing one per-round verifier challenge across **all branches and all
    /// groups within each branch**. Each branch carries its own degree groups
    /// in its own field config (i.e. a different prime); the shared challenge
    /// is sampled once per round as an integer in $[0, q^*)$ via
    /// `q_star_cfg`, then lifted into each branch's field via
    /// `F::from_with_cfg` (a no-op type cast when $q^* \le q_i$).
    ///
    /// Proves, for every branch $b$ and every group $g$ in that branch:
    ///
    /// $$
    /// \sum_{x \in \{0, 1\}^{\text{num\\_vars}}} G_{b, g}(x) =
    /// \text{claimed\\_sum}_{b, g}
    /// $$
    ///
    /// where $G_{b, g}(x) = \text{comb\\_fn}_{b, g}(\text{mles}_{b, g}(x))$.
    ///
    /// Designed to be used as a subprotocol within a larger system: takes
    /// the FS transcript (`transcript`) as input and returns the **internal
    /// ProverState** alongside the sumcheck proof for every branch. Claimed
    /// sums are derived by the prover during the first round.
    ///
    /// The single-branch case (`branches.len() == 1`) is the natural
    /// degenerate form; pass `q_star_cfg = &branches[0].1` and the lift is
    /// the identity.
    ///
    /// # Arguments
    ///
    /// * `transcript`: Fiat-Shamir transcript.
    /// * `branches`: One entry per branch: `(groups, F::Config)`. Each branch
    ///   contributes one or more degree groups that share that branch's
    ///   `F::Config`.
    /// * `num_vars`: Number of variables (must be consistent across all groups
    ///   in all branches).
    /// * `q_star_cfg`: Field configuration used for transcript metadata absorbs
    ///   and per-round challenge squeezes. For `fq-unify`, this is the smallest
    ///   of the per-branch moduli (so every branch can losslessly cast the
    ///   shared integer into its own field).
    ///
    /// # Returns
    ///
    /// One `(proof, prover_states)` tuple per branch, in the order branches
    /// were provided.
    ///
    /// # Panics
    ///
    /// * If `num_vars == 0`.
    /// * If `branches` is empty or any branch has no groups.
    #[allow(clippy::type_complexity)]
    pub fn prove_as_subprotocol(
        transcript: &mut impl Transcript,
        branches: Vec<(Vec<MultiDegreeSumcheckGroup<F>>, &F::Config)>,
        num_vars: usize,
        q_star_cfg: &F::Config,
    ) -> Vec<(MultiDegreeSumcheckProof<F>, Vec<SumcheckProverState<F>>)>
    where
        F: InnerTransparentField + Send + Sync,
        F::Integer: ConstTranscribable + Zero,
    {
        assert!(
            num_vars > 0,
            "Attempts to prove a constant: num_vars must be > 0"
        );
        assert!(!branches.is_empty(), "need at least one branch");
        for (groups, _) in &branches {
            assert!(!groups.is_empty(), "every branch needs at least one group");
        }

        let num_branches = branches.len();
        let mut buf = vec![0; F::Integer::NUM_BYTES];

        // Metadata: absorb (num_vars, num_branches) under `q_star_cfg` so the
        // transcript layout is canonical across branches.
        let nvars_field = F::from_with_cfg(num_vars as u64, q_star_cfg);
        let nbranches_field = F::from_with_cfg(num_branches as u64, q_star_cfg);
        transcript.absorb_random_field(&nvars_field, &mut buf);
        transcript.absorb_random_field(&nbranches_field, &mut buf);

        // Per-branch state, one `Vec` per branch.
        let mut per_branch_group_messages: Vec<Vec<Vec<SumcheckProverMsg<F>>>> =
            Vec::with_capacity(num_branches);
        let mut per_branch_claimed_sums: Vec<Vec<F>> = Vec::with_capacity(num_branches);
        let mut per_branch_prover_states: Vec<Vec<SumcheckProverState<F>>> =
            Vec::with_capacity(num_branches);
        let mut per_branch_comb_fns: Vec<Vec<CombFn<F>>> = Vec::with_capacity(num_branches);
        let mut per_branch_fast_paths: Vec<Vec<Option<Box<dyn Round1FastPath<F>>>>> =
            Vec::with_capacity(num_branches);
        let mut per_branch_cfg: Vec<&F::Config> = Vec::with_capacity(num_branches);

        for (groups, cfg) in branches {
            let num_groups = groups.len();
            let ngroups_field = F::from_with_cfg(num_groups as u64, q_star_cfg);
            transcript.absorb_random_field(&ngroups_field, &mut buf);

            let group_messages: Vec<Vec<SumcheckProverMsg<F>>> = (0..num_groups)
                .map(|_| Vec::with_capacity(num_vars))
                .collect();
            let mut prover_states: Vec<SumcheckProverState<F>> = Vec::with_capacity(num_groups);
            let mut comb_fns: Vec<CombFn<F>> = Vec::with_capacity(num_groups);
            let mut fast_paths: Vec<Option<Box<dyn Round1FastPath<F>>>> =
                Vec::with_capacity(num_groups);

            for group in groups {
                let degree_field = F::from_with_cfg(group.degree as u64, q_star_cfg);
                transcript.absorb_random_field(&degree_field, &mut buf);

                prover_states.push(SumcheckProverState::new(group.poly, num_vars, group.degree));
                comb_fns.push(group.comb_fn);
                fast_paths.push(group.round_1_fast_path);
            }

            per_branch_group_messages.push(group_messages);
            per_branch_claimed_sums.push(Vec::with_capacity(num_groups));
            per_branch_prover_states.push(prover_states);
            per_branch_comb_fns.push(comb_fns);
            per_branch_fast_paths.push(fast_paths);
            per_branch_cfg.push(cfg);
        }

        // Per-branch last challenge in each branch's field. The underlying
        // integer is shared across branches; each branch lifts it locally.
        let mut per_branch_verifier_msg: Vec<Option<F>> = vec![None; num_branches];

        for round in 1..=num_vars {
            // 1. Each branch produces its round messages (per-branch cfg).
            for b in 0..num_branches {
                let cfg = &per_branch_cfg[b];
                let verifier_msg = &per_branch_verifier_msg[b];
                let round_msgs: Vec<SumcheckProverMsg<F>> =
                    cfg_iter_mut!(per_branch_prover_states[b])
                        .zip(cfg_iter!(per_branch_comb_fns[b]))
                        .zip(cfg_iter!(per_branch_fast_paths[b]))
                        .map(|((state, comb_fn), fast_path)| {
                            if round == 1
                                && let Some(fast_path) = fast_path.as_ref()
                            {
                                // First round: per-group dispatch to fast path if available
                                let out = fast_path.round_1_message(cfg);
                                state.asserted_sum = Some(out.asserted_sum);
                                state.round = 1;
                                SumcheckProverMsg(NatEvaluatedPolyWithoutConstant::new(
                                    out.tail_evaluations,
                                ))
                            } else {
                                state.prove_round(verifier_msg, comb_fn, cfg)
                            }
                        })
                        .collect();

                for msg in &round_msgs {
                    transcript.absorb_random_field_slice(&msg.0.tail_evaluations, &mut buf);
                }

                for (j, msg) in round_msgs.into_iter().enumerate() {
                    per_branch_group_messages[b][j].push(msg);
                }
            }

            // 2. Sample one shared integer challenge in [0, q*) via q_star_cfg.
            let shared_chal_q_star: F = transcript.get_field_challenge(q_star_cfg);
            transcript.absorb_random_field(&shared_chal_q_star, &mut buf);
            let shared_chal_int = shared_chal_q_star.lift_to_integer();

            // 3. Per branch: lift the shared integer into its field and feed each group.
            //    Install fast-path post-fold MLEs on round 1.
            for b in 0..num_branches {
                let chal_b = F::from_with_cfg(shared_chal_int.clone(), per_branch_cfg[b]);
                if round == 1 {
                    per_branch_prover_states[b]
                        .iter_mut()
                        .zip(per_branch_fast_paths[b].iter_mut())
                        .for_each(|(state, fp_slot)| {
                            if let Some(fp) = fp_slot.take() {
                                state.mles = fp.fold_with_challenge(&chal_b, per_branch_cfg[b]);
                                state.skip_next_fold = true;
                            }
                        });
                }
                per_branch_verifier_msg[b] = Some(chal_b);
            }
        }

        // Finalize per branch (mirrors the original single-branch tail).
        let output = cfg_into_iter!(per_branch_prover_states)
            .zip(cfg_into_iter!(per_branch_group_messages))
            .zip(cfg_into_iter!(per_branch_claimed_sums))
            .zip(cfg_into_iter!(per_branch_verifier_msg))
            .map(
                |(((mut prover_states, group_messages), mut claimed_sums), last_chal)| {
                    prover_states.iter_mut().for_each(|state| {
                        let sum = state
                            .asserted_sum
                            .clone()
                            .expect("asserted sum should be recorded after the first prover round");
                        claimed_sums.push(sum);

                        if let Some(ref vmsg) = last_chal {
                            state.randomness.push(vmsg.clone());
                        }
                    });

                    let degrees = prover_states.iter().map(|s| s.max_degree).collect();
                    let proof = MultiDegreeSumcheckProof {
                        group_messages,
                        claimed_sums,
                        degrees,
                    };
                    (proof, prover_states)
                },
            )
            .collect();

        output
    }

    /// Multi-degree sumcheck verifier.
    ///
    /// Mirror of [`prove_as_subprotocol`]: drives one or more branches of
    /// multi-degree sumchecks in lockstep, sharing one per-round challenge
    /// sampled in $[0, q^*)$ via `q_star_cfg` and lifted into each branch's
    /// field. Verifies, for every branch $b$ and every group $g$:
    ///
    /// $$
    /// \sum_{x \in \{0, 1\}^{\text{num\\_vars}}} G_{b, g}(x) =
    /// \text{claimed\\_sum}_{b, g}
    /// $$
    ///
    /// where $G_{b, g}(x) = \text{comb\\_fn}_{b, g}(\text{mles}_{b, g}(x))$.
    ///
    /// Returns one `MultiDegreeSubClaims<F>` per branch: shared evaluation
    /// point `r*` (in that branch's field) and per-group expected
    /// evaluations. The caller must verify each branch's MLE combination at
    /// its `r*` equals its expected evaluation.
    ///
    /// The single-branch case (`proofs.len() == 1`) is the natural degenerate
    /// form; pass `q_star_cfg = &proofs[0].1`.
    ///
    /// # Arguments
    ///
    /// * `transcript`: Fiat-Shamir transcript (must match prover state at the
    ///   start of the sumcheck).
    /// * `num_vars`: Number of variables (sumcheck rounds).
    /// * `proofs`: One `(proof, F::Config)` per branch.
    /// * `q_star_cfg`: Field configuration used for transcript metadata absorbs
    ///   and per-round challenge squeezes (mirror of the prover).
    ///
    /// # Panics
    ///
    /// * If `num_vars == 0`.
    /// * If `proofs` is empty or any branch's proof has no groups.
    pub fn verify_as_subprotocol(
        transcript: &mut impl Transcript,
        num_vars: usize,
        proofs: &[(&MultiDegreeSumcheckProof<F>, &F::Config)],
        q_star_cfg: &F::Config,
    ) -> Result<Vec<MultiDegreeSubClaims<F>>, SumCheckError<F>>
    where
        F: InnerTransparentField,
        F::Integer: ConstTranscribable,
    {
        assert!(
            num_vars > 0,
            "Attempts to prove a constant: num_vars must be > 0"
        );
        assert!(!proofs.is_empty(), "need at least one branch");

        let num_branches = proofs.len();
        let mut buf = vec![0; F::Integer::NUM_BYTES];

        // Metadata: (num_vars, num_branches) under q_star_cfg.
        let nvars_field = F::from_with_cfg(num_vars as u64, q_star_cfg);
        let nbranches_field = F::from_with_cfg(num_branches as u64, q_star_cfg);
        transcript.absorb_random_field(&nvars_field, &mut buf);
        transcript.absorb_random_field(&nbranches_field, &mut buf);

        let mut per_branch_verifier_states: Vec<Vec<VerifierState<F>>> =
            Vec::with_capacity(num_branches);
        for (proof, cfg) in proofs {
            let num_groups = proof.degrees.len();
            assert!(num_groups != 0, "every branch needs at least one group");
            let ngroups_field = F::from_with_cfg(num_groups as u64, q_star_cfg);
            transcript.absorb_random_field(&ngroups_field, &mut buf);

            let states: Vec<VerifierState<F>> = (0..num_groups)
                .map(|j| {
                    let degree = proof.degrees[j];
                    let degree_field = F::from_with_cfg(degree as u64, q_star_cfg);
                    transcript.absorb_random_field(&degree_field, &mut buf);
                    VerifierState::new(num_vars, degree, *cfg)
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
                states.len(),
                proof.group_messages.len(),
                "verifier states ({}) must match proof groups ({})",
                states.len(),
                proof.group_messages.len(),
            );

            per_branch_verifier_states.push(states);
        }

        for i in 0..num_vars {
            // Absorb all branches' messages in branch-major order to match
            // the prover.
            for (proof, _) in proofs {
                proof.group_messages.iter().for_each(|msg| {
                    transcript.absorb_random_field_slice(&msg[i].0.tail_evaluations, &mut buf)
                });
            }

            // One shared integer challenge per round under q_star_cfg.
            let shared_chal_q_star: F = transcript.get_field_challenge(q_star_cfg);
            transcript.absorb_random_field(&shared_chal_q_star, &mut buf);
            let shared_chal_int = shared_chal_q_star.lift_to_integer();

            for (b, (proof, cfg)) in proofs.iter().enumerate() {
                let chal_b = F::from_with_cfg(shared_chal_int.clone(), cfg);
                per_branch_verifier_states[b]
                    .iter_mut()
                    .zip(proof.group_messages.iter())
                    .for_each(|(state, msg)| {
                        state.verify_round_with_challenge(&msg[i], chal_b.clone())
                    });
            }
        }

        // TODO: parallelize when multiple lookup groups exist
        let mut output = Vec::with_capacity(num_branches);
        for (b, (proof, _)) in proofs.iter().enumerate() {
            let states = std::mem::take(&mut per_branch_verifier_states[b]);
            let mut shared_point: Option<Vec<F>> = None;
            let mut expected_evaluations = Vec::with_capacity(states.len());
            for (j, state) in states.into_iter().enumerate() {
                let subclaim = state.check_and_generate_subclaim(proof.claimed_sums[j].clone())?;
                if let Some(ref p) = shared_point {
                    debug_assert_eq!(p, &subclaim.point);
                } else {
                    shared_point = Some(subclaim.point);
                }
                expected_evaluations.push(subclaim.expected_evaluation);
            }
            output.push(MultiDegreeSubClaims {
                point: shared_point.expect("at least one group"),
                expected_evaluations,
            });
        }

        Ok(output)
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
    use zinc_transcript::Blake3Transcript;

    const_monty_params!(TestParams, U128, "00000000b933426489189cb5b47d567f");
    type F = ConstMontyField<TestParams, { U128::LIMBS }>;

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

        // Prove (single-branch shape: one branch with `q_star_cfg = cfg`).
        let mut pt = Blake3Transcript::new();
        let mut outputs = MultiDegreeSumcheck::<F>::prove_as_subprotocol(
            &mut pt,
            vec![(vec![g0, g1], cfg)],
            num_vars,
            cfg,
        );
        let (proof, _states) = outputs.pop().expect("single branch");

        // Verify
        let mut vt = Blake3Transcript::new();
        let mut subclaims_vec = MultiDegreeSumcheck::<F>::verify_as_subprotocol(
            &mut vt,
            num_vars,
            &[(&proof, cfg)],
            cfg,
        )
        .expect("verification should succeed");
        let subclaims = subclaims_vec.pop().expect("single branch");

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

        let mut pt = Blake3Transcript::new();
        let mut outputs = MultiDegreeSumcheck::<F>::prove_as_subprotocol(
            &mut pt,
            vec![(vec![g], cfg)],
            num_vars,
            cfg,
        );
        let (proof, _) = outputs.pop().expect("single branch");

        let mut vt = Blake3Transcript::new();
        let mut subclaims_vec = MultiDegreeSumcheck::<F>::verify_as_subprotocol(
            &mut vt,
            num_vars,
            &[(&proof, cfg)],
            cfg,
        )
        .expect("verification should succeed");
        let subclaims = subclaims_vec.pop().expect("single branch");

        let point = &subclaims.point;
        let eq_eval = zinc_poly::utils::eq_eval(point, &r, F::from(1u32)).unwrap();
        let a_eval = mle.clone().evaluate_with_config(point, cfg).unwrap();

        assert_eq!(subclaims.expected_evaluations[0], eq_eval * a_eval);
    }

    /// Two branches sharing the same `F::Config` (i.e. $q^* = q_i$ for every
    /// branch): every branch must accept the shared per-round challenge in
    /// its field, and the resulting subclaims must verify per branch.
    ///
    /// Exercises the multi-branch behavior of the unified API (one shared
    /// integer challenge per round, lifted into each branch).
    #[test]
    fn two_branches_same_cfg() {
        let num_vars = 3;
        let cfg = &();
        let q_star_cfg = cfg;

        // Branch A: degree-2 group on (a, b).
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
        let r_a: Vec<F> = vec![F::from(5u32), F::from(7u32), F::from(11u32)];
        let eq_r_a = build_eq_x_r_inner(&r_a, cfg).unwrap();
        let group_a = MultiDegreeSumcheckGroup::new(
            2,
            vec![eq_r_a.clone(), a_mle.clone(), b_mle.clone()],
            Box::new(|v: &[F]| v[0] * (v[1] + v[2])),
        );

        // Branch B: degree-3 group on (a, b), different evaluation point.
        let r_b: Vec<F> = vec![F::from(2u32), F::from(3u32), F::from(5u32)];
        let eq_r_b = build_eq_x_r_inner(&r_b, cfg).unwrap();
        let group_b = MultiDegreeSumcheckGroup::new(
            3,
            vec![eq_r_b.clone(), a_mle.clone(), b_mle.clone()],
            Box::new(|v: &[F]| v[0] * v[1] * v[2]),
        );

        // Prover.
        let mut pt = Blake3Transcript::new();
        let mut outputs = MultiDegreeSumcheck::<F>::prove_as_subprotocol(
            &mut pt,
            vec![(vec![group_a], cfg), (vec![group_b], cfg)],
            num_vars,
            q_star_cfg,
        );
        assert_eq!(outputs.len(), 2);
        let (proof_b, _) = outputs.pop().unwrap();
        let (proof_a, _) = outputs.pop().unwrap();

        // Verifier.
        let mut vt = Blake3Transcript::new();
        let subclaims = MultiDegreeSumcheck::<F>::verify_as_subprotocol(
            &mut vt,
            num_vars,
            &[(&proof_a, cfg), (&proof_b, cfg)],
            q_star_cfg,
        )
        .expect("verification should succeed");
        assert_eq!(subclaims.len(), 2);

        // Both branches must land at the same shared point (same cfg →
        // same lift).
        assert_eq!(subclaims[0].point(), subclaims[1].point());

        // Per-branch subclaim checks against the polynomial identity.
        let point = subclaims[0].point();
        let eq_a_eval = zinc_poly::utils::eq_eval(point, &r_a, F::from(1u32)).unwrap();
        let eq_b_eval = zinc_poly::utils::eq_eval(point, &r_b, F::from(1u32)).unwrap();
        let a_eval = a_mle.evaluate_with_config(point, cfg).unwrap();
        let b_eval = b_mle.evaluate_with_config(point, cfg).unwrap();

        assert_eq!(
            subclaims[0].expected_evaluations()[0],
            eq_a_eval * (a_eval + b_eval),
        );
        assert_eq!(
            subclaims[1].expected_evaluations()[0],
            eq_b_eval * a_eval * b_eval,
        );
    }
}
