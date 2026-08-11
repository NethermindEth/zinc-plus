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

use crypto_primitives::{BaseFieldConfig, ProjectPrimitiveIntegersWithConfig, SetConfig};
use itertools::Itertools;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::marker::PhantomData;
use zinc_poly::mle::DenseMultilinearExtension;
use zinc_transcript::traits::{ConstTranscribable, GenTranscribable, Transcribable, Transcript};
use zinc_utils::{add, cfg_into_iter, cfg_iter, cfg_iter_mut, mul};

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
/// `fix_variables(cfg, &[r_1])` would produce on the standard
/// path.
pub trait Round1FastPath<C: SetConfig>: Send + Sync {
    /// Closed-form computation of the round-1 polynomial tail plus the
    /// asserted sum `p_1(0) + p_1(1)`.
    fn round_1_message(&self, config: &C) -> Round1Output<C::Element>;

    /// Closed-form fold of the group MLEs by the verifier's first
    /// challenge `r_1`. Returns the half-size MLEs in the same order
    /// `prepare_sumcheck_group` would have produced for the standard
    /// path.
    fn fold_with_challenge(
        self: Box<Self>,
        challenge: &C::Element,
        config: &C,
    ) -> Vec<DenseMultilinearExtension<C::Element>>;
}

/// A single degree group for the multi-degree sumcheck: (degree, mles,
/// comb_fn).
pub struct MultiDegreeSumcheckGroup<C: SetConfig> {
    degree: usize,
    poly: Vec<DenseMultilinearExtension<C::Element>>,
    comb_fn: CombFn<C::Element>,
    round_1_fast_path: Option<Box<dyn Round1FastPath<C>>>,
}

impl<C: SetConfig> MultiDegreeSumcheckGroup<C> {
    pub fn new(
        degree: usize,
        poly: Vec<DenseMultilinearExtension<C::Element>>,
        comb_fn: CombFn<C::Element>,
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
        poly: Vec<DenseMultilinearExtension<C::Element>>,
        comb_fn: CombFn<C::Element>,
        round_1_fast: Box<dyn Round1FastPath<C>>,
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
    /// Maps every field element through `f`, preserving structure — used to
    /// lift elements into wire integers and to project wire integers back
    /// into elements at the (de)serialization boundary.
    pub fn try_map<T, E>(
        &self,
        f: impl FnMut(&F) -> Result<T, E> + Copy,
    ) -> Result<MultiDegreeSumcheckProof<T>, E> {
        Ok(MultiDegreeSumcheckProof {
            group_messages: self
                .group_messages
                .iter()
                .map(|msgs| msgs.iter().map(|m| m.try_map(f)).try_collect())
                .try_collect()?,
            claimed_sums: self.claimed_sums.iter().map(f).try_collect()?,
            degrees: self.degrees.clone(),
        })
    }

    /// Needed by the verifier to check against expected
    /// sums before running the sumcheck.
    pub fn claimed_sums(&self) -> &[F] {
        &self.claimed_sums
    }

    #[cfg(test)]
    pub fn group_messages_mut(&mut self) -> &mut [Vec<SumcheckProverMsg<F>>] {
        &mut self.group_messages
    }
}

/// The proof is transcribed as raw field elements without field metadata:
/// the field config is bound into the transcript separately, at the top
/// level of the surrounding proof.
impl<F: ConstTranscribable> GenTranscribable for MultiDegreeSumcheckProof<F> {
    fn read_transcription_bytes_exact(bytes: &[u8]) -> Self {
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
            let msg_bytes = mul!(deg, F::NUM_BYTES);
            let mut msgs = Vec::with_capacity(num_vars);
            for _ in 0..num_vars {
                let tail_evaluations: Vec<F> =
                    Vec::read_transcription_bytes_exact(&bytes[..msg_bytes]);
                msgs.push(SumcheckProverMsg(NatEvaluatedPolyWithoutConstant {
                    tail_evaluations,
                }));
                bytes = &bytes[msg_bytes..];
            }
            group_messages.push(msgs);
        }

        let mut claimed_sums = Vec::with_capacity(num_groups);
        for _ in 0..num_groups {
            let cs = F::read_transcription_bytes_exact(&bytes[..F::NUM_BYTES]);
            claimed_sums.push(cs);
            bytes = &bytes[F::NUM_BYTES..];
        }

        Self {
            group_messages,
            claimed_sums,
            degrees,
        }
    }

    fn write_transcription_bytes_exact(&self, mut buf: &mut [u8]) {
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
                let evals = &msg.0.tail_evaluations;
                let end = mul!(evals.len(), F::NUM_BYTES);
                evals.write_transcription_bytes_exact(&mut buf[..end]);
                buf = &mut buf[end..];
            }
        }

        for cs in &self.claimed_sums {
            cs.write_transcription_bytes_exact(&mut buf[..F::NUM_BYTES]);
            buf = &mut buf[F::NUM_BYTES..];
        }
    }
}

impl<F: ConstTranscribable> Transcribable for MultiDegreeSumcheckProof<F> {
    fn get_num_bytes(&self) -> usize {
        let num_groups = self.group_messages.len();
        let num_vars = self.group_messages[0].len();
        // total_evals = Σ_g (degree_g × num_vars)
        let total_evals: usize = self.degrees.iter().map(|&d| mul!(d, num_vars)).sum();

        // [num_groups][num_vars][deg₀..degₙ][evals...][claimed_sums]
        let header = add!(u32::NUM_BYTES, u32::NUM_BYTES);
        let degrees = mul!(num_groups, u32::NUM_BYTES);
        let eval_data = mul!(total_evals, F::NUM_BYTES);
        let claimed = mul!(num_groups, F::NUM_BYTES);

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

pub struct MultiDegreeSumcheck<C>(PhantomData<C>);

impl<C> MultiDegreeSumcheck<C>
where
    C: BaseFieldConfig + ProjectPrimitiveIntegersWithConfig,
    C::Integer: ConstTranscribable,
{
    /// Multi-degree sumcheck prover.
    ///
    /// Drives one or more **families** of multi-degree sumchecks in lockstep
    /// (interleaved), sharing one per-round verifier challenge across **all
    /// families and all groups within each family**. Each family carries
    /// its own degree groups in its own field config (i.e. a different
    /// prime); the shared challenge is sampled once per round as an integer
    /// in $[0, q^*)$ via `q_star_cfg`, then lifted into each family's field
    /// via `cfg.project` (a no-op type cast when $q^* \le q_i$).
    ///
    /// Proves, for every family $f$ and every group $g$ in that family:
    ///
    /// $$
    /// \sum_{x \in \{0, 1\}^{\text{num\\_vars}}} G_{f, g}(x) =
    /// \text{claimed\\_sum}_{f, g}
    /// $$
    ///
    /// where $G_{f, g}(x) = \text{comb\\_fn}_{f, g}(\text{mles}_{f, g}(x))$.
    ///
    /// Designed to be used as a subprotocol within a larger system: takes
    /// the FS transcript (`transcript`) as input and returns the **internal
    /// ProverState** alongside the sumcheck proof for every family. Claimed
    /// sums are derived by the prover during the first round.
    ///
    /// The single-family case (`families.len() == 1`) is the natural
    /// degenerate form; pass `q_star_cfg = &families[0].1` and the lift is
    /// the identity.
    ///
    /// # Arguments
    ///
    /// * `transcript`: Fiat-Shamir transcript.
    /// * `families`: One entry per family: `(groups, F::Config)`. Each family
    ///   contributes one or more degree groups that share that family's
    ///   `F::Config`.
    /// * `num_vars`: Number of variables (must be consistent across all groups
    ///   in all families).
    /// * `q_star_cfg`: Field configuration used for transcript metadata absorbs
    ///   and per-round challenge squeezes. This is the smallest of the
    ///   per-family moduli (so every family can losslessly cast the shared
    ///   integer into its own field).
    ///
    /// # Returns
    ///
    /// One `(proof, prover_states)` tuple per family, in the order families
    /// were provided.
    ///
    /// # Panics
    ///
    /// * If `num_vars == 0`.
    /// * If `families` is empty or any family has no groups.
    #[allow(clippy::type_complexity)]
    pub fn prove_as_subprotocol(
        transcript: &mut impl Transcript,
        families: Vec<(Vec<MultiDegreeSumcheckGroup<C>>, &C)>,
        num_vars: usize,
        q_star_cfg: &C,
    ) -> Vec<(
        MultiDegreeSumcheckProof<C::Element>,
        Vec<SumcheckProverState<C>>,
    )> {
        assert!(
            num_vars > 0,
            "Attempts to prove a constant: num_vars must be > 0"
        );
        assert!(!families.is_empty(), "need at least one family");
        for (groups, _) in &families {
            assert!(!groups.is_empty(), "every family needs at least one group");
        }

        let num_families = families.len();
        let mut buf = vec![0; <C::Integer as ConstTranscribable>::NUM_BYTES];

        // Metadata: absorb (num_vars, num_families) under `q_star_cfg` so the
        // transcript layout is canonical across families.
        let nvars_field = q_star_cfg.project(&(num_vars as u64));
        let nfamilies_field = q_star_cfg.project(&(num_families as u64));
        transcript.absorb_field_element(q_star_cfg, &nvars_field, &mut buf);
        transcript.absorb_field_element(q_star_cfg, &nfamilies_field, &mut buf);

        // Per-family state, one `Vec` per family.
        let mut per_family_group_messages: Vec<Vec<Vec<SumcheckProverMsg<C::Element>>>> =
            Vec::with_capacity(num_families);
        let mut per_family_claimed_sums: Vec<Vec<C::Element>> = Vec::with_capacity(num_families);
        let mut per_family_prover_states: Vec<Vec<SumcheckProverState<C>>> =
            Vec::with_capacity(num_families);
        let mut per_family_comb_fns: Vec<Vec<CombFn<C::Element>>> =
            Vec::with_capacity(num_families);
        let mut per_family_fast_paths: Vec<Vec<Option<Box<dyn Round1FastPath<C>>>>> =
            Vec::with_capacity(num_families);
        let mut per_family_cfg: Vec<&C> = Vec::with_capacity(num_families);

        for (groups, cfg) in families {
            let num_groups = groups.len();
            let ngroups_field = q_star_cfg.project(&(num_groups as u64));
            transcript.absorb_field_element(q_star_cfg, &ngroups_field, &mut buf);

            let group_messages: Vec<Vec<SumcheckProverMsg<C::Element>>> = (0..num_groups)
                .map(|_| Vec::with_capacity(num_vars))
                .collect();
            let mut prover_states: Vec<SumcheckProverState<C>> = Vec::with_capacity(num_groups);
            let mut comb_fns: Vec<CombFn<C::Element>> = Vec::with_capacity(num_groups);
            let mut fast_paths: Vec<Option<Box<dyn Round1FastPath<C>>>> =
                Vec::with_capacity(num_groups);

            for group in groups {
                let degree_field = q_star_cfg.project(&(group.degree as u64));
                transcript.absorb_field_element(q_star_cfg, &degree_field, &mut buf);

                prover_states.push(SumcheckProverState::new(group.poly, num_vars, group.degree));
                comb_fns.push(group.comb_fn);
                fast_paths.push(group.round_1_fast_path);
            }

            per_family_group_messages.push(group_messages);
            per_family_claimed_sums.push(Vec::with_capacity(num_groups));
            per_family_prover_states.push(prover_states);
            per_family_comb_fns.push(comb_fns);
            per_family_fast_paths.push(fast_paths);
            per_family_cfg.push(cfg);
        }

        // Per-family last challenge in each family's field. The underlying
        // integer is shared across families; each family lifts it locally.
        let mut per_family_verifier_msg: Vec<Option<C::Element>> = vec![None; num_families];

        for round in 1..=num_vars {
            // 1. Each family produces its round messages (per-family cfg).
            for b in 0..num_families {
                let cfg = per_family_cfg[b];
                let verifier_msg = &per_family_verifier_msg[b];
                let round_msgs: Vec<SumcheckProverMsg<C::Element>> =
                    cfg_iter_mut!(per_family_prover_states[b])
                        .zip(cfg_iter!(per_family_comb_fns[b]))
                        .zip(cfg_iter!(per_family_fast_paths[b]))
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
                    transcript.absorb_field_element_slice(cfg, &msg.0.tail_evaluations, &mut buf);
                }

                for (j, msg) in round_msgs.into_iter().enumerate() {
                    per_family_group_messages[b][j].push(msg);
                }
            }

            // 2. Sample one shared integer challenge in [0, q*) via q_star_cfg.
            let shared_chal_q_star: C::Element = transcript.get_field_challenge(q_star_cfg);
            transcript.absorb_field_element(q_star_cfg, &shared_chal_q_star, &mut buf);
            let shared_chal_int = q_star_cfg.lift(&shared_chal_q_star);

            // 3. Per family: lift the shared integer into its field and feed each group.
            //    Install fast-path post-fold MLEs on round 1.
            for b in 0..num_families {
                let chal_b = per_family_cfg[b].project(&shared_chal_int);
                if round == 1 {
                    per_family_prover_states[b]
                        .iter_mut()
                        .zip(per_family_fast_paths[b].iter_mut())
                        .for_each(|(state, fp_slot)| {
                            if let Some(fp) = fp_slot.take() {
                                state.mles = fp.fold_with_challenge(&chal_b, per_family_cfg[b]);
                                state.skip_next_fold = true;
                            }
                        });
                }
                per_family_verifier_msg[b] = Some(chal_b);
            }
        }

        // Finalize per family (mirrors the original single-family tail).
        cfg_into_iter!(per_family_prover_states)
            .zip(cfg_into_iter!(per_family_group_messages))
            .zip(cfg_into_iter!(per_family_claimed_sums))
            .zip(cfg_into_iter!(per_family_verifier_msg))
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
            .collect()
    }

    /// Multi-degree sumcheck verifier.
    ///
    /// Mirror of [`prove_as_subprotocol`]: drives one or more families of
    /// multi-degree sumchecks in lockstep, sharing one per-round challenge
    /// sampled in $[0, q^*)$ via `q_star_cfg` and lifted into each family's
    /// field. Verifies, for every family $f$ and every group $g$:
    ///
    /// $$
    /// \sum_{x \in \{0, 1\}^{\text{num\\_vars}}} G_{f, g}(x) =
    /// \text{claimed\\_sum}_{f, g}
    /// $$
    ///
    /// where $G_{f, g}(x) = \text{comb\\_fn}_{f, g}(\text{mles}_{f, g}(x))$.
    ///
    /// Returns one `MultiDegreeSubClaims<F>` per family: shared evaluation
    /// point `r*` (in that family's field) and per-group expected
    /// evaluations. The caller must verify each family's MLE combination at
    /// its `r*` equals its expected evaluation.
    ///
    /// The single-family case (`proofs.len() == 1`) is the natural degenerate
    /// form; pass `q_star_cfg = &proofs[0].1`.
    ///
    /// # Arguments
    ///
    /// * `transcript`: Fiat-Shamir transcript (must match prover state at the
    ///   start of the sumcheck).
    /// * `num_vars`: Number of variables (sumcheck rounds).
    /// * `proofs`: One `(proof, F::Config)` per family.
    /// * `q_star_cfg`: Field configuration used for transcript metadata absorbs
    ///   and per-round challenge squeezes (mirror of the prover).
    ///
    /// # Panics
    ///
    /// * If `num_vars == 0`.
    /// * If `proofs` is empty or any family's proof has no groups.
    pub fn verify_as_subprotocol(
        transcript: &mut impl Transcript,
        num_vars: usize,
        proofs: &[(&MultiDegreeSumcheckProof<C::Element>, &C)],
        q_star_cfg: &C,
    ) -> Result<Vec<MultiDegreeSubClaims<C::Element>>, SumCheckError<C::Element>> {
        assert!(
            num_vars > 0,
            "Attempts to prove a constant: num_vars must be > 0"
        );
        assert!(!proofs.is_empty(), "need at least one family");

        let num_families = proofs.len();
        let mut buf = vec![0; <C::Integer as ConstTranscribable>::NUM_BYTES];

        // Metadata: (num_vars, num_families) under q_star_cfg.
        let nvars_field = q_star_cfg.project(&(num_vars as u64));
        let nfamilies_field = q_star_cfg.project(&(num_families as u64));
        transcript.absorb_field_element(q_star_cfg, &nvars_field, &mut buf);
        transcript.absorb_field_element(q_star_cfg, &nfamilies_field, &mut buf);

        let mut per_family_verifier_states: Vec<Vec<VerifierState<C>>> =
            Vec::with_capacity(num_families);
        for (proof, cfg) in proofs {
            let num_groups = proof.degrees.len();
            assert!(num_groups != 0, "every family needs at least one group");
            let ngroups_field = q_star_cfg.project(&(num_groups as u64));
            transcript.absorb_field_element(q_star_cfg, &ngroups_field, &mut buf);

            let states: Vec<VerifierState<C>> = (0..num_groups)
                .map(|j| {
                    let degree = proof.degrees[j];
                    let degree_field = q_star_cfg.project(&(degree as u64));
                    transcript.absorb_field_element(q_star_cfg, &degree_field, &mut buf);
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

            per_family_verifier_states.push(states);
        }

        for i in 0..num_vars {
            // Absorb all families' messages in family-major order to match
            // the prover.
            for (proof, cfg) in proofs {
                proof.group_messages.iter().for_each(|msg| {
                    transcript.absorb_field_element_slice(
                        *cfg,
                        &msg[i].0.tail_evaluations,
                        &mut buf,
                    )
                });
            }

            // One shared integer challenge per round under q_star_cfg.
            let shared_chal_q_star: C::Element = transcript.get_field_challenge(q_star_cfg);
            transcript.absorb_field_element(q_star_cfg, &shared_chal_q_star, &mut buf);
            let shared_chal_int = q_star_cfg.lift(&shared_chal_q_star);

            for (b, (proof, cfg)) in proofs.iter().enumerate() {
                let chal_b = cfg.project(&shared_chal_int);
                per_family_verifier_states[b]
                    .iter_mut()
                    .zip(proof.group_messages.iter())
                    .for_each(|(state, msg)| {
                        state.verify_round_with_challenge(&msg[i], chal_b.clone())
                    });
            }
        }

        // TODO: parallelize when multiple lookup groups exist
        let mut output = Vec::with_capacity(num_families);
        for (b, (proof, _)) in proofs.iter().enumerate() {
            let states = std::mem::take(&mut per_family_verifier_states[b]);
            let mut shared_point: Option<Vec<C::Element>> = None;
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
#[allow(
    clippy::arithmetic_side_effects,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::redundant_clone
)]
mod tests {
    use super::*;
    use crypto_bigint::{U128, const_monty_params};
    use crypto_primitives::{FixedConfig, crypto_bigint_const_monty::ConstMontyField};
    use zinc_poly::utils::build_eq_x_r;
    use zinc_transcript::Blake3Transcript;

    const_monty_params!(TestParams, U128, "00000000b933426489189cb5b47d567f");
    type F = ConstMontyField<TestParams, { U128::LIMBS }>;
    type Cfg = FixedConfig<F>;

    /// Two degree groups sharing the same evaluation point.
    ///
    /// - Group 0 (degree 2): `eq(y, r) · (a(y) + b(y))`
    /// - Group 1 (degree 3): `eq(y, r) · a(y) · b(y)`
    #[test]
    fn multi_degree_two_groups() {
        let num_vars = 3;
        let cfg = &Cfg::default();

        let a_vals: Vec<F> = (0_u32..8).map(|i| F::from(i + 1)).collect();
        let b_vals: Vec<F> = (0_u32..8).map(|i| F::from(i + 10)).collect();

        let a_mle =
            DenseMultilinearExtension::from_evaluations_vec(num_vars, a_vals, F::from(0_u32));
        let b_mle =
            DenseMultilinearExtension::from_evaluations_vec(num_vars, b_vals, F::from(0_u32));

        let r: Vec<F> = vec![F::from(5_u32), F::from(7_u32), F::from(11_u32)];
        let eq_r = build_eq_x_r(cfg, &r).unwrap();

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

        // Prove (single-family shape: one family with `q_star_cfg = cfg`).
        let mut pt = Blake3Transcript::new();
        let mut outputs = MultiDegreeSumcheck::<Cfg>::prove_as_subprotocol(
            &mut pt,
            vec![(vec![g0, g1], cfg)],
            num_vars,
            cfg,
        );
        let (proof, _states) = outputs.pop().expect("single family");

        // Verify
        let mut vt = Blake3Transcript::new();
        let mut subclaims_vec = MultiDegreeSumcheck::<Cfg>::verify_as_subprotocol(
            &mut vt,
            num_vars,
            &[(&proof, cfg)],
            cfg,
        )
        .expect("verification should succeed");
        let subclaims = subclaims_vec.pop().expect("single family");

        assert_eq!(subclaims.expected_evaluations.len(), 2);

        // Check final evaluations manually
        let point = &subclaims.point;
        let eq_eval = zinc_poly::utils::eq_eval(cfg, point, &r).unwrap();
        let a_eval = a_mle.evaluate(cfg, point).unwrap();
        let b_eval = b_mle.evaluate(cfg, point).unwrap();

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
        let cfg = &Cfg::default();

        let vals: Vec<F> = (0_u32..4).map(|i| F::from(i + 1)).collect();
        let mle = DenseMultilinearExtension::from_evaluations_vec(num_vars, vals, F::from(0_u32));

        let r: Vec<F> = vec![F::from(3_u32), F::from(7_u32)];
        let eq_r = build_eq_x_r(cfg, &r).unwrap();

        let g = MultiDegreeSumcheckGroup::new(
            2,
            vec![eq_r.clone(), mle.clone()],
            Box::new(|v: &[F]| v[0] * v[1]),
        );

        let mut pt = Blake3Transcript::new();
        let mut outputs = MultiDegreeSumcheck::<Cfg>::prove_as_subprotocol(
            &mut pt,
            vec![(vec![g], cfg)],
            num_vars,
            cfg,
        );
        let (proof, _) = outputs.pop().expect("single family");

        let mut vt = Blake3Transcript::new();
        let mut subclaims_vec = MultiDegreeSumcheck::<Cfg>::verify_as_subprotocol(
            &mut vt,
            num_vars,
            &[(&proof, cfg)],
            cfg,
        )
        .expect("verification should succeed");
        let subclaims = subclaims_vec.pop().expect("single family");

        let point = &subclaims.point;
        let eq_eval = zinc_poly::utils::eq_eval(cfg, point, &r).unwrap();
        let a_eval = mle.clone().evaluate(cfg, point).unwrap();

        assert_eq!(subclaims.expected_evaluations[0], eq_eval * a_eval);
    }

    /// Two families sharing the same `F::Config` (i.e. $q^* = q_i$ for every
    /// family): every family must accept the shared per-round challenge in
    /// its field, and the resulting subclaims must verify per family.
    ///
    /// Exercises the multi-family behavior of the unified API (one shared
    /// integer challenge per round, lifted into each family).
    #[test]
    fn two_families_same_cfg() {
        let num_vars = 3;
        let cfg = &Cfg::default();
        let q_star_cfg = cfg;

        // Group A: degree-2 group on (a, b).
        let a_vals: Vec<F> = (0_u32..8).map(|i| F::from(i + 1)).collect();
        let b_vals: Vec<F> = (0_u32..8).map(|i| F::from(i + 10)).collect();
        let a_mle =
            DenseMultilinearExtension::from_evaluations_vec(num_vars, a_vals, F::from(0_u32));
        let b_mle =
            DenseMultilinearExtension::from_evaluations_vec(num_vars, b_vals, F::from(0_u32));
        let r_a: Vec<F> = vec![F::from(5_u32), F::from(7_u32), F::from(11_u32)];
        let eq_r_a = build_eq_x_r(cfg, &r_a).unwrap();
        let group_a = MultiDegreeSumcheckGroup::new(
            2,
            vec![eq_r_a.clone(), a_mle.clone(), b_mle.clone()],
            Box::new(|v: &[F]| v[0] * (v[1] + v[2])),
        );

        // Group B: degree-3 group on (a, b), different evaluation point.
        let r_b: Vec<F> = vec![F::from(2_u32), F::from(3_u32), F::from(5_u32)];
        let eq_r_b = build_eq_x_r(cfg, &r_b).unwrap();
        let group_b = MultiDegreeSumcheckGroup::new(
            3,
            vec![eq_r_b.clone(), a_mle.clone(), b_mle.clone()],
            Box::new(|v: &[F]| v[0] * v[1] * v[2]),
        );

        // Prover.
        let mut pt = Blake3Transcript::new();
        let mut outputs = MultiDegreeSumcheck::<Cfg>::prove_as_subprotocol(
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
        let subclaims = MultiDegreeSumcheck::<Cfg>::verify_as_subprotocol(
            &mut vt,
            num_vars,
            &[(&proof_a, cfg), (&proof_b, cfg)],
            q_star_cfg,
        )
        .expect("verification should succeed");
        assert_eq!(subclaims.len(), 2);

        // Both families must land at the same shared point (same cfg -> same lift).
        assert_eq!(subclaims[0].point(), subclaims[1].point());

        // Per-family subclaim checks against the polynomial identity.
        let point = subclaims[0].point();
        let eq_a_eval = zinc_poly::utils::eq_eval(cfg, point, &r_a).unwrap();
        let eq_b_eval = zinc_poly::utils::eq_eval(cfg, point, &r_b).unwrap();
        let a_eval = a_mle.evaluate(cfg, point).unwrap();
        let b_eval = b_mle.evaluate(cfg, point).unwrap();

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
