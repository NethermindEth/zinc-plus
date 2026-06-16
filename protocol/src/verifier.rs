use super::*;
use crypto_primitives::{ConstIntSemiring, FromPrimitiveWithConfig, FromWithConfig};
use itertools::Itertools;
use std::io::Cursor;
use zinc_piop::{
    combined_poly_resolver::CombinedPolyResolver,
    ideal_check::{self, IdealCheckProtocol},
    lookup::booleanity::{BooleanityChecker, BooleanityProof},
    multipoint_eval::{self, MultipointEval},
    projections::{
        ProjectedScalars, ProjectedTrace, project_scalars, project_scalars_to_field,
        project_trace_coeffs_row_major,
    },
    sumcheck::multi_degree::MultiDegreeSumcheck,
};
use zinc_poly::{EvaluatablePolynomial, univariate::dynamic::over_field::DynamicPolynomialF};
use zinc_transcript::{
    Blake3Transcript,
    traits::{ConstTranscribable, Transcript},
};
use zinc_uair::{
    Uair, UairSignature, UairTrace,
    constraint_counter::count_constraints,
    ideal::{Ideal, IdealCheck},
    ideal_collector::IdealOrZero,
};
use zinc_utils::{
    add, from_ref::FromRef, inner_transparent_field::InnerTransparentField,
    mul_by_scalar::MulByScalar, projectable_to_field::ProjectableToField,
};
use zip_plus::{
    pcs::structs::{ZipPlus, ZipPlusParams, ZipTypes},
    pcs_transcript::PcsVerifierTranscript,
};

//
// Per-prime F_q[X] branch helpers
//

// FIXME

//
// Shared base
//

/// Persistent verifier infrastructure carried across every step.
#[derive(Clone, Debug)]
pub struct VerifierBase<'a, Zt: ZincTypes<D, FD>, const D: usize, const FD: usize> {
    num_vars: usize,
    uair_signature: UairSignature,
    pcs_transcript: PcsVerifierTranscript,
    public_trace: &'a UairTrace<'a, Zt::Int, Zt::Int, D, D>,

    // Commitment info
    vp_bin: &'a ZipPlusParams<Zt::BinaryZt, Zt::BinaryLc>,
    vp_arb: &'a ZipPlusParams<Zt::ArbitraryZt, Zt::ArbitraryLc>,
    vp_int: &'a ZipPlusParams<Zt::IntZt, Zt::IntLc>,
}

//
// Type-state structs
//

/// After step 0 (transcript reconstruction).
#[derive(Clone, Debug)]
pub struct VerifierTranscriptReconstructed<
    'a,
    Zt: ZincTypes<D, FD>,
    U: Uair,
    F: PrimeField,
    IdealOverF,
    const D: usize,
    const FD: usize,
> {
    base: VerifierBase<'a, Zt, D, FD>,

    // Proof leftovers
    proof_commitments: (ZipPlusCommitment, ZipPlusCommitment, ZipPlusCommitment),
    proof_ideal_check: IdealCheckProof<F>,
    proof_cpr: CombinedPolyResolverProof<F>,
    proof_combined_sumcheck: MultiDegreeSumcheckProof<F>,
    proof_multipoint_eval: MultipointEvalProof<F>,
    proof_witness_lifted_evals: Vec<DynamicPolynomialF<F>>,
    proof_lookup_proof: Option<BatchedLookupProof<F>>,
    proof_booleanity: Option<BooleanityProof<F>>,
    proof_ideal_checks_fq: Vec<IdealCheckProof<F>>,
    proof_cpr_fq: Vec<CombinedPolyResolverProof<F>>,
    proof_combined_sumchecks_fq: Vec<MultiDegreeSumcheckProof<F>>,
    _phantom: PhantomData<(U, IdealOverF)>,
}

/// After step 1 (prime projection).
#[derive(Clone, Debug)]
pub struct VerifierPrimeProjected<
    'a,
    Zt: ZincTypes<D, FD>,
    U: Uair,
    F: PrimeField,
    IdealOverF,
    const D: usize,
    const FD: usize,
> {
    base: VerifierBase<'a, Zt, D, FD>,
    field_cfg: F::Config,

    // Proof leftovers
    proof_commitments: (ZipPlusCommitment, ZipPlusCommitment, ZipPlusCommitment),
    proof_ideal_check: IdealCheckProof<F>,
    proof_cpr: CombinedPolyResolverProof<F>,
    proof_combined_sumcheck: MultiDegreeSumcheckProof<F>,
    proof_multipoint_eval: MultipointEvalProof<F>,
    proof_witness_lifted_evals: Vec<DynamicPolynomialF<F>>,
    proof_lookup_proof: Option<BatchedLookupProof<F>>,
    proof_booleanity: Option<BooleanityProof<F>>,
    proof_ideal_checks_fq: Vec<IdealCheckProof<F>>,
    proof_cpr_fq: Vec<CombinedPolyResolverProof<F>>,
    proof_combined_sumchecks_fq: Vec<MultiDegreeSumcheckProof<F>>,
    _phantom: PhantomData<(U, IdealOverF)>,
}

/// After step 2 (ideal check). `project_ideal` has been consumed.
#[derive(Clone, Debug)]
pub struct VerifierIdealChecked<
    'a,
    Zt: ZincTypes<D, FD>,
    U: Uair,
    F: PrimeField,
    IdealOverF,
    const D: usize,
    const FD: usize,
> {
    base: VerifierBase<'a, Zt, D, FD>,
    field_cfg: F::Config,
    /// Per-branch field configs (`[0]` = $Q[X]$, `[i >= 1]` =
    /// $F_{q_{i-1}}[X]$), kept for the next step's shared `fq-unify` $\psi$
    /// projecting element.
    all_field_cfgs: Vec<F::Config>,
    /// Index of $q^*$ in `all_field_cfgs`, computed once in step 2 and
    /// threaded forward so step 3 can recover `q_star_cfg` by indexing.
    q_star_idx: usize,
    /// Per-branch IC subclaims (length `n + 1`). `[0]` is the Q[X]
    /// branch's subclaim; `[i + 1]` is the per-prime $F_{q_i}[X]$
    /// subclaim. Previously only the Q[X] subclaim was kept; the per-prime
    /// subclaims were squeezed and discarded. They are now retained so
    /// step 4 (CPR verify) can drive a per-prime `prepare_verifier` call
    /// for each fq branch (Phase F.2.c).
    ic_subclaims: Vec<ideal_check::VerifierSubclaim<F>>,

    // Proof leftovers
    proof_commitments: (ZipPlusCommitment, ZipPlusCommitment, ZipPlusCommitment),
    proof_cpr: CombinedPolyResolverProof<F>,
    proof_combined_sumcheck: MultiDegreeSumcheckProof<F>,
    proof_multipoint_eval: MultipointEvalProof<F>,
    proof_witness_lifted_evals: Vec<DynamicPolynomialF<F>>,
    proof_lookup_proof: Option<BatchedLookupProof<F>>,
    proof_booleanity: Option<BooleanityProof<F>>,
    /// Per-prime CPR proofs (one per declared prime).
    proof_cpr_fq: Vec<CombinedPolyResolverProof<F>>,
    /// Per-prime multi-degree sumcheck proofs (one per declared prime).
    proof_combined_sumchecks_fq: Vec<MultiDegreeSumcheckProof<F>>,
    _phantom: PhantomData<(U, IdealOverF)>,
}

/// After step 3 (eval projection). `project_scalar` has been consumed.
#[derive(Clone, Debug)]
pub struct VerifierEvalProjected<
    'a,
    Zt: ZincTypes<D, FD>,
    U: Uair,
    F: PrimeField,
    IdealOverF,
    const D: usize,
    const FD: usize,
> {
    base: VerifierBase<'a, Zt, D, FD>,
    field_cfg: F::Config,
    /// Per-branch field configs (`[0]` = $Q[X]$, `[i >= 1]` =
    /// $F_{q_{i-1}}[X]$), kept for later per-prime CPR/MP/PCS phases.
    #[allow(dead_code)] // Consumed by Phase F.2.c in step4_sumcheck_verify.
    all_field_cfgs: Vec<F::Config>,
    /// Index of $q^*$ in `all_field_cfgs`, kept for later phases that need
    /// to re-sample shared challenges in $[0, q^*)$.
    #[allow(dead_code)] // Consumed by Phase F.2.c in step4_sumcheck_verify.
    q_star_idx: usize,
    /// Per-branch IC subclaims (length `n + 1`). `[0]` feeds the Q[X] CPR
    /// `prepare_verifier`; `[i + 1]` will feed each per-prime CPR
    /// `prepare_verifier` in Phase F.2.c.
    ic_subclaims: Vec<ideal_check::VerifierSubclaim<F>>,
    /// Per-branch $\psi$-projecting elements: integer sampled mod $q^*$
    /// and projected onto each of `all_field_cfgs`.
    projecting_elements: Vec<F>,
    /// Q[X] branch's $\psi$-projected scalars.
    projected_scalars_f: ProjectedScalars<U::Scalar, F>,
    /// Per-prime $\psi$-projected scalars (one per declared prime). Built
    /// in step 3 from each branch's `projecting_elements[i + 1]` and the
    /// UAIR-author-supplied `project_scalar` closure (applied with
    /// `all_field_cfgs[i + 1]`). Consumed in step 4 (CPR finalize per
    /// branch).
    #[allow(dead_code)] // Consumed by Phase F.2.c.
    projected_scalars_f_fq: Vec<ProjectedScalars<U::Scalar, F>>,

    // Proof leftovers
    proof_commitments: (ZipPlusCommitment, ZipPlusCommitment, ZipPlusCommitment),
    proof_cpr: CombinedPolyResolverProof<F>,
    proof_combined_sumcheck: MultiDegreeSumcheckProof<F>,
    proof_multipoint_eval: MultipointEvalProof<F>,
    proof_witness_lifted_evals: Vec<DynamicPolynomialF<F>>,
    proof_lookup_proof: Option<BatchedLookupProof<F>>,
    proof_booleanity: Option<BooleanityProof<F>>,
    /// Per-prime CPR proofs (one per declared prime), consumed in step 4.
    proof_cpr_fq: Vec<CombinedPolyResolverProof<F>>,
    /// Per-prime multi-degree sumcheck proofs (one per declared prime),
    /// consumed in step 4 by the lockstep verifier driver.
    proof_combined_sumchecks_fq: Vec<MultiDegreeSumcheckProof<F>>,
    _phantom: PhantomData<(U, IdealOverF)>,
}

/// After step 4 (sumcheck verify).
#[derive(Clone, Debug)]
pub struct VerifierSumchecked<
    'a,
    Zt: ZincTypes<D, FD>,
    F: PrimeField,
    IdealOverF,
    const D: usize,
    const FD: usize,
> {
    base: VerifierBase<'a, Zt, D, FD>,
    field_cfg: F::Config,
    /// Per-branch field configs (carried for later `fq-unify` phases).
    #[allow(dead_code)] // Used by later `fq-unify` phases.
    all_field_cfgs: Vec<F::Config>,
    /// Index of $q^*$ in `all_field_cfgs` (carried for later phases).
    #[allow(dead_code)] // Used by later `fq-unify` phases.
    q_star_idx: usize,
    /// Per-branch $\psi$-projecting elements: integer sampled mod $q^*$
    /// and projected onto each of `all_field_cfgs`.
    projecting_elements: Vec<F>,
    /// Per-branch CPR batching challenges $\alpha$: `[0]` was consumed by
    /// the Q[X] CPR verifier; `[i >= 1]` will drive the per-prime CPRs in
    /// Phase F+.
    #[allow(dead_code)] // Used by later `fq-unify` phases.
    folding_challenges: Vec<F>,
    /// CPR subclaim's evaluation point ($r^\star$)
    cpr_eval_point: Vec<F>,
    cpr_up_evals: Vec<F>,
    cpr_down_evals: Vec<F>,
    /// `bit_slice_evals` carried over from booleanity's `finalize_verifier`,
    /// to be collapsed into the appended `up_evals` entries
    /// $c_j = \sum_i b_{j,i}\,(\alpha')^i$.
    ///
    /// `None` iff there are no witness binary-poly columns.
    bool_bit_slice_evals: Option<Vec<F>>,
    /// Fresh challenge sampled after `bit_slice_evals` were absorbed by
    /// booleanity's `finalize_verifier`. Consumed in step 5 (bridge
    /// scalars) and step 6 (appended $\alpha'$-projected open_evals).
    ///
    /// `None` iff there are no witness binary-poly columns.
    alpha_prime_f: Option<F>,

    // Proof leftovers
    proof_commitments: (ZipPlusCommitment, ZipPlusCommitment, ZipPlusCommitment),
    proof_multipoint_eval: MultipointEvalProof<F>,
    proof_witness_lifted_evals: Vec<DynamicPolynomialF<F>>,
    proof_lookup_proof: Option<BatchedLookupProof<F>>,
    _phantom: PhantomData<IdealOverF>,
}

/// After step 5 (multi-point eval).
#[derive(Clone, Debug)]
pub struct VerifierMultipointEvaled<
    'a,
    Zt: ZincTypes<D, FD>,
    F: PrimeField,
    IdealOverF,
    const D: usize,
    const FD: usize,
> {
    base: VerifierBase<'a, Zt, D, FD>,
    field_cfg: F::Config,
    /// Per-branch $\psi$-projecting elements: integer sampled mod $q^*$
    /// and projected onto each of `all_field_cfgs`.
    projecting_elements: Vec<F>,
    // See VerifierSumchecked::alpha_prime_f
    alpha_prime_f: Option<F>,
    mp_subclaim: multipoint_eval::Subclaim<F>,

    // Proof leftovers
    proof_commitments: (ZipPlusCommitment, ZipPlusCommitment, ZipPlusCommitment),
    proof_witness_lifted_evals: Vec<DynamicPolynomialF<F>>,
    proof_lookup_proof: Option<BatchedLookupProof<F>>,
    _phantom: PhantomData<IdealOverF>,
}

/// After step 6 (lifted evals verification).
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct VerifierLiftedEvalsChecked<
    'a,
    Zt: ZincTypes<D, FD>,
    F: PrimeField,
    IdealOverF,
    const D: usize,
    const FD: usize,
> {
    base: VerifierBase<'a, Zt, D, FD>,
    field_cfg: F::Config,
    mp_subclaim: multipoint_eval::Subclaim<F>,
    all_lifted_evals: Vec<DynamicPolynomialF<F>>,

    // Proof leftovers
    proof_commitments: (ZipPlusCommitment, ZipPlusCommitment, ZipPlusCommitment),
    proof_lookup_proof: Option<BatchedLookupProof<F>>,
    _phantom: PhantomData<IdealOverF>,
}

/// After step 7 (PCS verify). Ready for
/// [`finish`](VerifierPcsVerified::finish).
#[derive(Clone, Debug)]
pub struct VerifierPcsVerified<IdealOverF> {
    _phantom: PhantomData<IdealOverF>,
}

//
// Step implementations
//

impl<Zt, U, F, const D: usize, const FD: usize> ZincPlusPiop<Zt, U, F, D, FD>
where
    Zt: ZincTypes<D, FD>,
    U: Uair,
    F: PrimeField,
    F::Integer: ConstTranscribable,
{
    /// Step 0: Verifier entry point.
    /// Reconstruct Fiat-Shamir transcript from commitments and public data.
    #[allow(clippy::type_complexity)]
    pub fn step0_reconstruct_transcript<'a, IdealOverF>(
        (vp_bin, vp_arb, vp_int): &'a (
            ZipPlusParams<Zt::BinaryZt, Zt::BinaryLc>,
            ZipPlusParams<Zt::ArbitraryZt, Zt::ArbitraryLc>,
            ZipPlusParams<Zt::IntZt, Zt::IntLc>,
        ),
        mut proof: Proof<F>,
        public_trace: &'a UairTrace<'a, Zt::Int, Zt::Int, D, D>,
        num_vars: usize,
    ) -> Result<VerifierTranscriptReconstructed<'a, Zt, U, F, IdealOverF, D, FD>, ProtocolError<F>>
    where
        IdealOverF: Ideal,
    {
        assert!(
            num_vars > 0,
            "Attempt to verify a constant: num_vars must be > 0"
        );
        let zip_proof = std::mem::take(&mut proof.zip);
        let uair_signature = U::signature();
        let mut base = VerifierBase {
            num_vars,
            uair_signature,
            public_trace,
            pcs_transcript: PcsVerifierTranscript {
                fs_transcript: Blake3Transcript::default(),
                stream: Cursor::new(zip_proof),
            },
            vp_bin,
            vp_arb,
            vp_int,
        };

        for comm in [
            &proof.commitments.0,
            &proof.commitments.1,
            &proof.commitments.2,
        ] {
            base.pcs_transcript.fs_transcript.absorb_slice(&comm.root);
        }

        absorb_public_columns(
            &mut base.pcs_transcript.fs_transcript,
            &base.public_trace.binary_poly,
        );
        absorb_public_columns(
            &mut base.pcs_transcript.fs_transcript,
            &base.public_trace.arbitrary_poly,
        );
        absorb_public_columns(
            &mut base.pcs_transcript.fs_transcript,
            &base.public_trace.int,
        );

        Ok(VerifierTranscriptReconstructed {
            base,
            proof_commitments: proof.commitments,
            proof_ideal_check: proof.ideal_check,
            proof_cpr: proof.cpr_proof,
            proof_combined_sumcheck: proof.combined_sumcheck,
            proof_multipoint_eval: proof.multipoint_eval,
            proof_witness_lifted_evals: proof.witness_lifted_evals,
            proof_lookup_proof: proof.lookup_proof,
            proof_booleanity: proof.booleanity_proof,
            proof_ideal_checks_fq: proof.ideal_checks_fq,
            proof_cpr_fq: proof.cpr_proofs_fq,
            proof_combined_sumchecks_fq: proof.combined_sumchecks_fq,
            _phantom: PhantomData,
        })
    }
}

impl<'a, Zt, U, F, IdealOverF, const D: usize, const FD: usize>
    VerifierTranscriptReconstructed<'a, Zt, U, F, IdealOverF, D, FD>
where
    Zt: ZincTypes<D, FD>,
    F: InnerTransparentField + FromPrimitiveWithConfig + FromRef<F> + Send + Sync + 'static,
    F::Integer: ConstIntSemiring + ConstTranscribable + Send + Sync + FromRef<Zt::Fmod>,
    U: Uair,
    IdealOverF: Ideal,
{
    /// Step 1: Prime projection. Samples the random field configuration.
    #[allow(clippy::type_complexity)]
    pub fn step1_prime_projection(
        mut self,
    ) -> Result<VerifierPrimeProjected<'a, Zt, U, F, IdealOverF, D, FD>, ProtocolError<F>> {
        let field_cfg = self
            .base
            .pcs_transcript
            .fs_transcript
            .get_random_field_cfg::<F, Zt::Fmod, Zt::PrimeTest>();

        Ok(VerifierPrimeProjected {
            base: self.base,
            field_cfg,
            proof_commitments: self.proof_commitments,
            proof_ideal_check: self.proof_ideal_check,
            proof_cpr: self.proof_cpr,
            proof_combined_sumcheck: self.proof_combined_sumcheck,
            proof_multipoint_eval: self.proof_multipoint_eval,
            proof_witness_lifted_evals: self.proof_witness_lifted_evals,
            proof_lookup_proof: self.proof_lookup_proof,
            proof_booleanity: self.proof_booleanity,
            proof_ideal_checks_fq: self.proof_ideal_checks_fq,
            proof_cpr_fq: self.proof_cpr_fq,
            proof_combined_sumchecks_fq: self.proof_combined_sumchecks_fq,
            _phantom: PhantomData,
        })
    }
}

impl<'a, Zt, U, F, IdealOverF, const D: usize, const FD: usize>
    VerifierPrimeProjected<'a, Zt, U, F, IdealOverF, D, FD>
where
    Zt: ZincTypes<D, FD>,
    Zt::Int: ProjectableToField<F>,
    <Zt::ArbitraryZt as ZipTypes>::Eval: ProjectableToField<F>,
    F: InnerTransparentField
        + FromPrimitiveWithConfig
        + for<'b> FromWithConfig<&'b Zt::Int>
        + for<'b> FromWithConfig<&'b Zt::CombR>
        + for<'b> FromWithConfig<&'b Zt::Chal>
        + for<'b> MulByScalar<&'b F>
        + FromRef<F>
        + Send
        + Sync
        + 'static,
    F::Integer:
        ConstIntSemiring + ConstTranscribable + Send + Sync + FromRef<Zt::Fmod> + FromRef<u64>,
    U: Uair + 'static,
    IdealOverF: Ideal + IdealCheck<DynamicPolynomialF<F>>,
{
    /// Step 2: Ideal check verification.
    ///
    /// `project_fq_ideal` is only invoked when the UAIR declares at least
    /// one prime; legacy UAIRs can pass `|_, _| unreachable!()`.
    ///
    /// **`fq-unify` evaluation point.** Mirrors the prover: sample one
    /// shared $\mathbf r \in [0, q^*)^\mu$ from the transcript and lift it
    /// into each branch's field via `F::from_with_cfg`, then run the
    /// per-branch verifications in `branch_idx` order.
    ///
    /// TODO(fq-soundness): the per-prime ideal-check subclaims produced
    /// here are *discarded* -- this only verifies that each combined
    /// polynomial $e_{i,t}$ lies in its claimed ideal, not that
    /// $e_{i,t}$ was correctly derived from the committed trace. The
    /// per-prime CPR/sumcheck/multipoint-eval/PCS-open chain that closes
    /// the soundness loop is NYI; see `Proof::fq_ideal_checks` doc.
    #[allow(clippy::type_complexity)]
    pub fn step2_ideal_check(
        mut self,
        project_ideal: impl Fn(&IdealOrZero<U::Ideal>, &F::Config) -> IdealOverF,
        project_fq_ideal: impl Fn(&IdealOrZero<U::FqIdeal>, &F::Config) -> IdealOverF,
    ) -> Result<VerifierIdealChecked<'a, Zt, U, F, IdealOverF, D, FD>, ProtocolError<F>> {
        // The Q[X]-branch ideal check only consumes Q[X] constraints; F_q[X]
        // constraints are handled by the per-prime branch below.
        let num_constraints = count_constraints::<U>();
        let primes = self.base.uair_signature.primes().to_vec();

        if primes.len() != self.proof_ideal_checks_fq.len() {
            // Honest prover always emits one proof per declared prime
            return Err(ProtocolError::FqIdealCheck {
                prime_idx: self.proof_ideal_checks_fq.len(),
                q: "<Length mismatch>".to_owned(),
                source: IdealCheckError::IdealCollectorError(
                    ideal_check::BatchedIdealCheckError::LengthMismatch {
                        num_ideals: primes.len(),
                        provided_values: self.proof_ideal_checks_fq.len(),
                    },
                ),
            });
        }

        // `fq-unify`: rebuild per-branch field configs, then sample the
        // shared evaluation point from the transcript exactly as the
        // prover does. Branch 0 = Q[X] (random sampled prime); branches
        // i >= 1 = declared primes in `primes()` order.
        let all_field_cfgs = build_all_cfgs::<F>(&self.base.uair_signature, self.field_cfg.clone());
        let q_star_idx = shared_challenge::compute_q_star_idx::<F>(&all_field_cfgs);
        let q_star_cfg = &all_field_cfgs[q_star_idx];
        let shared_eval_points: Vec<Vec<F>> = shared_challenge::sample_shared_field_challenges::<F>(
            &mut self.base.pcs_transcript.fs_transcript,
            self.base.num_vars,
            q_star_cfg,
            &all_field_cfgs,
        );

        let mut ic_subclaims: Vec<ideal_check::VerifierSubclaim<F>> =
            Vec::with_capacity(all_field_cfgs.len());
        let q_subclaim = IdealCheckProtocol::<U>::verify_as_subprotocol::<_, IdealOverF, _, _>(
            &mut self.base.pcs_transcript.fs_transcript,
            self.proof_ideal_check,
            /* branch_idx = */ 0,
            num_constraints.q,
            &shared_eval_points[0],
            |ideal| project_ideal(ideal, &self.field_cfg),
            |_| unreachable!("Q[X] branch"),
            &self.field_cfg,
        )?;
        ic_subclaims.push(q_subclaim);

        // Per-prime F_q[X] ideal-check verifications. The transcript
        // ordering MUST match the prover:
        // Q[X] first, then per-prime in `primes()` order. Subclaims are
        // collected and threaded forward to Phase F.2.c (per-prime CPR
        // `prepare_verifier`).
        for (prime_idx, (cfg_q_i, fq_proof)) in all_field_cfgs[1..]
            .iter()
            .zip(self.proof_ideal_checks_fq)
            .enumerate()
        {
            let branch_idx = add!(prime_idx, 1);
            let fq_subclaim =
                IdealCheckProtocol::<U>::verify_as_subprotocol::<_, IdealOverF, _, _>(
                    &mut self.base.pcs_transcript.fs_transcript,
                    fq_proof,
                    branch_idx,
                    num_constraints.for_prime(prime_idx),
                    &shared_eval_points[branch_idx],
                    |_| unreachable!("F_q[X] branch"),
                    // The lifted F_q[X] ideal's coefficient modulus must match
                    // the per-prime field config
                    |ideal| project_fq_ideal(ideal, cfg_q_i),
                    cfg_q_i,
                )
                .map_err(|source| ProtocolError::FqIdealCheck {
                    prime_idx,
                    q: F::modulus(cfg_q_i).to_string(),
                    source,
                })?;
            ic_subclaims.push(fq_subclaim);
        }

        Ok(VerifierIdealChecked {
            base: self.base,
            field_cfg: self.field_cfg,
            all_field_cfgs,
            q_star_idx,
            ic_subclaims,
            proof_commitments: self.proof_commitments,
            proof_cpr: self.proof_cpr,
            proof_combined_sumcheck: self.proof_combined_sumcheck,
            proof_multipoint_eval: self.proof_multipoint_eval,
            proof_witness_lifted_evals: self.proof_witness_lifted_evals,
            proof_lookup_proof: self.proof_lookup_proof,
            proof_booleanity: self.proof_booleanity,
            proof_cpr_fq: self.proof_cpr_fq,
            proof_combined_sumchecks_fq: self.proof_combined_sumchecks_fq,
            _phantom: PhantomData,
        })
    }
}

impl<'a, Zt, U, F, IdealOverF, const D: usize, const FD: usize>
    VerifierIdealChecked<'a, Zt, U, F, IdealOverF, D, FD>
where
    Zt: ZincTypes<D, FD>,
    F: InnerTransparentField
        + for<'b> FromWithConfig<&'b Zt::Chal>
        + FromRef<F>
        + Send
        + Sync
        + 'static,
    F::Integer: ConstIntSemiring + ConstTranscribable + Send + Sync + FromRef<Zt::Fmod>,
    U: Uair + 'static,
    IdealOverF: Ideal,
{
    /// Step 3: Evaluation projection. Consumes `project_scalar`.
    ///
    /// **`fq-unify` projecting element.** Mirrors the prover: sample a
    /// shared integer $a \in [0, q^*)$ and lift it into each branch's
    /// field. The $Q[X]$ branch consumes `projecting_elements[0]`;
    /// per-prime branches consume `projecting_elements[i + 1]`
    /// (Phase F.2.c).
    ///
    /// Also builds per-prime $\psi$-projected scalars from
    /// `project_scalar(., all_field_cfgs[i + 1])` and threads them
    /// forward as `projected_scalars_f_fq` for the per-prime CPR
    /// `finalize_verifier` in step 4.
    pub fn step3_eval_projection(
        mut self,
        project_scalar: impl Fn(&U::Scalar, &F::Config) -> DynamicPolynomialF<F>,
    ) -> Result<VerifierEvalProjected<'a, Zt, U, F, IdealOverF, D, FD>, ProtocolError<F>> {
        let q_star_cfg = &self.all_field_cfgs[self.q_star_idx];
        let projecting_elements: Vec<F> = shared_challenge::sample_shared_field_challenge::<F>(
            &mut self.base.pcs_transcript.fs_transcript,
            q_star_cfg,
            &self.all_field_cfgs,
        );

        // Q[X] branch.
        let projected_scalars_fx = project_scalars::<F, U>(|s| project_scalar(s, &self.field_cfg));
        let projected_scalars_f =
            project_scalars_to_field(projected_scalars_fx, &projecting_elements[0])
                .map_err(|(_s, _f, e)| ProtocolError::ScalarProjection(e))?;

        // Per-prime $F_{q_i}[X]$ branches.
        let n_fq = self.all_field_cfgs.len().saturating_sub(1);
        let mut projected_scalars_f_fq: Vec<ProjectedScalars<U::Scalar, F>> =
            Vec::with_capacity(n_fq);
        for prime_idx in 0..n_fq {
            let branch_idx = add!(prime_idx, 1);
            let cfg_i = &self.all_field_cfgs[branch_idx];
            let projected_scalars_fx_i =
                project_scalars::<F, U>(|s| project_scalar(s, cfg_i));
            let projected_scalars_f_i =
                project_scalars_to_field(projected_scalars_fx_i, &projecting_elements[branch_idx])
                    .map_err(|(_s, _f, e)| ProtocolError::ScalarProjection(e))?;
            projected_scalars_f_fq.push(projected_scalars_f_i);
        }

        Ok(VerifierEvalProjected {
            base: self.base,
            field_cfg: self.field_cfg,
            all_field_cfgs: self.all_field_cfgs,
            q_star_idx: self.q_star_idx,
            ic_subclaims: self.ic_subclaims,
            projecting_elements,
            projected_scalars_f,
            projected_scalars_f_fq,
            proof_commitments: self.proof_commitments,
            proof_cpr: self.proof_cpr,
            proof_combined_sumcheck: self.proof_combined_sumcheck,
            proof_multipoint_eval: self.proof_multipoint_eval,
            proof_witness_lifted_evals: self.proof_witness_lifted_evals,
            proof_lookup_proof: self.proof_lookup_proof,
            proof_booleanity: self.proof_booleanity,
            proof_cpr_fq: self.proof_cpr_fq,
            proof_combined_sumchecks_fq: self.proof_combined_sumchecks_fq,
            _phantom: PhantomData,
        })
    }
}

impl<'a, Zt, U, F, IdealOverF, const D: usize, const FD: usize>
    VerifierEvalProjected<'a, Zt, U, F, IdealOverF, D, FD>
where
    Zt: ZincTypes<D, FD>,
    Zt::Int: ProjectableToField<F>,
    <Zt::ArbitraryZt as ZipTypes>::Eval: ProjectableToField<F>,
    F: InnerTransparentField
        + FromPrimitiveWithConfig
        + for<'b> FromWithConfig<&'b Zt::Int>
        + for<'b> FromWithConfig<&'b Zt::CombR>
        + for<'b> FromWithConfig<&'b Zt::Chal>
        + for<'b> MulByScalar<&'b F>
        + FromRef<F>
        + Send
        + Sync
        + 'static,
    F::Integer: ConstIntSemiring + ConstTranscribable + Send + Sync + FromRef<Zt::Fmod>,
    U: Uair + 'static,
    IdealOverF: Ideal,
{
    /// Step 4: Sumcheck verification (CPR + optional booleanity +
    /// future lookup groups), followed by the squeeze of the bridge
    /// challenge $\alpha'$ when booleanity ran.
    pub fn step4_sumcheck_verify(
        mut self,
    ) -> Result<VerifierSumchecked<'a, Zt, F, IdealOverF, D, FD>, ProtocolError<F>> {
        let num_constraints = count_constraints::<U>();

        // Per-prime sub-proof length sanity check
        let n_fq = self.all_field_cfgs.len().saturating_sub(1);
        if self.proof_cpr_fq.len() != n_fq
            || self.proof_combined_sumchecks_fq.len() != n_fq
        {
            return Err(ProtocolError::FqIdealCheck {
                prime_idx: self.proof_cpr_fq.len(),
                q: "<fq sub-proof length mismatch>".to_owned(),
                source: IdealCheckError::IdealCollectorError(
                    ideal_check::BatchedIdealCheckError::LengthMismatch {
                        num_ideals: n_fq,
                        provided_values: self.proof_cpr_fq.len(),
                    },
                ),
            });
        }

        // `fq-unify`: mirror the prover by sampling one shared CPR
        // batching challenge $\alpha$ in $[0, q^*)$. `[0]` feeds the Q[X]
        // CPR; `[i + 1]` feeds the per-prime CPRs.
        let q_star_cfg = &self.all_field_cfgs[self.q_star_idx];
        let folding_challenges: Vec<F> = shared_challenge::sample_shared_field_challenge::<F>(
            &mut self.base.pcs_transcript.fs_transcript,
            q_star_cfg,
            &self.all_field_cfgs,
        );

        // -------- Q[X] branch: CPR pre-sumcheck ------------------------
        let q_cpr_verifier_ancillary = CombinedPolyResolver::prepare_verifier::<U>(
            &self.proof_cpr,
            self.proof_combined_sumcheck.claimed_sums()[0].clone(),
            &self.ic_subclaims[0],
            /* branch_idx = */ 0,
            num_constraints.q,
            self.base.num_vars,
            &self.projecting_elements[0],
            &folding_challenges[0],
            &self.field_cfg,
        )?;

        // Booleanity pre-sumcheck: squeezes the zerocheck point `r`
        // (num_vars field elements) and the batching challenge `alpha`,
        // in that order.
        let sig = self.base.uair_signature.clone();
        let num_pub_bin = sig.public_cols().num_binary_poly_cols();
        let num_total_bin = sig.total_cols().num_binary_poly_cols();
        let bin_wit_present = num_total_bin > num_pub_bin;

        let bool_verifier_ancillary = if bin_wit_present {
            // booleanity is group index 1 (right after CPR) on the Q-branch.
            let bool_claimed_sum = self
                .proof_combined_sumcheck
                .claimed_sums()
                .get(1)
                .ok_or(ProtocolError::BooleanityProofMissing)?;
            let num_wit_bin = num_total_bin.saturating_sub(num_pub_bin);
            let anc = BooleanityChecker::<F>::prepare_verifier(
                &mut self.base.pcs_transcript.fs_transcript,
                bool_claimed_sum,
                num_wit_bin,
                D,
                self.base.num_vars,
                &self.field_cfg,
            )
            .map_err(ProtocolError::Booleanity)?;
            Some(anc)
        } else {
            None
        };

        // -------- Per-prime branches: CPR pre-sumcheck ------------------
        let mut fq_cpr_ancillaries: Vec<_> = Vec::with_capacity(n_fq);
        for prime_idx in 0..n_fq {
            let branch_idx = add!(prime_idx, 1);
            let cfg_i = &self.all_field_cfgs[branch_idx];
            let anc_i = CombinedPolyResolver::prepare_verifier::<U>(
                &self.proof_cpr_fq[prime_idx],
                self.proof_combined_sumchecks_fq[prime_idx].claimed_sums()[0].clone(),
                &self.ic_subclaims[branch_idx],
                branch_idx,
                num_constraints.for_prime(prime_idx),
                self.base.num_vars,
                &self.projecting_elements[branch_idx],
                &folding_challenges[branch_idx],
                cfg_i,
            )?;
            fq_cpr_ancillaries.push(anc_i);
        }

        // -------- Lockstep multi-degree sumcheck verify ----------------
        // Branch 0 = Q[X] (CPR + optional booleanity); branches i >= 1
        // = per-prime CPR.
        let mut branch_proofs: Vec<(&MultiDegreeSumcheckProof<F>, &F::Config)> =
            Vec::with_capacity(add!(n_fq, 1));
        branch_proofs.push((&self.proof_combined_sumcheck, &self.field_cfg));
        for prime_idx in 0..n_fq {
            let branch_idx = add!(prime_idx, 1);
            branch_proofs.push((
                &self.proof_combined_sumchecks_fq[prime_idx],
                &self.all_field_cfgs[branch_idx],
            ));
        }
        let mut subclaims_iter = MultiDegreeSumcheck::verify_as_subprotocol(
            &mut self.base.pcs_transcript.fs_transcript,
            self.base.num_vars,
            &branch_proofs,
            q_star_cfg,
        )
        .map_err(CombinedPolyResolverError::SumcheckError)?
        .into_iter();

        // -------- Q[X] branch finalize ---------------------------------
        let md_subclaims = subclaims_iter
            .next()
            .expect("Q[X] branch subclaim always present");

        let cpr_subclaim = CombinedPolyResolver::finalize_verifier::<U>(
            &mut self.base.pcs_transcript.fs_transcript,
            self.proof_cpr,
            md_subclaims.point().to_vec(),
            md_subclaims.expected_evaluations()[0].clone(),
            q_cpr_verifier_ancillary,
            &self.projected_scalars_f,
            /* branch_idx = */ 0,
            &self.field_cfg,
        )?;

        // Booleanity -> multipoint-eval `alpha_prime` bridge.
        // After `finalize_verifier` (residue check + transcript absorption of
        // `bit_slice_evals`), squeeze `alpha_prime` and carry both forward to the
        // next step.
        let bool_bit_slice_evals: Option<Vec<F>> = if let Some(anc) = bool_verifier_ancillary {
            let booleanity_proof = self
                .proof_booleanity
                .take()
                .ok_or(ProtocolError::BooleanityProofMissing)?;
            let expected_eval = md_subclaims
                .expected_evaluations()
                .get(1)
                .ok_or(ProtocolError::BooleanityProofMissing)?;

            // Sumcheck residue check at r* against the bit-slice evals.
            // Absorbs bit_slice_evals.
            let bool_subclaim = BooleanityChecker::<F>::finalize_verifier(
                &mut self.base.pcs_transcript.fs_transcript,
                booleanity_proof,
                md_subclaims.point().to_vec(),
                expected_eval,
                anc,
                &self.field_cfg,
            )
            .map_err(ProtocolError::Booleanity)?;

            Some(bool_subclaim.bit_slice_evals)
        } else {
            None
        };

        // -------- Per-prime branch finalize ---------------------------
        // Mirror the prover's loop: pop next subclaim per branch, call
        // `finalize_verifier` under that branch's cfg + projected scalars.
        for (prime_idx, (cpr_proof_i, cpr_ancillary_i)) in self
            .proof_cpr_fq
            .into_iter()
            .zip(fq_cpr_ancillaries)
            .enumerate()
        {
            let branch_idx = add!(prime_idx, 1);
            let cfg_i = &self.all_field_cfgs[branch_idx];
            let md_subclaims_i = subclaims_iter
                .next()
                .expect("per-prime sumcheck subclaim always present");
            CombinedPolyResolver::finalize_verifier::<U>(
                &mut self.base.pcs_transcript.fs_transcript,
                cpr_proof_i,
                md_subclaims_i.point().to_vec(),
                md_subclaims_i.expected_evaluations()[0].clone(),
                cpr_ancillary_i,
                &self.projected_scalars_f_fq[prime_idx],
                branch_idx,
                cfg_i,
            )?;
        }

        // Squeeze alpha_prime in the same transcript order as the prover.
        let alpha_prime_f: Option<F> = bool_bit_slice_evals.as_ref().map(|_| {
            self.base
                .pcs_transcript
                .fs_transcript
                .get_field_challenge(&self.field_cfg)
        });

        let cpr_eval_point = cpr_subclaim.evaluation_point;

        let _ = &self.proof_lookup_proof;

        Ok(VerifierSumchecked {
            base: self.base,
            field_cfg: self.field_cfg,
            all_field_cfgs: self.all_field_cfgs,
            q_star_idx: self.q_star_idx,
            projecting_elements: self.projecting_elements,
            folding_challenges,
            cpr_eval_point,
            cpr_up_evals: cpr_subclaim.up_evals,
            cpr_down_evals: cpr_subclaim.down_evals,
            bool_bit_slice_evals,
            alpha_prime_f,
            proof_commitments: self.proof_commitments,
            proof_multipoint_eval: self.proof_multipoint_eval,
            proof_witness_lifted_evals: self.proof_witness_lifted_evals,
            proof_lookup_proof: self.proof_lookup_proof,
            _phantom: PhantomData,
        })
    }
}

impl<'a, Zt, F, IdealOverF, const D: usize, const FD: usize>
    VerifierSumchecked<'a, Zt, F, IdealOverF, D, FD>
where
    Zt: ZincTypes<D, FD>,
    F: InnerTransparentField + FromPrimitiveWithConfig + FromRef<F> + Send + Sync + 'static,
    F::Integer: ConstIntSemiring + ConstTranscribable + Send + Sync + FromRef<Zt::Fmod>,
    IdealOverF: Ideal,
{
    /// Step 5: Multi-point evaluation sumcheck.
    ///
    /// When the booleanity argument ran, this step appends one extra
    /// scalar up_eval $c_j = \sum_i b_{j,i}\,(\alpha')^i$ per witness
    /// binary-poly column (derived from the in-flight `bit_slice_evals`).
    /// These collapse the bit-slice claims at $r^\star$ into the
    /// multipoint-eval consistency equation; the matching
    /// $\alpha'$-projected `open_evals` are produced in step 6.
    /// `down_evals` are passed through unchanged.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn step5_multipoint_eval<U: Uair>(
        mut self,
    ) -> Result<VerifierMultipointEvaled<'a, Zt, F, IdealOverF, D, FD>, ProtocolError<F>> {
        let up_evals: Vec<F> = if let (Some(bit_slice_evals), Some(alpha_prime)) =
            (&self.bool_bit_slice_evals, &self.alpha_prime_f)
        {
            let sig = self.base.uair_signature.clone();
            let num_pub_bin = sig.public_cols().num_binary_poly_cols();
            let num_total_bin = sig.total_cols().num_binary_poly_cols();
            let num_wit_bin = num_total_bin.saturating_sub(num_pub_bin);
            let extra = alpha_prime_bridge_up_evals::<F, D>(
                bit_slice_evals,
                num_wit_bin,
                alpha_prime,
                &self.field_cfg,
            );
            self.cpr_up_evals.iter().cloned().chain(extra).collect()
        } else {
            self.cpr_up_evals.clone()
        };

        let mp_subclaim = MultipointEval::verify_as_subprotocol(
            &mut self.base.pcs_transcript.fs_transcript,
            self.proof_multipoint_eval,
            &self.cpr_eval_point,
            &up_evals,
            &self.cpr_down_evals,
            self.base.uair_signature.shifts(),
            self.base.num_vars,
            &self.field_cfg,
        )?;

        Ok(VerifierMultipointEvaled {
            base: self.base,
            field_cfg: self.field_cfg,
            projecting_elements: self.projecting_elements,
            alpha_prime_f: self.alpha_prime_f,
            mp_subclaim,
            proof_commitments: self.proof_commitments,
            proof_witness_lifted_evals: self.proof_witness_lifted_evals,
            proof_lookup_proof: self.proof_lookup_proof,
            _phantom: PhantomData,
        })
    }
}

impl<'a, Zt, F, IdealOverF, const D: usize, const FD: usize>
    VerifierMultipointEvaled<'a, Zt, F, IdealOverF, D, FD>
where
    Zt: ZincTypes<D, FD>,
    Zt::Int: ProjectableToField<F>,
    <Zt::ArbitraryZt as ZipTypes>::Eval: ProjectableToField<F>,
    F: InnerTransparentField
        + FromPrimitiveWithConfig
        + for<'b> FromWithConfig<&'b Zt::Int>
        + for<'b> FromWithConfig<&'b Zt::Chal>
        + for<'b> MulByScalar<&'b F>
        + FromRef<F>
        + Send
        + Sync
        + 'static,
    F::Integer: ConstIntSemiring + ConstTranscribable + Send + Sync + FromRef<Zt::Fmod>,
    IdealOverF: Ideal,
{
    /// Step 6: Recompute public lifted_evals, assemble full set, verify
    /// multipoint eval subclaim, and absorb all lifted_evals into transcript.
    ///
    /// All columns are projected at the original $\psi_a$
    /// `projecting_elements[0]` (so shifts continue to be bound through
    /// the $\psi_a$ chain). When the booleanity argument ran, additional
    /// $\alpha'$-projected `open_evals` are appended for the witness
    /// binary-poly columns; these match the appended `up_evals` from
    /// step 5 and close the Schwartz-Zippel bridge to the bit-slice claims.
    pub fn step6_lifted_evals<U: Uair>(
        mut self,
    ) -> Result<VerifierLiftedEvalsChecked<'a, Zt, F, IdealOverF, D, FD>, ProtocolError<F>> {
        let r_0 = &self.mp_subclaim.sumcheck_subclaim.point;

        let pub_cols = self.base.uair_signature.public_cols();
        let num_pub_bin = pub_cols.num_binary_poly_cols();
        let num_pub_arb = pub_cols.num_arbitrary_poly_cols();
        let num_pub_int = pub_cols.num_int_cols();

        let wit_cols = self.base.uair_signature.witness_cols();
        let num_wit_bin = wit_cols.num_binary_poly_cols();
        let num_wit_arb = wit_cols.num_arbitrary_poly_cols();

        let public_lifted = if add!(add!(num_pub_bin, num_pub_arb), num_pub_int) > 0 {
            let projected_public = project_trace_coeffs_row_major::<F, Zt::Int, Zt::Int, D, D>(
                self.base.public_trace,
                &self.field_cfg,
            );
            compute_lifted_evals::<F, D>(
                r_0,
                &self.base.public_trace.binary_poly,
                &ProjectedTrace::RowMajor(projected_public),
                &self.field_cfg,
            )
        } else {
            Vec::new()
        };

        let witness_lifted_evals = &self.proof_witness_lifted_evals;

        let all_lifted_evals: Vec<_> = public_lifted[..num_pub_bin]
            .iter()
            .chain(&witness_lifted_evals[..num_wit_bin])
            .chain(&public_lifted[num_pub_bin..add!(num_pub_bin, num_pub_arb)])
            .chain(&witness_lifted_evals[num_wit_bin..add!(num_wit_bin, num_wit_arb)])
            .chain(&public_lifted[add!(num_pub_bin, num_pub_arb)..])
            .chain(&witness_lifted_evals[add!(num_wit_bin, num_wit_arb)..])
            .cloned()
            .collect();

        // Project every column at the original \psi_a projecting element.
        // Shifts continue to consume the \psi_a-projected open_evals.
        let mut open_evals: Vec<F> = all_lifted_evals
            .iter()
            .map(|bar_u| bar_u.evaluate_at_point(&self.projecting_elements[0]))
            .collect::<Result<Vec<_>, _>>()
            .map_err(ProtocolError::LiftedEvalProjection)?;

        // Append \alpha'-projected open_evals for the witness binary-poly
        // columns. Their position matches the appended up_evals from
        // step 5 (indices `[num_total_cols, num_total_cols + num_wit_bin)`
        // in the extended MP column list). No ShiftSpec references these
        // appended indices, so the shift_at_r0 selectors are unaffected.
        if let Some(alpha_prime) = &self.alpha_prime_f {
            for bar_u in &witness_lifted_evals[..num_wit_bin] {
                open_evals.push(
                    bar_u
                        .evaluate_at_point(alpha_prime)
                        .map_err(ProtocolError::LiftedEvalProjection)?,
                );
            }
        }

        MultipointEval::verify_subclaim(
            &self.mp_subclaim,
            &open_evals,
            self.base.uair_signature.shifts(),
            &self.field_cfg,
        )?;

        let mut transcription_buf: Vec<u8> = vec![0; F::Integer::NUM_BYTES];
        for bar_u in &all_lifted_evals {
            self.base
                .pcs_transcript
                .fs_transcript
                .absorb_random_field_slice(&bar_u.coeffs, &mut transcription_buf);
        }

        Ok(VerifierLiftedEvalsChecked {
            base: self.base,
            field_cfg: self.field_cfg,
            mp_subclaim: self.mp_subclaim,
            all_lifted_evals,
            proof_commitments: self.proof_commitments,
            proof_lookup_proof: self.proof_lookup_proof,
            _phantom: PhantomData,
        })
    }
}

impl<'a, Zt, F, IdealOverF, const D: usize, const FD: usize>
    VerifierLiftedEvalsChecked<'a, Zt, F, IdealOverF, D, FD>
where
    Zt: ZincTypes<D, FD>,
    Zt::Int: ProjectableToField<F>,
    <Zt::BinaryZt as ZipTypes>::Cw: ProjectableToField<F>,
    <Zt::ArbitraryZt as ZipTypes>::Eval: ProjectableToField<F>,
    <Zt::ArbitraryZt as ZipTypes>::Cw: ProjectableToField<F>,
    <Zt::IntZt as ZipTypes>::Cw: ProjectableToField<F>,
    F: InnerTransparentField
        + FromPrimitiveWithConfig
        + for<'b> FromWithConfig<&'b Zt::Int>
        + for<'b> FromWithConfig<&'b Zt::CombR>
        + for<'b> FromWithConfig<&'b Zt::Chal>
        + for<'b> MulByScalar<&'b F>
        + FromRef<F>
        + Send
        + Sync
        + 'static,
    F::Integer: ConstIntSemiring + ConstTranscribable + Send + Sync + FromRef<Zt::Fmod>,
    IdealOverF: Ideal,
{
    /// Step 7: PCS verification at `r_0` (witness columns only).
    pub fn step7_pcs_verify<U: Uair, const CHECK_FOR_OVERFLOW: bool>(
        mut self,
    ) -> Result<VerifierPcsVerified<IdealOverF>, ProtocolError<F>> {
        let r_0 = &self.mp_subclaim.sumcheck_subclaim.point;
        let commitments = &self.proof_commitments;

        let pub_cols = self.base.uair_signature.public_cols();
        let num_pub_bin = pub_cols.num_binary_poly_cols();
        let num_pub_arb = pub_cols.num_arbitrary_poly_cols();
        let num_pub_int = pub_cols.num_int_cols();

        let total = self.base.uair_signature.total_cols();
        let num_total_bin = total.num_binary_poly_cols();
        let num_total_arb = total.num_arbitrary_poly_cols();

        let pcs_transcript = &mut self.base.pcs_transcript;
        let field_cfg = &self.field_cfg;
        let all_lifted_evals = &self.all_lifted_evals;

        let zero = F::zero_with_cfg(field_cfg);

        macro_rules! verify_pcs_batch {
            // Non-folded variant
            ($Zt:ty, $Lc:ty, $vp:expr, $idx:tt, $pt:expr, [$evals_range:expr]) => {{
                verify_pcs_batch!(
                    $Zt,
                    $Lc,
                    $vp,
                    $idx,
                    $pt,
                    [$evals_range],
                    |bar_u: &DynamicPolynomialF<F>, alphas: &[_]| {
                        let mut eval_j = zero.clone();
                        for (coeff, alpha) in bar_u.coeffs.iter().zip(alphas.iter()) {
                            let mut term = F::from_with_cfg(alpha, field_cfg);
                            term *= coeff;
                            eval_j += &term;
                        }
                        eval_j
                    }
                )
            }};

            // Universal variant with custom eval_j computation (used for folded columns)
            ($Zt:ty, $Lc:ty, $vp:expr, $idx:tt, $pt:expr, [$evals_range:expr], $compute_eval_j:expr) => {{
                let comm = &commitments.$idx;
                if comm.batch_size > 0 {
                    let per_poly_alphas = ZipPlus::<$Zt, $Lc>::sample_alphas(
                        &mut pcs_transcript.fs_transcript,
                        comm.batch_size,
                    );
                    let mut eval_f = zero.clone();
                    for (bar_u, alphas) in all_lifted_evals[$evals_range]
                        .iter()
                        .zip(per_poly_alphas.iter())
                    {
                        let eval_j = $compute_eval_j(bar_u, alphas);
                        eval_f += eval_j;
                    }
                    ZipPlus::<$Zt, $Lc>::verify_with_alphas::<F, CHECK_FOR_OVERFLOW>(
                        pcs_transcript,
                        $vp,
                        comm,
                        field_cfg,
                        $pt,
                        &eval_f,
                        &per_poly_alphas,
                    )
                    .map_err(|e| ProtocolError::PcsVerification($idx, e))?;
                }
            }};
        }

        // Folded witness columns are proved using the extended evaluation point
        // `r_0_ext = r_0 || folding_challenges`.
        let num_folding_challenges = Zt::BinaryFold::FOLDING_FACTOR.ilog2();
        let folding_challenges = (0..num_folding_challenges)
            .map(|_| {
                let g_chal: Zt::Chal = pcs_transcript.fs_transcript.get_challenge();
                F::from_with_cfg(&g_chal, &self.field_cfg)
            })
            .collect_vec();
        let mut r_0_ext = r_0.clone();
        r_0_ext.extend_from_slice(&folding_challenges);

        verify_pcs_batch!(
            Zt::BinaryZt,
            Zt::BinaryLc,
            self.base.vp_bin,
            0,
            &r_0_ext,
            [num_pub_bin..num_total_bin],
            |bar_u: &DynamicPolynomialF<F>, alphas: &[_]| {
                Zt::BinaryFold::fold_eval_claim(
                    &bar_u.coeffs,
                    alphas,
                    &folding_challenges,
                    field_cfg,
                )
            }
        );
        verify_pcs_batch!(
            Zt::ArbitraryZt,
            Zt::ArbitraryLc,
            self.base.vp_arb,
            1,
            &r_0,
            [add!(num_total_bin, num_pub_arb)..add!(num_total_bin, num_total_arb)]
        );
        verify_pcs_batch!(
            Zt::IntZt,
            Zt::IntLc,
            self.base.vp_int,
            2,
            &r_0,
            [add!(add!(num_total_bin, num_total_arb), num_pub_int)..]
        );

        Ok(VerifierPcsVerified {
            _phantom: PhantomData,
        })
    }
}

impl<IdealOverF: Ideal> VerifierPcsVerified<IdealOverF> {
    /// Complete verification.
    pub fn finish<F: PrimeField>(self) -> Result<(), ProtocolError<F>> {
        Ok(())
    }
}

//
// verify() wrapper
//

impl<Zt, U, F, const D: usize, const FD: usize> ZincPlusPiop<Zt, U, F, D, FD>
where
    Zt: ZincTypes<D, FD>,
    Zt::Int: ProjectableToField<F>,
    <Zt::BinaryZt as ZipTypes>::Cw: ProjectableToField<F>,
    <Zt::ArbitraryZt as ZipTypes>::Eval: ProjectableToField<F>,
    <Zt::ArbitraryZt as ZipTypes>::Cw: ProjectableToField<F>,
    <Zt::IntZt as ZipTypes>::Cw: ProjectableToField<F>,
    F: InnerTransparentField
        + FromPrimitiveWithConfig
        + for<'a> FromWithConfig<&'a Zt::Int>
        + for<'a> FromWithConfig<&'a Zt::CombR>
        + for<'a> FromWithConfig<&'a Zt::Chal>
        + for<'a> MulByScalar<&'a F>
        + FromRef<F>
        + Send
        + Sync
        + 'static,
    F::Integer:
        ConstIntSemiring + ConstTranscribable + Send + Sync + FromRef<Zt::Fmod> + FromRef<u64>,
    U: Uair + 'static,
{
    /// Zinc+ full PIOP verifier.
    ///
    /// Runs all verification steps in sequence and returns `Ok(())` on
    /// success. For per-step control, start with
    /// [`Self::step0_reconstruct_transcript`] and chain the individual
    /// `stepN_*` methods.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub fn verify<IdealOverF, const CHECK_FOR_OVERFLOW: bool>(
        vp: &(
            ZipPlusParams<Zt::BinaryZt, Zt::BinaryLc>,
            ZipPlusParams<Zt::ArbitraryZt, Zt::ArbitraryLc>,
            ZipPlusParams<Zt::IntZt, Zt::IntLc>,
        ),
        proof: Proof<F>,
        public_trace: &UairTrace<Zt::Int, Zt::Int, D, D>,
        num_vars: usize,
        project_scalar: impl Fn(&U::Scalar, &F::Config) -> DynamicPolynomialF<F>,
        project_ideal: impl Fn(&IdealOrZero<U::Ideal>, &F::Config) -> IdealOverF,
        project_fq_ideal: impl Fn(&IdealOrZero<U::FqIdeal>, &F::Config) -> IdealOverF,
    ) -> Result<(), ProtocolError<F>>
    where
        IdealOverF: Ideal + IdealCheck<DynamicPolynomialF<F>>,
    {
        ZincPlusPiop::<Zt, U, F, D, FD>::step0_reconstruct_transcript::<IdealOverF>(
            vp,
            proof,
            public_trace,
            num_vars,
        )?
        .step1_prime_projection()?
        .step2_ideal_check(project_ideal, project_fq_ideal)?
        .step3_eval_projection(project_scalar)?
        .step4_sumcheck_verify()?
        .step5_multipoint_eval::<U>()?
        .step6_lifted_evals::<U>()?
        .step7_pcs_verify::<U, CHECK_FOR_OVERFLOW>()?
        .finish::<F>()
    }
}

/// Test-only accessors for internal state, needed for tampering tests.
#[cfg(test)]
pub mod test_helpers {
    use super::*;

    #[cfg(test)]
    impl<'a, Zt: ZincTypes<D, FD>, const D: usize, const FD: usize> VerifierBase<'a, Zt, D, FD> {
        #[cfg(test)]
        pub fn fs_transcript_mut(&mut self) -> &mut Blake3Transcript {
            &mut self.pcs_transcript.fs_transcript
        }
    }

    #[cfg(test)]
    impl<'a, Zt, U, F, IdealOverF, const D: usize, const FD: usize>
        VerifierEvalProjected<'a, Zt, U, F, IdealOverF, D, FD>
    where
        Zt: ZincTypes<D, FD>,
        F: PrimeField,
        U: Uair,
    {
        pub fn projecting_element_f(&self) -> &F {
            &self.projecting_elements[0]
        }

        pub fn field_cfg(&self) -> &F::Config {
            &self.field_cfg
        }

        pub fn fs_transcript_mut(&mut self) -> &mut Blake3Transcript {
            self.base.fs_transcript_mut()
        }

        pub fn proof_combined_sumcheck(&self) -> &MultiDegreeSumcheckProof<F> {
            &self.proof_combined_sumcheck
        }

        pub fn ic_subclaim(&self) -> &ideal_check::VerifierSubclaim<F> {
            &self.ic_subclaims[0]
        }

        pub fn proof_cpr(&self) -> &CombinedPolyResolverProof<F> {
            &self.proof_cpr
        }

        pub fn num_vars(&self) -> usize {
            self.base.num_vars
        }

        pub fn uair_signature(&self) -> &UairSignature {
            &self.base.uair_signature
        }
    }
}
