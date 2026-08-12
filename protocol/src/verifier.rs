use super::*;
use itertools::Itertools;
use std::io::Cursor;
use zinc_piop::{
    combined_poly_resolver::CombinedPolyResolver,
    ideal_check::{self, IdealCheckProtocol},
    lookup::booleanity::{BoolVerifierAncillary, BooleanityChecker, BooleanityProof},
    multipoint_eval::{self, MultipointEval, MultipointEvalFamilyInputs},
    projections::{
        ProjectedScalars, ProjectedTrace, project_scalars, project_scalars_to_field,
        project_trace_coeffs_row_major,
    },
    sumcheck::multi_degree::MultiDegreeSumcheck,
};
use zinc_poly::univariate::dynamic::{DynamicPolynomialConfig, HasDynamicPolynomialConfig};
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
use zinc_utils::{add, powers, projectable_to_field::ProjectableToField, sub};
use zip_plus::{
    pcs::structs::{ZipPlus, ZipPlusParams, ZipTypes},
    pcs_transcript::PcsVerifierTranscript,
};

//
// Shared base
//

/// Persistent verifier infrastructure carried across every step.
#[derive(Clone, Debug)]
pub struct VerifierBase<'a, Zt: ZincTypes<D, FD>, const D: usize, const FD: usize> {
    num_vars: usize,
    uair_signature: UairSignature<Zt::Fmod>,
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
    C: BaseFieldConfig,
    IdealOverF,
    const D: usize,
    const FD: usize,
> {
    base: VerifierBase<'a, Zt, D, FD>,

    // Proof leftovers, still in wire form (canonical lifted integers)
    proof_commitments: (ZipPlusCommitment, ZipPlusCommitment, ZipPlusCommitment),
    proof_ideal_check: IdealCheckProof<Zt::Fmod>,
    proof_cpr: CombinedPolyResolverProof<Zt::Fmod>,
    proof_combined_sumcheck: MultiDegreeSumcheckProof<Zt::Fmod>,
    proof_multipoint_eval: MultipointEvalProof<Zt::Fmod>,
    /// Per-constraint-family witness-only lifted MLE evals. Layout:
    /// `[0]` = Q-family ($q_0$), `[i]` = declared prime $i - 1$ for
    /// $i = 1, \ldots, n$. Length = `1 + n_fq`.
    proof_witness_lifted_evals: Vec<Vec<DynamicPolynomial<Zt::Fmod>>>,
    /// Witness-only lifted MLE evals for $q''$ prime.
    /// `None` when there's no $F_q[X]$ constraints, in which case we assume
    /// $q'' := q_0$ and thus `proof_witness_lifted_evals[0]` is used for
    /// PCS verification.
    proof_witness_lifted_evals_pp: Option<Vec<DynamicPolynomial<Zt::Fmod>>>,
    proof_lookup_proof: Option<BatchedLookupProof<Zt::Fmod>>,
    proof_booleanity: Option<BooleanityProof<Zt::Fmod>>,
    proof_affine_booleanity: Option<BooleanityProof<Zt::Fmod>>,
    proof_ideal_checks_fq: Vec<IdealCheckProof<Zt::Fmod>>,
    proof_cpr_fq: Vec<CombinedPolyResolverProof<Zt::Fmod>>,
    proof_combined_sumchecks_fq: Vec<MultiDegreeSumcheckProof<Zt::Fmod>>,
    proof_multipoint_evals_fq: Vec<MultipointEvalProof<Zt::Fmod>>,
    _phantom: PhantomData<(U, C, IdealOverF)>,
}

/// After step 1 (prime projection).
#[derive(Clone, Debug)]
pub struct VerifierPrimeProjected<
    'a,
    Zt: ZincTypes<D, FD>,
    U: Uair,
    C: BaseFieldConfig,
    IdealOverF,
    const D: usize,
    const FD: usize,
> {
    base: VerifierBase<'a, Zt, D, FD>,
    field_cfg: C,

    // Proof leftovers
    proof_commitments: (ZipPlusCommitment, ZipPlusCommitment, ZipPlusCommitment),
    proof_ideal_check: IdealCheckProof<C::Element>,
    proof_cpr: CombinedPolyResolverProof<C::Element>,
    proof_combined_sumcheck: MultiDegreeSumcheckProof<C::Element>,
    proof_multipoint_eval: MultipointEvalProof<C::Element>,
    /// Per-constraint-family witness-only lifted MLE evals (see
    /// [`VerifierTranscriptReconstructed`] doc). Length = `1 + n_fq`.
    proof_witness_lifted_evals: Vec<Vec<DynamicPolynomial<C::Element>>>,
    /// Witness-only lifted MLE evals for $q''$ prime (see
    /// [`VerifierTranscriptReconstructed`] doc)
    proof_witness_lifted_evals_pp: Option<Vec<DynamicPolynomial<Zt::Fmod>>>,
    proof_lookup_proof: Option<BatchedLookupProof<C::Element>>,
    proof_booleanity: Option<BooleanityProof<C::Element>>,
    proof_affine_booleanity: Option<BooleanityProof<C::Element>>,
    proof_ideal_checks_fq: Vec<IdealCheckProof<C::Element>>,
    proof_cpr_fq: Vec<CombinedPolyResolverProof<C::Element>>,
    proof_combined_sumchecks_fq: Vec<MultiDegreeSumcheckProof<C::Element>>,
    proof_multipoint_evals_fq: Vec<MultipointEvalProof<C::Element>>,
    _phantom: PhantomData<(U, IdealOverF)>,
}

/// After step 2 (ideal check). `project_ideal` has been consumed.
#[derive(Clone, Debug)]
pub struct VerifierIdealChecked<
    'a,
    Zt: ZincTypes<D, FD>,
    U: Uair,
    C: BaseFieldConfig,
    IdealOverF,
    const D: usize,
    const FD: usize,
> {
    base: VerifierBase<'a, Zt, D, FD>,
    field_cfg: C,
    /// Per-family field configs (`[0]` = $Q[X]$, `[i >= 1]` =
    /// $F_{q_{i-1}}[X]$), kept for the next step's shared $\psi$
    /// projecting element.
    all_field_cfgs: Vec<C>,
    /// Index of $q^*$ in `all_field_cfgs`, computed once in step 2 and
    /// threaded forward so step 3 can recover `q_star_cfg` by indexing.
    q_star_idx: usize,
    /// Per-family IC subclaims (length `n + 1`). `[0]` is the Q[X]
    /// family's subclaim; `[i + 1]` is the per-prime $F_{q_i}[X]$
    /// subclaim. Previously only the Q[X] subclaim was kept; the per-prime
    /// subclaims were squeezed and discarded. They are now retained so
    /// step 4 (CPR verify) can drive a per-prime `prepare_verifier` call
    /// for each fq family.
    ic_subclaims: Vec<ideal_check::VerifierSubclaim<C::Element>>,

    // Proof leftovers
    proof_commitments: (ZipPlusCommitment, ZipPlusCommitment, ZipPlusCommitment),
    proof_cpr: CombinedPolyResolverProof<C::Element>,
    proof_combined_sumcheck: MultiDegreeSumcheckProof<C::Element>,
    proof_multipoint_eval: MultipointEvalProof<C::Element>,
    /// Per-constraint-family witness-only lifted MLE evals (see
    /// [`VerifierTranscriptReconstructed`] doc). Length = `1 + n_fq`.
    proof_witness_lifted_evals: Vec<Vec<DynamicPolynomial<C::Element>>>,
    /// Witness-only lifted MLE evals for $q''$ prime (see
    /// [`VerifierTranscriptReconstructed`] doc)
    proof_witness_lifted_evals_pp: Option<Vec<DynamicPolynomial<Zt::Fmod>>>,
    proof_lookup_proof: Option<BatchedLookupProof<C::Element>>,
    proof_booleanity: Option<BooleanityProof<C::Element>>,
    proof_affine_booleanity: Option<BooleanityProof<C::Element>>,
    /// Per-prime CPR proofs (one per declared prime).
    proof_cpr_fq: Vec<CombinedPolyResolverProof<C::Element>>,
    /// Per-prime multi-degree sumcheck proofs (one per declared prime).
    proof_combined_sumchecks_fq: Vec<MultiDegreeSumcheckProof<C::Element>>,
    /// Per-prime multipoint-eval proofs (one per declared prime).
    proof_multipoint_evals_fq: Vec<MultipointEvalProof<C::Element>>,
    _phantom: PhantomData<(U, IdealOverF)>,
}

/// After step 3 (eval projection). `project_scalar` has been consumed.
#[derive(Clone, Debug)]
pub struct VerifierEvalProjected<
    'a,
    Zt: ZincTypes<D, FD>,
    U: Uair,
    C: BaseFieldConfig,
    IdealOverF,
    const D: usize,
    const FD: usize,
> {
    base: VerifierBase<'a, Zt, D, FD>,
    field_cfg: C,
    /// Per-family field configs (`[0]` = $Q[X]$, `[i >= 1]` =
    /// $F_{q_{i-1}}[X]$), kept for later per-prime CPR/MP/PCS steps.
    all_field_cfgs: Vec<C>,
    /// Index of $q^*$ in `all_field_cfgs`, kept for later steps that need
    /// to re-sample shared challenges in $[0, q^*)$.
    q_star_idx: usize,
    /// Per-family IC subclaims (length `n + 1`). `[0]` feeds the Q[X] CPR
    /// `prepare_verifier`; `[i + 1]` will feed each per-prime CPR
    /// `prepare_verifier` in step 4.
    ic_subclaims: Vec<ideal_check::VerifierSubclaim<C::Element>>,
    /// Per-family $\psi$-projecting elements: integer sampled mod $q^*$
    /// and projected onto each of `all_field_cfgs`.
    projecting_elements: Vec<C::Element>,
    /// Q[X] family's $\psi$-projected scalars.
    projected_scalars_f: ProjectedScalars<U::Scalar, C::Element>,
    /// Per-prime $\psi$-projected scalars (one per declared prime). Built
    /// in step 3 from each family's `projecting_elements[i + 1]` and the
    /// UAIR-author-supplied `project_scalar` closure (applied with
    /// `all_field_cfgs[i + 1]`). Consumed in step 4 (CPR finalize per
    /// family).
    projected_scalars_f_fq: Vec<ProjectedScalars<U::Scalar, C::Element>>,

    // Proof leftovers
    proof_commitments: (ZipPlusCommitment, ZipPlusCommitment, ZipPlusCommitment),
    proof_cpr: CombinedPolyResolverProof<C::Element>,
    proof_combined_sumcheck: MultiDegreeSumcheckProof<C::Element>,
    proof_multipoint_eval: MultipointEvalProof<C::Element>,
    /// Per-constraint-family witness-only lifted MLE evals (see
    /// [`VerifierTranscriptReconstructed`] doc). Length = `1 + n_fq`.
    proof_witness_lifted_evals: Vec<Vec<DynamicPolynomial<C::Element>>>,
    /// Witness-only lifted MLE evals for $q''$ prime (see
    /// [`VerifierTranscriptReconstructed`] doc)
    proof_witness_lifted_evals_pp: Option<Vec<DynamicPolynomial<Zt::Fmod>>>,
    proof_lookup_proof: Option<BatchedLookupProof<C::Element>>,
    proof_booleanity: Option<BooleanityProof<C::Element>>,
    proof_affine_booleanity: Option<BooleanityProof<C::Element>>,
    /// Per-prime CPR proofs (one per declared prime).
    proof_cpr_fq: Vec<CombinedPolyResolverProof<C::Element>>,
    /// Per-prime multi-degree sumcheck proofs (one per declared prime),
    /// consumed in step 4 by the lockstep verifier driver.
    proof_combined_sumchecks_fq: Vec<MultiDegreeSumcheckProof<C::Element>>,
    /// Per-prime multipoint-eval proofs (one per declared prime),
    /// consumed in step 5 by the lockstep multipoint-eval verifier.
    proof_multipoint_evals_fq: Vec<MultipointEvalProof<C::Element>>,
    _phantom: PhantomData<(U, IdealOverF)>,
}

/// After step 4 (sumcheck verify).
#[derive(Clone, Debug)]
pub struct VerifierSumchecked<
    'a,
    Zt: ZincTypes<D, FD>,
    C: BaseFieldConfig,
    IdealOverF,
    const D: usize,
    const FD: usize,
> {
    base: VerifierBase<'a, Zt, D, FD>,
    field_cfg: C,
    /// Per-family field configs (carried for downstream steps).
    all_field_cfgs: Vec<C>,
    /// Index of $q^*$ in `all_field_cfgs` (carried for downstream steps).
    q_star_idx: usize,
    /// Per-family $\psi$-projecting elements: integer sampled mod $q^*$
    /// and projected onto each of `all_field_cfgs`.
    projecting_elements: Vec<C::Element>,
    /// CPR subclaim's evaluation point ($r^\star$)
    cpr_eval_point: Vec<C::Element>,
    cpr_up_evals: Vec<C::Element>,
    cpr_bit_op_evals: Vec<C::Element>,
    cpr_down_evals: Vec<C::Element>,
    /// Per-prime CPR subclaim evaluation points (r*, lifted into
    /// each family's field). Length `n_fq`. Empty for UAIRs with no
    /// declared fq primes.
    cpr_eval_points_fq: Vec<Vec<C::Element>>,
    /// Per-prime CPR subclaim `up_evals`. Length `n_fq`. Consumed in step 5.
    cpr_up_evals_fq: Vec<Vec<C::Element>>,
    /// Per-prime CPR subclaim `bit_op_evals`. Length `n_fq`. Consumed in step
    /// 5.
    cpr_bit_op_evals_fq: Vec<Vec<C::Element>>,
    /// Per-prime CPR subclaim `down_evals`. Length `n_fq`. Consumed in step 5.
    cpr_down_evals_fq: Vec<Vec<C::Element>>,
    /// `bit_slice_evals` carried over from booleanity's `finalize_verifier`,
    /// to be collapsed into the appended `up_evals` entries
    /// $c_j = \sum_i b_{j,i}\,(\alpha')^i$.
    ///
    /// `None` iff there are no witness binary-poly columns.
    bool_bit_slice_evals: Option<Vec<C::Element>>,
    /// Affine-virtual bit-slice evaluations carried separately from the
    /// committed-column Booleanity proof.
    affine_bool_bit_slice_evals: Option<Vec<C::Element>>,
    /// Fresh challenge sampled after `bit_slice_evals` were absorbed by
    /// booleanity's `finalize_verifier`. Consumed in step 5 (bridge
    /// scalars) and step 6 (appended $\alpha'$-projected open_evals).
    ///
    /// `None` iff there are neither witness binary-poly columns nor affine
    /// virtual booleanity targets.
    alpha_prime_f: Option<C::Element>,

    // Proof leftovers
    proof_commitments: (ZipPlusCommitment, ZipPlusCommitment, ZipPlusCommitment),
    proof_multipoint_eval: MultipointEvalProof<C::Element>,
    proof_multipoint_evals_fq: Vec<MultipointEvalProof<C::Element>>,
    /// Per-constraint-family witness-only lifted MLE evals (see
    /// [`VerifierTranscriptReconstructed`] doc). Length = `1 + n_fq`.
    proof_witness_lifted_evals: Vec<Vec<DynamicPolynomial<C::Element>>>,
    /// Witness-only lifted MLE evals for $q''$ prime (see
    /// [`VerifierTranscriptReconstructed`] doc)
    proof_witness_lifted_evals_pp: Option<Vec<DynamicPolynomial<Zt::Fmod>>>,
    proof_lookup_proof: Option<BatchedLookupProof<C::Element>>,
    _phantom: PhantomData<IdealOverF>,
}

/// After step 5 (multi-point eval).
#[derive(Clone, Debug)]
pub struct VerifierMultipointEvaled<
    'a,
    Zt: ZincTypes<D, FD>,
    C: BaseFieldConfig,
    IdealOverF,
    const D: usize,
    const FD: usize,
> {
    base: VerifierBase<'a, Zt, D, FD>,
    field_cfg: C,
    /// Per-family field configs (carried for downstream steps).
    all_field_cfgs: Vec<C>,
    /// Per-family $\psi$-projecting elements: integer sampled mod $q^*$
    /// and projected onto each of `all_field_cfgs`.
    projecting_elements: Vec<C::Element>,
    // See VerifierSumchecked::alpha_prime_f
    alpha_prime_f: Option<C::Element>,
    mp_subclaim: multipoint_eval::Subclaim<C::Element>,
    /// Per-prime multipoint-eval subclaims, one per declared prime, produced
    /// by step 5's lockstep multipoint-eval verifier. Empty for UAIRs with
    /// no declared fq primes. Consumed in step 6.
    mp_subclaims_fq: Vec<multipoint_eval::Subclaim<C::Element>>,

    // Proof leftovers
    proof_commitments: (ZipPlusCommitment, ZipPlusCommitment, ZipPlusCommitment),
    /// Per-constraint-family witness-only lifted MLE evals (see
    /// [`VerifierTranscriptReconstructed`] doc). Length = `1 + n_fq`.
    proof_witness_lifted_evals: Vec<Vec<DynamicPolynomial<C::Element>>>,
    /// Witness-only lifted MLE evals for $q''$ prime (see
    /// [`VerifierTranscriptReconstructed`] doc)
    proof_witness_lifted_evals_pp: Option<Vec<DynamicPolynomial<Zt::Fmod>>>,
    proof_lookup_proof: Option<BatchedLookupProof<C::Element>>,
    _phantom: PhantomData<IdealOverF>,
}

/// After step 6 (lifted evals verification).
#[derive(Clone, Debug)]
pub struct VerifierLiftedEvalsChecked<
    'a,
    Zt: ZincTypes<D, FD>,
    C: BaseFieldConfig,
    IdealOverF,
    const D: usize,
    const FD: usize,
> {
    base: VerifierBase<'a, Zt, D, FD>,
    /// `q''`-family witness-only lifted MLE evaluations at `r* =  r_0 mod q''`.
    /// Consumed by PCS verify.
    lifted_evals_pp: Vec<DynamicPolynomial<C::Element>>,
    /// PCS-only prime cfg sampled at step 6 start (mirror of prover step 7).
    q_pp_cfg: C,
    /// $r^\star = r_0 \bmod q''$ — the PCS evaluation point.
    r_star: Vec<C::Element>,

    // Proof leftovers
    proof_commitments: (ZipPlusCommitment, ZipPlusCommitment, ZipPlusCommitment),
    /// Lookup proof component, threaded through the verifier state
    /// machine. Not yet consumed: the `BatchedLookupProof` verifier chain
    /// is not implemented (see step 5 / step 6 TODOs). Retained so the
    /// proof shape stays stable once lookup verification lands.
    #[allow(dead_code)]
    proof_lookup_proof: Option<BatchedLookupProof<C::Element>>,
    _phantom: PhantomData<IdealOverF>,
}

/// After step 7 (PCS verify). Ready for
/// [`finish`](VerifierPcsVerified::finish).
#[derive(Clone, Debug)]
pub struct VerifierPcsVerified<IdealOverF> {
    pcs_transcript: PcsVerifierTranscript,
    _phantom: PhantomData<IdealOverF>,
}

fn prepare_booleanity_verifier_group<C>(
    transcript: &mut Blake3Transcript,
    claimed_sums: &[C::Element],
    group_idx: &mut usize,
    num_cols: usize,
    bit_width: usize,
    num_vars: usize,
    field_cfg: &C,
) -> Result<Option<BoolVerifierAncillary<C::Element>>, ProtocolError<C::Element>>
where
    C: BaseFieldConfig + ProjectPrimitiveIntegersWithConfig + 'static,
    C::Integer: ConstTranscribable,
{
    if num_cols == 0 {
        return Ok(None);
    }

    let claimed_sum = claimed_sums
        .get(*group_idx)
        .ok_or(ProtocolError::BooleanityProofMissing)?;
    let ancillary = BooleanityChecker::<C>::prepare_verifier(
        transcript,
        claimed_sum,
        num_cols,
        bit_width,
        num_vars,
        field_cfg,
    )
    .map_err(ProtocolError::Booleanity)?;
    *group_idx = add!(*group_idx, 1);
    Ok(Some(ancillary))
}

fn finalize_booleanity_verifier_group<C>(
    transcript: &mut Blake3Transcript,
    proof: Option<BooleanityProof<C::Element>>,
    shared_point: &[C::Element],
    expected_evaluations: &[C::Element],
    group_idx: &mut usize,
    ancillary: Option<BoolVerifierAncillary<C::Element>>,
    field_cfg: &C,
) -> Result<Option<Vec<C::Element>>, ProtocolError<C::Element>>
where
    C: BaseFieldConfig + ProjectPrimitiveIntegersWithConfig + 'static,
    C::Integer: ConstTranscribable,
{
    let (proof, ancillary) = match (proof, ancillary) {
        (None, None) => return Ok(None),
        (Some(proof), Some(ancillary)) => (proof, ancillary),
        _ => return Err(ProtocolError::BooleanityProofMissing),
    };
    let expected_eval = expected_evaluations
        .get(*group_idx)
        .ok_or(ProtocolError::BooleanityProofMissing)?;
    *group_idx = add!(*group_idx, 1);

    let subclaim = BooleanityChecker::<C>::finalize_verifier(
        transcript,
        proof,
        shared_point.to_vec(),
        expected_eval,
        ancillary,
        field_cfg,
    )
    .map_err(ProtocolError::Booleanity)?;
    Ok(Some(subclaim.bit_slice_evals))
}

//
// Step implementations
//

impl<Zt, U, C, const D: usize, const FD: usize> ZincPlusPiop<Zt, U, C, D, FD>
where
    Zt: ZincTypes<D, FD>,
    U: Uair<Prime = Zt::Fmod>,
    C: BaseFieldConfig<Integer = Zt::Fmod>,
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
        mut proof: Proof<Zt::Fmod>,
        public_trace: &'a UairTrace<'a, Zt::Int, Zt::Int, D, D>,
        num_vars: usize,
    ) -> Result<
        VerifierTranscriptReconstructed<'a, Zt, U, C, IdealOverF, D, FD>,
        ProtocolError<C::Element>,
    >
    where
        IdealOverF: Ideal,
    {
        assert!(
            num_vars > 0,
            "Attempt to verify a constant: num_vars must be > 0"
        );
        let uair_signature = U::signature();
        let zip_proof = std::mem::take(&mut proof.zip);
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
            base.pcs_transcript.fs_transcript.absorb_bytes(&comm.root);
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
            proof_witness_lifted_evals_pp: proof.witness_lifted_evals_pp,
            proof_lookup_proof: proof.lookup_proof,
            proof_booleanity: proof.booleanity_proof,
            proof_affine_booleanity: proof.affine_booleanity_proof,
            proof_ideal_checks_fq: proof.ideal_checks_fq,
            proof_cpr_fq: proof.cpr_proofs_fq,
            proof_combined_sumchecks_fq: proof.combined_sumchecks_fq,
            proof_multipoint_evals_fq: proof.multipoint_evals_fq,
            _phantom: PhantomData,
        })
    }
}

impl<'a, Zt, U, C, IdealOverF, const D: usize, const FD: usize>
    VerifierTranscriptReconstructed<'a, Zt, U, C, IdealOverF, D, FD>
where
    Zt: ZincTypes<D, FD>,
    C: BaseFieldConfig<Integer = Zt::Fmod> + Clone + Send + Sync + 'static,
    U: Uair,
    IdealOverF: Ideal,
{
    /// Step 1: Prime projection. Samples the random field configuration.
    #[allow(clippy::type_complexity)]
    pub fn step1_prime_projection(
        mut self,
    ) -> Result<VerifierPrimeProjected<'a, Zt, U, C, IdealOverF, D, FD>, ProtocolError<C::Element>>
    {
        let field_cfg = self
            .base
            .pcs_transcript
            .fs_transcript
            .get_random_field_cfg::<C, Zt::Fmod, Zt::PrimeTest>();

        // Project the wire proof (canonical lifted integers) into the
        // per-family fields, now that every constraint-family config is
        // known ($q_0$ sampled above, declared primes from the signature).
        // Rejects non-canonical encodings. The $q''$-family section stays
        // in wire form until step 7 samples $q''$.
        let all_field_cfgs = build_all_cfgs::<C>(&self.base.uair_signature, field_cfg.clone());
        macro_rules! project {
            ($cfg:expr, $section:expr) => {
                $section.try_map(|int| project_canonical($cfg, int))
            };
        }
        macro_rules! project_fq_vec {
            ($section:expr) => {
                $section
                    .iter()
                    .zip(&all_field_cfgs[1..])
                    .map(|(p, cfg)| project!(cfg, p))
                    .try_collect()
            };
        }
        let n_families = all_field_cfgs.len();
        if self.proof_witness_lifted_evals.len() != n_families {
            return Err(ProtocolError::FamilyCountMismatch {
                got: self.proof_witness_lifted_evals.len(),
                expected: n_families,
            });
        }
        for fq_len in [
            self.proof_ideal_checks_fq.len(),
            self.proof_cpr_fq.len(),
            self.proof_combined_sumchecks_fq.len(),
            self.proof_multipoint_evals_fq.len(),
        ] {
            if fq_len != sub!(n_families, 1) {
                return Err(ProtocolError::FamilyCountMismatch {
                    got: fq_len,
                    expected: sub!(n_families, 1),
                });
            }
        }
        let proof_witness_lifted_evals = self
            .proof_witness_lifted_evals
            .iter()
            .zip(&all_field_cfgs)
            .map(|(polys, cfg)| {
                polys
                    .iter()
                    .map(|poly| poly.try_map(|int| project_canonical(cfg, int)))
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(VerifierPrimeProjected {
            base: self.base,
            proof_commitments: self.proof_commitments,
            proof_ideal_check: project!(&field_cfg, self.proof_ideal_check)?,
            proof_cpr: project!(&field_cfg, self.proof_cpr)?,
            proof_combined_sumcheck: project!(&field_cfg, self.proof_combined_sumcheck)?,
            proof_multipoint_eval: project!(&field_cfg, self.proof_multipoint_eval)?,
            proof_witness_lifted_evals,
            proof_witness_lifted_evals_pp: self.proof_witness_lifted_evals_pp,
            proof_lookup_proof: match &self.proof_lookup_proof {
                Some(p) => Some(project!(&field_cfg, p)?),
                None => None,
            },
            proof_booleanity: match &self.proof_booleanity {
                Some(p) => Some(project!(&field_cfg, p)?),
                None => None,
            },
            proof_affine_booleanity: match &self.proof_affine_booleanity {
                Some(p) => Some(project!(&field_cfg, p)?),
                None => None,
            },
            proof_ideal_checks_fq: project_fq_vec!(self.proof_ideal_checks_fq)?,
            proof_cpr_fq: project_fq_vec!(self.proof_cpr_fq)?,
            proof_combined_sumchecks_fq: project_fq_vec!(self.proof_combined_sumchecks_fq)?,
            proof_multipoint_evals_fq: project_fq_vec!(self.proof_multipoint_evals_fq)?,
            field_cfg,
            _phantom: PhantomData,
        })
    }
}

impl<'a, Zt, U, C, IdealOverF, const D: usize, const FD: usize>
    VerifierPrimeProjected<'a, Zt, U, C, IdealOverF, D, FD>
where
    Zt: ZincTypes<D, FD>,
    Zt::Int: ProjectableToField<C>,
    <Zt::ArbitraryZt as ZipTypes>::Eval: ProjectableToField<C>,
    C: BaseFieldConfig<Integer = Zt::Fmod>
        + ProjectPrimitiveIntegersWithConfig
        + ProjectElementWithConfig<Zt::Int>
        + ProjectElementWithConfig<Zt::CombR>
        + ProjectElementWithConfig<Zt::Chal>
        + Clone
        + Send
        + Sync
        + 'static,
    U: Uair + 'static,
    IdealOverF: Ideal + for<'cfg> IdealCheck<DynamicPolynomialConfig<'cfg, C>>,
{
    /// Step 2: Ideal check verification.
    ///
    /// **Shared evaluation point.**
    ///
    /// Mirrors the prover: sample one shared $r \in [0, q^*)^\mu$ from the
    /// transcript and lift it into each family's field via
    /// `cfg.project`, then run the per-family verifications in
    /// `family_idx` order.
    ///
    /// `project_fq_ideal` is only invoked when the UAIR declares at least
    /// one prime; UAIRs with $Q[X]$ only constraints can pass
    /// `|_, _| unreachable!()`.
    #[allow(clippy::type_complexity)]
    pub fn step2_ideal_check(
        mut self,
        project_ideal: impl Fn(&IdealOrZero<U::Ideal>, &C) -> IdealOverF,
        project_fq_ideal: impl Fn(&IdealOrZero<U::FqIdeal>, &C) -> IdealOverF,
    ) -> Result<VerifierIdealChecked<'a, Zt, U, C, IdealOverF, D, FD>, ProtocolError<C::Element>>
    {
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

        // Rebuild per-family field configs, then sample the
        // shared evaluation point from the transcript exactly as the
        // prover does. Family 0 = Q[X] (random sampled prime); families
        // i >= 1 = declared primes in `primes()` order.
        let all_field_cfgs = build_all_cfgs::<C>(&self.base.uair_signature, self.field_cfg.clone());
        let q_star_idx = shared_challenge::compute_q_star_idx::<C>(&all_field_cfgs);
        let q_star_cfg = &all_field_cfgs[q_star_idx];
        let shared_eval_points: Vec<Vec<C::Element>> =
            shared_challenge::sample_shared_field_challenges::<C>(
                &mut self.base.pcs_transcript.fs_transcript,
                self.base.num_vars,
                q_star_cfg,
                &all_field_cfgs,
            );

        // Q[X]-family ideal check verification.
        let mut ic_subclaims: Vec<ideal_check::VerifierSubclaim<C::Element>> =
            Vec::with_capacity(all_field_cfgs.len());
        let q_subclaim = IdealCheckProtocol::<U>::verify_as_subprotocol::<_, IdealOverF, _, _>(
            &mut self.base.pcs_transcript.fs_transcript,
            self.proof_ideal_check,
            /* family_idx = */ 0,
            num_constraints.q,
            &shared_eval_points[0],
            |ideal| project_ideal(ideal, &self.field_cfg),
            |_| unreachable!("Q[X] family"),
            &self.field_cfg,
        )?;
        ic_subclaims.push(q_subclaim);

        // Per-prime F_q[X] ideal-check verifications.
        // Subclaims are collected and threaded forward to step 4's per-prime CPR
        // `prepare_verifier`.
        for (prime_idx, (cfg_q_i, fq_proof)) in all_field_cfgs[1..]
            .iter()
            .zip(self.proof_ideal_checks_fq)
            .enumerate()
        {
            let family_idx = add!(prime_idx, 1);
            let fq_subclaim =
                IdealCheckProtocol::<U>::verify_as_subprotocol::<_, IdealOverF, _, _>(
                    &mut self.base.pcs_transcript.fs_transcript,
                    fq_proof,
                    family_idx,
                    num_constraints.for_prime(prime_idx),
                    &shared_eval_points[family_idx],
                    |_| unreachable!("F_q[X] family"),
                    |ideal| project_fq_ideal(ideal, cfg_q_i),
                    cfg_q_i,
                )
                .map_err(|source| ProtocolError::FqIdealCheck {
                    prime_idx,
                    q: cfg_q_i.modulus().to_string(),
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
            proof_witness_lifted_evals_pp: self.proof_witness_lifted_evals_pp,
            proof_lookup_proof: self.proof_lookup_proof,
            proof_booleanity: self.proof_booleanity,
            proof_affine_booleanity: self.proof_affine_booleanity,
            proof_cpr_fq: self.proof_cpr_fq,
            proof_combined_sumchecks_fq: self.proof_combined_sumchecks_fq,
            proof_multipoint_evals_fq: self.proof_multipoint_evals_fq,
            _phantom: PhantomData,
        })
    }
}

impl<'a, Zt, U, C, IdealOverF, const D: usize, const FD: usize>
    VerifierIdealChecked<'a, Zt, U, C, IdealOverF, D, FD>
where
    Zt: ZincTypes<D, FD>,
    C: BaseFieldConfig<Integer = Zt::Fmod>
        + ProjectElementWithConfig<Zt::Chal>
        + Clone
        + Send
        + Sync
        + 'static,
    U: Uair<Prime = Zt::Fmod> + 'static,
    IdealOverF: Ideal,
{
    /// Step 3: Evaluation projection. Consumes `project_scalar`.
    ///
    /// **Shared projecting element.**
    ///
    /// Mirrors the prover: sample a shared integer $a \in [0, q^*)$ and lift it
    /// into each family's field. The $Q[X]$ family consumes
    /// `projecting_elements[0]`; per-prime families consume
    /// `projecting_elements[i + 1]`.
    ///
    /// Also builds per-prime $\psi$-projected scalars from
    /// `project_scalar(., all_field_cfgs[i + 1])` and threads them
    /// forward as `projected_scalars_f_fq` for the per-prime CPR
    /// `finalize_verifier` in step 4.
    pub fn step3_eval_projection(
        mut self,
        project_scalar: impl Fn(&U::Scalar, &C) -> DynamicPolynomial<C::Element>,
    ) -> Result<VerifierEvalProjected<'a, Zt, U, C, IdealOverF, D, FD>, ProtocolError<C::Element>>
    {
        let q_star_cfg = &self.all_field_cfgs[self.q_star_idx];
        let projecting_elements: Vec<C::Element> =
            shared_challenge::sample_shared_field_challenge::<C>(
                &mut self.base.pcs_transcript.fs_transcript,
                q_star_cfg,
                &self.all_field_cfgs,
            );

        // Q[X] family.
        let projected_scalars_fx =
            project_scalars::<C, U>(&self.field_cfg, |s| project_scalar(s, &self.field_cfg));
        let projected_scalars_f = project_scalars_to_field(
            &self.field_cfg,
            projected_scalars_fx,
            &projecting_elements[0],
        )
        .map_err(|(_s, _f, e)| ProtocolError::ScalarProjection(e))?;

        // Per-prime F_q[X] families.
        let projected_scalars_f_fq: Result<Vec<ProjectedScalars<U::Scalar, C::Element>>, _> = self
            .all_field_cfgs
            .iter()
            .zip(projecting_elements.iter())
            .skip(1) // Skip Q[X] family
            .map(|(cfg_i, projecting_element)| {
                let projected_scalars_fx_i =
                    project_scalars::<C, U>(cfg_i, |s| project_scalar(s, cfg_i));
                project_scalars_to_field(cfg_i, projected_scalars_fx_i, projecting_element)
                    .map_err(|(_s, _f, e)| ProtocolError::ScalarProjection(e))
            })
            .collect();
        let projected_scalars_f_fq = projected_scalars_f_fq?;

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
            proof_witness_lifted_evals_pp: self.proof_witness_lifted_evals_pp,
            proof_lookup_proof: self.proof_lookup_proof,
            proof_booleanity: self.proof_booleanity,
            proof_affine_booleanity: self.proof_affine_booleanity,
            proof_cpr_fq: self.proof_cpr_fq,
            proof_combined_sumchecks_fq: self.proof_combined_sumchecks_fq,
            proof_multipoint_evals_fq: self.proof_multipoint_evals_fq,
            _phantom: PhantomData,
        })
    }
}

impl<'a, Zt, U, C, IdealOverF, const D: usize, const FD: usize>
    VerifierEvalProjected<'a, Zt, U, C, IdealOverF, D, FD>
where
    Zt: ZincTypes<D, FD>,
    Zt::Int: ProjectableToField<C>,
    <Zt::ArbitraryZt as ZipTypes>::Eval: ProjectableToField<C>,
    C: BaseFieldConfig<Integer = Zt::Fmod>
        + ProjectPrimitiveIntegersWithConfig
        + ProjectElementWithConfig<Zt::Int>
        + ProjectElementWithConfig<Zt::CombR>
        + ProjectElementWithConfig<Zt::Chal>
        + Clone
        + Send
        + Sync
        + 'static,
    U: Uair<Prime = Zt::Fmod> + 'static,
    IdealOverF: Ideal,
{
    /// Step 4: Sumcheck verification (CPR + optional booleanity +
    /// future lookup groups), followed by the squeeze of the bridge
    /// challenge $\alpha'$ when booleanity ran.
    pub fn step4_sumcheck_verify(
        mut self,
    ) -> Result<VerifierSumchecked<'a, Zt, C, IdealOverF, D, FD>, ProtocolError<C::Element>> {
        let num_constraints = count_constraints::<U>();

        // Per-prime sub-proof length sanity check
        let n_fq = self.all_field_cfgs.len().saturating_sub(1);
        if self.proof_cpr_fq.len() != n_fq || self.proof_combined_sumchecks_fq.len() != n_fq {
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

        // Sample one shared CPR batching challenge \alpha in [0, q*)
        let q_star_cfg = &self.all_field_cfgs[self.q_star_idx];
        let folding_challenges: Vec<C::Element> =
            shared_challenge::sample_shared_field_challenge::<C>(
                &mut self.base.pcs_transcript.fs_transcript,
                q_star_cfg,
                &self.all_field_cfgs,
            );

        // -------- Q[X] family: CPR pre-sumcheck ------------------------
        let q_cpr_verifier_ancillary = CombinedPolyResolver::prepare_verifier::<U>(
            &self.proof_cpr,
            self.proof_combined_sumcheck.claimed_sums()[0].clone(),
            &self.ic_subclaims[0],
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
        let num_wit_bin = num_total_bin.saturating_sub(num_pub_bin);
        let num_affine_virtuals = sig.affine_virtual_specs().len();
        let mut bool_group_idx = 1;

        let bool_verifier_ancillary = prepare_booleanity_verifier_group(
            &mut self.base.pcs_transcript.fs_transcript,
            self.proof_combined_sumcheck.claimed_sums(),
            &mut bool_group_idx,
            num_wit_bin,
            D,
            self.base.num_vars,
            &self.field_cfg,
        )?;
        let affine_bool_verifier_ancillary = prepare_booleanity_verifier_group(
            &mut self.base.pcs_transcript.fs_transcript,
            self.proof_combined_sumcheck.claimed_sums(),
            &mut bool_group_idx,
            num_affine_virtuals,
            D,
            self.base.num_vars,
            &self.field_cfg,
        )?;

        // -------- Per-prime families: CPR pre-sumcheck ------------------
        let mut fq_cpr_ancillaries: Vec<_> = Vec::with_capacity(n_fq);
        for prime_idx in 0..n_fq {
            let family_idx = add!(prime_idx, 1);
            let cfg_i = &self.all_field_cfgs[family_idx];
            let anc_i = CombinedPolyResolver::prepare_verifier::<U>(
                &self.proof_cpr_fq[prime_idx],
                self.proof_combined_sumchecks_fq[prime_idx].claimed_sums()[0].clone(),
                &self.ic_subclaims[family_idx],
                num_constraints.for_prime(prime_idx),
                self.base.num_vars,
                &self.projecting_elements[family_idx],
                &folding_challenges[family_idx],
                cfg_i,
            )?;
            fq_cpr_ancillaries.push(anc_i);
        }

        // -------- Lockstep multi-degree sumcheck verify ----------------
        // Family 0 = Q[X] (CPR + optional booleanity);
        // Families i >= 1 = per-prime CPR.
        let mut family_proofs: Vec<(&MultiDegreeSumcheckProof<C::Element>, &C)> =
            Vec::with_capacity(add!(n_fq, 1));
        family_proofs.push((&self.proof_combined_sumcheck, &self.field_cfg));
        for prime_idx in 0..n_fq {
            let family_idx = add!(prime_idx, 1);
            family_proofs.push((
                &self.proof_combined_sumchecks_fq[prime_idx],
                &self.all_field_cfgs[family_idx],
            ));
        }
        let all_md_subclaims = MultiDegreeSumcheck::verify_as_subprotocol(
            &mut self.base.pcs_transcript.fs_transcript,
            self.base.num_vars,
            &family_proofs,
            q_star_cfg,
        )
        .map_err(CombinedPolyResolverError::SumcheckError)?;

        // -------- Q[X] family finalize ---------------------------------
        let q_md_subclaims = all_md_subclaims
            .first()
            .expect("Q[X] family subclaim always present");

        let cpr_subclaim = CombinedPolyResolver::finalize_verifier::<U>(
            &mut self.base.pcs_transcript.fs_transcript,
            self.proof_cpr,
            q_md_subclaims.point().to_vec(),
            q_md_subclaims.expected_evaluations()[0].clone(),
            q_cpr_verifier_ancillary,
            &self.projected_scalars_f,
            /* family_idx = */ 0,
            &self.field_cfg,
        )?;

        // Booleanity -> multipoint-eval `alpha_prime` bridge.
        // After `finalize_verifier` (residue check + transcript absorption of
        // `bit_slice_evals`), squeeze `alpha_prime` and carry both forward to the
        // next step.
        let mut bool_group_idx = 1;
        let bool_bit_slice_evals = finalize_booleanity_verifier_group(
            &mut self.base.pcs_transcript.fs_transcript,
            self.proof_booleanity.take(),
            q_md_subclaims.point(),
            q_md_subclaims.expected_evaluations(),
            &mut bool_group_idx,
            bool_verifier_ancillary,
            &self.field_cfg,
        )?;
        let affine_bool_bit_slice_evals = finalize_booleanity_verifier_group(
            &mut self.base.pcs_transcript.fs_transcript,
            self.proof_affine_booleanity.take(),
            q_md_subclaims.point(),
            q_md_subclaims.expected_evaluations(),
            &mut bool_group_idx,
            affine_bool_verifier_ancillary,
            &self.field_cfg,
        )?;

        // -------- Per-prime family finalize ---------------------------
        // Mirror the prover's loop: pop next subclaim per family, call
        // `finalize_verifier` under that family's cfg + projected scalars.
        // Capture per-family CPR subclaims so step 5 (lockstep MP-eval)
        // can feed them per family.
        let n_fq = self.all_field_cfgs.len().saturating_sub(1);
        let mut cpr_eval_points_fq: Vec<Vec<C::Element>> = Vec::with_capacity(n_fq);
        let mut cpr_up_evals_fq: Vec<Vec<C::Element>> = Vec::with_capacity(n_fq);
        let mut cpr_bit_op_evals_fq: Vec<Vec<C::Element>> = Vec::with_capacity(n_fq);
        let mut cpr_down_evals_fq: Vec<Vec<C::Element>> = Vec::with_capacity(n_fq);
        for (prime_idx, ((cpr_proof_i, cpr_ancillary_i), md_subclaims_i)) in self
            .proof_cpr_fq
            .into_iter()
            .zip(fq_cpr_ancillaries)
            .zip(all_md_subclaims.iter().skip(1))
            .enumerate()
        {
            let family_idx = add!(prime_idx, 1);
            let cfg_i = &self.all_field_cfgs[family_idx];
            let cpr_subclaim_i = CombinedPolyResolver::finalize_verifier::<U>(
                &mut self.base.pcs_transcript.fs_transcript,
                cpr_proof_i,
                md_subclaims_i.point().to_vec(),
                md_subclaims_i.expected_evaluations()[0].clone(),
                cpr_ancillary_i,
                &self.projected_scalars_f_fq[prime_idx],
                family_idx,
                cfg_i,
            )?;
            cpr_eval_points_fq.push(cpr_subclaim_i.evaluation_point);
            cpr_up_evals_fq.push(cpr_subclaim_i.up_evals);
            cpr_bit_op_evals_fq.push(cpr_subclaim_i.bit_op_evals);
            cpr_down_evals_fq.push(cpr_subclaim_i.down_evals);
        }

        // Squeeze alpha_prime in the same transcript order as the prover.
        let alpha_prime_f: Option<C::Element> =
            (bool_bit_slice_evals.is_some() || affine_bool_bit_slice_evals.is_some()).then(|| {
                self.base
                    .pcs_transcript
                    .fs_transcript
                    .get_field_challenge(&self.field_cfg)
            });

        let cpr_eval_point = cpr_subclaim.evaluation_point;

        assert!(
            self.proof_lookup_proof.is_none(),
            "Arbitrary lookup argument is not supported yet!"
        );

        Ok(VerifierSumchecked {
            base: self.base,
            field_cfg: self.field_cfg,
            all_field_cfgs: self.all_field_cfgs,
            q_star_idx: self.q_star_idx,
            projecting_elements: self.projecting_elements,
            cpr_eval_point,
            cpr_up_evals: cpr_subclaim.up_evals,
            cpr_bit_op_evals: cpr_subclaim.bit_op_evals,
            cpr_down_evals: cpr_subclaim.down_evals,
            cpr_eval_points_fq,
            cpr_up_evals_fq,
            cpr_bit_op_evals_fq,
            cpr_down_evals_fq,
            bool_bit_slice_evals,
            affine_bool_bit_slice_evals,
            alpha_prime_f,
            proof_commitments: self.proof_commitments,
            proof_multipoint_eval: self.proof_multipoint_eval,
            proof_multipoint_evals_fq: self.proof_multipoint_evals_fq,
            proof_witness_lifted_evals: self.proof_witness_lifted_evals,
            proof_witness_lifted_evals_pp: self.proof_witness_lifted_evals_pp,
            proof_lookup_proof: self.proof_lookup_proof,
            _phantom: PhantomData,
        })
    }
}

impl<'a, Zt, C, IdealOverF, const D: usize, const FD: usize>
    VerifierSumchecked<'a, Zt, C, IdealOverF, D, FD>
where
    Zt: ZincTypes<D, FD>,
    C: BaseFieldConfig<Integer = Zt::Fmod>
        + ProjectPrimitiveIntegersWithConfig
        + Clone
        + Send
        + Sync
        + 'static,
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
    ) -> Result<VerifierMultipointEvaled<'a, Zt, C, IdealOverF, D, FD>, ProtocolError<C::Element>>
    {
        // Length-mismatch guard
        let n_fq = self.cpr_eval_points_fq.len();
        if self.proof_multipoint_evals_fq.len() != n_fq {
            return Err(ProtocolError::FqIdealCheck {
                prime_idx: self.proof_multipoint_evals_fq.len(),
                q: "<fq mp-eval sub-proof length mismatch>".to_owned(),
                source: IdealCheckError::IdealCollectorError(
                    ideal_check::BatchedIdealCheckError::LengthMismatch {
                        num_ideals: n_fq,
                        provided_values: self.proof_multipoint_evals_fq.len(),
                    },
                ),
            });
        }

        // Q[X] family up_evals, extended only by the committed-column
        // Booleanity bridge. Affine virtuals stay separate and are checked
        // directly against their source-column projections below.
        let sig = self.base.uair_signature.clone();
        let num_pub_bin = sig.public_cols().num_binary_poly_cols();
        let num_total_bin = sig.total_cols().num_binary_poly_cols();
        let num_wit_bin = num_total_bin.saturating_sub(num_pub_bin);
        let num_affine_virtuals = sig.affine_virtual_specs().len();
        let alpha_prime = self.alpha_prime_f.as_ref();

        let witness_bridge_evals = if num_wit_bin == 0 {
            Vec::new()
        } else {
            collapse_bit_slice_evals::<C, D>(
                self.bool_bit_slice_evals
                    .as_ref()
                    .ok_or(ProtocolError::BooleanityProofMissing)?,
                num_wit_bin,
                alpha_prime.ok_or(ProtocolError::BooleanityProofMissing)?,
                &self.field_cfg,
            )
        };

        if num_affine_virtuals > 0 {
            let alpha_prime = alpha_prime.ok_or(ProtocolError::BooleanityProofMissing)?;
            let alpha_powers = powers(&self.field_cfg, alpha_prime, D);
            let mut source_bridge_evals: Vec<C::Element> = self
                .base
                .public_trace
                .binary_poly
                .iter()
                .map(|col| {
                    project_binary_col_at_field::<C, D>(col, &alpha_powers, &self.field_cfg)
                        .evaluate(&self.field_cfg, &self.cpr_eval_point)
                        .map_err(ProtocolError::AffineVirtualSourceProjection)
                })
                .collect::<Result<_, _>>()?;
            debug_assert_eq!(source_bridge_evals.len(), num_pub_bin);
            source_bridge_evals.extend(witness_bridge_evals.iter().cloned());

            let affine_bridge_evals = collapse_bit_slice_evals::<C, D>(
                self.affine_bool_bit_slice_evals
                    .as_ref()
                    .ok_or(ProtocolError::BooleanityProofMissing)?,
                num_affine_virtuals,
                alpha_prime,
                &self.field_cfg,
            );
            let expected_affine_evals = expected_affine_virtual_bridge_evals::<C, _, D>(
                &sig,
                &source_bridge_evals,
                alpha_prime,
                &self.field_cfg,
            );
            for (spec_index, (got, expected)) in affine_bridge_evals
                .into_iter()
                .zip(expected_affine_evals)
                .enumerate()
            {
                if got != expected {
                    return Err(ProtocolError::AffineVirtualBridgeMismatch {
                        spec_index,
                        got,
                        expected,
                    });
                }
            }
        }

        let q_up_evals: Vec<C::Element> = self
            .cpr_up_evals
            .iter()
            .cloned()
            .chain(witness_bridge_evals)
            .collect();

        // Per-prime families: zero-pad up_evals to match Q's column count
        //
        // Witness binary-poly columns live in Q[X] only.
        // Pad F_q[X] evals with zeros to share one lockstep `gammas` vector and one
        // shared r_0 across all families.
        let fq_up_evals_ext: Vec<Vec<C::Element>> = (0..n_fq)
            .map(|prime_idx| {
                let family_idx = add!(prime_idx, 1);
                let zero_i = self.all_field_cfgs[family_idx].zero();
                let mut up_i = self.cpr_up_evals_fq[prime_idx].clone();
                up_i.extend((0..num_wit_bin).map(|_| zero_i.clone()));
                up_i
            })
            .collect();

        let shifts = self.base.uair_signature.shifts();
        let num_vars = self.base.num_vars;
        let q_star_cfg = self.all_field_cfgs[self.q_star_idx].clone();

        // Single (n+1)-family lockstep MP-eval verifier
        let mut all_proofs: Vec<multipoint_eval::Proof<C::Element>> =
            Vec::with_capacity(add!(n_fq, 1));
        all_proofs.push(self.proof_multipoint_eval);
        all_proofs.extend(self.proof_multipoint_evals_fq);

        let mut all_families: Vec<MultipointEvalFamilyInputs<'_, C>> =
            Vec::with_capacity(add!(n_fq, 1));
        // The verifier-side MP precheck only consumes the claimed eval vectors;
        // bit-op MLE bodies exist on the prover side only.
        all_families.push(MultipointEvalFamilyInputs {
            field_cfg: &self.field_cfg,
            trace_mles: &[],
            bit_op_mles: &[],
            eval_point: &self.cpr_eval_point,
            up_evals: &q_up_evals,
            bit_op_evals: &self.cpr_bit_op_evals,
            down_evals: &self.cpr_down_evals,
        });

        for (prime_idx, up_evals) in fq_up_evals_ext.iter().enumerate() {
            let family_idx = add!(prime_idx, 1);
            all_families.push(MultipointEvalFamilyInputs {
                field_cfg: &self.all_field_cfgs[family_idx],
                trace_mles: &[],
                bit_op_mles: &[],
                eval_point: &self.cpr_eval_points_fq[prime_idx],
                up_evals,
                bit_op_evals: &self.cpr_bit_op_evals_fq[prime_idx],
                down_evals: &self.cpr_down_evals_fq[prime_idx],
            });
        }

        let mut subclaims_iter = MultipointEval::verify_as_subprotocol(
            &mut self.base.pcs_transcript.fs_transcript,
            all_proofs,
            all_families,
            shifts,
            num_vars,
            &q_star_cfg,
        )?
        .into_iter();

        let mp_subclaim = subclaims_iter.next().expect("Q-family subclaim present");
        let mp_subclaims_fq: Vec<multipoint_eval::Subclaim<C::Element>> = subclaims_iter.collect();
        debug_assert_eq!(mp_subclaims_fq.len(), n_fq);

        Ok(VerifierMultipointEvaled {
            base: self.base,
            field_cfg: self.field_cfg,
            all_field_cfgs: self.all_field_cfgs,
            projecting_elements: self.projecting_elements,
            alpha_prime_f: self.alpha_prime_f,
            mp_subclaim,
            mp_subclaims_fq,
            proof_commitments: self.proof_commitments,
            proof_witness_lifted_evals: self.proof_witness_lifted_evals,
            proof_witness_lifted_evals_pp: self.proof_witness_lifted_evals_pp,
            proof_lookup_proof: self.proof_lookup_proof,
            _phantom: PhantomData,
        })
    }
}

impl<'a, Zt, C, IdealOverF, const D: usize, const FD: usize>
    VerifierMultipointEvaled<'a, Zt, C, IdealOverF, D, FD>
where
    Zt: ZincTypes<D, FD>,
    Zt::Int: ProjectableToField<C>,
    <Zt::ArbitraryZt as ZipTypes>::Eval: ProjectableToField<C>,
    C: BaseFieldConfig<Integer = Zt::Fmod>
        + ProjectPrimitiveIntegersWithConfig
        + ProjectElementWithConfig<Zt::Int>
        + ProjectElementWithConfig<Zt::Chal>
        + Clone
        + Send
        + Sync
        + 'static,
    IdealOverF: Ideal,
{
    /// Step 6: Per-family lifted-eval consistency check.
    ///
    /// **When $F_q[X]$ constraints are present:**
    ///
    /// 1. **Sample $q''$** (mirror of prover step 7 start).
    /// 2. For each family $i \in \{0, ..., n\}$ (Q + declared primes):
    ///    - Recompute the public-column lifted evals under family $i$'s cfg,
    ///      interleave with the sent witness-only lifted evals to get the full
    ///      all-column lifted-eval list `all_lifted_evals[i]`.
    ///    - Project each lifted eval at `projecting_elements[i]` to get
    ///      `open_evals_i`.
    ///    - For Q-family (i = 0) when booleanity ran: append $\alpha'$
    ///      projections of witness-bin lifts.
    ///    - For F_q families (i >= 1): append zero entries (matching the
    ///      zero-padded MP `up_evals` on F_q families).
    ///    - Call `MultipointEval::verify_subclaim` against the matching MP
    ///      subclaim, family by family.
    /// 3. Also recompute the $q''$-family lifted evals and absorb every
    ///    family's coefficients into the FS transcript in the same order as the
    ///    prover.
    ///
    /// **When only $Q[X]$ constraints are present:**
    ///
    /// We alias $q'' := q_0$ and $r^* := r_0$ instead of sampling $q''$
    /// randomly. The rest proceeds as described above, but we use
    /// `proof_witness_lifted_evals[0]` instead of
    /// `proof_witness_lifted_evals_pp`.
    ///
    /// **Soundness**: per-family arithmetic in single prime fields. With
    /// the shared $r_0$ integer endpoint, the $n + 1$ constraint-family
    /// `verify_subclaim` calls each bind their family's lifted eval to
    /// that family's residue of the committed trace; the $q''$-family
    /// lifted eval is bound independently by the PCS open in step 7.
    #[allow(clippy::arithmetic_side_effects, clippy::too_many_lines)]
    pub fn step6_lifted_evals<U: Uair>(
        mut self,
    ) -> Result<VerifierLiftedEvalsChecked<'a, Zt, C, IdealOverF, D, FD>, ProtocolError<C::Element>>
    {
        let n_fq = self.mp_subclaims_fq.len();
        // Expected length of `proof_witness_lifted_evals`: one entry per
        // constraint family (Q-family at index 0 plus the `n_fq` declared
        // primes at indices 1..=n_fq).
        let n_families = add!(n_fq, 1);

        // Length-mismatch guard.
        if self.proof_witness_lifted_evals.len() != n_families {
            return Err(ProtocolError::FqIdealCheck {
                prime_idx: self.proof_witness_lifted_evals.len(),
                q: "<witness-lifted-evals family count mismatch>".to_owned(),
                source: IdealCheckError::IdealCollectorError(
                    ideal_check::BatchedIdealCheckError::LengthMismatch {
                        num_ideals: n_families,
                        provided_values: self.proof_witness_lifted_evals.len(),
                    },
                ),
            });
        }

        let pub_cols = self.base.uair_signature.public_cols();
        let num_pub_bin = pub_cols.num_binary_poly_cols();
        let num_pub_arb = pub_cols.num_arbitrary_poly_cols();
        let num_pub_int = pub_cols.num_int_cols();

        let wit_cols = self.base.uair_signature.witness_cols();
        let num_wit_bin = wit_cols.num_binary_poly_cols();
        let num_wit_arb = wit_cols.num_arbitrary_poly_cols();
        let num_wit_int = wit_cols.num_int_cols();

        // Since the entire vector is absorbed into the FS transcript below,
        // prevent a malicious prover from adding entries not tied to a
        // commitment.
        let num_wit_total = add!(add!(num_wit_bin, num_wit_arb), num_wit_int);

        for (family_idx, witness_lifted_i) in self.proof_witness_lifted_evals.iter().enumerate() {
            if witness_lifted_i.len() != num_wit_total {
                return Err(ProtocolError::WitnessLiftedEvalsLengthMismatch {
                    family_idx,
                    got: witness_lifted_i.len(),
                    expected: num_wit_total,
                });
            }
        }

        let expected_pp_len = if n_fq == 0 { 0 } else { num_wit_total };
        let actual_pp_len = self
            .proof_witness_lifted_evals_pp
            .as_ref()
            .map(|v| v.len())
            .unwrap_or(0);
        if actual_pp_len != expected_pp_len {
            return Err(ProtocolError::WitnessLiftedEvalsPpLengthMismatch {
                got: actual_pp_len,
                expected: expected_pp_len,
            });
        }

        // Sample q'' (mirror of prover step 7 start)
        //
        // With no F_q[X] constraints, q'' is aliased to q_0 and r* = r0.
        let q_pp_cfg = if n_fq == 0 {
            self.field_cfg.clone()
        } else {
            self.base
                .pcs_transcript
                .fs_transcript
                .get_random_field_cfg::<C, Zt::Fmod, Zt::PrimeTest>()
        };

        // $q''$ is now known: project the last wire-form proof section,
        // rejecting non-canonical encodings.
        let proof_witness_lifted_evals_pp = self
            .proof_witness_lifted_evals_pp
            .as_ref()
            .map(|polys| {
                polys
                    .iter()
                    .map(|poly| poly.try_map(|int| project_canonical(&q_pp_cfg, int)))
                    .collect::<Result<Vec<_>, _>>()
            })
            .transpose()?;
        // r* = r0 mod q'' (the underlying integer of mp_subclaim.r0 is
        // the shared challenge in [0, q*); lift into q'' cfg).
        // Aliased to r0 when q'' = q0.
        let r_star: Vec<C::Element> = if n_fq == 0 {
            self.mp_subclaim.r0.clone()
        } else {
            self.mp_subclaim
                .r0
                .iter()
                .map(|x| q_pp_cfg.project(&self.field_cfg.lift(x)))
                .collect()
        };

        // Helper: assemble all-column lifted evals from per-family public
        // (recomputed) + sent witness lifts.
        let assemble_all = |public_lifted: &[DynamicPolynomial<C::Element>],
                            witness_lifted: &[DynamicPolynomial<C::Element>]|
         -> Vec<DynamicPolynomial<C::Element>> {
            public_lifted[..num_pub_bin]
                .iter()
                .chain(&witness_lifted[..num_wit_bin])
                .chain(&public_lifted[num_pub_bin..add!(num_pub_bin, num_pub_arb)])
                .chain(&witness_lifted[num_wit_bin..add!(num_wit_bin, num_wit_arb)])
                .chain(&public_lifted[add!(num_pub_bin, num_pub_arb)..])
                .chain(&witness_lifted[add!(num_wit_bin, num_wit_arb)..])
                .cloned()
                .collect()
        };

        // Helper: recompute public-only lifted evals under family i's cfg
        // at family i's r_0 endpoint.
        let recompute_public_lifted =
            |family_r0: &[C::Element], family_cfg: &C| -> Vec<DynamicPolynomial<C::Element>> {
                if add!(add!(num_pub_bin, num_pub_arb), num_pub_int) == 0 {
                    return Vec::new();
                }
                let projected_public = project_trace_coeffs_row_major::<C, Zt::Int, Zt::Int, D, D>(
                    self.base.public_trace,
                    family_cfg,
                );
                compute_lifted_evals::<C, D>(
                    family_r0,
                    &self.base.public_trace.binary_poly,
                    &ProjectedTrace::RowMajor(projected_public),
                    family_cfg,
                )
            };

        // Q-family (i = 0)
        let q_r_0 = self.mp_subclaim.r0.clone();
        let q_public_lifted = recompute_public_lifted(&q_r_0, &self.field_cfg);
        let q_witness_lifted = self.proof_witness_lifted_evals[0].clone();
        let q_all_lifted = assemble_all(&q_public_lifted, &q_witness_lifted);

        let q_poly_cfg = self.field_cfg.dyn_poly_cfg();
        let mut q_open_evals: Vec<C::Element> = q_all_lifted
            .iter()
            .map(|bar_u| {
                q_poly_cfg
                    .evaluate_at_point(bar_u, &self.projecting_elements[0])
                    .map_err(ProtocolError::LiftedEvalProjection)
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Booleanity bridge: append $\alpha'$-projected witness-bin lifts.
        if let Some(alpha_prime) = &self.alpha_prime_f {
            for bar_u in &q_witness_lifted[..num_wit_bin] {
                q_open_evals.push(
                    q_poly_cfg
                        .evaluate_at_point(bar_u, alpha_prime)
                        .map_err(ProtocolError::LiftedEvalProjection)?,
                );
            }
        }

        MultipointEval::verify_subclaim(
            &self.mp_subclaim,
            &q_open_evals,
            &derive_bit_op_open_evals::<C, D>(
                self.base.uair_signature.bit_op_specs(),
                &q_all_lifted,
                &self.projecting_elements[0],
                &self.field_cfg,
            )?,
            self.base.uair_signature.shifts(),
            &self.field_cfg,
        )?;

        // Per-prime families (i >= 1)
        for prime_idx in 0..n_fq {
            let family_idx = add!(prime_idx, 1);
            let cfg_i = &self.all_field_cfgs[family_idx];
            let r_0_i = self.mp_subclaims_fq[prime_idx].r0.clone();
            let witness_lifted_i = &self.proof_witness_lifted_evals[family_idx];
            let public_lifted_i = recompute_public_lifted(&r_0_i, cfg_i);
            let all_lifted_i = assemble_all(&public_lifted_i, witness_lifted_i);

            let mut open_evals_i: Vec<C::Element> = all_lifted_i
                .iter()
                .map(|bar_u| {
                    cfg_i
                        .dyn_poly_cfg()
                        .evaluate_at_point(bar_u, &self.projecting_elements[family_idx])
                        .map_err(ProtocolError::LiftedEvalProjection)
                })
                .collect::<Result<Vec<_>, _>>()?;

            // Booleanity-bridge slots on fq families are zero-padded
            // (witness-bin lives in Q[X] only).
            if self.alpha_prime_f.is_some() {
                let zero_i = cfg_i.zero();
                open_evals_i.extend((0..num_wit_bin).map(|_| zero_i.clone()));
            }

            MultipointEval::verify_subclaim(
                &self.mp_subclaims_fq[prime_idx],
                &open_evals_i,
                &derive_bit_op_open_evals::<C, D>(
                    self.base.uair_signature.bit_op_specs(),
                    &all_lifted_i,
                    &self.projecting_elements[family_idx],
                    cfg_i,
                )?,
                self.base.uair_signature.shifts(),
                cfg_i,
            )?;
        }

        // Absorb all families' coefficients into the FS transcript in the same uniform
        // order as the prover
        let mut transcription_buf: Vec<u8> = vec![0; C::Integer::NUM_BYTES];
        debug_assert_eq!(
            self.all_field_cfgs.len(),
            self.proof_witness_lifted_evals.len()
        );
        for (cfg_i, witness_lifted_i) in self
            .all_field_cfgs
            .iter()
            .zip(&self.proof_witness_lifted_evals)
        {
            for bar_u in witness_lifted_i {
                self.base
                    .pcs_transcript
                    .fs_transcript
                    .absorb_field_element_slice(cfg_i, &bar_u.coeffs, &mut transcription_buf);
            }
        }
        if let Some(ref lifted_evals_pp) = proof_witness_lifted_evals_pp {
            for bar_u in lifted_evals_pp.iter() {
                self.base
                    .pcs_transcript
                    .fs_transcript
                    .absorb_field_element_slice(&q_pp_cfg, &bar_u.coeffs, &mut transcription_buf);
            }
        }

        // When q'' was aliased to q0, the PCS check reuses Q[X] family's witness lift.
        let lifted_evals_pp = if n_fq == 0 {
            self.proof_witness_lifted_evals[0].clone()
        } else {
            let Some(lifted_evals_pp) = proof_witness_lifted_evals_pp else {
                return Err(ProtocolError::WitnessLiftedEvalsPpLengthMismatch {
                    got: 0,
                    expected: expected_pp_len,
                });
            };
            lifted_evals_pp
        };

        Ok(VerifierLiftedEvalsChecked {
            base: self.base,
            lifted_evals_pp,
            q_pp_cfg,
            r_star,
            proof_commitments: self.proof_commitments,
            proof_lookup_proof: self.proof_lookup_proof,
            _phantom: PhantomData,
        })
    }
}

impl<'a, Zt, C, IdealOverF, const D: usize, const FD: usize>
    VerifierLiftedEvalsChecked<'a, Zt, C, IdealOverF, D, FD>
where
    Zt: ZincTypes<D, FD>,
    C: BaseFieldConfig<Integer = Zt::Fmod>
        + ProjectPrimitiveIntegersWithConfig
        + ProjectElementWithConfig<Zt::CombR>
        + ProjectElementWithConfig<Zt::Chal>
        + Clone
        + Send
        + Sync
        + 'static,
    IdealOverF: Ideal,
{
    /// Step 7: PCS verification at $r^\star := r_0 \bmod q''$,
    /// using the $q''$-family lifted evals sent by the prover and
    /// recovered into `lifted_evals_pp` during step 6.
    ///
    /// $q''$ was sampled at the start of step 6.
    /// Here we sample binary folding challenges under $q''$ and run the
    /// per-commitment PCS verify.
    ///
    /// Per-poly claims for `verify_with_alphas` are computed directly from
    /// `lifted_evals_pp[witness_range]` — no per-coefficient $\phi_{q''}$
    /// lift is needed because the prover already sent each $\bar
    /// u_j^{(q'')} \in F_{q''}[X]$.
    pub fn step7_pcs_verify<U: Uair, const CHECK_FOR_OVERFLOW: bool>(
        mut self,
    ) -> Result<VerifierPcsVerified<IdealOverF>, ProtocolError<C::Element>> {
        let commitments = &self.proof_commitments;

        let wit_cols = self.base.uair_signature.witness_cols();
        let num_wit_bin = wit_cols.num_binary_poly_cols();
        let num_wit_arb = wit_cols.num_arbitrary_poly_cols();

        let pcs_transcript = &mut self.base.pcs_transcript;
        let lifted_evals_pp = &self.lifted_evals_pp;
        let q_pp_cfg = &self.q_pp_cfg;
        let r_star = &self.r_star;

        let zero = q_pp_cfg.zero();

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
                    |bar_u: &DynamicPolynomial<C::Element>, alphas: &[_]| {
                        let mut eval_j = zero.clone();
                        for (coeff, alpha) in bar_u.coeffs.iter().zip(alphas.iter()) {
                            // bar_u is already in F_{q''}; just batch with alphas.
                            let term = q_pp_cfg.mul(&q_pp_cfg.project(alpha), coeff);
                            q_pp_cfg.add_assign(&mut eval_j, &term);
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
                    for (bar_u, alphas) in lifted_evals_pp[$evals_range]
                        .iter()
                        .zip(per_poly_alphas.iter())
                    {
                        let eval_j = $compute_eval_j(bar_u, alphas);
                        q_pp_cfg.add_assign(&mut eval_f, &eval_j);
                    }
                    ZipPlus::<$Zt, $Lc>::verify_with_alphas::<C, CHECK_FOR_OVERFLOW>(
                        pcs_transcript,
                        $vp,
                        comm,
                        q_pp_cfg,
                        $pt,
                        &eval_f,
                        &per_poly_alphas,
                    )
                    .map_err(|e| ProtocolError::PcsVerification($idx, e))?;
                }
            }};
        }

        // Folded witness columns are proved using the extended evaluation
        // point `r_star_ext = r_star || folding_challenges`.
        // Folding challenges are sampled fresh under q''.
        let num_folding_challenges = Zt::BinaryFold::FOLDING_FACTOR.ilog2();
        let folding_challenges = (0..num_folding_challenges)
            .map(|_| {
                let g_chal: Zt::Chal = pcs_transcript.fs_transcript.get_challenge();
                q_pp_cfg.project(&g_chal)
            })
            .collect_vec();
        let mut r_star_ext = r_star.clone();
        r_star_ext.extend_from_slice(&folding_challenges);

        // Witness-only ranges inside `lifted_evals_pp` (layout
        // `[wit_bin..., wit_arb..., wit_int...]`, same as `witness_only`
        // in prover step 7).
        verify_pcs_batch!(
            Zt::BinaryZt,
            Zt::BinaryLc,
            self.base.vp_bin,
            0,
            &r_star_ext,
            [0..num_wit_bin],
            |bar_u: &DynamicPolynomial<C::Element>, alphas: &[_]| {
                // bar_u is already in F_{q''}; use coeffs directly.
                Zt::BinaryFold::fold_eval_claim(
                    &bar_u.coeffs,
                    alphas,
                    &folding_challenges,
                    q_pp_cfg,
                )
            }
        );
        verify_pcs_batch!(
            Zt::ArbitraryZt,
            Zt::ArbitraryLc,
            self.base.vp_arb,
            1,
            r_star,
            [num_wit_bin..add!(num_wit_bin, num_wit_arb)]
        );
        verify_pcs_batch!(
            Zt::IntZt,
            Zt::IntLc,
            self.base.vp_int,
            2,
            r_star,
            [add!(num_wit_bin, num_wit_arb)..]
        );

        Ok(VerifierPcsVerified {
            pcs_transcript: self.base.pcs_transcript,
            _phantom: PhantomData,
        })
    }
}

impl<IdealOverF: Ideal> VerifierPcsVerified<IdealOverF> {
    /// Complete verification.
    ///
    /// Asserts that the proof stream has been fully consumed.
    pub fn finish<E: SetElement>(self) -> Result<(), ProtocolError<E>> {
        self.pcs_transcript.check_eof()?;
        Ok(())
    }
}

fn derive_bit_op_open_evals<C: BaseFieldConfig, const D: usize>(
    specs: &[zinc_uair::BitOpSpec],
    all_lifted: &[DynamicPolynomial<C::Element>],
    projecting_element: &C::Element,
    field_cfg: &C,
) -> Result<Vec<C::Element>, ProtocolError<C::Element>> {
    specs
        .iter()
        .map(|spec| {
            let source = &all_lifted[spec.source_col()];
            let transformed = spec.op().transform::<C, D>(source, field_cfg);
            field_cfg
                .dyn_poly_cfg()
                .evaluate_at_point(&transformed, projecting_element)
                .map_err(ProtocolError::LiftedEvalProjection)
        })
        .collect()
}

//
// verify() wrapper
//

impl<Zt, U, C, const D: usize, const FD: usize> ZincPlusPiop<Zt, U, C, D, FD>
where
    Zt: ZincTypes<D, FD>,
    Zt::Int: ProjectableToField<C>,
    <Zt::ArbitraryZt as ZipTypes>::Eval: ProjectableToField<C>,
    C: BaseFieldConfig<Integer = Zt::Fmod>
        + ProjectPrimitiveIntegersWithConfig
        + ProjectElementWithConfig<Zt::Int>
        + ProjectElementWithConfig<Zt::CombR>
        + ProjectElementWithConfig<Zt::Chal>
        + Clone
        + Send
        + Sync
        + 'static,
    U: Uair<Prime = Zt::Fmod> + 'static,
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
        proof: Proof<Zt::Fmod>,
        public_trace: &UairTrace<Zt::Int, Zt::Int, D, D>,
        num_vars: usize,
        project_scalar: impl Fn(&U::Scalar, &C) -> DynamicPolynomial<C::Element>,
        project_ideal: impl Fn(&IdealOrZero<U::Ideal>, &C) -> IdealOverF,
        project_fq_ideal: impl Fn(&IdealOrZero<U::FqIdeal>, &C) -> IdealOverF,
    ) -> Result<(), ProtocolError<C::Element>>
    where
        IdealOverF: Ideal + for<'cfg> IdealCheck<DynamicPolynomialConfig<'cfg, C>>,
    {
        ZincPlusPiop::<Zt, U, C, D, FD>::step0_reconstruct_transcript::<IdealOverF>(
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
        .finish::<C::Element>()
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
    impl<'a, Zt, U, C, IdealOverF, const D: usize, const FD: usize>
        VerifierEvalProjected<'a, Zt, U, C, IdealOverF, D, FD>
    where
        Zt: ZincTypes<D, FD>,
        C: BaseFieldConfig,
        U: Uair,
    {
        pub fn projecting_element_f(&self) -> &C::Element {
            &self.projecting_elements[0]
        }

        pub fn field_cfg(&self) -> &C {
            &self.field_cfg
        }

        pub fn fs_transcript_mut(&mut self) -> &mut Blake3Transcript {
            self.base.fs_transcript_mut()
        }

        pub fn proof_combined_sumcheck(&self) -> &MultiDegreeSumcheckProof<C::Element> {
            &self.proof_combined_sumcheck
        }

        pub fn ic_subclaim(&self) -> &ideal_check::VerifierSubclaim<C::Element> {
            &self.ic_subclaims[0]
        }

        pub fn proof_cpr(&self) -> &CombinedPolyResolverProof<C::Element> {
            &self.proof_cpr
        }

        pub fn num_vars(&self) -> usize {
            self.base.num_vars
        }

        pub fn uair_signature(&self) -> &UairSignature<Zt::Fmod> {
            &self.base.uair_signature
        }
    }
}
