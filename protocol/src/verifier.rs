use super::*;
use crypto_primitives::{ConstIntSemiring, FromPrimitiveWithConfig, FromWithConfig};
use itertools::Itertools;
use num_traits::Zero;
use std::{collections::HashMap, io::Cursor};
use zinc_piop::{
    combined_poly_resolver::{self, CombinedPolyResolver},
    ideal_check::{self, IdealCheckProtocol},
    lookup::booleanity::{BoolVerifierSubclaim, BooleanityChecker, BooleanityProof},
    multipoint_eval::{self, MultipointEval},
    projections::{
        ProjectedTrace, project_scalars, project_scalars_to_field, project_trace_coeffs_row_major,
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
    proof_resolver: CombinedPolyResolverProof<F>,
    proof_combined_sumcheck: MultiDegreeSumcheckProof<F>,
    proof_multipoint_eval: MultipointEvalProof<F>,
    proof_witness_lifted_evals: Vec<DynamicPolynomialF<F>>,
    proof_lookup_proof: Option<BatchedLookupProof<F>>,
    proof_booleanity: Option<BooleanityProof<F>>,
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
    proof_resolver: CombinedPolyResolverProof<F>,
    proof_combined_sumcheck: MultiDegreeSumcheckProof<F>,
    proof_multipoint_eval: MultipointEvalProof<F>,
    proof_witness_lifted_evals: Vec<DynamicPolynomialF<F>>,
    proof_lookup_proof: Option<BatchedLookupProof<F>>,
    proof_booleanity: Option<BooleanityProof<F>>,
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
    ic_subclaim: ideal_check::VerifierSubclaim<F>,

    // Proof leftovers
    proof_commitments: (ZipPlusCommitment, ZipPlusCommitment, ZipPlusCommitment),
    proof_resolver: CombinedPolyResolverProof<F>,
    proof_combined_sumcheck: MultiDegreeSumcheckProof<F>,
    proof_multipoint_eval: MultipointEvalProof<F>,
    proof_witness_lifted_evals: Vec<DynamicPolynomialF<F>>,
    proof_lookup_proof: Option<BatchedLookupProof<F>>,
    proof_booleanity: Option<BooleanityProof<F>>,
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
    ic_subclaim: ideal_check::VerifierSubclaim<F>,
    projecting_element_f: F,
    projected_scalars_f: HashMap<U::Scalar, F>,

    // Proof leftovers
    proof_commitments: (ZipPlusCommitment, ZipPlusCommitment, ZipPlusCommitment),
    proof_resolver: CombinedPolyResolverProof<F>,
    proof_combined_sumcheck: MultiDegreeSumcheckProof<F>,
    proof_multipoint_eval: MultipointEvalProof<F>,
    proof_witness_lifted_evals: Vec<DynamicPolynomialF<F>>,
    proof_lookup_proof: Option<BatchedLookupProof<F>>,
    proof_booleanity: Option<BooleanityProof<F>>,
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
    projecting_element_f: F,
    cpr_subclaim: combined_poly_resolver::VerifierSubclaim<F>,
    bool_subclaim: Option<BoolVerifierSubclaim<F>>,

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
    projecting_element_f: F,
    mp_subclaim: multipoint_eval::Subclaim<F>,
    /// Indicates the booleanity argument participated in the sumcheck,
    /// so step 6 must append D coefficients per witness binary-poly col
    /// to `open_evals`.
    booleanity_present: bool,

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
    F::Inner: ConstTranscribable,
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
    ) -> Result<
        VerifierTranscriptReconstructed<'a, Zt, U, F, IdealOverF, D, FD>,
        ProtocolError<F, IdealOverF>,
    >
    where
        IdealOverF: Ideal,
    {
        let zip_proof = std::mem::take(&mut proof.zip);
        let mut base = VerifierBase {
            num_vars,
            uair_signature: U::signature(),
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
            proof_resolver: proof.resolver,
            proof_combined_sumcheck: proof.combined_sumcheck,
            proof_multipoint_eval: proof.multipoint_eval,
            proof_witness_lifted_evals: proof.witness_lifted_evals,
            proof_lookup_proof: proof.lookup_proof,
            proof_booleanity: proof.booleanity_proof,
            _phantom: PhantomData,
        })
    }
}

impl<'a, Zt, U, F, IdealOverF, const D: usize, const FD: usize>
    VerifierTranscriptReconstructed<'a, Zt, U, F, IdealOverF, D, FD>
where
    Zt: ZincTypes<D, FD>,
    F: InnerTransparentField + FromPrimitiveWithConfig + FromRef<F> + Send + Sync + 'static,
    F::Inner: ConstIntSemiring + ConstTranscribable + Send + Sync + Zero + Default,
    F::Modulus: ConstTranscribable + FromRef<Zt::Fmod>,
    U: Uair,
    IdealOverF: Ideal,
{
    /// Step 1: Prime projection. Samples the random field configuration.
    #[allow(clippy::type_complexity)]
    pub fn step1_prime_projection(
        mut self,
    ) -> Result<VerifierPrimeProjected<'a, Zt, U, F, IdealOverF, D, FD>, ProtocolError<F, IdealOverF>>
    {
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
            proof_resolver: self.proof_resolver,
            proof_combined_sumcheck: self.proof_combined_sumcheck,
            proof_multipoint_eval: self.proof_multipoint_eval,
            proof_witness_lifted_evals: self.proof_witness_lifted_evals,
            proof_lookup_proof: self.proof_lookup_proof,
            proof_booleanity: self.proof_booleanity,
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
    F::Inner: ConstIntSemiring + ConstTranscribable + Send + Sync + Zero + Default,
    F::Modulus: ConstTranscribable + FromRef<Zt::Fmod>,
    U: Uair + 'static,
    IdealOverF: Ideal + IdealCheck<DynamicPolynomialF<F>>,
{
    /// Step 2: Ideal check verification. Consumes `project_ideal`.
    #[allow(clippy::type_complexity)]
    pub fn step2_ideal_check(
        mut self,
        project_ideal: impl Fn(&IdealOrZero<U::Ideal>, &F::Config) -> IdealOverF,
    ) -> Result<VerifierIdealChecked<'a, Zt, U, F, IdealOverF, D, FD>, ProtocolError<F, IdealOverF>>
    {
        let num_constraints = count_constraints::<U>();

        let ic_subclaim = U::verify_as_subprotocol::<_, IdealOverF, _>(
            &mut self.base.pcs_transcript.fs_transcript,
            self.proof_ideal_check,
            num_constraints,
            self.base.num_vars,
            |ideal| project_ideal(ideal, &self.field_cfg),
            &self.field_cfg,
        )?;

        Ok(VerifierIdealChecked {
            base: self.base,
            field_cfg: self.field_cfg,
            ic_subclaim,
            proof_commitments: self.proof_commitments,
            proof_resolver: self.proof_resolver,
            proof_combined_sumcheck: self.proof_combined_sumcheck,
            proof_multipoint_eval: self.proof_multipoint_eval,
            proof_witness_lifted_evals: self.proof_witness_lifted_evals,
            proof_lookup_proof: self.proof_lookup_proof,
            proof_booleanity: self.proof_booleanity,
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
    F::Inner: ConstIntSemiring + ConstTranscribable + Send + Sync + Zero + Default,
    F::Modulus: ConstTranscribable + FromRef<Zt::Fmod>,
    U: Uair + 'static,
    IdealOverF: Ideal,
{
    /// Step 3: Evaluation projection. Consumes `project_scalar`.
    pub fn step3_eval_projection(
        mut self,
        project_scalar: impl Fn(&U::Scalar, &F::Config) -> DynamicPolynomialF<F>,
    ) -> Result<VerifierEvalProjected<'a, Zt, U, F, IdealOverF, D, FD>, ProtocolError<F, IdealOverF>>
    {
        let projecting_element: Zt::Chal = self.base.pcs_transcript.fs_transcript.get_challenge();
        let projecting_element_f: F = F::from_with_cfg(&projecting_element, &self.field_cfg);

        let projected_scalars_fx = project_scalars::<F, U>(|s| project_scalar(s, &self.field_cfg));
        let projected_scalars_f =
            project_scalars_to_field(projected_scalars_fx, &projecting_element_f)
                .map_err(|(_s, _f, e)| ProtocolError::ScalarProjection(e))?;

        Ok(VerifierEvalProjected {
            base: self.base,
            field_cfg: self.field_cfg,
            ic_subclaim: self.ic_subclaim,
            projecting_element_f,
            projected_scalars_f,
            proof_commitments: self.proof_commitments,
            proof_resolver: self.proof_resolver,
            proof_combined_sumcheck: self.proof_combined_sumcheck,
            proof_multipoint_eval: self.proof_multipoint_eval,
            proof_witness_lifted_evals: self.proof_witness_lifted_evals,
            proof_lookup_proof: self.proof_lookup_proof,
            proof_booleanity: self.proof_booleanity,
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
    F::Inner: ConstIntSemiring + ConstTranscribable + Send + Sync + Zero + Default,
    F::Modulus: ConstTranscribable + FromRef<Zt::Fmod>,
    U: Uair + 'static,
    IdealOverF: Ideal,
{
    /// Step 4: Sumcheck verification (CPR + optional booleanity + lookup
    /// groups).
    pub fn step4_sumcheck_verify(
        mut self,
    ) -> Result<VerifierSumchecked<'a, Zt, F, IdealOverF, D, FD>, ProtocolError<F, IdealOverF>>
    {
        let num_constraints = count_constraints::<U>();

        // CPR pre-sumcheck: squeezes folding challenge \alpha.
        let cpr_verifier_ancillary = CombinedPolyResolver::prepare_verifier::<U>(
            &mut self.base.pcs_transcript.fs_transcript,
            &self.proof_resolver,
            self.proof_combined_sumcheck.claimed_sums()[0].clone(),
            &self.ic_subclaim,
            num_constraints,
            self.base.num_vars,
            &self.projecting_element_f,
            &self.field_cfg,
        )?;

        // Booleanity pre-sumcheck: squeezes r, \gamma, \delta in that order.
        // Determined statically from the UAIR signature, so prover and
        // verifier always agree on the presence flag.
        let sig = self.base.uair_signature.clone();
        let num_pub_bin = sig.public_cols().num_binary_poly_cols();
        let num_total_bin = sig.total_cols().num_binary_poly_cols();
        let bool_present = num_total_bin > num_pub_bin;

        let bool_verifier_ancillary = if bool_present {
            // booleanity is group index 1 (right after CPR)
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

        let md_subclaims = MultiDegreeSumcheck::verify_as_subprotocol(
            &mut self.base.pcs_transcript.fs_transcript,
            self.base.num_vars,
            &self.proof_combined_sumcheck,
            &self.field_cfg,
        )
        .map_err(CombinedPolyResolverError::SumcheckError)?;

        let cpr_subclaim = CombinedPolyResolver::finalize_verifier::<U>(
            &mut self.base.pcs_transcript.fs_transcript,
            self.proof_resolver,
            md_subclaims.point().to_vec(),
            md_subclaims.expected_evaluations()[0].clone(),
            cpr_verifier_ancillary,
            &self.projected_scalars_f,
            &self.field_cfg,
        )?;

        let bool_subclaim = if let Some(anc) = bool_verifier_ancillary {
            let booleanity_proof = self
                .proof_booleanity
                .take()
                .ok_or(ProtocolError::BooleanityProofMissing)?;
            let expected_evaluation = md_subclaims
                .expected_evaluations()
                .get(1)
                .ok_or(ProtocolError::BooleanityProofMissing)?;
            Some(
                BooleanityChecker::<F>::finalize_verifier(
                    &mut self.base.pcs_transcript.fs_transcript,
                    booleanity_proof,
                    md_subclaims.point().to_vec(),
                    expected_evaluation,
                    anc,
                    &self.field_cfg,
                )
                .map_err(ProtocolError::Booleanity)?,
            )
        } else {
            None
        };

        let _ = &self.proof_lookup_proof;

        Ok(VerifierSumchecked {
            base: self.base,
            field_cfg: self.field_cfg,
            projecting_element_f: self.projecting_element_f,
            cpr_subclaim,
            bool_subclaim,
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
    F::Inner: ConstIntSemiring + ConstTranscribable + Send + Sync + Zero + Default,
    F::Modulus: ConstTranscribable + FromRef<Zt::Fmod>,
    IdealOverF: Ideal,
{
    /// Step 5: Multi-point evaluation sumcheck.
    pub fn step5_multipoint_eval<U: Uair>(
        mut self,
    ) -> Result<VerifierMultipointEvaled<'a, Zt, F, IdealOverF, D, FD>, ProtocolError<F, IdealOverF>>
    {
        // When the booleanity argument is present, the bit-slice claims
        // produced by [`BooleanityChecker::finalize_verifier`] at `r*` are
        // appended to `up_evals` and folded into the same `r_0` as the CPR
        // claims. The corresponding scalar evaluations at `r_0` are
        // discharged for free in step 6 against `lifted_evals.coeffs`.
        let extended_up_evals: Vec<F> = if let Some(ref bs) = self.bool_subclaim {
            let mut v = self.cpr_subclaim.up_evals;
            v.extend_from_slice(&bs.bit_slice_evals);
            v
        } else {
            self.cpr_subclaim.up_evals
        };

        let mp_subclaim = MultipointEval::verify_as_subprotocol(
            &mut self.base.pcs_transcript.fs_transcript,
            self.proof_multipoint_eval,
            &self.cpr_subclaim.evaluation_point,
            &extended_up_evals,
            &self.cpr_subclaim.down_evals,
            self.base.uair_signature.shifts(),
            self.base.num_vars,
            &self.field_cfg,
        )?;

        Ok(VerifierMultipointEvaled {
            base: self.base,
            field_cfg: self.field_cfg,
            projecting_element_f: self.projecting_element_f,
            mp_subclaim,
            booleanity_present: self.bool_subclaim.is_some(),
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
    F::Inner: ConstIntSemiring + ConstTranscribable + Send + Sync + Zero + Default,
    F::Modulus: ConstTranscribable + FromRef<Zt::Fmod>,
    IdealOverF: Ideal,
{
    /// Step 6: Recompute public lifted_evals, assemble full set, verify
    /// multipoint eval subclaim, and absorb all lifted_evals into transcript.
    pub fn step6_lifted_evals<U: Uair>(
        mut self,
    ) -> Result<
        VerifierLiftedEvalsChecked<'a, Zt, F, IdealOverF, D, FD>,
        ProtocolError<F, IdealOverF>,
    > {
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

        let mut open_evals: Vec<F> = all_lifted_evals
            .iter()
            .map(|bar_u| bar_u.evaluate_at_point(&self.projecting_element_f))
            .collect::<Result<Vec<_>, _>>()
            .map_err(ProtocolError::LiftedEvalProjection)?;

        // When the booleanity argument was active, the extended `up_evals`
        // passed to MultipointEval included one entry per (witness bin-poly
        // column, bit position). Their corresponding `open_evals` at r_0
        // are precisely `lifted_evals[j].coeffs[i]` (already a scalar in
        // F_q, not a polynomial).
        if self.booleanity_present {
            let zero = F::zero_with_cfg(&self.field_cfg);
            for bar_u in all_lifted_evals.iter().skip(num_pub_bin).take(num_wit_bin) {
                for i in 0..D {
                    open_evals.push(bar_u.coeffs.get(i).cloned().unwrap_or_else(|| zero.clone()));
                }
            }
        }

        MultipointEval::verify_subclaim(
            &self.mp_subclaim,
            &open_evals,
            self.base.uair_signature.shifts(),
            &self.field_cfg,
        )?;

        let mut transcription_buf: Vec<u8> = vec![0; F::Inner::NUM_BYTES];
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
    F::Inner: ConstIntSemiring + ConstTranscribable + Send + Sync + Zero + Default,
    F::Modulus: ConstTranscribable + FromRef<Zt::Fmod>,
    IdealOverF: Ideal,
{
    /// Step 7: PCS verification at `r_0` (witness columns only).
    pub fn step7_pcs_verify<U: Uair, const CHECK_FOR_OVERFLOW: bool>(
        mut self,
    ) -> Result<VerifierPcsVerified<IdealOverF>, ProtocolError<F, IdealOverF>> {
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
    pub fn finish<F: PrimeField>(self) -> Result<(), ProtocolError<F, IdealOverF>> {
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
    F::Inner: ConstIntSemiring + ConstTranscribable + Send + Sync + Zero + Default,
    F::Modulus: ConstTranscribable + FromRef<Zt::Fmod>,
    U: Uair + 'static,
{
    /// Zinc+ full PIOP verifier.
    ///
    /// Runs all verification steps in sequence and returns `Ok(())` on
    /// success. For per-step control, starts with
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
    ) -> Result<(), ProtocolError<F, IdealOverF>>
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
        .step2_ideal_check(project_ideal)?
        .step3_eval_projection(project_scalar)?
        .step4_sumcheck_verify()?
        .step5_multipoint_eval::<U>()?
        .step6_lifted_evals::<U>()?
        .step7_pcs_verify::<U, CHECK_FOR_OVERFLOW>()?
        .finish::<F>()
    }
}
