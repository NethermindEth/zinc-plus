use super::*;
use crypto_primitives::{ConstIntSemiring, FromPrimitiveWithConfig, FromWithConfig};
use num_traits::Zero;
use std::{collections::HashMap, io::Cursor};
use zinc_piop::{
    combined_poly_resolver::{self, CombinedPolyResolver},
    ideal_check::{self, IdealCheckProtocol},
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
pub struct VerifierBase<'a, Zt: ZincTypes<D>, const D: usize> {
    num_vars: usize,
    uair_signature: UairSignature,
    pcs_transcript: PcsVerifierTranscript,
    public_trace: &'a UairTrace<'a, Zt::Int, Zt::Int, D>,

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
    Zt: ZincTypes<D>,
    U: Uair,
    F: PrimeField,
    IdealOverF,
    const D: usize,
> {
    base: VerifierBase<'a, Zt, D>,

    // Proof leftovers
    proof_commitments: (ZipPlusCommitment, ZipPlusCommitment, ZipPlusCommitment),
    proof_ideal_check: IdealCheckProof<F>,
    proof_resolver: CombinedPolyResolverProof<F>,
    proof_combined_sumcheck: MultiDegreeSumcheckProof<F>,
    proof_multipoint_eval: MultipointEvalProof<F>,
    proof_witness_lifted_evals: Vec<DynamicPolynomialF<F>>,
    proof_lookup_proof: Option<BatchedLookupProof<F>>,
    _phantom: PhantomData<(U, IdealOverF)>,
}

/// After step 1 (prime projection).
#[derive(Clone, Debug)]
pub struct VerifierPrimeProjected<
    'a,
    Zt: ZincTypes<D>,
    U: Uair,
    F: PrimeField,
    IdealOverF,
    const D: usize,
> {
    base: VerifierBase<'a, Zt, D>,
    field_cfg: F::Config,

    // Proof leftovers
    proof_commitments: (ZipPlusCommitment, ZipPlusCommitment, ZipPlusCommitment),
    proof_ideal_check: IdealCheckProof<F>,
    proof_resolver: CombinedPolyResolverProof<F>,
    proof_combined_sumcheck: MultiDegreeSumcheckProof<F>,
    proof_multipoint_eval: MultipointEvalProof<F>,
    proof_witness_lifted_evals: Vec<DynamicPolynomialF<F>>,
    proof_lookup_proof: Option<BatchedLookupProof<F>>,
    _phantom: PhantomData<(U, IdealOverF)>,
}

/// After step 2 (ideal check). `project_ideal` has been consumed.
#[derive(Clone, Debug)]
pub struct VerifierIdealChecked<
    'a,
    Zt: ZincTypes<D>,
    U: Uair,
    F: PrimeField,
    IdealOverF,
    const D: usize,
> {
    base: VerifierBase<'a, Zt, D>,
    field_cfg: F::Config,
    ic_subclaim: ideal_check::VerifierSubclaim<F>,

    // Proof leftovers
    proof_commitments: (ZipPlusCommitment, ZipPlusCommitment, ZipPlusCommitment),
    proof_resolver: CombinedPolyResolverProof<F>,
    proof_combined_sumcheck: MultiDegreeSumcheckProof<F>,
    proof_multipoint_eval: MultipointEvalProof<F>,
    proof_witness_lifted_evals: Vec<DynamicPolynomialF<F>>,
    proof_lookup_proof: Option<BatchedLookupProof<F>>,
    _phantom: PhantomData<(U, IdealOverF)>,
}

/// After step 3 (eval projection). `project_scalar` has been consumed.
#[derive(Clone, Debug)]
pub struct VerifierEvalProjected<
    'a,
    Zt: ZincTypes<D>,
    U: Uair,
    F: PrimeField,
    IdealOverF,
    const D: usize,
> {
    base: VerifierBase<'a, Zt, D>,
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
    _phantom: PhantomData<(U, IdealOverF)>,
}

/// After step 4 (sumcheck verify).
#[derive(Clone, Debug)]
pub struct VerifierSumchecked<'a, Zt: ZincTypes<D>, F: PrimeField, IdealOverF, const D: usize> {
    base: VerifierBase<'a, Zt, D>,
    field_cfg: F::Config,
    projecting_element_f: F,
    cpr_subclaim: combined_poly_resolver::VerifierSubclaim<F>,

    // Proof leftovers
    proof_commitments: (ZipPlusCommitment, ZipPlusCommitment, ZipPlusCommitment),
    proof_multipoint_eval: MultipointEvalProof<F>,
    proof_witness_lifted_evals: Vec<DynamicPolynomialF<F>>,
    proof_lookup_proof: Option<BatchedLookupProof<F>>,
    _phantom: PhantomData<IdealOverF>,
}

/// After step 5 (multi-point eval).
#[derive(Clone, Debug)]
pub struct VerifierMultipointEvaled<'a, Zt: ZincTypes<D>, F: PrimeField, IdealOverF, const D: usize>
{
    base: VerifierBase<'a, Zt, D>,
    field_cfg: F::Config,
    projecting_element_f: F,
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
    Zt: ZincTypes<D>,
    F: PrimeField,
    IdealOverF,
    const D: usize,
> {
    base: VerifierBase<'a, Zt, D>,
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

impl<Zt, U, F, const D: usize> ZincPlusPiop<Zt, U, F, D>
where
    Zt: ZincTypes<D>,
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
        public_trace: &'a UairTrace<'a, Zt::Int, Zt::Int, D>,
        num_vars: usize,
    ) -> Result<
        VerifierTranscriptReconstructed<'a, Zt, U, F, IdealOverF, D>,
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
            _phantom: PhantomData,
        })
    }
}

impl<'a, Zt, U, F, IdealOverF, const D: usize>
    VerifierTranscriptReconstructed<'a, Zt, U, F, IdealOverF, D>
where
    Zt: ZincTypes<D>,
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
    ) -> Result<VerifierPrimeProjected<'a, Zt, U, F, IdealOverF, D>, ProtocolError<F, IdealOverF>>
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
            _phantom: PhantomData,
        })
    }
}

impl<'a, Zt, U, F, IdealOverF, const D: usize> VerifierPrimeProjected<'a, Zt, U, F, IdealOverF, D>
where
    Zt: ZincTypes<D>,
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
    ) -> Result<VerifierIdealChecked<'a, Zt, U, F, IdealOverF, D>, ProtocolError<F, IdealOverF>>
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
            _phantom: PhantomData,
        })
    }
}

impl<'a, Zt, U, F, IdealOverF, const D: usize> VerifierIdealChecked<'a, Zt, U, F, IdealOverF, D>
where
    Zt: ZincTypes<D>,
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
    ) -> Result<VerifierEvalProjected<'a, Zt, U, F, IdealOverF, D>, ProtocolError<F, IdealOverF>>
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
            _phantom: PhantomData,
        })
    }
}

impl<'a, Zt, U, F, IdealOverF, const D: usize> VerifierEvalProjected<'a, Zt, U, F, IdealOverF, D>
where
    Zt: ZincTypes<D>,
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
    /// Step 4: Sumcheck verification (CPR + lookup groups).
    pub fn step4_sumcheck_verify(
        mut self,
    ) -> Result<VerifierSumchecked<'a, Zt, F, IdealOverF, D>, ProtocolError<F, IdealOverF>> {
        let num_constraints = count_constraints::<U>();

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

        let _ = &self.proof_lookup_proof;

        Ok(VerifierSumchecked {
            base: self.base,
            field_cfg: self.field_cfg,
            projecting_element_f: self.projecting_element_f,
            cpr_subclaim,
            proof_commitments: self.proof_commitments,
            proof_multipoint_eval: self.proof_multipoint_eval,
            proof_witness_lifted_evals: self.proof_witness_lifted_evals,
            proof_lookup_proof: self.proof_lookup_proof,
            _phantom: PhantomData,
        })
    }
}

impl<'a, Zt, F, IdealOverF, const D: usize> VerifierSumchecked<'a, Zt, F, IdealOverF, D>
where
    Zt: ZincTypes<D>,
    F: InnerTransparentField + FromPrimitiveWithConfig + FromRef<F> + Send + Sync + 'static,
    F::Inner: ConstIntSemiring + ConstTranscribable + Send + Sync + Zero + Default,
    F::Modulus: ConstTranscribable + FromRef<Zt::Fmod>,
    IdealOverF: Ideal,
{
    /// Step 5: Multi-point evaluation sumcheck.
    pub fn step5_multipoint_eval<U: Uair>(
        mut self,
    ) -> Result<VerifierMultipointEvaled<'a, Zt, F, IdealOverF, D>, ProtocolError<F, IdealOverF>>
    {
        let mp_subclaim = MultipointEval::verify_as_subprotocol(
            &mut self.base.pcs_transcript.fs_transcript,
            self.proof_multipoint_eval,
            &self.cpr_subclaim.evaluation_point,
            &self.cpr_subclaim.up_evals,
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
            proof_commitments: self.proof_commitments,
            proof_witness_lifted_evals: self.proof_witness_lifted_evals,
            proof_lookup_proof: self.proof_lookup_proof,
            _phantom: PhantomData,
        })
    }
}

impl<'a, Zt, F, IdealOverF, const D: usize> VerifierMultipointEvaled<'a, Zt, F, IdealOverF, D>
where
    Zt: ZincTypes<D>,
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
    ) -> Result<VerifierLiftedEvalsChecked<'a, Zt, F, IdealOverF, D>, ProtocolError<F, IdealOverF>>
    {
        let r_0 = &self.mp_subclaim.sumcheck_subclaim.point;

        let pub_cols = self.base.uair_signature.public_cols();
        let num_pub_bin = pub_cols.num_binary_poly_cols();
        let num_pub_arb = pub_cols.num_arbitrary_poly_cols();
        let num_pub_int = pub_cols.num_int_cols();

        let wit_cols = self.base.uair_signature.witness_cols();
        let num_wit_bin = wit_cols.num_binary_poly_cols();
        let num_wit_arb = wit_cols.num_arbitrary_poly_cols();

        let public_lifted = if add!(add!(num_pub_bin, num_pub_arb), num_pub_int) > 0 {
            let projected_public = project_trace_coeffs_row_major::<F, Zt::Int, Zt::Int, D>(
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

        let open_evals: Vec<F> = all_lifted_evals
            .iter()
            .map(|bar_u| bar_u.evaluate_at_point(&self.projecting_element_f))
            .collect::<Result<Vec<_>, _>>()
            .map_err(ProtocolError::LiftedEvalProjection)?;

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

impl<'a, Zt, F, IdealOverF, const D: usize> VerifierLiftedEvalsChecked<'a, Zt, F, IdealOverF, D>
where
    Zt: ZincTypes<D>,
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

        macro_rules! verify_pcs_batch {
            ($Zt:ty, $Lc:ty, $vp:expr, $idx:tt, [$evals_range:expr]) => {{
                let comm = &commitments.$idx;
                if comm.batch_size > 0 {
                    let per_poly_alphas = ZipPlus::<$Zt, $Lc>::sample_alphas(
                        &mut pcs_transcript.fs_transcript,
                        comm.batch_size,
                    );
                    let mut eval_f = F::zero_with_cfg(field_cfg);
                    for (bar_u, alphas) in all_lifted_evals[$evals_range]
                        .iter()
                        .zip(per_poly_alphas.iter())
                    {
                        for (coeff, alpha) in bar_u.coeffs.iter().zip(alphas.iter()) {
                            let mut term = F::from_with_cfg(alpha, field_cfg);
                            term *= coeff;
                            eval_f += &term;
                        }
                    }
                    ZipPlus::<$Zt, $Lc>::verify_with_alphas::<F, CHECK_FOR_OVERFLOW>(
                        pcs_transcript,
                        $vp,
                        comm,
                        field_cfg,
                        r_0,
                        &eval_f,
                        &per_poly_alphas,
                    )
                    .map_err(|e| ProtocolError::PcsVerification($idx, e))?;
                }
            }};
        }

        verify_pcs_batch!(
            Zt::BinaryZt,
            Zt::BinaryLc,
            self.base.vp_bin,
            0,
            [num_pub_bin..num_total_bin]
        );
        verify_pcs_batch!(
            Zt::ArbitraryZt,
            Zt::ArbitraryLc,
            self.base.vp_arb,
            1,
            [add!(num_total_bin, num_pub_arb)..add!(num_total_bin, num_total_arb)]
        );
        verify_pcs_batch!(
            Zt::IntZt,
            Zt::IntLc,
            self.base.vp_int,
            2,
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

// ── verify() wrapper ───────────────────────────────────────────────────

impl<Zt, U, F, const D: usize> ZincPlusPiop<Zt, U, F, D>
where
    Zt: ZincTypes<D>,
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
        public_trace: &UairTrace<Zt::Int, Zt::Int, D>,
        num_vars: usize,
        project_scalar: impl Fn(&U::Scalar, &F::Config) -> DynamicPolynomialF<F>,
        project_ideal: impl Fn(&IdealOrZero<U::Ideal>, &F::Config) -> IdealOverF,
    ) -> Result<(), ProtocolError<F, IdealOverF>>
    where
        IdealOverF: Ideal + IdealCheck<DynamicPolynomialF<F>>,
    {
        ZincPlusPiop::<Zt, U, F, D>::step0_reconstruct_transcript::<IdealOverF>(
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
