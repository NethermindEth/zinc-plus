use super::*;
use crypto_primitives::{ConstIntSemiring, FromPrimitiveWithConfig, FromWithConfig};
use num_traits::Zero;
use std::io::Cursor;
use zinc_piop::{
    combined_poly_resolver::{self, CombinedPolyResolver},
    ideal_check::{self, IdealCheckProtocol},
    lookup::booleanity::{
        compute_bit_slices_flat, compute_virtual_binary_poly_closing_overrides,
        compute_virtual_closing_overrides, finalize_booleanity_verifier,
        prepare_booleanity_verifier, verify_bit_decomposition_consistency,
    },
    multipoint_eval::{self, MultipointEval},
    projections::{
        ProjectedTrace, ScalarMap, project_scalars, project_scalars_to_field,
        project_trace_coeffs_row_major,
    },
    sumcheck::multi_degree::MultiDegreeSumcheck,
};
use zinc_poly::{
    EvaluatablePolynomial, mle::MultilinearExtensionWithConfig,
    univariate::dynamic::over_field::DynamicPolynomialF,
};
use zinc_transcript::{
    Blake3Transcript,
    traits::{ConstTranscribable, Transcript},
};
use zinc_uair::{
    BitOp, Uair, UairSignature, UairTrace,
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

/// Drop the witness binary_poly column evals the UAIR opted out of
/// (sorted, dedup'd `skip_indices` relative to the witness slice). The
/// surviving evals line up positionally with the bit-slice blocks
/// `bit_slice_evals[col*D..col*D+D]`, so the bit-decomposition
/// consistency check pairs the right parent eval with the right block.
fn filter_skipped_parent_evals<F: Clone>(
    witness_parent_evals: &[F],
    skip_indices: &[usize],
) -> Vec<F> {
    if skip_indices.is_empty() {
        return witness_parent_evals.to_vec();
    }
    witness_parent_evals
        .iter()
        .enumerate()
        .filter(|(i, _)| !skip_indices.contains(i))
        .map(|(_, e)| e.clone())
        .collect()
}

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
    projected_scalars_f: ScalarMap<U::Scalar, F>,

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
    /// Step 1: Prime projection. Builds the field configuration from the
    /// fixed projecting prime (secp256k1 base prime; see `crate::fixed_prime`).
    #[allow(clippy::type_complexity)]
    pub fn step1_prime_projection(
        self,
    ) -> Result<VerifierPrimeProjected<'a, Zt, U, F, IdealOverF, D>, ProtocolError<F, IdealOverF>>
    {
        // `fixed-prime` branch: use the secp256k1 base field prime as the
        // projecting prime instead of drawing one from the transcript.
        // See `crate::fixed_prime` for the soundness caveat.
        let field_cfg = crate::fixed_prime::secp256k1_field_cfg::<F, Zt::Fmod>();

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
        + for<'b> FromWithConfig<&'b <Zt::BinaryZt as ZipTypes>::CombR>
        + for<'b> FromWithConfig<&'b <Zt::ArbitraryZt as ZipTypes>::CombR>
        + for<'b> FromWithConfig<&'b <Zt::IntZt as ZipTypes>::CombR>
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
        project_scalar: impl Fn(&U::Scalar, &F::Config) -> DynamicPolynomialF<F> + Sync,
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
        + for<'b> FromWithConfig<&'b <Zt::BinaryZt as ZipTypes>::CombR>
        + for<'b> FromWithConfig<&'b <Zt::ArbitraryZt as ZipTypes>::CombR>
        + for<'b> FromWithConfig<&'b <Zt::IntZt as ZipTypes>::CombR>
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
    /// Step 4: Sumcheck verification (CPR + algebraic booleanity).
    pub fn step4_sumcheck_verify(
        mut self,
    ) -> Result<VerifierSumchecked<'a, Zt, F, IdealOverF, D>, ProtocolError<F, IdealOverF>> {
        let num_constraints = count_constraints::<U>();
        let num_pub_bin = self
            .base
            .uair_signature
            .public_cols()
            .num_binary_poly_cols();
        let num_total_bin =
            self.base.uair_signature.total_cols().num_binary_poly_cols();
        let bool_skip = self.base.uair_signature.booleanity_skip_indices();
        // Booleanity covers: witness binary_poly cols (minus
        // `booleanity_skip_indices`), packed virtual binary_poly cols,
        // declared int bit cols, and virtual booleanity linear-combo cols.
        let num_int_bit_cols =
            self.base.uair_signature.int_witness_bit_cols().len();
        let num_virtual_cols =
            self.base.uair_signature.virtual_booleanity_cols().len();
        let num_virtual_bp_cols =
            self.base.uair_signature.virtual_binary_poly_cols().len();
        let num_bit_slices = ((num_total_bin - num_pub_bin) - bool_skip.len()) * D
            + num_virtual_bp_cols * D
            + num_int_bit_cols
            + num_virtual_cols;
        let num_shifted_bit_slices =
            self.base.uair_signature.shifted_bit_slice_specs().len() * D;

        let cpr_verifier_ancillary = CombinedPolyResolver::prepare_verifier::<U>(
            &mut self.base.pcs_transcript.fs_transcript,
            &self.proof_resolver,
            self.proof_combined_sumcheck.claimed_sums()[0].clone(),
            &self.ic_subclaim,
            num_constraints,
            num_bit_slices,
            num_shifted_bit_slices,
            self.base.num_vars,
            &self.projecting_element_f,
            &self.field_cfg,
        )?;

        // 4b: Booleanity verifier prep — samples α_b, validates that the
        // booleanity group's claimed sum is zero (zerocheck).
        let bool_verifier_ancillary_opt = if num_bit_slices > 0 {
            let bool_claimed_sum =
                self.proof_combined_sumcheck.claimed_sums()[1].clone();
            prepare_booleanity_verifier::<F>(
                &mut self.base.pcs_transcript.fs_transcript,
                bool_claimed_sum,
                num_bit_slices,
                &self.ic_subclaim.evaluation_point,
                &self.field_cfg,
            )
            .map_err(ProtocolError::Booleanity)?
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

        // 4c: Booleanity verifier finalize — validates the booleanity
        // group's expected_evaluation at r*. For declared int bit cols
        // and virtual cols we substitute `up_eval` (or a reconstructed
        // linear-combo eval) into the closing in place of the
        // bit-slice eval emitted by the prover; this ties the
        // booleanity-bound MLE to the committed sources without a
        // separate equality check.
        let int_offset = self.base.uair_signature.total_cols().num_binary_poly_cols()
            + self.base.uair_signature.total_cols().num_arbitrary_poly_cols();
        let num_pub_int = self.base.uair_signature.public_cols().num_int_cols();
        let num_wit_int = self.base.uair_signature.witness_cols().num_int_cols();
        let num_binary_bit_slices_for_overrides = (num_total_bin - num_pub_bin) * D;
        let virtual_bp_specs = self.base.uair_signature.virtual_binary_poly_cols();
        let virtual_specs = self.base.uair_signature.virtual_booleanity_cols();

        // Public binary_poly bit slice evals at the shared sumcheck
        // point — verifier computes locally from public_trace; reused
        // by both virtual-bool and virtual-binary-poly overrides.
        let public_bit_slice_evals: Vec<F> = if !virtual_bp_specs.is_empty()
            || !virtual_specs.is_empty()
        {
            let public_bit_slice_mles = compute_bit_slices_flat::<F, D>(
                &self.base.public_trace.binary_poly,
                &self.field_cfg,
            );
            public_bit_slice_mles
                .into_iter()
                .map(|mle| mle.evaluate_with_config(md_subclaims.point(), &self.field_cfg))
                .collect::<Result<Vec<_>, _>>()
                .map_err(ProtocolError::ShiftedBitSliceEval)?
        } else {
            Vec::new()
        };

        // closing_overrides_tail layout (in trailing-position order):
        //   [virtual_binary_poly_per_bit (V_b * D),
        //    int_witness_bit_col evals (E_int),
        //    virtual_booleanity per-spec evals (V_f)]
        // The virtual binary_poly section sits *between* the genuine
        // binary col bit slices and the extra_bit_cols, matching the
        // prover's binary_cols append order.
        let mut closing_overrides_tail: Vec<F> = Vec::new();
        if !virtual_bp_specs.is_empty() {
            let virtual_bp_overrides = compute_virtual_binary_poly_closing_overrides::<F, D>(
                virtual_bp_specs,
                &cpr_subclaim.bit_slice_evals[..num_binary_bit_slices_for_overrides],
                &cpr_subclaim.shifted_bit_slice_evals,
                &public_bit_slice_evals,
                &self.field_cfg,
            );
            closing_overrides_tail.extend(virtual_bp_overrides);
        }
        closing_overrides_tail.extend(
            self.base
                .uair_signature
                .int_witness_bit_cols()
                .iter()
                .map(|&idx| cpr_subclaim.up_evals[int_offset + idx].clone()),
        );
        if !virtual_specs.is_empty() {
            let int_witness_up_evals: Vec<F> = (0..num_wit_int)
                .map(|i| {
                    cpr_subclaim.up_evals[int_offset + num_pub_int + i].clone()
                })
                .collect();
            let virtual_overrides = compute_virtual_closing_overrides::<F, D>(
                virtual_specs,
                &cpr_subclaim.bit_slice_evals[..num_binary_bit_slices_for_overrides],
                &cpr_subclaim.shifted_bit_slice_evals,
                &public_bit_slice_evals,
                &int_witness_up_evals,
                &self.field_cfg,
            );
            closing_overrides_tail.extend(virtual_overrides);
        }
        if let Some(ba) = bool_verifier_ancillary_opt {
            finalize_booleanity_verifier::<F>(
                &mut self.base.pcs_transcript.fs_transcript,
                &cpr_subclaim.bit_slice_evals,
                &closing_overrides_tail,
                md_subclaims.point(),
                md_subclaims.expected_evaluations()[1].clone(),
                ba,
                &self.field_cfg,
            )
            .map_err(ProtocolError::Booleanity)?;
        }

        // Bit-decomposition consistency: each genuine (non-skipped)
        // binary_poly column's F-projected MLE eval at r* must equal
        // Σ_i a^i · bit_slice_eval[i], where `a` is the projecting
        // element used in step 3 (ψ_a). Skip the columns the UAIR
        // opted out of (`booleanity_skip_indices`) — their bit-slice
        // evals don't appear in `bit_slice_evals`, so we filter
        // `up_evals` accordingly. Virtual binary_poly cols' bit slices
        // and extra (int / virtual-bool) bit cols follow in
        // `bit_slice_evals`; they're not parented in `up_evals` and
        // are bound via closing overrides below.
        let parent_evals_for_bool = filter_skipped_parent_evals(
            &cpr_subclaim.up_evals[num_pub_bin..num_total_bin],
            bool_skip,
        );
        let num_kept_binary_bit_slices = parent_evals_for_bool.len() * D;
        verify_bit_decomposition_consistency(
            &parent_evals_for_bool,
            &cpr_subclaim.bit_slice_evals[..num_kept_binary_bit_slices],
            &self.projecting_element_f,
            D,
        )
        .map_err(ProtocolError::Booleanity)?;

        // Shifted bit-slice consistency: tie each spec's emitted bit
        // slices to the corresponding `down_eval` (= parent col at
        // shifted point) via the same projection-element trick.
        let shifted_down_indices =
            self.base.uair_signature.shifted_bit_slice_down_indices();
        let shifted_parent_evals: Vec<F> = shifted_down_indices
            .iter()
            .map(|&i| cpr_subclaim.down_evals[i].clone())
            .collect();
        verify_bit_decomposition_consistency(
            &shifted_parent_evals,
            &cpr_subclaim.shifted_bit_slice_evals,
            &self.projecting_element_f,
            D,
        )
        .map_err(ProtocolError::Booleanity)?;

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
    /// Step 5: Multi-point evaluation sumcheck (under ψ_α).
    ///
    /// CPR's `up_evals` (one per base trace col) are extended with
    /// `bit_op_down_evals` (one per `BitOpSpec`), forming the `up`
    /// claim list mp_eval consumes. The shift list is unchanged.
    /// At r_0 mp_eval needs `open_evals` of length
    /// `num_total + num_bit_op` — the bit-op slots get derived in
    /// Step 6 by applying the bit-op locally to each source's
    /// lifted eval (free arithmetic in F_q[X]).
    pub fn step5_multipoint_eval<U: Uair>(
        mut self,
    ) -> Result<VerifierMultipointEvaled<'a, Zt, F, IdealOverF, D>, ProtocolError<F, IdealOverF>>
    {
        let cpr_eval_point = self.cpr_subclaim.evaluation_point.clone();

        let mut up_evals_with_bit_op = self.cpr_subclaim.up_evals.clone();
        up_evals_with_bit_op.extend(self.cpr_subclaim.bit_op_down_evals.iter().cloned());

        let mp_subclaim = MultipointEval::verify_as_subprotocol(
            &mut self.base.pcs_transcript.fs_transcript,
            self.proof_multipoint_eval,
            &cpr_eval_point,
            &up_evals_with_bit_op,
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
    ///
    /// For each `BitOpSpec(src, op)` declared by the UAIR, the verifier
    /// derives the bit-op slot's `open_eval` locally by applying `op` to
    /// the source's `lifted_eval` in F_q[X] and ψ_α-projecting. This
    /// closes the bit-op consistency loop without any new wire data:
    /// `MLE[op(src)](r_0) == op(MLE[src])(r_0)` because bit-op (acts on
    /// bit-position within a cell) and the MLE evaluation (acts on the
    /// row index) commute. Mismatches surface as
    /// `MultipointEval(ClaimMismatch)` from `verify_subclaim`.
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

        // Derive bit-op virtual lifted evals from the source's lifted
        // eval via the structural permutation (rot_c / shift_r_c). They
        // are appended to `open_evals` in the same order mp_eval used
        // to extend the up-claim list.
        let mut open_evals: Vec<F> = all_lifted_evals
            .iter()
            .map(|bar_u| bar_u.evaluate_at_point(&self.projecting_element_f))
            .collect::<Result<Vec<_>, _>>()
            .map_err(ProtocolError::LiftedEvalProjection)?;
        for spec in self.base.uair_signature.bit_op_specs() {
            let bar_u_src = &all_lifted_evals[spec.source_col()];
            let op_e_src = match spec.op() {
                BitOp::Rot(c) => bar_u_src.rot_c(c),
                BitOp::ShiftR(c) => bar_u_src.shift_r_c(c),
            };
            let psi = op_e_src
                .evaluate_at_point(&self.projecting_element_f)
                .map_err(ProtocolError::LiftedEvalProjection)?;
            open_evals.push(psi);
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
        + for<'b> FromWithConfig<&'b <Zt::BinaryZt as ZipTypes>::CombR>
        + for<'b> FromWithConfig<&'b <Zt::ArbitraryZt as ZipTypes>::CombR>
        + for<'b> FromWithConfig<&'b <Zt::IntZt as ZipTypes>::CombR>
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
    Zt::Int: ProjectableToField<F> + num_traits::Zero,
    <Zt::BinaryZt as ZipTypes>::Cw: ProjectableToField<F>,
    <Zt::ArbitraryZt as ZipTypes>::Eval: ProjectableToField<F>,
    <Zt::ArbitraryZt as ZipTypes>::Cw: ProjectableToField<F>,
    <Zt::IntZt as ZipTypes>::Cw: ProjectableToField<F>,
    F: InnerTransparentField
        + FromPrimitiveWithConfig
        + for<'a> FromWithConfig<&'a Zt::Int>
        + for<'a> FromWithConfig<&'a <Zt::BinaryZt as ZipTypes>::CombR>
        + for<'a> FromWithConfig<&'a <Zt::ArbitraryZt as ZipTypes>::CombR>
        + for<'a> FromWithConfig<&'a <Zt::IntZt as ZipTypes>::CombR>
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
        project_scalar: impl Fn(&U::Scalar, &F::Config) -> DynamicPolynomialF<F> + Sync,
        project_ideal: impl Fn(&IdealOrZero<U::Ideal>, &F::Config) -> IdealOverF,
    ) -> Result<(), ProtocolError<F, IdealOverF>>
    where
        IdealOverF: Ideal + IdealCheck<DynamicPolynomialF<F>>,
    {
        // Verifier-side public-column structural checks. UAIRs that
        // need to enforce structural properties of public columns
        // (compensator-zero on active rows, tail-corrector-zero on
        // inner rows, etc.) discharge them here, by direct row-wise
        // inspection of public_trace, before any algebraic check
        // begins. Default impl is a no-op for UAIRs that don't need
        // such checks.
        U::verify_public_structure(public_trace, num_vars)
            .map_err(ProtocolError::PublicStructure)?;

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

//
// Folded verifier (1× fold counterpart of `verify`).
//
// Mirrors the unfolded verifier but expects the binary commitment to be
// over `BinaryPoly<HALF_D>` split columns at length `2n`, and verifies the
// binary PCS opening at the extended point `(r_0 ‖ γ)`. The verifier
// recomputes the binary PCS `eval_f` from the proof's polynomial-valued
// `witness_lifted_evals` by splitting the 32-coefficient lifted polynomial
// into low/high halves and projecting each via the split commitment's
// per-poly alphas before γ-interpolating.
//

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn verify_folded<
    ZtF,
    U,
    F,
    IdealOverF,
    const D: usize,
    const HALF_D: usize,
    const CHECK_FOR_OVERFLOW: bool,
>(
    vp: &(
        ZipPlusParams<ZtF::BinaryZt, ZtF::BinaryLc>,
        ZipPlusParams<ZtF::ArbitraryZt, ZtF::ArbitraryLc>,
        ZipPlusParams<ZtF::IntZt, ZtF::IntLc>,
    ),
    mut proof: Proof<F>,
    public_trace: &UairTrace<ZtF::Int, ZtF::Int, D>,
    num_vars: usize,
    project_scalar: impl Fn(&U::Scalar, &F::Config) -> DynamicPolynomialF<F> + Sync,
    project_ideal: impl Fn(&IdealOrZero<U::Ideal>, &F::Config) -> IdealOverF,
) -> Result<(), ProtocolError<F, IdealOverF>>
where
    ZtF: crate::FoldedZincTypes<D, HALF_D>,
    ZtF::Int: ProjectableToField<F> + num_traits::Zero,
    <ZtF::ArbitraryZt as ZipTypes>::Eval: ProjectableToField<F>,
    <ZtF::BinaryZt as ZipTypes>::Cw: ProjectableToField<F>,
    <ZtF::ArbitraryZt as ZipTypes>::Cw: ProjectableToField<F>,
    <ZtF::IntZt as ZipTypes>::Cw: ProjectableToField<F>,
    U: Uair + 'static,
    F: InnerTransparentField
        + FromPrimitiveWithConfig
        + for<'b> FromWithConfig<&'b ZtF::Int>
        + for<'b> FromWithConfig<&'b <ZtF::BinaryZt as ZipTypes>::CombR>
        + for<'b> FromWithConfig<&'b <ZtF::ArbitraryZt as ZipTypes>::CombR>
        + for<'b> FromWithConfig<&'b <ZtF::IntZt as ZipTypes>::CombR>
        + for<'b> FromWithConfig<&'b ZtF::Chal>
        + for<'b> MulByScalar<&'b F>
        + FromRef<F>
        + Send
        + Sync
        + 'static,
    F::Inner: ConstIntSemiring + ConstTranscribable + Send + Sync + Zero + Default,
    F::Modulus: ConstTranscribable + FromRef<ZtF::Fmod>,
    IdealOverF: Ideal + IdealCheck<DynamicPolynomialF<F>>,
{
    // Verifier-side public-column structural checks (compensator/
    // corrector zero-pinning, etc.). UAIRs that don't need extra
    // structural checks fall through this with a no-op default impl.
    U::verify_public_structure(public_trace, num_vars)
        .map_err(ProtocolError::PublicStructure)?;

    // ── Step 0: Reconstruct transcript ──────────────────────────────────
    let zip_proof = std::mem::take(&mut proof.zip);
    let (vp_bin_split, vp_arb, vp_int) = vp;
    let uair_signature = U::signature();
    let mut pcs_transcript = PcsVerifierTranscript {
        fs_transcript: Blake3Transcript::default(),
        stream: Cursor::new(zip_proof),
    };
    for comm in [
        &proof.commitments.0,
        &proof.commitments.1,
        &proof.commitments.2,
    ] {
        pcs_transcript.fs_transcript.absorb_slice(&comm.root);
    }
    absorb_public_columns(&mut pcs_transcript.fs_transcript, &public_trace.binary_poly);
    absorb_public_columns(
        &mut pcs_transcript.fs_transcript,
        &public_trace.arbitrary_poly,
    );
    absorb_public_columns(&mut pcs_transcript.fs_transcript, &public_trace.int);

    // ── Step 1: Prime projection ────────────────────────────────────────
    // `fixed-prime` branch: match the non-folded path and use the
    // secp256k1 base prime as the projecting prime. See the prover-side
    // comment in `prove_folded` for why UAIRs with EC arithmetic need
    // the fixed prime here.
    let field_cfg = crate::fixed_prime::secp256k1_field_cfg::<F, ZtF::Fmod>();

    // ── Step 2: Ideal check ─────────────────────────────────────────────
    let num_constraints = count_constraints::<U>();
    let ic_subclaim = U::verify_as_subprotocol::<_, IdealOverF, _>(
        &mut pcs_transcript.fs_transcript,
        proof.ideal_check,
        num_constraints,
        num_vars,
        |ideal| project_ideal(ideal, &field_cfg),
        &field_cfg,
    )?;

    // ── Step 3: Eval projection ─────────────────────────────────────────
    let projecting_element: ZtF::Chal = pcs_transcript.fs_transcript.get_challenge();
    let projecting_element_f: F = F::from_with_cfg(&projecting_element, &field_cfg);

    let projected_scalars_fx = project_scalars::<F, U>(|s| project_scalar(s, &field_cfg));
    let projected_scalars_f =
        project_scalars_to_field(projected_scalars_fx, &projecting_element_f)
            .map_err(|(_s, _f, e)| ProtocolError::ScalarProjection(e))?;

    // ── Step 4: Sumcheck verify (CPR + algebraic booleanity) ────────────
    let num_pub_bin = uair_signature.public_cols().num_binary_poly_cols();
    let num_total_bin = uair_signature.total_cols().num_binary_poly_cols();
    let bool_skip = uair_signature.booleanity_skip_indices();
    let num_int_bit_cols = uair_signature.int_witness_bit_cols().len();
    let num_virtual_cols = uair_signature.virtual_booleanity_cols().len();
    let num_virtual_bp_cols = uair_signature.virtual_binary_poly_cols().len();
    let num_bit_slices = ((num_total_bin - num_pub_bin) - bool_skip.len()) * D
        + num_virtual_bp_cols * D
        + num_int_bit_cols
        + num_virtual_cols;
    let num_shifted_bit_slices =
        uair_signature.shifted_bit_slice_specs().len() * D;
    let cpr_verifier_ancillary = CombinedPolyResolver::prepare_verifier::<U>(
        &mut pcs_transcript.fs_transcript,
        &proof.resolver,
        proof.combined_sumcheck.claimed_sums()[0].clone(),
        &ic_subclaim,
        num_constraints,
        num_bit_slices,
        num_shifted_bit_slices,
        num_vars,
        &projecting_element_f,
        &field_cfg,
    )?;

    let bool_verifier_ancillary_opt = if num_bit_slices > 0 {
        let bool_claimed_sum = proof.combined_sumcheck.claimed_sums()[1].clone();
        prepare_booleanity_verifier::<F>(
            &mut pcs_transcript.fs_transcript,
            bool_claimed_sum,
            num_bit_slices,
            &ic_subclaim.evaluation_point,
            &field_cfg,
        )
        .map_err(ProtocolError::Booleanity)?
    } else {
        None
    };

    let md_subclaims = MultiDegreeSumcheck::verify_as_subprotocol(
        &mut pcs_transcript.fs_transcript,
        num_vars,
        &proof.combined_sumcheck,
        &field_cfg,
    )
    .map_err(combined_poly_resolver::CombinedPolyResolverError::SumcheckError)?;
    let cpr_subclaim = CombinedPolyResolver::finalize_verifier::<U>(
        &mut pcs_transcript.fs_transcript,
        proof.resolver,
        md_subclaims.point().to_vec(),
        md_subclaims.expected_evaluations()[0].clone(),
        cpr_verifier_ancillary,
        &projected_scalars_f,
        &field_cfg,
    )?;

    // Booleanity: substitute up_evals (for int bit cols) and computed
    // linear-combo evals (for virtual cols) into the closing.
    let int_offset = uair_signature.total_cols().num_binary_poly_cols()
        + uair_signature.total_cols().num_arbitrary_poly_cols();
    let num_pub_int = uair_signature.public_cols().num_int_cols();
    let num_wit_int = uair_signature.witness_cols().num_int_cols();
    let num_binary_bit_slices = (num_total_bin - num_pub_bin) * D;
    let virtual_bp_specs = uair_signature.virtual_binary_poly_cols();
    let virtual_specs = uair_signature.virtual_booleanity_cols();
    let public_bit_slice_evals: Vec<F> = if !virtual_bp_specs.is_empty()
        || !virtual_specs.is_empty()
    {
        let public_bit_slice_mles = compute_bit_slices_flat::<F, D>(
            &public_trace.binary_poly,
            &field_cfg,
        );
        public_bit_slice_mles
            .into_iter()
            .map(|mle| mle.evaluate_with_config(md_subclaims.point(), &field_cfg))
            .collect::<Result<Vec<_>, _>>()
            .map_err(ProtocolError::ShiftedBitSliceEval)?
    } else {
        Vec::new()
    };
    let mut closing_overrides_tail: Vec<F> = Vec::new();
    if !virtual_bp_specs.is_empty() {
        let virtual_bp_overrides = compute_virtual_binary_poly_closing_overrides::<F, D>(
            virtual_bp_specs,
            &cpr_subclaim.bit_slice_evals[..num_binary_bit_slices],
            &cpr_subclaim.shifted_bit_slice_evals,
            &public_bit_slice_evals,
            &field_cfg,
        );
        closing_overrides_tail.extend(virtual_bp_overrides);
    }
    closing_overrides_tail.extend(
        uair_signature
            .int_witness_bit_cols()
            .iter()
            .map(|&idx| cpr_subclaim.up_evals[int_offset + idx].clone()),
    );
    if !virtual_specs.is_empty() {
        let int_witness_up_evals: Vec<F> = (0..num_wit_int)
            .map(|i| cpr_subclaim.up_evals[int_offset + num_pub_int + i].clone())
            .collect();
        let virtual_overrides = compute_virtual_closing_overrides::<F, D>(
            virtual_specs,
            &cpr_subclaim.bit_slice_evals[..num_binary_bit_slices],
            &cpr_subclaim.shifted_bit_slice_evals,
            &public_bit_slice_evals,
            &int_witness_up_evals,
            &field_cfg,
        );
        closing_overrides_tail.extend(virtual_overrides);
    }
    if let Some(ba) = bool_verifier_ancillary_opt {
        finalize_booleanity_verifier::<F>(
            &mut pcs_transcript.fs_transcript,
            &cpr_subclaim.bit_slice_evals,
            &closing_overrides_tail,
            md_subclaims.point(),
            md_subclaims.expected_evaluations()[1].clone(),
            ba,
            &field_cfg,
        )
        .map_err(ProtocolError::Booleanity)?;
    }

    let parent_evals_for_bool = filter_skipped_parent_evals(
        &cpr_subclaim.up_evals[num_pub_bin..num_total_bin],
        bool_skip,
    );
    let num_kept_binary_bit_slices = parent_evals_for_bool.len() * D;
    verify_bit_decomposition_consistency(
        &parent_evals_for_bool,
        &cpr_subclaim.bit_slice_evals[..num_kept_binary_bit_slices],
        &projecting_element_f,
        D,
    )
    .map_err(ProtocolError::Booleanity)?;

    // Shifted bit-slice consistency (see unfolded `verify` for rationale).
    let shifted_down_indices = uair_signature.shifted_bit_slice_down_indices();
    let shifted_parent_evals: Vec<F> = shifted_down_indices
        .iter()
        .map(|&i| cpr_subclaim.down_evals[i].clone())
        .collect();
    verify_bit_decomposition_consistency(
        &shifted_parent_evals,
        &cpr_subclaim.shifted_bit_slice_evals,
        &projecting_element_f,
        D,
    )
    .map_err(ProtocolError::Booleanity)?;

    // ── Step 5: Multipoint eval ─────────────────────────────────────────
    let cpr_eval_point = cpr_subclaim.evaluation_point.clone();
    let mut up_evals_with_bit_op = cpr_subclaim.up_evals.clone();
    up_evals_with_bit_op.extend(cpr_subclaim.bit_op_down_evals.iter().cloned());
    let mp_subclaim = MultipointEval::verify_as_subprotocol(
        &mut pcs_transcript.fs_transcript,
        proof.multipoint_eval,
        &cpr_eval_point,
        &up_evals_with_bit_op,
        &cpr_subclaim.down_evals,
        uair_signature.shifts(),
        num_vars,
        &field_cfg,
    )?;
    let r_0 = mp_subclaim.sumcheck_subclaim.point.clone();

    // ── Step 6: Lifted evals ────────────────────────────────────────────
    let pub_cols = uair_signature.public_cols();
    let num_pub_bin = pub_cols.num_binary_poly_cols();
    let num_pub_arb = pub_cols.num_arbitrary_poly_cols();
    let num_pub_int = pub_cols.num_int_cols();
    let wit_cols = uair_signature.witness_cols();
    let num_wit_bin = wit_cols.num_binary_poly_cols();
    let num_wit_arb = wit_cols.num_arbitrary_poly_cols();

    let public_lifted = if add!(add!(num_pub_bin, num_pub_arb), num_pub_int) > 0 {
        let projected_public =
            project_trace_coeffs_row_major::<F, ZtF::Int, ZtF::Int, D>(public_trace, &field_cfg);
        crate::compute_lifted_evals::<F, D>(
            &r_0,
            &public_trace.binary_poly,
            &ProjectedTrace::RowMajor(projected_public),
            &field_cfg,
        )
    } else {
        Vec::new()
    };

    let witness_lifted_evals = &proof.witness_lifted_evals;
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
        .map(|bar_u| bar_u.evaluate_at_point(&projecting_element_f))
        .collect::<Result<Vec<_>, _>>()
        .map_err(ProtocolError::LiftedEvalProjection)?;
    // Bit-op virtual MLE consistency: derive the bit-op slot's
    // open_eval locally by applying the structural permutation
    // (rot_c / shift_r_c) to the source's lifted eval, then
    // ψ_α-projecting. Mismatch surfaces as
    // `MultipointEval(ClaimMismatch)`.
    for spec in uair_signature.bit_op_specs() {
        let bar_u_src = &all_lifted_evals[spec.source_col()];
        let op_e_src = match spec.op() {
            BitOp::Rot(c) => bar_u_src.rot_c(c),
            BitOp::ShiftR(c) => bar_u_src.shift_r_c(c),
        };
        let psi = op_e_src
            .evaluate_at_point(&projecting_element_f)
            .map_err(ProtocolError::LiftedEvalProjection)?;
        open_evals.push(psi);
    }

    MultipointEval::verify_subclaim(
        &mp_subclaim,
        &open_evals,
        uair_signature.shifts(),
        &field_cfg,
    )?;

    let mut transcription_buf: Vec<u8> = vec![0; F::Inner::NUM_BYTES];
    for bar_u in &all_lifted_evals {
        pcs_transcript
            .fs_transcript
            .absorb_random_field_slice(&bar_u.coeffs, &mut transcription_buf);
    }

    // Sample γ — the folding challenge — matching the prover's order
    // (after lifted_evals, before any PCS verify).
    let gamma: F = {
        let g_chal: ZtF::Chal = pcs_transcript.fs_transcript.get_challenge();
        F::from_with_cfg(&g_chal, &field_cfg)
    };
    let mut r0_ext = r_0.clone();
    r0_ext.push(gamma.clone());

    // ── Step 7: PCS verify ──────────────────────────────────────────────
    let total = uair_signature.total_cols();
    let num_total_bin = total.num_binary_poly_cols();
    let num_total_arb = total.num_arbitrary_poly_cols();

    // Binary: split commitment, opening at extended point. eval_f =
    // sum_j [(1−γ) · <a', bar_u_j.coeffs[0..HALF_D]> +
    //               γ  · <a', bar_u_j.coeffs[HALF_D..D]>],
    // where a' are the split-commitment per-poly alphas (length HALF_D)
    // sampled by the PCS, and bar_u_j is the witness lifted_eval for the
    // j-th binary column.
    {
        let comm = &proof.commitments.0;
        if comm.batch_size > 0 {
            let per_poly_alphas =
                ZipPlus::<ZtF::BinaryZt, ZtF::BinaryLc>::sample_alphas(
                    &mut pcs_transcript.fs_transcript,
                    comm.batch_size,
                );

            let one = F::one_with_cfg(&field_cfg);
            let one_minus_gamma = one - gamma.clone();
            let zero = F::zero_with_cfg(&field_cfg);
            let mut eval_f = zero.clone();

            for (bar_u, alphas) in all_lifted_evals[num_pub_bin..num_total_bin]
                .iter()
                .zip(per_poly_alphas.iter())
            {
                debug_assert_eq!(alphas.len(), HALF_D);
                let mut c1 = zero.clone();
                let mut c2 = zero.clone();
                for l in 0..HALF_D {
                    let a_l: F = F::from_with_cfg(&alphas[l], &field_cfg);

                    if let Some(coeff_lo) = bar_u.coeffs.get(l) {
                        let mut term = a_l.clone();
                        term *= coeff_lo;
                        c1 += &term;
                    }
                    if let Some(coeff_hi) = bar_u.coeffs.get(l + HALF_D) {
                        let mut term = a_l;
                        term *= coeff_hi;
                        c2 += &term;
                    }
                }
                let mut folded = one_minus_gamma.clone();
                folded *= c1;
                let mut g_c2 = gamma.clone();
                g_c2 *= c2;
                folded += &g_c2;
                eval_f += &folded;
            }

            ZipPlus::<ZtF::BinaryZt, ZtF::BinaryLc>::verify_with_alphas::<F, CHECK_FOR_OVERFLOW>(
                &mut pcs_transcript,
                vp_bin_split,
                comm,
                &field_cfg,
                &r0_ext,
                &eval_f,
                &per_poly_alphas,
            )
            .map_err(|e| ProtocolError::PcsVerification(0, e))?;
        }
    }

    // Arbitrary: standard verify at r_0 (unchanged from unfolded).
    {
        let comm = &proof.commitments.1;
        if comm.batch_size > 0 {
            let per_poly_alphas =
                ZipPlus::<ZtF::ArbitraryZt, ZtF::ArbitraryLc>::sample_alphas(
                    &mut pcs_transcript.fs_transcript,
                    comm.batch_size,
                );
            let mut eval_f = F::zero_with_cfg(&field_cfg);
            for (bar_u, alphas) in all_lifted_evals
                [add!(num_total_bin, num_pub_arb)..add!(num_total_bin, num_total_arb)]
                .iter()
                .zip(per_poly_alphas.iter())
            {
                for (coeff, alpha) in bar_u.coeffs.iter().zip(alphas.iter()) {
                    let mut term = F::from_with_cfg(alpha, &field_cfg);
                    term *= coeff;
                    eval_f += &term;
                }
            }
            ZipPlus::<ZtF::ArbitraryZt, ZtF::ArbitraryLc>::verify_with_alphas::<
                F,
                CHECK_FOR_OVERFLOW,
            >(
                &mut pcs_transcript,
                vp_arb,
                comm,
                &field_cfg,
                &r_0,
                &eval_f,
                &per_poly_alphas,
            )
            .map_err(|e| ProtocolError::PcsVerification(1, e))?;
        }
    }

    // Int: standard verify at r_0.
    {
        let comm = &proof.commitments.2;
        if comm.batch_size > 0 {
            let per_poly_alphas = ZipPlus::<ZtF::IntZt, ZtF::IntLc>::sample_alphas(
                &mut pcs_transcript.fs_transcript,
                comm.batch_size,
            );
            let mut eval_f = F::zero_with_cfg(&field_cfg);
            for (bar_u, alphas) in all_lifted_evals
                [add!(add!(num_total_bin, num_total_arb), num_pub_int)..]
                .iter()
                .zip(per_poly_alphas.iter())
            {
                for (coeff, alpha) in bar_u.coeffs.iter().zip(alphas.iter()) {
                    let mut term = F::from_with_cfg(alpha, &field_cfg);
                    term *= coeff;
                    eval_f += &term;
                }
            }
            ZipPlus::<ZtF::IntZt, ZtF::IntLc>::verify_with_alphas::<F, CHECK_FOR_OVERFLOW>(
                &mut pcs_transcript,
                vp_int,
                comm,
                &field_cfg,
                &r_0,
                &eval_f,
                &per_poly_alphas,
            )
            .map_err(|e| ProtocolError::PcsVerification(2, e))?;
        }
    }

    Ok(())
}

//
// Folded verifier (4× fold counterpart of `verify_folded`).
//
// Mirrors `verify_folded` but expects the binary commitment to be over
// `BinaryPoly<QUARTER_D>` columns of length `4n`, opens at the doubly-
// extended point `(r_0 ‖ γ₁ ‖ γ₂)`, and reconstructs the binary PCS
// `eval_f` from the proof's `witness_lifted_evals` coefficient eighths.
//
// # The folding formula
//
// After two chained 2× splits, a witness column `v` with `BinaryPoly<D>`
// entries (D=32) becomes a column `v''` with `BinaryPoly<QUARTER_D>`
// entries (QUARTER_D=8) and `4n` length. The MLE has `num_vars + 2`
// variables. By induction on `split_column`'s "low halves first, high
// halves second" layout, evaluating `v''_proj_via_a'` at `(r_0 ‖ γ₁ ‖ γ₂)`
// equals (per polynomial):
//
// ```text
// (1−γ₁)(1−γ₂) · ⟨a', coeffs[0..8]⟩    (block 00, bits  0..8)
//   + γ₁(1−γ₂) · ⟨a', coeffs[16..24]⟩  (block 10, bits 16..24)
//   + (1−γ₁)γ₂ · ⟨a', coeffs[8..16]⟩   (block 01, bits  8..16)
//   + γ₁γ₂     · ⟨a', coeffs[24..32]⟩  (block 11, bits 24..32)
// ```
//
// where `a'` are the split commitment's per-poly alphas (length QUARTER_D)
// and `coeffs[k..k+8]` is the slice of `lifted_eval_v.coeffs`
// corresponding to bits `[k, k+8)` of `v`. The bit-range pairing
// `(0..8, 16..24, 8..16, 24..32)` reflects the bit-reverse permutation
// induced by chained 2× splits. `eval_f` is the sum across all binary
// witness columns.
//

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn verify_folded_4x<
    ZtF,
    U,
    F,
    IdealOverF,
    const D: usize,
    const HALF_D: usize,
    const QUARTER_D: usize,
    const CHECK_FOR_OVERFLOW: bool,
>(
    vp: &(
        ZipPlusParams<ZtF::BinaryZt, ZtF::BinaryLc>,
        ZipPlusParams<ZtF::ArbitraryZt, ZtF::ArbitraryLc>,
        ZipPlusParams<ZtF::IntZt, ZtF::IntLc>,
    ),
    mut proof: Proof<F>,
    public_trace: &UairTrace<ZtF::Int, ZtF::Int, D>,
    num_vars: usize,
    project_scalar: impl Fn(&U::Scalar, &F::Config) -> DynamicPolynomialF<F> + Sync,
    project_ideal: impl Fn(&IdealOrZero<U::Ideal>, &F::Config) -> IdealOverF,
) -> Result<(), ProtocolError<F, IdealOverF>>
where
    ZtF: crate::FoldedZincTypes<D, QUARTER_D>,
    ZtF::Int: ProjectableToField<F> + num_traits::Zero,
    <ZtF::ArbitraryZt as ZipTypes>::Eval: ProjectableToField<F>,
    <ZtF::BinaryZt as ZipTypes>::Cw: ProjectableToField<F>,
    <ZtF::ArbitraryZt as ZipTypes>::Cw: ProjectableToField<F>,
    <ZtF::IntZt as ZipTypes>::Cw: ProjectableToField<F>,
    U: Uair + 'static,
    F: InnerTransparentField
        + FromPrimitiveWithConfig
        + for<'b> FromWithConfig<&'b ZtF::Int>
        + for<'b> FromWithConfig<&'b <ZtF::BinaryZt as ZipTypes>::CombR>
        + for<'b> FromWithConfig<&'b <ZtF::ArbitraryZt as ZipTypes>::CombR>
        + for<'b> FromWithConfig<&'b <ZtF::IntZt as ZipTypes>::CombR>
        + for<'b> FromWithConfig<&'b ZtF::Chal>
        + for<'b> MulByScalar<&'b F>
        + FromRef<F>
        + Send
        + Sync
        + 'static,
    F::Inner: ConstIntSemiring + ConstTranscribable + Send + Sync + Zero + Default,
    F::Modulus: ConstTranscribable + FromRef<ZtF::Fmod>,
    IdealOverF: Ideal + IdealCheck<DynamicPolynomialF<F>>,
{
    // Statically enforce the splitting tree D → HALF_D → QUARTER_D.
    debug_assert_eq!(D, 2 * HALF_D, "verify_folded_4x: D must equal 2 * HALF_D");
    debug_assert_eq!(
        HALF_D,
        2 * QUARTER_D,
        "verify_folded_4x: HALF_D must equal 2 * QUARTER_D",
    );

    // Verifier-side public-column structural checks (compensator/
    // corrector zero-pinning, etc.). Default impl is a no-op.
    U::verify_public_structure(public_trace, num_vars)
        .map_err(ProtocolError::PublicStructure)?;

    // ── Step 0: Reconstruct transcript ──────────────────────────────────
    let zip_proof = std::mem::take(&mut proof.zip);
    let (vp_bin_split2, vp_arb, vp_int) = vp;
    let uair_signature = U::signature();
    let mut pcs_transcript = PcsVerifierTranscript {
        fs_transcript: Blake3Transcript::default(),
        stream: Cursor::new(zip_proof),
    };
    for comm in [
        &proof.commitments.0,
        &proof.commitments.1,
        &proof.commitments.2,
    ] {
        pcs_transcript.fs_transcript.absorb_slice(&comm.root);
    }
    absorb_public_columns(&mut pcs_transcript.fs_transcript, &public_trace.binary_poly);
    absorb_public_columns(
        &mut pcs_transcript.fs_transcript,
        &public_trace.arbitrary_poly,
    );
    absorb_public_columns(&mut pcs_transcript.fs_transcript, &public_trace.int);

    // ── Step 1: Prime projection ────────────────────────────────────────
    // `fixed-prime` branch: match the non-folded path and use the
    // secp256k1 base prime as the projecting prime. See the prover-side
    // comment in `prove_folded` for why UAIRs with EC arithmetic need
    // the fixed prime here.
    let field_cfg = crate::fixed_prime::secp256k1_field_cfg::<F, ZtF::Fmod>();

    // ── Step 2: Ideal check ─────────────────────────────────────────────
    let num_constraints = count_constraints::<U>();
    let ic_subclaim = U::verify_as_subprotocol::<_, IdealOverF, _>(
        &mut pcs_transcript.fs_transcript,
        proof.ideal_check,
        num_constraints,
        num_vars,
        |ideal| project_ideal(ideal, &field_cfg),
        &field_cfg,
    )?;

    // ── Step 3: Eval projection ─────────────────────────────────────────
    let projecting_element: ZtF::Chal = pcs_transcript.fs_transcript.get_challenge();
    let projecting_element_f: F = F::from_with_cfg(&projecting_element, &field_cfg);

    let projected_scalars_fx = project_scalars::<F, U>(|s| project_scalar(s, &field_cfg));
    let projected_scalars_f =
        project_scalars_to_field(projected_scalars_fx, &projecting_element_f)
            .map_err(|(_s, _f, e)| ProtocolError::ScalarProjection(e))?;

    // ── Step 4: Sumcheck verify (CPR + algebraic booleanity) ────────────
    let num_pub_bin = uair_signature.public_cols().num_binary_poly_cols();
    let num_total_bin = uair_signature.total_cols().num_binary_poly_cols();
    let bool_skip = uair_signature.booleanity_skip_indices();
    let num_int_bit_cols = uair_signature.int_witness_bit_cols().len();
    let num_virtual_cols = uair_signature.virtual_booleanity_cols().len();
    let num_virtual_bp_cols = uair_signature.virtual_binary_poly_cols().len();
    let num_bit_slices = ((num_total_bin - num_pub_bin) - bool_skip.len()) * D
        + num_virtual_bp_cols * D
        + num_int_bit_cols
        + num_virtual_cols;
    let num_shifted_bit_slices =
        uair_signature.shifted_bit_slice_specs().len() * D;
    let cpr_verifier_ancillary = CombinedPolyResolver::prepare_verifier::<U>(
        &mut pcs_transcript.fs_transcript,
        &proof.resolver,
        proof.combined_sumcheck.claimed_sums()[0].clone(),
        &ic_subclaim,
        num_constraints,
        num_bit_slices,
        num_shifted_bit_slices,
        num_vars,
        &projecting_element_f,
        &field_cfg,
    )?;

    let bool_verifier_ancillary_opt = if num_bit_slices > 0 {
        let bool_claimed_sum = proof.combined_sumcheck.claimed_sums()[1].clone();
        prepare_booleanity_verifier::<F>(
            &mut pcs_transcript.fs_transcript,
            bool_claimed_sum,
            num_bit_slices,
            &ic_subclaim.evaluation_point,
            &field_cfg,
        )
        .map_err(ProtocolError::Booleanity)?
    } else {
        None
    };

    let md_subclaims = MultiDegreeSumcheck::verify_as_subprotocol(
        &mut pcs_transcript.fs_transcript,
        num_vars,
        &proof.combined_sumcheck,
        &field_cfg,
    )
    .map_err(combined_poly_resolver::CombinedPolyResolverError::SumcheckError)?;
    let cpr_subclaim = CombinedPolyResolver::finalize_verifier::<U>(
        &mut pcs_transcript.fs_transcript,
        proof.resolver,
        md_subclaims.point().to_vec(),
        md_subclaims.expected_evaluations()[0].clone(),
        cpr_verifier_ancillary,
        &projected_scalars_f,
        &field_cfg,
    )?;

    // Booleanity: substitute up_evals (for int bit cols) and computed
    // linear-combo evals (for virtual cols) into the closing.
    let int_offset = uair_signature.total_cols().num_binary_poly_cols()
        + uair_signature.total_cols().num_arbitrary_poly_cols();
    let num_pub_int = uair_signature.public_cols().num_int_cols();
    let num_wit_int = uair_signature.witness_cols().num_int_cols();
    let num_binary_bit_slices = (num_total_bin - num_pub_bin) * D;
    let virtual_bp_specs = uair_signature.virtual_binary_poly_cols();
    let virtual_specs = uair_signature.virtual_booleanity_cols();
    let public_bit_slice_evals: Vec<F> = if !virtual_bp_specs.is_empty()
        || !virtual_specs.is_empty()
    {
        let public_bit_slice_mles = compute_bit_slices_flat::<F, D>(
            &public_trace.binary_poly,
            &field_cfg,
        );
        public_bit_slice_mles
            .into_iter()
            .map(|mle| mle.evaluate_with_config(md_subclaims.point(), &field_cfg))
            .collect::<Result<Vec<_>, _>>()
            .map_err(ProtocolError::ShiftedBitSliceEval)?
    } else {
        Vec::new()
    };
    let mut closing_overrides_tail: Vec<F> = Vec::new();
    if !virtual_bp_specs.is_empty() {
        let virtual_bp_overrides = compute_virtual_binary_poly_closing_overrides::<F, D>(
            virtual_bp_specs,
            &cpr_subclaim.bit_slice_evals[..num_binary_bit_slices],
            &cpr_subclaim.shifted_bit_slice_evals,
            &public_bit_slice_evals,
            &field_cfg,
        );
        closing_overrides_tail.extend(virtual_bp_overrides);
    }
    closing_overrides_tail.extend(
        uair_signature
            .int_witness_bit_cols()
            .iter()
            .map(|&idx| cpr_subclaim.up_evals[int_offset + idx].clone()),
    );
    if !virtual_specs.is_empty() {
        let int_witness_up_evals: Vec<F> = (0..num_wit_int)
            .map(|i| cpr_subclaim.up_evals[int_offset + num_pub_int + i].clone())
            .collect();
        let virtual_overrides = compute_virtual_closing_overrides::<F, D>(
            virtual_specs,
            &cpr_subclaim.bit_slice_evals[..num_binary_bit_slices],
            &cpr_subclaim.shifted_bit_slice_evals,
            &public_bit_slice_evals,
            &int_witness_up_evals,
            &field_cfg,
        );
        closing_overrides_tail.extend(virtual_overrides);
    }
    if let Some(ba) = bool_verifier_ancillary_opt {
        finalize_booleanity_verifier::<F>(
            &mut pcs_transcript.fs_transcript,
            &cpr_subclaim.bit_slice_evals,
            &closing_overrides_tail,
            md_subclaims.point(),
            md_subclaims.expected_evaluations()[1].clone(),
            ba,
            &field_cfg,
        )
        .map_err(ProtocolError::Booleanity)?;
    }

    let parent_evals_for_bool = filter_skipped_parent_evals(
        &cpr_subclaim.up_evals[num_pub_bin..num_total_bin],
        bool_skip,
    );
    let num_kept_binary_bit_slices = parent_evals_for_bool.len() * D;
    verify_bit_decomposition_consistency(
        &parent_evals_for_bool,
        &cpr_subclaim.bit_slice_evals[..num_kept_binary_bit_slices],
        &projecting_element_f,
        D,
    )
    .map_err(ProtocolError::Booleanity)?;

    // Shifted bit-slice consistency (see unfolded `verify` for rationale).
    let shifted_down_indices = uair_signature.shifted_bit_slice_down_indices();
    let shifted_parent_evals: Vec<F> = shifted_down_indices
        .iter()
        .map(|&i| cpr_subclaim.down_evals[i].clone())
        .collect();
    verify_bit_decomposition_consistency(
        &shifted_parent_evals,
        &cpr_subclaim.shifted_bit_slice_evals,
        &projecting_element_f,
        D,
    )
    .map_err(ProtocolError::Booleanity)?;

    // ── Step 5: Multipoint eval ─────────────────────────────────────────
    let cpr_eval_point = cpr_subclaim.evaluation_point.clone();
    let mut up_evals_with_bit_op = cpr_subclaim.up_evals.clone();
    up_evals_with_bit_op.extend(cpr_subclaim.bit_op_down_evals.iter().cloned());
    let mp_subclaim = MultipointEval::verify_as_subprotocol(
        &mut pcs_transcript.fs_transcript,
        proof.multipoint_eval,
        &cpr_eval_point,
        &up_evals_with_bit_op,
        &cpr_subclaim.down_evals,
        uair_signature.shifts(),
        num_vars,
        &field_cfg,
    )?;
    let r_0 = mp_subclaim.sumcheck_subclaim.point.clone();

    // ── Step 6: Lifted evals ────────────────────────────────────────────
    let pub_cols = uair_signature.public_cols();
    let num_pub_bin = pub_cols.num_binary_poly_cols();
    let num_pub_arb = pub_cols.num_arbitrary_poly_cols();
    let num_pub_int = pub_cols.num_int_cols();
    let wit_cols = uair_signature.witness_cols();
    let num_wit_bin = wit_cols.num_binary_poly_cols();
    let num_wit_arb = wit_cols.num_arbitrary_poly_cols();

    let public_lifted = if add!(add!(num_pub_bin, num_pub_arb), num_pub_int) > 0 {
        let projected_public =
            project_trace_coeffs_row_major::<F, ZtF::Int, ZtF::Int, D>(public_trace, &field_cfg);
        crate::compute_lifted_evals::<F, D>(
            &r_0,
            &public_trace.binary_poly,
            &ProjectedTrace::RowMajor(projected_public),
            &field_cfg,
        )
    } else {
        Vec::new()
    };

    let witness_lifted_evals = &proof.witness_lifted_evals;
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
        .map(|bar_u| bar_u.evaluate_at_point(&projecting_element_f))
        .collect::<Result<Vec<_>, _>>()
        .map_err(ProtocolError::LiftedEvalProjection)?;
    // Bit-op virtual MLE consistency at r_0 — see `verify_folded`.
    for spec in uair_signature.bit_op_specs() {
        let bar_u_src = &all_lifted_evals[spec.source_col()];
        let op_e_src = match spec.op() {
            BitOp::Rot(c) => bar_u_src.rot_c(c),
            BitOp::ShiftR(c) => bar_u_src.shift_r_c(c),
        };
        let psi = op_e_src
            .evaluate_at_point(&projecting_element_f)
            .map_err(ProtocolError::LiftedEvalProjection)?;
        open_evals.push(psi);
    }

    MultipointEval::verify_subclaim(
        &mp_subclaim,
        &open_evals,
        uair_signature.shifts(),
        &field_cfg,
    )?;

    let mut transcription_buf: Vec<u8> = vec![0; F::Inner::NUM_BYTES];
    for bar_u in &all_lifted_evals {
        pcs_transcript
            .fs_transcript
            .absorb_random_field_slice(&bar_u.coeffs, &mut transcription_buf);
    }

    // Sample γ₁ then γ₂, matching the prover's order.
    let gamma1: F = {
        let g_chal: ZtF::Chal = pcs_transcript.fs_transcript.get_challenge();
        F::from_with_cfg(&g_chal, &field_cfg)
    };
    let gamma2: F = {
        let g_chal: ZtF::Chal = pcs_transcript.fs_transcript.get_challenge();
        F::from_with_cfg(&g_chal, &field_cfg)
    };
    let mut r0_ext = r_0.clone();
    r0_ext.push(gamma1.clone());
    r0_ext.push(gamma2.clone());

    // ── Step 7: PCS verify ──────────────────────────────────────────────
    let total = uair_signature.total_cols();
    let num_total_bin = total.num_binary_poly_cols();
    let num_total_arb = total.num_arbitrary_poly_cols();

    // Binary: split² commitment, opening at (r_0 ‖ γ₁ ‖ γ₂). eval_f is the
    // four-corner bilinear interpolation of the per-coefficient projections
    // of `bar_u.coeffs` quarter-slices, summed across binary witness cols.
    {
        let comm = &proof.commitments.0;
        if comm.batch_size > 0 {
            let per_poly_alphas =
                ZipPlus::<ZtF::BinaryZt, ZtF::BinaryLc>::sample_alphas(
                    &mut pcs_transcript.fs_transcript,
                    comm.batch_size,
                );

            let one = F::one_with_cfg(&field_cfg);
            let one_minus_g1 = one.clone() - gamma1.clone();
            let one_minus_g2 = one - gamma2.clone();
            // Bilinear weights for the four corners of (γ₁, γ₂):
            //   w00 = (1-γ₁)(1-γ₂),  w10 = γ₁(1-γ₂)
            //   w01 = (1-γ₁)γ₂,      w11 = γ₁γ₂
            let w00 = one_minus_g1.clone() * one_minus_g2.clone();
            let w10 = gamma1.clone() * one_minus_g2;
            let w01 = one_minus_g1 * gamma2.clone();
            let w11 = gamma1 * gamma2;

            let zero = F::zero_with_cfg(&field_cfg);
            let mut eval_f = zero.clone();

            for (bar_u, alphas) in all_lifted_evals[num_pub_bin..num_total_bin]
                .iter()
                .zip(per_poly_alphas.iter())
            {
                debug_assert_eq!(alphas.len(), QUARTER_D);

                // Per-quarter projections via the QUARTER_D-length per-poly
                // alphas. Coefficient bit-ranges follow the bit-reverse
                // permutation from chained 2× splits:
                //   c00 ↔ bits [0..8],   c10 ↔ bits [16..24]
                //   c01 ↔ bits [8..16],  c11 ↔ bits [24..32]
                let mut c00 = zero.clone();
                let mut c10 = zero.clone();
                let mut c01 = zero.clone();
                let mut c11 = zero.clone();
                for l in 0..QUARTER_D {
                    let a_l: F = F::from_with_cfg(&alphas[l], &field_cfg);

                    if let Some(coeff) = bar_u.coeffs.get(l) {
                        let mut term = a_l.clone();
                        term *= coeff;
                        c00 += &term;
                    }
                    if let Some(coeff) = bar_u.coeffs.get(l + HALF_D) {
                        let mut term = a_l.clone();
                        term *= coeff;
                        c10 += &term;
                    }
                    if let Some(coeff) = bar_u.coeffs.get(l + QUARTER_D) {
                        let mut term = a_l.clone();
                        term *= coeff;
                        c01 += &term;
                    }
                    if let Some(coeff) = bar_u.coeffs.get(l + HALF_D + QUARTER_D) {
                        let mut term = a_l;
                        term *= coeff;
                        c11 += &term;
                    }
                }

                let mut folded = w00.clone();
                folded *= c00;
                let mut t = w10.clone();
                t *= c10;
                folded += &t;
                let mut t = w01.clone();
                t *= c01;
                folded += &t;
                let mut t = w11.clone();
                t *= c11;
                folded += &t;

                eval_f += &folded;
            }

            ZipPlus::<ZtF::BinaryZt, ZtF::BinaryLc>::verify_with_alphas::<F, CHECK_FOR_OVERFLOW>(
                &mut pcs_transcript,
                vp_bin_split2,
                comm,
                &field_cfg,
                &r0_ext,
                &eval_f,
                &per_poly_alphas,
            )
            .map_err(|e| ProtocolError::PcsVerification(0, e))?;
        }
    }

    // Arbitrary: standard verify at r_0 (unchanged from unfolded).
    {
        let comm = &proof.commitments.1;
        if comm.batch_size > 0 {
            let per_poly_alphas =
                ZipPlus::<ZtF::ArbitraryZt, ZtF::ArbitraryLc>::sample_alphas(
                    &mut pcs_transcript.fs_transcript,
                    comm.batch_size,
                );
            let mut eval_f = F::zero_with_cfg(&field_cfg);
            for (bar_u, alphas) in all_lifted_evals
                [add!(num_total_bin, num_pub_arb)..add!(num_total_bin, num_total_arb)]
                .iter()
                .zip(per_poly_alphas.iter())
            {
                for (coeff, alpha) in bar_u.coeffs.iter().zip(alphas.iter()) {
                    let mut term = F::from_with_cfg(alpha, &field_cfg);
                    term *= coeff;
                    eval_f += &term;
                }
            }
            ZipPlus::<ZtF::ArbitraryZt, ZtF::ArbitraryLc>::verify_with_alphas::<
                F,
                CHECK_FOR_OVERFLOW,
            >(
                &mut pcs_transcript,
                vp_arb,
                comm,
                &field_cfg,
                &r_0,
                &eval_f,
                &per_poly_alphas,
            )
            .map_err(|e| ProtocolError::PcsVerification(1, e))?;
        }
    }

    // Int: standard verify at r_0.
    {
        let comm = &proof.commitments.2;
        if comm.batch_size > 0 {
            let per_poly_alphas = ZipPlus::<ZtF::IntZt, ZtF::IntLc>::sample_alphas(
                &mut pcs_transcript.fs_transcript,
                comm.batch_size,
            );
            let mut eval_f = F::zero_with_cfg(&field_cfg);
            for (bar_u, alphas) in all_lifted_evals
                [add!(add!(num_total_bin, num_total_arb), num_pub_int)..]
                .iter()
                .zip(per_poly_alphas.iter())
            {
                for (coeff, alpha) in bar_u.coeffs.iter().zip(alphas.iter()) {
                    let mut term = F::from_with_cfg(alpha, &field_cfg);
                    term *= coeff;
                    eval_f += &term;
                }
            }
            ZipPlus::<ZtF::IntZt, ZtF::IntLc>::verify_with_alphas::<F, CHECK_FOR_OVERFLOW>(
                &mut pcs_transcript,
                vp_int,
                comm,
                &field_cfg,
                &r_0,
                &eval_f,
                &per_poly_alphas,
            )
            .map_err(|e| ProtocolError::PcsVerification(2, e))?;
        }
    }

    Ok(())
}
