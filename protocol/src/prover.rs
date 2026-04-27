use super::*;
use crypto_primitives::{ConstIntSemiring, FromPrimitiveWithConfig, FromWithConfig};
use num_traits::Zero;
use std::{collections::HashMap, fmt::Debug};
use zinc_piop::{
    combined_poly_resolver::CombinedPolyResolver,
    ideal_check::IdealCheckProtocol,
    lookup::booleanity::compute_bit_slices_flat,
    multipoint_eval::{MultipointEval, Proof as MultipointEvalProof},
    projections::{
        ColumnMajorTrace, ProjectedTrace, RowMajorTrace, evaluate_trace_to_column_mles,
        project_scalars, project_scalars_to_field, project_trace_coeffs_column_major,
        project_trace_coeffs_row_major,
    },
    sumcheck::multi_degree::MultiDegreeSumcheck,
};
use zinc_poly::univariate::dynamic::over_field::DynamicPolynomialF;
use zinc_transcript::traits::{ConstTranscribable, Transcript};
use zinc_uair::{
    Uair, UairSignature, UairTrace, constraint_counter::count_constraints,
    degree_counter::count_max_degree,
};
use zinc_utils::{
    add, cfg_join, from_ref::FromRef, inner_transparent_field::InnerTransparentField,
    mul_by_scalar::MulByScalar, projectable_to_field::ProjectableToField,
};
use zip_plus::{
    pcs::structs::{ZipPlus, ZipPlusHint, ZipPlusParams, ZipTypes},
    pcs_transcript::PcsProverTranscript,
};

//
// Shared base
//

/// Persistent prover infrastructure carried across every step: the
/// Fiat-Shamir transcript, PCS parameters/hints/commitments, and trace
/// reference.
#[derive(Clone, Debug)]
pub struct ProverBase<'a, Zt: ZincTypes<D>, U: Uair, F: PrimeField, const D: usize> {
    num_vars: usize,
    uair_signature: UairSignature,
    pcs_transcript: PcsProverTranscript,
    trace: &'a UairTrace<'static, Zt::Int, Zt::Int, D>,

    // Commitment info
    pp_bin: &'a ZipPlusParams<Zt::BinaryZt, Zt::BinaryLc>,
    pp_arb: &'a ZipPlusParams<Zt::ArbitraryZt, Zt::ArbitraryLc>,
    pp_int: &'a ZipPlusParams<Zt::IntZt, Zt::IntLc>,
    hint_bin: Option<ZipPlusHint<<Zt::BinaryZt as ZipTypes>::Cw>>,
    hint_arb: Option<ZipPlusHint<<Zt::ArbitraryZt as ZipTypes>::Cw>>,
    hint_int: Option<ZipPlusHint<<Zt::IntZt as ZipTypes>::Cw>>,
    commitment_bin: ZipPlusCommitment,
    commitment_arb: ZipPlusCommitment,
    commitment_int: ZipPlusCommitment,

    _phantom: PhantomData<(U, F)>,
}

//
// Type-state structs
//

/// After step 1 via [`step1_combined`](ProverCommitted::step1_combined)
/// (row-major / "combined" projection). `project_scalar` has been consumed.
#[derive(Clone, Debug)]
pub struct ProverProjectedCombined<'a, Zt: ZincTypes<D>, U: Uair, F: PrimeField, const D: usize> {
    base: ProverBase<'a, Zt, U, F, D>,
    field_cfg: F::Config,
    projected_trace: RowMajorTrace<F>,
    projected_scalars_fx: HashMap<U::Scalar, DynamicPolynomialF<F>>,
}

/// After step 1 via [`step1_mle_first`](ProverCommitted::step1_mle_first)
/// (column-major / MLE-first projection). `project_scalar` has been consumed.
#[derive(Clone, Debug)]
pub struct ProverProjectedMleFirst<'a, Zt: ZincTypes<D>, U: Uair, F: PrimeField, const D: usize> {
    base: ProverBase<'a, Zt, U, F, D>,
    field_cfg: F::Config,
    projected_trace: ColumnMajorTrace<F>,
    projected_scalars_fx: HashMap<U::Scalar, DynamicPolynomialF<F>>,
}

/// After step 1 via [`step1_hybrid`](ProverCommitted::step1_hybrid). Carries
/// **both** layouts because the hybrid ideal-check feeds linear constraints
/// through the MLE-first lane (column-major) and non-linear constraints
/// through the combined-poly lane (row-major). `project_scalar` has been
/// consumed.
#[derive(Clone, Debug)]
pub struct ProverProjectedHybrid<'a, Zt: ZincTypes<D>, U: Uair, F: PrimeField, const D: usize> {
    base: ProverBase<'a, Zt, U, F, D>,
    field_cfg: F::Config,
    row_major_trace: RowMajorTrace<F>,
    column_major_trace: ColumnMajorTrace<F>,
    projected_scalars_fx: HashMap<U::Scalar, DynamicPolynomialF<F>>,
}

/// After step 2 (ideal check).
#[derive(Clone, Debug)]
pub struct ProverIdealChecked<'a, Zt: ZincTypes<D>, U: Uair, F: PrimeField, const D: usize> {
    base: ProverBase<'a, Zt, U, F, D>,
    field_cfg: F::Config,
    projected_trace: ProjectedTrace<F>,
    projected_scalars_fx: HashMap<U::Scalar, DynamicPolynomialF<F>>,

    // New
    ic_proof: IdealCheckProof<F>,
    ic_eval_point: Vec<F>,
}

/// After step 3 (eval projection). `projected_scalars_fx` has been consumed.
#[derive(Clone, Debug)]
pub struct ProverEvalProjected<'a, Zt: ZincTypes<D>, U: Uair, F: PrimeField, const D: usize> {
    base: ProverBase<'a, Zt, U, F, D>,
    field_cfg: F::Config,
    projected_trace: ProjectedTrace<F>,
    ic_proof: IdealCheckProof<F>,
    ic_eval_point: Vec<F>,

    // New
    projected_trace_f: Vec<DenseMultilinearExtension<F::Inner>>,
    projected_scalars_f: HashMap<U::Scalar, F>,
}

/// After step 4 (sumcheck).
#[allow(clippy::type_complexity)]
#[derive(Clone, Debug)]
pub struct ProverSumchecked<'a, Zt: ZincTypes<D>, U: Uair, F: PrimeField, const D: usize> {
    base: ProverBase<'a, Zt, U, F, D>,
    field_cfg: F::Config,
    projected_trace: ProjectedTrace<F>,
    ic_proof: IdealCheckProof<F>,
    projected_trace_f: Vec<DenseMultilinearExtension<F::Inner>>,

    // New
    cpr_proof: CombinedPolyResolverProof<F>,
    cpr_eval_point: Vec<F>,
    combined_sumcheck: MultiDegreeSumcheckProof<F>,
    lookup_proof: Option<BatchedLookupProof<F>>,
}

/// After step 5 (multipoint eval).
#[derive(Clone, Debug)]
pub struct ProverMultipointEvaled<'a, Zt: ZincTypes<D>, U: Uair, F: PrimeField, const D: usize> {
    base: ProverBase<'a, Zt, U, F, D>,
    field_cfg: F::Config,
    projected_trace: ProjectedTrace<F>,
    ic_proof: IdealCheckProof<F>,
    cpr_proof: CombinedPolyResolverProof<F>,
    combined_sumcheck: MultiDegreeSumcheckProof<F>,
    lookup_proof: Option<BatchedLookupProof<F>>,

    // New
    mp_proof: MultipointEvalProof<F>,
    r_0: Vec<F>,
}

/// After step 6 (lift-and-project).
#[derive(Clone, Debug)]
pub struct ProverLifted<'a, Zt: ZincTypes<D>, U: Uair, F: PrimeField, const D: usize> {
    base: ProverBase<'a, Zt, U, F, D>,
    field_cfg: F::Config,
    ic_proof: IdealCheckProof<F>,
    cpr_proof: CombinedPolyResolverProof<F>,
    combined_sumcheck: MultiDegreeSumcheckProof<F>,
    lookup_proof: Option<BatchedLookupProof<F>>,
    mp_proof: MultipointEvalProof<F>,
    r_0: Vec<F>,

    // New
    lifted_evals: Vec<DynamicPolynomialF<F>>,
}

/// After step 7 (PCS open). No new fields are added here, but the PCS
/// transcript has been updated with the opening proof.
/// Ready for generating the final proof object in
/// [`finish`](ProverPcsOpened::finish).
#[derive(Clone, Debug)]
pub struct ProverPcsOpened<'a, Zt: ZincTypes<D>, U: Uair, F: PrimeField, const D: usize> {
    base: ProverBase<'a, Zt, U, F, D>,
    ic_proof: IdealCheckProof<F>,
    cpr_proof: CombinedPolyResolverProof<F>,
    combined_sumcheck: MultiDegreeSumcheckProof<F>,
    lookup_proof: Option<BatchedLookupProof<F>>,
    mp_proof: MultipointEvalProof<F>,
    lifted_evals: Vec<DynamicPolynomialF<F>>,
}

//
// Step implementations
//

/// Prover uses common type bounds across all steps, so we use a helper macro to
/// define them
macro_rules! impl_with_type_bounds {
    ($type_name:ident { $($code:tt)* }) => {
        impl<'a, Zt, U, F, const D: usize> $type_name<'a, Zt, U, F, D>
        where
            Zt: ZincTypes<D>,
            Zt::Int: ProjectableToField<F>,
            <Zt::ArbitraryZt as ZipTypes>::Eval: ProjectableToField<F>,
            U: Uair + 'static,
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
            F::Inner:
                ConstIntSemiring + ConstTranscribable + FromRef<Zt::Fmod> + Send + Sync + Zero + Default,
            F::Modulus: ConstTranscribable + FromRef<Zt::Fmod>,
        {
            $($code)*
        }
    };
}

impl<Zt, U, F, const D: usize> ZincPlusPiop<Zt, U, F, D>
where
    Zt: ZincTypes<D>,
    U: Uair,
    F: PrimeField,
    F::Inner: ConstTranscribable,
{
    /// Step 0: Prover entry point.
    /// Commit *witness* columns via Zip+ PCS, absorb roots and public
    /// data into the Fiat-Shamir transcript.
    #[allow(clippy::type_complexity)]
    pub fn step0_commit<'a>(
        (pp_bin, pp_arb, pp_int): &'a (
            ZipPlusParams<Zt::BinaryZt, Zt::BinaryLc>,
            ZipPlusParams<Zt::ArbitraryZt, Zt::ArbitraryLc>,
            ZipPlusParams<Zt::IntZt, Zt::IntLc>,
        ),
        trace: &'a UairTrace<'static, Zt::Int, Zt::Int, D>,
        num_vars: usize,
    ) -> Result<ProverBase<'a, Zt, U, F, D>, ProtocolError<F, U::Ideal>> {
        let uair_signature = U::signature();
        let public_trace = trace.public(&uair_signature);
        let witness_trace = trace.witness(&uair_signature);

        let (res_bin, (res_arb, res_int)) = cfg_join!(
            commit_optionally(pp_bin, &witness_trace.binary_poly),
            commit_optionally(pp_arb, &witness_trace.arbitrary_poly),
            commit_optionally(pp_int, &witness_trace.int),
        );
        let (hint_bin, commitment_bin) = res_bin?;
        let (hint_arb, commitment_arb) = res_arb?;
        let (hint_int, commitment_int) = res_int?;

        let mut pcs_transcript = PcsProverTranscript::new_from_commitments(
            [&commitment_bin, &commitment_arb, &commitment_int].into_iter(),
        );

        absorb_public_columns(&mut pcs_transcript.fs_transcript, &public_trace.binary_poly);
        absorb_public_columns(
            &mut pcs_transcript.fs_transcript,
            &public_trace.arbitrary_poly,
        );
        absorb_public_columns(&mut pcs_transcript.fs_transcript, &public_trace.int);

        Ok(ProverBase {
            num_vars,
            uair_signature,
            pcs_transcript,
            trace,
            pp_bin,
            pp_arb,
            pp_int,
            hint_bin,
            hint_arb,
            hint_int,
            commitment_bin,
            commitment_arb,
            commitment_int,
            _phantom: PhantomData,
        })
    }
}

impl_with_type_bounds!(ProverBase
{
    #[allow(clippy::type_complexity)]
    fn project_common<S: Fn(&U::Scalar, &F::Config) -> DynamicPolynomialF<F>>(
        &mut self,
        project_scalar: S,
    ) -> Result<(F::Config, HashMap<U::Scalar, DynamicPolynomialF<F>>), ProtocolError<F, U::Ideal>>
    {
        // `fixed-prime` branch: use the secp256k1 base field prime as the
        // projecting prime instead of drawing one from the transcript.
        // See `crate::fixed_prime` for the soundness caveat.
        let field_cfg = crate::fixed_prime::secp256k1_field_cfg::<F, Zt::Fmod>();

        let projected_scalars_fx = project_scalars::<F, U>(|s| project_scalar(s, &field_cfg));
        Ok((field_cfg, projected_scalars_fx))
    }

    /// Step 1 (combined / row-major): Prime projection
    /// (`\phi_q`: `Z[X] -> F_q[X]`). Samples a random prime, projects the
    /// full trace and scalars using the row-major layout.
    /// Works for both linear and non-linear constraints.
    pub fn step1_combined<S: Fn(&U::Scalar, &F::Config) -> DynamicPolynomialF<F>>(
        mut self,
        project_scalar: S,
    ) -> Result<ProverProjectedCombined<'a, Zt, U, F, D>, ProtocolError<F, U::Ideal>> {
        let (field_cfg, projected_scalars_fx) = self.project_common(project_scalar)?;

        let projected_trace = project_trace_coeffs_row_major(self.trace, &field_cfg);
        Ok(ProverProjectedCombined {
            base: self,
            field_cfg,
            projected_trace,
            projected_scalars_fx,
        })
    }

    /// Step 1 (MLE-first / column-major): Prime projection
    /// (`\phi_q`: `Z[X] -> F_q[X]`). Samples a random prime, projects the
    /// full trace and scalars using the column-major layout.
    /// Only suitable for linear constraints.
    pub fn step1_mle_first<S: Fn(&U::Scalar, &F::Config) -> DynamicPolynomialF<F>>(
        mut self,
        project_scalar: S,
    ) -> Result<ProverProjectedMleFirst<'a, Zt, U, F, D>, ProtocolError<F, U::Ideal>> {
        let (field_cfg, projected_scalars_fx) = self.project_common(project_scalar)?;

        let projected_trace = project_trace_coeffs_column_major(self.trace, &field_cfg);
        Ok(ProverProjectedMleFirst {
            base: self,
            field_cfg,
            projected_trace,
            projected_scalars_fx,
        })
    }

    /// Step 1 (hybrid): Prime projection that produces **both** layouts.
    /// Used when the UAIR has a mix of linear and non-linear constraints,
    /// so the ideal-check can route them through their respective fast/slow
    /// lanes. Costs roughly the sum of `step1_combined` and
    /// `step1_mle_first` projection times.
    pub fn step1_hybrid<S: Fn(&U::Scalar, &F::Config) -> DynamicPolynomialF<F>>(
        mut self,
        project_scalar: S,
    ) -> Result<ProverProjectedHybrid<'a, Zt, U, F, D>, ProtocolError<F, U::Ideal>> {
        let (field_cfg, projected_scalars_fx) = self.project_common(project_scalar)?;

        let row_major_trace = project_trace_coeffs_row_major(self.trace, &field_cfg);
        let column_major_trace = project_trace_coeffs_column_major(self.trace, &field_cfg);
        Ok(ProverProjectedHybrid {
            base: self,
            field_cfg,
            row_major_trace,
            column_major_trace,
            projected_scalars_fx,
        })
    }
});

impl_with_type_bounds!(ProverProjectedCombined
{
    /// Step 2 (combined): Ideal check via `prove_combined` on the row-major
    /// trace. Works for both linear and non-linear constraints.
    pub fn step2_ideal_check(
        mut self,
    ) -> Result<ProverIdealChecked<'a, Zt, U, F, D>, ProtocolError<F, U::Ideal>> {
        let num_constraints = count_constraints::<U>();

        let (ic_proof, ic_prover_state) = U::prove_combined(
            &mut self.base.pcs_transcript.fs_transcript,
            &self.projected_trace,
            &self.projected_scalars_fx,
            num_constraints,
            self.base.num_vars,
            &self.field_cfg,
        )?;

        Ok(ProverIdealChecked {
            base: self.base,
            field_cfg: self.field_cfg,
            projected_trace: ProjectedTrace::RowMajor(self.projected_trace),
            projected_scalars_fx: self.projected_scalars_fx,
            ic_proof,
            ic_eval_point: ic_prover_state.evaluation_point,
        })
    }
});

impl_with_type_bounds!(ProverProjectedMleFirst
{
    /// Step 2 (MLE-first): Ideal check via `prove_linear` on the column-major
    /// trace. Only suitable for linear constraints.
    pub fn step2_ideal_check(
        mut self,
    ) -> Result<ProverIdealChecked<'a, Zt, U, F, D>, ProtocolError<F, U::Ideal>> {
        let num_constraints = count_constraints::<U>();

        let (ic_proof, ic_prover_state) = U::prove_linear(
            &mut self.base.pcs_transcript.fs_transcript,
            &self.projected_trace,
            &self.projected_scalars_fx,
            num_constraints,
            self.base.num_vars,
            &self.field_cfg,
        )?;

        Ok(ProverIdealChecked {
            base: self.base,
            field_cfg: self.field_cfg,
            projected_trace: ProjectedTrace::ColumnMajor(self.projected_trace),
            projected_scalars_fx: self.projected_scalars_fx,
            ic_proof,
            ic_eval_point: ic_prover_state.evaluation_point,
        })
    }
});

impl_with_type_bounds!(ProverProjectedHybrid
{
    /// Step 2 (hybrid): Ideal check via `prove_hybrid`, which routes linear
    /// constraints through the MLE-first lane and non-linear constraints
    /// through the combined-poly lane, merging into a single per-constraint
    /// claim list. Works for any UAIR; useful when the UAIR has a mix of
    /// linear and non-linear constraints.
    pub fn step2_ideal_check(
        mut self,
    ) -> Result<ProverIdealChecked<'a, Zt, U, F, D>, ProtocolError<F, U::Ideal>> {
        let num_constraints = count_constraints::<U>();

        let (ic_proof, ic_prover_state) = U::prove_hybrid(
            &mut self.base.pcs_transcript.fs_transcript,
            &self.row_major_trace,
            &self.column_major_trace,
            &self.projected_scalars_fx,
            num_constraints,
            self.base.num_vars,
            &self.field_cfg,
        )?;

        // Downstream Steps 3-7 work generically over `ProjectedTrace`. Pass
        // the row-major arm — it's what step3_eval_projection's
        // `evaluate_trace_to_column_mles` consumes most directly.
        Ok(ProverIdealChecked {
            base: self.base,
            field_cfg: self.field_cfg,
            projected_trace: ProjectedTrace::RowMajor(self.row_major_trace),
            projected_scalars_fx: self.projected_scalars_fx,
            ic_proof,
            ic_eval_point: ic_prover_state.evaluation_point,
        })
    }
});

impl_with_type_bounds!(ProverIdealChecked
{
    /// Step 3: Evaluation projection (`\psi_a`: `F_q[X] -> F_q`). Samples
    /// `a in F_q`, evaluates polynomials at `X = a`.
    pub fn step3_eval_projection(
        mut self,
    ) -> Result<ProverEvalProjected<'a, Zt, U, F, D>, ProtocolError<F, U::Ideal>> {
        let projecting_element: Zt::Chal = self.base.pcs_transcript.fs_transcript.get_challenge();
        let projecting_element_f: F = F::from_with_cfg(&projecting_element, &self.field_cfg);

        let projected_trace_f =
            evaluate_trace_to_column_mles(&self.projected_trace, &projecting_element_f);

        let projected_scalars_f =
            project_scalars_to_field(self.projected_scalars_fx, &projecting_element_f)
                .map_err(|(_s, _f, e)| ProtocolError::ScalarProjection(e))?;

        Ok(ProverEvalProjected {
            base: self.base,
            field_cfg: self.field_cfg,
            projected_trace: self.projected_trace,
            ic_proof: self.ic_proof,
            ic_eval_point: self.ic_eval_point,
            projected_trace_f,
            projected_scalars_f,
        })
    }
});

impl_with_type_bounds!(ProverEvalProjected
{
    /// Step 4: Combined CPR + Lookup multi-degree sumcheck over F_q.
    /// Batches the CPR constraint claim (degree `max_deg+2`) with lookup groups
    /// (one per table type) into a single sumcheck sharing one evaluation point `r*`.
    /// Produces `up_evals` and `down_evals` (CPR) and lookup auxiliary witnesses at `r*`.
    pub fn step4_sumcheck(
        mut self,
    ) -> Result<ProverSumchecked<'a, Zt, U, F, D>, ProtocolError<F, U::Ideal>> {
        let num_constraints = count_constraints::<U>();
        // Sumcheck protocol degree must accommodate the actual fold
        // polynomial's per-variable degree, including `assert_zero`
        // constraints. Although their values are identically zero on the
        // hypercube for an honest prover, the polynomial expression has
        // non-trivial per-variable degree (e.g. `b·(b-1)·s_accum` is
        // degree 3), and it appears in the fold via
        // `ConstraintFolder::assert_zero`. Their inclusion is what binds
        // the committed witness to the assert_zero claims (otherwise a
        // dishonest prover could violate them undetected).
        let max_degree = count_max_degree::<U>();

        let bit_slices = compute_bit_slices_flat::<F, D>(
            &self.base.trace.binary_poly,
            &self.field_cfg,
        );

        let (cpr_group, cpr_ancillary) = CombinedPolyResolver::prepare_sumcheck_group::<U>(
            &mut self.base.pcs_transcript.fs_transcript,
            self.projected_trace_f.clone(),
            bit_slices,
            &self.ic_eval_point,
            &self.projected_scalars_f,
            num_constraints,
            self.base.num_vars,
            max_degree,
            &self.field_cfg,
        )?;

        // 4b: Lookup prepare — placeholder
        let lookup_specs = &self.base.uair_signature;
        let groups = vec![cpr_group];
        // TODO: for each LookupGroup from group_lookup_specs(lookup_specs):
        //   - call prepare_batched_lookup_group(transcript, instance, &field_cfg)
        //   - push triple into groups, collect pending proofs + metas
        let _ = lookup_specs; // suppress unused warning until logup is implemented

        // 4c: Multi-degree sumcheck width CPR group and lookup groups
        let (combined_sumcheck, md_states) = MultiDegreeSumcheck::prove_as_subprotocol(
            &mut self.base.pcs_transcript.fs_transcript,
            groups,
            self.base.num_vars,
            &self.field_cfg,
        );
        // 4c: Finalize up_evals, down_evals
        let (cpr_proof, cpr_prover_state) = CombinedPolyResolver::finalize_prover(
            &mut self.base.pcs_transcript.fs_transcript,
            md_states.into_iter().next().expect("one CPR group"),
            cpr_ancillary,
            &self.field_cfg,
        )?;

        // TODO: build BatchedLookupProof from collected lookup_proofs + lookup_metas
        let lookup_proof = None;

        Ok(ProverSumchecked {
            base: self.base,
            field_cfg: self.field_cfg,
            projected_trace: self.projected_trace,
            ic_proof: self.ic_proof,
            projected_trace_f: self.projected_trace_f,
            cpr_proof,
            cpr_eval_point: cpr_prover_state.evaluation_point,
            combined_sumcheck,
            lookup_proof,
        })
    }
});

impl_with_type_bounds!(ProverSumchecked
{
    /// Step 5: Multi-point evaluation sumcheck. Combines `up_evals` and
    /// `down_evals` at `r'` into a single evaluation point `r_0`.
    /// Only the sumcheck proof is sent; scalar evaluations at `r_0` are derived from the
    /// polynomial-valued `lifted_evals` in Step 6
    pub fn step5_multipoint_eval(
        mut self,
    ) -> Result<ProverMultipointEvaled<'a, Zt, U, F, D>, ProtocolError<F, U::Ideal>> {
        let (mp_proof, mp_prover_state) = MultipointEval::prove_as_subprotocol(
            &mut self.base.pcs_transcript.fs_transcript,
            &self.projected_trace_f,
            &self.cpr_eval_point,
            &self.cpr_proof.up_evals,
            &self.cpr_proof.down_evals,
            self.base.uair_signature.shifts(),
            &self.field_cfg,
        )?;

        Ok(ProverMultipointEvaled {
            base: self.base,
            field_cfg: self.field_cfg,
            projected_trace: self.projected_trace,
            ic_proof: self.ic_proof,
            cpr_proof: self.cpr_proof,
            combined_sumcheck: self.combined_sumcheck,
            lookup_proof: self.lookup_proof,
            mp_proof,
            r_0: mp_prover_state.eval_point,
        })
    }
});

impl_with_type_bounds!(ProverMultipointEvaled
{
    /// Step 6: Lift-and-project. Computes per-column polynomial MLE
    /// evaluations at `r_0` in `F_q[X]` and absorbs them into the transcript.
    pub fn step6_lift_and_project(
        mut self,
    ) -> Result<ProverLifted<'a, Zt, U, F, D>, ProtocolError<F, U::Ideal>> {
        // Compute per-column polynomial MLE evaluations at r_0 in F_q[X]
        // (after \phi_q but before \psi_a). The verifier derives the scalar
        // open_evals via \psi_a for the sumcheck consistency check, and
        // supplies these to the Zip+ PCS for alpha-projection.
        let lifted_evals = compute_lifted_evals::<F, D>(
            &self.r_0,
            &self.base.trace.binary_poly,
            &self.projected_trace,
            &self.field_cfg,
        );

        let mut transcription_buf: Vec<u8> = vec![0; F::Inner::NUM_BYTES];
        for bar_u in &lifted_evals {
            self.base
                .pcs_transcript
                .fs_transcript
                .absorb_random_field_slice(&bar_u.coeffs, &mut transcription_buf);
        }

        Ok(ProverLifted {
            base: self.base,
            field_cfg: self.field_cfg,
            ic_proof: self.ic_proof,
            cpr_proof: self.cpr_proof,
            combined_sumcheck: self.combined_sumcheck,
            lookup_proof: self.lookup_proof,
            mp_proof: self.mp_proof,
            r_0: self.r_0,
            lifted_evals,
        })
    }
});

impl_with_type_bounds!(ProverLifted
{
    /// Step 7: PCS open at `r_0` (witness columns only).
    pub fn step7_pcs_open<const CHECK_FOR_OVERFLOW: bool>(
        mut self,
    ) -> Result<ProverPcsOpened<'a, Zt, U, F, D>, ProtocolError<F, U::Ideal>> {
        let witness_trace = self.base.trace.witness(&self.base.uair_signature);

        if let Some(hint_bin) = &self.base.hint_bin {
            let _ = ZipPlus::<Zt::BinaryZt, Zt::BinaryLc>::prove_f::<_, CHECK_FOR_OVERFLOW>(
                &mut self.base.pcs_transcript,
                self.base.pp_bin,
                &witness_trace.binary_poly,
                &self.r_0,
                hint_bin,
                &self.field_cfg,
            )?;
        }
        if let Some(hint_arb) = &self.base.hint_arb {
            let _ = ZipPlus::<Zt::ArbitraryZt, Zt::ArbitraryLc>::prove_f::<_, CHECK_FOR_OVERFLOW>(
                &mut self.base.pcs_transcript,
                self.base.pp_arb,
                &witness_trace.arbitrary_poly,
                &self.r_0,
                hint_arb,
                &self.field_cfg,
            )?;
        }
        if let Some(hint_int) = &self.base.hint_int {
            let _ = ZipPlus::<Zt::IntZt, Zt::IntLc>::prove_f::<_, CHECK_FOR_OVERFLOW>(
                &mut self.base.pcs_transcript,
                self.base.pp_int,
                &witness_trace.int,
                &self.r_0,
                hint_int,
                &self.field_cfg,
            )?;
        }

        Ok(ProverPcsOpened {
            base: self.base,
            ic_proof: self.ic_proof,
            cpr_proof: self.cpr_proof,
            combined_sumcheck: self.combined_sumcheck,
            lookup_proof: self.lookup_proof,
            mp_proof: self.mp_proof,
            lifted_evals: self.lifted_evals,
        })
    }
});

impl_with_type_bounds!(ProverPcsOpened
{
    /// Assemble the final proof from accumulated state.
    pub fn finish(self) -> Result<Proof<F>, ProtocolError<F, U::Ideal>> {
        let sig = self.base.uair_signature;
        let zip_proof = self.base.pcs_transcript.stream.into_inner();
        let commitments = (
            self.base.commitment_bin,
            self.base.commitment_arb,
            self.base.commitment_int,
        );

        let lifted_evals = self.lifted_evals;

        // Extract witness-only lifted evals (public columns come first in trace).
        let pub_cols = sig.public_cols();
        let num_pub_bin = pub_cols.num_binary_poly_cols();
        let num_pub_arb = pub_cols.num_arbitrary_poly_cols();
        let num_pub_int = pub_cols.num_int_cols();
        let total = sig.total_cols();
        let num_total_bin = total.num_binary_poly_cols();
        let num_total_arb = total.num_arbitrary_poly_cols();
        let witness = sig.witness_cols();
        let witness_arb_offset = add!(num_total_bin, num_pub_arb);
        let witness_arb_end = add!(witness_arb_offset, witness.num_arbitrary_poly_cols());
        let witness_int_offset = add!(add!(num_total_bin, num_total_arb), num_pub_int);
        let witness_lifted_evals: Vec<_> = lifted_evals[num_pub_bin..num_total_bin]
            .iter()
            .chain(&lifted_evals[witness_arb_offset..witness_arb_end])
            .chain(&lifted_evals[witness_int_offset..])
            .cloned()
            .collect();

        Ok(Proof {
            commitments,
            ideal_check: self.ic_proof,
            resolver: self.cpr_proof,
            combined_sumcheck: self.combined_sumcheck,
            multipoint_eval: self.mp_proof,
            zip: zip_proof,
            witness_lifted_evals,
            lookup_proof: self.lookup_proof,
        })
    }
});

//
// prove() wrapper
//

impl<Zt, U, F, const D: usize> ZincPlusPiop<Zt, U, F, D>
where
    Zt: ZincTypes<D>,
    Zt::Int: ProjectableToField<F>,
    <Zt::ArbitraryZt as ZipTypes>::Eval: ProjectableToField<F>,
    F: InnerTransparentField
        + FromPrimitiveWithConfig
        + for<'a> FromWithConfig<&'a Zt::Int>
        + for<'a> FromWithConfig<&'a Zt::CombR>
        + for<'a> FromWithConfig<&'a Zt::Chal>
        + for<'a> FromWithConfig<&'a Zt::Pt>
        + for<'a> MulByScalar<&'a F>
        + FromRef<F>
        + Send
        + Sync
        + 'static,
    F::Inner:
        ConstIntSemiring + ConstTranscribable + FromRef<Zt::Fmod> + Send + Sync + Zero + Default,
    F::Modulus: ConstTranscribable + FromRef<Zt::Fmod>,
    U: Uair + 'static,
{
    /// Zinc+ full PIOP prover.
    ///
    /// Runs all protocol steps in sequence and returns the assembled proof.
    /// For per-step control, start with [`Self::step0_commit`] and chain the
    /// individual `stepN_*` methods.
    ///
    /// # Path selection
    ///
    /// - `MLE_FIRST = false`: always uses the Combined path (row-major,
    ///   `prove_combined`). Works for any UAIR.
    /// - `MLE_FIRST = true`: dispatches at runtime based on per-constraint
    ///   degree:
    ///   - All non-zero-ideal constraints linear → MLE-first lane
    ///     (`prove_linear`).
    ///   - All non-zero-ideal constraints non-linear → falls back to
    ///     Combined.
    ///   - Mixed → Hybrid lane (`prove_hybrid`), which routes linear
    ///     constraints through MLE-first and non-linear through Combined,
    ///     merging into a single per-constraint claim list.
    ///   So `MLE_FIRST = true` is always safe and picks the cheapest
    ///   applicable path automatically.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub fn prove<const MLE_FIRST: bool, const CHECK_FOR_OVERFLOW: bool>(
        pp: &(
            ZipPlusParams<Zt::BinaryZt, Zt::BinaryLc>,
            ZipPlusParams<Zt::ArbitraryZt, Zt::ArbitraryLc>,
            ZipPlusParams<Zt::IntZt, Zt::IntLc>,
        ),
        trace: &UairTrace<'static, Zt::Int, Zt::Int, D>,
        num_vars: usize,
        project_scalar: impl Fn(&U::Scalar, &F::Config) -> DynamicPolynomialF<F>,
    ) -> Result<Proof<F>, ProtocolError<F, U::Ideal>> {
        let committed = Self::step0_commit(pp, trace, num_vars)?;

        let ideal_checked = if MLE_FIRST {
            // Classify constraints by degree, ignoring zero-ideal (their
            // value is substituted with ZERO regardless of lane).
            let mask = zinc_uair::degree_counter::linear_constraint_mask::<U>();
            let ideals = zinc_uair::ideal_collector::collect_ideals::<U>(
                zinc_uair::constraint_counter::count_constraints::<U>(),
            )
            .ideals;
            let (mut any_linear, mut any_nonlinear) = (false, false);
            for (m, i) in mask.iter().zip(ideals.iter()) {
                if i.is_zero_ideal() {
                    continue;
                }
                if *m { any_linear = true } else { any_nonlinear = true }
            }
            match (any_linear, any_nonlinear) {
                (true, false) => committed
                    .step1_mle_first(project_scalar)?
                    .step2_ideal_check()?,
                (false, _) => committed
                    .step1_combined(project_scalar)?
                    .step2_ideal_check()?,
                (true, true) => committed
                    .step1_hybrid(project_scalar)?
                    .step2_ideal_check()?,
            }
        } else {
            committed
                .step1_combined(project_scalar)?
                .step2_ideal_check()?
        };

        ideal_checked
            .step3_eval_projection()?
            .step4_sumcheck()?
            .step5_multipoint_eval()?
            .step6_lift_and_project()?
            .step7_pcs_open::<CHECK_FOR_OVERFLOW>()?
            .finish()
    }
}

#[allow(clippy::type_complexity)]
fn commit_optionally<Zt: ZipTypes, Lc: LinearCode<Zt>>(
    pp: &ZipPlusParams<Zt, Lc>,
    trace: &[DenseMultilinearExtension<Zt::Eval>],
) -> Result<(Option<ZipPlusHint<Zt::Cw>>, ZipPlusCommitment), ZipError> {
    if trace.is_empty() {
        Ok((
            None,
            ZipPlusCommitment {
                root: Default::default(),
                batch_size: 0,
            },
        ))
    } else {
        let (hint, commitment) = ZipPlus::commit(pp, trace)?;
        Ok((Some(hint), commitment))
    }
}

//
// Folded prover (1× fold, 2× column splitting on binary witness columns).
//
// The PIOP runs on the original `BinaryPoly<D>` trace; only the binary
// commit (step 0) and the binary PCS open (step 7) differ from the
// unfolded path. Steps 1–6 are inlined here so we don't duplicate the
// entire type-state chain on a separate `FoldedZincTypes` parameter.
//

/// Folded counterpart of [`ZincPlusPiop::prove`]. Runs the unfolded PIOP
/// on the original `BinaryPoly<D>` trace, but commits and opens binary
/// witness columns as `BinaryPoly<HALF_D>` halves of length `2n`. Returns
/// the same [`Proof<F>`] shape — the only difference visible to the
/// verifier is the binary PCS sub-proof.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn prove_folded<
    ZtF,
    U,
    F,
    const D: usize,
    const HALF_D: usize,
    const CHECK_FOR_OVERFLOW: bool,
>(
    pp: &(
        ZipPlusParams<ZtF::BinaryZt, ZtF::BinaryLc>,
        ZipPlusParams<ZtF::ArbitraryZt, ZtF::ArbitraryLc>,
        ZipPlusParams<ZtF::IntZt, ZtF::IntLc>,
    ),
    trace: &UairTrace<'static, ZtF::Int, ZtF::Int, D>,
    num_vars: usize,
    project_scalar: impl Fn(&U::Scalar, &F::Config) -> DynamicPolynomialF<F> + Sync,
) -> Result<Proof<F>, ProtocolError<F, U::Ideal>>
where
    ZtF: crate::FoldedZincTypes<D, HALF_D>,
    ZtF::Int: ProjectableToField<F>,
    <ZtF::ArbitraryZt as ZipTypes>::Eval: ProjectableToField<F>,
    BinaryPoly<HALF_D>: ProjectableToField<F>,
    U: Uair + 'static,
    F: InnerTransparentField
        + FromPrimitiveWithConfig
        + for<'a> FromWithConfig<&'a ZtF::Int>
        + for<'a> FromWithConfig<&'a ZtF::CombR>
        + for<'a> FromWithConfig<&'a ZtF::Chal>
        + for<'a> FromWithConfig<&'a ZtF::Pt>
        + for<'a> MulByScalar<&'a F>
        + FromRef<F>
        + Send
        + Sync
        + 'static,
    F::Inner:
        ConstIntSemiring + ConstTranscribable + FromRef<ZtF::Fmod> + Send + Sync + Zero + Default,
    F::Modulus: ConstTranscribable + FromRef<ZtF::Fmod>,
{
    let (pp_bin_split, pp_arb, pp_int) = pp;
    let uair_signature = U::signature();
    let public_trace = trace.public(&uair_signature);
    let witness_trace = trace.witness(&uair_signature);

    // ── Step 0: Commit (split binary, regular arb/int) ──────────────────
    let split_binary_witness: Vec<DenseMultilinearExtension<BinaryPoly<HALF_D>>> =
        zip_plus::pcs::folding::split_columns::<D, HALF_D>(&witness_trace.binary_poly);

    let (res_bin, (res_arb, res_int)) = cfg_join!(
        commit_optionally(pp_bin_split, &split_binary_witness),
        commit_optionally(pp_arb, &witness_trace.arbitrary_poly),
        commit_optionally(pp_int, &witness_trace.int),
    );
    let (hint_bin_split, commitment_bin) = res_bin?;
    let (hint_arb, commitment_arb) = res_arb?;
    let (hint_int, commitment_int) = res_int?;

    let mut pcs_transcript = PcsProverTranscript::new_from_commitments(
        [&commitment_bin, &commitment_arb, &commitment_int].into_iter(),
    );

    absorb_public_columns(&mut pcs_transcript.fs_transcript, &public_trace.binary_poly);
    absorb_public_columns(
        &mut pcs_transcript.fs_transcript,
        &public_trace.arbitrary_poly,
    );
    absorb_public_columns(&mut pcs_transcript.fs_transcript, &public_trace.int);

    // ── Step 1 (combined): Prime projection ─────────────────────────────
    let field_cfg = pcs_transcript
        .fs_transcript
        .get_random_field_cfg::<F, ZtF::Fmod, ZtF::PrimeTest>();

    let (projected_trace_rm, projected_scalars_fx) = cfg_join!(
        project_trace_coeffs_row_major::<F, _, _, D>(trace, &field_cfg),
        project_scalars::<F, U>(|s| project_scalar(s, &field_cfg)),
    );

    // ── Step 2 (combined): Ideal check ──────────────────────────────────
    let num_constraints = count_constraints::<U>();
    let (ic_proof, ic_prover_state) = U::prove_combined(
        &mut pcs_transcript.fs_transcript,
        &projected_trace_rm,
        &projected_scalars_fx,
        num_constraints,
        num_vars,
        &field_cfg,
    )?;
    let ic_eval_point = ic_prover_state.evaluation_point;
    let projected_trace = ProjectedTrace::RowMajor(projected_trace_rm);

    // ── Step 3: Eval projection (ψ_a) ───────────────────────────────────
    let projecting_element: ZtF::Chal = pcs_transcript.fs_transcript.get_challenge();
    let projecting_element_f: F = F::from_with_cfg(&projecting_element, &field_cfg);

    let projected_trace_f =
        evaluate_trace_to_column_mles(&projected_trace, &projecting_element_f);
    let projected_scalars_f =
        project_scalars_to_field(projected_scalars_fx, &projecting_element_f)
            .map_err(|(_s, _f, e)| ProtocolError::ScalarProjection(e))?;

    // ── Step 4: CPR + multi-degree sumcheck ─────────────────────────────
    let max_degree = count_max_degree::<U>();
    let bit_slices = compute_bit_slices_flat::<F, D>(&trace.binary_poly, &field_cfg);
    let (cpr_group, cpr_ancillary) = CombinedPolyResolver::prepare_sumcheck_group::<U>(
        &mut pcs_transcript.fs_transcript,
        projected_trace_f.clone(),
        bit_slices,
        &ic_eval_point,
        &projected_scalars_f,
        num_constraints,
        num_vars,
        max_degree,
        &field_cfg,
    )?;
    let groups = vec![cpr_group];
    let (combined_sumcheck, md_states) = MultiDegreeSumcheck::prove_as_subprotocol(
        &mut pcs_transcript.fs_transcript,
        groups,
        num_vars,
        &field_cfg,
    );
    let (cpr_proof, cpr_prover_state) = CombinedPolyResolver::finalize_prover(
        &mut pcs_transcript.fs_transcript,
        md_states.into_iter().next().expect("one CPR group"),
        cpr_ancillary,
        &field_cfg,
    )?;
    let lookup_proof: Option<BatchedLookupProof<F>> = None;

    // ── Step 5: Multi-point evaluation sumcheck ─────────────────────────
    let (mp_proof, mp_prover_state) = MultipointEval::prove_as_subprotocol(
        &mut pcs_transcript.fs_transcript,
        &projected_trace_f,
        &cpr_prover_state.evaluation_point,
        &cpr_proof.up_evals,
        &cpr_proof.down_evals,
        uair_signature.shifts(),
        &field_cfg,
    )?;
    let r_0 = mp_prover_state.eval_point;

    // ── Step 6: Lift-and-project + sample γ for folding ─────────────────
    let lifted_evals =
        compute_lifted_evals::<F, D>(&r_0, &trace.binary_poly, &projected_trace, &field_cfg);
    let mut transcription_buf: Vec<u8> = vec![0; F::Inner::NUM_BYTES];
    for bar_u in &lifted_evals {
        pcs_transcript
            .fs_transcript
            .absorb_random_field_slice(&bar_u.coeffs, &mut transcription_buf);
    }
    // After absorbing lifted_evals, sample the folding challenge γ. The
    // verifier replays this exact step before any PCS verify.
    let gamma: F = {
        let g_chal: ZtF::Chal = pcs_transcript.fs_transcript.get_challenge();
        F::from_with_cfg(&g_chal, &field_cfg)
    };

    // ── Step 7: PCS open. Binary uses split columns at extended point ───
    let mut r0_ext = r_0.clone();
    r0_ext.push(gamma);

    if let Some(hint_bin) = &hint_bin_split {
        let _ = ZipPlus::<ZtF::BinaryZt, ZtF::BinaryLc>::prove_f::<_, CHECK_FOR_OVERFLOW>(
            &mut pcs_transcript,
            pp_bin_split,
            &split_binary_witness,
            &r0_ext,
            hint_bin,
            &field_cfg,
        )?;
    }
    if let Some(hint_arb) = &hint_arb {
        let _ =
            ZipPlus::<ZtF::ArbitraryZt, ZtF::ArbitraryLc>::prove_f::<_, CHECK_FOR_OVERFLOW>(
                &mut pcs_transcript,
                pp_arb,
                &witness_trace.arbitrary_poly,
                &r_0,
                hint_arb,
                &field_cfg,
            )?;
    }
    if let Some(hint_int) = &hint_int {
        let _ = ZipPlus::<ZtF::IntZt, ZtF::IntLc>::prove_f::<_, CHECK_FOR_OVERFLOW>(
            &mut pcs_transcript,
            pp_int,
            &witness_trace.int,
            &r_0,
            hint_int,
            &field_cfg,
        )?;
    }

    // ── Assemble the proof ──────────────────────────────────────────────
    let zip_proof = pcs_transcript.stream.into_inner();
    let commitments = (commitment_bin, commitment_arb, commitment_int);

    let pub_cols = uair_signature.public_cols();
    let num_pub_bin = pub_cols.num_binary_poly_cols();
    let num_pub_arb = pub_cols.num_arbitrary_poly_cols();
    let num_pub_int = pub_cols.num_int_cols();
    let total = uair_signature.total_cols();
    let num_total_bin = total.num_binary_poly_cols();
    let num_total_arb = total.num_arbitrary_poly_cols();
    let witness = uair_signature.witness_cols();
    let witness_arb_offset = add!(num_total_bin, num_pub_arb);
    let witness_arb_end = add!(witness_arb_offset, witness.num_arbitrary_poly_cols());
    let witness_int_offset = add!(add!(num_total_bin, num_total_arb), num_pub_int);
    let witness_lifted_evals: Vec<_> = lifted_evals[num_pub_bin..num_total_bin]
        .iter()
        .chain(&lifted_evals[witness_arb_offset..witness_arb_end])
        .chain(&lifted_evals[witness_int_offset..])
        .cloned()
        .collect();

    Ok(Proof {
        commitments,
        ideal_check: ic_proof,
        resolver: cpr_proof,
        combined_sumcheck,
        multipoint_eval: mp_proof,
        zip: zip_proof,
        witness_lifted_evals,
        lookup_proof,
    })
}

//
// Folded prover (4× fold = 2 rounds, D → HALF_D → QUARTER_D).
//
// Splits each `BinaryPoly<D>` witness column into a length-4n column of
// `BinaryPoly<QUARTER_D>` entries (via two chained 2× splits), commits the
// twice-split column, and opens it at the doubly-extended PIOP point
// `(r_0 ‖ γ₁ ‖ γ₂)`. Compared to 1× fold, each codeword element is a quarter
// the original size — column-opening data drops accordingly.
//
// As with `prove_folded`, the PIOP runs unchanged on the original
// `BinaryPoly<D>` trace; the verifier reconstructs the binary PCS `eval_f`
// from `witness_lifted_evals` coefficient eighths (see `verify_folded_4x`).
//

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn prove_folded_4x<
    ZtF,
    U,
    F,
    const D: usize,
    const HALF_D: usize,
    const QUARTER_D: usize,
    const CHECK_FOR_OVERFLOW: bool,
>(
    pp: &(
        ZipPlusParams<ZtF::BinaryZt, ZtF::BinaryLc>,
        ZipPlusParams<ZtF::ArbitraryZt, ZtF::ArbitraryLc>,
        ZipPlusParams<ZtF::IntZt, ZtF::IntLc>,
    ),
    trace: &UairTrace<'static, ZtF::Int, ZtF::Int, D>,
    num_vars: usize,
    project_scalar: impl Fn(&U::Scalar, &F::Config) -> DynamicPolynomialF<F> + Sync,
) -> Result<Proof<F>, ProtocolError<F, U::Ideal>>
where
    ZtF: crate::FoldedZincTypes<D, QUARTER_D>,
    ZtF::Int: ProjectableToField<F>,
    <ZtF::ArbitraryZt as ZipTypes>::Eval: ProjectableToField<F>,
    BinaryPoly<HALF_D>: ProjectableToField<F>,
    BinaryPoly<QUARTER_D>: ProjectableToField<F>,
    U: Uair + 'static,
    F: InnerTransparentField
        + FromPrimitiveWithConfig
        + for<'a> FromWithConfig<&'a ZtF::Int>
        + for<'a> FromWithConfig<&'a ZtF::CombR>
        + for<'a> FromWithConfig<&'a ZtF::Chal>
        + for<'a> FromWithConfig<&'a ZtF::Pt>
        + for<'a> MulByScalar<&'a F>
        + FromRef<F>
        + Send
        + Sync
        + 'static,
    F::Inner:
        ConstIntSemiring + ConstTranscribable + FromRef<ZtF::Fmod> + Send + Sync + Zero + Default,
    F::Modulus: ConstTranscribable + FromRef<ZtF::Fmod>,
{
    let (pp_bin_split2, pp_arb, pp_int) = pp;
    let uair_signature = U::signature();
    let public_trace = trace.public(&uair_signature);
    let witness_trace = trace.witness(&uair_signature);

    // ── Step 0: Commit (split binary twice, regular arb/int) ────────────
    // First split: D → HALF_D, length n → 2n.
    let split1: Vec<DenseMultilinearExtension<BinaryPoly<HALF_D>>> =
        zip_plus::pcs::folding::split_columns::<D, HALF_D>(&witness_trace.binary_poly);
    // Second split: HALF_D → QUARTER_D, length 2n → 4n.
    let split2: Vec<DenseMultilinearExtension<BinaryPoly<QUARTER_D>>> =
        zip_plus::pcs::folding::split_columns::<HALF_D, QUARTER_D>(&split1);
    drop(split1);

    let (res_bin, (res_arb, res_int)) = cfg_join!(
        commit_optionally(pp_bin_split2, &split2),
        commit_optionally(pp_arb, &witness_trace.arbitrary_poly),
        commit_optionally(pp_int, &witness_trace.int),
    );
    let (hint_bin_split, commitment_bin) = res_bin?;
    let (hint_arb, commitment_arb) = res_arb?;
    let (hint_int, commitment_int) = res_int?;

    let mut pcs_transcript = PcsProverTranscript::new_from_commitments(
        [&commitment_bin, &commitment_arb, &commitment_int].into_iter(),
    );

    absorb_public_columns(&mut pcs_transcript.fs_transcript, &public_trace.binary_poly);
    absorb_public_columns(
        &mut pcs_transcript.fs_transcript,
        &public_trace.arbitrary_poly,
    );
    absorb_public_columns(&mut pcs_transcript.fs_transcript, &public_trace.int);

    // ── Step 1 (combined): Prime projection ─────────────────────────────
    let field_cfg = pcs_transcript
        .fs_transcript
        .get_random_field_cfg::<F, ZtF::Fmod, ZtF::PrimeTest>();

    let (projected_trace_rm, projected_scalars_fx) = cfg_join!(
        project_trace_coeffs_row_major::<F, _, _, D>(trace, &field_cfg),
        project_scalars::<F, U>(|s| project_scalar(s, &field_cfg)),
    );

    // ── Step 2 (combined): Ideal check ──────────────────────────────────
    let num_constraints = count_constraints::<U>();
    let (ic_proof, ic_prover_state) = U::prove_combined(
        &mut pcs_transcript.fs_transcript,
        &projected_trace_rm,
        &projected_scalars_fx,
        num_constraints,
        num_vars,
        &field_cfg,
    )?;
    let ic_eval_point = ic_prover_state.evaluation_point;
    let projected_trace = ProjectedTrace::RowMajor(projected_trace_rm);

    // ── Step 3: Eval projection (ψ_a) ───────────────────────────────────
    let projecting_element: ZtF::Chal = pcs_transcript.fs_transcript.get_challenge();
    let projecting_element_f: F = F::from_with_cfg(&projecting_element, &field_cfg);

    let projected_trace_f =
        evaluate_trace_to_column_mles(&projected_trace, &projecting_element_f);
    let projected_scalars_f =
        project_scalars_to_field(projected_scalars_fx, &projecting_element_f)
            .map_err(|(_s, _f, e)| ProtocolError::ScalarProjection(e))?;

    // ── Step 4: CPR + multi-degree sumcheck ─────────────────────────────
    let max_degree = count_max_degree::<U>();
    let bit_slices = compute_bit_slices_flat::<F, D>(&trace.binary_poly, &field_cfg);
    let (cpr_group, cpr_ancillary) = CombinedPolyResolver::prepare_sumcheck_group::<U>(
        &mut pcs_transcript.fs_transcript,
        projected_trace_f.clone(),
        bit_slices,
        &ic_eval_point,
        &projected_scalars_f,
        num_constraints,
        num_vars,
        max_degree,
        &field_cfg,
    )?;
    let groups = vec![cpr_group];
    let (combined_sumcheck, md_states) = MultiDegreeSumcheck::prove_as_subprotocol(
        &mut pcs_transcript.fs_transcript,
        groups,
        num_vars,
        &field_cfg,
    );
    let (cpr_proof, cpr_prover_state) = CombinedPolyResolver::finalize_prover(
        &mut pcs_transcript.fs_transcript,
        md_states.into_iter().next().expect("one CPR group"),
        cpr_ancillary,
        &field_cfg,
    )?;
    let lookup_proof: Option<BatchedLookupProof<F>> = None;

    // ── Step 5: Multi-point evaluation sumcheck ─────────────────────────
    let (mp_proof, mp_prover_state) = MultipointEval::prove_as_subprotocol(
        &mut pcs_transcript.fs_transcript,
        &projected_trace_f,
        &cpr_prover_state.evaluation_point,
        &cpr_proof.up_evals,
        &cpr_proof.down_evals,
        uair_signature.shifts(),
        &field_cfg,
    )?;
    let r_0 = mp_prover_state.eval_point;

    // ── Step 6: Lift-and-project + sample γ₁ and γ₂ ─────────────────────
    let lifted_evals =
        compute_lifted_evals::<F, D>(&r_0, &trace.binary_poly, &projected_trace, &field_cfg);
    let mut transcription_buf: Vec<u8> = vec![0; F::Inner::NUM_BYTES];
    for bar_u in &lifted_evals {
        pcs_transcript
            .fs_transcript
            .absorb_random_field_slice(&bar_u.coeffs, &mut transcription_buf);
    }
    // After absorbing lifted_evals, sample γ₁ then γ₂. Order matters — the
    // verifier replays in the same order before any PCS verify.
    let gamma1: F = {
        let g_chal: ZtF::Chal = pcs_transcript.fs_transcript.get_challenge();
        F::from_with_cfg(&g_chal, &field_cfg)
    };
    let gamma2: F = {
        let g_chal: ZtF::Chal = pcs_transcript.fs_transcript.get_challenge();
        F::from_with_cfg(&g_chal, &field_cfg)
    };

    // ── Step 7: PCS open ────────────────────────────────────────────────
    // Binary opens the twice-split column at point (r_0 ‖ γ₁ ‖ γ₂).
    let mut r0_ext = r_0.clone();
    r0_ext.push(gamma1);
    r0_ext.push(gamma2);

    if let Some(hint_bin) = &hint_bin_split {
        let _ = ZipPlus::<ZtF::BinaryZt, ZtF::BinaryLc>::prove_f::<_, CHECK_FOR_OVERFLOW>(
            &mut pcs_transcript,
            pp_bin_split2,
            &split2,
            &r0_ext,
            hint_bin,
            &field_cfg,
        )?;
    }
    if let Some(hint_arb) = &hint_arb {
        let _ =
            ZipPlus::<ZtF::ArbitraryZt, ZtF::ArbitraryLc>::prove_f::<_, CHECK_FOR_OVERFLOW>(
                &mut pcs_transcript,
                pp_arb,
                &witness_trace.arbitrary_poly,
                &r_0,
                hint_arb,
                &field_cfg,
            )?;
    }
    if let Some(hint_int) = &hint_int {
        let _ = ZipPlus::<ZtF::IntZt, ZtF::IntLc>::prove_f::<_, CHECK_FOR_OVERFLOW>(
            &mut pcs_transcript,
            pp_int,
            &witness_trace.int,
            &r_0,
            hint_int,
            &field_cfg,
        )?;
    }

    // ── Assemble the proof ──────────────────────────────────────────────
    let zip_proof = pcs_transcript.stream.into_inner();
    let commitments = (commitment_bin, commitment_arb, commitment_int);

    let pub_cols = uair_signature.public_cols();
    let num_pub_bin = pub_cols.num_binary_poly_cols();
    let num_pub_arb = pub_cols.num_arbitrary_poly_cols();
    let num_pub_int = pub_cols.num_int_cols();
    let total = uair_signature.total_cols();
    let num_total_bin = total.num_binary_poly_cols();
    let num_total_arb = total.num_arbitrary_poly_cols();
    let witness = uair_signature.witness_cols();
    let witness_arb_offset = add!(num_total_bin, num_pub_arb);
    let witness_arb_end = add!(witness_arb_offset, witness.num_arbitrary_poly_cols());
    let witness_int_offset = add!(add!(num_total_bin, num_total_arb), num_pub_int);
    let witness_lifted_evals: Vec<_> = lifted_evals[num_pub_bin..num_total_bin]
        .iter()
        .chain(&lifted_evals[witness_arb_offset..witness_arb_end])
        .chain(&lifted_evals[witness_int_offset..])
        .cloned()
        .collect();

    Ok(Proof {
        commitments,
        ideal_check: ic_proof,
        resolver: cpr_proof,
        combined_sumcheck,
        multipoint_eval: mp_proof,
        zip: zip_proof,
        witness_lifted_evals,
        lookup_proof,
    })
}
