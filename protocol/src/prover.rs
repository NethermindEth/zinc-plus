use super::*;
use crypto_primitives::{ConstIntSemiring, FromPrimitiveWithConfig, FromWithConfig};
use num_traits::Zero;
use std::fmt::Debug;
use zinc_piop::{
    combined_poly_resolver::CombinedPolyResolver,
    ideal_check::IdealCheckProtocol,
    lookup::logup_gkr::{LookupArgument, LookupArgumentSubclaim},
    multipoint_eval::{MultipointEval, Proof as MultipointEvalProof},
    projections::{
        ColumnMajorTrace, ProjectedTrace, RowMajorTrace, ScalarMap,
        evaluate_trace_to_column_mles, project_scalars, project_scalars_to_field,
        project_trace_coeffs_column_major, project_trace_coeffs_row_major,
    },
    sumcheck::multi_degree::MultiDegreeSumcheck,
};
use zinc_uair::LookupTableType;
use zinc_poly::univariate::dynamic::over_field::DynamicPolynomialF;
use zinc_transcript::traits::{ConstTranscribable, Transcript};
use zinc_uair::{
    Uair, UairSignature, UairTrace, constraint_counter::count_constraints,
    degree_counter::count_effective_max_degree,
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
    projected_scalars_fx: ScalarMap<U::Scalar, DynamicPolynomialF<F>>,
}

/// After step 1 via [`step1_mle_first`](ProverCommitted::step1_mle_first)
/// (column-major / MLE-first projection). `project_scalar` has been consumed.
#[derive(Clone, Debug)]
pub struct ProverProjectedMleFirst<'a, Zt: ZincTypes<D>, U: Uair, F: PrimeField, const D: usize> {
    base: ProverBase<'a, Zt, U, F, D>,
    field_cfg: F::Config,
    projected_trace: ColumnMajorTrace<F>,
    projected_scalars_fx: ScalarMap<U::Scalar, DynamicPolynomialF<F>>,
}

/// After step 2 (ideal check).
#[derive(Clone, Debug)]
pub struct ProverIdealChecked<'a, Zt: ZincTypes<D>, U: Uair, F: PrimeField, const D: usize> {
    base: ProverBase<'a, Zt, U, F, D>,
    field_cfg: F::Config,
    projected_trace: ProjectedTrace<F>,
    projected_scalars_fx: ScalarMap<U::Scalar, DynamicPolynomialF<F>>,

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
    projected_scalars_f: ScalarMap<U::Scalar, F>,
    /// Projecting element `a` used for ψ_a (persisted across later
    /// steps so lookup-table construction for `BitPoly` lookups can
    /// project bit-polys into the same field element).
    projecting_element_f: F,
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
    /// Carried through from step3 for step4b's BitPoly lookup-table
    /// construction.
    projecting_element_f: F,

    // New
    cpr_proof: CombinedPolyResolverProof<F>,
    cpr_eval_point: Vec<F>,
    combined_sumcheck: MultiDegreeSumcheckProof<F>,
    lookup_proof: Vec<LookupArgumentProof<F>>,
    /// Prover-only state from step4b for later binding (step5b, NYI).
    lookup_subclaims: Vec<LookupArgumentSubclaim<F>>,
}

/// After step 5 (multipoint eval + optional reducer). The `r_0` field
/// holds the *final opening point*: it is the multipoint-eval output
/// when there are no lookups, and the `MultiPointReducer` output
/// `r_final` when there are.
#[derive(Clone, Debug)]
pub struct ProverMultipointEvaled<'a, Zt: ZincTypes<D>, U: Uair, F: PrimeField, const D: usize> {
    base: ProverBase<'a, Zt, U, F, D>,
    field_cfg: F::Config,
    projected_trace: ProjectedTrace<F>,
    ic_proof: IdealCheckProof<F>,
    cpr_proof: CombinedPolyResolverProof<F>,
    combined_sumcheck: MultiDegreeSumcheckProof<F>,
    lookup_proof: Vec<LookupArgumentProof<F>>,

    // New
    mp_proof: MultipointEvalProof<F>,
    r_0: Vec<F>,
    lookup_reducer: Option<crate::LookupReducerProof<F>>,
}

/// After step 6 (lift-and-project).
#[derive(Clone, Debug)]
pub struct ProverLifted<'a, Zt: ZincTypes<D>, U: Uair, F: PrimeField, const D: usize> {
    base: ProverBase<'a, Zt, U, F, D>,
    field_cfg: F::Config,
    ic_proof: IdealCheckProof<F>,
    cpr_proof: CombinedPolyResolverProof<F>,
    combined_sumcheck: MultiDegreeSumcheckProof<F>,
    lookup_proof: Vec<LookupArgumentProof<F>>,
    mp_proof: MultipointEvalProof<F>,
    r_0: Vec<F>,
    lookup_reducer: Option<crate::LookupReducerProof<F>>,

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
    lookup_proof: Vec<LookupArgumentProof<F>>,
    mp_proof: MultipointEvalProof<F>,
    lifted_evals: Vec<DynamicPolynomialF<F>>,
    lookup_reducer: Option<crate::LookupReducerProof<F>>,
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
    ) -> Result<(F::Config, ScalarMap<U::Scalar, DynamicPolynomialF<F>>), ProtocolError<F, U::Ideal>>
    {
        let field_cfg = self
            .pcs_transcript
            .fs_transcript
            .get_random_field_cfg::<F, Zt::Fmod, Zt::PrimeTest>();

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
            projecting_element_f,
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
        // Skip zero-ideal constraints in the fq_sumcheck — their values are
        // identically zero on the hypercube, and ConstraintFolder::assert_zero
        // is a no-op. The effective max degree is the max over remaining
        // (non-zero-ideal) constraints only.
        let max_degree = count_effective_max_degree::<U>();

        let (cpr_group, cpr_ancillary) = CombinedPolyResolver::prepare_sumcheck_group::<U>(
            &mut self.base.pcs_transcript.fs_transcript,
            self.projected_trace_f.clone(),
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

        // Lookup argument is wired separately in step4b_lookup. For UAIRs
        // without lookup specs these stay empty.
        let lookup_proof: Vec<LookupArgumentProof<F>> = Vec::new();
        let lookup_subclaims: Vec<LookupArgumentSubclaim<F>> = Vec::new();

        Ok(ProverSumchecked {
            base: self.base,
            field_cfg: self.field_cfg,
            projected_trace: self.projected_trace,
            ic_proof: self.ic_proof,
            projected_trace_f: self.projected_trace_f,
            projecting_element_f: self.projecting_element_f,
            cpr_proof,
            cpr_eval_point: cpr_prover_state.evaluation_point,
            combined_sumcheck,
            lookup_proof,
            lookup_subclaims,
        })
    }
});

impl_with_type_bounds!(ProverSumchecked
{
    /// Step 4b: Per-lookup-group logup-GKR argument.
    ///
    /// Groups the UAIR's `LookupColumnSpec`s by `LookupTableType` via
    /// [`group_lookup_specs`], then runs one `LookupArgument::prove`
    /// per group — batching all `L >= 1` witness columns in the group
    /// under a single logup-GKR call. The resulting proofs are placed
    /// into `self.lookup_proof` and threaded through into the final
    /// `Proof<F>` via `finish()`; the subclaims (trace-level eval
    /// point ρ_row_g and claimed component evals) are kept as
    /// prover-only state for the step5/step5b binding.
    ///
    /// No-op for UAIRs with no lookup specs.
    ///
    /// # Multiplicity column convention (temporary)
    ///
    /// For this integration the multiplicity column for group `g` is
    /// taken to be the committed int column at *flat* index
    /// `total_cols - num_groups + g`. That is, the last `num_groups`
    /// int columns are treated as multiplicities, one per group. A
    /// future iteration will let the UAIR declare them explicitly.
    pub fn step4b_lookup(
        mut self,
    ) -> Result<Self, ProtocolError<F, U::Ideal>> {
        use zinc_piop::lookup::group_lookup_specs;

        let lookup_specs = self.base.uair_signature.lookup_specs().to_vec();
        if lookup_specs.is_empty() {
            return Ok(self);
        }

        let groups = group_lookup_specs(&lookup_specs);
        let num_cols = self.projected_trace_f.len();
        let num_groups = groups.len();

        for (group_idx, group) in groups.iter().enumerate() {
            let mult_idx = num_cols
                .checked_sub(num_groups)
                .and_then(|n| n.checked_add(group_idx))
                .expect("trace must contain one multiplicity col per lookup group");
            // Multiplicity must not appear in any lookup expression within
            // the group (the GKR argument would double-count).
            for expr in &group.expressions {
                for (wit_idx, _) in expr.terms.iter() {
                    assert_ne!(
                        *wit_idx, mult_idx,
                        "lookup expression cannot reference the multiplicity column"
                    );
                }
            }

            // Synthesize one dense MLE per affine expression. For
            // `AffineExpr::single(idx)` this is just a clone of
            // `projected_trace_f[idx]`. For real combos it computes
            // `Σ c_k · col_k + constant` row-by-row.
            let synthesized: Vec<DenseMultilinearExtension<F::Inner>> = group
                .expressions
                .iter()
                .map(|expr| {
                    synthesize_affine_mle::<F>(
                        expr,
                        &self.projected_trace_f,
                        &self.field_cfg,
                    )
                })
                .collect();
            let wit_mles: Vec<&DenseMultilinearExtension<F::Inner>> =
                synthesized.iter().collect();
            let mul_mle = &self.projected_trace_f[mult_idx];
            let table_mle = build_table_mle::<F>(
                &group.table_type,
                self.base.num_vars,
                &self.field_cfg,
                &self.projecting_element_f,
            );

            let (proof, subclaim) = LookupArgument::<F>::prove(
                &mut self.base.pcs_transcript.fs_transcript,
                &wit_mles,
                &table_mle,
                mul_mle,
                &self.field_cfg,
            );

            self.lookup_proof.push(proof);
            self.lookup_subclaims.push(subclaim);
        }

        Ok(self)
    }

    /// Step 5: Multi-point evaluation sumcheck. Combines `up_evals` and
    /// `down_evals` at `r'` into a single evaluation point `r_0`.
    ///
    /// Then, when there are lookup groups, runs `MultiPointReducer` as
    /// step 5b to fold the lookup `ρ_row_g` claims alongside `r_0` into
    /// a fresh `r_final`, which replaces `r_0` as the final opening
    /// point consumed by step6 / step7.
    ///
    /// When no lookups are present, the flow is unchanged (`r_final == r_0`
    /// and `lookup_reducer == None`).
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
        let r_0 = mp_prover_state.eval_point;

        // Step 5b: if the UAIR declared lookups, run MultiPointReducer to
        // fold (r_0, witness_evals_at_r_0) with each (ρ_row_g,
        // witness_evals_at_ρ_row_g) into a fresh r_final.
        //
        // The reducer operates on the *witness-only* column subset so
        // that its `r_final` eval claims are directly verifiable by the
        // PCS opening (which only commits witness columns). The verifier
        // re-derives public evaluations from the public trace and
        // splices them in when closing the mp_subclaim.
        let (opening_point, lookup_reducer) = if self.lookup_subclaims.is_empty() {
            (r_0, None)
        } else {
            use zinc_piop::multipoint_reducer::{MultiClaim, MultiPointReducer};
            use zinc_poly::mle::MultilinearExtensionWithConfig;

            let witness_full_indices =
                crate::witness_full_col_indices(&self.base.uair_signature);
            let witness_mles: Vec<DenseMultilinearExtension<F::Inner>> = witness_full_indices
                .iter()
                .map(|&idx| self.projected_trace_f[idx].clone())
                .collect();

            let witness_evals_at_r_0: Vec<F> = witness_mles
                .iter()
                .map(|mle| {
                    mle.clone()
                        .evaluate_with_config(&r_0, &self.field_cfg)
                        .expect("witness MLE eval at r_0")
                })
                .collect();

            let mut claims = Vec::with_capacity(1 + self.lookup_subclaims.len());
            claims.push(MultiClaim {
                point: r_0.clone(),
                evals: witness_evals_at_r_0.clone(),
            });

            let mut witness_evals_at_rho_row = Vec::with_capacity(self.lookup_subclaims.len());
            for sub in &self.lookup_subclaims {
                let evals: Vec<F> = witness_mles
                    .iter()
                    .map(|mle| {
                        mle.clone()
                            .evaluate_with_config(&sub.rho_row, &self.field_cfg)
                            .expect("witness MLE eval at ρ_row_g")
                    })
                    .collect();
                witness_evals_at_rho_row.push(evals.clone());
                claims.push(MultiClaim {
                    point: sub.rho_row.clone(),
                    evals,
                });
            }

            let (reducer_proof, reducer_sub) = MultiPointReducer::<F>::prove(
                &mut self.base.pcs_transcript.fs_transcript,
                &witness_mles,
                &claims,
                &self.field_cfg,
            )
            .expect("MultiPointReducer::prove on honest claims");

            (
                reducer_sub.r_final,
                Some(crate::LookupReducerProof {
                    witness_evals_at_r_0,
                    witness_evals_at_rho_row,
                    reducer_proof,
                }),
            )
        };

        Ok(ProverMultipointEvaled {
            base: self.base,
            field_cfg: self.field_cfg,
            projected_trace: self.projected_trace,
            ic_proof: self.ic_proof,
            cpr_proof: self.cpr_proof,
            combined_sumcheck: self.combined_sumcheck,
            lookup_proof: self.lookup_proof,
            mp_proof,
            r_0: opening_point,
            lookup_reducer,
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
            lookup_reducer: self.lookup_reducer,
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
            lookup_reducer: self.lookup_reducer,
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
            lookup_reducer: self.lookup_reducer,
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
            committed
                .step1_mle_first(project_scalar)?
                .step2_ideal_check()?
        } else {
            committed
                .step1_combined(project_scalar)?
                .step2_ideal_check()?
        };

        ideal_checked
            .step3_eval_projection()?
            .step4_sumcheck()?
            .step4b_lookup()?
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

/// Materialise one dense MLE from an [`AffineExpr`] over the projected
/// trace. At each hypercube index `b` the returned MLE stores
///
/// ```text
///   Σ_{(k, c) in expr.terms} c · projected_trace_f[k].evaluations[b] + constant
/// ```
///
/// lifted from `i64` coefficients to the target field via
/// [`i64_to_field`](crate::i64_to_field). Used by step 4b to feed the
/// logup-GKR argument with a synthesized witness column per lookup spec.
#[allow(clippy::arithmetic_side_effects)]
pub(crate) fn synthesize_affine_mle<F>(
    expr: &zinc_uair::AffineExpr,
    projected_trace_f: &[DenseMultilinearExtension<F::Inner>],
    cfg: &F::Config,
) -> DenseMultilinearExtension<F::Inner>
where
    F: PrimeField + InnerTransparentField + FromPrimitiveWithConfig,
    F::Inner: ConstIntSemiring + Send + Sync + Zero + Default,
{
    assert!(!projected_trace_f.is_empty(), "empty trace in synthesize_affine_mle");
    let num_vars = projected_trace_f[0].num_vars;
    let row_count = 1usize << num_vars;
    let zero_inner = F::zero_with_cfg(cfg).into_inner();

    // Precompute lifted coefficients once; the row loop is dominated by
    // per-row adds/muls in F so this avoids `i64_to_field` per cell.
    let f_terms: Vec<(usize, F)> = expr
        .terms
        .iter()
        .map(|(idx, c)| (*idx, crate::i64_to_field::<F>(*c, cfg)))
        .collect();
    let f_constant: F = crate::i64_to_field::<F>(expr.constant, cfg);

    let evals: Vec<F::Inner> = (0..row_count)
        .map(|b| {
            let mut acc = f_constant.clone();
            for (col_idx, coeff) in &f_terms {
                let v = F::new_unchecked_with_cfg(
                    projected_trace_f[*col_idx].evaluations[b].clone(),
                    cfg,
                );
                acc = acc + coeff.clone() * &v;
            }
            acc.into_inner()
        })
        .collect();

    DenseMultilinearExtension::from_evaluations_vec(num_vars, evals, zero_inner)
}

/// Build the canonical table MLE for a lookup group's table type.
///
/// The MLE has `num_vars` variables (so `2^num_vars` entries). The
/// table's effective width is `chunk_width.unwrap_or(width)`: when a
/// non-`None` `chunk_width` is present, the table shrinks to match
/// the chunk, so a 32-bit parent value can be range-checked in
/// 2^chunk_width-sized slices (UAIR author supplies the chunk-column
/// decomposition + per-chunk lookup specs; the framework just sizes
/// the table accordingly). The first `2^effective_width` entries
/// encode the table; any remaining entries are padded with 0.
///
/// * `Word { width, chunk_width }`: entry `j = j as F` (integer in
///   `[0, 2^effective_width)`).
/// * `BitPoly { width, chunk_width }`: entry
///   `j = ψ_a(bit-poly of j) = Σ_i bit_i(j)·a^i`, summed to
///   `effective_width` bits. Required for the protocol's ψ_a-projected
///   view of bit-poly columns to line up with the table.
#[allow(clippy::arithmetic_side_effects)]
pub(crate) fn build_table_mle<F>(
    table_type: &LookupTableType,
    num_vars: usize,
    cfg: &F::Config,
    projecting_element_f: &F,
) -> DenseMultilinearExtension<F::Inner>
where
    F: PrimeField
        + InnerTransparentField
        + FromPrimitiveWithConfig
        + for<'a> MulByScalar<&'a F>,
{
    let effective_width = match table_type {
        LookupTableType::Word { width, chunk_width }
        | LookupTableType::BitPoly { width, chunk_width } => chunk_width.unwrap_or(*width),
    };
    assert!(
        effective_width <= num_vars,
        "table width {effective_width} exceeds num_vars {num_vars}"
    );
    let row_count = 1usize << num_vars;
    let table_size = 1usize << effective_width;
    let zero_inner = F::zero_with_cfg(cfg).into_inner();

    let entry_at = |j: u64| -> F::Inner {
        match table_type {
            LookupTableType::Word { .. } => {
                F::from_with_cfg(j, cfg).into_inner()
            }
            LookupTableType::BitPoly { .. } => {
                // Horner: Σ_{i=0..effective_width} bit_i(j) · a^i,
                // evaluated bottom-up so each step adds a bit times
                // the running power of `a`.
                let mut acc = F::zero_with_cfg(cfg);
                let mut pow = F::from_with_cfg(1u64, cfg);
                for i in 0..effective_width {
                    if (j >> i) & 1 == 1 {
                        acc = acc + pow.clone();
                    }
                    if i + 1 < effective_width {
                        pow = pow * projecting_element_f;
                    }
                }
                acc.into_inner()
            }
        }
    };

    let mut evals = Vec::with_capacity(row_count);
    for j in 0..row_count {
        if j < table_size {
            evals.push(entry_at(j as u64));
        } else {
            evals.push(zero_inner.clone());
        }
    }
    DenseMultilinearExtension::from_evaluations_vec(num_vars, evals, zero_inner)
}
