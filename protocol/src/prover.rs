use super::*;
use crypto_primitives::{ConstIntSemiring, FromPrimitiveWithConfig, FromWithConfig};
use num_traits::Zero;
use std::{borrow::Cow, fmt::Debug};
use zinc_piop::{
    combined_poly_resolver::CombinedPolyResolver,
    ideal_check::IdealCheckProtocol,
    lookup::booleanity::{BooleanityChecker, BooleanityProof},
    multipoint_eval::{MultipointEval, Proof as MultipointEvalProof},
    projections::{
        ColumnMajorTrace, ProjectedScalars, ProjectedTrace, RowMajorTrace,
        evaluate_trace_to_column_mles, project_scalars, project_scalars_to_field,
        project_trace_coeffs_column_major, project_trace_coeffs_row_major,
    },
    sumcheck::multi_degree::MultiDegreeSumcheck,
};
use zinc_poly::{
    mle::DenseMultilinearExtension, univariate::dynamic::over_field::DynamicPolynomialF,
};
use zinc_transcript::traits::{ConstTranscribable, Transcript};
use zinc_uair::{
    Uair, UairSignature, UairTrace, constraint_counter::count_constraints,
    degree_counter::count_max_degree,
};
use zinc_utils::{
    add, cfg_iter, cfg_join, from_ref::FromRef, inner_transparent_field::InnerTransparentField,
    montgomery_inner_product::MontgomeryIntegerInnerProduct, mul_by_scalar::MulByScalar, powers,
    projectable_to_field::ProjectableToField,
};
use zip_plus::{
    pcs::structs::{ZipPlus, ZipPlusHint, ZipPlusParams, ZipTypes},
    pcs_transcript::PcsProverTranscript,
};

//
// Type-state structs
//

/// Initial prover state, before commitment: the UAIR signature, the
/// caller-provided original trace, and the folded witness trace.
#[derive(Clone, Debug)]
pub struct ProverFolded<
    'a,
    Zt: ZincTypes<D, FD>,
    U: Uair,
    F: PrimeField,
    const D: usize,
    const FD: usize,
> {
    uair_signature: UairSignature,
    original_trace: &'a UairTrace<'static, Zt::Int, Zt::Int, D, D>,
    folded_witness_trace: UairTrace<'a, Zt::Int, Zt::Int, FD, D>,

    _phantom: PhantomData<(&'a u8, U, F)>,
}

/// Persistent prover infrastructure carried across every subsequent
/// step: the Fiat-Shamir transcript, PCS parameters/hints/commitments,
/// and trace reference.
/// Obtained after step 1 via [`step1_commit`](ProverFolded::step1_commit).
#[derive(Clone, Debug)]
pub struct ProverCommitted<
    'a,
    Zt: ZincTypes<D, FD>,
    U: Uair,
    F: PrimeField,
    const D: usize,
    const FD: usize,
> {
    num_vars: usize,
    uair_signature: UairSignature,
    original_trace: &'a UairTrace<'static, Zt::Int, Zt::Int, D, D>,
    folded_witness_trace: UairTrace<'a, Zt::Int, Zt::Int, FD, D>,
    pcs_transcript: PcsProverTranscript,

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

/// After step 2 via [`step2_combined`](ProverCommitted::step2_combined)
/// (row-major / "combined" projection).
#[derive(Clone, Debug)]
pub struct ProverProjectedCombined<
    'a,
    Zt: ZincTypes<D, FD>,
    U: Uair,
    F: PrimeField,
    const D: usize,
    const FD: usize,
> {
    base: ProverCommitted<'a, Zt, U, F, D, FD>,
    field_cfg: F::Config,
    projected_trace: RowMajorTrace<F>,
    projected_scalars_fx: ProjectedScalars<U::Scalar, DynamicPolynomialF<F>>,
}

/// After step 2 via [`step2_mle_first`](ProverCommitted::step2_mle_first)
/// (column-major / MLE-first projection).
#[derive(Clone, Debug)]
pub struct ProverProjectedMleFirst<
    'a,
    Zt: ZincTypes<D, FD>,
    U: Uair,
    F: PrimeField,
    const D: usize,
    const FD: usize,
> {
    base: ProverCommitted<'a, Zt, U, F, D, FD>,
    field_cfg: F::Config,
    projected_trace: ColumnMajorTrace<F>,
    projected_scalars_fx: ProjectedScalars<U::Scalar, DynamicPolynomialF<F>>,
}

/// After step 3 (ideal check).
#[derive(Clone, Debug)]
pub struct ProverIdealChecked<
    'a,
    Zt: ZincTypes<D, FD>,
    U: Uair,
    F: PrimeField,
    const D: usize,
    const FD: usize,
> {
    base: ProverCommitted<'a, Zt, U, F, D, FD>,
    field_cfg: F::Config,
    projected_trace: ProjectedTrace<F>,
    projected_scalars_fx: ProjectedScalars<U::Scalar, DynamicPolynomialF<F>>,

    // New
    ic_proof: IdealCheckProof<F>,
    ic_eval_point: Vec<F>,
}

/// After step 4 (eval projection). `projected_scalars_fx` has been consumed.
#[derive(Clone, Debug)]
pub struct ProverEvalProjected<
    'a,
    Zt: ZincTypes<D, FD>,
    U: Uair,
    F: PrimeField,
    const D: usize,
    const FD: usize,
> {
    base: ProverCommitted<'a, Zt, U, F, D, FD>,
    field_cfg: F::Config,
    projected_trace: ProjectedTrace<F>,
    ic_proof: IdealCheckProof<F>,
    ic_eval_point: Vec<F>,

    // New
    projected_trace_f: Vec<DenseMultilinearExtension<F::Inner>>,
    projected_scalars_f: ProjectedScalars<U::Scalar, F>,
}

/// After step 5 (sumcheck).
#[allow(clippy::type_complexity)]
#[derive(Clone, Debug)]
pub struct ProverSumchecked<
    'a,
    Zt: ZincTypes<D, FD>,
    U: Uair,
    F: PrimeField,
    const D: usize,
    const FD: usize,
> {
    base: ProverCommitted<'a, Zt, U, F, D, FD>,
    field_cfg: F::Config,
    projected_trace: ProjectedTrace<F>,
    ic_proof: IdealCheckProof<F>,
    /// Trace MLEs at the original $\psi_a$ projecting element, as built
    /// by `evaluate_trace_to_column_mles` in a previous step.
    ///
    /// Carried forward so that the next step can prepend them when assembling
    /// the multipoint-eval inputs (and optionally append $\alpha'$-projected
    /// witness-bin MLEs as the Schwartz-Zippel bridge).
    projected_trace_f: Vec<DenseMultilinearExtension<F::Inner>>,

    // New
    cpr_proof: CombinedPolyResolverProof<F>,
    cpr_eval_point: Vec<F>,
    combined_sumcheck: MultiDegreeSumcheckProof<F>,
    lookup_proof: Option<BatchedLookupProof<F>>,
    booleanity_proof: Option<BooleanityProof<F>>,
    /// Fresh challenge sampled after `bit_slice_evals` were absorbed by
    /// booleanity's `finalize_prover`. Used by the next step to (a) build the
    /// extra $\alpha'$-projected witness-bin trace MLEs and (b) compute
    /// the per-column bridge scalars $c_j = \sum_i (\alpha')^i b_{j,i}$
    /// appended to multipoint-eval's `up_evals`.
    ///
    /// `None` iff there are no witness binary-poly columns (no booleanity
    /// argument).
    alpha_prime_f: Option<F>,
}

/// After step 6 (multipoint eval).
#[derive(Clone, Debug)]
pub struct ProverMultipointEvaled<
    'a,
    Zt: ZincTypes<D, FD>,
    U: Uair,
    F: PrimeField,
    const D: usize,
    const FD: usize,
> {
    base: ProverCommitted<'a, Zt, U, F, D, FD>,
    field_cfg: F::Config,
    projected_trace: ProjectedTrace<F>,
    ic_proof: IdealCheckProof<F>,
    cpr_proof: CombinedPolyResolverProof<F>,
    combined_sumcheck: MultiDegreeSumcheckProof<F>,
    lookup_proof: Option<BatchedLookupProof<F>>,
    booleanity_proof: Option<BooleanityProof<F>>,

    // New
    mp_proof: MultipointEvalProof<F>,
    r_0: Vec<F>,
}

/// After step 7 (lift-and-project).
#[derive(Clone, Debug)]
pub struct ProverLifted<
    'a,
    Zt: ZincTypes<D, FD>,
    U: Uair,
    F: PrimeField,
    const D: usize,
    const FD: usize,
> {
    base: ProverCommitted<'a, Zt, U, F, D, FD>,
    field_cfg: F::Config,
    ic_proof: IdealCheckProof<F>,
    cpr_proof: CombinedPolyResolverProof<F>,
    combined_sumcheck: MultiDegreeSumcheckProof<F>,
    lookup_proof: Option<BatchedLookupProof<F>>,
    booleanity_proof: Option<BooleanityProof<F>>,
    mp_proof: MultipointEvalProof<F>,
    r_0: Vec<F>,

    // New
    lifted_evals: Vec<DynamicPolynomialF<F>>,
}

/// After step 8 (PCS open). No new fields are added here, but the PCS
/// transcript has been updated with the opening proof.
/// Ready for generating the final proof object in
/// [`finish`](ProverPcsOpened::finish).
#[derive(Clone, Debug)]
pub struct ProverPcsOpened<
    'a,
    Zt: ZincTypes<D, FD>,
    U: Uair,
    F: PrimeField,
    const D: usize,
    const FD: usize,
> {
    base: ProverCommitted<'a, Zt, U, F, D, FD>,
    ic_proof: IdealCheckProof<F>,
    cpr_proof: CombinedPolyResolverProof<F>,
    combined_sumcheck: MultiDegreeSumcheckProof<F>,
    lookup_proof: Option<BatchedLookupProof<F>>,
    booleanity_proof: Option<BooleanityProof<F>>,
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
        impl<'a, Zt, U, F, const D: usize, const FD: usize> $type_name<'a, Zt, U, F, D, FD>
        where
            Zt: ZincTypes<D, FD>,
            Zt::Int: ProjectableToField<F>,
            <Zt::ArbitraryZt as ZipTypes>::Eval: ProjectableToField<F>,
            U: Uair + 'static,
            F: InnerTransparentField
                + FromPrimitiveWithConfig
                + for<'b> FromWithConfig<&'b Zt::Int>
                + for<'b> FromWithConfig<&'b Zt::Chal>
                + for<'b> MulByScalar<&'b F>
                + FromRef<F>
                + MontgomeryIntegerInnerProduct<Zt::CombR>
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

impl<Zt, U, F, const D: usize, const FD: usize> ZincPlusPiop<Zt, U, F, D, FD>
where
    Zt: ZincTypes<D, FD>,
    U: Uair,
    F: PrimeField,
    F::Inner: ConstTranscribable,
{
    /// Step 0: Folding the trace.
    #[allow(clippy::type_complexity)]
    pub fn step0_fold<'a>(
        trace: &'a UairTrace<'static, Zt::Int, Zt::Int, D, D>,
    ) -> Result<ProverFolded<'a, Zt, U, F, D, FD>, ProtocolError<F, U::Ideal>> {
        let uair_signature = U::signature();
        let witness_trace = trace.witness(&uair_signature);

        let folded_bin_witness_trace = cfg_iter!(witness_trace.binary_poly)
            .map(Zt::BinaryFold::fold_trace_mle)
            .collect();

        let folded_witness_trace = UairTrace {
            binary_poly: Cow::Owned(folded_bin_witness_trace),
            arbitrary_poly: witness_trace.arbitrary_poly.clone(),
            int: witness_trace.int.clone(),
        };

        Ok(ProverFolded {
            uair_signature,
            original_trace: trace,
            folded_witness_trace,
            _phantom: PhantomData,
        })
    }
}

impl_with_type_bounds!(ProverFolded
{
    /// Step 1: Commitment.
    /// Commit *witness* columns via Zip+ PCS, absorb roots and public
    /// data into the Fiat-Shamir transcript.
    #[allow(clippy::type_complexity)]
    pub fn step1_commit(
        self,
        (pp_bin, pp_arb, pp_int): &'a (
            ZipPlusParams<Zt::BinaryZt, Zt::BinaryLc>,
            ZipPlusParams<Zt::ArbitraryZt, Zt::ArbitraryLc>,
            ZipPlusParams<Zt::IntZt, Zt::IntLc>,
        ),
        num_vars: usize,
    ) -> Result<ProverCommitted<'a, Zt, U, F, D, FD>, ProtocolError<F, U::Ideal>> {
        let sig = &self.uair_signature;
        let public_trace = self.original_trace.public(sig);

        let (res_bin, (res_arb, res_int)) = cfg_join!(
            commit_optionally(pp_bin, &self.folded_witness_trace.binary_poly),
            commit_optionally(pp_arb, &self.folded_witness_trace.arbitrary_poly),
            commit_optionally(pp_int, &self.folded_witness_trace.int),
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

        Ok(ProverCommitted {
            num_vars,
            uair_signature: self.uair_signature,
            original_trace: self.original_trace,
            folded_witness_trace: self.folded_witness_trace,
            pcs_transcript,
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
});

impl_with_type_bounds!(ProverCommitted
{
    #[allow(clippy::type_complexity)]
    fn project_common<S: Fn(&U::Scalar, &F::Config) -> DynamicPolynomialF<F>>(
        &mut self,
        project_scalar: S,
    ) -> Result<(F::Config, ProjectedScalars<U::Scalar, DynamicPolynomialF<F>>), ProtocolError<F, U::Ideal>>
    {
        let field_cfg = self
            .pcs_transcript
            .fs_transcript
            .get_random_field_cfg::<F, Zt::Fmod, Zt::PrimeTest>();

        let projected_scalars_fx = project_scalars::<F, U>(|s| project_scalar(s, &field_cfg));
        Ok((field_cfg, projected_scalars_fx))
    }

    /// Step 2 (combined / row-major): Prime projection
    /// (`\phi_q`: `Z[X] -> F_q[X]`). Samples a random prime, projects the
    /// full trace and scalars using the row-major layout.
    /// Works for both linear and non-linear constraints.
    pub fn step2_combined<S: Fn(&U::Scalar, &F::Config) -> DynamicPolynomialF<F>>(
        mut self,
        project_scalar: S,
    ) -> Result<ProverProjectedCombined<'a, Zt, U, F, D, FD>, ProtocolError<F, U::Ideal>> {
        let (field_cfg, projected_scalars_fx) = self.project_common(project_scalar)?;

        let projected_trace = project_trace_coeffs_row_major(self.original_trace, &field_cfg);
        Ok(ProverProjectedCombined {
            base: self,
            field_cfg,
            projected_trace,
            projected_scalars_fx,
        })
    }

    /// Step 2 (MLE-first / column-major): Prime projection
    /// (`\phi_q`: `Z[X] -> F_q[X]`). Samples a random prime, projects the
    /// full trace and scalars using the column-major layout.
    pub fn step2_mle_first<S: Fn(&U::Scalar, &F::Config) -> DynamicPolynomialF<F>>(
        mut self,
        project_scalar: S,
    ) -> Result<ProverProjectedMleFirst<'a, Zt, U, F, D, FD>, ProtocolError<F, U::Ideal>> {
        let (field_cfg, projected_scalars_fx) = self.project_common(project_scalar)?;

        let projected_trace = project_trace_coeffs_column_major(self.original_trace, &field_cfg);
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
    /// Step 3 (combined): Ideal check via `prove_combined` on the row-major
    /// trace. Works for both linear and non-linear constraints.
    pub fn step3_ideal_check(
        mut self,
    ) -> Result<ProverIdealChecked<'a, Zt, U, F, D, FD>, ProtocolError<F, U::Ideal>> {
        let num_constraints = count_constraints::<U>();

        let (ic_proof, ic_prover_state) = U::prove_combined::<_, D>(
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
    /// Step 3 (MLE-first): Ideal check via `prove_mle_first` on the
    /// column-major trace. Works for any UAIR: linear non-zero-ideal
    /// constraints go through the column-major MLE-first path, non-linear
    /// non-zero-ideal constraints fall back to the row-major path (with an
    /// internal transpose), and zero-ideal constraints are short-circuited
    /// to zero.
    pub fn step3_ideal_check(
        mut self,
    ) -> Result<ProverIdealChecked<'a, Zt, U, F, D, FD>, ProtocolError<F, U::Ideal>> {
        let num_constraints = count_constraints::<U>();

        let (ic_proof, ic_prover_state) = U::prove_mle_first::<_, D>(
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
    /// Step 4: Evaluation projection (`\psi_a`: `F_q[X] -> F_q`). Samples
    /// `a in F_q`, evaluates polynomials at `X = a`.
    pub fn step4_eval_projection(
        mut self,
    ) -> Result<ProverEvalProjected<'a, Zt, U, F, D, FD>, ProtocolError<F, U::Ideal>> {
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
    /// Step 5: Combined CPR + Booleanity + Lookup multi-degree sumcheck over F_q.
    /// Batches the CPR constraint claim (degree `max_deg+2`), the booleanity
    /// argument (degree 3), and lookup groups (one per table type) into a
    /// single sumcheck sharing one evaluation point `r*`. Produces
    /// `up_evals`/`down_evals` (CPR), `bit_slice_evals` (booleanity), and
    /// lookup auxiliary witnesses at `r*`.
    ///
    /// After booleanity's `finalize_prover` absorbs `bit_slice_evals` into
    /// the transcript, this step squeezes a fresh challenge $\alpha'$ and
    /// stores it on `ProverSumchecked`. The actual Schwartz-Zippel bridge
    /// is installed in `step6_multipoint_eval`, which appends one extra
    /// $\alpha'$-projected witness-bin column MLE (and matching up_eval
    /// $c_j = \sum_i b_{j,i} (\alpha')^i$) per witness binary-poly column
    /// to the multipoint-eval inputs. Shifts continue to reference the
    /// original $\psi_a$-projected witness-bin slot, so `down_evals` are
    /// untouched: shifted booleanity is inherited from un-shifted
    /// booleanity (same committed column).
    ///
    /// The PCS chain (`step6_multipoint_eval` + lifted-evals + Zip+ open)
    /// closes the bridge: at the random sumcheck output $r_0$,
    /// $\overline{u_j}(\alpha') = \widetilde{g_j}(r_0)$ pins the appended
    /// column's multilinear extension to the true $\alpha'$-projection
    /// $g_j$ of the committed $u_j$, replacing the previous
    /// underconstrained $\psi_a$ linear pin-down (sound only for $D=1$).
    pub fn step5_sumcheck(
        mut self,
    ) -> Result<ProverSumchecked<'a, Zt, U, F, D, FD>, ProtocolError<F, U::Ideal>> {
        let num_constraints = count_constraints::<U>();
        let max_degree = count_max_degree::<U>();

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

        let mut groups = vec![cpr_group];

        // Booleanity: prepare optional group over witness binary-poly cols.
        let sig = &self.base.uair_signature;
        let num_pub_bin = sig.public_cols().num_binary_poly_cols();
        let num_total_bin = sig.total_cols().num_binary_poly_cols();
        let trace_wit_bin_poly = &self.base.original_trace.binary_poly[num_pub_bin..num_total_bin];

        let bool_ancillary = if !trace_wit_bin_poly.is_empty() {
            let (bool_group, anc) = BooleanityChecker::prepare_sumcheck_group::<D>(
                &mut self.base.pcs_transcript.fs_transcript,
                trace_wit_bin_poly,
                self.base.num_vars,
                &self.field_cfg,
            )
            .map_err(ProtocolError::Booleanity)?;
            groups.push(bool_group);
            Some(anc)
        } else {
            None
        };

        // TODO: for each LookupGroup from group_lookup_specs(lookup_specs):
        //   - call prepare_batched_lookup_group(transcript, instance, &field_cfg)
        //   - push triple into groups, collect pending proofs + metas

        // Multi-degree sumcheck over CPR (+ booleanity + lookup groups).
        let (combined_sumcheck, md_states) = MultiDegreeSumcheck::prove_as_subprotocol(
            &mut self.base.pcs_transcript.fs_transcript,
            groups,
            self.base.num_vars,
            &self.field_cfg,
        );

        let mut md_iter = md_states.into_iter();

        let (cpr_proof, cpr_prover_state) = CombinedPolyResolver::finalize_prover(
            &mut self.base.pcs_transcript.fs_transcript,
            md_iter.next().expect("CPR group always present"),
            cpr_ancillary,
            &self.field_cfg,
        )?;

        let booleanity_proof = if let Some(anc) = bool_ancillary {
            let state = md_iter.next().expect("booleanity group present");
            Some(
                BooleanityChecker::finalize_prover(
                    &mut self.base.pcs_transcript.fs_transcript,
                    state,
                    anc,
                    &self.field_cfg,
                )
                .map_err(ProtocolError::Booleanity)?,
            )
        } else {
            None
        };

        // TODO: build BatchedLookupProof from collected lookup_proofs + lookup_metas
        let lookup_proof = None;

        // Booleanity -> multipoint-eval `alpha_prime` bridge: squeeze
        // \alpha' after `bit_slice_evals` were absorbed by `finalize_prover`.
        // The challenge is consumed in `step6_multipoint_eval` to build the
        // extra appended trace MLEs / up_evals (one per witness binary-poly
        // column). `cpr_proof.up_evals` / `cpr_proof.down_evals` and
        // `projected_trace_f` are carried through unmodified.
        let alpha_prime_f: Option<F> = booleanity_proof.as_ref().map(|_| {
            self.base
                .pcs_transcript
                .fs_transcript
                .get_field_challenge(&self.field_cfg)
        });

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
            booleanity_proof,
            alpha_prime_f,
        })
    }
});

impl_with_type_bounds!(ProverSumchecked
{
    /// Step 6: Multi-point evaluation sumcheck. Combines `up_evals` and
    /// `down_evals` at `r*` into a single evaluation point `r_0`.
    /// Only the sumcheck proof is sent; scalar evaluations at `r_0` are derived from the
    /// polynomial-valued `lifted_evals` in Step 7.
    ///
    /// When the booleanity argument ran (witness binary-poly columns
    /// present), the multipoint-eval inputs are *extended* with one extra
    /// $\alpha'$-projected column MLE and one extra scalar up_eval
    /// $c_j = \sum_i b_{j,i}\,(\alpha')^{i}$ per witness binary-poly column.
    /// The appended columns are placed at indices
    /// `[num_total_cols, num_total_cols + num_wit_bin)`; no `ShiftSpec`
    /// references those indices, so `down_evals`/`shifts` are untouched and
    /// shifted booleanity is inherited from the un-shifted column (which
    /// continues to live at its original $\psi_a$-projected slot).
    /// See the module-level `BooleanityChecker` docs for the soundness
    /// argument (Schwartz-Zippel at $\alpha'$ + MP/PCS chain at $r_0$).
    #[allow(clippy::arithmetic_side_effects)]
    pub fn step6_multipoint_eval(
        mut self,
    ) -> Result<ProverMultipointEvaled<'a, Zt, U, F, D, FD>, ProtocolError<F, U::Ideal>> {
        let (trace_mles, up_evals) = if let Some(alpha_prime) = &self.alpha_prime_f {
            let sig = &self.base.uair_signature;
            let num_pub_bin = sig.public_cols().num_binary_poly_cols();
            let num_total_bin = sig.total_cols().num_binary_poly_cols();
            let num_wit_bin = num_total_bin.saturating_sub(num_pub_bin);

            // Project the witness binary-poly columns at \alpha' directly from the
            // committed BinaryPoly<D> data:
            // More efficient that generic `evaluate_trace_to_column_mles` path.
            let one = F::one_with_cfg(&self.field_cfg);
            let alpha_powers: Vec<F> = powers(alpha_prime.clone(), one, D);
            let bin_cols = &self.base.original_trace.binary_poly[num_pub_bin..num_total_bin];
            let extra_trace_mles: Vec<DenseMultilinearExtension<F::Inner>> = cfg_iter!(bin_cols)
                .map(|col| project_binary_col_at_field::<F, D>(col, &alpha_powers, &self.field_cfg))
                .collect();
            debug_assert_eq!(extra_trace_mles.len(), num_wit_bin);

            let bp = self
                .booleanity_proof
                .as_ref()
                .expect("booleanity_proof present iff alpha_prime_f is Some");
            let extra_up_evals = alpha_prime_bridge_up_evals::<F, D>(
                &bp.bit_slice_evals,
                num_wit_bin,
                alpha_prime,
                &self.field_cfg,
            );

            let mut trace_mles = self.projected_trace_f;
            trace_mles.extend(extra_trace_mles);
            let mut up_evals = self.cpr_proof.up_evals.clone();
            up_evals.extend(extra_up_evals);
            (trace_mles, up_evals)
        } else {
            (self.projected_trace_f, self.cpr_proof.up_evals.clone())
        };

        let (mp_proof, mp_prover_state) = MultipointEval::prove_as_subprotocol(
            &mut self.base.pcs_transcript.fs_transcript,
            &trace_mles,
            &self.cpr_eval_point,
            &up_evals,
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
            booleanity_proof: self.booleanity_proof,
            mp_proof,
            r_0: mp_prover_state.eval_point,
        })
    }
});

impl_with_type_bounds!(ProverMultipointEvaled
{
    /// Step 7: Lift-and-project. Computes per-column polynomial MLE
    /// evaluations at `r_0` in `F_q[X]` and absorbs them into the transcript.
    pub fn step7_lift_and_project(
        mut self,
    ) -> Result<ProverLifted<'a, Zt, U, F, D, FD>, ProtocolError<F, U::Ideal>> {
        // Compute per-column polynomial MLE evaluations at r_0 in F_q[X]
        // (after \phi_q but before \psi_a). The verifier derives the scalar
        // open_evals via \psi_a for the sumcheck consistency check, and
        // supplies these to the Zip+ PCS for alpha-projection.
        let lifted_evals = compute_lifted_evals(
            &self.r_0,
            &self.base.original_trace.binary_poly,
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
            booleanity_proof: self.booleanity_proof,
            mp_proof: self.mp_proof,
            r_0: self.r_0,
            lifted_evals,
        })
    }
});

impl_with_type_bounds!(ProverLifted
{
    /// Step 8: PCS open at `r_0` (witness columns only).
    pub fn step8_pcs_open<const CHECK_FOR_OVERFLOW: bool>(
        mut self,
    ) -> Result<ProverPcsOpened<'a, Zt, U, F, D, FD>, ProtocolError<F, U::Ideal>> {
        let witness_trace = &self.base.folded_witness_trace;

        // Folded witness columns are proved using the extended evaluation point
        // `r_0_ext = r_0 || folding_challenges`.
        let mut r_0_ext = self.r_0.clone();
        let num_folding_challenges = Zt::BinaryFold::FOLDING_FACTOR.ilog2();
        (0..num_folding_challenges).for_each(|_| {
            let g_chal: Zt::Chal = self.base.pcs_transcript.fs_transcript.get_challenge();
            let gamma = F::from_with_cfg(&g_chal, &self.field_cfg);
            r_0_ext.push(gamma);
        });

        if let Some(hint_bin) = &self.base.hint_bin {
            let _ = ZipPlus::<Zt::BinaryZt, Zt::BinaryLc>::prove_f::<_, CHECK_FOR_OVERFLOW>(
                &mut self.base.pcs_transcript,
                self.base.pp_bin,
                &witness_trace.binary_poly,
                &r_0_ext,
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
            booleanity_proof: self.booleanity_proof,
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
            cpr_proof: self.cpr_proof,
            combined_sumcheck: self.combined_sumcheck,
            multipoint_eval: self.mp_proof,
            zip: zip_proof,
            witness_lifted_evals,
            lookup_proof: self.lookup_proof,
            booleanity_proof: self.booleanity_proof,
        })
    }
});

//
// prove() wrapper
//

impl<Zt, U, F, const D: usize, const FD: usize> ZincPlusPiop<Zt, U, F, D, FD>
where
    Zt: ZincTypes<D, FD>,
    Zt::Int: ProjectableToField<F>,
    <Zt::ArbitraryZt as ZipTypes>::Eval: ProjectableToField<F>,
    F: InnerTransparentField
        + FromPrimitiveWithConfig
        + for<'a> FromWithConfig<&'a Zt::Int>
        + for<'a> FromWithConfig<&'a Zt::Chal>
        + for<'a> FromWithConfig<&'a Zt::Pt>
        + for<'a> MulByScalar<&'a F>
        + FromRef<F>
        + MontgomeryIntegerInnerProduct<Zt::CombR>
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
    /// For per-step control, start with [`Self::step0_fold`] and chain the
    /// individual `stepN_*` methods.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub fn prove<const MLE_FIRST: bool, const CHECK_FOR_OVERFLOW: bool>(
        pp: &(
            ZipPlusParams<Zt::BinaryZt, Zt::BinaryLc>,
            ZipPlusParams<Zt::ArbitraryZt, Zt::ArbitraryLc>,
            ZipPlusParams<Zt::IntZt, Zt::IntLc>,
        ),
        trace: &UairTrace<'static, Zt::Int, Zt::Int, D, D>,
        num_vars: usize,
        project_scalar: impl Fn(&U::Scalar, &F::Config) -> DynamicPolynomialF<F>,
    ) -> Result<Proof<F>, ProtocolError<F, U::Ideal>> {
        let committed = Self::step0_fold(trace)?.step1_commit(pp, num_vars)?;

        let ideal_checked = if MLE_FIRST {
            committed
                .step2_mle_first(project_scalar)?
                .step3_ideal_check()?
        } else {
            committed
                .step2_combined(project_scalar)?
                .step3_ideal_check()?
        };

        ideal_checked
            .step4_eval_projection()?
            .step5_sumcheck()?
            .step6_multipoint_eval()?
            .step7_lift_and_project()?
            .step8_pcs_open::<CHECK_FOR_OVERFLOW>()?
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

/// Project a single witness binary-poly column at a field element by
/// evaluating each bit-packed cell $\sum_i \text{bit}_i \cdot X^i$ at
/// $X = \alpha$.
///
/// Used to build the appended $\alpha'$-projected witness-binary-poly MLEs that
/// participate in `MultipointEval` as the Schwartz-Zippel bridge from
/// booleanity into the PCS chain.
#[allow(clippy::arithmetic_side_effects)]
fn project_binary_col_at_field<F, const D: usize>(
    col: &DenseMultilinearExtension<BinaryPoly<D>>,
    alpha_powers: &[F],
    field_cfg: &F::Config,
) -> DenseMultilinearExtension<F::Inner>
where
    F: PrimeField,
    F::Inner: Clone + Send + Sync,
{
    debug_assert_eq!(alpha_powers.len(), D);
    let zero = F::zero_with_cfg(field_cfg);

    // Sequential row loop: per-row work is at most `D` conditional adds.
    // The caller parallelizes the outer loop over witness binary-poly columns.
    let evaluations: Vec<F::Inner> = col
        .evaluations
        .iter()
        .map(|entry| {
            let mut acc = zero.clone();
            for (i, bit) in entry.iter().enumerate() {
                if bit.into_inner() {
                    acc += &alpha_powers[i];
                }
            }
            acc.into_inner()
        })
        .collect();

    DenseMultilinearExtension {
        num_vars: col.num_vars,
        evaluations,
    }
}
