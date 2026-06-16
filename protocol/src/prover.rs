use super::*;
use crypto_primitives::{ConstIntSemiring, FixedSemiring, FromPrimitiveWithConfig, FromWithConfig};
use std::{borrow::Cow, fmt::Debug};
use zinc_piop::{
    combined_poly_resolver::CombinedPolyResolver,
    ideal_check::IdealCheckProtocol,
    lookup::booleanity::{BooleanityChecker, BooleanityProof},
    multipoint_eval::{MultipointEval, MultipointEvalBranchInputs, Proof as MultipointEvalProof},
    projections::{
        ColumnMajorTrace, ProjectedScalars, ProjectedTrace, RowMajorTrace,
        evaluate_trace_to_column_mles, project_scalars, project_scalars_to_field,
        project_trace_coeffs_column_major, project_trace_coeffs_row_major,
    },
    sumcheck::multi_degree::{MultiDegreeSumcheck, MultiDegreeSumcheckGroup},
};
use zinc_poly::{
    mle::DenseMultilinearExtension,
    univariate::dynamic::{
        over_field::DynamicPolynomialF, over_fixed_semiring::DynamicPolynomialFS,
    },
};
use zinc_transcript::traits::{ConstTranscribable, Transcript};
use zinc_uair::{
    Uair, UairSignature, UairTrace, constraint_counter::count_constraints,
    degree_counter::count_max_degree,
};
use zinc_utils::{
    add, cfg_iter, cfg_join, from_ref::FromRef, inner_transparent_field::InnerTransparentField,
    mul_by_scalar::MulByScalar, powers, projectable_to_field::ProjectableToField,
};
use zip_plus::{
    pcs::structs::{ZipPlus, ZipPlusHint, ZipPlusParams, ZipTypes},
    pcs_transcript::PcsProverTranscript,
};

//
// Per-prime F_q[X] branch helpers
//

// FIXME

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
    /// Field configs for all constraint branches, starting with randomly
    /// sampled `field_cfg` (for $Q[X]$ constraints, always present) followed by
    /// config for each $q_i$ for $F_{q_i}[X]$ constraints.
    all_field_cfgs: Vec<F::Config>,
    /// Index of $q^* := \min_i q_i$ in `all_field_cfgs`.
    q_star_idx: usize,
    /// Per-prime $\mathbb{F}_{q_i}[X]$ projections (one entry per prime in
    /// `UairSignature::primes()`), pre-staged in step 2 so step 3's per-prime
    /// ideal check can read them. Empty for legacy UAIRs.
    ///
    /// TODO(fq-perf): the row-major projection is duplicated -- once for the
    /// Q[X] branch and once per prime here. The `fq-unify` optimization
    /// would emit all projections in one trace sweep.
    fq_staging: Vec<FqProjStaging<U, F>>,
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
    /// Field configs for all constraint branches, starting with randomly
    /// sampled `field_cfg` (for $Q[X]$ constraints, always present) followed by
    /// config for each $q_i$ for $F_{q_i}[X]$ constraints.
    all_field_cfgs: Vec<F::Config>,
    /// Index of $q^* := \min_i q_i$ in `all_field_cfgs`.
    q_star_idx: usize,
    /// Per-prime $\mathbb{F}_{q_i}[X]$ projections, column-major layout
    /// counterpart of [`ProverProjectedCombined::fq_staging`].
    fq_staging: Vec<FqProjStaging<U, F>>,
}

/// Per-prime $\phi_{q_i}$ projection of the integer trace and UAIR scalars,
/// pre-built at step 2 for step 3's per-prime ideal check (and threaded
/// forward through step 4 into the per-prime CPR / sumcheck / MP-eval
/// chain under `fq-unify`). The trace layout (row- vs column-major)
/// matches the variant chosen at step 2 and is carried inside
/// [`ProjectedTrace`].
///
/// Field config is stored separately on the parent state (see
/// `all_field_cfgs`) and is not needed here.
#[derive(Clone, Debug)]
pub struct FqProjStaging<U: Uair, F: PrimeField> {
    projected_trace: ProjectedTrace<F>,
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
    /// Field configs for all constraint branches, starting with randomly
    /// sampled `field_cfg` (for $Q[X]$ constraints, always present) followed by
    /// config for each $q_i$ for $F_{q_i}[X]$ constraints.
    all_field_cfgs: Vec<F::Config>,
    /// Index of $q^* := \min_i q_i$ in `all_field_cfgs`.
    q_star_idx: usize,
    projected_trace: ProjectedTrace<F>,
    projected_scalars_fx: ProjectedScalars<U::Scalar, DynamicPolynomialF<F>>,
    /// Per-prime $\phi_{q_i}$ projections from step 2, threaded forward so
    /// step 4 can build per-prime $\psi$-projected trace/scalars and step 5
    /// can drive the per-prime CPR sumcheck branches. Empty for UAIRs
    /// with no declared fq primes.
    #[allow(dead_code)] // Consumed by Phase F.2.b in step4_eval_projection.
    fq_staging: Vec<FqProjStaging<U, F>>,

    // New
    ic_proof: IdealCheckProof<F>,
    /// Per-branch IC evaluation points (full `Vec<Vec<F>>` of size
    /// `n + 1`, sampled once at step 3 via `sample_shared_field_challenges`
    /// and lifted into each branch's field). `[0]` is consumed by the
    /// Q[X] CPR in step 5; `[i + 1]` will drive the per-prime CPR.
    ic_eval_points: Vec<Vec<F>>,
    /// Per-prime $\mathbb{F}_{q_i}[X]$ ideal-check proofs, one per declared
    /// prime in `base.uair_signature.primes()`, in order.
    ///
    /// TODO(fq-soundness): each entry is currently only a standalone
    /// ideal-membership check on the per-prime combined polynomial
    /// $e_{i,t}$; the downstream per-prime CPR + sumcheck +
    /// multipoint-eval + PCS-open chain that ties $e_{i,t}$ back to the
    /// committed trace via $\phi_{q_i}(\hat f_0)$ is **not** present yet.
    /// See [`Proof::ideal_checks_fq`] for the soundness gap and the planned
    /// unification optimization (one shared $\mathbf r \in [0, q^*)^\mu$).
    ic_proof_fq: Vec<IdealCheckProof<F>>,
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
    /// Per-branch field configs, kept for the per-prime CPR/sumcheck/MP
    /// chain in later phases. `[0]` = $Q[X]$ branch.
    #[allow(dead_code)] // Consumed by Phase F.2.b in step5_sumcheck.
    all_field_cfgs: Vec<F::Config>,
    /// Index of $q^* := \min_i q_i$ in `all_field_cfgs`.
    #[allow(dead_code)] // Consumed by Phase F.2.b in step5_sumcheck.
    q_star_idx: usize,
    /// Per-branch $\psi$-projecting elements: integer sampled mod $q^*$
    /// and projected onto each of `all_field_cfgs`. Length `n + 1`.
    /// `[0]` was consumed by step 4 to build `projected_trace_f`;
    /// `[i + 1]` was consumed to build `projected_trace_f_fq[i]`. Carried
    /// forward for later phases (H/I) that need the integer endpoint.
    #[allow(dead_code)] // Consumed by Phase H / I.
    projecting_elements: Vec<F>,
    projected_trace: ProjectedTrace<F>,
    ic_proof: IdealCheckProof<F>,
    /// Per-branch IC evaluation points (full $\text{n+1} \times \mu$
    /// matrix). `[0]` feeds the Q[X] CPR; `[i + 1]` feeds the per-prime
    /// CPRs (Phase F.2.b).
    ic_eval_points: Vec<Vec<F>>,
    ic_proof_fq: Vec<IdealCheckProof<F>>,

    // New
    projected_trace_f: Vec<DenseMultilinearExtension<F::Inner>>,
    projected_scalars_f: ProjectedScalars<U::Scalar, F>,
    /// Per-prime $\psi$-projected trace MLEs (one entry per declared prime
    /// in `UairSignature::primes()`). Built in step 4 from each
    /// `fq_staging[i].projected_trace` using `projecting_elements[i + 1]`.
    /// Consumed by per-prime CPR `prepare_sumcheck_group` in step 5 (Phase
    /// F.2.b). Empty for UAIRs with no declared fq primes.
    #[allow(dead_code)] // Consumed by Phase F.2.b.
    projected_trace_f_fq: Vec<Vec<DenseMultilinearExtension<F::Inner>>>,
    /// Per-prime $\psi$-projected scalars (one entry per declared prime).
    /// Built in step 4 from each `fq_staging[i].projected_scalars_fx`
    /// using `projecting_elements[i + 1]`. Consumed in step 5 by the
    /// per-prime CPR. Empty for UAIRs with no declared fq primes.
    #[allow(dead_code)] // Consumed by Phase F.2.b.
    projected_scalars_f_fq: Vec<ProjectedScalars<U::Scalar, F>>,
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
    /// Per-branch field configs (carried for later `fq-unify` phases).
    #[allow(dead_code)] // Used by later `fq-unify` phases.
    all_field_cfgs: Vec<F::Config>,
    /// Index of $q^*$ in `all_field_cfgs` (carried for later phases).
    #[allow(dead_code)] // Used by later `fq-unify` phases.
    q_star_idx: usize,
    /// Per-branch CPR batching challenges $\alpha$ (length `n + 1`).
    /// `[0]` was consumed by the Q[X] CPR in step 5; `[i + 1]` was
    /// consumed by the per-prime CPRs in the same step.
    #[allow(dead_code)] // Used by later `fq-unify` phases.
    folding_challenges: Vec<F>,
    projected_trace: ProjectedTrace<F>,
    ic_proof: IdealCheckProof<F>,
    ic_proof_fq: Vec<IdealCheckProof<F>>,
    /// Trace MLEs at the original $\psi_a$ projecting element, as built
    /// by `evaluate_trace_to_column_mles` in a previous step.
    ///
    /// Carried forward so that the next step can prepend them when assembling
    /// the multipoint-eval inputs (and optionally append $\alpha'$-projected
    /// witness-bin MLEs as the Schwartz-Zippel bridge).
    projected_trace_f: Vec<DenseMultilinearExtension<F::Inner>>,
    /// Per-prime $\psi$-projected trace MLEs (one entry per declared
    /// prime). Threaded forward from step 4 by `step5_sumcheck` so that
    /// Phase G's lockstep multipoint-eval can build per-prime MP branches.
    /// Empty for UAIRs with no declared fq primes.
    projected_trace_f_fq: Vec<Vec<DenseMultilinearExtension<F::Inner>>>,

    // New
    cpr_proof: CombinedPolyResolverProof<F>,
    cpr_eval_point: Vec<F>,
    combined_sumcheck: MultiDegreeSumcheckProof<F>,
    /// Per-prime CPR proofs (one per declared prime in
    /// `UairSignature::primes()`), produced by F.2.b's per-prime CPR
    /// finalize. Empty for UAIRs with no declared fq primes.
    cpr_proofs_fq: Vec<CombinedPolyResolverProof<F>>,
    /// Per-prime CPR sumcheck endpoints `r^*_i`, lifted into each branch's
    /// field. Empty for UAIRs with no declared fq primes. Consumed by
    /// Phase G's lockstep multipoint-eval.
    cpr_eval_points_fq: Vec<Vec<F>>,
    /// Per-prime multi-degree sumcheck proofs (one per declared prime).
    /// Empty for UAIRs with no declared fq primes.
    combined_sumchecks_fq: Vec<MultiDegreeSumcheckProof<F>>,
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
    /// Per-branch field configs (carried for later phases that need the
    /// per-prime cfgs, e.g. Phase H/I).
    #[allow(dead_code)] // Used by later `fq-unify` phases.
    all_field_cfgs: Vec<F::Config>,
    projected_trace: ProjectedTrace<F>,
    ic_proof: IdealCheckProof<F>,
    ic_proof_fq: Vec<IdealCheckProof<F>>,
    cpr_proof: CombinedPolyResolverProof<F>,
    combined_sumcheck: MultiDegreeSumcheckProof<F>,
    /// Per-prime CPR proofs threaded forward from `ProverSumchecked`.
    cpr_proofs_fq: Vec<CombinedPolyResolverProof<F>>,
    /// Per-prime multi-degree sumcheck proofs threaded forward.
    combined_sumchecks_fq: Vec<MultiDegreeSumcheckProof<F>>,
    lookup_proof: Option<BatchedLookupProof<F>>,
    booleanity_proof: Option<BooleanityProof<F>>,

    // New
    mp_proof: MultipointEvalProof<F>,
    r_0: Vec<F>,
    /// Per-prime multipoint-eval proofs (one per declared prime in
    /// `UairSignature::primes()`), produced by Phase G's lockstep
    /// multipoint-eval. Empty for UAIRs with no declared fq primes.
    mp_proofs_fq: Vec<MultipointEvalProof<F>>,
    /// Per-prime sumcheck output points $r_0$ (one per declared prime,
    /// lifted into each branch's field — the underlying integer is shared
    /// with the Q-branch `r_0` thanks to the lockstep sumcheck). Empty for
    /// UAIRs with no declared fq primes. Carried forward for Phase H/I.
    #[allow(dead_code)] // Consumed by Phase H / I.
    r_0_fq: Vec<Vec<F>>,
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
> where
    F::Integer: FixedSemiring,
{
    base: ProverCommitted<'a, Zt, U, F, D, FD>,
    /// Q-branch field cfg ($q_0$). Retained for diagnostic and future use
    /// (e.g. consistency checks against per-branch configs); the PCS open
    /// in `step8_pcs_open` runs under a freshly sampled $q''$ instead.
    #[allow(dead_code)] // PCS open uses freshly-sampled $q''$, not $q_0$.
    field_cfg: F::Config,
    /// Per-branch field configs.
    #[allow(dead_code)] // Used by later `fq-unify` phases.
    all_field_cfgs: Vec<F::Config>,
    ic_proof: IdealCheckProof<F>,
    ic_proof_fq: Vec<IdealCheckProof<F>>,
    cpr_proof: CombinedPolyResolverProof<F>,
    combined_sumcheck: MultiDegreeSumcheckProof<F>,
    cpr_proofs_fq: Vec<CombinedPolyResolverProof<F>>,
    combined_sumchecks_fq: Vec<MultiDegreeSumcheckProof<F>>,
    lookup_proof: Option<BatchedLookupProof<F>>,
    booleanity_proof: Option<BooleanityProof<F>>,
    mp_proof: MultipointEvalProof<F>,
    r_0: Vec<F>,
    /// Per-prime multipoint-eval proofs threaded forward from
    /// `ProverMultipointEvaled`.
    mp_proofs_fq: Vec<MultipointEvalProof<F>>,
    /// Per-prime sumcheck output points $r_0$ threaded forward.
    #[allow(dead_code)] // Consumed by Phase H / I.
    r_0_fq: Vec<Vec<F>>,

    // New
    /// Integer-coefficient lifted MLE evaluations at $r_0$, one per
    /// trace column. Public columns first, then witness columns,
    /// interleaved by type. See [`compute_lifted_evals`].
    lifted_evals: Vec<DynamicPolynomialFS<F::Integer>>,
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
> where
    F::Integer: FixedSemiring,
{
    base: ProverCommitted<'a, Zt, U, F, D, FD>,
    ic_proof: IdealCheckProof<F>,
    ic_proof_fq: Vec<IdealCheckProof<F>>,
    cpr_proof: CombinedPolyResolverProof<F>,
    combined_sumcheck: MultiDegreeSumcheckProof<F>,
    cpr_proofs_fq: Vec<CombinedPolyResolverProof<F>>,
    combined_sumchecks_fq: Vec<MultiDegreeSumcheckProof<F>>,
    lookup_proof: Option<BatchedLookupProof<F>>,
    booleanity_proof: Option<BooleanityProof<F>>,
    mp_proof: MultipointEvalProof<F>,
    /// Per-prime multipoint-eval proofs threaded forward.
    mp_proofs_fq: Vec<MultipointEvalProof<F>>,
    lifted_evals: Vec<DynamicPolynomialFS<F::Integer>>,
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
                + for<'b> FromWithConfig<&'b Zt::CombR>
                + for<'b> FromWithConfig<&'b Zt::Chal>
                + for<'b> MulByScalar<&'b F>
                + FromRef<F>
                + Send
                + Sync
                + 'static,
            F::Integer:
                ConstIntSemiring + ConstTranscribable + FromRef<Zt::Fmod> + FromRef<u64> + Send + Sync,
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
    F::Integer: FixedSemiring + ConstTranscribable,
{
    /// Step 0: Folding the trace.
    #[allow(clippy::type_complexity)]
    pub fn step0_fold<'a>(
        trace: &'a UairTrace<'static, Zt::Int, Zt::Int, D, D>,
    ) -> Result<ProverFolded<'a, Zt, U, F, D, FD>, ProtocolError<F>> {
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
    ) -> Result<ProverCommitted<'a, Zt, U, F, D, FD>, ProtocolError<F>> {
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
    ) -> Result<(F::Config, ProjectedScalars<U::Scalar, DynamicPolynomialF<F>>), ProtocolError<F>>
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
    pub fn step2_combined<S: Fn(&U::Scalar, &F::Config) -> DynamicPolynomialF<F> + Copy>(
        mut self,
        project_scalar: S,
    ) -> Result<ProverProjectedCombined<'a, Zt, U, F, D, FD>, ProtocolError<F>> {
        let (field_cfg, projected_scalars_fx) = self.project_common(project_scalar)?;
        let all_field_cfgs = build_all_cfgs::<F>(&self.uair_signature, field_cfg.clone());

        let projected_trace = project_trace_coeffs_row_major(self.original_trace, &field_cfg);

        // Per-prime F_q[X] staging: project trace + scalars under each
        // `phi_{q_i}` deterministically. `project_scalar` is reused with the
        // per-prime cfg.
        let fq_cfgs = &all_field_cfgs[1..];
        let mut fq_staging: Vec<FqProjStaging<U, F>> = Vec::with_capacity(fq_cfgs.len());
        for cfg_q_i in fq_cfgs.iter() {
            let projected_trace_i =
                project_trace_coeffs_row_major(self.original_trace, cfg_q_i);
            let projected_scalars_i = project_scalars::<F, U>(|s| project_scalar(s, cfg_q_i));
            fq_staging.push(FqProjStaging {
                projected_trace: ProjectedTrace::RowMajor(projected_trace_i),
                projected_scalars_fx: projected_scalars_i,
            });
        }

        let q_star_idx = shared_challenge::compute_q_star_idx::<F>(&all_field_cfgs);

        Ok(ProverProjectedCombined {
            base: self,
            field_cfg,
            projected_trace,
            projected_scalars_fx,
            all_field_cfgs,
            q_star_idx,
            fq_staging,
        })
    }

    /// Step 2 (MLE-first / column-major): Prime projection
    /// (`\phi_q`: `Z[X] -> F_q[X]`). Samples a random prime, projects the
    /// full trace and scalars using the column-major layout.
    pub fn step2_mle_first<S: Fn(&U::Scalar, &F::Config) -> DynamicPolynomialF<F> + Copy>(
        mut self,
        project_scalar: S,
    ) -> Result<ProverProjectedMleFirst<'a, Zt, U, F, D, FD>, ProtocolError<F>> {
        let (field_cfg, projected_scalars_fx) = self.project_common(project_scalar)?;
        let all_field_cfgs = build_all_cfgs::<F>(&self.uair_signature, field_cfg.clone());

        let projected_trace = project_trace_coeffs_column_major(self.original_trace, &field_cfg);

        let fq_cfgs = &all_field_cfgs[1..];
        let mut fq_staging: Vec<FqProjStaging<U, F>> = Vec::with_capacity(fq_cfgs.len());
        for cfg_q_i in fq_cfgs.iter() {
            let projected_trace_i =
                project_trace_coeffs_column_major(self.original_trace, cfg_q_i);
            let projected_scalars_i = project_scalars::<F, U>(|s| project_scalar(s, cfg_q_i));
            fq_staging.push(FqProjStaging {
                projected_trace: ProjectedTrace::ColumnMajor(projected_trace_i),
                projected_scalars_fx: projected_scalars_i,
            });
        }

        let q_star_idx = shared_challenge::compute_q_star_idx::<F>(&all_field_cfgs);

        Ok(ProverProjectedMleFirst {
            base: self,
            field_cfg,
            projected_trace,
            projected_scalars_fx,
            all_field_cfgs,
            q_star_idx,
            fq_staging,
        })
    }
});

impl_with_type_bounds!(ProverProjectedCombined
{
    /// Step 3 (combined): Ideal check via `prove_combined` on the row-major
    /// trace. Works for both linear and non-linear constraints.
    ///
    /// Also runs one per-prime $\mathbb{F}_{q_i}[X]$ ideal check per
    /// declared prime in `UairSignature::primes()`, in order. The per-prime
    /// trace and scalars are projected deterministically with `q_i`'s
    /// `field_cfg`.
    ///
    /// **`fq-unify` evaluation point.** All $n + 1$ branches share a single
    /// MLE evaluation point $\mathbf r \in [0, q^*)^\mu$ sampled once from
    /// the transcript at the start of this step. Each branch lifts the
    /// shared integer vector into its own field via $F::from\_with\_cfg$.
    /// Since each shared integer is strictly less than every $q_i$, the
    /// lift is a type cast: all branches agree on the underlying integer.
    ///
    /// TODO(fq-soundness): the per-prime claims produced here are only
    /// ideal-membership checks on the combined polynomials $e_{i,t}$; the
    /// per-prime CPR + sumcheck + multipoint-eval + PCS-open chain that
    /// ties $e_{i,t}$ to the committed trace is not implemented yet.
    /// Adding it means duplicating the post-step3 chain (steps 5..=8) once
    /// per declared prime -- or, equivalently, performing the
    /// `fq-unify` optimization that lets the chain be run once and shared.
    pub fn step3_ideal_check(
        mut self,
    ) -> Result<ProverIdealChecked<'a, Zt, U, F, D, FD>, ProtocolError<F>> {
        let num_constraints = count_constraints::<U>();

        // `fq-unify`: sample one shared evaluation point in `[0, q*)^mu`
        // up-front and lift it into each branch's field.
        let q_star_cfg = &self.all_field_cfgs[self.q_star_idx];
        let shared_eval_points: Vec<Vec<F>> =
            shared_challenge::sample_shared_field_challenges::<F>(
                &mut self.base.pcs_transcript.fs_transcript,
                self.base.num_vars,
                q_star_cfg,
                &self.all_field_cfgs,
            );

        let (ic_proof, _) = IdealCheckProtocol::<U>::prove_combined::<_, D>(
            &mut self.base.pcs_transcript.fs_transcript,
            &self.projected_trace,
            &self.projected_scalars_fx,
            /* branch_idx = */ 0,
            num_constraints.q,
            &shared_eval_points[0],
            &self.field_cfg,
        )?;

        // Per-prime F_q[X] ideal checks, in `primes()` order. Uses the
        // per-prime trace/scalar projections pre-built in step 2.
        let fq_cfgs = &self.all_field_cfgs[1..];
        let mut ic_proof_fq: Vec<IdealCheckProof<F>> = Vec::with_capacity(fq_cfgs.len());
        for (prime_idx, (cfg_q_i, staging)) in
            fq_cfgs.iter().zip(self.fq_staging.iter()).enumerate()
        {
            let branch_idx = add!(prime_idx, 1);
            let ProjectedTrace::RowMajor(ref trace_row) = staging.projected_trace else {
                unreachable!("should be row-major staging")
            };
            let (ic_proof_i, _ic_prover_state_i) = IdealCheckProtocol::<U>::prove_combined::<_, D>(
                &mut self.base.pcs_transcript.fs_transcript,
                trace_row,
                &staging.projected_scalars_fx,
                branch_idx,
                num_constraints.for_prime(prime_idx),
                &shared_eval_points[branch_idx],
                cfg_q_i,
            )
            .map_err(|source| ProtocolError::FqIdealCheck {
                prime_idx,
                q: F::modulus(cfg_q_i).to_string(),
                source,
            })?;

            ic_proof_fq.push(ic_proof_i);
        }

        Ok(ProverIdealChecked {
            base: self.base,
            field_cfg: self.field_cfg,
            all_field_cfgs: self.all_field_cfgs,
            q_star_idx: self.q_star_idx,
            projected_trace: ProjectedTrace::RowMajor(self.projected_trace),
            projected_scalars_fx: self.projected_scalars_fx,
            fq_staging: self.fq_staging,
            ic_proof,
            ic_eval_points: shared_eval_points,
            ic_proof_fq,
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
    ///
    /// **`fq-unify` evaluation point.** See the row-major
    /// [`step3_ideal_check`](ProverProjectedCombined::step3_ideal_check)
    /// for the shared $\mathbf r \in [0, q^*)^\mu$ design; same shape here.
    pub fn step3_ideal_check(
        mut self,
    ) -> Result<ProverIdealChecked<'a, Zt, U, F, D, FD>, ProtocolError<F>> {
        // The Q[X]-branch ideal check only consumes Q[X] constraints; F_q[X]
        // constraints are handled by the per-prime branch below.
        let num_constraints = count_constraints::<U>();

        // `fq-unify`: shared evaluation point in `[0, q*)^mu`, lifted per
        // branch. Mirror of the row-major variant.
        let q_star_cfg = &self.all_field_cfgs[self.q_star_idx];
        let shared_eval_points: Vec<Vec<F>> =
            shared_challenge::sample_shared_field_challenges::<F>(
                &mut self.base.pcs_transcript.fs_transcript,
                self.base.num_vars,
                q_star_cfg,
                &self.all_field_cfgs,
            );

        let (ic_proof, _) = IdealCheckProtocol::<U>::prove_mle_first::<_, D>(
            &mut self.base.pcs_transcript.fs_transcript,
            &self.projected_trace,
            &self.projected_scalars_fx,
            /* branch_idx = */ 0,
            num_constraints.q,
            &shared_eval_points[0],
            &self.field_cfg,
        )?;

        // Per-prime F_q[X] ideal checks (MLE-first / column-major), in
        // `primes()` order. Uses the per-prime trace/scalar projections
        // pre-built in step 2. See `step3_ideal_check` on
        // `ProverProjectedCombined` for the TODO(fq-*) notes -- same caveats.
        let fq_cfgs = &self.all_field_cfgs[1..];
        let mut ic_proof_fq: Vec<IdealCheckProof<F>> = Vec::with_capacity(fq_cfgs.len());
        for (prime_idx, (cfg_q_i, staging)) in
            fq_cfgs.iter().zip(self.fq_staging.iter()).enumerate()
        {
            let branch_idx = add!(prime_idx, 1);
            let ProjectedTrace::ColumnMajor(ref trace_col) = staging.projected_trace else {
                unreachable!("should be column-major staging")
            };
            let (ic_proof_i, _ic_prover_state_i) = IdealCheckProtocol::<U>::prove_mle_first::<_, D>(
                &mut self.base.pcs_transcript.fs_transcript,
                trace_col,
                &staging.projected_scalars_fx,
                branch_idx,
                num_constraints.for_prime(prime_idx),
                &shared_eval_points[branch_idx],
                cfg_q_i,
            )
            .map_err(|source| ProtocolError::FqIdealCheck {
                prime_idx,
                q: F::modulus(cfg_q_i).to_string(),
                source,
            })?;

            ic_proof_fq.push(ic_proof_i);
        }

        Ok(ProverIdealChecked {
            base: self.base,
            field_cfg: self.field_cfg,
            all_field_cfgs: self.all_field_cfgs,
            q_star_idx: self.q_star_idx,
            projected_trace: ProjectedTrace::ColumnMajor(self.projected_trace),
            projected_scalars_fx: self.projected_scalars_fx,
            fq_staging: self.fq_staging,
            ic_proof,
            ic_eval_points: shared_eval_points,
            ic_proof_fq,
        })
    }
});

impl_with_type_bounds!(ProverIdealChecked
{
    /// Step 4: Evaluation projection ($\psi_a$: $F_q[X] \to F_q$).
    ///
    /// **`fq-unify` projecting element.** Sample one shared integer
    /// $a \in [0, q^*)$ once via [`shared_challenge::sample_shared_field_challenge`]
    /// and lift it into each branch's field. The $Q[X]$ branch consumes
    /// `projecting_elements[0]`; per-prime branches consume
    /// `projecting_elements[i + 1]` (Phase F.2.b).
    ///
    /// Also builds the per-prime $\psi$-projected trace MLEs / scalars from
    /// each `fq_staging[i]` using `projecting_elements[i + 1]`, and threads
    /// them forward as `projected_trace_f_fq` / `projected_scalars_f_fq`
    /// for the per-prime CPR sumcheck in step 5.
    pub fn step4_eval_projection(
        mut self,
    ) -> Result<ProverEvalProjected<'a, Zt, U, F, D, FD>, ProtocolError<F>> {
        let q_star_cfg = &self.all_field_cfgs[self.q_star_idx];
        let projecting_elements: Vec<F> = shared_challenge::sample_shared_field_challenge::<F>(
            &mut self.base.pcs_transcript.fs_transcript,
            q_star_cfg,
            &self.all_field_cfgs,
        );

        // Q[X] branch: $\psi_a$-projected trace MLEs + projected scalars.
        let projected_trace_f =
            evaluate_trace_to_column_mles(&self.projected_trace, &projecting_elements[0]);

        let projected_scalars_f =
            project_scalars_to_field(self.projected_scalars_fx, &projecting_elements[0])
                .map_err(|(_s, _f, e)| ProtocolError::ScalarProjection(e))?;

        // Per-prime $F_{q_i}[X]$ branches: same construction with each
        // branch's $\psi$ projecting element.
        let mut projected_trace_f_fq: Vec<Vec<DenseMultilinearExtension<F::Inner>>> =
            Vec::with_capacity(self.fq_staging.len());
        let mut projected_scalars_f_fq: Vec<ProjectedScalars<U::Scalar, F>> =
            Vec::with_capacity(self.fq_staging.len());
        for (prime_idx, staging) in self.fq_staging.into_iter().enumerate() {
            let branch_idx = add!(prime_idx, 1);
            let trace_f_i =
                evaluate_trace_to_column_mles(&staging.projected_trace, &projecting_elements[branch_idx]);
            let scalars_f_i =
                project_scalars_to_field(staging.projected_scalars_fx, &projecting_elements[branch_idx])
                    .map_err(|(_s, _f, e)| ProtocolError::ScalarProjection(e))?;
            projected_trace_f_fq.push(trace_f_i);
            projected_scalars_f_fq.push(scalars_f_i);
        }

        Ok(ProverEvalProjected {
            base: self.base,
            field_cfg: self.field_cfg,
            all_field_cfgs: self.all_field_cfgs,
            q_star_idx: self.q_star_idx,
            projecting_elements,
            projected_trace: self.projected_trace,
            ic_proof: self.ic_proof,
            ic_eval_points: self.ic_eval_points,
            ic_proof_fq: self.ic_proof_fq,
            projected_trace_f,
            projected_scalars_f,
            projected_trace_f_fq,
            projected_scalars_f_fq,
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
    ) -> Result<ProverSumchecked<'a, Zt, U, F, D, FD>, ProtocolError<F>> {
        let num_constraints = count_constraints::<U>();
        let max_degree = count_max_degree::<U>();

        // `fq-unify`: sample one shared CPR batching challenge $\alpha$ in
        // $[0, q^*)$ and lift it into each branch's field. The Q[X] branch
        // consumes `folding_challenges[0]`; per-prime branches consume
        // `folding_challenges[i + 1]`.
        let q_star_cfg_owned = self.all_field_cfgs[self.q_star_idx].clone();
        let folding_challenges: Vec<F> = shared_challenge::sample_shared_field_challenge::<F>(
            &mut self.base.pcs_transcript.fs_transcript,
            &q_star_cfg_owned,
            &self.all_field_cfgs,
        );

        // ------------- Q[X] branch groups -----------------
        // TODO(#185): once protocol-level prover materializes bit-op virtual
        // MLEs, pass them here. For now no UAIR on `main` declares
        // `bit_op_specs`, so passing an empty vec keeps behaviour identical.
        let bit_op_down_mles = Vec::new();
        let (q_cpr_group, q_cpr_ancillary) = CombinedPolyResolver::prepare_sumcheck_group::<U>(
            self.projected_trace_f.clone(),
            bit_op_down_mles,
            &self.ic_eval_points[0],
            &self.projected_scalars_f,
            /* branch_idx = */ 0,
            num_constraints.q,
            self.base.num_vars,
            max_degree,
            &folding_challenges[0],
            &self.field_cfg,
        )?;

        let mut q_groups = vec![q_cpr_group];

        // Booleanity: prepare optional group over witness binary-poly cols.
        // Lives in the Q[X] branch only (binary witness columns are the
        // $\mathbb Z$ side).
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
            q_groups.push(bool_group);
            Some(anc)
        } else {
            None
        };

        // TODO: for each LookupGroup from group_lookup_specs(lookup_specs):
        //   - call prepare_batched_lookup_group(transcript, instance, &field_cfg)
        //   - push triple into groups, collect pending proofs + metas

        // ------------- Per-prime $F_{q_i}[X]$ branch groups --------------
        // One CPR group per declared prime. No booleanity, no lookups in
        // the fq branches (by design — binary witnesses live in Q[X]).
        let n_fq = self.projected_trace_f_fq.len();
        let mut fq_cpr_ancillaries: Vec<_> = Vec::with_capacity(n_fq);
        let mut fq_branch_groups: Vec<Vec<MultiDegreeSumcheckGroup<F>>> =
            Vec::with_capacity(n_fq);
        for prime_idx in 0..n_fq {
            let branch_idx = add!(prime_idx, 1);
            let cfg_i = &self.all_field_cfgs[branch_idx];
            let trace_f_i = self.projected_trace_f_fq[prime_idx].clone();
            let scalars_f_i = &self.projected_scalars_f_fq[prime_idx];
            let eval_point_i = &self.ic_eval_points[branch_idx];
            let folding_i = &folding_challenges[branch_idx];
            let (cpr_group_i, cpr_ancillary_i) =
                CombinedPolyResolver::prepare_sumcheck_group::<U>(
                    trace_f_i,
                    Vec::new(),
                    eval_point_i,
                    scalars_f_i,
                    branch_idx,
                    num_constraints.for_prime(prime_idx),
                    self.base.num_vars,
                    max_degree,
                    folding_i,
                    cfg_i,
                )?;
            fq_branch_groups.push(vec![cpr_group_i]);
            fq_cpr_ancillaries.push(cpr_ancillary_i);
        }

        // ------------- Lockstep multi-degree sumcheck --------------------
        // Branch 0 = Q[X] with CPR + optional booleanity; branches i >= 1
        // = per-prime CPR. Shared per-round challenges in $[0, q^*)$.
        let mut md_sc_branches: Vec<(Vec<MultiDegreeSumcheckGroup<F>>, &F::Config)> =
            Vec::with_capacity(add!(n_fq, 1));
        md_sc_branches.push((q_groups, &self.field_cfg));
        for (prime_idx, groups) in fq_branch_groups.into_iter().enumerate() {
            let branch_idx = add!(prime_idx, 1);
            md_sc_branches.push((groups, &self.all_field_cfgs[branch_idx]));
        }

        let mut sumcheck_outputs = MultiDegreeSumcheck::prove_as_subprotocol(
            &mut self.base.pcs_transcript.fs_transcript,
            md_sc_branches,
            self.base.num_vars,
            &q_star_cfg_owned,
        )
        .into_iter();

        // ------------- Q[X] branch finalize ------------------------------
        let (combined_sumcheck, md_states) =
            sumcheck_outputs.next().expect("Q[X] branch always present");
        let mut md_iter = md_states.into_iter();

        let (cpr_proof, cpr_prover_state) = CombinedPolyResolver::finalize_prover::<U>(
            &mut self.base.pcs_transcript.fs_transcript,
            md_iter.next().expect("CPR group always present"),
            q_cpr_ancillary,
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

        // ------------- Per-prime branch finalize -------------------------
        // For each fq branch: the multi-degree sumcheck handed back a
        // `Vec<SumcheckProverState>` with exactly one entry (just the CPR
        // group). Finalize CPR per branch under that branch's cfg.
        let mut cpr_proofs_fq: Vec<CombinedPolyResolverProof<F>> = Vec::with_capacity(n_fq);
        let mut cpr_eval_points_fq: Vec<Vec<F>> = Vec::with_capacity(n_fq);
        let mut combined_sumchecks_fq: Vec<MultiDegreeSumcheckProof<F>> =
            Vec::with_capacity(n_fq);
        for (prime_idx, cpr_ancillary_i) in fq_cpr_ancillaries.into_iter().enumerate() {
            let branch_idx = add!(prime_idx, 1);
            let cfg_i = &self.all_field_cfgs[branch_idx];
            let (sumcheck_i, states_i) =
                sumcheck_outputs.next().expect("fq branch sumcheck output");
            let mut states_iter_i = states_i.into_iter();
            let (cpr_proof_i, cpr_state_i) = CombinedPolyResolver::finalize_prover::<U>(
                &mut self.base.pcs_transcript.fs_transcript,
                states_iter_i.next().expect("CPR group always present"),
                cpr_ancillary_i,
                cfg_i,
            )?;
            combined_sumchecks_fq.push(sumcheck_i);
            cpr_proofs_fq.push(cpr_proof_i);
            cpr_eval_points_fq.push(cpr_state_i.evaluation_point);
        }

        // Booleanity -> multipoint-eval `alpha_prime` bridge: squeeze
        // \alpha' after `bit_slice_evals` were absorbed by `finalize_prover`.
        let alpha_prime_f: Option<F> = booleanity_proof.as_ref().map(|_| {
            self.base
                .pcs_transcript
                .fs_transcript
                .get_field_challenge(&self.field_cfg)
        });

        Ok(ProverSumchecked {
            base: self.base,
            field_cfg: self.field_cfg,
            all_field_cfgs: self.all_field_cfgs,
            q_star_idx: self.q_star_idx,
            folding_challenges,
            projected_trace: self.projected_trace,
            ic_proof: self.ic_proof,
            ic_proof_fq: self.ic_proof_fq,
            projected_trace_f: self.projected_trace_f,
            projected_trace_f_fq: self.projected_trace_f_fq,
            cpr_proof,
            cpr_eval_point: cpr_prover_state.evaluation_point,
            combined_sumcheck,
            cpr_proofs_fq,
            cpr_eval_points_fq,
            combined_sumchecks_fq,
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
    /// $c_j = \sum_i b_{j,i}\,(\alpha')^{i}$ per witness binary-poly column,
    /// placed at indices `[num_total_cols, num_total_cols + num_wit_bin)`.
    /// On the Q-branch these carry the real bit-slice bridge claim; on the
    /// per-prime branches they are **zero-padded** so all branches share
    /// the same column layout (UAIR rule D6: binary witness columns live
    /// only in Q[X], so per-prime branches have no booleanity-bridge
    /// claim to make there). The shared layout lets a single $(n+1)$-branch
    /// lockstep MP-eval produce a single shared $r_0$ across all branches,
    /// which Phases H/I rely on for the lift-to-$\mathbb Z$ + $q''$-anchored
    /// PCS open. No `ShiftSpec` references those indices, so
    /// `down_evals`/`shifts` are untouched and shifted booleanity is
    /// inherited from the un-shifted column (which continues to live at
    /// its original $\psi_a$-projected slot).
    /// See the module-level `BooleanityChecker` docs for the soundness
    /// argument (Schwartz-Zippel at $\alpha'$ + MP/PCS chain at $r_0$).
    #[allow(clippy::arithmetic_side_effects, clippy::too_many_lines)]
    pub fn step6_multipoint_eval(
        mut self,
    ) -> Result<ProverMultipointEvaled<'a, Zt, U, F, D, FD>, ProtocolError<F>> {
        let n_fq = self.projected_trace_f_fq.len();
        let q_star_cfg = self.all_field_cfgs[self.q_star_idx].clone();
        let shifts = self.base.uair_signature.shifts();
        let num_vars = self.base.num_vars;

        // --- Q[X] branch (booleanity-bridge appended iff alpha_prime present)
        //
        // When booleanity ran, the Q-branch gets extra columns appended:
        //   - `extra_trace_mles[j]`: the $\alpha'$-projection of witness
        //     binary column $j$, as a DenseMultilinearExtension over Q[X]'s
        //     inner type.
        //   - `extra_up_evals[j] = c_j = \sum_i b_{j,i}\,(\alpha')^i$: the
        //     $\alpha'$-batched bit_slice_evals at the booleanity endpoint.
        // These extensions tie booleanity's $r^*$-anchored bit-slice claims
        // into the MP-eval sumcheck so that the MP endpoint $r_0$ also binds
        // them; the verifier's downstream lifted-eval projection at
        // $\alpha'$ closes the loop.
        let (q_trace_mles, q_up_evals, num_wit_bin) = if let Some(alpha_prime) =
            &self.alpha_prime_f
        {
            let sig = &self.base.uair_signature;
            let num_pub_bin = sig.public_cols().num_binary_poly_cols();
            let num_total_bin = sig.total_cols().num_binary_poly_cols();
            let num_wit_bin = num_total_bin.saturating_sub(num_pub_bin);

            // Project the witness binary-poly columns at \alpha' directly from the
            // committed BinaryPoly<D> data:
            // More efficient than the generic `evaluate_trace_to_column_mles` path.
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
            (trace_mles, up_evals, num_wit_bin)
        } else {
            (self.projected_trace_f, self.cpr_proof.up_evals.clone(), 0)
        };

        // --- Per-prime branches (zero-padded to match Q-branch column count)
        //
        // Each fq branch's trace MLEs and up_evals are extended with
        // `num_wit_bin` zero entries, padding to the same column count as
        // the (booleanity-extended) Q-branch. This preserves the lockstep
        // shape (shared `gammas` and one $r_0$ across all branches) without
        // making any non-trivial claim on the per-prime branches: zero on
        // both sides of the MP-eval sumcheck contributes zero.
        let extension_size = 1usize << num_vars;
        let mut fq_trace_mles_ext: Vec<Vec<DenseMultilinearExtension<F::Inner>>> =
            Vec::with_capacity(n_fq);
        let mut fq_up_evals_ext: Vec<Vec<F>> = Vec::with_capacity(n_fq);
        for prime_idx in 0..n_fq {
            let branch_idx = add!(prime_idx, 1);
            let cfg_i = &self.all_field_cfgs[branch_idx];
            let zero_i = F::zero_with_cfg(cfg_i);
            let zero_inner_i = zero_i.inner();

            let mut trace_i = self.projected_trace_f_fq[prime_idx].clone();
            if num_wit_bin > 0 {
                let zero_mle = DenseMultilinearExtension::from_evaluations_vec(
                    num_vars,
                    vec![zero_inner_i.clone(); extension_size],
                    zero_inner_i.clone(),
                );
                trace_i.extend((0..num_wit_bin).map(|_| zero_mle.clone()));
            }
            fq_trace_mles_ext.push(trace_i);

            let mut up_i = self.cpr_proofs_fq[prime_idx].up_evals.clone();
            up_i.extend((0..num_wit_bin).map(|_| zero_i.clone()));
            fq_up_evals_ext.push(up_i);
        }

        // --- Single $(n+1)$-branch lockstep MP-eval ---------------------
        //
        // Branch 0 = Q[X] (with booleanity-bridge cols when applicable);
        // branches $i \ge 1$ = per-prime fq branches (zero-padded to the
        // same column count). All branches share one $r_0$ in $[0, q^*)$.
        let mut all_branches: Vec<MultipointEvalBranchInputs<'_, F>> =
            Vec::with_capacity(add!(n_fq, 1));
        all_branches.push(MultipointEvalBranchInputs {
            trace_mles: &q_trace_mles,
            eval_point: &self.cpr_eval_point,
            up_evals: &q_up_evals,
            down_evals: &self.cpr_proof.down_evals,
            field_cfg: &self.field_cfg,
        });
        for prime_idx in 0..n_fq {
            let branch_idx = add!(prime_idx, 1);
            all_branches.push(MultipointEvalBranchInputs {
                trace_mles: &fq_trace_mles_ext[prime_idx],
                eval_point: &self.cpr_eval_points_fq[prime_idx],
                up_evals: &fq_up_evals_ext[prime_idx],
                down_evals: &self.cpr_proofs_fq[prime_idx].down_evals,
                field_cfg: &self.all_field_cfgs[branch_idx],
            });
        }

        let mut outputs_iter = MultipointEval::prove_as_subprotocol(
            &mut self.base.pcs_transcript.fs_transcript,
            all_branches,
            shifts,
            &q_star_cfg,
        )?
        .into_iter();

        let (mp_proof_q, q_state) = outputs_iter.next().expect("Q-branch present");
        let r_0_q = q_state.eval_point;

        let mut mp_proofs_fq: Vec<MultipointEvalProof<F>> = Vec::with_capacity(n_fq);
        let mut r_0_fq: Vec<Vec<F>> = Vec::with_capacity(n_fq);
        for (proof_i, state_i) in outputs_iter {
            mp_proofs_fq.push(proof_i);
            r_0_fq.push(state_i.eval_point);
        }

        Ok(ProverMultipointEvaled {
            base: self.base,
            field_cfg: self.field_cfg,
            all_field_cfgs: self.all_field_cfgs,
            projected_trace: self.projected_trace,
            ic_proof: self.ic_proof,
            ic_proof_fq: self.ic_proof_fq,
            cpr_proof: self.cpr_proof,
            combined_sumcheck: self.combined_sumcheck,
            cpr_proofs_fq: self.cpr_proofs_fq,
            combined_sumchecks_fq: self.combined_sumchecks_fq,
            lookup_proof: self.lookup_proof,
            booleanity_proof: self.booleanity_proof,
            mp_proof: mp_proof_q,
            r_0: r_0_q,
            mp_proofs_fq,
            r_0_fq,
        })
    }
});

impl_with_type_bounds!(ProverMultipointEvaled
{
    /// Step 7: Lift-and-project. Computes per-column **integer-coefficient**
    /// polynomial MLE evaluations at `r_0` ($\bar u_j \in \mathbb{Z}[X]$)
    /// and absorbs them into the transcript.
    ///
    /// The verifier per-branch lifts each $\bar u_j$ into branch $i$'s
    /// field via $\phi_{q_i}$, then evaluates at
    /// `projecting_elements[i]` to derive the scalar `open_evals` for the
    /// MP-eval sumcheck consistency check on each branch.
    pub fn step7_lift_and_project(
        mut self,
    ) -> Result<ProverLifted<'a, Zt, U, F, D, FD>, ProtocolError<F>> {
        // Compute per-column polynomial MLE evaluations at r_0 as
        // integer-coefficient polynomials. See `compute_lifted_evals_int`
        // for the stepping-stone soundness caveat (today the integers sit
        // in [0, q_0)).
        let lifted_evals = compute_lifted_evals(
            &self.r_0,
            &self.base.original_trace.binary_poly,
            &self.projected_trace,
            &self.field_cfg,
        );

        // **Today** (stepping-stone implementation): the eq-sum accumulator runs
        // in $\mathbb{F} = \mathbb{F}_{q_0}$ exactly as in
        // [`compute_lifted_evals`], then each coefficient is lifted back to
        // $\mathbb{F}$::Integer via [`Field::lift_to_integer`]. This means the
        // integer representation sits in $[0, q_0)$ and is therefore only
        // soundly bound modulo $q_0$.
        //
        // **TODO(fq-soundness)**: replace the inner arithmetic with a wider
        // generic accumulator (over `Zt::Int` / `F::Integer` directly via
        // `Semiring` traits, no concrete `BigInt`) so the integer representation
        // is the (unique) element of $[0, \prod_i q_i \cdot q'')$. That gives
        // per-branch soundness for fq branches via
        // $\phi_{q_i}(\bar u_j) \cdot \psi_{\text{proj}[i]} = \mathrm{up\\_evals}[i][j]$.
        // Until then, the per-branch projection check is structurally in place
        // but binds only the $q_0$ branch.
        let lifted_evals: Vec<_> = lifted_evals.iter().map(|p| p.lift_to_integers()).collect();

        // Absorb integer coefficients into the transcript
        let mut transcription_buf: Vec<u8> = vec![0; F::Integer::NUM_BYTES];
        for bar_u in &lifted_evals {
            self.base
                .pcs_transcript
                .fs_transcript
                .absorb_random_int_slice(&bar_u.coeffs, &mut transcription_buf);
        }

        Ok(ProverLifted {
            base: self.base,
            field_cfg: self.field_cfg,
            all_field_cfgs: self.all_field_cfgs,
            ic_proof: self.ic_proof,
            ic_proof_fq: self.ic_proof_fq,
            cpr_proof: self.cpr_proof,
            combined_sumcheck: self.combined_sumcheck,
            cpr_proofs_fq: self.cpr_proofs_fq,
            combined_sumchecks_fq: self.combined_sumchecks_fq,
            lookup_proof: self.lookup_proof,
            booleanity_proof: self.booleanity_proof,
            mp_proof: self.mp_proof,
            r_0: self.r_0,
            mp_proofs_fq: self.mp_proofs_fq,
            r_0_fq: self.r_0_fq,
            lifted_evals,
        })
    }
});

impl_with_type_bounds!(ProverLifted
{
    /// Step 8: PCS open at $\mathbf r^* := \mathbf r_0 \bmod q''$, where $q''$
    /// is a freshly sampled PCS-only prime.
    ///
    /// Per the `fq-unify` design, the PCS opening prime $q''$ is decoupled
    /// from the constraint primes ($q_0$ and the declared $q_1, \dots, q_n$).
    /// This anchors the witness-polynomial commitments to a single fresh
    /// prime, so PCS soundness is governed entirely by $q''$ and is
    /// independent of the constraint moduli.
    ///
    /// **Transcript ordering**: $q''$ is sampled first (before the binary
    /// folding challenges) so that the folding randomness is bound to the
    /// PCS prime, not the other way around. Mirrored in
    /// [`step7_pcs_verify`](crate::ZincPlusPiop::step7_pcs_verify).
    pub fn step8_pcs_open<const CHECK_FOR_OVERFLOW: bool>(
        mut self,
    ) -> Result<ProverPcsOpened<'a, Zt, U, F, D, FD>, ProtocolError<F>> {
        let witness_trace = &self.base.folded_witness_trace;

        // Sample q'', the PCS-only prime. Decoupled from q0 and the
        // declared q_i; the only role of q'' is to anchor the PCS open.
        let q_pp_cfg = self
            .base
            .pcs_transcript
            .fs_transcript
            .get_random_field_cfg::<F, Zt::Fmod, Zt::PrimeTest>();

        // Build (r* = r0 mod q'') component-wise.
        let r_star: Vec<F> = self
            .r_0
            .iter()
            .map(|x| F::from_with_cfg(x.lift_to_integer(), &q_pp_cfg))
            .collect();

        // Folded witness columns are proved using the extended evaluation
        // point `r_star_ext = r_star || folding_challenges`. Folding
        // challenges are sampled fresh under $q''$.
        let mut r_star_ext = r_star.clone();
        let num_folding_challenges = Zt::BinaryFold::FOLDING_FACTOR.ilog2();
        (0..num_folding_challenges).for_each(|_| {
            let g_chal: Zt::Chal = self.base.pcs_transcript.fs_transcript.get_challenge();
            let gamma = F::from_with_cfg(&g_chal, &q_pp_cfg);
            r_star_ext.push(gamma);
        });

        if let Some(hint_bin) = &self.base.hint_bin {
            let _ = ZipPlus::<Zt::BinaryZt, Zt::BinaryLc>::prove_f::<_, CHECK_FOR_OVERFLOW>(
                &mut self.base.pcs_transcript,
                self.base.pp_bin,
                &witness_trace.binary_poly,
                &r_star_ext,
                hint_bin,
                &q_pp_cfg,
            )?;
        }
        if let Some(hint_arb) = &self.base.hint_arb {
            let _ = ZipPlus::<Zt::ArbitraryZt, Zt::ArbitraryLc>::prove_f::<_, CHECK_FOR_OVERFLOW>(
                &mut self.base.pcs_transcript,
                self.base.pp_arb,
                &witness_trace.arbitrary_poly,
                &r_star,
                hint_arb,
                &q_pp_cfg,
            )?;
        }
        if let Some(hint_int) = &self.base.hint_int {
            let _ = ZipPlus::<Zt::IntZt, Zt::IntLc>::prove_f::<_, CHECK_FOR_OVERFLOW>(
                &mut self.base.pcs_transcript,
                self.base.pp_int,
                &witness_trace.int,
                &r_star,
                hint_int,
                &q_pp_cfg,
            )?;
        }

        Ok(ProverPcsOpened {
            base: self.base,
            ic_proof: self.ic_proof,
            ic_proof_fq: self.ic_proof_fq,
            cpr_proof: self.cpr_proof,
            combined_sumcheck: self.combined_sumcheck,
            cpr_proofs_fq: self.cpr_proofs_fq,
            combined_sumchecks_fq: self.combined_sumchecks_fq,
            lookup_proof: self.lookup_proof,
            booleanity_proof: self.booleanity_proof,
            mp_proof: self.mp_proof,
            mp_proofs_fq: self.mp_proofs_fq,
            lifted_evals: self.lifted_evals,
        })
    }
});

impl_with_type_bounds!(ProverPcsOpened
{
    /// Assemble the final proof from accumulated state.
    pub fn finish(self) -> Result<Proof<F>, ProtocolError<F>> {
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
            ideal_checks_fq: self.ic_proof_fq,
            cpr_proofs_fq: self.cpr_proofs_fq,
            combined_sumchecks_fq: self.combined_sumchecks_fq,
            multipoint_evals_fq: self.mp_proofs_fq,
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
        + for<'a> FromWithConfig<&'a Zt::CombR>
        + for<'a> FromWithConfig<&'a Zt::Chal>
        + for<'a> FromWithConfig<&'a Zt::Pt>
        + for<'a> MulByScalar<&'a F>
        + FromRef<F>
        + Send
        + Sync
        + 'static,
    F::Integer:
        ConstIntSemiring + ConstTranscribable + FromRef<Zt::Fmod> + FromRef<u64> + Send + Sync,
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
        project_scalar: impl Fn(&U::Scalar, &F::Config) -> DynamicPolynomialF<F> + Copy,
    ) -> Result<Proof<F>, ProtocolError<F>> {
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
    F::Integer: Clone + Send + Sync,
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
