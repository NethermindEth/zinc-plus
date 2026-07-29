use super::*;
use itertools::Itertools;
use std::{
    borrow::Cow,
    fmt::{Debug, Display},
};
use zinc_piop::{
    combined_poly_resolver::CombinedPolyResolver,
    ideal_check::{IdealCheckProtocol, Proof as IdealCheckProof},
    lookup::booleanity::{BooleanityChecker, BooleanityProof},
    multipoint_eval::{MultipointEval, MultipointEvalFamilyInputs, Proof as MultipointEvalProof},
    projections::{
        ColumnMajorTrace, ProjectedScalars, ProjectedTrace, RowMajorTrace,
        build_bit_op_virtual_mle, evaluate_trace_to_column_mles, project_scalars,
        project_scalars_to_field, project_trace_coeffs_column_major,
        project_trace_coeffs_row_major,
    },
    sumcheck::multi_degree::{MultiDegreeSumcheck, MultiDegreeSumcheckGroup},
};
use zinc_poly::{mle::DenseMultilinearExtension, univariate::dynamic::DynamicPolynomial};
use zinc_transcript::traits::{ConstTranscribable, Transcript};
use zinc_uair::{
    Uair, UairSignature, UairTrace, constraint_counter::count_constraints,
    degree_counter::count_max_degree,
};
use zinc_utils::{
    add, cfg_iter, cfg_join, mul_by_scalar::MulByScalar, powers,
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
    C: BaseFieldConfig,
    const D: usize,
    const FD: usize,
> {
    uair_signature: UairSignature<Zt::Fmod>,
    original_trace: &'a UairTrace<'static, Zt::Int, Zt::Int, D, D>,
    folded_witness_trace: UairTrace<'a, Zt::Int, Zt::Int, FD, D>,

    _phantom: PhantomData<(&'a u8, U, C)>,
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
    C: BaseFieldConfig,
    const D: usize,
    const FD: usize,
> {
    num_vars: usize,
    uair_signature: UairSignature<Zt::Fmod>,
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

    _phantom: PhantomData<(U, C)>,
}

/// After step 2 via [`step2_combined`](ProverCommitted::step2_combined)
/// (row-major / "combined" projection).
#[derive(Clone, Debug)]
pub struct ProverProjectedCombined<
    'a,
    Zt: ZincTypes<D, FD>,
    U: Uair,
    C: BaseFieldConfig,
    const D: usize,
    const FD: usize,
> {
    base: ProverCommitted<'a, Zt, U, C, D, FD>,
    field_cfg: C,
    projected_trace: RowMajorTrace<C::Element>,
    projected_scalars_fx: ProjectedScalars<U::Scalar, DynamicPolynomial<C::Element>>,
    /// Field configs for all constraint families, starting with randomly
    /// sampled `field_cfg` (for $Q[X]$ constraints, always present) followed by
    /// config for each $q_i$ for $F_{q_i}[X]$ constraints.
    all_field_cfgs: Vec<C>,
    /// Index of $q^* := \min_i q_i$ in `all_field_cfgs`.
    q_star_idx: usize,
    /// Per-prime $F_{q_i}[X]$ projections (one entry per prime in
    /// `UairSignature::primes()`), pre-staged in step 2 so step 3's per-prime
    /// ideal check can read them. Empty for UAIRs with $Q[X]$ only constraints.
    ///
    /// TODO(perf): the row-major projection is duplicated -- once for the
    ///   Q[X] family and once per prime here. A future optimization could
    ///   emit all projections in one trace sweep.
    fq_staging: Vec<FqProjStaging<U, C::Element>>,
}

/// After step 2 via [`step2_mle_first`](ProverCommitted::step2_mle_first)
/// (column-major / MLE-first projection).
#[derive(Clone, Debug)]
pub struct ProverProjectedMleFirst<
    'a,
    Zt: ZincTypes<D, FD>,
    U: Uair,
    C: BaseFieldConfig,
    const D: usize,
    const FD: usize,
> {
    base: ProverCommitted<'a, Zt, U, C, D, FD>,
    field_cfg: C,
    projected_trace: ColumnMajorTrace<C::Element>,
    projected_scalars_fx: ProjectedScalars<U::Scalar, DynamicPolynomial<C::Element>>,
    /// Field configs for all constraint families, starting with randomly
    /// sampled `field_cfg` (for $Q[X]$ constraints, always present) followed by
    /// config for each $q_i$ for $F_{q_i}[X]$ constraints.
    all_field_cfgs: Vec<C>,
    /// Index of $q^* := \min_i q_i$ in `all_field_cfgs`.
    q_star_idx: usize,
    /// Per-prime $F_{q_i}[X]$ projections, column-major layout
    /// counterpart of [`ProverProjectedCombined::fq_staging`].
    ///
    /// TODO(perf): the column-major projection is duplicated -- once for the
    ///   Q[X] family and once per prime here. A future optimization could
    ///   emit all projections in one trace sweep.
    fq_staging: Vec<FqProjStaging<U, C::Element>>,
}

/// Per-prime $\phi_{q_i}$ projection of the integer trace and UAIR scalars,
/// pre-built at step 2 for step 3's per-prime ideal check (and threaded
/// forward through step 4 into the per-prime CPR / sumcheck / MP-eval
/// chain). The trace layout (row- vs column-major) matches the variant
/// chosen at step 2 and is carried inside [`ProjectedTrace`].
///
/// Field config is stored separately on the parent state (see
/// `all_field_cfgs`) and is not needed here.
#[derive(Clone, Debug)]
pub struct FqProjStaging<U: Uair, F: Clone> {
    projected_trace: ProjectedTrace<F>,
    projected_scalars_fx: ProjectedScalars<U::Scalar, DynamicPolynomial<F>>,
}

/// After step 3 (ideal check).
#[derive(Clone, Debug)]
pub struct ProverIdealChecked<
    'a,
    Zt: ZincTypes<D, FD>,
    U: Uair,
    C: BaseFieldConfig,
    const D: usize,
    const FD: usize,
> {
    base: ProverCommitted<'a, Zt, U, C, D, FD>,
    field_cfg: C,
    /// Field configs for all constraint families, starting with randomly
    /// sampled `field_cfg` (for $Q[X]$ constraints, always present) followed by
    /// config for each $q_i$ for $F_{q_i}[X]$ constraints.
    all_field_cfgs: Vec<C>,
    /// Index of $q^* := \min_i q_i$ in `all_field_cfgs`.
    q_star_idx: usize,
    projected_trace: ProjectedTrace<C::Element>,
    projected_scalars_fx: ProjectedScalars<U::Scalar, DynamicPolynomial<C::Element>>,
    /// Per-prime $\phi_{q_i}$ projections from step 2, threaded forward so
    /// step 4 can build per-prime $\psi$-projected trace/scalars and step 5
    /// can drive the per-prime CPR sumcheck families. Empty for UAIRs
    /// with no declared fq primes.
    fq_staging: Vec<FqProjStaging<U, C::Element>>,

    // New
    ic_proof: IdealCheckProof<C::Element>,
    /// Per-family IC evaluation points (full `Vec<Vec<C::Element>>` of size
    /// `n + 1`, sampled once at step 3 via `sample_shared_field_challenges`
    /// and lifted into each family's field). `[0]` is consumed by the
    /// Q[X] CPR in step 5; `[i + 1]` will drive the per-prime CPR.
    ic_eval_points: Vec<Vec<C::Element>>,
    /// Per-prime $F_{q_i}[X]$ ideal-check proofs, one per declared
    /// prime in `base.uair_signature.primes()`, in order.
    ic_proof_fq: Vec<IdealCheckProof<C::Element>>,
}

/// After step 4 (eval projection). `projected_scalars_fx` has been consumed.
#[derive(Clone, Debug)]
pub struct ProverEvalProjected<
    'a,
    Zt: ZincTypes<D, FD>,
    U: Uair,
    C: BaseFieldConfig,
    const D: usize,
    const FD: usize,
> {
    base: ProverCommitted<'a, Zt, U, C, D, FD>,
    field_cfg: C,
    /// Per-family field configs, kept for the per-prime CPR/sumcheck/MP
    /// chain in later phases. `[0]` = $Q[X]$ family.
    all_field_cfgs: Vec<C>,
    /// Index of $q^* := \min_i q_i$ in `all_field_cfgs`.
    q_star_idx: usize,
    projected_trace: ProjectedTrace<C::Element>,
    /// Per-prime $\phi_{q_i}$-projected coefficient traces, threaded
    /// forward from step 2's `fq_staging`. Reused by step 7
    /// (`step7_lift_and_project`) to avoid re-projecting the integer trace
    /// per declared prime. Empty for UAIRs with no declared fq primes.
    projected_trace_fq: Vec<ProjectedTrace<C::Element>>,
    ic_proof: IdealCheckProof<C::Element>,
    /// Per-family IC evaluation points (full $\text{n+1} \times \mu$
    /// matrix). `[0]` feeds the Q[X] CPR; `[i + 1]` feeds the per-prime
    /// CPRs in step 5.
    ic_eval_points: Vec<Vec<C::Element>>,
    ic_proof_fq: Vec<IdealCheckProof<C::Element>>,

    // New
    projected_trace_f: Vec<DenseMultilinearExtension<C::Element>>,
    /// Q[X] family bit-op virtual MLEs, in `UairSignature::bit_op_specs()`
    /// order.
    bit_op_mles: Vec<DenseMultilinearExtension<C::Element>>,
    projected_scalars_f: ProjectedScalars<U::Scalar, C::Element>,
    /// Per-prime $\psi$-projected trace MLEs (one entry per declared prime
    /// in `UairSignature::primes()`). Built in step 4 from each
    /// `fq_staging[i].projected_trace` using `projecting_elements[i + 1]`.
    /// Consumed by per-prime CPR `prepare_sumcheck_group` in step 5.
    /// Empty for UAIRs with no declared fq primes.
    projected_trace_f_fq: Vec<Vec<DenseMultilinearExtension<C::Element>>>,
    /// Per-prime family bit-op virtual MLEs, one vector per declared prime,
    /// each in `UairSignature::bit_op_specs()` order.
    bit_op_mles_fq: Vec<Vec<DenseMultilinearExtension<C::Element>>>,
    /// Per-prime $\psi$-projected scalars (one entry per declared prime).
    /// Built in step 4 from each `fq_staging[i].projected_scalars_fx`
    /// using `projecting_elements[i + 1]`. Consumed in step 5 by the
    /// per-prime CPR. Empty for UAIRs with no declared fq primes.
    projected_scalars_f_fq: Vec<ProjectedScalars<U::Scalar, C::Element>>,
}

/// After step 5 (sumcheck).
#[allow(clippy::type_complexity)]
#[derive(Clone, Debug)]
pub struct ProverSumchecked<
    'a,
    Zt: ZincTypes<D, FD>,
    U: Uair,
    C: BaseFieldConfig,
    const D: usize,
    const FD: usize,
> {
    base: ProverCommitted<'a, Zt, U, C, D, FD>,
    field_cfg: C,
    /// Per-family field configs (carried for downstream steps).
    all_field_cfgs: Vec<C>,
    /// Index of $q^*$ in `all_field_cfgs` (carried for downstream steps).
    q_star_idx: usize,
    projected_trace: ProjectedTrace<C::Element>,
    /// Per-prime $\phi_{q_i}$-projected coefficient traces, threaded
    /// forward from step 4 for reuse by step 7's lifted-eval computation.
    /// Empty for UAIRs with no declared fq primes.
    projected_trace_fq: Vec<ProjectedTrace<C::Element>>,
    ic_proof: IdealCheckProof<C::Element>,
    ic_proof_fq: Vec<IdealCheckProof<C::Element>>,
    /// Trace MLEs at the original $\psi_a$ projecting element, as built
    /// by `evaluate_trace_to_column_mles` in a previous step.
    ///
    /// Carried forward so that the next step can prepend them when assembling
    /// the multipoint-eval inputs (and optionally append $\alpha'$-projected
    /// witness-bin MLEs as the Schwartz-Zippel bridge).
    projected_trace_f: Vec<DenseMultilinearExtension<C::Element>>,
    /// Q[X] family bit-op virtual MLEs, carried to multipoint eval.
    bit_op_mles: Vec<DenseMultilinearExtension<C::Element>>,
    /// Per-prime $\psi$-projected trace MLEs (one entry per declared
    /// prime). Threaded forward from step 4 by `step5_sumcheck` so that
    /// step 6's lockstep multipoint-eval can build per-prime MP families.
    /// Empty for UAIRs with no declared fq primes.
    projected_trace_f_fq: Vec<Vec<DenseMultilinearExtension<C::Element>>>,
    /// Per-prime bit-op virtual MLEs, carried to multipoint eval.
    bit_op_mles_fq: Vec<Vec<DenseMultilinearExtension<C::Element>>>,

    // New
    cpr_proof: CombinedPolyResolverProof<C::Element>,
    cpr_eval_point: Vec<C::Element>,
    combined_sumcheck: MultiDegreeSumcheckProof<C::Element>,
    /// Per-prime CPR proofs (one per declared prime in
    /// `UairSignature::primes()`), produced by each per-prime CPR finalize
    /// in step 5. Empty for UAIRs with no declared fq primes.
    cpr_proofs_fq: Vec<CombinedPolyResolverProof<C::Element>>,
    /// Per-prime CPR sumcheck endpoints `r^*_i`, lifted into each family's
    /// field. Empty for UAIRs with no declared fq primes. Consumed by
    /// step 6's lockstep multipoint-eval.
    cpr_eval_points_fq: Vec<Vec<C::Element>>,
    /// Per-prime multi-degree sumcheck proofs (one per declared prime).
    /// Empty for UAIRs with no declared fq primes.
    combined_sumchecks_fq: Vec<MultiDegreeSumcheckProof<C::Element>>,
    lookup_proof: Option<BatchedLookupProof<C::Element>>,
    booleanity_proof: Option<BooleanityProof<C::Element>>,
    /// Fresh challenge sampled after `bit_slice_evals` were absorbed by
    /// booleanity's `finalize_prover`. Used by the next step to (a) build the
    /// extra $\alpha'$-projected witness-bin trace MLEs and (b) compute
    /// the per-column bridge scalars $c_j = \sum_i (\alpha')^i b_{j,i}$
    /// appended to multipoint-eval's `up_evals`.
    ///
    /// `None` iff there are no witness binary-poly columns (no booleanity
    /// argument).
    alpha_prime_f: Option<C::Element>,
}

/// After step 6 (multipoint eval).
#[derive(Clone, Debug)]
pub struct ProverMultipointEvaled<
    'a,
    Zt: ZincTypes<D, FD>,
    U: Uair,
    C: BaseFieldConfig,
    const D: usize,
    const FD: usize,
> {
    base: ProverCommitted<'a, Zt, U, C, D, FD>,
    field_cfg: C,
    /// Per-family field configs (carried for downstream steps that need
    /// the per-prime cfgs).
    all_field_cfgs: Vec<C>,
    projected_trace: ProjectedTrace<C::Element>,
    /// Per-prime $\phi_{q_i}$-projected coefficient traces, threaded
    /// forward from step 5 for reuse by step 7's lifted-eval computation.
    /// Empty for UAIRs with no declared fq primes.
    projected_trace_fq: Vec<ProjectedTrace<C::Element>>,
    ic_proof: IdealCheckProof<C::Element>,
    ic_proof_fq: Vec<IdealCheckProof<C::Element>>,
    cpr_proof: CombinedPolyResolverProof<C::Element>,
    combined_sumcheck: MultiDegreeSumcheckProof<C::Element>,
    /// Per-prime CPR proofs threaded forward from `ProverSumchecked`.
    cpr_proofs_fq: Vec<CombinedPolyResolverProof<C::Element>>,
    /// Per-prime multi-degree sumcheck proofs threaded forward.
    combined_sumchecks_fq: Vec<MultiDegreeSumcheckProof<C::Element>>,
    lookup_proof: Option<BatchedLookupProof<C::Element>>,
    booleanity_proof: Option<BooleanityProof<C::Element>>,

    // New
    mp_proof: MultipointEvalProof<C::Element>,
    r_0: Vec<C::Element>,
    /// Per-prime multipoint-eval proofs (one per declared prime in
    /// `UairSignature::primes()`), produced by step 6's lockstep
    /// multipoint-eval. Empty for UAIRs with no declared fq primes.
    mp_proofs_fq: Vec<MultipointEvalProof<C::Element>>,
    /// Per-prime sumcheck output points $r_0$ (one per declared prime,
    /// lifted into each family's field — the underlying integer is shared
    /// with the Q-family `r_0` thanks to the lockstep sumcheck). Empty for
    /// UAIRs with no declared fq primes. Consumed in step 7.
    r_0_fq: Vec<Vec<C::Element>>,
}

/// After step 7 (lift-and-project).
#[derive(Clone, Debug)]
pub struct ProverLifted<
    'a,
    Zt: ZincTypes<D, FD>,
    U: Uair,
    C: BaseFieldConfig,
    const D: usize,
    const FD: usize,
> {
    base: ProverCommitted<'a, Zt, U, C, D, FD>,
    field_cfg: C,
    ic_proof: IdealCheckProof<C::Element>,
    ic_proof_fq: Vec<IdealCheckProof<C::Element>>,
    cpr_proof: CombinedPolyResolverProof<C::Element>,
    combined_sumcheck: MultiDegreeSumcheckProof<C::Element>,
    cpr_proofs_fq: Vec<CombinedPolyResolverProof<C::Element>>,
    combined_sumchecks_fq: Vec<MultiDegreeSumcheckProof<C::Element>>,
    lookup_proof: Option<BatchedLookupProof<C::Element>>,
    booleanity_proof: Option<BooleanityProof<C::Element>>,
    mp_proof: MultipointEvalProof<C::Element>,
    /// Per-prime multipoint-eval proofs threaded forward from
    /// `ProverMultipointEvaled`.
    mp_proofs_fq: Vec<MultipointEvalProof<C::Element>>,

    /// Per-constraint-family **witness-only** lifted MLE evaluations at
    /// $r_0$ (or family-specific $r_0^{(i)}$). Layout: index `0` is the
    /// Q-family ($q_0$); indices `1..=n` are the declared primes in
    /// `UairSignature::primes()` order. Length = `1 + r_0_fq.len()`.
    /// The verifier recomputes the public-column half per family.
    lifted_evals: Vec<Vec<DynamicPolynomial<C::Element>>>,
    /// $q''$-family lifted MLE evaluations at $r_0 \bmod q''$, witness
    /// columns only. Used directly by step8's PCS open as the
    /// $\phi_{q''}$-projected claim. Tracked separately from
    /// `lifted_evals` because the $q''$-family is PCS-only (no
    /// per-family constraint check).
    /// If no $F_q[X]$ constraints are present, this will be `None` to indicate
    /// `q'' := q0` and this is identical to `lifted_evals`.
    lifted_evals_pp: Option<Vec<DynamicPolynomial<C::Element>>>,
    /// PCS-only prime cfg sampled at step 7 start.
    q_pp_cfg: C,
    /// $r^\star = r_0 \bmod q''$ — the PCS evaluation
    /// point for step 8.
    r_star: Vec<C::Element>,
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
    C: BaseFieldConfig,
    const D: usize,
    const FD: usize,
> {
    base: ProverCommitted<'a, Zt, U, C, D, FD>,
    field_cfg: C,
    /// PCS-only prime cfg sampled at step 7 start, needed to lift the
    /// $q''$-family section into wire integers.
    q_pp_cfg: C,
    ic_proof: IdealCheckProof<C::Element>,
    ic_proof_fq: Vec<IdealCheckProof<C::Element>>,
    cpr_proof: CombinedPolyResolverProof<C::Element>,
    combined_sumcheck: MultiDegreeSumcheckProof<C::Element>,
    cpr_proofs_fq: Vec<CombinedPolyResolverProof<C::Element>>,
    combined_sumchecks_fq: Vec<MultiDegreeSumcheckProof<C::Element>>,
    lookup_proof: Option<BatchedLookupProof<C::Element>>,
    booleanity_proof: Option<BooleanityProof<C::Element>>,
    mp_proof: MultipointEvalProof<C::Element>,
    /// Per-prime multipoint-eval proofs threaded forward.
    mp_proofs_fq: Vec<MultipointEvalProof<C::Element>>,
    /// Per-constraint-family witness-only lifted evals. Index `0` is the
    /// Q-family; indices `1..=n` are the declared primes (in
    /// `UairSignature::primes()` order). Length = `1 + mp_proofs_fq.len()`.
    lifted_evals: Vec<Vec<DynamicPolynomial<C::Element>>>,
    /// $q''$-family witness-only lifted evals (PCS-only family).
    lifted_evals_pp: Option<Vec<DynamicPolynomial<C::Element>>>,
}

//
// Step implementations
//

/// Prover uses common type bounds across all steps, so we use a helper macro to
/// define them
macro_rules! impl_with_type_bounds {
    ($type_name:ident { $($code:tt)* }) => {
        impl<'a, Zt, U, C, const D: usize, const FD: usize> $type_name<'a, Zt, U, C, D, FD>
        where
            Zt: ZincTypes<D, FD>,
            Zt::Int: ProjectableToField<C>,
            Zt::CombR: MulByScalar<Zt::Chal>,
            <Zt::ArbitraryZt as ZipTypes>::Eval: ProjectableToField<C>,
            U: Uair<Prime = Zt::Fmod> + 'static,
            C: BaseFieldConfig<Integer = Zt::Fmod>
                + ProjectPrimitiveIntegersWithConfig
                + ProjectElementWithConfig<Zt::Int>
                + ProjectElementWithConfig<Zt::CombR>
                + ProjectElementWithConfig<Zt::Chal>
                + 'static,
            C::Element: ConstTranscribable,
        {
            $($code)*
        }
    };
}

impl<Zt, U, C, const D: usize, const FD: usize> ZincPlusPiop<Zt, U, C, D, FD>
where
    Zt: ZincTypes<D, FD>,
    U: Uair<Prime = Zt::Fmod>,
    C: BaseFieldConfig,
    C::Integer: ConstTranscribable,
{
    /// Step 0: Folding the trace.
    #[allow(clippy::type_complexity)]
    pub fn step0_fold<'a>(
        trace: &'a UairTrace<'static, Zt::Int, Zt::Int, D, D>,
    ) -> Result<ProverFolded<'a, Zt, U, C, D, FD>, ProtocolError<C::Element>> {
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
    ) -> Result<ProverCommitted<'a, Zt, U, C, D, FD>, ProtocolError<C::Element>> {
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
    fn project_common<S: Fn(&U::Scalar, &C) -> DynamicPolynomial<C::Element>>(
        &mut self,
        project_scalar: S,
    ) -> Result<(C, ProjectedScalars<U::Scalar, DynamicPolynomial<C::Element>>), ProtocolError<C::Element>>
    {
        let field_cfg = self
            .pcs_transcript
            .fs_transcript
            .get_random_field_cfg::<C, Zt::Fmod, Zt::PrimeTest>();

        let projected_scalars_fx = project_scalars::<C, U>(&field_cfg, |s| project_scalar(s, &field_cfg));
        Ok((field_cfg, projected_scalars_fx))
    }

    /// Step 2 (combined / row-major): Prime projection
    /// (`\phi_q`: `Z[X] -> F_q[X]`). Samples a random prime, projects the
    /// full trace and scalars using the row-major layout.
    /// Works for both linear and non-linear constraints.
    pub fn step2_combined<S: Fn(&U::Scalar, &C) -> DynamicPolynomial<C::Element> + Copy>(
        mut self,
        project_scalar: S,
    ) -> Result<ProverProjectedCombined<'a, Zt, U, C, D, FD>, ProtocolError<C::Element>> {
        let (field_cfg, projected_scalars_fx) = self.project_common(project_scalar)?;
        let all_field_cfgs = build_all_cfgs::<C>(&self.uair_signature, field_cfg.clone());

        let projected_trace = project_trace_coeffs_row_major(self.original_trace, &field_cfg);

        // Per-prime F_q[X] staging: project trace + scalars under each
        // `phi_{q_i}` deterministically. `project_scalar` is reused with the
        // per-prime cfg.
        let fq_cfgs = &all_field_cfgs[1..];
        let mut fq_staging: Vec<FqProjStaging<U, C::Element>> = Vec::with_capacity(fq_cfgs.len());
        for cfg_q_i in fq_cfgs.iter() {
            let projected_trace_i =
                project_trace_coeffs_row_major(self.original_trace, cfg_q_i);
            let projected_scalars_i = project_scalars::<C, U>(cfg_q_i, |s| project_scalar(s, cfg_q_i));
            fq_staging.push(FqProjStaging {
                projected_trace: ProjectedTrace::RowMajor(projected_trace_i),
                projected_scalars_fx: projected_scalars_i,
            });
        }

        let q_star_idx = shared_challenge::compute_q_star_idx::<C>(&all_field_cfgs);

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
    pub fn step2_mle_first<S: Fn(&U::Scalar, &C) -> DynamicPolynomial<C::Element> + Copy>(
        mut self,
        project_scalar: S,
    ) -> Result<ProverProjectedMleFirst<'a, Zt, U, C, D, FD>, ProtocolError<C::Element>> {
        let (field_cfg, projected_scalars_fx) = self.project_common(project_scalar)?;
        let all_field_cfgs = build_all_cfgs::<C>(&self.uair_signature, field_cfg.clone());

        let projected_trace = project_trace_coeffs_column_major(self.original_trace, &field_cfg);

        let fq_cfgs = &all_field_cfgs[1..];
        let mut fq_staging: Vec<FqProjStaging<U, C::Element>> = Vec::with_capacity(fq_cfgs.len());
        for cfg_q_i in fq_cfgs.iter() {
            let projected_trace_i =
                project_trace_coeffs_column_major(self.original_trace, cfg_q_i);
            let projected_scalars_i = project_scalars::<C, U>(cfg_q_i, |s| project_scalar(s, cfg_q_i));
            fq_staging.push(FqProjStaging {
                projected_trace: ProjectedTrace::ColumnMajor(projected_trace_i),
                projected_scalars_fx: projected_scalars_i,
            });
        }

        let q_star_idx = shared_challenge::compute_q_star_idx::<C>(&all_field_cfgs);

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
    /// Also runs one per-prime $F_{q_i}[X]$ ideal check per
    /// declared prime in `UairSignature::primes()`, in order. The per-prime
    /// trace and scalars are projected deterministically with `q_i`'s
    /// `field_cfg`.
    ///
    /// **Shared evaluation point.** All $n + 1$ families share a single
    /// MLE evaluation point $r \in [0, q^*)^\mu$ sampled once from
    /// the transcript at the start of this step. Each family lifts the
    /// shared integer vector into its own field via $C::Element::from\_with\_cfg$.
    /// Since each shared integer is strictly less than every $q_i$, the
    /// lift is a type cast: all families agree on the underlying integer.
    pub fn step3_ideal_check(
        mut self,
    ) -> Result<ProverIdealChecked<'a, Zt, U, C, D, FD>, ProtocolError<C::Element>> {
        let num_constraints = count_constraints::<U>();

        // Sample one shared evaluation point in `[0, q*)^mu`
        // up-front and lift it into each family's field.
        let q_star_cfg = &self.all_field_cfgs[self.q_star_idx];
        let shared_eval_points: Vec<Vec<C::Element>> =
            shared_challenge::sample_shared_field_challenges::<C>(
                &mut self.base.pcs_transcript.fs_transcript,
                self.base.num_vars,
                q_star_cfg,
                &self.all_field_cfgs,
            );

        let ic_proof = IdealCheckProtocol::<U>::prove_combined::<_, D>(
            &mut self.base.pcs_transcript.fs_transcript,
            &self.projected_trace,
            &self.projected_scalars_fx,
            /* family_idx = */ 0,
            num_constraints.q,
            &shared_eval_points[0],
            &self.field_cfg,
        )?;

        // Per-prime F_q[X] ideal checks, in `primes()` order. Uses the
        // per-prime trace/scalar projections pre-built in step 2.
        let fq_cfgs = &self.all_field_cfgs[1..];
        let mut ic_proof_fq: Vec<IdealCheckProof<C::Element>> = Vec::with_capacity(fq_cfgs.len());
        for (prime_idx, (cfg_q_i, staging)) in
            fq_cfgs.iter().zip(self.fq_staging.iter()).enumerate()
        {
            let family_idx = add!(prime_idx, 1);
            let ProjectedTrace::RowMajor(ref trace_row) = staging.projected_trace else {
                unreachable!("should be row-major staging")
            };
            let ic_proof_i = IdealCheckProtocol::<U>::prove_combined::<_, D>(
                &mut self.base.pcs_transcript.fs_transcript,
                trace_row,
                &staging.projected_scalars_fx,
                family_idx,
                num_constraints.for_prime(prime_idx),
                &shared_eval_points[family_idx],
                cfg_q_i,
            )
            .map_err(|source| ProtocolError::FqIdealCheck {
                prime_idx,
                q: cfg_q_i.modulus().to_string(),
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
    /// **Shared evaluation point.** See the row-major
    /// [`step3_ideal_check`](ProverProjectedCombined::step3_ideal_check)
    /// for the shared $r \in [0, q^*)^\mu$ design; same shape here.
    pub fn step3_ideal_check(
        mut self,
    ) -> Result<ProverIdealChecked<'a, Zt, U, C, D, FD>, ProtocolError<C::Element>> {
        // The Q[X]-family ideal check only consumes Q[X] constraints; F_q[X]
        // constraints are handled by the per-prime family below.
        let num_constraints = count_constraints::<U>();

        // Shared evaluation point in `[0, q*)^mu`, lifted per
        // family. Mirror of the row-major variant.
        let q_star_cfg = &self.all_field_cfgs[self.q_star_idx];
        let shared_eval_points: Vec<Vec<C::Element>> =
            shared_challenge::sample_shared_field_challenges::<C>(
                &mut self.base.pcs_transcript.fs_transcript,
                self.base.num_vars,
                q_star_cfg,
                &self.all_field_cfgs,
            );

        let ic_proof = IdealCheckProtocol::<U>::prove_mle_first::<_, D>(
            &mut self.base.pcs_transcript.fs_transcript,
            &self.projected_trace,
            &self.projected_scalars_fx,
            /* family_idx = */ 0,
            num_constraints.q,
            &shared_eval_points[0],
            &self.field_cfg,
        )?;

        // Per-prime F_q[X] ideal checks (MLE-first / column-major), in
        // `primes()` order. Uses the per-prime trace/scalar projections
        // pre-built in step 2.
        let fq_cfgs = &self.all_field_cfgs[1..];
        let mut ic_proof_fq: Vec<IdealCheckProof<C::Element>> = Vec::with_capacity(fq_cfgs.len());
        for (prime_idx, (cfg_q_i, staging)) in
            fq_cfgs.iter().zip(self.fq_staging.iter()).enumerate()
        {
            let family_idx = add!(prime_idx, 1);
            let ProjectedTrace::ColumnMajor(ref trace_col) = staging.projected_trace else {
                unreachable!("should be column-major staging")
            };
            let ic_proof_i = IdealCheckProtocol::<U>::prove_mle_first::<_, D>(
                &mut self.base.pcs_transcript.fs_transcript,
                trace_col,
                &staging.projected_scalars_fx,
                family_idx,
                num_constraints.for_prime(prime_idx),
                &shared_eval_points[family_idx],
                cfg_q_i,
            )
            .map_err(|source| ProtocolError::FqIdealCheck {
                prime_idx,
                q: cfg_q_i.modulus().to_string(),
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
    /// **Shared projecting element.**
    ///
    /// Sample one shared integer $a \in [0, q^*)$ once via [`shared_challenge::sample_shared_field_challenge`]
    /// and lift it into each family's field. The $Q[X]$ family consumes
    /// `projecting_elements[0]`; per-prime families consume
    /// `projecting_elements[i + 1]`.
    ///
    /// Also builds the per-prime $\psi$-projected trace MLEs / scalars from
    /// each `fq_staging[i]` using `projecting_elements[i + 1]`, and threads
    /// them forward as `projected_trace_f_fq` / `projected_scalars_f_fq`
    /// for the per-prime CPR sumcheck in step 5.
    pub fn step4_eval_projection(
        mut self,
    ) -> Result<ProverEvalProjected<'a, Zt, U, C, D, FD>, ProtocolError<C::Element>> {
        let q_star_cfg = &self.all_field_cfgs[self.q_star_idx];
        let projecting_elements: Vec<C::Element> = shared_challenge::sample_shared_field_challenge::<C>(
            &mut self.base.pcs_transcript.fs_transcript,
            q_star_cfg,
            &self.all_field_cfgs,
        );

        // Q[X] family: $\psi_a$-projected trace MLEs + projected scalars.
        let projected_trace_f = evaluate_trace_to_column_mles(
            &self.field_cfg,
            &self.projected_trace,
            &projecting_elements[0],
        );

        let bit_op_specs = self.base.uair_signature.bit_op_specs().to_vec();
        let bit_op_mles = bit_op_specs
            .iter()
            .map(|spec| {
                build_bit_op_virtual_mle::<C, D>(
                    &self.projected_trace,
                    spec,
                    &projecting_elements[0],
                    &self.field_cfg,
                )
            })
            .collect();

        let projected_scalars_f = project_scalars_to_field(
            &self.field_cfg,
            self.projected_scalars_fx,
            &projecting_elements[0],
        )
        .map_err(|(_s, _f, e)| ProtocolError::ScalarProjection(e))?;

        // Per-prime $F_{q_i}[X]$ families: same construction with each
        // family's $\psi$ projecting element. The per-prime
        // $\phi_{q_i}$-projected coefficient traces are retained and
        // threaded forward (`projected_trace_fq`) so step 7's
        // lifted-eval computation can reuse them instead of
        // re-projecting from the integer trace.
        let n_fq = self.fq_staging.len();
        let mut projected_trace_fq: Vec<ProjectedTrace<C::Element>> = Vec::with_capacity(n_fq);
        let mut projected_trace_f_fq: Vec<Vec<DenseMultilinearExtension<C::Element>>> =
            Vec::with_capacity(n_fq);
        let mut bit_op_mles_fq: Vec<Vec<DenseMultilinearExtension<C::Element>>> =
            Vec::with_capacity(n_fq);
        let mut projected_scalars_f_fq: Vec<ProjectedScalars<U::Scalar, C::Element>> =
            Vec::with_capacity(n_fq);
        for (prime_idx, staging) in self.fq_staging.into_iter().enumerate() {
            let family_idx = add!(prime_idx, 1);
            let FqProjStaging {
                projected_trace: projected_trace_i,
                projected_scalars_fx: scalars_fx_i,
            } = staging;
            let trace_f_i = evaluate_trace_to_column_mles(
                &self.all_field_cfgs[family_idx],
                &projected_trace_i,
                &projecting_elements[family_idx],
            );
            let bit_op_mles_i = bit_op_specs
                .iter()
                .map(|spec| {
                    build_bit_op_virtual_mle::<C, D>(
                        &projected_trace_i,
                        spec,
                        &projecting_elements[family_idx],
                        &self.all_field_cfgs[family_idx],
                    )
                })
                .collect();
            let scalars_f_i = project_scalars_to_field(
                &self.all_field_cfgs[family_idx],
                scalars_fx_i,
                &projecting_elements[family_idx],
            )
            .map_err(|(_s, _f, e)| ProtocolError::ScalarProjection(e))?;
            projected_trace_fq.push(projected_trace_i);
            projected_trace_f_fq.push(trace_f_i);
            bit_op_mles_fq.push(bit_op_mles_i);
            projected_scalars_f_fq.push(scalars_f_i);
        }

        Ok(ProverEvalProjected {
            base: self.base,
            field_cfg: self.field_cfg,
            all_field_cfgs: self.all_field_cfgs,
            q_star_idx: self.q_star_idx,
            projected_trace: self.projected_trace,
            projected_trace_fq,
            ic_proof: self.ic_proof,
            ic_eval_points: self.ic_eval_points,
            ic_proof_fq: self.ic_proof_fq,
            projected_trace_f,
            bit_op_mles,
            projected_scalars_f,
            projected_trace_f_fq,
            bit_op_mles_fq,
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
    ) -> Result<ProverSumchecked<'a, Zt, U, C, D, FD>, ProtocolError<C::Element>> {
        let num_constraints = count_constraints::<U>();
        let max_degree = count_max_degree::<U>();

        // Sample one shared CPR batching challenge $\alpha$ in
        // $[0, q^*)$ and lift it into each family's field. The Q[X] family
        // consumes `folding_challenges[0]`; per-prime families consume
        // `folding_challenges[i + 1]`.
        let q_star_cfg_owned = self.all_field_cfgs[self.q_star_idx].clone();
        let folding_challenges: Vec<C::Element> = shared_challenge::sample_shared_field_challenge::<C>(
            &mut self.base.pcs_transcript.fs_transcript,
            &q_star_cfg_owned,
            &self.all_field_cfgs,
        );

        // ------------- Q[X] family groups -----------------
        let (q_cpr_group, q_cpr_ancillary) = CombinedPolyResolver::prepare_sumcheck_group::<U>(
            self.projected_trace_f.clone(),
            self.bit_op_mles.clone(),
            &self.ic_eval_points[0],
            &self.projected_scalars_f,
            /* family_idx = */ 0,
            num_constraints.q,
            self.base.num_vars,
            max_degree,
            &folding_challenges[0],
            &self.field_cfg,
        )?;

        let mut q_groups = vec![q_cpr_group];

        // Booleanity: prepare optional group over witness binary-poly cols.
        // Lives in the Q[X] family only.
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

        // ------------- Per-prime $F_{q_i}[X]$ family groups --------------
        // One CPR group per declared prime. No booleanity, no lookups in
        // the fq families (by design — binary witnesses live in Q[X]).
        let n_fq = self.projected_trace_f_fq.len();
        let mut fq_cpr_ancillaries: Vec<_> = Vec::with_capacity(n_fq);
        let mut fq_family_groups: Vec<Vec<MultiDegreeSumcheckGroup<C>>> =
            Vec::with_capacity(n_fq);
        for prime_idx in 0..n_fq {
            let family_idx = add!(prime_idx, 1);
            let cfg_i = &self.all_field_cfgs[family_idx];
            let trace_f_i = self.projected_trace_f_fq[prime_idx].clone();
            let scalars_f_i = &self.projected_scalars_f_fq[prime_idx];
            let eval_point_i = &self.ic_eval_points[family_idx];
            let folding_i = &folding_challenges[family_idx];
            let (cpr_group_i, cpr_ancillary_i) =
                CombinedPolyResolver::prepare_sumcheck_group::<U>(
                    trace_f_i,
                    self.bit_op_mles_fq[prime_idx].clone(),
                    eval_point_i,
                    scalars_f_i,
                    family_idx,
                    num_constraints.for_prime(prime_idx),
                    self.base.num_vars,
                    max_degree,
                    folding_i,
                    cfg_i,
                )?;
            fq_family_groups.push(vec![cpr_group_i]);
            fq_cpr_ancillaries.push(cpr_ancillary_i);
        }

        // ------------- Lockstep multi-degree sumcheck --------------------
        // Family 0 = Q[X] with CPR + optional booleanity; families i >= 1
        // = per-prime CPR. Shared per-round challenges in $[0, q^*)$.
        let mut md_sc_families: Vec<(Vec<MultiDegreeSumcheckGroup<C>>, &C)> =
            Vec::with_capacity(add!(n_fq, 1));
        md_sc_families.push((q_groups, &self.field_cfg));
        for (prime_idx, groups) in fq_family_groups.into_iter().enumerate() {
            let family_idx = add!(prime_idx, 1);
            md_sc_families.push((groups, &self.all_field_cfgs[family_idx]));
        }

        let mut sumcheck_outputs = MultiDegreeSumcheck::prove_as_subprotocol(
            &mut self.base.pcs_transcript.fs_transcript,
            md_sc_families,
            self.base.num_vars,
            &q_star_cfg_owned,
        )
        .into_iter();

        // ------------- Q[X] family finalize ------------------------------
        let (combined_sumcheck, md_states) =
            sumcheck_outputs.next().expect("Q[X] family always present");
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

        // ------------- Per-prime family finalize -------------------------
        // For each fq family: the multi-degree sumcheck handed back a
        // `Vec<SumcheckProverState>` with exactly one entry (just the CPR
        // group). Finalize CPR per family under that family's cfg.
        let mut cpr_proofs_fq: Vec<CombinedPolyResolverProof<C::Element>> = Vec::with_capacity(n_fq);
        let mut cpr_eval_points_fq: Vec<Vec<C::Element>> = Vec::with_capacity(n_fq);
        let mut combined_sumchecks_fq: Vec<MultiDegreeSumcheckProof<C::Element>> =
            Vec::with_capacity(n_fq);
        for (prime_idx, cpr_ancillary_i) in fq_cpr_ancillaries.into_iter().enumerate() {
            let family_idx = add!(prime_idx, 1);
            let cfg_i = &self.all_field_cfgs[family_idx];
            let (sumcheck_i, states_i) =
                sumcheck_outputs.next().expect("fq family sumcheck output");
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
        let alpha_prime_f: Option<C::Element> = booleanity_proof.as_ref().map(|_| {
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
            projected_trace: self.projected_trace,
            projected_trace_fq: self.projected_trace_fq,
            ic_proof: self.ic_proof,
            ic_proof_fq: self.ic_proof_fq,
            projected_trace_f: self.projected_trace_f,
            bit_op_mles: self.bit_op_mles,
            projected_trace_f_fq: self.projected_trace_f_fq,
            bit_op_mles_fq: self.bit_op_mles_fq,
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
    /// On the Q-family these carry the real bit-slice bridge claim; on the
    /// per-prime families they are **zero-padded** so all families share
    /// the same column layout (UAIR rule D6: binary witness columns live
    /// only in Q[X], so per-prime families have no booleanity-bridge
    /// claim to make there). The shared layout lets a single $(n+1)$-family
    /// lockstep MP-eval produce a single shared $r_0$ across all families,
    /// which Phases H/I rely on for the lift-to-$Z$ + $q''$-anchored
    /// PCS open. No `ShiftSpec` references those indices, so
    /// `down_evals`/`shifts` are untouched and shifted booleanity is
    /// inherited from the un-shifted column (which continues to live at
    /// its original $\psi_a$-projected slot).
    /// See the module-level `BooleanityChecker` docs for the soundness
    /// argument (Schwartz-Zippel at $\alpha'$ + MP/PCS chain at $r_0$).
    #[allow(clippy::arithmetic_side_effects, clippy::too_many_lines)]
    pub fn step6_multipoint_eval(
        mut self,
    ) -> Result<ProverMultipointEvaled<'a, Zt, U, C, D, FD>, ProtocolError<C::Element>> {
        let n_fq = self.projected_trace_f_fq.len();
        let q_star_cfg = self.all_field_cfgs[self.q_star_idx].clone();
        let shifts = self.base.uair_signature.shifts();
        let num_vars = self.base.num_vars;

        // --- Q[X] family (booleanity-bridge appended iff alpha_prime present)
        //
        // When booleanity ran, the Q-family gets extra columns appended:
        //   - `extra_trace_mles[j]`: the $\alpha'$-projection of witness
        //     binary column $j$, as a DenseMultilinearExtension over Q[X]'s
        //     inner type.
        //   - `extra_up_evals[j] = c_j = \sum_i b_{j,i}\,(\alpha')^i$: the
        //     $\alpha'$-batched bit_slice_evals at the booleanity endpoint.
        // These extensions tie booleanity's $r^*$-anchored bit-slice claims
        // into the MP-eval sumcheck so that the MP endpoint $r_0$ also binds
        // them; the verifier's downstream lifted-eval projection at
        // $\alpha'$ closes the loop.
        let mut projected_trace_f = self.projected_trace_f;
        let (q_up_evals, num_wit_bin) = if let Some(alpha_prime) =
            &self.alpha_prime_f
        {
            let sig = &self.base.uair_signature;
            let num_pub_bin = sig.public_cols().num_binary_poly_cols();
            let num_total_bin = sig.total_cols().num_binary_poly_cols();
            let num_wit_bin = num_total_bin.saturating_sub(num_pub_bin);

            // Project the witness binary-poly columns at \alpha' directly from the
            // committed BinaryPoly<D> data:
            // More efficient than the generic `evaluate_trace_to_column_mles` path.
            let alpha_powers: Vec<C::Element> = powers(&self.field_cfg, alpha_prime, D);
            let bin_cols = &self.base.original_trace.binary_poly[num_pub_bin..num_total_bin];
            let extra_trace_mles: Vec<DenseMultilinearExtension<C::Element>> = cfg_iter!(bin_cols)
                .map(|col| project_binary_col_at_field::<C, D>(col, &alpha_powers, &self.field_cfg))
                .collect();
            debug_assert_eq!(extra_trace_mles.len(), num_wit_bin);

            let bp = self
                .booleanity_proof
                .as_ref()
                .expect("booleanity_proof present iff alpha_prime_f is Some");
            let extra_up_evals = alpha_prime_bridge_up_evals::<C, D>(
                &bp.bit_slice_evals,
                num_wit_bin,
                alpha_prime,
                &self.field_cfg,
            );

            projected_trace_f.extend(extra_trace_mles);

            let mut up_evals = self.cpr_proof.up_evals.clone();
            up_evals.extend(extra_up_evals);
            (up_evals, num_wit_bin)
        } else {
            (self.cpr_proof.up_evals.clone(), 0)
        };

        // --- Per-prime families (zero-padded to match Q-family column count)
        //
        // Each fq family's trace MLEs and up_evals are extended with
        // `num_wit_bin` zero entries, padding to the same column count as
        // the (booleanity-extended) Q-family. This preserves the lockstep
        // shape (shared `gammas` and one $r_0$ across all families) without
        // making any non-trivial claim on the per-prime families: zero on
        // both sides of the MP-eval sumcheck contributes zero.
        let extension_size = 1usize << num_vars;
        let mut fq_trace_mles_padded: Vec<Vec<DenseMultilinearExtension<C::Element>>> =
            Vec::with_capacity(n_fq);
        let mut fq_up_evals_padded: Vec<Vec<C::Element>> = Vec::with_capacity(n_fq);
        for (prime_idx, mut trace_i) in self.projected_trace_f_fq.into_iter().enumerate() {
            let family_idx = add!(prime_idx, 1);
            let cfg_i = &self.all_field_cfgs[family_idx];
            let zero_i = cfg_i.zero();

            if num_wit_bin > 0 {
                let zero_mle = DenseMultilinearExtension::from_evaluations_vec(
                    num_vars,
                    vec![zero_i.clone(); extension_size],
                    zero_i.clone(),
                );
                trace_i.extend((0..num_wit_bin).map(|_| zero_mle.clone()));
            }
            fq_trace_mles_padded.push(trace_i);

            let mut up_i = self.cpr_proofs_fq[prime_idx].up_evals.clone();
            up_i.extend((0..num_wit_bin).map(|_| zero_i.clone()));
            fq_up_evals_padded.push(up_i);
        }

        // --- Single (n+1)-family lockstep MP-eval ---------------------
        //
        // Family 0 = Q[X] (with booleanity-bridge cols when applicable);
        // families i >= 1 = per-prime F_q[X] families (zero-padded to the
        // same column count). All families share one r_0 in [0, q*).
        let mut all_families: Vec<MultipointEvalFamilyInputs<'_, C>> =
            Vec::with_capacity(add!(n_fq, 1));
        all_families.push(MultipointEvalFamilyInputs {
            field_cfg: &self.field_cfg,
            trace_mles: &projected_trace_f,
            bit_op_mles: &self.bit_op_mles,
            eval_point: &self.cpr_eval_point,
            up_evals: &q_up_evals,
            bit_op_evals: &self.cpr_proof.bit_op_evals,
            down_evals: &self.cpr_proof.down_evals,
        });
        for prime_idx in 0..n_fq {
            let family_idx = add!(prime_idx, 1);
            all_families.push(MultipointEvalFamilyInputs {
                field_cfg: &self.all_field_cfgs[family_idx],
                trace_mles: &fq_trace_mles_padded[prime_idx],
                bit_op_mles: &self.bit_op_mles_fq[prime_idx],
                eval_point: &self.cpr_eval_points_fq[prime_idx],
                up_evals: &fq_up_evals_padded[prime_idx],
                bit_op_evals: &self.cpr_proofs_fq[prime_idx].bit_op_evals,
                down_evals: &self.cpr_proofs_fq[prime_idx].down_evals,
            });
        }

        let mut outputs_iter = MultipointEval::prove_as_subprotocol(
            &mut self.base.pcs_transcript.fs_transcript,
            all_families,
            shifts,
            &q_star_cfg,
        )?
        .into_iter();

        let (mp_proof_q, q_state) = outputs_iter.next().expect("Q-family present");
        let r_0_q = q_state.eval_point;

        let mut mp_proofs_fq: Vec<MultipointEvalProof<C::Element>> = Vec::with_capacity(n_fq);
        let mut r_0_fq: Vec<Vec<C::Element>> = Vec::with_capacity(n_fq);
        for (proof_i, state_i) in outputs_iter {
            mp_proofs_fq.push(proof_i);
            r_0_fq.push(state_i.eval_point);
        }

        Ok(ProverMultipointEvaled {
            base: self.base,
            field_cfg: self.field_cfg,
            all_field_cfgs: self.all_field_cfgs,
            projected_trace: self.projected_trace,
            projected_trace_fq: self.projected_trace_fq,
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
    /// Step 7: Lift-and-project.
    ///
    /// 1. **Sample $q''$.** A fresh PCS-only prime, decoupled from the
    ///    constraint primes $q_0, q_1, \dots, q_n$. Sampled here (start of
    ///    step 7) — before any lifted evals are produced, so that the
    ///    $q''$-family lift can be computed and sent in this step. This
    ///    is post-commitments and post-$r_0$, so soundness is preserved
    ///    (the prover cannot influence $q''$).
    /// 2. **Compute per-family lifted MLE evaluations** at $r_0$:
    ///    - Q-family ($q_0$): all columns (public + witness) interleaved by
    ///      UAIR column layout, stored locally; only witness columns make
    ///      it into the proof.
    ///    - Per declared prime $q_i$ (i ≥ 1): witness-only lifted evals
    ///      under that family's field cfg. The $\phi_{q_i}$-projected
    ///      coefficient trace was already built in step 2 (`fq_staging`)
    ///      and threaded through `projected_trace_fq`; this step runs
    ///      `compute_lifted_evals` on it at $r_0$ lifted into family
    ///      $i$'s field.
    ///    - $q''$-family: witness-only lifted evals under $q''$. Same
    ///      pattern as fq families; the $r_0$ lifted into $F_{q''}$
    ///      gives $r^\star = r_0 \bmod q''$ which doubles
    ///      as the PCS evaluation point in step 8.
    /// 3. **Absorb** each family's coefficients into the FS transcript in
    ///    a deterministic order: Q-family first, then each declared prime
    ///    in `primes()` order, then $q''$.
    ///
    /// **Soundness**: with $r_0$ shared across all constraint families
    /// (a consequence of the lockstep sumcheck), the per-family MP-eval
    /// consistency check in `step6_lifted_evals` (verifier) binds each
    /// $\bar u_j^{(i)}$ to
    /// the prover's actual $q_i$-projected trace at $r_0$. The
    /// $q''$-family lift is independently bound to the trace by the PCS
    /// open at $r^\star$ in step 8.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn step7_lift_and_project(
        mut self,
    ) -> Result<ProverLifted<'a, Zt, U, C, D, FD>, ProtocolError<C::Element>> {
        let n_fq = self.r_0_fq.len();

        // --- Sample q'' (PCS-only prime) ---
        //
        // The fresh PCS-only prime exists to decouple the witness-trace
        // commitment from *which* of several constraint primes opens it. With
        // no F_q[X] constraints, there is only q_0, so this decoupling buys
        // nothing: we alias q'' := q_0; r* := r_0 and skip q'' lift.
        let (q_pp_cfg, r_star) = if n_fq == 0 {
            (self.field_cfg.clone(), self.r_0.clone())
        } else {
            let cfg = self.base
                .pcs_transcript
                .fs_transcript
                .get_random_field_cfg::<C, Zt::Fmod, Zt::PrimeTest>();
            let r_star = self
                .r_0
                .iter()
                .map(|x| self.field_cfg.lift(x))
                .map(|x| cfg.project(&x))
                .collect();
            (cfg, r_star)
        };

        // Witness-col extraction helper. UAIR's column layout interleaves
        // public and witness blocks per type (bin / arb / int), so we slice
        // out the witness sub-blocks and concatenate.
        let sig = self.base.uair_signature.clone();
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

        let witness_only =
            |all: &[DynamicPolynomial<C::Element>]| -> Vec<DynamicPolynomial<C::Element>> {
                all[num_pub_bin..num_total_bin]
                    .iter()
                    .chain(&all[witness_arb_offset..witness_arb_end])
                    .chain(&all[witness_int_offset..])
                    .cloned()
                    .collect()
            };

        // --- Per-constraint-family witness-only lifted evals ---
        // Index 0: Q-family (q_0) at r_0. Indices 1..=n: declared primes
        // at family-specific r_0_fq[i-1]. Length = 1 + n_fq.
        let mut lifted_evals: Vec<Vec<DynamicPolynomial<C::Element>>> =
            Vec::with_capacity(add!(n_fq, 1));

        // Q-family (index 0): compute all-col lifted evals, then keep
        // witness-only. We need the all-col version momentarily for the
        // `compute_lifted_evals` call signature (which projects from the
        // already-projected trace), but only the witness slice is sent.
        let q_lifted_all = compute_lifted_evals(
            &self.r_0,
            &self.base.original_trace.binary_poly,
            &self.projected_trace,
            &self.field_cfg,
        );
        lifted_evals.push(witness_only(&q_lifted_all));

        // Declared-prime families (indices 1..=n). Reuse the per-prime
        // $\phi_{q_i}$-projected coefficient traces threaded forward from
        // step 2's `fq_staging` (via steps 4--6) — same layout as the
        // Q-family's `projected_trace`, just under each $q_i$'s cfg.
        debug_assert_eq!(self.projected_trace_fq.len(), n_fq);
        for (prime_idx, projected_trace_i) in self.projected_trace_fq.iter().enumerate() {
            let family_idx = add!(prime_idx, 1);
            let cfg_i = &self.all_field_cfgs[family_idx];
            let r_0_i = &self.r_0_fq[prime_idx];
            let lifted_evals_i = compute_lifted_evals(
                r_0_i,
                &self.base.original_trace.binary_poly,
                projected_trace_i,
                cfg_i,
            );
            lifted_evals.push(witness_only(&lifted_evals_i));
        }

        // q'' family: witness-only lifted evals (PCS-only)
        // Compute the q''-projected witness lift at r*.
        // When q'' is aliased to q_0 (no F_q[X] constraints), this lift is
        // identical to the Q-family lift already computed, so we reuse it
        // while avoiding duplication.
        let lifted_evals_pp = if n_fq == 0 {
            None
        } else {
            let projected_trace_pp = project_trace_coeffs_row_major::<C, Zt::Int, Zt::Int, D, D>(
                self.base.original_trace,
                &q_pp_cfg,
            );
            let lifted_evals_pp_full = compute_lifted_evals(
                &r_star,
                &self.base.original_trace.binary_poly,
                &ProjectedTrace::RowMajor(projected_trace_pp),
                &q_pp_cfg,
            );
            Some(witness_only(&lifted_evals_pp_full))
        };

        // --- Absorb all per-family coefficients into the transcript ---
        // Uniform order: each constraint family's witness-only lifted
        // evals, then the q'' family. Mirrored in step6_lifted_evals.
        let mut transcription_buf: Vec<u8> = vec![0; C::Element::NUM_BYTES];
        for lifted_i in &lifted_evals {
            for bar_u in lifted_i {
                self.base
                    .pcs_transcript
                    .fs_transcript
                    .absorb_field_element_slice(&bar_u.coeffs, &mut transcription_buf);
            }
        }
        if let Some(ref lifted_pp) = lifted_evals_pp {
            for bar_u in lifted_pp.iter() {
                self.base
                    .pcs_transcript
                    .fs_transcript
                    .absorb_field_element_slice(&bar_u.coeffs, &mut transcription_buf);
            }
        }

        Ok(ProverLifted {
            base: self.base,
            field_cfg: self.field_cfg,
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
            lifted_evals,
            lifted_evals_pp,
            q_pp_cfg,
            r_star,
        })
    }
});

impl_with_type_bounds!(ProverLifted
{
    /// Step 8: PCS open at $r^\star := r_0 \bmod q''$, where
    /// $q''$ was sampled at the start of step 7.
    ///
    /// The PCS opening prime $q''$ is decoupled from the constraint primes
    /// ($q_0$ and the declared $q_1, \dots, q_n$). This anchors the
    /// witness-polynomial commitments to a single fresh prime, so PCS
    /// soundness is governed entirely by $q''$ and is independent of the
    /// constraint moduli.
    ///
    /// **Transcript ordering**: $q''$ was already sampled at the start of
    /// step 7 (so that the $q''$-family lifted evals could be computed and
    /// sent there). Step 8 only samples the binary folding challenges
    /// under $q''$, then calls the PCS opens. Mirrored in
    /// [`step7_pcs_verify`](crate::ZincPlusPiop::step7_pcs_verify).
    pub fn step8_pcs_open<const CHECK_FOR_OVERFLOW: bool>(
        mut self,
    ) -> Result<ProverPcsOpened<'a, Zt, U, C, D, FD>, ProtocolError<C::Element>> {
        let witness_trace = &self.base.folded_witness_trace;
        let q_pp_cfg = &self.q_pp_cfg;
        let r_star = &self.r_star;

        // Folded witness columns are proved using the extended evaluation
        // point `r_star_ext = r_star || folding_challenges`. Folding
        // challenges are sampled fresh under $q''$.
        let mut r_star_ext = r_star.clone();
        let num_folding_challenges = Zt::BinaryFold::FOLDING_FACTOR.ilog2();
        (0..num_folding_challenges).for_each(|_| {
            let g_chal: Zt::Chal = self.base.pcs_transcript.fs_transcript.get_challenge();
            let gamma = q_pp_cfg.project(&g_chal);
            r_star_ext.push(gamma);
        });

        if let Some(hint_bin) = &self.base.hint_bin {
            let _ = ZipPlus::<Zt::BinaryZt, Zt::BinaryLc>::prove_f::<_, CHECK_FOR_OVERFLOW>(
                &mut self.base.pcs_transcript,
                self.base.pp_bin,
                &witness_trace.binary_poly,
                &r_star_ext,
                hint_bin,
                q_pp_cfg,
            )?;
        }
        if let Some(hint_arb) = &self.base.hint_arb {
            let _ = ZipPlus::<Zt::ArbitraryZt, Zt::ArbitraryLc>::prove_f::<_, CHECK_FOR_OVERFLOW>(
                &mut self.base.pcs_transcript,
                self.base.pp_arb,
                &witness_trace.arbitrary_poly,
                r_star,
                hint_arb,
                q_pp_cfg,
            )?;
        }
        if let Some(hint_int) = &self.base.hint_int {
            let _ = ZipPlus::<Zt::IntZt, Zt::IntLc>::prove_f::<_, CHECK_FOR_OVERFLOW>(
                &mut self.base.pcs_transcript,
                self.base.pp_int,
                &witness_trace.int,
                r_star,
                hint_int,
                q_pp_cfg,
            )?;
        }

        Ok(ProverPcsOpened {
            base: self.base,
            field_cfg: self.field_cfg,
            q_pp_cfg: self.q_pp_cfg,
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
            lifted_evals_pp: self.lifted_evals_pp,
        })
    }
});

impl_with_type_bounds!(ProverPcsOpened
{
    /// Assemble the final proof from accumulated state, lifting every field
    /// element into its canonical integer
    pub fn finish(self) -> Result<Proof<Zt::Fmod>, ProtocolError<C::Element>> {
        let zip_proof = self.base.pcs_transcript.stream.into_inner();
        let commitments = (
            self.base.commitment_bin,
            self.base.commitment_arb,
            self.base.commitment_int,
        );
        let all_cfgs = build_all_cfgs::<C>(&self.base.uair_signature, self.field_cfg.clone());

        // Helpers
        macro_rules! lift {
            ($cfg:expr, $section:expr) => {
                $section.try_map(|e| Ok::<Zt::Fmod, ProtocolError<C::Element>>($cfg.lift(e)))
            };
        }
        macro_rules! lift_fq_vec {
            ($section:expr) => {
                 $section
                    .iter()
                    .enumerate()
                    .map(|(i, p)| lift!(all_cfgs[add!(i, 1)], p))
                    .try_collect()
            };
        }
        let lift_polys = |cfg: &C,
                          polys: &[DynamicPolynomial<C::Element>]|
         -> Result<Vec<DynamicPolynomial<Zt::Fmod>>, ProtocolError<C::Element>> {
            polys
                .iter()
                .map(|p| p.try_map(|e| Ok(cfg.lift(e))))
                .collect()
        };

        let witness_lifted_evals = self
            .lifted_evals
            .iter()
            .enumerate()
            .map(|(i, polys)| lift_polys(&all_cfgs[i], polys))
            .collect::<Result<Vec<_>, _>>()?;
        let witness_lifted_evals_pp = self
            .lifted_evals_pp
            .as_ref()
            .map(|polys| lift_polys(&self.q_pp_cfg, polys))
            .transpose()?;

        Ok(Proof {
            commitments,
            ideal_check: lift!(self.field_cfg, self.ic_proof)?,
            cpr_proof: lift!(self.field_cfg, self.cpr_proof)?,
            combined_sumcheck: lift!(self.field_cfg, self.combined_sumcheck)?,
            multipoint_eval: lift!(self.field_cfg, self.mp_proof)?,
            zip: zip_proof,
            witness_lifted_evals,
            lookup_proof: match &self.lookup_proof {
                Some(p) => Some(lift!(self.field_cfg, p)?),
                None => None,
            },
            booleanity_proof: match &self.booleanity_proof {
                Some(p) => Some(lift!(self.field_cfg, p)?),
                None => None,
            },
            ideal_checks_fq: lift_fq_vec!(self.ic_proof_fq)?,
            cpr_proofs_fq: lift_fq_vec!(self.cpr_proofs_fq)?,
            combined_sumchecks_fq: lift_fq_vec!(self.combined_sumchecks_fq)?,
            multipoint_evals_fq: lift_fq_vec!(self.mp_proofs_fq)?,
            witness_lifted_evals_pp,
        })
    }
});

//
// prove() wrapper
//

impl<Zt, U, C, const D: usize, const FD: usize> ZincPlusPiop<Zt, U, C, D, FD>
where
    Zt: ZincTypes<D, FD>,
    Zt::Int: ProjectableToField<C>,
    Zt::CombR: MulByScalar<Zt::Chal>,
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
    C::Element: ConstTranscribable,
    C::Integer: Display,
    U: Uair<Prime = Zt::Fmod> + 'static,
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
        project_scalar: impl Fn(&U::Scalar, &C) -> DynamicPolynomial<C::Element> + Copy,
    ) -> Result<Proof<Zt::Fmod>, ProtocolError<C::Element>> {
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
fn project_binary_col_at_field<C, const D: usize>(
    col: &DenseMultilinearExtension<BinaryPoly<D>>,
    alpha_powers: &[C::Element],
    field_cfg: &C,
) -> DenseMultilinearExtension<C::Element>
where
    C: BaseFieldConfig,
{
    debug_assert_eq!(alpha_powers.len(), D);
    let zero = field_cfg.zero();

    // Sequential row loop: per-row work is at most `D` conditional adds.
    // The caller parallelizes the outer loop over witness binary-poly columns.
    let evaluations: Vec<C::Element> = col
        .evaluations
        .iter()
        .map(|entry| {
            let mut acc = zero.clone();
            for (i, bit) in entry.iter().enumerate() {
                if *bit.inner() {
                    field_cfg.add_assign(&mut acc, &alpha_powers[i]);
                }
            }
            acc
        })
        .collect();

    DenseMultilinearExtension {
        num_vars: col.num_vars,
        evaluations,
    }
}
