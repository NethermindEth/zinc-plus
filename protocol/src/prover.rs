use super::*;
use crate::constraint_system::{ConstraintEndpoints, ConstraintSystem, Layout};
use crypto_primitives::{FromPrimitiveWithConfig, FromWithConfig};
use std::{borrow::Cow, fmt::Debug};
use zinc_piop::projections::{self, ProjectedTrace};
use zinc_poly::{
    mle::DenseMultilinearExtension, univariate::dynamic::over_field::DynamicPolynomialF,
};
use zinc_transcript::traits::{ConstTranscribable, Transcript};
use zinc_uair::UairTrace;
use zinc_utils::{
    add, cfg_join, from_ref::FromRef, inner_transparent_field::InnerTransparentField,
    mul_by_scalar::MulByScalar, projectable_to_field::ProjectableToField,
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
    CS,
    F: PrimeField,
    const D: usize,
    const FD: usize,
> {
    layout: Layout<Zt::Fmod>,
    original_trace: &'a UairTrace<'static, Zt::Int, Zt::Int, D, D>,
    folded_witness_trace: UairTrace<'a, Zt::Int, Zt::Int, FD, D>,

    _phantom: PhantomData<(&'a u8, CS, F)>,
}

/// Persistent prover infrastructure carried across every subsequent
/// step: the Fiat-Shamir transcript, PCS parameters/hints/commitments,
/// and trace reference.
/// Obtained after step 1 via [`step1_commit`](ProverFolded::step1_commit).
#[derive(Clone, Debug)]
pub struct ProverCommitted<
    'a,
    Zt: ZincTypes<D, FD>,
    CS,
    F: PrimeField,
    const D: usize,
    const FD: usize,
> {
    num_vars: usize,
    layout: Layout<Zt::Fmod>,
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

    _phantom: PhantomData<(CS, F)>,
}

/// After step 2 (prime projection, `\phi_q`). Scalar-free: the substrate no
/// longer projects UAIR scalars (that moves into the constraint frontend).
///
/// Holds the per-family $\phi_{q_i}$-projected coefficient traces in family
/// order `[Q[X] (q_0), q_1, .., q_n]` — retained because step 7's lift-and-
/// project reuses them, and a borrow is handed to the constraint frontend's
/// `prove_constraints`.
#[derive(Clone, Debug)]
pub struct ProverProjected<
    'a,
    Zt: ZincTypes<D, FD>,
    CS,
    F: PrimeField,
    const D: usize,
    const FD: usize,
> {
    base: ProverCommitted<'a, Zt, CS, F, D, FD>,
    /// Field configs for all constraint families, starting with randomly
    /// sampled `field_cfg` (for $Q[X]$ constraints, always present) followed by
    /// config for each $q_i$ for $F_{q_i}[X]$ constraints.
    all_field_cfgs: Vec<F::Config>,
    /// Per-family $\phi_{q_i}$-projected coefficient traces, in family order
    /// `[0] = Q[X]`, `[i] = q_i`. Layout (row- vs column-major) matches the
    /// variant chosen at step 2.
    projected_traces: Vec<ProjectedTrace<F>>,
    _phantom: PhantomData<CS>,
}

/// After the constraint argument (`CS::prove_constraints`), which now
/// also runs the lockstep multipoint-eval (former step 6) internally.
///
/// Carries the WHOLE constraint sub-proof (`CS::ConstraintProof`) plus the
/// multipoint-eval endpoints and the retained per-family
/// $\phi_{q_i}$-projected traces for step 7's lift.
#[allow(clippy::type_complexity)]
#[derive(Clone, Debug)]
pub struct ProverConstrained<
    'a,
    Zt: ZincTypes<D, FD>,
    CS: ConstraintSystem,
    F: PrimeField,
    const D: usize,
    const FD: usize,
> {
    base: ProverCommitted<'a, Zt, CS, F, D, FD>,
    field_cfg: F::Config,
    /// Per-family field configs (carried for downstream steps).
    all_field_cfgs: Vec<F::Config>,
    /// Per-family $\phi_{q_i}$-projected coefficient traces, in family order
    /// (`[0]` = Q[X]). Retained for step 7's lift-and-project.
    projected_traces: Vec<ProjectedTrace<F>>,

    /// The whole constraint-argument sub-proof returned by the frontend.
    constraint_proof: CS::ConstraintProof,

    /// Shared evaluation endpoint the frontend's multipoint-eval reduced to,
    /// received via `ConstraintEndpoints` (NOT computed here — the substrate
    /// no longer runs MP-eval). Consumed by step 7's lift-and-project.
    r_0: Vec<F>,
    r_0_fq: Vec<Vec<F>>,
}

/// After step 7 (lift-and-project).
#[derive(Clone, Debug)]
pub struct ProverLifted<
    'a,
    Zt: ZincTypes<D, FD>,
    CS: ConstraintSystem,
    F: PrimeField,
    const D: usize,
    const FD: usize,
> {
    base: ProverCommitted<'a, Zt, CS, F, D, FD>,
    /// The whole constraint-argument sub-proof returned by the frontend.
    constraint_proof: CS::ConstraintProof,

    /// Per-constraint-family **witness-only** lifted MLE evaluations at
    /// $r_0$ (or family-specific $r_0^{(i)}$). Layout: index `0` is the
    /// Q-family ($q_0$); indices `1..=n` are the declared primes in
    /// `UairSignature::primes()` order. Length = `1 + n`.
    /// The verifier recomputes the public-column half per family.
    lifted_evals: Vec<Vec<DynamicPolynomialF<F>>>,
    /// $q''$-family lifted MLE evaluations at $r_0 \bmod q''$, witness
    /// columns only. Used directly by step8's PCS open as the
    /// $\phi_{q''}$-projected claim. Tracked separately from
    /// `lifted_evals` because the $q''$-family is PCS-only (no
    /// per-family constraint check).
    /// If no $F_q[X]$ constraints are present, this will be `None` to indicate
    /// `q'' := q0` and this is identical to `lifted_evals`.
    lifted_evals_pp: Option<Vec<DynamicPolynomialF<F>>>,
    /// PCS-only prime cfg sampled at step 7 start.
    q_pp_cfg: F::Config,
    /// $r^\star = r_0 \bmod q''$ — the PCS evaluation
    /// point for step 8.
    r_star: Vec<F>,
}

/// After step 8 (PCS open). No new fields are added here, but the PCS
/// transcript has been updated with the opening proof.
/// Ready for generating the final proof object in
/// [`finish`](ProverPcsOpened::finish).
#[derive(Clone, Debug)]
pub struct ProverPcsOpened<
    'a,
    Zt: ZincTypes<D, FD>,
    CS: ConstraintSystem,
    F: PrimeField,
    const D: usize,
    const FD: usize,
> {
    base: ProverCommitted<'a, Zt, CS, F, D, FD>,
    /// The whole constraint-argument sub-proof returned by the frontend.
    constraint_proof: CS::ConstraintProof,
    /// Per-constraint-family witness-only lifted evals. Index `0` is the
    /// Q-family; indices `1..=n` are the declared primes (in
    /// `UairSignature::primes()` order). Length = `1 + n`.
    lifted_evals: Vec<Vec<DynamicPolynomialF<F>>>,
    /// $q''$-family witness-only lifted evals (PCS-only family).
    lifted_evals_pp: Option<Vec<DynamicPolynomialF<F>>>,
}

//
// Step implementations
//

/// Prover uses common type bounds across all steps, so we use a helper macro to
/// define them
macro_rules! impl_with_type_bounds {
    ($type_name:ident { $($code:tt)* }) => {
        impl<'a, Zt, CS, F, const D: usize, const FD: usize> $type_name<'a, Zt, CS, F, D, FD>
        where
            Zt: ZincTypes<D, FD>,
            Zt::Int: ProjectableToField<F>,
            <Zt::ArbitraryZt as ZipTypes>::Eval: ProjectableToField<F>,
            CS: ConstraintSystem<Field = F, Prime = Zt::Fmod>,
            F: InnerTransparentField<Integer = Zt::Fmod>
                + FromPrimitiveWithConfig
                + for<'b> FromWithConfig<&'b Zt::Int>
                + for<'b> FromWithConfig<&'b Zt::CombR>
                + for<'b> FromWithConfig<&'b Zt::Chal>
                + for<'b> MulByScalar<&'b F>
                + FromRef<F>
                + Send
                + Sync
                + 'static,
        {
            $($code)*
        }
    };
}

impl<Zt, CS, F, const D: usize, const FD: usize> ZincPlusPiop<Zt, CS, F, D, FD>
where
    Zt: ZincTypes<D, FD>,
    CS: ConstraintSystem<Field = F, Prime = Zt::Fmod>,
    F: PrimeField,
    F::Integer: ConstTranscribable,
{
    /// Step 0: Folding the trace.
    ///
    /// Uses the layout from the constraint system (`cs.layout()`) to slice the
    /// witness columns out of the caller-provided trace.
    #[allow(clippy::type_complexity)]
    pub fn step0_fold<'a>(
        trace: &'a UairTrace<'static, Zt::Int, Zt::Int, D, D>,
        cs: &CS,
    ) -> Result<ProverFolded<'a, Zt, CS, F, D, FD>, ProtocolError<F>> {
        let layout: Layout<Zt::Fmod> = cs.layout().clone();
        let witness_trace = layout.witness_of(trace);

        let folded_bin_witness_trace = cfg_iter!(witness_trace.binary_poly)
            .map(Zt::BinaryFold::fold_trace_mle)
            .collect();

        let folded_witness_trace = UairTrace {
            binary_poly: Cow::Owned(folded_bin_witness_trace),
            arbitrary_poly: witness_trace.arbitrary_poly.clone(),
            int: witness_trace.int.clone(),
        };

        Ok(ProverFolded {
            layout,
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
    ) -> Result<ProverCommitted<'a, Zt, CS, F, D, FD>, ProtocolError<F>> {
        let layout = &self.layout;
        let public_trace = layout.public_of(self.original_trace);

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
            layout: self.layout,
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
    fn project_common<P>(
        mut self,
        project_trace_coeffs: P,
    ) -> Result<ProverProjected<'a, Zt, CS, F, D, FD>, ProtocolError<F>>
    where
        P: for<'b> Fn(&'b UairTrace<'static, Zt::Int, Zt::Int, D, D>, &'b F::Config) -> ProjectedTrace<F>
    {
        // Sample the random Q[X] prime q_0 and build the per-family field configs [q_0, q_1, .., q_n].
        let field_cfg = self
            .pcs_transcript
            .fs_transcript
            .get_random_field_cfg::<F, F::Integer, Zt::PrimeTest>();
        let all_field_cfgs = build_all_cfgs::<F>(&self.layout, field_cfg);

        let projected_traces = all_field_cfgs
            .iter()
            .map(|cfg| project_trace_coeffs(&self.original_trace, cfg))
            .collect();

        Ok(ProverProjected {
            base: self,
            all_field_cfgs,
            projected_traces,
            _phantom: PhantomData,
        })
    }

    /// Step 2 (combined / row-major): Prime projection
    /// (`\phi_q`: `Z[X] -> F_q[X]`). Samples a random prime and projects the
    /// full trace using the row-major layout, per family. Scalar projection no
    /// longer happens here — it moved into the constraint frontend (Option A).
    /// Works for both linear and non-linear constraints.
    pub fn step2_combined(
        self,
    ) -> Result<ProverProjected<'a, Zt, CS, F, D, FD>, ProtocolError<F>> {
        self.project_common(|original_trace, cfg| {
            let proj = projections::project_trace_coeffs_row_major(original_trace, cfg);
            ProjectedTrace::RowMajor(proj)
        })
    }

    /// Step 2 (MLE-first / column-major): Prime projection
    /// (`\phi_q`: `Z[X] -> F_q[X]`). Samples a random prime and projects the
    /// full trace using the column-major layout, per family.
    pub fn step2_mle_first(
        self,
    ) -> Result<ProverProjected<'a, Zt, CS, F, D, FD>, ProtocolError<F>> {
        self.project_common(|original_trace, cfg| {
            let proj = projections::project_trace_coeffs_column_major(original_trace, cfg);
            ProjectedTrace::ColumnMajor(proj)
        })
    }
});

impl_with_type_bounds!(ProverProjected
{
    /// Constraint argument (substrate side).
    ///
    /// Hands the per-family `phi_q`-projected traces to the constraint system's
    /// [`ConstraintSystem::prove_constraints`], which runs the (relocated)
    /// ideal-check / `psi_a` scalar & bit-op / constraint-sumcheck + booleanity
    /// argument **and** the lockstep multipoint-eval (former step 6) that
    /// reduces every per-family claim to the shared endpoint $r_0$. The
    /// substrate receives the whole sub-proof and the
    /// [`ConstraintEndpoints`](crate::constraint_system::ConstraintEndpoints)
    /// ($r_0$ and the per-prime $r_0^{(i)}$); its remaining job is step 7's
    /// lift-and-project + step 8's PCS open at that point.
    ///
    /// The per-family projected traces are *retained* for step 7's
    /// lift-and-project.
    #[allow(clippy::type_complexity)]
    pub fn step_constraints(
        mut self,
        cs: &CS,
    ) -> Result<ProverConstrained<'a, Zt, CS, F, D, FD>, ProtocolError<F>> {
        let num_vars = self.base.num_vars;

        // --- Constraint argument (former steps 3--6, multipoint-eval incl.). ---
        let (constraint_proof, endpoints) = cs.prove_constraints(
            &mut self.base.pcs_transcript.fs_transcript,
            &self.projected_traces,
            &self.all_field_cfgs,
            num_vars,
        )?;

        let ConstraintEndpoints { r_0, r_0_fq } = endpoints;

        let field_cfg = self.all_field_cfgs[0].clone();

        Ok(ProverConstrained {
            base: self.base,
            field_cfg,
            all_field_cfgs: self.all_field_cfgs,
            projected_traces: self.projected_traces,
            constraint_proof,
            r_0,
            r_0_fq,
        })
    }
});

impl_with_type_bounds!(ProverConstrained
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
    ) -> Result<ProverLifted<'a, Zt, CS, F, D, FD>, ProtocolError<F>> {
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
                .get_random_field_cfg::<F, F::Integer, Zt::PrimeTest>();
            let r_star =  self.r_0
                .iter()
                .map(|x| F::from_with_cfg(x.lift_to_integer(), &cfg))
                .collect();
            (cfg, r_star)
        };

        // Witness-col extraction helper. UAIR's column layout interleaves
        // public and witness blocks per type (bin / arb / int), so we slice
        // out the witness sub-blocks and concatenate.
        let sig = self.base.layout.clone();
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
            |all: &[DynamicPolynomialF<F>]| -> Vec<DynamicPolynomialF<F>> {
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
        let mut lifted_evals: Vec<Vec<DynamicPolynomialF<F>>> =
            Vec::with_capacity(add!(n_fq, 1));

        // Q-family (index 0): compute all-col lifted evals, then keep
        // witness-only. We need the all-col version momentarily for the
        // `compute_lifted_evals` call signature (which projects from the
        // already-projected trace), but only the witness slice is sent.
        let q_lifted_all = compute_lifted_evals(
            &self.r_0,
            &self.base.original_trace.binary_poly,
            &self.projected_traces[0],
            &self.field_cfg,
        );
        lifted_evals.push(witness_only(&q_lifted_all));

        // Declared-prime families (indices 1..=n). Reuse the per-prime
        // $\phi_{q_i}$-projected coefficient traces threaded forward from
        // step 2 — same layout as the Q-family's projected trace, just under
        // each $q_i$'s cfg.
        debug_assert_eq!(self.projected_traces.len(), add!(n_fq, 1));
        for (prime_idx, projected_trace_i) in self.projected_traces[1..].iter().enumerate() {
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
            let projected_trace_pp = projections::project_trace_coeffs_row_major::<F, Zt::Int, Zt::Int, D, D>(
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
        let mut transcription_buf: Vec<u8> = vec![0; F::Integer::NUM_BYTES];
        for lifted_i in &lifted_evals {
            for bar_u in lifted_i {
                self.base
                    .pcs_transcript
                    .fs_transcript
                    .absorb_random_field_slice(&bar_u.coeffs, &mut transcription_buf);
            }
        }
        if let Some(ref lifted_pp) = lifted_evals_pp {
            for bar_u in lifted_pp.iter() {
                self.base
                    .pcs_transcript
                    .fs_transcript
                    .absorb_random_field_slice(&bar_u.coeffs, &mut transcription_buf);
            }
        }

        Ok(ProverLifted {
            base: self.base,
            constraint_proof: self.constraint_proof,
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
    /// under $q''$, then calls the PCS opens. Mirrored by the substrate PCS
    /// verify inside the verifier's `finish_verify`.
    pub fn step8_pcs_open<const CHECK_FOR_OVERFLOW: bool>(
        mut self,
    ) -> Result<ProverPcsOpened<'a, Zt, CS, F, D, FD>, ProtocolError<F>> {
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
            let gamma = F::from_with_cfg(&g_chal, q_pp_cfg);
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
            constraint_proof: self.constraint_proof,
            lifted_evals: self.lifted_evals,
            lifted_evals_pp: self.lifted_evals_pp,
        })
    }
});

impl_with_type_bounds!(ProverPcsOpened
{
    /// Assemble the final proof from accumulated state.
    pub fn finish(self) -> Result<Proof<F, CS::ConstraintProof>, ProtocolError<F>> {
        let zip_proof = self.base.pcs_transcript.stream.into_inner();
        let commitments = (
            self.base.commitment_bin,
            self.base.commitment_arb,
            self.base.commitment_int,
        );

        Ok(Proof {
            commitments,
            zip: zip_proof,
            witness_lifted_evals: self.lifted_evals,
            witness_lifted_evals_pp: self.lifted_evals_pp,
            constraint_proof: self.constraint_proof,
        })
    }
});

//
// prove() wrapper
//

impl<Zt, CS, F, const D: usize, const FD: usize> ZincPlusPiop<Zt, CS, F, D, FD>
where
    Zt: ZincTypes<D, FD>,
    Zt::Int: ProjectableToField<F>,
    <Zt::ArbitraryZt as ZipTypes>::Eval: ProjectableToField<F>,
    F: InnerTransparentField<Integer = Zt::Fmod>
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
    CS: ConstraintSystem<Field = F, Prime = Zt::Fmod>,
{
    /// Zinc+ full PIOP prover.
    ///
    /// Runs all protocol steps in sequence and returns the assembled proof.
    /// For per-step control, start with [`Self::step0_fold`] and chain the
    /// individual `stepN_*` methods.
    ///
    /// The constraint argument (former `project_scalar` closure and the
    /// ideal-check / sumcheck / multipoint-eval engine) lives inside the
    /// caller-built `cs` frontend; the prover routes through the
    /// [`ConstraintSystem`] seam.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub fn prove<const MLE_FIRST: bool, const CHECK_FOR_OVERFLOW: bool>(
        pp: &(
            ZipPlusParams<Zt::BinaryZt, Zt::BinaryLc>,
            ZipPlusParams<Zt::ArbitraryZt, Zt::ArbitraryLc>,
            ZipPlusParams<Zt::IntZt, Zt::IntLc>,
        ),
        trace: &UairTrace<'static, Zt::Int, Zt::Int, D, D>,
        num_vars: usize,
        cs: &CS,
    ) -> Result<Proof<F, CS::ConstraintProof>, ProtocolError<F>> {
        let committed = Self::step0_fold(trace, cs)?.step1_commit(pp, num_vars)?;

        let projected = if MLE_FIRST {
            committed.step2_mle_first()?
        } else {
            committed.step2_combined()?
        };

        projected
            .step_constraints(cs)?
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
