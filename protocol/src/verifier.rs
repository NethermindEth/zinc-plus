use super::*;
use crate::constraint_system::{ConstraintEndpoints, ConstraintSystem, Layout};
use crypto_primitives::{FromPrimitiveWithConfig, FromWithConfig};
use itertools::Itertools;
use std::io::Cursor;
use zinc_piop::projections::{ProjectedTrace, project_trace_coeffs_row_major};
use zinc_poly::univariate::dynamic::over_field::DynamicPolynomialF;
use zinc_transcript::{
    Blake3Transcript,
    traits::{ConstTranscribable, Transcript},
};
use zinc_uair::{
    UairTrace,
    ideal::{Ideal, IdealCheck},
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
    layout: Layout<Zt::Fmod>,
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

/// After step 0 (transcript reconstruction). Holds the constraint sub-proof
/// (`CS::ConstraintProof`) plus the substrate [`Proof`] leftovers the later
/// steps consume.
#[derive(Clone, Debug)]
pub struct VerifierTranscriptReconstructed<
    'a,
    Zt: ZincTypes<D, FD>,
    CS: ConstraintSystem,
    F: PrimeField,
    IdealOverF,
    const D: usize,
    const FD: usize,
> {
    base: VerifierBase<'a, Zt, D, FD>,

    // Proof leftovers
    proof_commitments: (ZipPlusCommitment, ZipPlusCommitment, ZipPlusCommitment),
    constraint_proof: CS::ConstraintProof,
    /// Per-constraint-family witness-only lifted MLE evals. Layout:
    /// `[0]` = Q-family ($q_0$), `[i]` = declared prime $i - 1$ for
    /// $i = 1, \ldots, n$. Length = `1 + n_fq`.
    proof_witness_lifted_evals: Vec<Vec<DynamicPolynomialF<F>>>,
    /// Witness-only lifted MLE evals for $q''$ prime.
    /// `None` when there's no $F_q[X]$ constraints, in which case we assume
    /// $q'' := q_0$ and thus `proof_witness_lifted_evals[0]` is used for
    /// PCS verification.
    proof_witness_lifted_evals_pp: Option<Vec<DynamicPolynomialF<F>>>,
    /// Fixed working field from `cs.working_field()` (`None` ⇒ sample at step
    /// 1, as UAIR does). Mirror of the prover's
    /// `ProverFolded::working_field`.
    working_field: Option<F::Config>,
    _phantom: PhantomData<IdealOverF>,
}

/// After step 1 (prime projection).
#[derive(Clone, Debug)]
pub struct VerifierPrimeProjected<
    'a,
    Zt: ZincTypes<D, FD>,
    CS: ConstraintSystem,
    F: PrimeField,
    IdealOverF,
    const D: usize,
    const FD: usize,
> {
    base: VerifierBase<'a, Zt, D, FD>,
    field_cfg: F::Config,

    // Proof leftovers
    proof_commitments: (ZipPlusCommitment, ZipPlusCommitment, ZipPlusCommitment),
    constraint_proof: CS::ConstraintProof,
    /// Per-constraint-family witness-only lifted MLE evals (see
    /// [`VerifierTranscriptReconstructed`] doc). Length = `1 + n_fq`.
    proof_witness_lifted_evals: Vec<Vec<DynamicPolynomialF<F>>>,
    /// Witness-only lifted MLE evals for $q''$ prime (see
    /// [`VerifierTranscriptReconstructed`] doc)
    proof_witness_lifted_evals_pp: Option<Vec<DynamicPolynomialF<F>>>,
    _phantom: PhantomData<IdealOverF>,
}

//
// Step implementations
//

impl<Zt, CS, F, const D: usize, const FD: usize> ZincPlusPiop<Zt, CS, F, D, FD>
where
    Zt: ZincTypes<D, FD>,
    CS: ConstraintSystem<Field = F, Prime = Zt::Fmod>,
    F: PrimeField<Integer = Zt::Fmod>,
{
    /// Step 0: Verifier entry point.
    /// Reconstruct Fiat-Shamir transcript from commitments and public data.
    ///
    /// The constraint sub-proof (`proof.constraint_proof`) is moved straight
    /// into the type-state; there is no longer any repacking of flat `Proof`
    /// fields.
    #[allow(clippy::type_complexity)]
    pub fn step0_reconstruct_transcript<'a, IdealOverF>(
        (vp_bin, vp_arb, vp_int): &'a (
            ZipPlusParams<Zt::BinaryZt, Zt::BinaryLc>,
            ZipPlusParams<Zt::ArbitraryZt, Zt::ArbitraryLc>,
            ZipPlusParams<Zt::IntZt, Zt::IntLc>,
        ),
        mut proof: Proof<F, CS::ConstraintProof>,
        public_trace: &'a UairTrace<'a, Zt::Int, Zt::Int, D, D>,
        num_vars: usize,
        cs: &CS,
    ) -> Result<VerifierTranscriptReconstructed<'a, Zt, CS, F, IdealOverF, D, FD>, ProtocolError<F>>
    where
        IdealOverF: Ideal,
    {
        assert!(
            num_vars > 0,
            "Attempt to verify a constant: num_vars must be > 0"
        );
        let zip_proof = std::mem::take(&mut proof.zip);
        let mut base = VerifierBase {
            num_vars,
            layout: cs.layout().clone(),
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
            constraint_proof: proof.constraint_proof,
            proof_witness_lifted_evals: proof.witness_lifted_evals,
            proof_witness_lifted_evals_pp: proof.witness_lifted_evals_pp,
            working_field: cs.working_field(),
            _phantom: PhantomData,
        })
    }
}

impl<'a, Zt, CS, F, IdealOverF, const D: usize, const FD: usize>
    VerifierTranscriptReconstructed<'a, Zt, CS, F, IdealOverF, D, FD>
where
    Zt: ZincTypes<D, FD>,
    F: InnerTransparentField<Integer = Zt::Fmod>
        + FromPrimitiveWithConfig
        + FromRef<F>
        + Send
        + Sync
        + 'static,
    CS: ConstraintSystem<Field = F>,
    IdealOverF: Ideal,
{
    /// Step 1: Prime projection. Samples the random field configuration.
    #[allow(clippy::type_complexity)]
    pub fn step1_prime_projection(
        mut self,
    ) -> Result<VerifierPrimeProjected<'a, Zt, CS, F, IdealOverF, D, FD>, ProtocolError<F>> {
        // Mirror the prover: use the fixed working field if the constraint
        // system supplies one, else sample a fresh random prime.
        let field_cfg = match self.working_field.take() {
            Some(cfg) => cfg,
            None => self
                .base
                .pcs_transcript
                .fs_transcript
                .get_random_field_cfg::<F, Zt::Fmod, Zt::PrimeTest>(),
        };

        Ok(VerifierPrimeProjected {
            base: self.base,
            field_cfg,
            proof_commitments: self.proof_commitments,
            constraint_proof: self.constraint_proof,
            proof_witness_lifted_evals: self.proof_witness_lifted_evals,
            proof_witness_lifted_evals_pp: self.proof_witness_lifted_evals_pp,
            _phantom: PhantomData,
        })
    }
}

impl<'a, Zt, CS, F, IdealOverF, const D: usize, const FD: usize>
    VerifierPrimeProjected<'a, Zt, CS, F, IdealOverF, D, FD>
where
    Zt: ZincTypes<D, FD>,
    Zt::Int: ProjectableToField<F>,
    <Zt::BinaryZt as ZipTypes>::Cw: ProjectableToField<F>,
    <Zt::ArbitraryZt as ZipTypes>::Eval: ProjectableToField<F>,
    <Zt::ArbitraryZt as ZipTypes>::Cw: ProjectableToField<F>,
    <Zt::IntZt as ZipTypes>::Cw: ProjectableToField<F>,
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
    CS: ConstraintSystem<Field = F, Prime = Zt::Fmod>,
    IdealOverF: Ideal + IdealCheck<DynamicPolynomialF<F>>,
    F::Integer: Ord + num_traits::Zero + Default + Send + Sync,
{
    /// Steps 2--7: run the frontend constraint verify, the substrate
    /// lift-and-project + per-family lifted-eval consistency check, and the
    /// Zip+ PCS verify.
    ///
    /// # Frontend seam (steps 2--5)
    ///
    /// The ideal-check verify, `psi_a` scalar projection, constraint sumcheck +
    /// booleanity verify, and lockstep multipoint-eval verify are all delegated
    /// to [`UairFrontend::verify_constraints`], mirroring how the prover routes
    /// its constraint argument. It returns the shared endpoint(s)
    /// [`ConstraintEndpoints`] and the opaque tail-state
    /// [`UairVerifierClaims`].
    ///
    /// # Substrate lift-and-project (step 6)
    ///
    /// For each family the substrate recomputes the public-column lifted evals,
    /// interleaves them with the sent witness-only lifts to form the full
    /// (public + witness, layout-interleaved) lifted-eval list, then calls
    /// [`UairFrontend::verify_lifted_evals`] which closes every per-family
    /// multipoint-eval subclaim. Afterwards every family's coefficients are
    /// absorbed into the FS transcript in the same uniform order as the prover.
    ///
    /// **When only $Q[X]$ constraints are present** we alias $q'' := q_0$ and
    /// $r^* := r_0$ instead of sampling $q''$ randomly, reusing
    /// `witness_lifted_evals[0]` for the PCS check.
    ///
    /// # PCS verify (step 7)
    ///
    /// Samples the binary folding challenges under $q''$ and runs the
    /// per-commitment PCS verify at $r^\star := r_0 \bmod q''$.
    #[allow(clippy::arithmetic_side_effects, clippy::too_many_lines)]
    pub fn finish_verify<const CHECK_FOR_OVERFLOW: bool>(
        mut self,
        cs: &CS,
        project_scalar: impl Fn(&CS::Scalar, &F::Config) -> DynamicPolynomialF<F>,
        project_ideal: impl Fn(&CS::IdealSource, &F::Config) -> IdealOverF,
        project_fq_ideal: impl Fn(&CS::FqIdealSource, &F::Config) -> IdealOverF,
    ) -> Result<(), ProtocolError<F>> {
        let num_vars = self.base.num_vars;

        // Per-family field configs [q_0, q_1, .., q_n].
        let all_field_cfgs = build_all_cfgs::<F>(&self.base.layout, self.field_cfg.clone());
        let n_fq = all_field_cfgs.len().saturating_sub(1);

        // ---- Constraint argument (former verifier steps 2--5). ----
        let (endpoints, claims): (ConstraintEndpoints<F>, CS::VerifierClaims) = cs
            .verify_constraints::<IdealOverF>(
                &mut self.base.pcs_transcript.fs_transcript,
                &self.constraint_proof,
                &all_field_cfgs,
                num_vars,
                project_ideal,
                project_fq_ideal,
                project_scalar,
            )?;
        let ConstraintEndpoints { r_0, r_0_fq } = endpoints;

        // ---- Step 6: lift-and-project (substrate). ----
        let n_families = add!(n_fq, 1);

        // Length-mismatch guard.
        if self.proof_witness_lifted_evals.len() != n_families {
            return Err(ProtocolError::FqIdealCheck {
                prime_idx: self.proof_witness_lifted_evals.len(),
                q: "<witness-lifted-evals family count mismatch>".to_owned(),
                source: IdealCheckError::IdealCollectorError(
                    zinc_piop::ideal_check::BatchedIdealCheckError::LengthMismatch {
                        num_ideals: n_families,
                        provided_values: self.proof_witness_lifted_evals.len(),
                    },
                ),
            });
        }

        let pub_cols = self.base.layout.public_cols();
        let num_pub_bin = pub_cols.num_binary_poly_cols();
        let num_pub_arb = pub_cols.num_arbitrary_poly_cols();
        let num_pub_int = pub_cols.num_int_cols();

        let wit_cols = self.base.layout.witness_cols();
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

        // Sample q'' (mirror of prover step 7 start). With no F_q[X]
        // constraints, q'' is aliased to q_0 and r* = r0.
        let q_pp_cfg = if n_fq == 0 {
            self.field_cfg.clone()
        } else {
            self.base
                .pcs_transcript
                .fs_transcript
                .get_random_field_cfg::<F, Zt::Fmod, Zt::PrimeTest>()
        };
        // r* = r0 mod q'' (aliased to r0 when q'' = q0).
        let r_star: Vec<F> = if n_fq == 0 {
            r_0.clone()
        } else {
            r_0.iter()
                .map(|x| F::from_with_cfg(x.lift_to_integer(), &q_pp_cfg))
                .collect()
        };

        // Helper: assemble all-column lifted evals from per-family public
        // (recomputed) + sent witness lifts.
        let assemble_all = |public_lifted: &[DynamicPolynomialF<F>],
                            witness_lifted: &[DynamicPolynomialF<F>]|
         -> Vec<DynamicPolynomialF<F>> {
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
            |family_r0: &[F], family_cfg: &F::Config| -> Vec<DynamicPolynomialF<F>> {
                if add!(add!(num_pub_bin, num_pub_arb), num_pub_int) == 0 {
                    return Vec::new();
                }
                let projected_public = project_trace_coeffs_row_major::<F, Zt::Int, Zt::Int, D, D>(
                    self.base.public_trace,
                    family_cfg,
                );
                compute_lifted_evals::<F, D>(
                    family_r0,
                    &self.base.public_trace.binary_poly,
                    &ProjectedTrace::RowMajor(projected_public),
                    family_cfg,
                )
            };

        // Assemble each family's FULL (public + witness) lifted evals at r_0.
        let mut per_family_all_lifted: Vec<Vec<DynamicPolynomialF<F>>> =
            Vec::with_capacity(n_families);

        // Q-family (i = 0).
        let q_public_lifted = recompute_public_lifted(&r_0, &self.field_cfg);
        per_family_all_lifted.push(assemble_all(
            &q_public_lifted,
            &self.proof_witness_lifted_evals[0],
        ));

        // Per-prime families (i >= 1).
        for (prime_idx, r_0_i) in r_0_fq.iter().enumerate() {
            let family_idx = add!(prime_idx, 1);
            let cfg_i = &all_field_cfgs[family_idx];
            let public_lifted_i = recompute_public_lifted(r_0_i, cfg_i);
            per_family_all_lifted.push(assemble_all(
                &public_lifted_i,
                &self.proof_witness_lifted_evals[family_idx],
            ));
        }

        // Close every per-family multipoint-eval subclaim (frontend seam).
        cs.verify_lifted_evals(&claims, &per_family_all_lifted, &all_field_cfgs)?;

        // Absorb all families' coefficients into the FS transcript in the same
        // uniform order as the prover.
        let mut transcription_buf: Vec<u8> = vec![0; F::Integer::NUM_BYTES];
        for witness_lifted_i in &self.proof_witness_lifted_evals {
            for bar_u in witness_lifted_i {
                self.base
                    .pcs_transcript
                    .fs_transcript
                    .absorb_random_field_slice(&bar_u.coeffs, &mut transcription_buf);
            }
        }
        if let Some(ref lifted_evals_pp) = self.proof_witness_lifted_evals_pp {
            for bar_u in lifted_evals_pp.iter() {
                self.base
                    .pcs_transcript
                    .fs_transcript
                    .absorb_random_field_slice(&bar_u.coeffs, &mut transcription_buf);
            }
        }

        // When q'' was aliased to q0, the PCS check reuses Q[X] family's
        // witness lift.
        let lifted_evals_pp = if n_fq == 0 {
            self.proof_witness_lifted_evals[0].clone()
        } else {
            let Some(lifted_evals_pp) = self.proof_witness_lifted_evals_pp else {
                return Err(ProtocolError::WitnessLiftedEvalsPpLengthMismatch {
                    got: 0,
                    expected: expected_pp_len,
                });
            };
            lifted_evals_pp
        };

        // ---- Step 7: PCS verify. ----
        Self::pcs_verify::<CHECK_FOR_OVERFLOW>(
            &mut self.base,
            &self.proof_commitments,
            &lifted_evals_pp,
            &q_pp_cfg,
            &r_star,
        )
    }

    /// Step 7: PCS verification at $r^\star := r_0 \bmod q''$, using the
    /// $q''$-family lifted evals sent by the prover.
    ///
    /// Per-poly claims for `verify_with_alphas` are computed directly from
    /// `lifted_evals_pp[witness_range]` — no per-coefficient $\phi_{q''}$ lift
    /// is needed because the prover already sent each $\bar u_j^{(q'')} \in
    /// F_{q''}[X]$.
    #[allow(clippy::arithmetic_side_effects)]
    fn pcs_verify<const CHECK_FOR_OVERFLOW: bool>(
        base: &mut VerifierBase<'a, Zt, D, FD>,
        commitments: &(ZipPlusCommitment, ZipPlusCommitment, ZipPlusCommitment),
        lifted_evals_pp: &[DynamicPolynomialF<F>],
        q_pp_cfg: &F::Config,
        r_star: &[F],
    ) -> Result<(), ProtocolError<F>> {
        let wit_cols = base.layout.witness_cols();
        let num_wit_bin = wit_cols.num_binary_poly_cols();
        let num_wit_arb = wit_cols.num_arbitrary_poly_cols();

        let vp_bin = base.vp_bin;
        let vp_arb = base.vp_arb;
        let vp_int = base.vp_int;
        let pcs_transcript = &mut base.pcs_transcript;

        let zero = F::zero_with_cfg(q_pp_cfg);

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
                            // bar_u is already in F_{q''}; just batch with alphas.
                            let mut term = F::from_with_cfg(alpha, q_pp_cfg);
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
                    for (bar_u, alphas) in lifted_evals_pp[$evals_range]
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
                F::from_with_cfg(&g_chal, q_pp_cfg)
            })
            .collect_vec();
        let mut r_star_ext = r_star.to_vec();
        r_star_ext.extend_from_slice(&folding_challenges);

        // Witness-only ranges inside `lifted_evals_pp` (layout
        // `[wit_bin..., wit_arb..., wit_int...]`, same as `witness_only` in
        // prover step 7).
        verify_pcs_batch!(
            Zt::BinaryZt,
            Zt::BinaryLc,
            vp_bin,
            0,
            &r_star_ext,
            [0..num_wit_bin],
            |bar_u: &DynamicPolynomialF<F>, alphas: &[_]| {
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
            vp_arb,
            1,
            r_star,
            [num_wit_bin..add!(num_wit_bin, num_wit_arb)]
        );
        verify_pcs_batch!(
            Zt::IntZt,
            Zt::IntLc,
            vp_int,
            2,
            r_star,
            [add!(num_wit_bin, num_wit_arb)..]
        );

        Ok(())
    }
}

//
// verify() wrapper
//

impl<Zt, CS, F, const D: usize, const FD: usize> ZincPlusPiop<Zt, CS, F, D, FD>
where
    Zt: ZincTypes<D, FD>,
    Zt::Int: ProjectableToField<F>,
    <Zt::BinaryZt as ZipTypes>::Cw: ProjectableToField<F>,
    <Zt::ArbitraryZt as ZipTypes>::Eval: ProjectableToField<F>,
    <Zt::ArbitraryZt as ZipTypes>::Cw: ProjectableToField<F>,
    <Zt::IntZt as ZipTypes>::Cw: ProjectableToField<F>,
    F: InnerTransparentField<Integer = Zt::Fmod>
        + FromPrimitiveWithConfig
        + for<'a> FromWithConfig<&'a Zt::Int>
        + for<'a> FromWithConfig<&'a Zt::CombR>
        + for<'a> FromWithConfig<&'a Zt::Chal>
        + for<'a> MulByScalar<&'a F>
        + FromRef<F>
        + Send
        + Sync
        + 'static,
    F::Integer: Ord + num_traits::Zero + Default + Send + Sync,
    CS: ConstraintSystem<Field = F, Prime = Zt::Fmod>,
{
    /// Zinc+ full PIOP verifier.
    ///
    /// Runs all verification steps in sequence and returns `Ok(())` on
    /// success. Steps 0--1 (transcript reconstruct + prime projection) are
    /// substrate; steps 2--5 (constraint argument) are delegated to the
    /// [`ConstraintSystem`] seam; steps 6--7 (lift-and-project + PCS verify)
    /// return to the substrate.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub fn verify<IdealOverF, const CHECK_FOR_OVERFLOW: bool>(
        vp: &(
            ZipPlusParams<Zt::BinaryZt, Zt::BinaryLc>,
            ZipPlusParams<Zt::ArbitraryZt, Zt::ArbitraryLc>,
            ZipPlusParams<Zt::IntZt, Zt::IntLc>,
        ),
        proof: Proof<F, CS::ConstraintProof>,
        public_trace: &UairTrace<Zt::Int, Zt::Int, D, D>,
        num_vars: usize,
        cs: &CS,
        project_scalar: impl Fn(&CS::Scalar, &F::Config) -> DynamicPolynomialF<F>,
        project_ideal: impl Fn(&CS::IdealSource, &F::Config) -> IdealOverF,
        project_fq_ideal: impl Fn(&CS::FqIdealSource, &F::Config) -> IdealOverF,
    ) -> Result<(), ProtocolError<F>>
    where
        IdealOverF: Ideal + IdealCheck<DynamicPolynomialF<F>>,
    {
        ZincPlusPiop::<Zt, CS, F, D, FD>::step0_reconstruct_transcript::<IdealOverF>(
            vp,
            proof,
            public_trace,
            num_vars,
            cs,
        )?
        .step1_prime_projection()?
        .finish_verify::<CHECK_FOR_OVERFLOW>(
            cs,
            project_scalar,
            project_ideal,
            project_fq_ideal,
        )
    }
}

//
// Test-only stepwise entry + accessors, needed for tampering tests that drive
// the constraint-verify transcript replay individually.
//
#[cfg(test)]
pub mod test_helpers {
    use super::*;
    use crate::uair_frontend::{UairConstraintProof, UairFrontend};
    use zinc_piop::{
        ideal_check::{self, IdealCheckProtocol},
        projections::{ProjectedScalars, project_scalars, project_scalars_to_field},
    };
    use zinc_uair::{Uair, constraint_counter::count_constraints, ideal_collector::IdealOrZero};

    /// After the (test-only) stepwise ideal check.
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
        all_field_cfgs: Vec<F::Config>,
        q_star_idx: usize,
        ic_subclaims: Vec<ideal_check::VerifierSubclaim<F>>,
        constraint_proof: UairConstraintProof<F>,
        _phantom: PhantomData<(U, IdealOverF)>,
    }

    /// After the (test-only) stepwise eval projection.
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
        #[allow(dead_code)]
        all_field_cfgs: Vec<F::Config>,
        #[allow(dead_code)]
        q_star_idx: usize,
        ic_subclaims: Vec<ideal_check::VerifierSubclaim<F>>,
        projecting_elements: Vec<F>,
        #[allow(dead_code)]
        projected_scalars_f: ProjectedScalars<U::Scalar, F>,
        #[allow(dead_code)]
        projected_scalars_f_fq: Vec<ProjectedScalars<U::Scalar, F>>,
        constraint_proof: UairConstraintProof<F>,
        _phantom: PhantomData<(U, IdealOverF)>,
    }

    impl<'a, 'cs, Zt, U, F, IdealOverF, const D: usize, const FD: usize>
        VerifierPrimeProjected<'a, Zt, UairFrontend<'cs, U, F, D, FD>, F, IdealOverF, D, FD>
    where
        Zt: ZincTypes<D, FD>,
        Zt::Int: ProjectableToField<F>,
        <Zt::ArbitraryZt as ZipTypes>::Eval: ProjectableToField<F>,
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
        U: Uair<Prime = Zt::Fmod> + 'static,
        IdealOverF: Ideal + IdealCheck<DynamicPolynomialF<F>>,
        F::Integer: Ord + num_traits::Zero + Default + Send + Sync,
    {
        /// Step 2 (test-only): ideal check verification. Mirrors
        /// [`UairFrontend::verify_constraints`](crate::uair_frontend::UairFrontend)'s
        /// step-2 transcript order verbatim.
        #[allow(clippy::type_complexity)]
        pub fn step2_ideal_check(
            mut self,
            project_ideal: impl Fn(&IdealOrZero<U::Ideal>, &F::Config) -> IdealOverF,
            project_fq_ideal: impl Fn(&IdealOrZero<U::FqIdeal>, &F::Config) -> IdealOverF,
        ) -> Result<VerifierIdealChecked<'a, Zt, U, F, IdealOverF, D, FD>, ProtocolError<F>>
        {
            let num_constraints = count_constraints::<U>();
            let all_field_cfgs = build_all_cfgs::<F>(&self.base.layout, self.field_cfg.clone());
            let q_star_idx = shared_challenge::compute_q_star_idx::<F>(&all_field_cfgs);
            let q_star_cfg = &all_field_cfgs[q_star_idx];
            let shared_eval_points: Vec<Vec<F>> =
                shared_challenge::sample_shared_field_challenges::<F>(
                    &mut self.base.pcs_transcript.fs_transcript,
                    self.base.num_vars,
                    q_star_cfg,
                    &all_field_cfgs,
                );

            let mut ic_subclaims: Vec<ideal_check::VerifierSubclaim<F>> =
                Vec::with_capacity(all_field_cfgs.len());
            let q_subclaim = IdealCheckProtocol::<U>::verify_as_subprotocol::<_, IdealOverF, _, _>(
                &mut self.base.pcs_transcript.fs_transcript,
                self.constraint_proof.ideal_check.clone(),
                0,
                num_constraints.q,
                &shared_eval_points[0],
                |ideal| project_ideal(ideal, &self.field_cfg),
                |_| unreachable!("Q[X] family"),
            )?;
            ic_subclaims.push(q_subclaim);

            for (prime_idx, (cfg_q_i, fq_proof)) in all_field_cfgs[1..]
                .iter()
                .zip(self.constraint_proof.ideal_checks_fq.iter())
                .enumerate()
            {
                let family_idx = add!(prime_idx, 1);
                let fq_subclaim =
                    IdealCheckProtocol::<U>::verify_as_subprotocol::<_, IdealOverF, _, _>(
                        &mut self.base.pcs_transcript.fs_transcript,
                        fq_proof.clone(),
                        family_idx,
                        num_constraints.for_prime(prime_idx),
                        &shared_eval_points[family_idx],
                        |_| unreachable!("F_q[X] family"),
                        |ideal| project_fq_ideal(ideal, cfg_q_i),
                    )
                    .map_err(|source| ProtocolError::FqIdealCheck {
                        prime_idx,
                        q: F::modulus(cfg_q_i).to_string(),
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
                constraint_proof: self.constraint_proof,
                _phantom: PhantomData,
            })
        }
    }

    impl<'a, Zt, U, F, IdealOverF, const D: usize, const FD: usize>
        VerifierIdealChecked<'a, Zt, U, F, IdealOverF, D, FD>
    where
        Zt: ZincTypes<D, FD>,
        F: InnerTransparentField<Integer = Zt::Fmod>
            + for<'b> FromWithConfig<&'b Zt::Chal>
            + FromRef<F>
            + Send
            + Sync
            + 'static,
        U: Uair<Prime = Zt::Fmod> + 'static,
        IdealOverF: Ideal,
        F::Integer: Ord,
    {
        /// Step 3 (test-only): evaluation projection.
        pub fn step3_eval_projection(
            mut self,
            project_scalar: impl Fn(&U::Scalar, &F::Config) -> DynamicPolynomialF<F>,
        ) -> Result<VerifierEvalProjected<'a, Zt, U, F, IdealOverF, D, FD>, ProtocolError<F>>
        {
            let q_star_cfg = &self.all_field_cfgs[self.q_star_idx];
            let projecting_elements: Vec<F> = shared_challenge::sample_shared_field_challenge::<F>(
                &mut self.base.pcs_transcript.fs_transcript,
                q_star_cfg,
                &self.all_field_cfgs,
            );

            let projected_scalars_fx =
                project_scalars::<F, U>(|s| project_scalar(s, &self.field_cfg));
            let projected_scalars_f =
                project_scalars_to_field(projected_scalars_fx, &projecting_elements[0])
                    .map_err(|(_s, _f, e)| ProtocolError::ScalarProjection(e))?;

            let projected_scalars_f_fq: Vec<ProjectedScalars<U::Scalar, F>> = self
                .all_field_cfgs
                .iter()
                .zip(projecting_elements.iter())
                .skip(1)
                .map(|(cfg_i, projecting_element)| {
                    let projected_scalars_fx_i =
                        project_scalars::<F, U>(|s| project_scalar(s, cfg_i));
                    project_scalars_to_field(projected_scalars_fx_i, projecting_element)
                        .map_err(|(_s, _f, e)| ProtocolError::ScalarProjection(e))
                })
                .collect::<Result<_, _>>()?;

            Ok(VerifierEvalProjected {
                base: self.base,
                field_cfg: self.field_cfg,
                all_field_cfgs: self.all_field_cfgs,
                q_star_idx: self.q_star_idx,
                ic_subclaims: self.ic_subclaims,
                projecting_elements,
                projected_scalars_f,
                projected_scalars_f_fq,
                constraint_proof: self.constraint_proof,
                _phantom: PhantomData,
            })
        }
    }

    impl<'a, Zt: ZincTypes<D, FD>, const D: usize, const FD: usize> VerifierBase<'a, Zt, D, FD> {
        pub fn fs_transcript_mut(&mut self) -> &mut Blake3Transcript {
            &mut self.pcs_transcript.fs_transcript
        }
    }

    impl<'a, Zt, U, F, IdealOverF, const D: usize, const FD: usize>
        VerifierEvalProjected<'a, Zt, U, F, IdealOverF, D, FD>
    where
        Zt: ZincTypes<D, FD>,
        F: PrimeField,
        U: Uair,
    {
        pub fn projecting_element_f(&self) -> &F {
            &self.projecting_elements[0]
        }

        pub fn field_cfg(&self) -> &F::Config {
            &self.field_cfg
        }

        pub fn fs_transcript_mut(&mut self) -> &mut Blake3Transcript {
            self.base.fs_transcript_mut()
        }

        pub fn proof_combined_sumcheck(&self) -> &MultiDegreeSumcheckProof<F> {
            &self.constraint_proof.combined_sumcheck
        }

        pub fn ic_subclaim(&self) -> &ideal_check::VerifierSubclaim<F> {
            &self.ic_subclaims[0]
        }

        pub fn proof_cpr(&self) -> &CombinedPolyResolverProof<F> {
            &self.constraint_proof.cpr_proof
        }

        pub fn num_vars(&self) -> usize {
            self.base.num_vars
        }

        pub fn layout(&self) -> &Layout<Zt::Fmod> {
            &self.base.layout
        }
    }
}
