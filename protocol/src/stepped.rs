//! Timing-instrumented variants of the Zinc+ prover and verifier.
//!
//! These report per-step `Duration`s so that benchmarks can measure
//! each protocol phase independently.

use super::*;
use crypto_primitives::{ConstIntSemiring, FromPrimitiveWithConfig, FromWithConfig};
use num_traits::Zero;
use std::{
    io::Cursor,
    time::{Duration, Instant},
};
use zinc_piop::{
    combined_poly_resolver::CombinedPolyResolver,
    ideal_check::IdealCheckProtocol,
    multipoint_eval::MultipointEval,
    projections::{
        ProjectedTrace, evaluate_trace_to_column_mles, project_scalars, project_scalars_to_field,
        project_trace_coeffs_column_major, project_trace_coeffs_row_major,
    },
    sumcheck::multi_degree::MultiDegreeSumcheck,
};
use zinc_poly::{EvaluatablePolynomial, univariate::dynamic::over_field::DynamicPolynomialF};
use zinc_transcript::{
    Blake3Transcript,
    traits::{ConstTranscribable, Transcript},
};
use zinc_uair::{
    Uair, UairTrace,
    constraint_counter::count_constraints,
    degree_counter::count_max_degree,
    ideal::{Ideal, IdealCheck},
    ideal_collector::IdealOrZero,
};
use zinc_utils::{
    add, cfg_join, from_ref::FromRef, inner_transparent_field::InnerTransparentField,
    mul_by_scalar::MulByScalar, projectable_to_field::ProjectableToField,
};
use zip_plus::{
    pcs::structs::{ZipPlus, ZipPlusHint, ZipPlusParams, ZipTypes},
    pcs_transcript::{PcsProverTranscript, PcsVerifierTranscript},
};

/// Per-step durations recorded by [`ZincPlusPiop::prove_with_timings`].
#[derive(Clone, Debug)]
pub struct ProverStepTimings {
    pub commit: Duration,
    pub prime_projection: Duration,
    pub ideal_check: Duration,
    pub eval_projection: Duration,
    pub combined_sumcheck: Duration,
    pub multipoint_eval: Duration,
    pub lift_and_project: Duration,
    pub pcs_open: Duration,
    pub proof_assembly: Duration,
}

impl ProverStepTimings {
    pub fn as_labeled_slice(&self) -> Vec<(&'static str, Duration)> {
        vec![
            ("0: Commit", self.commit),
            ("1: Prime projection", self.prime_projection),
            ("2: Ideal check", self.ideal_check),
            ("3: Eval projection", self.eval_projection),
            ("4: Combined sumcheck", self.combined_sumcheck),
            ("5: Multi-point eval", self.multipoint_eval),
            ("6: Lift-and-project", self.lift_and_project),
            ("7: PCS open", self.pcs_open),
            ("8: Proof assembly", self.proof_assembly),
        ]
    }
}

/// Per-step durations recorded by [`ZincPlusPiop::verify_with_timings`].
#[derive(Clone, Debug)]
pub struct VerifierStepTimings {
    pub transcript_reconstruct: Duration,
    pub prime_projection: Duration,
    pub ideal_check: Duration,
    pub eval_projection: Duration,
    pub sumcheck_verify: Duration,
    pub multipoint_eval: Duration,
    pub lifted_evals: Duration,
    pub pcs_verify: Duration,
}

impl VerifierStepTimings {
    pub fn as_labeled_slice(&self) -> Vec<(&'static str, Duration)> {
        vec![
            ("0: Transcript reconstruct", self.transcript_reconstruct),
            ("1: Prime projection", self.prime_projection),
            ("2: Ideal check", self.ideal_check),
            ("3: Eval projection", self.eval_projection),
            ("4: Sumcheck verify", self.sumcheck_verify),
            ("5: Multi-point eval", self.multipoint_eval),
            ("6: Lifted evals", self.lifted_evals),
            ("7: PCS verify", self.pcs_verify),
        ]
    }
}

// ── Prover ─────────────────────────────────────────────────────────────

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
    /// Same as [`prove`](ZincPlusPiop::prove) but returns per-step timings.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub fn prove_with_timings<const MLE_FIRST: bool, const CHECK_FOR_OVERFLOW: bool>(
        (pp_bin, pp_arb, pp_int): &(
            ZipPlusParams<Zt::BinaryZt, Zt::BinaryLc>,
            ZipPlusParams<Zt::ArbitraryZt, Zt::ArbitraryLc>,
            ZipPlusParams<Zt::IntZt, Zt::IntLc>,
        ),
        trace: &UairTrace<'static, Zt::Int, Zt::Int, D>,
        num_vars: usize,
        project_scalar: impl Fn(&U::Scalar, &F::Config) -> DynamicPolynomialF<F>,
    ) -> Result<(Proof<F>, ProverStepTimings), ProtocolError<F, U::Ideal>> {
        let sig = U::signature();
        let public_trace = trace.public(&sig);
        let witness_trace = trace.witness(&sig);

        // === Step 0: Commit ===
        let t = Instant::now();

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

        let commit = t.elapsed();

        // === Step 1: Prime projection ===
        let t = Instant::now();

        let field_cfg = pcs_transcript
            .fs_transcript
            .get_random_field_cfg::<F, Zt::Fmod, Zt::PrimeTest>();

        let projected_scalars_fx = project_scalars::<F, U>(|s| project_scalar(s, &field_cfg));
        let num_constraints = count_constraints::<U>();

        let projected_trace = if MLE_FIRST {
            ProjectedTrace::ColumnMajor(project_trace_coeffs_column_major(trace, &field_cfg))
        } else {
            ProjectedTrace::RowMajor(project_trace_coeffs_row_major(trace, &field_cfg))
        };

        let prime_projection = t.elapsed();

        // === Step 2: Ideal check ===
        let t = Instant::now();

        let (ic_proof, ic_prover_state) = match &projected_trace {
            ProjectedTrace::ColumnMajor(t) => U::prove_linear(
                &mut pcs_transcript.fs_transcript,
                t,
                &projected_scalars_fx,
                num_constraints,
                num_vars,
                &field_cfg,
            ),
            ProjectedTrace::RowMajor(t) => U::prove_combined(
                &mut pcs_transcript.fs_transcript,
                t,
                &projected_scalars_fx,
                num_constraints,
                num_vars,
                &field_cfg,
            ),
        }?;

        let ideal_check = t.elapsed();

        // === Step 3: Eval projection ===
        let t = Instant::now();

        let projecting_element: Zt::Chal = pcs_transcript.fs_transcript.get_challenge();
        let projecting_element_f: F = F::from_with_cfg(&projecting_element, &field_cfg);

        let projected_trace_f =
            evaluate_trace_to_column_mles(&projected_trace, &projecting_element_f);

        let projected_scalars_f =
            project_scalars_to_field(projected_scalars_fx, &projecting_element_f)
                .map_err(|(_s, _f, e)| ProtocolError::ScalarProjection(e))?;

        let max_degree = count_max_degree::<U>();

        let eval_projection = t.elapsed();

        // === Step 4: Combined sumcheck ===
        let t = Instant::now();

        let (cpr_group, cpr_ancillary) = CombinedPolyResolver::prepare_sumcheck_group::<U>(
            &mut pcs_transcript.fs_transcript,
            projected_trace_f.clone(),
            &ic_prover_state.evaluation_point,
            &projected_scalars_f,
            num_constraints,
            num_vars,
            max_degree,
            &field_cfg,
        )?;

        let lookup_specs = U::signature();
        let groups = vec![cpr_group];
        let _ = lookup_specs;

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

        let lookup_proof = None;

        let combined_sumcheck_time = t.elapsed();

        // === Step 5: Multi-point eval ===
        let t = Instant::now();

        let uair_sig = U::signature();
        let (mp_proof, mp_prover_state) = MultipointEval::prove_as_subprotocol(
            &mut pcs_transcript.fs_transcript,
            &projected_trace_f,
            &cpr_prover_state.evaluation_point,
            &cpr_proof.up_evals,
            &cpr_proof.down_evals,
            uair_sig.shifts(),
            &field_cfg,
        )?;

        let multipoint_eval = t.elapsed();

        // === Step 6: Lift-and-project ===
        let t = Instant::now();

        let r_0 = &mp_prover_state.eval_point;

        let lifted_evals =
            compute_lifted_evals::<F, D>(r_0, &trace.binary_poly, &projected_trace, &field_cfg);

        let mut transcription_buf: Vec<u8> = vec![0; F::Inner::NUM_BYTES];
        for bar_u in &lifted_evals {
            pcs_transcript
                .fs_transcript
                .absorb_random_field_slice(&bar_u.coeffs, &mut transcription_buf);
        }

        let lift_and_project = t.elapsed();

        // === Step 7: PCS open ===
        let t = Instant::now();

        if let Some(hint_bin) = &hint_bin {
            let _ = ZipPlus::<Zt::BinaryZt, Zt::BinaryLc>::prove_f::<_, CHECK_FOR_OVERFLOW>(
                &mut pcs_transcript,
                pp_bin,
                &witness_trace.binary_poly,
                r_0,
                hint_bin,
                &field_cfg,
            )?;
        }
        if let Some(hint_arb) = &hint_arb {
            let _ = ZipPlus::<Zt::ArbitraryZt, Zt::ArbitraryLc>::prove_f::<_, CHECK_FOR_OVERFLOW>(
                &mut pcs_transcript,
                pp_arb,
                &witness_trace.arbitrary_poly,
                r_0,
                hint_arb,
                &field_cfg,
            )?;
        }
        if let Some(hint_int) = &hint_int {
            let _ = ZipPlus::<Zt::IntZt, Zt::IntLc>::prove_f::<_, CHECK_FOR_OVERFLOW>(
                &mut pcs_transcript,
                pp_int,
                &witness_trace.int,
                r_0,
                hint_int,
                &field_cfg,
            )?;
        }

        let pcs_open = t.elapsed();

        // === Proof assembly ===
        let t = Instant::now();

        let zip_proof = pcs_transcript.stream.into_inner();
        let commitments = (commitment_bin, commitment_arb, commitment_int);

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

        let proof = Proof {
            commitments,
            ideal_check: ic_proof,
            resolver: cpr_proof,
            combined_sumcheck,
            multipoint_eval: mp_proof,
            zip: zip_proof,
            witness_lifted_evals,
            lookup_proof,
        };

        let proof_assembly = t.elapsed();

        let timings = ProverStepTimings {
            commit,
            prime_projection,
            ideal_check,
            eval_projection,
            combined_sumcheck: combined_sumcheck_time,
            multipoint_eval,
            lift_and_project,
            pcs_open,
            proof_assembly,
        };

        Ok((proof, timings))
    }
}

// ── Verifier ───────────────────────────────────────────────────────────

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
    /// Same as [`verify`](ZincPlusPiop::verify) but returns per-step timings.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub fn verify_with_timings<IdealOverF, const CHECK_FOR_OVERFLOW: bool>(
        (vp_bin, vp_arb, vp_int): &(
            ZipPlusParams<Zt::BinaryZt, Zt::BinaryLc>,
            ZipPlusParams<Zt::ArbitraryZt, Zt::ArbitraryLc>,
            ZipPlusParams<Zt::IntZt, Zt::IntLc>,
        ),
        proof: Proof<F>,
        public_trace: &UairTrace<Zt::Int, Zt::Int, D>,
        num_vars: usize,
        project_scalar: impl Fn(&U::Scalar, &F::Config) -> DynamicPolynomialF<F>,
        project_ideal: impl Fn(&IdealOrZero<U::Ideal>, &F::Config) -> IdealOverF,
    ) -> Result<VerifierStepTimings, ProtocolError<F, IdealOverF>>
    where
        IdealOverF: Ideal + IdealCheck<DynamicPolynomialF<F>>,
    {
        // === Step 0: Reconstruct transcript ===
        let t = Instant::now();

        let mut pcs_transcript = PcsVerifierTranscript {
            fs_transcript: Blake3Transcript::default(),
            stream: Cursor::new(proof.zip),
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

        let transcript_reconstruct = t.elapsed();

        // === Step 1: Prime projection ===
        let t = Instant::now();

        let field_cfg = pcs_transcript
            .fs_transcript
            .get_random_field_cfg::<F, Zt::Fmod, Zt::PrimeTest>();

        let num_constraints = count_constraints::<U>();

        let prime_projection = t.elapsed();

        // === Step 2: Ideal check ===
        let t = Instant::now();

        let ic_subclaim = U::verify_as_subprotocol::<_, IdealOverF, _>(
            &mut pcs_transcript.fs_transcript,
            proof.ideal_check,
            num_constraints,
            num_vars,
            |ideal| project_ideal(ideal, &field_cfg),
            &field_cfg,
        )?;

        let ideal_check = t.elapsed();

        // === Step 3: Eval projection ===
        let t = Instant::now();

        let projecting_element: Zt::Chal = pcs_transcript.fs_transcript.get_challenge();
        let projecting_element_f: F = F::from_with_cfg(&projecting_element, &field_cfg);

        let projected_scalars_fx = project_scalars::<F, U>(|s| project_scalar(s, &field_cfg));
        let projected_scalars_f =
            project_scalars_to_field(projected_scalars_fx, &projecting_element_f)
                .map_err(|(_s, _f, e)| ProtocolError::ScalarProjection(e))?;

        let eval_projection = t.elapsed();

        // === Step 4: Sumcheck verify ===
        let t = Instant::now();

        let cpr_verifier_ancillary = CombinedPolyResolver::prepare_verifier::<U>(
            &mut pcs_transcript.fs_transcript,
            &proof.resolver,
            proof.combined_sumcheck.claimed_sums()[0].clone(),
            &ic_subclaim,
            num_constraints,
            num_vars,
            &projecting_element_f,
            &field_cfg,
        )?;

        let md_subclaims = MultiDegreeSumcheck::verify_as_subprotocol(
            &mut pcs_transcript.fs_transcript,
            num_vars,
            &proof.combined_sumcheck,
            &field_cfg,
        )
        .map_err(CombinedPolyResolverError::SumcheckError)?;

        let cpr_subclaim = CombinedPolyResolver::finalize_verifier::<U>(
            &mut pcs_transcript.fs_transcript,
            proof.resolver,
            md_subclaims.point().to_vec(),
            md_subclaims.expected_evaluations()[0].clone(),
            cpr_verifier_ancillary,
            &projected_scalars_f,
            &field_cfg,
        )?;

        let _ = &proof.lookup_proof;

        let sumcheck_verify = t.elapsed();

        // === Step 5: Multi-point eval ===
        let t = Instant::now();

        let uair_sig = U::signature();
        let mp_subclaim = MultipointEval::verify_as_subprotocol(
            &mut pcs_transcript.fs_transcript,
            proof.multipoint_eval,
            &cpr_subclaim.evaluation_point,
            &cpr_subclaim.up_evals,
            &cpr_subclaim.down_evals,
            uair_sig.shifts(),
            num_vars,
            &field_cfg,
        )?;

        let multipoint_eval = t.elapsed();

        // === Step 6: Lifted evals ===
        let t = Instant::now();

        let r_0 = &mp_subclaim.sumcheck_subclaim.point;

        let pub_cols = uair_sig.public_cols();
        let num_pub_bin = pub_cols.num_binary_poly_cols();
        let num_pub_arb = pub_cols.num_arbitrary_poly_cols();
        let num_pub_int = pub_cols.num_int_cols();

        let wit_cols = uair_sig.witness_cols();
        let num_wit_bin = wit_cols.num_binary_poly_cols();
        let num_wit_arb = wit_cols.num_arbitrary_poly_cols();

        let public_lifted = if add!(add!(num_pub_bin, num_pub_arb), num_pub_int) > 0 {
            let projected_public =
                project_trace_coeffs_row_major::<F, Zt::Int, Zt::Int, D>(public_trace, &field_cfg);
            compute_lifted_evals::<F, D>(
                r_0,
                &public_trace.binary_poly,
                &ProjectedTrace::RowMajor(projected_public),
                &field_cfg,
            )
        } else {
            Vec::new()
        };

        let all_lifted_evals: Vec<_> = public_lifted[..num_pub_bin]
            .iter()
            .chain(&proof.witness_lifted_evals[..num_wit_bin])
            .chain(&public_lifted[num_pub_bin..add!(num_pub_bin, num_pub_arb)])
            .chain(&proof.witness_lifted_evals[num_wit_bin..add!(num_wit_bin, num_wit_arb)])
            .chain(&public_lifted[add!(num_pub_bin, num_pub_arb)..])
            .chain(&proof.witness_lifted_evals[add!(num_wit_bin, num_wit_arb)..])
            .cloned()
            .collect();

        let open_evals: Vec<F> = all_lifted_evals
            .iter()
            .map(|bar_u| bar_u.evaluate_at_point(&projecting_element_f))
            .collect::<Result<Vec<_>, _>>()
            .map_err(ProtocolError::LiftedEvalProjection)?;

        MultipointEval::verify_subclaim(&mp_subclaim, &open_evals, uair_sig.shifts(), &field_cfg)?;

        let mut transcription_buf: Vec<u8> = vec![0; F::Inner::NUM_BYTES];
        for bar_u in &all_lifted_evals {
            pcs_transcript
                .fs_transcript
                .absorb_random_field_slice(&bar_u.coeffs, &mut transcription_buf);
        }

        let lifted_evals = t.elapsed();

        // === Step 7: PCS verify ===
        let t = Instant::now();

        macro_rules! verify_pcs_batch {
            ($Zt:ty, $Lc:ty, $vp:expr, $idx:tt, [$evals_range:expr]) => {{
                let comm = &proof.commitments.$idx;
                if comm.batch_size > 0 {
                    let per_poly_alphas = ZipPlus::<$Zt, $Lc>::sample_alphas(
                        &mut pcs_transcript.fs_transcript,
                        comm.batch_size,
                    );
                    let mut eval_f = F::zero_with_cfg(&field_cfg);
                    for (bar_u, alphas) in all_lifted_evals[$evals_range]
                        .iter()
                        .zip(per_poly_alphas.iter())
                    {
                        for (coeff, alpha) in bar_u.coeffs.iter().zip(alphas.iter()) {
                            let mut term = F::from_with_cfg(alpha, &field_cfg);
                            term *= coeff;
                            eval_f += &term;
                        }
                    }
                    ZipPlus::<$Zt, $Lc>::verify_with_alphas::<F, CHECK_FOR_OVERFLOW>(
                        &mut pcs_transcript,
                        $vp,
                        comm,
                        &field_cfg,
                        r_0,
                        &eval_f,
                        &per_poly_alphas,
                    )
                    .map_err(|e| ProtocolError::PcsVerification($idx, e))?;
                }
            }};
        }

        let total = uair_sig.total_cols();
        let num_total_bin = total.num_binary_poly_cols();
        let num_total_arb = total.num_arbitrary_poly_cols();
        verify_pcs_batch!(
            Zt::BinaryZt,
            Zt::BinaryLc,
            vp_bin,
            0,
            [num_pub_bin..num_total_bin]
        );
        verify_pcs_batch!(
            Zt::ArbitraryZt,
            Zt::ArbitraryLc,
            vp_arb,
            1,
            [add!(num_total_bin, num_pub_arb)..add!(num_total_bin, num_total_arb)]
        );
        verify_pcs_batch!(
            Zt::IntZt,
            Zt::IntLc,
            vp_int,
            2,
            [add!(add!(num_total_bin, num_total_arb), num_pub_int)..]
        );

        let pcs_verify = t.elapsed();

        Ok(VerifierStepTimings {
            transcript_reconstruct,
            prime_projection,
            ideal_check,
            eval_projection,
            sumcheck_verify,
            multipoint_eval,
            lifted_evals,
            pcs_verify,
        })
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
