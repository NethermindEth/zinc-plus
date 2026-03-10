use super::*;
use crypto_primitives::{ConstIntSemiring, FromPrimitiveWithConfig, FromWithConfig};
use num_traits::Zero;
use std::io::Cursor;
use zinc_piop::{
    batched_shift::BatchedShift,
    combined_poly_resolver::CombinedPolyResolver,
    ideal_check::IdealCheckProtocol,
    projections::{project_scalars, project_scalars_to_field},
};
use zinc_poly::{EvaluatablePolynomial, univariate::dynamic::over_field::DynamicPolynomialF};
use zinc_transcript::{
    KeccakTranscript,
    traits::{ConstTranscribable, Transcript},
};
use zinc_uair::{
    Uair,
    constraint_counter::count_constraints,
    degree_counter::count_max_degree,
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
    /// Zinc+ full PIOP Verifier.
    ///
    /// Verifies all steps. Both `up_evals` and `down_evals` (shifted MLE
    /// claims) are fully verified: `up_evals` by the Zip+ PCS (Step 5a),
    /// `down_evals` via the batched shift protocol (Step 4.5) and PCS
    /// evaluate-only (Step 5b).
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub fn verify<IdealOverF, const CHECK_FOR_OVERFLOW: bool>(
        (vp_bin, vp_arb, vp_int): &(
            ZipPlusParams<Zt::BinaryZt, Zt::BinaryLc>,
            ZipPlusParams<Zt::ArbitraryZt, Zt::ArbitraryLc>,
            ZipPlusParams<Zt::IntZt, Zt::IntLc>,
        ),
        proof: Proof<F>,
        num_vars: usize,
        project_scalar: impl Fn(&U::Scalar, &F::Config) -> DynamicPolynomialF<F>,
        project_ideal: impl Fn(&IdealOrZero<U::Ideal>, &F::Config) -> IdealOverF,
    ) -> Result<(), ProtocolError<F, IdealOverF>>
    where
        IdealOverF: Ideal + IdealCheck<DynamicPolynomialF<F>>,
    {
        // === Step 0: Reconstruct transcript from commitments ===
        // The verifier creates a PcsVerifierTranscript from the PCS proof bytes.
        let mut pcs_transcript = PcsVerifierTranscript {
            fs_transcript: KeccakTranscript::default(),
            stream: Cursor::new(proof.zip),
        };
        for comm in [
            &proof.commitments.0,
            &proof.commitments.1,
            &proof.commitments.2,
        ] {
            pcs_transcript.fs_transcript.absorb_slice(&comm.root);
        }

        // === Step 1: Prime projection ===
        let field_cfg = pcs_transcript
            .fs_transcript
            .get_random_field_cfg::<F, Zt::Fmod, Zt::PrimeTest>();

        let num_constraints = count_constraints::<U>();

        // === Step 2: Verify ideal check ===
        let ic_subclaim = IdealCheckProtocol::verify_as_subprotocol::<U, IdealOverF, _>(
            &mut pcs_transcript.fs_transcript,
            proof.ideal_check,
            num_constraints,
            num_vars,
            |ideal| project_ideal(ideal, &field_cfg),
            &field_cfg,
        )?;

        // === Step 3: Evaluation projection ===
        // Sample projecting element as Zt::Chal (matching the prover).
        let projecting_element: Zt::Chal = pcs_transcript.fs_transcript.get_challenge();
        let projecting_element_f: F = F::from_with_cfg(&projecting_element, &field_cfg);

        // Verifier independently computes projected scalars (public data).
        let projected_scalars_fx = project_scalars::<F, U>(|s| project_scalar(s, &field_cfg));
        let projected_scalars_f =
            project_scalars_to_field(projected_scalars_fx, &projecting_element_f)
                .map_err(|(_s, _f, e)| ProtocolError::ScalarProjection(e))?;

        let max_degree = count_max_degree::<U>();

        // === Step 4: Verify finite-field PIOP ===
        let cpr_subclaim = CombinedPolyResolver::verify_as_subprotocol::<U>(
            &mut pcs_transcript.fs_transcript,
            proof.resolver,
            num_constraints,
            num_vars,
            max_degree,
            &projecting_element_f,
            &projected_scalars_f,
            ic_subclaim,
            &field_cfg,
        )?;

        // === Step 4.5: Lift-and-project verification ===
        // Absorb the prover's lifted_evals into the transcript (matching
        // prover Step 4.5). Then check ψ_a consistency: evaluating each
        // lifted_eval_j(X) at the projecting element must recover up_eval_j.
        let mut transcription_buf: Vec<u8> = vec![0; F::Inner::NUM_BYTES];
        for bar_u in &proof.lifted_evals {
            pcs_transcript
                .fs_transcript
                .absorb_random_field_slice(&bar_u.coeffs, &mut transcription_buf);
        }

        for (j, (bar_u, up_eval)) in proof
            .lifted_evals
            .iter()
            .zip(cpr_subclaim.up_evals.iter())
            .enumerate()
        {
            let psi_a_val = bar_u
                .evaluate_at_point(&projecting_element_f)
                .map_err(ProtocolError::LiftedEvalProjection)?;
            if psi_a_val != *up_eval {
                return Err(ProtocolError::LiftedEvalMismatch {
                    column: j,
                    expected: up_eval.clone(),
                    actual: psi_a_val,
                });
            }
        }

        // === Step 4.5: Verify batched shift ===
        let bs_subclaim = BatchedShift::verify_as_subprotocol(
            &mut pcs_transcript.fs_transcript,
            proof.batched_shift,
            &cpr_subclaim.evaluation_point,
            &cpr_subclaim.down_evals,
            num_vars,
            &field_cfg,
        )?;

        // === Step 5a: PCS verify (check witness MLE evaluation claims at r') ====
        // After the sumcheck, the verifier uses the Zip+ PCS to confirm
        // that the committed witness MLEs actually evaluate to the claimed
        // up_evals at the sumcheck challenge point.
        //
        // TODO: Once we add public inputs, compute public input MLE evaluations
        //       at cpr_subclaim.evaluation_point directly from public data here,
        //       then include them in the constraint recomputation check.

        macro_rules! verify_pcs_batch {
            ($Zt:ty, $Lc:ty, $vp:expr, $idx:tt, [$evals_range:expr]) => {{
                let comm = &proof.commitments.$idx;
                if comm.batch_size > 0 {
                    let per_poly_alphas = ZipPlus::<$Zt, $Lc>::sample_alphas(
                        &mut pcs_transcript.fs_transcript,
                        comm.batch_size,
                    );
                    let mut eval_f = F::zero_with_cfg(&field_cfg);
                    for (bar_u, alphas) in proof.lifted_evals[$evals_range]
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
                        &cpr_subclaim.evaluation_point,
                        &eval_f,
                        &per_poly_alphas,
                    )
                    .map_err(|e| ProtocolError::PcsVerification($idx, e))?;
                }
            }};
        }

        let (n_bin, n_arb, _n_int) = proof.num_witness_cols;
        verify_pcs_batch!(Zt::BinaryZt, Zt::BinaryLc, vp_bin, 0, [..n_bin]);
        verify_pcs_batch!(
            Zt::ArbitraryZt,
            Zt::ArbitraryLc,
            vp_arb,
            1,
            [n_bin..add!(n_bin, n_arb)]
        );
        verify_pcs_batch!(Zt::IntZt, Zt::IntLc, vp_int, 2, [add!(n_bin, n_arb)..]);

        // === Step 5b: PCS verify evaluate-only at shift point ρ ===
        // Absorb lifted_evals_shift (matching prover Step 6b).
        for bar_u in &proof.lifted_evals_shift {
            pcs_transcript
                .fs_transcript
                .absorb_random_field_slice(&bar_u.coeffs, &mut transcription_buf);
        }

        // Check ψ_a consistency: ψ_a(lifted_evals_shift_j) == shift_eval_j.
        for (j, (bar_u, shift_eval)) in proof
            .lifted_evals_shift
            .iter()
            .zip(bs_subclaim.shift_evals.iter())
            .enumerate()
        {
            let psi_a_val = bar_u
                .evaluate_at_point(&projecting_element_f)
                .map_err(ProtocolError::ShiftLiftedEvalProjection)?;
            if psi_a_val != *shift_eval {
                return Err(ProtocolError::ShiftLiftedEvalMismatch {
                    column: j,
                    expected: shift_eval.clone(),
                    actual: psi_a_val,
                });
            }
        }

        // Verify evaluate-only for each committed batch at ρ.
        macro_rules! verify_pcs_eval_only_batch {
            ($Zt:ty, $Lc:ty, $vp:expr, $idx:tt, [$evals_range:expr]) => {{
                let comm = &proof.commitments.$idx;
                if comm.batch_size > 0 {
                    let per_poly_alphas = ZipPlus::<$Zt, $Lc>::sample_alphas(
                        &mut pcs_transcript.fs_transcript,
                        comm.batch_size,
                    );
                    let mut eval_f = F::zero_with_cfg(&field_cfg);
                    for (bar_u, alphas) in proof.lifted_evals_shift[$evals_range]
                        .iter()
                        .zip(per_poly_alphas.iter())
                    {
                        for (coeff, alpha) in bar_u.coeffs.iter().zip(alphas.iter()) {
                            let mut term = F::from_with_cfg(alpha, &field_cfg);
                            term *= coeff;
                            eval_f += &term;
                        }
                    }
                    ZipPlus::<$Zt, $Lc>::verify_evaluate_only::<F, CHECK_FOR_OVERFLOW>(
                        &mut pcs_transcript,
                        $vp,
                        &field_cfg,
                        &bs_subclaim.shift_point,
                        &eval_f,
                    )
                    .map_err(|e| ProtocolError::PcsVerification($idx, e))?;
                }
            }};
        }

        verify_pcs_eval_only_batch!(Zt::BinaryZt, Zt::BinaryLc, vp_bin, 0, [..n_bin]);
        verify_pcs_eval_only_batch!(
            Zt::ArbitraryZt,
            Zt::ArbitraryLc,
            vp_arb,
            1,
            [n_bin..add!(n_bin, n_arb)]
        );
        verify_pcs_eval_only_batch!(Zt::IntZt, Zt::IntLc, vp_int, 2, [add!(n_bin, n_arb)..]);

        Ok(())
    }
}
