use super::*;
use crypto_primitives::{ConstIntSemiring, FromPrimitiveWithConfig, FromWithConfig};
use num_traits::Zero;
use std::io::Cursor;
use zinc_piop::{
    combined_poly_resolver::CombinedPolyResolver,
    ideal_check::IdealCheckProtocol,
    projections::{project_scalars, project_scalars_to_field},
};
use zinc_poly::{
    mle::{DenseMultilinearExtension, MultilinearExtensionWithConfig},
    univariate::dynamic::over_field::DynamicPolynomialF,
};
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
    /// Verifies all steps and returns a [`Subclaim`]. The `up_evals` are
    /// already verified by the Zip+ PCS (Step 5); the `down_evals` (shifted
    /// MLE claims) still need to be checked via [`resolve_subclaim`].
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
    ) -> Result<Subclaim<F>, ProtocolError<F, IdealOverF>>
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

        // === Step 5: PCS verify (check witness MLE evaluation claims) ====
        // After the sumcheck, the verifier uses the Zip+ PCS to confirm
        // that the committed witness MLEs actually evaluate to the claimed
        // up_evals at the sumcheck challenge point.
        //
        // TODO: Once we add public inputs, compute public input MLE evaluations
        //       at cpr_subclaim.evaluation_point directly from public data here,
        //       then include them in the constraint recomputation check.

        if proof.commitments.0.batch_size > 0 {
            ZipPlus::<Zt::BinaryZt, Zt::BinaryLc>::verify::<F, CHECK_FOR_OVERFLOW>(
                &mut pcs_transcript,
                vp_bin,
                &proof.commitments.0,
                &field_cfg,
                &cpr_subclaim.evaluation_point,
                &cpr_subclaim.up_evals[0],
            )
            .map_err(|e| ProtocolError::PcsVerification(0, e))?;
        }

        if proof.commitments.1.batch_size > 0 {
            ZipPlus::<Zt::ArbitraryZt, Zt::ArbitraryLc>::verify::<F, CHECK_FOR_OVERFLOW>(
                &mut pcs_transcript,
                vp_arb,
                &proof.commitments.1,
                &field_cfg,
                &cpr_subclaim.evaluation_point,
                &cpr_subclaim.up_evals[1],
            )
            .map_err(|e| ProtocolError::PcsVerification(1, e))?;
        }

        if proof.commitments.2.batch_size > 0 {
            ZipPlus::<Zt::IntZt, Zt::IntLc>::verify::<F, CHECK_FOR_OVERFLOW>(
                &mut pcs_transcript,
                vp_int,
                &proof.commitments.2,
                &field_cfg,
                &cpr_subclaim.evaluation_point,
                &cpr_subclaim.up_evals[2],
            )
            .map_err(|e| ProtocolError::PcsVerification(2, e))?;
        }

        Ok(Subclaim {
            evaluation_point: cpr_subclaim.evaluation_point,
            up_evals: cpr_subclaim.up_evals,
            down_evals: cpr_subclaim.down_evals,
        })
    }

    /// Subclaim resolution (shifted-MLE evaluation check)
    ///
    /// Verify the "down" (next-row) MLE evaluation claims from the subclaim.
    ///
    /// The "up" evaluations are already verified by the Zip+ PCS inside
    /// [`verify`] (Step 5). The "down" evaluations correspond to the shifted
    /// trace MLE (rows 1..n, zero-padded), which the PCS does not yet open.
    /// Until the PCS is extended to also open shifted MLEs, this function
    /// checks them directly against the prover's auxiliary projected trace.
    pub fn resolve_subclaim(
        subclaim: &Subclaim<F>,
        projected_trace_f: &[DenseMultilinearExtension<F::Inner>],
        field_cfg: &F::Config,
    ) -> Result<(), ProtocolError<F, U::Ideal>>
    where
        DenseMultilinearExtension<F::Inner>: MultilinearExtensionWithConfig<F>,
    {
        let num_cols = projected_trace_f.len();

        // Check "down" evaluations (shifted/next-row columns).
        // The shifted trace drops the first row and zero-pads, matching
        // the CombinedPolyResolver's convention.
        for (i, (mle, expected)) in projected_trace_f
            .iter()
            .zip(subclaim.down_evals.iter())
            .enumerate()
        {
            let shifted: DenseMultilinearExtension<F::Inner> =
                mle.iter().skip(1).cloned().collect();

            let actual = shifted
                .evaluate_with_config(&subclaim.evaluation_point, field_cfg)
                .map_err(ProtocolError::MleEvaluation)?;

            if actual != *expected {
                return Err(ProtocolError::SubclaimMismatch {
                    column: add!(num_cols, i),
                    expected: expected.clone(),
                    actual,
                });
            }
        }

        Ok(())
    }
}
