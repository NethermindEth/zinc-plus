use super::*;
use crypto_primitives::{ConstIntSemiring, FromPrimitiveWithConfig, FromWithConfig};
use num_traits::Zero;
use std::io::Cursor;
use zinc_piop::{
    combined_poly_resolver::CombinedPolyResolver,
    ideal_check::IdealCheckProtocol,
    multipoint_eval::MultipointEval,
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
    /// Zinc+ full PIOP verifier.
    ///
    /// `up_evals` and `down_evals` from the F_q sumcheck (Step 4) are reduced
    /// via the multi-point evaluation sumcheck (Step 5) to a single evaluation
    /// point `r_0`. The scalar `open_evals` at `r_0` are derived from the
    /// polynomial-valued `lifted_evals` via `\psi_a`, then used
    /// to check the sumcheck consistency. A single Zip+ PCS invocation
    /// (Step 6) confirms the `lifted_evals`.
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

        // === Step 2: Ideal check ===
        let ic_subclaim = U::verify_as_subprotocol::<F, IdealOverF, _>(
            &mut pcs_transcript.fs_transcript,
            proof.ideal_check,
            num_constraints,
            num_vars,
            |ideal| project_ideal(ideal, &field_cfg),
            &field_cfg,
        )?;

        // === Step 3: Evaluation projection (\psi_a) ===
        let projecting_element: Zt::Chal = pcs_transcript.fs_transcript.get_challenge();
        let projecting_element_f: F = F::from_with_cfg(&projecting_element, &field_cfg);

        let projected_scalars_fx = project_scalars::<F, U>(|s| project_scalar(s, &field_cfg));
        let projected_scalars_f =
            project_scalars_to_field(projected_scalars_fx, &projecting_element_f)
                .map_err(|(_s, _f, e)| ProtocolError::ScalarProjection(e))?;

        let max_degree = count_max_degree::<U>();

        // === Step 4: Sumcheck over F_q ===
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

        // === Step 5: Multi-point evaluation sumcheck ===
        // Derive scalar open_evals from lifted_evals via \psi_a, then pass
        // them to the multipoint eval verifier for the consistency check.
        let open_evals: Vec<F> = proof
            .lifted_evals
            .iter()
            .map(|bar_u| bar_u.evaluate_at_point(&projecting_element_f))
            .collect::<Result<Vec<_>, _>>()
            .map_err(ProtocolError::LiftedEvalProjection)?;

        let mp_subclaim = MultipointEval::verify_as_subprotocol(
            &mut pcs_transcript.fs_transcript,
            proof.multipoint_eval,
            &cpr_subclaim.evaluation_point,
            &cpr_subclaim.up_evals,
            &cpr_subclaim.down_evals,
            &open_evals,
            num_vars,
            &field_cfg,
        )?;

        // Absorb lifted_evals into transcript
        let mut transcription_buf: Vec<u8> = vec![0; F::Inner::NUM_BYTES];
        for bar_u in &proof.lifted_evals {
            pcs_transcript
                .fs_transcript
                .absorb_random_field_slice(&bar_u.coeffs, &mut transcription_buf);
        }

        // === Step 6: PCS verify at r_0 ===
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
                        &mp_subclaim.eval_point,
                        &eval_f,
                        &per_poly_alphas,
                    )
                    .map_err(|e| ProtocolError::PcsVerification($idx, e))?;
                }
            }};
        }

        let sig = U::signature();
        let (n_bin, n_arb, _n_int) = (sig.binary_poly_cols, sig.arbitrary_poly_cols, sig.int_cols);
        verify_pcs_batch!(Zt::BinaryZt, Zt::BinaryLc, vp_bin, 0, [..n_bin]);
        verify_pcs_batch!(
            Zt::ArbitraryZt,
            Zt::ArbitraryLc,
            vp_arb,
            1,
            [n_bin..add!(n_bin, n_arb)]
        );
        verify_pcs_batch!(Zt::IntZt, Zt::IntLc, vp_int, 2, [add!(n_bin, n_arb)..]);

        Ok(())
    }
}
