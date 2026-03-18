use super::*;
use crypto_primitives::{ConstIntSemiring, FromPrimitiveWithConfig, FromWithConfig};
use num_traits::Zero;
use std::io::Cursor;
use zinc_piop::{
    combined_poly_resolver::CombinedPolyResolver,
    ideal_check::IdealCheckProtocol,
    multipoint_eval::MultipointEval,
    projections::{
        ProjectedTrace, project_scalars, project_scalars_to_field, project_trace_coeffs_row_major,
    },
};
use zinc_poly::{EvaluatablePolynomial, univariate::dynamic::over_field::DynamicPolynomialF};
use zinc_transcript::{
    KeccakTranscript,
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
    /// Zinc+ full PIOP verifier.
    ///
    /// `up_evals` and `down_evals` from the F_q sumcheck (Step 4) are reduced
    /// via the multi-point evaluation sumcheck (Step 5) to a single evaluation
    /// point `r_0`. The verifier recomputes public `lifted_evals` from public
    /// data at `r_0`, interleaves them with the witness `lifted_evals` from
    /// the proof, derives scalar `open_evals` via `\psi_a`, and checks the
    /// multipoint eval consistency. A Zip+ PCS invocation (Step 7) confirms
    /// the witness `lifted_evals`.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub fn verify<IdealOverF, const CHECK_FOR_OVERFLOW: bool>(
        (vp_bin, vp_arb, vp_int): &(
            ZipPlusParams<Zt::BinaryZt, Zt::BinaryLc>,
            ZipPlusParams<Zt::ArbitraryZt, Zt::ArbitraryLc>,
            ZipPlusParams<Zt::IntZt, Zt::IntLc>,
        ),
        proof: Proof<F>,
        public_trace: &UairTrace<Zt::Int, Zt::Int, D>,
        num_vars: usize,
        project_scalar: impl Fn(&U::Scalar, &F::Config) -> DynamicPolynomialF<F> + Sync,
        project_ideal: impl Fn(&IdealOrZero<U::Ideal>, &F::Config) -> IdealOverF,
    ) -> Result<(), ProtocolError<F, IdealOverF>>
    where
        IdealOverF: Ideal + IdealCheck<DynamicPolynomialF<F>>,
    {
        // === Step 0: Reconstruct transcript from commitments + public data ===
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

        absorb_public_columns(&mut pcs_transcript.fs_transcript, &public_trace.binary_poly);
        absorb_public_columns(
            &mut pcs_transcript.fs_transcript,
            &public_trace.arbitrary_poly,
        );
        absorb_public_columns(&mut pcs_transcript.fs_transcript, &public_trace.int);

        // === Step 1: Prime projection ===
        let field_cfg = pcs_transcript
            .fs_transcript
            .get_random_field_cfg::<F, Zt::Fmod, Zt::PrimeTest>();

        let num_constraints = count_constraints::<U>();

        // === Step 2: Ideal check ===
        let ic_subclaim = U::verify_as_subprotocol::<_, IdealOverF, _>(
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
        let mp_subclaim = MultipointEval::verify_as_subprotocol(
            &mut pcs_transcript.fs_transcript,
            proof.multipoint_eval,
            &cpr_subclaim.evaluation_point,
            &cpr_subclaim.up_evals,
            &cpr_subclaim.down_evals,
            num_vars,
            &field_cfg,
        )?;

        let r_0 = &mp_subclaim.sumcheck_subclaim.point;

        // === Step 6: Recompute public lifted_evals, assemble full set ===
        //
        // The proof carries only witness lifted_evals. The verifier
        // independently computes public lifted_evals from public data at r_0,
        // then interleaves them with the witness portion to reconstruct the
        // full lifted_evals in canonical order:
        //   [pub_bin, wit_bin, pub_arb, wit_arb, pub_int, wit_int]
        let sig = U::signature();
        let num_pub_bin = sig.public_binary_poly_cols;
        let num_pub_arb = sig.public_arbitrary_poly_cols;
        let num_pub_int = sig.public_int_cols;

        let (num_wit_bin, num_wit_arb) = (
            sig.witness_binary_poly_cols,
            sig.witness_arbitrary_poly_cols,
        );

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

        // Derive scalar open_evals via \psi_a and finalize the multipoint
        // eval consistency check.
        let open_evals: Vec<F> = all_lifted_evals
            .iter()
            .map(|bar_u| bar_u.evaluate_at_point(&projecting_element_f))
            .collect::<Result<Vec<_>, _>>()
            .map_err(ProtocolError::LiftedEvalProjection)?;

        MultipointEval::<F>::verify_subclaim(&mp_subclaim, &open_evals, &field_cfg)?;

        // Absorb all lifted_evals into transcript (same order as prover).
        let mut transcription_buf: Vec<u8> = vec![0; F::Inner::NUM_BYTES];
        for bar_u in &all_lifted_evals {
            pcs_transcript
                .fs_transcript
                .absorb_random_field_slice(&bar_u.coeffs, &mut transcription_buf);
        }

        // === Step 7: PCS verify at r_0 (witness columns only) ===

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

        let num_total_bin = sig.total_binary_poly_cols();
        let num_total_arb = sig.total_arbitrary_poly_cols();
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

        Ok(())
    }
}
