use super::*;
use crypto_primitives::{ConstIntSemiring, FromPrimitiveWithConfig, FromWithConfig, PrimeField};
use num_traits::Zero;
use zinc_piop::{
    combined_poly_resolver::CombinedPolyResolver,
    ideal_check::IdealCheckProtocol,
    projections::{
        project_scalars, project_scalars_to_field, project_trace_coeffs, project_trace_to_field,
    },
};
use zinc_poly::{
    mle::DenseMultilinearExtension, univariate::dynamic::over_field::DynamicPolynomialF,
};
use zinc_transcript::traits::{ConstTranscribable, Transcribable, Transcript};
use zinc_uair::{
    Uair, constraint_counter::count_constraints, degree_counter::count_max_degree, ideal::Ideal,
};
use zinc_utils::{
    from_ref::FromRef, inner_transparent_field::InnerTransparentField, mul_by_scalar::MulByScalar,
    projectable_to_field::ProjectableToField,
};
use zip_plus::{
    ZipError,
    code::LinearCode,
    pcs::structs::{ZipPlus, ZipPlusCommitment, ZipPlusHint, ZipPlusParams, ZipTypes},
    pcs_transcript::PcsProverTranscript,
};

impl<Zt, U, F, const D: usize> ZincPlusPiop<Zt, U, F, D>
where
    Zt: WitnessZincTypes<D>,
    Zt::Int: ProjectableToField<F>,
    <Zt::ArbitraryZt as ZipTypes>::Eval: ProjectableToField<F>,
    F: InnerTransparentField
        + FromPrimitiveWithConfig
        + FromWithConfig<Zt::Int>
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
    /// Zinc+ PIOP Prover (Algorithm 1 from the paper, Steps 1–5).
    ///
    /// # Protocol flow (paper Section 2.2 "Combining the three steps"):
    ///
    /// 0. **Commit**: commit each witness column via Zip+ PCS, absorb roots.
    /// 1. **Prime projection** (φ_q: Q\[X\] → F_q\[X\]): sample random prime q
    ///    from transcript, project trace and scalars.
    /// 2. **Ideal check**: sample r ∈ F_q^μ, prover sends MLE evaluations,
    ///    verifier checks ideal membership.
    /// 3. **Evaluation projection** (ψ_a: F_q\[X\] → F_q): sample a ∈ F_q,
    ///    evaluate polynomials at X = a.
    /// 4. **Finite-field PIOP**: sumcheck over F_q to prove the projected
    ///    claim.
    /// 5. **PCS open**: Zip+ test + evaluate for each committed column, proving
    ///    witness MLE evaluations at the sumcheck challenge point.
    ///
    /// Returns the proof and auxiliary data (for subclaim resolution without
    /// PCS).
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub fn prove<const CHECK_FOR_OVERFLOW: bool>(
        (pp_bin, pp_arb, pp_int): &(
            ZipPlusParams<Zt::BinaryZt, Zt::BinaryLc>,
            ZipPlusParams<Zt::ArbitraryZt, Zt::ArbitraryLc>,
            ZipPlusParams<Zt::IntZt, Zt::IntLc>,
        ),
        trace_bin_poly: &[DenseMultilinearExtension<<Zt::BinaryZt as ZipTypes>::Eval>],
        trace_arb_poly: &[DenseMultilinearExtension<<Zt::ArbitraryZt as ZipTypes>::Eval>],
        trace_int: &[DenseMultilinearExtension<<Zt::IntZt as ZipTypes>::Eval>],
        num_vars: usize,
        project_scalar: impl Fn(&U::Scalar, &F::Config) -> DynamicPolynomialF<F>,
    ) -> Result<(Proof<F>, ProverAux<F>), ProtocolError<F, U::Ideal>> {
        // === Step 0: Commit to witness traces ===

        // Commit each witness column via Zip+ PCS.
        fn pcs_commit_witness<Zt, Lc>(
            pp: &ZipPlusParams<Zt, Lc>,
            witness: &[DenseMultilinearExtension<Zt::Eval>],
        ) -> Result<(Vec<ZipPlusHint<Zt::Cw>>, Vec<ZipPlusCommitment>), ZipError>
        where
            Zt: ZipTypes,
            Lc: LinearCode<Zt>,
        {
            let mut hints = Vec::with_capacity(witness.len());
            let mut commitments = Vec::with_capacity(witness.len());
            for col in witness {
                let (hint, comm) = ZipPlus::<Zt, Lc>::commit(pp, col)?;
                hints.push(hint);
                commitments.push(comm);
            }
            Ok((hints, commitments))
        }

        let (hints_bin, commitments_bin) = pcs_commit_witness(pp_bin, trace_bin_poly)?;
        let (hints_arb, commitments_arb) = pcs_commit_witness(pp_arb, trace_arb_poly)?;
        let (hints_int, commitments_int) = pcs_commit_witness(pp_int, trace_int)?;

        // Create the main transcript
        let mut pcs_transcript = PcsProverTranscript::new_from_commitments(
            commitments_bin
                .iter()
                .chain(commitments_arb.iter())
                .chain(commitments_int.iter()),
        )?;
        // TODO: Absorb public inputs as well once they are part of the protocol,
        //       or this will open up a soundness vulnerability!

        // === Step 1: Prime projection (φ_q: Q[X] → F_q[X]) ===

        // Sample a random prime modulus from the Fiat-Shamir transcript.
        let field_cfg = pcs_transcript
            .fs_transcript
            .get_random_field_cfg::<F, Zt::Fmod, Zt::PrimeTest>();

        // Project the witness trace from Q[X] to F_q[X].
        let projected_trace = project_trace_coeffs::<F, Zt::Int, Zt::Int, D>(
            trace_bin_poly,
            trace_arb_poly,
            trace_int,
            &field_cfg,
        );

        // Project UAIR scalars from Q[X] to F_q[X].
        let projected_scalars_fx = project_scalars::<F, U>(|s| project_scalar(s, &field_cfg));

        let num_constraints = count_constraints::<U>();

        // === Step 2: Randomized ideal check ===
        let (ic_proof, ic_prover_state) = IdealCheckProtocol::prove_as_subprotocol::<U>(
            &mut pcs_transcript.fs_transcript,
            &projected_trace,
            &projected_scalars_fx,
            num_constraints,
            num_vars,
            &field_cfg,
        )?;

        // === Step 3: Evaluation projection (ψ_a: F_q[X] → F_q) ===
        // Sample the projecting element as Zt::Chal (matching the Zip+ PCS convention),
        // then convert to F for PIOP use.
        let projecting_element: Zt::Chal = pcs_transcript.fs_transcript.get_challenge();
        let projecting_element_f: F = F::from_with_cfg(&projecting_element, &field_cfg);

        // Project trace from F_q[X] to F_q by evaluating each polynomial at X = a.
        let projected_trace_f =
            project_trace_to_field::<F, D>(&[], &projected_trace, &[], &projecting_element_f);

        // Project scalars from F_q[X] to F_q.
        let projected_scalars_f =
            project_scalars_to_field(projected_scalars_fx, &projecting_element_f)
                .map_err(|(_s, _f, e)| ProtocolError::ScalarProjection(e))?;

        let max_degree = count_max_degree::<U>();

        // === Step 4: Finite-field PIOP (sumcheck over F_q) ===
        let (cpr_proof, cpr_prover_state) = CombinedPolyResolver::prove_as_subprotocol::<U>(
            &mut pcs_transcript.fs_transcript,
            projected_trace_f.clone(),
            &ic_prover_state.evaluation_point,
            &projected_scalars_f,
            num_constraints,
            num_vars,
            max_degree,
            &field_cfg,
        )?;

        // === Step 5: PCS open (prove witness MLE evaluations) ===
        // After the sumcheck, the prover must prove that the committed
        // witness MLEs evaluate to the claimed values (up_evals / down_evals)
        // at the sumcheck challenge point r'.
        // This corresponds to Step 12 "Oracle evaluations at sumcheck point"
        // in the AirZinc protocol (rolled_out_argument.tex).
        //
        // TODO: Once we add public inputs, the verifier will compute public
        //       input MLE evaluations at the sumcheck point directly from
        //       public data. The PCS only covers witness columns.
        let eval_point = &cpr_prover_state.evaluation_point;

        pcs_prove_witness_evaluations::<Zt::BinaryZt, Zt::BinaryLc, _, _, CHECK_FOR_OVERFLOW>(
            &mut pcs_transcript,
            pp_bin,
            trace_bin_poly,
            &hints_bin,
            &commitments_bin,
            eval_point,
            &field_cfg,
            &projecting_element,
        )?;
        pcs_prove_witness_evaluations::<Zt::ArbitraryZt, Zt::ArbitraryLc, _, _, CHECK_FOR_OVERFLOW>(
            &mut pcs_transcript,
            pp_arb,
            trace_arb_poly,
            &hints_arb,
            &commitments_arb,
            eval_point,
            &field_cfg,
            &projecting_element,
        )?;
        pcs_prove_witness_evaluations::<Zt::IntZt, Zt::IntLc, _, _, CHECK_FOR_OVERFLOW>(
            &mut pcs_transcript,
            pp_int,
            trace_int,
            &hints_int,
            &commitments_int,
            eval_point,
            &field_cfg,
            &projecting_element,
        )?;

        let zip_proof = pcs_transcript.stream.into_inner();
        let commitments = commitments_bin
            .into_iter()
            .chain(commitments_arb)
            .chain(commitments_int)
            .collect();

        Ok((
            Proof {
                num_witness_cols: (trace_bin_poly.len(), trace_arb_poly.len(), trace_int.len()),
                zip_commitments: commitments,
                ideal_check: ic_proof,
                resolver: cpr_proof,
                zip_proof,
            },
            ProverAux {
                field_cfg,
                projected_trace_f,
            },
        ))
    }
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn pcs_prove_witness_evaluations<Zt, Lc, F, I, const CHECK_FOR_OVERFLOW: bool>(
    pcs_transcript: &mut PcsProverTranscript,
    pp: &ZipPlusParams<Zt, Lc>,
    witness: &[DenseMultilinearExtension<Zt::Eval>],
    hints: &[ZipPlusHint<Zt::Cw>],
    commitments: &[ZipPlusCommitment],
    eval_point: &[F],
    field_cfg: &F::Config,
    projecting_element: &Zt::Chal,
) -> Result<(), ProtocolError<F, I>>
where
    Zt: ZipTypes,
    Zt::Eval: ProjectableToField<F>,
    Lc: LinearCode<Zt>,
    F: PrimeField + for<'a> FromWithConfig<&'a Zt::Chal> + for<'a> MulByScalar<&'a F> + FromRef<F>,
    F::Inner: Transcribable,
    F::Modulus: FromRef<Zt::Fmod> + Transcribable,
    I: Ideal,
{
    for ((hint, col), _comm) in hints.iter().zip(witness.iter()).zip(commitments.iter()) {
        // Proximity test
        ZipPlus::<Zt, Lc>::test::<CHECK_FOR_OVERFLOW>(pcs_transcript, pp, col, hint)?;

        // Evaluation proof
        let _eval_f: F = ZipPlus::<Zt, Lc>::evaluate_f::<F, CHECK_FOR_OVERFLOW>(
            pcs_transcript,
            pp,
            col,
            eval_point,
            field_cfg,
            projecting_element,
        )?;
    }
    Ok(())
}
