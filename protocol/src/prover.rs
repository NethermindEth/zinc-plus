use super::*;
use crypto_primitives::{ConstIntSemiring, FromPrimitiveWithConfig, FromWithConfig};
use num_traits::Zero;
use zinc_piop::{
    batched_shift::BatchedShift,
    combined_poly_resolver::CombinedPolyResolver,
    ideal_check::IdealCheckProtocol,
    projections::{
        project_scalars, project_scalars_to_field, project_trace_coeffs, project_trace_to_field,
    },
};
use zinc_poly::{
    mle::{DenseMultilinearExtension, MultilinearExtensionWithConfig},
    univariate::dynamic::over_field::DynamicPolynomialF,
};
use zinc_transcript::traits::{ConstTranscribable, Transcribable, Transcript};
use zinc_uair::{Uair, constraint_counter::count_constraints, degree_counter::count_max_degree};
use zinc_utils::{
    from_ref::FromRef, inner_transparent_field::InnerTransparentField, mul_by_scalar::MulByScalar,
    projectable_to_field::ProjectableToField,
};
use zip_plus::{
    pcs::structs::{ZipPlus, ZipPlusParams, ZipTypes},
    pcs_transcript::PcsProverTranscript,
};

impl<Zt, U, F, const D: usize> ZincPlusPiop<Zt, U, F, D>
where
    Zt: ZincTypes<D>,
    Zt::Int: ProjectableToField<F>,
    <Zt::ArbitraryZt as ZipTypes>::Eval: ProjectableToField<F>,
    F: InnerTransparentField
        + FromPrimitiveWithConfig
        + FromWithConfig<Zt::Int>
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
    /// Zinc+ full PIOP Prover.
    ///
    /// # Protocol flow:
    ///
    /// 0. **Commit**: commit each witness column via Zip+ PCS, absorb roots.
    /// 1. **Prime projection** (φ_q: Q\[X\] → F_q\[X\]): sample random prime q
    ///    from transcript, project trace and scalars.
    /// 2. **Ideal check**: sample r ∈ F_q^μ, prover sends MLE evaluations,
    ///    verifier checks ideal membership.
    /// 3. **Evaluation projection** (ψ_a: F_q\[X\] → F_q): sample a ∈ F_q,
    ///    evaluate polynomials at X = a.
    /// 4. **Finite-field PIOP**: sumcheck over F_q to prove the projected
    ///    claim. 4.5. **Batched shift**: sumcheck reducing shifted-MLE claims
    ///    to standard MLE claims at a new point ρ.
    /// 5. ?????
    /// 6a. **PCS open**: Zip+ test + evaluate for each committed column at the
    ///     sumcheck challenge point r'.
    /// 6b. **PCS evaluate-only**: Zip+ evaluate for each committed column at
    ///     the shift point ρ (proximity already established in 5a).
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
    ) -> Result<Proof<F>, ProtocolError<F, U::Ideal>> {
        // === Step 0: Commit to witness traces ===

        // Commit each witness column via Zip+ PCS.
        macro_rules! commit_optionally {
            ($pp:expr, $trace:expr) => {
                if $trace.is_empty() {
                    (
                        None,
                        ZipPlusCommitment {
                            root: Default::default(),
                            batch_size: 0,
                        },
                    )
                } else {
                    let (hint, commitment) = ZipPlus::commit($pp, $trace)?;
                    (Some(hint), commitment)
                }
            };
        }
        let (hint_bin, commitment_bin) = commit_optionally!(pp_bin, trace_bin_poly);
        let (hint_arb, commitment_arb) = commit_optionally!(pp_arb, trace_arb_poly);
        let (hint_int, commitment_int) = commit_optionally!(pp_int, trace_int);

        // Create the main transcript
        let mut pcs_transcript = PcsProverTranscript::new_from_commitments(
            [&commitment_bin, &commitment_arb, &commitment_int].into_iter(),
        )?;
        // TODO: Absorb public inputs as well once they are part of the protocol,
        //       or this will open up a soundness vulnerability!

        // === Step 1: Prime projection (φ_q: Z[X] → F_q[X]) ===

        // Sample a random prime modulus from the Fiat-Shamir transcript.
        let field_cfg = pcs_transcript
            .fs_transcript
            .get_random_field_cfg::<F, Zt::Fmod, Zt::PrimeTest>();

        // Project the witness trace from Z[X] to F_q[X].
        let projected_trace = project_trace_coeffs::<F, Zt::Int, Zt::Int, D>(
            trace_bin_poly,
            trace_arb_poly,
            trace_int,
            &field_cfg,
        );

        // Project UAIR scalars from Z[X] to F_q[X].
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

        // === Step 4.5: Lift-and-project (oracle evaluations at sumcheck point) ===
        // Compute per-column unprojected polynomial MLE evaluations at the
        // sumcheck challenge point. These are in F_q[X] (after \phi_q but before
        // \psi_a), so the verifier can check \psi_a(lifted_eval_j) == up_eval_j and
        // supply them to the Zip+ PCS for alpha-projection.
        //
        // For each column, we evaluate the MLE coefficient-by-coefficient:
        //   lifted_eval_j = Σ_ℓ (Σ_b eq(b, r') * c_{j,b,ℓ}) * X^ℓ
        // where c_{j,b,ℓ} is the ℓ-th coefficient of the φ_q-projected entry
        // at position b.
        let eval_point = &cpr_prover_state.evaluation_point;
        let zero_inner = F::zero_with_cfg(&field_cfg).into_inner();

        let lifted_evals: Vec<DynamicPolynomialF<F>> = projected_trace
            .iter()
            .map(|col_mle| {
                let max_degree = col_mle
                    .iter()
                    .flat_map(|entry| entry.degree())
                    .max()
                    .unwrap_or(0);

                let coeffs: Vec<F> = (0..=max_degree)
                    .map(|l| {
                        let coeff_mle: DenseMultilinearExtension<F::Inner> = col_mle
                            .iter()
                            .map(|entry| {
                                entry
                                    .coeffs
                                    .get(l)
                                    .map(|f| f.inner().clone())
                                    .unwrap_or_else(|| zero_inner.clone())
                            })
                            .collect();

                        coeff_mle
                            .evaluate_with_config(eval_point, &field_cfg)
                            .expect("lifted eval: coefficient MLE evaluation failed")
                    })
                    .collect();

                DynamicPolynomialF { coeffs }
            })
            .collect();

        // Absorb lifted_evals into the Fiat-Shamir transcript before PCS.
        let mut transcription_buf: Vec<u8> = vec![0; F::Inner::NUM_BYTES];
        for bar_u in &lifted_evals {
            pcs_transcript
                .fs_transcript
                .absorb_random_field_slice(&bar_u.coeffs, &mut transcription_buf);
        }

        // === Step 5: Batched shift (reduce down_evals to shift_evals at ρ) ===
        let (bs_proof, bs_prover_state) = BatchedShift::prove_as_subprotocol(
            &mut pcs_transcript.fs_transcript,
            &projected_trace_f,
            &cpr_prover_state.evaluation_point,
            &cpr_proof.down_evals,
            &field_cfg,
        )?;

        // === Step 6a: PCS open (prove witness MLE evaluations at r') ===
        // After the sumcheck, the prover must prove that the committed
        // witness MLEs evaluate to the claimed values (up_evals) at the
        // sumcheck challenge point r'.
        //
        // The Zip+ PCS proves that the committed polynomial-valued traces
        // evaluate (under alpha-projection) consistently with the lifted_evals.
        // The verifier checks ψ_a(lifted_eval_j) == up_eval_j to tie back to
        // the sumcheck.
        //
        // TODO: Once we add public inputs, the verifier will compute public
        //       input MLE evaluations at the sumcheck point directly from
        //       public data. The PCS only covers witness columns.

        if let Some(hint_bin) = &hint_bin {
            let _ = ZipPlus::<Zt::BinaryZt, Zt::BinaryLc>::prove_f::<_, CHECK_FOR_OVERFLOW>(
                &mut pcs_transcript,
                pp_bin,
                trace_bin_poly,
                eval_point,
                hint_bin,
                &field_cfg,
            )?;
        }
        if let Some(hint_arb) = &hint_arb {
            let _ = ZipPlus::<Zt::ArbitraryZt, Zt::ArbitraryLc>::prove_f::<_, CHECK_FOR_OVERFLOW>(
                &mut pcs_transcript,
                pp_arb,
                trace_arb_poly,
                eval_point,
                hint_arb,
                &field_cfg,
            )?;
        }
        if let Some(hint_int) = &hint_int {
            let _ = ZipPlus::<Zt::IntZt, Zt::IntLc>::prove_f::<_, CHECK_FOR_OVERFLOW>(
                &mut pcs_transcript,
                pp_int,
                trace_int,
                eval_point,
                hint_int,
                &field_cfg,
            )?;
        }

        // === Step 6b: PCS evaluate-only at shift point ρ ===
        // Proximity was already established in Step 5a, so we only need
        // evaluation consistency proofs at the new point ρ.
        let shift_point = &bs_prover_state.shift_point;

        pcs_prove_shift_evaluations::<Zt::BinaryZt, Zt::BinaryLc, _, _, CHECK_FOR_OVERFLOW>(
            &mut pcs_transcript,
            pp_bin,
            trace_bin_poly,
            shift_point,
            &field_cfg,
            &projecting_element,
        )?;
        pcs_prove_shift_evaluations::<Zt::ArbitraryZt, Zt::ArbitraryLc, _, _, CHECK_FOR_OVERFLOW>(
            &mut pcs_transcript,
            pp_arb,
            trace_arb_poly,
            shift_point,
            &field_cfg,
            &projecting_element,
        )?;
        pcs_prove_shift_evaluations::<Zt::IntZt, Zt::IntLc, _, _, CHECK_FOR_OVERFLOW>(
            &mut pcs_transcript,
            pp_int,
            trace_int,
            shift_point,
            &field_cfg,
            &projecting_element,
        )?;

        let zip_proof = pcs_transcript.stream.into_inner();
        let commitments = (commitment_bin, commitment_arb, commitment_int);

        Ok(Proof {
            num_witness_cols: (trace_bin_poly.len(), trace_arb_poly.len(), trace_int.len()),
            commitments,
            ideal_check: ic_proof,
            resolver: cpr_proof,
            batched_shift: bs_proof,
                zip: zip_proof,
                lifted_evals,
            })
    }
}

fn pcs_prove_shift_evaluations<Zt, Lc, F, I, const CHECK_FOR_OVERFLOW: bool>(
    pcs_transcript: &mut PcsProverTranscript,
    pp: &ZipPlusParams<Zt, Lc>,
    witness: &[DenseMultilinearExtension<Zt::Eval>],
    shift_point: &[F],
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
    for col in witness {
        // Evaluate-only (no proximity test needed)
        let _eval_f: F = ZipPlus::<Zt, Lc>::evaluate_f::<F, CHECK_FOR_OVERFLOW>(
            pcs_transcript,
            pp,
            col,
            shift_point,
            field_cfg,
            projecting_element,
        )?;
    }
    Ok(())
}
