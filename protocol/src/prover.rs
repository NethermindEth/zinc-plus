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
    mle::DenseMultilinearExtension, univariate::dynamic::over_field::DynamicPolynomialF,
};
use zinc_transcript::traits::{ConstTranscribable, Transcript};
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
    /// 5. **Lift-and-project**: compute per-column unprojected polynomial MLE
    ///    evaluations at the sumcheck challenge point r' (lifted_evals) and at
    ///    the shift point ρ (lifted_evals_shift). Absorb into transcript.
    /// 6a. **PCS open**: Zip+ test + evaluate for each committed column at the
    ///     sumcheck challenge point r'.
    /// 6b. **PCS evaluate-only**: Zip+ evaluate for each committed column at
    ///     the shift point ρ (proximity already established in 6a).
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

        let lifted_evals =
            compute_lifted_evals::<F, D>(eval_point, trace_bin_poly, &projected_trace, &field_cfg);

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
        // Compute per-column lifted evaluations at ρ (same approach as
        // Step 4.5 but at the shift point instead of the sumcheck point).
        let shift_point = &bs_prover_state.shift_point;

        let lifted_evals_shift =
            compute_lifted_evals::<F, D>(shift_point, trace_bin_poly, &projected_trace, &field_cfg);

        // Absorb lifted_evals_shift into transcript before PCS evaluate-only.
        for bar_u in &lifted_evals_shift {
            pcs_transcript
                .fs_transcript
                .absorb_random_field_slice(&bar_u.coeffs, &mut transcription_buf);
        }

        // Prove evaluate-only for each committed batch at ρ.
        if let Some(_hint_bin) = &hint_bin {
            let _ = ZipPlus::<Zt::BinaryZt, Zt::BinaryLc>::prove_evaluate_only_f::<
                _,
                CHECK_FOR_OVERFLOW,
            >(
                &mut pcs_transcript,
                pp_bin,
                trace_bin_poly,
                shift_point,
                &field_cfg,
            )?;
        }
        if let Some(_hint_arb) = &hint_arb {
            let _ = ZipPlus::<Zt::ArbitraryZt, Zt::ArbitraryLc>::prove_evaluate_only_f::<
                _,
                CHECK_FOR_OVERFLOW,
            >(
                &mut pcs_transcript,
                pp_arb,
                trace_arb_poly,
                shift_point,
                &field_cfg,
            )?;
        }
        if let Some(_hint_int) = &hint_int {
            let _ = ZipPlus::<Zt::IntZt, Zt::IntLc>::prove_evaluate_only_f::<_, CHECK_FOR_OVERFLOW>(
                &mut pcs_transcript,
                pp_int,
                trace_int,
                shift_point,
                &field_cfg,
            )?;
        }

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
            lifted_evals_shift,
        })
    }
}

/// Compute per-column lifted MLE evaluations at `point`, with optimizations:
///
/// - **Binary columns** (first `trace_bin_poly.len()` entries): the inner
///   product with the eq table is computed with conditional additions only (no
///   multiplications), exploiting the fact that every coefficient is 0 or 1.
/// - **Non-binary columns** (remaining entries in `projected_trace`): standard
///   multiply-accumulate against the precomputed eq table, avoiding the
///   clone-heavy `evaluate_with_config` path.
///
/// The eq(point, *) table is built once and reused across all columns.
#[allow(clippy::arithmetic_side_effects)]
fn compute_lifted_evals<F, const D: usize>(
    point: &[F],
    trace_bin_poly: &[DenseMultilinearExtension<BinaryPoly<D>>],
    projected_trace: &[DenseMultilinearExtension<DynamicPolynomialF<F>>],
    field_cfg: &F::Config,
) -> Vec<DynamicPolynomialF<F>>
where
    F: InnerTransparentField,
{
    let eq_table = zinc_poly::utils::build_eq_x_r_vec(point, field_cfg)
        .expect("compute_lifted_evals: eq table build failed");

    let n_bin = trace_bin_poly.len();
    let zero = F::zero_with_cfg(field_cfg);

    let mut result = Vec::with_capacity(projected_trace.len());

    for col in trace_bin_poly {
        let mut coeffs = vec![zero.clone(); D];
        for (b, entry) in col.iter().enumerate() {
            for (l, coeff) in entry.iter().enumerate() {
                if coeff.into_inner() {
                    coeffs[l] += &eq_table[b];
                }
            }
        }
        result.push(DynamicPolynomialF::new_trimmed(coeffs));
    }

    for col_mle in &projected_trace[n_bin..] {
        let num_coeffs = col_mle
            .iter()
            .map(|entry| entry.coeffs.len())
            .max()
            .unwrap_or(0);

        let mut coeffs = vec![zero.clone(); num_coeffs];
        for (b, entry) in col_mle.iter().enumerate() {
            for (l, coeff) in entry.coeffs.iter().enumerate() {
                let mut term = eq_table[b].clone();
                term *= coeff;
                coeffs[l] += &term;
            }
        }
        result.push(DynamicPolynomialF::new_trimmed(coeffs));
    }

    result
}
