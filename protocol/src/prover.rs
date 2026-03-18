use super::*;
use crypto_primitives::{ConstIntSemiring, FromPrimitiveWithConfig, FromWithConfig};
use num_traits::Zero;
use zinc_piop::{
    combined_poly_resolver::CombinedPolyResolver,
    ideal_check::IdealCheckProtocol,
    multipoint_eval::MultipointEval,
    projections::{
        evaluate_trace_to_column_mles, project_scalars, project_scalars_to_field,
        project_trace_coeffs_row_major,
    },
};
use zinc_poly::{
    mle::DenseMultilinearExtension, univariate::dynamic::over_field::DynamicPolynomialF,
};
use zinc_transcript::traits::{ConstTranscribable, Transcript};
use zinc_uair::{Uair, constraint_counter::count_constraints, degree_counter::count_max_degree};
use zinc_utils::{
    add, from_ref::FromRef, inner_transparent_field::InnerTransparentField,
    mul_by_scalar::MulByScalar, projectable_to_field::ProjectableToField,
};
use zip_plus::{
    pcs::structs::{ZipPlus, ZipPlusHint, ZipPlusParams, ZipTypes},
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
    /// Zinc+ full PIOP prover.
    ///
    /// The trace arrays contain public columns first, then witness columns,
    /// within each type group. The split is derived from `U::signature()`.
    ///
    /// # Protocol steps
    ///
    /// 0. **Commit**: commit only *witness* columns via Zip+ PCS, absorb roots
    ///    and public data.
    /// 1. **Prime projection** (`\phi_q`: `Z[X] -> F_q[X]`): sample random
    ///    prime q, project full trace and scalars.
    /// 2. **Ideal check**: sample `r in F_q^mu`, prover sends MLE evaluations,
    ///    verifier checks ideal membership.
    /// 3. **Evaluation projection** (`\psi_a`: `F_q[X] -> F_q`): sample `a in
    ///    F_q`, evaluate polynomials at `X = a`.
    /// 4. **F_q sumcheck**: finite-field sumcheck proving the projected
    ///    constraint claim. Produces `up_evals` and `down_evals` at point `r'`.
    /// 5. **Multi-point evaluation sumcheck**: single sumcheck combining
    ///    `up_evals` and `down_evals` at `r'` into a single evaluation point
    ///    `r_0`. Only the sumcheck proof is sent; scalar evaluations at `r_0`
    ///    are derived from the polynomial-valued `lifted_evals` in Step 6.
    /// 6. **Lift-and-project**: compute per-column polynomial MLE evaluations
    ///    at `r_0` (in `F_q[X]`, before `\psi_a`). Absorb into transcript.
    /// 7. **PCS open**: Zip+ prove for each committed *witness* column at
    ///    `r_0`.
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
        project_scalar: impl Fn(&U::Scalar, &F::Config) -> DynamicPolynomialF<F> + Sync,
    ) -> Result<Proof<F>, ProtocolError<F, U::Ideal>> {
        let sig = U::signature();
        let num_pub_bin = sig.public_binary_poly_cols;
        let num_pub_arb = sig.public_arbitrary_poly_cols;
        let num_pub_int = sig.public_int_cols;

        let witness_bin = &trace_bin_poly[num_pub_bin..];
        let witness_arb = &trace_arb_poly[num_pub_arb..];
        let witness_int = &trace_int[num_pub_int..];

        // === Step 0: Commit only witness columns ===
        fn commit_optionally<Zt2: ZipTypes, Lc2: LinearCode<Zt2>>(
            pp: &ZipPlusParams<Zt2, Lc2>,
            trace: &[DenseMultilinearExtension<Zt2::Eval>],
        ) -> Result<
            (Option<ZipPlusHint<Zt2::Cw>>, ZipPlusCommitment),
            ZipError,
        > {
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

        #[cfg(feature = "parallel")]
        let ((res_bin, res_arb), res_int) = rayon::join(
            || {
                rayon::join(
                    || commit_optionally(pp_bin, witness_bin),
                    || commit_optionally(pp_arb, witness_arb),
                )
            },
            || commit_optionally(pp_int, witness_int),
        );

        #[cfg(not(feature = "parallel"))]
        let (res_bin, res_arb, res_int) = (
            commit_optionally(pp_bin, witness_bin),
            commit_optionally(pp_arb, witness_arb),
            commit_optionally(pp_int, witness_int),
        );

        let (hint_bin, commitment_bin) = res_bin?;
        let (hint_arb, commitment_arb) = res_arb?;
        let (hint_int, commitment_int) = res_int?;

        let mut pcs_transcript = PcsProverTranscript::new_from_commitments(
            [&commitment_bin, &commitment_arb, &commitment_int].into_iter(),
        );

        absorb_public_columns(
            &mut pcs_transcript.fs_transcript,
            &trace_bin_poly[..num_pub_bin],
        );
        absorb_public_columns(
            &mut pcs_transcript.fs_transcript,
            &trace_arb_poly[..num_pub_arb],
        );
        absorb_public_columns(&mut pcs_transcript.fs_transcript, &trace_int[..num_pub_int]);

        // === Step 1: Prime projection (\phi_q: Z[X] -> F_q[X]) ===

        let field_cfg = pcs_transcript
            .fs_transcript
            .get_random_field_cfg::<F, Zt::Fmod, Zt::PrimeTest>();

        let projected_scalars_fx = project_scalars::<F, U>(|s| project_scalar(s, &field_cfg));
        let num_constraints = count_constraints::<U>();

        let projected_trace = project_trace_coeffs_row_major::<F, Zt::Int, Zt::Int, D>(
            trace_bin_poly,
            trace_arb_poly,
            trace_int,
            &field_cfg,
        );

        // === Step 2: Ideal check ===
        let (ic_proof, ic_prover_state) = U::prove_combined(
            &mut pcs_transcript.fs_transcript,
            &projected_trace,
            &projected_scalars_fx,
            num_constraints,
            num_vars,
            &field_cfg,
        )?;

        // === Step 3: Evaluation projection (\psi_a: F_q[X] -> F_q) ===
        let projecting_element: Zt::Chal = pcs_transcript.fs_transcript.get_challenge();
        let projecting_element_f: F = F::from_with_cfg(&projecting_element, &field_cfg);

        // Project trace from F_q[X] to F_q by evaluating each polynomial at X = a.
        let projected_trace_f =
            evaluate_trace_to_column_mles(&projected_trace, &projecting_element_f);

        // Project scalars from F_q[X] to F_q.
        let projected_scalars_f =
            project_scalars_to_field(projected_scalars_fx, &projecting_element_f)
                .map_err(|(_s, _f, e)| ProtocolError::ScalarProjection(e))?;

        let max_degree = count_max_degree::<U>();

        // === Step 4: Sumcheck over F_q ===
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

        // === Step 5: Multi-point evaluation sumcheck ===
        // Combines up_evals and down_evals at r' into a single evaluation
        // point r_0 via one sumcheck.
        let (mp_proof, mp_prover_state) = MultipointEval::prove_as_subprotocol(
            &mut pcs_transcript.fs_transcript,
            &projected_trace_f,
            &cpr_prover_state.evaluation_point,
            &cpr_proof.up_evals,
            &cpr_proof.down_evals,
            &field_cfg,
        )?;

        // === Step 6: Lift-and-project at r_0 ===
        // Compute per-column polynomial MLE evaluations at r_0 in F_q[X]
        // (after \phi_q but before \psi_a). The verifier derives the scalar
        // open_evals via \psi_a for the sumcheck consistency check, and
        // supplies these to the Zip+ PCS for alpha-projection.
        let r_0 = &mp_prover_state.eval_point;

        let lifted_evals =
            compute_lifted_evals::<F, D>(r_0, trace_bin_poly, &projected_trace, &field_cfg);

        let mut transcription_buf: Vec<u8> = vec![0; F::Inner::NUM_BYTES];
        for bar_u in &lifted_evals {
            pcs_transcript
                .fs_transcript
                .absorb_random_field_slice(&bar_u.coeffs, &mut transcription_buf);
        }

        // === Step 7: PCS open at r_0 (witness columns only) ===
        if let Some(hint_bin) = &hint_bin {
            let _ = ZipPlus::<Zt::BinaryZt, Zt::BinaryLc>::prove_f::<_, CHECK_FOR_OVERFLOW>(
                &mut pcs_transcript,
                pp_bin,
                witness_bin,
                r_0,
                hint_bin,
                &field_cfg,
            )?;
        }
        if let Some(hint_arb) = &hint_arb {
            let _ = ZipPlus::<Zt::ArbitraryZt, Zt::ArbitraryLc>::prove_f::<_, CHECK_FOR_OVERFLOW>(
                &mut pcs_transcript,
                pp_arb,
                witness_arb,
                r_0,
                hint_arb,
                &field_cfg,
            )?;
        }
        if let Some(hint_int) = &hint_int {
            let _ = ZipPlus::<Zt::IntZt, Zt::IntLc>::prove_f::<_, CHECK_FOR_OVERFLOW>(
                &mut pcs_transcript,
                pp_int,
                witness_int,
                r_0,
                hint_int,
                &field_cfg,
            )?;
        }

        let zip_proof = pcs_transcript.stream.into_inner();
        let commitments = (commitment_bin, commitment_arb, commitment_int);

        // Remember that the public columns come first in the input trace arrays
        let num_total_bin = sig.total_binary_poly_cols();
        let num_total_arb = sig.total_arbitrary_poly_cols();
        let witness_arb_offset = add!(num_total_bin, num_pub_arb);
        let witness_arb_end = add!(witness_arb_offset, sig.witness_arbitrary_poly_cols);
        let witness_int_offset = add!(add!(num_total_bin, num_total_arb), num_pub_int);
        let witness_lifted_evals: Vec<_> = lifted_evals[num_pub_bin..num_total_bin]
            .iter()
            .chain(&lifted_evals[witness_arb_offset..witness_arb_end])
            .chain(&lifted_evals[witness_int_offset..])
            .cloned()
            .collect();

        Ok(Proof {
            commitments,
            ideal_check: ic_proof,
            resolver: cpr_proof,
            multipoint_eval: mp_proof,
            zip: zip_proof,
            witness_lifted_evals,
        })
    }
}
