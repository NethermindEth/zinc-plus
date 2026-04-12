use super::*;
use crypto_primitives::{ConstIntSemiring, FromPrimitiveWithConfig, FromWithConfig};
use num_traits::Zero;
use std::io::Cursor;
use zinc_piop::{
    combined_poly_resolver::{CombinedPolyResolver, VerifierSubclaim},
    ideal_check::IdealCheckProtocol,
    lookup::{
        LogupFinalizerInput, LogupVerifierPreSumcheckData, LookupAuxEvals, LookupGroup,
        group_lookup_specs,
        logup::LogupProtocol,
        utils::{bitpoly_decomp_base, word_decomp_base},
    },
    multipoint_eval::{MultipointEval, MultipointEvalData},
    projections::{
        ProjectedTrace, project_scalars, project_scalars_to_field, project_trace_coeffs_row_major,
    },
    sumcheck::multi_degree::{MultiDegreeSubClaims, MultiDegreeSumcheck},
};
use zinc_poly::{EvaluatablePolynomial, univariate::dynamic::over_field::DynamicPolynomialF};
use zinc_transcript::{
    Blake3Transcript,
    traits::{ConstTranscribable, Transcript},
};
use zinc_uair::{
    ColumnLayout, LookupTableType, Uair, UairTrace,
    constraint_counter::count_constraints,
    ideal::{Ideal, IdealCheck},
    ideal_collector::IdealOrZero,
};
use zinc_utils::{
    add, from_ref::FromRef, inner_transparent_field::InnerTransparentField, mul,
    mul_by_scalar::MulByScalar, powers, projectable_to_field::ProjectableToField,
};
use zip_plus::{
    pcs::structs::{ZipPlus, ZipPlusParams, ZipTypes},
    pcs_transcript::PcsVerifierTranscript,
};

/// Flat proof-provided evaluation slices for lookup verification.
struct LookupEvalsFlat<'a, F> {
    aux: &'a [F],
    chunk: &'a [F],
}

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
    /// `up_evals` and `down_evals` from the combined CPR+Lookup multi-degree
    /// sumcheck (Step 4) are reduced via the multi-point evaluation
    /// sumcheck (Step 5) to a single evaluation point `r_0`. The verifier
    /// recomputes public `lifted_evals` from public data at `r_0`,
    /// interleaves them with the witness `lifted_evals` from the proof,
    /// derives scalar `open_evals` via `\psi_a`, and checks the multipoint
    /// eval consistency. A Zip+ PCS invocation (Step 7) confirms
    /// the witness `lifted_evals`.
    #[allow(clippy::type_complexity)]
    pub fn verify<IdealOverF, const CHECK_FOR_OVERFLOW: bool>(
        (vp_bin, vp_arb, vp_int, vp_lookup): &(
            ZipPlusParams<Zt::BinaryZt, Zt::BinaryLc>,
            ZipPlusParams<Zt::ArbitraryZt, Zt::ArbitraryLc>,
            ZipPlusParams<Zt::IntZt, Zt::IntLc>,
            ZipPlusParams<Zt::LookupZt, Zt::LookupLc>,
        ),
        proof: Proof<F>,
        public_trace: &UairTrace<Zt::Int, Zt::Int, D>,
        num_vars: usize,
        project_scalar: impl Fn(&U::Scalar, &F::Config) -> DynamicPolynomialF<F>,
        project_ideal: impl Fn(&IdealOrZero<U::Ideal>, &F::Config) -> IdealOverF,
    ) -> Result<(), ProtocolError<F, IdealOverF>>
    where
        IdealOverF: Ideal + IdealCheck<DynamicPolynomialF<F>>,
    {
        // === Step 0: Reconstruct transcript from commitments + public data ===
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

        // === Step 4: Sumcheck over F_q ===
        // 4a: CPR prepare_verifier (samples folding challenge, checks claimed sum)
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

        // 4a: Lookup pre-sumcheck
        let sig = U::signature();
        let lookup_groups_info = group_lookup_specs(sig.lookup_specs());
        let (lookup_aux_evals, lookup_chunk_evals, lookup_lifted_evals, lookup_chunk_lifted_evals) =
            match proof.lookup.as_ref() {
                Some(lookup) => (
                    &lookup.aux_evals[..],
                    &lookup.chunk_evals[..],
                    &lookup.lifted_evals[..],
                    &lookup.chunk_lifted_evals[..],
                ),
                None => (
                    &[][..],
                    &[][..],
                    &[][..] as &[DynamicPolynomialF<F>],
                    &[][..],
                ),
            };

        let mut lookup_verifier_pre_sumcheck_data = Vec::new();
        if let Some(ref lookup) = proof.lookup {
            let mut chunk_comm_idx = 0;
            for (g, group_info) in lookup_groups_info.iter().enumerate() {
                // For decomposed groups, absorb chunk commitment before m
                if group_info.table_type.chunk_width().is_some() {
                    let cc = &lookup.chunk_commitments[chunk_comm_idx];
                    pcs_transcript.fs_transcript.absorb_slice(&cc.root);
                    chunk_comm_idx = add!(chunk_comm_idx, 1);
                }
                let (comm_m, comm_uv) = &lookup.commitments[g];
                let data = LogupProtocol::<F>::build_verifier_pre_sumcheck(
                    &mut pcs_transcript.fs_transcript,
                    &comm_m.root,
                    &comm_uv.root,
                    num_vars,
                    &field_cfg,
                );
                lookup_verifier_pre_sumcheck_data.push(data);
            }
        }

        // 4b: Multi-degree sumcheck verify
        let md_subclaims = MultiDegreeSumcheck::verify_as_subprotocol(
            &mut pcs_transcript.fs_transcript,
            num_vars,
            &proof.combined_sumcheck,
            &field_cfg,
        )
        .map_err(CombinedPolyResolverError::SumcheckError)?;

        // 4c: CPR finalize_verifier generates subclaim
        let cpr_subclaim = CombinedPolyResolver::finalize_verifier::<U>(
            &mut pcs_transcript.fs_transcript,
            proof.resolver,
            md_subclaims.point().to_vec(),
            md_subclaims.expected_evaluations()[0].clone(),
            cpr_verifier_ancillary,
            &projected_scalars_f,
            &field_cfg,
        )?;

        // 4d: Lookup finalize — verify batched LogUp + reconstruction at r*
        Self::finalize_lookup_groups(
            &lookup_groups_info,
            &lookup_verifier_pre_sumcheck_data,
            &md_subclaims,
            &cpr_subclaim,
            &LookupEvalsFlat {
                aux: lookup_aux_evals,
                chunk: lookup_chunk_evals,
            },
            &projecting_element_f,
            &field_cfg,
        )?;

        // Absorb lookup aux + chunk evals into transcript
        let mut transcription_buf: Vec<u8> = vec![0; F::Inner::NUM_BYTES];
        let all_lookup_evals: Vec<F> = lookup_aux_evals
            .iter()
            .chain(lookup_chunk_evals.iter())
            .cloned()
            .collect();
        pcs_transcript
            .fs_transcript
            .absorb_random_field_slice(&all_lookup_evals, &mut transcription_buf);

        // === Step 5: Multi-point evaluation sumcheck ===
        let uair_sig = U::signature();
        let mp_subclaim = MultipointEval::verify_as_subprotocol(
            &mut pcs_transcript.fs_transcript,
            proof.multipoint_eval,
            &all_lookup_evals,
            MultipointEvalData {
                eval_point: &cpr_subclaim.evaluation_point,
                up_evals: &cpr_subclaim.up_evals,
                down_evals: &cpr_subclaim.down_evals,
                shifts: uair_sig.shifts(),
            },
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
        let total: &ColumnLayout = uair_sig.total_cols().into();
        let public: &ColumnLayout = uair_sig.public_cols().into();
        let [wit_bin, wit_arb, wit_int] = total.witness_ranges(public);

        let public_lifted = if public.cols() > 0 {
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

        let (num_pub_bin, num_pub_arb) = (
            public.num_binary_poly_cols(),
            public.num_arbitrary_poly_cols(),
        );
        let (num_wit_bin, num_wit_arb) = (wit_bin.len(), wit_arb.len());
        let wit = &proof.witness_lifted_evals;
        let all_lifted_evals: Vec<_> = public_lifted[..num_pub_bin]
            .iter()
            .chain(&wit[..num_wit_bin])
            .chain(&public_lifted[num_pub_bin..add!(num_pub_bin, num_pub_arb)])
            .chain(&wit[num_wit_bin..add!(num_wit_bin, num_wit_arb)])
            .chain(&public_lifted[add!(num_pub_bin, num_pub_arb)..])
            .chain(&wit[add!(num_wit_bin, num_wit_arb)..])
            .chain(lookup_lifted_evals.iter())
            .chain(lookup_chunk_lifted_evals.iter())
            .cloned()
            .collect();

        // Derive scalar open_evals via \psi_a and finalize the multipoint
        // eval consistency check.
        let open_evals: Vec<F> = all_lifted_evals
            .iter()
            .map(|lifted| lifted.evaluate_at_point(&projecting_element_f))
            .collect::<Result<Vec<_>, _>>()
            .map_err(ProtocolError::LiftedEvalProjection)?;

        MultipointEval::verify_subclaim(&mp_subclaim, &open_evals, uair_sig.shifts(), &field_cfg)?;

        // Absorb all lifted_evals into transcript (same order as prover).
        for lifted in &all_lifted_evals {
            pcs_transcript
                .fs_transcript
                .absorb_random_field_slice(&lifted.coeffs, &mut transcription_buf);
        }

        // === Step 7: PCS verify at r_0 (witness columns only) ===

        let mut lifted_offset = wit_bin.start;
        macro_rules! pcs_verify {
            ($Zt:ty, $Lc:ty, $vp:expr, $comm:expr, $err_idx:expr) => {
                if $comm.batch_size > 0 {
                    let end = add!(lifted_offset, $comm.batch_size);
                    let per_poly_alphas = ZipPlus::<$Zt, $Lc>::sample_alphas(
                        &mut pcs_transcript.fs_transcript,
                        $comm.batch_size,
                    );
                    let mut eval_f = F::zero_with_cfg(&field_cfg);
                    for (lifted, alphas) in all_lifted_evals[lifted_offset..end]
                        .iter()
                        .zip(per_poly_alphas.iter())
                    {
                        for (coeff, alpha) in lifted.coeffs.iter().zip(alphas.iter()) {
                            let mut term = F::from_with_cfg(alpha, &field_cfg);
                            term *= coeff;
                            eval_f += &term;
                        }
                    }
                    ZipPlus::<$Zt, $Lc>::verify_with_alphas::<F, CHECK_FOR_OVERFLOW>(
                        &mut pcs_transcript,
                        $vp,
                        $comm,
                        &field_cfg,
                        r_0,
                        &eval_f,
                        &per_poly_alphas,
                    )
                    .map_err(|e| ProtocolError::PcsVerification($err_idx, e))?;
                    #[allow(unused_assignments)]
                    {
                        lifted_offset = end;
                    }
                }
            };
        }
        pcs_verify!(Zt::BinaryZt, Zt::BinaryLc, vp_bin, &proof.commitments.0, 0);
        lifted_offset = wit_arb.start;
        pcs_verify!(
            Zt::ArbitraryZt,
            Zt::ArbitraryLc,
            vp_arb,
            &proof.commitments.1,
            1
        );
        lifted_offset = wit_int.start;
        pcs_verify!(Zt::IntZt, Zt::IntLc, vp_int, &proof.commitments.2, 2);

        // PCS verify for lookup aux + chunks at r_0
        lifted_offset = wit_int.end;
        if let Some(ref lk) = proof.lookup {
            let mut chunk_comm_iter = lk.chunk_commitments.iter();
            for (gi, (comm_m, comm_u)) in lookup_groups_info.iter().zip(lk.commitments.iter()) {
                pcs_verify!(Zt::LookupZt, Zt::LookupLc, vp_lookup, comm_m, 3);
                pcs_verify!(Zt::LookupZt, Zt::LookupLc, vp_lookup, comm_u, 4);
                if gi.table_type.is_decomposed() {
                    let comm_chunk = chunk_comm_iter.next().expect("chunk commitment");
                    pcs_verify!(Zt::LookupZt, Zt::LookupLc, vp_lookup, comm_chunk, 5);
                }
            }
        }

        Ok(())
    }

    /// Verify batched LogUp identities and reconstruction checks for each
    /// lookup group.
    fn finalize_lookup_groups(
        groups: &[LookupGroup],
        pre_sumcheck: &[LogupVerifierPreSumcheckData<F>],
        md_subclaims: &MultiDegreeSubClaims<F>,
        cpr_subclaim: &VerifierSubclaim<F>,
        lookup_evals: &LookupEvalsFlat<F>,
        projecting_element: &F,
        field_cfg: &F::Config,
    ) -> Result<(), LookupError> {
        let aux_evals_flat = lookup_evals.aux;
        let chunk_evals_flat = lookup_evals.chunk;
        let mut md_group_idx = 1; // skip CPR group at index 0
        let mut aux_offset = 0;
        let mut chunk_offset = 0;

        for (g, group_info) in groups.iter().enumerate() {
            let num_lk_cols = group_info.column_indices.len();
            let num_chunks = group_info.table_type.num_chunks();
            let num_logup_cols = mul!(num_lk_cols, num_chunks);
            let is_decomposed = group_info.table_type.is_decomposed();

            let expected = [
                md_subclaims.expected_evaluations()[md_group_idx].clone(),
                md_subclaims.expected_evaluations()[add!(md_group_idx, 1)].clone(),
            ];

            // For decomposed: L*K chunk evals; non-decomposed: L trace evals
            let decomp_w_evals: Vec<F> = if is_decomposed {
                chunk_evals_flat[chunk_offset..add!(chunk_offset, num_logup_cols)].to_vec()
            } else {
                group_info
                    .column_indices
                    .iter()
                    .map(|&idx| cpr_subclaim.up_evals[idx].clone())
                    .collect()
            };

            let aux_evals: Vec<LookupAuxEvals<F>> = (0..num_logup_cols)
                .map(|j| {
                    let m_idx = add!(aux_offset, j);
                    let u_idx = add!(add!(aux_offset, num_logup_cols), j);
                    LookupAuxEvals {
                        m_eval: aux_evals_flat[m_idx].clone(),
                        u_eval: aux_evals_flat[u_idx].clone(),
                    }
                })
                .collect();

            LogupProtocol::<F>::finalize_verifier(
                &pre_sumcheck[g],
                LogupFinalizerInput {
                    subclaim_point: md_subclaims.point(),
                    expected_evaluations: &expected,
                    w_evals: &decomp_w_evals,
                    aux_evals: &aux_evals,
                },
                group_info,
                projecting_element,
                field_cfg,
            )?;

            // Reconstruction check
            if is_decomposed {
                let reconstruction_expected =
                    &md_subclaims.expected_evaluations()[add!(md_group_idx, 2)];

                let trace_evals: Vec<F> = group_info
                    .column_indices
                    .iter()
                    .map(|&idx| cpr_subclaim.up_evals[idx].clone())
                    .collect();

                let one = F::one_with_cfg(field_cfg);
                let eq_val = zinc_poly::utils::eq_eval(
                    md_subclaims.point(),
                    &pre_sumcheck[g].r,
                    one.clone(),
                )?;

                let decomp_base: F = match &group_info.table_type {
                    LookupTableType::BitPoly {
                        chunk_width: Some(c),
                        ..
                    } => bitpoly_decomp_base(*c, projecting_element),
                    LookupTableType::Word {
                        chunk_width: Some(c),
                        ..
                    } => word_decomp_base(*c, field_cfg),
                    _ => unreachable!("is_decomposed guard"),
                };
                let bases = powers(decomp_base, one.clone(), num_chunks);
                let gamma = &pre_sumcheck[g].gamma;
                let gamma_pows = powers(gamma.clone(), one, num_lk_cols);

                let mut computed = F::zero_with_cfg(field_cfg);
                for (col_l, witness_eval) in trace_evals.iter().enumerate() {
                    let chunk_start = add!(chunk_offset, mul!(col_l, num_chunks));
                    let chunk_end = add!(chunk_start, num_chunks);
                    let chunk_slice = &chunk_evals_flat[chunk_start..chunk_end];
                    let mut reconstructed = F::zero_with_cfg(field_cfg);
                    for (idx, base) in bases.iter().enumerate() {
                        reconstructed += &(base.clone() * &chunk_slice[idx]);
                    }
                    let diff = witness_eval.clone() - &reconstructed;
                    computed += &(gamma_pows[col_l].clone() * &diff);
                }
                computed *= &eq_val;
                if computed != *reconstruction_expected {
                    return Err(LookupError::DecompositionInconsistent);
                }
            }

            md_group_idx = add!(md_group_idx, if is_decomposed { 3 } else { 2 });
            aux_offset = add!(aux_offset, mul!(2, num_logup_cols));
            if is_decomposed {
                chunk_offset = add!(chunk_offset, num_logup_cols);
            }
        }

        Ok(())
    }
}
