use super::*;
use crypto_primitives::{ConstIntSemiring, FromPrimitiveWithConfig, FromWithConfig};
use num_traits::Zero;
use zinc_piop::{
    combined_poly_resolver::CombinedPolyResolver,
    ideal_check::IdealCheckProtocol,
    lookup::{
        LogupProverAncillary, group_lookup_specs,
        logup::LogupProtocol,
        utils::{batch_inverse_shifted, compute_multiplicities},
    },
    multipoint_eval::{MultipointEval, MultipointEvalData},
    projections::{
        ProjectedTrace, evaluate_trace_to_column_mles, project_scalars, project_scalars_to_field,
        project_trace_coeffs_column_major, project_trace_coeffs_row_major,
    },
    sumcheck::multi_degree::{MultiDegreeSumcheck, MultiDegreeSumcheckGroup},
};
use zinc_poly::{
    mle::MultilinearExtensionWithConfig, univariate::dynamic::over_field::DynamicPolynomialF,
};
use zinc_transcript::traits::{ConstTranscribable, Transcript};
use zinc_uair::{
    ColumnLayout, Uair, UairTrace, constraint_counter::count_constraints,
    degree_counter::count_max_degree,
};
use zinc_utils::{
    cfg_join, from_ref::FromRef, inner_transparent_field::InnerTransparentField,
    mul_by_scalar::MulByScalar, projectable_to_field::ProjectableToField,
};
use zip_plus::{
    pcs::structs::{ZipPlus, ZipPlusCommitment, ZipPlusHint, ZipPlusParams, ZipTypes},
    pcs_transcript::PcsProverTranscript,
};

/// Per-group PCS data for lookup auxiliary columns, needed for
/// Step 7 PCS opening at r_0.
struct LookupGroupPcsData<Zt: ZipTypes> {
    hint_m: ZipPlusHint<Zt::Cw>,
    hint_u: ZipPlusHint<Zt::Cw>,
    pcs_m_mles: Vec<DenseMultilinearExtension<Zt::Eval>>,
    pcs_u_mles: Vec<DenseMultilinearExtension<Zt::Eval>>,
    comm_m: ZipPlusCommitment,
    comm_u: ZipPlusCommitment,
    /// Chunk PCS data, present only for decomposed lookups.
    chunk: Option<ChunkPcsData<Zt>>,
}

struct ChunkPcsData<Zt: ZipTypes> {
    hint: ZipPlusHint<Zt::Cw>,
    mles: Vec<DenseMultilinearExtension<Zt::Eval>>,
    comm: ZipPlusCommitment,
}

impl<Zt, U, F, const D: usize> ZincPlusPiop<Zt, U, F, D>
where
    Zt: ZincTypes<D>,
    Zt::Int: ProjectableToField<F>,
    <Zt::ArbitraryZt as ZipTypes>::Eval: ProjectableToField<F>,
    <Zt::LookupZt as ZipTypes>::Eval: FromRef<F::Inner>,
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
    /// Zinc+ full PIOP prover.
    ///
    /// The trace arrays contain public columns first, then witness columns,
    /// within each type group. The split is derived from `U::signature()`.
    ///
    /// # Const parameters
    ///
    /// - `MLE_FIRST`: when `true`, the ideal check step uses the MLE-first
    ///   approach (`prove_linear`) with a column-major projected trace; when
    ///   `false`, the combined polynomial approach (`prove_combined`) with a
    ///   row-major projected trace is used. MLE-first is only valid for linear
    ///   UAIRs (no polynomial multiplications in constraints).
    /// - `CHECK_FOR_OVERFLOW`: propagated to the PCS prover.
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
    /// 4. **Combined CPR + Lookup multi-degree sumcheck**: batches the CPR
    ///    constraint claim (degree `max_deg+2`) with lookup groups (one per
    ///    table type) into a single sumcheck sharing one evaluation point `r*`.
    ///    Produces `up_evals` and `down_evals` (CPR), then evaluates lookup
    ///    auxiliary and chunk inner MLEs at `r*` and absorbs them.
    /// 5. **Multi-point evaluation sumcheck**: batches trace MLEs and lookup
    ///    aux + chunk inner MLEs with challenge, then reduces `up_evals` and
    ///    `down_evals` into a single evaluation point `r_0` via one sumcheck.
    ///    Scalar evaluations at `r_0` are derived from `lifted_evals` in Step
    ///    6.
    /// 6. **Lift-and-project**: compute per-column polynomial MLE evaluations
    ///    at `r_0` (in `F_q[X]`, before `\psi_a`). Absorb into transcript.
    /// 7. **PCS open**: Zip+ prove for each committed column set (witness
    ///    trace, lookup m/u, and chunk if decomposed) at `r_0`.
    #[allow(clippy::type_complexity)]
    pub fn prove<const MLE_FIRST: bool, const CHECK_FOR_OVERFLOW: bool>(
        (pp_bin, pp_arb, pp_int, pp_lookup): &(
            ZipPlusParams<Zt::BinaryZt, Zt::BinaryLc>,
            ZipPlusParams<Zt::ArbitraryZt, Zt::ArbitraryLc>,
            ZipPlusParams<Zt::IntZt, Zt::IntLc>,
            ZipPlusParams<Zt::LookupZt, Zt::LookupLc>,
        ),
        trace: &UairTrace<'static, Zt::Int, Zt::Int, D>,
        num_vars: usize,
        project_scalar: impl Fn(&U::Scalar, &F::Config) -> DynamicPolynomialF<F>,
    ) -> Result<Proof<F>, ProtocolError<F, U::Ideal>> {
        let sig = U::signature();
        let public_trace = trace.public(&sig);
        let witness_trace = trace.witness(&sig);

        // === Step 0: Commit only witness columns ===
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

        // === Step 1: Prime projection (\phi_q: Z[X] -> F_q[X]) ===

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

        // === Step 2: Ideal check ===
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
        // 4a: CPR + Lookup prepare → sumcheck group
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

        // Prepare Lookup groups
        let (lookup_groups, lookup_pcs_data, lookup_aux_f, lookup_chunk_f) =
            Self::prepare_lookup_groups(
                &mut pcs_transcript,
                pp_lookup,
                &projected_trace_f,
                num_vars,
                &projecting_element_f,
                &field_cfg,
            )?;

        // 4b: Multi-degree sumcheck with CPR group and lookup groups
        let mut groups = vec![cpr_group];
        groups.extend(lookup_groups);
        let (combined_sumcheck, md_states) = MultiDegreeSumcheck::prove_as_subprotocol(
            &mut pcs_transcript.fs_transcript,
            groups,
            num_vars,
            &field_cfg,
        );

        // 4c: Finalize up_evals, down_evals
        let (cpr_proof, cpr_prover_state) = CombinedPolyResolver::finalize_prover(
            &mut pcs_transcript.fs_transcript,
            md_states.into_iter().next().expect("one CPR group"),
            cpr_ancillary,
            &field_cfg,
        )?;

        // 4d: Compute lookup aux + chunk evals at r* and absorb
        let lookup_aux_inner_mles = to_inner_mles(&lookup_aux_f, num_vars, &field_cfg);
        let lookup_chunk_inner_mles = to_inner_mles(&lookup_chunk_f, num_vars, &field_cfg);
        let r_star = &cpr_prover_state.evaluation_point;
        let lookup_aux_evals = eval_inner_mles(&lookup_aux_inner_mles, r_star, &field_cfg);
        let lookup_chunk_evals = eval_inner_mles(&lookup_chunk_inner_mles, r_star, &field_cfg);

        let mut transcription_buf: Vec<u8> = vec![0; F::Inner::NUM_BYTES];
        pcs_transcript
            .fs_transcript
            .absorb_random_field_slice(&lookup_aux_evals, &mut transcription_buf);
        pcs_transcript
            .fs_transcript
            .absorb_random_field_slice(&lookup_chunk_evals, &mut transcription_buf);

        // === Step 5: Multi-point evaluation sumcheck ===
        // Extend trace_mles and up_evals with lookup (aux + chunks).
        // Combines up_evals and down_evals at r' into a single evaluation
        // point r_0 via one sumcheck.
        let uair_sig = U::signature();
        let all_lookup_inner_mles: Vec<_> = lookup_aux_inner_mles
            .iter()
            .chain(lookup_chunk_inner_mles.iter())
            .cloned()
            .collect();
        let (mp_proof, mp_prover_state) = MultipointEval::prove_as_subprotocol(
            &mut pcs_transcript.fs_transcript,
            &projected_trace_f,
            &all_lookup_inner_mles,
            MultipointEvalData {
                eval_point: &cpr_prover_state.evaluation_point,
                up_evals: &cpr_proof.up_evals,
                down_evals: &cpr_proof.down_evals,
                shifts: uair_sig.shifts(),
            },
            &field_cfg,
        )?;

        // === Step 6: Lift-and-project at r_0 ===
        // Compute per-column polynomial MLE evaluations at r_0 in F_q[X]
        // (after \phi_q but before \psi_a). The verifier derives the scalar
        // open_evals via \psi_a for the sumcheck consistency check, and
        // supplies these to the Zip+ PCS for alpha-projection.
        let r_0 = &mp_prover_state.eval_point;
        let lifted_evals =
            compute_lifted_evals::<F, D>(r_0, &trace.binary_poly, &projected_trace, &field_cfg);
        let lookup_lifted_evals = inner_mles_to_lifted(&lookup_aux_inner_mles, r_0, &field_cfg);
        let lookup_chunk_lifted_evals =
            inner_mles_to_lifted(&lookup_chunk_inner_mles, r_0, &field_cfg);

        for lifted in &lifted_evals {
            pcs_transcript
                .fs_transcript
                .absorb_random_field_slice(&lifted.coeffs, &mut transcription_buf);
        }
        for lifted in lookup_lifted_evals
            .iter()
            .chain(lookup_chunk_lifted_evals.iter())
        {
            pcs_transcript
                .fs_transcript
                .absorb_random_field_slice(&lifted.coeffs, &mut transcription_buf);
        }

        // === Step 7: PCS open at r_0 (witness columns only) ===
        macro_rules! pcs_open {
            ($zt:ty, $lc:ty, $pp:expr, $mles:expr, $hint:expr) => {
                if let Some(hint) = $hint {
                    let _ = ZipPlus::<$zt, $lc>::prove_f::<_, CHECK_FOR_OVERFLOW>(
                        &mut pcs_transcript,
                        $pp,
                        $mles,
                        r_0,
                        hint,
                        &field_cfg,
                    )?;
                }
            };
        }

        pcs_open!(
            Zt::BinaryZt,
            Zt::BinaryLc,
            pp_bin,
            &witness_trace.binary_poly,
            &hint_bin
        );
        pcs_open!(
            Zt::ArbitraryZt,
            Zt::ArbitraryLc,
            pp_arb,
            &witness_trace.arbitrary_poly,
            &hint_arb
        );
        pcs_open!(Zt::IntZt, Zt::IntLc, pp_int, &witness_trace.int, &hint_int);

        for data in &lookup_pcs_data {
            pcs_open!(
                Zt::LookupZt,
                Zt::LookupLc,
                pp_lookup,
                &data.pcs_m_mles,
                &Some(&data.hint_m)
            );
            pcs_open!(
                Zt::LookupZt,
                Zt::LookupLc,
                pp_lookup,
                &data.pcs_u_mles,
                &Some(&data.hint_u)
            );
            if let Some(ref chunk) = data.chunk {
                pcs_open!(
                    Zt::LookupZt,
                    Zt::LookupLc,
                    pp_lookup,
                    &chunk.mles,
                    &Some(&chunk.hint)
                );
            }
        }

        let zip_proof = pcs_transcript.stream.into_inner();
        let commitments = (commitment_bin, commitment_arb, commitment_int);

        // Extract witness-only lifted evals (public columns come first in each
        // segment).
        let total: &ColumnLayout = sig.total_cols().into();
        let public: &ColumnLayout = sig.public_cols().into();
        let [wit_bin, wit_arb, wit_int] = total.witness_ranges(public);
        let witness_lifted_evals: Vec<_> = lifted_evals[wit_bin]
            .iter()
            .chain(&lifted_evals[wit_arb])
            .chain(&lifted_evals[wit_int])
            .cloned()
            .collect();

        let lookup = if lookup_pcs_data.is_empty() {
            None
        } else {
            let chunk_commitments: Vec<_> = lookup_pcs_data
                .iter()
                .filter_map(|d| d.chunk.as_ref().map(|c| c.comm.clone()))
                .collect();
            Some(LookupProofData {
                commitments: lookup_pcs_data
                    .iter()
                    .map(|d| (d.comm_m.clone(), d.comm_u.clone()))
                    .collect(),
                aux_evals: lookup_aux_evals,
                lifted_evals: lookup_lifted_evals,
                chunk_commitments,
                chunk_evals: lookup_chunk_evals,
                chunk_lifted_evals: lookup_chunk_lifted_evals,
            })
        };

        Ok(Proof {
            commitments,
            ideal_check: ic_proof,
            resolver: cpr_proof,
            combined_sumcheck,
            multipoint_eval: mp_proof,
            zip: zip_proof,
            witness_lifted_evals,
            lookup,
        })
    }

    /// Prepare lookup sumcheck groups for all lookup specs in the UAIR.
    ///
    /// For each group of L columns sharing the same table type:
    ///
    /// **Non-decomposed** (`chunk_width = None`):
    /// 1. Extract witnesses + full table
    /// 2. Commit m (L cols), squeeze β, commit u (L cols), squeeze r, γ
    /// 3. `build_sumcheck_groups` → 2 groups
    ///
    /// **Decomposed** (`chunk_width = Some(c)`):
    /// 1. Construct decomposed witnesses and subtables + L×K chunks
    /// 2. Commit chunks (L*K), commit m (L*K), squeeze β, commit u (L*K),
    ///    squeeze r, γ
    /// 3. `build_sumcheck_groups` with L*K virtual cols → 2 groups
    /// 4. `build_reconstruction_group` → 1 group
    ///
    /// Returns `(sumcheck_groups, pcs_data, aux_f, chunk_f)`:
    /// - `aux_f`: per-group `[m_0..m_{N-1}, u_0..u_{N-1}]` (N = L or L*K)
    /// - `chunk_f`: per-group flattened chunk columns (empty for
    ///   non-decomposed)
    #[allow(clippy::type_complexity)]
    fn prepare_lookup_groups(
        pcs_transcript: &mut PcsProverTranscript,
        pp_lookup: &ZipPlusParams<Zt::LookupZt, Zt::LookupLc>,
        projected_trace_f: &[DenseMultilinearExtension<F::Inner>],
        num_vars: usize,
        projecting_element_f: &F,
        field_cfg: &F::Config,
    ) -> Result<
        (
            Vec<MultiDegreeSumcheckGroup<F>>,
            Vec<LookupGroupPcsData<Zt::LookupZt>>,
            Vec<Vec<F>>,
            Vec<Vec<F>>,
        ),
        ProtocolError<F, U::Ideal>,
    >
    where
        F::Inner: ConstTranscribable + Zero + Default,
        F::Modulus: ConstTranscribable,
        <Zt::LookupZt as ZipTypes>::Eval: FromRef<F::Inner>,
    {
        // Convert field element to the canonical integer
        let f_to_eval = |f: &F| -> <Zt::LookupZt as ZipTypes>::Eval {
            <Zt::LookupZt as ZipTypes>::Eval::from_ref(&f.retrieve_canonical())
        };

        let sig = U::signature();
        let lookup_groups_info = group_lookup_specs(sig.lookup_specs());

        let mut all_groups = Vec::new();
        let mut all_pcs_data = Vec::new();
        let mut all_aux_f: Vec<Vec<F>> = Vec::new();
        let mut all_chunk_f: Vec<Vec<F>> = Vec::new();

        let eval_zero = f_to_eval(&F::zero_with_cfg(field_cfg));
        let mk_lookup_mle =
            |vals: &[F]| -> DenseMultilinearExtension<<Zt::LookupZt as ZipTypes>::Eval> {
                DenseMultilinearExtension::from_evaluations_vec(
                    num_vars,
                    vals.iter().map(&f_to_eval).collect(),
                    eval_zero.clone(),
                )
            };

        for group_info in &lookup_groups_info {
            let (witnesses, _full_table) = LogupProtocol::<F>::extract_witnesses_and_table(
                projected_trace_f,
                group_info,
                projecting_element_f,
                field_cfg,
            );

            let decomp = LogupProtocol::<F>::extract_decomposed(
                &witnesses,
                group_info,
                projecting_element_f,
                field_cfg,
            );

            // If decomposed determine columns + table for LogUp
            let (logup_cols, logup_table, chunk_pcs) = if let Some(ref decomp) = decomp {
                // Flatten chunks[l][k] → L*K virtual columns
                let virtual_cols: Vec<Vec<F>> = decomp
                    .chunks
                    .iter()
                    .flat_map(|per_col| per_col.iter().cloned())
                    .collect();

                // Commit L*K chunk columns (Round LC — before β)
                let chunk_mles: Vec<_> = virtual_cols.iter().map(|c| mk_lookup_mle(c)).collect();
                let (h_chunk, c_chunk) =
                    ZipPlus::<Zt::LookupZt, Zt::LookupLc>::commit(pp_lookup, &chunk_mles)?;
                pcs_transcript.fs_transcript.absorb_slice(&c_chunk.root);

                let chunk_pcs = ChunkPcsData {
                    hint: h_chunk,
                    mles: chunk_mles,
                    comm: c_chunk,
                };
                (virtual_cols, decomp.subtable.clone(), Some(chunk_pcs))
            } else {
                (witnesses.clone(), _full_table, None)
            };

            // Compute multiplicities against effective table, commit in one batch
            let num_logup_cols = logup_cols.len();
            let mut m_vecs = Vec::with_capacity(num_logup_cols);
            let mut m_mles = Vec::with_capacity(num_logup_cols);
            for col in &logup_cols {
                let m = compute_multiplicities(col, &logup_table, field_cfg)
                    .ok_or(LookupError::WitnessNotInTable)?;
                m_mles.push(mk_lookup_mle(&m));
                m_vecs.push(m);
            }
            let (hint_m, comm_m) =
                ZipPlus::<Zt::LookupZt, Zt::LookupLc>::commit(pp_lookup, &m_mles)?;
            pcs_transcript.fs_transcript.absorb_slice(&comm_m.root);

            // Squeeze β (shared across all columns in this group)
            let beta: F = pcs_transcript.fs_transcript.get_field_challenge(field_cfg);

            // Compute inverse witnesses + shared table inverse, commit only u
            let mut u_vecs = Vec::with_capacity(num_logup_cols);
            let mut u_mles = Vec::with_capacity(num_logup_cols);
            for col in &logup_cols {
                let u = batch_inverse_shifted(&beta, col);
                u_mles.push(mk_lookup_mle(&u));
                u_vecs.push(u);
            }
            let v = batch_inverse_shifted(&beta, &logup_table);
            let (hint_u, comm_u) =
                ZipPlus::<Zt::LookupZt, Zt::LookupLc>::commit(pp_lookup, &u_mles)?;
            pcs_transcript.fs_transcript.absorb_slice(&comm_u.root);

            // Squeeze r then γ
            let r: Vec<F> = pcs_transcript
                .fs_transcript
                .get_field_challenges(num_vars, field_cfg);
            let gamma: F = pcs_transcript.fs_transcript.get_field_challenge(field_cfg);

            // Build 2 batched LogUp sumcheck groups
            let col_refs: Vec<&[F]> = logup_cols.iter().map(|c| c.as_slice()).collect();
            let auxs: Vec<LogupProverAncillary<'_, F>> = m_vecs
                .iter()
                .zip(u_vecs.iter())
                .map(|(m, u)| LogupProverAncillary {
                    multiplicities: m,
                    inverse_witness: u,
                    inverse_table: &v,
                })
                .collect();
            let grps = LogupProtocol::build_sumcheck_groups(
                &col_refs,
                &logup_table,
                &auxs,
                &beta,
                &gamma,
                &r,
                field_cfg,
            )?;
            all_groups.extend(grps);

            // Optionally build reconstruction group for decomposed lookups
            if let Some(ref decomp) = decomp {
                let witness_refs: Vec<&[F]> = witnesses.iter().map(|w| w.as_slice()).collect();
                let recon_grps = LogupProtocol::build_reconstruction_group(
                    &witness_refs,
                    &decomp.chunks,
                    &decomp.decomp_bases,
                    &gamma,
                    &r,
                    field_cfg,
                )?;
                all_groups.extend(recon_grps);

                // Collect chunk F vectors for eval at r* / lifted eval / PCS
                for per_col in &decomp.chunks {
                    for chunk in per_col {
                        all_chunk_f.push(chunk.clone());
                    }
                }
            }

            // Collect aux F vectors: [m_0..m_{N-1}, u_0..u_{N-1}]
            for m in &m_vecs {
                all_aux_f.push(m.clone());
            }
            for u in &u_vecs {
                all_aux_f.push(u.clone());
            }

            all_pcs_data.push(LookupGroupPcsData {
                hint_m,
                hint_u,
                pcs_m_mles: m_mles,
                pcs_u_mles: u_mles,
                comm_m,
                comm_u,
                chunk: chunk_pcs,
            });
        }

        Ok((all_groups, all_pcs_data, all_aux_f, all_chunk_f))
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

fn to_inner_mles<F: InnerTransparentField>(
    vecs: &[Vec<F>],
    num_vars: usize,
    cfg: &F::Config,
) -> Vec<DenseMultilinearExtension<F::Inner>>
where
    F::Inner: Clone + Default,
{
    let inner_zero = F::zero_with_cfg(cfg).inner().clone();
    vecs.iter()
        .map(|vals| {
            DenseMultilinearExtension::from_evaluations_vec(
                num_vars,
                vals.iter().map(|f| f.inner().clone()).collect(),
                inner_zero.clone(),
            )
        })
        .collect()
}

fn eval_inner_mles<F: InnerTransparentField>(
    mles: &[DenseMultilinearExtension<F::Inner>],
    point: &[F],
    cfg: &F::Config,
) -> Vec<F>
where
    DenseMultilinearExtension<F::Inner>: MultilinearExtensionWithConfig<F>,
{
    mles.iter()
        .map(|mle| {
            mle.clone()
                .evaluate_with_config(point, cfg)
                .expect("MLE evaluation should succeed")
        })
        .collect()
}

fn inner_mles_to_lifted<F: InnerTransparentField>(
    mles: &[DenseMultilinearExtension<F::Inner>],
    point: &[F],
    cfg: &F::Config,
) -> Vec<DynamicPolynomialF<F>>
where
    DenseMultilinearExtension<F::Inner>: MultilinearExtensionWithConfig<F>,
{
    eval_inner_mles(mles, point, cfg)
        .into_iter()
        .map(|eval| DynamicPolynomialF::new_trimmed(vec![eval]))
        .collect()
}
