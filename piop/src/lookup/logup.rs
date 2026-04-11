//! LogUp sumcheck group construction and verifier finalization.
//!
//! Provides building blocks for the LogUp lookup argument within the
//! Zinc+ multi-degree sumcheck pipeline. The LogUp identity is:
//!
//! ```text
//! Σ_i 1/(β − w_i) = Σ_j m_j/(β − T_j)
//! ```
//!
//! The prover commits multiplicities `m` and inverse witnesses
//! `u = 1/(β − w)` via Zip+ PCS. The table inverse `v = 1/(β − T)` is
//! NOT committed — the verifier computes it from the public table.
//!
//! ## γ-Batched sumcheck groups
//!
//! For L witness columns sharing the same table, a random challenge γ
//! collapses 2·L individual groups into exactly **2 groups**:
//!
//! - **Group 0** (zerocheck, degree 3): `(Σ_l γ^l · (d_l·u_l − 1)) · eq(r, y)`
//!   where `d_l = β − w_l`. Enforces `u_l = 1/(β − w_l)` pointwise for all L
//!   columns.
//! - **Group 1** (sumcheck, degree 2, claimed_sum = 0): `Σ_l γ^l · (u_l −
//!   m_l·v)`. Enforces `Σ u_l = Σ m_l·v` for all L columns simultaneously.
//!
//! When L=1 this degenerates to the unbatched case (γ^0 = 1).
//!
//! ## Committed model
//!
//! In the full pipeline (`protocol/src/prover.rs`), m and u are committed
//! via PCS and opened at `r_0` through the multipoint eval sumcheck.
//! The functions here build sumcheck groups and check subclaim evaluations.

use crypto_primitives::FromPrimitiveWithConfig;
use num_traits::Zero;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::marker::PhantomData;
use zinc_poly::{
    mle::DenseMultilinearExtension,
    utils::{build_eq_x_r_inner, build_eq_x_r_vec, eq_eval},
};
use zinc_transcript::traits::{ConstTranscribable, Transcript};
use zinc_uair::LookupTableType;
use zinc_utils::{cfg_iter, inner_transparent_field::InnerTransparentField, log2};

use crate::{
    lookup::{
        LogupFinalizerInput, LogupProverAncillary, LookupGroup, utils::batch_inverse_shifted,
    },
    sumcheck::multi_degree::MultiDegreeSumcheckGroup,
};

use super::{
    structs::{LogupVerifierPreSumcheckData, LookupError},
    utils::{generate_bitpoly_table, generate_word_table},
};

/// LogUp sumcheck group builder and verifier finalizer.
pub struct LogupProtocol<F: InnerTransparentField>(PhantomData<F>);

impl<F: InnerTransparentField + FromPrimitiveWithConfig + Send + Sync> LogupProtocol<F> {
    /// Build two γ-batched LogUp sumcheck groups from L columns.
    ///
    /// Takes L witnesses and their auxiliary vectors. Returns exactly
    /// 2 groups regardless of L: `[group_0, group_1]`
    ///
    /// - Group 0 (degree 3, zerocheck): `(Σ_l γ^l · (d_l·u_l − 1)) · eq(r, y)`
    /// - Group 1 (degree 2, sumcheck, claimed_sum = 0): `Σ_l γ^l · (u_l −
    ///   m_l·v)`
    ///
    /// When L=1, γ^0=1 so this degenerates to the unbatched case.
    /// The caller is responsible for PCS commitment, transcript operations,
    /// and squeezing β and r before calling this function.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn build_sumcheck_groups(
        witnesses: &[&[F]],
        table: &[F],
        auxs: &[LogupProverAncillary<'_, F>],
        beta: &F,
        gamma: &F,
        r: &[F],
        field_cfg: &F::Config,
    ) -> Result<Vec<MultiDegreeSumcheckGroup<F>>, LookupError>
    where
        F::Inner: Zero + Default + Send + Sync,
        F: 'static,
    {
        let num_cols = witnesses.len();
        assert_eq!(
            num_cols,
            auxs.len(),
            "witnesses and auxs must have same length"
        );

        let zero = F::zero_with_cfg(field_cfg);
        let one = F::one_with_cfg(field_cfg);

        let w_num_vars = if num_cols > 0 {
            log2(witnesses[0].len().next_power_of_two()) as usize
        } else {
            0
        };
        let t_num_vars = log2(table.len().next_power_of_two()) as usize;
        let num_vars = w_num_vars.max(t_num_vars);

        let eq_r = build_eq_x_r_inner(r, field_cfg)?;

        // ---- Build MLEs ----
        let inner_zero = zero.inner().clone();
        let mk_mle = |data: &[F]| -> DenseMultilinearExtension<F::Inner> {
            DenseMultilinearExtension::from_evaluations_vec(
                num_vars,
                cfg_iter!(data).map(|x| x.inner().clone()).collect(),
                inner_zero.clone(),
            )
        };

        let beta_inner = beta.inner();

        let gamma_pows = zinc_utils::powers(gamma.clone(), one.clone(), num_cols);

        // Group 0 MLEs: [eq, d_0, u_0, d_1, u_1, ...] where
        // d = (β − w): denominator for witness inverse check
        let mut group0_mles = Vec::with_capacity(1 + 2 * num_cols);
        group0_mles.push(eq_r);
        for (witness, aux) in witnesses.iter().zip(auxs.iter()) {
            let d_mle = DenseMultilinearExtension::from_evaluations_vec(
                num_vars,
                cfg_iter!(*witness)
                    .map(|w_i| F::sub_inner(beta_inner, w_i.inner(), field_cfg))
                    .collect(),
                inner_zero.clone(),
            );
            group0_mles.push(d_mle);
            group0_mles.push(mk_mle(aux.inverse_witness));
        }

        // Group 1 MLEs: [u_0, m_0, u_1, m_1, ..., v]
        let mut group1_mles = Vec::with_capacity(2 * num_cols + 1);
        for aux in auxs.iter() {
            group1_mles.push(mk_mle(aux.inverse_witness));
            group1_mles.push(mk_mle(aux.multiplicities));
        }
        group1_mles.push(mk_mle(auxs[0].inverse_table));

        // group0: (d_l·u_l − 1) · eq where d_l = (β − w_l)
        let one0 = one.clone();
        let zero0 = zero.clone();
        let gamma_powers_copy0 = gamma_pows.clone();
        let l0 = num_cols;
        let group_0 = MultiDegreeSumcheckGroup::new(
            3,
            group0_mles,
            Box::new(move |v: &[F]| {
                // v[0] = eq, v[1+2l] = d_l, v[2+2l] = u_l
                let mut sum = zero0.clone();
                for l in 0..l0 {
                    let d = &v[1 + 2 * l];
                    let u = &v[2 + 2 * l];
                    let term = d.clone() * u - &one0;
                    sum += &(gamma_powers_copy0[l].clone() * &term);
                }
                sum * &v[0]
            }),
        );

        // group1: (u − m·v)
        let zero1 = zero;
        let gp1 = gamma_pows;
        let l1 = num_cols;
        let group_1 = MultiDegreeSumcheckGroup::new(
            2,
            group1_mles,
            Box::new(move |v: &[F]| {
                // v[2l] = u_l, v[2l+1] = m_l, v[2*L] = v (shared)
                let v_shared = &v[2 * l1];
                let mut sum = zero1.clone();
                for l in 0..l1 {
                    let u = &v[2 * l];
                    let m = &v[2 * l + 1];
                    let term = u.clone() - &(m.clone() * v_shared);
                    sum += &(gp1[l].clone() * &term);
                }
                sum
            }),
        );

        Ok(vec![group_0, group_1])
    }

    /// Extract witness columns and the shared lookup table for a group.
    ///
    /// Returns `(witnesses, table)` where `witnesses[i]` corresponds to
    /// `group_info.column_indices[i]`. The table is generated from
    /// `group_info.table_type`.
    pub fn extract_witnesses_and_table(
        projected_trace_f: &[DenseMultilinearExtension<F::Inner>],
        group_info: &super::LookupGroup,
        projecting_element_f: &F,
        field_cfg: &F::Config,
    ) -> (Vec<Vec<F>>, Vec<F>) {
        let witnesses: Vec<Vec<F>> = group_info
            .column_indices
            .iter()
            .map(|&col_idx| {
                projected_trace_f[col_idx]
                    .iter()
                    .map(|inner| F::new_unchecked_with_cfg(inner.clone(), field_cfg))
                    .collect()
            })
            .collect();

        let table: Vec<F> = match &group_info.table_type {
            LookupTableType::BitPoly { width, .. } => {
                generate_bitpoly_table(*width, projecting_element_f, field_cfg)
            }
            LookupTableType::Word { width, .. } => generate_word_table(*width, field_cfg),
        };

        (witnesses, table)
    }

    /// Pre-sumcheck transcript sync for LogUp.
    ///
    /// Absorbs PCS commitment roots into the transcript
    /// and squeezes the challenges β and r, mirroring the prover's
    /// `prepare_lookup_groups`. Must run BEFORE the multi-degree sumcheck
    /// verify to keep the transcript in sync.
    pub fn build_verifier_pre_sumcheck(
        transcript: &mut impl Transcript,
        comm_m_root: &[u8],
        comm_u_root: &[u8],
        num_vars: usize,
        field_cfg: &F::Config,
    ) -> LogupVerifierPreSumcheckData<F>
    where
        F::Inner: ConstTranscribable,
    {
        // Absorb multiplicity commitment root, squeeze β
        transcript.absorb_slice(comm_m_root);
        let beta: F = transcript.get_field_challenge(field_cfg);

        // Absorb inverse witness root, squeeze r then γ
        transcript.absorb_slice(comm_u_root);
        let r: Vec<F> = transcript.get_field_challenges(num_vars, field_cfg);
        let gamma: F = transcript.get_field_challenge(field_cfg);

        LogupVerifierPreSumcheckData { r, beta, gamma }
    }

    /// Post-sumcheck finalization for the γ-batched LogUp verifier.
    ///
    /// Given the subclaim point `x*` and two expected evaluations (one
    /// per γ-batched group), checks:
    ///
    /// - Group 0: `Σ_l γ^l · (d_l·u_l − 1) · eq_val == expected[0]` where d_l =
    ///   (β − w_l)
    /// - Group 1: `Σ_l γ^l · (u_l − m_l·v_eval) == expected[1]`
    ///
    /// `v_eval` is computed once from the public table + β.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn finalize_verifier(
        pre_sumcheck_data: &LogupVerifierPreSumcheckData<F>,
        input: LogupFinalizerInput<'_, F>,
        group_info: &LookupGroup,
        projecting_element_f: &F,
        field_cfg: &F::Config,
    ) -> Result<(), LookupError>
    where
        F::Inner: ConstTranscribable + Zero,
    {
        let one = F::one_with_cfg(field_cfg);
        let LogupFinalizerInput {
            subclaim_point,
            expected_evaluations,
            w_evals,
            aux_evals,
        } = input;
        let LogupVerifierPreSumcheckData { r, beta, gamma } = pre_sumcheck_data;
        let num_cols = w_evals.len();
        assert_eq!(num_cols, aux_evals.len());

        // Evaluate eq(x*, r)
        let eq_val = eq_eval(subclaim_point, r, one.clone())?;

        // Compute v_eval from public table
        let table: Vec<F> = match &group_info.table_type {
            LookupTableType::BitPoly { width, .. } => {
                generate_bitpoly_table(*width, projecting_element_f, field_cfg)
            }
            LookupTableType::Word { width, .. } => generate_word_table(*width, field_cfg),
        };

        let eq_at_point = build_eq_x_r_vec(subclaim_point, field_cfg)?;
        let zero = F::zero_with_cfg(field_cfg);

        let v = batch_inverse_shifted(beta, &table);
        let v_eval: F = v
            .iter()
            .zip(eq_at_point.iter())
            .fold(zero.clone(), |acc, (v, e)| acc + &(v.clone() * e));

        let gamma_pows = zinc_utils::powers(gamma.clone(), one.clone(), num_cols);

        // Group 0: Σ_l γ^l · (d_l·u_l − 1) · eq_val, where d_l = (β − w_l)
        let mut computed_0 = zero.clone();
        for l in 0..num_cols {
            let w_l = &w_evals[l];
            let u_l = &aux_evals[l].u_eval;
            let term = (beta.clone() - w_l) * u_l - &one;
            computed_0 += &(gamma_pows[l].clone() * &term);
        }
        computed_0 *= &eq_val;
        if computed_0 != expected_evaluations[0] {
            return Err(LookupError::FinalEvaluationMismatch);
        }

        // Group 1: Σ_l γ^l · (u_l − m_l · v_eval)
        let mut computed_1 = zero;
        for l in 0..num_cols {
            let u_l = &aux_evals[l].u_eval;
            let m_l = &aux_evals[l].m_eval;
            let term = u_l.clone() - &(m_l.clone() * &v_eval);
            computed_1 += &(gamma_pows[l].clone() * &term);
        }
        if computed_1 != expected_evaluations[1] {
            return Err(LookupError::FinalEvaluationMismatch);
        }

        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::arithmetic_side_effects, clippy::cast_possible_truncation)]
mod tests {
    use super::*;
    use crate::sumcheck::multi_degree::MultiDegreeSumcheck;
    use crypto_bigint::{U128, const_monty_params};
    use crypto_primitives::crypto_bigint_const_monty::ConstMontyField;
    use num_traits::ConstZero;
    use zinc_poly::mle::MultilinearExtensionWithConfig;
    use zinc_transcript::Blake3Transcript;

    use super::super::{
        LogupFinalizerInput, LookupAuxEvals, LookupGroup,
        utils::{batch_inverse_shifted, compute_multiplicities},
    };

    const N: usize = 2;
    const_monty_params!(TestParams, U128, "00000000b933426489189cb5b47d567f");
    type F = ConstMontyField<TestParams, N>;

    fn make_transcript() -> Blake3Transcript {
        let mut t = Blake3Transcript::default();
        t.absorb_slice(b"logup-test");
        t
    }

    struct LogupTestHarness {
        w_evals: Vec<F>,
        aux_evals: Vec<LookupAuxEvals<F>>,
        pre: LogupVerifierPreSumcheckData<F>,
        expected_evaluations: Vec<F>,
        x_star: Vec<F>,
        group_info: LookupGroup,
    }

    fn setup_logup_harness(witnesses: &[Vec<F>], table: &[F]) -> LogupTestHarness {
        let num_vars = zinc_utils::log2(witnesses[0].len().next_power_of_two()) as usize;
        let num_cols = witnesses.len();
        let beta = F::from(17u32);
        let gamma = F::from(7u32);

        let mut m_vecs = Vec::with_capacity(num_cols);
        let mut u_vecs = Vec::with_capacity(num_cols);
        let v = batch_inverse_shifted(&beta, table);

        for w in witnesses {
            let m = compute_multiplicities(w, table, &()).expect("witness should be in table");
            let u = batch_inverse_shifted(&beta, w);
            m_vecs.push(m);
            u_vecs.push(u);
        }

        let r: Vec<F> = (0..num_vars).map(|i| F::from((i + 3) as u32)).collect();

        let witness_refs: Vec<&[F]> = witnesses.iter().map(|w| w.as_slice()).collect();
        let auxs: Vec<super::super::LogupProverAncillary<'_, F>> = m_vecs
            .iter()
            .zip(u_vecs.iter())
            .map(|(m, u)| super::super::LogupProverAncillary {
                multiplicities: m,
                inverse_witness: u,
                inverse_table: &v,
            })
            .collect();

        let groups = LogupProtocol::build_sumcheck_groups(
            &witness_refs,
            table,
            &auxs,
            &beta,
            &gamma,
            &r,
            &(),
        )
        .expect("build_sumcheck_groups should succeed");

        assert_eq!(groups.len(), 2);

        let mut pt = make_transcript();
        let (md_proof, _) =
            MultiDegreeSumcheck::prove_as_subprotocol(&mut pt, groups, num_vars, &());
        let mut vt = make_transcript();
        let md_sub = MultiDegreeSumcheck::verify_as_subprotocol(&mut vt, num_vars, &md_proof, &())
            .expect("sumcheck verify should succeed");

        let x_star = md_sub.point().to_vec();
        let inner_zero = F::ZERO.into_inner();
        let mk_mle = |data: &[F]| -> DenseMultilinearExtension<_> {
            DenseMultilinearExtension::from_evaluations_vec(
                num_vars,
                data.iter().map(|f| f.into_inner()).collect(),
                inner_zero,
            )
        };

        let w_evals: Vec<F> = witnesses
            .iter()
            .map(|w| mk_mle(w).evaluate_with_config(&x_star, &()).unwrap())
            .collect();
        let aux_evals: Vec<LookupAuxEvals<F>> = m_vecs
            .iter()
            .zip(u_vecs.iter())
            .map(|(m, u)| LookupAuxEvals {
                u_eval: mk_mle(u).evaluate_with_config(&x_star, &()).unwrap(),
                m_eval: mk_mle(m).evaluate_with_config(&x_star, &()).unwrap(),
            })
            .collect();

        let table_width = zinc_utils::log2(table.len().next_power_of_two()) as usize;
        LogupTestHarness {
            w_evals,
            aux_evals,
            pre: LogupVerifierPreSumcheckData { r, beta, gamma },
            expected_evaluations: md_sub.expected_evaluations().to_vec(),
            x_star,
            group_info: LookupGroup {
                table_type: LookupTableType::Word {
                    width: table_width,
                    chunk_width: None,
                },
                column_indices: (0..num_cols).collect(),
            },
        }
    }

    impl LogupTestHarness {
        fn finalize_with(
            &self,
            w_evals: &[F],
            aux_evals: &[LookupAuxEvals<F>],
        ) -> Result<(), LookupError> {
            let fin = LogupFinalizerInput {
                subclaim_point: &self.x_star,
                expected_evaluations: &self.expected_evaluations,
                w_evals,
                aux_evals,
            };
            LogupProtocol::<F>::finalize_verifier(&self.pre, fin, &self.group_info, &F::ZERO, &())
        }

        fn finalize_honest(&self) -> Result<(), LookupError> {
            self.finalize_with(&self.w_evals, &self.aux_evals)
        }
    }

    /// Full γ-batched LogUp pipeline: `setup_logup_harness` then honest
    /// finalize.
    fn run_logup_roundtrip(witnesses: &[Vec<F>], table: &[F]) {
        setup_logup_harness(witnesses, table)
            .finalize_honest()
            .expect("finalize_verifier should succeed");
    }

    #[test]
    fn logup_roundtrip_small() {
        let table: Vec<F> = (0..4u32).map(F::from).collect();
        let witness: Vec<F> = vec![0u32, 1, 1, 3].into_iter().map(F::from).collect();
        run_logup_roundtrip(&[witness], &table);
    }

    #[test]
    fn logup_roundtrip_all_same() {
        let table: Vec<F> = (0..4u32).map(F::from).collect();
        let witness: Vec<F> = vec![2u32; 4].into_iter().map(F::from).collect();
        run_logup_roundtrip(&[witness], &table);
    }

    #[test]
    fn logup_roundtrip_full_table() {
        let table: Vec<F> = (0..8u32).map(F::from).collect();
        let witness: Vec<F> = (0..8u32).map(F::from).collect();
        run_logup_roundtrip(&[witness], &table);
    }

    #[test]
    fn logup_roundtrip_multi_column() {
        let table: Vec<F> = (0..4u32).map(F::from).collect();
        let w0: Vec<F> = vec![0u32, 1, 1, 3].into_iter().map(F::from).collect();
        let w1: Vec<F> = vec![3u32, 3, 0, 2].into_iter().map(F::from).collect();
        let w2: Vec<F> = vec![1u32, 2, 2, 0].into_iter().map(F::from).collect();
        run_logup_roundtrip(&[w0, w1, w2], &table);
    }

    #[test]
    fn logup_reject_invalid_witness() {
        let table: Vec<F> = (0..4u32).map(F::from).collect();
        let witness: Vec<F> = vec![0u32, 5].into_iter().map(F::from).collect();

        let result = compute_multiplicities(&witness, &table, &());
        assert!(
            result.is_none(),
            "witness entry 5 not in table should return None"
        );
    }

    fn single_col_harness() -> LogupTestHarness {
        let table: Vec<F> = (0..4u32).map(F::from).collect();
        let witness: Vec<F> = vec![0u32, 1, 1, 3].into_iter().map(F::from).collect();
        setup_logup_harness(&[witness], &table)
    }

    fn two_col_harness() -> LogupTestHarness {
        let table: Vec<F> = (0..4u32).map(F::from).collect();
        let w0: Vec<F> = vec![0u32, 1, 1, 3].into_iter().map(F::from).collect();
        let w1: Vec<F> = vec![3u32, 3, 0, 2].into_iter().map(F::from).collect();
        setup_logup_harness(&[w0, w1], &table)
    }

    #[test]
    fn logup_wrong_u_eval_rejected() {
        let h = single_col_harness();
        assert!(h.finalize_honest().is_ok());
        let mut bad = h.aux_evals.clone();
        bad[0].u_eval += F::from(1u32);
        assert!(h.finalize_with(&h.w_evals, &bad).is_err());
    }

    #[test]
    fn logup_wrong_m_eval_rejected() {
        let h = single_col_harness();
        let mut bad = h.aux_evals.clone();
        bad[0].m_eval += F::from(1u32);
        assert!(h.finalize_with(&h.w_evals, &bad).is_err());
    }

    #[test]
    fn logup_wrong_w_eval_rejected() {
        let h = single_col_harness();
        let mut bad_w = h.w_evals.clone();
        bad_w[0] += F::from(1u32);
        assert!(h.finalize_with(&bad_w, &h.aux_evals).is_err());
    }

    #[test]
    fn logup_wrong_gamma_rejected() {
        let mut h = two_col_harness();
        assert!(h.finalize_honest().is_ok());
        h.pre.gamma = F::from(999u32);
        assert!(h.finalize_honest().is_err());
    }

    #[test]
    fn logup_multi_col_wrong_single_aux_rejected() {
        let h = two_col_harness();
        assert!(h.finalize_honest().is_ok());
        let mut bad = h.aux_evals.clone();
        bad[1].m_eval += F::from(1u32);
        assert!(h.finalize_with(&h.w_evals, &bad).is_err());
    }
}
