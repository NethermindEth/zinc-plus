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
//! ## Sumcheck groups
//!
//! Two groups are produced per witness column:
//!
//! - **Group 0** (zerocheck, degree 3): `(d·u − 1)·eq(r, y)` where `d = β − w`.
//!   Enforces `u = 1/(β − w)` pointwise.
//! - **Group 1** (sumcheck, degree 2): `u(y) − m(y)·v(y)`. Enforces `Σ u = Σ
//!   m·v` (the LogUp sum identity). this is sumcheck with claimed sum = 0.
//!
//! The verifier checks `claimed_sum == 0` for both groups, then
//! verifies the subclaim evaluations at the shared point `x*`.
//!
//! ## Committed model
//!
//! In the full pipeline (`protocol/src/prover.rs`), m and u are committed
//! via PCS and opened at `r_0` through the multipoint eval sumcheck.
//! The functions here are build sumcheck groups and check subclaim evaluations.

use crypto_primitives::FromPrimitiveWithConfig;
use num_traits::Zero;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::marker::PhantomData;
use zinc_poly::{
    mle::DenseMultilinearExtension,
    utils::{build_eq_x_r_inner, build_eq_x_r_vec},
};
use zinc_transcript::traits::{ConstTranscribable, Transcript};
use zinc_uair::LookupTableType;
use zinc_utils::{cfg_iter, inner_transparent_field::InnerTransparentField};

use crate::{
    lookup::{
        LogupFinalizerInput, LogupProverAncillary, LookupGroup, utils::batch_inverse_shifted,
    },
    sumcheck::multi_degree::MultiDegreeSumcheckGroup,
};

use super::{
    structs::{LogupVerifierPreSumcheckData, LookupAuxEvals, LookupError},
    utils::{generate_bitpoly_table, generate_word_table},
};

/// LogUp sumcheck group builder and verifier finalizer.
pub struct LogupProtocol<F: InnerTransparentField>(PhantomData<F>);

impl<F: InnerTransparentField + FromPrimitiveWithConfig + Send + Sync> LogupProtocol<F> {
    /// Build two LogUp sumcheck groups from pre-computed vectors.
    ///
    /// Returns `[group_0, group_1]` where:
    /// - Group 0 (degree 3): `(d·u − 1)·eq(r, y)` — zerocheck for inverse
    ///   correctness.
    /// - Group 1 (degree 2): `u − m·v` — sumcheck for the LogUp sum identity
    ///   with claimed_sum = 0.
    ///
    /// The caller is responsible for PCS commitment, transcript operations,
    /// and squeezing β and r before calling this function.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn build_sumcheck_groups(
        witness: &[F],
        table: &[F],
        aux: &LogupProverAncillary<'_, F>,
        beta: &F,
        r: &[F],
        field_cfg: &F::Config,
    ) -> Result<Vec<MultiDegreeSumcheckGroup<F>>, LookupError>
    where
        F::Inner: Zero + Default + Send + Sync,
        F: 'static,
    {
        let zero = F::zero_with_cfg(field_cfg);
        let one = F::one_with_cfg(field_cfg);

        let w_num_vars = zinc_utils::log2(witness.len().next_power_of_two()) as usize;
        let t_num_vars = zinc_utils::log2(table.len().next_power_of_two()) as usize;
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

        // d = (β − w): denominator for witness inverse check
        let d_mle = DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            cfg_iter!(witness)
                .map(|w_i| F::sub_inner(beta_inner, w_i.inner(), field_cfg))
                .collect(),
            inner_zero.clone(),
        );

        let u_mle = mk_mle(aux.inverse_witness);
        let v_mle = mk_mle(aux.inverse_table);
        let m_mle = mk_mle(aux.multiplicities);

        // group0: (d·u − 1) · eq where d = (β − w)
        let one0 = one.clone();
        let group_0 = MultiDegreeSumcheckGroup::new(
            3,
            vec![eq_r.clone(), d_mle, u_mle.clone()],
            Box::new(move |v: &[F]| (v[1].clone() * &v[2] - &one0) * &v[0]),
        );

        // group1: (u − m·v)
        let group_1 = MultiDegreeSumcheckGroup::new(
            2,
            vec![u_mle, m_mle.clone(), v_mle.clone()],
            Box::new(move |v: &[F]| v[0].clone() - &(v[1].clone() * &v[2])),
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
        use super::utils::{generate_bitpoly_table, generate_word_table};
        use zinc_uair::LookupTableType;

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

        // Absorb inverse witness root, squeeze r
        transcript.absorb_slice(comm_u_root);
        let r: Vec<F> = transcript.get_field_challenges(num_vars, field_cfg);

        LogupVerifierPreSumcheckData { r, beta }
    }

    /// Post-sumcheck finalization for the LogUp verifier.
    ///
    /// Given the subclaim point `x*` and the two expected evaluations
    /// from the multi-degree sumcheck, verifies both LogUp identities:
    ///
    /// - Group 0: `((β − w_eval)·u_eval − 1)·eq_val == expected[0]`
    /// - Group 1: `u_eval − m_eval·v_eval == expected[1]`
    ///
    /// `v_eval` is computed internally from the public table + β (not
    /// from the proof) — the table inverse is never committed.
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
        let LogupVerifierPreSumcheckData { r, beta } = pre_sumcheck_data;
        let LogupFinalizerInput {
            subclaim_point,
            expected_evaluations,
            w_eval,
            aux_evals: LookupAuxEvals { u_eval, m_eval },
        } = input;

        // Evaluate eq(x*, r)
        let eq_val = zinc_poly::utils::eq_eval(subclaim_point, r, one.clone())?;

        // Evaluate (β − T) at x*: the verifier regenerates the table
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
            .fold(zero, |acc, (v, e)| acc + &(v.clone() * e));

        // Group 0: inverse correctness — (d·u − 1)·eq where d = (β − w)
        let computed_0 = ((beta.clone() - w_eval) * u_eval - &one) * &eq_val;
        if computed_0 != expected_evaluations[0] {
            return Err(LookupError::FinalEvaluationMismatch);
        }

        // Group 1: (u − m·v)
        let computed_1 = u_eval.clone() - &(m_eval.clone() * &v_eval);
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
        LogupFinalizerInput, LookupGroup,
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

    /// Run the full LogUp pipeline on a single (witness, table) pair:
    /// compute aux → build sumcheck groups → run multi-degree sumcheck →
    /// finalize_verifier identity checks.
    fn run_logup_roundtrip(witness: &[F], table: &[F]) {
        let num_vars = zinc_utils::log2(witness.len().next_power_of_two()) as usize;

        let m = compute_multiplicities(witness, table, &()).expect("witness should be in table");
        let beta = F::from(17u32);
        let u = batch_inverse_shifted(&beta, witness);
        let v = batch_inverse_shifted(&beta, table);

        let r: Vec<F> = (0..num_vars).map(|i| F::from((i + 3) as u32)).collect();

        let aux = super::super::LogupProverAncillary {
            multiplicities: &m,
            inverse_witness: &u,
            inverse_table: &v,
        };
        let groups = LogupProtocol::build_sumcheck_groups(witness, table, &aux, &beta, &r, &())
            .expect("build_sumcheck_groups should succeed");

        assert_eq!(groups.len(), 2);

        // Run the multi-degree sumcheck (prover side)
        let mut prover_transcript = make_transcript();
        let (md_proof, _md_states) = MultiDegreeSumcheck::prove_as_subprotocol(
            &mut prover_transcript,
            groups,
            num_vars,
            &(),
        );

        // All the claimed sums should be zero (the identities sum to 0
        // over the boolean hypercube when witness exists in table).
        for (g, cs) in md_proof.claimed_sums().iter().enumerate() {
            assert_eq!(*cs, F::ZERO, "group {g} claimed sum should be zero");
        }

        // Verify the sumcheck
        let mut verifier_transcript = make_transcript();
        let md_subclaims = MultiDegreeSumcheck::verify_as_subprotocol(
            &mut verifier_transcript,
            num_vars,
            &md_proof,
            &(),
        )
        .expect("sumcheck verify should succeed");

        // Compute w_eval, u_eval, m_eval at subclaim point x*
        let x_star = md_subclaims.point();
        let inner_zero = F::ZERO.into_inner();
        let mk_mle = |data: &[F]| -> DenseMultilinearExtension<_> {
            DenseMultilinearExtension::from_evaluations_vec(
                num_vars,
                data.iter().map(|f| f.into_inner()).collect(),
                inner_zero,
            )
        };

        let w_eval: F = mk_mle(witness)
            .evaluate_with_config(x_star, &())
            .expect("eval should succeed");
        let u_eval: F = mk_mle(&u)
            .evaluate_with_config(x_star, &())
            .expect("eval should succeed");
        let m_eval: F = mk_mle(&m)
            .evaluate_with_config(x_star, &())
            .expect("eval should succeed");

        let table_width = zinc_utils::log2(table.len().next_power_of_two()) as usize;
        let group_info = LookupGroup {
            table_type: LookupTableType::Word {
                width: table_width,
                chunk_width: None,
            },
            column_indices: vec![0],
        };

        let pre = LogupVerifierPreSumcheckData { r, beta };
        let aux_evals = LookupAuxEvals { u_eval, m_eval };
        let fin_input = LogupFinalizerInput {
            subclaim_point: x_star,
            expected_evaluations: md_subclaims.expected_evaluations(),
            w_eval: &w_eval,
            aux_evals: &aux_evals,
        };

        LogupProtocol::<F>::finalize_verifier(&pre, fin_input, &group_info, &F::ZERO, &())
            .expect("finalize_verifier should succeed");
    }

    #[test]
    fn logup_roundtrip_small() {
        let table: Vec<F> = (0..4u32).map(F::from).collect();
        let witness: Vec<F> = vec![0u32, 1, 1, 3].into_iter().map(F::from).collect();
        run_logup_roundtrip(&witness, &table);
    }

    #[test]
    fn logup_roundtrip_all_same() {
        let table: Vec<F> = (0..4u32).map(F::from).collect();
        let witness: Vec<F> = vec![2u32; 4].into_iter().map(F::from).collect();
        run_logup_roundtrip(&witness, &table);
    }

    #[test]
    fn logup_roundtrip_full_table() {
        let table: Vec<F> = (0..8u32).map(F::from).collect();
        let witness: Vec<F> = (0..8u32).map(F::from).collect();
        run_logup_roundtrip(&witness, &table);
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

    #[test]
    fn logup_wrong_aux_eval_rejected() {
        let table: Vec<F> = (0..4u32).map(F::from).collect();
        let witness: Vec<F> = vec![0u32, 1, 1, 3].into_iter().map(F::from).collect();
        let num_vars = 2;

        let m = compute_multiplicities(&witness, &table, &()).unwrap();
        let beta = F::from(17u32);
        let u = batch_inverse_shifted(&beta, &witness);
        let v = batch_inverse_shifted(&beta, &table);
        let r: Vec<F> = (0..num_vars).map(|i| F::from((i + 3) as u32)).collect();

        let aux = super::super::LogupProverAncillary {
            multiplicities: &m,
            inverse_witness: &u,
            inverse_table: &v,
        };
        let groups =
            LogupProtocol::build_sumcheck_groups(&witness, &table, &aux, &beta, &r, &()).unwrap();

        let mut prover_transcript = make_transcript();
        let (md_proof, _) = MultiDegreeSumcheck::prove_as_subprotocol(
            &mut prover_transcript,
            groups,
            num_vars,
            &(),
        );

        let mut verifier_transcript = make_transcript();
        let md_subclaims = MultiDegreeSumcheck::verify_as_subprotocol(
            &mut verifier_transcript,
            num_vars,
            &md_proof,
            &(),
        )
        .unwrap();

        let x_star = md_subclaims.point();
        let inner_zero = F::ZERO.into_inner();
        let mk_mle = |data: &[F]| -> DenseMultilinearExtension<_> {
            DenseMultilinearExtension::from_evaluations_vec(
                num_vars,
                data.iter().map(|f| f.into_inner()).collect(),
                inner_zero,
            )
        };
        let w_eval: F = mk_mle(&witness).evaluate_with_config(x_star, &()).unwrap();
        let u_eval: F = mk_mle(&u).evaluate_with_config(x_star, &()).unwrap();
        let m_eval: F = mk_mle(&m).evaluate_with_config(x_star, &()).unwrap();

        let group_info = LookupGroup {
            table_type: LookupTableType::Word {
                width: 2,
                chunk_width: None,
            },
            column_indices: vec![0],
        };
        let pre = LogupVerifierPreSumcheckData { r, beta };

        // Corrupt u_eval
        let bad_aux = LookupAuxEvals {
            u_eval: u_eval + F::from(1u32),
            m_eval,
        };
        let fin_input = LogupFinalizerInput {
            subclaim_point: x_star,
            expected_evaluations: md_subclaims.expected_evaluations(),
            w_eval: &w_eval,
            aux_evals: &bad_aux,
        };
        let result =
            LogupProtocol::<F>::finalize_verifier(&pre, fin_input, &group_info, &F::ZERO, &());
        assert!(result.is_err(), "corrupted u_eval should be rejected");
    }
}
