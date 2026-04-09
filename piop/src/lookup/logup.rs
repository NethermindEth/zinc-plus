//! Core LogUp protocol implementation.
//!
//! Implements the LogUp PIOP for proving that every entry of a witness
//! vector belongs to a prescribed lookup table.
//!
//! ## Protocol overview
//!
//! Given witness `w ∈ F_q^W` and table `T ∈ F_q^N`, the LogUp protocol
//! proves `w_i ∈ {T_j}` for all `i` via a log-derivative identity:
//!
//! ```text
//! Σ_i 1/(β − w_i) = Σ_j m_j/(β − T_j)
//! ```
//!
//! The prover sends multiplicities `m`, inverse vectors `u`, `v` in the
//! clear, then runs a batched sumcheck to verify correctness.

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
use zinc_utils::{cfg_iter, inner_transparent_field::InnerTransparentField};

use crate::sumcheck::MLSumcheck;

use super::{
    structs::{LogupProof, LogupProverState, LogupVerifierSubClaim, LookupError},
    tables::{batch_inverse, batch_inverse_shifted, compute_multiplicities},
};

/// The core LogUp protocol.
pub struct LogupProtocol<F: InnerTransparentField>(PhantomData<F>);

impl<F: InnerTransparentField + FromPrimitiveWithConfig + Send + Sync> LogupProtocol<F> {
    /// Prover for the LogUp protocol.
    ///
    /// # Arguments
    ///
    /// - `transcript`: Fiat-Shamir transcript.
    /// - `witness`: The witness vector as field elements (projected trace
    ///   column). Length `W` (will be padded to next power of two).
    /// - `table`: The lookup table entries. Length `N` (must be a power of
    ///   two).
    /// - `field_cfg`: Field configuration.
    ///
    /// # Returns
    ///
    /// `(LogupProof, LogupProverState)` on success, or a `LookupError`.
    #[allow(clippy::arithmetic_side_effects, clippy::too_many_arguments)]
    pub fn prove_as_subprotocol(
        transcript: &mut impl Transcript,
        witness: &[F],
        table: &[F],
        field_cfg: &F::Config,
    ) -> Result<(LogupProof<F>, LogupProverState<F>), LookupError<F>>
    where
        F::Inner: ConstTranscribable + Zero + Default + Send + Sync,
        F::Modulus: ConstTranscribable,
    {
        let witness_len = witness.len();
        let zero = F::zero_with_cfg(field_cfg);
        let one = F::one_with_cfg(field_cfg);

        // ---- Step 1: Compute multiplicities ----
        let multiplicities = compute_multiplicities(witness, table, field_cfg)
            .ok_or(LookupError::WitnessNotInTable)?;

        // ---- Step 2: Absorb multiplicities into transcript ----
        let mut buf = vec![0u8; F::Inner::NUM_BYTES];
        transcript.absorb_random_field_slice(&multiplicities, &mut buf);

        // ---- Step 3: Get challenge β ----
        let beta: F = transcript.get_field_challenge(field_cfg);

        // ---- Step 4: Compute inverse vectors ----
        // u_i = 1 / (β − w_i)  — fused subtraction + batch inverse
        let inverse_witness = batch_inverse_shifted(&beta, witness);

        // v_j = 1 / (β − T_j)  — fused subtraction + batch inverse
        let inverse_table = batch_inverse_shifted(&beta, table);

        // ---- Step 5: Absorb inverse vectors into transcript ----
        transcript.absorb_random_field_slice(&inverse_witness, &mut buf);
        transcript.absorb_random_field_slice(&inverse_table, &mut buf);

        // ---- Step 6: Get batching challenge γ and random evaluation point r ----
        let gamma: F = transcript.get_field_challenge(field_cfg);

        // We run the sumcheck over the larger of the two domains.
        // Pad witness-side and table-side MLEs to a common hypercube.
        let w_num_vars = zinc_utils::log2(witness_len.next_power_of_two()) as usize;
        let t_num_vars = zinc_utils::log2(table.len().next_power_of_two()) as usize;
        let num_vars = w_num_vars.max(t_num_vars);

        // Get a random evaluation point for the eq polynomial.
        let r: Vec<F> = transcript.get_field_challenges(num_vars, field_cfg);
        let eq_r = build_eq_x_r_inner(&r, field_cfg)?;

        // ---- Step 7: Build MLEs for the sumcheck ----
        // Precompute (β − w) once as a vector so the combination
        // function avoids cloning β at every evaluation point.
        //
        // MLE layout (table inverse checked directly, not via sumcheck):
        //   0: eq(y, r)
        //   1: d̃(y) = β − w̃(y) — precomputed shifted witness
        //   2: ũ(y)             — inverse witness MLE
        //   3: ṽ(y)             — inverse table MLE
        //   4: m̃(y)             — multiplicity MLE

        let inner_zero = F::zero_with_cfg(field_cfg).inner().clone();

        let mk_mle = |data: &[F]| -> DenseMultilinearExtension<F::Inner> {
            DenseMultilinearExtension::from_evaluations_vec(
                num_vars,
                cfg_iter!(data).map(|x| x.inner().clone()).collect(),
                inner_zero.clone(),
            )
        };

        // d_i = β − w_i  computed directly at the Inner level to avoid
        // allocating intermediate Vec<F> elements.
        let beta_inner = beta.inner();
        let d_mle = DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            cfg_iter!(witness)
                .map(|w_i| F::sub_inner(beta_inner, w_i.inner(), field_cfg))
                .collect(),
            inner_zero.clone(),
        );

        let u_mle = mk_mle(&inverse_witness);
        let v_mle = mk_mle(&inverse_table);
        let m_mle = mk_mle(&multiplicities);

        let mles = vec![eq_r, d_mle, u_mle, v_mle, m_mle];

        // ---- Step 8: Run batched sumcheck ----
        // Two identities batched with γ (table inverse checked
        // directly by the verifier, saving one MLE + one identity):
        //
        //   γ^0 · [d̃(y) · ũ(y) − 1] · eq(y, r)            (witness inverse correctness)
        //   γ^1 · [ũ(y) − m̃(y) · ṽ(y)] · eq(y, r)          (log-derivative balance)
        //
        // Total claimed sum = 0.

        let gamma_clone = gamma.clone();
        let one_clone = one.clone();

        let comb_fn = move |vals: &[F]| -> F {
            let eq_val = &vals[0];
            let d_val = &vals[1]; // β − w (precomputed)
            let u_val = &vals[2];
            let v_val = &vals[3];
            let m_val = &vals[4];

            // Identity 1: d · u − 1  (d = β − w, no β clone needed)
            let id1 = d_val.clone() * u_val - &one_clone;
            // Identity 2: u − m · v
            let id2 = u_val.clone() - &(m_val.clone() * v_val);

            // Batch: γ^0 · id1 + γ^1 · id2, all multiplied by eq(y, r)
            let batched = id1 + id2 * &gamma_clone;
            batched * eq_val
        };

        // degree: eq is linear (deg 1), each identity has degree 2
        // total degree = 1 + 2 = 3
        let degree = 3;

        let (sumcheck_proof, sumcheck_prover_state) = MLSumcheck::prove_as_subprotocol(
            transcript,
            mles,
            num_vars,
            degree,
            comb_fn,
            field_cfg,
        );

        // ---- Step 9: Verify multiplicity sum = W (prover side) ----
        // The verifier will check this directly, but we include it for
        // completeness.
        debug_assert!({
            let sum: F = multiplicities.iter().cloned().fold(zero.clone(), |a, b| a + &b);
            sum == F::from_with_cfg(witness_len as u64, field_cfg)
        });

        Ok((
            LogupProof {
                multiplicities,
                inverse_witness,
                inverse_table,
                sumcheck_proof,
            },
            LogupProverState {
                evaluation_point: sumcheck_prover_state.randomness,
            },
        ))
    }

    /// Verifier for the LogUp protocol.
    ///
    /// # Arguments
    ///
    /// - `transcript`: Fiat-Shamir transcript (must be in the same state as
    ///   the prover's transcript at the start of the LogUp protocol).
    /// - `proof`: The `LogupProof` received from the prover.
    /// - `table`: The lookup table entries (verifier also knows the table).
    /// - `witness_len`: The length of the witness vector.
    /// - `field_cfg`: Field configuration.
    ///
    /// # Returns
    ///
    /// `LogupVerifierSubClaim` on success, or a `LookupError`.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn verify_as_subprotocol(
        transcript: &mut impl Transcript,
        proof: &LogupProof<F>,
        table: &[F],
        witness_len: usize,
        field_cfg: &F::Config,
    ) -> Result<LogupVerifierSubClaim<F>, LookupError<F>>
    where
        F::Inner: ConstTranscribable + Zero,
        F::Modulus: ConstTranscribable,
    {
        let zero = F::zero_with_cfg(field_cfg);
        let one = F::one_with_cfg(field_cfg);

        // ---- Step 1: Absorb multiplicities ----
        let mut buf = vec![0u8; F::Inner::NUM_BYTES];
        transcript.absorb_random_field_slice(&proof.multiplicities, &mut buf);

        // ---- Step 2: Get challenge β ----
        let beta: F = transcript.get_field_challenge(field_cfg);

        // ---- Step 3: Absorb inverse vectors ----
        transcript.absorb_random_field_slice(&proof.inverse_witness, &mut buf);
        transcript.absorb_random_field_slice(&proof.inverse_table, &mut buf);

        // ---- Step 4: Get batching challenge γ and random evaluation point r ----
        let gamma: F = transcript.get_field_challenge(field_cfg);

        let w_num_vars = zinc_utils::log2(witness_len.next_power_of_two()) as usize;
        let t_num_vars = zinc_utils::log2(table.len().next_power_of_two()) as usize;
        let num_vars = w_num_vars.max(t_num_vars);

        let r: Vec<F> = transcript.get_field_challenges(num_vars, field_cfg);

        // ---- Step 5: Verify sumcheck ----
        let subclaim = MLSumcheck::verify_as_subprotocol(
            transcript,
            num_vars,
            3, // degree
            &proof.sumcheck_proof,
            field_cfg,
        )?;

        // ---- Step 6: Verify the final evaluation claim ----
        // The subclaim says: the combination polynomial evaluated at
        // `subclaim.point` should equal `subclaim.expected_evaluation`.
        // We need to verify this by evaluating all MLEs at `subclaim.point`.
        let eval_point = &subclaim.point;

        // Evaluate eq(point, r)
        let eq_val = zinc_poly::utils::eq_eval(eval_point, &r, one.clone())?;

        // Build eq(·, eval_point) once and reuse for all MLE evaluations.
        let eq_at_point = build_eq_x_r_vec(eval_point, field_cfg)?;

        let eval_at_point = |data: &[F]| -> F {
            data.iter()
                .zip(eq_at_point.iter())
                .fold(zero.clone(), |acc, (d, e)| acc + &(d.clone() * e))
        };

        // ---- Step 6a: Direct table inverse check ----
        // Verify (β − T[j]) · v[j] = 1 for all j. This replaces the
        // table-inverse identity that was previously inside the sumcheck,
        // saving one MLE and one identity in the prover.
        for (j, (t_j, v_j)) in table.iter().zip(proof.inverse_table.iter()).enumerate() {
            let check = (beta.clone() - t_j) * v_j;
            if check != one {
                return Err(LookupError::TableInverseIncorrect { index: j });
            }
        }

        // Compute d̃(x*) directly: since d[i] = β − w[i] = 1/u[i],
        // we have d̃(x*) = Σ_j (1/u[j]) · eq(j, x*).
        let u_inv = batch_inverse(&proof.inverse_witness);
        let d_eval = eval_at_point(&u_inv);
        let u_eval = eval_at_point(&proof.inverse_witness);
        let v_eval = eval_at_point(&proof.inverse_table);
        let m_eval = eval_at_point(&proof.multiplicities);

        // Recompute the combination function at the subclaim point
        // (matches the prover's 2-identity comb_fn)
        let id1 = d_eval * &u_eval - &one;
        let id2 = u_eval.clone() - &(m_eval * &v_eval);

        let batched = id1 + id2 * &gamma;
        let expected = batched * &eq_val;

        if expected != subclaim.expected_evaluation {
            return Err(LookupError::FinalEvaluationMismatch {
                expected: subclaim.expected_evaluation.clone(),
                got: expected,
            });
        }

        // ---- Step 7: Check multiplicity sum = witness_len ----
        let m_sum: F = proof
            .multiplicities
            .iter()
            .cloned()
            .fold(zero, |a, b| a + &b);
        let expected_sum = F::from_with_cfg(witness_len as u64, field_cfg);
        if m_sum != expected_sum {
            return Err(LookupError::MultiplicitySumMismatch {
                expected: witness_len as u64,
                got: {
                    // Approximate — exact value depends on the field representation.
                    // For error reporting, we use a placeholder.
                    0
                },
            });
        }

        Ok(LogupVerifierSubClaim {
            evaluation_point: subclaim.point,
            expected_evaluation: subclaim.expected_evaluation,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crypto_bigint::{U128, const_monty_params};
    use crypto_primitives::crypto_bigint_const_monty::ConstMontyField;
    use zinc_transcript::KeccakTranscript;

    const N: usize = 2;
    const_monty_params!(TestParams, U128, "00000000b933426489189cb5b47d567f");
    type F = ConstMontyField<TestParams, N>;

    #[test]
    fn logup_prove_verify_small() {
        // Table: {0, 1, 2, 3}
        let table: Vec<F> = (0..4u32).map(F::from).collect();

        // Witness: [0, 1, 1, 3] — all entries are in the table.
        let witness: Vec<F> = vec![0u32, 1, 1, 3].into_iter().map(F::from).collect();

        let mut prover_transcript = KeccakTranscript::new();
        let (proof, _prover_state) = LogupProtocol::<F>::prove_as_subprotocol(
            &mut prover_transcript,
            &witness,
            &table,
            &(),
        )
        .expect("prover should succeed");

        let mut verifier_transcript = KeccakTranscript::new();
        let _subclaim = LogupProtocol::<F>::verify_as_subprotocol(
            &mut verifier_transcript,
            &proof,
            &table,
            witness.len(),
            &(),
        )
        .expect("verifier should accept");
    }

    #[test]
    fn logup_reject_invalid_witness() {
        let table: Vec<F> = (0..4u32).map(F::from).collect();

        // Witness contains 5, which is NOT in the table.
        let witness: Vec<F> = vec![0u32, 5].into_iter().map(F::from).collect();

        let mut prover_transcript = KeccakTranscript::new();
        let result = LogupProtocol::<F>::prove_as_subprotocol(
            &mut prover_transcript,
            &witness,
            &table,
            &(),
        );

        assert!(result.is_err());
    }
}
