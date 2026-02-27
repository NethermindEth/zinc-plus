//! Decomposition + LogUp protocol for large lookup tables.
//!
//! For tables of size 2^{K·c} (e.g. BitPoly(32)), the table is
//! decomposed into K sub-tables of size 2^c each. The prover
//! demonstrates:
//!
//! 1. **Decomposition consistency**: each witness entry `w_i` equals
//!    `Σ_k shifts[k] · chunks[k][i]`.
//! 2. **Sub-table membership**: each chunk belongs to the sub-table,
//!    via K LogUp invocations (sharing the same sub-table).
//!
//! All auxiliary vectors are sent in the clear.

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
    structs::{
        DecompLogupProof, DecompLogupProverState, DecompLogupVerifierSubClaim,
        DecompLookupInstance, LookupError,
    },
    tables::{batch_inverse_shifted, build_table_index, compute_multiplicities_with_index},
};

/// The Decomposition + LogUp protocol for large tables.
pub struct DecompLogupProtocol<F: InnerTransparentField>(PhantomData<F>);

impl<F: InnerTransparentField + FromPrimitiveWithConfig + Send + Sync> DecompLogupProtocol<F> {
    /// Prover for the Decomposition + LogUp protocol.
    ///
    /// Supports an arbitrary number K of chunks. The witness is
    /// decomposed as `w_i = Σ_k shifts[k] · chunks[k][i]`, and each
    /// chunk is proven to lie in the shared sub-table via LogUp.
    ///
    /// # Arguments
    ///
    /// - `transcript`: Fiat-Shamir transcript.
    /// - `instance`: A `DecompLookupInstance` with K chunks.
    /// - `field_cfg`: Field configuration.
    #[allow(clippy::arithmetic_side_effects, clippy::too_many_arguments)]
    pub fn prove_as_subprotocol(
        transcript: &mut impl Transcript,
        instance: &DecompLookupInstance<F>,
        field_cfg: &F::Config,
    ) -> Result<(DecompLogupProof<F>, DecompLogupProverState<F>), LookupError<F>>
    where
        F::Inner: ConstTranscribable + Zero + Default + Send + Sync,
    {
        let witness = &instance.witness;
        let subtable = &instance.subtable;
        let _shifts = &instance.shifts;
        let chunks = &instance.chunks;
        let num_chunks = chunks.len();
        let witness_len = witness.len();

        let zero = F::zero_with_cfg(field_cfg);
        let one = F::one_with_cfg(field_cfg);
        let mut buf = vec![0u8; F::Inner::NUM_BYTES];

        // ---- Step 1: Send decomposition chunks in the clear ----
        for chunk in chunks {
            transcript.absorb_random_field_slice(chunk, &mut buf);
        }

        // ---- Step 2: Compute multiplicities for each chunk ----
        // Build the table index once and reuse it across all chunks.
        let table_index = build_table_index(subtable);
        let chunk_multiplicities: Vec<Vec<F>> = cfg_iter!(chunks)
            .map(|chunk| {
                compute_multiplicities_with_index(chunk, &table_index, subtable.len(), field_cfg)
                    .ok_or(LookupError::WitnessNotInTable)
            })
            .collect::<Result<_, _>>()?;

        // Aggregate multiplicities across chunks: m_agg[j] = Σ_k m_k[j]
        let aggregated_multiplicities: Vec<F> = {
            let mut agg = vec![zero.clone(); subtable.len()];
            for m in &chunk_multiplicities {
                for (a, mk) in agg.iter_mut().zip(m.iter()) {
                    *a += mk;
                }
            }
            agg
        };

        transcript.absorb_random_field_slice(&aggregated_multiplicities, &mut buf);

        // ---- Step 3: Get challenge β for LogUp ----
        let beta: F = transcript.get_field_challenge(field_cfg);

        // ---- Step 4: Compute inverse vectors for each chunk ----
        let inverse_witnesses: Vec<Vec<F>> = cfg_iter!(chunks)
            .map(|chunk| batch_inverse_shifted(&beta, chunk))
            .collect();

        // Inverse table vector (shared across all chunks)
        let v_table = batch_inverse_shifted(&beta, subtable);

        // Absorb inverse vectors
        for u in &inverse_witnesses {
            transcript.absorb_random_field_slice(u, &mut buf);
        }
        transcript.absorb_random_field_slice(&v_table, &mut buf);

        // ---- Step 5: Get batching challenge ----
        let gamma: F = transcript.get_field_challenge(field_cfg);

        // Compute dimensions
        let w_num_vars = zinc_utils::log2(witness_len.next_power_of_two()) as usize;
        let t_num_vars = zinc_utils::log2(subtable.len().next_power_of_two()) as usize;
        let num_vars = w_num_vars.max(t_num_vars);

        // Get random evaluation point for eq polynomial
        let r: Vec<F> = transcript.get_field_challenges(num_vars, field_cfg);
        let eq_r = build_eq_x_r_inner(&r, field_cfg)?;

        let inner_zero = zero.inner().clone();

        // ---- Step 6: Build MLEs ----
        // Precompute (β − chunk_k) once per chunk so the combination
        // function avoids cloning β at every evaluation point.
        //
        // MLE layout (K = num_chunks):
        //   0:               eq(y, r)
        //   1   .. K:        d_k(y) = β − chunk_k(y)  — K shifted chunk MLEs
        //   K+1 .. 2K:       ũ_k(y)                    — K inverse witness MLEs
        //   2K+1:            ṽ(y)                      — inverse table
        //   2K+2:            m̃_agg(y)                 — aggregated multiplicity MLE
        //
        // Total: 2K + 3 MLEs.

        let mk_mle = |data: &[F]| -> DenseMultilinearExtension<F::Inner> {
            DenseMultilinearExtension::from_evaluations_vec(
                num_vars,
                cfg_iter!(data).map(|x| x.inner().clone()).collect(),
                inner_zero.clone(),
            )
        };

        // d_k[i] = β − chunk_k[i]  computed directly at the Inner level
        // to avoid allocating intermediate Vec<F> elements.
        let beta_inner = beta.inner();
        let mk_shifted_mle =
            |chunk: &[F]| -> DenseMultilinearExtension<F::Inner> {
                DenseMultilinearExtension::from_evaluations_vec(
                    num_vars,
                    cfg_iter!(chunk)
                        .map(|c_i| F::sub_inner(beta_inner, c_i.inner(), field_cfg))
                        .collect(),
                    inner_zero.clone(),
                )
            };

        let mut mles = Vec::with_capacity(2 * num_chunks + 3);
        mles.push(eq_r);                             // [0]: eq
        for chunk in chunks {                         // [1..K]: β − chunk
            mles.push(mk_shifted_mle(chunk));
        }
        for u in &inverse_witnesses {                 // [K+1..2K]: inverse witnesses
            mles.push(mk_mle(u));
        }
        mles.push(mk_mle(&v_table));                 // [2K+1]: inverse table
        mles.push(mk_mle(&aggregated_multiplicities)); // [2K+2]: aggregated multiplicities

        // ---- Step 7: Run batched sumcheck ----
        // K + 1 identities batched with γ (decomp consistency removed,
        // table inverse checked directly by verifier):
        //
        // γ^{0}..γ^{K−1}: [d_k · u_k − 1] · eq          (inverse correctness, K)
        // γ^{K}:           [Σ_k u_k − m_agg · v] · eq    (aggregated log-deriv balance)
        //
        // Total claimed sum = 0.

        // Precompute gamma powers: γ^0 .. γ^{K}
        let mut gamma_powers = Vec::with_capacity(num_chunks + 1);
        let mut gp = one.clone();
        for _ in 0..=num_chunks {
            gamma_powers.push(gp.clone());
            gp *= &gamma;
        }

        let one_clone = one.clone();
        let k = num_chunks;

        let comb_fn = move |vals: &[F]| -> F {
            let eq_val = &vals[0];
            let mut batched = {
                // Identity 0: inverse correctness for chunk 0
                let d_val = &vals[1];       // β − c_0 (precomputed)
                let u_val = &vals[1 + k];
                let id = d_val.clone() * u_val - &one_clone;
                id * &gamma_powers[0]
            };

            // Identities 1..K−1: inverse correctness for remaining chunks
            for i in 1..k {
                let d_val = &vals[1 + i];   // β − c_i (precomputed)
                let u_val = &vals[1 + k + i];
                let id = d_val.clone() * u_val - &one_clone;
                batched += &(id * &gamma_powers[i]);
            }

            // Identity K: aggregated log-derivative balance
            // Σ_k u_k − m_agg · v
            let v_val = &vals[2 * k + 1];
            let m_agg_val = &vals[2 * k + 2];
            let mut u_sum = vals[1 + k].clone();
            for i in 1..k {
                u_sum += &vals[1 + k + i];
            }
            let id_balance = u_sum - &(m_agg_val.clone() * v_val);
            batched += &(id_balance * &gamma_powers[k]);

            batched * eq_val
        };

        // degree: eq is linear (1), identities have degree 2 (products of 2 MLEs)
        // max degree = 1 + 2 = 3
        let degree = 3;

        let (sumcheck_proof, sumcheck_prover_state) = MLSumcheck::prove_as_subprotocol(
            transcript,
            mles,
            num_vars,
            degree,
            comb_fn,
            field_cfg,
        );

        // ---- Step 8: Verify multiplicity sums (prover-side sanity check) ----
        debug_assert!({
            let expected = F::from_with_cfg((num_chunks * witness_len) as u64, field_cfg);
            let sum: F = aggregated_multiplicities
                .iter()
                .cloned()
                .fold(zero.clone(), |a, b| a + &b);
            sum == expected
        });

        Ok((
            DecompLogupProof {
                chunk_vectors: chunks.to_vec(),
                sumcheck_proof,
                aggregated_multiplicities,
                chunk_inverse_witnesses: inverse_witnesses,
                inverse_table: v_table,
            },
            DecompLogupProverState {
                evaluation_point: sumcheck_prover_state.randomness,
            },
        ))
    }

    /// Verifier for the Decomposition + LogUp protocol.
    ///
    /// Supports an arbitrary number K of chunks.
    ///
    /// # Arguments
    ///
    /// - `transcript`: Fiat-Shamir transcript.
    /// - `proof`: The `DecompLogupProof` received from the prover.
    /// - `subtable`: The sub-table entries.
    /// - `shifts`: The shift factors for the decomposition (K entries).
    /// - `witness_len`: Length of the original witness.
    /// - `field_cfg`: Field configuration.
    #[allow(clippy::arithmetic_side_effects, clippy::too_many_arguments)]
    pub fn verify_as_subprotocol(
        transcript: &mut impl Transcript,
        proof: &DecompLogupProof<F>,
        subtable: &[F],
        _shifts: &[F],
        witness_len: usize,
        field_cfg: &F::Config,
    ) -> Result<DecompLogupVerifierSubClaim<F>, LookupError<F>>
    where
        F::Inner: ConstTranscribable + Zero + Default,
    {
        let zero = F::zero_with_cfg(field_cfg);
        let one = F::one_with_cfg(field_cfg);
        let mut buf = vec![0u8; F::Inner::NUM_BYTES];

        let num_chunks = proof.chunk_vectors.len();
        let chunks = &proof.chunk_vectors;
        let agg_mults = &proof.aggregated_multiplicities;
        let inverse_witnesses = &proof.chunk_inverse_witnesses;
        let v_table = &proof.inverse_table;

        // ---- Mirror transcript operations ----

        // Absorb chunks
        for chunk in chunks {
            transcript.absorb_random_field_slice(chunk, &mut buf);
        }

        // Absorb aggregated multiplicities
        transcript.absorb_random_field_slice(agg_mults, &mut buf);

        // Get β
        let beta: F = transcript.get_field_challenge(field_cfg);

        // Absorb inverse vectors
        for u in inverse_witnesses {
            transcript.absorb_random_field_slice(u, &mut buf);
        }
        transcript.absorb_random_field_slice(v_table, &mut buf);

        // Get γ
        let gamma: F = transcript.get_field_challenge(field_cfg);

        // Compute dimensions
        let w_num_vars = zinc_utils::log2(witness_len.next_power_of_two()) as usize;
        let t_num_vars = zinc_utils::log2(subtable.len().next_power_of_two()) as usize;
        let num_vars = w_num_vars.max(t_num_vars);

        // Get evaluation point
        let r: Vec<F> = transcript.get_field_challenges(num_vars, field_cfg);

        // ---- Verify sumcheck ----
        let subclaim = MLSumcheck::verify_as_subprotocol(
            transcript,
            num_vars,
            3,
            &proof.sumcheck_proof,
            field_cfg,
        )?;

        // ---- Verify final evaluation claim ----
        let eval_point = &subclaim.point;

        let eq_val = zinc_poly::utils::eq_eval(eval_point, &r, one.clone())?;

        // Build eq(·, eval_point) once and reuse for all MLE evaluations.
        let eq_at_point = build_eq_x_r_vec(eval_point, field_cfg)?;

        let eval_at_point = |data: &[F]| -> F {
            data.iter()
                .zip(eq_at_point.iter())
                .fold(zero.clone(), |acc, (d, e)| acc + &(d.clone() * e))
        };

        // ---- Direct table inverse check ----
        // Verify (β − T[j]) · v[j] = 1 for all j. This replaces the
        // table-inverse identity that was previously inside the sumcheck.
        for (j, (t_j, v_j)) in subtable.iter().zip(v_table.iter()).enumerate() {
            let check = (beta.clone() - t_j) * v_j;
            if check != one {
                return Err(LookupError::TableInverseIncorrect { index: j });
            }
        }

        // Evaluate data MLEs at the subclaim point via inner product
        let chunk_evals: Vec<F> = chunks
            .iter()
            .map(|c| eval_at_point(c))
            .collect();

        let u_evals: Vec<F> = inverse_witnesses
            .iter()
            .map(|u| eval_at_point(u))
            .collect();

        let v_eval = eval_at_point(v_table);
        let m_agg_eval = eval_at_point(agg_mults);

        // Recompute combination function at the subclaim point
        // (matches the prover's K+1 identity comb_fn)

        // Precompute gamma powers: γ^0 .. γ^K
        let mut gamma_powers = Vec::with_capacity(num_chunks + 1);
        let mut gp = one.clone();
        for _ in 0..=num_chunks {
            gamma_powers.push(gp.clone());
            gp *= &gamma;
        }

        // Identities 0..K−1: inverse correctness for each chunk
        let mut batched = {
            let id = (beta.clone() - &chunk_evals[0]) * &u_evals[0] - &one;
            id * &gamma_powers[0]
        };
        for k in 1..num_chunks {
            let id = (beta.clone() - &chunk_evals[k]) * &u_evals[k] - &one;
            batched += &(id * &gamma_powers[k]);
        }

        // Identity K: aggregated log-derivative balance
        let mut u_sum = u_evals[0].clone();
        for k in 1..num_chunks {
            u_sum += &u_evals[k];
        }
        let id_balance = u_sum - &(m_agg_eval * &v_eval);
        batched += &(id_balance * &gamma_powers[num_chunks]);

        let expected = batched * &eq_val;

        if expected != subclaim.expected_evaluation {
            return Err(LookupError::FinalEvaluationMismatch {
                expected: subclaim.expected_evaluation.clone(),
                got: expected,
            });
        }

        // ---- Verify aggregated multiplicity sum = K × witness_len ----
        let expected_sum = F::from_with_cfg((num_chunks * witness_len) as u64, field_cfg);
        let m_sum: F = agg_mults.iter().cloned().fold(zero.clone(), |a, b| a + &b);
        if m_sum != expected_sum {
            return Err(LookupError::MultiplicitySumMismatch {
                expected: (num_chunks * witness_len) as u64,
                got: 0,
            });
        }

        Ok(DecompLogupVerifierSubClaim {
            evaluation_point: subclaim.point,
            expected_evaluation: subclaim.expected_evaluation,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lookup::tables::{bitpoly_shift, generate_bitpoly_table};
    use crypto_bigint::{U128, const_monty_params};
    use crypto_primitives::crypto_bigint_const_monty::ConstMontyField;
    use zinc_transcript::KeccakTranscript;

    const N: usize = 2;
    const_monty_params!(TestParams, U128, "00000000b933426489189cb5b47d567f");
    type F = ConstMontyField<TestParams, N>;

    /// Test with K=2 chunks: BitPoly(4) → two BitPoly(2) sub-tables.
    #[test]
    fn decomp_logup_2_chunks_bitpoly() {
        let a = F::from(3u32);

        let chunk_width = 2;
        let num_chunks = 2;
        let subtable = generate_bitpoly_table(chunk_width, &a, &());
        let full_table = generate_bitpoly_table(chunk_width * num_chunks, &a, &());

        // shifts = [a^0 = 1, a^2]
        let shifts: Vec<F> = (0..num_chunks)
            .map(|k| bitpoly_shift(k * chunk_width, &a))
            .collect();

        // Witness: pick some entries from the full table.
        let witness = vec![
            full_table[0].clone(),  // index 0: lo=0, hi=0
            full_table[3].clone(),  // index 3: lo=T[3], hi=T[0]
            full_table[5].clone(),  // index 5: lo=T[1], hi=T[1]
            full_table[15].clone(), // index 15: all bits set
        ];

        // Masks for extracting chunk_width-bit groups
        let mask = (1usize << chunk_width) - 1;
        let indices = [0usize, 3, 5, 15];
        let chunks: Vec<Vec<F>> = (0..num_chunks)
            .map(|k| {
                indices
                    .iter()
                    .map(|&idx| subtable[(idx >> (k * chunk_width)) & mask].clone())
                    .collect()
            })
            .collect();

        let instance = DecompLookupInstance {
            witness: witness.clone(),
            subtable: subtable.clone(),
            shifts: shifts.clone(),
            chunks,
        };

        let mut prover_transcript = KeccakTranscript::new();
        let (proof, _) = DecompLogupProtocol::<F>::prove_as_subprotocol(
            &mut prover_transcript,
            &instance,
            &(),
        )
        .expect("prover should succeed");

        let mut verifier_transcript = KeccakTranscript::new();
        let _subclaim = DecompLogupProtocol::<F>::verify_as_subprotocol(
            &mut verifier_transcript,
            &proof,
            &subtable,
            &shifts,
            witness.len(),
            &(),
        )
        .expect("verifier should accept");
    }

    /// Test with K=4 chunks: BitPoly(8) → four BitPoly(2) sub-tables.
    #[test]
    fn decomp_logup_4_chunks_bitpoly() {
        let a = F::from(3u32);

        let chunk_width = 2;
        let num_chunks = 4;
        let subtable = generate_bitpoly_table(chunk_width, &a, &());
        let full_table = generate_bitpoly_table(chunk_width * num_chunks, &a, &());

        // shifts = [1, a^2, a^4, a^6]
        let shifts: Vec<F> = (0..num_chunks)
            .map(|k| bitpoly_shift(k * chunk_width, &a))
            .collect();

        // Pick some entries from the full 256-entry table.
        let indices = [0usize, 42, 127, 200, 255, 170, 85, 13];
        let witness: Vec<F> = indices.iter().map(|&i| full_table[i].clone()).collect();

        // Decompose into 4 chunks of 2 bits each.
        let mask = (1usize << chunk_width) - 1;
        let chunks: Vec<Vec<F>> = (0..num_chunks)
            .map(|k| {
                indices
                    .iter()
                    .map(|&idx| subtable[(idx >> (k * chunk_width)) & mask].clone())
                    .collect()
            })
            .collect();

        let instance = DecompLookupInstance {
            witness: witness.clone(),
            subtable: subtable.clone(),
            shifts: shifts.clone(),
            chunks,
        };

        let mut prover_transcript = KeccakTranscript::new();
        let (proof, _) = DecompLogupProtocol::<F>::prove_as_subprotocol(
            &mut prover_transcript,
            &instance,
            &(),
        )
        .expect("prover should succeed");

        let mut verifier_transcript = KeccakTranscript::new();
        let _subclaim = DecompLogupProtocol::<F>::verify_as_subprotocol(
            &mut verifier_transcript,
            &proof,
            &subtable,
            &shifts,
            witness.len(),
            &(),
        )
        .expect("verifier should accept");
    }
}
