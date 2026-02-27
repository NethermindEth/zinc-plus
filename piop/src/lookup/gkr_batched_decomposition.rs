//! GKR-based Batched Decomposition + LogUp protocol for multiple witnesses.
//!
//! For L witnesses, each decomposed into K chunks, proves all L lookups
//! using K·L GKR fractional sumchecks (one per chunk per witness) and one
//! for the table side (with total aggregated multiplicities Σ_ℓ m_agg^(ℓ)).
//!
//! ## Root cross-check
//!
//! Let the chunk GKR for lookup ℓ, chunk k prove `P_{ℓ,k}/Q_{ℓ,k}`.
//! The table GKR proves `P_t/Q_t`.
//! The verifier checks:
//!
//! ```text
//! Σ_{ℓ=0..L} Σ_{k=0..K} P_{ℓ,k}/Q_{ℓ,k} == P_t/Q_t
//! ```
//!
//! implemented as an iterative fraction-addition cross-check (no field inversion).

use crypto_primitives::FromPrimitiveWithConfig;
use num_traits::Zero;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::marker::PhantomData;
use zinc_poly::utils::build_eq_x_r_vec;
use zinc_transcript::traits::{ConstTranscribable, Transcript};
use zinc_utils::{cfg_iter, inner_transparent_field::InnerTransparentField};

use super::{
    gkr_logup::{FractionLayer, build_fraction_tree, gkr_fraction_prove, gkr_fraction_verify},
    structs::{
        BatchedDecompLookupInstance, GkrBatchedDecompLogupProof, GkrBatchedDecompLogupProverState,
        GkrBatchedDecompLogupVerifierSubClaim, LookupError,
    },
    tables::{build_table_index, compute_multiplicities_with_index},
};

/// The GKR Batched Decomposition + LogUp protocol for multiple witnesses.
pub struct GkrBatchedDecompLogupProtocol<F: InnerTransparentField>(PhantomData<F>);

impl<F: InnerTransparentField + FromPrimitiveWithConfig + Send + Sync>
    GkrBatchedDecompLogupProtocol<F>
{
    /// Prover for the GKR Batched Decomposition + LogUp protocol.
    ///
    /// All L witness vectors must have the **same length** and share the
    /// same sub-table and shift factors.
    #[allow(clippy::arithmetic_side_effects, clippy::too_many_arguments)]
    pub fn prove_as_subprotocol(
        transcript: &mut impl Transcript,
        instance: &BatchedDecompLookupInstance<F>,
        field_cfg: &F::Config,
    ) -> Result<
        (
            GkrBatchedDecompLogupProof<F>,
            GkrBatchedDecompLogupProverState<F>,
        ),
        LookupError<F>,
    >
    where
        F::Inner: ConstTranscribable + Zero + Default + Send + Sync,
    {
        let subtable = &instance.subtable;
        let chunks = &instance.chunks; // [L][K][N]
        let num_lookups = chunks.len();
        let num_chunks = if num_lookups > 0 { chunks[0].len() } else { 0 };
        let witness_len =
            if num_lookups > 0 && num_chunks > 0 { chunks[0][0].len() } else { 0 };

        let zero = F::zero_with_cfg(field_cfg);
        let one = F::one_with_cfg(field_cfg);
        let mut buf = vec![0u8; F::Inner::NUM_BYTES];

        // ---- Step 1: Send all chunk vectors in the clear ----
        for chunk_set in chunks {
            for chunk in chunk_set {
                transcript.absorb_random_field_slice(chunk, &mut buf);
            }
        }

        // ---- Step 2: Compute multiplicities for every (ℓ, k) pair in parallel ----
        let table_index = build_table_index(subtable);
        let all_chunk_multiplicities: Vec<Vec<Vec<F>>> = cfg_iter!(chunks)
            .map(|lookup_chunks| {
                lookup_chunks
                    .iter()
                    .map(|chunk| {
                        compute_multiplicities_with_index(
                            chunk,
                            &table_index,
                            subtable.len(),
                            field_cfg,
                        )
                        .ok_or(LookupError::WitnessNotInTable)
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<_, _>>()?;

        // ---- Step 3: Aggregate multiplicities per witness ----
        let aggregated_multiplicities: Vec<Vec<F>> = all_chunk_multiplicities
            .iter()
            .map(|lookup_mults| {
                let mut agg = vec![zero.clone(); subtable.len()];
                for m in lookup_mults {
                    for (a, mk) in agg.iter_mut().zip(m.iter()) {
                        *a += mk;
                    }
                }
                agg
            })
            .collect();

        // ---- Step 4: Absorb all aggregated multiplicities ----
        for m in &aggregated_multiplicities {
            transcript.absorb_random_field_slice(m, &mut buf);
        }

        // ---- Step 5: Shared β challenge ----
        let beta: F = transcript.get_field_challenge(field_cfg);

        // ---- Step 6a: Pre-compute all tree inputs ----
        let w_num_vars = zinc_utils::log2(witness_len.next_power_of_two()) as usize;
        let t_num_vars = zinc_utils::log2(subtable.len().next_power_of_two()) as usize;
        let w_size = 1usize << w_num_vars;
        let t_size = 1usize << t_num_vars;

        // Build total aggregated multiplicity (Σ_ℓ m_agg^(ℓ)) for the table side.
        let mut total_agg_mult = vec![zero.clone(); subtable.len()];
        for m in &aggregated_multiplicities {
            for (a, m_l) in total_agg_mult.iter_mut().zip(m.iter()) {
                *a += m_l;
            }
        }

        // ---- Step 6b: Build ALL fraction trees in parallel ----
        // Tree building is pure computation (no transcript), so all L×K chunk trees
        // and the single table tree are built concurrently.
        let chunk_trees: Vec<Vec<Vec<FractionLayer<F>>>> = cfg_iter!(chunks)
            .map(|chunk_set| {
                cfg_iter!(chunk_set)
                    .map(|chunk| {
                        let mut leaf_p = vec![one.clone(); chunk.len()];
                        let mut leaf_q: Vec<F> =
                            chunk.iter().map(|c_i| beta.clone() - c_i).collect();
                        leaf_p.resize(w_size, zero.clone());
                        leaf_q.resize(w_size, one.clone());
                        build_fraction_tree(leaf_p, leaf_q)
                    })
                    .collect()
            })
            .collect();

        let mut t_leaf_p = total_agg_mult.clone();
        let mut t_leaf_q: Vec<F> =
            cfg_iter!(subtable).map(|t_j| beta.clone() - t_j).collect();
        t_leaf_p.resize(t_size, zero.clone());
        t_leaf_q.resize(t_size, one.clone());
        let table_tree = build_fraction_tree(t_leaf_p, t_leaf_q);

        // ---- Step 6c: GKR-prove sequentially (transcript-bound) ----
        let mut chunk_gkr_proofs = Vec::with_capacity(num_lookups);
        for trees in &chunk_trees {
            let mut witness_chunk_proofs = Vec::with_capacity(num_chunks);
            for tree in trees {
                witness_chunk_proofs.push(gkr_fraction_prove(transcript, tree, field_cfg));
            }
            chunk_gkr_proofs.push(witness_chunk_proofs);
        }
        let table_gkr_proof = gkr_fraction_prove(transcript, &table_tree, field_cfg);

        // ---- Debug: root cross-check (Σ_{ℓ,k} P_{ℓ,k}/Q_{ℓ,k} == P_t/Q_t) ----
        debug_assert!({
            let mut acc_p = zero.clone();
            let mut acc_q = one.clone();
            for lookup_proofs in &chunk_gkr_proofs {
                for gkr in lookup_proofs {
                    let new_p =
                        acc_p.clone() * &gkr.root_q + &(gkr.root_p.clone() * &acc_q);
                    let new_q = acc_q.clone() * &gkr.root_q;
                    acc_p = new_p;
                    acc_q = new_q;
                }
            }
            let lhs = acc_p * &table_gkr_proof.root_q;
            let rhs = table_gkr_proof.root_p.clone() * &acc_q;
            lhs == rhs
        });

        // ---- Debug: multiplicity sum per witness = K × witness_len ----
        debug_assert!({
            let expected =
                F::from_with_cfg((num_chunks * witness_len) as u64, field_cfg);
            aggregated_multiplicities.iter().all(|agg| {
                let sum: F = agg.iter().cloned().fold(zero.clone(), |a, b| a + &b);
                sum == expected
            })
        });

        Ok((
            GkrBatchedDecompLogupProof {
                chunk_vectors: chunks.clone(),
                aggregated_multiplicities,
                chunk_gkr_proofs,
                table_gkr_proof,
            },
            GkrBatchedDecompLogupProverState {
                chunk_eval_points: vec![vec![Vec::new(); num_chunks]; num_lookups],
                table_eval_point: Vec::new(),
            },
        ))
    }

    /// Verifier for the GKR Batched Decomposition + LogUp protocol.
    #[allow(clippy::arithmetic_side_effects, clippy::too_many_arguments)]
    pub fn verify_as_subprotocol(
        transcript: &mut impl Transcript,
        proof: &GkrBatchedDecompLogupProof<F>,
        subtable: &[F],
        _shifts: &[F],
        num_lookups: usize,
        witness_len: usize,
        field_cfg: &F::Config,
    ) -> Result<GkrBatchedDecompLogupVerifierSubClaim<F>, LookupError<F>>
    where
        F::Inner: ConstTranscribable + Zero,
    {
        let zero = F::zero_with_cfg(field_cfg);
        let one = F::one_with_cfg(field_cfg);
        let mut buf = vec![0u8; F::Inner::NUM_BYTES];

        let num_chunks =
            if num_lookups > 0 { proof.chunk_vectors[0].len() } else { 0 };
        let chunks = &proof.chunk_vectors;
        let agg_mults = &proof.aggregated_multiplicities;

        // ---- Mirror transcript: absorb all chunk vectors ----
        for chunk_set in chunks {
            for chunk in chunk_set {
                transcript.absorb_random_field_slice(chunk, &mut buf);
            }
        }

        // ---- Absorb all aggregated multiplicities ----
        for m in agg_mults {
            transcript.absorb_random_field_slice(m, &mut buf);
        }

        // ---- Shared β challenge ----
        let beta: F = transcript.get_field_challenge(field_cfg);

        let w_num_vars = zinc_utils::log2(witness_len.next_power_of_two()) as usize;
        let t_num_vars = zinc_utils::log2(subtable.len().next_power_of_two()) as usize;

        // ---- Verify GKR proofs for each (ℓ, k) and check leaf claims ----
        let mut chunk_eval_points = Vec::with_capacity(num_lookups);
        let mut chunk_expected_evals = Vec::with_capacity(num_lookups);

        // Accumulate fraction sum for root cross-check: acc = Σ_{ℓ,k} P_{ℓ,k}/Q_{ℓ,k}
        let mut acc_p = zero.clone();
        let mut acc_q = one.clone();

        for (l, chunk_set) in proof.chunk_gkr_proofs.iter().enumerate() {
            let mut witness_eval_points = Vec::with_capacity(num_chunks);
            let mut witness_expected_evals = Vec::with_capacity(num_chunks);
            for (k, gkr_proof) in chunk_set.iter().enumerate() {
                let result =
                    gkr_fraction_verify(transcript, gkr_proof, w_num_vars, field_cfg)?;

                // Leaf numerator must be the MLE of all-ones (= 1).
                if result.expected_p != one {
                    return Err(LookupError::GkrLeafMismatch);
                }

                // Evaluate c̃_{ℓ,k} at the GKR leaf eval point.
                let c_eval = if result.point.is_empty() {
                    chunks[l][k][0].clone()
                } else {
                    let eq_at_pt = build_eq_x_r_vec(&result.point, field_cfg)?;
                    chunks[l][k]
                        .iter()
                        .zip(eq_at_pt.iter())
                        .fold(zero.clone(), |acc, (c, e)| acc + &(c.clone() * e))
                };

                // Leaf denominator: expected_q == β − c̃_{ℓ,k}(r_{ℓ,k})
                let expected_q = beta.clone() - &c_eval;
                if expected_q != result.expected_q {
                    return Err(LookupError::GkrLeafMismatch);
                }

                // Accumulate into fraction sum for cross-check.
                let new_p = acc_p.clone() * &gkr_proof.root_q
                    + &(gkr_proof.root_p.clone() * &acc_q);
                let new_q = acc_q.clone() * &gkr_proof.root_q;
                acc_p = new_p;
                acc_q = new_q;

                witness_eval_points.push(result.point.clone());
                witness_expected_evals.push(c_eval);
            }
            chunk_eval_points.push(witness_eval_points);
            chunk_expected_evals.push(witness_expected_evals);
        }

        // ---- Verify GKR proof for table side ----
        let table_result = gkr_fraction_verify(
            transcript,
            &proof.table_gkr_proof,
            t_num_vars,
            field_cfg,
        )?;

        // ---- Root cross-check: Σ_{ℓ,k} P_{ℓ,k}/Q_{ℓ,k} == P_t/Q_t ----
        // Implemented as: acc_p · Q_t == P_t · acc_q
        let lhs = acc_p * &proof.table_gkr_proof.root_q;
        let rhs = proof.table_gkr_proof.root_p.clone() * &acc_q;
        if lhs != rhs {
            return Err(LookupError::GkrRootMismatch);
        }

        // ---- Verify leaf-level claims for table side ----
        // Reconstruct total aggregated multiplicity (Σ_ℓ m_agg^(ℓ)).
        let mut total_agg_mult = vec![zero.clone(); subtable.len()];
        for m in agg_mults {
            for (a, m_l) in total_agg_mult.iter_mut().zip(m.iter()) {
                *a += m_l;
            }
        }

        let (m_eval, t_eval) = if table_result.point.is_empty() {
            (total_agg_mult[0].clone(), subtable[0].clone())
        } else {
            let eq_at_t = build_eq_x_r_vec(&table_result.point, field_cfg)?;
            let m_e: F = total_agg_mult
                .iter()
                .zip(eq_at_t.iter())
                .fold(zero.clone(), |acc, (m_j, eq_j)| acc + &(m_j.clone() * eq_j));
            let t_e: F = subtable
                .iter()
                .zip(eq_at_t.iter())
                .fold(zero.clone(), |acc, (t_j, eq_j)| acc + &(t_j.clone() * eq_j));
            (m_e, t_e)
        };
        if m_eval != table_result.expected_p {
            return Err(LookupError::GkrLeafMismatch);
        }
        let expected_q_t = beta.clone() - &t_eval;
        if expected_q_t != table_result.expected_q {
            return Err(LookupError::GkrLeafMismatch);
        }

        // ---- Verify aggregated multiplicity sum per witness = K × witness_len ----
        let expected_sum =
            F::from_with_cfg((num_chunks * witness_len) as u64, field_cfg);
        for (_ell, agg) in agg_mults.iter().enumerate() {
            let m_sum: F = agg.iter().cloned().fold(zero.clone(), |a, b| a + &b);
            if m_sum != expected_sum {
                return Err(LookupError::MultiplicitySumMismatch {
                    expected: (num_chunks * witness_len) as u64,
                    got: 0,
                });
            }
        }

        Ok(GkrBatchedDecompLogupVerifierSubClaim {
            chunk_eval_points,
            chunk_expected_evals,
            table_eval_point: table_result.point,
            mult_expected_eval: m_eval,
            table_expected_eval: t_eval,
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

    #[test]
    fn gkr_batched_decomp_logup_2_lookups_2_chunks() {
        let a = F::from(3u32);
        let chunk_width = 2;
        let num_chunks = 2;
        let num_lookups = 2;
        let subtable = generate_bitpoly_table(chunk_width, &a, &());
        let full_table = generate_bitpoly_table(chunk_width * num_chunks, &a, &());
        let shifts: Vec<F> = (0..num_chunks)
            .map(|k| bitpoly_shift(k * chunk_width, &a))
            .collect();
        let indices = [[0usize, 3], [5, 15]];
        let mut witnesses = Vec::new();
        let mut all_chunks = Vec::new();
        for idxs in indices.iter() {
            let witness: Vec<F> = idxs.iter().map(|&i| full_table[i].clone()).collect();
            let mask = (1usize << chunk_width) - 1;
            let chunks: Vec<Vec<F>> = (0..num_chunks)
                .map(|k| idxs.iter().map(|&idx| subtable[(idx >> (k * chunk_width)) & mask].clone()).collect())
                .collect();
            witnesses.push(witness);
            all_chunks.push(chunks);
        }
        let instance = BatchedDecompLookupInstance {
            witnesses: witnesses.clone(),
            subtable: subtable.clone(),
            shifts: shifts.clone(),
            chunks: all_chunks,
        };
        let mut prover_transcript = KeccakTranscript::new();
        let (proof, _) = GkrBatchedDecompLogupProtocol::<F>::prove_as_subprotocol(
            &mut prover_transcript,
            &instance,
            &(),
        ).expect("prover should succeed");
        let mut verifier_transcript = KeccakTranscript::new();
        let _subclaim = GkrBatchedDecompLogupProtocol::<F>::verify_as_subprotocol(
            &mut verifier_transcript,
            &proof,
            &subtable,
            &shifts,
            num_lookups,
            witnesses[0].len(),
            &(),
        ).expect("verifier should accept");
    }

    #[test]
    fn gkr_batched_decomp_logup_reject_bad_chunk() {
        let a = F::from(3u32);
        let chunk_width = 2;
        let num_chunks = 2;
        let _num_lookups = 1;
        let subtable = generate_bitpoly_table(chunk_width, &a, &());
        let full_table = generate_bitpoly_table(chunk_width * num_chunks, &a, &());
        let shifts: Vec<F> = (0..num_chunks)
            .map(|k| bitpoly_shift(k * chunk_width, &a))
            .collect();
        let witness = vec![full_table[0].clone(), full_table[3].clone()];
        let bad_entry = F::from(999u32);
        let chunks = vec![
            vec![subtable[0].clone(), subtable[3].clone()],
            vec![subtable[0].clone(), bad_entry],
        ];
        let instance = BatchedDecompLookupInstance {
            witnesses: vec![witness],
            subtable: subtable.clone(),
            shifts,
            chunks: vec![chunks],
        };
        let mut prover_transcript = KeccakTranscript::new();
        let result = GkrBatchedDecompLogupProtocol::<F>::prove_as_subprotocol(
            &mut prover_transcript,
            &instance,
            &(),
        );
        assert!(result.is_err());
    }
}
