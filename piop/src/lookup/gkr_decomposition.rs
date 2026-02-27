//! GKR-based Decomposition + LogUp protocol for large lookup tables.
//!
//! Combines the Lasso-style decomposition of
//! [`super::decomposition::DecompLogupProtocol`] with the GKR fractional
//! sumcheck of [`super::gkr_logup`], eliminating inverse vectors.
//!
//! For a table of size `2^{K·c}` decomposed into K sub-tables of size
//! `2^c`, the prover demonstrates:
//!
//! 1. **Sub-table membership**: each chunk belongs to the sub-table,
//!    proved via K GKR fractional sumchecks (one per chunk) plus one
//!    for the table side (with aggregated multiplicities).
//!
//! 2. **Decomposition consistency**: `w_i = Σ_k shifts[k] · chunks[k][i]`
//!    is left to the outer protocol (identical to the non-GKR variant).
//!
//! ## Advantage over [`DecompLogupProtocol`]
//!
//! No inverse vectors `u_k`, `v` are sent. The proof contains only
//! chunks, aggregated multiplicities, and K+1 GKR proofs — saving
//! `O(K·W + N)` field elements at the cost of `O(K·log²(W) + log²(N))`
//! GKR layer proof elements.

use crypto_primitives::FromPrimitiveWithConfig;
use num_traits::Zero;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::marker::PhantomData;
use zinc_poly::utils::build_eq_x_r_vec;
use zinc_transcript::traits::{ConstTranscribable, Transcript};
use zinc_utils::{cfg_iter, inner_transparent_field::InnerTransparentField};

use super::{
    gkr_logup::{build_fraction_tree, gkr_fraction_prove, gkr_fraction_verify},
    structs::{
        DecompLookupInstance, GkrDecompLogupProof, GkrDecompLogupProverState,
        GkrDecompLogupVerifierSubClaim, LookupError,
    },
    tables::{build_table_index, compute_multiplicities_with_index},
};

/// The GKR Decomposition + LogUp protocol for large tables.
pub struct GkrDecompLogupProtocol<F: InnerTransparentField>(PhantomData<F>);

impl<F: InnerTransparentField + FromPrimitiveWithConfig + Send + Sync> GkrDecompLogupProtocol<F> {
    /// Prover for the GKR Decomposition + LogUp protocol.
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
    ) -> Result<(GkrDecompLogupProof<F>, GkrDecompLogupProverState<F>), LookupError<F>>
    where
        F::Inner: ConstTranscribable + Zero + Default + Send + Sync,
    {
        let subtable = &instance.subtable;
        let chunks = &instance.chunks;
        let num_chunks = chunks.len();
        let witness_len = chunks[0].len();

        let zero = F::zero_with_cfg(field_cfg);
        let one = F::one_with_cfg(field_cfg);
        let mut buf = vec![0u8; F::Inner::NUM_BYTES];

        // ---- Step 1: Send decomposition chunks in the clear ----
        for chunk in chunks {
            transcript.absorb_random_field_slice(chunk, &mut buf);
        }

        // ---- Step 2: Compute multiplicities for each chunk ----
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

        // ---- Step 4: Build fraction trees for each chunk ----
        let w_num_vars = zinc_utils::log2(witness_len.next_power_of_two()) as usize;
        let t_num_vars = zinc_utils::log2(subtable.len().next_power_of_two()) as usize;

        let w_size = 1usize << w_num_vars;
        let t_size = 1usize << t_num_vars;

        let mut chunk_trees = Vec::with_capacity(num_chunks);
        for chunk in chunks {
            // Leaves: (1, β − c_k[i]), padded with (0, 1)
            let mut leaf_p = vec![one.clone(); chunk.len()];
            let mut leaf_q: Vec<F> = cfg_iter!(chunk)
                .map(|c_i| beta.clone() - c_i)
                .collect();
            leaf_p.resize(w_size, zero.clone());
            leaf_q.resize(w_size, one.clone());
            chunk_trees.push(build_fraction_tree(leaf_p, leaf_q));
        }

        // ---- Step 5: Build fraction tree for the table side ----
        let mut t_leaf_p = aggregated_multiplicities.clone();
        let mut t_leaf_q: Vec<F> = cfg_iter!(subtable)
            .map(|t_j| beta.clone() - t_j)
            .collect();
        t_leaf_p.resize(t_size, zero.clone());
        t_leaf_q.resize(t_size, one.clone());
        let table_tree = build_fraction_tree(t_leaf_p, t_leaf_q);

        // ---- Step 6: GKR proofs for each chunk ----
        let mut chunk_gkr_proofs = Vec::with_capacity(num_chunks);
        for tree in &chunk_trees {
            chunk_gkr_proofs.push(gkr_fraction_prove(transcript, tree, field_cfg));
        }

        // ---- Step 7: GKR proof for table side ----
        let table_gkr_proof = gkr_fraction_prove(transcript, &table_tree, field_cfg);

        // ---- Step 8: Root cross-check (prover-side debug) ----
        // Σ_k P_wk/Q_wk == P_t/Q_t
        // Computed as: acc = acc_P/acc_Q iteratively, then acc_P·Q_t == P_t·acc_Q
        debug_assert!({
            let mut acc_p = zero.clone();
            let mut acc_q = one.clone();
            for gkr in &chunk_gkr_proofs {
                // acc + P/Q = (acc_p·Q + P·acc_q) / (acc_q·Q)
                let new_p = acc_p.clone() * &gkr.root_q + &(gkr.root_p.clone() * &acc_q);
                let new_q = acc_q.clone() * &gkr.root_q;
                acc_p = new_p;
                acc_q = new_q;
            }
            let lhs = acc_p * &table_gkr_proof.root_q;
            let rhs = table_gkr_proof.root_p.clone() * &acc_q;
            lhs == rhs
        });

        // ---- Step 9: Verify multiplicity sum (prover-side debug) ----
        debug_assert!({
            let expected = F::from_with_cfg((num_chunks * witness_len) as u64, field_cfg);
            let sum: F = aggregated_multiplicities
                .iter()
                .cloned()
                .fold(zero, |a, b| a + &b);
            sum == expected
        });

        Ok((
            GkrDecompLogupProof {
                chunk_vectors: chunks.to_vec(),
                aggregated_multiplicities,
                chunk_gkr_proofs,
                table_gkr_proof,
            },
            GkrDecompLogupProverState {
                chunk_eval_points: vec![Vec::new(); num_chunks], // recovered by verifier
                table_eval_point: Vec::new(),
            },
        ))
    }

    /// Verifier for the GKR Decomposition + LogUp protocol.
    ///
    /// # Arguments
    ///
    /// - `transcript`: Fiat-Shamir transcript.
    /// - `proof`: The `GkrDecompLogupProof` received from the prover.
    /// - `subtable`: The sub-table entries.
    /// - `_shifts`: Shift factors (unused; decomposition consistency is
    ///   checked by the outer protocol).
    /// - `witness_len`: Length of each chunk (= length of original witness).
    /// - `field_cfg`: Field configuration.
    #[allow(clippy::arithmetic_side_effects, clippy::too_many_arguments)]
    pub fn verify_as_subprotocol(
        transcript: &mut impl Transcript,
        proof: &GkrDecompLogupProof<F>,
        subtable: &[F],
        _shifts: &[F],
        witness_len: usize,
        field_cfg: &F::Config,
    ) -> Result<GkrDecompLogupVerifierSubClaim<F>, LookupError<F>>
    where
        F::Inner: ConstTranscribable + Zero,
    {
        let zero = F::zero_with_cfg(field_cfg);
        let one = F::one_with_cfg(field_cfg);
        let mut buf = vec![0u8; F::Inner::NUM_BYTES];

        let num_chunks = proof.chunk_vectors.len();
        let chunks = &proof.chunk_vectors;
        let agg_mults = &proof.aggregated_multiplicities;

        // ---- Mirror transcript: absorb chunks ----
        for chunk in chunks {
            transcript.absorb_random_field_slice(chunk, &mut buf);
        }

        // ---- Absorb aggregated multiplicities ----
        transcript.absorb_random_field_slice(agg_mults, &mut buf);

        // ---- Get β ----
        let beta: F = transcript.get_field_challenge(field_cfg);

        let w_num_vars = zinc_utils::log2(witness_len.next_power_of_two()) as usize;
        let t_num_vars = zinc_utils::log2(subtable.len().next_power_of_two()) as usize;

        // ---- Verify GKR proofs for each chunk ----
        let mut chunk_results = Vec::with_capacity(num_chunks);
        for gkr_proof in &proof.chunk_gkr_proofs {
            let result = gkr_fraction_verify(transcript, gkr_proof, w_num_vars, field_cfg)?;
            chunk_results.push(result);
        }

        // ---- Verify GKR proof for table side ----
        let table_result =
            gkr_fraction_verify(transcript, &proof.table_gkr_proof, t_num_vars, field_cfg)?;

        // ---- Root cross-check ----
        // Σ_k P_wk/Q_wk == P_t/Q_t
        //
        // Iterate: acc = Σ fractions, then check acc_P·Q_t == P_t·acc_Q
        let mut acc_p = zero.clone();
        let mut acc_q = one.clone();
        for gkr_proof in &proof.chunk_gkr_proofs {
            let new_p =
                acc_p.clone() * &gkr_proof.root_q + &(gkr_proof.root_p.clone() * &acc_q);
            let new_q = acc_q.clone() * &gkr_proof.root_q;
            acc_p = new_p;
            acc_q = new_q;
        }
        let lhs = acc_p * &proof.table_gkr_proof.root_q;
        let rhs = proof.table_gkr_proof.root_p.clone() * &acc_q;
        if lhs != rhs {
            return Err(LookupError::GkrRootMismatch);
        }

        // ---- Verify leaf-level claims for each chunk ----
        // Each chunk tree has leaves (p, q) = (1, β − c_k(x)).
        // Leaf claim: expected_p = 1, expected_q = β − c_k̃(r_k).
        let mut chunk_eval_points = Vec::with_capacity(num_chunks);
        let mut chunk_expected_evals = Vec::with_capacity(num_chunks);

        for (k, result) in chunk_results.iter().enumerate() {
            // Check numerator = 1 (MLE of constant 1 evaluates to 1)
            if result.expected_p != one {
                return Err(LookupError::GkrLeafMismatch);
            }

            // Evaluate c_k̃ at the GKR eval point
            let c_k_eval = if result.point.is_empty() {
                chunks[k][0].clone()
            } else {
                let eq_at_pt = build_eq_x_r_vec(&result.point, field_cfg)?;
                chunks[k]
                    .iter()
                    .zip(eq_at_pt.iter())
                    .fold(zero.clone(), |acc, (c, e)| acc + &(c.clone() * e))
            };

            // Check denominator: expected_q == β − c_k̃(r_k)
            let expected_q = beta.clone() - &c_k_eval;
            if expected_q != result.expected_q {
                return Err(LookupError::GkrLeafMismatch);
            }

            chunk_eval_points.push(result.point.clone());
            chunk_expected_evals.push(c_k_eval);
        }

        // ---- Verify leaf-level claims for table side ----
        // Table tree has leaves (p, q) = (m_agg(x), β − T(x)).
        let (m_eval, t_eval) = if table_result.point.is_empty() {
            (agg_mults[0].clone(), subtable[0].clone())
        } else {
            let eq_at_t = build_eq_x_r_vec(&table_result.point, field_cfg)?;
            let m_e: F = agg_mults
                .iter()
                .zip(eq_at_t.iter())
                .fold(zero.clone(), |acc, (m_j, eq_j)| {
                    acc + &(m_j.clone() * eq_j)
                });
            let t_e: F = subtable
                .iter()
                .zip(eq_at_t.iter())
                .fold(zero.clone(), |acc, (t_j, eq_j)| {
                    acc + &(t_j.clone() * eq_j)
                });
            (m_e, t_e)
        };

        if m_eval != table_result.expected_p {
            return Err(LookupError::GkrLeafMismatch);
        }

        let expected_q = beta.clone() - &t_eval;
        if expected_q != table_result.expected_q {
            return Err(LookupError::GkrLeafMismatch);
        }

        // ---- Verify aggregated multiplicity sum = K × witness_len ----
        let expected_sum = F::from_with_cfg((num_chunks * witness_len) as u64, field_cfg);
        let m_sum: F = agg_mults.iter().cloned().fold(zero, |a, b| a + &b);
        if m_sum != expected_sum {
            return Err(LookupError::MultiplicitySumMismatch {
                expected: (num_chunks * witness_len) as u64,
                got: 0,
            });
        }

        Ok(GkrDecompLogupVerifierSubClaim {
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

    /// Test with K=2 chunks: BitPoly(4) → two BitPoly(2) sub-tables.
    #[test]
    fn gkr_decomp_logup_2_chunks_bitpoly() {
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
            full_table[0].clone(),
            full_table[3].clone(),
            full_table[5].clone(),
            full_table[15].clone(),
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
        let (proof, _) = GkrDecompLogupProtocol::<F>::prove_as_subprotocol(
            &mut prover_transcript,
            &instance,
            &(),
        )
        .expect("prover should succeed");

        let mut verifier_transcript = KeccakTranscript::new();
        let _subclaim = GkrDecompLogupProtocol::<F>::verify_as_subprotocol(
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
    fn gkr_decomp_logup_4_chunks_bitpoly() {
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
        let (proof, _) = GkrDecompLogupProtocol::<F>::prove_as_subprotocol(
            &mut prover_transcript,
            &instance,
            &(),
        )
        .expect("prover should succeed");

        let mut verifier_transcript = KeccakTranscript::new();
        let _subclaim = GkrDecompLogupProtocol::<F>::verify_as_subprotocol(
            &mut verifier_transcript,
            &proof,
            &subtable,
            &shifts,
            witness.len(),
            &(),
        )
        .expect("verifier should accept");
    }

    /// Test with K=4 chunks and a larger witness (16 entries).
    #[test]
    fn gkr_decomp_logup_4_chunks_larger_witness() {
        let a = F::from(3u32);

        let chunk_width = 2;
        let num_chunks = 4;
        let subtable = generate_bitpoly_table(chunk_width, &a, &());
        let full_table = generate_bitpoly_table(chunk_width * num_chunks, &a, &());

        let shifts: Vec<F> = (0..num_chunks)
            .map(|k| bitpoly_shift(k * chunk_width, &a))
            .collect();

        // 16 entries with repetitions
        let indices = [
            0usize, 1, 2, 3, 42, 42, 127, 127, 200, 200, 255, 255, 170, 85, 13, 0,
        ];
        let witness: Vec<F> = indices.iter().map(|&i| full_table[i].clone()).collect();

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
        let (proof, _) = GkrDecompLogupProtocol::<F>::prove_as_subprotocol(
            &mut prover_transcript,
            &instance,
            &(),
        )
        .expect("prover should succeed");

        let mut verifier_transcript = KeccakTranscript::new();
        let _subclaim = GkrDecompLogupProtocol::<F>::verify_as_subprotocol(
            &mut verifier_transcript,
            &proof,
            &subtable,
            &shifts,
            witness.len(),
            &(),
        )
        .expect("verifier should accept");
    }

    /// Reject a chunk that contains an entry not in the subtable.
    #[test]
    fn gkr_decomp_logup_reject_bad_chunk() {
        let a = F::from(3u32);

        let chunk_width = 2;
        let num_chunks = 2;
        let subtable = generate_bitpoly_table(chunk_width, &a, &());
        let full_table = generate_bitpoly_table(chunk_width * num_chunks, &a, &());

        let shifts: Vec<F> = (0..num_chunks)
            .map(|k| bitpoly_shift(k * chunk_width, &a))
            .collect();

        let witness = vec![full_table[0].clone(), full_table[3].clone()];

        // First chunk is valid, second chunk has an entry NOT in subtable
        let bad_entry = F::from(999u32); // not in the 4-element subtable
        let chunks = vec![
            vec![subtable[0].clone(), subtable[3].clone()],
            vec![subtable[0].clone(), bad_entry],
        ];

        let instance = DecompLookupInstance {
            witness,
            subtable: subtable.clone(),
            shifts,
            chunks,
        };

        let mut prover_transcript = KeccakTranscript::new();
        let result = GkrDecompLogupProtocol::<F>::prove_as_subprotocol(
            &mut prover_transcript,
            &instance,
            &(),
        );

        assert!(result.is_err());
    }

    /// Test with K=2 chunks, single-entry witness.
    #[test]
    fn gkr_decomp_logup_single_entry() {
        let a = F::from(3u32);

        let chunk_width = 2;
        let num_chunks = 2;
        let subtable = generate_bitpoly_table(chunk_width, &a, &());
        let full_table = generate_bitpoly_table(chunk_width * num_chunks, &a, &());

        let shifts: Vec<F> = (0..num_chunks)
            .map(|k| bitpoly_shift(k * chunk_width, &a))
            .collect();

        // Single witness entry: index 7
        let idx = 7usize;
        let witness = vec![full_table[idx].clone()];

        let mask = (1usize << chunk_width) - 1;
        let chunks: Vec<Vec<F>> = (0..num_chunks)
            .map(|k| vec![subtable[(idx >> (k * chunk_width)) & mask].clone()])
            .collect();

        let instance = DecompLookupInstance {
            witness: witness.clone(),
            subtable: subtable.clone(),
            shifts: shifts.clone(),
            chunks,
        };

        let mut prover_transcript = KeccakTranscript::new();
        let (proof, _) = GkrDecompLogupProtocol::<F>::prove_as_subprotocol(
            &mut prover_transcript,
            &instance,
            &(),
        )
        .expect("prover should succeed");

        let mut verifier_transcript = KeccakTranscript::new();
        let _subclaim = GkrDecompLogupProtocol::<F>::verify_as_subprotocol(
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
