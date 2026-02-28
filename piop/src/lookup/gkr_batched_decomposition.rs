//! Batched GKR Decomposition + LogUp protocol.
//!
//! Given L witness vectors that all look up into the same decomposed
//! table (e.g. BitPoly(32) → K sub-tables of BitPoly(8)), this
//! protocol proves all L·K chunk lookups using **two GKR fractional
//! sumchecks** — one combined witness tree and one combined table tree.
//!
//! ## Key advantage over [`super::batched_decomposition`]
//!
//! - **No inverse vectors** (`u`, `v`) are sent — eliminated by the GKR
//!   fractional sumcheck structure.
//! - **No chunk vectors** are sent — assumed committed by the PCS
//!   (Zip+ column commitments serve as chunk commitments).
//! - Only **aggregated multiplicities** are sent in the clear.
//!
//! ## Protocol overview
//!
//! 1. Prover sends aggregated multiplicities `m_agg^(ℓ)` for each lookup.
//! 2. Challenges β (shift) and α (batching) are derived.
//! 3. A combined witness fraction tree is built:
//!    leaf `(ℓ, k, i)` → numerator `α^ℓ`, denominator `β − c_k^(ℓ)[i]`.
//! 4. A combined table fraction tree is built:
//!    leaf `j` → numerator `Σ_ℓ α^ℓ · m_agg^(ℓ)[j]`, denominator `β − T[j]`.
//! 5. Two GKR fractional sumchecks prove the sums match (root cross-check).
//! 6. Verifier leaf-check reduces to chunk MLE evaluations (provided by PCS).

use crypto_primitives::FromPrimitiveWithConfig;
use num_traits::Zero;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::marker::PhantomData;
use zinc_poly::utils::build_eq_x_r_vec;
use zinc_transcript::traits::{ConstTranscribable, Transcript};
use zinc_utils::{cfg_iter, inner_transparent_field::InnerTransparentField};

use super::gkr_logup::{
    build_fraction_tree, gkr_fraction_prove, gkr_fraction_verify,
};
use super::structs::{
    BatchedDecompLookupInstance, GkrBatchedDecompLogupProof,
    GkrBatchedDecompLogupProverState, GkrBatchedDecompLogupVerifierSubClaim,
    LookupError,
};
use super::tables::{build_table_index, compute_multiplicities_with_index};

/// The batched GKR Decomposition + LogUp protocol.
pub struct GkrBatchedDecompLogupProtocol<F: InnerTransparentField>(PhantomData<F>);

impl<F: InnerTransparentField + FromPrimitiveWithConfig + Send + Sync>
    GkrBatchedDecompLogupProtocol<F>
{
    /// Prover for the batched GKR Decomposition + LogUp protocol.
    ///
    /// All L witness vectors must have the **same length** and share
    /// the same sub-table / shift factors.
    ///
    /// # Arguments
    ///
    /// - `transcript`: Fiat-Shamir transcript.
    /// - `instance`: The batched lookup instance (witnesses, chunks, subtable, shifts).
    /// - `field_cfg`: Field configuration.
    ///
    /// # Returns
    ///
    /// `(GkrBatchedDecompLogupProof, GkrBatchedDecompLogupProverState)` on success.
    #[allow(clippy::arithmetic_side_effects, clippy::too_many_arguments)]
    pub fn prove_as_subprotocol(
        transcript: &mut impl Transcript,
        instance: &BatchedDecompLookupInstance<F>,
        field_cfg: &F::Config,
    ) -> Result<
        (GkrBatchedDecompLogupProof<F>, GkrBatchedDecompLogupProverState<F>),
        LookupError<F>,
    >
    where
        F::Inner: ConstTranscribable + Zero + Default + Send + Sync,
        F::Config: Sync,
    {
        let witnesses = &instance.witnesses;
        let subtable = &instance.subtable;
        let all_chunks = &instance.chunks;

        let num_lookups = witnesses.len(); // L
        let num_chunks = instance.shifts.len(); // K
        let witness_len = witnesses[0].len(); // W
        let table_len = subtable.len(); // N

        let zero = F::zero_with_cfg(field_cfg);
        let one = F::one_with_cfg(field_cfg);
        let mut buf = vec![0u8; F::Inner::NUM_BYTES];

        // ---- Step 1: Compute per-chunk multiplicities, then aggregate ----
        let table_index = build_table_index(subtable);
        let all_chunk_multiplicities: Vec<Vec<Vec<F>>> = cfg_iter!(all_chunks)
            .map(|lookup_chunks| {
                lookup_chunks
                    .iter()
                    .map(|chunk| {
                        compute_multiplicities_with_index(
                            chunk, &table_index, table_len, field_cfg,
                        )
                        .ok_or(LookupError::WitnessNotInTable)
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<_, _>>()?;

        let all_aggregated_multiplicities: Vec<Vec<F>> = all_chunk_multiplicities
            .iter()
            .map(|lookup_mults| {
                let mut agg = vec![zero.clone(); table_len];
                for m in lookup_mults {
                    for (a, mk) in agg.iter_mut().zip(m.iter()) {
                        *a += mk;
                    }
                }
                agg
            })
            .collect();

        // ---- Step 2: Absorb aggregated multiplicities ----
        for agg in &all_aggregated_multiplicities {
            transcript.absorb_random_field_slice(agg, &mut buf);
        }

        // ---- Step 3: Challenges β and α ----
        let beta: F = transcript.get_field_challenge(field_cfg);
        let alpha: F = transcript.get_field_challenge(field_cfg);

        // Precompute α powers: α^0, α^1, ..., α^{L-1}
        let mut alpha_powers = Vec::with_capacity(num_lookups);
        let mut ap = one.clone();
        for _ in 0..num_lookups {
            alpha_powers.push(ap.clone());
            ap *= &alpha;
        }

        // ---- Step 4: Build combined witness fraction tree ----
        // Total leaves = L * K * W, padded to next power of 2.
        let total_witness_leaves = num_lookups * num_chunks * witness_len;
        let w_num_vars = zinc_utils::log2(total_witness_leaves.next_power_of_two()) as usize;
        let w_size = 1usize << w_num_vars;

        // Leaf ordering: flat index = ℓ * (K * W) + k * W + i
        // leaf_p[idx] = α^ℓ
        // leaf_q[idx] = β - chunks[ℓ][k][i]
        let mut w_leaf_p = Vec::with_capacity(w_size);
        let mut w_leaf_q = Vec::with_capacity(w_size);

        for ell in 0..num_lookups {
            for k in 0..num_chunks {
                for i in 0..witness_len {
                    w_leaf_p.push(alpha_powers[ell].clone());
                    w_leaf_q.push(beta.clone() - &all_chunks[ell][k][i]);
                }
            }
        }
        // Pad with (0, 1) — zero fraction, doesn't affect the sum.
        w_leaf_p.resize(w_size, zero.clone());
        w_leaf_q.resize(w_size, one.clone());

        let witness_tree = build_fraction_tree(w_leaf_p, w_leaf_q);

        // ---- Step 5: Build combined table fraction tree ----
        // Leaves = N (subtable size), padded to next power of 2.
        let t_num_vars = zinc_utils::log2(table_len.next_power_of_two()) as usize;
        let t_size = 1usize << t_num_vars;

        // leaf_p[j] = Σ_ℓ α^ℓ · m_agg^(ℓ)[j]
        // leaf_q[j] = β - T[j]
        let mut t_leaf_p = Vec::with_capacity(t_size);
        let mut t_leaf_q = Vec::with_capacity(t_size);

        for j in 0..table_len {
            let mut combined_mult = zero.clone();
            for ell in 0..num_lookups {
                combined_mult += &(alpha_powers[ell].clone() * &all_aggregated_multiplicities[ell][j]);
            }
            t_leaf_p.push(combined_mult);
            t_leaf_q.push(beta.clone() - &subtable[j]);
        }
        t_leaf_p.resize(t_size, zero.clone());
        t_leaf_q.resize(t_size, one.clone());

        let table_tree = build_fraction_tree(t_leaf_p, t_leaf_q);

        // ---- Step 6: GKR fractional sumcheck for witness tree ----
        let witness_gkr = gkr_fraction_prove(transcript, &witness_tree, field_cfg);

        // ---- Step 7: GKR fractional sumcheck for table tree ----
        let table_gkr = gkr_fraction_prove(transcript, &table_tree, field_cfg);

        // ---- Step 8: Prover-side root cross-check (debug) ----
        debug_assert!({
            let lhs = witness_gkr.root_p.clone() * &table_gkr.root_q;
            let rhs = table_gkr.root_p.clone() * &witness_gkr.root_q;
            lhs == rhs
        });

        // ---- Sanity check: multiplicity sums ----
        debug_assert!({
            let expected_per_lookup = F::from_with_cfg((num_chunks * witness_len) as u64, field_cfg);
            all_aggregated_multiplicities.iter().all(|agg| {
                let sum: F = agg.iter().cloned().fold(zero.clone(), |a, b| a + &b);
                sum == expected_per_lookup
            })
        });

        // ---- Recover evaluation points from GKR ----
        // The prover re-runs the verifier logic to extract the evaluation
        // points. We use a separate transcript re-derive (or just track
        // them during the prove).  For now, we return empty and let the
        // caller re-derive via the verifier.
        //
        // Actually, the prover needs these for PCS integration, so we
        // re-derive them by running the verifier-side challenge extraction
        // in a cloned transcript.  But since the prover already built the
        // trees, we can extract the points directly by tracking r_k during
        // the prove loop.  The gkr_fraction_prove function doesn't return
        // r_k, so we re-derive it here.
        //
        // For efficiency, we reconstruct by re-running the verification
        // on the already-produced proofs with a fresh transcript that tracks
        // the same state.
        let witness_eval_point = recover_eval_point_from_proof(&witness_gkr, w_num_vars, field_cfg);
        let table_eval_point = recover_eval_point_from_proof(&table_gkr, t_num_vars, field_cfg);

        Ok((
            GkrBatchedDecompLogupProof {
                aggregated_multiplicities: all_aggregated_multiplicities,
                witness_gkr,
                table_gkr,
            },
            GkrBatchedDecompLogupProverState {
                witness_eval_point,
                table_eval_point,
                witness_num_vars: w_num_vars,
                table_num_vars: t_num_vars,
                alpha: alpha.clone(),
                beta: beta.clone(),
                num_lookups,
                num_chunks,
                witness_len,
            },
        ))
    }

    /// Verifier for the batched GKR Decomposition + LogUp protocol.
    ///
    /// # Arguments
    ///
    /// - `transcript`: Fiat-Shamir transcript.
    /// - `proof`: The GKR batched decomp logup proof.
    /// - `subtable`: The shared sub-table entries.
    /// - `shifts`: Shift factors (K entries).
    /// - `num_lookups`: Number of lookups L.
    /// - `witness_len`: Length of each witness vector W.
    /// - `field_cfg`: Field configuration.
    ///
    /// # Returns
    ///
    /// `GkrBatchedDecompLogupVerifierSubClaim` on success, reducing the
    /// lookup verification to chunk MLE evaluations that the PCS must provide.
    #[allow(clippy::arithmetic_side_effects, clippy::too_many_arguments)]
    pub fn verify_as_subprotocol(
        transcript: &mut impl Transcript,
        proof: &GkrBatchedDecompLogupProof<F>,
        subtable: &[F],
        shifts: &[F],
        num_lookups: usize,
        witness_len: usize,
        field_cfg: &F::Config,
    ) -> Result<GkrBatchedDecompLogupVerifierSubClaim<F>, LookupError<F>>
    where
        F::Inner: ConstTranscribable + Zero + Default + Send + Sync,
        F::Config: Sync,
    {
        let zero = F::zero_with_cfg(field_cfg);
        let one = F::one_with_cfg(field_cfg);
        let mut buf = vec![0u8; F::Inner::NUM_BYTES];

        let num_chunks = shifts.len();
        let table_len = subtable.len();
        let all_agg_mults = &proof.aggregated_multiplicities;

        // ---- Step 1: Absorb aggregated multiplicities ----
        for agg in all_agg_mults {
            transcript.absorb_random_field_slice(agg, &mut buf);
        }

        // ---- Step 2: Challenges β and α ----
        let beta: F = transcript.get_field_challenge(field_cfg);
        let alpha: F = transcript.get_field_challenge(field_cfg);

        let mut alpha_powers = Vec::with_capacity(num_lookups);
        let mut ap = one.clone();
        for _ in 0..num_lookups {
            alpha_powers.push(ap.clone());
            ap *= &alpha;
        }

        // ---- Step 3: Compute tree dimensions ----
        let total_witness_leaves = num_lookups * num_chunks * witness_len;
        let w_num_vars = zinc_utils::log2(total_witness_leaves.next_power_of_two()) as usize;
        let t_num_vars = zinc_utils::log2(table_len.next_power_of_two()) as usize;

        // ---- Step 4: Verify witness-side GKR ----
        let witness_result = gkr_fraction_verify(
            transcript,
            &proof.witness_gkr,
            w_num_vars,
            field_cfg,
        )?;

        // ---- Step 5: Verify table-side GKR ----
        let table_result = gkr_fraction_verify(
            transcript,
            &proof.table_gkr,
            t_num_vars,
            field_cfg,
        )?;

        // ---- Step 6: Cross-check roots ----
        // P_w · Q_t == P_t · Q_w  ⟺  P_w/Q_w == P_t/Q_t
        let lhs = proof.witness_gkr.root_p.clone() * &proof.table_gkr.root_q;
        let rhs = proof.table_gkr.root_p.clone() * &proof.witness_gkr.root_q;
        if lhs != rhs {
            return Err(LookupError::GkrRootMismatch);
        }

        // ---- Step 7: Verify table-side leaf claims ----
        // Table leaves: p_j = Σ_ℓ α^ℓ · m_agg^(ℓ)[j], q_j = β − T[j]
        // The verifier can compute these directly.
        if table_result.point.is_empty() {
            // 0-variable case: single entry.
            let expected_p = {
                let mut val = zero.clone();
                for ell in 0..num_lookups {
                    val += &(alpha_powers[ell].clone() * &all_agg_mults[ell][0]);
                }
                val
            };
            let expected_q = beta.clone() - &subtable[0];

            if expected_p != table_result.expected_p || expected_q != table_result.expected_q {
                return Err(LookupError::GkrLeafMismatch);
            }
        } else {
            let eq_at_t = build_eq_x_r_vec(&table_result.point, field_cfg)?;

            // p̃_t(r_t) = Σ_j eq(j, r_t) · [Σ_ℓ α^ℓ · m_agg^(ℓ)[j]]
            let mut p_eval = zero.clone();
            for j in 0..table_len {
                let mut combined_mult = zero.clone();
                for ell in 0..num_lookups {
                    combined_mult += &(alpha_powers[ell].clone() * &all_agg_mults[ell][j]);
                }
                p_eval += &(combined_mult * &eq_at_t[j]);
            }
            // Pad entries contribute 0 to p.

            // q̃_t(r_t) = Σ_j eq(j, r_t) · (β − T[j]) + Σ_{j≥N} eq(j, r_t) · 1
            let mut q_eval = zero.clone();
            for j in 0..table_len {
                q_eval += &((beta.clone() - &subtable[j]) * &eq_at_t[j]);
            }
            // Padding entries have q = 1.
            for j in table_len..eq_at_t.len() {
                q_eval += &eq_at_t[j];
            }

            if p_eval != table_result.expected_p {
                return Err(LookupError::GkrLeafMismatch);
            }
            if q_eval != table_result.expected_q {
                return Err(LookupError::GkrLeafMismatch);
            }
        }

        // ---- Step 8: Verify multiplicity sums ----
        let expected_sum = F::from_with_cfg((num_chunks * witness_len) as u64, field_cfg);
        for ell in 0..num_lookups {
            let m_sum: F = all_agg_mults[ell].iter().cloned().fold(zero.clone(), |a, b| a + &b);
            if m_sum != expected_sum {
                return Err(LookupError::MultiplicitySumMismatch {
                    expected: (num_chunks * witness_len) as u64,
                    got: 0,
                });
            }
        }

        // ---- Step 9: Witness-side leaf subclaim ----
        // Witness leaves: p = α^ℓ (structured), q = β − chunk value.
        // The verifier can compute the expected p̃_w(r_w) from α and the
        // tree structure.  For q̃_w(r_w), the verifier needs the combined
        // chunk MLE evaluation from PCS.
        //
        // p̃_w(r_w):
        // Leaf ordering: flat idx = ℓ*(K*W) + k*W + i, padded to 2^{d_w}.
        // p[idx] = α^ℓ for idx in [ℓ*K*W, (ℓ+1)*K*W), and 0 for padding.
        //
        // This is a structured MLE: each α^ℓ is repeated K*W times.
        // We can evaluate it using eq weights over the 'ℓ' coordinates.
        let expected_p_eval = compute_witness_p_eval(
            &witness_result.point,
            &alpha_powers,
            num_lookups,
            num_chunks,
            witness_len,
            w_num_vars,
            field_cfg,
        );

        if expected_p_eval != witness_result.expected_p {
            return Err(LookupError::GkrLeafMismatch);
        }

        // q̃_w(r_w) = β · (sum of eq weights over data region) - combined_chunk_eval
        //           + (sum of eq weights over padding region)     [padding has q=1]
        //
        // The combined_chunk_eval is what the PCS must provide.
        // We return the expected q value so the pipeline can derive the
        // chunk evaluation from it.

        Ok(GkrBatchedDecompLogupVerifierSubClaim {
            witness_eval_point: witness_result.point,
            expected_witness_q_eval: witness_result.expected_q,
            expected_witness_p_eval: witness_result.expected_p,
            table_eval_point: table_result.point,
        })
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute the expected evaluation of the witness numerator MLE p̃_w(r).
///
/// The witness numerator is structured: p[ℓ*K*W + k*W + i] = α^ℓ for
/// data entries, 0 for padding.  This is evaluated efficiently using
/// the structure.
#[allow(clippy::arithmetic_side_effects)]
fn compute_witness_p_eval<F>(
    point: &[F],
    alpha_powers: &[F],
    num_lookups: usize,
    num_chunks: usize,
    witness_len: usize,
    num_vars: usize,
    field_cfg: &F::Config,
) -> F
where
    F: InnerTransparentField + FromPrimitiveWithConfig + Send + Sync,
    F::Config: Sync,
{
    let zero = F::zero_with_cfg(field_cfg);

    if point.is_empty() {
        // 0-variable: single leaf, return α^0 = 1 (if there's data)
        return if num_lookups > 0 { alpha_powers[0].clone() } else { zero };
    }

    // Build the full eq vector and sum over data positions.
    let total_size = 1usize << num_vars;
    let block_size = num_chunks * witness_len; // K * W entries per lookup

    let eq_vec = build_eq_x_r_vec(point, field_cfg)
        .expect("eq vector construction should succeed");

    let mut result = zero;
    for ell in 0..num_lookups {
        let base = ell * block_size;
        let end = base + block_size;
        if end > total_size {
            break;
        }
        let mut block_sum = F::zero_with_cfg(field_cfg);
        for j in base..end {
            block_sum += &eq_vec[j];
        }
        result += &(alpha_powers[ell].clone() * &block_sum);
    }

    result
}

/// Recover the evaluation point from a GKR fraction proof by replaying
/// the transcript challenge derivation.
///
/// The evaluation point is built round-by-round: for round 0, it's `(λ_0)`;
/// for round k ≥ 1, it's `(sumcheck_randomness, λ_k)`.
/// Since we don't have the transcript state, we extract from the proof
/// structure by noting that the sumcheck proofs contain their randomness
/// implicitly (the verifier derives it from the proof messages).
///
/// For now, we return an empty vector; the prover can re-derive by
/// running verify on the same transcript state.  The pipeline integration
/// handles this by running verify after prove.
#[allow(dead_code)]
fn recover_eval_point_from_proof<F: InnerTransparentField + FromPrimitiveWithConfig>(
    _proof: &super::structs::GkrFractionProof<F>,
    _num_vars: usize,
    _field_cfg: &F::Config,
) -> Vec<F> {
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lookup::tables::{bitpoly_shift, generate_bitpoly_table};
    use crate::lookup::structs::BatchedDecompLookupInstance;
    use crypto_bigint::{U128, const_monty_params};
    use crypto_primitives::crypto_bigint_const_monty::ConstMontyField;
    use zinc_transcript::KeccakTranscript;

    const N: usize = 2;
    const_monty_params!(TestParams, U128, "00000000b933426489189cb5b47d567f");
    type F = ConstMontyField<TestParams, N>;

    /// Helper: build a single lookup from indices into the full table.
    fn make_lookup(
        indices: &[usize],
        subtable: &[F],
        chunk_width: usize,
        num_chunks: usize,
        shifts: &[F],
    ) -> (Vec<F>, Vec<Vec<F>>) {
        let mask = (1usize << chunk_width) - 1;
        let chunks: Vec<Vec<F>> = (0..num_chunks)
            .map(|k| {
                indices
                    .iter()
                    .map(|&idx| {
                        subtable[(idx >> (k * chunk_width)) & mask].clone()
                    })
                    .collect()
            })
            .collect();
        let witness: Vec<F> = (0..indices.len())
            .map(|i| {
                let mut val = F::from(0u32);
                for k in 0..num_chunks {
                    val += &(shifts[k].clone() * &chunks[k][i]);
                }
                val
            })
            .collect();
        (witness, chunks)
    }

    /// GKR batched lookup with L=3, K=2, BitPoly(4).
    #[test]
    fn gkr_batched_3_lookups_2_chunks() {
        let a = F::from(3u32);
        let chunk_width = 2;
        let num_chunks = 2;
        let subtable = generate_bitpoly_table(chunk_width, &a, &());
        let shifts: Vec<F> = (0..num_chunks)
            .map(|k| bitpoly_shift(k * chunk_width, &a))
            .collect();

        let (w0, c0) = make_lookup(&[0, 3, 5, 15], &subtable, chunk_width, num_chunks, &shifts);
        let (w1, c1) = make_lookup(&[1, 2, 7, 10], &subtable, chunk_width, num_chunks, &shifts);
        let (w2, c2) = make_lookup(&[15, 15, 0, 8], &subtable, chunk_width, num_chunks, &shifts);

        let instance = BatchedDecompLookupInstance {
            witnesses: vec![w0, w1, w2],
            subtable: subtable.clone(),
            shifts: shifts.clone(),
            chunks: vec![c0, c1, c2],
        };

        let mut pt = KeccakTranscript::new();
        let (proof, _state) =
            GkrBatchedDecompLogupProtocol::<F>::prove_as_subprotocol(
                &mut pt, &instance, &(),
            )
            .expect("prover should succeed");

        let mut vt = KeccakTranscript::new();
        let _sub = GkrBatchedDecompLogupProtocol::<F>::verify_as_subprotocol(
            &mut vt,
            &proof,
            &subtable,
            &shifts,
            3,
            4,
            &(),
        )
        .expect("verifier should accept");
    }

    /// GKR batched lookup with L=5, K=4, BitPoly(8).
    #[test]
    fn gkr_batched_5_lookups_4_chunks() {
        let a = F::from(3u32);
        let chunk_width = 2;
        let num_chunks = 4;
        let subtable = generate_bitpoly_table(chunk_width, &a, &());
        let shifts: Vec<F> = (0..num_chunks)
            .map(|k| bitpoly_shift(k * chunk_width, &a))
            .collect();

        let index_sets: Vec<Vec<usize>> = vec![
            vec![0, 42, 127, 200, 255, 170, 85, 13],
            vec![1, 2, 3, 4, 5, 6, 7, 8],
            vec![100, 200, 50, 150, 250, 10, 90, 180],
            vec![255, 0, 128, 64, 32, 16, 8, 4],
            vec![17, 34, 68, 136, 119, 238, 221, 187],
        ];

        let mut witnesses = Vec::new();
        let mut all_chunks = Vec::new();
        for indices in &index_sets {
            let (w, c) = make_lookup(indices, &subtable, chunk_width, num_chunks, &shifts);
            witnesses.push(w);
            all_chunks.push(c);
        }

        let instance = BatchedDecompLookupInstance {
            witnesses,
            subtable: subtable.clone(),
            shifts: shifts.clone(),
            chunks: all_chunks,
        };

        let mut pt = KeccakTranscript::new();
        let (proof, _state) =
            GkrBatchedDecompLogupProtocol::<F>::prove_as_subprotocol(
                &mut pt, &instance, &(),
            )
            .expect("prover should succeed");

        let mut vt = KeccakTranscript::new();
        let _sub = GkrBatchedDecompLogupProtocol::<F>::verify_as_subprotocol(
            &mut vt,
            &proof,
            &subtable,
            &shifts,
            5,
            8,
            &(),
        )
        .expect("verifier should accept");
    }

    /// Single lookup (L=1) should work correctly.
    #[test]
    fn gkr_batched_single_lookup() {
        let a = F::from(3u32);
        let chunk_width = 2;
        let num_chunks = 2;
        let subtable = generate_bitpoly_table(chunk_width, &a, &());
        let shifts: Vec<F> = (0..num_chunks)
            .map(|k| bitpoly_shift(k * chunk_width, &a))
            .collect();

        let (w, c) = make_lookup(&[0, 5, 10, 15], &subtable, chunk_width, num_chunks, &shifts);

        let instance = BatchedDecompLookupInstance {
            witnesses: vec![w],
            subtable: subtable.clone(),
            shifts: shifts.clone(),
            chunks: vec![c],
        };

        let mut pt = KeccakTranscript::new();
        let (proof, _state) =
            GkrBatchedDecompLogupProtocol::<F>::prove_as_subprotocol(
                &mut pt, &instance, &(),
            )
            .expect("prover should succeed");

        let mut vt = KeccakTranscript::new();
        let _sub = GkrBatchedDecompLogupProtocol::<F>::verify_as_subprotocol(
            &mut vt,
            &proof,
            &subtable,
            &shifts,
            1,
            4,
            &(),
        )
        .expect("verifier should accept");
    }
}
