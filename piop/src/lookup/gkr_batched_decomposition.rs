//! Batched GKR Decomposition + LogUp protocol.
//!
//! Given L witness vectors that all look up into the same decomposed
//! table (e.g. BitPoly(32) → K sub-tables of BitPoly(8)), this
//! protocol proves all L·K chunk lookups using a **batched GKR
//! fractional sumcheck** — L separate per-lookup witness trees with
//! layer-wise batched sumchecks, plus one combined table tree.
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
//! 3. For each lookup ℓ, a per-lookup witness fraction tree is built:
//!    leaf `(k, i)` → numerator `1`, denominator `β − c_k^(ℓ)[i]`.
//! 4. A combined table fraction tree is built:
//!    leaf `j` → numerator `Σ_ℓ α^ℓ · m_agg^(ℓ)[j]`, denominator `β − T[j]`.
//! 5. A batched GKR fractional sumcheck processes all L witness trees
//!    layer-by-layer in a single combined sumcheck per layer.
//! 6. A single GKR fractional sumcheck proves the table tree.
//! 7. Cross-check: `Σ_ℓ α^ℓ · P_w^(ℓ)/Q_w^(ℓ) == P_t/Q_t`.
//! 8. Verifier leaf-check reduces to chunk MLE evaluations (provided by PCS).

use crypto_primitives::FromPrimitiveWithConfig;
use num_traits::Zero;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::marker::PhantomData;
use zinc_poly::utils::build_eq_x_r_vec;
use zinc_transcript::traits::{ConstTranscribable, Transcript};
use zinc_utils::{cfg_iter, cfg_into_iter, inner_transparent_field::InnerTransparentField};

use super::gkr_logup::{
    build_fraction_tree, build_fraction_tree_ones_leaf, gkr_fraction_prove, gkr_fraction_verify,
    batched_gkr_fraction_prove, batched_gkr_fraction_verify,
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

        // ---- Step 4: Build L per-lookup witness fraction trees ----
        // Each tree has K*W leaves (padded to next power of 2).
        // leaf_p[(k, i)] = 1   (numerators are all 1 per-tree)
        // leaf_q[(k, i)] = β − chunks[ℓ][k][i]
        let per_lookup_leaves = num_chunks * witness_len; // K * W
        let w_num_vars = zinc_utils::log2(per_lookup_leaves.next_power_of_two()) as usize;
        let w_size = 1usize << w_num_vars;

        let leaf_p_all_ones = per_lookup_leaves == w_size;
        let witness_trees: Vec<_> = (0..num_lookups)
            .map(|ell| {
                let mut leaf_q = Vec::with_capacity(w_size);
                for k in 0..num_chunks {
                    for i in 0..witness_len {
                        leaf_q.push(beta.clone() - &all_chunks[ell][k][i]);
                    }
                }
                if leaf_p_all_ones {
                    build_fraction_tree_ones_leaf(one.clone(), leaf_q)
                } else {
                    let mut leaf_p = vec![one.clone(); per_lookup_leaves];
                    leaf_p.resize(w_size, zero.clone());
                    leaf_q.resize(w_size, one.clone());
                    build_fraction_tree(leaf_p, leaf_q)
                }
            })
            .collect();

        // ---- Step 5: Build combined table fraction tree ----
        // Leaves = N (subtable size), padded to next power of 2.
        let t_num_vars = zinc_utils::log2(table_len.next_power_of_two()) as usize;
        let t_size = 1usize << t_num_vars;

        // leaf_p[j] = Σ_ℓ α^ℓ · m_agg^(ℓ)[j]
        // leaf_q[j] = β - T[j]
        let (mut t_leaf_p, mut t_leaf_q): (Vec<F>, Vec<F>) =
            cfg_into_iter!(0..table_len, 256)
                .map(|j| {
                    let mut combined_mult = zero.clone();
                    for ell in 0..num_lookups {
                        combined_mult += &(alpha_powers[ell].clone()
                            * &all_aggregated_multiplicities[ell][j]);
                    }
                    (combined_mult, beta.clone() - &subtable[j])
                })
                .unzip();
        t_leaf_p.resize(t_size, zero.clone());
        t_leaf_q.resize(t_size, one.clone());

        let table_tree = build_fraction_tree(t_leaf_p, t_leaf_q);

        // ---- Step 6: Batched GKR fractional sumcheck for L witness trees ----
        let witness_result =
            batched_gkr_fraction_prove(transcript, &witness_trees, field_cfg);
        let witness_eval_point = witness_result.eval_point;

        // ---- Step 7: GKR fractional sumcheck for table tree ----
        let (table_gkr, table_eval_point) =
            gkr_fraction_prove(transcript, &table_tree, field_cfg);

        // ---- Step 8: Prover-side root cross-check (debug) ----
        // Σ_ℓ α^ℓ · P_w^(ℓ)/Q_w^(ℓ) == P_t/Q_t
        // ⟺  (Σ_ℓ α^ℓ · P_w^(ℓ) · Π_{j≠ℓ} Q_w^(j)) · Q_t == P_t · Π_ℓ Q_w^(ℓ)
        debug_assert!({
            // Simple form: cross-multiply with all Q_w and Q_t
            let witness_gkr = &witness_result.proof;
            let q_w_product: F = witness_gkr.roots_q.iter().cloned()
                .fold(one.clone(), |acc, q| acc * &q);
            let mut lhs = zero.clone();
            for ell in 0..num_lookups {
                let mut others_q = one.clone();
                for j in 0..num_lookups {
                    if j != ell {
                        others_q *= &witness_gkr.roots_q[j];
                    }
                }
                lhs += &(alpha_powers[ell].clone() * &witness_gkr.roots_p[ell] * &others_q);
            }
            lhs *= &table_gkr.root_q;
            let rhs = table_gkr.root_p.clone() * &q_w_product;
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

        Ok((
            GkrBatchedDecompLogupProof {
                aggregated_multiplicities: all_aggregated_multiplicities,
                witness_gkr: witness_result.proof,
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
    /// Uses L separate per-lookup witness trees (batched layer-wise)
    /// and a single α-batched table tree.
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
    /// `GkrBatchedDecompLogupVerifierSubClaim` on success.
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
        let per_lookup_leaves = num_chunks * witness_len; // K * W per-tree
        let w_num_vars = zinc_utils::log2(per_lookup_leaves.next_power_of_two()) as usize;
        let t_num_vars = zinc_utils::log2(table_len.next_power_of_two()) as usize;

        // ---- Step 4: Verify witness-side batched GKR (L trees) ----
        let witness_result = batched_gkr_fraction_verify(
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
        // Σ_ℓ α^ℓ · P_w^(ℓ)/Q_w^(ℓ) == P_t/Q_t
        // ⟺  (Σ_ℓ α^ℓ · P_w^(ℓ) · Π_{j≠ℓ} Q_w^(j)) · Q_t == P_t · Π_ℓ Q_w^(ℓ)
        // Uses prefix/suffix products for O(L) instead of O(L²) multiplications.
        {
            let roots_q = &proof.witness_gkr.roots_q;
            let q_w_product: F = roots_q.iter().cloned()
                .fold(one.clone(), |acc, q| acc * &q);
            let mut lhs = zero.clone();
            if num_lookups <= 1 {
                if num_lookups == 1 {
                    lhs += &(alpha_powers[0].clone() * &proof.witness_gkr.roots_p[0]);
                }
            } else {
                let mut prefix = Vec::with_capacity(num_lookups);
                prefix.push(one.clone());
                for i in 1..num_lookups {
                    prefix.push(prefix[i - 1].clone() * &roots_q[i - 1]);
                }
                let mut suffix = vec![one.clone(); num_lookups];
                for i in (0..num_lookups - 1).rev() {
                    suffix[i] = suffix[i + 1].clone() * &roots_q[i + 1];
                }
                for ell in 0..num_lookups {
                    let others_q = prefix[ell].clone() * &suffix[ell];
                    lhs += &(alpha_powers[ell].clone() * &proof.witness_gkr.roots_p[ell] * &others_q);
                }
            }
            lhs *= &proof.table_gkr.root_q;
            let rhs = proof.table_gkr.root_p.clone() * &q_w_product;
            if lhs != rhs {
                return Err(LookupError::GkrRootMismatch);
            }
        }

        // ---- Step 7 + 8: Verify table-side leaf claims and multiplicity sums ----
        // Precompute combined multiplicities and check sums in one pass.
        let expected_mult_sum = F::from_with_cfg((num_chunks * witness_len) as u64, field_cfg);
        let combined_mults: Vec<F> = {
            let mut combined = vec![zero.clone(); table_len];
            for ell in 0..num_lookups {
                let alpha_ell = &alpha_powers[ell];
                let mut m_sum = zero.clone();
                for j in 0..table_len {
                    let scaled = alpha_ell.clone() * &all_agg_mults[ell][j];
                    combined[j] += &scaled;
                    m_sum += &all_agg_mults[ell][j];
                }
                if m_sum != expected_mult_sum {
                    return Err(LookupError::MultiplicitySumMismatch {
                        expected: (num_chunks * witness_len) as u64,
                        got: 0,
                    });
                }
            }
            combined
        };

        if table_result.point.is_empty() {
            let expected_p = if table_len > 0 { combined_mults[0].clone() } else { zero.clone() };
            let expected_q = beta.clone() - &subtable[0];
            if expected_p != table_result.expected_p || expected_q != table_result.expected_q {
                return Err(LookupError::GkrLeafMismatch);
            }
        } else {
            let eq_at_t = build_eq_x_r_vec(&table_result.point, field_cfg)?;
            let mut p_eval = zero.clone();
            let mut q_eval = zero.clone();
            for j in 0..table_len {
                p_eval += &(combined_mults[j].clone() * &eq_at_t[j]);
                q_eval += &((beta.clone() - &subtable[j]) * &eq_at_t[j]);
            }
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

        // ---- Step 9: Witness-side leaf subclaims ----
        // Each per-lookup tree ℓ has:
        //   p^(ℓ)[(k, i)] = 1                 (all-ones numerator)
        //   q^(ℓ)[(k, i)] = β − c_k^(ℓ)[i]   (data), 1 (padding)
        //
        // The verifier can compute expected_p^(ℓ)(r) = MLE of all-ones
        // over the K*W data region = Σ_{j < K*W} eq(j, r).
        //
        // For q̃^(ℓ)(r), the verifier needs the combined chunk MLE
        // evaluation from PCS (returned as the expected value).
        //
        // All trees share the same point r, so we compute the expected
        // p once and check it against every tree's expected_p.
        let expected_witness_p = compute_witness_ones_p_eval(
            &witness_result.point,
            num_chunks,
            witness_len,
            w_num_vars,
            field_cfg,
        );

        for ell in 0..num_lookups {
            if expected_witness_p != witness_result.expected_ps[ell] {
                return Err(LookupError::GkrLeafMismatch);
            }
        }

        // Return the first tree's q evaluation as the expected q value.
        // All trees share the same r point, so the PCS needs to provide
        // chunk MLE evaluations at this shared point.
        Ok(GkrBatchedDecompLogupVerifierSubClaim {
            witness_eval_point: witness_result.point,
            expected_witness_q_eval: witness_result.expected_qs[0].clone(),
            expected_witness_p_eval: witness_result.expected_ps[0].clone(),
            table_eval_point: table_result.point,
        })
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute the expected evaluation of the per-tree witness numerator MLE.
///
/// Each per-tree witness numerator has `p[j] = 1` for `j < K*W` (data
/// region) and `0` for padding.  The MLE evaluation at point `r` is:
///   p̃(r) = Σ_{j < K*W} eq(j, r)
///
/// This is independent of the tree index ℓ — all trees share the same
/// structure and the same point.
#[allow(clippy::arithmetic_side_effects)]
fn compute_witness_ones_p_eval<F>(
    point: &[F],
    num_chunks: usize,
    witness_len: usize,
    num_vars: usize,
    field_cfg: &F::Config,
) -> F
where
    F: InnerTransparentField + FromPrimitiveWithConfig + Send + Sync,
    F::Inner: Clone + Send + Sync,
    F::Config: Sync,
{
    let one = F::one_with_cfg(field_cfg);

    if point.is_empty() {
        // 0-variable: single leaf = 1 (if there's data)
        return if num_chunks * witness_len > 0 { one } else { F::zero_with_cfg(field_cfg) };
    }

    let data_entries = num_chunks * witness_len; // K * W
    let total_size = 1usize << num_vars;
    debug_assert!(data_entries <= total_size);

    let eq_vec = build_eq_x_r_vec(point, field_cfg)
        .expect("eq vector construction should succeed");

    // Sum eq weights over data entries.
    let mut result = F::zero_with_cfg(field_cfg);
    for j in 0..data_entries {
        result += &eq_vec[j];
    }
    result
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
