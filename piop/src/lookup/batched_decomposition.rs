//! Batched Decomposition + LogUp protocol.
//!
//! Given L witness vectors that all look up into the same decomposed
//! table (e.g. BitPoly(32) → K sub-tables of BitPoly(8)), this
//! protocol batches **everything** into a single sumcheck:
//!
//! - A single β challenge is shared across all L lookups.
//! - The table inverse vector `v = 1/(β − T)` is computed once.
//! - All `1 + L·(2K+1)` identities are γ-batched into one
//!   combination function, yielding one sumcheck with
//!   `3 + L·(3K+1)` MLEs and degree 3.
//!
//! This amortises the dominant sumcheck cost across all lookups.

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
use zinc_utils::{cfg_into_iter, cfg_iter, inner_transparent_field::InnerTransparentField};

use crate::sumcheck::MLSumcheck;

use super::{
    structs::{
        BatchedDecompLogupProof, BatchedDecompLogupProverState,
        BatchedDecompLogupVerifierSubClaim, BatchedDecompLookupInstance, LookupError,
        LookupSumcheckGroup, LookupVerifierPreSumcheck,
    },
    tables::{
        batch_inverse, batch_inverse_shifted, build_table_index,
        compute_multiplicities_with_index,
    },
};

/// The batched Decomposition + LogUp protocol.
pub struct BatchedDecompLogupProtocol<F: InnerTransparentField>(PhantomData<F>);

impl<F: InnerTransparentField + FromPrimitiveWithConfig + Send + Sync>
    BatchedDecompLogupProtocol<F>
{
    /// Prover for the batched Decomposition + LogUp protocol.
    ///
    /// All L witness vectors must have the **same length** and share
    /// the same sub-table / shift factors.
    ///
    /// # MLE layout
    ///
    /// The L·(K+1) identities are precomputed into a single aggregate
    /// polynomial H, so https://eprint.iacr.org/2023/1284the sumcheck operates on only **2 MLEs** at
    /// degree 2:
    ///
    /// ```text
    ///   [0]:  eq(y, r)
    ///   [1]:  H(y)   — precomputed batched identity polynomial
    /// ```
    ///
    /// where H[j] = Σ_{ℓ,k} γ^{…}·((β−c^(ℓ)_k[j])·u^(ℓ)_k[j] − 1)
    ///            + Σ_{ℓ}   γ^{…}·(Σ_k u^(ℓ)_k[j] − m̃_agg^(ℓ)[j]·ṽ[j])
    ///
    /// # Identities (γ-batched)
    ///
    /// ```text
    ///   For ℓ = 0..L (offset = ℓ·(K+1)):
    ///     γ^{off}..γ^{off+K−1}:  [(β−c^(ℓ)_k)·u^(ℓ)_k−1]·eq  (inv correctness)
    ///     γ^{off+K}: [Σ_k u^(ℓ)_k − m_agg^(ℓ)·v]·eq           (log-deriv balance)
    /// Total identities: L·(K+1)
    /// ```
    #[allow(clippy::arithmetic_side_effects, clippy::too_many_arguments)]
    pub fn prove_as_subprotocol(
        transcript: &mut impl Transcript,
        instance: &BatchedDecompLookupInstance<F>,
        field_cfg: &F::Config,
    ) -> Result<
        (BatchedDecompLogupProof<F>, BatchedDecompLogupProverState<F>),
        LookupError<F>,
    >
    where
        F::Inner: ConstTranscribable + Zero + Default + Send + Sync,
    {
        let witnesses = &instance.witnesses;
        let subtable = &instance.subtable;
        let shifts = &instance.shifts;
        let all_chunks = &instance.chunks;

        let num_lookups = witnesses.len();
        let num_chunks = shifts.len();
        let witness_len = witnesses[0].len();

        let zero = F::zero_with_cfg(field_cfg);
        let one = F::one_with_cfg(field_cfg);
        let mut buf = vec![0u8; F::Inner::NUM_BYTES];

        // NOTE: Chunk vectors are NOT absorbed into the transcript.
        // The PCS commitment (already in the transcript) binds the
        // parent column and hence its unique chunk decomposition.
        // This allows us to omit chunk vectors from the proof entirely.

        // ---- Step 1: Compute multiplicities per chunk, then aggregate per lookup ----
        let table_index = build_table_index(subtable);
        let all_chunk_multiplicities: Vec<Vec<Vec<F>>> = cfg_iter!(all_chunks)
            .map(|lookup_chunks| {
                lookup_chunks
                    .iter()
                    .map(|chunk| {
                        compute_multiplicities_with_index(
                            chunk, &table_index, subtable.len(), field_cfg,
                        )
                        .ok_or(LookupError::WitnessNotInTable)
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<_, _>>()?;

        let all_aggregated_multiplicities: Vec<Vec<F>> = all_chunk_multiplicities
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

        for agg in &all_aggregated_multiplicities {
            transcript.absorb_random_field_slice(agg, &mut buf);
        }

        // ---- Step 3: Shared β challenge ----
        let beta: F = transcript.get_field_challenge(field_cfg);

        // ---- Step 4: Inverse vectors ----
        let all_inverse_witnesses: Vec<Vec<Vec<F>>> = cfg_iter!(all_chunks)
            .map(|lookup_chunks| {
                lookup_chunks
                    .iter()
                    .map(|chunk| batch_inverse_shifted(&beta, chunk))
                    .collect()
            })
            .collect();

        let v_table = batch_inverse_shifted(&beta, subtable);

        for lookup_invs in &all_inverse_witnesses {
            for u in lookup_invs {
                transcript.absorb_random_field_slice(u, &mut buf);
            }
        }
        transcript.absorb_random_field_slice(&v_table, &mut buf);

        // ---- Step 5: Batching challenge γ ----
        let gamma: F = transcript.get_field_challenge(field_cfg);

        // Dimensions
        let w_num_vars = zinc_utils::log2(witness_len.next_power_of_two()) as usize;
        let t_num_vars =
            zinc_utils::log2(subtable.len().next_power_of_two()) as usize;
        let num_vars = w_num_vars.max(t_num_vars);

        // ---- Step 6: Precompute γ powers ----
        let num_identities = num_lookups * (num_chunks + 1);
        let mut gamma_powers = Vec::with_capacity(num_identities);
        let mut gp = one.clone();
        for _ in 0..num_identities {
            gamma_powers.push(gp.clone());
            gp *= &gamma;
        }

        // ---- Step 7: Precompute batched identity polynomial H ----
        //
        // H[j] = Σ_{ℓ} [ Σ_k γ^{base+k}·((β−c^(ℓ)_k[j])·u^(ℓ)_k[j] − 1)
        //                  + γ^{base+K}·(Σ_k u^(ℓ)_k[j] − m_agg^(ℓ)[j]·v[j]) ]
        //
        // Evaluating H outside the sumcheck reduces the number of MLEs
        // from 2+L·(2K+1) to just 2 (eq and H) and the degree from 3
        // to 2, giving a large constant-factor speedup in the prover.
        let n = 1usize << num_vars;
        let subtable_len = subtable.len();
        let h_evaluations: Vec<F> = cfg_into_iter!(0..n)
            .map(|j| {
                let mut acc = zero.clone();
                let v_j = if j < subtable_len { &v_table[j] } else { &zero };

                for ell in 0..num_lookups {
                    let base_id = ell * (num_chunks + 1);

                    // Inverse-correctness identities: (β − c_k) · u_k − 1
                    for k_idx in 0..num_chunks {
                        let c_j = if j < witness_len {
                            &all_chunks[ell][k_idx][j]
                        } else {
                            &zero
                        };
                        let u_j = if j < witness_len {
                            &all_inverse_witnesses[ell][k_idx][j]
                        } else {
                            &zero
                        };
                        let id = (beta.clone() - c_j) * u_j - &one;
                        acc += &(id * &gamma_powers[base_id + k_idx]);
                    }

                    // Balance identity: Σ_k u_k − m_agg · v
                    let m_agg_j = if j < subtable_len {
                        &all_aggregated_multiplicities[ell][j]
                    } else {
                        &zero
                    };
                    let mut u_sum = if j < witness_len {
                        all_inverse_witnesses[ell][0][j].clone()
                    } else {
                        zero.clone()
                    };
                    for k_idx in 1..num_chunks {
                        if j < witness_len {
                            u_sum += &all_inverse_witnesses[ell][k_idx][j];
                        }
                    }
                    let balance = u_sum - &(m_agg_j.clone() * v_j);
                    acc += &(balance * &gamma_powers[base_id + num_chunks]);
                }

                acc
            })
            .collect();

        // ---- Step 8: Build MLEs (eq and H only) ----
        let r: Vec<F> = transcript.get_field_challenges(num_vars, field_cfg);
        let eq_r = build_eq_x_r_inner(&r, field_cfg)?;

        let inner_zero = zero.inner().clone();
        let h_mle = DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            cfg_iter!(h_evaluations).map(|x| x.inner().clone()).collect(),
            inner_zero,
        );

        let mles = vec![eq_r, h_mle];

        // ---- Step 9: Run sumcheck (degree 2: product of two linear MLEs) ----
        let comb_fn = move |vals: &[F]| -> F {
            vals[0].clone() * &vals[1]
        };

        let degree = 2;

        let (sumcheck_proof, sumcheck_prover_state) =
            MLSumcheck::prove_as_subprotocol(
                transcript, mles, num_vars, degree, comb_fn, field_cfg,
            );
        // ---- Sanity check: aggregated multiplicity sums ----
        debug_assert!({
            let expected = F::from_with_cfg((num_chunks * witness_len) as u64, field_cfg);
            all_aggregated_multiplicities.iter().all(|agg| {
                let sum: F =
                    agg.iter().cloned().fold(zero.clone(), |a, b| a + &b);
                sum == expected
            })
        });

        Ok((
            BatchedDecompLogupProof {
                sumcheck_proof,
                aggregated_multiplicities: all_aggregated_multiplicities,
                chunk_inverse_witnesses: all_inverse_witnesses,
                inverse_table: v_table,
            },
            BatchedDecompLogupProverState {
                evaluation_point: sumcheck_prover_state.randomness,
            },
        ))
    }

    /// Build the sumcheck degree group for batched LogUp **without**
    /// running the sumcheck.
    ///
    /// Does all pre-sumcheck work (absorb chunks/mults/inverses, draw
    /// β/γ/r, precompute H) and returns a `(degree, mles, comb_fn)`
    /// triple plus ancillary data.
    ///
    /// After the multi-degree sumcheck completes, call
    /// [`finalize_prover`] to assemble the final proof.
    #[allow(clippy::arithmetic_side_effects, clippy::too_many_arguments)]
    pub fn build_prover_group(
        transcript: &mut impl Transcript,
        instance: &BatchedDecompLookupInstance<F>,
        field_cfg: &F::Config,
    ) -> Result<LookupSumcheckGroup<F>, LookupError<F>>
    where
        F::Inner: ConstTranscribable + Zero + Default + Send + Sync,
    {
        let witnesses = &instance.witnesses;
        let subtable = &instance.subtable;
        let shifts = &instance.shifts;
        let all_chunks = &instance.chunks;

        let num_lookups = witnesses.len();
        let num_chunks = shifts.len();
        let witness_len = witnesses[0].len();

        let zero = F::zero_with_cfg(field_cfg);
        let one = F::one_with_cfg(field_cfg);
        let mut buf = vec![0u8; F::Inner::NUM_BYTES];

        // NOTE: Chunk vectors are NOT absorbed into the transcript.
        // The PCS commitment binds the parent column and hence its chunks.

        // ---- Step 1: Compute multiplicities ----
        let table_index = build_table_index(subtable);
        let all_chunk_multiplicities: Vec<Vec<Vec<F>>> = cfg_iter!(all_chunks)
            .map(|lookup_chunks| {
                lookup_chunks
                    .iter()
                    .map(|chunk| {
                        compute_multiplicities_with_index(
                            chunk, &table_index, subtable.len(), field_cfg,
                        )
                        .ok_or(LookupError::WitnessNotInTable)
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<_, _>>()?;

        let all_aggregated_multiplicities: Vec<Vec<F>> = all_chunk_multiplicities
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

        for agg in &all_aggregated_multiplicities {
            transcript.absorb_random_field_slice(agg, &mut buf);
        }

        // ---- Step 3: Shared β challenge ----
        let beta: F = transcript.get_field_challenge(field_cfg);

        // ---- Step 4: Inverse vectors ----
        let all_inverse_witnesses: Vec<Vec<Vec<F>>> = cfg_iter!(all_chunks)
            .map(|lookup_chunks| {
                lookup_chunks
                    .iter()
                    .map(|chunk| batch_inverse_shifted(&beta, chunk))
                    .collect()
            })
            .collect();

        let v_table = batch_inverse_shifted(&beta, subtable);

        for lookup_invs in &all_inverse_witnesses {
            for u in lookup_invs {
                transcript.absorb_random_field_slice(u, &mut buf);
            }
        }
        transcript.absorb_random_field_slice(&v_table, &mut buf);

        // ---- Step 5: Batching challenge γ ----
        let gamma: F = transcript.get_field_challenge(field_cfg);

        // Dimensions
        let w_num_vars =
            zinc_utils::log2(witness_len.next_power_of_two()) as usize;
        let t_num_vars =
            zinc_utils::log2(subtable.len().next_power_of_two()) as usize;
        let num_vars = w_num_vars.max(t_num_vars);

        // ---- Step 6: Precompute γ powers ----
        let num_identities = num_lookups * (num_chunks + 1);
        let mut gamma_powers = Vec::with_capacity(num_identities);
        let mut gp = one.clone();
        for _ in 0..num_identities {
            gamma_powers.push(gp.clone());
            gp *= &gamma;
        }

        // ---- Step 7: Precompute H ----
        let n = 1usize << num_vars;
        let subtable_len = subtable.len();
        let h_evaluations: Vec<F> = cfg_into_iter!(0..n)
            .map(|j| {
                let mut acc = zero.clone();
                let v_j = if j < subtable_len {
                    &v_table[j]
                } else {
                    &zero
                };

                for ell in 0..num_lookups {
                    let base_id = ell * (num_chunks + 1);

                    for k_idx in 0..num_chunks {
                        let c_j = if j < witness_len {
                            &all_chunks[ell][k_idx][j]
                        } else {
                            &zero
                        };
                        let u_j = if j < witness_len {
                            &all_inverse_witnesses[ell][k_idx][j]
                        } else {
                            &zero
                        };
                        let id = (beta.clone() - c_j) * u_j - &one;
                        acc += &(id * &gamma_powers[base_id + k_idx]);
                    }

                    let m_agg_j = if j < subtable_len {
                        &all_aggregated_multiplicities[ell][j]
                    } else {
                        &zero
                    };
                    let mut u_sum = if j < witness_len {
                        all_inverse_witnesses[ell][0][j].clone()
                    } else {
                        zero.clone()
                    };
                    for k_idx in 1..num_chunks {
                        if j < witness_len {
                            u_sum +=
                                &all_inverse_witnesses[ell][k_idx][j];
                        }
                    }
                    let balance = u_sum - &(m_agg_j.clone() * v_j);
                    acc += &(balance * &gamma_powers[base_id + num_chunks]);
                }

                acc
            })
            .collect();

        // ---- Step 8: Build MLEs ----
        let r: Vec<F> =
            transcript.get_field_challenges(num_vars, field_cfg);
        let eq_r = build_eq_x_r_inner(&r, field_cfg)?;

        let inner_zero = zero.inner().clone();
        let h_mle = DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            cfg_iter!(h_evaluations)
                .map(|x| x.inner().clone())
                .collect(),
            inner_zero,
        );

        let mles = vec![eq_r, h_mle];

        let comb_fn: Box<dyn Fn(&[F]) -> F + Send + Sync> =
            Box::new(move |vals: &[F]| -> F {
                vals[0].clone() * &vals[1]
            });

        Ok(LookupSumcheckGroup {
            degree: 2,
            mles,
            comb_fn,
            num_vars,
            aggregated_multiplicities: all_aggregated_multiplicities,
            chunk_inverse_witnesses: all_inverse_witnesses,
            inverse_table: v_table,
        })
    }

    /// Assemble the final lookup proof from the pre-sumcheck data
    /// and the sumcheck proof extracted from the multi-degree result.
    pub fn finalize_prover(
        pre: LookupSumcheckGroup<F>,
        sumcheck_proof: crate::sumcheck::SumcheckProof<F>,
        evaluation_point: Vec<F>,
    ) -> (
        BatchedDecompLogupProof<F>,
        BatchedDecompLogupProverState<F>,
    ) {
        (
            BatchedDecompLogupProof {
                sumcheck_proof,
                aggregated_multiplicities: pre.aggregated_multiplicities,
                chunk_inverse_witnesses: pre.chunk_inverse_witnesses,
                inverse_table: pre.inverse_table,
            },
            BatchedDecompLogupProverState {
                evaluation_point,
            },
        )
    }

    /// Pre-sumcheck verification for batched LogUp.
    ///
    /// Mirrors the prover's transcript operations (absorb chunks,
    /// mults, inverses; draw β, γ, r) and returns data needed for
    /// post-sumcheck recomputation of H(x*).
    #[allow(clippy::arithmetic_side_effects, clippy::too_many_arguments)]
    pub fn build_verifier_pre_sumcheck(
        transcript: &mut impl Transcript,
        proof: &BatchedDecompLogupProof<F>,
        subtable: &[F],
        shifts: &[F],
        num_lookups: usize,
        witness_len: usize,
        field_cfg: &F::Config,
    ) -> Result<LookupVerifierPreSumcheck<F>, LookupError<F>>
    where
        F::Inner: ConstTranscribable + Zero + Default,
    {
        let zero = F::zero_with_cfg(field_cfg);
        let one = F::one_with_cfg(field_cfg);
        let mut buf = vec![0u8; F::Inner::NUM_BYTES];

        let num_chunks = shifts.len();
        let all_agg_mults = &proof.aggregated_multiplicities;
        let all_inv_w = &proof.chunk_inverse_witnesses;
        let v_table = &proof.inverse_table;

        // NOTE: Chunk vectors are not absorbed — binding comes from PCS.
        for agg in all_agg_mults {
            transcript.absorb_random_field_slice(agg, &mut buf);
        }

        let beta: F = transcript.get_field_challenge(field_cfg);

        for lookup_invs in all_inv_w {
            for u in lookup_invs {
                transcript.absorb_random_field_slice(u, &mut buf);
            }
        }
        transcript.absorb_random_field_slice(v_table, &mut buf);

        let gamma: F = transcript.get_field_challenge(field_cfg);

        let w_num_vars =
            zinc_utils::log2(witness_len.next_power_of_two()) as usize;
        let t_num_vars =
            zinc_utils::log2(subtable.len().next_power_of_two()) as usize;
        let num_vars = w_num_vars.max(t_num_vars);

        let r: Vec<F> =
            transcript.get_field_challenges(num_vars, field_cfg);

        // Verify table inverse correctness.
        for (j, (t_j, v_j)) in subtable.iter().zip(v_table.iter()).enumerate()
        {
            let check = (beta.clone() - t_j) * v_j;
            if check != one {
                return Err(LookupError::TableInverseIncorrect { index: j });
            }
        }

        // Verify aggregated multiplicity sums.
        let expected_mult_sum =
            F::from_with_cfg((num_chunks * witness_len) as u64, field_cfg);
        for ell in 0..num_lookups {
            let m_sum: F = all_agg_mults[ell]
                .iter()
                .cloned()
                .fold(zero.clone(), |a, b| a + &b);
            if m_sum != expected_mult_sum {
                return Err(LookupError::MultiplicitySumMismatch {
                    expected: (num_chunks * witness_len) as u64,
                    got: 0,
                });
            }
        }

        Ok(LookupVerifierPreSumcheck {
            num_vars,
            r,
            beta,
            gamma,
            num_lookups,
            num_chunks,
            witness_len,
            shifts: shifts.to_vec(),
        })
    }

    /// Post-sumcheck finalization for the batched LogUp verifier.
    ///
    /// Reconstructs chunk vectors from inverse witnesses (`c_k[j] = β − 1/u_k[j]`),
    /// recomputes H(x*) at the subclaim point, checks it matches the expected
    /// evaluation from the sumcheck, and verifies the decomposition consistency
    /// against the PCS-committed parent column evaluations.
    ///
    /// # Arguments
    ///
    /// - `parent_column_evals`: for each lookup ℓ, the MLE evaluation of the
    ///   parent (projected) column at the subclaim point. Used for the
    ///   decomposition check: `Σ_k shift_k · c̃_k(x*) = parent_eval(x*)`.
    ///   Pass an empty slice to skip the decomposition check (e.g. in tests
    ///   without a PCS).
    #[allow(clippy::arithmetic_side_effects, clippy::too_many_arguments)]
    pub fn finalize_verifier(
        pre: &LookupVerifierPreSumcheck<F>,
        proof: &BatchedDecompLogupProof<F>,
        subclaim_point: &[F],
        subclaim_expected_evaluation: &F,
        parent_column_evals: &[F],
        field_cfg: &F::Config,
    ) -> Result<BatchedDecompLogupVerifierSubClaim<F>, LookupError<F>>
    where
        F::Inner: ConstTranscribable + Zero + Default,
    {
        let zero = F::zero_with_cfg(field_cfg);
        let one = F::one_with_cfg(field_cfg);

        let eq_val =
            zinc_poly::utils::eq_eval(subclaim_point, &pre.r, one.clone())?;

        let eq_at_point =
            build_eq_x_r_vec(subclaim_point, field_cfg)?;

        // ── Recompute H(x*) ──────────────────────────────────────────
        //
        // The prover's combination function has L*(K+1) identities:
        //   - L*K inverse-correctness identities (one per chunk per lookup)
        //   - L   balance identities (one per lookup)
        //
        // The inverse-correctness identity evaluates to 0 at every
        // j < witness_len (where c_k[j] = β − 1/u_k[j]). At padding
        // positions j >= witness_len, the identity evaluates to −1.
        // Its MLE evaluation at x* is therefore:
        //   ĩd_k(x*) = 0 · eq_sum_W + (−1)·(1 − eq_sum_W) = eq_sum_W − 1
        // where eq_sum_W = Σ_{j<W} eq(j, x*).
        //
        // This is the SAME value for all chunks k and lookups ℓ, so
        // the total inverse-correctness contribution is:
        //   (eq_sum_W − 1) · Σ_{ℓ,k} γ^(ℓ*(K+1) + k)
        //
        // When witness_len == 2^num_vars (no padding), eq_sum_W = 1
        // and the contribution is 0.

        let n = eq_at_point.len();
        let w = pre.witness_len;

        // Compute gamma powers for balance identities:
        //   gamma^(ell*(K+1) + K) for ell = 0..L
        let mut gamma_balance_powers = Vec::with_capacity(pre.num_lookups);
        // Also accumulate Σ_{ℓ,k} γ^(ℓ*(K+1)+k) for the padding correction.
        let mut inv_corr_gamma_sum = zero.clone();
        {
            let mut gp = one.clone();
            // S_K = Σ_{k=0}^{K-1} γ^k  (partial sum of first K powers within a stride)
            let mut s_k = one.clone();
            for _ in 1..pre.num_chunks {
                gp *= &pre.gamma;
                s_k += &gp;
            }
            // gp is now γ^(K-1); multiply once more for γ^K
            gp *= &pre.gamma;
            // γ^(K+1)
            let gamma_stride = gp.clone() * &pre.gamma;

            // For each lookup ℓ, record the balance power and accumulate
            // the inverse-correctness gamma sum.
            let mut stride_power = one.clone(); // γ^(ℓ*(K+1))
            for _ in 0..pre.num_lookups {
                gamma_balance_powers.push(stride_power.clone() * &gp); // γ^(ℓ*(K+1)+K)
                inv_corr_gamma_sum += &(stride_power.clone() * &s_k); // Σ_k γ^(ℓ*(K+1)+k)
                stride_power *= &gamma_stride;
            }
        }

        let v_table = &proof.inverse_table;
        let all_inv_w = &proof.chunk_inverse_witnesses;
        let all_agg_mults = &proof.aggregated_multiplicities;

        let v_eq: Vec<F> = v_table
            .iter()
            .zip(eq_at_point.iter())
            .map(|(v_j, eq_j)| v_j.clone() * eq_j)
            .collect();

        // Helper: inner product of data with eq_at_point.
        let eval_at_point = |data: &[F]| -> F {
            data.iter()
                .zip(eq_at_point.iter())
                .fold(zero.clone(), |acc, (d, e)| acc + &(d.clone() * e))
        };

        // eq_sum_W = Σ_{j<W} eq(j, x*).  Equals 1 when W = 2^num_vars.
        let eq_sum_w: F = if w < n {
            eq_at_point[..w]
                .iter()
                .fold(zero.clone(), |acc, e| acc + e)
        } else {
            one.clone()
        };

        let mut h_eval = zero.clone();

        // Inverse-correctness padding correction.
        // ĩd_k(x*) = eq_sum_W − 1 for every chunk k and lookup ℓ.
        if w < n {
            let id_pad = eq_sum_w.clone() - &one;
            h_eval += &(id_pad * &inv_corr_gamma_sum);
        }

        // For each lookup, also accumulate per-chunk MLE evaluations
        // for the decomposition check.
        let mut all_chunk_evals: Vec<Vec<F>> = Vec::with_capacity(pre.num_lookups);

        for ell in 0..pre.num_lookups {
            let mut u_sum_eval = zero.clone();
            let mut chunk_evals_for_ell = Vec::with_capacity(pre.num_chunks);

            for k_idx in 0..pre.num_chunks {
                let u_k = &all_inv_w[ell][k_idx];

                // ũ_k(x*) = Σ_j u_k[j] · eq(j, x*)
                let u_eval = eval_at_point(u_k);
                u_sum_eval += &u_eval;

                // c̃_k(x*) for the decomposition check.
                // c_k[j] = β − 1/u_k[j], so
                // c̃_k(x*) = β · eq_sum_W − Σ_{j<W} eq_j/u_k[j]
                //
                // Use batch_inverse (Montgomery's trick) instead of
                // per-element divisions.
                let inv_u_k = batch_inverse(u_k);
                let inv_u_eval = eval_at_point(&inv_u_k);
                let c_eval = pre.beta.clone() * &eq_sum_w - &inv_u_eval;
                chunk_evals_for_ell.push(c_eval);
            }

            all_chunk_evals.push(chunk_evals_for_ell);

            // Balance identity: u_sum_eval − mv_eval
            let mv_eval: F = all_agg_mults[ell]
                .iter()
                .zip(v_eq.iter())
                .fold(zero.clone(), |acc, (m_j, ve_j)| {
                    acc + &(m_j.clone() * ve_j)
                });

            let balance = u_sum_eval - &mv_eval;
            h_eval += &(balance * &gamma_balance_powers[ell]);
        }

        let expected = h_eval * &eq_val;

        if expected != *subclaim_expected_evaluation {
            return Err(LookupError::FinalEvaluationMismatch {
                expected: subclaim_expected_evaluation.clone(),
                got: expected,
            });
        }

        // ── Decomposition consistency check ─────────────────────────
        //
        // For each lookup ℓ, verify that the reconstructed chunk
        // evaluations are consistent with the PCS-committed parent
        // column: Σ_k shift_k · c̃_k^(ℓ)(x*) = parent_eval_ℓ(x*).
        if !parent_column_evals.is_empty() {
            for (ell, parent_eval) in parent_column_evals.iter().enumerate() {
                let mut recomposed = zero.clone();
                for (k_idx, shift_k) in pre.shifts.iter().enumerate() {
                    recomposed += &(shift_k.clone() * &all_chunk_evals[ell][k_idx]);
                }
                if recomposed != *parent_eval {
                    return Err(LookupError::DecompositionInconsistent);
                }
            }
        }

        Ok(BatchedDecompLogupVerifierSubClaim {
            evaluation_point: subclaim_point.to_vec(),
            expected_evaluation: subclaim_expected_evaluation.clone(),
        })
    }

    /// Verifier for the batched Decomposition + LogUp protocol.
    ///
    /// Reconstructs chunk vectors from inverse witnesses.
    /// Does NOT perform the decomposition consistency check against
    /// PCS parent evaluations (use `finalize_verifier` for that).
    #[allow(clippy::arithmetic_side_effects, clippy::too_many_arguments)]
    pub fn verify_as_subprotocol(
        transcript: &mut impl Transcript,
        proof: &BatchedDecompLogupProof<F>,
        subtable: &[F],
        shifts: &[F],
        num_lookups: usize,
        witness_len: usize,
        field_cfg: &F::Config,
    ) -> Result<BatchedDecompLogupVerifierSubClaim<F>, LookupError<F>>
    where
        F::Inner: ConstTranscribable + Zero + Default,
    {
        let zero = F::zero_with_cfg(field_cfg);
        let one = F::one_with_cfg(field_cfg);
        let mut buf = vec![0u8; F::Inner::NUM_BYTES];

        let num_chunks = shifts.len();
        let all_agg_mults = &proof.aggregated_multiplicities;
        let all_inv_w = &proof.chunk_inverse_witnesses;
        let v_table = &proof.inverse_table;

        // ---- Mirror transcript operations (no chunk absorption) ----
        for agg in all_agg_mults {
            transcript.absorb_random_field_slice(agg, &mut buf);
        }

        let beta: F = transcript.get_field_challenge(field_cfg);

        for lookup_invs in all_inv_w {
            for u in lookup_invs {
                transcript.absorb_random_field_slice(u, &mut buf);
            }
        }
        transcript.absorb_random_field_slice(v_table, &mut buf);

        let gamma: F = transcript.get_field_challenge(field_cfg);

        let w_num_vars =
            zinc_utils::log2(witness_len.next_power_of_two()) as usize;
        let t_num_vars =
            zinc_utils::log2(subtable.len().next_power_of_two()) as usize;
        let num_vars = w_num_vars.max(t_num_vars);

        let r: Vec<F> = transcript.get_field_challenges(num_vars, field_cfg);

        // ---- Verify sumcheck (degree 2: product of two linear MLEs) ----
        let subclaim = MLSumcheck::verify_as_subprotocol(
            transcript,
            num_vars,
            2,
            &proof.sumcheck_proof,
            field_cfg,
        )?;

        let eval_point = &subclaim.point;
        let eq_val =
            zinc_poly::utils::eq_eval(eval_point, &r, one.clone())?;

        // Build eq(·, eval_point) once and reuse for MLE evaluations.
        let eq_at_point = build_eq_x_r_vec(eval_point, field_cfg)?;

        // ---- Direct table inverse check ----
        for (j, (t_j, v_j)) in subtable.iter().zip(v_table.iter()).enumerate() {
            let check = (beta.clone() - t_j) * v_j;
            if check != one {
                return Err(LookupError::TableInverseIncorrect { index: j });
            }
        }

        // ---- Recompute H(x*) using chunks reconstructed from inverses ----
        //
        // The inverse-correctness identities evaluate to 0 at actual
        // witness positions and −1 at padding positions.
        let n = eq_at_point.len();
        let mut gamma_balance_powers = Vec::with_capacity(num_lookups);
        let mut inv_corr_gamma_sum = zero.clone();
        {
            let mut gp = one.clone();
            let mut s_k = one.clone();
            for _ in 1..num_chunks {
                gp *= &gamma;
                s_k += &gp;
            }
            gp *= &gamma; // γ^K
            let gamma_stride = gp.clone() * &gamma; // γ^(K+1)

            let mut stride_power = one.clone();
            for _ in 0..num_lookups {
                gamma_balance_powers.push(stride_power.clone() * &gp);
                inv_corr_gamma_sum += &(stride_power.clone() * &s_k);
                stride_power *= &gamma_stride;
            }
        }

        let v_eq: Vec<F> = v_table
            .iter()
            .zip(eq_at_point.iter())
            .map(|(v_j, eq_j)| v_j.clone() * eq_j)
            .collect();

        let eval_at_point = |data: &[F]| -> F {
            data.iter()
                .zip(eq_at_point.iter())
                .fold(zero.clone(), |acc, (d, e)| acc + &(d.clone() * e))
        };

        let mut h_eval = zero.clone();

        // Padding correction for inverse-correctness identities.
        if witness_len < n {
            let eq_sum_w: F = eq_at_point[..witness_len]
                .iter()
                .fold(zero.clone(), |acc, e| acc + e);
            let id_pad = eq_sum_w - &one;
            h_eval += &(id_pad * &inv_corr_gamma_sum);
        }

        for ell in 0..num_lookups {
            let mut u_sum_eval = zero.clone();

            for k_idx in 0..num_chunks {
                let u_k = &all_inv_w[ell][k_idx];
                let u_eval = eval_at_point(u_k);
                u_sum_eval += &u_eval;
            }

            let mv_eval: F = all_agg_mults[ell]
                .iter()
                .zip(v_eq.iter())
                .fold(zero.clone(), |acc, (m_j, ve_j)| {
                    acc + &(m_j.clone() * ve_j)
                });

            let balance = u_sum_eval - &mv_eval;
            h_eval += &(balance * &gamma_balance_powers[ell]);
        }

        let expected = h_eval * &eq_val;

        if expected != subclaim.expected_evaluation {
            return Err(LookupError::FinalEvaluationMismatch {
                expected: subclaim.expected_evaluation.clone(),
                got: expected,
            });
        }

        // ---- Verify aggregated multiplicity sums ----
        let expected_sum =
            F::from_with_cfg((num_chunks * witness_len) as u64, field_cfg);

        for ell in 0..num_lookups {
            let m_sum: F =
                all_agg_mults[ell].iter().cloned().fold(zero.clone(), |a, b| a + &b);
            if m_sum != expected_sum {
                return Err(LookupError::MultiplicitySumMismatch {
                    expected: (num_chunks * witness_len) as u64,
                    got: 0,
                });
            }
        }

        Ok(BatchedDecompLogupVerifierSubClaim {
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

    /// Batched lookup with L=3 lookups, K=2 chunks, BitPoly(4).
    #[test]
    fn batched_decomp_logup_3_lookups_2_chunks() {
        let a = F::from(3u32);
        let chunk_width = 2;
        let num_chunks = 2;
        let subtable = generate_bitpoly_table(chunk_width, &a, &());
        let _full_table =
            generate_bitpoly_table(chunk_width * num_chunks, &a, &());
        let shifts: Vec<F> = (0..num_chunks)
            .map(|k| bitpoly_shift(k * chunk_width, &a))
            .collect();

        // Three lookups, each with 4 witness entries (padded to same len).
        let (w0, c0) = make_lookup(
            &[0, 3, 5, 15],
            &subtable,
            chunk_width,
            num_chunks,
            &shifts,
        );
        let (w1, c1) = make_lookup(
            &[1, 2, 7, 10],
            &subtable,
            chunk_width,
            num_chunks,
            &shifts,
        );
        let (w2, c2) = make_lookup(
            &[15, 15, 0, 8],
            &subtable,
            chunk_width,
            num_chunks,
            &shifts,
        );

        let instance = BatchedDecompLookupInstance {
            witnesses: vec![w0, w1, w2],
            subtable: subtable.clone(),
            shifts: shifts.clone(),
            chunks: vec![c0, c1, c2],
        };

        let mut pt = KeccakTranscript::new();
        let (proof, _) =
            BatchedDecompLogupProtocol::<F>::prove_as_subprotocol(
                &mut pt, &instance, &(),
            )
            .expect("prover should succeed");

        let mut vt = KeccakTranscript::new();
        let _sub = BatchedDecompLogupProtocol::<F>::verify_as_subprotocol(
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

    /// Batched lookup with L=5 lookups, K=4 chunks, BitPoly(8).
    #[test]
    fn batched_decomp_logup_5_lookups_4_chunks() {
        let a = F::from(3u32);
        let chunk_width = 2;
        let num_chunks = 4;
        let subtable = generate_bitpoly_table(chunk_width, &a, &());
        let _full_table =
            generate_bitpoly_table(chunk_width * num_chunks, &a, &());
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
            let (w, c) = make_lookup(
                indices,
                &subtable,
                chunk_width,
                num_chunks,
                &shifts,
            );
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
        let (proof, _) =
            BatchedDecompLogupProtocol::<F>::prove_as_subprotocol(
                &mut pt, &instance, &(),
            )
            .expect("prover should succeed");

        let mut vt = KeccakTranscript::new();
        let _sub = BatchedDecompLogupProtocol::<F>::verify_as_subprotocol(
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

    /// Single lookup (L=1) should behave identically to the
    /// non-batched protocol (up to transcript differences).
    #[test]
    fn batched_decomp_logup_single_lookup() {
        let a = F::from(3u32);
        let chunk_width = 2;
        let num_chunks = 2;
        let subtable = generate_bitpoly_table(chunk_width, &a, &());
        let shifts: Vec<F> = (0..num_chunks)
            .map(|k| bitpoly_shift(k * chunk_width, &a))
            .collect();

        let (w, c) = make_lookup(
            &[0, 5, 10, 15],
            &subtable,
            chunk_width,
            num_chunks,
            &shifts,
        );

        let instance = BatchedDecompLookupInstance {
            witnesses: vec![w],
            subtable: subtable.clone(),
            shifts: shifts.clone(),
            chunks: vec![c],
        };

        let mut pt = KeccakTranscript::new();
        let (proof, _) =
            BatchedDecompLogupProtocol::<F>::prove_as_subprotocol(
                &mut pt, &instance, &(),
            )
            .expect("prover should succeed");

        let mut vt = KeccakTranscript::new();
        let _sub = BatchedDecompLogupProtocol::<F>::verify_as_subprotocol(
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
