//! Top-level GKR-LogUp prover and verifier per lookup group.
//!
//! Implements the chunks-in-clear polynomial-valued lift design:
//!
//! - **Chunks are NOT sent in the proof and NOT separately committed.**
//!   The prover sends per-`(ell, k)` polynomial-valued chunk lifts
//!   `c_k'^(ell) = MLE[v_k^(ell)](r_inner) ∈ F_q[X]_{<chunk_width}` —
//!   `chunk_width` field elements per (lookup, chunk).
//! - The witness-side GKR runs over ψ_a-projected chunk values; its
//!   leaf identity at the descent point `r = (r_inner, r_outer)`
//!   reduces to `expected_qs[ell] = β - Σ_k eq_outer(k, r_outer) ·
//!   ψ_a(c_k'^(ell))`.
//! - The verifier sub-claim returned to the protocol layer is the
//!   combined parent polynomial `c^(ell) = Σ_k X^{k·chunk_width} ·
//!   c_k'^(ell) = MLE[v^(ell)](r_inner)`. The protocol layer binds
//!   this to the parent column's PCS commitment by opening Zip+ at
//!   `r_inner` (a second opening, beyond the step-7 one at `r_0`).
//!
//! See `IMPLEMENTATION.md` (gleaming-pony plan) for the full design
//! discussion.

use crypto_primitives::{FromPrimitiveWithConfig, PrimeField};
use num_traits::Zero;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use zinc_poly::{
    mle::DenseMultilinearExtension,
    univariate::{binary::BinaryPoly, dynamic::over_field::DynamicPolynomialF},
    utils::build_eq_x_r_vec,
};
use zinc_transcript::traits::{ConstTranscribable, Transcript};
use zinc_uair::LookupTableType;
use zinc_utils::{cfg_iter, cfg_into_iter, inner_transparent_field::InnerTransparentField};

use super::gkr::{
    batched_gkr_fraction_prove, batched_gkr_fraction_verify, build_fraction_tree,
    build_fraction_tree_ones_leaf, gkr_fraction_prove, gkr_fraction_verify,
};
use super::structs::{
    GkrLogupError, GkrLogupGroupMeta, GkrLogupGroupProof, GkrLogupGroupSubclaim,
};
use super::tables::{
    build_table_index, compute_multiplicities_with_index, generate_bitpoly_table,
};

// ---------------------------------------------------------------------------
// Public input shape for `prove_group`.
// ---------------------------------------------------------------------------

/// Inputs to [`prove_group`] for a single lookup group of binary_poly
/// columns. MVP supports `binary_poly<D>` parents with
/// `LookupTableType::BitPoly { width: D, chunk_width: Some(cw) }` where
/// `cw` divides `D`. Each parent column appears as a
/// `DenseMultilinearExtension<BinaryPoly<D>>` value (the trace's
/// committed binary_poly column).
pub struct BinaryPolyLookupInstance<'a, F: PrimeField, const D: usize> {
    /// L parent column MLEs (binary_poly-valued).
    pub parent_columns: Vec<&'a DenseMultilinearExtension<BinaryPoly<D>>>,
    /// L flat-trace column indices, mirrored into the proof's group meta.
    pub parent_column_indices: Vec<usize>,
    /// Lookup table type — must be `BitPoly { width: D, chunk_width: Some(cw) }`.
    pub table_type: LookupTableType,
    /// Projecting element `a` used by ψ_a, threaded from step 3.
    pub projecting_element_f: &'a F,
    /// Number of MLE variables of each parent column (= log2(W)).
    pub n_vars: usize,
}

// ---------------------------------------------------------------------------
// Prover
// ---------------------------------------------------------------------------

/// Prove one GKR-LogUp lookup group with chunks-in-clear poly-lift.
///
/// Returns the lookup proof, the group meta to embed in the outer
/// proof, and the verifier sub-claim the protocol layer must discharge
/// against the parent column's PCS commitment via a Zip+ opening at
/// `r_inner`.
#[allow(clippy::arithmetic_side_effects)]
pub fn prove_group<F, const D: usize>(
    transcript: &mut impl Transcript,
    instance: &BinaryPolyLookupInstance<'_, F, D>,
    field_cfg: &F::Config,
) -> Result<
    (GkrLogupGroupProof<F>, GkrLogupGroupMeta, GkrLogupGroupSubclaim<F>),
    GkrLogupError<F>,
>
where
    F: InnerTransparentField + FromPrimitiveWithConfig + Send + Sync,
    F::Inner: ConstTranscribable + Zero + Default + Send + Sync,
    F::Modulus: ConstTranscribable,
    F::Config: Sync,
{
    let (width, chunk_width) = match instance.table_type {
        LookupTableType::BitPoly { width, chunk_width: Some(cw) } => (width, cw),
        LookupTableType::BitPoly { width, chunk_width: None } => (width, width),
        _ => {
            return Err(GkrLogupError::WitnessNotInTable);
        }
    };
    assert_eq!(width, D, "table width must equal binary_poly degree D");
    assert!(chunk_width > 0 && width % chunk_width == 0, "chunk_width must divide width");
    let num_chunks = width / chunk_width;
    let num_lookups = instance.parent_columns.len();
    let n_vars = instance.n_vars;
    let witness_len = 1usize << n_vars;

    // ---- Step 1: ψ_a-project chunks for the GKR ----
    // chunks_psi[ell][k][i] = Σ_{p=0..chunk_width} bit_{k·cw + p}(v^(ell)[i]) · a^p
    let a = instance.projecting_element_f.clone();
    let one = F::one_with_cfg(field_cfg);
    let zero = F::zero_with_cfg(field_cfg);

    // Precompute powers of `a` of length `chunk_width`.
    let mut a_powers_chunk = Vec::with_capacity(chunk_width);
    let mut acc = one.clone();
    for _ in 0..chunk_width {
        a_powers_chunk.push(acc.clone());
        acc = acc * &a;
    }

    let chunks_psi: Vec<Vec<Vec<F>>> = cfg_iter!(instance.parent_columns)
        .map(|parent| {
            (0..num_chunks)
                .map(|k| {
                    (0..witness_len)
                        .map(|i| {
                            let bp = &parent.evaluations[i];
                            let coeffs = bp.inner().coeffs;
                            let mut sum = zero.clone();
                            for p in 0..chunk_width {
                                if coeffs[k * chunk_width + p].into_inner() {
                                    sum = sum + &a_powers_chunk[p];
                                }
                            }
                            sum
                        })
                        .collect()
                })
                .collect()
        })
        .collect();

    // ---- Step 2: Build subtable T = ψ_a({0,1}^{<chunk_width}[X]) ----
    let subtable: Vec<F> = generate_bitpoly_table(chunk_width, &a, field_cfg);
    let table_len = subtable.len();
    let table_index = build_table_index::<F>(&subtable);

    // ---- Step 3: Multiplicities ----
    let agg_mults: Vec<Vec<F>> = chunks_psi
        .iter()
        .map(|lookup_chunks| {
            let mut agg = vec![zero.clone(); table_len];
            for chunk in lookup_chunks {
                let m = compute_multiplicities_with_index(
                    chunk,
                    &table_index,
                    table_len,
                    field_cfg,
                )
                .ok_or(GkrLogupError::WitnessNotInTable)?;
                for (a, mk) in agg.iter_mut().zip(m.iter()) {
                    *a = a.clone() + mk;
                }
            }
            Ok::<Vec<F>, GkrLogupError<F>>(agg)
        })
        .collect::<Result<_, _>>()?;

    // ---- Step 4: Absorb agg multiplicities, sample β, α ----
    let mut buf = vec![0u8; F::Inner::NUM_BYTES];
    for agg in &agg_mults {
        transcript.absorb_random_field_slice(agg, &mut buf);
    }
    let beta: F = transcript.get_field_challenge(field_cfg);
    let alpha: F = transcript.get_field_challenge(field_cfg);

    // α^ell powers.
    let mut alpha_powers = Vec::with_capacity(num_lookups);
    let mut ap = one.clone();
    for _ in 0..num_lookups {
        alpha_powers.push(ap.clone());
        ap = ap * &alpha;
    }

    // ---- Step 5: Build L witness fraction trees ----
    let per_lookup_leaves = num_chunks * witness_len; // K · W (always a power of 2)
    let w_num_vars = zinc_utils::log2(per_lookup_leaves.next_power_of_two()) as usize;
    let w_size = 1usize << w_num_vars;
    let leaves_already_pow2 = per_lookup_leaves == w_size;

    let witness_trees: Vec<_> = (0..num_lookups)
        .map(|ell| {
            let mut leaf_q = Vec::with_capacity(w_size);
            for k in 0..num_chunks {
                for i in 0..witness_len {
                    leaf_q.push(beta.clone() - &chunks_psi[ell][k][i]);
                }
            }
            if leaves_already_pow2 {
                build_fraction_tree_ones_leaf(one.clone(), leaf_q)
            } else {
                let mut leaf_p = vec![one.clone(); per_lookup_leaves];
                leaf_p.resize(w_size, zero.clone());
                leaf_q.resize(w_size, one.clone());
                build_fraction_tree(leaf_p, leaf_q)
            }
        })
        .collect();

    // ---- Step 6: Build α-batched table fraction tree ----
    let t_num_vars = zinc_utils::log2(table_len.next_power_of_two()) as usize;
    let t_size = 1usize << t_num_vars;
    let (mut t_leaf_p, mut t_leaf_q): (Vec<F>, Vec<F>) = cfg_into_iter!(0..table_len, 256)
        .map(|j| {
            let mut combined_mult = zero.clone();
            for ell in 0..num_lookups {
                combined_mult = combined_mult + &(alpha_powers[ell].clone() * &agg_mults[ell][j]);
            }
            (combined_mult, beta.clone() - &subtable[j])
        })
        .unzip();
    t_leaf_p.resize(t_size, zero.clone());
    t_leaf_q.resize(t_size, one.clone());
    let table_tree = build_fraction_tree(t_leaf_p, t_leaf_q);

    // ---- Step 7: Witness GKR ----
    let witness_result = batched_gkr_fraction_prove(transcript, &witness_trees, field_cfg);

    // ---- Step 8: Table GKR ----
    let (table_gkr, _table_eval_point) = gkr_fraction_prove(transcript, &table_tree, field_cfg);

    // ---- Step 9: Polynomial-valued chunk lifts ----
    //
    // For each (ell, k), c_k'^(ell) = MLE[v_k^(ell)](r_inner) is a
    // polynomial in F_q[X]_{<chunk_width}. We compute the parent's
    // full lifted eval (D coefficients) and split into chunks of
    // chunk_width coefficients each.
    let r_full = &witness_result.eval_point;
    assert!(
        r_full.len() >= n_vars,
        "GKR descent must have at least n_vars row variables"
    );
    let r_inner: Vec<F> = r_full[..n_vars].to_vec();

    // Batch the L parent lifts so the eq(·, r_inner) table is built
    // once and the bit walks run in parallel across the L parents.
    let parent_lifts =
        compute_binary_poly_lifts::<F, D>(&instance.parent_columns, &r_inner, field_cfg);
    let chunk_lifts: Vec<Vec<DynamicPolynomialF<F>>> = parent_lifts
        .into_iter()
        .map(|parent_lifted| {
            (0..num_chunks)
                .map(|k| {
                    let lo = k * chunk_width;
                    let hi = lo + chunk_width;
                    DynamicPolynomialF::new_trimmed(parent_lifted.coeffs[lo..hi].to_vec())
                })
                .collect()
        })
        .collect();

    // ---- Step 10: Combine chunks → parent claim at r_inner ----
    let combined_polynomial: Vec<DynamicPolynomialF<F>> = (0..num_lookups)
        .map(|ell| combine_chunks::<F>(&chunk_lifts[ell], chunk_width, width, &zero))
        .collect();

    // ---- Sanity (debug) ----
    debug_assert!({
        // Multiplicity sum invariant.
        let expected_per_lookup =
            F::from_with_cfg((num_chunks * witness_len) as u64, field_cfg);
        agg_mults.iter().all(|agg| {
            let sum: F = agg.iter().cloned().fold(zero.clone(), |a, b| a + &b);
            sum == expected_per_lookup
        })
    });

    let meta = GkrLogupGroupMeta {
        table_type: instance.table_type.clone(),
        num_lookups,
        num_chunks,
        chunk_width,
        witness_len,
        parent_columns: instance.parent_column_indices.clone(),
    };
    let proof = GkrLogupGroupProof {
        chunk_lifts,
        aggregated_multiplicities: agg_mults,
        witness_gkr: witness_result.proof,
        table_gkr,
        bin_lifts_at_r_inner: Vec::new(),
    };
    let subclaim = GkrLogupGroupSubclaim {
        r_inner,
        combined_polynomial,
        parent_columns: meta.parent_columns.clone(),
    };
    Ok((proof, meta, subclaim))
}

// ---------------------------------------------------------------------------
// Verifier
// ---------------------------------------------------------------------------

/// Verify one GKR-LogUp lookup group's proof. Returns the verifier
/// sub-claim that the protocol layer must discharge by opening Zip+ on
/// the parent column at `subclaim.r_inner` and matching against
/// `subclaim.combined_polynomial`.
#[allow(clippy::arithmetic_side_effects)]
pub fn verify_group<F>(
    transcript: &mut impl Transcript,
    proof: &GkrLogupGroupProof<F>,
    meta: &GkrLogupGroupMeta,
    projecting_element_f: &F,
    field_cfg: &F::Config,
) -> Result<GkrLogupGroupSubclaim<F>, GkrLogupError<F>>
where
    F: InnerTransparentField + FromPrimitiveWithConfig + Send + Sync,
    F::Inner: ConstTranscribable + Zero + Default + Send + Sync,
    F::Modulus: ConstTranscribable,
    F::Config: Sync,
{
    let (width, chunk_width) = match &meta.table_type {
        LookupTableType::BitPoly { width, chunk_width: Some(cw) } => (*width, *cw),
        LookupTableType::BitPoly { width, chunk_width: None } => (*width, *width),
        _ => return Err(GkrLogupError::WitnessNotInTable),
    };
    assert!(chunk_width > 0 && width % chunk_width == 0);

    let num_lookups = meta.num_lookups;
    let num_chunks = meta.num_chunks;
    let witness_len = meta.witness_len;
    let n_vars = (witness_len as f64).log2() as usize;
    assert_eq!(1usize << n_vars, witness_len, "witness_len must be a power of 2");
    assert_eq!(num_chunks, width / chunk_width);
    assert_eq!(proof.chunk_lifts.len(), num_lookups);
    assert_eq!(proof.aggregated_multiplicities.len(), num_lookups);

    let zero = F::zero_with_cfg(field_cfg);
    let one = F::one_with_cfg(field_cfg);

    // ---- Reconstruct subtable + shifts ----
    let subtable = generate_bitpoly_table::<F>(chunk_width, projecting_element_f, field_cfg);
    let table_len = subtable.len();

    // ---- Step 1: Absorb agg multiplicities, sample β, α ----
    let mut buf = vec![0u8; F::Inner::NUM_BYTES];
    for agg in &proof.aggregated_multiplicities {
        transcript.absorb_random_field_slice(agg, &mut buf);
    }
    let beta: F = transcript.get_field_challenge(field_cfg);
    let alpha: F = transcript.get_field_challenge(field_cfg);

    let mut alpha_powers = Vec::with_capacity(num_lookups);
    let mut ap = one.clone();
    for _ in 0..num_lookups {
        alpha_powers.push(ap.clone());
        ap = ap * &alpha;
    }

    // ---- Step 2: Witness + table GKR verify ----
    let per_lookup_leaves = num_chunks * witness_len;
    let w_num_vars = zinc_utils::log2(per_lookup_leaves.next_power_of_two()) as usize;
    let t_num_vars = zinc_utils::log2(table_len.next_power_of_two()) as usize;

    let witness_result =
        batched_gkr_fraction_verify(transcript, &proof.witness_gkr, w_num_vars, field_cfg)?;
    let table_result =
        gkr_fraction_verify(transcript, &proof.table_gkr, t_num_vars, field_cfg)?;

    // ---- Step 3: Cross-check roots ----
    {
        let roots_q = &proof.witness_gkr.roots_q;
        let q_w_product: F = roots_q.iter().cloned().fold(one.clone(), |acc, q| acc * &q);
        let mut lhs = zero.clone();
        if num_lookups == 1 {
            lhs = lhs + &(alpha_powers[0].clone() * &proof.witness_gkr.roots_p[0]);
        } else if num_lookups > 1 {
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
                lhs = lhs
                    + &(alpha_powers[ell].clone() * &proof.witness_gkr.roots_p[ell] * &others_q);
            }
        }
        lhs = lhs * &proof.table_gkr.root_q;
        let rhs = proof.table_gkr.root_p.clone() * &q_w_product;
        if lhs != rhs {
            return Err(GkrLogupError::GkrRootMismatch);
        }
    }

    // ---- Step 4: Multiplicity sums + table-side leaf check ----
    let expected_mult_sum = F::from_with_cfg((num_chunks * witness_len) as u64, field_cfg);
    let combined_mults: Vec<F> = {
        let mut combined = vec![zero.clone(); table_len];
        for ell in 0..num_lookups {
            let alpha_ell = &alpha_powers[ell];
            let mut m_sum = zero.clone();
            for j in 0..table_len {
                let scaled = alpha_ell.clone() * &proof.aggregated_multiplicities[ell][j];
                combined[j] = combined[j].clone() + &scaled;
                m_sum = m_sum + &proof.aggregated_multiplicities[ell][j];
            }
            if m_sum != expected_mult_sum {
                return Err(GkrLogupError::MultiplicitySumMismatch {
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
            return Err(GkrLogupError::GkrLeafMismatch);
        }
    } else {
        let eq_at_t = build_eq_x_r_vec(&table_result.point, field_cfg)?;
        let mut p_eval = zero.clone();
        let mut q_eval = zero.clone();
        for j in 0..table_len {
            p_eval = p_eval + &(combined_mults[j].clone() * &eq_at_t[j]);
            q_eval = q_eval + &((beta.clone() - &subtable[j]) * &eq_at_t[j]);
        }
        for j in table_len..eq_at_t.len() {
            q_eval = q_eval + &eq_at_t[j];
        }
        if p_eval != table_result.expected_p {
            return Err(GkrLogupError::GkrLeafMismatch);
        }
        if q_eval != table_result.expected_q {
            return Err(GkrLogupError::GkrLeafMismatch);
        }
    }

    // ---- Step 5: Witness-side leaf check using chunk lifts ----
    //
    // r = (r_inner, r_outer) with r_inner of length n_vars (low bits)
    // and r_outer of length log2(K) (high bits).
    let r_full = &witness_result.point;
    assert_eq!(r_full.len(), w_num_vars);
    assert_eq!(w_num_vars, n_vars + zinc_utils::log2(num_chunks) as usize);
    let r_inner: Vec<F> = r_full[..n_vars].to_vec();
    let r_outer: Vec<F> = r_full[n_vars..].to_vec();

    // expected_p^(ell)(r) should equal Σ_{j<K·W} eq(j, r) — but K·W is
    // a power of 2 in MVP, so this is simply 1 (the all-ones MLE over
    // the full hypercube evaluates to 1 at any point).
    //
    // (For non-power-of-2 case we'd call compute_witness_ones_p_eval.)
    let expected_p_value = if per_lookup_leaves == (1usize << w_num_vars) {
        one.clone()
    } else {
        let eq_at_r = build_eq_x_r_vec(r_full, field_cfg)?;
        let mut s = zero.clone();
        for j in 0..per_lookup_leaves {
            s = s + &eq_at_r[j];
        }
        s
    };
    for ell in 0..num_lookups {
        if witness_result.expected_ps[ell] != expected_p_value {
            return Err(GkrLogupError::GkrLeafMismatch);
        }
    }

    // For q^(ell), reconstruct from chunk lifts:
    //   expected_qs[ell] = β - Σ_k eq_outer(k, r_outer) · ψ_a(c_k'^(ell))
    //                       + padding_correction
    // (no padding when K·W is a power of 2)
    let eq_at_outer = build_eq_x_r_vec(&r_outer, field_cfg)?;
    for ell in 0..num_lookups {
        if proof.chunk_lifts[ell].len() != num_chunks {
            return Err(GkrLogupError::GkrLeafMismatch);
        }
        let mut psi_combined = zero.clone();
        for k in 0..num_chunks {
            let psi = eval_at_projecting_element::<F>(
                &proof.chunk_lifts[ell][k],
                projecting_element_f,
                field_cfg,
            );
            psi_combined = psi_combined + &(eq_at_outer[k].clone() * &psi);
        }
        let mut padding_correction = zero.clone();
        if per_lookup_leaves != (1usize << w_num_vars) {
            let eq_at_full = build_eq_x_r_vec(r_full, field_cfg)?;
            for j in per_lookup_leaves..eq_at_full.len() {
                padding_correction = padding_correction + &eq_at_full[j];
            }
        }
        let expected_q_local = beta.clone() - &psi_combined + &padding_correction;
        if expected_q_local != witness_result.expected_qs[ell] {
            return Err(GkrLogupError::GkrLeafMismatch);
        }
    }

    // ---- Step 6: Combine chunk lifts into parent polynomial claim ----
    let combined_polynomial: Vec<DynamicPolynomialF<F>> = (0..num_lookups)
        .map(|ell| combine_chunks::<F>(&proof.chunk_lifts[ell], chunk_width, width, &zero))
        .collect();

    Ok(GkrLogupGroupSubclaim {
        r_inner,
        combined_polynomial,
        parent_columns: meta.parent_columns.clone(),
    })
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute the polynomial-valued MLE evaluation of a `binary_poly<D>`
/// column at `point ∈ F_q^{n_vars}`. Returns a `DynamicPolynomialF<F>`
/// of degree `< D` whose coefficient `p` equals
/// `Σ_i eq(i, point) · bit_p(parent[i])`.
///
/// Convenience wrapper around [`compute_binary_poly_lifts`] for the
/// single-column case. Callers with multiple columns at the same
/// `point` should call [`compute_binary_poly_lifts`] directly to share
/// the eq-table build and parallelize across columns.
pub fn compute_binary_poly_lift<F, const D: usize>(
    parent: &DenseMultilinearExtension<BinaryPoly<D>>,
    point: &[F],
    field_cfg: &F::Config,
) -> DynamicPolynomialF<F>
where
    F: InnerTransparentField + FromPrimitiveWithConfig + Send + Sync,
    F::Inner: Send + Sync,
    F::Config: Sync,
{
    compute_binary_poly_lifts::<F, D>(&[parent], point, field_cfg)
        .into_iter()
        .next()
        .expect("single-col lift")
}

/// Compute polynomial-valued MLE evaluations of a batch of
/// `binary_poly<D>` columns at a shared `point ∈ F_q^{n_vars}`.
///
/// Builds the `eq(·, point)` table ONCE (an O(2^n_vars) operation),
/// then parallelizes the per-column bit-conditional sum across rayon
/// threads. Each column's inner accumulation uses `+=` on `coeffs[p]`
/// (no per-add allocation).
///
/// Drop-in replacement for calling [`compute_binary_poly_lift`] N
/// times in a serial loop — saves N-1 redundant eq builds plus N-fold
/// parallelism on the `O(N · 2^n_vars · D)` bit walk.
#[allow(clippy::arithmetic_side_effects)]
pub fn compute_binary_poly_lifts<F, const D: usize>(
    cols: &[&DenseMultilinearExtension<BinaryPoly<D>>],
    point: &[F],
    field_cfg: &F::Config,
) -> Vec<DynamicPolynomialF<F>>
where
    F: InnerTransparentField + FromPrimitiveWithConfig + Send + Sync,
    F::Inner: Send + Sync,
    F::Config: Sync,
{
    let zero = F::zero_with_cfg(field_cfg);
    let eq_table = build_eq_x_r_vec(point, field_cfg)
        .expect("compute_binary_poly_lifts: eq table build failed");
    cfg_iter!(cols)
        .map(|col| {
            let mut coeffs = vec![zero.clone(); D];
            for (i, entry) in col.iter().enumerate() {
                let bits = entry.inner();
                for (p, c) in bits.coeffs.iter().enumerate() {
                    if c.into_inner() {
                        coeffs[p] += &eq_table[i];
                    }
                }
            }
            DynamicPolynomialF::new_trimmed(coeffs)
        })
        .collect()
}

/// Combine K chunk lifts into the parent's combined polynomial:
///   `combined = Σ_k X^{k · chunk_width} · chunks[k]`
/// where the result has `width` coefficients (`width = K · chunk_width`).
pub fn combine_chunks<F: PrimeField>(
    chunks: &[DynamicPolynomialF<F>],
    chunk_width: usize,
    width: usize,
    zero: &F,
) -> DynamicPolynomialF<F> {
    let mut coeffs = vec![zero.clone(); width];
    for (k, chunk) in chunks.iter().enumerate() {
        let lo = k * chunk_width;
        for (p, c) in chunk.coeffs.iter().enumerate() {
            if lo + p < width {
                coeffs[lo + p] = c.clone();
            }
        }
    }
    DynamicPolynomialF::new_trimmed(coeffs)
}

/// Evaluate a `DynamicPolynomialF<F>` at the projecting element `a`
/// (`ψ_a` on a polynomial of degree < some bound). Horner from the
/// highest coefficient down.
#[allow(clippy::arithmetic_side_effects)]
fn eval_at_projecting_element<F: PrimeField>(
    poly: &DynamicPolynomialF<F>,
    a: &F,
    field_cfg: &F::Config,
) -> F {
    let mut acc = F::zero_with_cfg(field_cfg);
    for c in poly.coeffs.iter().rev() {
        acc = acc * a + c;
    }
    acc
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crypto_bigint::{U128, const_monty_params};
    use crypto_primitives::{Field, crypto_bigint_const_monty::ConstMontyField};
    use rand::{Rng, RngCore, SeedableRng, rngs::StdRng};
    use zinc_transcript::Blake3Transcript;

    const_monty_params!(TestParams, U128, "00000000b933426489189cb5b47d567f");
    type F = ConstMontyField<TestParams, { U128::LIMBS }>;

    fn rand_binary_poly_col(
        n_vars: usize,
        rng: &mut impl RngCore,
    ) -> DenseMultilinearExtension<BinaryPoly<32>> {
        let len = 1usize << n_vars;
        let evals: Vec<BinaryPoly<32>> =
            (0..len).map(|_| BinaryPoly::<32>::from(rng.next_u32())).collect();
        DenseMultilinearExtension::from_evaluations_vec(n_vars, evals, BinaryPoly::<32>::zero())
    }

    #[test]
    fn round_trip_l1_k4_bitpoly32() {
        let cfg = ();
        let mut rng = StdRng::seed_from_u64(42);
        let n_vars = 6; // W = 64
        let parent = rand_binary_poly_col(n_vars, &mut rng);

        let a: F = F::from(rng.next_u64());
        let table_type = LookupTableType::BitPoly { width: 32, chunk_width: Some(8) };

        let instance = BinaryPolyLookupInstance::<'_, F, 32> {
            parent_columns: vec![&parent],
            parent_column_indices: vec![0],
            table_type: table_type.clone(),
            projecting_element_f: &a,
            n_vars,
        };

        let mut p_ts = Blake3Transcript::new();
        let (proof, meta, prover_sub) =
            prove_group::<F, 32>(&mut p_ts, &instance, &cfg).expect("prove");

        let mut v_ts = Blake3Transcript::new();
        let verifier_sub =
            verify_group::<F>(&mut v_ts, &proof, &meta, &a, &cfg).expect("verify");

        assert_eq!(prover_sub.r_inner, verifier_sub.r_inner);
        assert_eq!(prover_sub.combined_polynomial, verifier_sub.combined_polynomial);

        // Sanity: combined_polynomial[0] should equal MLE[parent](r_inner).
        let parent_lift =
            compute_binary_poly_lift::<F, 32>(&parent, &verifier_sub.r_inner, &cfg);
        assert_eq!(verifier_sub.combined_polynomial[0], parent_lift);
    }

    #[test]
    fn round_trip_l2_k4_bitpoly32() {
        let cfg = ();
        let mut rng = StdRng::seed_from_u64(7);
        let n_vars = 5;
        let p1 = rand_binary_poly_col(n_vars, &mut rng);
        let p2 = rand_binary_poly_col(n_vars, &mut rng);

        let a: F = F::from(rng.next_u64());
        let table_type = LookupTableType::BitPoly { width: 32, chunk_width: Some(8) };
        let instance = BinaryPolyLookupInstance::<'_, F, 32> {
            parent_columns: vec![&p1, &p2],
            parent_column_indices: vec![0, 1],
            table_type,
            projecting_element_f: &a,
            n_vars,
        };

        let mut p_ts = Blake3Transcript::new();
        let (proof, meta, _) = prove_group::<F, 32>(&mut p_ts, &instance, &cfg).expect("prove");
        let mut v_ts = Blake3Transcript::new();
        let sub = verify_group::<F>(&mut v_ts, &proof, &meta, &a, &cfg).expect("verify");
        assert_eq!(sub.combined_polynomial.len(), 2);
    }

    #[test]
    fn tampered_chunk_lift_rejected() {
        let cfg = ();
        let mut rng = StdRng::seed_from_u64(99);
        let n_vars = 5;
        let parent = rand_binary_poly_col(n_vars, &mut rng);
        let a: F = F::from(rng.next_u64());
        let table_type = LookupTableType::BitPoly { width: 32, chunk_width: Some(8) };
        let instance = BinaryPolyLookupInstance::<'_, F, 32> {
            parent_columns: vec![&parent],
            parent_column_indices: vec![0],
            table_type,
            projecting_element_f: &a,
            n_vars,
        };
        let mut p_ts = Blake3Transcript::new();
        let (mut proof, meta, _) = prove_group::<F, 32>(&mut p_ts, &instance, &cfg).expect("prove");

        // Tamper a chunk lift coefficient.
        proof.chunk_lifts[0][0].coeffs[0] =
            proof.chunk_lifts[0][0].coeffs[0].clone() + F::from(1u64);

        let mut v_ts = Blake3Transcript::new();
        let res = verify_group::<F>(&mut v_ts, &proof, &meta, &a, &cfg);
        assert!(res.is_err(), "verifier must reject tampered chunk lift");
    }

    #[test]
    fn tampered_multiplicity_rejected() {
        let cfg = ();
        let mut rng = StdRng::seed_from_u64(123);
        let n_vars = 5;
        let parent = rand_binary_poly_col(n_vars, &mut rng);
        let a: F = F::from(rng.next_u64());
        let table_type = LookupTableType::BitPoly { width: 32, chunk_width: Some(8) };
        let instance = BinaryPolyLookupInstance::<'_, F, 32> {
            parent_columns: vec![&parent],
            parent_column_indices: vec![0],
            table_type,
            projecting_element_f: &a,
            n_vars,
        };
        let mut p_ts = Blake3Transcript::new();
        let (mut proof, meta, _) = prove_group::<F, 32>(&mut p_ts, &instance, &cfg).expect("prove");

        proof.aggregated_multiplicities[0][0] =
            proof.aggregated_multiplicities[0][0].clone() + F::from(1u64);

        let mut v_ts = Blake3Transcript::new();
        let res = verify_group::<F>(&mut v_ts, &proof, &meta, &a, &cfg);
        assert!(res.is_err(), "verifier must reject tampered multiplicity");
    }
}
