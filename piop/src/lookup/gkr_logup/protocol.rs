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
    BatchedGkrFractionProveResult,
};
use super::structs::{
    GkrFractionProof, GkrLogupError, GkrLogupGroupMeta, GkrLogupGroupProof,
    GkrLogupGroupSubclaim,
};
use super::tables::{generate_bitpoly_table, generate_word_table, word_shift};

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

    // ---- Step 1: Extract integer chunk indices directly from BitPoly bits ----
    //
    // For each (ell, k, i), the chunk's value is the integer
    //   n_{k,i} = Σ_{p=0..chunk_width} bit_{k·cw+p}(v^(ell)[i]) · 2^p
    // and the ψ_a-projected scalar is `subtable[n_{k,i}]` (where the
    // subtable is laid out so position `n` ↔ ψ_a of the BitPoly with
    // bit pattern `n`). This skips all field arithmetic for the
    // chunk-projection step — bit walks dominate.
    //
    // `chunks_idx[ell][k][i] ∈ [0, 2^chunk_width)` and is reused below
    // for both the multiplicity histogram (Step 3) and the witness
    // fraction tree's `leaf_q` construction (Step 5), avoiding any
    // intermediate `chunks_psi: Vec<Vec<Vec<F>>>` materialization.
    let a = instance.projecting_element_f.clone();
    let zero = F::zero_with_cfg(field_cfg);

    let chunks_idx: Vec<Vec<Vec<u32>>> = cfg_iter!(instance.parent_columns)
        .map(|parent| {
            // For each row, pack the entry's bits into a u32 once via
            // the BinaryPoly iter abstraction (works under both the
            // `Vec<Boolean>` and packed-u64 representations). Then
            // extract per-chunk indices by shifting.
            let row_packed: Vec<u32> = (0..witness_len)
                .map(|i| {
                    let mut packed: u32 = 0;
                    for (idx, b) in parent.evaluations[i].iter().enumerate() {
                        if b.into_inner() {
                            packed |= 1u32 << idx;
                        }
                    }
                    packed
                })
                .collect();
            let chunk_mask: u32 = if chunk_width == 32 {
                u32::MAX
            } else {
                (1u32 << chunk_width) - 1
            };
            (0..num_chunks)
                .map(|k| {
                    let shift = (k * chunk_width) as u32;
                    row_packed
                        .iter()
                        .map(|&p| (p >> shift) & chunk_mask)
                        .collect()
                })
                .collect()
        })
        .collect();

    // ---- Step 2: Build subtable T = ψ_a({0,1}^{<chunk_width}[X]) ----
    let subtable: Vec<F> = generate_bitpoly_table(chunk_width, &a, field_cfg);

    // ---- Steps 3-8: multiplicities, challenges, fraction trees, GKR ----
    //
    // Shared with every other table type: from here to the lifts, the
    // proof is a function of the chunk indices and the subtable alone.
    let FractionPhase { agg_mults, witness_result, table_gkr } = prove_fraction_phase(
        transcript,
        &chunks_idx,
        &subtable,
        num_lookups,
        num_chunks,
        witness_len,
        field_cfg,
    );

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
        .map(|mut parent_lifted| {
            // `compute_binary_poly_lifts` returns trimmed polys; if the parent
            // column has structurally-zero high bits across all rows (e.g.
            // SHA's `S = w >> k` columns), the trimmed length can be < width.
            // Zero-pad up to `width` so chunk slicing always sees full chunks.
            if parent_lifted.coeffs.len() < width {
                parent_lifted.coeffs.resize(width, zero.clone());
            }
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

/// The half of a lookup group's proof that does not care what the parent
/// columns are made of: multiplicities over the chunk indices, the
/// transcript's challenges, the witness and table fraction trees, and
/// both GKR runs.
///
/// Everything above this point differs per table type -- how a column's
/// cells become chunk indices, and which subtable those index into --
/// and everything below it differs again, in how the parent's lift is
/// taken. In between, a chunk index is a chunk index.
pub(super) struct FractionPhase<F: PrimeField> {
    pub agg_mults: Vec<Vec<F>>,
    pub witness_result: BatchedGkrFractionProveResult<F>,
    pub table_gkr: GkrFractionProof<F>,
}

#[allow(clippy::arithmetic_side_effects)]
pub(super) fn prove_fraction_phase<F>(
    transcript: &mut impl Transcript,
    chunks_idx: &[Vec<Vec<u32>>],
    subtable: &[F],
    num_lookups: usize,
    num_chunks: usize,
    witness_len: usize,
    field_cfg: &F::Config,
) -> FractionPhase<F>
where
    F: InnerTransparentField + FromPrimitiveWithConfig + Send + Sync,
    F::Inner: ConstTranscribable + Zero + Default + Send + Sync,
    F::Modulus: ConstTranscribable,
    F::Config: Sync,
{
    let one = F::one_with_cfg(field_cfg);
    let zero = F::zero_with_cfg(field_cfg);
    let table_len = subtable.len();

    // ---- Step 3: Multiplicities via per-ell histogram ----
    //
    // Direct array tally on integer chunk indices — no hashmap. Saves
    // L·K·W hashmap lookups (~2M at typical sizes) and frees the
    // table-index hashmap allocation.
    let agg_mults: Vec<Vec<F>> = cfg_iter!(chunks_idx)
        .map(|lookup_chunks| {
            let mut counts = vec![0u64; table_len];
            for chunk in lookup_chunks {
                for &n in chunk {
                    counts[n as usize] += 1;
                }
            }
            counts
                .into_iter()
                .map(|c| F::from_with_cfg(c, field_cfg))
                .collect()
        })
        .collect();

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

    // Pre-compute β − subtable[n] for each n ∈ [0, table_len). Per-leaf
    // construction below becomes a single index + clone (no field op).
    let beta_minus_subtable: Vec<F> = subtable
        .iter()
        .map(|s| beta.clone() - s)
        .collect();

    // L witness fraction trees built in parallel — each tree's
    // construction is independent (different ell), and `build_fraction_tree`
    // itself is the heavy part (O(K·W) field ops per tree).
    let witness_trees: Vec<_> = cfg_into_iter!(0..num_lookups)
        .map(|ell| {
            let mut leaf_q = Vec::with_capacity(w_size);
            for k in 0..num_chunks {
                for i in 0..witness_len {
                    let n = chunks_idx[ell][k][i] as usize;
                    leaf_q.push(beta_minus_subtable[n].clone());
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

    FractionPhase { agg_mults, witness_result, table_gkr }
}

/// Inputs to [`prove_group_word`] for a single lookup group of integer
/// columns against `LookupTableType::Word { width, chunk_width }`.
///
/// Where a BitPoly parent is a column of polynomials, a Word parent is a
/// column of integers -- the trace's `int` group -- so the cell already
/// holds the number the table is indexed by, and no projection is needed
/// to recover it.
pub struct WordLookupInstance<'a, F: PrimeField, I> {
    /// L parent column MLEs (integer-valued).
    pub parent_columns: Vec<&'a DenseMultilinearExtension<I>>,
    /// L flat-trace column indices, mirrored into the proof's group meta.
    pub parent_column_indices: Vec<usize>,
    /// Lookup table type -- must be `Word { width, chunk_width }`.
    pub table_type: LookupTableType,
    /// Projecting element, threaded from step 3 for transcript parity.
    pub projecting_element_f: &'a F,
    /// Number of MLE variables of each parent column (= log2(W)).
    pub n_vars: usize,
}

/// Read an integer cell as the unsigned number the Word table indexes by.
///
/// `ConstTranscribable` is the only integer-shaped thing `Zt::Int` is
/// guaranteed to offer, and it is enough: the transcription is the
/// value's own little-endian bytes. A negative cell transcribes with its
/// high bits set, so it reads back far above any `2^width` and simply is
/// not in the table -- which is the refusal a range check is for, not a
/// special case to write.
fn word_of<I: ConstTranscribable>(cell: &I) -> u128 {
    let mut buf = vec![0u8; I::NUM_BYTES];
    cell.write_transcription_bytes_exact(&mut buf);
    let mut value: u128 = 0;
    for (i, byte) in buf.iter().take(16).enumerate() {
        value |= (*byte as u128) << (8 * i);
    }
    // Any byte beyond the low 16 that is set puts the cell out of every
    // Word table we can build, so saturate rather than wrap into range.
    if buf.iter().skip(16).any(|b| *b != 0) {
        return u128::MAX;
    }
    value
}

/// The Word half of [`prove_group`]: same proof, different parents.
#[allow(clippy::arithmetic_side_effects)]
pub fn prove_group_word<F, I>(
    transcript: &mut impl Transcript,
    instance: &WordLookupInstance<'_, F, I>,
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
    I: ConstTranscribable + Clone + Send + Sync,
{
    let (width, chunk_width) = match instance.table_type {
        LookupTableType::Word { width, chunk_width: Some(cw) } => (width, cw),
        LookupTableType::Word { width, chunk_width: None } => (width, width),
        _ => return Err(GkrLogupError::WitnessNotInTable),
    };
    assert!(chunk_width > 0 && width % chunk_width == 0, "chunk_width must divide width");
    assert!(chunk_width < 32, "chunk_width must leave a u32 chunk index");
    let num_chunks = width / chunk_width;
    // The witness fraction tree has num_chunks * witness_len leaves. Both
    // the BitPoly path and this one only ever build power-of-two trees --
    // BitPoly because its chunk count always is one -- so the padded case
    // is untested rather than supported. Refuse the shape here, where the
    // reason is legible, rather than emit a proof the verifier rejects.
    assert!(
        num_chunks.is_power_of_two(),
        "width / chunk_width must be a power of two, got {num_chunks}"
    );
    let num_lookups = instance.parent_columns.len();
    let n_vars = instance.n_vars;
    let witness_len = 1usize << n_vars;
    let zero = F::zero_with_cfg(field_cfg);

    // ---- Step 1: chunk indices straight off the integers ----
    //
    // No bit walk and no reverse lookup: the cell is the number, so a
    // chunk is a shift and a mask. A cell too wide for the table lands
    // outside it by construction and the lookup refuses.
    let chunk_mask: u128 = (1u128 << chunk_width) - 1;
    let mut chunks_idx: Vec<Vec<Vec<u32>>> = Vec::with_capacity(num_lookups);
    for parent in &instance.parent_columns {
        let mut per_chunk = vec![Vec::with_capacity(witness_len); num_chunks];
        for i in 0..witness_len {
            let value = word_of(&parent.evaluations[i]);
            for (k, out) in per_chunk.iter_mut().enumerate() {
                let shifted = value
                    .checked_shr((k * chunk_width) as u32)
                    .unwrap_or(0);
                let idx = shifted & chunk_mask;
                // Saturating: an out-of-table cell must miss the table,
                // never alias into it.
                out.push(if value >= (1u128 << width) {
                    u32::MAX
                } else {
                    idx as u32
                });
            }
        }
        chunks_idx.push(per_chunk);
    }

    // ---- Step 2: the Word subtable is the integers themselves ----
    let subtable: Vec<F> = generate_word_table(chunk_width, field_cfg);
    if chunks_idx
        .iter()
        .flatten()
        .flatten()
        .any(|n| *n as usize >= subtable.len())
    {
        return Err(GkrLogupError::WitnessNotInTable);
    }

    let FractionPhase { agg_mults, witness_result, table_gkr } = prove_fraction_phase(
        transcript,
        &chunks_idx,
        &subtable,
        num_lookups,
        num_chunks,
        witness_len,
        field_cfg,
    );

    // ---- Step 9: the parent's lift is one evaluation ----
    //
    // An integer cell has no coefficients to walk, so the lift of a
    // chunk is the multilinear evaluation of that chunk's own column at
    // r_inner, carried as a degree-0 polynomial.
    let r_full = &witness_result.eval_point;
    assert!(r_full.len() >= n_vars, "GKR descent must have at least n_vars row variables");
    let r_inner: Vec<F> = r_full[..n_vars].to_vec();
    let eq_table = build_eq_x_r_vec(&r_inner, field_cfg)
        .map_err(|_| GkrLogupError::WitnessNotInTable)?;

    let chunk_lifts: Vec<Vec<DynamicPolynomialF<F>>> = chunks_idx
        .iter()
        .map(|per_chunk| {
            per_chunk
                .iter()
                .map(|chunk| {
                    let mut acc = zero.clone();
                    for (i, n) in chunk.iter().enumerate() {
                        let term = F::from_with_cfg(*n as u64, field_cfg);
                        acc = acc + &(eq_table[i].clone() * &term);
                    }
                    DynamicPolynomialF::new_trimmed(vec![acc])
                })
                .collect()
        })
        .collect();

    // ---- Step 10: place value, not coefficient position ----
    let combined_polynomial: Vec<DynamicPolynomialF<F>> = (0..num_lookups)
        .map(|ell| combine_chunks_word::<F>(&chunk_lifts[ell], chunk_width, field_cfg))
        .collect();

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
        LookupTableType::Word { width, chunk_width: Some(cw) } => (*width, *cw),
        LookupTableType::Word { width, chunk_width: None } => (*width, *width),
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
    // The subtable a chunk index reads against. For Word it is the
    // integers themselves, so psi_a is the identity there and the leaf
    // check below needs no case of its own.
    let subtable = match &meta.table_type {
        LookupTableType::Word { .. } => generate_word_table::<F>(chunk_width, field_cfg),
        _ => generate_bitpoly_table::<F>(chunk_width, projecting_element_f, field_cfg),
    };
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
        .map(|ell| match &meta.table_type {
            // Word chunks carry place value, so they recombine to one
            // field element; BitPoly chunks are coefficient blocks.
            LookupTableType::Word { .. } => {
                combine_chunks_word::<F>(&proof.chunk_lifts[ell], chunk_width, field_cfg)
            }
            _ => combine_chunks::<F>(&proof.chunk_lifts[ell], chunk_width, width, &zero),
        })
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
                for (p, c) in entry.iter().enumerate() {
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

/// Combine K Word chunk lifts into the parent's claim at `r_inner`.
///
/// A Word cell is an integer, not a polynomial, so its chunks carry
/// place value rather than coefficient position: the parent is
/// `Σ_k 2^{k · chunk_width} · chunk_k`, one field element, which rides
/// as a degree-0 polynomial so the rest of the protocol -- which only
/// ever evaluates these at the projecting element -- needs no case.
#[allow(clippy::arithmetic_side_effects)]
pub fn combine_chunks_word<F: PrimeField + FromPrimitiveWithConfig>(
    chunks: &[DynamicPolynomialF<F>],
    chunk_width: usize,
    field_cfg: &F::Config,
) -> DynamicPolynomialF<F> {
    let mut acc = F::zero_with_cfg(field_cfg);
    let mut place = F::one_with_cfg(field_cfg);
    let shift = word_shift::<F>(chunk_width, field_cfg);
    for chunk in chunks {
        let value = chunk
            .coeffs
            .first()
            .cloned()
            .unwrap_or_else(|| F::zero_with_cfg(field_cfg));
        acc = acc + &(place.clone() * &value);
        place = place * &shift;
    }
    DynamicPolynomialF::new_trimmed(vec![acc])
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

    /// A column of integers proves and verifies against Word(16), and
    /// the parent claim the verifier is left holding is the integers'
    /// own multilinear evaluation -- place value, not coefficients.
    #[test]
    fn round_trip_word16_over_int_column() {
        let cfg = ();
        let n_vars = 5; // W = 32
        let witness_len = 1usize << n_vars;
        // Values inside Word(16).
        let cells: Vec<i64> = (0..witness_len).map(|i| ((i * 613) % 65536) as i64).collect();
        let parent = DenseMultilinearExtension::from_evaluations_vec(n_vars, cells.clone(), 0i64);

        let a: F = F::from(7u64);
        let table_type = LookupTableType::Word { width: 16, chunk_width: Some(8) };
        let instance = WordLookupInstance::<'_, F, i64> {
            parent_columns: vec![&parent],
            parent_column_indices: vec![0],
            table_type: table_type.clone(),
            projecting_element_f: &a,
            n_vars,
        };

        let mut p_ts = Blake3Transcript::new();
        let (proof, meta, prover_sub) =
            prove_group_word::<F, i64>(&mut p_ts, &instance, &cfg).expect("prove");

        let mut v_ts = Blake3Transcript::new();
        let verifier_sub =
            verify_group::<F>(&mut v_ts, &proof, &meta, &a, &cfg).expect("verify");

        assert_eq!(prover_sub.r_inner, verifier_sub.r_inner);
        assert_eq!(prover_sub.combined_polynomial, verifier_sub.combined_polynomial);

        // The parent claim is Σ_i eq(i, r) · cell_i, as a degree-0 poly.
        let eq = build_eq_x_r_vec(&verifier_sub.r_inner, &cfg).expect("eq");
        let mut expected = F::from(0u64);
        for (i, c) in cells.iter().enumerate() {
            expected = expected + &(eq[i].clone() * &F::from(*c as u64));
        }
        assert_eq!(
            verifier_sub.combined_polynomial[0],
            DynamicPolynomialF::new_trimmed(vec![expected])
        );
    }

    /// The witness fraction tree is built over `num_chunks · witness_len`
    /// leaves and only the power-of-two case is exercised, so a chunk
    /// count that is not one is refused where the reason is readable
    /// rather than surfacing later as a verifier rejection.
    #[test]
    #[should_panic(expected = "must be a power of two")]
    fn a_chunk_count_that_is_not_a_power_of_two_is_refused() {
        let cfg = ();
        let n_vars = 5;
        let cells: Vec<i64> = (0..(1usize << n_vars)).map(|i| (i % 256) as i64).collect();
        let parent = DenseMultilinearExtension::from_evaluations_vec(n_vars, cells, 0i64);
        let a: F = F::from(7u64);
        let instance = WordLookupInstance::<'_, F, i64> {
            parent_columns: vec![&parent],
            parent_column_indices: vec![0],
            // 20 / 4 = 5 chunks.
            table_type: LookupTableType::Word { width: 20, chunk_width: Some(4) },
            projecting_element_f: &a,
            n_vars,
        };
        let mut ts = Blake3Transcript::new();
        let _ = prove_group_word::<F, i64>(&mut ts, &instance, &cfg);
    }

    /// The whole point: a cell outside the table cannot be proved. A
    /// negative slack is exactly this case -- it reads back above every
    /// 2^width -- so the range check refuses rather than proving.
    #[test]
    fn a_cell_outside_the_word_table_is_refused() {
        let cfg = ();
        let n_vars = 5;
        let witness_len = 1usize << n_vars;
        let a: F = F::from(7u64);
        let table_type = LookupTableType::Word { width: 16, chunk_width: Some(8) };

        for bad in [70_000i64, -1i64, -96i64] {
            let mut cells: Vec<i64> = (0..witness_len).map(|i| (i % 256) as i64).collect();
            cells[3] = bad;
            let parent = DenseMultilinearExtension::from_evaluations_vec(n_vars, cells, 0i64);
            let instance = WordLookupInstance::<'_, F, i64> {
                parent_columns: vec![&parent],
                parent_column_indices: vec![0],
                table_type: table_type.clone(),
                projecting_element_f: &a,
                n_vars,
            };
            let mut ts = Blake3Transcript::new();
            assert!(
                prove_group_word::<F, i64>(&mut ts, &instance, &cfg).is_err(),
                "a cell of {bad} is not in Word(16) and must not prove"
            );
        }
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
