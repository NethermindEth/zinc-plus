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

use std::collections::HashMap;

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
use super::tables::{
    generate_bitpoly_table, generate_prescribed_table, generate_word_table,
    prescribed_multiplicities, word_shift,
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
    let agg_mults = histogram_multiplicities::<F>(&chunks_idx, subtable.len(), field_cfg);
    let FractionPhase { witness_result, table_gkr } = prove_fraction_phase(
        transcript,
        &WitnessLeaves::Indexed(&chunks_idx),
        &subtable,
        &agg_mults,
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
/// columns are made of: the transcript's challenges, the witness and
/// table fraction trees, and both GKR runs.
///
/// Everything above this point differs per table type -- how a column's
/// cells become chunk indices, which subtable those index into, and where
/// the table's multiplicities come from -- and everything below it
/// differs again, in how the parent's lift is taken. In between, a chunk
/// index is a chunk index.
pub(super) struct FractionPhase<F: PrimeField> {
    pub witness_result: BatchedGkrFractionProveResult<F>,
    pub table_gkr: GkrFractionProof<F>,
}

/// Where a group's witness leaves come from.
///
/// A whole-column lookup gives every leaf the numerator one and reads its
/// denominator out of the table, so a cell is an index into it. A
/// selection reads the cell's own value -- the columns hold cells the
/// table says nothing about -- and the numerator is what says whether the
/// selection picked the cell.
pub(super) enum WitnessLeaves<'a, F: PrimeField> {
    /// `chunks_idx[ell][k][i]` is the subtable position of chunk `k` of
    /// row `i` of the `ell`-th parent column.
    Indexed(&'a [Vec<Vec<u32>>]),
    /// `cells` is the group's columns laid end to end, and
    /// `selections[ell]` names the positions in it that selection reads,
    /// once per time it reads them.
    Selected {
        cells: &'a [F],
        selections: &'a [Vec<u32>],
    },
}

/// The multiplicity of each table entry among the chunk indices: the
/// histogram a table with nothing prescribed about it has to be told.
#[allow(clippy::arithmetic_side_effects)]
pub(super) fn histogram_multiplicities<F>(
    chunks_idx: &[Vec<Vec<u32>>],
    table_len: usize,
    field_cfg: &F::Config,
) -> Vec<Vec<F>>
where
    F: PrimeField + FromPrimitiveWithConfig + Send + Sync,
    F::Config: Sync,
{
    // Direct array tally on integer chunk indices — no hashmap. Saves
    // L·K·W hashmap lookups (~2M at typical sizes) and frees the
    // table-index hashmap allocation.
    cfg_iter!(chunks_idx)
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
        .collect()
}

#[allow(clippy::arithmetic_side_effects, clippy::too_many_arguments)]
pub(super) fn prove_fraction_phase<F>(
    transcript: &mut impl Transcript,
    leaves: &WitnessLeaves<'_, F>,
    subtable: &[F],
    agg_mults: &[Vec<F>],
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

    // ---- Step 4: Absorb agg multiplicities, sample β, α ----
    let mut buf = vec![0u8; F::Inner::NUM_BYTES];
    for agg in agg_mults {
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

    // L witness fraction trees built in parallel — each tree's
    // construction is independent (different ell), and `build_fraction_tree`
    // itself is the heavy part (O(K·W) field ops per tree).
    let witness_trees: Vec<_> = match leaves {
        WitnessLeaves::Indexed(chunks_idx) => {
            // Pre-compute β − subtable[n] for each n ∈ [0, table_len). Per-leaf
            // construction below becomes a single index + clone (no field op).
            let beta_minus_subtable: Vec<F> =
                subtable.iter().map(|s| beta.clone() - s).collect();
            cfg_into_iter!(0..num_lookups)
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
                .collect()
        }
        WitnessLeaves::Selected { cells, selections } => {
            // Every selection reads the same cells, so the denominators are
            // built once and only the numerator tells the trees apart. A
            // cell nobody selected gets a zero there and contributes
            // nothing, which is also what the padding gets -- so a selected
            // group has no pad entry to pin and no row to slide a value into.
            let mut leaf_q: Vec<F> = cells.iter().map(|c| beta.clone() - c).collect();
            leaf_q.resize(w_size, one.clone());
            cfg_into_iter!(0..num_lookups)
                .map(|ell| {
                    let mut leaf_p = vec![zero.clone(); w_size];
                    for &j in &selections[ell] {
                        leaf_p[j as usize] += &one;
                    }
                    build_fraction_tree(leaf_p, leaf_q.clone())
                })
                .collect()
        }
    };

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

    FractionPhase { witness_result, table_gkr }
}

/// Inputs to [`prove_group_word`] and [`prove_group_prescribed`] for a
/// single lookup group of integer columns.
///
/// Where a BitPoly parent is a column of polynomials, these parents are
/// columns of integers -- the trace's `int` group -- so the cell already
/// holds the number the table is indexed by, and no projection is needed
/// to recover it.
pub struct IntLookupInstance<'a, F: PrimeField, I> {
    /// L parent column MLEs (integer-valued).
    pub parent_columns: Vec<&'a DenseMultilinearExtension<I>>,
    /// L flat-trace column indices, mirrored into the proof's group meta.
    pub parent_column_indices: Vec<usize>,
    /// Lookup table type -- must read integer columns.
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
    instance: &IntLookupInstance<'_, F, I>,
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

    let agg_mults = histogram_multiplicities::<F>(&chunks_idx, subtable.len(), field_cfg);
    let FractionPhase { witness_result, table_gkr } = prove_fraction_phase(
        transcript,
        &WitnessLeaves::Indexed(&chunks_idx),
        &subtable,
        &agg_mults,
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

/// A prescribed cell is looked up whole, so its group has one chunk and
/// no width of its own; this is the width the rest of the protocol asks
/// for and never divides anything by.
const PRESCRIBED_CHUNK_WIDTH: usize = 1;

/// The table a `Prescribed` type names and the multiplicity of each of
/// its entries, both of which anyone holding the type can build.
///
/// Refused when the type cannot mean what it says: a pad that is one of
/// the values would leave that entry's multiplicity ambiguous, and values
/// that outnumber the rows cannot all be laid down.
fn prescribed_table_side<F>(
    values: &[u64],
    pad: u64,
    witness_len: usize,
    field_cfg: &F::Config,
) -> Result<(Vec<F>, Vec<F>), GkrLogupError<F>>
where
    F: PrimeField + FromPrimitiveWithConfig,
{
    if values.contains(&pad) || values.len() > witness_len {
        return Err(GkrLogupError::MalformedPrescribedTable);
    }
    Ok((
        generate_prescribed_table(values, pad, field_cfg),
        prescribed_multiplicities(values.len(), witness_len, field_cfg),
    ))
}

/// The prescribed half of [`prove_group`]: the column is the table.
///
/// A cell is looked up whole -- one chunk, since the table is a multiset
/// and not a range to decompose -- and the multiplicities are the table's
/// own rather than a count of what the witness happened to hold. That is
/// what makes the LogUp identity say multiset equality: the column holds
/// each value once and the pad in every other row, or the identity fails.
#[allow(clippy::arithmetic_side_effects)]
pub fn prove_group_prescribed<F, I>(
    transcript: &mut impl Transcript,
    instance: &IntLookupInstance<'_, F, I>,
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
    let (values, pad) = match &instance.table_type {
        LookupTableType::Prescribed { values, pad } => (values, *pad),
        _ => return Err(GkrLogupError::WitnessNotInTable),
    };
    let num_lookups = instance.parent_columns.len();
    let n_vars = instance.n_vars;
    let witness_len = 1usize << n_vars;
    let zero = F::zero_with_cfg(field_cfg);

    // ---- Steps 1-2: the table, and where each cell sits in it ----
    //
    // A value the table does not name has no place to sit, so the column
    // is refused here rather than proved against a table it misses.
    let (subtable, multiplicities) =
        prescribed_table_side::<F>(values, pad, witness_len, field_cfg)?;
    let mut position: HashMap<u128, u32> = HashMap::with_capacity(subtable.len());
    for (index, entry) in values.iter().chain(std::iter::once(&pad)).enumerate() {
        position.insert(
            u128::from(*entry),
            u32::try_from(index).expect("a table this long has no index"),
        );
    }
    let mut chunks_idx: Vec<Vec<Vec<u32>>> = Vec::with_capacity(num_lookups);
    for parent in &instance.parent_columns {
        let mut chunk = Vec::with_capacity(witness_len);
        for i in 0..witness_len {
            let value = word_of(&parent.evaluations[i]);
            match position.get(&value) {
                Some(index) => chunk.push(*index),
                None => return Err(GkrLogupError::WitnessNotInTable),
            }
        }
        chunks_idx.push(vec![chunk]);
    }

    // Every column is checked against the same prescribed table, so every
    // one of them carries the same multiplicities.
    let agg_mults = vec![multiplicities; num_lookups];
    let FractionPhase { witness_result, table_gkr } = prove_fraction_phase(
        transcript,
        &WitnessLeaves::Indexed(&chunks_idx),
        &subtable,
        &agg_mults,
        num_lookups,
        1,
        witness_len,
        field_cfg,
    );

    // ---- Step 9: the entry a cell indexes is the cell's own value ----
    //
    // So the single chunk's lift is the parent column's multilinear
    // evaluation at r_inner: the same number the witness leaf check
    // reconstructs and the protocol layer binds against the commitment.
    let r_full = &witness_result.eval_point;
    assert!(r_full.len() >= n_vars, "GKR descent must have at least n_vars row variables");
    let r_inner: Vec<F> = r_full[..n_vars].to_vec();
    let eq_table = build_eq_x_r_vec(&r_inner, field_cfg)?;

    let chunk_lifts: Vec<Vec<DynamicPolynomialF<F>>> = chunks_idx
        .iter()
        .map(|per_chunk| {
            per_chunk
                .iter()
                .map(|chunk| {
                    let mut acc = zero.clone();
                    for (i, n) in chunk.iter().enumerate() {
                        acc += &(eq_table[i].clone() * &subtable[*n as usize]);
                    }
                    DynamicPolynomialF::new_trimmed(vec![acc])
                })
                .collect()
        })
        .collect();
    let combined_polynomial: Vec<DynamicPolynomialF<F>> = chunk_lifts
        .iter()
        .map(|lifts| combine_chunks_word::<F>(lifts, PRESCRIBED_CHUNK_WIDTH, field_cfg))
        .collect();

    let meta = GkrLogupGroupMeta {
        table_type: instance.table_type.clone(),
        num_lookups,
        num_chunks: 1,
        chunk_width: PRESCRIBED_CHUNK_WIDTH,
        witness_len,
        parent_columns: instance.parent_column_indices.clone(),
    };
    let proof = GkrLogupGroupProof {
        chunk_lifts,
        // The verifier builds this table's multiplicities itself, so the
        // proof does not carry them.
        aggregated_multiplicities: Vec::new(),
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

/// Inputs to [`prove_group_selected`] for a group whose table names the
/// cells it speaks for.
///
/// The columns arrive already projected into the field: a cell no
/// selection names still sits in a denominator, and no table says what it
/// may be, so its value has to be the one the commitment carries rather
/// than one a table can index.
pub struct SelectedLookupInstance<'a, F: PrimeField> {
    /// The C columns this group's selections reach across.
    pub parent_columns: Vec<&'a DenseMultilinearExtension<F::Inner>>,
    /// C flat-trace column indices, mirrored into the proof's group meta.
    pub parent_column_indices: Vec<usize>,
    /// Lookup table type -- must be `Selected`.
    pub table_type: LookupTableType,
    /// Number of MLE variables of each column (= log2(W)).
    pub n_vars: usize,
}

/// The flat leaf position of every cell each selection names, over the
/// group's columns laid end to end.
///
/// Refused when a selection names a cell the group does not have: a slot
/// past its columns or a row past its rows indexes nothing at all.
#[allow(clippy::arithmetic_side_effects)]
fn selection_positions<F: PrimeField>(
    selections: &[Vec<(u32, u32)>],
    num_columns: usize,
    witness_len: usize,
) -> Result<Vec<Vec<u32>>, GkrLogupError<F>> {
    selections
        .iter()
        .map(|selection| {
            selection
                .iter()
                .map(|&(slot, row)| {
                    let (slot, row) = (slot as usize, row as usize);
                    if slot >= num_columns || row >= witness_len {
                        return Err(GkrLogupError::MalformedSelection);
                    }
                    u32::try_from(slot * witness_len + row)
                        .map_err(|_| GkrLogupError::MalformedSelection)
                })
                .collect()
        })
        .collect()
}

/// The selected half of [`prove_group`]: the table names its own cells.
///
/// Where a prescribed table speaks for a whole column and pads the rest,
/// a selection speaks for the cells the signature names and says nothing
/// about the others. So the group runs one tree per selection over the
/// same columns, the numerator carrying the selection and the denominator
/// the column's own value -- and an unselected cell, numerator zero,
/// contributes exactly nothing to either side.
#[allow(clippy::arithmetic_side_effects)]
pub fn prove_group_selected<F>(
    transcript: &mut impl Transcript,
    instance: &SelectedLookupInstance<'_, F>,
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
    let (values, selections) = match &instance.table_type {
        LookupTableType::Selected { values, selections } => (values, selections),
        _ => return Err(GkrLogupError::WitnessNotInTable),
    };
    let num_columns = instance.parent_columns.len();
    let num_lookups = selections.len();
    let n_vars = instance.n_vars;
    let witness_len = 1usize << n_vars;
    let positions = selection_positions::<F>(selections, num_columns, witness_len)?;

    // The cells the selections read: the group's columns end to end, in
    // the order the specs declare them.
    let cells: Vec<F> = instance
        .parent_columns
        .iter()
        .flat_map(|column| {
            column
                .evaluations
                .iter()
                .map(|value| F::new_unchecked_with_cfg(value.clone(), field_cfg))
        })
        .collect();

    // The table is the multiset itself, each entry once. Nothing here
    // comes from the witness, so the proof carries no multiplicities.
    let subtable: Vec<F> = values
        .iter()
        .map(|value| F::from_with_cfg(*value, field_cfg))
        .collect();
    let agg_mults = vec![vec![F::one_with_cfg(field_cfg); values.len()]; num_lookups];

    let FractionPhase { witness_result, table_gkr } = prove_fraction_phase(
        transcript,
        &WitnessLeaves::Selected { cells: &cells, selections: &positions },
        &subtable,
        &agg_mults,
        num_lookups,
        num_columns,
        witness_len,
        field_cfg,
    );

    // One claim per column, shared by every selection that reaches into
    // it: the column's own multilinear evaluation at r_inner, which is
    // what the protocol layer binds against the commitment.
    let r_full = &witness_result.eval_point;
    assert!(r_full.len() >= n_vars, "GKR descent must have at least n_vars row variables");
    let r_inner: Vec<F> = r_full[..n_vars].to_vec();
    let eq_table = build_eq_x_r_vec(&r_inner, field_cfg)?;
    let column_lifts: Vec<DynamicPolynomialF<F>> = cells
        .chunks_exact(witness_len)
        .map(|column| {
            let mut acc = F::zero_with_cfg(field_cfg);
            for (eq, cell) in eq_table.iter().zip(column) {
                acc += &(eq.clone() * cell);
            }
            DynamicPolynomialF::new_trimmed(vec![acc])
        })
        .collect();

    let meta = GkrLogupGroupMeta {
        table_type: instance.table_type.clone(),
        num_lookups,
        num_chunks: num_columns,
        chunk_width: PRESCRIBED_CHUNK_WIDTH,
        witness_len,
        parent_columns: instance.parent_column_indices.clone(),
    };
    let proof = GkrLogupGroupProof {
        // Every selection reads the same columns, so the group's lifts are
        // one row rather than one per selection.
        chunk_lifts: vec![column_lifts.clone()],
        aggregated_multiplicities: Vec::new(),
        witness_gkr: witness_result.proof,
        table_gkr,
        bin_lifts_at_r_inner: Vec::new(),
    };
    let subclaim = GkrLogupGroupSubclaim {
        r_inner,
        combined_polynomial: column_lifts,
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
        LookupTableType::Prescribed { .. } | LookupTableType::Selected { .. } => {
            (PRESCRIBED_CHUNK_WIDTH, PRESCRIBED_CHUNK_WIDTH)
        }
    };
    assert!(chunk_width > 0 && width % chunk_width == 0);

    // Every selection reads the group's columns, so the group carries one
    // row of lifts; a whole-column lookup reads only its own parent, so it
    // carries one row per parent.
    let selections = match &meta.table_type {
        LookupTableType::Selected { selections, .. } => Some(selections.as_slice()),
        _ => None,
    };
    let num_lookups = meta.num_lookups;
    let num_chunks = meta.num_chunks;
    let witness_len = meta.witness_len;
    let n_vars = (witness_len as f64).log2() as usize;
    assert_eq!(1usize << n_vars, witness_len, "witness_len must be a power of 2");
    match selections {
        // A selection's leaves run over the group's own columns, a block
        // each; every other table decomposes a cell into chunks.
        Some(sels) => {
            assert_eq!(num_chunks, meta.parent_columns.len());
            assert_eq!(num_lookups, sels.len());
        }
        None => assert_eq!(num_chunks, width / chunk_width),
    }
    let lift_rows = if selections.is_some() { 1 } else { num_lookups };
    if proof.chunk_lifts.len() != lift_rows {
        return Err(GkrLogupError::GkrLeafMismatch);
    }

    let zero = F::zero_with_cfg(field_cfg);
    let one = F::one_with_cfg(field_cfg);

    // ---- Reconstruct subtable + shifts ----
    // The subtable a chunk index reads against. For Word it is the
    // integers themselves, so psi_a is the identity there and the leaf
    // check below needs no case of its own. A prescribed or selected
    // table is its values, and its multiplicities come with them: the
    // verifier builds that whole side and takes none of it from the proof.
    let (subtable, prescribed_mults) = match &meta.table_type {
        LookupTableType::Word { .. } => (generate_word_table::<F>(chunk_width, field_cfg), None),
        LookupTableType::Prescribed { values, pad } => {
            let (table, multiplicities) =
                prescribed_table_side::<F>(values, *pad, witness_len, field_cfg)?;
            (table, Some(vec![multiplicities; num_lookups]))
        }
        LookupTableType::Selected { values, .. } => {
            // No pad: a cell no selection names is not looked up at all,
            // so there is no row left over for the table to account for.
            let table: Vec<F> = values
                .iter()
                .map(|value| F::from_with_cfg(*value, field_cfg))
                .collect();
            let multiplicities = vec![one.clone(); values.len()];
            (table, Some(vec![multiplicities; num_lookups]))
        }
        _ => (
            generate_bitpoly_table::<F>(chunk_width, projecting_element_f, field_cfg),
            None,
        ),
    };
    let table_len = subtable.len();
    let aggregated_multiplicities = prescribed_mults
        .as_deref()
        .unwrap_or(&proof.aggregated_multiplicities);
    assert_eq!(aggregated_multiplicities.len(), num_lookups);

    // ---- Step 1: Absorb agg multiplicities, sample β, α ----
    let mut buf = vec![0u8; F::Inner::NUM_BYTES];
    for agg in aggregated_multiplicities {
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
    //
    // How many leaves each tree's numerators turn on: a whole-column
    // lookup turns on every leaf it has, a selection only the cells it
    // names. The table side has to account for exactly that many.
    let leaves_read = |ell: usize| match selections {
        Some(sels) => sels[ell].len(),
        None => num_chunks * witness_len,
    };
    let combined_mults: Vec<F> = {
        let mut combined = vec![zero.clone(); table_len];
        for ell in 0..num_lookups {
            let alpha_ell = &alpha_powers[ell];
            let mut m_sum = zero.clone();
            for j in 0..table_len {
                let scaled = alpha_ell.clone() * &aggregated_multiplicities[ell][j];
                combined[j] = combined[j].clone() + &scaled;
                m_sum = m_sum + &aggregated_multiplicities[ell][j];
            }
            let expected = leaves_read(ell) as u64;
            if m_sum != F::from_with_cfg(expected, field_cfg) {
                return Err(GkrLogupError::MultiplicitySumMismatch { expected, got: 0 });
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
    assert_eq!(
        w_num_vars,
        n_vars + zinc_utils::log2(num_chunks.next_power_of_two()) as usize
    );
    let r_inner: Vec<F> = r_full[..n_vars].to_vec();
    let r_outer: Vec<F> = r_full[n_vars..].to_vec();

    // A single chunk descends on no outer variables at all, and the empty
    // product is one.
    let eq_at_outer = match r_outer.is_empty() {
        true => vec![one.clone()],
        false => build_eq_x_r_vec(&r_outer, field_cfg)?,
    };
    // How much of the outer cube the group's leaves cover. The rest is
    // padding, whose numerator is zero and denominator one, so it adds
    // `1 - covered` to every q and nothing to any p. Whenever the leaf
    // count is a power of two this is one and both corrections vanish.
    let covered: F = eq_at_outer[..num_chunks]
        .iter()
        .cloned()
        .fold(zero.clone(), |acc, eq| acc + &eq);
    let padding = one.clone() - &covered;

    // expected_p^(ell)(r) is the numerator MLE at the descent point: the
    // covered mass where every leaf is read, and Σ eq over the named cells
    // where a selection says which. The selection is declared, so the
    // verifier sums those eq terms itself -- nothing about it is
    // committed, and it costs the cells it names rather than the trace.
    for ell in 0..num_lookups {
        let expected_p = match selections {
            Some(sels) => sels[ell].iter().fold(zero.clone(), |acc, &(slot, row)| {
                let eq_row = eq_at_index(row as usize, &r_inner, &one);
                acc + &(eq_at_outer[slot as usize].clone() * &eq_row)
            }),
            None => covered.clone(),
        };
        if witness_result.expected_ps[ell] != expected_p {
            return Err(GkrLogupError::GkrLeafMismatch);
        }
    }

    // For q^(ell), reconstruct from the lifts of the columns the tree's
    // denominators read:
    //   expected_qs[ell] = β·covered - Σ_k eq_outer(k, r_outer) · ψ_a(c_k')
    //                       + padding
    for ell in 0..num_lookups {
        let lifts = &proof.chunk_lifts[if selections.is_some() { 0 } else { ell }];
        if lifts.len() != num_chunks {
            return Err(GkrLogupError::GkrLeafMismatch);
        }
        let mut psi_combined = zero.clone();
        for k in 0..num_chunks {
            let psi =
                eval_at_projecting_element::<F>(&lifts[k], projecting_element_f, field_cfg);
            psi_combined = psi_combined + &(eq_at_outer[k].clone() * &psi);
        }
        let expected_q_local = beta.clone() * &covered - &psi_combined + &padding;
        if expected_q_local != witness_result.expected_qs[ell] {
            return Err(GkrLogupError::GkrLeafMismatch);
        }
    }

    // ---- Step 6: Combine chunk lifts into parent polynomial claim ----
    let combined_polynomial: Vec<DynamicPolynomialF<F>> = match &meta.table_type {
        // A selection's chunks are whole columns, each standing for
        // itself, so the group's claim is one lift per column and there
        // is nothing to recombine.
        LookupTableType::Selected { .. } => proof.chunk_lifts[0].clone(),
        // A cell read as a number recombines to one field element by
        // place value; BitPoly chunks are coefficient blocks.
        LookupTableType::BitPoly { .. } => (0..num_lookups)
            .map(|ell| combine_chunks::<F>(&proof.chunk_lifts[ell], chunk_width, width, &zero))
            .collect(),
        _ => (0..num_lookups)
            .map(|ell| {
                combine_chunks_word::<F>(&proof.chunk_lifts[ell], chunk_width, field_cfg)
            })
            .collect(),
    };

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

/// `eq(index, r)` read off the index's bits rather than out of a table:
/// a verifier wanting a handful of these pays for those and not for the
/// whole cube. Agrees with `build_eq_x_r_vec`, whose entry `j` pairs
/// `r[nu]` with bit `nu` of `j`.
#[allow(clippy::arithmetic_side_effects)]
fn eq_at_index<F: PrimeField>(index: usize, r: &[F], one: &F) -> F {
    r.iter().enumerate().fold(one.clone(), |acc, (nu, r_nu)| {
        acc * &match (index >> nu) & 1 {
            1 => r_nu.clone(),
            _ => one.clone() - r_nu,
        }
    })
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
    type Inner = <F as Field>::Inner;

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
        let instance = IntLookupInstance::<'_, F, i64> {
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

    /// A column holding a permutation of 1..=9 and nothing else but pad
    /// is the multiset the table prescribes, so it proves and verifies,
    /// and the claim the verifier is left holding is the column's own
    /// multilinear evaluation.
    #[test]
    fn round_trip_prescribed_permutation() {
        let cfg = ();
        let n_vars = 4; // W = 16: nine values and seven pads.
        let cells: Vec<i64> = vec![4, 9, 2, 3, 5, 7, 8, 1, 6, 0, 0, 0, 0, 0, 0, 0];
        let parent = DenseMultilinearExtension::from_evaluations_vec(n_vars, cells.clone(), 0i64);

        let a: F = F::from(7u64);
        let instance = IntLookupInstance::<'_, F, i64> {
            parent_columns: vec![&parent],
            parent_column_indices: vec![0],
            table_type: LookupTableType::Prescribed { values: (1..=9).collect(), pad: 0 },
            projecting_element_f: &a,
            n_vars,
        };

        let mut p_ts = Blake3Transcript::new();
        let (proof, meta, prover_sub) =
            prove_group_prescribed::<F, i64>(&mut p_ts, &instance, &cfg).expect("prove");
        assert!(
            proof.aggregated_multiplicities.is_empty(),
            "a prescribed table's multiplicities are the verifier's own"
        );

        let mut v_ts = Blake3Transcript::new();
        let verifier_sub =
            verify_group::<F>(&mut v_ts, &proof, &meta, &a, &cfg).expect("verify");

        assert_eq!(prover_sub.r_inner, verifier_sub.r_inner);
        assert_eq!(prover_sub.combined_polynomial, verifier_sub.combined_polynomial);

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

    /// A 3x3 latin square laid down one row per column, over `rows` rows
    /// so every column has cells no selection names.
    fn latin_square(square: [[u32; 3]; 3], rows: usize) -> Vec<DenseMultilinearExtension<Inner>> {
        let n_vars = zinc_utils::log2(rows) as usize;
        let zero = F::from(0u32).inner().clone();
        square
            .iter()
            .map(|row| {
                let mut evals: Vec<Inner> =
                    row.iter().map(|v| F::from(*v).inner().clone()).collect();
                evals.resize(rows, zero.clone());
                DenseMultilinearExtension::from_evaluations_vec(n_vars, evals, zero.clone())
            })
            .collect()
    }

    /// Three row selections and three column selections over the three
    /// columns: the shape a sudoku's obligations take, at the size a unit
    /// test can read.
    fn latin_selections() -> Vec<Vec<(u32, u32)>> {
        let rows = (0..3u32).map(|r| (0..3u32).map(|p| (r, p)).collect());
        let columns = (0..3u32).map(|p| (0..3u32).map(|r| (r, p)).collect());
        rows.chain(columns).collect()
    }

    fn latin_table(selections: Vec<Vec<(u32, u32)>>) -> LookupTableType {
        LookupTableType::Selected { values: (1..=3).collect(), selections }
    }

    fn prove_latin(
        columns: &[DenseMultilinearExtension<Inner>],
        table_type: LookupTableType,
    ) -> Result<
        (GkrLogupGroupProof<F>, GkrLogupGroupMeta, GkrLogupGroupSubclaim<F>),
        GkrLogupError<F>,
    > {
        let instance = SelectedLookupInstance::<'_, F> {
            parent_columns: columns.iter().collect(),
            parent_column_indices: (0..columns.len()).collect(),
            table_type,
            n_vars: zinc_utils::log2(columns[0].evaluations.len()) as usize,
        };
        prove_group_selected::<F>(&mut Blake3Transcript::new(), &instance, &())
    }

    /// Six selections over three columns -- neither count a power of two --
    /// prove and verify as one group, and the claim the verifier is left
    /// holding is one lift per column rather than one per selection.
    #[test]
    fn round_trip_selected_latin_square() {
        let a: F = F::from(7u64);
        let columns = latin_square([[1, 2, 3], [2, 3, 1], [3, 1, 2]], 8);
        let (proof, meta, prover_sub) =
            prove_latin(&columns, latin_table(latin_selections())).expect("prove");
        assert_eq!(meta.num_lookups, 6);
        assert_eq!(proof.chunk_lifts.len(), 1, "every selection reads the same columns");
        assert!(proof.aggregated_multiplicities.is_empty());

        let mut v_ts = Blake3Transcript::new();
        let sub = verify_group::<F>(&mut v_ts, &proof, &meta, &a, &()).expect("verify");
        assert_eq!(prover_sub.combined_polynomial, sub.combined_polynomial);

        let eq = build_eq_x_r_vec(&sub.r_inner, &()).expect("eq");
        for (slot, column) in columns.iter().enumerate() {
            let expected = column.evaluations.iter().zip(eq.iter()).fold(
                F::from(0u64),
                |acc, (cell, eq_i)| {
                    acc + &(eq_i.clone() * &F::new_unchecked_with_cfg(cell.clone(), &()))
                },
            );
            assert_eq!(
                sub.combined_polynomial[slot],
                DynamicPolynomialF::new_trimmed(vec![expected])
            );
        }
    }

    /// A cell no selection names is genuinely unconstrained: the lookup
    /// says nothing about it, so a column carrying anything at all in its
    /// unnamed rows still proves and verifies.
    #[test]
    fn an_unnamed_cell_is_unconstrained() {
        let a: F = F::from(7u64);
        let mut columns = latin_square([[1, 2, 3], [2, 3, 1], [3, 1, 2]], 8);
        columns[0].evaluations[5] = F::from(4242u64).inner().clone();
        let (proof, meta, _) =
            prove_latin(&columns, latin_table(latin_selections())).expect("prove");
        let mut v_ts = Blake3Transcript::new();
        verify_group::<F>(&mut v_ts, &proof, &meta, &a, &()).expect("verify");
    }

    /// The teeth: a value slid out of a named cell into an unnamed one of
    /// the same column. Every cell of the column is still a value of the
    /// table, and the selection is simply one short -- there is no pad
    /// entry for the missing one to hide behind.
    #[test]
    fn a_value_slid_out_of_a_selection_is_refused() {
        let a: F = F::from(7u64);
        let mut columns = latin_square([[1, 2, 3], [2, 3, 1], [3, 1, 2]], 8);
        columns[0].evaluations[5] = columns[0].evaluations[2].clone();
        columns[0].evaluations[2] = F::from(0u64).inner().clone();
        let (proof, meta, _) =
            prove_latin(&columns, latin_table(latin_selections())).expect("prove");
        let mut v_ts = Blake3Transcript::new();
        let res = verify_group::<F>(&mut v_ts, &proof, &meta, &a, &());
        assert!(
            matches!(res, Err(GkrLogupError::GkrRootMismatch)),
            "a selection one value short must fail the LogUp identity, got {res:?}"
        );
    }

    /// A selection naming a cell the group's columns do not have indexes
    /// nothing at all, so it is refused where the reason is legible.
    #[test]
    fn a_selection_outside_the_columns_is_refused() {
        let a: F = F::from(7u64);
        let columns = latin_square([[1, 2, 3], [2, 3, 1], [3, 1, 2]], 8);
        for bad in [(3u32, 0u32), (0, 8)] {
            let table = latin_table(vec![vec![(0, 0), (0, 1), bad]]);
            assert!(matches!(
                prove_latin(&columns, table),
                Err(GkrLogupError::MalformedSelection)
            ));
        }
    }

    /// The teeth: every cell is in the table and the column is still the
    /// wrong multiset -- a nine twice and no eight. The prover can build
    /// this proof, and the verifier, counting the table side itself, must
    /// reject it.
    #[test]
    fn a_repeated_value_is_not_the_prescribed_multiset() {
        let cfg = ();
        let n_vars = 4;
        let cells: Vec<i64> = vec![4, 9, 2, 3, 5, 7, 9, 1, 6, 0, 0, 0, 0, 0, 0, 0];
        let parent = DenseMultilinearExtension::from_evaluations_vec(n_vars, cells, 0i64);
        let a: F = F::from(7u64);
        let instance = IntLookupInstance::<'_, F, i64> {
            parent_columns: vec![&parent],
            parent_column_indices: vec![0],
            table_type: LookupTableType::Prescribed { values: (1..=9).collect(), pad: 0 },
            projecting_element_f: &a,
            n_vars,
        };

        let mut p_ts = Blake3Transcript::new();
        let (proof, meta, _) =
            prove_group_prescribed::<F, i64>(&mut p_ts, &instance, &cfg).expect("prove");
        let mut v_ts = Blake3Transcript::new();
        let res = verify_group::<F>(&mut v_ts, &proof, &meta, &a, &cfg);
        assert!(
            matches!(res, Err(GkrLogupError::GkrRootMismatch)),
            "a repeated value must fail the LogUp identity itself, got {res:?}"
        );
    }

    /// A cell the table never names has no place in it, so the column is
    /// refused where the reason is legible rather than proved against a
    /// table it misses.
    #[test]
    fn a_value_outside_the_prescribed_table_is_refused() {
        let cfg = ();
        let n_vars = 4;
        let a: F = F::from(7u64);
        for bad in [10i64, -1i64] {
            let mut cells: Vec<i64> = vec![4, 9, 2, 3, 5, 7, 8, 1, 6, 0, 0, 0, 0, 0, 0, 0];
            cells[0] = bad;
            let parent = DenseMultilinearExtension::from_evaluations_vec(n_vars, cells, 0i64);
            let instance = IntLookupInstance::<'_, F, i64> {
                parent_columns: vec![&parent],
                parent_column_indices: vec![0],
                table_type: LookupTableType::Prescribed { values: (1..=9).collect(), pad: 0 },
                projecting_element_f: &a,
                n_vars,
            };
            let mut ts = Blake3Transcript::new();
            assert!(
                prove_group_prescribed::<F, i64>(&mut ts, &instance, &cfg).is_err(),
                "a cell of {bad} is not in the prescribed table and must not prove"
            );
        }
    }

    /// A pad that is one of the values would stand for two multiplicities
    /// at once, and values that outnumber the rows cannot all be laid
    /// down: neither table is a multiset a column could hold.
    #[test]
    fn a_table_that_is_not_a_multiset_is_refused() {
        let cfg = ();
        let n_vars = 2; // W = 4.
        let parent = DenseMultilinearExtension::from_evaluations_vec(n_vars, vec![1i64; 4], 0i64);
        let a: F = F::from(7u64);
        for table_type in [
            LookupTableType::Prescribed { values: vec![1, 2, 3], pad: 2 },
            LookupTableType::Prescribed { values: (1..=9).collect(), pad: 0 },
        ] {
            let instance = IntLookupInstance::<'_, F, i64> {
                parent_columns: vec![&parent],
                parent_column_indices: vec![0],
                table_type,
                projecting_element_f: &a,
                n_vars,
            };
            let mut ts = Blake3Transcript::new();
            assert!(prove_group_prescribed::<F, i64>(&mut ts, &instance, &cfg).is_err());
        }
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
        let instance = IntLookupInstance::<'_, F, i64> {
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
            let instance = IntLookupInstance::<'_, F, i64> {
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
