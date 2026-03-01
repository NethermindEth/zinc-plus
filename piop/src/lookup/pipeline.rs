//! Pipeline integration for batched decomposed LogUp.
//!
//! This module provides [`prove_batched_lookup`] and
//! [`verify_batched_lookup`] — high-level functions that the prover /
//! verifier pipeline calls to enforce typing constraints on trace columns.
//!
//! ## Usage
//!
//! The caller supplies:
//! - A list of [`LookupColumnSpec`]s describing which columns need lookup
//!   verification and which table type (BitPoly or Word) they should be
//!   checked against.
//! - The projected trace columns as `&[Vec<F>]` (one `Vec<F>` per column,
//!   each element is a field element obtained by evaluating the binary
//!   polynomial / integer at the projecting element).
//! - A Fiat-Shamir transcript (shared with the rest of the pipeline).
//!
//! Columns with the same table type are **batched** into a single
//! [`BatchedDecompLogupProtocol`] instance, amortising the sumcheck cost.
//!
//! ## Example table types
//!
//! | Type | Set | Table size | Decomposition |
//! |------|-----|-----------|---------------|
/// | `BitPoly { width: 32 }` | `{0,1}^{<32}[X]` | 2^32 → 4 × 2^8 | K=4, chunk_width=8 |
/// | `Word { width: 32 }` | `[0, 2^32 − 1]` | 2^32 → 4 × 2^8 | K=4, chunk_width=8 |

use crypto_primitives::{FromPrimitiveWithConfig, PrimeField};
use num_traits::Zero;
use zinc_transcript::traits::{ConstTranscribable, Transcript};
use zinc_utils::inner_transparent_field::InnerTransparentField;

use super::batched_decomposition::BatchedDecompLogupProtocol;
use super::gkr_batched_decomposition::GkrBatchedDecompLogupProtocol;
use super::structs::{
    BatchedDecompLogupProof,
    BatchedDecompLogupVerifierSubClaim, BatchedDecompLookupInstance,
    GkrBatchedDecompLogupProof, GkrBatchedDecompLogupProverState,
    GkrBatchedDecompLogupVerifierSubClaim,
    LookupColumnSpec, LookupError, LookupGroup, LookupTableType,
    group_lookup_specs,
};
use super::tables::{
    bitpoly_shift,
    decompose_bitpoly_column, decompose_raw_indices_to_chunks, decompose_word_column,
    generate_bitpoly_table, generate_word_table,
    word_shift,
};

// ── Constants ───────────────────────────────────────────────────────────────

/// Default chunk width for decomposition.
///
/// Tables of width ≤ `DECOMP_THRESHOLD` are used directly (K=1).
/// Tables of width > `DECOMP_THRESHOLD` are split into chunks of this width.
///
/// Using chunk_width=8 gives sub-tables of size 2^8 = 256, keeping memory
/// usage reasonable even with many lookup columns (e.g. 20+ columns would
/// need ~40 MB of multiplicities at chunk_width=16 vs ~160 KB at 8).
const DEFAULT_CHUNK_WIDTH: usize = 8;

/// Below this width, no decomposition is needed — the full table fits
/// comfortably in memory and we use a single-chunk "decomposition" (K=1).
const DECOMP_THRESHOLD: usize = 8;

// ── Proof container ─────────────────────────────────────────────────────────

/// Complete lookup proof for the pipeline.
///
/// Contains one [`BatchedDecompLogupProof`] per lookup group (i.e. per
/// distinct table type). The groups are ordered deterministically.
#[derive(Clone, Debug)]
pub struct PipelineLookupProof<F: PrimeField> {
    /// Per-group proofs, in the same order as the groups returned by
    /// [`group_lookup_specs`].
    pub group_proofs: Vec<BatchedDecompLogupProof<F>>,
    /// Per-group metadata needed by the verifier: (table_type, column_count, witness_len).
    pub group_meta: Vec<LookupGroupMeta>,
}

/// Metadata for one lookup group (stored in the proof so the verifier
/// can reconstruct tables and shifts).
#[derive(Clone, Debug)]
pub struct LookupGroupMeta {
    /// The table type for this group.
    pub table_type: LookupTableType,
    /// Number of columns (= L, the batch size).
    pub num_columns: usize,
    /// Length of each witness vector (= number of rows in the trace).
    pub witness_len: usize,
    /// Original trace column indices for each witness in the group.
    ///
    /// `original_column_indices[ℓ]` is the index into the full trace
    /// (and therefore into the CPR `up_evals`) of the ℓ-th lookup column
    /// in this group.  The verifier uses this to extract parent column
    /// evaluations for the decomposition consistency check.
    pub original_column_indices: Vec<usize>,
}

/// Prover state returned after running the lookup pipeline step.
#[derive(Clone, Debug)]
pub struct PipelineLookupProverState<F: PrimeField> {
    /// Evaluation points from each group's sumcheck.
    pub evaluation_points: Vec<Vec<F>>,
}

// ── GKR proof container ─────────────────────────────────────────────────────

/// Complete GKR-based lookup proof for the pipeline.
///
/// Contains one [`GkrBatchedDecompLogupProof`] per lookup group.
/// Chunks are NOT included (assumed committed via PCS).
#[derive(Clone, Debug)]
pub struct GkrPipelineLookupProof<F: PrimeField> {
    /// Per-group GKR proofs.
    pub group_proofs: Vec<GkrBatchedDecompLogupProof<F>>,
    /// Per-group metadata.
    pub group_meta: Vec<LookupGroupMeta>,
}

/// Prover state returned after running the GKR lookup pipeline step.
#[derive(Clone, Debug)]
pub struct GkrPipelineLookupProverState<F: PrimeField> {
    /// Per-group prover states (contains evaluation points, α, β, etc.).
    pub group_states: Vec<GkrBatchedDecompLogupProverState<F>>,
}

// ── Prover ──────────────────────────────────────────────────────────────────

/// Run the batched decomposed LogUp prover for all specified lookup columns.
///
/// Groups columns by table type, generates the decomposed instance for
/// each group, and calls [`BatchedDecompLogupProtocol::prove_as_subprotocol`].
///
/// # Arguments
///
/// - `transcript`: shared Fiat-Shamir transcript.
/// - `columns`: projected trace columns — `columns[col_idx][row]` is the
///   field element for column `col_idx`, row `row`. All columns must have
///   the same length.
/// - `specs`: lookup specifications (which columns, which table types).
/// - `projecting_element`: the element `a` used to project BinaryPoly types
///   to the field. Only needed for `BitPoly` table types; for `Word` types
///   this is unused.
/// - `field_cfg`: PIOP field configuration.
///
/// # Returns
///
/// `(PipelineLookupProof, PipelineLookupProverState)` on success.
#[allow(clippy::arithmetic_side_effects, clippy::too_many_arguments)]
pub fn prove_batched_lookup<F>(
    transcript: &mut impl Transcript,
    columns: &[Vec<F>],
    specs: &[LookupColumnSpec],
    projecting_element: &F,
    field_cfg: &F::Config,
) -> Result<(PipelineLookupProof<F>, PipelineLookupProverState<F>), LookupError<F>>
where
    F: InnerTransparentField + FromPrimitiveWithConfig + Send + Sync,
    F::Inner: ConstTranscribable + Zero + Default + Send + Sync,
    F::Config: Sync,
{
    let groups = group_lookup_specs(specs);
    let mut group_proofs = Vec::with_capacity(groups.len());
    let mut group_meta = Vec::with_capacity(groups.len());
    let mut eval_points = Vec::with_capacity(groups.len());

    for group in &groups {
        let instance = build_lookup_instance(
            columns,
            group,
            projecting_element,
            field_cfg,
        )?;

        let witness_len = instance.witnesses[0].len();

        let (proof, state) =
            BatchedDecompLogupProtocol::<F>::prove_as_subprotocol(
                transcript,
                &instance,
                field_cfg,
            )?;

        group_proofs.push(proof);
        group_meta.push(LookupGroupMeta {
            table_type: group.table_type.clone(),
            num_columns: group.column_indices.len(),
            witness_len,
            original_column_indices: group.column_indices.clone(),
        });
        eval_points.push(state.evaluation_point);
    }

    Ok((
        PipelineLookupProof {
            group_proofs,
            group_meta,
        },
        PipelineLookupProverState {
            evaluation_points: eval_points,
        },
    ))
}

/// Like [`prove_batched_lookup`], but accepts precomputed raw integer
/// indices for each column instead of reverse-mapping from projected
/// field elements.
///
/// This avoids building the full `2^width` lookup table in
/// [`decompose_bitpoly_column`] / [`decompose_word_column`], which
/// would be prohibitive for large widths (e.g. 32, where the table
/// has 4 billion entries).
///
/// # Arguments
///
/// - `columns`: projected field columns (same as [`prove_batched_lookup`]).
/// - `raw_indices`: for each column `i`, `raw_indices[i][row]` is the
///   integer index in `[0, 2^width)` for that cell.
///
/// All other arguments are identical to [`prove_batched_lookup`].
#[allow(clippy::arithmetic_side_effects, clippy::too_many_arguments)]
pub fn prove_batched_lookup_with_indices<F>(
    transcript: &mut impl Transcript,
    columns: &[Vec<F>],
    raw_indices: &[Vec<usize>],
    specs: &[LookupColumnSpec],
    projecting_element: &F,
    field_cfg: &F::Config,
) -> Result<(PipelineLookupProof<F>, PipelineLookupProverState<F>), LookupError<F>>
where
    F: InnerTransparentField + FromPrimitiveWithConfig + Send + Sync,
    F::Inner: ConstTranscribable + Zero + Default + Send + Sync,
    F::Config: Sync,
{
    let groups = group_lookup_specs(specs);
    let mut group_proofs = Vec::with_capacity(groups.len());
    let mut group_meta = Vec::with_capacity(groups.len());
    let mut eval_points = Vec::with_capacity(groups.len());

    for group in &groups {
        let instance = build_lookup_instance_from_indices(
            columns,
            raw_indices,
            group,
            projecting_element,
            field_cfg,
        )?;

        let witness_len = instance.witnesses[0].len();

        let (proof, state) =
            BatchedDecompLogupProtocol::<F>::prove_as_subprotocol(
                transcript,
                &instance,
                field_cfg,
            )?;

        group_proofs.push(proof);
        group_meta.push(LookupGroupMeta {
            table_type: group.table_type.clone(),
            num_columns: group.column_indices.len(),
            witness_len,
            original_column_indices: group.column_indices.clone(),
        });
        eval_points.push(state.evaluation_point);
    }

    Ok((
        PipelineLookupProof {
            group_proofs,
            group_meta,
        },
        PipelineLookupProverState {
            evaluation_points: eval_points,
        },
    ))
}

/// Build a [`BatchedDecompLookupInstance`] using precomputed raw integer
/// indices, avoiding the full-table reverse lookup.
///
/// Public alias for pipeline integration (used by batched CPR+lookup).
#[allow(clippy::arithmetic_side_effects)]
pub fn build_lookup_instance_from_indices_pub<F>(
    columns: &[Vec<F>],
    raw_indices: &[Vec<usize>],
    group: &LookupGroup,
    projecting_element: &F,
    field_cfg: &F::Config,
) -> Result<BatchedDecompLookupInstance<F>, LookupError<F>>
where
    F: InnerTransparentField + FromPrimitiveWithConfig + Send + Sync,
    F::Config: Sync,
{
    build_lookup_instance_from_indices(columns, raw_indices, group, projecting_element, field_cfg)
}

#[allow(clippy::arithmetic_side_effects)]
fn build_lookup_instance_from_indices<F>(
    columns: &[Vec<F>],
    raw_indices: &[Vec<usize>],
    group: &LookupGroup,
    projecting_element: &F,
    field_cfg: &F::Config,
) -> Result<BatchedDecompLookupInstance<F>, LookupError<F>>
where
    F: InnerTransparentField + FromPrimitiveWithConfig + Send + Sync,
    F::Config: Sync,
{
    let (subtable, shifts) = generate_table_and_shifts(
        &group.table_type,
        projecting_element,
        field_cfg,
    );

    let chunk_width = compute_chunk_width(&group.table_type);
    let total_width = table_type_width(&group.table_type);

    let mut witnesses = Vec::with_capacity(group.column_indices.len());
    let mut all_chunks = Vec::with_capacity(group.column_indices.len());

    for &col_idx in &group.column_indices {
        witnesses.push(columns[col_idx].clone());

        let chunks = decompose_raw_indices_to_chunks(
            &raw_indices[col_idx],
            total_width,
            chunk_width,
            &subtable,
        );
        all_chunks.push(chunks);
    }

    Ok(BatchedDecompLookupInstance {
        witnesses,
        subtable,
        shifts,
        chunks: all_chunks,
    })
}

// ── GKR Prover ──────────────────────────────────────────────────────────────

/// Run the GKR batched decomposed LogUp prover for all lookup columns.
///
/// Like [`prove_batched_lookup_with_indices`] but uses the GKR fractional
/// sumcheck variant. Chunks are NOT included in the proof (assumed
/// committed via PCS).
#[allow(clippy::arithmetic_side_effects, clippy::too_many_arguments)]
pub fn prove_gkr_batched_lookup_with_indices<F>(
    transcript: &mut impl Transcript,
    columns: &[Vec<F>],
    raw_indices: &[Vec<usize>],
    specs: &[LookupColumnSpec],
    projecting_element: &F,
    field_cfg: &F::Config,
) -> Result<(GkrPipelineLookupProof<F>, GkrPipelineLookupProverState<F>), LookupError<F>>
where
    F: InnerTransparentField + FromPrimitiveWithConfig + Send + Sync,
    F::Inner: ConstTranscribable + Zero + Default + Send + Sync,
    F::Config: Sync,
{
    let groups = group_lookup_specs(specs);
    let mut group_proofs = Vec::with_capacity(groups.len());
    let mut group_meta = Vec::with_capacity(groups.len());
    let mut group_states = Vec::with_capacity(groups.len());

    for group in &groups {
        let instance = build_lookup_instance_from_indices(
            columns,
            raw_indices,
            group,
            projecting_element,
            field_cfg,
        )?;

        let witness_len = instance.witnesses[0].len();

        let (proof, state) =
            GkrBatchedDecompLogupProtocol::<F>::prove_as_subprotocol(
                transcript,
                &instance,
                field_cfg,
            )?;

        group_proofs.push(proof);
        group_meta.push(LookupGroupMeta {
            table_type: group.table_type.clone(),
            num_columns: group.column_indices.len(),
            witness_len,
            original_column_indices: group.column_indices.clone(),
        });
        group_states.push(state);
    }

    Ok((
        GkrPipelineLookupProof {
            group_proofs,
            group_meta,
        },
        GkrPipelineLookupProverState {
            group_states,
        },
    ))
}

// ── Verifier ────────────────────────────────────────────────────────────────

/// Run the batched decomposed LogUp verifier for all lookup groups.
///
/// Reconstructs the sub-tables and shifts from the group metadata,
/// then calls [`BatchedDecompLogupProtocol::verify_as_subprotocol`] for
/// each group.
#[allow(clippy::arithmetic_side_effects, clippy::too_many_arguments)]
pub fn verify_batched_lookup<F>(
    transcript: &mut impl Transcript,
    proof: &PipelineLookupProof<F>,
    projecting_element: &F,
    field_cfg: &F::Config,
) -> Result<Vec<BatchedDecompLogupVerifierSubClaim<F>>, LookupError<F>>
where
    F: InnerTransparentField + FromPrimitiveWithConfig + Send + Sync,
    F::Inner: ConstTranscribable + Zero + Default + Send + Sync,
    F::Config: Sync,
{
    let mut subclaims = Vec::with_capacity(proof.group_proofs.len());

    for (group_proof, meta) in proof.group_proofs.iter().zip(proof.group_meta.iter()) {
        let (subtable, shifts) = generate_table_and_shifts(
            &meta.table_type,
            projecting_element,
            field_cfg,
        );

        let subclaim = BatchedDecompLogupProtocol::<F>::verify_as_subprotocol(
            transcript,
            group_proof,
            &subtable,
            &shifts,
            meta.num_columns,
            meta.witness_len,
            field_cfg,
        )?;

        subclaims.push(subclaim);
    }

    Ok(subclaims)
}

/// Run the GKR batched decomposed LogUp verifier for all lookup groups.
///
/// Reconstructs the sub-tables and shifts from the group metadata,
/// then calls [`GkrBatchedDecompLogupProtocol::verify_as_subprotocol`]
/// for each group.
#[allow(clippy::arithmetic_side_effects, clippy::too_many_arguments)]
pub fn verify_gkr_batched_lookup<F>(
    transcript: &mut impl Transcript,
    proof: &GkrPipelineLookupProof<F>,
    projecting_element: &F,
    field_cfg: &F::Config,
) -> Result<Vec<GkrBatchedDecompLogupVerifierSubClaim<F>>, LookupError<F>>
where
    F: InnerTransparentField + FromPrimitiveWithConfig + Send + Sync,
    F::Inner: ConstTranscribable + Zero + Default + Send + Sync,
    F::Config: Sync,
{
    let mut subclaims = Vec::with_capacity(proof.group_proofs.len());

    for (group_proof, meta) in proof.group_proofs.iter().zip(proof.group_meta.iter()) {
        let (subtable, shifts) = generate_table_and_shifts(
            &meta.table_type,
            projecting_element,
            field_cfg,
        );

        let subclaim = GkrBatchedDecompLogupProtocol::<F>::verify_as_subprotocol(
            transcript,
            group_proof,
            &subtable,
            &shifts,
            meta.num_columns,
            meta.witness_len,
            field_cfg,
        )?;

        subclaims.push(subclaim);
    }

    Ok(subclaims)
}

// ── Internal helpers ────────────────────────────────────────────────────────

/// Build a [`BatchedDecompLookupInstance`] for a single lookup group.
#[allow(clippy::arithmetic_side_effects)]
fn build_lookup_instance<F>(
    columns: &[Vec<F>],
    group: &LookupGroup,
    projecting_element: &F,
    field_cfg: &F::Config,
) -> Result<BatchedDecompLookupInstance<F>, LookupError<F>>
where
    F: InnerTransparentField + FromPrimitiveWithConfig + Send + Sync,
    F::Config: Sync,
{
    let (subtable, shifts) = generate_table_and_shifts(
        &group.table_type,
        projecting_element,
        field_cfg,
    );

    let chunk_width = compute_chunk_width(&group.table_type);
    let _total_width = table_type_width(&group.table_type);

    let mut witnesses = Vec::with_capacity(group.column_indices.len());
    let mut all_chunks = Vec::with_capacity(group.column_indices.len());

    for &col_idx in &group.column_indices {
        let witness = &columns[col_idx];
        witnesses.push(witness.clone());

        let chunks = match &group.table_type {
            LookupTableType::BitPoly { width } => {
                decompose_bitpoly_column(
                    witness,
                    *width,
                    chunk_width,
                    projecting_element,
                    &subtable,
                )
                .ok_or(LookupError::WitnessNotInTable)?
            }
            LookupTableType::Word { width } => {
                decompose_word_column(
                    witness,
                    *width,
                    chunk_width,
                    field_cfg,
                )
                .ok_or(LookupError::WitnessNotInTable)?
            }
        };
        all_chunks.push(chunks);
    }

    Ok(BatchedDecompLookupInstance {
        witnesses,
        subtable,
        shifts,
        chunks: all_chunks,
    })
}

/// Generate the sub-table and shift factors for a given table type.
///
/// Public for pipeline integration.
#[allow(clippy::arithmetic_side_effects)]
pub fn generate_table_and_shifts<F>(
    table_type: &LookupTableType,
    projecting_element: &F,
    field_cfg: &F::Config,
) -> (Vec<F>, Vec<F>)
where
    F: PrimeField + FromPrimitiveWithConfig + Send + Sync,
    F::Config: Sync,
{
    let chunk_width = compute_chunk_width(table_type);
    let total_width = table_type_width(table_type);
    let num_chunks = total_width / chunk_width;

    match table_type {
        LookupTableType::BitPoly { .. } => {
            let subtable = generate_bitpoly_table(chunk_width, projecting_element, field_cfg);
            let shifts: Vec<F> = (0..num_chunks)
                .map(|k| bitpoly_shift(k * chunk_width, projecting_element))
                .collect();
            (subtable, shifts)
        }
        LookupTableType::Word { .. } => {
            let subtable = generate_word_table(chunk_width, field_cfg);
            let shifts: Vec<F> = (0..num_chunks)
                .map(|k| word_shift(k * chunk_width, field_cfg))
                .collect();
            (subtable, shifts)
        }
    }
}

/// Compute the chunk width for a given table type.
fn compute_chunk_width(table_type: &LookupTableType) -> usize {
    let total = table_type_width(table_type);
    if total <= DECOMP_THRESHOLD {
        total // No decomposition needed; K=1
    } else {
        DEFAULT_CHUNK_WIDTH
    }
}

/// Extract the total width from a table type.
fn table_type_width(table_type: &LookupTableType) -> usize {
    match table_type {
        LookupTableType::BitPoly { width } => *width,
        LookupTableType::Word { width } => *width,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lookup::tables::generate_bitpoly_table;
    use crypto_bigint::{U128, const_monty_params};
    use crypto_primitives::crypto_bigint_const_monty::ConstMontyField;
    use zinc_transcript::KeccakTranscript;

    const N: usize = 2;
    const_monty_params!(TestParams, U128, "00000000b933426489189cb5b47d567f");
    type F = ConstMontyField<TestParams, N>;

    /// Helper: generate a column with valid BitPoly entries.
    fn make_bitpoly_column(
        indices: &[usize],
        width: usize,
        projecting_element: &F,
    ) -> Vec<F> {
        let table = generate_bitpoly_table(width, projecting_element, &());
        indices.iter().map(|&i| table[i].clone()).collect()
    }

    /// Helper: generate a column with valid Word entries.
    fn make_word_column(values: &[u32]) -> Vec<F> {
        values.iter().map(|&v| F::from(v)).collect()
    }

    #[test]
    fn pipeline_single_bitpoly_column() {
        let a = F::from(3u32);
        // BitPoly(4) — small enough for no decomposition
        let col = make_bitpoly_column(&[0, 3, 5, 15], 4, &a);

        let specs = vec![LookupColumnSpec {
            column_index: 0,
            table_type: LookupTableType::BitPoly { width: 4 },
        }];

        let mut pt = KeccakTranscript::new();
        let (proof, _state) = prove_batched_lookup(
            &mut pt,
            &[col],
            &specs,
            &a,
            &(),
        )
        .expect("prover should succeed");

        assert_eq!(proof.group_proofs.len(), 1);

        let mut vt = KeccakTranscript::new();
        let subclaims = verify_batched_lookup(
            &mut vt,
            &proof,
            &a,
            &(),
        )
        .expect("verifier should accept");

        assert_eq!(subclaims.len(), 1);
    }

    #[test]
    fn pipeline_batched_bitpoly_columns() {
        let a = F::from(3u32);
        // Three columns all in BitPoly(4)
        let col0 = make_bitpoly_column(&[0, 3, 5, 15], 4, &a);
        let col1 = make_bitpoly_column(&[1, 2, 7, 10], 4, &a);
        let col2 = make_bitpoly_column(&[15, 15, 0, 8], 4, &a);

        let specs = vec![
            LookupColumnSpec { column_index: 0, table_type: LookupTableType::BitPoly { width: 4 } },
            LookupColumnSpec { column_index: 1, table_type: LookupTableType::BitPoly { width: 4 } },
            LookupColumnSpec { column_index: 2, table_type: LookupTableType::BitPoly { width: 4 } },
        ];

        let mut pt = KeccakTranscript::new();
        let (proof, _) = prove_batched_lookup(
            &mut pt,
            &[col0, col1, col2],
            &specs,
            &a,
            &(),
        )
        .expect("prover should succeed");

        // All 3 columns have the same table type → single group
        assert_eq!(proof.group_proofs.len(), 1);
        assert_eq!(proof.group_meta[0].num_columns, 3);

        let mut vt = KeccakTranscript::new();
        verify_batched_lookup(&mut vt, &proof, &a, &())
            .expect("verifier should accept");
    }

    #[test]
    fn pipeline_word_column() {
        let a = F::from(3u32); // Not used for Word, but required
        // Word(8) — values in [0, 255]
        let col = make_word_column(&[0, 42, 127, 255]);

        let specs = vec![LookupColumnSpec {
            column_index: 0,
            table_type: LookupTableType::Word { width: 8 },
        }];

        let mut pt = KeccakTranscript::new();
        let (proof, _) = prove_batched_lookup(
            &mut pt,
            &[col],
            &specs,
            &a,
            &(),
        )
        .expect("prover should succeed");

        let mut vt = KeccakTranscript::new();
        verify_batched_lookup(&mut vt, &proof, &a, &())
            .expect("verifier should accept");
    }

    #[test]
    fn pipeline_mixed_table_types() {
        let a = F::from(3u32);
        // col0: BitPoly(4), col1: Word(8), col2: BitPoly(4)
        let col0 = make_bitpoly_column(&[0, 3, 5, 15], 4, &a);
        let col1 = make_word_column(&[0, 42, 127, 255]);
        let col2 = make_bitpoly_column(&[1, 2, 7, 10], 4, &a);

        let specs = vec![
            LookupColumnSpec { column_index: 0, table_type: LookupTableType::BitPoly { width: 4 } },
            LookupColumnSpec { column_index: 1, table_type: LookupTableType::Word { width: 8 } },
            LookupColumnSpec { column_index: 2, table_type: LookupTableType::BitPoly { width: 4 } },
        ];

        let mut pt = KeccakTranscript::new();
        let (proof, _) = prove_batched_lookup(
            &mut pt,
            &[col0, col1, col2],
            &specs,
            &a,
            &(),
        )
        .expect("prover should succeed");

        // Two groups: BitPoly(4) with 2 columns, Word(8) with 1 column
        assert_eq!(proof.group_proofs.len(), 2);

        let mut vt = KeccakTranscript::new();
        verify_batched_lookup(&mut vt, &proof, &a, &())
            .expect("verifier should accept");
    }

    #[test]
    fn pipeline_invalid_entry_fails() {
        let a = F::from(3u32);
        // Put an entry that's not in BitPoly(4) table
        let mut col = make_bitpoly_column(&[0, 3, 5, 15], 4, &a);
        col[2] = F::from(999u32); // Invalid

        let specs = vec![LookupColumnSpec {
            column_index: 0,
            table_type: LookupTableType::BitPoly { width: 4 },
        }];

        let mut pt = KeccakTranscript::new();
        let result = prove_batched_lookup(
            &mut pt,
            &[col],
            &specs,
            &a,
            &(),
        );

        assert!(result.is_err());
    }
}
