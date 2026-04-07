//! Data structures for the lookup protocol.
//!
//! Proof types, prover/verifier intermediates, instance types, and error
//! definitions live here. Specification types (`LookupTableType`,
//! `LookupColumnSpec`) live in `zinc-uair` and are re-exported from the
//! parent module.

use std::marker::PhantomData;
use thiserror::Error;
use zinc_uair::LookupTableType;

use crypto_primitives::PrimeField;
use zinc_poly::mle::DenseMultilinearExtension;

use crate::CombFn;

// ---------------------------------------------------------------------------
// Instance types (input to the lookup protocols)
// ---------------------------------------------------------------------------

/// Single-witness decomposition lookup instance (Phase 4a).
///
/// The witness is decomposed as:
/// `w[i] = shifts[0]*chunks[0][i] + … + shifts[K-1]*chunks[K-1][i]`
///
/// All chunks look up into the same sub-table.
#[derive(Clone, Debug)]
pub struct DecompLookupInstance<F> {
    /// The witness column (projected trace column).
    pub witness: Vec<F>,
    /// The sub-table entries (e.g. projected BitPoly(8) or Word(8)).
    pub subtable: Vec<F>,
    /// Shift factors, one per chunk.
    /// For `BitPoly(32)` with 4 chunks of width 8: `[1, a^8, a^16, a^24]`.
    pub shifts: Vec<F>,
    /// Precomputed decomposition chunks.
    /// `chunks[k][i]` is the k-th chunk of witness entry `i`.
    pub chunks: Vec<Vec<F>>,
}

/// Batched decomposition lookup instance (Phase 4b).
///
/// L witness vectors (all the same length) all look up into the same
/// decomposed table. Shifts and sub-table are shared; each witness
/// has its own chunk decomposition.
///
/// `witnesses[l][i] = Σ_k shifts[k] · chunks[l][k][i]`
#[derive(Clone, Debug)]
pub struct BatchedDecompLookupInstance<F> {
    /// L witness vectors, each of the same length.
    pub witnesses: Vec<Vec<F>>,
    /// The shared sub-table entries.
    pub subtable: Vec<F>,
    /// Shift factors (K entries, same for every witness).
    pub shifts: Vec<F>,
    /// Per-witness chunk decompositions.
    /// `chunks[l][k][i]` = k-th chunk of the l-th witness, entry i.
    pub chunks: Vec<Vec<Vec<F>>>,
}

// ---------------------------------------------------------------------------
// Per-group proof (BatchedDecompLogup)
// ---------------------------------------------------------------------------

/// Proof for one lookup group (columns sharing the same table type).
///
/// Does **not** contain a sumcheck proof — the sumcheck is shared via
/// the protocol-level multi-degree sumcheck (Phase 2a). This struct
/// carries only the auxiliary vectors the verifier needs to reconstruct
/// evaluations at the shared point.
///
/// Chunk vectors are **not** included — the verifier reconstructs them
/// from the inverse witnesses: `c_k[j] = β − 1/u_k[j]`. Soundness
/// follows from the PCS commitment binding the parent column.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchedDecompLogupProof<F> {
    /// Per-witness aggregated multiplicity vectors:
    /// `aggregated_multiplicities[l][j] = Σ_k m_k^(l)[j]`.
    pub aggregated_multiplicities: Vec<Vec<F>>,
    /// Per-witness per-chunk inverse witness vectors:
    /// `chunk_inverse_witnesses[l][k][i] = 1 / (β − chunk[l][k][i])`.
    pub chunk_inverse_witnesses: Vec<Vec<Vec<F>>>,
    /// Shared inverse table vector: `inverse_table[j] = 1 / (β − T[j])`.
    pub inverse_table: Vec<F>,
}

// ---------------------------------------------------------------------------
// Per-group metadata (carried in the proof for the verifier)
// ---------------------------------------------------------------------------

/// Describes how a lookup witness column was derived from the trace.
///
/// Carried in [`LookupGroupMeta`] so the verifier can reconstruct the
/// parent evaluation without re-receiving the lookup specs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LookupWitnessSource {
    /// Standard column lookup: parent eval = `up_evals[column_index]`.
    Column {
        /// Original trace column index.
        column_index: usize,
    },
    /// Affine-combination lookup: parent eval = `Σ coeff·up_evals[col] +
    /// offset`.
    Affine {
        /// `(column_index, coefficient)` pairs.
        terms: Vec<(usize, i64)>,
        /// Constant bit-polynomial offset encoded as a u32 bit pattern.
        constant_offset_bits: u32,
    },
}

/// Per-group metadata stored in the proof so the verifier can reconstruct
/// tables and column layout without being passed the original lookup specs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LookupGroupMeta {
    /// Table type for this group (determines subtable generation).
    pub table_type: LookupTableType,
    /// Number of witness columns batched into this group (L).
    pub num_columns: usize,
    /// Number of rows in each witness vector (trace length).
    pub witness_len: usize,
    /// Per-witness source descriptors.
    pub witness_sources: Vec<LookupWitnessSource>,
}

// ---------------------------------------------------------------------------
// Complete lookup proof
// ---------------------------------------------------------------------------

/// Top-level proof: one [`BatchedDecompLogupProof`] per lookup group
/// (groups formed by batching columns with the same [`LookupTableType`]).
/// Carries [`LookupGroupMeta`] per group so the verifier needs no
/// external specs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchedLookupProof<F> {
    /// Per-group proofs, in group order.
    pub group_proofs: Vec<BatchedDecompLogupProof<F>>,
    /// Per-group metadata needed by the verifier.
    pub group_meta: Vec<LookupGroupMeta>,
}

impl<F> BatchedLookupProof<F> {
    /// Construct an empty proof (no lookup groups).
    pub fn empty() -> Self {
        Self {
            group_proofs: Vec::new(),
            group_meta: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Sumcheck group (prepare → multi-degree sumcheck → finalize)
// ---------------------------------------------------------------------------

/// A prepared lookup sumcheck group for one table type.
///
/// Produced by `prepare_batched_lookup_group` (Phase 4b), consumed by
/// `MultiDegreeSumcheck::prove` (Phase 2a). Holds the sumcheck MLEs
/// and combination function *plus* the ancillary data needed to assemble
/// the final [`BatchedDecompLogupProof`] after the shared sumcheck.
pub struct LookupSumcheckGroup<F: PrimeField> {
    /// Sumcheck degree (always 2 for the precomputed-H variant).
    pub degree: usize,
    /// MLEs: `[eq_r, H]`.
    pub mles: Vec<DenseMultilinearExtension<F::Inner>>,
    /// Combination function: `|vals| vals[0] * vals[1]`.
    pub comb_fn: CombFn<F>,
    /// Number of sumcheck variables for this group.
    pub num_vars: usize,
    /// Ancillary proof data (multiplicities, inverses) carried through
    /// to the final [`BatchedDecompLogupProof`].
    pub proof: BatchedDecompLogupProof<F>,
    /// Group metadata carried into [`BatchedLookupProof::group_meta`].
    pub meta: LookupGroupMeta,
}

// ---------------------------------------------------------------------------
// Verifier pre-sumcheck state
// ---------------------------------------------------------------------------

/// Pre-sumcheck verification data for batched LogUp.
///
/// Holds the transcript challenges and dimensions computed before the
/// multi-degree sumcheck that are needed by `finalize_verifier`.
pub struct LookupVerifierPreSumcheck<F> {
    /// Number of sumcheck variables.
    pub num_vars: usize,
    /// The random evaluation point `r` drawn for `eq(y, r)`.
    pub r: Vec<F>,
    /// The β challenge used for shifted inversions.
    pub beta: F,
    /// The γ batching challenge.
    pub gamma: F,
    /// Number of lookups (L).
    pub num_lookups: usize,
    /// Number of chunks (K).
    pub num_chunks: usize,
    /// Witness vector length.
    pub witness_len: usize,
    /// Shift factors (K entries) — needed for decomposition check.
    pub shifts: Vec<F>,
    /// Raw subtable values — needed for the decomposition consistency
    /// check without `batch_inverse` in the verifier.
    pub subtable: Vec<F>,
}

// ---------------------------------------------------------------------------
// Grouping utility
// ---------------------------------------------------------------------------

/// A group of columns that all look up into the same decomposed table.
///
/// Produced by [`group_lookup_specs`] and consumed by the pipeline.
#[derive(Clone, Debug)]
pub struct LookupGroup {
    /// The shared table type.
    pub table_type: LookupTableType,
    /// Indices of all columns in this group.
    pub column_indices: Vec<usize>,
}

/// Groups a list of [`LookupColumnSpec`](zinc_uair::LookupColumnSpec)s
/// by their table type.
///
/// Columns with the same `LookupTableType` are batched into a single
/// `BatchedDecompLogupProtocol` instance.
pub fn group_lookup_specs(specs: &[zinc_uair::LookupColumnSpec]) -> Vec<LookupGroup> {
    use std::collections::BTreeMap;

    let mut map: BTreeMap<String, (LookupTableType, Vec<usize>)> = BTreeMap::new();
    for spec in specs {
        let key = format!("{:?}", spec.table_type);
        map.entry(key)
            .or_insert_with(|| (spec.table_type.clone(), Vec::new()))
            .1
            .push(spec.column_index);
    }
    map.into_values()
        .map(|(table_type, column_indices)| LookupGroup {
            table_type,
            column_indices,
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors from the lookup protocol.
#[derive(Debug, Error)]
pub enum LookupError<F> {
    #[error("lookup not implemented")]
    NotImplemented,

    #[error("witness entry not found in lookup table")]
    WitnessNotInTable,

    #[error("table inverse vector is incorrect at index {index}")]
    TableInverseIncorrect { index: usize },

    #[error("decomposition consistency check failed")]
    DecompositionInconsistent,

    #[error("multiplicity sum mismatch: expected {expected}, got {got}")]
    MultiplicitySumMismatch { expected: u64, got: u64 },

    #[error("final evaluation check failed")]
    FinalEvaluationMismatch,

    #[doc(hidden)]
    #[error("internal")]
    _Marker(PhantomData<F>),
}
