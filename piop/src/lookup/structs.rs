//! Data structures for the lookup protocol.
//!
//! Proof types, prover/verifier intermediates and error
//! definitions live here. Specification types (`LookupTableType`,
//! `LookupColumnSpec`) live in `zinc-uair` and are re-exported from the
//! parent module.

use std::collections::BTreeMap;
use thiserror::Error;
use zinc_poly::utils::ArithErrors;
use zinc_uair::LookupTableType;

// ---------------------------------------------------------------------------
// Per-group proof (BatchedDecompLogup)
// ---------------------------------------------------------------------------

/// Proof for one lookup group (columns sharing the same table type).
///
/// Does **not** contain a sumcheck proof — the sumcheck is shared via
/// the protocol-level multi-degree sumcheck. This struct
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
    /// offset`. Currently only needed for BitPoly
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
#[derive(Clone, Default, Debug, PartialEq, Eq)]
pub struct BatchedLookupProof<F> {
    /// Per-group proofs, in group order.
    pub group_proofs: Vec<BatchedDecompLogupProof<F>>,
    /// Per-group metadata needed by the verifier.
    pub group_meta: Vec<LookupGroupMeta>,
}

// ---------------------------------------------------------------------------
// Core LogUp types
// ---------------------------------------------------------------------------

/// Verifier subclaim from the LogUp protocol.
#[derive(Clone, Debug)]
pub struct LogupVerifierSubClaim<F> {
    /// Evaluation point from the sumcheck subclaim.
    pub evaluation_point: Vec<F>,
    /// Expected evaluation at the subclaim point.
    pub expected_evaluation: F,
}

/// Ancillary prover data for LogUp sumcheck group construction.
///
/// Borrows the pre-computed auxiliary vectors needed by
/// [`LogupProtocol::build_sumcheck_groups`].
#[derive(Clone, Debug)]
pub struct LogupProverAncillary<'a, F> {
    /// Multiplicity vector.
    pub multiplicities: &'a [F],
    /// Inverse witness vector `u = 1/(β − w)`.
    pub inverse_witness: &'a [F],
    /// Inverse table vector `v = 1/(β − T)`.
    pub inverse_table: &'a [F],
}

/// Pre-sumcheck verifier data for the core LogUp protocol.
///
/// Produced by `build_verifier_pre_sumcheck`, holds the transcript
/// challenges needed for `finalize_verifier`.
#[derive(Clone, Debug)]
pub struct LogupVerifierPreSumcheckData<F> {
    /// Random evaluation point `r` for `eq(y, r)`.
    pub r: Vec<F>,
    /// The β challenge for LogUp.
    pub beta: F,
    /// The γ challenge for batching multiple lookup columns.
    pub gamma: F,
}

/// Scalar MLE evaluations of lookup auxiliary columns at the subclaim
/// point r*, obtained from PCS openings via MultipointEval.
///
/// Used by `finalize_verifier` to check the LogUp identities
#[derive(Clone, Debug)]
pub struct LookupAuxEvals<F> {
    pub u_eval: F,
    pub m_eval: F,
}

/// Bundle of per-group evaluation data passed to
/// [`LogupProtocol::finalize_verifier`].
///
/// Carries evaluation data for the 2 batched LogUp groups.
/// For non-decomposed lookups, `w_evals` and `aux_evals` have L entries.
/// For decomposed lookups (K chunks per column), both have L*K entries
///
/// The reconstruction check for decomposed lookups is handled at the
/// protocol level (verifier.rs), not here.
#[derive(Clone, Debug)]
pub struct LogupFinalizerInput<'a, F> {
    /// The multi-degree sumcheck subclaim point (r*).
    pub subclaim_point: &'a [F],
    /// Expected evaluations from the sumcheck subclaim for the 2
    /// batched LogUp groups.
    pub expected_evaluations: &'a [F],
    /// Evaluations of the witness column polynomials at r*.
    /// L entries for non-decomposed, L*K chunk evals for decomposed.
    pub w_evals: &'a [F],
    /// Per-column auxiliary evaluations (u, m) at r*.
    /// Length L for non-decomposed, L*K for decomposed.
    pub aux_evals: &'a [LookupAuxEvals<F>],
}

// ---------------------------------------------------------------------------
// Decomposition info
// ---------------------------------------------------------------------------

/// Bundle of decomposition data for a single lookup group with
/// `chunk_width.is_some()`.
///
/// Produced by [`LogupProtocol::extract_decomposed`] and consumed by the
/// prover to build reconstruction sumcheck groups and commit chunk columns.
#[derive(Clone, Debug)]
pub struct DecompInfo<F> {
    /// The smaller subtable (2^chunk_width entries) used for LogUp.
    pub subtable: Vec<F>,
    /// Positional bases `[1, base, base^2, …]` of length K.
    pub decomp_bases: Vec<F>,
    /// Decomposed chunk columns. `chunks[l][k]` is chunk k of column l.
    pub chunks: Vec<Vec<Vec<F>>>,
}

// ---------------------------------------------------------------------------
// Grouping utility
// ---------------------------------------------------------------------------

/// A group of columns that all look up into the same decomposed table.
///
/// Produced by [`group_lookup_specs`] and consumed by the pipeline.
#[derive(Debug)]
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
    let mut map: BTreeMap<LookupTableType, Vec<usize>> = BTreeMap::new();
    for spec in specs {
        map.entry(spec.table_type.clone())
            .or_default()
            .push(spec.column_index);
    }
    map.into_iter()
        .map(|(table_type, column_indices)| LookupGroup {
            table_type,
            column_indices,
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum LookupError {
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

    #[error("malformed proof: expected {expected} evaluations, got {got}")]
    WrongEvaluationCount { expected: usize, got: usize },

    #[error("malformed proof: w_evals length ({w}) != aux_evals length ({aux})")]
    EvalLengthMismatch { w: usize, aux: usize },

    #[error("arithmetic error: {0}")]
    Arith(#[from] ArithErrors),
}
