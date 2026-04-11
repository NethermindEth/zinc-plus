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
// Core LogUp types
// ---------------------------------------------------------------------------

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
/// point x*, obtained from PCS openings via MultipointEval.
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
/// Carries L columns' evaluation data for the two γ-batched groups.
#[derive(Clone, Debug)]
pub struct LogupFinalizerInput<'a, F> {
    /// The multi-degree sumcheck subclaim point (x*).
    pub subclaim_point: &'a [F],
    /// Expected evaluations from the sumcheck subclaim (len=2, one
    /// per γ-batched group).
    pub expected_evaluations: &'a [F],
    /// Evaluations of the L witness column polynomials at x*.
    pub w_evals: &'a [F],
    /// Per-column auxiliary evaluations (u, m) at x*.
    pub aux_evals: &'a [LookupAuxEvals<F>],
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

    #[error("arithmetic error: {0}")]
    Arith(#[from] ArithErrors),
}
