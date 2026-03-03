//! Data structures for the lookup protocol.
//!
//! Contains proof types, prover state, verifier sub-claims, and error
//! definitions for both the core LogUp protocol and the
//! decomposition-based LogUp protocol for large tables.

use crypto_primitives::PrimeField;
use thiserror::Error;
use zinc_poly::{EvaluationError, utils::ArithErrors};

use crate::sumcheck::{SumCheckError, SumcheckProof};

// ---------------------------------------------------------------------------
// LogUp
// ---------------------------------------------------------------------------

/// Proof for the LogUp protocol.
///
/// Since all auxiliary vectors are sent in the clear, the proof
/// contains these vectors plus the batched sumcheck proof.
#[derive(Clone, Debug)]
pub struct LogupProof<F: PrimeField> {
    /// Multiplicity vector: `m[j]` counts how many times `T[j]` appears in
    /// the witness.
    pub multiplicities: Vec<F>,
    /// Inverse witness vector: `u[i] = 1 / (β − w[i])`.
    pub inverse_witness: Vec<F>,
    /// Inverse table vector: `v[j] = 1 / (β − T[j])`.
    pub inverse_table: Vec<F>,
    /// The batched sumcheck proof covering the three LogUp identities.
    pub sumcheck_proof: SumcheckProof<F>,
}

/// Prover state after running the LogUp protocol.
#[derive(Clone, Debug)]
pub struct LogupProverState<F: PrimeField> {
    /// The evaluation point produced by the sumcheck.
    pub evaluation_point: Vec<F>,
}

/// Verifier sub-claim after verifying the LogUp proof.
#[derive(Clone, Debug)]
pub struct LogupVerifierSubClaim<F: PrimeField> {
    /// The evaluation point produced by the sumcheck verifier.
    pub evaluation_point: Vec<F>,
    /// The expected evaluation at the sub-claim point.
    pub expected_evaluation: F,
}

// ---------------------------------------------------------------------------
// Decomposition + LogUp
// ---------------------------------------------------------------------------

/// Proof for the Decomposition + LogUp protocol.
///
/// Used for large tables (e.g. 2^{32}) that are decomposed into two
/// sub-tables of size 2^{k} each (typically k = 16).
#[derive(Clone, Debug)]
pub struct DecompLogupProof<F: PrimeField> {
    /// Decomposition chunk vectors sent in the clear.
    /// For a 2-chunk decomposition: `[chunk_lo, chunk_hi]`.
    pub chunk_vectors: Vec<Vec<F>>,
    /// The batched sumcheck proof covering the decomposition consistency
    /// check **and** both LogUp sub-protocol identities.
    pub sumcheck_proof: SumcheckProof<F>,
    /// Aggregated multiplicity vector: `m_agg[j] = Σ_k m_k[j]`, where
    /// `m_k[j]` counts how many times `T[j]` appears in chunk `k`.
    pub aggregated_multiplicities: Vec<F>,
    /// Inverse witness vectors for each chunk's LogUp (lo, then hi).
    pub chunk_inverse_witnesses: Vec<Vec<F>>,
    /// Inverse table vector (shared between both chunks when the
    /// sub-table is the same).
    pub inverse_table: Vec<F>,
}

/// Prover state after running the Decomposition + LogUp protocol.
#[derive(Clone, Debug)]
pub struct DecompLogupProverState<F: PrimeField> {
    /// The evaluation point produced by the sumcheck.
    pub evaluation_point: Vec<F>,
}

/// Verifier sub-claim after verifying the Decomposition + LogUp proof.
#[derive(Clone, Debug)]
pub struct DecompLogupVerifierSubClaim<F: PrimeField> {
    /// The evaluation point produced by the sumcheck verifier.
    pub evaluation_point: Vec<F>,
    /// The expected evaluation at the sub-claim point.
    pub expected_evaluation: F,
}

/// Describes a decomposition-based lookup instance with K chunks.
///
/// The witness is decomposed as:
/// `w_i = shifts[0]*chunks[0][i] + shifts[1]*chunks[1][i] + … + shifts[K-1]*chunks[K-1][i]`
///
/// All chunks look up into the same sub-table.
#[derive(Clone, Debug)]
pub struct DecompLookupInstance<F: PrimeField> {
    /// The witness column (projected trace column).
    pub witness: Vec<F>,
    /// The sub-table entries (e.g. projected BitPoly(8) or Word(8)).
    pub subtable: Vec<F>,
    /// Shift factors, one per chunk.
    ///
    /// For `BitPoly(32)` with 4 chunks of width 8:
    /// `shifts = [1, a^8, a^16, a^24]`.
    pub shifts: Vec<F>,
    /// Precomputed decomposition chunks.
    ///
    /// `chunks[k][i]` is the k-th chunk of witness entry `i`.
    /// Each chunk vector has length equal to the witness length.
    pub chunks: Vec<Vec<F>>,
}

// ---------------------------------------------------------------------------
// Batched Decomposition + LogUp
// ---------------------------------------------------------------------------

/// Instance for the **batched** Decomposition + LogUp protocol.
///
/// Multiple witness vectors (all the same length) all look up into
/// the same decomposed table.  The shifts and sub-table are shared;
/// each witness has its own chunk decomposition.
///
/// `witnesses[ℓ][i] = Σ_k shifts[k] · chunks[ℓ][k][i]`
#[derive(Clone, Debug)]
pub struct BatchedDecompLookupInstance<F: PrimeField> {
    /// L witness vectors, each of the same length.
    pub witnesses: Vec<Vec<F>>,
    /// The shared sub-table entries.
    pub subtable: Vec<F>,
    /// Shift factors (K entries, same for every witness).
    pub shifts: Vec<F>,
    /// Per-witness chunk decompositions.
    ///
    /// `chunks[ℓ][k][i]` = k-th chunk of the ℓ-th witness, entry i.
    pub chunks: Vec<Vec<Vec<F>>>,
}

/// Proof for the **batched** Decomposition + LogUp protocol.
///
/// A single sumcheck covers all L lookups simultaneously.
///
/// Chunk vectors are **not** included — the verifier reconstructs
/// them from the inverse witnesses: `c_k[j] = β − 1/u_k[j]`.
/// Soundness follows from the PCS commitment binding the parent
/// column (and hence its chunks), combined with the decomposition
/// check `Σ_k shift_k · c̃_k(x*) = parent_eval(x*)`.
#[derive(Clone, Debug)]
pub struct BatchedDecompLogupProof<F: PrimeField> {
    /// The single batched sumcheck proof.
    pub sumcheck_proof: SumcheckProof<F>,
    /// Per-witness aggregated multiplicity vectors:
    /// `aggregated_multiplicities[ℓ][j] = Σ_k m_k^(ℓ)[j]`.
    pub aggregated_multiplicities: Vec<Vec<F>>,
    /// Per-witness inverse witness vectors: `chunk_inverse_witnesses[ℓ][k]`.
    pub chunk_inverse_witnesses: Vec<Vec<Vec<F>>>,
    /// Shared inverse table vector (computed once from β).
    pub inverse_table: Vec<F>,
}

/// Prover state after running the batched Decomposition + LogUp protocol.
#[derive(Clone, Debug)]
pub struct BatchedDecompLogupProverState<F: PrimeField> {
    /// The evaluation point produced by the sumcheck.
    pub evaluation_point: Vec<F>,
}

/// Verifier sub-claim after verifying the batched Decomposition + LogUp proof.
#[derive(Clone, Debug)]
pub struct BatchedDecompLogupVerifierSubClaim<F: PrimeField> {
    /// The evaluation point produced by the sumcheck verifier.
    pub evaluation_point: Vec<F>,
    /// The expected evaluation at the sub-claim point.
    pub expected_evaluation: F,
}

/// Intermediate state from the lookup pre-sumcheck phase.
///
/// Contains the sumcheck degree group and ancillary data needed to
/// assemble the final proof after the multi-degree sumcheck.
pub struct LookupSumcheckGroup<F: PrimeField> {
    /// Sumcheck degree (always 2 for the precomputed-H variant).
    pub degree: usize,
    /// MLEs: `[eq_r, H]`.
    pub mles: Vec<zinc_poly::mle::DenseMultilinearExtension<F::Inner>>,
    /// Combination function: `|vals| vals[0] * vals[1]`.
    pub comb_fn: Box<dyn Fn(&[F]) -> F + Send + Sync>,
    /// Number of sumcheck variables for this group.
    pub num_vars: usize,
    /// Ancillary data — aggregated multiplicities (for proof).
    pub aggregated_multiplicities: Vec<Vec<F>>,
    /// Ancillary data — chunk inverse witnesses (for proof).
    pub chunk_inverse_witnesses: Vec<Vec<Vec<F>>>,
    /// Ancillary data — inverse table (for proof).
    pub inverse_table: Vec<F>,
}

/// Pre-sumcheck verification data for batched LogUp.
///
/// Holds the transcript challenges and dimensions computed before the
/// sumcheck that are needed by [`finalize_verifier`].
pub struct LookupVerifierPreSumcheck<F: PrimeField> {
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
    /// Raw subtable values (T entries) — used by the optimized
    /// decomposition check to avoid `batch_inverse` in the verifier.
    pub subtable: Vec<F>,
}

// ---------------------------------------------------------------------------
// GKR LogUp
// ---------------------------------------------------------------------------

/// Proof for a single GKR fractional sumcheck.
///
/// Proves `Σ_{x ∈ {0,1}^d} p(x)/q(x) = root_p/root_q` using a
/// layer-by-layer GKR protocol with `d` levels.
#[derive(Clone, Debug)]
pub struct GkrFractionProof<F: PrimeField> {
    /// Root numerator `P = Σ p_i · Π_{j≠i} q_j`.
    pub root_p: F,
    /// Root denominator `Q = Π q_i`.
    pub root_q: F,
    /// Per-layer proofs, one for each GKR level (0..d).
    pub layer_proofs: Vec<GkrLayerProof<F>>,
}

/// Proof for a single GKR layer.
///
/// At GKR layer `k` (with `2^k` entries), the prover runs a sumcheck
/// over `k` variables (for `k ≥ 1`) or a direct check (for `k = 0`),
/// then reveals the four child-MLE evaluations at the subclaim point.
#[derive(Clone, Debug)]
pub struct GkrLayerProof<F: PrimeField> {
    /// Sumcheck proof for this layer (`None` for layer 0 which has 0 variables).
    pub sumcheck_proof: Option<SumcheckProof<F>>,
    /// Evaluation of the left-child numerator MLE at the subclaim point.
    pub p_left: F,
    /// Evaluation of the right-child numerator MLE at the subclaim point.
    pub p_right: F,
    /// Evaluation of the left-child denominator MLE at the subclaim point.
    pub q_left: F,
    /// Evaluation of the right-child denominator MLE at the subclaim point.
    pub q_right: F,
}

/// Proof for the GKR Batched Decomposition + LogUp protocol.
///
/// Replaces [`BatchedDecompLogupProof`] when chunks are committed via
/// the PCS (not sent in the clear). Only multiplicities and GKR layer
/// proofs are sent.
///
/// The witness side uses **L separate fraction trees** (one per lookup)
/// with layer-wise batched sumchecks, while the table side uses a
/// single α-batched tree.
#[derive(Clone, Debug)]
pub struct GkrBatchedDecompLogupProof<F: PrimeField> {
    /// Per-lookup aggregated multiplicity vectors:
    /// `aggregated_multiplicities[ℓ][j] = Σ_k m_k^(ℓ)[j]`.
    pub aggregated_multiplicities: Vec<Vec<F>>,
    /// Batched GKR fractional sumcheck proof for L witness trees.
    pub witness_gkr: BatchedGkrFractionProof<F>,
    /// GKR fractional sumcheck proof for the combined table tree.
    pub table_gkr: GkrFractionProof<F>,
}

/// Proof for L batched GKR fractional sumchecks run layer-by-layer.
///
/// All L trees share the same depth and challenge trajectory; the
/// per-layer sumcheck is batched into a single sumcheck over 1 + 4L
/// MLEs at degree 3.
#[derive(Clone, Debug)]
pub struct BatchedGkrFractionProof<F: PrimeField> {
    /// Per-tree root numerators: `roots_p[ℓ]`.
    pub roots_p: Vec<F>,
    /// Per-tree root denominators: `roots_q[ℓ]`.
    pub roots_q: Vec<F>,
    /// Per-layer proofs, one for each GKR level (0..d).
    pub layer_proofs: Vec<BatchedGkrLayerProof<F>>,
}

/// Proof for a single layer of the batched GKR protocol.
///
/// At layer k, the L trees share one sumcheck proof (for k ≥ 1) or a
/// direct check (k = 0), and each tree contributes 4 evaluations.
#[derive(Clone, Debug)]
pub struct BatchedGkrLayerProof<F: PrimeField> {
    /// Shared sumcheck proof for this layer (`None` for k = 0).
    pub sumcheck_proof: Option<SumcheckProof<F>>,
    /// Per-tree left-child numerator evaluations: `p_lefts[ℓ]`.
    pub p_lefts: Vec<F>,
    /// Per-tree right-child numerator evaluations: `p_rights[ℓ]`.
    pub p_rights: Vec<F>,
    /// Per-tree left-child denominator evaluations: `q_lefts[ℓ]`.
    pub q_lefts: Vec<F>,
    /// Per-tree right-child denominator evaluations: `q_rights[ℓ]`.
    pub q_rights: Vec<F>,
}

/// Prover state after running the GKR batched decomposition + LogUp protocol.
#[derive(Clone, Debug)]
pub struct GkrBatchedDecompLogupProverState<F: PrimeField> {
    /// Evaluation point at the witness-tree leaf level: `r ∈ F^{d_w}`.
    ///
    /// Decomposes as `(r_high, r_low)` where `r_low ∈ F^{log₂(W)}`
    /// is the point at which chunk column MLEs need to be opened.
    pub witness_eval_point: Vec<F>,
    /// Evaluation point at the table-tree leaf level.
    pub table_eval_point: Vec<F>,
    /// Number of witness variables (log₂ of padded witness-tree size).
    pub witness_num_vars: usize,
    /// Number of table variables.
    pub table_num_vars: usize,
    /// Batching challenge α used to combine lookups.
    pub alpha: F,
    /// Challenge β.
    pub beta: F,
    /// Number of lookups L.
    pub num_lookups: usize,
    /// Number of chunks K.
    pub num_chunks: usize,
    /// Witness length W (before padding).
    pub witness_len: usize,
}

/// Verifier sub-claim after verifying the GKR batched decomposition + LogUp proof.
#[derive(Clone, Debug)]
pub struct GkrBatchedDecompLogupVerifierSubClaim<F: PrimeField> {
    /// The evaluation point at the witness-tree leaf level.
    pub witness_eval_point: Vec<F>,
    /// Expected evaluation of the combined chunk MLE at `witness_eval_point`.
    ///
    /// The verifier needs `q̃_w(r) = β − combined_chunks(r)`, where
    /// `combined_chunks(r)` is the MLE of all L·K chunk columns flattened.
    /// This value must be provided by PCS openings.
    pub expected_witness_q_eval: F,
    /// Expected evaluation of the witness numerator MLE (α-weight pattern).
    /// Computable by the verifier from α and the tree structure.
    pub expected_witness_p_eval: F,
    /// The table evaluation point.
    pub table_eval_point: Vec<F>,
}

// ---------------------------------------------------------------------------
// Lookup specification (pipeline integration)
// ---------------------------------------------------------------------------

/// Describes the type of lookup table a column should be checked against.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum LookupTableType {
    /// Column entries are in `{0,1}^{<w}[X]` — binary polynomials of degree
    /// less than `width`, projected at element `a`.
    ///
    /// Full table size = `2^width`; for large `width` (e.g. 32) the table
    /// is decomposed into `K` sub-tables of size `2^{width/K}`.
    BitPoly {
        /// Total width of the binary polynomial (e.g. 32).
        width: usize,
        /// Optional override for the chunk width used during decomposition.
        /// When `None`, the default chunk width (8) is used for tables
        /// wider than the decomposition threshold.
        chunk_width: Option<usize>,
    },
    /// Column entries are in `[0, 2^width − 1]` — unsigned integers that
    /// fit in `width` bits.
    ///
    /// Full table size = `2^width`; for large `width` the table is
    /// decomposed into `K` sub-tables of size `2^{width/K}`.
    Word {
        /// Total bit-width (e.g. 32).
        width: usize,
        /// Optional override for the chunk width used during decomposition.
        /// When `None`, the default chunk width (8) is used.
        chunk_width: Option<usize>,
    },
}

/// Specifies that a particular trace column should be looked up against
/// a prescribed table.
#[derive(Clone, Debug)]
pub struct LookupColumnSpec {
    /// Index of the trace column (0-based into the projected field-element
    /// trace, i.e. after projection to `F`).
    pub column_index: usize,
    /// The lookup table type this column should be checked against.
    pub table_type: LookupTableType,
}

/// Specifies an **affine-combination lookup**: an integer-linear combination
/// of trace columns whose result should lie in a prescribed lookup table.
///
/// The lookup expression evaluated at each row is:
///
/// ```text
/// Σ_j (terms[j].1 · trace[terms[j].0][row]) + constant_offset
/// ```
///
/// where `constant_offset` is the field projection of a constant
/// bit-polynomial (0 for no offset).
///
/// For example, the SHA-256 Ch lookup:
///   `ê[t] + ê[t−1] − 2·ch_ef_hat[t] ∈ BitPoly{32}`
/// is expressed as terms = [(COL_E_HAT, 1), (COL_E_TM1, 1), (COL_CH_EF_HAT, -2)]
/// with constant_offset_bits = 0.
#[derive(Clone, Debug)]
pub struct AffineLookupSpec {
    /// `(column_index, coefficient)` pairs. Column indices are 0-based
    /// into the full trace.
    pub terms: Vec<(usize, i64)>,
    /// Constant bit-polynomial offset, as a u32 where bit `k` represents
    /// the coefficient of `X^k`. Set to 0 for no offset. For the SHA-256
    /// "ones" polynomial `1_w = Σ_{i=0}^{31} X^i`, use `0xFFFF_FFFF`.
    pub constant_offset_bits: u32,
    /// The lookup table type this combination should be checked against.
    pub table_type: LookupTableType,
}

/// Describes how a lookup verifier obtains the "parent evaluation" for
/// the decomposition consistency check.
///
/// For standard column lookups, the parent eval is simply the CPR
/// evaluation of that trace column. For affine-combination lookups,
/// the parent eval is the same affine combination applied to the CPR
/// evaluations.
#[derive(Clone, Debug)]
pub enum LookupWitnessSource {
    /// Standard column lookup — parent eval = `up_evals[column_index]`.
    Column {
        /// Original trace column index.
        column_index: usize,
    },
    /// Affine-combination lookup — parent eval = `Σ coeff·up_evals[col] + offset`.
    Affine {
        /// `(column_index, coefficient)` pairs.
        terms: Vec<(usize, i64)>,
        /// Constant bit-polynomial offset (u32 bit pattern).
        constant_offset_bits: u32,
    },
}

/// A group of columns that all look up into the same decomposed table.
///
/// Produced by [`group_lookup_specs`] and consumed by the pipeline
/// integration layer.
#[derive(Clone, Debug)]
pub struct LookupGroup {
    /// The shared table type.
    pub table_type: LookupTableType,
    /// Indices of all columns in this group.
    pub column_indices: Vec<usize>,
}

/// Groups a list of [`LookupColumnSpec`]s by their table type.
///
/// Columns with the same `LookupTableType` are batched into a single
/// [`BatchedDecompLogupProtocol`] instance.
pub fn group_lookup_specs(specs: &[LookupColumnSpec]) -> Vec<LookupGroup> {
    use std::collections::BTreeMap;

    // Use a deterministic ordering key so the groups are reproducible.
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

/// Errors that can occur during the LogUp or Decomposition + LogUp protocols.
#[derive(Debug, Error)]
pub enum LookupError<F: PrimeField> {
    /// A sumcheck sub-protocol failed.
    #[error("sumcheck error in lookup: {0}")]
    SumCheckError(SumCheckError<F>),
    /// Failed to build the eq polynomial.
    #[error("failed to build eq polynomial: {0}")]
    EqBuildError(ArithErrors),
    /// Failed to evaluate an MLE.
    #[error("MLE evaluation error: {0}")]
    MleEvaluationError(EvaluationError),
    /// A witness entry is not in the lookup table.
    #[error("witness entry not found in lookup table")]
    WitnessNotInTable,
    /// The prover's inverse table vector is incorrect: (β − T_j) · v_j ≠ 1.
    #[error("table inverse vector is incorrect at index {index}")]
    TableInverseIncorrect {
        /// Index of the first failing entry.
        index: usize,
    },
    /// Decomposition consistency failure.
    #[error("decomposition consistency check failed")]
    DecompositionInconsistent,
    /// Multiplicity sum does not match witness length.
    #[error("multiplicity sum mismatch: expected {expected}, got {got}")]
    MultiplicitySumMismatch {
        /// Expected sum (witness length).
        expected: u64,
        /// Actual sum.
        got: u64,
    },
    /// The verifier's final evaluation check failed.
    #[error("final evaluation check failed: expected {expected:?}, got {got:?}")]
    FinalEvaluationMismatch {
        /// Expected evaluation.
        expected: F,
        /// Actual evaluation.
        got: F,
    },
    /// GKR root cross-check failed: P_w · Q_t ≠ P_t · Q_w.
    #[error("GKR root cross-check failed")]
    GkrRootMismatch,
    /// GKR leaf-level claim mismatch.
    #[error("GKR leaf-level claim mismatch")]
    GkrLeafMismatch,
    /// GKR layer-0 direct check failed.
    #[error("GKR layer-0 check failed: expected {expected:?}, got {got:?}")]
    GkrLayer0Mismatch {
        /// Expected value.
        expected: F,
        /// Actual value.
        got: F,
    },
}

impl<F: PrimeField> From<SumCheckError<F>> for LookupError<F> {
    fn from(e: SumCheckError<F>) -> Self {
        Self::SumCheckError(e)
    }
}

impl<F: PrimeField> From<ArithErrors> for LookupError<F> {
    fn from(e: ArithErrors) -> Self {
        Self::EqBuildError(e)
    }
}

impl<F: PrimeField> From<EvaluationError> for LookupError<F> {
    fn from(e: EvaluationError) -> Self {
        Self::MleEvaluationError(e)
    }
}
