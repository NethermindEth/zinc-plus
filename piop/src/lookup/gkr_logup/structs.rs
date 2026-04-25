//! Proof, error, and intermediate types for the GKR-LogUp lookup
//! protocol with polynomial-valued chunk lifts.
//!
//! Distinct from `super::super::structs` (the legacy/stub data types
//! for a different lookup design) — the GKR-LogUp module keeps its own
//! types so the existing scaffolding stays untouched.

use crypto_primitives::PrimeField;
use thiserror::Error;
use zinc_poly::{
    EvaluationError, univariate::dynamic::over_field::DynamicPolynomialF, utils::ArithErrors,
};
use zinc_uair::LookupTableType;

use crate::sumcheck::{SumCheckError, SumcheckProof};

// ---------------------------------------------------------------------------
// GKR fraction-tree proof types (ported verbatim from agentic-approach-v1).
// ---------------------------------------------------------------------------

/// Proof for a single GKR fractional sumcheck.
///
/// Proves `Σ_{x ∈ {0,1}^d} p(x)/q(x) = root_p/root_q` using a
/// layer-by-layer GKR protocol with `d` levels.
#[derive(Clone, Debug, PartialEq, Eq)]
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
#[derive(Clone, Debug, PartialEq, Eq)]
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

/// Proof for L batched GKR fractional sumchecks run layer-by-layer.
///
/// All L trees share the same depth and challenge trajectory; the
/// per-layer sumcheck is batched into a single sumcheck over 1 + 4L
/// MLEs at degree 3.
#[derive(Clone, Debug, PartialEq, Eq)]
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
#[derive(Clone, Debug, PartialEq, Eq)]
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

// ---------------------------------------------------------------------------
// Top-level lookup proof + group payload (chunks-in-clear poly-lift design)
// ---------------------------------------------------------------------------

/// Per-group proof for one [`LookupTableType`] in the new GKR-LogUp
/// protocol.
///
/// **Chunks are neither sent nor committed.** The prover sends only:
///   1. polynomial-valued **chunk lifts** `c_k'^(ell) = MLE[v_k^(ell)](r_inner) ∈ F_q[X]_{<chunk_width}`,
///      one per `(ell, k)` — `chunk_width` field elements each;
///   2. aggregated multiplicities `m_agg^(ell)[j]`;
///   3. the witness-side and table-side GKR fractional sumcheck proofs.
///
/// The protocol layer (caller of `verify_group`) receives a
/// [`GkrLogupGroupSubclaim`] containing the verifier-reconstructed
/// combined polynomial `c^(ell) = Σ_k X^{k·chunk_width} · c_k'^(ell)`
/// at the GKR descent row-half `r_inner`. The caller binds that
/// polynomial to the parent column's PCS commitment via a Zip+ opening
/// at `r_inner`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GkrLogupGroupProof<F: PrimeField> {
    /// `chunk_lifts[ell][k]` is the polynomial-valued chunk MLE
    /// `c_k'^(ell) = MLE[v_k^(ell)](r_inner) ∈ F_q[X]_{<chunk_width}`.
    pub chunk_lifts: Vec<Vec<DynamicPolynomialF<F>>>,
    /// `aggregated_multiplicities[ell]` of length `subtable_size`.
    pub aggregated_multiplicities: Vec<Vec<F>>,
    /// Batched witness-side GKR fractional sumcheck (L trees).
    pub witness_gkr: BatchedGkrFractionProof<F>,
    /// Single-tree table-side GKR fractional sumcheck.
    pub table_gkr: GkrFractionProof<F>,
}

/// Static metadata for one lookup group, ported alongside the proof so
/// the verifier can rebuild the subtable, shifts, and column locations
/// without having the original `LookupColumnSpec`s on hand.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GkrLogupGroupMeta {
    pub table_type: LookupTableType,
    pub num_lookups: usize,
    pub num_chunks: usize,
    pub chunk_width: usize,
    pub witness_len: usize,
    /// Full-trace column indices of the L parent columns in this group.
    pub parent_columns: Vec<usize>,
}

/// Per-lookup verifier sub-claim emitted by `verify_group` for the
/// protocol layer's Zip+ binding.
///
/// For each lookup `ell`, the protocol must verify a Zip+ opening of
/// the parent column `parent_columns[ell]` at point `r_inner`, expecting
/// the polynomial-valued evaluation `combined_polynomial[ell]`.
#[derive(Clone, Debug)]
pub struct GkrLogupGroupSubclaim<F: PrimeField> {
    /// Row-half of the GKR descent point — the point at which the
    /// parent column's MLE must be opened (length = `n_vars`).
    pub r_inner: Vec<F>,
    /// Per-lookup combined polynomial-valued claim at `r_inner`.
    /// `combined_polynomial[ell] ∈ F_q[X]_{<width}` should equal
    /// `MLE[v^(ell)](r_inner)` if all chunk lifts are honest.
    pub combined_polynomial: Vec<DynamicPolynomialF<F>>,
    /// Pass-through of `GkrLogupGroupMeta::parent_columns` for caller
    /// convenience.
    pub parent_columns: Vec<usize>,
}

/// Top-level GKR-LogUp lookup proof: one per-group payload + per-group
/// metadata, in group order. Empty (`groups: vec![]`) when the UAIR
/// declares no lookup specs.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GkrLogupLookupProof<F: PrimeField> {
    pub groups: Vec<GkrLogupGroupProof<F>>,
    pub group_meta: Vec<GkrLogupGroupMeta>,
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors specific to the GKR-LogUp lookup protocol with polynomial-
/// valued chunk lifts.
///
/// Distinct from the legacy `super::super::structs::LookupError` so the
/// existing stub scaffolding is unchanged.
#[derive(Debug, Error)]
pub enum GkrLogupError<F: PrimeField> {
    /// Wraps an underlying sumcheck failure.
    #[error("sumcheck error: {0}")]
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
    /// Multiplicity sum does not match the expected witness count.
    #[error("multiplicity sum mismatch: expected {expected}, got {got}")]
    MultiplicitySumMismatch { expected: u64, got: u64 },
    /// GKR root cross-check failed.
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
    /// Final-evaluation check (within a GKR layer subclaim) failed.
    #[error("final evaluation check failed: expected {expected:?}, got {got:?}")]
    FinalEvaluationMismatch {
        /// Expected evaluation.
        expected: F,
        /// Actual evaluation.
        got: F,
    },
    /// Polynomial-valued chunk lift `c_k'` projects to a value other
    /// than the GKR-bound scalar chunk claim `c_k`.
    #[error("chunk lift consistency check failed: psi(c_k') != c_k")]
    ChunkLiftMismatch,
    /// The Zip+ opening of the parent column at the GKR descent point
    /// disagrees with the verifier's reconstructed combined polynomial.
    #[error("parent PCS opening mismatch at GKR descent point")]
    ParentPcsMismatch,
}

impl<F: PrimeField> From<SumCheckError<F>> for GkrLogupError<F> {
    fn from(e: SumCheckError<F>) -> Self {
        Self::SumCheckError(e)
    }
}

impl<F: PrimeField> From<ArithErrors> for GkrLogupError<F> {
    fn from(e: ArithErrors) -> Self {
        Self::EqBuildError(e)
    }
}

impl<F: PrimeField> From<EvaluationError> for GkrLogupError<F> {
    fn from(e: EvaluationError) -> Self {
        Self::MleEvaluationError(e)
    }
}
