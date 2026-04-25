//! Proof, error, and intermediate types for the GKR-LogUp lookup
//! protocol with polynomial-valued chunk lifts.
//!
//! Distinct from `super::super::structs` (the legacy/stub data types
//! for a different lookup design) — the GKR-LogUp module keeps its own
//! types so the existing scaffolding stays untouched.

use crypto_primitives::PrimeField;
use thiserror::Error;
use zinc_poly::{EvaluationError, utils::ArithErrors};

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
