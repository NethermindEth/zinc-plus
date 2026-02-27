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
#[derive(Clone, Debug)]
pub struct BatchedDecompLogupProof<F: PrimeField> {
    /// Per-witness chunk vectors: `chunk_vectors[ℓ][k]`.
    pub chunk_vectors: Vec<Vec<Vec<F>>>,
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

// ---------------------------------------------------------------------------
// GKR LogUp
// ---------------------------------------------------------------------------

/// Proof for a single GKR fractional sumcheck.
///
/// Proves `Σ_{x ∈ {0,1}^n} p(x)/q(x) = P_root/Q_root` using a layered
/// GKR circuit that adds fractions pairwise in a binary tree.
#[derive(Clone, Debug)]
pub struct GkrFractionProof<F: PrimeField> {
    /// Root numerator of the fraction tree.
    pub root_p: F,
    /// Root denominator of the fraction tree.
    pub root_q: F,
    /// Per-layer proofs, one for each GKR round (d rounds total).
    ///
    /// `layer_proofs[0]` is the root→level-1 round (no sumcheck, just
    /// evaluations). `layer_proofs[k]` for k ≥ 1 contains a sumcheck
    /// proof plus evaluations.
    pub layer_proofs: Vec<GkrLayerProof<F>>,
}

/// Proof data for one GKR layer transition.
///
/// For round 0 (root → level 1), `sumcheck_proof` is `None` since
/// there are zero sumcheck variables (direct algebraic check).
/// For rounds 1..(d−1), it contains a sumcheck proof.
#[derive(Clone, Debug)]
pub struct GkrLayerProof<F: PrimeField> {
    /// Sumcheck proof for this layer transition (None for round 0).
    pub sumcheck_proof: Option<SumcheckProof<F>>,
    /// `p̃_{k+1}(s, 0)` — left-child numerator at the subclaim point.
    pub p_left: F,
    /// `p̃_{k+1}(s, 1)` — right-child numerator at the subclaim point.
    pub p_right: F,
    /// `q̃_{k+1}(s, 0)` — left-child denominator at the subclaim point.
    pub q_left: F,
    /// `q̃_{k+1}(s, 1)` — right-child denominator at the subclaim point.
    pub q_right: F,
}

/// Proof for the GKR LogUp protocol.
///
/// Uses the GKR protocol to prove the log-derivative identity without
/// requiring inverse vectors. Only multiplicities are sent in the clear.
#[derive(Clone, Debug)]
pub struct GkrLogupProof<F: PrimeField> {
    /// Multiplicity vector: `m[j]` counts how many times `T[j]` appears
    /// in the witness.
    pub multiplicities: Vec<F>,
    /// GKR fractional sumcheck proof for the witness side:
    /// `Σ_x 1/(β − w(x))`.
    pub witness_gkr: GkrFractionProof<F>,
    /// GKR fractional sumcheck proof for the table side:
    /// `Σ_x m(x)/(β − T(x))`.
    pub table_gkr: GkrFractionProof<F>,
}

/// Prover state after running the GKR LogUp protocol.
#[derive(Clone, Debug)]
pub struct GkrLogupProverState<F: PrimeField> {
    /// Evaluation point for the witness polynomial (from the witness GKR).
    pub witness_eval_point: Vec<F>,
    /// Evaluation point for the table/multiplicity polynomials (from the
    /// table GKR).
    pub table_eval_point: Vec<F>,
}

/// Verifier sub-claim after verifying the GKR LogUp proof.
#[derive(Clone, Debug)]
pub struct GkrLogupVerifierSubClaim<F: PrimeField> {
    /// Evaluation point for the witness polynomial.
    pub witness_eval_point: Vec<F>,
    /// Expected evaluation of the witness MLE: `w̃(r_w)`.
    pub witness_expected_eval: F,
    /// Evaluation point for the multiplicity/table polynomials.
    pub table_eval_point: Vec<F>,
    /// Expected evaluation of the multiplicity MLE: `m̃(r_t)`.
    pub mult_expected_eval: F,
    /// Expected evaluation of the table MLE: `T̃(r_t)`.
    pub table_expected_eval: F,
}

// ---------------------------------------------------------------------------
// GKR Decomposition + LogUp
// ---------------------------------------------------------------------------

/// Proof for the GKR Decomposition + LogUp protocol.
///
/// Replaces the inverse vectors of [`DecompLogupProof`] with one GKR
/// fractional-sumcheck proof per chunk plus one for the table side.
#[derive(Clone, Debug)]
pub struct GkrDecompLogupProof<F: PrimeField> {
    /// Decomposition chunk vectors sent in the clear.
    ///
    /// `chunk_vectors[k][i]` is the k-th chunk of witness entry `i`.
    pub chunk_vectors: Vec<Vec<F>>,
    /// Aggregated multiplicity vector: `m_agg[j] = Σ_k m_k[j]`.
    pub aggregated_multiplicities: Vec<F>,
    /// K GKR fractional-sumcheck proofs, one per chunk.
    ///
    /// Proves `Σ_i 1/(β − c_k[i])` for each chunk k.
    pub chunk_gkr_proofs: Vec<GkrFractionProof<F>>,
    /// GKR fractional-sumcheck proof for the table side.
    ///
    /// Proves `Σ_j m_agg[j]/(β − T[j])`.
    pub table_gkr_proof: GkrFractionProof<F>,
}

/// Prover state after running the GKR Decomposition + LogUp protocol.
#[derive(Clone, Debug)]
pub struct GkrDecompLogupProverState<F: PrimeField> {
    /// Per-chunk evaluation points from the GKR proofs.
    pub chunk_eval_points: Vec<Vec<F>>,
    /// Evaluation point for the table/multiplicity polynomials.
    pub table_eval_point: Vec<F>,
}

/// Verifier sub-claim after verifying the GKR Decomposition + LogUp proof.
#[derive(Clone, Debug)]
pub struct GkrDecompLogupVerifierSubClaim<F: PrimeField> {
    /// Per-chunk evaluation points.
    pub chunk_eval_points: Vec<Vec<F>>,
    /// Expected evaluation of each chunk MLE: `c̃_k(r_k)`.
    pub chunk_expected_evals: Vec<F>,
    /// Evaluation point for the table/multiplicity polynomials.
    pub table_eval_point: Vec<F>,
    /// Expected evaluation of the aggregated multiplicity MLE: `m̃_agg(r_t)`.
    pub mult_expected_eval: F,
    /// Expected evaluation of the table MLE: `T̃(r_t)`.
    pub table_expected_eval: F,
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// GKR Batched Decomposition + LogUp
// ---------------------------------------------------------------------------

/// Proof for the GKR Batched Decomposition + LogUp protocol.
///
/// For L witnesses, each decomposed into K chunks, proves all L lookups
/// using K·L GKR fractional sumchecks (one per chunk per witness) and one for the table side.
#[derive(Clone, Debug)]
pub struct GkrBatchedDecompLogupProof<F: PrimeField> {
    /// Per-witness chunk vectors: `chunk_vectors[ℓ][k][i]`.
    pub chunk_vectors: Vec<Vec<Vec<F>>>,
    /// Per-witness aggregated multiplicity vectors: `aggregated_multiplicities[ℓ][j]`.
    pub aggregated_multiplicities: Vec<Vec<F>>,
    /// Per-witness, per-chunk GKR proofs: `chunk_gkr_proofs[ℓ][k]`.
    pub chunk_gkr_proofs: Vec<Vec<GkrFractionProof<F>>>,
    /// GKR fractional-sumcheck proof for the table side (aggregated multiplicities).
    pub table_gkr_proof: GkrFractionProof<F>,
}

/// Prover state after running the GKR Batched Decomposition + LogUp protocol.
#[derive(Clone, Debug)]
pub struct GkrBatchedDecompLogupProverState<F: PrimeField> {
    /// Per-witness, per-chunk evaluation points from the GKR proofs.
    pub chunk_eval_points: Vec<Vec<Vec<F>>>,
    /// Evaluation point for the table/multiplicity polynomials.
    pub table_eval_point: Vec<F>,
}

/// Verifier sub-claim after verifying the GKR Batched Decomposition + LogUp proof.
#[derive(Clone, Debug)]
pub struct GkrBatchedDecompLogupVerifierSubClaim<F: PrimeField> {
    /// Per-witness, per-chunk evaluation points.
    pub chunk_eval_points: Vec<Vec<Vec<F>>>,
    /// Expected evaluation of each chunk MLE: `c̃_{ℓ,k}(r_{ℓ,k})`.
    pub chunk_expected_evals: Vec<Vec<F>>,
    /// Evaluation point for the table/multiplicity polynomials.
    pub table_eval_point: Vec<F>,
    /// Expected evaluation of the aggregated multiplicity MLE: `m̃_agg(r_t)`.
    pub mult_expected_eval: F,
    /// Expected evaluation of the table MLE: `T̃(r_t)`.
    pub table_expected_eval: F,
}

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
    /// GKR root fraction mismatch: witness and table sums disagree.
    #[error("GKR root fraction mismatch: P_w·Q_t ≠ P_t·Q_w")]
    GkrRootMismatch,
    /// GKR layer-0 direct check failed.
    #[error("GKR layer-0 check failed: expected {expected:?}, got {got:?}")]
    GkrLayer0Mismatch {
        /// Expected value.
        expected: F,
        /// Actual value.
        got: F,
    },
    /// GKR leaf evaluation claim mismatch.
    #[error("GKR leaf claim mismatch")]
    GkrLeafMismatch,
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
