//! Core data structures for the Zinc+ SNARK.

use crypto_primitives::PrimeField;
use zip_plus::batched_pcs::structs::{BatchedZipPlusCommitment, BatchedZipPlusProof};

/// A complete Zinc+ SNARK proof.
///
/// Contains the PIOP interaction data (ideal check + combined polynomial
/// resolver) and batched PCS opening proofs for both the original
/// ("up") and shifted ("down") trace column polynomials.
///
/// Each entry in the `up_pcs_proofs` / `down_pcs_proofs` vectors
/// corresponds to one PCS type batch. For single-type UAIRs there is
/// one entry; for multi-type UAIRs (e.g., binary poly + integer) there
/// is one entry per type.
#[derive(Debug, Clone)]
pub struct ZincProof<F: PrimeField> {
    /// Ideal check proof from the PIOP.
    pub ic_proof: zinc_piop::ideal_check::Proof<F>,
    /// Combined polynomial resolver proof from the PIOP.
    pub resolver_proof: zinc_piop::combined_poly_resolver::Proof<F>,
    /// Batched PCS proofs for the original trace column polynomials.
    /// One entry per PCS type batch.
    pub up_pcs_proofs: Vec<BatchedZipPlusProof>,
    /// Batched PCS proofs for the shifted trace column polynomials.
    /// One entry per PCS type batch.
    pub down_pcs_proofs: Vec<BatchedZipPlusProof>,
}

/// Commitments to all trace columns in a Zinc+ SNARK instance.
///
/// Each entry in the vectors corresponds to one PCS type batch
/// (single shared Merkle tree per batch).
#[derive(Debug, Clone)]
pub struct ZincCommitments {
    /// Batched commitments to the original trace columns.
    pub up_commitments: Vec<BatchedZipPlusCommitment>,
    /// Batched commitments to the shifted (next-row) trace columns.
    pub down_commitments: Vec<BatchedZipPlusCommitment>,
}
