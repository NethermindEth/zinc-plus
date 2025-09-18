use ark_std::string::String;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum MerkleError {
    #[error("Invalid PCS opening: {0}")]
    InvalidPcsOpen(String),

    #[error("Invalid Merkle proof: {0}")]
    InvalidMerkleProof(String),

    #[error("Invalid Merkle path length: expected {expected}, got {actual}")]
    InvalidMerklePathLength { expected: usize, actual: usize },

    #[error("Invalid leaf index: {0} is out of bounds")]
    InvalidLeafIndex(usize),

    #[error("Invalid root hash")]
    InvalidRootHash,

    #[error("Failed to read merkle proof")]
    FailedMerkleProofReading,

    #[error("Failed to write merkle proof")]
    FailedMerkleProofWriting,
}
