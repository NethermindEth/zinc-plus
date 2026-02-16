use crate::{
    code::LinearCode,
    merkle::{MerkleTree, MtHash},
    pcs::structs::{ZipPlusParams, ZipTypes},
};
use crypto_primitives::DenseRowMatrix;
use std::marker::PhantomData;

/// Batched Zip+ is a Polynomial Commitment Scheme (PCS) that supports
/// committing to a batch of multilinear polynomials with a single shared
/// Merkle tree.
pub struct BatchedZipPlus<Zt: ZipTypes, Lc: LinearCode<Zt>>(PhantomData<(Zt, Lc)>);

/// Full data of a batched zip commitment to multiple multilinear polynomials,
/// including encoded rows per polynomial and a single shared Merkle tree,
/// kept by the prover for the testing phase.
#[derive(Debug)]
pub struct BatchedZipPlusHint<R> {
    /// The encoded rows of each polynomial's matrix representation.
    /// `cw_matrices[i]` is the codeword matrix for the i-th polynomial.
    pub cw_matrices: Vec<DenseRowMatrix<R>>,
    /// Single shared Merkle tree built from the concatenated columns of all
    /// codeword matrices.
    pub merkle_tree: MerkleTree,
}

impl<R> BatchedZipPlusHint<R> {
    pub fn new(
        cw_matrices: Vec<DenseRowMatrix<R>>,
        merkle_tree: MerkleTree,
    ) -> BatchedZipPlusHint<R> {
        BatchedZipPlusHint {
            cw_matrices,
            merkle_tree,
        }
    }

    /// Number of polynomials in the batch.
    pub fn batch_size(&self) -> usize {
        self.cw_matrices.len()
    }
}

/// The compact commitment to a batch of multilinear polynomials, consisting of
/// the single shared Merkle root, to be sent to the verifier.
#[derive(Clone, Debug, Default)]
pub struct BatchedZipPlusCommitment {
    /// Root of the shared merkle tree of all polynomials' codeword matrices.
    pub root: MtHash,
    /// Number of polynomials in the batch.
    pub batch_size: usize,
}

/// Proof obtained by the verifier after the testing phase with a batch of
/// polynomials.
#[derive(Clone, Debug)]
pub struct BatchedZipPlusTestTranscript(pub(crate) crate::pcs_transcript::PcsTranscript);

impl From<crate::pcs_transcript::PcsTranscript> for BatchedZipPlusTestTranscript {
    fn from(transcript: crate::pcs_transcript::PcsTranscript) -> Self {
        BatchedZipPlusTestTranscript(transcript)
    }
}

impl From<BatchedZipPlusTestTranscript> for crate::pcs_transcript::PcsTranscript {
    fn from(value: BatchedZipPlusTestTranscript) -> Self {
        value.0
    }
}

/// Proof obtained by the verifier after the evaluation phase of a batched PCS.
#[derive(Clone, Debug)]
pub struct BatchedZipPlusProof(pub(crate) Vec<u8>);

impl From<crate::pcs_transcript::PcsTranscript> for BatchedZipPlusProof {
    fn from(transcript: crate::pcs_transcript::PcsTranscript) -> Self {
        BatchedZipPlusProof(transcript.stream.into_inner())
    }
}

impl From<BatchedZipPlusProof> for crate::pcs_transcript::PcsTranscript {
    fn from(proof: BatchedZipPlusProof) -> Self {
        Self {
            fs_transcript: zinc_transcript::KeccakTranscript::default(),
            stream: std::io::Cursor::new(proof.0),
        }
    }
}

/// Re-use `ZipPlusParams` from the single-polynomial PCS — parameters are
/// the same for each polynomial in the batch.
pub type BatchedZipPlusParams<Zt, Lc> = ZipPlusParams<Zt, Lc>;
