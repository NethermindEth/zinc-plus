use crate::{
    pcs::structs::AsPackable,
    traits::{ConstTranscribable, Transcribable},
    utils::ReinterpretVector,
};
use itertools::Itertools;
use p3_commit::{BatchOpeningRef, Mmcs};
use p3_field::Packable;
use p3_matrix::{Dimensions, Matrix as P3Matrix};
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{CryptographicHasher, PseudoCompressionFunction};
use std::{
    fmt,
    fmt::{Display, Formatter},
};
use thiserror::Error;

pub const HASH_OUT_LEN: usize = blake3::OUT_LEN;

#[derive(Clone, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct MtHash(pub(crate) [u8; HASH_OUT_LEN]);

impl Default for MtHash {
    fn default() -> Self {
        MtHash([0; HASH_OUT_LEN])
    }
}

impl Display for MtHash {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let blake3_hash: blake3::Hash = self.0.into();
        <blake3::Hash as Display>::fmt(&blake3_hash, f)
    }
}

impl ConstTranscribable for MtHash {
    const NUM_BYTES: usize = HASH_OUT_LEN;

    fn read_transcription_bytes(buf: &[u8]) -> Self {
        assert_eq!(buf.len(), HASH_OUT_LEN);
        MtHash(buf.try_into().expect("Invalid buffer length for MtHash"))
    }

    fn write_transcription_bytes(&self, buf: &mut [u8]) {
        assert_eq!(buf.len(), HASH_OUT_LEN);
        buf.copy_from_slice(&self.0);
    }
}

impl<B> From<B> for MtHash
where
    B: Into<[u8; HASH_OUT_LEN]>,
{
    fn from(b: B) -> Self {
        MtHash(b.into())
    }
}

#[derive(Debug, Default, Clone)]
struct MtHasher;

impl<T: ConstTranscribable + Clone> CryptographicHasher<T, [u8; HASH_OUT_LEN]> for MtHasher {
    fn hash_iter<I>(&self, input: I) -> [u8; HASH_OUT_LEN]
    where
        I: IntoIterator<Item = T>,
    {
        let mut hasher = blake3::Hasher::new();
        let mut buf = vec![0_u8; T::NUM_BYTES];
        for item in input {
            item.write_transcription_bytes(&mut buf);
            hasher.update(&buf);
        }
        hasher.finalize().into()
    }
}

#[derive(Debug, Default, Clone)]
struct MtPerm;

impl PseudoCompressionFunction<[u8; HASH_OUT_LEN], 2> for MtPerm {
    fn compress(&self, input: [[u8; HASH_OUT_LEN]; 2]) -> [u8; HASH_OUT_LEN] {
        let mut hasher = blake3::Hasher::new();
        for ref item in input {
            hasher.update(item);
        }
        hasher.finalize().into()
    }
}

type Matrix<T> = TransposedMatrix<T>;
type MtMmcs<T> = MerkleTreeMmcs<T, u8, MtHasher, MtPerm, HASH_OUT_LEN>;
type P3MerkleTree<T> = p3_merkle_tree::MerkleTree<T, u8, Matrix<T>, HASH_OUT_LEN>;

#[derive(Debug, Default)]
pub struct MerkleTree<T>
where
    T: Packable + Transcribable + Clone + Send + Sync,
{
    inner: Option<MerkleTreeInner<T>>,
}

#[derive(Debug)]
struct MerkleTreeInner<T> {
    prover_data: P3MerkleTree<T>,
    matrix_dims: Dimensions,
}

impl<T> MerkleTree<T>
where
    T: Packable + ConstTranscribable + Clone + Send + Sync,
{
    pub fn new<S>(rows: Vec<Vec<S>>) -> Self
    where
        S: AsPackable<Packable = T>,
    {
        assert!(!rows.is_empty());
        let row_width = rows[0].len();
        assert!(row_width > 0);
        assert!(
            rows.iter().all(|row| row.len() == row_width),
            "All rows must have the same width"
        );
        assert!(row_width.is_power_of_two());

        // Each matrix row is hashed together to form a leaf in the Merkle tree.
        // Thus, we need to use a transposed matrix to have original columns as leaves.
        let matrix = {
            let rows = unsafe { ReinterpretVector::reinterpret_vector(rows) };
            // Safe because we just initialized all elements of `columns`, and
            // MaybeUninit<T> is #[repr(transparent)].
            Matrix::new(rows)
        };

        let matrix_dims = matrix.dimensions();
        let prover_data = P3MerkleTree::new::<T, _, _, _>(&MtHasher, &MtPerm, vec![matrix]);

        Self {
            inner: Some(MerkleTreeInner {
                prover_data,
                matrix_dims,
            }),
        }
    }

    pub fn root(&self) -> MtHash {
        MtHash(
            *self
                .inner
                .as_ref()
                .expect("Merkle tree not initialized")
                .prover_data
                .root()
                .as_ref(),
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MerkleProof {
    pub path: Vec<MtHash>,
    pub matrix_dims: Dimensions,
}

impl MerkleProof {
    pub fn new(path: Vec<MtHash>, matrix_dims: Dimensions) -> Self {
        MerkleProof { path, matrix_dims }
    }

    pub fn create_proof<T>(merkle_tree: &MerkleTree<T>, leaf: usize) -> Result<Self, MerkleError>
    where
        T: Packable + ConstTranscribable + Clone,
    {
        let mt = merkle_tree
            .inner
            .as_ref()
            .ok_or(MerkleError::InvalidRootHash)?;
        let prover = MtMmcs::<T>::new(MtHasher, MtPerm);
        let opening = prover.open_batch(leaf, &mt.prover_data);
        let path = opening.opening_proof.into_iter().map(MtHash).collect();
        Ok(Self::new(path, mt.matrix_dims))
    }

    pub fn verify<S, T>(
        &self,
        root: &MtHash,
        leaf_values: Vec<S>,
        leaf_index: usize,
    ) -> Result<(), MerkleError>
    where
        S: AsPackable<Packable = T>,
        T: Packable + ConstTranscribable + Clone,
    {
        let prover = MtMmcs::<T>::new(MtHasher, MtPerm);

        let leaf_values = unsafe { ReinterpretVector::reinterpret_vector(leaf_values) };

        let values = vec![leaf_values];
        let proof = self.path.iter().map(|h| h.0).collect_vec();
        let proof = BatchOpeningRef::new(&values, &proof);
        prover
            .verify_batch(&root.0.into(), &[self.matrix_dims], leaf_index, proof)
            .map_err(|e| {
                MerkleError::InvalidMerkleProof(format!("Failed to validate Merkle proof: {:?}", e))
            })
    }
}

impl Display for MerkleProof {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(f, "Merkle Path: {}", self.path.iter().join(", "))?;
        writeln!(f, "Matrix Dimensions: {}", self.matrix_dims)?;
        Ok(())
    }
}

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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TransposedMatrix<T> {
    columns: Vec<Vec<T>>,
    column_len: usize,
}

impl<T> TransposedMatrix<T> {
    pub fn new(columns: Vec<Vec<T>>) -> Self {
        let column_len = columns.first().map_or(0, Vec::len);
        Self {
            columns,
            column_len,
        }
    }
}

impl<T> p3_matrix::Matrix<T> for TransposedMatrix<T>
where
    T: Packable + ConstTranscribable + Clone + Send + Sync,
{
    fn width(&self) -> usize {
        self.columns.len()
    }

    fn height(&self) -> usize {
        self.column_len
    }

    unsafe fn get_unchecked(&self, r: usize, c: usize) -> T {
        unsafe { *self.columns.get_unchecked(c).get_unchecked(r) }
    }

    unsafe fn row_unchecked(
        &self,
        r: usize,
    ) -> impl IntoIterator<Item = T, IntoIter = impl Iterator<Item = T> + Send + Sync> {
        unsafe { self.columns.iter().map(move |col| *col.get_unchecked(r)) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crypto_bigint::Random;
    use crypto_primitives::crypto_bigint_int::Int;
    use rand::rng;

    #[test]
    fn test_merkle_proof() {
        const N: usize = 3;
        let leaves_len = 1024;
        let mut rng = rng();
        let leaves_data = (0..leaves_len)
            .map(|_| Int::random(&mut rng))
            .collect::<Vec<Int<N>>>();

        let merkle_tree = MerkleTree::new(vec![leaves_data.clone()]);

        // Print tree structure after merklizing
        let root = merkle_tree.root();
        // Create a proof for the first leaf
        for (i, leaf) in leaves_data.iter().enumerate() {
            let proof =
                MerkleProof::create_proof(&merkle_tree, i).expect("Merkle proof creation failed");

            // Verify the proof
            proof
                .verify(&root, vec![*leaf], i)
                .expect("Merkle proof verification failed");
        }
    }
}
