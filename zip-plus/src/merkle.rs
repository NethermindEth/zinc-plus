use crate::{add, traits::ConstTranscribable, utils::parallelize_into_iter_map_collect};
use itertools::Itertools;
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

#[derive(Debug, Default)]
pub struct MerkleTree {
    inner: Option<MerkleTreeInner>,
}

#[derive(Debug)]
struct MerkleTreeInner {
    /// First vector is leaves, last vector is root
    layers: Vec<Vec<MtHash>>,
}

impl MerkleTree {
    pub fn new<S>(rows: &[&[S]]) -> Self
    where
        S: ConstTranscribable + Clone + Send + Sync,
    {
        assert!(!rows.is_empty());
        let row_width = rows[0].len();
        assert!(row_width > 0);
        assert!(
            rows.iter().all(|row| row.len() == row_width),
            "All rows must have the same width"
        );
        assert!(row_width.is_power_of_two());

        let leaves = hash_leaves(rows, row_width);
        let inner = build_merkle_tree_from_leaves(leaves);

        Self { inner: Some(inner) }
    }

    pub fn root(&self) -> MtHash {
        self.inner
            .as_ref()
            .expect("Merkle tree not initialized")
            .layers
            .last()
            .and_then(|v| v.first())
            .cloned()
            .expect("Merkle tree has no root node")
    }
}

fn hash_leaves<S>(rows: &[&[S]], m_cols: usize) -> Vec<MtHash>
where
    S: ConstTranscribable + Send + Sync,
{
    parallelize_into_iter_map_collect(0..m_cols, |i| {
        let mut hasher = blake3::Hasher::new();
        let mut buf = vec![0_u8; S::NUM_BYTES];
        for row in rows.iter() {
            let v = &row[i];
            v.write_transcription_bytes(&mut buf);
            hasher.update(&buf);
        }
        hasher.finalize().into()
    })
}

#[allow(clippy::unwrap_used)] // Using unwrap here never panics
fn build_merkle_tree_from_leaves(nodes: Vec<MtHash>) -> MerkleTreeInner {
    if nodes.is_empty() {
        return MerkleTreeInner {
            layers: vec![vec![blake3::Hasher::new().finalize().into()]],
        };
    }
    assert!(
        nodes.len().is_power_of_two(),
        "Number of leaves must be a power of two"
    );
    let tree_height = nodes.len().trailing_zeros() as usize;
    let mut layers = Vec::with_capacity(tree_height);
    layers.push(nodes);

    loop {
        let (chunked_prev_layer, []) = layers.last().unwrap().as_chunks::<2>() else {
            unreachable!(
                "Leaves length must be a power of two, so we should always have an even number of nodes"
            )
        };
        let layer: Vec<MtHash> = chunked_prev_layer
            .iter()
            .map(|hash_pair: &[MtHash; 2]| {
                let mut hasher = blake3::Hasher::new();
                hasher.update(&hash_pair[0].0);
                hasher.update(&hash_pair[1].0);
                hasher.finalize().into()
            })
            .collect();
        let level_size = layer.len();
        layers.push(layer);
        if level_size == 1 {
            break; // We reached the root
        }
    }
    MerkleTreeInner { layers }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MerkleProof {
    /// Siblings along the path from leaf to root. Does not include the leaf and
    /// root hashes.
    pub siblings: Vec<MtHash>,
    /// Index of the leaf being proven
    pub leaf_index: usize,
    /// Total number of leaves in the tree
    pub tree_size: usize,
}

impl MerkleProof {
    pub fn new(path: Vec<MtHash>, leaf_index: usize, tree_size: usize) -> Self {
        assert!(!path.is_empty(), "Merkle proof path cannot be empty");
        assert!(leaf_index < tree_size, "Leaf index out of bounds");
        Self {
            siblings: path,
            leaf_index,
            tree_size,
        }
    }

    #[allow(clippy::arithmetic_side_effects)] // Using intentionally, overflow isn't possible
    pub fn create_proof(merkle_tree: &MerkleTree, leaf_index: usize) -> Result<Self, MerkleError> {
        let mt = merkle_tree
            .inner
            .as_ref()
            .ok_or(MerkleError::InvalidRootHash)?;

        let mut siblings = Vec::new();
        let mut layer_idx = 0;
        let mut current_layer = &mt.layers[layer_idx];
        let mut current_index = leaf_index;

        loop {
            // Determine if current node is left (even) or right (odd) child
            let is_left_child = current_index.is_multiple_of(2);

            if is_left_child {
                // Left child, sibling is on the right
                let sibling_index = current_index + 1;
                if sibling_index < current_layer.len() {
                    siblings.push(current_layer[sibling_index].clone());
                } else {
                    // We've reached the root
                    debug_assert_eq!(layer_idx, mt.layers.len() - 1);
                    break;
                }
            } else {
                // Right child, sibling is on the left
                let sibling_index = current_index - 1;
                siblings.push(current_layer[sibling_index].clone());
            }

            current_index /= 2;
            layer_idx += 1;
            current_layer = &mt.layers[layer_idx];
        }

        Ok(MerkleProof {
            siblings,
            leaf_index,
            tree_size: mt.layers[0].len(),
        })
    }

    pub fn verify<S>(
        &self,
        root: &MtHash,
        leaf_values: &[S],
        leaf_index: usize,
    ) -> Result<(), MerkleError>
    where
        S: ConstTranscribable,
    {
        if self.siblings.is_empty() {
            return Err(MerkleError::InvalidMerkleProof(
                "Merkle proof siblings was empty".to_owned(),
            ));
        }
        if leaf_index != self.leaf_index {
            return Err(MerkleError::InvalidLeafIndex(leaf_index));
        }

        let mut buf = vec![0_u8; S::NUM_BYTES];
        let mut current_hash: MtHash = {
            let mut hasher = blake3::Hasher::new();
            for v in leaf_values.iter() {
                v.write_transcription_bytes(&mut buf);
                hasher.update(&buf);
            }
            hasher.finalize().into()
        };
        let mut current_index = self.leaf_index;
        let mut level_size = self.tree_size;
        let mut siblings_iter = self.siblings.iter();

        while level_size > 1 {
            // Determine if current node is left (even) or right (odd) child
            let is_left_child = current_index.is_multiple_of(2);

            let sibling_hash = siblings_iter.next().ok_or(MerkleError::InvalidMerkleProof(
                "Not enough siblings in proof".to_owned(),
            ))?;

            let mut hasher = blake3::Hasher::new();
            if is_left_child {
                hasher.update(&current_hash.0);
                hasher.update(&sibling_hash.0);
            } else {
                hasher.update(&sibling_hash.0);
                hasher.update(&current_hash.0);
            }

            current_hash = hasher.finalize().into();

            current_index /= 2;
            level_size = level_size.div_ceil(2);
        }

        if siblings_iter.next().is_some() {
            return Err(MerkleError::InvalidMerklePathLength {
                expected: self.siblings.len(),
                actual: add!(self.siblings.len(), 1),
            });
        }

        if &current_hash != root {
            return Err(MerkleError::InvalidRootHash);
        }
        Ok(())
    }
}

impl Display for MerkleProof {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(f, "Merkle Path: {}", self.siblings.iter().join(", "))?;
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

    #[error("Invalid leaf index: {0}")]
    InvalidLeafIndex(usize),

    #[error("Invalid root hash")]
    InvalidRootHash,

    #[error("Failed to read merkle proof")]
    FailedMerkleProofReading,

    #[error("Failed to write merkle proof")]
    FailedMerkleProofWriting,
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

        let merkle_tree = MerkleTree::new(&[leaves_data.as_slice()]);

        // Print tree structure after merklizing
        let root = merkle_tree.root();
        // Create a proof for the first leaf
        for (i, leaf) in leaves_data.iter().enumerate() {
            let proof =
                MerkleProof::create_proof(&merkle_tree, i).expect("Merkle proof creation failed");

            // Verify the proof
            let result = proof.verify(&root, &[*leaf], i);
            assert!(
                result.is_ok(),
                "Merkle proof verification failed for leaf index {i}: {}",
                result.err().unwrap()
            );
        }
    }
}
