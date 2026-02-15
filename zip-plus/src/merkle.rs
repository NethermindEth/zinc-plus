use blake3::hazmat;
use itertools::Itertools;
use std::{
    fmt,
    fmt::{Display, Formatter},
};
use thiserror::Error;
use zinc_transcript::traits::ConstTranscribable;
use zinc_utils::{add, cfg_into_iter, cfg_iter, sub};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

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
        build_merkle_tree_from_leaves(leaves)
    }

    pub fn height(&self) -> usize {
        self.layers.len()
    }

    pub fn root(&self) -> MtHash {
        self.layers
            .last()
            .expect("Merkle tree must have at least one layer")
            .first()
            .cloned()
            .expect("Merkle tree must have a root")
    }

    /// Generates a Merkle proof for the element at the given index.
    pub fn prove(&self, leaf_index: usize) -> Result<MerkleProof, MerkleError> {
        let leaf_count = self.layers[0].len();

        if leaf_index >= leaf_count || leaf_count == 0 {
            return Err(MerkleError::InvalidLeafIndex(leaf_index));
        }

        // Calculate the sibling path using layer values.
        let siblings = build_sibling_path(leaf_index, &self.layers);

        Ok(MerkleProof {
            leaf_index,
            leaf_count,
            siblings,
        })
    }
}

/// Serialize all elements of `values` into a single contiguous byte buffer
/// and hash them with Blake3 in one `update` call.  This lets Blake3 process
/// full 1 KiB chunks with SIMD, which is significantly faster than the
/// per-element `update` approach.
fn hash_column<S: ConstTranscribable>(values: &[S]) -> MtHash {
    let elem_bytes = S::NUM_BYTES;
    let mut buf = vec![0_u8; values.len() * elem_bytes];
    for (i, v) in values.iter().enumerate() {
        let start = i * elem_bytes;
        v.write_transcription_bytes(&mut buf[start..start + elem_bytes]);
    }
    let mut hasher = blake3::Hasher::new();
    hasher.update(&buf);
    hasher.finalize().into()
}

/// Construct the leaves of the Merkle tree by hashing each column across all
/// rows.
///
/// For each column, serializes all row elements into a contiguous byte buffer
/// and feeds them to Blake3 in one `update` call (same strategy as
/// [`hash_column`], but avoids collecting the column into a temporary `Vec`).
fn hash_leaves<S>(rows: &[&[S]], m_cols: usize) -> Vec<MtHash>
where
    S: ConstTranscribable + Send + Sync,
{
    let num_rows = rows.len();
    let elem_bytes = S::NUM_BYTES;
    let col_bytes = num_rows * elem_bytes;

    cfg_into_iter!(0..m_cols)
        .map(|i| {
            let mut buf = vec![0_u8; col_bytes];
            for (r, row) in rows.iter().enumerate() {
                let start = r * elem_bytes;
                row[i].write_transcription_bytes(&mut buf[start..start + elem_bytes]);
            }
            let mut hasher = blake3::Hasher::new();
            hasher.update(&buf);
            hasher.finalize().into()
        })
        .collect()
}

/// Builds a Merkle tree from the given leaves, abusing blake3::hazmat module
/// for subtree merging.
fn build_merkle_tree_from_leaves(leaves: Vec<MtHash>) -> MerkleTree {
    let n = leaves.len();

    if n == 0 {
        return MerkleTree {
            layers: vec![vec![blake3::hash(&[]).into()]],
        };
    }
    assert!(
        n.is_power_of_two(),
        "Number of leaves must be a power of two"
    );

    if n == 1 {
        return MerkleTree {
            layers: vec![leaves],
        };
    }

    // Build all layers from bottom (leaves) to top (root)
    // layers[i] contains all contiguous subtree roots of size 2^i
    let root_layer_idx = n.trailing_zeros() as usize; // log2(n)
    let num_layers = add!(root_layer_idx, 1);
    let mut layers: Vec<Vec<MtHash>> = Vec::with_capacity(num_layers);

    // Layer 0: individual leaves
    layers.push(leaves);

    // Build each subsequent layer
    for layer_idx in 1..num_layers {
        let is_root_layer = layer_idx == root_layer_idx;

        let prev_layer = &layers[sub!(layer_idx, 1)];
        let (prev_layer_chunks, _) = prev_layer.as_chunks::<2>();

        let current_layer = cfg_iter!(prev_layer_chunks)
            .map(|[left, right]| {
                if is_root_layer {
                    hazmat::merge_subtrees_root(&left.0, &right.0, hazmat::Mode::Hash).into()
                } else {
                    hazmat::merge_subtrees_non_root(&left.0, &right.0, hazmat::Mode::Hash).into()
                }
            })
            .collect();

        layers.push(current_layer);
    }

    MerkleTree { layers }
}

#[allow(clippy::arithmetic_side_effects)] // Using intentionally, overflow isn't possible
fn build_sibling_path(target_index: usize, layers: &[Vec<MtHash>]) -> Vec<MtHash> {
    let mut siblings = Vec::new();
    let mut layer_idx = 0;
    let mut current_layer = &layers[layer_idx];
    let mut current_index = target_index;

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
                debug_assert_eq!(layer_idx, layers.len() - 1);
                debug_assert_eq!(current_layer.len(), 1);
                break;
            }
        } else {
            // Right child, sibling is on the left
            let sibling_index = current_index - 1;
            siblings.push(current_layer[sibling_index].clone());
        }

        current_index /= 2;
        layer_idx += 1;
        current_layer = &layers[layer_idx];
    }

    siblings
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MerkleProof {
    /// Index of the leaf being proven
    pub leaf_index: usize,
    /// Total number of leaves in the tree
    pub leaf_count: usize,
    /// The path of sibling chaining values (bottom-up order).
    pub siblings: Vec<MtHash>,
}

impl MerkleProof {
    pub fn new(leaf_index: usize, leaf_count: usize, siblings: Vec<MtHash>) -> Self {
        assert!(!siblings.is_empty(), "Merkle proof path cannot be empty");
        assert!(leaf_index < leaf_count, "Leaf index out of bounds");
        Self {
            leaf_index,
            leaf_count,
            siblings,
        }
    }

    /// Verifies the proof against a known root hash and the claimed element
    /// data.
    pub fn verify<S>(
        &self,
        root: &MtHash,
        column_values: &[S],
        leaf_index: usize,
    ) -> Result<(), MerkleError>
    where
        S: ConstTranscribable,
    {
        if leaf_index != self.leaf_index {
            return Err(MerkleError::InvalidLeafIndex(leaf_index));
        }

        let mut current_cv: MtHash = hash_column(column_values);

        if self.leaf_count == 1 {
            if self.leaf_index == 0 && self.siblings.is_empty() {
                // The root is just the hash of the single element.
                if &current_cv != root {
                    return Err(MerkleError::InvalidRootHash);
                }
                return Ok(());
            } else {
                return Err(MerkleError::InvalidMerkleProof(
                    "Single element Merkle proof is invalid".to_owned(),
                ));
            }
        }

        let directions = get_path_directions(self.leaf_count, self.leaf_index);

        if directions.len() != self.siblings.len() {
            return Err(MerkleError::InvalidMerklePathLength {
                expected: self.siblings.len(),
                actual: directions.len(),
            });
        }

        //  Walk up the tree
        let mut path_iter = self.siblings.iter().zip(directions.iter());

        // Pop the last element for the root merge.
        let Some((last_sibling, last_direction)) = path_iter.next_back() else {
            unreachable!("There should always be at least one sibling in the proof");
        };

        // Iterate over intermediate merges (non-root).
        for (sibling_cv, direction) in path_iter {
            let is_left = matches!(direction, PathDirection::Left);
            if is_left {
                current_cv = hazmat::merge_subtrees_non_root(
                    &current_cv.0,
                    &sibling_cv.0,
                    hazmat::Mode::Hash,
                )
                .into();
            } else {
                current_cv = hazmat::merge_subtrees_non_root(
                    &sibling_cv.0,
                    &current_cv.0,
                    hazmat::Mode::Hash,
                )
                .into();
            }
        }

        // Final root merge.
        let final_hash: MtHash = if matches!(last_direction, PathDirection::Left) {
            hazmat::merge_subtrees_root(&current_cv.0, &last_sibling.0, hazmat::Mode::Hash).into()
        } else {
            hazmat::merge_subtrees_root(&last_sibling.0, &current_cv.0, hazmat::Mode::Hash).into()
        };

        if &final_hash != root {
            return Err(MerkleError::InvalidRootHash);
        }
        Ok(())
    }

    /// Estimate the number of bytes that would be written to [[PcsTranscript]]
    /// when an instance of this type is transcribed.
    #[allow(clippy::arithmetic_side_effects)] // Overflow isn't possible
    pub fn estimate_transcribed_size(merkle_tree_height: usize) -> usize {
        // Note the proof does not include leaf layer, so we subtract 1.
        3 * u64::NUM_BYTES + (merkle_tree_height - 1) * MtHash::NUM_BYTES
    }
}

impl Display for MerkleProof {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(f, "Merkle Path: {}", self.siblings.iter().join(", "))?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PathDirection {
    Left,
    Right,
}

/// Helper to determine the path directions (leaf to root).
#[allow(clippy::arithmetic_side_effects)] // Intentional, no side effects possible.
fn get_path_directions(total_chunks: usize, target_index: usize) -> Vec<PathDirection> {
    let mut path = Vec::new();
    let mut current_size = total_chunks;
    let mut current_index = target_index;

    // Iterate top-down (Root to Leaf) to determine the path based on BLAKE3 rules.
    while current_size > 1 {
        // BLAKE3 split rule: largest power of two less than N
        // (or N/2 if N is power of 2).
        let split_len = current_size.next_power_of_two() / 2;

        if current_index < split_len {
            path.push(PathDirection::Left);
            current_size = split_len;
        } else {
            // Went right.
            path.push(PathDirection::Right);
            current_size -= split_len;
            current_index -= split_len;
        }
    }
    // Reverse the path so it is ordered from leaf to root (bottom-up) for
    // verification.
    path.reverse();
    path
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
            let proof = merkle_tree.prove(i).expect("Merkle proof creation failed");

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
