use crate::{sub, traits::ConstTranscribable};
use ark_std::cfg_into_iter;
use blake3::hazmat;
use itertools::Itertools;
use std::{
    fmt,
    fmt::{Display, Formatter},
};
use thiserror::Error;

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
    root: MtHash,
    /// The leaf layer of the tree (ChainingValues of the chunks).
    leaf_cvs: Vec<MtHash>,
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

    pub fn root(&self) -> MtHash {
        self.root.clone()
    }

    /// Generates a Merkle proof for the element at the given index.
    pub fn prove(&self, leaf_index: usize) -> Result<MerkleProof, MerkleError> {
        let leaf_count = self.leaf_cvs.len();

        if leaf_index >= leaf_count || leaf_count == 0 {
            return Err(MerkleError::InvalidLeafIndex(leaf_index));
        }

        // Calculate the sibling path.
        let siblings = find_path_recursive(&self.leaf_cvs, leaf_index);

        Ok(MerkleProof {
            leaf_index,
            leaf_count,
            siblings,
        })
    }
}

// This could've been a function, but macro performance is better.
macro_rules! hash_many {
    ($iter:expr, $t:tt) => {{
        let mut hasher = blake3::Hasher::new();
        let mut buf = vec![0_u8; <$t>::NUM_BYTES];
        for v in $iter {
            v.write_transcription_bytes(&mut buf);
            hasher.update(&buf);
        }
        hasher.finalize().into()
    }};
}

/// Find the roots of the left and right subtrees.
/// Do it parallel if the "parallel" feature is enabled.
macro_rules! get_left_right_subtree_roots {
    ($left:ident, $right:ident) => {{
        #[cfg(feature = "parallel")]
        let result = rayon::join(|| get_subtree_root($left), || get_subtree_root($right));
        #[cfg(not(feature = "parallel"))]
        let result = (get_subtree_root($left), get_subtree_root($right));
        result
    }};
}

fn hash_leaves<S>(rows: &[&[S]], m_cols: usize) -> Vec<MtHash>
where
    S: ConstTranscribable + Send + Sync,
{
    cfg_into_iter!(0..m_cols)
        .map(|i| {
            // Hash the i-th column across all rows.
            hash_many!(rows.iter().map(|row| &row[i]), S)
        })
        .collect()
}

fn build_merkle_tree_from_leaves(leaves: Vec<MtHash>) -> MerkleTree {
    if leaves.is_empty() {
        return MerkleTree {
            root: blake3::hash(&[]).into(),
            leaf_cvs: vec![],
        };
    }
    assert!(
        leaves.len().is_power_of_two(),
        "Number of leaves must be a power of two"
    );

    let root = if leaves.len() == 1 {
        // In this design, the root of a single-element tree is just the hash of the
        // element itself.
        leaves[0].clone()
    } else {
        // Build the tree structure recursively from the leaves.
        compute_root_from_leaves(&leaves)
    };

    MerkleTree {
        root,
        leaf_cvs: leaves,
    }
}

/// Helper to compute the root from leaves (len > 1). The top-level merge uses
/// merge_subtrees_root.
fn compute_root_from_leaves(cvs: &[MtHash]) -> MtHash {
    assert!(cvs.len() > 1);

    // Find the split point according to BLAKE3 rules.
    let split_len = cvs.len().next_power_of_two() / 2;
    let (left, right) = cvs.split_at(split_len);

    // Recursively get the roots of the subtrees (non-root merges).
    let (left_root, right_root) = get_left_right_subtree_roots!(left, right);

    // The final merge is a root merge.
    hazmat::merge_subtrees_root(&left_root.0, &right_root.0, hazmat::Mode::Hash).into()
}

/// Recursively merges a slice of ChainingValues into a single root CV for that
/// subtree.
fn get_subtree_root(cvs: &[MtHash]) -> MtHash {
    if cvs.len() == 1 {
        return cvs[0].clone();
    }
    let split_len = cvs.len().next_power_of_two() / 2;
    let (left, right) = cvs.split_at(split_len);
    let (left_root, right_root) = get_left_right_subtree_roots!(left, right);
    hazmat::merge_subtrees_non_root(&left_root.0, &right_root.0, hazmat::Mode::Hash).into()
}

/// Recursive helper to find the sibling path. This defines the BLAKE3
/// asymmetric geometry.
fn find_path_recursive(cvs: &[MtHash], target_index: usize) -> Vec<MtHash> {
    if cvs.len() <= 1 {
        return vec![];
    }

    // Find the split point (BLAKE3 rule: largest power of 2 less than N).
    let split_len = cvs.len().next_power_of_two() / 2;
    let (left, right) = cvs.split_at(split_len);

    let (path_from_child, sibling_root) = if target_index < split_len {
        // Target is in the left subtree. Sibling is the root of the right subtree.
        (
            find_path_recursive(left, target_index),
            get_subtree_root(right),
        )
    } else {
        // Target is in the right subtree. Sibling is the root of the left subtree.
        (
            find_path_recursive(right, sub!(target_index, split_len)),
            get_subtree_root(left),
        )
    };

    // Build the path from leaf towards the root (bottom-up).
    let mut path = path_from_child;
    path.push(sibling_root);
    path
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

        let mut current_cv: MtHash = hash_many!(column_values, S);

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
