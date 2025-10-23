use crate::{add, mul, sub, traits::ConstTranscribable};
use ark_std::cfg_chunks;
use blake3::{hazmat, hazmat::HasherExt};
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
    /// First vector is leaves, last vector is root
    layers: Vec<Vec<MtHash>>,
    /// How many consequent leaf hashes do we use to encode one element?
    num_leaves_per_element: usize,
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
        let n_cols = rows[0].len();
        // Calculate how many leaves (chunks) needed per column, rounded up to next
        // power of 2 to ensure proper BLAKE3 subtree alignment
        let leaves_needed = mul!(S::NUM_BYTES, n_cols).div_ceil(blake3::CHUNK_LEN);
        let num_leaves_per_element = leaves_needed.next_power_of_two();

        let leaves = hash_leaves(rows, row_width, num_leaves_per_element);
        build_merkle_tree_from_leaves(leaves, num_leaves_per_element)
    }

    pub fn root(&self) -> MtHash {
        self.layers
            .last()
            .expect("Merkle tree must have at least one layer")
            .first()
            .cloned()
            .expect("Merkle tree must have a root")
    }

    /// Generates a Merkle proof for the element (column) at the given index.
    /// The proof starts from the subtree root covering all leaves for this
    /// element. Uses pre-computed layer values for O(1) sibling lookups.
    pub fn prove(&self, element_index: usize) -> Result<MerkleProof, MerkleError> {
        let leaf_count = self.layers[0].len();
        let num_elements = leaf_count / self.num_leaves_per_element;

        if element_index >= num_elements {
            return Err(MerkleError::InvalidLeafIndex(element_index));
        }

        // Find the layer index for the subtree size corresponding to
        // num_leaves_per_element Layer i contains subtrees of size 2^i, so we
        // need layer log2(num_leaves_per_element)
        let subtree_layer_idx = self.num_leaves_per_element.trailing_zeros() as usize;

        // Calculate the sibling path starting from the subtree root.
        // The element's subtree is at index element_index in the subtree layer.
        let siblings = find_path_with_layers(element_index, subtree_layer_idx, &self.layers);

        Ok(MerkleProof {
            element_index,
            num_elements,
            num_leaves_per_element: self.num_leaves_per_element,
            siblings,
        })
    }
}

fn hash_leaves<S>(rows: &[&[S]], m_cols: usize, num_leaves_per_element: usize) -> Vec<MtHash>
where
    S: ConstTranscribable + Send + Sync,
{
    let transcription_len = S::NUM_BYTES;
    assert!(
        transcription_len <= blake3::CHUNK_LEN,
        "Transcription length of a single element must fit within BLAKE3 chunk length"
    );

    // Verify that leaves from each column will reside in the same BLAKE3 subtree.
    //
    // BLAKE3 tree structure:
    // - Uses asymmetric binary tree with splitting rule: next_power_of_two / 2
    // - A contiguous range of leaves forms a valid subtree only if properly aligned
    //
    // For each column producing `num_leaves_per_element` leaves:
    // - Column i produces leaves at indices [i * num_leaves_per_element, (i+1) *
    //   num_leaves_per_element)
    // - These leaves must form a contiguous subtree in BLAKE3's structure
    //
    // Sufficient condition for same-subtree guarantee:
    // 1. num_leaves_per_element must be a power of 2
    //    - This ensures each column's leaves can form a perfect binary subtree
    // 2. Each column's starting index must be aligned to num_leaves_per_element
    //    - Column i starts at index (i * num_leaves_per_element)
    //    - This is automatically aligned since both are multiples
    // 3. Total number of leaves must be a power of 2
    //    - This ensures the overall tree structure is valid
    //
    // Why power-of-2 is necessary:
    // - BLAKE3's splitting creates balanced subtrees for power-of-2 ranges
    // - Non-power-of-2 ranges get split asymmetrically, potentially spanning
    //   multiple subtrees
    // - Example: 3 leaves might split as 2+1, where they're in different subtrees
    //   at some level

    assert!(
        num_leaves_per_element.is_power_of_two(),
        "num_leaves_per_element must be a power of 2 to ensure all leaves from each column \
         reside in the same BLAKE3 subtree. Got: {}",
        num_leaves_per_element
    );

    let total_leaves = m_cols * num_leaves_per_element;
    assert!(
        total_leaves.is_power_of_two(),
        "Total number of leaves (m_cols * num_leaves_per_element) must be a power of 2. \
         Got: {} columns * {} leaves/column = {} total leaves",
        m_cols,
        num_leaves_per_element,
        total_leaves
    );

    // Additional verification: each column's starting index is naturally aligned
    // because start_idx = column_idx * num_leaves_per_element, and
    // num_leaves_per_element is a power of 2, so (start_idx %
    // num_leaves_per_element) == 0 for all columns. This alignment ensures that
    // the range [start_idx, start_idx + num_leaves_per_element) forms a valid
    // BLAKE3 subtree boundary.

    let values_per_chunk = blake3::CHUNK_LEN / transcription_len;

    // Helper to hash a buffer efficiently
    #[inline]
    fn hash_chunk(buf: &[u8]) -> MtHash {
        let mut hasher = blake3::Hasher::new();
        hasher.update(buf);
        MtHash::from(hasher.finalize_non_root())
    }

    (0..m_cols)
        .flat_map(|i| {
            // Hash the i-th column across all rows into chunks
            let mut buf = [0u8; blake3::CHUNK_LEN];

            // Produce leaves for this column
            let num_chunks = rows.len().div_ceil(values_per_chunk);
            let mut leaves = Vec::with_capacity(num_leaves_per_element.max(num_chunks));

            for chunk_idx in 0..num_chunks {
                let chunk_start = chunk_idx * values_per_chunk;
                let chunk_end = (chunk_start + values_per_chunk).min(rows.len());
                let chunk_size = chunk_end - chunk_start;

                // Write values into chunk buffer
                for (idx, row) in rows[chunk_start..chunk_end].iter().enumerate() {
                    let start = idx * transcription_len;
                    let end = start + transcription_len;
                    row[i].write_transcription_bytes(&mut buf[start..end]);
                }

                // Zero out the rest of the buffer if this chunk is partial
                if chunk_size < values_per_chunk {
                    let used_bytes = chunk_size * transcription_len;
                    buf[used_bytes..].fill(0);
                }

                leaves.push(hash_chunk(&buf));
            }

            // Pad with zero-filled chunks if necessary to reach num_leaves_per_element
            if leaves.len() < num_leaves_per_element {
                buf.fill(0);
                let zero_hash = hash_chunk(&buf);
                leaves.resize(num_leaves_per_element, zero_hash);
            }

            assert_eq!(
                leaves.len(),
                num_leaves_per_element,
                "Expected {} leaves per column, got {}",
                num_leaves_per_element,
                leaves.len()
            );

            leaves
        })
        .collect()
}

fn build_merkle_tree_from_leaves(leaves: Vec<MtHash>, num_leaves_per_element: usize) -> MerkleTree {
    let n = leaves.len();

    if n == 0 {
        return MerkleTree {
            layers: vec![vec![blake3::hash(&[]).into()]],
            num_leaves_per_element,
        };
    }
    assert!(
        n.is_power_of_two(),
        "Number of leaves must be a power of two"
    );

    if n == 1 {
        return MerkleTree {
            layers: vec![leaves],
            num_leaves_per_element,
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
        let current_layer = cfg_chunks!(prev_layer, 2)
            .map(|chunk| {
                let [left, right] = chunk else {
                    unreachable!();
                };
                if is_root_layer {
                    hazmat::merge_subtrees_root(&left.0, &right.0, hazmat::Mode::Hash).into()
                } else {
                    hazmat::merge_subtrees_non_root(&left.0, &right.0, hazmat::Mode::Hash).into()
                }
            })
            .collect();

        layers.push(current_layer);
    }

    MerkleTree {
        layers,
        num_leaves_per_element,
    }
}

/// Finds the sibling path starting from a subtree root at the given layer.
/// This enables proving subtrees rather than individual leaves.
///
/// # Arguments
/// * `subtree_index` - Index of the subtree in its layer
/// * `start_layer_idx` - Layer index where the subtree root resides
/// * `layers` - All tree layers
///
/// # Returns
/// Siblings in bottom-up order (subtree level to root)
#[allow(clippy::arithmetic_side_effects)] // Using intentionally, overflow isn't possible
fn find_path_with_layers(
    subtree_index: usize,
    start_layer_idx: usize,
    layers: &[Vec<MtHash>],
) -> Vec<MtHash> {
    let mut siblings = Vec::new();
    let mut layer_idx = start_layer_idx;
    let mut current_index = subtree_index;

    // Traverse from the subtree layer up to the root
    while layer_idx < layers.len() - 1 {
        let current_layer = &layers[layer_idx];

        // Determine if current node is left (even) or right (odd) child
        let is_left_child = current_index % 2 == 0;

        if is_left_child {
            // Left child, sibling is on the right
            let sibling_index = current_index + 1;
            if sibling_index < current_layer.len() {
                siblings.push(current_layer[sibling_index].clone());
            } else {
                // No right sibling, we're at the rightmost position
                // This should only happen if we're at the root already
                debug_assert_eq!(current_layer.len(), 1);
                break;
            }
        } else {
            // Right child, sibling is on the left
            let sibling_index = current_index - 1;
            siblings.push(current_layer[sibling_index].clone());
        }

        // Move to parent in next layer
        current_index /= 2;
        layer_idx += 1;
    }

    siblings
}

/// Computes the subtree root for a column by hashing its values into leaves
/// and merging them following BLAKE3's tree structure.
///
/// This mirrors the process in `hash_leaves` but for a single column during
/// verification.
fn compute_subtree_root_from_column<S>(
    column_values: &[S],
    num_leaves_per_element: usize,
) -> Result<MtHash, MerkleError>
where
    S: ConstTranscribable,
{
    let transcription_len = S::NUM_BYTES;

    if transcription_len > blake3::CHUNK_LEN {
        return Err(MerkleError::InvalidMerkleProof(
            "Element transcription length exceeds BLAKE3 chunk length".to_owned(),
        ));
    }

    // Hash the column values into chunks, producing leaves
    let mut buf = [0_u8; blake3::CHUNK_LEN];
    let values_per_chunk = blake3::CHUNK_LEN / transcription_len;

    // Helper to hash a buffer efficiently
    #[inline]
    fn hash_chunk(buf: &[u8]) -> MtHash {
        let mut hasher = blake3::Hasher::new();
        hasher.update(buf);
        MtHash::from(hasher.finalize_non_root())
    }

    let num_chunks = column_values.len().div_ceil(values_per_chunk);
    let mut leaves = Vec::with_capacity(num_leaves_per_element.max(num_chunks));

    for chunk_idx in 0..num_chunks {
        let chunk_start = chunk_idx * values_per_chunk;
        let chunk_end = (chunk_start + values_per_chunk).min(column_values.len());
        let chunk_size = chunk_end - chunk_start;

        // Write values into the chunk buffer
        for (idx, value) in column_values[chunk_start..chunk_end].iter().enumerate() {
            let start = idx * transcription_len;
            let end = start + transcription_len;
            value.write_transcription_bytes(&mut buf[start..end]);
        }

        // Zero out the rest of the buffer if this chunk is partial
        if chunk_size < values_per_chunk {
            let used_bytes = chunk_size * transcription_len;
            buf[used_bytes..].fill(0);
        }

        leaves.push(hash_chunk(&buf));
    }

    // Pad with zero-filled chunks if necessary to reach num_leaves_per_element
    if leaves.len() < num_leaves_per_element {
        buf.fill(0);
        let zero_hash = hash_chunk(&buf);
        leaves.resize(num_leaves_per_element, zero_hash);
    }

    // Merge the leaves into a subtree root following BLAKE3 structure
    if leaves.len() == 1 {
        Ok(leaves[0].clone())
    } else {
        Ok(merge_leaves_to_subtree_root(&leaves))
    }
}

/// Recursively merges leaves into a subtree root following BLAKE3's asymmetric
/// tree structure.
fn merge_leaves_to_subtree_root(leaves: &[MtHash]) -> MtHash {
    if leaves.len() == 1 {
        return leaves[0].clone();
    }

    if leaves.len() == 2 {
        return hazmat::merge_subtrees_non_root(&leaves[0].0, &leaves[1].0, hazmat::Mode::Hash)
            .into();
    }

    // BLAKE3 split rule: next_power_of_two / 2
    let split_len = leaves.len().next_power_of_two() / 2;
    let (left, right) = leaves.split_at(split_len);

    let left_root = merge_leaves_to_subtree_root(left);
    let right_root = merge_leaves_to_subtree_root(right);

    hazmat::merge_subtrees_non_root(&left_root.0, &right_root.0, hazmat::Mode::Hash).into()
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MerkleProof {
    /// Index of the element (column) being proven
    pub element_index: usize,
    /// Total number of elements (columns) in the tree
    pub num_elements: usize,
    /// Number of leaves per element (defines subtree size)
    pub num_leaves_per_element: usize,
    /// The path of sibling chaining values from subtree root to tree root
    /// (bottom-up order).
    pub siblings: Vec<MtHash>,
}

impl MerkleProof {
    pub fn new(
        element_index: usize,
        num_elements: usize,
        num_leaves_per_element: usize,
        siblings: Vec<MtHash>,
    ) -> Self {
        assert!(element_index < num_elements, "Element index out of bounds");
        assert!(
            num_leaves_per_element.is_power_of_two(),
            "num_leaves_per_element must be a power of 2"
        );
        Self {
            element_index,
            num_elements,
            num_leaves_per_element,
            siblings,
        }
    }

    /// Verifies the proof against a known root hash and the claimed element
    /// (column) data.
    ///
    /// This verification:
    /// 1. Computes the subtree root from the column values (hashing into
    ///    num_leaves_per_element leaves)
    /// 2. Walks up from the subtree root to the tree root using the sibling
    ///    path
    pub fn verify<S>(
        &self,
        root: &MtHash,
        column_values: &[S],
        element_index: usize,
    ) -> Result<(), MerkleError>
    where
        S: ConstTranscribable,
    {
        if element_index != self.element_index {
            return Err(MerkleError::InvalidLeafIndex(element_index));
        }

        // Step 1: Compute the subtree root for this element's leaves
        // This recomputes the BLAKE3 hashing that produced num_leaves_per_element
        // leaves and then merges them into a single subtree root
        let mut current_hash =
            compute_subtree_root_from_column(column_values, self.num_leaves_per_element)?;

        // Handle edge case: single element tree
        if self.num_elements == 1 {
            if self.element_index == 0 && self.siblings.is_empty() {
                // The root is just the subtree root itself
                if &current_hash != root {
                    return Err(MerkleError::InvalidRootHash);
                }
                return Ok(());
            } else {
                return Err(MerkleError::InvalidMerkleProof(
                    "Single element Merkle proof is invalid".to_owned(),
                ));
            }
        }

        // Step 2: Walk up from subtree root to tree root
        // Determine path directions starting from the subtree layer
        let subtree_layer_idx = self.num_leaves_per_element.trailing_zeros() as usize;
        let total_leaves = self.num_elements * self.num_leaves_per_element;
        let root_layer_idx = total_leaves.trailing_zeros() as usize;
        let num_layers_to_traverse = root_layer_idx - subtree_layer_idx;

        if self.siblings.len() != num_layers_to_traverse {
            return Err(MerkleError::InvalidMerklePathLength {
                expected: num_layers_to_traverse,
                actual: self.siblings.len(),
            });
        }

        // Compute path directions from the element's position in its layer
        let mut current_index = self.element_index;
        let mut is_left_directions = Vec::with_capacity(num_layers_to_traverse);

        for _ in 0..num_layers_to_traverse {
            is_left_directions.push(current_index % 2 == 0);
            current_index /= 2;
        }

        // Walk up the tree (all merges are non-root until the last one)
        let mut siblings_iter = self.siblings.iter();
        let mut directions_iter = is_left_directions.iter();

        // All merges except the last use non-root merge
        for _ in 0..num_layers_to_traverse.saturating_sub(1) {
            let sibling = siblings_iter.next().unwrap();
            let is_left = *directions_iter.next().unwrap();

            current_hash = if is_left {
                hazmat::merge_subtrees_non_root(&current_hash.0, &sibling.0, hazmat::Mode::Hash)
                    .into()
            } else {
                hazmat::merge_subtrees_non_root(&sibling.0, &current_hash.0, hazmat::Mode::Hash)
                    .into()
            };
        }

        // Final merge to root uses root merge function
        if let (Some(last_sibling), Some(&is_left)) = (siblings_iter.next(), directions_iter.next())
        {
            let final_hash: MtHash = if is_left {
                hazmat::merge_subtrees_root(&current_hash.0, &last_sibling.0, hazmat::Mode::Hash)
                    .into()
            } else {
                hazmat::merge_subtrees_root(&last_sibling.0, &current_hash.0, hazmat::Mode::Hash)
                    .into()
            };

            if &final_hash != root {
                return Err(MerkleError::InvalidRootHash);
            }
        } else {
            // No siblings means we're already at root (single element case, handled above)
            unreachable!("Should have been handled by single element case");
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
