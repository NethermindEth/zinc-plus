use blake3::{CHUNK_LEN, Hash, Hasher};
use blake3::hazmat::{ChainingValue, HasherExt, merge_subtrees_non_root, merge_subtrees_root, Mode};

/// Represents a proof for a single element within the Blake3 Merkle tree.
/// The proof consists of the data required to reconstruct a leaf hash (a chunk)
/// and the sibling path to prove that chunk's inclusion in the root.
pub struct Blake3MerkleProof {
    /// The original element data being proven.
    pub leaf_element_data: Vec<u8>,
    /// The full 1024-byte chunk (or less if it's the last one) that contains the element.
    pub chunk_bytes: Vec<u8>,
    /// The index of the chunk in the leaf layer of the tree (0, 1, 2,...).
    pub chunk_index: u64,
    /// The path of sibling chaining values from the leaf chunk up to the root.
    pub sibling_path: Vec<ChainingValue>,
}

impl Blake3MerkleProof {
    /// Verifies the proof against a known root hash.
    ///
    /// This function performs two main steps:
    /// 1. Reconstructs the leaf hash (ChainingValue) of the chunk containing the element.
    /// 2. Walks up the tree using the sibling path, hashing at each level, to re-calculate the root hash.
    ///
    /// Returns `true` if the calculated root hash matches the provided one.
    pub fn verify(&self, root: &Hash) -> bool {
        // 1. Reconstruct the leaf hash for the chunk containing the element.
        let mut chunk_hasher = Hasher::new();
        // The input offset is critical for ensuring the hash is correct for its position in the stream.
        chunk_hasher.set_input_offset(self.chunk_index * CHUNK_LEN as u64);
        chunk_hasher.update(&self.chunk_bytes);
        let mut current_cv = chunk_hasher.finalize_non_root();

        // 2. Walk up the tree, merging with siblings to recalculate the root.
        // The verification logic must perfectly mirror the tree's asymmetric construction.
        let mut current_chunk_index = self.chunk_index as usize;
        let mut total_chunks_in_subtree = Blake3MerkleTree::get_total_chunks(&self.leaf_element_data, &self.chunk_bytes);

        for sibling_cv in &self.sibling_path {
            // Determine the split point according to BLAKE3's rules.
            // The left subtree's size is the largest power of two less than or equal to the total number of chunks.
            let split_len = total_chunks_in_subtree.next_power_of_two() / 2;

            if split_len == 0 { // Should not happen in a valid proof path
                return false;
            }

            if current_chunk_index < split_len {
                // Our node is in the left subtree; the sibling is the root of the right subtree.
                current_cv = merge_subtrees_non_root(&current_cv, sibling_cv, Mode::Hash);
                total_chunks_in_subtree = split_len;
            } else {
                // Our node is in the right subtree; the sibling is the root of the left subtree.
                current_cv = merge_subtrees_non_root(sibling_cv, &current_cv, Mode::Hash);
                total_chunks_in_subtree -= split_len;
                current_chunk_index -= split_len;
            }
        }

        // The final merge operation must use `merge_subtrees_root` if the tree has more than one chunk.
        // In our simplified logic, the final `current_cv` should represent one of the two CVs that form the root.
        // However, our proof path gives us the *other* CV. We perform one final merge to get the root.
        let final_hash = if self.sibling_path.is_empty() {
            // This is a single-chunk tree, the root is just the chunk hash with the ROOT flag.
            let mut root_hasher = Hasher::new();
            root_hasher.update(&self.chunk_bytes);
            root_hasher.finalize()
        } else {
            // The last sibling in the path is the one we merge with to get the root.
            // We need to determine the final left/right order.
            let total_chunks = Blake3MerkleTree::get_total_chunks(&self.leaf_element_data, &self.chunk_bytes);
            let final_split = total_chunks.next_power_of_two() / 2;
            let final_sibling = self.sibling_path.last().unwrap();

            if self.chunk_index < final_split as u64 {
                 merge_subtrees_root(&current_cv, final_sibling, Mode::Hash).into()
            } else {
                 merge_subtrees_root(final_sibling, &current_cv, Mode::Hash).into()
            }
        };

        &final_hash == root
    }
}

/// A Merkle tree built using BLAKE3's native internal tree structure.
pub struct Blake3MerkleTree {
    root: Hash,
    /// All layers of the tree, from leaves to the two CVs that form the root.
    layers: Vec<Vec<ChainingValue>>,
    /// The serialized input data, required for generating proofs.
    serialized_data: Vec<u8>,
    /// The byte start and end positions of each original element within `serialized_data`.
    element_positions: Vec<(usize, usize)>,
}

impl Blake3MerkleTree {
    /// Constructs a new Blake3MerkleTree from a slice of string elements.
    pub fn new(elements: &[&str]) -> Self {
        // 1. Serialize all elements into a single byte vector.
        // We use a simple length-prefixing format (u32 length + data).
        let mut serialized_data = Vec::new();
        let mut element_positions = Vec::with_capacity(elements.len());
        for &element in elements {
            let start = serialized_data.len();
            let len_bytes = (element.len() as u32).to_le_bytes();
            serialized_data.extend_from_slice(&len_bytes);
            serialized_data.extend_from_slice(element.as_bytes());
            let end = serialized_data.len();
            element_positions.push((start, end));
        }

        // 2. Split the serialized data into 1024-byte chunks.
        let chunks: Vec<&[u8]> = serialized_data.chunks(CHUNK_LEN).collect();
        if chunks.is_empty() {
            let root = blake3::hash(&[]);
            return Self { root, layers: vec![], serialized_data, element_positions };
        }

        // 3. Hash each chunk to get the leaf layer (layer 0) of ChainingValues.
        let leaf_cvs: Vec<ChainingValue> = chunks
           .iter()
           .enumerate()
           .map(|(i, chunk_data)| {
                let mut hasher = Hasher::new();
                hasher.set_input_offset((i * CHUNK_LEN) as u64);
                hasher.update(chunk_data);
                hasher.finalize_non_root()
            })
           .collect();

        // 4. Recursively build parent layers according to BLAKE3's asymmetric structure.
        let mut layers = vec![leaf_cvs];
        while layers.last().unwrap().len() > 2 {
            let prev_layer = layers.last().unwrap();
            let next_layer = Self::build_parent_layer(prev_layer);
            layers.push(next_layer);
        }

        // 5. Compute the final root hash from the top-level ChainingValues.
        let top_cvs = layers.last().unwrap();
        let root = if top_cvs.len() == 1 {
            // This case handles a single-chunk input.
            let mut hasher = Hasher::new();
            hasher.update(chunks);
            hasher.finalize()
        } else {
            merge_subtrees_root(&top_cvs[0], &top_cvs[1], Mode::Hash).into()
        };

        Self { root, layers, serialized_data, element_positions }
    }

    /// Helper function to build one parent layer from a child layer.
    fn build_parent_layer(child_cvs: &[ChainingValue]) -> Vec<ChainingValue> {
        child_cvs
           .chunks(2)
           .map(|pair| {
                if pair.len() == 2 {
                    merge_subtrees_non_root(&pair[0], &pair[1], Mode::Hash)
                } else {
                    assert!(pair.len() == 1, "!!!");
                    pair[0]
                }
            })
           .collect()
    }

    /// Recursively merges a slice of ChainingValues into a single root CV for that subtree.
    fn get_subtree_root(cvs: &[ChainingValue]) -> ChainingValue {
        if cvs.len() == 1 {
            return cvs[0];
        }
        let split_len = cvs.len().next_power_of_two() / 2;
        let (left, right) = cvs.split_at(split_len);
        let left_root = Self::get_subtree_root(left);
        let right_root = Self::get_subtree_root(right);
        merge_subtrees_non_root(&left_root, &right_root, Mode::Hash)
    }

    /// Generates a Merkle proof for the element at the given index.
    pub fn prove(&self, element_index: usize) -> Option<Blake3MerkleProof> {
        if element_index >= self.element_positions.len() {
            return None;
        }

        // Find which chunk the element belongs to.
        let (element_start, element_end) = self.element_positions[element_index];
        let chunk_index = (element_start / CHUNK_LEN) as u64;

        let chunk_start = chunk_index as usize * CHUNK_LEN;
        let chunk_end = (chunk_start + CHUNK_LEN).min(self.serialized_data.len());

        let leaf_element_data = self.serialized_data[element_start..element_end].to_vec();
        let chunk_bytes = self.serialized_data[chunk_start..chunk_end].to_vec();

        // Find the sibling path recursively.
        let sibling_path = Self::find_path_recursive(&self.layers, chunk_index as usize);

        Some(Blake3MerkleProof {
            leaf_element_data,
            chunk_bytes,
            chunk_index,
            sibling_path,
        })
    }

    /// Recursive helper to find the sibling path for a given leaf index.
    fn find_path_recursive(cvs: &[ChainingValue], target_index: usize) -> Vec<ChainingValue> {
        if cvs.len() <= 1 {
            return vec![];
        }

        // Find the split point for this level of the tree.
        let split_len = cvs.len().next_power_of_two() / 2;
        let (left, right) = cvs.split_at(split_len);

        let (path_from_child, sibling_root) = if target_index < split_len {
            // Target is in the left subtree.
            (Self::find_path_recursive(left, target_index), Self::get_subtree_root(right))
        } else {
            // Target is in the right subtree.
            (Self::find_path_recursive(right, target_index - split_len), Self::get_subtree_root(left))
        };

        let mut path = path_from_child;
        path.push(sibling_root);
        path
    }

    /// A helper to get the total number of chunks from proof data.
    /// This is a bit of a hack for verification without passing the whole tree.
    /// A real implementation might include the total leaves count in the proof itself.
    fn get_total_chunks(leaf_element_data: &[u8], chunk_bytes: &[u8]) -> usize {
        // This is a simplified estimation. For a robust system,
        // the total number of leaves should be part of the proof.
        let prefix_len = 4; // u32 length prefix
        let total_len = prefix_len + leaf_element_data.len() + chunk_bytes.len();
        (total_len + CHUNK_LEN -1) / CHUNK_LEN
    }

    pub fn root(&self) -> &Hash {
        &self.root
    }
}
