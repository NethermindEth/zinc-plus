use blake3::Hash;
// Import necessary hazmat functions
use blake3::hazmat::{ChainingValue, Mode, merge_subtrees_non_root, merge_subtrees_root};

/// Represents a proof for a single element within the Blake3 Merkle tree.
#[derive(Clone)]
pub struct Blake3MerkleProof {
    /// The index of the element in the original list.
    pub element_index: usize,
    /// The total number of elements (leaves) in the tree. Essential for
    /// asymmetric verification.
    pub total_elements: usize,
    /// The path of sibling chaining values (bottom-up order).
    pub sibling_path: Vec<ChainingValue>,
    // Note: The element data itself is usually provided to the verifier
    // alongside the proof, rather than being stored inside it.
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PathDirection {
    Left,
    Right,
}

impl Blake3MerkleProof {
    /// Helper to determine the path directions (leaf to root).
    fn get_path_directions(total_chunks: usize, target_index: usize) -> Vec<PathDirection> {
        let mut path = Vec::new();
        let mut current_size = total_chunks;
        let mut current_index = target_index;

        // Iterate top-down (Root to Leaf) to determine the path based on BLAKE3 rules.
        while current_size > 1 {
            // BLAKE3 split rule: largest power of two less than N (or N/2 if N is power of
            // 2).
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

    // fn merge_subrees_helper(
    //     f: impl Fn(&ChainingValue, &ChainingValue, Mode) -> ChainingValue,
    //     current: &ChainingValue,
    //     other: &ChainingValue,
    //     direction: PathDirection,
    // ) -> ChainingValue {
    //     match direction {
    //         PathDirection::Left => f(current, other, Mode::Hash),
    //         PathDirection::Right => f(other, current, Mode::Hash),
    //     }
    // }

    /// Verifies the proof against a known root hash and the claimed element
    /// data.
    pub fn verify(&self, root: &Hash, claimed_element_data: &[u8]) -> bool {
        // 1. Hash the claimed element data to reconstruct the leaf CV.
        let element_hash = blake3::hash(claimed_element_data);
        let mut current_cv = ChainingValue::from(element_hash);

        // 2. Handle the single element case specifically.
        if self.total_elements == 1 {
            if self.element_index == 0 && self.sibling_path.is_empty() {
                // The root is just the hash of the single element.
                return &element_hash == root;
            } else {
                return false; // Invalid proof structure
            }
        }

        // 3. Determine the path directions (logic unchanged).
        let directions = Self::get_path_directions(self.total_elements, self.element_index);

        if directions.len() != self.sibling_path.len() {
            return false;
        }

        // 4. Walk up the tree (logic remains identical to the corrected PoC).
        let mut path_iter = self.sibling_path.iter().zip(directions.iter());

        // Pop the last element for the ROOT merge.
        let (last_sibling, last_direction) = match path_iter.next_back() {
            Some(pair) => pair,
            None => return false, // Unreachable if total_elements > 1
        };

        // Iterate over intermediate merges (non-root).
        for (sibling_cv, direction) in path_iter {
            let is_left = matches!(direction, PathDirection::Left);
            if is_left {
                current_cv = merge_subtrees_non_root(&current_cv, sibling_cv, Mode::Hash);
            } else {
                current_cv = merge_subtrees_non_root(sibling_cv, &current_cv, Mode::Hash);
            }
        }

        // 5. Final root merge.
        let final_hash: Hash = if matches!(last_direction, PathDirection::Left) {
            merge_subtrees_root(&current_cv, last_sibling, Mode::Hash).into()
        } else {
            merge_subtrees_root(last_sibling, &current_cv, Mode::Hash).into()
        };

        &final_hash == root
    }
}

/// A Merkle tree built using BLAKE3's native internal tree structure.
pub struct Blake3MerkleTree {
    pub root: Hash,
    /// The leaf layer of the tree (ChainingValues of the chunks).
    leaf_cvs: Vec<ChainingValue>,
}

// Assuming the Blake3MerkleTree struct is modified to remove
// serialized_data and element_positions.

impl Blake3MerkleTree {
    pub fn new(elements: &[&str]) -> Self {
        // 1. Hash each element individually to get the leaves.
        let leaf_cvs: Vec<ChainingValue> = elements
            .iter()
            .map(|&element| {
                // BLAKE3 handles the arbitrary size of the element efficiently.
                let hash = blake3::hash(element.as_bytes());

                // Convert the resulting Hash directly into a ChainingValue (CV).
                // This allows us to use these hashes as inputs for the hazmat functions.
                ChainingValue::from(hash)
            })
            .collect();

        if leaf_cvs.is_empty() {
            // Define the root of an empty tree (e.g., hash of empty string)
            return Self {
                root: blake3::hash(&[]),
                leaf_cvs: vec![],
            };
        }

        // 2. Build the tree structure recursively from the leaves.
        let root = if leaf_cvs.len() == 1 {
            // In this design, the root of a single-element tree is just the hash of the
            // element itself.
            Hash::from(leaf_cvs[0])
        } else {
            // The existing recursive logic (compute_root_from_leaves, get_subtree_root)
            // correctly implements the BLAKE3 asymmetric structure using hazmat merges.
            Self::compute_root_from_leaves(&leaf_cvs)
        };

        Self { root, leaf_cvs }
    }

    /// Helper to compute the root from leaves (len > 1). The top-level merge
    /// uses merge_subtrees_root.
    fn compute_root_from_leaves(cvs: &[ChainingValue]) -> Hash {
        assert!(cvs.len() > 1);

        // Find the split point according to BLAKE3 rules.
        let split_len = cvs.len().next_power_of_two() / 2;
        let (left, right) = cvs.split_at(split_len);

        // Recursively get the roots of the subtrees (non-root merges).
        let left_root = Self::get_subtree_root(left);
        let right_root = Self::get_subtree_root(right);

        // The final merge is a root merge.
        merge_subtrees_root(&left_root, &right_root, Mode::Hash).into()
    }

    /// Recursively merges a slice of ChainingValues into a single root CV for
    /// that subtree.
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
        let total_elements = self.leaf_cvs.len();

        // 1. Check if the index is valid.
        if element_index >= total_elements || total_elements == 0 {
            return None;
        }

        // 2. Calculate the sibling path.
        // We use the recursive helper function, passing the list of all leaves
        // (leaf_cvs) and the index of the element we are proving.
        let sibling_path = Self::find_path_recursive(&self.leaf_cvs, element_index);

        // 3. Construct the proof.
        Some(Blake3MerkleProof {
            element_index,
            total_elements,
            sibling_path,
        })
    }

    /// Recursive helper to find the sibling path (remains the same as the
    /// corrected PoC). This defines the BLAKE3 asymmetric geometry.
    fn find_path_recursive(cvs: &[ChainingValue], target_index: usize) -> Vec<ChainingValue> {
        if cvs.len() <= 1 {
            return vec![];
        }

        // Find the split point (BLAKE3 rule: largest power of 2 less than N).
        let split_len = cvs.len().next_power_of_two() / 2;
        let (left, right) = cvs.split_at(split_len);

        let (path_from_child, sibling_root) = if target_index < split_len {
            // Target is in the left subtree. Sibling is the root of the right subtree.
            (
                Self::find_path_recursive(left, target_index),
                Self::get_subtree_root(right),
            )
        } else {
            // Target is in the right subtree. Sibling is the root of the left subtree.
            (
                Self::find_path_recursive(right, target_index - split_len),
                Self::get_subtree_root(left),
            )
        };

        // Build the path from leaf towards the root (bottom-up).
        let mut path = path_from_child;
        path.push(sibling_root);
        path
    }
}
