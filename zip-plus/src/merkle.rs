use blake3::hazmat;
use itertools::Itertools;
use std::{
    fmt,
    fmt::{Display, Formatter},
    ops::Deref,
};
use thiserror::Error;
use zinc_transcript::traits::{ConstTranscribable, GenTranscribable};
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

impl Deref for MtHash {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Display for MtHash {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let blake3_hash: blake3::Hash = self.0.into();
        <blake3::Hash as Display>::fmt(&blake3_hash, f)
    }
}

impl GenTranscribable for MtHash {
    fn read_transcription_bytes_exact(buf: &[u8]) -> Self {
        assert_eq!(buf.len(), HASH_OUT_LEN);
        MtHash(buf.try_into().expect("Invalid buffer length for MtHash"))
    }

    fn write_transcription_bytes_exact(&self, buf: &mut [u8]) {
        assert_eq!(buf.len(), HASH_OUT_LEN);
        buf.copy_from_slice(&self.0);
    }
}

impl ConstTranscribable for MtHash {
    const NUM_BYTES: usize = HASH_OUT_LEN;
}

impl<B> From<B> for MtHash
where
    B: Into<[u8; HASH_OUT_LEN]>,
{
    fn from(b: B) -> Self {
        MtHash(b.into())
    }
}

#[derive(Debug, Clone, Default)]
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

    /// Build a Merkle tree directly from column-major leaf data.
    ///
    /// `columns[j]` is the slice of values that defines leaf `j`'s
    /// pre-image — read sequentially and serialised to a single
    /// Blake3 buffer. Equivalent to [`Self::new`] when
    /// `columns[j][r] == rows[r][j]`, but reads each leaf's input
    /// sequentially instead of as `num_rows` strided per-row
    /// fetches. Used by [`crate::pcs::phase_commit`] after building
    /// the column-major mirror of the encoded matrices, so the
    /// commit phase's leaf-hash pass touches the same memory the
    /// open phase later slices.
    pub fn new_from_columns<S>(columns: &[&[S]]) -> Self
    where
        S: ConstTranscribable + Send + Sync,
    {
        assert!(!columns.is_empty());
        let num_cols = columns.len();
        assert!(num_cols.is_power_of_two());
        // All columns must agree on length so the per-column buffer
        // size is well-defined; otherwise the tree is malformed.
        let col_len = columns[0].len();
        assert!(col_len > 0);
        assert!(
            columns.iter().all(|c| c.len() == col_len),
            "All columns must have the same length"
        );

        let leaves: Vec<MtHash> = cfg_iter!(columns).map(|col| hash_column(col)).collect();
        build_merkle_tree_from_leaves(leaves)
    }

    /// Like [`Self::new_from_column_groups`] but builds leaves directly
    /// from row-major `cw_matrices`, skipping the column-major
    /// intermediate (`cw_columns`) entirely. Per-leaf work goes into a
    /// small (G × batch × num_rows × elem_bytes) scratch buffer that
    /// stays in L1, so we avoid the write-allocate amplification of
    /// scattering into thousands of small `Vec` buffers and the second
    /// pass over the column slab to re-read for hashing.
    ///
    /// Produces the **same Merkle root** as [`Self::new_from_column_groups`]
    /// applied to the column-major mirror of `cw_matrices` (verified by
    /// a round-trip test): the byte layout of each leaf is
    /// `for l in 0..group_size: for m in 0..batch: for r in 0..num_rows:
    /// cw_matrices[m].data[r * codeword_len + g * group_size + l]`.
    pub fn new_from_row_major_grouped<R>(
        cw_matrices: &[crypto_primitives::DenseRowMatrix<R>],
        num_rows: usize,
        codeword_len: usize,
        group_size: usize,
    ) -> Self
    where
        R: ConstTranscribable + Send + Sync,
    {
        assert!(!cw_matrices.is_empty());
        assert!(group_size > 0 && group_size.is_power_of_two());
        assert!(codeword_len.is_power_of_two());
        assert_eq!(codeword_len % group_size, 0);
        let num_leaves = codeword_len / group_size;
        assert!(num_leaves.is_power_of_two());

        let group_indices: Vec<usize> = (0..num_leaves).collect();
        let leaves: Vec<MtHash> = cfg_iter!(group_indices)
            .map(|&g| {
                hash_grouped_leaf_from_row_major(
                    cw_matrices,
                    num_rows,
                    codeword_len,
                    g,
                    group_size,
                )
            })
            .collect();
        build_merkle_tree_from_leaves(leaves)
    }

    /// Like [`Self::new_from_columns`] but groups `group_size` consecutive
    /// columns into a single Merkle leaf. The leaf hash is `H(col_0 || …
    /// || col_{group_size-1})` (each column serialised contiguously, in
    /// the same order it appears in `columns`).
    ///
    /// Trades wire-side cost — each opening must carry all `group_size`
    /// columns of its group so the verifier can recompute the leaf hash
    /// — for fewer Blake3 invocations on the commit side. For small
    /// per-column buffers (e.g. a tall-thin codeword matrix with
    /// `num_rows = 2`), per-hash setup overhead dominates and grouping
    /// can cut commit time multi-fold.
    ///
    /// Constraints: `columns.len()` must be a multiple of `group_size`,
    /// `group_size` must be a power of two, and `columns.len() /
    /// group_size` must be a power of two.
    pub fn new_from_column_groups<S>(columns: &[&[S]], group_size: usize) -> Self
    where
        S: ConstTranscribable + Send + Sync,
    {
        assert!(!columns.is_empty());
        assert!(group_size > 0 && group_size.is_power_of_two());
        let num_cols = columns.len();
        assert!(num_cols.is_power_of_two());
        assert_eq!(
            num_cols % group_size,
            0,
            "columns.len() ({num_cols}) must be a multiple of group_size ({group_size})"
        );
        let num_leaves = num_cols / group_size;
        assert!(num_leaves.is_power_of_two());

        let col_len = columns[0].len();
        assert!(col_len > 0);
        assert!(
            columns.iter().all(|c| c.len() == col_len),
            "All columns must have the same length"
        );

        let group_indices: Vec<usize> = (0..num_leaves).collect();
        let leaves: Vec<MtHash> = cfg_iter!(group_indices)
            .map(|&g| {
                let start = g * group_size;
                let group_slice = &columns[start..start + group_size];
                hash_column_group(group_slice)
            })
            .collect();
        build_merkle_tree_from_leaves(leaves)
    }

    /// Build a Merkle tree over leaves derived from three groups of rows
    /// with potentially different element types. Any group may be empty,
    /// but at least one must be non-empty (and m_cols is taken from the
    /// first non-empty group).
    ///
    /// Each leaf is `H(row0_0[j] || ... || row0_n0[j] || row1_0[j] || ... ||
    /// row2_n2[j])`, i.e. the concatenation of column j across all rows
    /// from all three groups in fixed order (0, 1, 2). Used by
    /// [`crate::pcs::multi_zip::MultiZip3`] to commit two or three
    /// heterogeneous Zip+ instances under a single tree.
    pub fn new_combined_3<S0, S1, S2>(
        rows0: &[&[S0]],
        rows1: &[&[S1]],
        rows2: &[&[S2]],
    ) -> Self
    where
        S0: ConstTranscribable + Send + Sync,
        S1: ConstTranscribable + Send + Sync,
        S2: ConstTranscribable + Send + Sync,
    {
        let m_cols = rows0
            .first()
            .map(|r| r.len())
            .or_else(|| rows1.first().map(|r| r.len()))
            .or_else(|| rows2.first().map(|r| r.len()))
            .expect("new_combined_3 requires at least one non-empty group");
        assert!(m_cols > 0);
        assert!(
            rows0.iter().all(|r| r.len() == m_cols)
                && rows1.iter().all(|r| r.len() == m_cols)
                && rows2.iter().all(|r| r.len() == m_cols),
            "All rows across all three groups must have the same width"
        );
        assert!(m_cols.is_power_of_two());

        let leaves = hash_combined_leaves_3(rows0, rows1, rows2, m_cols);
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
#[allow(clippy::arithmetic_side_effects)]
fn hash_column<S: ConstTranscribable>(values: &[S]) -> MtHash {
    let elem_bytes = S::NUM_BYTES;
    let mut buf = vec![0_u8; values.len() * elem_bytes];
    for (i, v) in values.iter().enumerate() {
        let start = i * elem_bytes;
        v.write_transcription_bytes_exact(&mut buf[start..start + elem_bytes]);
    }
    let mut hasher = blake3::Hasher::new();
    hasher.update(&buf);
    hasher.finalize().into()
}

/// Hash one Merkle leaf directly from the row-major `cw_matrices`,
/// skipping the column-major mirror. Writes the canonical leaf byte
/// layout (matching [`hash_column_group`]) into a small per-leaf
/// scratch buffer, then hashes.
///
/// Reads are cache-friendly: for each `(m, r)` pair we read
/// `group_size` consecutive cells (one cacheline at G ≤ 8). Writes hit
/// `group_size` scattered positions in the L1-resident scratch buffer.
#[allow(clippy::arithmetic_side_effects)]
fn hash_grouped_leaf_from_row_major<R>(
    cw_matrices: &[crypto_primitives::DenseRowMatrix<R>],
    num_rows: usize,
    codeword_len: usize,
    group_idx: usize,
    group_size: usize,
) -> MtHash
where
    R: ConstTranscribable,
{
    let elem_bytes = R::NUM_BYTES;
    let batch = cw_matrices.len();
    let group_start = group_idx * group_size;
    let buf_len = group_size * batch * num_rows * elem_bytes;
    let mut buf = vec![0u8; buf_len];

    // Canonical leaf layout (matches `hash_column_group`):
    //   for l in 0..group_size:
    //     for m in 0..batch:
    //       for r in 0..num_rows:
    //         cell = cw_matrices[m].data[r * codeword_len + group_start + l]
    //
    // Iterate (m, r) outer, l inner so the source reads are contiguous
    // (one cacheline of `group_size` cells per (m, r)). Writes scatter
    // across the small scratch buffer, which stays in L1.
    let col_stride = batch * num_rows * elem_bytes;
    let m_stride = num_rows * elem_bytes;
    for m in 0..batch {
        let m_base = m * m_stride;
        let mat_data = &cw_matrices[m].data;
        for r in 0..num_rows {
            let r_offset = m_base + r * elem_bytes;
            let row_off = r * codeword_len + group_start;
            let cells = &mat_data[row_off..row_off + group_size];
            for (l, cell) in cells.iter().enumerate() {
                let byte_offset = l * col_stride + r_offset;
                cell.write_transcription_bytes_exact(&mut buf[byte_offset..byte_offset + elem_bytes]);
            }
        }
    }

    let mut hasher = blake3::Hasher::new();
    hasher.update(&buf);
    hasher.finalize().into()
}

/// Hash a group of columns into one Merkle leaf.
/// Layout: `H(col_0 || col_1 || … || col_{G-1})` where each column's
/// elements are serialised contiguously in `S`'s transcription format.
/// All columns must have the same length.
#[allow(clippy::arithmetic_side_effects)]
fn hash_column_group<S: ConstTranscribable>(columns: &[&[S]]) -> MtHash {
    assert!(!columns.is_empty());
    let elem_bytes = S::NUM_BYTES;
    let col_len = columns[0].len();
    debug_assert!(
        columns.iter().all(|c| c.len() == col_len),
        "hash_column_group: all columns must have the same length"
    );
    let mut buf = vec![0_u8; columns.len() * col_len * elem_bytes];
    for (g, col) in columns.iter().enumerate() {
        let group_offset = g * col_len * elem_bytes;
        for (i, v) in col.iter().enumerate() {
            let start = group_offset + i * elem_bytes;
            v.write_transcription_bytes_exact(&mut buf[start..start + elem_bytes]);
        }
    }
    let mut hasher = blake3::Hasher::new();
    hasher.update(&buf);
    hasher.finalize().into()
}

/// Construct the leaves of the Merkle tree by hashing each column across all
/// rows.
///
/// For each column, serializes all row elements into a single contiguous byte
/// buffer and feeds it to Blake3 in one `update` call.  This lets Blake3
/// process full 1 KiB chunks with SIMD, which is significantly faster than
/// the per-element `update` approach when columns are tall (many rows).
#[allow(clippy::arithmetic_side_effects)]
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
                row[i].write_transcription_bytes_exact(&mut buf[start..start + elem_bytes]);
            }
            let mut hasher = blake3::Hasher::new();
            hasher.update(&buf);
            hasher.finalize().into()
        })
        .collect()
}

/// Construct Merkle leaves over three groups of heterogeneous rows.
///
/// `leaf_j = H( bytes(rows0_*[j]) || bytes(rows1_*[j]) || bytes(rows2_*[j]) )`.
/// Group order is fixed (0, 1, 2) and within a group rows are concatenated in
/// the order given. Both prover (commit) and verifier (path verification) must
/// use the same convention.
#[allow(clippy::arithmetic_side_effects)]
fn hash_combined_leaves_3<S0, S1, S2>(
    rows0: &[&[S0]],
    rows1: &[&[S1]],
    rows2: &[&[S2]],
    m_cols: usize,
) -> Vec<MtHash>
where
    S0: ConstTranscribable + Send + Sync,
    S1: ConstTranscribable + Send + Sync,
    S2: ConstTranscribable + Send + Sync,
{
    let elem_bytes_0 = S0::NUM_BYTES;
    let elem_bytes_1 = S1::NUM_BYTES;
    let elem_bytes_2 = S2::NUM_BYTES;
    let col_bytes_0 = rows0.len() * elem_bytes_0;
    let col_bytes_1 = rows1.len() * elem_bytes_1;
    let col_bytes_2 = rows2.len() * elem_bytes_2;
    let col_bytes_total = col_bytes_0 + col_bytes_1 + col_bytes_2;

    cfg_into_iter!(0..m_cols)
        .map(|i| {
            let mut buf = vec![0_u8; col_bytes_total];
            let mut offset = 0;
            for row in rows0 {
                row[i].write_transcription_bytes_exact(&mut buf[offset..offset + elem_bytes_0]);
                offset += elem_bytes_0;
            }
            for row in rows1 {
                row[i].write_transcription_bytes_exact(&mut buf[offset..offset + elem_bytes_1]);
                offset += elem_bytes_1;
            }
            for row in rows2 {
                row[i].write_transcription_bytes_exact(&mut buf[offset..offset + elem_bytes_2]);
                offset += elem_bytes_2;
            }
            let mut hasher = blake3::Hasher::new();
            hasher.update(&buf);
            hasher.finalize().into()
        })
        .collect()
}

/// Hash a single combined column (the verifier-side counterpart to
/// `hash_combined_leaves_3`). Layout must match: bytes from group 0, then 1,
/// then 2.
#[allow(clippy::arithmetic_side_effects)]
fn hash_combined_column_3<S0, S1, S2>(col0: &[S0], col1: &[S1], col2: &[S2]) -> MtHash
where
    S0: ConstTranscribable,
    S1: ConstTranscribable,
    S2: ConstTranscribable,
{
    let eb0 = S0::NUM_BYTES;
    let eb1 = S1::NUM_BYTES;
    let eb2 = S2::NUM_BYTES;
    let total = col0.len() * eb0 + col1.len() * eb1 + col2.len() * eb2;
    let mut buf = vec![0_u8; total];
    let mut offset = 0;
    for v in col0 {
        v.write_transcription_bytes_exact(&mut buf[offset..offset + eb0]);
        offset += eb0;
    }
    for v in col1 {
        v.write_transcription_bytes_exact(&mut buf[offset..offset + eb1]);
        offset += eb1;
    }
    for v in col2 {
        v.write_transcription_bytes_exact(&mut buf[offset..offset + eb2]);
        offset += eb2;
    }
    let mut hasher = blake3::Hasher::new();
    hasher.update(&buf);
    hasher.finalize().into()
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
        self.verify_with_leaf(root, hash_column(column_values), leaf_index)
    }

    /// Verifies the proof for a leaf produced by hashing a *group* of
    /// `column_group.len()` consecutive columns (matching
    /// [`MerkleTree::new_from_column_groups`]'s leaf layout). The caller
    /// must pass the columns in the same order they appeared in the
    /// commit-side `columns` slice.
    pub fn verify_grouped<S>(
        &self,
        root: &MtHash,
        column_group: &[&[S]],
        leaf_index: usize,
    ) -> Result<(), MerkleError>
    where
        S: ConstTranscribable,
    {
        self.verify_with_leaf(root, hash_column_group(column_group), leaf_index)
    }

    /// Verifies the proof for a leaf produced by hashing three heterogeneous
    /// column slices (matching `MerkleTree::new_combined_3`'s leaf layout).
    pub fn verify_combined_3<S0, S1, S2>(
        &self,
        root: &MtHash,
        col0: &[S0],
        col1: &[S1],
        col2: &[S2],
        leaf_index: usize,
    ) -> Result<(), MerkleError>
    where
        S0: ConstTranscribable,
        S1: ConstTranscribable,
        S2: ConstTranscribable,
    {
        self.verify_with_leaf(root, hash_combined_column_3(col0, col1, col2), leaf_index)
    }

    fn verify_with_leaf(
        &self,
        root: &MtHash,
        leaf_hash: MtHash,
        leaf_index: usize,
    ) -> Result<(), MerkleError> {
        if leaf_index != self.leaf_index {
            return Err(MerkleError::InvalidLeafIndex(leaf_index));
        }

        let mut current_cv: MtHash = leaf_hash;

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
