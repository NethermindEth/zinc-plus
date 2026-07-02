//! A variable-density (CSR) sparse matrix for R1CS.
//!
//! Shaped after ProveKit's `SparseMatrix`
//! (`provekit/provekit/common/src/sparse_matrix.rs`): row-compressed storage
//! where each row holds a **variable** number of `(column, value)` entries,
//! sorted by column. This is the natural representation for R1CS matrices
//! emitted by a front-end compiler, and unlike
//! [`crypto_primitives::SparseMatrix`] (fixed number of non-zeros per row) it
//! does not force short rows to be padded with filler entries.
//!
//! We deliberately mirror only the *shape*, not ProveKit's storage
//! optimizations: there is no interner (values are stored inline) and no
//! delta-encoded serialization.
//!
//! # Layout status
//!
//! Step R1 of `docs/r1cs-frontend-plan.md`: this type is defined and
//! unit-tested standalone; [`R1csInstance`](crate::r1cs_frontend::R1csInstance)
//! is switched over to it in step R2 (replacing the fixed-density matrix + the
//! generic matrix-entry parameter).

use crypto_primitives::PrimeField;
use zinc_utils::{add, sub};

/// A sparse matrix stored in compressed-sparse-row (CSR) form.
///
/// `entries[row_starts[r]..row_starts[r+1]]` are row `r`'s `(column, value)`
/// pairs, kept sorted by column with no duplicate columns. Generic over the
/// value type `T` as a plain container; R1CS instantiates it at the field `F`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SparseMatrix<T> {
    /// Number of rows.
    pub num_rows: usize,
    /// Number of columns.
    pub num_cols: usize,
    /// CSR row offsets into `entries`; length `num_rows + 1`, with
    /// `row_starts[num_rows] == entries.len()`.
    row_starts: Vec<usize>,
    /// `(column, value)` pairs, grouped by row and sorted by column within each
    /// row.
    entries: Vec<(usize, T)>,
}

impl<T> SparseMatrix<T> {
    /// Build a matrix from per-row `(column, value)` lists.
    ///
    /// Each row is sorted by column. The number of rows is the number of lists
    /// consumed; `num_cols` is taken as given.
    ///
    /// # Panics
    /// - if any column index is `>= num_cols`;
    /// - if a row contains a duplicate column (callers must pre-merge, e.g. sum
    ///   coefficients on the same witness, exactly as ProveKit's
    ///   `canonicalize_terms` does before insertion).
    pub fn from_rows(num_cols: usize, rows: impl IntoIterator<Item = Vec<(usize, T)>>) -> Self {
        let mut row_starts = vec![0usize];
        let mut entries: Vec<(usize, T)> = Vec::new();

        for mut row in rows {
            row.sort_by_key(|(col, _)| *col);
            for pair in row.windows(2) {
                assert_ne!(
                    pair[0].0, pair[1].0,
                    "duplicate column {} in a row (pre-merge before building)",
                    pair[0].0
                );
            }
            if let Some((last_col, _)) = row.last() {
                assert!(
                    *last_col < num_cols,
                    "column {last_col} out of bounds (num_cols = {num_cols})"
                );
            }
            entries.extend(row);
            row_starts.push(entries.len());
        }

        let num_rows = sub!(row_starts.len(), 1);
        Self {
            num_rows,
            num_cols,
            row_starts,
            entries,
        }
    }

    /// Total number of stored (non-zero) entries.
    pub fn num_entries(&self) -> usize {
        self.entries.len()
    }

    /// Iterate row `r`'s `(column, &value)` entries, sorted by column.
    ///
    /// # Panics
    /// If `r >= num_rows`.
    pub fn iter_row(&self, r: usize) -> impl Iterator<Item = (usize, &T)> {
        let start = self.row_starts[r];
        let end = self.row_starts[add!(r, 1)];
        self.entries[start..end]
            .iter()
            .map(|(col, val)| (*col, val))
    }

    /// Iterate all `(row, column, &value)` entries in row-major, column-sorted
    /// order.
    pub fn iter(&self) -> impl Iterator<Item = (usize, usize, &T)> {
        (0..self.num_rows).flat_map(move |r| self.iter_row(r).map(move |(col, val)| (r, col, val)))
    }
}

impl<F: PrimeField> SparseMatrix<F> {
    /// Right-multiply by a vector: returns `M z` with
    /// `(Mz)[i] = \sum_{(j, v) \in \text{row } i} v \cdot z[j]`.
    ///
    /// # Panics
    /// If `z.len() != num_cols`.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn mul_vector(&self, z: &[F], cfg: &F::Config) -> Vec<F> {
        assert_eq!(
            z.len(),
            self.num_cols,
            "vector length {} must equal num_cols {}",
            z.len(),
            self.num_cols
        );
        (0..self.num_rows)
            .map(|r| {
                let mut acc = F::zero_with_cfg(cfg);
                for (col, val) in self.iter_row(r) {
                    acc += &(val.clone() * &z[col]);
                }
                acc
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crypto_bigint::U64;
    use crypto_primitives::{
        FromWithConfig, crypto_bigint_monty::MontyField, crypto_bigint_uint::Uint,
    };

    #[test]
    fn from_rows_builds_csr_and_iterates_sorted() {
        // 3x4 matrix; row 0 given unsorted, row 1 empty, row 2 single entry.
        let m = SparseMatrix::from_rows(
            4,
            vec![vec![(2usize, 20i64), (0, 10)], vec![], vec![(3, 30)]],
        );
        assert_eq!(m.num_rows, 3);
        assert_eq!(m.num_cols, 4);
        assert_eq!(m.num_entries(), 3);

        let row0: Vec<_> = m.iter_row(0).map(|(c, v)| (c, *v)).collect();
        assert_eq!(row0, vec![(0, 10), (2, 20)], "row 0 must be column-sorted");
        assert_eq!(m.iter_row(1).count(), 0, "row 1 is empty");
        let row2: Vec<_> = m.iter_row(2).map(|(c, v)| (c, *v)).collect();
        assert_eq!(row2, vec![(3, 30)]);

        let all: Vec<_> = m.iter().map(|(r, c, v)| (r, c, *v)).collect();
        assert_eq!(all, vec![(0, 0, 10), (0, 2, 20), (2, 3, 30)]);
    }

    #[test]
    fn empty_matrix_is_well_formed() {
        let m = SparseMatrix::<i64>::from_rows(0, Vec::<Vec<(usize, i64)>>::new());
        assert_eq!(m.num_rows, 0);
        assert_eq!(m.num_entries(), 0);
        assert_eq!(m.iter().count(), 0);
    }

    #[test]
    #[should_panic(expected = "duplicate column")]
    fn from_rows_rejects_duplicate_columns() {
        let _ = SparseMatrix::from_rows(4, vec![vec![(1usize, 1i64), (1, 2)]]);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn from_rows_rejects_out_of_bounds_column() {
        let _ = SparseMatrix::from_rows(2, vec![vec![(5usize, 1i64)]]);
    }

    #[test]
    fn mul_vector_matches_hand_computation() {
        const L: usize = U64::LIMBS * 3;
        type F = MontyField<L>;
        let cfg = F::make_cfg(&Uint::from(0xFFFF_FFFF_FFFF_FFC5_u64)).expect("prime");
        let f = |x: i64| F::from_with_cfg(&x, &cfg);

        // 2x3 matrix: row0 = 3*z0 + 5*z2, row1 = 7*z1.
        let m = SparseMatrix::from_rows(3, vec![vec![(0usize, f(3)), (2, f(5))], vec![(1, f(7))]]);
        let z = vec![f(2), f(4), f(6)];
        // row0: 3*2 + 5*6 = 36 ; row1: 7*4 = 28.
        assert_eq!(m.mul_vector(&z, &cfg), vec![f(36), f(28)]);
    }
}
