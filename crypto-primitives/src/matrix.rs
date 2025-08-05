/// A matrix, rectangular table of values
pub trait Matrix<T> {
    /// Number of rows in this matrix
    fn num_rows(&self) -> usize;

    /// Number of columns in this matrix
    fn num_cols(&self) -> usize;
}

/// Sparse matrix is a matrix with a fixed number non-zero of elements per row
#[derive(Clone, Debug)]
pub struct SparseMatrix<T> {
    /// Number of rows
    pub num_rows: usize,

    /// Number of columns
    pub num_cols: usize,

    // TODO(alex): Make variable?
    /// Number of non-zero elements per row
    pub density: usize,

    /// Rows of sparse matrix
    pub cells: Vec<(usize, T)>,
}

impl<T> SparseMatrix<T> {
    /// Iterate over slices, where each slice corresponds to a row
    #[inline]
    pub fn rows(&self) -> impl Iterator<Item = &[(usize, T)]> {
        self.cells.chunks(self.density)
    }
}

impl<T> Matrix<T> for SparseMatrix<T> {
    /// Number of rows
    #[inline]
    fn num_rows(&self) -> usize {
        self.num_rows
    }

    /// Number of columns
    #[inline]
    fn num_cols(&self) -> usize {
        self.num_cols
    }
}

impl<T, T2> From<&SparseMatrix<T2>> for SparseMatrix<T>
where
    T: for<'a> From<&'a T2>,
{
    fn from(other: &SparseMatrix<T2>) -> Self {
        SparseMatrix {
            num_rows: other.num_cols,
            num_cols: other.num_cols,
            density: other.density,
            cells: other
                .cells
                .iter()
                .map(|(idx, val)| (*idx, val.into()))
                .collect(),
        }
    }
}
