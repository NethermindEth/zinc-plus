use alloc::vec::Vec;

/// A matrix, rectangular table of values
pub trait Matrix<T> {
    /// Number of rows in this matrix
    fn num_rows(&self) -> usize;

    /// Number of columns in this matrix
    fn num_cols(&self) -> usize;

    fn rows<'a>(&'a self) -> impl Iterator<Item = impl Iterator<Item = (usize, &'a T)>>
    where
        T: 'a;
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

    fn rows<'a>(&'a self) -> impl Iterator<Item = impl Iterator<Item = (usize, &'a T)>>
    where
        T: 'a,
    {
        self.cells
            .chunks(self.density)
            .map(|chunk| chunk.iter().map(|v| (v.0, &v.1)))
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

#[derive(Clone, Debug)]
pub struct DenseRowMatrix<T> {
    /// Number of columns
    pub num_cols: usize,

    /// Rows of dense matrix
    pub rows: Vec<Vec<T>>,
}

impl<T> Matrix<T> for DenseRowMatrix<T> {
    #[inline]
    fn num_rows(&self) -> usize {
        self.rows.len()
    }

    #[inline]
    fn num_cols(&self) -> usize {
        self.num_cols
    }

    fn rows<'a>(&'a self) -> impl Iterator<Item = impl Iterator<Item = (usize, &'a T)>>
    where
        T: 'a,
    {
        self.rows.iter().map(|row| row.iter().enumerate())
    }
}

impl<T> From<Vec<Vec<T>>> for DenseRowMatrix<T> {
    fn from(value: Vec<Vec<T>>) -> Self {
        let num_cols = value.iter().map(|row| row.len()).max().unwrap_or(0);
        DenseRowMatrix {
            num_cols,
            rows: value,
        }
    }
}
