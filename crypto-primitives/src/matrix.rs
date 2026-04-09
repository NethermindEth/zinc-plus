use alloc::vec::Vec;
use core::mem::{ManuallyDrop, MaybeUninit};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// A matrix, rectangular table of values
pub trait Matrix<T> {
    /// Number of rows in this matrix
    fn num_rows(&self) -> usize;

    /// Number of columns in this matrix
    fn num_cols(&self) -> usize;

    fn cells<'a>(&'a self) -> impl Iterator<Item = impl Iterator<Item = (usize, &'a T)>>
    where
        T: 'a;

    fn cells_mut<'a>(
        &'a mut self,
    ) -> impl Iterator<Item = impl Iterator<Item = (usize, &'a mut T)>>
    where
        T: 'a;

    fn is_empty(&self) -> bool {
        self.num_rows() == 0 || self.num_cols() == 0
    }

    #[cfg(feature = "parallel")]
    fn par_cells<'a>(
        &'a self,
    ) -> impl ParallelIterator<Item = impl Iterator<Item = (usize, &'a T)>>
    where
        T: 'a + Send + Sync;

    #[cfg(feature = "parallel")]
    fn par_cells_mut<'a>(
        &'a mut self,
    ) -> impl ParallelIterator<Item = impl Iterator<Item = (usize, &'a mut T)>>
    where
        T: 'a + Send + Sync;
}

/// Sparse matrix is a matrix with a fixed number non-zero of elements per row
#[derive(Clone, Debug, Default, Eq, PartialEq)]
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

    fn cells<'a>(&'a self) -> impl Iterator<Item = impl Iterator<Item = (usize, &'a T)>>
    where
        T: 'a,
    {
        self.cells
            .chunks(self.density)
            .map(|chunk| chunk.iter().map(|v| (v.0, &v.1)))
    }

    fn cells_mut<'a>(&'a mut self) -> impl Iterator<Item = impl Iterator<Item = (usize, &'a mut T)>>
    where
        T: 'a,
    {
        self.cells
            .chunks_mut(self.density)
            .map(|chunk| chunk.iter_mut().map(|v| (v.0, &mut v.1)))
    }

    #[cfg(feature = "parallel")]
    fn par_cells<'a>(&'a self) -> impl ParallelIterator<Item = impl Iterator<Item = (usize, &'a T)>>
    where
        T: 'a + Send + Sync,
    {
        self.cells
            .par_chunks(self.density)
            .map(|chunk| chunk.iter().map(|v| (v.0, &v.1)))
    }

    #[cfg(feature = "parallel")]
    fn par_cells_mut<'a>(
        &'a mut self,
    ) -> impl ParallelIterator<Item = impl Iterator<Item = (usize, &'a mut T)>>
    where
        T: 'a + Send + Sync,
    {
        self.cells
            .par_chunks_mut(self.density)
            .map(|chunk| chunk.iter_mut().map(|v| (v.0, &mut v.1)))
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

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct DenseRowMatrix<T> {
    /// Number of rows
    pub num_rows: usize,

    /// Number of columns
    pub num_cols: usize,

    /// Linearized rows of dense matrix
    pub data: Vec<T>,
}

impl<T: Clone> DenseRowMatrix<T> {
    pub fn uninit(num_rows: usize, num_cols: usize) -> DenseRowMatrix<MaybeUninit<T>> {
        let full_capacity = num_rows
            .checked_mul(num_cols)
            .expect("Overflow in matrix size calculation");
        let mut data = Vec::with_capacity(full_capacity);
        // Safety: It's safe to create an uninitialized vector of `MaybeUninit<T>`
        unsafe { data.set_len(full_capacity) };
        DenseRowMatrix {
            num_rows,
            num_cols,
            data,
        }
    }

    pub fn as_rows(&self) -> impl Iterator<Item = &[T]> {
        self.data.chunks_exact(self.num_cols)
    }

    pub fn as_rows_mut(&mut self) -> impl Iterator<Item = &mut [T]> {
        self.data.chunks_exact_mut(self.num_cols)
    }

    #[cfg(feature = "parallel")]
    pub fn as_rows_par(&self) -> impl ParallelIterator<Item = &[T]>
    where
        T: Send + Sync,
    {
        self.data.par_chunks_exact(self.num_cols)
    }

    #[cfg(feature = "parallel")]
    pub fn as_rows_mut_par(&mut self) -> impl ParallelIterator<Item = &mut [T]>
    where
        T: Send + Sync,
    {
        self.data.par_chunks_exact_mut(self.num_cols)
    }

    pub fn to_rows_slices(&self) -> Vec<&[T]> {
        self.as_rows().collect()
    }

    pub fn to_rows_slices_mut(&mut self) -> Vec<&mut [T]> {
        self.as_rows_mut().collect()
    }

    pub fn to_rows(&self) -> Vec<Vec<T>> {
        self.data
            .chunks_exact(self.num_cols)
            .map(|row| row.to_vec())
            .collect()
    }
}

impl<T> DenseRowMatrix<MaybeUninit<T>> {
    /// Mark the matrix as initialized, converting it to a `DenseRowMatrix<T>`.
    /// Attempts to do so in place without copying the data.
    ///
    /// # Safety
    /// The caller must ensure that all elements in `self.data` are initialized.
    pub unsafe fn init(self) -> DenseRowMatrix<T> {
        // Prevent `source` from being dropped while we steal its parts.
        let mut v = ManuallyDrop::new(self.data);
        let (ptr, len, cap) = (v.as_mut_ptr(), v.len(), v.capacity());
        // Safety: MaybeUninit<T> can be safely converted to T if all elements are
        // initialized.
        let data = unsafe { Vec::from_raw_parts(ptr.cast::<T>(), len, cap) };
        DenseRowMatrix {
            num_rows: self.num_rows,
            num_cols: self.num_cols,
            data,
        }
    }
}

impl<T> Matrix<T> for DenseRowMatrix<T> {
    #[inline]
    fn num_rows(&self) -> usize {
        self.num_rows
    }

    #[inline]
    fn num_cols(&self) -> usize {
        self.num_cols
    }

    fn cells<'a>(&'a self) -> impl Iterator<Item = impl Iterator<Item = (usize, &'a T)>>
    where
        T: 'a,
    {
        self.data
            .chunks_exact(self.num_cols)
            .map(|row| row.iter().enumerate())
    }

    fn cells_mut<'a>(&'a mut self) -> impl Iterator<Item = impl Iterator<Item = (usize, &'a mut T)>>
    where
        T: 'a,
    {
        self.data
            .chunks_exact_mut(self.num_cols)
            .map(|row| row.iter_mut().enumerate())
    }

    #[cfg(feature = "parallel")]
    fn par_cells<'a>(&'a self) -> impl ParallelIterator<Item = impl Iterator<Item = (usize, &'a T)>>
    where
        T: 'a + Send + Sync,
    {
        self.data
            .par_chunks_exact(self.num_cols)
            .map(|row| row.iter().enumerate())
    }

    #[cfg(feature = "parallel")]
    fn par_cells_mut<'a>(
        &'a mut self,
    ) -> impl ParallelIterator<Item = impl Iterator<Item = (usize, &'a mut T)>>
    where
        T: 'a + Send + Sync,
    {
        self.data
            .par_chunks_exact_mut(self.num_cols)
            .map(|row| row.iter_mut().enumerate())
    }
}

impl<T> From<Vec<Vec<T>>> for DenseRowMatrix<T> {
    fn from(value: Vec<Vec<T>>) -> Self {
        let num_cols = value.iter().map(|row| row.len()).max().unwrap_or(0);
        DenseRowMatrix {
            num_rows: value.len(),
            num_cols,
            data: value.into_iter().flatten().collect(),
        }
    }
}
