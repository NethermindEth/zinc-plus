#![allow(non_snake_case)]

use ark_std::{collections::BTreeSet, fmt::Debug, iter, marker::PhantomData, vec, vec::Vec};
use itertools::Itertools;

use super::pcs::structs::ZipTranscript;
use crate::{
    traits::{Field, FieldMap, Integer, Words, ZipTypes},
    utils::expand,
};

pub trait LinearCode<ZT: ZipTypes>: Sync + Send {
    /// Length of each input row before encoding
    fn row_len(&self) -> usize;

    /// Length of each encoded codeword (output length after encoding)
    fn codeword_len(&self) -> usize;

    /// Number of columns to open during verification (security parameter)
    fn num_column_opening(&self) -> usize;

    /// Number of proximity tests to perform (security parameter)
    fn num_proximity_testing(&self) -> usize;

    /// Encodes a row of cryptographic integers using this linear encoding scheme.
    ///
    /// This function is optimized for the prover's context where we work with cryptographic integers.
    /// It's more efficient than `encode_f` as it avoids field conversions.
    ///
    /// # Parameters
    /// - `row`: Slice of cryptographic integers to encode
    ///
    /// # Returns
    /// A vector of cryptographic integers representing the encoded row
    fn encode(&self, row: &[ZT::N]) -> Vec<ZT::M> {
        self.encode_wide(row)
    }

    /// Encodes a row of cryptographic integers using this linear encoding scheme.
    ///
    /// This function is optimized for the prover's context where we work with cryptographic integers.
    /// It's more efficient than `encode_f` as it avoids field conversions.
    ///
    /// # Parameters
    /// - `row`: Slice of cryptographic integers to encode
    ///
    /// # Returns
    /// A vector of cryptographic integers representing the encoded row
    fn encode_wide<In, Out>(&self, row: &[In]) -> Vec<Out>
    where
        In: Integer,
        Out: Integer + for<'a> From<&'a In> + for<'a> From<&'a ZT::L>;

    /// Encodes a row of field elements using this linear encoding scheme.
    ///
    /// This function is used when working with field elements directly and performs the encoding
    /// by first converting the sparse matrices to field elements.
    ///
    /// # Parameters
    /// - `row`: Slice of field elements to encode
    /// - `field`: Field configuration for the conversion
    ///
    /// # Returns
    /// A vector of field elements representing the encoded row
    fn encode_f<F: Field>(&self, row: &[F], field: F::R) -> Vec<F>
    where
        ZT::L: FieldMap<F, Output = F>;
}

/// A linear code implementation used for the Zip PCS.
///
/// # Type Parameters
/// - `I`: The input cryptographic integer type. Represents the field elements being encoded.
/// - `L`: The matrix element type. A larger cryptographic integer type used for sparse matrix
///   operations to prevent overflow during encoding. Must be at least as large as `I`.
#[derive(Clone, Debug)]
pub struct ZipLinearCode<ZT: ZipTypes> {
    /// Length of each input row before encoding
    row_len: usize,

    /// Length of each encoded codeword (output length after encoding)
    codeword_len: usize,

    /// Number of columns to open during verification (security parameter)
    num_column_opening: usize,

    /// Number of proximity tests to perform (security parameter)
    num_proximity_testing: usize,

    /// First sparse matrix used in the encoding process
    a: SparseMatrixZ<ZT::L>,

    /// Second sparse matrix used in the encoding process
    b: SparseMatrixZ<ZT::L>,

    phantom: PhantomData<ZT>,
}

pub trait LinearCodeSpec: Debug {
    fn num_column_opening(&self) -> usize;

    /// A.k.a. inverse rate, the ratio of codeword length to input row length.
    /// Has to be at a power of 2.
    fn repetition_factor(&self) -> usize;

    fn num_proximity_testing(&self, _log2_q: usize, _n: usize, _n_0: usize) -> usize;
}

// Figure 2 in [GLSTW21](https://eprint.iacr.org/2021/1043.pdf).
#[derive(Debug)]
pub struct DefaultLinearCodeSpec;
impl LinearCodeSpec for DefaultLinearCodeSpec {
    fn num_column_opening(&self) -> usize {
        1000
    }

    fn repetition_factor(&self) -> usize {
        2
    }

    fn num_proximity_testing(&self, _log2_q: usize, _n: usize, _n_0: usize) -> usize {
        1
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SparseMatrixDimension {
    /// Number of rows
    n: usize,
    /// Number of columns
    m: usize,
    /// Number of non-zero elements per row
    d: usize,
}

impl ark_std::fmt::Display for SparseMatrixDimension {
    fn fmt(&self, f: &mut ark_std::fmt::Formatter<'_>) -> ark_std::fmt::Result {
        write!(
            f,
            "{}x{} matrix with {} non-zero elements per row",
            self.n, self.m, self.d
        )
    }
}

impl SparseMatrixDimension {
    fn new(n: usize, m: usize, d: usize) -> Self {
        Self { n, m, d }
    }
}

/// Sparse matrix over a ring of integers.
#[derive(Clone, Debug)]
pub struct SparseMatrixZ<I: Integer> {
    dimension: SparseMatrixDimension,
    cells: Vec<(usize, I)>,
}

impl<L: Integer> SparseMatrixZ<L> {
    /// Creates a new sparse matrix with the given dimension and samples its cells using the
    /// provided transcript.
    fn sample_new<T: ZipTranscript<L>>(
        dimension: SparseMatrixDimension,
        transcript: &mut T,
    ) -> Self {
        let cells = iter::repeat_with(|| {
            let mut columns = BTreeSet::<usize>::new();
            transcript.sample_unique_columns(0..dimension.m, &mut columns, dimension.d);
            columns
                .into_iter()
                .map(|column| (column, transcript.get_encoding_element()))
                .collect_vec()
        })
        .take(dimension.n)
        .flatten()
        .collect();
        Self { dimension, cells }
    }

    pub fn rows(&self) -> impl Iterator<Item = &[(usize, L)]> {
        self.cells.chunks(self.dimension.d)
    }

    /// Multiplies the sparse matrix by a vector of cryptographic integers.
    pub fn mat_vec_mul<N: Integer, M: Integer + for<'a> From<&'a N> + for<'a> From<&'a L>>(
        &self,
        vector: &[N],
    ) -> Vec<M> {
        assert_eq!(
            self.dimension.m,
            vector.len(),
            "Vector length must match matrix column dimension"
        );

        let mut result = vec![M::from_i64(0i64); self.dimension.n];

        self.rows().enumerate().for_each(|(row_idx, cells)| {
            let mut sum = M::ZERO;
            for (column, coeff) in cells.iter() {
                sum += &(expand::<L, M>(coeff) * expand::<N, M>(&vector[*column]));
            }
            result[row_idx] = sum;
        });

        result
    }

    pub fn to_dense(&self) -> Vec<Vec<L>> {
        let mut r: Vec<Vec<L>> = vec![vec![L::ZERO; self.dimension.m]; self.dimension.n];
        for (row_i, (col_i, value)) in self.cells.iter().enumerate() {
            r[row_i][*col_i] = value.clone();
        }
        r
    }
}

pub fn steps(start: i64) -> impl Iterator<Item = i64> {
    steps_by(start, 1i64)
}

pub fn steps_by(start: i64, step: i64) -> impl Iterator<Item = i64> {
    iter::successors(Some(start), move |state| Some(step + *state))
}
