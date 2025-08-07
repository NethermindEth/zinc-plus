#![allow(non_snake_case)]

use ark_std::{fmt::Debug, iter, vec::Vec};

use crate::traits::{Field, FieldMap, Integer, ZipTypes};

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

pub fn steps(start: i64) -> impl Iterator<Item = i64> {
    steps_by(start, 1i64)
}

pub fn steps_by(start: i64, step: i64) -> impl Iterator<Item = i64> {
    iter::successors(Some(start), move |state| Some(step + *state))
}
