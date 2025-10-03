pub mod raa;

use crate::{
    pcs::structs::ZipTypes,
    traits::{FromRef},
};
use crypto_primitives::{PrimeField, Ring};

pub trait LinearCode<Zt: ZipTypes>: Sync + Send {
    /// Repetition factor, a.k.a. inverse rate, the ratio of codeword length to
    /// input row length. Has to be at a power of 2.
    ///
    /// Note: Ideally, this should be a generic constant, but due to the fact
    /// that generic parameters may not be used in const operations, this
    /// makes using it too much of a hassle.
    const REPETITION_FACTOR: usize;

    fn new(poly_size: usize, check_for_overflows: bool) -> Self;

    /// Length of each input row before encoding
    fn row_len(&self) -> usize;

    /// Length of each encoded codeword (output length after encoding)
    fn codeword_len(&self) -> usize;

    /// Encodes a row of cryptographic integers using this linear encoding
    /// scheme.
    ///
    /// This function is optimized for the prover's context where we work with
    /// cryptographic integers. It's more efficient than `encode_f` as it
    /// avoids field conversions.
    ///
    /// # Parameters
    /// - `row`: Slice of cryptographic integers to encode
    ///
    /// # Returns
    /// A vector of cryptographic integers representing the encoded row
    fn encode(&self, row: &[Zt::Eval]) -> Vec<Zt::Cw>;

    /// Encodes a row of cryptographic integers using this linear encoding
    /// scheme.
    ///
    /// This function is optimized for the prover's context where we work with
    /// cryptographic integers. It's more efficient than `encode_f` as it
    /// avoids field conversions.
    ///
    /// # Parameters
    /// - `row`: Slice of cryptographic integers to encode
    ///
    /// # Returns
    /// A vector of cryptographic integers representing the encoded row
    fn encode_wide(&self, row: &[Zt::CombR]) -> Vec<Zt::CombR>;

    /// Encodes a row of field elements using this linear encoding scheme.
    ///
    /// This function is used when working with field elements directly and
    /// performs the encoding by first converting the sparse matrices to
    /// field elements.
    ///
    /// # Parameters
    /// - `row`: Slice of field elements to encode
    /// - `field`: Field configuration for the conversion
    ///
    /// # Returns
    /// A vector of field elements representing the encoded row
    fn encode_f<F>(&self, row: &[F]) -> Vec<F>
    where
        F: PrimeField + FromRef<F>;
}
