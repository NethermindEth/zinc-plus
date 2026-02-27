pub mod iprs;
pub mod raa;
pub mod raa_sign_flip;

use crate::pcs::structs::ZipTypes;
use crypto_primitives::FromPrimitiveWithConfig;
use zinc_utils::from_ref::FromRef;

pub trait LinearCode<Zt: ZipTypes>: Sync + Send {
    /// Repetition factor, a.k.a. inverse rate, the ratio of codeword length to
    /// input row length. Has to be at a power of 2.
    ///
    /// Note: Ideally, this should be a generic constant, but due to the fact
    /// that generic parameters may not be used in const operations, this
    /// makes using it too much of a hassle.
    const REPETITION_FACTOR: usize;

    fn new(row_len: usize) -> Self;

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
        F: FromPrimitiveWithConfig + FromRef<F>;

    /// Computes the encoding of `row` only at the given output `positions`.
    ///
    /// Returns `positions.len()` values, where the i-th result is the
    /// codeword element that would appear at `positions[i]` in the full
    /// encoding of `row`. This is much cheaper than a full `encode_wide`
    /// when only a small subset of output positions is needed (e.g. for
    /// verifier spot-checks).
    ///
    /// Default implementation falls back to `encode_wide` and indexes.
    fn encode_wide_at_positions(
        &self,
        row: &[Zt::CombR],
        positions: &[usize],
    ) -> Vec<Zt::CombR> {
        let full = self.encode_wide(row);
        positions.iter().map(|&p| full[p].clone()).collect()
    }

    /// Computes the field-element encoding of `row` only at the given output
    /// `positions`.
    ///
    /// Returns `positions.len()` values, where the i-th result is the
    /// codeword element that would appear at `positions[i]` in the full
    /// encoding of `row`. Default implementation falls back to `encode_f`.
    fn encode_f_at_positions<F>(&self, row: &[F], positions: &[usize]) -> Vec<F>
    where
        F: FromPrimitiveWithConfig + FromRef<F> + Send + Sync,
    {
        let full = self.encode_f(row);
        positions.iter().map(|&p| full[p].clone()).collect()
    }
}
