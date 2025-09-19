use crypto_primitives::crypto_bigint_int::Int;
use std::collections::BTreeSet;

/// Trait for types that can be transcribed to and from a byte representation.
/// Byte order is not specified, but it must be portable across platforms.
pub trait Transcribable {
    /// Number of bytes required to represent this type.
    const NUM_BYTES: usize;

    /// Creates a new instance from a byte buffer.
    /// The buffer must be exactly `NUM_BYTES` long.
    fn from_transcription_bytes(bytes: &[u8]) -> Self;

    /// Transcribes the current instance into a byte buffer.
    /// Buffer must be exactly `NUM_BYTES` long.
    fn to_transcription_bytes(&self, buf: &mut [u8]);
}

pub trait Transcript {
    fn get_encoding_element<const LIMBS: usize>(&mut self) -> Int<LIMBS>;

    fn get_u64(&mut self) -> u64;

    fn sample_unique_columns(
        &mut self,
        range: ark_std::ops::Range<usize>,
        columns: &mut BTreeSet<usize>,
        count: usize,
    ) -> usize;
}
