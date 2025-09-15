use std::collections::BTreeSet;
use crypto_bigint::Int;

pub trait FromBits {
    fn from_be_bits(bits: &[bool]) -> Self;
    fn from_le_bits(bits: &[bool]) -> Self;
}

pub trait ConstNumBytes {
    /// Number of bytes required to represent this type.
    const NUM_BYTES: usize;
}

pub trait MapIterable: Sized {
    fn map_iterable<'a, const M: usize, I: IntoIterator<Item = &'a Int<M>>>(
        iterable: I,
    ) -> Vec<Self>;
}

// Out own version of FromBytes and ToBytes traits, allowing us to implement them for types
// that do not implement the num_traits versions directly, but have a compatible interface.

pub trait FromBytes {
    fn from_be_bytes(bytes: &[u8]) -> Self;
    fn from_le_bytes(bytes: &[u8]) -> Self;
}

pub trait ToBytes {
    fn to_be_bytes(&self) -> Vec<u8>;
    fn to_le_bytes(&self) -> Vec<u8>;
}

pub trait Transcribable: FromBytes + ToBytes + ConstNumBytes {}
impl<T: FromBytes + ToBytes + ConstNumBytes> Transcribable for T {}

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
