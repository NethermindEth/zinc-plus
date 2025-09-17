use crypto_bigint::modular::{ConstMontyForm, ConstMontyParams};
use crypto_bigint::{Uint, Word};
use itertools::Itertools;
use crate::traits::{ConstNumBytes, FromBytes, ToBytes};
use crate::utils::WORD_FACTOR;


//
// Making ConstMontyForm Transcribable
//

impl<Mod: ConstMontyParams<LIMBS>, const LIMBS: usize> ConstNumBytes for ConstMontyForm<Mod, LIMBS> {
    const NUM_BYTES: usize = 8 * LIMBS * WORD_FACTOR;
}

// impl<Mod: ConstMontyParams<LIMBS>, const LIMBS: usize> Transcribable for ConstMontyForm<Mod, LIMBS> {}

impl<Mod: ConstMontyParams<LIMBS>, const LIMBS: usize> FromBytes for ConstMontyForm<Mod, LIMBS> {
    fn from_be_bytes(bytes: &[u8]) -> Self {
        // crypto_bigint::Uint stores limbs in least-to-most significant order.
        // However, we're encoding each limb in big-endian order.
        let (chunked, rem) = bytes.as_chunks::<8>();
        assert!(rem.is_empty(), "Invalid byte slice length for ConstMontyForm");
        let words = chunked.into_iter().flat_map(|chunk| {
            let (chunked, rem) = chunk.as_chunks::<{ 8 / WORD_FACTOR }>();
            assert!(rem.is_empty(), "Invalid byte slice length for ConstMontyForm");
            chunked.into_iter().rev().map(|chunk| Word::from_be_bytes(*chunk))
        })
            .collect_array::<LIMBS>()
            .expect("Invalid length for ConstMontyForm");
        let u = Uint::<LIMBS>::from_words(words);
        ConstMontyForm::from_montgomery(u)
    }

    fn from_le_bytes(bytes: &[u8]) -> Self {
        // crypto_bigint::Uint stores limbs in least-to-most significant order.
        // It matches little-endian order ef limbs encoding, so platform pointer width does not matter.
        let (chunked, rem) = bytes.as_chunks::<{ 8 / WORD_FACTOR }>();
        assert!(rem.is_empty(), "Invalid byte slice length for ConstMontyForm");
        let words = chunked.into_iter().map(|chunk| Word::from_le_bytes(*chunk))
            .collect_array::<LIMBS>()
            .expect("Invalid length for ConstMontyForm");
        let u = Uint::<LIMBS>::from_words(words);
        ConstMontyForm::from_montgomery(u)
    }
}

impl<Mod: ConstMontyParams<LIMBS>, const LIMBS: usize> ToBytes for ConstMontyForm<Mod, LIMBS> {
    fn to_be_bytes(&self) -> Vec<u8> {
        // crypto_bigint::Uint stores limbs in least-to-most significant order.
        // However, we're encoding each limb in big-endian order.
        // TODO: Test if this really works cross-platform
        let be = self.as_montgomery().as_words().map(|w| w.to_be_bytes());
        be.chunks(WORD_FACTOR).flat_map(|chunk| {
            // Reverse the order of bytes in each limb, also saturating the output with zeroes if necessary
            ZeroBytesIterator.take(chunk.len() - WORD_FACTOR).chain(
                chunk.into_iter().rev().cloned()
            ).flatten()
        }).collect_vec()
    }

    fn to_le_bytes(&self) -> Vec<u8> {
        // crypto_bigint::Uint stores limbs in least-to-most significant order.
        // It matches little-endian order ef limbs encoding, so platform pointer width does not matter.
        self.as_montgomery().as_words().iter().flat_map(|w| w.to_le_bytes()).collect_vec()
    }
}

/// An iterator that yields zero bytes, used for padding in `to_be_bytes` and `to_le_bytes`.
/// Each element is zero bytes chunk equivalent to encoding a zero limb in the
/// `ConstMontyForm` representation.
#[derive(Debug, Clone, Copy)]
pub struct ZeroBytesIterator;

impl Iterator for ZeroBytesIterator {
    type Item = [u8; 8 / WORD_FACTOR];

    fn next(&mut self) -> Option<Self::Item> {
        Some([0; {8 / WORD_FACTOR}])
    }
}

impl DoubleEndedIterator for ZeroBytesIterator {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.next()
    }
}
