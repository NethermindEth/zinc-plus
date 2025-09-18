use crate::{
    add, rem, sub,
    traits::{ToBytes, Transcribable, Transcript},
    utils::WORD_FACTOR,
};
use ark_std::vec::Vec;
use crypto_primitives::{PrimeField, crypto_bigint_int::Int};
use sha3::{Digest, Keccak256};

/// A cryptographic transcript implementation using the Keccak-256 hash
/// function. Used for Fiat-Shamir transformations in zero-knowledge proof
/// systems.
#[derive(Clone)]
pub struct KeccakTranscript {
    /// The underlying Keccak-256 hasher that maintains the transcript state.
    hasher: Keccak256,
}

impl Default for KeccakTranscript {
    fn default() -> Self {
        Self::new()
    }
}

impl KeccakTranscript {
    pub fn new() -> Self {
        Self {
            hasher: Keccak256::new(),
        }
    }

    /// Absorbs arbitrary bytes into the transcript.
    /// This updates the internal state of the hasher with the provided data.
    pub fn absorb(&mut self, v: &[u8]) {
        self.hasher.update(v);
    }

    /// Generates a specified number of pseudorandom bytes based on the current
    /// transcript state. Uses a counter-based approach to generate enough
    /// bytes from the hasher.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn get_random_bytes(&mut self, length: usize) -> Vec<u8> {
        let mut result = Vec::with_capacity(length);
        let mut counter = 0;
        while result.len() < length {
            let mut temp_hasher = self.hasher.clone();
            temp_hasher.update(i32::to_be_bytes(counter));
            let hash = temp_hasher.finalize();
            result.extend_from_slice(&hash);

            counter += 1;
        }

        result.truncate(length);
        result
    }

    /// Absorbs a field element into the transcript.
    /// Delegates to the field element's implementation of
    /// absorb_into_transcript.
    pub fn absorb_random_field<F>(&mut self, v: &F)
    where
        F: PrimeField,
        F::Inner: ToBytes,
    {
        self.absorb(&[0x3]);
        self.absorb(&F::MODULUS.to_be_bytes());
        self.absorb(&[0x5]);

        self.absorb(&[0x1]);
        self.absorb(&v.inner().to_be_bytes());
        self.absorb(&[0x3])
    }

    /// Absorbs a slice of field elements into the transcript.
    /// Processes each field element in the slice sequentially.
    pub fn absorb_slice<F>(&mut self, slice: &[F])
    where
        F: PrimeField,
        F::Inner: ToBytes,
    {
        for field_element in slice.iter() {
            self.absorb_random_field(field_element);
        }
    }

    /// Generates a pseudorandom [Integer] as a challenge based on the current
    /// transcript state.
    pub fn get_challenge<T: Transcribable>(&mut self) -> T {
        let challenge = self.get_random_bytes(T::NUM_BYTES);
        self.hasher.update([0x12]);
        self.hasher.update(&challenge);
        self.hasher.update([0x34]);
        T::from_be_bytes(&challenge)
    }

    /// Generates pseudorandom [CryptoInt]s as challenges based on the current
    /// transcript state.
    pub fn get_challenges<T: Transcribable>(&mut self, n: usize) -> Vec<T> {
        (0..n).map(|_| self.get_challenge()).collect()
    }

    /// Generates a pseudorandom `usize` within the given range bounds based on
    /// the current transcript state.
    #[allow(clippy::unwrap_used)]
    fn get_usize_in_range(&mut self, range: &ark_std::ops::Range<usize>) -> usize {
        let challenge = self.hasher.clone().finalize();

        self.hasher.update([0x88]);
        self.hasher.update(challenge);
        self.hasher.update([0x11]);

        let num = usize::from_be_bytes(challenge[..size_of::<usize>()].try_into().unwrap());
        add!(range.start, rem!(num, sub!(range.end, range.start)))
    }
}

impl Transcript for KeccakTranscript {
    #[allow(clippy::cast_possible_wrap)]
    fn get_encoding_element<const LIMBS: usize>(&mut self) -> Int<LIMBS> {
        let byte = self.get_random_bytes(1)[0];
        // cancels all bits and depends only on whether the random byte LSB is 0 or 1
        let bit = byte & 1;
        crypto_bigint::Int::from(bit as i8).into()
    }

    #[allow(clippy::unwrap_used)]
    fn get_u64(&mut self) -> u64 {
        let bytes = self.get_challenge::<Int<WORD_FACTOR>>().to_be_bytes();
        assert_eq!(
            bytes.len(),
            size_of::<u64>(),
            "Expected 8 bytes for u64, got {} bytes",
            bytes.len()
        );
        u64::from_be_bytes(bytes.try_into().unwrap())
    }

    #[allow(clippy::arithmetic_side_effects)]
    fn sample_unique_columns(
        &mut self,
        range: ark_std::ops::Range<usize>,
        columns: &mut ark_std::collections::BTreeSet<usize>,
        count: usize,
    ) -> usize {
        let mut added = 0;
        while added < count {
            let candidate = self.get_usize_in_range(&range);
            if columns.insert(candidate) {
                added += 1;
            }
        }
        added
    }
}
