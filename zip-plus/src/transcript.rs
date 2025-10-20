use crate::traits::{Transcribable, Transcript};
use ark_std::vec::Vec;
use crypto_primitives::PrimeField;
use sha3::{Digest, Keccak256};

/// A cryptographic transcript implementation using the Keccak-256 hash
/// function. Used for Fiat-Shamir transformations in zero-knowledge proof
/// systems.
#[derive(Debug, Clone)]
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

    /// Generates a specified number of pseudorandom bytes based on the current
    /// transcript state. Uses a counter-based approach to generate enough
    /// bytes from the hasher.
    ///
    /// Note that this does NOT update the internal state of the hasher
    #[allow(clippy::arithmetic_side_effects)]
    fn get_random_bytes(&mut self, length: usize) -> Vec<u8> {
        let mut result = Vec::with_capacity(length);
        let mut counter = 0;
        while result.len() < length {
            let mut temp_hasher = self.hasher.clone();
            temp_hasher.update(i32::to_le_bytes(counter));
            let hash = temp_hasher.finalize();
            result.extend_from_slice(&hash);

            counter += 1;
        }

        result.truncate(length);
        result
    }

    /// Absorbs arbitrary bytes into the transcript.
    /// This updates the internal state of the hasher with the provided data.
    pub fn absorb(&mut self, v: &[u8]) {
        self.hasher.update(v);
    }

    /// Absorbs a field element into the transcript.
    /// Delegates to the field element's implementation of
    /// absorb_into_transcript.
    pub fn absorb_random_field<F>(&mut self, v: &F)
    where
        F: PrimeField,
        F::Inner: Transcribable,
    {
        let mut buf = vec![0; F::Inner::NUM_BYTES];
        self.absorb(&[0x3]);
        F::MODULUS.write_transcription_bytes(&mut buf);
        self.absorb(&buf);
        self.absorb(&[0x5]);

        self.absorb(&[0x1]);
        v.inner().write_transcription_bytes(&mut buf);
        self.absorb(&buf);
        self.absorb(&[0x3])
    }

    /// Absorbs a slice of field elements into the transcript.
    /// Processes each field element in the slice sequentially.
    pub fn absorb_slice<F>(&mut self, slice: &[F])
    where
        F: PrimeField,
        F::Inner: Transcribable,
    {
        for field_element in slice.iter() {
            self.absorb_random_field(field_element);
        }
    }
}

impl Transcript for KeccakTranscript {
    fn get_challenge<T: Transcribable>(&mut self) -> T {
        let challenge = self.get_random_bytes(T::NUM_BYTES);
        self.hasher.update([0x12]);
        self.hasher.update(&challenge);
        self.hasher.update([0x34]);
        T::read_transcription_bytes(&challenge)
    }
}
