pub mod traits;

use crate::traits::{ConstTranscribable, Transcribable, Transcript};
use crypto_primitives::{ConstIntSemiring, PrimeField};
use sha3::{Digest, Keccak256};
use zinc_primality::PrimalityTest;

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
    fn fill_with_random_bytes(&mut self, buf: &mut [u8]) {
        let mut counter = 0;
        let mut filled_length = 0;
        while filled_length < buf.len() {
            let mut temp_hasher = self.hasher.clone();
            temp_hasher.update(i32::to_le_bytes(counter));
            let hash = temp_hasher.finalize();
            let start = filled_length;
            let end = (filled_length + hash.len()).min(buf.len());
            buf[start..end].copy_from_slice(&hash[0..(end - start)]);

            filled_length += hash.len();
            counter += 1;
        }
    }

    fn gen_random<R: ConstTranscribable>(&mut self, buf: &mut [u8]) -> R {
        self.fill_with_random_bytes(buf);
        self.absorb(buf);
        R::read_transcription_bytes(buf)
    }
}

impl Transcript for KeccakTranscript {
    fn get_challenge<T: ConstTranscribable>(&mut self) -> T {
        let mut buf = vec![0u8; T::NUM_BYTES];
        self.fill_with_random_bytes(&mut buf);
        self.hasher.update([0x12]);
        self.hasher.update(&mut buf);
        self.hasher.update([0x34]);
        T::read_transcription_bytes(&buf)
    }

    #[allow(clippy::arithmetic_side_effects)]
    fn get_prime<R: ConstIntSemiring + ConstTranscribable, T: PrimalityTest<R>>(&mut self) -> R {
        let buf = &mut vec![0u8; R::NUM_BYTES];
        loop {
            let mut prime_candidate: R = self.gen_random(buf);
            if prime_candidate.is_zero() {
                continue;
            }
            if prime_candidate.is_even() {
                prime_candidate -= R::ONE;
            }
            if T::is_probably_prime(&prime_candidate) {
                return prime_candidate;
            }
        }
    }

    /// Absorbs arbitrary bytes into the transcript.
    /// This updates the internal state of the hasher with the provided data.
    fn absorb(&mut self, v: &[u8]) {
        self.hasher.update(v);
    }

    /// Absorbs a field element into the transcript.
    /// Delegates to the field element's implementation of
    /// absorb_into_transcript.
    // Note: Currently this only works for fields whose modulus and inner element
    // have the same byte length
    fn absorb_random_field<F>(&mut self, v: &F, buf: &mut [u8])
    where
        F: PrimeField,
        F::Inner: Transcribable,
        F::Modulus: Transcribable,
    {
        debug_assert_eq!(F::Inner::LENGTH_NUM_BYTES, F::Modulus::LENGTH_NUM_BYTES);
        debug_assert_eq!(
            F::Inner::get_num_bytes(v.inner()),
            F::Modulus::get_num_bytes(&v.modulus())
        );
        self.absorb(&[0x3]);
        v.modulus().write_transcription_bytes(buf);
        self.absorb(buf);
        self.absorb(&[0x5]);

        self.absorb(&[0x1]);
        v.inner().write_transcription_bytes(buf);
        self.absorb(buf);
        self.absorb(&[0x3])
    }
}
