pub mod traits;

use crate::traits::{ConstTranscribable, Transcribable, Transcript};
use crypto_primitives::{ConstIntSemiring, PrimeField};
use zinc_primality::PrimalityTest;

/// A cryptographic transcript implementation using the BLAKE3 hash function.
/// Used for Fiat-Shamir transformations in zero-knowledge proof systems.
///
/// All absorbed data is fed into a single incremental BLAKE3 hasher, so Blake3
/// sees one contiguous message and can apply its tree-based SIMD optimisations.
/// When squeezing challenges, the hasher is cloned and finalised in XOF mode,
/// keeping the cost proportional to the data absorbed since the last squeeze
/// rather than re-hashing from scratch.
#[derive(Debug, Clone)]
pub struct Blake3Transcript {
    /// The underlying BLAKE3 hasher that maintains the transcript state.
    hasher: blake3::Hasher,
}

/// Backward-compatible alias.
pub type KeccakTranscript = Blake3Transcript;

impl Default for Blake3Transcript {
    fn default() -> Self {
        Self::new()
    }
}

impl Blake3Transcript {
    pub fn new() -> Self {
        Self {
            hasher: blake3::Hasher::new(),
        }
    }

    /// Generates pseudorandom bytes by cloning the current hasher state and
    /// finalising the clone in XOF (extendable output function) mode.
    ///
    /// Note that this does NOT update the internal hasher state.
    fn fill_with_random_bytes(&self, buf: &mut [u8]) {
        let mut output_reader = self.hasher.clone().finalize_xof();
        output_reader.fill(buf);
    }

    fn gen_random<R: ConstTranscribable>(&mut self, buf: &mut [u8]) -> R {
        self.fill_with_random_bytes(buf);
        self.absorb(buf);
        R::read_transcription_bytes(buf)
    }
}

impl Transcript for Blake3Transcript {
    fn get_challenge<T: ConstTranscribable>(&mut self) -> T {
        let mut buf = vec![0u8; T::NUM_BYTES];
        self.fill_with_random_bytes(&mut buf);
        self.hasher.update(&[0x12]);
        self.hasher.update(&buf);
        self.hasher.update(&[0x34]);
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
    /// This updates the internal hasher state with the provided data.
    fn absorb(&mut self, v: &[u8]) {
        self.hasher.update(v);
    }

    /// Absorbs a field element into the transcript via a single `update` call.
    /// Serialises the separator-delimited modulus and inner value into one
    /// contiguous buffer so Blake3 sees a larger chunk.
    #[allow(clippy::arithmetic_side_effects)]
    fn absorb_random_field<F>(&mut self, v: &F, buf: &mut [u8])
    where
        F: PrimeField,
        F::Inner: Transcribable,
    {
        let n = buf.len();
        // Layout: [0x3] || modulus (n bytes) || [0x5] || [0x1] || inner (n bytes) || [0x3]
        let total = 2 * n + 4;
        let mut combined = vec![0u8; total];
        combined[0] = 0x3;
        v.modulus()
            .write_transcription_bytes(&mut combined[1..1 + n]);
        combined[1 + n] = 0x5;
        combined[2 + n] = 0x1;
        v.inner()
            .write_transcription_bytes(&mut combined[3 + n..3 + 2 * n]);
        combined[3 + 2 * n] = 0x3;
        self.hasher.update(&combined);
    }

    /// Absorbs a slice of field elements in a single `update` call.
    /// All elements are serialised into one contiguous buffer, letting Blake3
    /// process the data in large SIMD-friendly chunks instead of many tiny
    /// per-element updates.
    #[allow(clippy::arithmetic_side_effects)]
    fn absorb_random_field_slice<F>(&mut self, v: &[F], buf: &mut [u8])
    where
        F: PrimeField,
        F::Inner: Transcribable,
    {
        if v.is_empty() {
            return;
        }
        let n = buf.len();
        // Per element: [0x3] || modulus (n bytes) || [0x5] || [0x1] || inner (n bytes) || [0x3]
        let per_elem = 2 * n + 4;
        let total = per_elem * v.len();
        let mut combined = vec![0u8; total];
        for (i, elem) in v.iter().enumerate() {
            let off = i * per_elem;
            combined[off] = 0x3;
            elem.modulus()
                .write_transcription_bytes(&mut combined[off + 1..off + 1 + n]);
            combined[off + 1 + n] = 0x5;
            combined[off + 2 + n] = 0x1;
            elem.inner()
                .write_transcription_bytes(&mut combined[off + 3 + n..off + 3 + 2 * n]);
            combined[off + 3 + 2 * n] = 0x3;
        }
        self.hasher.update(&combined);
    }
}
