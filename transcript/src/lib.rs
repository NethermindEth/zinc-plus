pub mod pow;
pub mod traits;

use crate::traits::{ConstTranscribable, GenTranscribable, Transcript};
use crypto_primitives::{BaseFieldConfig, ConstIntSemiring};
use std::io::ErrorKind;
use thiserror::Error;
use zinc_primality::PrimalityTest;

/// A cryptographic transcript implementation using the BLAKE3 hash
/// function. Used for Fiat-Shamir transformations in zero-knowledge proof
/// systems.
#[derive(Debug, Clone)]
pub struct Blake3Transcript {
    /// The underlying BLAKE3 hasher that maintains the transcript state.
    hasher: blake3::Hasher,
}

impl Default for Blake3Transcript {
    fn default() -> Self {
        Self::new()
    }
}

/// Domain-separation label bound into every transcript at construction.
///
/// Absorbed before any protocol data, so transcripts belonging to different
/// protocols, or to different versions of this one, cannot coincide even on
/// byte-identical prover messages. The per-message tags applied by
/// [`Transcript::absorb_bytes`] delimit values within a transcript; this label
/// separates one transcript's whole message schedule from another's.
///
/// Bump the version suffix on any change to the wire encoding or to the
/// message schedule of the protocol.
const DOMAIN_SEPARATOR: &[u8] = b"zinc-plus/transcript/v1";

impl Blake3Transcript {
    pub fn new() -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(DOMAIN_SEPARATOR);
        Self { hasher }
    }

    fn derive_seed(&self, context: &'static str) -> [u8; 32] {
        let transcript_digest = self.hasher.finalize();
        blake3::derive_key(context, transcript_digest.as_bytes())
    }

    /// Derives the proof-of-work seed from the current transcript prefix
    /// without changing the transcript state.
    ///
    /// The operation has a fixed BLAKE3 derive-key context. Keeping it
    /// non-mutating makes the subsequent nonce message the only transcript
    /// state change made by grinding.
    pub fn derive_pow_seed(&self) -> [u8; 32] {
        self.derive_seed(crate::pow::POW_SEED_CONTEXT)
    }

    /// Generates a specified number of pseudorandom bytes based on the current
    /// transcript state. Uses a counter-based approach to generate enough
    /// bytes from the hasher.
    ///
    /// Note that this does NOT update the internal state of the hasher
    #[allow(clippy::arithmetic_side_effects)]
    fn fill_with_random_bytes(&mut self, buf: &mut [u8]) {
        self.hasher.finalize_xof().fill(buf);
    }

    fn gen_random<R: ConstTranscribable>(&mut self, buf: &mut [u8]) -> R {
        self.fill_with_random_bytes(buf);
        self.absorb_bytes(buf);
        R::read_transcription_bytes_exact(buf)
    }
}

impl Transcript for Blake3Transcript {
    fn get_challenge<T: ConstTranscribable>(&mut self) -> T {
        let mut buf = vec![0u8; T::NUM_BYTES];
        self.fill_with_random_bytes(&mut buf);
        self.hasher.update(&[0x12]);
        self.hasher.update(&buf);
        self.hasher.update(&[0x34]);
        T::read_transcription_bytes_exact(&buf)
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

    fn absorb_inner(&mut self, v: &[u8]) {
        self.hasher.update(v);
    }
}

pub fn read_field_cfg<C>(bytes: &[u8]) -> C
where
    C: BaseFieldConfig,
    C::Integer: ConstTranscribable,
{
    let mod_size = C::Integer::NUM_BYTES;
    let modulus = C::Integer::read_transcription_bytes_exact(&bytes[..mod_size]);
    C::new(&modulus).expect("valid field modulus in proof transcription")
}

pub fn append_field_cfg<'a, C>(buf: &'a mut [u8], modulus: &C::Integer) -> &'a mut [u8]
where
    C: BaseFieldConfig,
    C::Integer: ConstTranscribable,
{
    let mod_size = C::Integer::NUM_BYTES;
    let (buf, rest) = buf.split_at_mut(mod_size);
    modulus.write_transcription_bytes_exact(buf);
    rest
}

#[derive(Clone, Debug, PartialEq, Error)]
#[error("{1}")]
pub struct TranscriptError(pub ErrorKind, pub String);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seed_derivation_is_stable_domain_separated_and_non_mutating() {
        let mut transcript = Blake3Transcript::new();
        transcript.absorb_bytes(b"shared prefix");
        let mut untouched = transcript.clone();

        let seed = transcript.derive_pow_seed();
        assert_eq!(seed, transcript.derive_pow_seed());
        assert_ne!(
            seed,
            transcript.derive_seed("zinc-plus/test/seed-b/v1"),
            "different operations must not share derived seeds"
        );
        let mut different_prefix = transcript.clone();
        different_prefix.absorb_bytes(b"different suffix");
        assert_ne!(
            seed,
            different_prefix.derive_pow_seed(),
            "the proof-of-work seed must bind the transcript prefix"
        );
        assert_eq!(
            transcript.get_challenge::<u64>(),
            untouched.get_challenge::<u64>(),
            "deriving a seed must not consume or absorb a challenge"
        );
    }
}
