pub mod traits;

use crate::traits::{ConstTranscribable, GenTranscribable, Transcript};
use crypto_primitives::{BaseFieldConfig, ConstIntSemiring};
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

impl Blake3Transcript {
    pub fn new() -> Self {
        Self {
            hasher: blake3::Hasher::new(),
        }
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
        self.absorb_inner(buf);
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
