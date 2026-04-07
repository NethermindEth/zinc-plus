pub mod traits;

use crate::traits::{ConstTranscribable, GenTranscribable, Transcript};
use crypto_primitives::{ConstIntSemiring, PrimeField};
use sha3::{Digest, Keccak256};
use zinc_primality::PrimalityTest;
use zinc_utils::add;

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
        self.absorb_inner(buf);
        R::read_transcription_bytes_exact(buf)
    }
}

impl Transcript for KeccakTranscript {
    fn get_challenge<T: ConstTranscribable>(&mut self) -> T {
        let mut buf = vec![0u8; T::NUM_BYTES];
        self.fill_with_random_bytes(&mut buf);
        self.hasher.update([0x12]);
        self.hasher.update(&mut buf);
        self.hasher.update([0x34]);
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

pub fn read_field_cfg<F>(bytes: &[u8]) -> F::Config
where
    F: PrimeField,
    F::Modulus: ConstTranscribable,
{
    let mod_size = F::Modulus::NUM_BYTES;
    let modulus = F::Modulus::read_transcription_bytes_exact(&bytes[..mod_size]);
    F::make_cfg(&modulus).expect("valid field modulus in proof transcription")
}

pub fn read_field_vec_with_cfg<F>(bytes: &[u8], field_cfg: &F::Config) -> Vec<F>
where
    F: PrimeField,
    F::Inner: ConstTranscribable,
{
    let inner_size = F::Inner::NUM_BYTES;
    bytes
        .chunks_exact(inner_size)
        .map(F::Inner::read_transcription_bytes_exact)
        .map(|inner| F::new_unchecked_with_cfg(inner, field_cfg))
        .collect()
}

pub fn append_field_cfg<'a, F>(buf: &'a mut [u8], modulus: &F::Modulus) -> &'a mut [u8]
where
    F: PrimeField,
    F::Modulus: ConstTranscribable,
{
    let mod_size = F::Modulus::NUM_BYTES;
    let (buf, rest) = buf.split_at_mut(mod_size);
    modulus.write_transcription_bytes_exact(buf);
    rest
}

pub fn append_field_vec_inner<'a, F>(buf: &'a mut [u8], slice: &[F]) -> &'a mut [u8]
where
    F: PrimeField,
    F::Inner: ConstTranscribable,
{
    let inner_size = F::Inner::NUM_BYTES;
    let mut offset = 0;
    for elem in slice {
        let offset_end = add!(offset, inner_size);
        elem.inner()
            .write_transcription_bytes_exact(&mut buf[offset..offset_end]);
        offset = offset_end;
    }
    &mut buf[offset..]
}
