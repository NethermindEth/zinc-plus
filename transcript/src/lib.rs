pub mod traits;

use crate::traits::{ConstTranscribable, GenTranscribable, Transcript};
use crypto_primitives::{ConstIntSemiring, PrimeField};
use zinc_primality::PrimalityTest;
use zinc_utils::{add, from_ref::FromRef};

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

    /// Like [`Transcript::get_prime`] but constrains the prime to fit in
    /// the low `byte_size` bytes of `R::NUM_BYTES`. Bytes above that
    /// stay zero. `R` is little-endian (limb 0 first), so this gives a
    /// prime `< 2^(byte_size*8)`.
    ///
    /// Used by the dual-prime path's Z-branch to draw a smaller prime
    /// (e.g. 192-bit for `byte_size = 24`) inside a wider field type
    /// like `Uint<4>`. Both prover and verifier must call this with
    /// the same `byte_size` so transcripts agree.
    ///
    /// Panics if `byte_size > R::NUM_BYTES`.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn get_prime_with_byte_size<
        R: ConstIntSemiring + ConstTranscribable,
        T: PrimalityTest<R>,
    >(
        &mut self,
        byte_size: usize,
    ) -> R {
        assert!(
            byte_size <= R::NUM_BYTES,
            "byte_size ({byte_size}) must not exceed R::NUM_BYTES ({})",
            R::NUM_BYTES,
        );
        let buf = &mut vec![0u8; R::NUM_BYTES];
        loop {
            // Fill only the low `byte_size` bytes; the rest stay zero.
            // The transcript only absorbs the bytes we actually drew —
            // absorbing the implicit zeros would just be padding noise.
            for byte in buf[..byte_size].iter_mut() {
                *byte = 0;
            }
            self.fill_with_random_bytes(&mut buf[..byte_size]);
            self.absorb_inner(&buf[..byte_size]);
            let mut prime_candidate: R = R::read_transcription_bytes_exact(buf);
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

    /// Helper that mirrors [`Transcript::get_random_field_cfg`] but
    /// constrains the modulus to the low `byte_size` bytes of
    /// `F::Modulus::NUM_BYTES`. See [`get_prime_with_byte_size`].
    pub fn get_random_field_cfg_with_byte_size<F, FMod, T>(
        &mut self,
        byte_size: usize,
    ) -> F::Config
    where
        F: PrimeField,
        FMod: ConstTranscribable + ConstIntSemiring,
        F::Modulus: FromRef<FMod>,
        T: PrimalityTest<FMod>,
    {
        let prime = self.get_prime_with_byte_size::<FMod, T>(byte_size);
        F::make_cfg(&F::Modulus::from_ref(&prime))
            .expect("prime is guaranteed to be prime")
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
