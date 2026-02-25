pub mod traits;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::traits::{ConstTranscribable, GenTranscribable, Transcript};
use crypto_primitives::{ConstIntSemiring, PrimeField};
use itertools::Itertools;
use std::iter;
use zinc_primality::PrimalityTest;
use zinc_utils::add;

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

        // Set MSB to 1
        buf[R::NUM_BYTES - 1] |= 1 << 7;
        let msb_one: R = R::read_transcription_bytes_exact(buf);
        let two = R::ONE + R::ONE;

        #[cfg(feature = "parallel")]
        let num_threads = rayon::current_num_threads();

        loop {
            let mut prime_candidate: R = self.gen_random(buf);
            if prime_candidate.is_zero() {
                continue;
            }
            if prime_candidate.is_even() {
                prime_candidate -= R::ONE;
            }

            // Set the MSB to one to ensure the candidate is of the correct bit length
            if prime_candidate < msb_one {
                prime_candidate += &msb_one;
            }

            let iter = iter::successors(Some(prime_candidate), |current| current.checked_add(&two));

            if let Some(prime) = {
                #[cfg(feature = "parallel")]
                {
                    // Unfortunately, `rayon::par_bridge` does not preserve the order of the
                    // iterator, so we use this workaround to get a well-defined
                    // order.
                    iter.chunks(num_threads * 1_000)
                        .into_iter()
                        .find_map(|chunk| {
                            let chunk = chunk.collect_vec();
                            chunk
                                .into_par_iter()
                                .by_exponential_blocks()
                                .find_first(|v| T::is_probably_prime(v))
                        })
                }

                #[cfg(not(feature = "parallel"))]
                {
                    let mut iter = iter;
                    iter.find(|v| T::is_probably_prime(v))
                }
            } {
                return prime;
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
