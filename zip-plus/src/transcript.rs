use ark_std::vec::Vec;
use sha3::{Digest, Keccak256};
use crypto_primitives::PrimeField;
use crate::{
    field::{Int},
    pcs::{structs::ZipTranscript, utils::AsWords},
    traits::{BigInteger, Integer, PrimitiveConversion, Words},
};

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
        F::Inner: BigInteger,
    {
        self.absorb(&[0x3]);
        self.absorb(&F::MODULUS.to_bytes_be());
        self.absorb(&[0x5]);

        self.absorb(&[0x1]);
        self.absorb(&v.inner().to_bytes_be());
        self.absorb(&[0x3])
    }

    /// Absorbs a slice of field elements into the transcript.
    /// Processes each field element in the slice sequentially.
    pub fn absorb_slice<F>(&mut self, slice: &[F])
    where
        F: PrimeField,
        F::Inner: BigInteger,
    {
        for field_element in slice.iter() {
            self.absorb_random_field(field_element);
        }
    }

    /// Internal helper that generates two 128-bit limbs from the current
    /// transcript state. Updates the transcript state.
    fn get_challenge_limbs(&mut self) -> (u128, u128) {
        let challenge = self.hasher.clone().finalize();

        let lo = u128::from_be_bytes(challenge[0..16].try_into().unwrap());
        let hi = u128::from_be_bytes(challenge[16..32].try_into().unwrap());

        self.hasher.update([0x00]);
        self.hasher.update(challenge);
        self.hasher.update([0x01]);

        (lo, hi)
    }

    /// Generates a pseudorandom field element as a challenge based on the
    /// current transcript state.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn get_challenge<F>(&mut self) -> F
    where
        F: PrimeField,
        F::Inner: BigInteger,
    {
        let (lo, hi) = self.get_challenge_limbs();
        let modulus = F::MODULUS;
        let challenge_num_bits = modulus.num_bits() - 1;
        if <F::Inner as BigInteger>::W::num_words() == 1 {
            let lo_mask = (1u64 << challenge_num_bits) - 1;

            let truncated_lo = lo as u64 & lo_mask;

            let challenge = F::from(truncated_lo);
            return challenge;
        }
        if challenge_num_bits < 128 {
            let lo_mask = (1u128 << challenge_num_bits) - 1;

            let truncated_lo = lo & lo_mask;

            let challenge: F = F::from(truncated_lo);
            challenge
        } else if challenge_num_bits >= 256 {
            let two_to_128 = F::from(F::Inner::from_bits_le(&(0..196).map(|i| i == 128).collect::<Vec<bool>>()));

            let challenge: F = F::from(lo) + two_to_128 * F::from(hi);
            challenge
        } else {
            let hi_bits_to_keep = challenge_num_bits - 128;
            let hi_mask = (1u128 << hi_bits_to_keep) - 1;

            let truncated_hi = hi & hi_mask;

            let two_to_128 = F::from(F::Inner::from_bits_le(&(0..196).map(|i| i == 128).collect::<Vec<bool>>()));

            let ret: F = F::from(lo) + two_to_128 * F::from(truncated_hi);
            ret
        }
    }

    /// Generates pseudorandom field elements as challenges based on the current
    /// transcript state.
    pub fn get_challenges<F>(&mut self, n: usize) -> Vec<F>
    where
        F: PrimeField,
        F::Inner: BigInteger
    {
        let mut challenges = Vec::with_capacity(n);
        challenges.extend((0..n).map(|_| self.get_challenge::<F>()));
        challenges
    }

    /// Generates a pseudorandom [Integer] as a challenge based on the current
    /// transcript state.
    pub fn get_integer_challenge<I: Integer>(&mut self) -> I {
        let mut words = I::W::default();

        for i in 0..I::W::num_words() {
            let mut challenge = [0u8; 8];
            challenge.copy_from_slice(self.get_random_bytes(8).as_slice());
            self.hasher.update([0x12]);
            self.hasher.update(challenge);
            self.hasher.update([0x34]);
            words[i] = PrimitiveConversion::from_primitive(u64::from_le_bytes(challenge));
        }

        I::from_words(words)
    }

    /// Generates pseudorandom [CryptoInt]s as challenges based on the current
    /// transcript state.
    pub fn get_integer_challenges<I: Integer>(&mut self, n: usize) -> Vec<I> {
        (0..n).map(|_| self.get_integer_challenge()).collect()
    }

    /// Generates a pseudorandom `usize` within the given range bounds based on
    /// the current transcript state.
    fn get_usize_in_range(&mut self, range: &ark_std::ops::Range<usize>) -> usize {
        let challenge = self.hasher.clone().finalize();

        self.hasher.update([0x88]);
        self.hasher.update(challenge);
        self.hasher.update([0x11]);

        let num = usize::from_le_bytes(challenge[..8].try_into().unwrap());
        range.start + (num % (range.end - range.start))
    }
}

impl<I: Integer> ZipTranscript<I> for KeccakTranscript {
    fn get_encoding_element(&mut self) -> I {
        let byte = self.get_random_bytes(1)[0];
        // cancels all bits and depends only on whether the random byte LSB is 0 or 1
        let bit = byte & 1;
        I::from(bit as i8)
    }

    fn get_u64(&mut self) -> u64 {
        self.get_integer_challenge::<Int<1>>().as_words()[0]
    }

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
#[cfg(test)]
mod tests {
    use ark_std::str::FromStr;

    use super::KeccakTranscript;
    use crate::{
        define_field_config,
        field::{BigInt, RandomField},
    };

    define_field_config!(
        FC,
        "3618502788666131213697322783095070105623107215331596699973092056135872020481"
    );

    #[test]
    fn test_keccak_transcript() {
        let mut transcript = KeccakTranscript::new();

        transcript.absorb(b"This is a test string!");
        let challenge: RandomField<32, FC<32>> = transcript.get_challenge();

        let expected_bigint = BigInt::<32>::from_str(
            "693058076479703886486101269644733982722902192016595549603371045888466087870",
        )
        .unwrap();
        let expected_field = expected_bigint.into();

        assert_eq!(challenge, expected_field);
    }
}
