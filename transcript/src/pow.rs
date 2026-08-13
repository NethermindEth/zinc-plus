//! Pure proof-of-work search and checking primitives.
//!
//! Protocol code is responsible for deriving a seed from the transcript
//! prefix and for absorbing the resulting nonce before the protected
//! challenge. This module deliberately does not mutate a transcript.

#[cfg(feature = "parallel")]
use rayon::prelude::*;
use zinc_utils::add;

/// Domain used to derive the proof-of-work seed from a Fiat-Shamir transcript.
pub const POW_SEED_CONTEXT: &str = "zinc-plus/fiat-shamir/query-pow-seed/v1";

/// Highest protocol-level grinding difficulty accepted by Zinc+.
///
/// This keeps an accidentally enabled configuration within a practical
/// `u64` nonce-search budget.
pub const MAX_GRINDING_BITS: u32 = 32;

/// Bound the deterministic search chunk selected from the configured
/// difficulty. Small difficulties avoid oversized speculative work; larger
/// difficulties amortize Rayon scheduling across a wider range.
const MIN_SEARCH_CHUNK_LOG2: u32 = 12;
const MAX_SEARCH_CHUNK_LOG2: u32 = 24;

/// Keep cheap predicate calls from being dominated by Rayon scheduling.
#[cfg(feature = "parallel")]
const MIN_SEARCH_LEN: usize = 1 << 8;

fn search_chunk_len(bits: u32) -> u32 {
    let log2 = bits
        .saturating_div(2)
        .saturating_add(8)
        .clamp(MIN_SEARCH_CHUNK_LOG2, MAX_SEARCH_CHUNK_LOG2);
    1u32.checked_shl(log2)
        .expect("PoW search chunk exponent is bounded below 32")
}

/// Hashes the fixed 64-byte message `seed || nonce_le || zero_padding` with
/// BLAKE3 and returns the first eight digest bytes as a big-endian integer.
/// A valid witness therefore has zeroes in the conventional leading bits of
/// the BLAKE3 digest.
///
/// The 24 zero bytes make the work message exactly one BLAKE3 block and are
/// part of the protocol encoding.
#[must_use]
pub fn pow_hash(seed: &[u8; 32], nonce: u64) -> u64 {
    let mut input = [0u8; blake3::BLOCK_LEN];
    input[..32].copy_from_slice(seed);
    input[32..40].copy_from_slice(&nonce.to_le_bytes());
    let digest = blake3::hash(&input);
    u64::from_be_bytes(
        digest.as_bytes()[..8]
            .try_into()
            .expect("BLAKE3 digest is 32 bytes"),
    )
}

/// Returns whether `nonce` supplies at least `bits` bits of work for `seed`.
///
/// `bits == 0` accepts every nonce. Values above 64 reject every nonce.
#[must_use]
pub fn check_pow(seed: &[u8; 32], nonce: u64, bits: u32) -> bool {
    pow_hash(seed, nonce).leading_zeros() >= bits
}

/// Finds the smallest nonce satisfying [`check_pow`].
///
/// Expected work is approximately `2^bits` BLAKE3 evaluations. The result is
/// deterministic with and without the `parallel` feature.
///
/// # Panics
/// Panics when `bits >= 64`, which is not a feasible search configuration.
#[must_use]
pub fn find_pow_nonce(seed: &[u8; 32], bits: u32) -> u64 {
    assert!(bits < 64, "grinding bit count must be less than 64");
    if bits == 0 {
        return 0;
    }
    let chunk_len = search_chunk_len(bits);
    let mut chunk_start = 0u64;
    loop {
        if let Some(nonce) = search_chunk(seed, chunk_start, chunk_len, bits) {
            return nonce;
        }
        chunk_start = add!(
            chunk_start,
            u64::from(chunk_len),
            "PoW nonce search exhausted the u64 nonce space"
        );
    }
}

#[cfg(feature = "parallel")]
fn search_chunk(seed: &[u8; 32], start: u64, len: u32, bits: u32) -> Option<u64> {
    (0..len)
        .into_par_iter()
        .with_min_len(MIN_SEARCH_LEN)
        .map(|offset| add!(start, u64::from(offset)))
        .find_first(|nonce| check_pow(seed, *nonce, bits))
}

#[cfg(not(feature = "parallel"))]
fn search_chunk(seed: &[u8; 32], start: u64, len: u32, bits: u32) -> Option<u64> {
    (0..len)
        .map(|offset| add!(start, u64::from(offset)))
        .find(|nonce| check_pow(seed, *nonce, bits))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_seed() -> [u8; 32] {
        *blake3::hash(b"zinc-plus PoW test seed").as_bytes()
    }

    #[test]
    fn hash_wire_vectors_are_stable() {
        assert_eq!(pow_hash(&[0u8; 32], 0), 0x4d00_6976_636a_8696);
        assert_eq!(pow_hash(&[0u8; 32], 1), 0xbb4f_adcb_0b8f_eab0);
        assert_eq!(pow_hash(&[0x42u8; 32], 42), 0xe8d7_1fdd_a38e_84f3);
    }

    #[test]
    fn search_chunks_scale_with_difficulty_and_remain_bounded() {
        assert_eq!(search_chunk_len(1), 1 << 12);
        assert_eq!(search_chunk_len(12), 1 << 14);
        assert_eq!(search_chunk_len(16), 1 << 16);
        assert_eq!(search_chunk_len(20), 1 << 18);
        assert_eq!(search_chunk_len(24), 1 << 20);
        assert_eq!(search_chunk_len(32), 1 << 24);
        assert_eq!(search_chunk_len(u32::MAX), 1 << 24);
    }

    #[test]
    fn find_then_check_roundtrip() {
        let seed = test_seed();
        for bits in [0, 1, 4, 8] {
            let nonce = find_pow_nonce(&seed, bits);
            assert!(check_pow(&seed, nonce, bits));
        }
    }

    #[test]
    fn found_nonce_is_minimal_and_deterministic() {
        let seed = test_seed();
        let nonce = find_pow_nonce(&seed, 8);
        assert_eq!(nonce, 66, "the deterministic proof nonce changed");
        assert_eq!(nonce, find_pow_nonce(&seed, 8));
        assert!((0..nonce).all(|smaller| !check_pow(&seed, smaller, 8)));
    }

    #[test]
    fn search_advances_past_an_empty_chunk() {
        const BITS: u32 = 12;

        let chunk_len = search_chunk_len(BITS);
        let seed = (0u64..100)
            .map(|counter| *blake3::hash(&counter.to_le_bytes()).as_bytes())
            .find(|seed| search_chunk(seed, 0, chunk_len, BITS).is_none())
            .expect("a deterministic seed with no solution in its first chunk");

        let nonce = find_pow_nonce(&seed, BITS);
        assert!(nonce >= u64::from(chunk_len));
        assert!(check_pow(&seed, nonce, BITS));
    }

    #[test]
    fn zero_and_out_of_range_bits_are_explicit() {
        let seed = test_seed();
        assert_eq!(find_pow_nonce(&seed, 0), 0);
        assert!(check_pow(&seed, 0, 0));
        assert!(!check_pow(&seed, 0, 65));
    }

    #[test]
    #[should_panic(expected = "less than 64")]
    fn search_rejects_64_bits() {
        let _ = find_pow_nonce(&test_seed(), 64);
    }
}
