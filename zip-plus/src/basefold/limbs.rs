//! Balanced base-`2^p` limb decomposition of integer vectors.
//!
//! The basefold variant of Zip+ keeps the per-round folding state as a small
//! number of *limb vectors* with balanced `p`-bit digits: a vector `w` with
//! `|w[i]| < 2^{p*k - 1}` is represented as `k` vectors `v_1, ..., v_k` with
//! entries in `[-2^{p-1}, 2^{p-1})` such that `w = sum_j 2^{p*(j-1)} * v_j`.
//! Folding with fresh challenges keeps the digit width fixed (the radix
//! weights never enter the fold), and for `p >= lambda + 2` the limb count
//! stabilizes at `k = 2`.
//!
//! Implementation note: digit extraction works on the sign/magnitude
//! representation (unsigned shifts only) followed by a single carry pass to
//! rebalance digits into `[-2^{p-1}, 2^{p-1})`. This avoids any reliance on
//! signed-shift semantics of the underlying big-integer type.

use super::BasefoldError;
use crypto_primitives::{IntRing, crypto_bigint_int::Int};
use num_traits::{ConstOne, ConstZero};
use zinc_utils::{add, sub};

/// Number of balanced base-`2^p` digits used for values of at most `bits`
/// bits: the smallest `k` such that `bits <= p*k - 1`. Representability is
/// guaranteed with margin: `k` balanced digits cover all
/// `|x| <= 2^{pk-1} - 2^{p(k-1)}`, and every caller bound below stays under
/// that (see `next_limb_count`).
pub fn num_limbs_for(bits: u32, p: u32) -> usize {
    debug_assert!(p > 0);
    add!(bits, 1u32).div_ceil(p) as usize
}

/// The limb-count recursion of the basefold rounds: folding the `2 * k_prev`
/// limb halves with challenges in `[0, 2^lambda)` and balanced digits
/// (`|d| <= 2^{p-1}`) yields values bounded by
/// `2 * k * (2^lambda - 1) * 2^{p-1} < 2^{lambda + p + ceil(log2(k_prev))}`
/// (the factor 2 and the half-range cancel). For `p >= lambda + 2` this
/// stabilizes at `k = 2`, with the fold bound strictly below the balanced
/// representability limit of two digits.
pub fn next_limb_count(lambda: u32, p: u32, k_prev: usize) -> usize {
    next_limb_count_arity(lambda, p, k_prev, 1)
}

/// Arity-`2^s` variant of [`next_limb_count`]: folding the `2^s * k_prev`
/// limb pieces with challenges in `[0, 2^lambda)` and balanced digits yields
/// values bounded by `2^s * k * (2^lambda - 1) * 2^{p-1}
/// < 2^{lambda + p + s - 1 + ceil(log2(k_prev))}`. Stabilizes at `k = 2`
/// for `p >= lambda + s + 1`.
pub fn next_limb_count_arity(lambda: u32, p: u32, k_prev: usize, log_arity: u32) -> usize {
    debug_assert!(k_prev >= 1 && log_arity >= 1);
    // ceil(log2(k_prev))
    let log_k = sub!(usize::BITS, sub!(k_prev, 1usize).leading_zeros());
    num_limbs_for(add!(add!(lambda, p), add!(log_k, sub!(log_arity, 1u32))), p)
}

/// Balanced base-`2^p` decomposition of a single integer.
///
/// Returns `num_limbs` digits in `[-2^{p-1}, 2^{p-1})`, least significant
/// first, with `x = sum_j 2^{p*j} * digit_j`. Errors if `x` is not
/// representable in that range.
#[allow(clippy::arithmetic_side_effects)] // digit arithmetic bounded by construction
pub fn decompose_entry_balanced<const NIN: usize, const ND: usize>(
    x: &Int<NIN>,
    p: u32,
    num_limbs: usize,
) -> Result<Vec<Int<ND>>, BasefoldError> {
    // Digits live in [-2^{p-1}, 2^{p-1}); the carry pass needs one spare bit
    // beyond `p` and one sign bit.
    let capacity = sub!(Int::<ND>::BITS, 2u32);
    if p > capacity || p == 0 {
        return Err(BasefoldError::DigitWidth { p, capacity });
    }
    if p >= Int::<NIN>::BITS {
        return Err(BasefoldError::DigitWidth {
            p,
            capacity: sub!(Int::<NIN>::BITS, 1u32),
        });
    }

    let negative = x.is_negative();
    let mut magnitude = x.inner().abs();
    let mask =
        (crypto_bigint::Uint::<NIN>::ONE << p).wrapping_sub(&crypto_bigint::Uint::<NIN>::ONE);

    // Unsigned digit extraction from the magnitude, then sign application.
    let mut digits: Vec<Int<ND>> = Vec::with_capacity(num_limbs);
    for _ in 0..num_limbs {
        let chunk: crypto_bigint::Uint<ND> = (magnitude & mask).resize();
        // Top two bits are clear since `chunk < 2^p <= 2^{ND*64 - 2}`, so the
        // word reinterpretation is value-preserving.
        let digit = Int::<ND>::new(crypto_bigint::Int::from_words(chunk.to_words()));
        digits.push(if negative { -digit } else { digit });
        magnitude >>= p;
    }
    if magnitude != crypto_bigint::Uint::ZERO {
        return Err(BasefoldError::ValueTooWide { num_limbs, p });
    }

    // Rebalance into [-2^{p-1}, 2^{p-1}) with a single carry pass.
    let full = Int::<ND>::ONE << p;
    let half = Int::<ND>::ONE << sub!(p, 1u32);
    let neg_half = -half;
    let mut carry = Int::<ND>::ZERO;
    for d in digits.iter_mut() {
        let v = add!(*d, carry);
        if v >= half {
            *d = sub!(v, full);
            carry = Int::ONE;
        } else if v < neg_half {
            *d = add!(v, full);
            carry = -Int::<ND>::ONE;
        } else {
            *d = v;
            carry = Int::ZERO;
        }
    }
    if carry != Int::ZERO {
        return Err(BasefoldError::ValueTooWide { num_limbs, p });
    }
    Ok(digits)
}

/// Balanced decomposition of a vector into `num_limbs` limb vectors
/// (limb `j` collects digit `j` of every entry).
pub fn decompose_balanced<const NIN: usize, const ND: usize>(
    w: &[Int<NIN>],
    p: u32,
    num_limbs: usize,
) -> Result<Vec<Vec<Int<ND>>>, BasefoldError> {
    let mut limbs: Vec<Vec<Int<ND>>> = (0..num_limbs)
        .map(|_| Vec::with_capacity(w.len()))
        .collect();
    for x in w {
        let digits = decompose_entry_balanced::<NIN, ND>(x, p, num_limbs)?;
        for (limb, digit) in limbs.iter_mut().zip(digits) {
            limb.push(digit);
        }
    }
    Ok(limbs)
}

/// Recombine limb vectors: `w[i] = sum_j 2^{p*(j-1)} * limbs[j][i]`,
/// evaluated Horner-style in the (wider) output width. Left shifts are
/// overflow-checked by the underlying big-integer type.
#[allow(clippy::arithmetic_side_effects)]
pub fn recombine<const ND: usize, const NW: usize>(
    limbs: &[Vec<Int<ND>>],
    p: u32,
) -> Vec<Int<NW>> {
    let Some(first) = limbs.first() else {
        return Vec::new();
    };
    let len = first.len();
    debug_assert!(limbs.iter().all(|l| l.len() == len));
    (0..len)
        .map(|i| {
            let mut acc = Int::<NW>::ZERO;
            for limb in limbs.iter().rev() {
                acc = add!(acc << p, limb[i].resize::<NW>());
            }
            acc
        })
        .collect()
}

#[cfg(test)]
#[allow(clippy::arithmetic_side_effects, clippy::unwrap_used)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng, rngs::StdRng};

    fn random_int<const N: usize>(rng: &mut StdRng, bits: u32) -> Int<N> {
        // Random signed value with |x| < 2^bits.
        let mut words = [0u64; N];
        let full_words = (bits / 64) as usize;
        for w in words.iter_mut().take(full_words) {
            *w = rng.random();
        }
        let rem = bits % 64;
        if rem > 0 && full_words < N {
            words[full_words] = rng.random::<u64>() & ((1u64 << rem) - 1);
        }
        let mag = Int::<N>::new(crypto_bigint::Int::from_words(words));
        if rng.random::<bool>() { -mag } else { mag }
    }

    #[test]
    fn roundtrip_small_values() {
        for x in -1000i32..1000 {
            let v = Int::<2>::from(x);
            let digits = decompose_entry_balanced::<2, 2>(&v, 34, 2).unwrap();
            let rec = recombine::<2, 2>(&[vec![digits[0]], vec![digits[1]]], 34);
            assert_eq!(rec[0], v, "roundtrip failed for {x}");
        }
    }

    #[test]
    fn roundtrip_random_wide() {
        let mut rng = StdRng::seed_from_u64(7);
        let p = 130u32;
        for _ in 0..200 {
            // |x| < 2^{2p-2} < 2^{2p-1}, representable in two digits.
            let x = random_int::<6>(&mut rng, 2 * p - 2);
            let digits = decompose_entry_balanced::<6, 3>(&x, p, 2).unwrap();
            let half = Int::<3>::ONE << (p - 1);
            for d in &digits {
                assert!(*d < half && *d >= -half, "digit out of balanced range");
            }
            let limbs: Vec<Vec<Int<3>>> = digits.into_iter().map(|d| vec![d]).collect();
            let rec = recombine::<3, 6>(&limbs, p);
            assert_eq!(rec[0], x);
        }
    }

    #[test]
    fn too_wide_value_errors() {
        let p = 34u32;
        // +2^{pk-1} is not representable in balanced digits (the range is
        // asymmetric: -2^{pk-1} is, +2^{pk-1} is not).
        let too_big = Int::<2>::ONE << (2 * p - 1);
        assert!(decompose_entry_balanced::<2, 2>(&too_big, p, 2).is_err());
        // The exact positive maximum: (2^{p-1}-1) * (1 + 2^p).
        let max_ok = ((Int::<2>::ONE << (p - 1)) - Int::<2>::ONE)
            * ((Int::<2>::ONE << p) + Int::<2>::ONE);
        assert!(decompose_entry_balanced::<2, 2>(&max_ok, p, 2).is_ok());
        assert!(
            decompose_entry_balanced::<2, 2>(&add!(max_ok, Int::<2>::ONE), p, 2).is_err()
        );
        let min_ok = -(Int::<2>::ONE << (2 * p - 1));
        assert!(decompose_entry_balanced::<2, 2>(&min_ok, p, 2).is_ok());
    }

    #[test]
    fn limb_count_recursion_stabilizes_at_two() {
        // p = lambda + 2 keeps k = 2 (draft Lemma 5.2 at arity 2).
        let lambda = 128u32;
        let p = lambda + 2;
        assert_eq!(num_limbs_for(p - 1, p), 1);
        let mut k = 1usize;
        for _ in 0..50 {
            k = next_limb_count(lambda, p, k);
            assert!(k <= 2);
        }
        assert_eq!(k, 2);
        // p = lambda + 1 is not enough.
        assert!(next_limb_count(lambda, lambda + 1, 2) > 2);
        // Arity 8 (s = 3): stable at two limbs iff p >= lambda + s + 1.
        assert_eq!(next_limb_count_arity(lambda, lambda + 4, 2, 3), 2);
        assert!(next_limb_count_arity(lambda, lambda + 3, 2, 3) > 2);
    }

    #[test]
    fn stability_under_simulated_folds() {
        // Simulate R rounds: fold 2k limb halves with fresh 128-bit
        // challenges, re-decompose, check the digits stay in range with
        // k = 2 limbs throughout.
        let mut rng = StdRng::seed_from_u64(42);
        let lambda = 128u32;
        let p = lambda + 2;
        let len = 64usize;
        let mut limbs: Vec<Vec<Int<3>>> = vec![
            (0..len)
                .map(|_| random_int::<3>(&mut rng, p - 1))
                .collect(),
        ];
        for round in 0..6 {
            let k = limbs.len();
            let half_len = limbs[0].len() / 2;
            let mut folded: Vec<Int<8>> = vec![Int::ZERO; half_len];
            for limb in &limbs {
                let alpha_e = random_int::<8>(&mut rng, lambda).inner().abs();
                let alpha_o = random_int::<8>(&mut rng, lambda).inner().abs();
                let alpha_e = Int::<8>::new(crypto_bigint::Int::from_words(alpha_e.to_words()));
                let alpha_o = Int::<8>::new(crypto_bigint::Int::from_words(alpha_o.to_words()));
                for (i, acc) in folded.iter_mut().enumerate() {
                    use zinc_utils::mul_by_scalar::MulByScalar;
                    let e = limb[2 * i].resize::<8>().mul_by_scalar::<true>(&alpha_e).unwrap();
                    let o = limb[2 * i + 1].resize::<8>().mul_by_scalar::<true>(&alpha_o).unwrap();
                    *acc = add!(*acc, add!(e, o));
                }
            }
            let k_next = next_limb_count(lambda, p, k);
            assert_eq!(k_next, 2, "round {round}: limb count escaped 2");
            limbs = decompose_balanced::<8, 3>(&folded, p, k_next).unwrap();
            let rec = recombine::<3, 8>(&limbs, p);
            assert_eq!(rec, folded, "round {round}: recombination mismatch");
        }
    }
}
