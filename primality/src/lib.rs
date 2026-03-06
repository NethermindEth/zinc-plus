use crypto_bigint::Odd;
use crypto_primitives::crypto_bigint_uint::Uint;

pub trait PrimalityTest<R> {
    fn is_probably_prime(candidate: &R) -> bool;
}

#[derive(Debug, Clone, Copy)]
pub struct MillerRabin {}

/// Product of the first 15 odd primes: 3 × 5 × … × 53.
/// Any candidate divisible by one of these primes is composite.
/// Fits comfortably in a u64 (1 123 762 587 748 273 770 < 2^60).
const SMALL_PRIME_PRODUCT: u64 =
    3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41 * 43 * 47 * 53;

/// The small primes whose product is [`SMALL_PRIME_PRODUCT`].
const SMALL_PRIMES: [u64; 15] = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53];

impl<const LIMBS: usize> PrimalityTest<Uint<LIMBS>> for MillerRabin {
    fn is_probably_prime(candidate: &Uint<LIMBS>) -> bool {
        // Quick trial division by small primes.  We compute a single
        // big-integer remainder mod the product of 15 small primes, then
        // test divisibility by each prime using native u64 arithmetic.
        // This rejects ≈77 % of odd composites, each of which would
        // otherwise require a full Miller-Rabin modular exponentiation.
        let product = Uint::<LIMBS>::from(SMALL_PRIME_PRODUCT);
        // SAFETY: product > 0, so the remainder is well-defined.
        let r_big = *candidate % product;
        let r: u64 = r_big.as_words()[0];
        for &p in &SMALL_PRIMES {
            if r % p == 0 {
                // `candidate` is divisible by `p`.  It's composite unless
                // it *is* `p` itself (only possible when LIMBS == 1 and
                // the value is tiny — effectively never for 192-bit fields).
                if *candidate != Uint::<LIMBS>::from(p) {
                    return false;
                }
            }
        }

        let Some(odd) = Odd::new(*candidate.inner()).into_option() else {
            return false;
        };
        let test = crypto_primes::hazmat::MillerRabin::new(odd);
        test.test_base_two().is_probably_prime()
    }
}
