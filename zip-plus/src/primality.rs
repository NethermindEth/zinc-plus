use crate::utils::UintSemiring;
use crypto_bigint::{Odd, Uint};
use crypto_primes::hazmat::MillerRabin;

pub trait PrimalityTest<R> {
    fn is_probably_prime(candidate: &R) -> bool;
}

impl<const LIMBS: usize> PrimalityTest<UintSemiring<LIMBS>> for MillerRabin<Uint<LIMBS>> {
    fn is_probably_prime(candidate: &UintSemiring<LIMBS>) -> bool {
        let Some(odd) = Odd::new(candidate.0).into_option() else {
            return false;
        };
        let test = Self::new(odd);
        test.test_base_two().is_probably_prime()
    }
}
