use crypto_bigint::Odd;
use crypto_primitives::crypto_bigint_uint::Uint;

pub trait PrimalityTest<R> {
    fn is_probably_prime(candidate: &R) -> bool;
}

#[derive(Debug, Clone, Copy)]
pub struct MillerRabin {}

impl<const LIMBS: usize> PrimalityTest<Uint<LIMBS>> for MillerRabin {
    fn is_probably_prime(candidate: &Uint<LIMBS>) -> bool {
        let Some(odd) = Odd::new(*candidate.inner()).into_option() else {
            return false;
        };
        let test = crypto_primes::hazmat::MillerRabin::new(odd);
        test.test_base_two().is_probably_prime()
    }
}
