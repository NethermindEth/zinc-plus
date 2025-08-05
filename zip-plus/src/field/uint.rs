use ark_std::ops::{Mul, SubAssign};
use crypto_bigint::{Integer, Odd, Uint as CryptoUint};
use crypto_primes::hazmat::MillerRabin;
use num_traits::One;

use crate::{
    field::{biginteger::Words, Int},
    traits::{types::PrimalityTest, FromBytes, Uinteger},
};

#[derive(Clone)]
pub struct Uint<const N: usize>(pub(crate) CryptoUint<N>);

impl<const N: usize> Mul<Self> for Uint<N> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self(self.0 * rhs.0)
    }
}

impl<const N: usize> One for Uint<N> {
    fn one() -> Self {
        Self(CryptoUint::ONE)
    }
}

impl<'a, const N: usize> SubAssign<&'a Self> for Uint<N> {
    fn sub_assign(&mut self, rhs: &'a Self) {
        self.0 -= rhs.0;
    }
}

impl<const N: usize> Uinteger for Uint<N> {
    type W = Words<N>;
    type Int = Int<N>;
    type PrimalityTest = MillerRabin<CryptoUint<N>>;

    fn from_words(words: Words<N>) -> Self {
        Self(CryptoUint::from_words(words.0))
    }

    fn as_int(&self) -> Self::Int {
        Int(self.0.as_int())
    }

    fn to_words(self) -> Words<N> {
        Words(self.0.to_words())
    }

    fn is_even(&self) -> bool {
        self.0.is_even().unwrap_u8() == 1
    }
}

impl<const N: usize> FromBytes for Uint<N> {
    fn from_bytes_le(bytes: &[u8]) -> Option<Self> {
        Some(Self(CryptoUint::from_le_slice(bytes)))
    }

    fn from_bytes_be(bytes: &[u8]) -> Option<Self> {
        Some(Self(CryptoUint::from_be_slice(bytes)))
    }
}

impl<const N: usize> PrimalityTest<Uint<N>> for MillerRabin<CryptoUint<N>> {
    type Inner = CryptoUint<N>;

    fn new(candidate: Uint<N>) -> Self {
        Self::new(Odd::new(candidate.0).unwrap())
    }

    fn is_probably_prime(&self) -> bool {
        self.test_base_two().is_probably_prime()
    }
}
