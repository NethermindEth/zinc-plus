#![allow(non_snake_case)]

use crate::traits::{FieldMap, FromBytes, Words as WordsTrait};
use ark_ff::UniformRand;
use crypto_bigint::Random;
use num_traits::ConstZero;
use std::marker::PhantomData;

mod arithmetic;
mod biginteger;
mod comparison;
pub mod config;
mod constant;
mod int;
mod uint;

pub use biginteger::{
    BigInt, BigInteger64, BigInteger128, BigInteger256, BigInteger320, BigInteger384,
    BigInteger448, BigInteger768, BigInteger832, Words, signed_mod_reduction,
};
pub use config::FieldConfig;
pub use int::Int;
pub use uint::Uint;

#[derive(Copy, Clone)]
pub struct RandomField<const N: usize, FC: FieldConfig<BigInt<N>>> {
    pub value: BigInt<N>,
    phantom_data: PhantomData<FC>,
}

use crate::field::config::{FieldConfigBase, FieldConfigOps};
use crate::{
    traits::{BigInteger, Field},
    transcript::KeccakTranscript,
};

impl<const N: usize, FC: FieldConfig<BigInt<N>>> RandomField<N, FC> {
    /// Convert from `BigInteger` to `RandomField`
    ///
    /// If `BigInteger` is greater then field modulus return `None`
    pub fn from_bigint(value: BigInt<N>) -> Option<RandomField<N, FC>> {
        if value >= FC::modulus() {
            None
        } else {
            let mut r = value;
            FC::mul_assign(&mut r, &FC::r_squared());

            Some(RandomField::new_unchecked(r))
        }
    }

    pub fn from_i64(value: i64) -> Option<RandomField<N, FC>> {
        if BigInt::from(value.unsigned_abs()) >= FC::modulus() {
            None
        } else {
            let mut r = value.unsigned_abs().into();
            FC::mul_assign(&mut r, &FC::r_squared());

            let mut elem = RandomField::new_unchecked(r);
            if value.is_negative() {
                elem = -elem;
            }
            Some(elem)
        }
    }

    #[inline]
    pub fn into_bigint(self) -> BigInt<N> {
        self.value.demontgomery(&FC::modulus(), FC::inv())
    }
}

impl<const N: usize, FC: FieldConfig<BigInt<N>>> Field for RandomField<N, FC> {
    type B = BigInt<N>;
    type C = FC;
    type W = Words<N>;
    type I = Int<N>;
    type U = Uint<N>;

    fn new_unchecked(value: BigInt<N>) -> Self {
        Self {
            value,
            phantom_data: PhantomData,
        }
    }

    fn rand<R: ark_std::rand::Rng + ?Sized>(rng: &mut R) -> Self {
        loop {
            let mut value = BigInt::rand(rng);
            let modulus = FC::modulus();
            let shave_bits = 64 * N - modulus.num_bits() as usize;
            // Mask away the unused bits at the beginning.
            assert!(shave_bits <= 64);
            let mask = if shave_bits == 64 {
                0
            } else {
                u64::MAX >> shave_bits
            };

            let val = value.last_mut();
            *val &= mask;

            if value < modulus {
                return value.map_to_field();
            }
        }
    }

    #[inline(always)]
    fn value(&self) -> &BigInt<N> {
        &self.value
    }

    #[inline(always)]
    fn value_mut(&mut self) -> &mut BigInt<N> {
        &mut self.value
    }

    fn absorb_into_transcript(&self, transcript: &mut KeccakTranscript) {
        transcript.absorb(&[0x3]);
        transcript.absorb(&FC::modulus().to_bytes_be());
        transcript.absorb(&[0x5]);

        transcript.absorb(&[0x1]);
        transcript.absorb(&self.value.to_bytes_be());
        transcript.absorb(&[0x3])
    }
}

impl<const N: usize, FC: FieldConfig<BigInt<N>>> UniformRand for RandomField<N, FC> {
    fn rand<R: ark_std::rand::Rng + ?Sized>(rng: &mut R) -> Self {
        let value = BigInt::rand(rng);

        Self {
            value,
            phantom_data: PhantomData,
        }
    }
}

impl<const N: usize, FC: FieldConfig<BigInt<N>>> Random for RandomField<N, FC> {
    fn random(rng: &mut (impl ark_std::rand::RngCore + ?Sized)) -> Self {
        let value = BigInt::rand(rng);

        Self {
            value,
            phantom_data: PhantomData,
        }
    }
}

impl<const N: usize, FC: FieldConfig<BigInt<N>>> ark_std::fmt::Debug for RandomField<N, FC> {
    fn fmt(&self, f: &mut ark_std::fmt::Formatter<'_>) -> ark_std::fmt::Result {
        write!(f, "{} in Z_{}", self.clone().into_bigint(), FC::modulus())
    }
}

impl<const N: usize, FC: FieldConfig<BigInt<N>>> ark_std::fmt::Display for RandomField<N, FC> {
    fn fmt(&self, f: &mut ark_std::fmt::Formatter<'_>) -> ark_std::fmt::Result {
        write!(f, "{}", self.clone().into_bigint())
    }
}

impl<const N: usize, FC: FieldConfig<BigInt<N>>> Default for RandomField<N, FC> {
    fn default() -> Self {
        Self {
            value: BigInt::ZERO,
            phantom_data: PhantomData,
        }
    }
}

unsafe impl<const N: usize, FC: FieldConfig<BigInt<N>>> Send for RandomField<N, FC> {}
unsafe impl<const N: usize, FC: FieldConfig<BigInt<N>>> Sync for RandomField<N, FC> {}

impl<const N: usize, FC: FieldConfig<BigInt<N>>> From<u128> for RandomField<N, FC> {
    fn from(value: u128) -> Self {
        value.map_to_field()
    }
}

macro_rules! impl_from_uint {
    ($type:ty) => {
        impl<const N: usize, FC: FieldConfig<BigInt<N>>> From<$type> for RandomField<N, FC> {
            fn from(value: $type) -> Self {
                value.map_to_field()
            }
        }
    };
}

impl_from_uint!(u64);
impl_from_uint!(u32);
impl_from_uint!(u16);
impl_from_uint!(u8);

impl<const N: usize, FC: FieldConfig<BigInt<N>>> From<bool> for RandomField<N, FC> {
    fn from(value: bool) -> Self {
        value.map_to_field()
    }
}

impl<const N: usize, FC: FieldConfig<BigInt<N>>> FromBytes for RandomField<N, FC> {
    fn from_bytes_le(bytes: &[u8]) -> Option<Self> {
        Self::from_bigint(BigInt::<N>::from_bytes_le(bytes)?)
    }

    fn from_bytes_be(bytes: &[u8]) -> Option<Self> {
        Self::from_bigint(BigInt::<N>::from_bytes_be(bytes)?)
    }
}

// Implementation of FieldMap for BigInt<N>
impl<F: Field, const M: usize> FieldMap<F> for BigInt<M>
where
    for<'a> Int<M>: From<&'a F::B>,
    F::B: From<Int<M>>,
    for<'a> F::I: From<&'a BigInt<M>>,
{
    type Output = F;

    fn map_to_field(&self) -> Self::Output {
        let mut value = if M > F::W::num_words() {
            let modulus: Int<M> = (&F::C::modulus()).into();
            let mut value: Int<M> = self.into();
            value %= modulus;

            F::B::from(value)
        } else {
            let modulus: F::I = (&F::C::modulus()).into();
            let mut value: F::I = self.into();
            value %= modulus;

            value.into()
        };

        F::C::mul_assign(&mut value, &F::C::r_squared());

        F::new_unchecked(value)
    }
}

// Implementation of FieldMap for reference to BigInt<N>
impl<F: Field, const M: usize> FieldMap<F> for &BigInt<M>
where
    BigInt<M>: FieldMap<F, Output = F>,
{
    type Output = F;
    fn map_to_field(&self) -> Self::Output {
        (*self).map_to_field()
    }
}
