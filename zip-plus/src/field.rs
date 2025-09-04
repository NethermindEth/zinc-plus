#![allow(non_snake_case)]

use std::iter::{Product, Sum};
use crate::traits::{Words as WordsTrait};
use ark_ff::UniformRand;
use crypto_bigint::Random;
use num_traits::{CheckedAdd, CheckedMul, CheckedNeg, CheckedShl, CheckedShr, CheckedSub, ConstOne, ConstZero, Pow};
use std::marker::PhantomData;
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Shl, Shr, Sub, SubAssign};

mod arithmetic;
mod biginteger;
mod comparison;
pub mod config;
mod conversion;
mod constant;
mod int;
mod uint;

pub use biginteger::{
    BigInt, BigInteger64, BigInteger128, BigInteger256, BigInteger320, BigInteger384,
    BigInteger448, BigInteger768, BigInteger832, Words, signed_mod_reduction,
};
pub use config::FieldConfig;
use crypto_primitives::{ConstRing, Field, IntRing, PrimeField, Ring};
pub use int::Int;
pub use uint::Uint;

use crate::{
    field::config::{FieldConfigBase, FieldConfigOps},
    traits::{BigInteger},
    transcript::KeccakTranscript,
};

#[derive(Copy, Clone)]
pub struct RandomField<const N: usize, FC: FieldConfig<BigInt<N>>> {
    pub value: BigInt<N>,
    phantom_data: PhantomData<FC>,
}

impl<const N: usize, FC: FieldConfig<BigInt<N>>> Ring for RandomField<N, FC> {}

impl<const N: usize, FC: FieldConfig<BigInt<N>>> ConstRing for RandomField<N, FC> {}

impl<const N: usize, FC: FieldConfig<BigInt<N>>> IntRing for RandomField<N, FC> {}

impl<const N: usize, FC: FieldConfig<BigInt<N>>> Field for RandomField<N, FC> {}

impl<const N: usize, FC: FieldConfig<BigInt<N>>> PrimeField for RandomField<N, FC> {
    type Inner = BigInt<N>;
    const MODULUS: Self::Inner = FC::modulus();

    #[inline(always)]
    fn new_unchecked(value: Self::Inner) -> Self {
        Self {
            value,
            phantom_data: PhantomData,
        }
    }

    #[inline(always)]
    fn inner(&self) -> &Self::Inner {
        &self.value
    }
}

impl<const N: usize, FC: FieldConfig<BigInt<N>>> RandomField<N, FC> {
    #[inline]
    pub fn into_bigint(self) -> BigInt<N> {
        self.value.demontgomery(&FC::modulus(), FC::inv())
    }
}

impl<const N: usize, FC: FieldConfig<BigInt<N>>> Random for RandomField<N, FC> {
    fn random(rng: &mut (impl ark_std::rand::RngCore + ?Sized)) -> Self {
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
                return Self::from(value);
            }
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
