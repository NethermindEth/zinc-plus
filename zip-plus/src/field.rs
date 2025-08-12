#![allow(non_snake_case)]

use std::marker::PhantomData;
use ark_ff::UniformRand;
use crypto_bigint::Random;
use num_traits::ConstZero;
use crate::traits::{Config, ConfigReference, FieldMap, FromBytes, Words as WordsTrait};

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
pub use config::{ConfigRef, DebugFieldConfig, FieldConfig};
pub use int::Int;
pub use uint::Uint;
#[derive(Copy, Clone)]
pub struct RandomField<'cfg, const N: usize, FC: ConstFieldConfig<N>> {
    pub value: BigInt<N>,
    pub config: ConfigRef<'cfg, N>,
    phantom_data: PhantomData<FC>
}

use crate::{
    traits::{BigInteger, Field},
    transcript::KeccakTranscript,
};
use crate::field::config::ConstFieldConfig;

impl<'cfg, const N: usize, FC: ConstFieldConfig<N>> RandomField<'cfg, N, FC> {

    pub fn config(&self) -> &FieldConfig<N>  {
        self.config.reference().expect("Field config cannot be none")
    }

    pub fn zero_with_config(config: ConfigRef<'cfg, N>) -> Self {
        Self {
            config,
            value: BigInt::ZERO,
            phantom_data: PhantomData,
        }
    }

    #[inline(always)]
    pub fn config_ptr(&self) -> ConfigRef<'cfg, N> {
        self.config
    }

    /// Convert from `BigInteger` to `RandomField`
    ///
    /// If `BigInteger` is greater then field modulus return `None`
    pub fn from_bigint<'a>(config: ConfigRef<'a, N>, value: BigInt<N>) -> Option<RandomField<'a, N, FC>>
    where
        'cfg: 'a,
    {
        let config_ref = config.reference().expect("Field config cannot be none");

        if value >= *config_ref.modulus() {
            None
        } else {
            let mut r = value;
            config_ref.mul_assign(&mut r, config_ref.r2());

            Some(RandomField::new_unchecked(config, r))
        }
    }

    pub fn from_i64(value: i64, config: ConfigRef<'cfg, N>) -> Option<RandomField<'cfg, N, FC>> {
        let config_ref = config.reference().expect("Field config cannot be none");

        if BigInt::from(value.unsigned_abs()) >= *config_ref.modulus() {
            None
        } else {
            let mut r = value.unsigned_abs().into();
            config_ref.mul_assign(&mut r, config_ref.r2());

            let mut elem = RandomField::new_unchecked(config, r);
            if value.is_negative() {
                elem = -elem;
            }
            Some(elem)
        }
    }

    #[inline]
    pub fn into_bigint(self) -> BigInt<N> {
        Self::demontgomery(self.config(), self.value)
    }

    #[inline]
    fn demontgomery(config: &FieldConfig<N>, value: BigInt<N>) -> BigInt<N> {
        value.demontgomery(config.modulus(), config.inv())
    }
}

impl<'cfg, const N: usize, FC: ConstFieldConfig<N>> Field for RandomField<'cfg, N, FC> {
    type B = BigInt<N>;
    type C = FieldConfig<N>;
    type R = ConfigRef<'cfg, N>;
    type W = Words<N>;
    type I = Int<N>;
    type U = Uint<N>;
    type DebugField = DebugRandomField;

    fn new_unchecked(config: ConfigRef<'cfg, N>, value: BigInt<N>) -> Self {
        Self { config, value, phantom_data: PhantomData }
    }

    fn rand_with_config<R: ark_std::rand::Rng + ?Sized>(rng: &mut R, config: Self::R) -> Self {
        loop {
            let mut value = BigInt::rand(rng);
            let modulus = config
                .reference()
                .expect("Field config cannot be none")
                .modulus();
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

            if value < *modulus {
                return value.map_to_field(config);
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
        let config = self.config();

        transcript.absorb(&[0x3]);
        transcript.absorb(&config.modulus().to_bytes_be());
        transcript.absorb(&[0x5]);

        transcript.absorb(&[0x1]);
        transcript.absorb(&self.value.to_bytes_be());
        transcript.absorb(&[0x3])
    }
}

impl<const N: usize, FC: ConstFieldConfig<N>> UniformRand for RandomField<'_, N, FC> {
    fn rand<R: ark_std::rand::Rng + ?Sized>(rng: &mut R) -> Self {
        let value = BigInt::rand(rng);

        Self { value, config: unsafe { ConfigRef::new(Box::leak(Box::new(FC::field_config()))) }, phantom_data: PhantomData }
    }
}

impl<const N: usize, FC: ConstFieldConfig<N>> Random for RandomField<'_, N, FC> {
    fn random(rng: &mut (impl ark_std::rand::RngCore + ?Sized)) -> Self {
        let value = BigInt::rand(rng);

        Self { value, config: unsafe { ConfigRef::new(Box::leak(Box::new(FC::field_config()))) }, phantom_data: PhantomData }
    }
}

impl<const N: usize, FC: ConstFieldConfig<N>> ark_std::fmt::Debug for RandomField<'_, N, FC> {
    fn fmt(&self, f: &mut ark_std::fmt::Formatter<'_>) -> ark_std::fmt::Result {
        write!(
            f,
            "{} in Z_{}",
            self.clone().into_bigint(),
            self.config_ptr().reference().unwrap().modulus()
        )
    }
}

impl<const N: usize, FC: ConstFieldConfig<N>> ark_std::fmt::Display for RandomField<'_, N, FC> {
    fn fmt(&self, f: &mut ark_std::fmt::Formatter<'_>) -> ark_std::fmt::Result {
        write!(f, "{}", self.clone().into_bigint())
    }
}

impl<const N: usize, FC: ConstFieldConfig<N>> Default for RandomField<'_, N, FC> {
    fn default() -> Self {
        Self {
            value: BigInt::ZERO,
            config: unsafe { ConfigRef::new(Box::leak(Box::new(FC::field_config()))) },
            phantom_data: PhantomData
        }
    }
}

unsafe impl<const N: usize, FC: ConstFieldConfig<N>> Send for RandomField<'_, N, FC> {}
unsafe impl<const N: usize, FC: ConstFieldConfig<N>> Sync for RandomField<'_, N, FC> {}

#[derive(Debug)]
pub enum DebugRandomField {
    Raw {
        value: num_bigint::BigInt,
    },

    Initialized {
        config: DebugFieldConfig,
        value: num_bigint::BigInt,
    },
}

impl<const N: usize, FC: ConstFieldConfig<N>> From<RandomField<'_, N, FC>> for DebugRandomField {
    fn from(value: RandomField<'_, N, FC>) -> Self {
        Self::Initialized {
            config: (*value.config.reference().unwrap()).into(),
            value: value.value.into(),
        }
    }
}

impl ark_std::fmt::Display for DebugRandomField {
    fn fmt(&self, f: &mut ark_std::fmt::Formatter<'_>) -> ark_std::fmt::Result {
        match self {
            Self::Raw { value } => {
                write!(f, "{value}")
            }
            self_ @ Self::Initialized { .. } => {
                write!(f, "{self_}")
            }
        }
    }
}

impl<const N: usize, FC: ConstFieldConfig<N>> From<u128> for RandomField<'_, N, FC> {
    fn from(value: u128) -> Self {
        value.map_to_field(unsafe { ConfigRef::new(Box::leak(Box::new(FC::field_config()))) })
    }
}

macro_rules! impl_from_uint {
    ($type:ty) => {
        impl<const N: usize, FC: ConstFieldConfig<N>> From<$type> for RandomField<'_, N, FC> {
            fn from(value: $type) -> Self {
                value.map_to_field(unsafe { ConfigRef::new(Box::leak(Box::new(FC::field_config()))) })
            }
        }
    };
}

impl_from_uint!(u64);
impl_from_uint!(u32);
impl_from_uint!(u16);
impl_from_uint!(u8);

impl<const N: usize, FC: ConstFieldConfig<N>> From<bool> for RandomField<'_, N, FC> {
    fn from(value: bool) -> Self {
        value.map_to_field(unsafe { ConfigRef::new(Box::leak(Box::new(FC::field_config()))) })
    }
}

impl<const N: usize, FC: ConstFieldConfig<N>> FromBytes for RandomField<'_, N, FC> {
    fn from_bytes_le(bytes: &[u8]) -> Option<Self> {
        Some(RandomField {
            value: BigInt::<N>::from_bytes_le(bytes)?,
            config: unsafe { ConfigRef::new(Box::leak(Box::new(FC::field_config()))) },
            phantom_data: PhantomData
        })
    }

    fn from_bytes_be(bytes: &[u8]) -> Option<Self> {
        Some(RandomField {
            value: BigInt::<N>::from_bytes_be(bytes)?,
            config: unsafe { ConfigRef::new(Box::leak(Box::new(FC::field_config()))) },
            phantom_data: PhantomData
        })
    }
}

impl<'cfg, const N: usize, FC: ConstFieldConfig<N>> RandomField<'cfg, N, FC> {
    pub fn from_bytes_le_with_config(config: ConfigRef<'cfg, N>, bytes: &[u8]) -> Option<Self> {
        let value = BigInt::<N>::from_bytes_le(bytes);

        Self::from_bigint(config, value?)
    }

    pub fn from_bytes_be_with_config(config: ConfigRef<'cfg, N>, bytes: &[u8]) -> Option<Self> {
        let value = BigInt::<N>::from_bytes_be(bytes);

        Self::from_bigint(config, value?)
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

    fn map_to_field(&self, config_ref: F::R) -> Self::Output {
        let config = match config_ref.reference() {
            Some(config) => config,
            None => panic!("Cannot convert BigInt to prime field element without a modulus"),
        };

        let mut value = if M > F::W::num_words() {
            let modulus: Int<M> = config.modulus().into();
            let mut value: Int<M> = self.into();
            value %= modulus;

            F::B::from(value)
        } else {
            let modulus: F::I = config.modulus().into();
            let mut value: F::I = self.into();
            value %= modulus;

            value.into()
        };

        config.mul_assign(&mut value, config.r2());

        F::new_unchecked(config_ref, value)
    }
}

// Implementation of FieldMap for reference to BigInt<N>
impl<F: Field, const M: usize> FieldMap<F> for &BigInt<M>
where
    BigInt<M>: FieldMap<F, Output = F>,
{
    type Output = F;
    fn map_to_field(&self, config_ref: F::R) -> Self::Output {
        (*self).map_to_field(config_ref)
    }
}
