#![allow(non_snake_case)]

use ark_ff::UniformRand;
use crypto_bigint::Random;
use num_traits::ConstZero;
use crate::traits::{Config, ConfigReference, FieldMap, FromBytes, Words as WordsTrait};

mod arithmetic;
mod biginteger;
mod comparison;
mod config;
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
pub struct RandomField<'cfg, const N: usize> {
    pub value: BigInt<N>,
    pub config: Option<ConfigRef<'cfg, N>>,
}

use crate::{
    traits::{BigInteger, Field},
    transcript::KeccakTranscript,
};

impl<'cfg, const N: usize> RandomField<'cfg, N> {
    pub fn with_either<R, I, A>(&self, raw_fn: R, init_fn: I) -> A
    where
        I: Fn(&FieldConfig<N>, &BigInt<N>) -> A,
        R: Fn(&BigInt<N>) -> A,
    {
        match self.config {
            None => raw_fn(&self.value),
            Some(config) => init_fn(
                config.reference().expect("Field config cannot be none"),
                &self.value,
            ),
        }
    }

    pub fn with_either_mut<R, I, A>(&mut self, raw_fn: R, init_fn: I) -> A
    where
        I: Fn(&FieldConfig<N>, &mut BigInt<N>) -> A,
        R: Fn(&mut BigInt<N>) -> A,
    {
        match self.config {
            None => raw_fn(&mut self.value),
            Some(config) => init_fn(
                config.reference().expect("Field config cannot be none"),
                &mut self.value,
            ),
        }
    }

    pub fn with_either_owned<R, I, A>(self, raw_fn: R, init_fn: I) -> A
    where
        I: Fn(&FieldConfig<N>, BigInt<N>) -> A,
        R: Fn(BigInt<N>) -> A,
    {
        match self.config {
            None => raw_fn(self.value),
            Some(config) => init_fn(
                config.reference().expect("Field config cannot be none"),
                self.value,
            ),
        }
    }

    pub fn with_aligned_config_mut<F, G, A>(
        &mut self,
        rhs: &Self,
        fn_with_config: F,
        fn_without_config: G,
    ) -> A
    where
        F: Fn(&mut BigInt<N>, &BigInt<N>, &FieldConfig<N>) -> A,
        G: Fn(&mut BigInt<N>, &BigInt<N>) -> A,
    {
        match (self.config, rhs.config) {
            (None, None) => fn_without_config(&mut self.value, &rhs.value),
            (Some(config), Some(_)) => fn_with_config(
                &mut self.value,
                &rhs.value,
                config.reference().expect("Field config cannot be none"),
            ),
            (Some(config), None) => {
                let rhs = rhs.with_config(config);
                fn_with_config(
                    &mut self.value,
                    &rhs.value,
                    config.reference().expect("Field config cannot be none"),
                )
            }
            (None, Some(config)) => {
                *self = self.with_config(config);
                fn_with_config(
                    &mut self.value,
                    &rhs.value,
                    config.reference().expect("Field config cannot be none"),
                )
            }
        }
    }

    pub fn zero_with_config(config: ConfigRef<'cfg, N>) -> Self {
        Self {
            config: Some(config),
            value: BigInt::ZERO,
        }
    }

    #[inline(always)]
    pub fn config_ptr(&self) -> ConfigRef<'cfg, N> {
        match self.config {
            None => ConfigRef::NONE,
            Some(config) => config,
        }
    }

    /// Convert from `BigInteger` to `RandomField`
    ///
    /// If `BigInteger` is greater then field modulus return `None`
    pub fn from_bigint<'a>(config: ConfigRef<'a, N>, value: BigInt<N>) -> Option<RandomField<'a, N>>
    where
        'cfg: 'a,
    {
        let config_ref = match config.reference() {
            Some(config) => config,
            None => return Some(Self { value, config: None }),
        };

        if value >= *config_ref.modulus() {
            None
        } else {
            let mut r = value;
            config_ref.mul_assign(&mut r, config_ref.r2());

            Some(RandomField::new_unchecked(config, r))
        }
    }

    pub fn from_i64(value: i64, config: ConfigRef<'cfg, N>) -> Option<RandomField<'cfg, N>> {
        let config_ref = match config.reference() {
            Some(config) => config,
            None => {
                panic!("Cannot convert signed integer to prime field element without a modulus")
            }
        };

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
        self.with_either_owned(|value| value, Self::demontgomery)
    }

    #[inline]
    fn demontgomery(config: &FieldConfig<N>, value: BigInt<N>) -> BigInt<N> {
        value.demontgomery(config.modulus(), config.inv())
    }
}

impl<'cfg, const N: usize> Field for RandomField<'cfg, N> {
    type B = BigInt<N>;
    type C = FieldConfig<N>;
    type R = ConfigRef<'cfg, N>;
    type W = Words<N>;
    type I = Int<N>;
    type U = Uint<N>;
    type DebugField = DebugRandomField;

    fn new_unchecked(config: ConfigRef<'cfg, N>, value: BigInt<N>) -> Self {
        Self { config: Some(config), value }
    }

    fn without_config(value: Self::B) -> Self {
        Self { value, config: None }
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

    fn with_config(self, config: Self::R) -> Self {
        let value = match self.config {
            None => {
                // Ideally we should do something like:
                //
                // ```
                // let modulus: BigInt<N> = unsafe { (*config).modulus };
                // *value = *value % modulus;
                // ```
                //
                // but we don't have `mod` out of the box.
                // So let's hope we don't exceed the modulus.

                // TODO: prettify this

                *Self::from_bigint(config, self.value)
                    .expect("Should not end up with a None here.")
                    .value()
            }
            Some(_) => { panic!("Cannot convert initialized field to raw field without a value") },
        };

        Self { config: Some(config), value }
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
        if let Some(config) = self.config {
            let config = config.reference().expect("Field config cannot be none");

            transcript.absorb(&[0x3]);
            transcript.absorb(&config.modulus().to_bytes_be());
            transcript.absorb(&[0x5]);
        }

        transcript.absorb(&[0x1]);
        transcript.absorb(&self.value.to_bytes_be());
        transcript.absorb(&[0x3])
    }
}

impl<const N: usize> UniformRand for RandomField<'_, N> {
    fn rand<R: ark_std::rand::Rng + ?Sized>(rng: &mut R) -> Self {
        let value = BigInt::rand(rng);

        Self { value, config: None }
    }
}

impl<const N: usize> Random for RandomField<'_, N> {
    fn random(rng: &mut (impl ark_std::rand::RngCore + ?Sized)) -> Self {
        let value = BigInt::rand(rng);

        Self { value, config: None }
    }
}

impl<const N: usize> ark_std::fmt::Debug for RandomField<'_, N> {
    fn fmt(&self, f: &mut ark_std::fmt::Formatter<'_>) -> ark_std::fmt::Result {
        match self.config {
            None => write!(f, "{}, no config", self.value),
            Some(_) => write!(
                f,
                "{} in Z_{}",
                self.into_bigint(),
                self.config_ptr().reference().unwrap().modulus()
            ),
        }
    }
}

impl<const N: usize> ark_std::fmt::Display for RandomField<'_, N> {
    fn fmt(&self, f: &mut ark_std::fmt::Formatter<'_>) -> ark_std::fmt::Result {
        // TODO: we should go back from Montgomery here.
        match self.config {
            None => {
                write!(f, "{}", self.value)
            }
            Some(_) => {
                write!(f, "{}", self.into_bigint())
            }
        }
    }
}

impl<const N: usize> Default for RandomField<'_, N> {
    fn default() -> Self {
        Self {
            value: BigInt::ZERO,
            config: None,
        }
    }
}

unsafe impl<const N: usize> Send for RandomField<'_, N> {}
unsafe impl<const N: usize> Sync for RandomField<'_, N> {}

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

impl<const N: usize> From<RandomField<'_, N>> for DebugRandomField {
    fn from(value: RandomField<'_, N>) -> Self {
        match value.config {
            None => Self::Raw {
                value: value.value.into(),
            },
            Some(config) => Self::Initialized {
                config: (*config.reference().unwrap()).into(),
                value: value.value.into(),
            },
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

impl<const N: usize> From<u128> for RandomField<'_, N> {
    fn from(value: u128) -> Self {
        let value = BigInt::from(value);

        RandomField { value, config: None }
    }
}

macro_rules! impl_from_uint {
    ($type:ty) => {
        impl<const N: usize> From<$type> for RandomField<'_, N> {
            fn from(value: $type) -> Self {
                let value = BigInt::from(value);
                RandomField { value, config: None }
            }
        }
    };
}

impl_from_uint!(u64);
impl_from_uint!(u32);
impl_from_uint!(u16);
impl_from_uint!(u8);

impl<const N: usize> From<bool> for RandomField<'_, N> {
    fn from(value: bool) -> Self {
        let value = BigInt::from(value as u8);
        RandomField { value, config: None }
    }
}

impl<const N: usize> FromBytes for RandomField<'_, N> {
    fn from_bytes_le(bytes: &[u8]) -> Option<Self> {
        Some(RandomField {
            value: BigInt::<N>::from_bytes_le(bytes)?,
            config: None,
        })
    }

    fn from_bytes_be(bytes: &[u8]) -> Option<Self> {
        Some(RandomField {
            value: BigInt::<N>::from_bytes_be(bytes)?,
            config: None,
        })
    }
}

impl<'cfg, const N: usize> RandomField<'cfg, N> {
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
