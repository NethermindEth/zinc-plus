#![allow(non_snake_case)]

use ark_ff::UniformRand;
use crypto_bigint::Random;

use crate::traits::{Config, ConfigReference, FieldMap, FromBytes, Words as WordsTrait};

mod arithmetic;
mod biginteger;
mod comparison;
mod config;
mod constant;
mod int;
mod uint;

pub use biginteger::{
    signed_mod_reduction, BigInt, BigInteger128, BigInteger256, BigInteger320, BigInteger384,
    BigInteger448, BigInteger64, BigInteger768, BigInteger832, Words,
};
pub use config::{ConfigRef, DebugFieldConfig, FieldConfig};
pub use int::Int;
pub use uint::Uint;
#[derive(Copy, Clone)]
pub enum RandomField<'cfg, const N: usize> {
    Raw {
        value: BigInt<N>,
    },
    Initialized {
        config: ConfigRef<'cfg, N>,
        value: BigInt<N>,
    },
}

use RandomField::*;

use crate::{
    traits::{BigInteger, Field},
    transcript::KeccakTranscript,
};

impl<'cfg, const N: usize> RandomField<'cfg, N> {
    pub fn is_raw(&self) -> bool {
        matches!(self, Raw { .. })
    }

    pub fn is_initialized(&self) -> bool {
        matches!(self, Initialized { .. })
    }

    pub fn with_raw_value_or<F, A>(&self, f: F, default: A) -> A
    where
        F: Fn(&BigInt<N>) -> A,
    {
        match self {
            Raw { value } => f(value),
            _ => default,
        }
    }

    pub fn with_raw_value_mut_or<F, A>(&mut self, f: F, default: A) -> A
    where
        F: Fn(&mut BigInt<N>) -> A,
    {
        match self {
            Raw { value } => f(value),
            _ => default,
        }
    }

    pub fn with_init_value<'a, F, A>(&'a self, f: F) -> Option<A>
    where
        F: Fn(&'a FieldConfig<N>, &'a BigInt<N>) -> A,
    {
        match self {
            Initialized { config, value } => Some(f(
                config.reference().expect("Field config cannot be none"),
                value,
            )),
            _ => None,
        }
    }

    pub fn with_init_value_or<'a, F, A>(&'a self, f: F, default: A) -> A
    where
        F: Fn(&'a FieldConfig<N>, &'a BigInt<N>) -> A,
    {
        match self {
            Initialized { config, value } => f(
                config.reference().expect("Field config cannot be none"),
                value,
            ),
            _ => default,
        }
    }

    pub fn with_either<'a, R, I, A>(&'a self, raw_fn: R, init_fn: I) -> A
    where
        I: Fn(&'a FieldConfig<N>, &'a BigInt<N>) -> A,
        R: Fn(&'a BigInt<N>) -> A,
    {
        match self {
            Raw { value } => raw_fn(value),
            Initialized { config, value } => init_fn(
                config.reference().expect("Field config cannot be none"),
                value,
            ),
        }
    }

    pub fn with_either_mut<'a, R, I, A>(&'a mut self, raw_fn: R, init_fn: I) -> A
    where
        I: Fn(&'a FieldConfig<N>, &'a mut BigInt<N>) -> A,
        R: Fn(&'a mut BigInt<N>) -> A,
    {
        match self {
            Raw { value } => raw_fn(value),
            Initialized { config, value } => init_fn(
                config.reference().expect("Field config cannot be none"),
                value,
            ),
        }
    }

    pub fn with_either_owned<R, I, A>(self, raw_fn: R, init_fn: I) -> A
    where
        I: Fn(&FieldConfig<N>, BigInt<N>) -> A,
        R: Fn(BigInt<N>) -> A,
    {
        match self {
            Raw { value } => raw_fn(value),
            Initialized { config, value } => init_fn(
                config.reference().expect("Field config cannot be none"),
                value,
            ),
        }
    }

    pub fn with_aligned_config_mut<F, G, A>(
        &mut self,
        rhs: &Self,
        with_config: F,
        without_config: G,
    ) -> A
    where
        F: Fn(&mut BigInt<N>, &BigInt<N>, &FieldConfig<N>) -> A,
        G: Fn(&mut BigInt<N>, &BigInt<N>) -> A,
    {
        match (self, rhs) {
            (Raw { value: value_self }, Raw { value: rhs }) => without_config(value_self, rhs),
            (
                Initialized {
                    value: value_self,
                    config,
                },
                Initialized {
                    value: value_rhs, ..
                },
            ) => with_config(
                value_self,
                value_rhs,
                config.reference().expect("Field config cannot be none"),
            ),
            (
                Initialized {
                    value: value_self,
                    config,
                },
                rhs @ Raw { .. },
            ) => {
                let rhs = (*rhs).set_config_owned(*config);
                with_config(
                    value_self,
                    rhs.value(),
                    config.reference().expect("Field config cannot be none"),
                )
            }
            (
                lhs @ Raw { .. },
                Initialized {
                    value: value_rhs,
                    config,
                },
            ) => {
                lhs.set_config(*config);

                with_config(
                    lhs.value_mut(),
                    value_rhs,
                    config.reference().expect("Field config cannot be none"),
                )
            }
        }
    }

    pub fn config_copied(&self) -> Option<FieldConfig<N>> {
        match self {
            Raw { .. } => None,
            Initialized { config, .. } => config.reference().copied(),
        }
    }

    pub fn zero_with_config(config: ConfigRef<'cfg, N>) -> Self {
        Initialized {
            config,
            value: BigInt::zero(),
        }
    }

    /// Config setter that can be used after a `RandomField::rand(...)` call.
    pub fn set_config_owned(mut self, config: ConfigRef<'cfg, N>) -> Self {
        self.set_config(config);
        self
    }

    #[inline(always)]
    pub fn config_ptr(&self) -> ConfigRef<'cfg, N> {
        match self {
            Raw { .. } => ConfigRef::NONE,
            Initialized { config, .. } => *config,
        }
    }

    /// Convert from `BigInteger` to `RandomField`
    ///
    /// If `BigInteger` is greater then field modulus return `None`
    pub fn from_bigint(config: ConfigRef<N>, value: BigInt<N>) -> Option<RandomField<N>> {
        let config_ref = match config.reference() {
            Some(config) => config,
            None => return Some(Raw { value }),
        };

        if value >= *config_ref.modulus() {
            None
        } else {
            let mut r = value;
            config_ref.mul_assign(&mut r, config_ref.r2());

            Some(RandomField::new_unchecked(config, r))
        }
    }

    pub fn from_i64(value: i64, config: ConfigRef<N>) -> Option<RandomField<N>> {
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
        Initialized { config, value }
    }

    fn without_config(value: Self::B) -> Self {
        Raw { value }
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

    fn set_config(&mut self, config: Self::R) {
        self.with_raw_value_mut_or(
            |value| {
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

                *value = *Self::from_bigint(config, *value)
                    .expect("Should not end up with a None here.")
                    .value();
            },
            (),
        );

        let value = ark_std::mem::take(self.value_mut());

        *self = Initialized { config, value }
    }

    #[inline(always)]
    fn value(&self) -> &BigInt<N> {
        match self {
            Raw { value } => value,
            Initialized { value, .. } => value,
        }
    }

    #[inline(always)]
    fn value_mut(&mut self) -> &mut BigInt<N> {
        match self {
            Raw { value } => value,
            Initialized { value, .. } => value,
        }
    }

    fn absorb_into_transcript(&self, transcript: &mut KeccakTranscript) {
        match self {
            Raw { value } => {
                transcript.absorb(&[0x1]);
                transcript.absorb(&value.to_bytes_be());
                transcript.absorb(&[0x3])
            }
            Initialized { config, value } => {
                let config = config.reference().expect("Field config cannot be none");

                transcript.absorb(&[0x3]);
                transcript.absorb(&config.modulus().to_bytes_be());
                transcript.absorb(&[0x5]);

                transcript.absorb(&[0x1]);
                transcript.absorb(&value.to_bytes_be());
                transcript.absorb(&[0x3])
            }
        }
    }
}

impl<const N: usize> UniformRand for RandomField<'_, N> {
    fn rand<R: ark_std::rand::Rng + ?Sized>(rng: &mut R) -> Self {
        let value = BigInt::rand(rng);

        Self::Raw { value }
    }
}

impl<const N: usize> Random for RandomField<'_, N> {
    fn random(rng: &mut (impl ark_std::rand::RngCore + ?Sized)) -> Self {
        let value = BigInt::rand(rng);

        Self::Raw { value }
    }
}

impl<const N: usize> ark_std::fmt::Debug for RandomField<'_, N> {
    fn fmt(&self, f: &mut ark_std::fmt::Formatter<'_>) -> ark_std::fmt::Result {
        match self {
            Raw { value } => write!(f, "{value}, no config"),
            self_ => write!(
                f,
                "{} in Z_{}",
                self_.into_bigint(),
                self.config_ptr().reference().unwrap().modulus()
            ),
        }
    }
}

impl<const N: usize> ark_std::fmt::Display for RandomField<'_, N> {
    fn fmt(&self, f: &mut ark_std::fmt::Formatter<'_>) -> ark_std::fmt::Result {
        // TODO: we should go back from Montgomery here.
        match self {
            Raw { value } => {
                write!(f, "{value}")
            }
            self_ @ Initialized { .. } => {
                write!(f, "{}", self_.into_bigint())
            }
        }
    }
}

impl<const N: usize> Default for RandomField<'_, N> {
    fn default() -> Self {
        Raw {
            value: BigInt::zero(),
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
        match value {
            Raw { value } => Self::Raw {
                value: value.into(),
            },
            Initialized { config, value } => Self::Initialized {
                config: (*config.reference().unwrap()).into(),
                value: value.into(),
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

        Raw { value }
    }
}

macro_rules! impl_from_uint {
    ($type:ty) => {
        impl<const N: usize> From<$type> for RandomField<'_, N> {
            fn from(value: $type) -> Self {
                let value = BigInt::from(value);
                Raw { value }
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
        Raw { value }
    }
}

impl<const N: usize> FromBytes for RandomField<'_, N> {
    fn from_bytes_le(bytes: &[u8]) -> Option<Self> {
        Some(Raw {
            value: BigInt::<N>::from_bytes_le(bytes)?,
        })
    }

    fn from_bytes_be(bytes: &[u8]) -> Option<Self> {
        Some(Raw {
            value: BigInt::<N>::from_bytes_be(bytes)?,
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

#[cfg(test)]
mod tests {
    use crate::{
        big_int,
        field::{config::ConfigRef, RandomField},
        field_config,
    };

    #[test]
    fn test_with_raw_value_or_for_raw_variant() {
        let raw_field: RandomField<'_, 1> = RandomField::Raw {
            value: big_int!(42),
        };

        assert_eq!(
            raw_field.with_raw_value_or(|v| *v, big_int!(99)),
            big_int!(42)
        );
    }

    #[test]
    fn test_with_raw_value_or_for_initialized_variant() {
        let config = field_config!(23);
        let config = ConfigRef::from(&config);
        let init_field: RandomField<'_, 1> = RandomField::Initialized {
            config,
            value: big_int!(10),
        };

        assert_eq!(
            init_field.with_raw_value_or(|v| *v, big_int!(99)),
            big_int!(99)
        );
    }
    #[test]
    fn test_with_init_value_or_initialized() {
        let config = field_config!(23);
        let config = ConfigRef::from(&config);
        let init_field: RandomField<'_, 1> = RandomField::Initialized {
            config,
            value: big_int!(10),
        };

        assert_eq!(
            init_field.with_init_value_or(|_, v| *v, big_int!(99)),
            big_int!(10)
        );
    }

    #[test]
    fn test_with_init_value_or_raw() {
        let raw_field: RandomField<'_, 1> = RandomField::Raw {
            value: big_int!(42),
        };

        assert_eq!(
            raw_field.with_init_value_or(|_, v| *v, big_int!(99)),
            big_int!(99)
        );
    }
}
