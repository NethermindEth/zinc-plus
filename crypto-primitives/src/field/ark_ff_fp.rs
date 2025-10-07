use super::*;
use ark_ff::{
    AdditiveGroup, BigInt, FftField, FpConfig, LegendreSymbol, MontBackend, MontConfig,
    SqrtPrecomputation,
    fields::{Field as ArkWrappedField, Fp as ArkWrappedFp, PrimeField as ArkPrimeField},
};
use ark_serialize::{
    CanonicalDeserialize, CanonicalDeserializeWithFlags, CanonicalSerialize,
    CanonicalSerializeWithFlags, Compress, Flags, Read, SerializationError, Valid, Validate, Write,
};
use core::{
    cmp::Ordering,
    fmt::{Display, Formatter, Result as FmtResult},
    hash::{Hash, Hasher},
    iter::{Product, Sum},
    marker::PhantomData,
    ops::{Add, AddAssign, Deref, Mul, MulAssign, Sub, SubAssign},
    str::FromStr,
};
use crypto_primitives_proc_macros::InfallibleCheckedOp;
use num_bigint::BigUint;
use num_traits::{
    CheckedAdd, CheckedDiv, CheckedMul, CheckedNeg, CheckedSub, ConstOne, ConstZero, One, Pow, Zero,
};

#[cfg(feature = "rand")]
use ark_std::{UniformRand, rand::prelude::*};
#[cfg(feature = "rand")]
use rand::distr::StandardUniform;

// Can't derive core traits because of the generic parameters
#[derive(InfallibleCheckedOp)]
#[infallible_checked_unary_op((CheckedNeg, neg))]
#[infallible_checked_binary_op((CheckedAdd, add), (CheckedSub, sub), (CheckedMul, mul))]
#[repr(transparent)]
pub struct Fp<P: FpConfig<N>, const N: usize>(ArkWrappedFp<P, N>);

impl<P: FpConfig<N>, const N: usize> Fp<P, N> {
    /// Wraps a given value into this wrapper type
    #[inline(always)]
    pub const fn new(value: ArkWrappedFp<P, N>) -> Self {
        Self(value)
    }

    /// Get the reference to the wrapped value
    #[inline(always)]
    pub const fn inner(&self) -> &ArkWrappedFp<P, N> {
        &self.0
    }

    /// Get the wrapped value, consuming self
    #[inline(always)]
    pub const fn into_inner(self) -> ArkWrappedFp<P, N> {
        self.0
    }
}

impl<T: MontConfig<N>, const N: usize> Fp<MontBackend<T, N>, N> {
    #[doc(hidden)]
    pub const R: BigInt<N> = T::R;
    #[doc(hidden)]
    pub const R2: BigInt<N> = T::R2;
    #[doc(hidden)]
    pub const INV: u64 = T::INV;

    #[inline]
    pub const fn new_from_bigint(element: BigInt<N>) -> Self {
        Self(ArkWrappedFp::<MontBackend<T, N>, N>::new(element))
    }
}

//
// Core traits
//

impl<P: FpConfig<N>, const N: usize> Debug for Fp<P, N> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        Debug::fmt(&self.0, f)
    }
}

impl<P: FpConfig<N>, const N: usize> Display for Fp<P, N> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "{}", self.0)
    }
}

impl<P: FpConfig<N>, const N: usize> Default for Fp<P, N> {
    #[inline(always)]
    fn default() -> Self {
        Self::zero()
    }
}

impl<P: FpConfig<N>, const N: usize> Deref for Fp<P, N> {
    type Target = ArkWrappedFp<P, N>;

    fn deref(&self) -> &Self::Target {
        self.inner()
    }
}

impl<P: FpConfig<N>, const N: usize> Clone for Fp<P, N> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

impl<P: FpConfig<N>, const N: usize> Copy for Fp<P, N> {}

impl<P: FpConfig<N>, const N: usize> PartialEq for Fp<P, N> {
    fn eq(&self, other: &Self) -> bool {
        self.0.eq(&other.0)
    }
}

impl<P: FpConfig<N>, const N: usize> Eq for Fp<P, N> {}

impl<P: FpConfig<N>, const N: usize> PartialOrd for Fp<P, N> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<P: FpConfig<N>, const N: usize> Ord for Fp<P, N> {
    fn cmp(&self, other: &Self) -> Ordering {
        Ord::cmp(&self.0, &other.0)
    }
}

impl<P: FpConfig<N>, const N: usize> Hash for Fp<P, N> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state)
    }
}

impl<P: FpConfig<N>, const N: usize> FromStr for Fp<P, N> {
    type Err = <ArkWrappedFp<P, N> as FromStr>::Err;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        ArkWrappedFp::<P, N>::from_str(s).map(Self)
    }
}

//
// Zero and One traits
//

impl<P: FpConfig<N>, const N: usize> Zero for Fp<P, N> {
    #[inline]
    fn zero() -> Self {
        <Self as ConstZero>::ZERO
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}

impl<P: FpConfig<N>, const N: usize> One for Fp<P, N> {
    fn one() -> Self {
        <Self as ConstOne>::ONE
    }
}

impl<P: FpConfig<N>, const N: usize> ConstZero for Fp<P, N> {
    const ZERO: Self = Self(P::ZERO);
}

impl<P: FpConfig<N>, const N: usize> ConstOne for Fp<P, N> {
    const ONE: Self = Self(P::ONE);
}

//
// Basic arithmetic operations
//

impl<P: FpConfig<N>, const N: usize> Neg for Fp<P, N> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self(self.0.neg())
    }
}

macro_rules! impl_basic_op {
    ($trait:ident, $method:ident) => {
        impl<P: FpConfig<N>, const N: usize> $trait for Fp<P, N> {
            type Output = Self;

            #[inline(always)]
            fn $method(self, rhs: Self) -> Self::Output {
                Self(self.0.$method(rhs.0))
            }
        }

        impl<P: FpConfig<N>, const N: usize> $trait<&Self> for Fp<P, N> {
            type Output = Self;

            #[inline(always)]
            fn $method(self, rhs: &Self) -> Self::Output {
                Self(self.0.$method(&rhs.0))
            }
        }

        impl<P: FpConfig<N>, const N: usize> $trait<&mut Self> for Fp<P, N> {
            type Output = Self;

            #[inline(always)]
            fn $method(self, rhs: &mut Self) -> Self::Output {
                Self(self.0.$method(&rhs.0))
            }
        }

        impl<P: FpConfig<N>, const N: usize> $trait for &Fp<P, N> {
            type Output = Fp<P, N>;

            #[inline(always)]
            fn $method(self, rhs: Self) -> Self::Output {
                Fp(self.0.$method(rhs.0))
            }
        }

        impl<P: FpConfig<N>, const N: usize> $trait<Fp<P, N>> for &Fp<P, N> {
            type Output = Fp<P, N>;

            #[inline(always)]
            fn $method(self, rhs: Fp<P, N>) -> Self::Output {
                Fp(self.0.$method(&rhs.0))
            }
        }
    };
}

impl_basic_op!(Add, add);
impl_basic_op!(Sub, sub);
impl_basic_op!(Mul, mul);

impl<P: FpConfig<N>, const N: usize> Div for Fp<P, N> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        self.div(&rhs)
    }
}

impl<P: FpConfig<N>, const N: usize> Div<&Self> for Fp<P, N> {
    type Output = Self;

    fn div(self, rhs: &Self) -> Self::Output {
        self.checked_div(rhs).expect("Division by zero")
    }
}

impl<P: FpConfig<N>, const N: usize> Div<&mut Self> for Fp<P, N> {
    type Output = Self;

    fn div(self, rhs: &mut Self) -> Self::Output {
        self.div(&*rhs)
    }
}

impl<P: FpConfig<N>, const N: usize> Pow<u32> for Fp<P, N> {
    type Output = Self;

    fn pow(self, rhs: u32) -> Self::Output {
        Self(self.0.pow([u64::from(rhs)]))
    }
}

impl<P: FpConfig<N>, const N: usize> Inv for Fp<P, N> {
    type Output = Option<Self>;

    fn inv(mut self) -> Self::Output {
        let _ = self.0.inverse_in_place()?;
        Some(self)
    }
}

//
// Checked arithmetic operations
// (Note: Field operations do not overflow)
//

impl<P: FpConfig<N>, const N: usize> CheckedDiv for Fp<P, N> {
    #[allow(clippy::arithmetic_side_effects)] // False alert
    fn checked_div(&self, rhs: &Self) -> Option<Self> {
        rhs.0.inverse().map(|inv| Self(self.0 * inv))
    }
}

//
// Arithmetic assign operations
//

macro_rules! impl_field_op_assign {
    ($trait:ident, $method:ident, $inner:ident) => {
        impl<P: FpConfig<N>, const N: usize> $trait for Fp<P, N> {
            fn $method(&mut self, rhs: Self) {
                // Use reference for inner call to avoid moves of rhs.0 where not needed
                *self = self.$inner(&rhs);
            }
        }

        impl<P: FpConfig<N>, const N: usize> $trait<&Self> for Fp<P, N> {
            fn $method(&mut self, rhs: &Self) {
                *self = self.$inner(rhs);
            }
        }

        impl<P: FpConfig<N>, const N: usize> $trait<&mut Self> for Fp<P, N> {
            fn $method(&mut self, rhs: &mut Self) {
                *self = self.$inner(rhs);
            }
        }
    };
}

impl_field_op_assign!(AddAssign, add_assign, add);
impl_field_op_assign!(SubAssign, sub_assign, sub);
impl_field_op_assign!(MulAssign, mul_assign, mul);
impl_field_op_assign!(DivAssign, div_assign, div);

//
// Aggregate operations
//

impl<P: FpConfig<N>, const N: usize> Sum for Fp<P, N> {
    #[allow(clippy::arithmetic_side_effects)] // False alert
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, x| acc + x)
    }
}

impl<'a, P: FpConfig<N>, const N: usize> Sum<&'a Self> for Fp<P, N> {
    #[allow(clippy::arithmetic_side_effects)] // False alert
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, x| acc + x)
    }
}

impl<P: FpConfig<N>, const N: usize> Product for Fp<P, N> {
    #[allow(clippy::arithmetic_side_effects)] // False alert
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |acc, x| acc * x)
    }
}

impl<'a, P: FpConfig<N>, const N: usize> Product<&'a Self> for Fp<P, N> {
    #[allow(clippy::arithmetic_side_effects)] // False alert
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |acc, x| acc * x)
    }
}

//
// Conversions
//

impl<P: FpConfig<N>, const N: usize> From<&Fp<P, N>> for Fp<P, N> {
    fn from(value: &Self) -> Self {
        *value
    }
}

macro_rules! impl_from_delegate {
    ($($t:ty),* $(,)?) => {
        $(
            impl<P: FpConfig<N>, const N: usize> From<$t> for Fp<P, N> {
                fn from(value: $t) -> Self {
                    Self(ArkWrappedFp::from(value))
                }
            }

            impl<P: FpConfig<N>, const N: usize> From<&$t> for Fp<P, N> {
                fn from(value: &$t) -> Self {
                    Self::from(*value)
                }
            }
        )*
    };
}

impl_from_delegate!(u8, u16, u32, u64, u128);
impl_from_delegate!(i8, i16, i32, i64, i128);

impl<P: FpConfig<N>, const N: usize> From<bool> for Fp<P, N> {
    fn from(value: bool) -> Self {
        if value { Self::one() } else { Self::zero() }
    }
}

impl<P: FpConfig<N>, const N: usize> From<BigInt<N>> for Fp<P, N> {
    fn from(value: BigInt<N>) -> Self {
        Self(ArkWrappedFp::from(value))
    }
}

impl<P: FpConfig<N>, const N: usize> From<Fp<P, N>> for BigInt<N> {
    fn from(value: Fp<P, N>) -> Self {
        value.0.into()
    }
}

impl<P: FpConfig<N>, const N: usize> From<BigUint> for Fp<P, N> {
    fn from(value: BigUint) -> Self {
        Self(ArkWrappedFp::from(value))
    }
}

impl<P: FpConfig<N>, const N: usize> From<Fp<P, N>> for BigUint {
    fn from(value: Fp<P, N>) -> Self {
        value.0.into()
    }
}

//
// Ring and Field
//

impl<P: FpConfig<N>, const N: usize> Ring for Fp<P, N> {}

impl<P: FpConfig<N>, const N: usize> IntRing for Fp<P, N> {}

impl<P: FpConfig<N>, const N: usize> Field for Fp<P, N> {}

impl<P: FpConfig<N>, const N: usize> PrimeField for Fp<P, N> {
    type Inner = BigInt<N>;
    const MODULUS: Self::Inner = P::MODULUS;

    fn new_unchecked(inner: Self::Inner) -> Self {
        Self(ArkWrappedFp(inner, PhantomData))
    }

    fn inner(&self) -> &Self::Inner {
        &self.0.0
    }
}

//
// RNG
//

#[cfg(feature = "rand")]
impl<P: FpConfig<N>, const N: usize> Distribution<Fp<P, N>> for StandardUniform {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Fp<P, N> {
        Fp(UniformRand::rand(rng))
    }
}

#[cfg(feature = "rand")]
impl<P: FpConfig<N>, const N: usize> UniformRand for Fp<P, N> {
    fn rand<R: ark_std::rand::Rng + ?Sized>(rng: &mut R) -> Self {
        Self(ArkWrappedFp::rand(rng))
    }
}

//
// Traits from ark-ff
//

impl<P: FpConfig<N>, const N: usize> CanonicalSerialize for Fp<P, N> {
    fn serialize_with_mode<W: Write>(
        &self,
        writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.0.serialize_with_mode(writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.0.serialized_size(compress)
    }
}

impl<P: FpConfig<N>, const N: usize> CanonicalSerializeWithFlags for Fp<P, N> {
    fn serialize_with_flags<W: Write, G: Flags>(
        &self,
        writer: W,
        flags: G,
    ) -> Result<(), SerializationError> {
        self.0.serialize_with_flags(writer, flags)
    }

    fn serialized_size_with_flags<G: Flags>(&self) -> usize {
        self.0.serialized_size_with_flags::<G>()
    }
}

impl<P: FpConfig<N>, const N: usize> CanonicalDeserialize for Fp<P, N> {
    fn deserialize_with_mode<R: Read>(
        reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        ArkWrappedFp::deserialize_with_mode(reader, compress, validate).map(Self)
    }
}

impl<P: FpConfig<N>, const N: usize> CanonicalDeserializeWithFlags for Fp<P, N> {
    fn deserialize_with_flags<R: Read, G: Flags>(
        reader: R,
    ) -> Result<(Self, G), SerializationError> {
        ArkWrappedFp::deserialize_with_flags(reader).map(|(field, flags)| (Self(field), flags))
    }
}

impl<P: FpConfig<N>, const N: usize> Valid for Fp<P, N> {
    fn check(&self) -> Result<(), SerializationError> {
        self.0.check()
    }
}

impl<P: FpConfig<N>, const N: usize> AdditiveGroup for Fp<P, N> {
    type Scalar = Self;
    const ZERO: Self = <Self as ConstZero>::ZERO;
}

impl<P: FpConfig<N>, const N: usize> zeroize::Zeroize for Fp<P, N> {
    fn zeroize(&mut self) {
        self.0.zeroize()
    }
}

impl<P: FpConfig<N>, const N: usize> ArkWrappedField for Fp<P, N> {
    type BasePrimeField = Self;
    const SQRT_PRECOMP: Option<SqrtPrecomputation<Self>> = {
        match ArkWrappedFp::SQRT_PRECOMP {
            None => None,
            Some(p) => Some(match p {
                SqrtPrecomputation::TonelliShanks {
                    two_adicity,
                    quadratic_nonresidue_to_trace,
                    trace_of_modulus_minus_one_div_two,
                } => SqrtPrecomputation::TonelliShanks {
                    two_adicity,
                    quadratic_nonresidue_to_trace: Self(quadratic_nonresidue_to_trace),
                    trace_of_modulus_minus_one_div_two,
                },
                SqrtPrecomputation::Case3Mod4 {
                    modulus_plus_one_div_four,
                } => SqrtPrecomputation::Case3Mod4 {
                    modulus_plus_one_div_four,
                },
                _ => panic!(
                    "Can't deal with a precomputation that is not Tonelli-Shanks or Case3Mod4"
                ),
            }),
        }
    };
    const ONE: Self = <Self as ConstOne>::ONE;

    fn extension_degree() -> u64 {
        ArkWrappedFp::<P, N>::extension_degree()
    }

    fn to_base_prime_field_elements(&self) -> impl Iterator<Item = Self::BasePrimeField> {
        self.0.to_base_prime_field_elements().map(Self)
    }

    fn from_base_prime_field_elems(
        elems: impl IntoIterator<Item = Self::BasePrimeField>,
    ) -> Option<Self> {
        ArkWrappedFp::from_base_prime_field_elems(elems.into_iter().map(|v| v.0)).map(Self)
    }

    fn from_base_prime_field(elem: Self::BasePrimeField) -> Self {
        Self(ArkWrappedFp::from_base_prime_field(elem.0))
    }

    fn from_random_bytes_with_flags<G: Flags>(bytes: &[u8]) -> Option<(Self, G)> {
        ArkWrappedFp::from_random_bytes_with_flags(bytes).map(|(field, flags)| (Self(field), flags))
    }

    fn legendre(&self) -> LegendreSymbol {
        self.0.legendre()
    }

    fn square(&self) -> Self {
        Self(self.0.square())
    }

    fn square_in_place(&mut self) -> &mut Self {
        self.0.square_in_place();
        self
    }

    fn inverse(&self) -> Option<Self> {
        self.0.inverse().map(Self)
    }

    fn inverse_in_place(&mut self) -> Option<&mut Self> {
        let _ = self.0.inverse_in_place()?;
        Some(self)
    }

    fn frobenius_map_in_place(&mut self, power: usize) {
        self.0.frobenius_map_in_place(power);
    }

    fn mul_by_base_prime_field(&self, elem: &Self::BasePrimeField) -> Self {
        Self(self.0.mul_by_base_prime_field(&elem.0))
    }
}

impl<P: FpConfig<N>, const N: usize> FftField for Fp<P, N> {
    const GENERATOR: Self = Self(ArkWrappedFp::<P, N>::GENERATOR);
    const TWO_ADICITY: u32 = ArkWrappedFp::<P, N>::TWO_ADICITY;
    const TWO_ADIC_ROOT_OF_UNITY: Self = Self(ArkWrappedFp::<P, N>::TWO_ADIC_ROOT_OF_UNITY);
}

impl<P: FpConfig<N>, const N: usize> ArkPrimeField for Fp<P, N> {
    type BigInt = <ArkWrappedFp<P, N> as ArkPrimeField>::BigInt;
    const MODULUS: Self::BigInt = P::MODULUS;
    const MODULUS_MINUS_ONE_DIV_TWO: Self::BigInt =
        <ArkWrappedFp<P, N> as ArkPrimeField>::MODULUS_MINUS_ONE_DIV_TWO;
    const MODULUS_BIT_SIZE: u32 = <ArkWrappedFp<P, N> as ArkPrimeField>::MODULUS_BIT_SIZE;
    const TRACE: Self::BigInt = <ArkWrappedFp<P, N> as ArkPrimeField>::TRACE;
    const TRACE_MINUS_ONE_DIV_TWO: Self::BigInt =
        <ArkWrappedFp<P, N> as ArkPrimeField>::TRACE_MINUS_ONE_DIV_TWO;

    fn from_bigint(repr: Self::BigInt) -> Option<Self> {
        ArkWrappedFp::<P, N>::from_bigint(repr).map(Self)
    }

    fn into_bigint(self) -> Self::BigInt {
        self.0.into_bigint()
    }
}

//
// Predefined fields of various sizes for convenience
//

pub type Fp64<P> = Fp<P, 1>;
pub type Fp128<P> = Fp<P, 2>;
pub type Fp192<P> = Fp<P, 3>;
pub type Fp256<P> = Fp<P, 4>;
pub type Fp320<P> = Fp<P, 5>;
pub type Fp384<P> = Fp<P, 6>;
pub type Fp448<P> = Fp<P, 7>;
pub type Fp512<P> = Fp<P, 8>;
pub type Fp576<P> = Fp<P, 9>;
pub type Fp640<P> = Fp<P, 10>;
pub type Fp704<P> = Fp<P, 11>;
pub type Fp768<P> = Fp<P, 12>;
pub type Fp832<P> = Fp<P, 13>;

#[macro_export]
macro_rules! mont_fp {
    ($c0:expr) => {{ $crate::field::ark_ff_fp::Fp::new(ark_ff::MontFp!($c0)) }};
}
