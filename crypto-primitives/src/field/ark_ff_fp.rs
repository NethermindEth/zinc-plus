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

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::{format, vec::Vec};
    use ark_ff::{MontBackend, MontConfig};
    use num_traits::{One, Zero};

    // Define a test prime field modulus
    // Using a 256-bit prime for testing: 2^256 - 2^32 - 977 (secp256k1 field prime)
    #[derive(MontConfig)]
    #[modulus = "115792089237316195423570985008687907853269984665640564039457584007908834671663"]
    #[generator = "3"]
    pub struct TestFpConfig;
    type TestFp = Fp<MontBackend<TestFpConfig, 4>, 4>;

    #[test]
    fn zero_one_basics() {
        let z = TestFp::zero();
        assert!(z.is_zero());
        let o = TestFp::one();
        assert!(!o.is_zero());
        assert_ne!(z, o);
    }

    #[test]
    fn basic_operations() {
        let a = TestFp::from(10_u64);
        let b = TestFp::from(5_u64);

        // Test addition
        let c = a + b;
        assert_eq!(c, TestFp::from(15_u64));

        // Test subtraction
        let d = a - b;
        assert_eq!(d, TestFp::from(5_u64));

        // Test multiplication
        let e = a * b;
        assert_eq!(e, TestFp::from(50_u64));
    }

    #[allow(clippy::op_ref)]
    #[test]
    fn reference_operations() {
        let a = TestFp::from(10_u64);
        let b = TestFp::from(5_u64);

        // Test reference-based addition
        let c = a + &b;
        assert_eq!(c, TestFp::from(15_u64));

        // Test reference-based subtraction
        let d = a - &b;
        assert_eq!(d, TestFp::from(5_u64));

        // Test reference-based multiplication
        let e = a * &b;
        assert_eq!(e, TestFp::from(50_u64));
    }

    #[test]
    fn from_unsigned_and_signed() {
        assert_eq!(TestFp::from(0_u64), TestFp::zero());
        assert_eq!(TestFp::from(1_u32), TestFp::one());
        assert_eq!(TestFp::from(-1_i32) + TestFp::one(), TestFp::zero());
        assert_eq!(TestFp::from(-5_i64) + TestFp::from(5_u64), TestFp::zero());
    }

    #[test]
    fn from_bool() {
        assert_eq!(TestFp::from(true), TestFp::one());
        assert_eq!(TestFp::from(false), TestFp::zero());

        let t: TestFp = true.into();
        let f: TestFp = false.into();
        assert_eq!(t, TestFp::one());
        assert_eq!(f, TestFp::zero());
    }

    #[test]
    fn basic_add_smoke() {
        let a: TestFp = 123_u64.into();
        let b: TestFp = 456_u64.into();
        assert_eq!(a + b, TestFp::from(579_u64));
    }

    #[test]
    fn add_wrapping_and_basic() {
        let a: TestFp = (-100_i64).into();
        let b: TestFp = 105_u64.into();
        let c = a + b;
        let d: TestFp = 5_u64.into();
        assert_eq!(c, d);
    }

    #[test]
    fn sub_basic() {
        let a: TestFp = 100_u64.into();
        let b: TestFp = 7_u64.into();
        assert_eq!(a - b, 93_u64.into());
    }

    #[test]
    fn mul_basic() {
        let a: TestFp = 100_u64.into();
        let b: TestFp = 7_u64.into();
        assert_eq!(a * b, 700_u64.into());
    }

    #[test]
    fn add_assign_basic() {
        let mut a: TestFp = 5_u64.into();
        a += TestFp::from(6_u64);
        assert_eq!(a, 11_u64.into());
    }

    #[test]
    fn sub_assign_basic() {
        let mut a: TestFp = 20_u64.into();
        a -= TestFp::from(7_u64);
        assert_eq!(a, 13_u64.into());
    }

    #[test]
    fn mul_assign_basic() {
        let mut a: TestFp = 11_u64.into();
        a *= TestFp::from(3_u64);
        assert_eq!(a, 33_u64.into());
    }

    #[test]
    fn neg_basic() {
        let a: TestFp = 9_u64.into();
        let neg_a = -a;

        assert_eq!(a + neg_a, TestFp::zero());
    }

    #[test]
    fn div_basic() {
        let num: TestFp = 11_u64.into();
        let den: TestFp = 5_u64.into();
        let q = num / den;
        assert_eq!(q * den, num);
    }

    #[test]
    #[should_panic(expected = "Division by zero")]
    fn div_by_zero_panics() {
        let a: TestFp = 7_u64.into();
        let zero = TestFp::zero();
        let _ = a / zero;
    }

    #[test]
    fn div_assign_basic() {
        let mut a: TestFp = 20_u64.into();
        let b: TestFp = 4_u64.into();
        a /= b;
        assert_eq!(a * b, 20_u64.into());
    }

    #[test]
    fn pow_operation() {
        // Test basic exponentiation
        let base = TestFp::from(2_u64);

        // 2^0 = 1
        assert_eq!(base.pow(0), TestFp::one());

        // 2^1 = 2
        assert_eq!(base.pow(1), base);

        // 2^3 = 8
        assert_eq!(base.pow(3), TestFp::from(8_u64));

        // 2^10 = 1024
        assert_eq!(base.pow(10), TestFp::from(1024_u64));

        // Test with different base
        let base = TestFp::from(3_u64);

        // 3^4 = 81
        assert_eq!(base.pow(4), TestFp::from(81_u64));

        // Test with base 1
        let base = TestFp::from(1_u64);
        assert_eq!(base.pow(1000), TestFp::from(1_u64));

        // Test with base 0
        let base = TestFp::from(0_u64);
        assert_eq!(base.pow(0), TestFp::one()); // 0^0 = 1 by convention
        assert_eq!(base.pow(10), TestFp::zero()); // 0^n = 0 for n > 0
    }

    #[test]
    fn inv_operation() {
        let a = TestFp::from(5_u64);
        let inv_a = a.inv().unwrap();
        assert_eq!(a * inv_a, TestFp::one());

        // Test that zero has no inverse
        let zero = TestFp::zero();
        assert!(zero.inv().is_none());
    }

    #[test]
    fn checked_neg() {
        // Test with positive number
        let a = TestFp::from(10_u64);
        let neg_a = a.checked_neg().unwrap();
        assert_eq!(neg_a, TestFp::from(-10_i64));

        // Test with negative number
        let b = TestFp::from(-5_i64);
        let neg_b = b.checked_neg().unwrap();
        assert_eq!(neg_b, TestFp::from(5_u64));

        // Test with zero
        let zero = TestFp::zero();
        let neg_zero = zero.checked_neg().unwrap();
        assert_eq!(neg_zero, zero);
    }

    #[test]
    fn checked_add() {
        let a = TestFp::from(10_u64);
        let b = TestFp::from(5_u64);

        let c = a.checked_add(&b).unwrap();
        assert_eq!(c, TestFp::from(15_u64));
    }

    #[test]
    fn checked_sub() {
        let a = TestFp::from(10_u64);
        let b = TestFp::from(5_u64);

        let d = a.checked_sub(&b).unwrap();
        assert_eq!(d, TestFp::from(5_u64));
    }

    #[test]
    fn checked_mul() {
        let a = TestFp::from(10_u64);
        let b = TestFp::from(5_u64);

        let e = a.checked_mul(&b).unwrap();
        assert_eq!(e, TestFp::from(50_u64));
    }

    #[test]
    fn checked_div() {
        let a = TestFp::from(10_u64);
        let b = TestFp::from(5_u64);
        let zero = TestFp::zero();

        // Normal division
        let c = a.checked_div(&b).unwrap();
        assert_eq!(c * b, a);

        // Division by zero
        assert!(a.checked_div(&zero).is_none());
    }

    #[allow(clippy::op_ref)]
    #[test]
    fn ref_and_value_combinations_add_sub_mul() {
        let a: TestFp = 42_u64.into();
        let b: TestFp = 123_u64.into();

        let r1 = a + b;
        let a1: TestFp = 42_u64.into();
        let b1: TestFp = 123_u64.into();
        let r2 = a1 + b1;
        let r3 = a1 + &b1;
        let a2: TestFp = 42_u64.into();
        let b2: TestFp = 123_u64.into();
        let r4 = &a2 + b2;
        assert_eq!(r1, r2);
        assert_eq!(r1, r3);
        assert_eq!(r1, r4);

        let a: TestFp = 88_u64.into();
        let b: TestFp = 59_u64.into();
        let s1 = a - b;
        let a1: TestFp = 88_u64.into();
        let b1: TestFp = 59_u64.into();
        let s2 = a1 - b1;
        let s3 = a1 - &b1;
        let a2: TestFp = 88_u64.into();
        let b2: TestFp = 59_u64.into();
        let s4 = &a2 - b2;
        assert_eq!(s1, s2);
        assert_eq!(s1, s3);
        assert_eq!(s1, s4);

        let a: TestFp = 9_u64.into();
        let b: TestFp = 14_u64.into();
        let m1 = a * b;
        let a1: TestFp = 9_u64.into();
        let b1: TestFp = 14_u64.into();
        let m2 = a1 * b1;
        let m3 = a1 * &b1;
        let a2: TestFp = 9_u64.into();
        let b2: TestFp = 14_u64.into();
        let m4 = &a2 * b2;
        assert_eq!(m1, m2);
        assert_eq!(m1, m3);
        assert_eq!(m1, m4);
    }

    #[test]
    fn assign_ops_with_refs_and_val() {
        let mut a: TestFp = 100_u64.into();
        let b: TestFp = 50_u64.into();
        a += b;
        assert_eq!(a, 150_u64.into());

        let mut c: TestFp = 100_u64.into();
        let d: TestFp = 50_u64.into();
        c += &d;
        assert_eq!(c, 150_u64.into());

        let mut e: TestFp = 100_u64.into();
        let f: TestFp = 30_u64.into();
        e -= f;
        assert_eq!(e, 70_u64.into());

        let mut g: TestFp = 100_u64.into();
        let h: TestFp = 30_u64.into();
        g -= &h;
        assert_eq!(g, 70_u64.into());

        let mut i: TestFp = 10_u64.into();
        let j: TestFp = 5_u64.into();
        i *= j;
        assert_eq!(i, 50_u64.into());

        let mut k: TestFp = 10_u64.into();
        let l: TestFp = 5_u64.into();
        k *= &l;
        assert_eq!(k, 50_u64.into());
    }

    #[test]
    fn aggregate_operations() {
        // Test Sum trait
        let values: Vec<TestFp> = [1_u64, 2_u64, 3_u64]
            .into_iter()
            .map(TestFp::from)
            .collect();
        let sum: TestFp = values.iter().sum();
        assert_eq!(sum, TestFp::from(6_u64));

        let sum2: TestFp = values.into_iter().sum();
        assert_eq!(sum2, TestFp::from(6_u64));

        // Test Product trait
        let values: Vec<TestFp> = [2_u64, 3_u64, 4_u64]
            .into_iter()
            .map(TestFp::from)
            .collect();
        let product: TestFp = values.iter().product();
        assert_eq!(product, TestFp::from(24_u64));

        let product2: TestFp = values.into_iter().product();
        assert_eq!(product2, TestFp::from(24_u64));

        // Test empty collections
        let empty_vec: Vec<TestFp> = Vec::new();
        let empty_sum: TestFp = empty_vec.iter().sum();
        assert_eq!(empty_sum, TestFp::zero());

        let empty_product: TestFp = empty_vec.iter().product();
        assert_eq!(empty_product, TestFp::one());
    }

    #[test]
    fn conversions() {
        // Test From<ArkWrappedFp> for Fp (via new)
        let inner = ark_ff::MontFp!("123");
        let wrapped = TestFp::new(inner);
        assert_eq!(wrapped, TestFp::from(123_u64));

        // Test inner() and into_inner()
        let value = TestFp::from(456_u64);
        let inner_ref = value.inner();
        assert_eq!(*inner_ref, ark_ff::MontFp!("456"));
        assert_eq!(value.into_inner(), ark_ff::MontFp!("456"));
    }

    #[test]
    fn from_primitive() {
        // Test from_u8
        let a = TestFp::from(42_u8);
        assert_eq!(a, TestFp::from(42_u64));

        // Test from_u16
        let b = TestFp::from(12345_u16);
        assert_eq!(b, TestFp::from(12345_u64));

        // Test from_u32
        let c = TestFp::from(1234567890_u32);
        assert_eq!(c, TestFp::from(1234567890_u64));

        // Test from_u64
        let d = TestFp::from(1234567890123456789_u64);
        assert_eq!(d, TestFp::from(1234567890123456789_u64));

        // Test from_i8
        let e = TestFp::from(-42_i8);
        assert_eq!(e, TestFp::from(-42_i64));

        // Test from_i16
        let f = TestFp::from(-12345_i16);
        assert_eq!(f, TestFp::from(-12345_i64));

        // Test from_i32
        let g = TestFp::from(-1234567890_i32);
        assert_eq!(g, TestFp::from(-1234567890_i64));

        // Test from_i64
        let h = TestFp::from(-1234567890123456789_i64);
        assert_eq!(h, TestFp::from(-1234567890123456789_i64));
    }

    #[test]
    fn from_bigint() {
        let bigint = BigInt::<4>::from(12345_u64);
        let fp: TestFp = bigint.into();
        assert_eq!(fp, TestFp::from(12345_u64));

        // Test round-trip conversion
        let original = TestFp::from(67890_u64);
        let bigint: BigInt<4> = original.into();
        let restored: TestFp = bigint.into();
        assert_eq!(original, restored);
    }

    #[allow(clippy::clone_on_copy)]
    #[test]
    fn clone_and_copy() {
        let a = TestFp::from(42_u64);
        let b = a; // Copy
        let c = a.clone(); // Clone

        assert_eq!(a, b);
        assert_eq!(a, c);
        assert_eq!(b, c);
    }

    #[test]
    fn equality_and_ordering() {
        let a = TestFp::from(10_u64);
        let b = TestFp::from(10_u64);
        let c = TestFp::from(20_u64);

        // Test equality
        assert_eq!(a, b);
        assert_ne!(a, c);

        // Test ordering
        assert!(a < c);
        assert!(c > a);
        assert!(a <= b);
        assert!(a >= b);
    }

    #[test]
    fn hash_trait() {
        use core::hash::{Hash, Hasher};

        // Simple hasher for testing
        struct TestHasher {
            state: u64,
        }

        impl Hasher for TestHasher {
            fn finish(&self) -> u64 {
                self.state
            }

            fn write(&mut self, bytes: &[u8]) {
                for &byte in bytes {
                    self.state = self.state.wrapping_mul(31).wrapping_add(u64::from(byte));
                }
            }
        }

        let a = TestFp::from(42_u64);
        let b = TestFp::from(42_u64);

        let mut hasher_a = TestHasher { state: 0 };
        a.hash(&mut hasher_a);
        let hash_a = hasher_a.finish();

        let mut hasher_b = TestHasher { state: 0 };
        b.hash(&mut hasher_b);
        let hash_b = hasher_b.finish();

        // Equal values should have equal hashes
        assert_eq!(hash_a, hash_b);
    }

    #[test]
    fn default_is_zero() {
        let default_fp = TestFp::default();
        assert_eq!(default_fp, TestFp::zero());
        assert!(default_fp.is_zero());
    }

    #[test]
    fn display_and_debug() {
        let a = TestFp::from(42_u64);

        // Test that Display and Debug don't panic
        let _display = format!("{}", a);
        let _debug = format!("{:?}", a);

        // Both should produce non-empty strings
        assert!(!_display.is_empty());
        assert!(!_debug.is_empty());
    }

    #[test]
    fn from_str() {
        // Test parsing from string
        let a: TestFp = "123".parse().unwrap();
        assert_eq!(a, TestFp::from(123_u64));

        let b: TestFp = "0".parse().unwrap();
        assert_eq!(b, TestFp::zero());

        let c: TestFp = "1".parse().unwrap();
        assert_eq!(c, TestFp::one());
    }

    #[test]
    fn deref_access() {
        let a = TestFp::from(42_u64);
        // Test that we can access inner methods via Deref
        let _ = a.is_zero();
        let _ = a.inverse();
    }
}
